# SPDX-License-Identifier: Apache-2.0
"""TPU model runner with Omni connector mixin support."""

import contextlib
from typing import Any
import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


class TPUOmniModelRunner(OmniConnectorModelRunnerMixin, TPUModelRunner):
    """TPU Model Runner with Omni Connector Mixin for inter-stage transfers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_intermediate_buffer: dict[str, dict[str, Any]] = {}
        self._omni_num_scheduled_tokens_np: np.ndarray | None = None
        self._omni_last_model_output: object | None = None
        self.omni_prefix_cache = None
        self._sampled_token_ids_cpu_override = None
        self._omni_query_start_loc_model_kwarg = False
        if hasattr(self, "model_config"):
            try:
                self.init_omni_connectors(self.model_config)
            except Exception:
                pass

    def _store_value(self, dest: dict, key: str, value: Any, gpu_keys: set) -> None:
        if isinstance(value, torch.Tensor):
            if key in gpu_keys:
                dest[key] = value.detach().clone()
            else:
                t = value.detach()
                dest[key] = t.to("cpu").contiguous() if t.is_cuda or t.device.type == "xla" else t.contiguous()
        elif isinstance(value, list):
            dest[key] = [
                (item.detach().to("cpu").contiguous() if isinstance(item, torch.Tensor) else item)
                for item in value
            ]
        else:
            dest[key] = value

    def _update_intermediate_buffer(self, req_id: str, upd: dict) -> None:
        if not isinstance(upd, dict) or not upd:
            return
        gpu_keys: set[tuple[str, str]] = set()
        if hasattr(self, "model") and hasattr(self.model, "gpu_resident_buffer_keys"):
            gpu_keys = self.model.gpu_resident_buffer_keys
        existing = self.model_intermediate_buffer.setdefault(req_id, {})
        for k, v in upd.items():
            if isinstance(v, dict):
                existing_sub = existing.setdefault(k, {})
                for qual, val in v.items():
                    self._store_value(existing_sub, qual, val, {q for tk, q in gpu_keys if tk == k})
            else:
                self._store_value(existing, k, v, set())
        req_state = getattr(self, "requests", {}).get(req_id)
        if req_state is not None:
            setattr(req_state, "additional_information_cpu", existing)

    def _sync_local_stage_payloads(self) -> None:
        """Move received full-payload stage inputs into model_intermediate_buffer."""
        cache = getattr(self, "_local_stage_payload_cache", None)
        if not cache:
            return
        lock = getattr(self, "_lock", None)
        ctx = lock if lock is not None else contextlib.nullcontext()
        with ctx:
            if not cache:
                return
            active_req_ids = set(getattr(self, "requests", {}))
            if hasattr(self, "input_batch") and hasattr(self.input_batch, "req_ids"):
                num_reqs = getattr(self.input_batch, "num_reqs", len(self.input_batch.req_ids))
                active_req_ids.update(self.input_batch.req_ids[:num_reqs])
            pending = set(getattr(self, "_full_payload_pending_broadcast_req_ids", set()))
            staged = {
                req_id: payload
                for req_id, payload in cache.items()
                if req_id not in pending and (not active_req_ids or req_id in active_req_ids) and isinstance(payload, dict)
            }
            for req_id in staged:
                cache.pop(req_id, None)
        for req_id, payload in staged.items():
            logger.info(
                "[TPUOmniModelRunner] Syncing stage payload into model_intermediate_buffer for req %s (keys=%s)",
                req_id,
                list(payload.keys()),
            )
            self._update_intermediate_buffer(req_id, payload)

    def _should_accumulate_full_payload_output(self) -> bool:
        """Gate send-side full-payload output accumulation on TPU."""
        if getattr(self, "_omni_connector", None) is None:
            return False
        if getattr(self, "_custom_process_func", None) is not None:
            return True
        model_config = getattr(self, "model_config", None)
        next_func = getattr(model_config, "custom_process_next_stage_input_func", None)
        if isinstance(next_func, str) and bool(next_func):
            return True
        return False

    def execute_model(self, scheduler_output: Any) -> Any:
        if hasattr(self, "_omni_connector") and self._omni_connector is not None:
            for request in getattr(scheduler_output, "pending_input_registrations", []):
                self.register_chunk_recv(request)
            self.recv_full_payload_inputs(scheduler_output)
            self._sync_local_stage_payloads()
            if getattr(self, "_pending_full_payload_send", None):
                flush_ids = set(getattr(scheduler_output, "finished_req_ids", set()))
                flush_ids.update({rid for rid in self._pending_full_payload_send if rid not in getattr(self, "requests", {})})
                if flush_ids:
                    logger.info("[TPUOmniModelRunner] Pre-step flushing full-payload outputs for %s", flush_ids)
                    active_reqs = getattr(self, "requests", {})
                    for rid in flush_ids:
                        req_state = active_reqs.get(rid)
                        if req_state is not None:
                            p_ids = getattr(req_state, "prompt_token_ids", []) or []
                            o_ids = getattr(req_state, "output_token_ids", []) or []
                            req_state.all_token_ids = list(p_ids) + list(o_ids)
                    self.flush_full_payload_outputs(flush_ids)

        output = super().execute_model(scheduler_output)

        if hasattr(self, "_omni_connector") and self._omni_connector is not None:
            real_output = getattr(output, "_model_runner_output", output)
            inter_stage_outputs = getattr(real_output, "inter_stage_outputs", None)
            if inter_stage_outputs and self._should_accumulate_full_payload_output():
                req_ids = getattr(real_output, "req_ids", [])
                logger.info(
                    "[TPUOmniModelRunner] Stage %s accumulating full payload: req_ids=%s, num_inter=%d",
                    getattr(self, "_stage_id", "?"),
                    req_ids,
                    len(inter_stage_outputs),
                )
                for i, rid in enumerate(req_ids):
                    if i < len(inter_stage_outputs) and inter_stage_outputs[i]:
                        req_state = getattr(self, "requests", {}).get(rid)
                        if req_state is not None:
                            if not hasattr(req_state, "request_id"):
                                req_state.request_id = rid
                            if not hasattr(req_state, "external_req_id"):
                                req_state.external_req_id = rid
                            p_ids = getattr(req_state, "prompt_token_ids", []) or []
                            o_ids = getattr(req_state, "output_token_ids", []) or []
                            req_state.all_token_ids = list(p_ids) + list(o_ids)
                        self.accumulate_full_payload_output(rid, inter_stage_outputs[i], req_state)

            if getattr(self, "_pending_full_payload_send", None):
                finished_ids = set(getattr(scheduler_output, "finished_req_ids", set()))
                active_reqs = getattr(self, "requests", {})
                for rid in list(self._pending_full_payload_send.keys()):
                    req_state = active_reqs.get(rid)
                    if req_state is not None and getattr(req_state, "is_finished", False):
                        finished_ids.add(rid)
                    elif req_state is None:
                        finished_ids.add(rid)
                for rid in finished_ids:
                    req_state = active_reqs.get(rid)
                    if req_state is not None:
                        p_ids = getattr(req_state, "prompt_token_ids", []) or []
                        o_ids = getattr(req_state, "output_token_ids", []) or []
                        req_state.all_token_ids = list(p_ids) + list(o_ids)
                if finished_ids:
                    logger.info("[TPUOmniModelRunner] Post-step flushing full-payload outputs for %s", finished_ids)
                    self.flush_full_payload_outputs(finished_ids)

        if hasattr(self, "attach_omni_connector_output"):
            try:
                output = self.attach_omni_connector_output(output)
            except Exception as e:
                logger.debug("[TPUOmniModelRunner] attach_omni_connector_output error: %s", e)
        return output
