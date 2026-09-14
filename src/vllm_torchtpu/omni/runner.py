# SPDX-License-Identifier: Apache-2.0
"""TPU model runner with Omni connector mixin support."""

import contextlib
import multiprocessing
import os
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

        # Resolve Stage ID robustly
        stage_id = None
        proc_name = multiprocessing.current_process().name
        proc_title = ""
        try:
            import setproctitle
            proc_title = setproctitle.getproctitle()
        except Exception:
            pass

        for cand in (0, 1, 2):
            if (
                f"stage{cand}" in proc_name
                or f"stage{cand}" in proc_title
                or f"stage_{cand}" in proc_name
                or f"stage_{cand}" in proc_title
            ):
                stage_id = cand
                break
        if stage_id is None:
            env_stage = os.environ.get("VLLM_OMNI_STAGE_ID")
            if env_stage is not None:
                try:
                    stage_id = int(env_stage)
                except ValueError:
                    pass
        if stage_id is None and hasattr(self, "model_config"):
            model_stage = getattr(self.model_config, "model_stage", None)
            if model_stage == "thinker":
                stage_id = 0
            elif model_stage == "talker":
                stage_id = 1
            elif model_stage == "code2wav":
                stage_id = 2
        if stage_id is None and hasattr(self, "model_config"):
            stage_id = getattr(self.model_config, "stage_id", 0)

        if hasattr(self, "model_config"):
            if stage_id is not None:
                self.model_config.stage_id = stage_id
            try:
                self.init_omni_connectors(self.model_config)
            except Exception as e:
                logger.warning("[TPUOmniModelRunner] init_omni_connectors error: %s", e)

        if stage_id is not None:
            self._stage_id = stage_id

        if self._stage_id == 0:
            self._next_stage_id = 1
            if getattr(self, "_custom_process_func", None) is None:
                try:
                    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
                        thinker2talker_full_payload,
                    )
                    self._custom_process_func = thinker2talker_full_payload
                    self._custom_process_func_path = (
                        "vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_full_payload"
                    )
                except Exception as e:
                    logger.warning("[TPUOmniModelRunner] Could not import thinker2talker_full_payload: %s", e)
        elif self._stage_id == 1:
            self._next_stage_id = 2
            if getattr(self, "_custom_process_func", None) is None:
                try:
                    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
                        talker2code2wav_full_payload,
                    )
                    self._custom_process_func = talker2code2wav_full_payload
                    self._custom_process_func_path = (
                        "vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav_full_payload"
                    )
                except Exception as e:
                    logger.warning("[TPUOmniModelRunner] Could not import talker2code2wav_full_payload: %s", e)
        elif self._stage_id == 2:
            self._next_stage_id = None
            self._custom_process_func = None

        logger.info(
            "[TPUOmniModelRunner] Initialized Stage %s: next_stage=%s, custom_func=%s, connector=%s",
            self._stage_id,
            self._next_stage_id,
            getattr(self, "_custom_process_func_path", None),
            type(getattr(self, "_omni_connector", None)).__name__,
        )

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

    def _sync_local_stage_payloads(self, scheduler_output: Any = None) -> None:
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
            if scheduler_output is not None:
                active_reqs = getattr(scheduler_output, "scheduled_new_reqs", []) or []
                active_req_ids.update([req.req_id for req in active_reqs if hasattr(req, "req_id")])
                active_req_ids.update(getattr(scheduler_output, "num_scheduled_tokens", {}).keys())
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
        if getattr(self, "_stage_id", None) == 2:
            return False
        if getattr(self, "_stage_id", None) in (0, 1):
            return True
        if getattr(self, "_custom_process_func", None) is not None:
            return True
        model_config = getattr(self, "model_config", None)
        next_func = getattr(model_config, "custom_process_next_stage_input_func", None)
        if isinstance(next_func, str) and bool(next_func):
            return True
        return False

    def _update_states(self, scheduler_output: Any) -> Any:
        ret = super()._update_states(scheduler_output)
        if hasattr(self, "_omni_connector") and self._omni_connector is not None:
            finished_ids = set(getattr(scheduler_output, "finished_req_ids", []))
            pending_ids = set(getattr(self, "_pending_full_payload_send", {}).keys())
            to_flush = finished_ids & pending_ids
            if to_flush:
                logger.info("[TPUOmniModelRunner] _update_states flushing full-payload outputs for %s", to_flush)
                active_reqs = getattr(self, "requests", {})
                for rid in to_flush:
                    req_state = active_reqs.get(rid)
                    if req_state is not None:
                        p_ids = getattr(req_state, "prompt_token_ids", []) or []
                        o_ids = getattr(req_state, "output_token_ids", []) or []
                        req_state.all_token_ids = list(p_ids) + list(o_ids)
                self.flush_full_payload_outputs(to_flush)
            cleanup_fn = getattr(self, "cleanup_finished_request", None)
            if cleanup_fn is not None and getattr(self, "_omni_connector_initialized", False):
                for rid in finished_ids:
                    try:
                        cleanup_fn(rid)
                    except Exception as e:
                        logger.debug("[TPUOmniModelRunner] cleanup_finished_request error: %s", e)
        return ret

    def execute_model(self, scheduler_output: Any) -> Any:
        if hasattr(self, "_omni_connector") and self._omni_connector is not None:
            for request in getattr(scheduler_output, "pending_input_registrations", []):
                self.register_chunk_recv(request)
            self.recv_full_payload_inputs(scheduler_output)
            self._sync_local_stage_payloads(scheduler_output)
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
            self._sync_local_stage_payloads(scheduler_output)

        if output is not None and hasattr(self, "attach_omni_connector_output"):
            try:
                output = self.attach_omni_connector_output(output)
            except Exception as e:
                logger.debug("[TPUOmniModelRunner] attach_omni_connector_output error in execute_model: %s", e)
        return output

    def sample_tokens(self, grammar_output: Any) -> Any:
        output = super().sample_tokens(grammar_output)
        if output is None:
            return None

        if hasattr(self, "_omni_connector") and self._omni_connector is not None:
            real_output = getattr(output, "_model_runner_output", output)
            inter_stage_outputs = getattr(real_output, "inter_stage_outputs", None)
            if inter_stage_outputs is None:
                inter_stage_outputs = getattr(real_output, "multimodal_outputs", None)

            if inter_stage_outputs and self._should_accumulate_full_payload_output():
                req_ids = getattr(real_output, "req_ids", [])
                logger.info(
                    "[TPUOmniModelRunner] Stage %s accumulating full payload: req_ids=%s, num_inter=%d",
                    getattr(self, "_stage_id", "?"),
                    req_ids,
                    len(inter_stage_outputs),
                )
                active_reqs = getattr(self, "requests", {})
                for i, rid in enumerate(req_ids):
                    if i < len(inter_stage_outputs) and inter_stage_outputs[i]:
                        req_state = active_reqs.get(rid)
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
                active_reqs = getattr(self, "requests", {})
                finished_ids = set()
                for rid in list(self._pending_full_payload_send.keys()):
                    req_state = active_reqs.get(rid)
                    if req_state is None:
                        finished_ids.add(rid)
                    elif getattr(req_state, "is_finished", False):
                        finished_ids.add(rid)
                    else:
                        sp = getattr(req_state, "sampling_params", None)
                        stop_token_ids = set()
                        if sp is not None:
                            if hasattr(sp, "stop_token_ids") and sp.stop_token_ids:
                                stop_token_ids.update(sp.stop_token_ids)
                            if getattr(sp, "stop_token_id", None) is not None:
                                stop_token_ids.add(sp.stop_token_id)
                        if getattr(self, "_stage_id", None) == 0:
                            stop_token_ids.update({151643, 151645})
                        elif getattr(self, "_stage_id", None) == 1:
                            stop_token_ids.update({2150, 4198})

                        out_ids = getattr(req_state, "output_token_ids", []) or []
                        if out_ids and out_ids[-1] in stop_token_ids:
                            logger.info(
                                "[TPUOmniModelRunner] Detected stop token %s for req %s",
                                out_ids[-1],
                                rid,
                            )
                            finished_ids.add(rid)
                            req_state.is_finished = True
                        elif sp is not None and getattr(sp, "max_tokens", None) is not None:
                            if len(out_ids) >= sp.max_tokens:
                                logger.info(
                                    "[TPUOmniModelRunner] Max tokens (%d) reached for req %s",
                                    sp.max_tokens,
                                    rid,
                                )
                                finished_ids.add(rid)
                                req_state.is_finished = True

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
                logger.debug("[TPUOmniModelRunner] attach_omni_connector_output error in sample_tokens: %s", e)

        return output
