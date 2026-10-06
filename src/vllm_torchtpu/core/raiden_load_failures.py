# SPDX-License-Identifier: Apache-2.0
"""Finish Raiden failed loads without releasing armed receive buffers early."""

from functools import wraps

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiKVConnectorWorkerMetadata,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import RequestStatus

from vllm_torchtpu.distributed.kv_transfer.raiden_load_failures import (
    RaidenLoadFailures,
    raiden_load_failure_feedback_enabled,
)


class _DecodeLoadFailures:
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.original_update = scheduler.update_from_output
        scheduler.update_from_output = self.update_from_output

    def update_from_output(self, scheduler_output, model_runner_output):
        kv_output = model_runner_output.kv_connector_output
        failed_req_ids: set[str] = set()
        if kv_output is not None:
            metadata = kv_output.kv_connector_worker_meta
            children = (
                metadata.metadata
                if isinstance(metadata, MultiKVConnectorWorkerMetadata)
                else (metadata,)
            )
            for child in children:
                if isinstance(child, RaidenLoadFailures):
                    failed_req_ids.update(child.failed_req_ids)
        # finish_requests preserves WAITING_FOR_REMOTE_KVS blocks until
        # finished_recving. Finish before consuming any stale model output.
        failed_requests = (
            self.scheduler.finish_requests(failed_req_ids, RequestStatus.FINISHED_ERROR)
            if failed_req_ids
            else []
        )
        outputs = self.original_update(scheduler_output, model_runner_output)
        for request in failed_requests:
            batch = outputs.setdefault(request.client_index, EngineCoreOutputs())
            batch.outputs.append(
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=FinishReason.ERROR,
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                    kv_transfer_params={
                        "kv_transfer_error": {
                            "status_code": 500,
                            "status_name": "model.KVTDecodeLoadFailure",
                            "status_message": (
                                "D-side failed to load KV cache data. "
                                "Category: internal."
                            ),
                        }
                    },
                )
            )
        return outputs


def patch_scheduler_init() -> None:
    """Install per-instance feedback only for the selected Raiden load mode."""
    if Scheduler.__dict__.get("_tpu_raiden_load_failures_patch", False):
        return
    original_init = Scheduler.__init__

    @wraps(original_init)
    def init(self, vllm_config, *args, **kwargs):
        original_init(self, vllm_config, *args, **kwargs)
        if raiden_load_failure_feedback_enabled(vllm_config):
            _DecodeLoadFailures(self)

    Scheduler.__init__ = init
    Scheduler._tpu_raiden_load_failures_patch = True
