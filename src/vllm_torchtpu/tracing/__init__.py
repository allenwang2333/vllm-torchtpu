# Copyright 2026 Google LLC
"""PyTorch Kineto and TorchTPU profiler tracing module."""
from vllm_torchtpu.tracing.annotation import (TraceAnnotation,
                                              is_trace_annotation_enabled)
from vllm_torchtpu.tracing.observability import (
    dump_moe_observability,
    get_current_layer,
    get_current_phase,
    is_observability_enabled,
    log_batch_stats,
    log_ep_rank_load,
    log_expert_load,
    log_gmm_shape,
    log_hbm_checkpoint,
    record_expert_routing_in_graph,
    register_layer_moe_buffer,
    set_current_layer,
    set_current_phase,
    trace_collective,
)
from vllm_torchtpu.tracing.utils import (extract_kv_lens_for_tracing,
                                         extract_request_ids_for_tracing,
                                         trim_request_id_suffix)

__all__ = [
    "TraceAnnotation",
    "is_trace_annotation_enabled",
    "extract_kv_lens_for_tracing",
    "extract_request_ids_for_tracing",
    "trim_request_id_suffix",
    "log_batch_stats",
    "log_expert_load",
    "log_ep_rank_load",
    "log_gmm_shape",
    "trace_collective",
    "log_hbm_checkpoint",
    "set_current_phase",
    "get_current_phase",
    "set_current_layer",
    "get_current_layer",
    "register_layer_moe_buffer",
    "record_expert_routing_in_graph",
    "dump_moe_observability",
    "is_observability_enabled",
]

