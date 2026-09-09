# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU Performance Observability & Tracing Suite.

Provides structured console logging and native Perfetto trace annotations
for the 6 core TPU inference metrics:
  1. Batch stats: phase, batch_size, num_prefill, num_decode, actual_tokens,
     bucketed_tokens, padding_tokens, avg/max_context_len (batch utilization)
  2. Expert load: layer_id, tokens_per_expert[256], active_experts, max/mean/p95 (hot expert)
  3. EP rank load: layer_id, tokens_per_ep_rank[32], max/mean, p95/mean, CV (rank straggler)
  4. GMM shape: expert M=num_tokens, histogram (tiny GMM detection)
  5. Collective semantic tags: layer, phase, collective_type, tensor_shape,
     bytes, group_size, logical_role (XProf communication attribution)
  6. HBM checkpoints: used/free/limit per rank (attention replication & memory budgeting)
"""

from contextlib import contextmanager
import inspect
import json
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.tracing.annotation import TraceAnnotation

logger = logging.getLogger("vllm_torchtpu.observability")


def is_observability_enabled() -> bool:
    """Check if observability logging and tracing are enabled."""
    val = os.getenv("TPU_OBSERVABILITY_ENABLE", "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    if val in ("1", "true", "yes", "on"):
        return True
    # Auto-enable if Perfetto, XProf, or torch profiler is active
    if os.getenv("ENABLE_PERFETTO", "").lower() == "true":
        return True
    if os.getenv("ENABLE_XPROF", "").lower() == "true":
        return True
    if os.getenv("TPU_PJRT_EVENT_TRACE_ENABLE", "") == "1":
        return True
    return True


def dump_every_n_steps() -> int:
    """Frequency of post-step observability dumps (default: 1 for initial steps, 10 for steady state)."""
    return int(os.getenv("TPU_OBSERVABILITY_DUMP_EVERY", "1"))


# -----------------------------------------------------------------------------
# Global Execution Context (Phase, Layer, Rank)
# -----------------------------------------------------------------------------
_THREAD_LOCAL = threading.local()


def set_current_phase(phase: str) -> None:
    """Set the current inference phase ('prefill', 'decode', 'balanced', etc.)."""
    _THREAD_LOCAL.current_phase = phase


def get_current_phase() -> str:
    """Get the current inference phase."""
    return getattr(_THREAD_LOCAL, "current_phase", "unknown")


def set_current_layer(layer_id: Any) -> None:
    """Set the current active layer identifier."""
    _THREAD_LOCAL.current_layer = str(layer_id)


def get_current_layer() -> str:
    """Get the current active layer identifier."""
    return getattr(_THREAD_LOCAL, "current_layer", "unknown")


def get_worker_rank() -> int:
    """Return the global worker rank."""
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    try:
        return int(jax.process_index())
    except Exception:
        pass
    for env_var in ("RANK", "TPU_WORKER_ID", "RAY_TPU_WORKER_ID"):
        val = os.getenv(env_var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


# -----------------------------------------------------------------------------
# 1. Batch Stats (Batch Utilization)
# -----------------------------------------------------------------------------
def log_batch_stats(
    phase: str,
    batch_size: int,
    num_prefill: int,
    num_decode: int,
    actual_tokens: int,
    bucketed_tokens: int,
    padding_tokens: int,
    avg_context_len: float,
    max_context_len: int,
) -> None:
    """Log batch composition stats and record a Perfetto trace event.

    Answers: Is batch utilization low? (actual vs bucketed padding overhead)
    """
    if not is_observability_enabled():
        return

    utilization_pct = (actual_tokens / max(1, bucketed_tokens)) * 100.0

    # Structured log
    logger.info(
        "[BATCH_STATS] phase=%s, batch_size=%d, num_prefill=%d, num_decode=%d, "
        "actual_tokens=%d, bucketed_tokens=%d, padding_tokens=%d, "
        "avg_context_len=%.1f, max_context_len=%d, batch_utilization=%.1f%%",
        phase,
        batch_size,
        num_prefill,
        num_decode,
        actual_tokens,
        bucketed_tokens,
        padding_tokens,
        avg_context_len,
        max_context_len,
        utilization_pct,
    )

    # Perfetto Trace Annotation
    with TraceAnnotation(
        "BatchStats",
        phase=phase,
        batch_size=batch_size,
        num_prefill=num_prefill,
        num_decode=num_decode,
        actual_tokens=actual_tokens,
        bucketed_tokens=bucketed_tokens,
        padding_tokens=padding_tokens,
        avg_context_len=round(avg_context_len, 1),
        max_context_len=max_context_len,
        batch_utilization_pct=round(utilization_pct, 1),
    ):
        pass


# -----------------------------------------------------------------------------
# 2. Expert Load (Hot Expert Detection)
# -----------------------------------------------------------------------------
def log_expert_load(
    layer_id: Any,
    tokens_per_expert: Union[List[int], np.ndarray, torch.Tensor],
    max_experts_to_log: int = 256,
) -> Dict[str, Any]:
    """Log per-layer expert load distribution and record Perfetto trace event.

    Answers: Are there hot experts causing load skew?
    """
    if not is_observability_enabled():
        return {}

    if isinstance(tokens_per_expert, torch.Tensor):
        tokens_arr = tokens_per_expert.detach().cpu().numpy().astype(np.int64)
    else:
        tokens_arr = np.asarray(tokens_per_expert, dtype=np.int64)

    total_experts = len(tokens_arr)
    active_experts = int(np.count_nonzero(tokens_arr))
    max_tokens = int(np.max(tokens_arr)) if total_experts > 0 else 0
    mean_tokens = float(np.mean(tokens_arr)) if total_experts > 0 else 0.0
    p95_tokens = float(np.percentile(tokens_arr, 95)) if total_experts > 0 else 0.0
    hot_expert_ratio = max_tokens / (mean_tokens + 1e-6)

    # Truncate array representation for readable logs if needed
    expert_slice = tokens_arr[:max_experts_to_log].tolist()

    logger.info(
        "[EXPERT_LOAD] layer=%s, active_experts=%d/%d, max=%d, mean=%.1f, p95=%.1f, "
        "hot_ratio=%.2fx, tokens_per_expert=%s",
        str(layer_id),
        active_experts,
        total_experts,
        max_tokens,
        mean_tokens,
        p95_tokens,
        hot_expert_ratio,
        expert_slice,
    )

    with TraceAnnotation(
        "ExpertLoad",
        layer=str(layer_id),
        active_experts=active_experts,
        total_experts=total_experts,
        max_tokens=max_tokens,
        mean_tokens=round(mean_tokens, 1),
        p95_tokens=round(p95_tokens, 1),
        hot_ratio=round(hot_expert_ratio, 2),
    ):
        pass

    return {
        "layer_id": str(layer_id),
        "tokens_per_expert": tokens_arr,
        "active_experts": active_experts,
        "max_tokens": max_tokens,
        "mean_tokens": mean_tokens,
        "p95_tokens": p95_tokens,
        "hot_ratio": hot_expert_ratio,
    }


# -----------------------------------------------------------------------------
# 3. EP Rank Load (Rank Straggler Detection)
# -----------------------------------------------------------------------------
def log_ep_rank_load(
    layer_id: Any,
    tokens_per_expert_or_rank: Union[List[int], np.ndarray, torch.Tensor],
    num_ep_ranks: int = 32,
) -> Dict[str, Any]:
    """Calculate and log EP rank load skew, max/mean, p95/mean, and CV.

    Answers: Are there rank stragglers holding back the DP/EP lockstep?
    """
    if not is_observability_enabled():
        return {}

    if isinstance(tokens_per_expert_or_rank, torch.Tensor):
        arr = tokens_per_expert_or_rank.detach().cpu().numpy().astype(np.float64)
    else:
        arr = np.asarray(tokens_per_expert_or_rank, dtype=np.float64)

    # If tokens_per_expert was passed instead of tokens_per_ep_rank, aggregate by rank
    if len(arr) != num_ep_ranks and len(arr) > num_ep_ranks:
        experts_per_rank = len(arr) // num_ep_ranks
        tokens_per_ep_rank = np.array([
            np.sum(arr[r * experts_per_rank:(r + 1) * experts_per_rank])
            for r in range(num_ep_ranks)
        ], dtype=np.float64)
    else:
        tokens_per_ep_rank = arr[:num_ep_ranks]

    actual_ep_ranks = len(tokens_per_ep_rank)
    mean_tokens = float(np.mean(tokens_per_ep_rank)) if actual_ep_ranks > 0 else 0.0
    max_tokens = float(np.max(tokens_per_ep_rank)) if actual_ep_ranks > 0 else 0.0
    p95_tokens = float(np.percentile(tokens_per_ep_rank, 95)) if actual_ep_ranks > 0 else 0.0
    std_tokens = float(np.std(tokens_per_ep_rank)) if actual_ep_ranks > 0 else 0.0

    max_over_mean = max_tokens / (mean_tokens + 1e-6)
    p95_over_mean = p95_tokens / (mean_tokens + 1e-6)
    cv = std_tokens / (mean_tokens + 1e-6)  # Coefficient of Variation
    straggler_rank = int(np.argmax(tokens_per_ep_rank)) if actual_ep_ranks > 0 else 0

    logger.info(
        "[EP_RANK_LOAD] layer=%s, ep_ranks=%d, max/mean=%.2f, p95/mean=%.2f, "
        "CV=%.3f, straggler_rank=%d, tokens_per_ep_rank=%s",
        str(layer_id),
        actual_ep_ranks,
        max_over_mean,
        p95_over_mean,
        cv,
        straggler_rank,
        tokens_per_ep_rank.astype(int).tolist(),
    )

    with TraceAnnotation(
        "EPRankLoad",
        layer=str(layer_id),
        ep_ranks=actual_ep_ranks,
        max_over_mean=round(max_over_mean, 2),
        p95_over_mean=round(p95_over_mean, 2),
        CV=round(cv, 3),
        straggler_rank=straggler_rank,
    ):
        pass

    return {
        "layer_id": str(layer_id),
        "tokens_per_ep_rank": tokens_per_ep_rank.astype(int).tolist(),
        "max_over_mean": max_over_mean,
        "p95_over_mean": p95_over_mean,
        "cv": cv,
        "straggler_rank": straggler_rank,
    }


# -----------------------------------------------------------------------------
# 4. GMM Shape (Tiny GMM Histogram)
# -----------------------------------------------------------------------------
def log_gmm_shape(
    layer_id: Any,
    tokens_per_expert: Union[List[int], np.ndarray, torch.Tensor],
) -> Dict[str, Any]:
    """Calculate and log expert token counts M and GMM shape histogram.

    Answers: Are there large numbers of tiny GMM matrix multiplications?
    """
    if not is_observability_enabled():
        return {}

    if isinstance(tokens_per_expert, torch.Tensor):
        tokens_arr = tokens_per_expert.detach().cpu().numpy().astype(np.int64)
    else:
        tokens_arr = np.asarray(tokens_per_expert, dtype=np.int64)

    active_experts = int(np.count_nonzero(tokens_arr))

    # Binned histogram for TPU matrix unit efficiency evaluation
    m0 = int(np.sum(tokens_arr == 0))
    m1_16 = int(np.sum((tokens_arr >= 1) & (tokens_arr <= 16)))
    m17_32 = int(np.sum((tokens_arr >= 17) & (tokens_arr <= 32)))
    m33_64 = int(np.sum((tokens_arr >= 33) & (tokens_arr <= 64)))
    m65_128 = int(np.sum((tokens_arr >= 65) & (tokens_arr <= 128)))
    m129_256 = int(np.sum((tokens_arr >= 129) & (tokens_arr <= 256)))
    m_gt256 = int(np.sum(tokens_arr > 256))

    tiny_gmm_count = m1_16 + m17_32
    tiny_pct = (tiny_gmm_count / max(1, active_experts)) * 100.0

    histogram_dict = {
        "M=0": m0,
        "M:1-16": m1_16,
        "M:17-32": m17_32,
        "M:33-64": m33_64,
        "M:65-128": m65_128,
        "M:129-256": m129_256,
        "M:>256": m_gt256,
    }

    logger.info(
        "[GMM_SHAPE] layer=%s, active_experts=%d, tiny_gmms=%d (%.1f%%), "
        "histogram=%s, M_tokens=%s",
        str(layer_id),
        active_experts,
        tiny_gmm_count,
        tiny_pct,
        json.dumps(histogram_dict),
        tokens_arr.tolist(),
    )

    with TraceAnnotation(
        "GMMShape",
        layer=str(layer_id),
        active_experts=active_experts,
        tiny_gmms=tiny_gmm_count,
        tiny_pct=round(tiny_pct, 1),
        m0=m0,
        m1_16=m1_16,
        m17_32=m17_32,
        m33_64=m33_64,
        m65_128=m65_128,
        m129_256=m129_256,
        m_gt256=m_gt256,
    ):
        pass

    return {
        "layer_id": str(layer_id),
        "tiny_gmm_count": tiny_gmm_count,
        "tiny_pct": tiny_pct,
        "histogram": histogram_dict,
    }


# -----------------------------------------------------------------------------
# 5. Collective Semantic Tags (XProf & Perfetto Communication Attribution)
# -----------------------------------------------------------------------------
@contextmanager
def trace_collective(
    collective_type: str = "unknown",
    tensor: Optional[torch.Tensor] = None,
    layer: Any = None,
    phase: Optional[str] = None,
    tensor_shape: Optional[Union[Tuple[int, ...], List[int]]] = None,
    element_size: int = 2,
    group_size: int = 1,
    logical_role: str = "unknown",
    **kwargs: Any,
):
    """Context manager wrapping collective communication with rich semantic tags.

    Attributes:
      layer: layer index or name
      phase: 'prefill', 'decode', etc.
      collective_type: 'AllGather', 'ReduceScatter', 'AllReduce', 'AllToAll'
      tensor_shape: shape of tensor being communicated
      bytes: size in bytes
      group_size: world size of communication group
      logical_role: 'moe_dispatch_hidden_states', 'moe_combine_output', etc.

    Answers: Exactly which model layer, phase, and logical operation generated
    the communication slice in XProf / Perfetto?
    """
    if not is_observability_enabled():
        yield
        return

    curr_layer = str(layer) if layer is not None else get_current_layer()
    curr_phase = str(phase) if phase is not None else get_current_phase()

    if tensor is not None:
        shape = list(tensor.shape)
        elem_size = tensor.element_size()
    elif tensor_shape is not None:
        shape = list(tensor_shape)
        elem_size = element_size
    else:
        shape = []
        elem_size = element_size

    nbytes = int(np.prod(shape)) * elem_size if len(shape) > 0 else 0
    mb = nbytes / (1024.0 * 1024.0)

    # Emit readable logger message
    logger.info(
        "[COLLECTIVE] layer=%s, phase=%s, type=%s, shape=%s, bytes=%d (%.2f MB), "
        "group_size=%d, role=%s",
        curr_layer,
        curr_phase,
        collective_type,
        shape,
        nbytes,
        mb,
        group_size,
        logical_role,
    )

    # Perfetto trace slice enclosing the collective operation
    trace_name = f"Collective:{collective_type}"
    with TraceAnnotation(
        trace_name,
        layer=curr_layer,
        phase=curr_phase,
        collective_type=collective_type,
        tensor_shape=str(shape),
        bytes=nbytes,
        mb=round(mb, 2),
        group_size=group_size,
        logical_role=logical_role,
    ):
        yield


# -----------------------------------------------------------------------------
# 6. HBM Checkpoints (Memory Budgeting & Headroom)
# -----------------------------------------------------------------------------
def log_hbm_checkpoint(
    checkpoint_name: str,
    rank: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Query and log HBM used/free/limit and utilization percentage per rank.

    Answers: Can attention replication and candidate configurations fit into HBM?
    """
    if not is_observability_enabled():
        return {}

    curr_rank = rank if rank is not None else get_worker_rank()

    try:
        if device is None and torch.cuda.is_available():
            device = torch.device(torch.cuda.current_device())
        free_bytes, limit_bytes = torch.accelerator.get_memory_info(device)
    except Exception as exc:
        try:
            free_bytes, limit_bytes = torch.cuda.mem_get_info(device)
        except Exception:
            logger.warning("Failed to query HBM info for checkpoint %s: %s", checkpoint_name, exc)
            return {}

    used_bytes = limit_bytes - free_bytes
    used_gb = used_bytes / (1024.0 ** 3)
    free_gb = free_bytes / (1024.0 ** 3)
    limit_gb = limit_bytes / (1024.0 ** 3)
    util_pct = (used_bytes / max(1, limit_bytes)) * 100.0

    logger.info(
        "[HBM_CHECKPOINT] checkpoint=%s, rank=%d, used=%.2f GB (%d B), "
        "free=%.2f GB (%d B), limit=%.2f GB (%d B), util=%.1f%%",
        checkpoint_name,
        curr_rank,
        used_gb,
        used_bytes,
        free_gb,
        free_bytes,
        limit_gb,
        limit_bytes,
        util_pct,
    )

    with TraceAnnotation(
        "HBMCheckpoint",
        checkpoint=checkpoint_name,
        rank=curr_rank,
        used_gb=round(used_gb, 2),
        free_gb=round(free_gb, 2),
        limit_gb=round(limit_gb, 2),
        util_pct=round(util_pct, 1),
    ):
        pass

    return {
        "checkpoint": checkpoint_name,
        "rank": curr_rank,
        "used_bytes": used_bytes,
        "free_bytes": free_bytes,
        "limit_bytes": limit_bytes,
        "used_gb": used_gb,
        "free_gb": free_gb,
        "limit_gb": limit_gb,
        "util_pct": util_pct,
    }


# -----------------------------------------------------------------------------
# In-Graph MoE Expert Load Accumulator (Zero Host-Device Sync During Forward)
# -----------------------------------------------------------------------------
_MOE_BUFFERS: Dict[str, torch.Tensor] = {}
_MOE_LOCK = threading.Lock()
_ACCUM_OP = None


def _accumulate_kernel(counts: jax.Array, topk_ids: jax.Array,
                       topk_weights: jax.Array) -> jax.Array:
    """In-graph accumulation of expert token counts without host sync."""
    num_experts = counts.shape[0]
    ids = topk_ids.reshape(-1).astype(jnp.int32)
    weights = topk_weights.reshape(-1)
    valid = (ids >= 0) & (ids < num_experts) & (weights > 0)
    safe_ids = jnp.where(valid, ids, 0)
    ones = jnp.where(valid, 1, 0).astype(counts.dtype)
    hist = jax.ops.segment_sum(
        ones,
        safe_ids,
        num_segments=num_experts,
        indices_are_sorted=False,
    )
    return counts + hist.astype(counts.dtype)


def _get_accum_op():
    global _ACCUM_OP
    if _ACCUM_OP is None:
        try:
            op = pallas.jax_op("pallas::expert_load_accum",
                               _accumulate_kernel,
                               donate_argnums=(0,))

            def _fake(counts, topk_ids, topk_weights):
                return torch.empty_like(counts)

            op.register_fake(_fake)
            _ACCUM_OP = op
        except Exception as e:
            logger.warning("Failed to initialize expert_load_accum custom op: %s", e)
            _ACCUM_OP = None
    return _ACCUM_OP


def register_layer_moe_buffer(layer, num_experts: int = 256,
                               device: Optional[torch.device] = None,
                               layer_name: Optional[str] = None) -> None:
    """Register persistent in-graph expert count buffer on an MoE layer."""
    if not is_observability_enabled():
        return
    name = layer_name or getattr(layer, "layer_name", None) or getattr(layer, "prefix", None) or f"layer_{len(_MOE_BUFFERS)}"
    if hasattr(layer, "_expert_load_counts") and getattr(layer, "_expert_load_counts") is not None:
        buf = getattr(layer, "_expert_load_counts")
        with _MOE_LOCK:
            _MOE_BUFFERS[name] = buf
        _get_accum_op()
        return

    if device is None:
        device = getattr(getattr(layer, "w13_weight", None), "device", None) or getattr(getattr(layer, "w2_weight", None), "device", None)
    if device is None:
        for p in layer.parameters():
            device = p.device
            break
    if device is None:
        for b in layer.buffers():
            device = b.device
            break
    if device is None:
        device = torch.device("tpu")

    buf = torch.zeros(num_experts, dtype=torch.int32, device=device)
    layer.register_buffer("_expert_load_counts", buf, persistent=False)
    with _MOE_LOCK:
        _MOE_BUFFERS[name] = buf
    _get_accum_op()


def record_expert_routing_in_graph(layer, topk_ids: torch.Tensor,
                                   topk_weights: torch.Tensor) -> None:
    """Record expert routing in the compiled graph."""
    if not is_observability_enabled():
        return
    buf = getattr(layer, "_expert_load_counts", None)
    if buf is None:
        return
    op = _ACCUM_OP
    if op is None:
        return
    new_counts = op(buf, topk_ids, topk_weights)
    buf.copy_(new_counts)


def dump_moe_observability(step_id: int, phase: str = "unknown") -> None:
    """Off-graph post-step dump for Logs 2, 3, 4 (Expert load, EP rank load, GMM shape)."""
    if not is_observability_enabled():
        return
    dump_every = dump_every_n_steps()
    if step_id > 5 and dump_every > 1 and step_id % dump_every != 0:
        return

    with _MOE_LOCK:
        items = list(_MOE_BUFFERS.items())
    if not items:
        return

    for name, buf in items:
        try:
            tokens_per_expert = buf.cpu().numpy()
        except Exception as exc:
            logger.warning("Failed to read expert load buffer for %s: %s", name, exc)
            continue

        # Emit Log 2, 3, 4
        log_expert_load(layer_id=name, tokens_per_expert=tokens_per_expert)
        log_ep_rank_load(layer_id=name, tokens_per_expert_or_rank=tokens_per_expert, num_ep_ranks=32)
        log_gmm_shape(layer_id=name, tokens_per_expert=tokens_per_expert)

        # Reset buffer for next step
        try:
            buf.zero_()
        except Exception as exc:
            pass

