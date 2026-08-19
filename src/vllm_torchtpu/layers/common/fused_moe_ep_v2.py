# SPDX-License-Identifier: Apache-2.0
"""JAX entry point for the **v2** fused expert-parallel MoE kernel.

Same contract as ``fused_moe_ep.py`` (v1) -- six positional operands, global
shapes, one ``[T, H]`` output sharded by token -- over a different program.
v1 is token-stationary: an outer loop over token blocks re-streams all of this
rank's expert weights for each one. v2 is weight-stationary: it plans the whole
transport once in XLA, visits only the experts that actually received rows, and
holds each expert's weights across all of that expert's row tiles with a
triple-buffered prefetch. At the 16384-token prefill point that is a 16x cut in
expert-weight traffic, which is why v2 is the prefill kernel and v1 was not.

Two things the v2 kernel wants that our loader does not already produce, both
handled here rather than at load time so that the GMM path keeps reading the
same tensors (one set of weights, two paths):

* **scales as exactly ``[E, 2F]`` / ``[E, H]``.** ``process_weights_after_loading``
  leaves the loader's singleton axes on (``[E, 1, 1, 2F]``), and the kernel
  layer *refuses* those rather than reshaping them, on the grounds that a
  reshape would then stand between the parameter and the call on every call.
  Dropping a singleton axis is a metadata-only view that XLA folds away.
* **f32 router logits.** v2 softmaxes and selects from what it is given; scoring
  in bf16 disagrees with ``torch.topk`` on a few tokens per thousand at
  serving-scale logit spreads (measured for v1).

The torch side is ``layers/vllm/fused_moe_ep.py``.
"""

# No `from __future__ import annotations` here: torch_tpu's bridge inspects
# these annotations by identity (`annotation is jax.Array`,
# pallas.py:_infer_static_argnums), and PEP 563 would hand it strings -- every
# tensor argument would then be classified static.
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v2 import (ROWBLK, WeightFormat,
                                                capacity_slab_rows,
                                                fused_ep_moe_v2,
                                                ragged_stride_bound)

# collective_id registry (one id per distinct sync pattern; two kernels sharing
# an id share a barrier semaphore, and the sync patterns are not compatible):
#   0     kernels/collectives/all_gather_matmul.py -- 2-neighbour ring, no caller
#   6     kernels/fused_moe/v2/kernel.py           -- all-pairs barrier, EP axis
#   7     kernels/fused_moe/v1/kernel.py           -- all-peer barrier on the EP axis
#   11-13 tests/kernels/experimental/pcp_streaming_rpa/*
# Upstream hard-codes 0 for v2, which is taken here. v1 and v2 never run in one
# process (the version switch picks one), but they get distinct ids anyway: a
# shared id is a silent cross-kernel barrier, not an error.
FUSED_EP_V2_COLLECTIVE_ID = 6

# The tile height the kernel computes its rows in, and the unit its no-drop slab
# rounds to. Not a free parameter: AOT-gated on this host, 256 hits a Mosaic
# vector-layout bug and 512 exceeds VMEM by the kernel's own accounting, so 128
# is the only usable value and is what upstream serves (`_TILE_M`).
CAPACITY = 128

# SMEM the kernel's scalar tables need *besides* the token-gather table, plus
# Mosaic's own scalar working set. Measured on tpu7x by holding the token count
# and growing the slab until the allocator refused: at a 262,144-row slab
# (1024 kB of table) it reports "Used 1.22M of 1.00M smem. Exceeded smem
# capacity by 230.7K", so everything else is ~231 kB. Rounded up.
_SMEM_OVERHEAD_BYTES = 236 * 1024

# Skew margin for the per-shard slab, and it is ON by default -- the drop-free
# slab is what makes the 16384-token bucket cost 15,620 us/layer against the GMM
# path's 3,264, and sizing it takes the MoE program's XLA temp from 919.9 MB to
# 490.4 MB at that point. The default lives HERE, not only in the serving
# adapter's env read, so that a caller driving the kernel straight from JAX gets
# the shipped program rather than the pre-fix one. Pass 0 for the drop-free
# slab. The derivation and the three bounds that justify 3.0 are in
# layers/vllm/fused_moe_ep.capacity_skew.
CAPACITY_SKEW = 3.0


def _slab_rows(tokens: int, top_k: int, num_experts: int, ep: int,
               skew: float) -> int:
    """The slab this call will actually allocate."""
    if skew > 0:
        return capacity_slab_rows(tokens, top_k, num_experts, ep, CAPACITY,
                                  skew)
    return ragged_stride_bound(tokens, top_k, num_experts, CAPACITY)


def max_node_tokens(top_k: int, num_experts: int, smem_bytes: int) -> int:
    """Most node-wide tokens whose routing plan still fits in one chip's SMEM.

    ``shard_token_gather`` returns one int32 per row of the drop-free slab and
    is a *scalar-prefetch* operand, so it lives in SMEM rather than HBM, and the
    slab is sized for every routed pair in the job landing on one shard. That
    makes SMEM -- not VMEM, not HBM -- the binding constraint on the token
    bucket: ~20,000 node-wide tokens at Qwen3.5's top-10 over 512 experts,
    against a shipped full-prefill point of 16,384.
    """
    rows = (smem_bytes - _SMEM_OVERHEAD_BYTES) // 4
    return max(0, (rows - (ROWBLK - 1) * num_experts - CAPACITY) // top_k)


def _check_smem(tokens: int, top_k: int, num_experts: int, ep: int,
                skew: float) -> None:
    """Refuse a bucket the scalar tables cannot fit, by name.

    Mosaic's own message for this is "Ran out of memory in memory space smem",
    which names neither the token count that caused it nor the setting that
    controls it, and it arrives at the first compile of a serving run.
    """
    smem = int(pltpu.get_tpu_info().smem_capacity_bytes)
    budget = smem - _SMEM_OVERHEAD_BYTES
    stride = _slab_rows(tokens, top_k, num_experts, ep, skew)
    if stride * 4 > budget:
        limit = max_node_tokens(top_k, num_experts, smem)
        raise ValueError(
            f"the fused EP MoE v2 kernel cannot take {tokens} node-wide "
            f"tokens: its drop-free slab is {stride} rows and the token-gather "
            f"table is one scalar-prefetch int32 per row, which is "
            f"{stride * 4 / 1024:.0f} kB of this chip's {smem / 1024:.0f} kB "
            f"SMEM against a ~{_SMEM_OVERHEAD_BYTES / 1024:.0f} kB budget for "
            f"everything else. The ceiling at top_k={top_k} over "
            f"{num_experts} experts is {limit} tokens on the drop-free slab; "
            "lower max_num_batched_tokens, size the slab with "
            "MOE_FUSED_EP_V2_CAPACITY_SKEW, or cap the kernel's engagement "
            "band with MOE_FUSED_EP_KERNEL_MAX_TOKENS.")


def _scales_2d(scale: Optional[jax.Array]) -> Optional[jax.Array]:
    """``[E, 1, 1, N]`` (the loader's shape) -> ``[E, N]`` (the kernel's)."""
    if scale is None:
        return None
    if scale.ndim == 2:
        return scale
    return scale.reshape(scale.shape[0], scale.shape[-1])


def fused_ep_moe_v2_func(
    tokens: jax.Array,
    gating_output: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: Optional[jax.Array],
    w2_scale: Optional[jax.Array],
    ep_ids: Optional[jax.Array] = None,
    *,
    mesh: "jax.sharding.Mesh",
    top_k: int,
    renormalize: bool,
    activation: str,
    capacity: int = CAPACITY,
    collective_id: int = FUSED_EP_V2_COLLECTIVE_ID,
    sharded_plan: bool = False,
    capacity_skew: Optional[float] = None,
    mesh_ep_ranks: Optional[tuple] = None,
    ep_size: Optional[int] = None,
) -> jax.Array:
    """Run one MoE layer over the EP mesh through the v2 kernel.

    Shapes are *global* (the torch bridge multiplies each local shape up by the
    partition spec): ``tokens`` ``[T, H]`` and ``gating_output`` ``[T, E]``
    sharded over the EP axis by token, ``w1`` ``[E, H, 2F]`` (gate|up
    concatenated, i.e. the GMM path's own layout, which is also v2's) and ``w2``
    ``[E, F, H]`` sharded by expert. Output is ``[T, H]``, sharded by token.

    ``ep_ids`` is ``arange(ep)`` sharded over the same axis, so each shard reads
    its own index out of a ``[1]`` operand. **Any caller that goes through
    ``jax.export`` -- which is every caller through the torch bridge -- has to
    supply it**, because a constant built inside the traced function is a
    sharded iota and the SPMD partitioner implements one of those by emitting
    the `partition-id` that the exported module is then refused for. Left None
    it is built here, which is correct and fast for a harness driving the
    kernel directly from jax and is what the standalone jax benchmark does.
    """
    skew = CAPACITY_SKEW if capacity_skew is None else capacity_skew
    ep = ep_size if ep_size is not None else mesh.shape[mesh.axis_names[-1]]
    _check_smem(tokens.shape[0], top_k, w1.shape[0], ep, skew)
    return fused_ep_moe_v2(
        tokens,
        w1,
        w2,
        _scales_2d(w1_scale),
        _scales_2d(w2_scale),
        gating_output.astype(jnp.float32),
        ep_ids=ep_ids,
        mesh_ep_ranks=mesh_ep_ranks,
        topk=top_k,
        renormalize=renormalize,
        mesh=mesh,
        capacity=capacity,
        weight_format=WeightFormat.FP8,
        act_fn=activation,
        collective_id=collective_id,
        sharded_plan=sharded_plan,
        capacity_skew=skew or None,
    )
