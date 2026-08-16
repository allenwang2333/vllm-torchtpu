# SPDX-License-Identifier: Apache-2.0
"""JAX entry point for the fused expert-parallel MoE kernel.

One shard_map'd Pallas program per MoE layer replaces the DP all-gather, the
rank-local permute/GMM/GMM/combine, and the DP reduce-scatter: each rank brings
its own token rows, the kernel pulls the rows routed to its local experts off
every peer, runs both GMMs, and DMAs each result row back to the rank that owns
that token. Routing (scoring, top-k, renormalize) happens inside the kernel
from the raw router logits.

The kernel reads the weights in the layout the GMM path already stores
(``w13 [E, H, 2F]``, gate|up concatenated on the last axis), so both MoE paths
share one set of loaded weights and neither needs a restaging pass at load.

The torch side is ``layers/vllm/fused_moe_ep.py``.
"""

# No `from __future__ import annotations` here: torch_tpu's bridge inspects
# these annotations by identity (`annotation is jax.Array`,
# pallas.py:_infer_static_argnums), and PEP 563 would hand it strings -- every
# tensor argument would then be classified static.
from typing import Optional

import jax

from vllm_torchtpu.kernels.fused_moe.v1.kernel import fused_ep_moe

# collective_id registry (one id per distinct sync pattern; two kernels sharing
# an id share a barrier semaphore, and the sync patterns are not compatible):
#   0     kernels/collectives/all_gather_matmul.py -- 2-neighbour ring, no caller
#   7     kernels/fused_moe/v1/kernel.py           -- all-peer barrier on the EP axis
#   11-13 tests/kernels/experimental/pcp_streaming_rpa/*
# The production PCP streaming RPA and GDN PCP kernels use no barrier semaphore.
FUSED_EP_COLLECTIVE_ID = 7


def fused_ep_moe_func(
    tokens: jax.Array,
    gating_output: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: Optional[jax.Array],
    w2_scale: Optional[jax.Array],
    *,
    mesh: "jax.sharding.Mesh",
    ep_axis_name: str,
    top_k: int,
    renormalize: bool,
    activation: str,
    scoring_fn: str,
    subc_quant_w1_sz: Optional[int],
    subc_quant_w2_sz: Optional[int],
    blocks: Optional[tuple] = None,
    collective_id: int = FUSED_EP_COLLECTIVE_ID,
    # The GMM path's weight layout, read in place: see the module docstring.
    w13_concat_layout: bool = True,
) -> jax.Array:
    """Run one MoE layer over the EP mesh; see the module docstring.

    Shapes are *global* (the torch bridge multiplies each local shape up by the
    partition spec): ``tokens`` ``[T, H]`` and ``gating_output`` ``[T, E]``
    sharded over the EP axis by token, ``w1`` ``[E, H, 2F]`` (gate|up
    concatenated, i.e. the GMM path's own layout) / ``w2`` ``[E, F, H]`` sharded
    by expert. Output is ``[T, H]``, sharded by token.
    """
    block_names = ("bt", "bf", "bd1", "bd2", "btc", "bfc", "bd1c", "bd2c")
    block_kwargs = dict(zip(block_names, blocks or ()))
    return fused_ep_moe(
        mesh=mesh,
        tokens=tokens,
        w1=w1,
        w2=w2,
        gating_output=gating_output,
        top_k=top_k,
        renormalize_topk_logits=renormalize,
        act_fn=activation,
        scoring_fn=scoring_fn,
        subc_quant_w1_sz=subc_quant_w1_sz,
        subc_quant_w2_sz=subc_quant_w2_sz,
        w13_concat_layout=w13_concat_layout,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        ep_axis_name=ep_axis_name,
        collective_id=collective_id,
        **block_kwargs,
    )
