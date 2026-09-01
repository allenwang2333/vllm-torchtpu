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
"""DP all-to-all MoE dispatch/combine around the fused GMM (Batch U).

The default DP=4 EP path brackets every MoE layer with three collectives on
the 4-member DP group: an all-gather of the (latent) hidden states, an
all-gather of the router logits, and a reduce-scatter of the expert output --
measured 3.38 ms/step on K3 (92 MoE layers). This module replaces them with
two all-to-alls:

- dispatch: route this rank's own tokens locally (the gate is replicated and
  the contract shift to pre-dispatch routing is the same one
  `pipelined_fused_moe_gmm` already makes), then send each token's packed row
  (hidden + topk weights + topk ids, one i32 payload so the exchange is a
  single collective) to every DP rank that owns at least one of its experts.
  Destination rank comes from the chip-block table (`dp_mesh`); non-local
  expert slots are masked by the GMM's existing ``experts_start`` logic, and
  padding slots carry ``topk_ids == -1``, the convention the GMM already
  uses.
- combine: each rank's GMM output row is the token's partial sum over the
  rank-local experts (weights already applied by the GMM); the reverse
  exchange returns every partial to the token's home rank, which index-adds
  them in destination order. The reduction order differs from the
  reduce-scatter it replaces, so the result is not bit-identical to the
  baseline path.

Two modes behind ``TPU_MOE_DP_A2A``:

- ``ragged``: sort by destination, exchange with `jax.lax.ragged_all_to_all`
  (in production in this tree at `gdn_attention.py`'s PCP exchange) with
  data-dependent offsets/sizes; moves only routed rows.
- ``padded``: every destination bucket is padded to the full local token
  count and exchanged with `jax.lax.all_to_all` (equal split, static sizes).
  The fallback if the ragged collective misbehaves -- at K3's topk=16 nearly
  every token touches all 4 DP ranks anyway, so the byte difference is small
  and the structural win (3 collectives -> 2, logits all-gather gone) is the
  same.

Both ops run as SPMD programs over an op-local 4-device DP mesh
(`dp_mesh.build_dp_mesh`), so the exchange is one custom op in the outer
graph per layer per direction; nothing per-layer lands in the model's own
compiled program.
"""

import functools
import inspect
from typing import Any, Optional

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.dp_mesh import (DP_AXIS_NAME, TP_AXIS_NAME,
                                               build_chip_block_dest_table,
                                               build_dp_mesh)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
    _PcpStreamingJaxCallable)
from vllm_torchtpu.layers.vllm.fused_moe import fused_moe_gmm
from vllm_torchtpu.layers.vllm.pipelined_fused_moe import \
    enable_pipelined_collective_and_compute
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def dp_a2a_requested() -> bool:
    """Whether the operator asked for the DP a2a exchange (arming is separate)."""
    return (envs.TPU_MOE_DP_A2A or "") in ("ragged", "padded")


def _pack16(x: jax.Array) -> jax.Array:
    """Pack a 16-bit dtype's rows into i32 lanes (little-endian pairs)."""
    u = jax.lax.bitcast_convert_type(x, jnp.uint16).astype(jnp.uint32)
    packed = u[:, 0::2] | (u[:, 1::2] << 16)
    return jax.lax.bitcast_convert_type(packed, jnp.int32)


def _unpack16(v: jax.Array, dtype: jnp.dtype) -> jax.Array:
    u = jax.lax.bitcast_convert_type(v, jnp.uint32)
    lo = (u & 0xFFFF).astype(jnp.uint16)
    hi = (u >> 16).astype(jnp.uint16)
    pair = jnp.stack([lo, hi], axis=-1).reshape(v.shape[0], -1)
    return jax.lax.bitcast_convert_type(pair, dtype)


def _pack_rows(hidden: jax.Array, topk_w: jax.Array,
               topk_ids: jax.Array) -> jax.Array:
    """Pack one (hidden, weights, ids) row set into a single i32 payload.

    Layout per row: ``H//2`` i32 lanes of packed 16-bit hidden, then ``K//2``
    lanes of packed 16-bit weights, then ``K`` lanes of i32 expert ids. One
    packed payload keeps the dispatch exchange a single collective: three
    separate exchanges would pay the ICI latency floor three times per layer.
    """
    return jnp.concatenate(
        [_pack16(hidden), _pack16(topk_w), topk_ids], axis=-1)


def _unpack_rows(packed: jax.Array, hidden_size: int, topk: int,
                 dtype: jnp.dtype) -> tuple[jax.Array, jax.Array, jax.Array]:
    h_end = hidden_size // 2
    w_end = h_end + topk // 2
    hidden = _unpack16(packed[:, :h_end], dtype)
    topk_w = _unpack16(packed[:, h_end:w_end], dtype)
    return hidden, topk_w, packed[:, w_end:]


def _fill_rows(num_rows: int, row_width: int, topk: int) -> jax.Array:
    """Receive buffer: zero hidden/weight lanes, ``-1`` expert-id lanes.

    Rows never written by the exchange must decode to masked-out slots
    (``topk_ids == -1``, the GMM's dump-bucket convention) with finite
    (zero) hidden/weight values: the SparseCore one-hot combine multiplies
    masked slots by a zero coefficient, and 0 * NaN would be NaN.
    """
    out = jnp.zeros((num_rows, row_width), jnp.int32)
    return out.at[:, -topk:].set(jnp.int32(-1))


def _exclusive_cumsum(x: jax.Array) -> jax.Array:
    return jnp.cumsum(x) - x


def _dispatch_local(hidden: jax.Array, topk_w: jax.Array, topk_ids: jax.Array,
                    table: jax.Array, rank: jax.Array | None = None, *,
                    mode: str, dp_size: int, experts_per_block: int,
                    axis: str):
    """Shard-map body: local [T, ...] shards in, received [P*T, ...] out."""
    num_tokens, hidden_size = hidden.shape
    topk = topk_ids.shape[1]
    dp = dp_size

    if mode == "padded":
        packed = _pack_rows(hidden, topk_w, topk_ids)
        send = jnp.broadcast_to(packed[None, ...],
                                (dp, num_tokens, packed.shape[1]))
        recv = jax.lax.all_to_all(send,
                                  axis,
                                  split_axis=0,
                                  concat_axis=0,
                                  tiled=True)
        return _unpack_rows(recv.reshape(dp * num_tokens, -1), hidden_size,
                            topk, hidden.dtype)

    # Ragged: send each token to every DP rank owning >=1 of its experts.
    # Slots the GMM would mask anyway (invalid id, or zero weight from padded
    # tokens) are excluded here too, so padded tokens never leave home.
    valid = (topk_ids >= 0) & (topk_w != 0)
    num_experts = table.shape[0] * experts_per_block
    clipped = jnp.clip(topk_ids, 0, num_experts - 1)
    dest = table[clipped // experts_per_block]  # [T, K] mesh indices
    hits = jax.nn.one_hot(dest, dp, dtype=jnp.int32) * valid[..., None].astype(
        jnp.int32)
    dest_mask = hits.sum(axis=1) > 0  # [T, P]

    # Stable argsort over (token, dest) pairs keyed by destination groups the
    # send buffer into one contiguous run per destination; invalid pairs sort
    # into a dump region at the end that the sizes below never cover.
    keys = jnp.where(dest_mask,
                     jnp.arange(dp, dtype=jnp.int32)[None, :], dp).reshape(-1)
    order = jnp.argsort(keys, stable=True)
    tok_of_pair = (order // dp).astype(jnp.int32)
    send = _pack_rows(hidden[tok_of_pair], topk_w[tok_of_pair],
                      topk_ids[tok_of_pair])

    counts = dest_mask.astype(jnp.int32).sum(axis=0)  # [P] per destination
    send_off = _exclusive_cumsum(counts)
    # Receive-side counts are data owned by the senders; a [P]->[P,P]
    # all-gather of the count vector is one tiny collective per layer.
    all_counts = jax.lax.all_gather(counts, axis)  # row r = rank r's counts
    # The rank arrives as a runtime tensor, not lax.axis_index: axis_index
    # lowers to partitionId, which this stack's SPMD partitioning rejects
    # (same lesson as fused_moe_ep's _RANK_BUFFER).
    me = rank.reshape(())
    recv_counts = jnp.take(all_counts, me, axis=1)
    recv_off = _exclusive_cumsum(recv_counts)

    out = _fill_rows(dp * num_tokens, send.shape[1], topk)
    recv = jax.lax.ragged_all_to_all(send,
                                     out,
                                     send_off,
                                     counts,
                                     recv_off,
                                     recv_counts,
                                     axis_name=axis)
    recv_h, recv_w, recv_ids = _unpack_rows(recv, hidden_size, topk,
                                            hidden.dtype)
    # The layout vectors go out as [N, 1] rather than [N]: the torch_tpu
    # bridge's shape conversion does not handle a 1-D spec sharded over both
    # mesh axes, and a 2-D [N, 1] P('dp', 'tp') spec stays honest even if TP
    # lanes ever diverge.
    return (recv_h, recv_w, recv_ids, tok_of_pair[:, None],
            send_off[:, None], counts[:, None], recv_off[:, None],
            recv_counts[:, None])


def _combine_local(gmm_out: jax.Array, tok_of_pair: jax.Array,
                   send_off: jax.Array, send_sizes: jax.Array,
                   recv_off: jax.Array, recv_sizes: jax.Array, *, mode: str,
                   dp_size: int, axis: str) -> jax.Array:
    """Shard-map body: received-layout [P*T, H] partials -> local [T, H]."""
    dp = dp_size
    num_tokens = gmm_out.shape[0] // dp

    if mode == "padded":
        # Block s of the receive layout holds the partials of src s's tokens;
        # the symmetric all_to_all returns block d = partials from dest d.
        blocks = gmm_out.reshape(dp, num_tokens, -1)
        back = jax.lax.all_to_all(blocks,
                                  axis,
                                  split_axis=0,
                                  concat_axis=0,
                                  tiled=True)
        return back.sum(axis=0)

    tok_of_pair = tok_of_pair[:, 0]
    send_off = send_off[:, 0]
    send_sizes = send_sizes[:, 0]
    recv_off = recv_off[:, 0]
    recv_sizes = recv_sizes[:, 0]
    # Reverse the dispatch exchange: rows src j sent me live at recv_off[j],
    # and on the home rank they land back at that rank's dispatch send
    # offsets -- i.e. in the sorted (token, dest) pair order the home rank
    # itself built, so tok_of_pair[j] is the home token of row j.
    out = jnp.zeros_like(gmm_out)
    back = jax.lax.ragged_all_to_all(gmm_out,
                                     out,
                                     recv_off,
                                     recv_sizes,
                                     send_off,
                                     send_sizes,
                                     axis_name=axis)
    # Dump-region rows of `back` were never written (zeros), so index-adding
    # them onto real tokens is a no-op and no separate dump slot is needed.
    return jnp.zeros((num_tokens, gmm_out.shape[1]),
                     gmm_out.dtype).at[tok_of_pair].add(back)


class _DpA2aContext:
    """Process-wide a2a state: mesh, ops, and the chip-block dest table."""

    def __init__(self, mode: str, dispatch_op: Any, combine_op: Any,
                 dest_table: torch.Tensor, rank_buffer: torch.Tensor,
                 dp_size: int) -> None:
        self.mode = mode
        self.dispatch_op = dispatch_op
        self.combine_op = combine_op
        self.dest_table = dest_table
        self.rank_buffer = rank_buffer
        self.dp_size = dp_size


_CTX: _DpA2aContext | None = None


def get_dp_a2a_context() -> _DpA2aContext | None:
    """The armed context, or None (off, refused, or not yet built)."""
    return _CTX


def _dp_a2a_jax_op(name: str, fn: Any, *, mesh: Any,
                   input_partition_specs: tuple,
                   output_partition_specs: Any):
    """Register ``fn`` as a torch custom op over the op-local DP mesh.

    This is the PCP-streaming registration pattern
    (``register_custom_kernel`` + ``call_custom_kernel``): the exported jax
    program stays behind a custom-kernel boundary and keeps its own 4-device
    device assignment, instead of being inlined into the model's 32-partition
    module -- the stock ``pallas.jax_op``/``sharded_jax_op`` path inlines, and
    the shard_map collectives then lower to replica groups that neither cover
    nor address the outer program's partitions (XLA verifier rejection at
    model compile). Unlike ``pcp_streaming_jax_op`` this mirrors the output
    spec tree exactly, so a single-array output takes a bare spec.
    """
    from jax.sharding import NamedSharding
    from torch_tpu._internal.pallas import pallas as pallas_impl

    signature = inspect.signature(inspect.unwrap(fn), eval_str=True)
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    out_shardings = jax.tree.map(
        lambda s: NamedSharding(mesh, s),
        output_partition_specs,
        is_leaf=lambda s: isinstance(s, PartitionSpec))
    jit_fn = jax.jit(fn,
                     static_argnums=static_argnums,
                     out_shardings=out_shardings,
                     keep_unused=True)
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"{name}_{id(fn)}", [], {
            "static_argnums": static_argnums,
            "output_partition_specs": str(output_partition_specs),
        })
    wrapped_fn = _PcpStreamingJaxCallable(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=tuple(input_partition_specs),
        static_argnums=static_argnums,
        donate_argnums=(),
    )
    return torch.library.custom_op(name, wrapped_fn, mutates_args=())


def _build_dispatch_op(mesh: Any, mode: str, dp_size: int,
                       experts_per_block: int):
    sharded = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME)

    if mode == "padded":

        def dispatch(hidden: jax.Array, topk_w: jax.Array, topk_ids: jax.Array,
                     table: jax.Array
                     ) -> tuple[jax.Array, jax.Array, jax.Array]:
            mapped = jax.shard_map(
                functools.partial(_dispatch_local,
                                  mode=mode,
                                  dp_size=dp_size,
                                  experts_per_block=experts_per_block,
                                  axis=DP_AXIS_NAME),
                mesh=mesh,
                in_specs=(sharded, sharded, sharded, PartitionSpec()),
                out_specs=(sharded, sharded, sharded),
                check_vma=False,
            )
            return mapped(hidden, topk_w, topk_ids, table)

        op = _dp_a2a_jax_op(
            f"pallas::dp_a2a_moe_dispatch_{mode}",
            dispatch,
            mesh=mesh,
            input_partition_specs=(sharded, sharded, sharded, PartitionSpec()),
            output_partition_specs=(sharded, sharded, sharded),
        )

        def _fake(hidden, topk_w, topk_ids, table):
            t, h = hidden.shape
            k = topk_ids.shape[1]
            return (torch.empty((dp_size * t, h),
                                dtype=hidden.dtype,
                                device=hidden.device),
                    torch.empty((dp_size * t, k),
                                dtype=topk_w.dtype,
                                device=topk_w.device),
                    torch.empty((dp_size * t, k),
                                dtype=torch.int32,
                                device=topk_ids.device))

        op.register_fake(_fake)
        return op

    def dispatch(hidden: jax.Array, topk_w: jax.Array, topk_ids: jax.Array,
                 table: jax.Array, rank: jax.Array
                 ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array,
                            jax.Array, jax.Array, jax.Array, jax.Array]:
        mapped = jax.shard_map(
            functools.partial(_dispatch_local,
                              mode=mode,
                              dp_size=dp_size,
                              experts_per_block=experts_per_block,
                              axis=DP_AXIS_NAME),
            mesh=mesh,
            in_specs=(sharded, sharded, sharded, PartitionSpec(), sharded),
            out_specs=(sharded, ) * 8,
            check_vma=False,
        )
        return mapped(hidden, topk_w, topk_ids, table, rank)

    op = _dp_a2a_jax_op(
        f"pallas::dp_a2a_moe_dispatch_{mode}",
        dispatch,
        mesh=mesh,
        input_partition_specs=(sharded, sharded, sharded, PartitionSpec(),
                               sharded),
        output_partition_specs=(sharded, ) * 8,
    )

    def _fake(hidden, topk_w, topk_ids, table, rank):
        t, h = hidden.shape
        k = topk_ids.shape[1]
        pt = dp_size * t
        return (torch.empty((pt, h), dtype=hidden.dtype, device=hidden.device),
                torch.empty((pt, k), dtype=topk_w.dtype, device=topk_w.device),
                torch.empty((pt, k), dtype=torch.int32,
                            device=topk_ids.device),
                torch.empty((pt, 1), dtype=torch.int32,
                            device=topk_ids.device),
                *[
                    torch.empty(
                        (dp_size, 1), dtype=torch.int32,
                        device=topk_ids.device) for _ in range(4)
                ])

    op.register_fake(_fake)
    return op


def _build_combine_op(mesh: Any, mode: str, dp_size: int):
    sharded = PartitionSpec(DP_AXIS_NAME, TP_AXIS_NAME)

    if mode == "padded":

        def combine(gmm_out: jax.Array) -> jax.Array:
            mapped = jax.shard_map(
                functools.partial(_combine_local,
                                  mode=mode,
                                  dp_size=dp_size,
                                  axis=DP_AXIS_NAME),
                mesh=mesh,
                in_specs=(sharded, ),
                out_specs=sharded,
                check_vma=False,
            )
            return mapped(gmm_out)

        op = _dp_a2a_jax_op(
            f"pallas::dp_a2a_moe_combine_{mode}",
            combine,
            mesh=mesh,
            input_partition_specs=(sharded, ),
            output_partition_specs=sharded,
        )

        def _fake(gmm_out):
            return torch.empty((gmm_out.shape[0] // dp_size, gmm_out.shape[1]),
                               dtype=gmm_out.dtype,
                               device=gmm_out.device)

        op.register_fake(_fake)
        return op

    def combine(gmm_out: jax.Array, tok_of_pair: jax.Array,
                send_off: jax.Array, send_sizes: jax.Array,
                recv_off: jax.Array, recv_sizes: jax.Array) -> jax.Array:
        mapped = jax.shard_map(
            functools.partial(_combine_local,
                              mode=mode,
                              dp_size=dp_size,
                              axis=DP_AXIS_NAME),
            mesh=mesh,
            in_specs=(sharded, ) * 6,
            out_specs=sharded,
            check_vma=False,
        )
        return mapped(gmm_out, tok_of_pair, send_off, send_sizes, recv_off,
                      recv_sizes)

    op = _dp_a2a_jax_op(
        f"pallas::dp_a2a_moe_combine_{mode}",
        combine,
        mesh=mesh,
        input_partition_specs=(sharded, ) * 6,
        output_partition_specs=sharded,
    )

    def _fake(gmm_out, *_layout):
        return torch.empty((gmm_out.shape[0] // dp_size, gmm_out.shape[1]),
                           dtype=gmm_out.dtype,
                           device=gmm_out.device)

    op.register_fake(_fake)
    return op


def prebuild_dp_a2a(layer: Any) -> bool:
    """Arm the DP a2a exchange, once per process. Called from
    ``process_weights_after_loading`` -- it runs collectives (mesh device
    mapping, chip-block table) and registers torch custom ops, all of which
    are forbidden inside a traced forward.

    Returns True when armed. Any refusal leaves the context unset; the quant
    method then keeps the default dispatch path, so a refusal is never a
    silent behavior change.
    """
    global _CTX
    if not dp_a2a_requested():
        return False
    if _CTX is not None:
        return True
    mode = envs.TPU_MOE_DP_A2A

    def refuse(reason: str) -> bool:
        logger.warning(
            "DP a2a MoE (TPU_MOE_DP_A2A=%s) not engaged: %s. The default "
            "all-gather/reduce-scatter dispatch serves every step.", mode,
            reason)
        return False

    if enable_pipelined_collective_and_compute():
        return refuse(
            "TPU_MOE_COLLECTION_CHUNK_SIZE is set and the pipelined exchange "
            "already owns the DP collectives; the two cannot compose")
    if not layer.moe_config.moe_parallel_config.use_ep:
        return refuse("the experts are replicated rather than "
                      "expert-parallel, so there is no exchange to save")
    from vllm_torchtpu.layers.vllm import moe_routing
    if moe_routing._SIMULATION_STRATEGY is not None:
        return refuse("VLLM_MOE_ROUTING_SIMULATION_STRATEGY is set; simulated "
                      "routing samples per device and the exchange assumes "
                      "identical routing across TP lanes")

    from vllm.distributed.parallel_state import get_dp_group
    dp_size = int(get_dp_group().world_size)
    if dp_size <= 1:
        return refuse("DP world size is 1")

    ep_size = int(layer.moe_config.ep_size)
    global_experts = int(layer.global_num_experts)
    if ep_size <= 1 or global_experts % ep_size != 0:
        return refuse(
            f"{global_experts} global experts do not divide evenly over "
            f"ep_size={ep_size}")
    experts_per_block = global_experts // ep_size

    topk = int(layer.moe_config.experts_per_token)
    if topk % 2:
        return refuse(f"topk={topk} is odd; the packed payload bitcasts "
                      "bf16 pairs into i32 lanes and needs an even topk")

    try:
        mesh = build_dp_mesh()
        dest_table = build_chip_block_dest_table(ep_size,
                                                 layer.w13_weight.device)
    except (RuntimeError, ValueError) as error:
        return refuse(str(error))
    if mesh is None:
        return refuse("no DP group")
    if int(mesh.shape[DP_AXIS_NAME]) != dp_size:
        return refuse(
            f"DP mesh size {int(mesh.shape[DP_AXIS_NAME])} != DP group "
            f"world {dp_size}")

    dispatch_op = _build_dispatch_op(mesh, mode, dp_size, experts_per_block)
    combine_op = _build_combine_op(mesh, mode, dp_size)
    rank_buffer = torch.tensor([[int(get_dp_group().rank_in_group)]],
                               dtype=torch.int32,
                               device=layer.w13_weight.device)
    _CTX = _DpA2aContext(mode, dispatch_op, combine_op, dest_table,
                         rank_buffer, dp_size)
    logger.info(
        "DP a2a MoE armed | mode=%s dp=%d experts_per_block=%d "
        "topk=%d", mode, dp_size, experts_per_block, topk)
    return True


def dp_a2a_fused_moe_gmm(
    ctx: _DpA2aContext,
    *,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: Optional[torch.Tensor],
    **gmm_kwargs,
) -> torch.Tensor:
    """``fused_moe_gmm`` with the DP exchange done as an all-to-all.

    Takes this rank's LOCAL rows and returns this rank's combined LOCAL rows,
    the same contract `pipelined_fused_moe_gmm` has. The GMM itself is
    unchanged: it receives the exchanged global expert ids and masks non-local
    slots via ``experts_start`` exactly as on the all-gather path.
    """
    # The packed payload bitcasts 16-bit rows into i32 lanes, so the weights
    # must arrive in the hidden dtype (the GMM would cast them anyway).
    if topk_weights.dtype != hidden_states.dtype:
        topk_weights = topk_weights.to(hidden_states.dtype)
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    if ctx.mode == "padded":
        recv_h, recv_w, recv_ids = ctx.dispatch_op(hidden_states, topk_weights,
                                                   topk_ids, ctx.dest_table)
        out = fused_moe_gmm(hidden_states=recv_h,
                            topk_weights=recv_w,
                            topk_ids=recv_ids,
                            experts_start=experts_start,
                            **gmm_kwargs)
        return ctx.combine_op(out)

    (recv_h, recv_w, recv_ids, tok_of_pair, send_off, send_sizes, recv_off,
     recv_sizes) = ctx.dispatch_op(hidden_states, topk_weights, topk_ids,
                                   ctx.dest_table, ctx.rank_buffer)
    out = fused_moe_gmm(hidden_states=recv_h,
                        topk_weights=recv_w,
                        topk_ids=recv_ids,
                        experts_start=experts_start,
                        **gmm_kwargs)
    return ctx.combine_op(out, tok_of_pair, send_off, send_sizes, recv_off,
                          recv_sizes)
