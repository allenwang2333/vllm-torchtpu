# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Torch bridge for kernels/experimental/segment_megakernel (TPU_SEGMENT_MEGAKERNEL).

A DeepSeek-V3.2-style decoder (GLM-5.2) runs, between MoE layer N's merged
sparse MLA attention output and the next layer's attention core, a chain of
small XLA ops and kernels: W_UV, o_proj and its TP all-reduce (one fused
kernel at <= 16 rows), the fused add + RMSNorm, the router, the MoE
(moe_tp_decode, with its own all-reduce),
the next layer's fused add + RMSNorm, fused q_a/kv_a, the q_a / kv_a norms,
the k_pe rope and the packing of the KV write rows for the SparseCore gather.
When layer N+1 is a skip layer (no indexer: it reuses the shared top-k), and
the step has at most 32 tokens, this module runs that whole segment as ONE
Pallas call per rank. The call returns exactly the three things layer N+1
needs: the new residual stream, q_c (the input of N+1's q_b_proj) and the KV
write rows (`native_sc_gather.pack_write_rows` layout, the gather's operand).

Dispatch (`run_layer`, called per layer from the patched DeepseekV2Model
forward in models/vllm/deepseek_v2_segment_patch.py): an armed layer N at a
backbone token bucket <= 32, with its KV cache bound, calls its attention with
`segment_tail=True` (the MLA wrapper returns the send-layout / fused small
all-to-all merge's output in latent space instead of applying W_UV and running
o_proj), runs the megakernel,
and hands `(residual, q_c, kv_rows)` to layer N+1, whose input_layernorm is
skipped and whose attention takes `segment_head=(q_c, kv_rows)`. Every other
case runs the layer as upstream does. The token-count test leaves a shape
guard, so the backbone warmup traces each side of it once
(compilation/shape_variants.py); both sides lift the same tensors (the op
takes the layers' own parameters and buffers, reshaped inside JAX; the only
new buffer is the padded q_a/kv_a weight, which replaces the original in place
and is cut back on the usual path).

`arm` runs once at the end of TPUModelRunner.load_model, after
moe_tp_decode's arming (the op reuses its mesh, rank operand and shared-expert
buffers). It is collective over the TP group: every rank must arm the same
layers (the in-kernel all-reduces pair up across ranks), so ranks compare an
arming signature and raise on a mismatch. Mirrors moe_tp_decode.py's bridge
(one pallas.jax_op per distinct statics, built at arming, shard_map over the
EP mesh with per-rank and replicated partition specs, a register_fake for the
dynamic token dimension).

Admission is narrow on purpose: the kernel reproduces the incumbent ops'
numerics only for the setup it was verified on -- TP16 x DCP8 with
VLLM_DCP_Q_REPLICATE, the send-layout DCP decode path (TPU_DCP_MLA_SEND_LAYOUT,
optionally TPU_MLA_FUSED_SMALL_A2A) with W_UV after the merge in bf16,
TPU_SMALL_ALL_REDUCE (valid rows) or TPU_FUSED_OPROJ_AR for o_proj's
all-reduce, o_proj on the fp8_skinny site and q_a/kv_a on xla_quantized_matmul,
moe_tp_decode on N, the KV write in the gather, the MLA decode prologue, an
fp8 KV cache with k_scale 1, bf16 NeoX rope. Anything else keeps the usual
path, with the reason logged.
"""

import dataclasses
import hashlib
import types
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.dcp import dcp_rank_operand, get_dcp_group
from vllm_torchtpu.kernels.experimental import segment_megakernel as kernel
from vllm_torchtpu.kernels.mla import kv_cache_utils as mla_kv_utils
from vllm_torchtpu.layers.adapter import moe_tp_decode, token_padding
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

TAIL_ATTR = "_segment_megakernel_tail"
HEAD_ATTR = "_segment_megakernel_head"

_OPS: dict[tuple, Any] = {}
# The armed head layers' MLA attention modules: the op reads the KV page
# geometry from them when it is exported (after the KV cache is allocated).
_HEAD_ATTNS: list[Any] = []


@dataclasses.dataclass(frozen=True)
class _Statics:
    hidden: int
    tp: int
    dcp: int
    heads_rank: int
    v_head: int
    q_lora: int
    kv_lora: int
    rope: int
    qkv_pad: int
    experts: int
    topk: int
    moe_i: int
    shared_i: int
    routed_scale: float
    rms_eps: float
    interleave: int

    def cfg(self, tokens: int) -> kernel.SegCfg:
        return kernel.SegCfg(
            T=tokens,
            hidden=self.hidden,
            tp=self.tp,
            dcp=self.dcp,
            heads_rank=self.heads_rank,
            v_head=self.v_head,
            q_lora=self.q_lora,
            kv_lora=self.kv_lora,
            rope=self.rope,
            qkv_pad=self.qkv_pad,
            experts=self.experts,
            topk=self.topk,
            moe_i=self.moe_i,
            shared_i=self.shared_i,
            routed_scale=self.routed_scale,
            rms_eps=self.rms_eps,
        )


@dataclasses.dataclass
class _Tail:
    """What an armed layer N needs at forward time. No module references:
    Dynamo names a module's tensors after the first path it reaches the module
    by, and a path through here would rename layer N+1's (and break the
    backbone warmup's hand-over, j-cf10a39a)."""

    op: Any
    dcp_rank: torch.Tensor  # the DCP attention ops' rank operand (one buffer)
    slot: torch.Tensor  # int32 [1, 1]: this boundary's row of the stacks


# The megakernel's outputs for layer N+1: (residual, q_c, kv_rows).
SegmentOut = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# All armed boundaries' small constants, one row / slice per boundary
# (`fstack`, `sw13s`, `sw2s`; kernel.SegCfg.fields()). One stack per tensor is
# too large for XLA to place in VMEM ahead of the call; per-layer operands of
# these sizes each got a serial copy there (j-7694e3a5).
_STACKS: dict[str, torch.Tensor] = {}
STACK_MIN_ROWS = 32
# What the stacks were built from: (decoder layers, [(boundary index N, statics)]).
_STACK_PLAN: list[tuple[list, list]] = []


# ---------------------------------------------------------------------------
# The op
# ---------------------------------------------------------------------------
_PER_RANK = (
    "lat",
    "wuv",
    "dcp_rank",
    "rank",
    "slot",
    "oproj_w",
    "w13",
    "w2",
    "s2",
    "fstack",
    "s13s",
    "sw13s",
    "sw2s",
)
_ARGS = (
    "lat",  # bf16 [T, heads, kv_lora]: N's merged attention output (latent)
    "res",  # bf16 [T, H] residual stream
    "cos",  # bf16 [T, rope / 2] (N+1's rotary table rows)
    "sin",
    "pad_mask",  # token-padding mask (rows >= n_valid are padding)
    "seq_lens",
    "block_tables",
    "query_start_loc",
    "dcp_rank",  # int32 [1]
    "rank",  # int32 [1, 1]: mesh index (moe_tp_decode's)
    "slot",  # int32 [1, 1]: the boundary's row of the constant stacks
    "wuv",  # bf16 [heads, kv_lora, v_head]: N's W_UV
    "oproj_w",  # fp8 [o_in, H]
    "gate",  # bf16 [E, H]
    "w13",
    "w2",
    "s2",
    "qkv_w",  # fp8 [H, qkv_pad] (the lane-padded copy)
    "fstack",  # f32 [boundaries, 1, row_width] (this rank's)
    "s13s",  # f32 [boundaries, E, 1, 2I]
    "sw13s",  # fp8 [boundaries, H, 2 Fs]
    "sw2s",  # fp8 [boundaries, Fs, H]
)


def _kv_geometry() -> tuple[int, int]:
    """(page_size, num_pages) of the armed heads' sparse MLA nope cache."""
    seen = set()
    for attn in _HEAD_ATTNS:
        spec = getattr(attn, "mla_kv_spec", None)
        if spec is None:
            raise RuntimeError(
                f"segment megakernel: {attn.layer_name} has no KV cache spec yet"
            )
        seen.add((int(spec[0].page_size), int(spec[0].num_pages)))
    if len(seen) != 1:
        raise RuntimeError(f"segment megakernel: KV geometries differ: {seen}")
    return next(iter(seen))


def _op_body(st: _Statics):
    def body(
        lat,
        res,
        cos,
        sin,
        pad_mask,
        seq_lens,
        block_tables,
        query_start_loc,
        dcp_rank,
        rank,
        slot,
        wuv,
        oproj_w,
        gate,
        w13,
        w2,
        s2,
        qkv_w,
        fstack,
        s13s,
        sw13s,
        sw2s,
    ):
        T = res.shape[0]  # static: one export per token bucket
        cfg = st.cfg(T)
        page_size, num_pages = _kv_geometry()
        n_valid = T - jnp.sum(pad_mask[:T].astype(jnp.int32))
        cfull, sfull = kernel.rope_lane_tables(cos, sin, cfg.m)
        # The step's KV destination rows on this DCP rank, exactly as
        # sparse_mla_attention_dcp builds them for the gather's KV write.
        spec = types.SimpleNamespace(page_size=page_size, num_pages=num_pages)
        page, row_slot = mla_kv_utils._get_dcp_page_and_slot(
            T,
            seq_lens,
            block_tables,
            query_start_loc,
            dcp_rank[0],
            nope_spec=spec,
            dcp_size=st.dcp,
            interleave_size=st.interleave,
        )
        dst = mla_kv_utils._dst_rows(page, row_slot, page_size)
        ops = dict(
            meta=kernel.pack_meta(n_valid, rank, slot, dst),
            lat=lat.reshape(T, -1),
            act=kernel.pack_act(res, cfull, sfull),
            wuv=wuv,
            oproj_w=oproj_w,
            gate=gate,
            w13=w13,
            w2=w2,
            s2=s2,
            qkv_w=qkv_w,
            fstack=fstack,
            s13s=s13s,
            sw13s=sw13s,
            sw2s=sw2s,
        )
        out = kernel.segment_call(cfg, ops)
        # int32 across the torch boundary (torch has no general uint32).
        kv_rows = lax.bitcast_convert_type(out["kvp_out"], jnp.int32)
        return out["res_out"], out["qc_out"], kv_rows

    return body


def _fake(res: torch.Tensor, q_lora: int):
    T = res.shape[0]
    rnd = kernel.KV_TOKEN_ROUND
    rows = 3 * ((T + rnd - 1) // rnd) * rnd
    return (
        torch.empty_like(res),
        res.new_empty((T, q_lora)),
        res.new_empty((rows, 128), dtype=torch.int32),
    )


def _get_op(mesh, st: _Statics):
    """One jax_op per distinct statics; built at arming, never in a trace."""
    key = (mesh, st)
    op = _OPS.get(key)
    if op is not None:
        return op
    axis = mesh.axis_names[0]
    specs = tuple(P(axis) if name in _PER_RANK else P() for name in _ARGS)
    out_specs = (P(axis), P(axis), P(axis))
    sharded = jax.shard_map(
        _op_body(st), mesh=mesh, in_specs=specs, out_specs=out_specs, check_vma=False
    )

    # torch_tpu reads the operand types from these annotations.
    def fn(
        lat: jax.Array,
        res: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        pad_mask: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        dcp_rank: jax.Array,
        rank: jax.Array,
        slot: jax.Array,
        wuv: jax.Array,
        oproj_w: jax.Array,
        gate: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
        s2: jax.Array,
        qkv_w: jax.Array,
        fstack: jax.Array,
        s13s: jax.Array,
        sw13s: jax.Array,
        sw2s: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        args = (
            lat,
            res,
            cos,
            sin,
            pad_mask,
            seq_lens,
            block_tables,
            query_start_loc,
            dcp_rank,
            rank,
            slot,
            wuv,
            oproj_w,
            gate,
            w13,
            w2,
            s2,
            qkv_w,
            fstack,
            s13s,
            sw13s,
            sw2s,
        )
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    name = "pallas::segment_megakernel" + (f"_{len(_OPS)}" if _OPS else "")
    # PartitionSpecs at runtime, as moe_tp_decode._get_op passes them.
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        name,
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=out_specs,
    )
    q_lora = st.q_lora
    op.register_fake(lambda lat, res, *a, **k: _fake(res, q_lora))
    _OPS[key] = op
    return op


# ---------------------------------------------------------------------------
# Arming
# ---------------------------------------------------------------------------
def _env_reason() -> str | None:
    need = {
        "USE_MOE_TP_DECODE_KERNEL": envs.USE_MOE_TP_DECODE_KERNEL,
        "TPU_MOE_SKIP_PADDED_TOKENS": envs.TPU_MOE_SKIP_PADDED_TOKENS,
        "TPU_SMALL_ALL_REDUCE": envs.TPU_SMALL_ALL_REDUCE,
        "TPU_SMALL_ALL_REDUCE_VALID_ROWS": envs.TPU_SMALL_ALL_REDUCE_VALID_ROWS,
        "TPU_DCP_LSE_MERGE_A2A": envs.TPU_DCP_LSE_MERGE_A2A,
        "TPU_DCP_LSE_MERGE_A2A_PACK": envs.TPU_DCP_LSE_MERGE_A2A_PACK,
        "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS": envs.TPU_DCP_LSE_MERGE_A2A_REAL_ROWS,
        "TPU_DCP_MLA_SEND_LAYOUT": envs.TPU_DCP_MLA_SEND_LAYOUT,
        "TPU_DCP_MLA_SPLIT_Q_LSE": envs.TPU_DCP_MLA_SPLIT_Q_LSE,
        "TPU_MLA_KV_WRITE_IN_GATHER": envs.TPU_MLA_KV_WRITE_IN_GATHER,
        "TPU_DCP_MLA_MASK_IN_KERNEL": envs.TPU_DCP_MLA_MASK_IN_KERNEL,
        "TPU_DCP_MLA_FUSED_PREP": envs.TPU_DCP_MLA_FUSED_PREP,
        "TPU_DCP_RANK_OPERAND": envs.TPU_DCP_RANK_OPERAND,
        "TPU_MLA_DECODE_PROLOGUE": envs.TPU_MLA_DECODE_PROLOGUE,
        "TPU_FP8_LINEAR_PALLAS": envs.TPU_FP8_LINEAR_PALLAS,
    }
    off = [k for k, v in need.items() if not v]
    if off:
        return f"needs {', '.join(off)}=1"
    if envs.FUSED_MOE_DECODE_KERNEL:
        return "FUSED_MOE_DECODE_KERNEL is set"
    if envs.TPU_MOE_ROUTER_HBM_PIN:
        # Its router logits are ~1e-4 one bf16 ulp off the XLA dot's.
        return "TPU_MOE_ROUTER_HBM_PIN is set (the kernel reproduces XLA's router)"
    if envs.TPU_DCP_MLA_SEND_LAYOUT_MIN_TOKENS > min(kernel.TOKEN_BUCKETS):
        return "TPU_DCP_MLA_SEND_LAYOUT_MIN_TOKENS leaves kernel buckets off it"
    if envs.TPU_SMALL_ALL_REDUCE_MAX_ROWS < kernel.MAX_TOKENS:
        return f"TPU_SMALL_ALL_REDUCE_MAX_ROWS < {kernel.MAX_TOKENS}"
    if envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS < kernel.MAX_TOKENS:
        return f"TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS < {kernel.MAX_TOKENS}"
    from vllm_torchtpu.layers.adapter import small_all_reduce_op

    if small_all_reduce_op._AR_VALID_OP is None:
        return "the small valid-rows all-reduce is not armed"
    return None


def _rms_reason(norm: Any, width: int) -> str | None:
    from vllm.model_executor.layers.layernorm import RMSNorm

    if type(norm).__name__ not in ("RMSNorm", "TpuRMSNorm") or not isinstance(
        norm, RMSNorm
    ):
        return f"{type(norm).__name__} is not a plain RMSNorm"
    if not norm.has_weight or norm.variance_size_override is not None:
        return "an RMSNorm without its own weight or with a variance override"
    if tuple(norm.weight.shape) != (width,) or norm.weight.dtype != torch.bfloat16:
        return f"an RMSNorm weight is not bf16 [{width}]"
    return None


def _fp8_linear(lin: Any, k: int, n: int | None) -> str | None:
    from vllm_torchtpu.layers.adapter.linear_common import (
        KEEP_VLLM_LAYOUT_ATTR,
        WEIGHT_FLIPPED_ATTR,
    )
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    name = getattr(lin, "prefix", None) or type(lin).__name__
    if not isinstance(lin.quant_method, VllmFp8LinearMethodTPU):
        return f"{name} is not VllmFp8LinearMethodTPU"
    if not getattr(lin, WEIGHT_FLIPPED_ATTR, False) or getattr(
        lin, KEEP_VLLM_LAYOUT_ATTR, False
    ):
        return f"{name} weight is not K-major"
    if lin.bias is not None:
        return f"{name} has a bias"
    w, s = lin.weight, lin.weight_scale
    if w.dtype != torch.float8_e4m3fn or s.dim() != 1 or s.dtype != torch.float32:
        return f"{name} is not FP8 with f32 per-channel scales"
    if w.shape[0] != k or (n is not None and w.shape[1] != n):
        return f"{name} weight {list(w.shape)} is not [{k}, {n}]"
    return None


def _moe_runner(layer: Any):
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    runners = [m for m in layer.mlp.modules() if isinstance(m, MoERunner)]
    return runners[0] if len(runners) == 1 else None


def _admit_pair(layer: Any, nxt: Any, topo_mesh: Any) -> _Statics | str:
    """Statics for the boundary (layer N -> nxt), or why it keeps the usual path."""
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
    from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        DeepseekScalingRotaryEmbedding,
    )
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention

    from vllm_torchtpu.layers.adapter.linear_common import (
        FP8_PALLAS_N_ATTR,
        FP8_PALLAS_SCHED_GROUP_ATTR,
        FP8_PALLAS_WEIGHT,
    )

    if not isinstance(layer.self_attn, DeepseekV2MLAAttention) or not isinstance(
        nxt.self_attn, DeepseekV2MLAAttention
    ):
        return "not an MLA decoder pair"
    # ---- layer N: the tail (attention output .. MoE)
    runner = _moe_runner(layer)
    if runner is None:
        return "layer N is not a MoE layer (dense FFN tails keep the usual path)"
    if getattr(layer.mlp, "experts", None) is not runner:
        return "layer N's MoE runner is not layer.mlp.experts"
    if not moe_tp_decode.moe_tp_decode_supported(runner._quant_method):
        return "layer N's MoE is not armed by moe_tp_decode"
    moe = runner.routed_experts
    armed = getattr(moe, moe_tp_decode.STATICS_ATTR, None)
    if armed is None:
        return "layer N's moe_tp_decode statics are missing"
    mstat, experts_start = armed
    if mstat.use_ep or experts_start != 0:
        return "layer N's MoE is expert parallel"
    if mstat.activation != "silu":
        return "layer N's MoE activation is not silu"
    gate = runner.gate
    w13 = moe.w13_weight
    experts, hidden, two_i = w13.shape
    sw13, ss13, sw2, ss2 = (getattr(moe, n) for n in moe_tp_decode._BUFFERS[:4])
    if gate is None or tuple(gate.weight.shape) != (experts, hidden):
        return "the router gate is not [E, H] (stock F.linear layout)"
    if gate.weight.dtype != torch.bfloat16 or getattr(gate, "bias", None) is not None:
        return "the router gate is not bf16 without bias"
    wrap = layer.self_attn.mla_attn
    if not (wrap.is_sparse and wrap.dcp_q_replicate) or wrap.g_proj is not None:
        return "layer N's attention is not sparse DCP query-replicated MLA"
    if wrap.gate_is_fused or wrap.kimi_preprocess is not None:
        return "layer N's attention has an output gate"
    attn = wrap.mla_attn
    if getattr(attn, "sparse_mla_dcp_send_op", None) is None:
        return "layer N has no send-layout DCP op (TPU_DCP_MLA_SEND_LAYOUT)"
    w_uv = getattr(attn, "W_UV", None)
    want_uv = (int(wrap.num_heads), int(wrap.kv_lora_rank), int(wrap.v_head_dim))
    if w_uv is None or tuple(w_uv.shape) != want_uv or w_uv.dtype != torch.bfloat16:
        return f"layer N's W_UV is not bf16 {list(want_uv)} (QUANTIZE_ON_LOAD?)"
    if getattr(attn, "W_UV_scale", None) is not None:
        return "layer N's W_UV carries a scale"
    o_proj = wrap.o_proj
    reason = _fp8_linear(o_proj, int(wrap.num_heads * wrap.v_head_dim), hidden)
    if reason is not None:
        return reason
    if (
        getattr(o_proj, FP8_PALLAS_N_ATTR, None) is None
        or getattr(o_proj, FP8_PALLAS_WEIGHT, None) is not None
        or getattr(o_proj, FP8_PALLAS_SCHED_GROUP_ATTR, None) is not None
    ):
        return "o_proj is not an unpadded fp8_skinny site (TPU_FP8_LINEAR_PALLAS_SITES)"
    if not getattr(o_proj, "reduce_results", False):
        return "o_proj does not all-reduce its output"
    # ---- layer N+1: the head (input_layernorm .. KV pack)
    nwrap = nxt.self_attn.mla_attn
    if not (nwrap.is_sparse and nwrap.dcp_q_replicate):
        return "layer N+1 is not sparse DCP query-replicated MLA"
    if nwrap.indexer is not None and not nwrap.skip_topk:
        return "layer N+1 runs an indexer (indexed boundaries keep the usual path)"
    if nwrap.q_lora_rank is None or nwrap.kimi_preprocess is not None:
        return "layer N+1 has no fused q_a/kv_a"
    if nwrap.gate_is_fused or nwrap.g_proj is not None:
        return "layer N+1's attention has an output gate"
    if not (nwrap.decode_prologue and nwrap._rope_neox_permuted):
        return "layer N+1 does not take the MLA decode prologue with NeoX rope"
    rope = nwrap.rotary_emb
    if not isinstance(rope, RotaryEmbedding) or isinstance(
        rope, DeepseekScalingRotaryEmbedding
    ):
        return "layer N+1's rope is not a plain RotaryEmbedding (bf16 cache)"
    if rope.rotary_dim != nwrap.qk_rope_head_dim or nwrap.qk_rope_head_dim != 64:
        return "layer N+1's rope does not cover the 64-wide rope head"
    nattn = nwrap.mla_attn
    if getattr(nattn, "sparse_mla_dcp_send_op", None) is None:
        return "layer N+1 has no send-layout DCP op (TPU_DCP_MLA_SEND_LAYOUT)"
    if nattn.kv_cache_quantized_dtype != torch.float8_e4m3fn:
        return "layer N+1's KV cache is not fp8"
    impl = nattn.impl
    _, k_scale, _ = impl._get_kv_scales(nattn)
    if k_scale not in (None, 1.0):
        return f"layer N+1's k_scale is {k_scale}, not 1"
    group = get_dcp_group()
    if group is None or getattr(impl, "dcp_size", 1) != int(group.world_size):
        return "layer N+1's DCP size differs from the DCP group's"
    qkv = nwrap.fused_qkv_a_proj
    width = nwrap.q_lora_rank + nwrap.kv_lora_rank + nwrap.qk_rope_head_dim
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        SEGMENT_QKV_LOGICAL_ATTR,
    )

    if getattr(qkv, SEGMENT_QKV_LOGICAL_ATTR, None) is not None:
        return "layer N+1 is already armed as a head"
    reason = _fp8_linear(qkv, hidden, width)
    if reason is not None:
        return reason
    if getattr(qkv, FP8_PALLAS_N_ATTR, None) is not None:
        return "fused_qkv_a_proj is a Pallas fp8 site (the kernel reproduces XLA's)"
    norms = (
        (layer.post_attention_layernorm, hidden),
        (nxt.input_layernorm, hidden),
        (nwrap.q_a_layernorm, nwrap.q_lora_rank),
        (nwrap.kv_a_layernorm, nwrap.kv_lora_rank),
    )
    for norm, w in norms:
        reason = _rms_reason(norm, w)
        if reason is not None:
            return reason
    eps = {float(n.variance_epsilon) for n, _ in norms}
    if len(eps) != 1:
        return f"the four RMSNorms use different eps {sorted(eps)}"
    dcp = int(group.world_size)
    st = _Statics(
        hidden=int(hidden),
        tp=int(topo_mesh.size),
        dcp=dcp,
        heads_rank=int(wrap.num_heads),
        v_head=int(wrap.v_head_dim),
        q_lora=int(nwrap.q_lora_rank),
        kv_lora=int(nwrap.kv_lora_rank),
        rope=int(nwrap.qk_rope_head_dim),
        qkv_pad=-(-width // 128) * 128,
        experts=int(experts),
        topk=int(mstat.topk),
        moe_i=int(two_i // 2),
        shared_i=int(sw2.shape[0]),
        routed_scale=float(mstat.routed_scale),
        rms_eps=eps.pop(),
        interleave=int(impl.dcp_interleave_size),
    )
    if st.tp != 16 or st.dcp != 8:
        return f"verified at TP16 x DCP8 only (TP {st.tp}, DCP {st.dcp})"
    if tuple(o_proj.weight.shape) != (st.heads_rank * st.v_head, st.hidden):
        return "o_proj's K is not this rank's merged heads"
    if (tuple(sw13.shape), tuple(ss13.shape)) != (
        (hidden, 2 * st.shared_i),
        (2 * st.shared_i,),
    ):
        return "the shared expert's buffers are not [H, 2Fs] / [2Fs]"
    for t in kernel.TOKEN_BUCKETS:
        reason = kernel.supported(st.cfg(t))
        if reason is not None:
            return f"T={t}: {reason}"
    return st


QKV_W_BUF = "_segment_megakernel_qkv_w"


def _pad_qkv(qkv: Any, n_pad: int, n_logical: int) -> None:
    """Register on fused_qkv_a_proj row-major copies of its weight / scale
    zero-padded to `n_pad` columns (the kernel's layout). The parameters stay
    as they are: the usual path's GEMM reads the weight in the layout its
    loader left (j-7694e3a5: on a row-major padded weight the 4096-token
    q_a/kv_a GEMM ran 155 us instead of 80). Costs n_pad x hidden fp8 of HBM
    per armed boundary."""
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        SEGMENT_QKV_LOGICAL_ATTR,
    )
    from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

    w = qkv.weight.data
    pad = n_pad - w.shape[1]
    # fresh_buffer: a dtype view handed to an op is re-materialized by a
    # tt_jit_as_strided program before every forward (j-da0b4092).
    w_pad = fresh_buffer(
        torch.nn.functional.pad(w.contiguous().view(torch.uint8), (0, pad)).view(
            w.dtype
        )
    )
    qkv.register_buffer(QKV_W_BUF, w_pad, persistent=False)
    setattr(qkv, SEGMENT_QKV_LOGICAL_ATTR, n_logical)
    synchronize_tensors([w_pad])


def _one_side_tensors(model: Any, i: int) -> list[torch.Tensor]:
    """Every tensor of armed boundary (i, i+1) that only one side of the
    token-count branch reads, through the module paths the usual forward
    uses."""
    layer, nxt = model.layers[i], model.layers[i + 1]
    moe = layer.mlp.experts.routed_experts
    nwrap = nxt.self_attn.mla_attn
    qkv = nwrap.fused_qkv_a_proj
    attn = layer.self_attn.mla_attn.mla_attn
    # The usual path reads W_UV after the merge and (TPU_DCP_MERGE_VSPACE)
    # retains the gathered copy beside it; the kernel side reads only W_UV.
    uv = [
        t
        for t in (
            attn.W_UV,
            getattr(attn, "W_UV_dcp_vspace", None),
            getattr(attn, "W_UV_dcp_vspace_scale", None),
        )
        if t is not None
    ]
    out = [
        getattr(layer, TAIL_ATTR).slot,
        *uv,
        layer.self_attn.mla_attn.o_proj.weight_scale,
        layer.post_attention_layernorm.weight,
        moe.w13_weight_scale_inv,
        *(getattr(moe, n) for n in moe_tp_decode._BUFFERS),
        nxt.input_layernorm.weight,
        nwrap.q_a_layernorm.weight,
        nwrap.kv_a_layernorm.weight,
        qkv.weight,
        qkv.weight_scale,
        getattr(qkv, QKV_W_BUF),
    ]
    return out


def _lift_tensors(model: Any) -> list[torch.Tensor]:
    tensors = list(_STACKS.values())
    for i, layer in enumerate(model.layers):
        if getattr(layer, TAIL_ATTR, None) is not None:
            tensors += _one_side_tensors(model, i)
    return tensors


# Lift ops by arity, and the chunk sizes the armed model's tensors split
# into: a torch op schema takes at most 64 arguments (positions + 63), and the
# served model lifts ~950 tensors (j-b0950221).
LIFT_MAX_TENSORS = 63
_LIFT_OPS: dict[int, Any] = {}
_LIFT_CHUNKS: list[int] = []


def lift_chunks(n: int, cap: int = LIFT_MAX_TENSORS) -> list[int]:
    """Sizes of consecutive chunks of n tensors, each at most `cap`."""
    return [min(cap, n - i) for i in range(0, n, cap)]


def _build_lift_op(n: int) -> Any:
    """positions + 0 * (one element of each of n tensors), in JAX: the
    integer zero folds away in XLA (no reads, no copies), but every tensor is
    an operand of the op, so Dynamo lifts it."""
    params = ", ".join(f"t{i}: jax.Array" for i in range(n))
    body = "".join(
        f"    z = z + t{i}.reshape(-1)[0].astype(jnp.int32) * 0\n" for i in range(n)
    )
    src = (
        f"def segment_lift(positions: jax.Array, {params}) -> jax.Array:\n"
        "    z = jnp.int32(0)\n"
        + body
        + "    return positions + z.astype(positions.dtype)\n"
    )
    ns: dict[str, Any] = {"jax": jax, "jnp": jnp}
    exec(src, ns)  # a fixed-arity signature: torch_tpu reads the annotations
    op = pallas.jax_op(f"pallas::segment_megakernel_lift_{n}", ns["segment_lift"])
    op.register_fake(lambda positions, *a, **k: torch.empty_like(positions))
    return op


def lift_segment_buffers(model: Any, positions: torch.Tensor) -> torch.Tensor:
    """`positions`, through an op that takes every tensor only one side of
    the token-count branch reads (the stacks and padded q_a/kv_a copies on the
    kernel side; the boundaries' own constants on the usual side), so both
    traces lift the same inputs, under the same names (this runs first in
    both), and the backbone warmup can hand buckets between them. Reading
    them in torch instead materialized relayout copies of the fp8 weights
    every step (twin j-8f7e86c3: layer 0 +337 us)."""
    if not _LIFT_CHUNKS:
        return positions
    tensors = _lift_tensors(model)
    start = 0
    for k in _LIFT_CHUNKS:  # positions threads through the chain
        positions = _LIFT_OPS[k](positions, *tensors[start : start + k])
        start += k
    return positions


def _compile_sizes_reason(vllm_config: Any) -> str | None:
    sizes = getattr(vllm_config.compilation_config, "compile_sizes", None) or []
    small = sorted(int(s) for s in sizes if int(s) <= kernel.MAX_TOKENS)
    bad = [s for s in small if s not in kernel.TOKEN_BUCKETS]
    if bad:
        return f"token buckets {bad} <= {kernel.MAX_TOKENS} are not kernel buckets"
    return None


def layer_device(layer: Any) -> torch.device:
    return layer.post_attention_layernorm.weight.device


def _consts_row(st: _Statics, layer: Any, nxt: Any) -> torch.Tensor:
    """Boundary (layer, nxt)'s fstack row (kernel.SegCfg.fields() order)."""
    moe = _moe_runner(layer).routed_experts
    sw13, ss13, sw2, ss2, bias, _, _ = (getattr(moe, n) for n in moe_tp_decode._BUFFERS)
    nwrap = nxt.self_attn.mla_attn
    consts = dict(
        oproj_s=layer.self_attn.mla_attn.o_proj.weight_scale,
        g_post=layer.post_attention_layernorm.weight,
        g_in=nxt.input_layernorm.weight,
        g_qa=nwrap.q_a_layernorm.weight,
        g_kva=nwrap.kv_a_layernorm.weight,
        bias=bias,
        ss13=ss13,
        ss2=ss2,
        qkv_s=nwrap.fused_qkv_a_proj.weight_scale,
    )
    parts = []
    for name, _, w in st.cfg(kernel.TOKEN_BUCKETS[0]).fields():
        v = consts[name].detach().reshape(-1).to(torch.float32)
        parts.append(torch.nn.functional.pad(v, (0, w - v.shape[0])))
    return torch.cat(parts)


def _stack_values(layers: list, plan: list) -> dict[str, torch.Tensor]:
    """The constant stacks' values from the layers' current tensors."""
    rows, s13s, sw13s, sw2s = [], [], [], []
    for i, st in plan:
        layer, nxt = layers[i], layers[i + 1]
        moe = _moe_runner(layer).routed_experts
        rows.append(_consts_row(st, layer, nxt))
        s13s.append(
            moe.w13_weight_scale_inv.detach().reshape(moe.w13_weight.shape[0], 1, -1)
        )
        sw13s.append(getattr(moe, moe_tp_decode._BUFFERS[0]).detach().contiguous())
        sw2s.append(getattr(moe, moe_tp_decode._BUFFERS[2]).detach().contiguous())
    # At least STACK_MIN_ROWS rows (zero padding): with few armed boundaries
    # (a twin has one) the stacks are small enough for XLA to place in VMEM
    # with a copy each, as it does the per-layer tensors; GLM-5.2 serves 56.
    pad = max(0, STACK_MIN_ROWS - len(rows))

    def stack(xs, view8=False):
        xs = [x.view(torch.uint8) if view8 else x for x in xs]
        xs = xs + [torch.zeros_like(xs[0])] * pad
        out = torch.stack(xs)
        return out.view(sw13s[0].dtype) if view8 else out

    return {
        "fstack": stack(rows)[:, None, :],
        "s13s": stack(s13s).to(torch.float32),
        "sw13s": stack(sw13s, view8=True),
        "sw2s": stack(sw2s, view8=True),
    }


def _build_stacks(layers: list, plan: list) -> None:
    """The constant stacks, one row / slice per armed boundary, as fresh
    buffers (a view would be re-materialized every forward)."""
    from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

    for name, value in _stack_values(layers, plan).items():
        _STACKS[name] = fresh_buffer(value)
    _STACK_PLAN[:] = [(layers, list(plan))]
    synchronize_tensors(list(_STACKS.values()))


def refresh_stacks() -> int:
    """Copy the armed boundaries' current constants into the stacks, in
    place (compiled programs keep reading the same buffers); returns the
    boundaries refreshed. The stacks are copies taken at arming, so a
    parameter changed in place after `arm` (the routing correction bias, the
    norm weights, the scales) would otherwise reach the megakernel stale while
    the usual path reads it live. Serving changes none after loading; a caller
    that does (the microbenchmark twin re-draws the routers) calls this after."""
    from vllm_torchtpu.utils import synchronize_tensors

    if not _STACK_PLAN:
        return 0
    layers, plan = _STACK_PLAN[0]
    for name, value in _stack_values(layers, plan).items():
        _STACKS[name].copy_(value.to(_STACKS[name].dtype))
    synchronize_tensors(list(_STACKS.values()))
    return len(plan)


def arm(model: torch.nn.Module, vllm_config: Any) -> int:
    """Arm every admitted MoE -> skip boundary of `model`'s decoder stack;
    returns how many. Collective over the TP group. The MTP drafter is not
    armed (it is traced once for every bucket)."""
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

    topo = moe_tp_decode._topology()  # collective: first, on every rank
    stacks = [m for m in model.modules() if isinstance(m, DeepseekV2Model)]
    inner = stacks[0] if len(stacks) == 1 else None
    layers = list(getattr(inner, "layers", []) or [])
    start = int(getattr(inner, "start_layer", 0))
    end = int(getattr(inner, "end_layer", len(layers)))
    plan: list[tuple[int, _Statics]] = []
    refused: dict[str, int] = {}

    def refuse(why: str) -> None:
        refused[why] = refused.get(why, 0) + 1

    if isinstance(topo, str):
        refuse(topo)
    else:
        global_reason = _env_reason() or _compile_sizes_reason(vllm_config)
        aux = set(getattr(inner, "aux_hidden_state_layers", ()) or ())
        # A head can also be a tail (skip -> skip): its forward consumes one
        # hand-off and produces the next.
        for i in range(start, end - 1):
            if global_reason is not None:
                refuse(global_reason)
                continue
            if (i + 1) in aux:
                refuse("layer N+1 feeds an aux hidden state")
                continue
            res = _admit_pair(layers[i], layers[i + 1], topo.mesh)
            if isinstance(res, str):
                refuse(res)
                continue
            plan.append((i, res))
    sig = hashlib.sha256(
        repr(([(i, repr(st)) for i, st in plan], len(layers))).encode()
    ).hexdigest()[:16]
    mesh_index = -1 if isinstance(topo, str) else topo.mesh_index
    moe_tp_decode._check_ranks_agree(
        dict(sig=sig, mesh_index=mesh_index, armed=len(plan))
    )
    if plan:
        _build_stacks(layers, plan)
    for k, (i, st) in enumerate(plan):
        assert not isinstance(topo, str)
        layer, nxt = layers[i], layers[i + 1]
        nwrap = nxt.self_attn.mla_attn
        width = nwrap.q_lora_rank + nwrap.kv_lora_rank + nwrap.qk_rope_head_dim
        _pad_qkv(nwrap.fused_qkv_a_proj, st.qkv_pad, width)
        nattn = nwrap.mla_attn
        # The head's send-layout op and (TPU_MLA_FUSED_SMALL_A2A) its fused
        # exchange op, taking the megakernel's KV write rows.
        nattn.sparse_mla_dcp_send_kvrows_op = nattn.impl._build_sparse_mla_dcp_op(
            nattn,
            st.dcp,
            st.interleave,
            q_rope=True,
            send_layout=True,
            kv_rows=True,
        )
        if getattr(nattn, "sparse_mla_dcp_fused_op", None) is not None:
            nattn.sparse_mla_dcp_fused_kvrows_op = (
                nattn.impl._build_sparse_mla_dcp_op(
                    nattn,
                    st.dcp,
                    st.interleave,
                    q_rope=True,
                    send_layout=True,
                    fused_exchange=True,
                    kv_rows=True,
                )
            )
        # TPU_GROUP_BATCHED_GATHER: a head that is also a group member (its
        # group gather ops were built at load, before this arming) takes the
        # rows its group head pre-gathered as well as the megakernel's rows.
        if getattr(nattn, "sparse_mla_dcp_send_pre_op", None) is not None:
            nattn.sparse_mla_dcp_send_kvrows_pre_op = (
                nattn.impl._build_sparse_mla_dcp_op(
                    nattn,
                    st.dcp,
                    st.interleave,
                    q_rope=True,
                    send_layout=True,
                    kv_rows=True,
                    group_pre=True,
                )
            )
            if getattr(nattn, "sparse_mla_dcp_fused_pre_op", None) is not None:
                nattn.sparse_mla_dcp_fused_kvrows_pre_op = (
                    nattn.impl._build_sparse_mla_dcp_op(
                        nattn,
                        st.dcp,
                        st.interleave,
                        q_rope=True,
                        send_layout=True,
                        fused_exchange=True,
                        kv_rows=True,
                        group_pre=True,
                    )
                )
        _HEAD_ATTNS.append(nattn)
        setattr(
            layer,
            TAIL_ATTR,
            _Tail(
                op=_get_op(topo.mesh, st),
                dcp_rank=dcp_rank_operand(),
                slot=torch.tensor([[k]], dtype=torch.int32, device=layer_device(layer)),
            ),
        )
        setattr(nxt, HEAD_ATTR, True)
    if plan:
        _LIFT_CHUNKS[:] = lift_chunks(len(_lift_tensors(inner)))
        for k in set(_LIFT_CHUNKS):
            if k not in _LIFT_OPS:
                _LIFT_OPS[k] = _build_lift_op(k)
    log = logger.info if plan else logger.warning
    log(
        "segment_megakernel: armed %d boundaries (layers N = %s, buckets %s, sig %s); "
        "refused: %s",
        len(plan),
        [i for i, _ in plan],
        list(kernel.TOKEN_BUCKETS),
        sig,
        refused or "none",
    )
    return len(plan)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
def _kv_bound(wrap: Any) -> bool:
    """Whether the head layer's KV cache is allocated (not the memory probe)."""
    from vllm.model_executor.layers.attention.attention import get_attention_context

    _, _, kv_cache, _ = get_attention_context(wrap.mla_attn.layer_name)
    return isinstance(kv_cache, (tuple, list))


def takes_tail(layer: Any, positions: torch.Tensor) -> bool:
    """Trace-time dispatch for layer N. The token-count test leaves a shape
    guard: the backbone warmup traces each side once."""
    tail = getattr(layer, TAIL_ATTR, None)
    if tail is None:
        return False
    from vllm.compilation import backends as vllm_compilation_backends

    if vllm_compilation_backends.model_tag != "backbone":
        return False
    if positions.shape[0] > kernel.MAX_TOKENS:
        return False
    # One KV allocation covers every layer: N's cache stands for N+1's.
    return _kv_bound(layer.self_attn.mla_attn)


def segment_forward(
    prev: Any,
    layer: Any,
    recv: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
) -> SegmentOut:
    """The megakernel for the boundary (prev = armed layer N, layer = N+1)
    from N's merge receive buffer. Called from N+1's step, and every tensor is
    read through the module paths the usual forward reads it through
    (`layers[i].mlp.experts.routed_experts...`): Dynamo names a lifted tensor
    after its first access path, and the backbone warmup hands a bucket
    between the two sides of the token-count branch only if both lift the
    same names."""
    from vllm.model_executor.layers.attention.attention import get_attention_context

    tail: _Tail = getattr(prev, TAIL_ATTR)
    wrap, nwrap = prev.self_attn.mla_attn, layer.self_attn.mla_attn
    md, _, _, _ = get_attention_context(nwrap.mla_attn.layer_name)
    padding = token_padding._padding_state
    assert padding is not None, "the segment megakernel needs the token-padding mask"
    # N+1's rope rows, as rope_half_split reads them (bf16 cache).
    cache = nwrap.rotary_emb._match_cos_sin_cache_dtype(residual)
    cos, sin = cache.index_select(0, positions.flatten()).chunk(2, dim=-1)
    runner = prev.mlp.experts
    moe = runner.routed_experts
    rank = getattr(moe, moe_tp_decode._BUFFERS[5])
    res_out, q_c, kv_rows = tail.op(
        recv,  # the merged latent [T, heads, kv_lora]
        residual,
        cos,
        sin,
        padding.local_padding_mask,
        md.seq_lens,
        md.block_tables,
        md.query_start_loc,
        tail.dcp_rank,
        rank,
        tail.slot,
        wrap.mla_attn.W_UV,
        wrap.o_proj.weight,
        runner.gate.weight,
        moe.w13_weight,
        moe.w2_weight,
        moe.w2_weight_scale_inv,
        getattr(nwrap.fused_qkv_a_proj, QKV_W_BUF),
        _STACKS["fstack"],
        _STACKS["s13s"],
        _STACKS["sw13s"],
        _STACKS["sw2s"],
    )
    return res_out, q_c, kv_rows


# Layer N's hand-off: (layer N, its merge receive buffer, the residual).
Pending = tuple[Any, torch.Tensor, torch.Tensor]


def run_layer(
    layer: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor | None,
    residual: torch.Tensor | None,
    llama_4_scaling: torch.Tensor | None,
    pending: Pending | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, Pending | None]:
    """One DeepseekV2DecoderLayer step with the segment hand-offs: `pending`
    is armed layer N-1's (or None); returns (hidden_states, residual,
    hand-off to N+1). Without either hand-off this is
    DeepseekV2DecoderLayer.forward on the MLA / no-sequence-parallel path."""
    tail = takes_tail(layer, positions)
    if pending is None and not tail:
        hs, res = layer(positions, hidden_states, residual, llama_4_scaling)
        return hs, res, None
    assert not layer.use_sequence_parallel_moe and not layer.use_mha
    attn_kwargs: dict[str, Any] = {}
    if pending is not None:
        prev, recv, prev_residual = pending
        residual, q_c, kv_rows = segment_forward(
            prev, layer, recv, prev_residual, positions
        )
        hidden_states = residual  # token-count template only
        attn_kwargs["segment_head"] = (q_c, kv_rows)
    elif residual is None:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
    else:
        hidden_states, residual = layer.input_layernorm(hidden_states, residual)
    if tail:
        attn_kwargs["segment_tail"] = True
    out = layer.self_attn.mla_attn(
        positions, hidden_states, llama_4_scaling, **attn_kwargs
    )
    if tail:
        return None, None, (layer, out, residual)
    hidden_states, residual = layer.post_attention_layernorm(out, residual)
    hidden_states = layer.mlp(hidden_states)
    return hidden_states, residual, None
