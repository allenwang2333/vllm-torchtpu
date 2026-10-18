# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Torch bridge for kernels/experimental/segment_megakernel (TPU_SEGMENT_MEGAKERNEL).

A DeepSeek-V3.2-style decoder (GLM-5.2) runs, between MoE layer N's sparse
MLA attention and the next layer's attention core, a chain of ~50 small XLA
ops and 3 kernels: the DCP LSE merge, o_proj and its TP all-reduce, the fused
add + RMSNorm, the router, the MoE (moe_tp_decode, with its own all-reduce),
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
`segment_tail=True` (the MLA wrapper returns the merge's packed all-to-all
receive buffer instead of merging it and running o_proj), runs the megakernel,
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
VLLM_DCP_Q_REPLICATE, TPU_DCP_MLA_UV_BEFORE_MERGE, the packed real-rows LSE
merge all-to-all, TPU_SMALL_ALL_REDUCE (valid rows) for o_proj's all-reduce,
o_proj on the fp8_skinny site and q_a/kv_a on xla_quantized_matmul,
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


# The megakernel's outputs for layer N+1: (residual, q_c, kv_rows).
SegmentOut = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# The op
# ---------------------------------------------------------------------------
_PER_RANK = (
    "a2a",
    "dcp_rank",
    "rank",
    "e0",
    "oproj_w",
    "oproj_s",
    "w13",
    "w2",
    "s13",
    "s2",
    "sw13",
    "ss13",
    "sw2",
    "ss2",
)
_ARGS = (
    "a2a",  # int16 [dcp, 1 + 4T, v_head]: N's packed merge receive buffer
    "res",  # bf16 [T, H] residual stream
    "cos",  # bf16 [T, rope / 2] (N+1's rotary table rows)
    "sin",
    "pad_mask",  # token-padding mask (rows >= n_valid are padding)
    "seq_lens",
    "block_tables",
    "query_start_loc",
    "dcp_rank",  # int32 [1]
    "rank",  # int32 [1, 1]: mesh index (moe_tp_decode's)
    # int32 [1, 1]: N's moe_tp_decode first expert (0 under TP). Unused; taken
    # so both sides of the token-count branch lift the same buffers.
    "e0",
    "oproj_w",  # fp8 [o_in, H]
    "oproj_s",  # f32 [H]
    "g_post",  # bf16 [H]
    "gate",  # bf16 [E, H]
    "bias",  # f32 [E]
    "w13",
    "w2",
    "s13",
    "s2",
    "sw13",
    "ss13",
    "sw2",
    "ss2",
    "g_in",  # bf16 [H]
    "qkv_w",  # fp8 [H, qkv_pad]
    "qkv_s",  # f32 [qkv_pad]
    "g_qa",  # bf16 [q_lora]
    "g_kva",  # bf16 [kv_lora]
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
        a2a,
        res,
        cos,
        sin,
        pad_mask,
        seq_lens,
        block_tables,
        query_start_loc,
        dcp_rank,
        rank,
        e0,
        oproj_w,
        oproj_s,
        g_post,
        gate,
        bias,
        w13,
        w2,
        s13,
        s2,
        sw13,
        ss13,
        sw2,
        ss2,
        g_in,
        qkv_w,
        qkv_s,
        g_qa,
        g_kva,
    ):
        T = res.shape[0]  # static: one export per token bucket
        cfg = st.cfg(T)
        page_size, num_pages = _kv_geometry()
        n_valid = T - jnp.sum(pad_mask[:T].astype(jnp.int32))
        # meta[3] is unread by the kernel; e0 rides there so it stays an
        # operand of the exported function (an unused one would be pruned).
        meta = jnp.stack(
            [n_valid, rank.reshape(()), jnp.int32(0), e0.reshape(())]
        ).astype(jnp.int32)
        cfull, sfull = kernel.rope_lane_tables(cos, sin, cfg.m)
        # The step's KV destination rows on this DCP rank, exactly as
        # sparse_mla_attention_dcp builds them for the gather's KV write.
        spec = types.SimpleNamespace(page_size=page_size, num_pages=num_pages)
        page, slot = mla_kv_utils._get_dcp_page_and_slot(
            T,
            seq_lens,
            block_tables,
            query_start_loc,
            dcp_rank[0],
            nope_spec=spec,
            dcp_size=st.dcp,
            interleave_size=st.interleave,
        )
        dst = mla_kv_utils._dst_rows(page, slot, page_size)
        row = lambda x: x.reshape(1, -1)  # noqa: E731
        ops = dict(
            meta=meta,
            a2a=a2a,
            res=res,
            cfull=cfull,
            sfull=sfull,
            g_post=row(g_post),
            g_in=row(g_in),
            g_qa=row(g_qa),
            g_kva=row(g_kva),
            bias=row(bias),
            enc=kernel.kv_dst_block(dst, cfg.tok_rows),
            oproj_w=oproj_w,
            oproj_s=row(oproj_s),
            gate=gate,
            w13=w13,
            w2=w2,
            s13=s13,
            s2=s2,
            sw13=sw13,
            ss13=row(ss13),
            sw2=sw2,
            ss2=row(ss2),
            qkv_w=qkv_w,
            qkv_s=row(qkv_s),
        )
        shapes = kernel.operand_shapes(cfg)
        ops = {k: v.reshape(shapes[k][0]) for k, v in ops.items()}
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
        a2a: jax.Array,
        res: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        pad_mask: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        dcp_rank: jax.Array,
        rank: jax.Array,
        e0: jax.Array,
        oproj_w: jax.Array,
        oproj_s: jax.Array,
        g_post: jax.Array,
        gate: jax.Array,
        bias: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
        s13: jax.Array,
        s2: jax.Array,
        sw13: jax.Array,
        ss13: jax.Array,
        sw2: jax.Array,
        ss2: jax.Array,
        g_in: jax.Array,
        qkv_w: jax.Array,
        qkv_s: jax.Array,
        g_qa: jax.Array,
        g_kva: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        args = (
            a2a,
            res,
            cos,
            sin,
            pad_mask,
            seq_lens,
            block_tables,
            query_start_loc,
            dcp_rank,
            rank,
            e0,
            oproj_w,
            oproj_s,
            g_post,
            gate,
            bias,
            w13,
            w2,
            s13,
            s2,
            sw13,
            ss13,
            sw2,
            ss2,
            g_in,
            qkv_w,
            qkv_s,
            g_qa,
            g_kva,
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
    op.register_fake(lambda a2a, res, *a, **k: _fake(res, q_lora))
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
        "TPU_DCP_MLA_UV_BEFORE_MERGE": envs.TPU_DCP_MLA_UV_BEFORE_MERGE,
        "TPU_DCP_LSE_MERGE_A2A": envs.TPU_DCP_LSE_MERGE_A2A,
        "TPU_DCP_LSE_MERGE_A2A_PACK": envs.TPU_DCP_LSE_MERGE_A2A_PACK,
        "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS": envs.TPU_DCP_LSE_MERGE_A2A_REAL_ROWS,
        "TPU_DCP_RS_OUT_PRECISION": envs.TPU_DCP_RS_OUT_PRECISION,
        "TPU_MLA_KV_WRITE_IN_GATHER": envs.TPU_MLA_KV_WRITE_IN_GATHER,
        "TPU_DCP_MLA_MASK_IN_KERNEL": envs.TPU_DCP_MLA_MASK_IN_KERNEL,
        "TPU_DCP_MLA_FUSED_PREP": envs.TPU_DCP_MLA_FUSED_PREP,
        "TPU_DCP_RANK_OPERAND": envs.TPU_DCP_RANK_OPERAND,
        "TPU_MLA_DECODE_PROLOGUE": envs.TPU_MLA_DECODE_PROLOGUE,
        "TPU_MLA_SPARSE_SPLIT_Q": envs.TPU_MLA_SPARSE_SPLIT_Q,
        "TPU_FP8_LINEAR_PALLAS": envs.TPU_FP8_LINEAR_PALLAS,
    }
    off = [k for k, v in need.items() if not v]
    if off:
        return f"needs {', '.join(off)}=1"
    if envs.FUSED_MOE_DECODE_KERNEL:
        return "FUSED_MOE_DECODE_KERNEL is set"
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
    if getattr(attn, "W_UV_dcp_qrep", None) is None:
        return "layer N's W_UV does not run before the merge"
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


def _pad_qkv(qkv: Any, n_pad: int, n_logical: int) -> None:
    """Replace fused_qkv_a_proj's weight / scale by copies zero-padded to
    `n_pad` columns (the kernel's layout); the usual path cuts its output back
    to `n_logical` (SEGMENT_QKV_LOGICAL_ATTR)."""
    from vllm.model_executor.utils import replace_parameter

    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        SEGMENT_QKV_LOGICAL_ATTR,
    )
    from vllm_torchtpu.utils import synchronize_tensors

    w, s = qkv.weight.data, qkv.weight_scale.data
    pad = n_pad - w.shape[1]
    w_pad = torch.nn.functional.pad(w.view(torch.uint8), (0, pad)).view(w.dtype)
    s_pad = torch.nn.functional.pad(s, (0, pad))
    replace_parameter(qkv, "weight", torch.nn.Parameter(w_pad, requires_grad=False))
    replace_parameter(
        qkv, "weight_scale", torch.nn.Parameter(s_pad, requires_grad=False)
    )
    setattr(qkv, SEGMENT_QKV_LOGICAL_ATTR, n_logical)
    synchronize_tensors([qkv.weight, qkv.weight_scale])


def _compile_sizes_reason(vllm_config: Any) -> str | None:
    sizes = getattr(vllm_config.compilation_config, "compile_sizes", None) or []
    small = sorted(int(s) for s in sizes if int(s) <= kernel.MAX_TOKENS)
    bad = [s for s in small if s not in kernel.TOKEN_BUCKETS]
    if bad:
        return f"token buckets {bad} <= {kernel.MAX_TOKENS} are not kernel buckets"
    return None


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
    for i, st in plan:
        assert not isinstance(topo, str)
        layer, nxt = layers[i], layers[i + 1]
        nwrap = nxt.self_attn.mla_attn
        width = nwrap.q_lora_rank + nwrap.kv_lora_rank + nwrap.qk_rope_head_dim
        _pad_qkv(nwrap.fused_qkv_a_proj, st.qkv_pad, width)
        nattn = nwrap.mla_attn
        nattn.sparse_mla_dcp_kvrows_op = nattn.impl._build_sparse_mla_dcp_op(
            nattn,
            st.dcp,
            st.interleave,
            q_rope=True,
            kv_rows=True,
        )
        _HEAD_ATTNS.append(nattn)
        setattr(
            layer,
            TAIL_ATTR,
            _Tail(
                op=_get_op(topo.mesh, st),
                dcp_rank=dcp_rank_operand(),
            ),
        )
        setattr(nxt, HEAD_ATTR, True)
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
    sw13, ss13, sw2, ss2, bias, rank, e0 = (
        getattr(moe, n) for n in moe_tp_decode._BUFFERS
    )
    qkv = nwrap.fused_qkv_a_proj
    res_out, q_c, kv_rows = tail.op(
        recv,
        residual,
        cos,
        sin,
        padding.local_padding_mask,
        md.seq_lens,
        md.block_tables,
        md.query_start_loc,
        tail.dcp_rank,
        rank,
        e0,
        wrap.o_proj.weight,
        wrap.o_proj.weight_scale,
        prev.post_attention_layernorm.weight,
        runner.gate.weight,
        bias,
        moe.w13_weight,
        moe.w2_weight,
        moe.w13_weight_scale_inv,
        moe.w2_weight_scale_inv,
        sw13,
        ss13,
        sw2,
        ss2,
        layer.input_layernorm.weight,
        qkv.weight,
        qkv.weight_scale,
        nwrap.q_a_layernorm.weight,
        nwrap.kv_a_layernorm.weight,
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
