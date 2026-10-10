# SPDX-License-Identifier: Apache-2.0
"""Trace-stable sequence-parallel forward and FP8 fusions for Qwen3Next and Qwen3.5 MoE.

Includes:
- Fused attention output sigmoid-gate + per-token FP8 quantization (`pallas_sigmoid_gate_quant`)
- Pre-transposed GDN `in_proj_qkvz` FP8 weights (`maybe_flip_gdn_in_proj_qkvz_weight`)
- 256-column-aligned FP8 router gate + shared_expert input quantization sharing
- In-place W8A8 FP8 quantization of `ParallelLMHead`
- Trace-stable sequence-parallel forwards when `TPU_MOE_SEQUENCE_PARALLEL=1`
"""

from collections.abc import Callable
import contextlib
import functools
from itertools import islice
import os
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import torch
import torch.nn.functional as F
from torch_tpu._internal import pallas
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.model_loader import base_loader
from vllm.model_executor.models import qwen3_next, qwen3_next_mtp
from vllm.sequence import IntermediateTensors

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

UPSTREAM_DECODER_LAYER_FORWARD = qwen3_next.Qwen3NextDecoderLayer.forward
UPSTREAM_MODEL_FORWARD = qwen3_next.Qwen3NextModel.forward
UPSTREAM_MTP_FORWARD = qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward
UPSTREAM_SPARSE_MOE_FORWARD = qwen3_next.Qwen3NextSparseMoeBlock.forward
UPSTREAM_ATTN_INIT = qwen3_next.Qwen3NextAttention.__init__
UPSTREAM_ATTN_FORWARD = qwen3_next.Qwen3NextAttention.forward
UPSTREAM_GEMMA_RMSNORM_FORWARD_NATIVE = GemmaRMSNorm.forward_native

PAL_GATE_Q_MAX_ROWS = 8192

_sigmoid_gate_quant_op: Callable | None = None
_sigmoid_gate_o_proj_op: Callable | None = None
_fp8_gate_and_se_op: Callable | None = None
_V6_PATCHES_INSTALLED = False
_step0_gather_indices: torch.Tensor | None = None
# `Qwen35MtpMegakernel` running the MTP layer, while a draft program compiled
# for it executes (see `mtp_megakernel_context`).
_mtp_megakernel: Any = None


@contextlib.contextmanager
def step0_post_attn_gather_context(indices: torch.Tensor | None):
    """Gather last-token rows immediately after self_attn.attn(q, k, v) on MTP step 0."""
    global _step0_gather_indices
    prev = _step0_gather_indices
    _step0_gather_indices = indices
    try:
        yield
    finally:
        _step0_gather_indices = prev


@contextlib.contextmanager
def mtp_megakernel_context(mk: Any):
    """Run the MTP layer and final norm through `mk.forward_layer`.

    A draft program is traced once, so a program first called inside this
    context must be called inside it every time, and a program traced outside
    it must never be.
    """
    global _mtp_megakernel
    prev = _mtp_megakernel
    _mtp_megakernel = mk
    try:
        yield
    finally:
        _mtp_megakernel = prev


def is_step0_post_attn_gather_installed(draft_model: Any) -> bool:
    """Return True if `draft_model` executes the step-0 post-attn gather path."""
    if draft_model is None:
        return False
    inner = getattr(draft_model, "model", draft_model)
    if getattr(inner, "_supports_step0_post_attn_gather", False) or getattr(
        draft_model, "_supports_step0_post_attn_gather", False
    ):
        return True
    return (
        qwen3_next.Qwen3NextAttention.forward is patched_qwen3_next_attn_forward
        and qwen3_next.Qwen3NextDecoderLayer.forward is patched_decoder_layer_forward
        and getattr(type(inner), "forward", None) is patched_mtp_forward
    )


def _legal_tile(rows: int, cap: int) -> int:
    for t in range(min(cap, rows), 31, -1):
        if rows % t == 0 and t % 32 == 0:
            return t
    return rows


def _select_gate_q_tile(rows: int, cols: int) -> int:
    """Select multiple-of-8 tile cap for [rows, cols] sigmoid-gate + FP8 quant.

    Keeps live fp32 arrays within the 64-VREG hardware file (preventing multi-MiB
    VMEM spills at N=2048) and ensures grid >= 6 at decode/verify batch sizes
    (N=96, 128) so double-buffered VMEM DMAs overlap with compute.
    """
    vregs_per_row = max(1, (cols + 1023) // 1024)
    vreg_cap = max(16, (64 // vregs_per_row // 8) * 8)
    if rows <= 256:
        dma_cap = max(8, (rows // 6 // 8) * 8)
        return min(vreg_cap, dma_cap)
    return vreg_cap


def _sigmoid_gate_quant_kernel(attn_ref, gate_ref, q_ref, s_ref):
    a = attn_ref[...].astype(jnp.float32)
    g = gate_ref[...].astype(jnp.float32)
    y = (a * jax.nn.sigmoid(g)).astype(jnp.bfloat16).astype(jnp.float32)
    amax = jnp.max(jnp.abs(y), axis=1, keepdims=True)
    scale = amax * jnp.float32(1.0 / 448.0)
    scale = jnp.where(scale == 0.0, 1.0, scale)
    scale_inv = jnp.float32(1.0) / scale
    q_ref[...] = jnp.clip(y * scale_inv, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    s_ref[...] = scale


def pallas_sigmoid_gate_quant(
    attn_out: jax.Array,
    gate: jax.Array,
    *,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fused attn_output * sigmoid(gate) + per-token FP8 quantization in Pallas."""
    orig_m, k = attn_out.shape
    if orig_m < 8:
        pad = 8 - orig_m
        attn_out = jnp.pad(attn_out, ((0, pad), (0, 0)))
        gate = jnp.pad(gate, ((0, pad), (0, 0)))
    m = attn_out.shape[0]
    bm = _legal_tile(m, _select_gate_q_tile(m, k))
    q_spec = pl.BlockSpec((bm, k), lambda i: (i, 0))
    s_spec = pl.BlockSpec((bm, 1), lambda i: (i, 0))
    q, s = pl.pallas_call(
        _sigmoid_gate_quant_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((m, k), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((m, 1), jnp.float32),
        ],
        grid=(m // bm,),
        in_specs=[q_spec, q_spec],
        out_specs=[q_spec, s_spec],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="pallas_sigmoid_gate_quant",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(attn_out, gate)
    if orig_m < 8:
        return q[:orig_m], s[:orig_m]
    return q, s


def sigmoid_gate_quant_reference(
    attn_out: jax.Array,
    gate: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Pure-JAX reference for attn_output * sigmoid(gate) + per-token FP8 quant."""
    a = attn_out.astype(jnp.float32)
    g = gate.astype(jnp.float32)
    y = (a * jax.nn.sigmoid(g)).astype(jnp.bfloat16)
    y_f32 = y.astype(jnp.float32)
    amax = jnp.max(jnp.abs(y_f32), axis=-1, keepdims=True)
    scale = amax * jnp.float32(1.0 / 448.0)
    scale = jnp.where(scale == 0.0, 1.0, scale)
    scale_inv = jnp.float32(1.0) / scale
    q = jnp.clip(y_f32 * scale_inv, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    return q, scale


def _get_sigmoid_gate_quant_op() -> Callable:
    global _sigmoid_gate_quant_op
    if _sigmoid_gate_quant_op is None:

        def _fn(
            attn_out: jax.Array,
            gate: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            m, k = attn_out.shape
            if m <= PAL_GATE_Q_MAX_ROWS and k % 128 == 0:
                return pallas_sigmoid_gate_quant(attn_out, gate)
            return sigmoid_gate_quant_reference(attn_out, gate)

        op = pallas.jax_op("pallas::qwen3_next_sigmoid_gate_quant", _fn)

        def _fake(attn_out: torch.Tensor, gate: torch.Tensor):
            return (
                attn_out.new_empty(attn_out.shape, dtype=torch.float8_e4m3fn),
                attn_out.new_empty((attn_out.shape[0], 1), dtype=torch.float32),
            )

        op.register_fake(_fake)
        _sigmoid_gate_quant_op = op
    linear_common._get_prequant_matmul_op()
    _get_sigmoid_gate_o_proj_op()
    return _sigmoid_gate_quant_op


def _get_sigmoid_gate_o_proj_op() -> Callable:
    global _sigmoid_gate_o_proj_op
    if _sigmoid_gate_o_proj_op is None:

        def _fn(
            attn_out: jax.Array,
            gate: jax.Array,
            weight: jax.Array,
            weight_scale: jax.Array,
        ) -> jax.Array:
            m, k = attn_out.shape
            if m <= PAL_GATE_Q_MAX_ROWS and k % 128 == 0:
                q, s = pallas_sigmoid_gate_quant(attn_out, gate)
            else:
                q, s = sigmoid_gate_quant_reference(attn_out, gate)
            return linear_common._prequant_matmul_jax(q, s, weight, weight_scale)

        op = pallas.jax_op("pallas::qwen3_next_sigmoid_gate_o_proj", _fn)

        def _fake(
            attn_out: torch.Tensor,
            gate: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ):
            return attn_out.new_empty(
                (attn_out.shape[0], weight.shape[-1]), dtype=torch.bfloat16
            )

        op.register_fake(_fake)
        _sigmoid_gate_o_proj_op = op
    return _sigmoid_gate_o_proj_op


def _is_fp8_per_channel_linear(proj: Any) -> bool:
    weight = getattr(proj, "weight", None)
    weight_scale = getattr(proj, "weight_scale", None)
    if weight is None or weight_scale is None:
        return False
    if not isinstance(weight, torch.Tensor) or not isinstance(
        weight_scale, torch.Tensor
    ):
        return False
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        return False
    if weight_scale.shape != (weight.shape[-1],):
        return False
    if not getattr(proj, WEIGHT_FLIPPED_ATTR, False):
        return False
    if getattr(proj, "bias", None) is not None and not getattr(
        proj, "skip_bias_add", False
    ):
        return False
    return True


def maybe_flip_gdn_in_proj_qkvz_weight(gdn_layer: Any) -> bool:
    """Flip GDN `in_proj_qkvz.weight` from `[n_out, n_in]` to contiguous `(k, n)`
    when PCP streaming is not enabled, eliminating per-step `.transpose(0, 1)`.
    """
    if gdn_layer is None or getattr(gdn_layer, "_pcp_streaming_configured", False):
        return False
    proj = getattr(gdn_layer, "in_proj_qkvz", None)
    if proj is None:
        return False
    if getattr(proj, WEIGHT_FLIPPED_ATTR, False):
        return True
    if not getattr(proj, KEEP_VLLM_LAYOUT_ATTR, False):
        return False
    if hasattr(proj, "weight_scale_inv"):
        return False
    weight = getattr(proj, "weight", None)
    weight_scale = getattr(proj, "weight_scale", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return False
    if weight.dtype != torch.float8_e4m3fn:
        return False
    if (
        not isinstance(weight_scale, torch.Tensor)
        or weight_scale.ndim != 1
        or weight_scale.shape != (weight.shape[0],)
    ):
        return False
    w_kn = weight.data.transpose(0, 1).contiguous()
    w_buf = torch.empty(
        w_kn.shape[0], w_kn.shape[1], dtype=w_kn.dtype, device=w_kn.device
    ).copy_(w_kn)
    param = torch.nn.Parameter(w_buf, requires_grad=False)
    if isinstance(proj, torch.nn.Module) and "weight" in getattr(
        proj, "_parameters", {}
    ):
        proj._parameters["weight"] = param
    else:
        proj.weight = param
    setattr(proj, KEEP_VLLM_LAYOUT_ATTR, False)
    setattr(proj, WEIGHT_FLIPPED_ATTR, True)
    if w_buf.device.type == "tpu":
        synchronize_tensors([proj.weight])
    return True


def _can_use_pal_gate_q(
    attn_layer: Any,
    attn_output: torch.Tensor,
    gate: torch.Tensor,
) -> bool:
    cached = getattr(attn_layer, "_pal_gate_q_eligible", None)
    if cached is False:
        return False
    if cached is None:
        o_proj = getattr(attn_layer, "o_proj", None)
        if o_proj is None or not _is_fp8_per_channel_linear(o_proj):
            return False
        if not getattr(o_proj, "input_is_parallel", True):
            return False
    if attn_output.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        return False
    if attn_output.shape != gate.shape or attn_output.shape[-1] % 128 != 0:
        return False
    return True


def _apply_pal_gate_q_o_proj(
    o_proj: Any,
    attn_output: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    orig_prefix = attn_output.shape[:-1]
    k = attn_output.shape[-1]
    attn_2d = attn_output.reshape(-1, k)
    gate_2d = gate.reshape(-1, k)
    output = _get_sigmoid_gate_o_proj_op()(
        attn_2d, gate_2d, o_proj.weight, o_proj.weight_scale
    )
    if getattr(o_proj, "reduce_results", False) and getattr(o_proj, "tp_size", 1) > 1:
        output = tensor_model_parallel_all_reduce(output)
    return output.reshape(*orig_prefix, output.shape[-1])


def _forward_qwen3_next_attn_from_qkv(
    attn_layer: Any,
    qkv: torch.Tensor,
    positions: torch.Tensor | None,
) -> torch.Tensor:
    q, k, v, gate = attn_layer._project_qkv_gate(qkv, positions)
    attn_output = attn_layer.attn(q, k, v)
    gather_idx = _step0_gather_indices
    if gather_idx is not None:
        attn_output = torch.index_select(attn_output, 0, gather_idx)
        if gate is not None:
            gate = torch.index_select(gate, 0, gather_idx)
    if gate is not None:
        if _can_use_pal_gate_q(attn_layer, attn_output, gate):
            return _apply_pal_gate_q_o_proj(attn_layer.o_proj, attn_output, gate)
        attn_output = attn_output * torch.sigmoid(gate)
    output, _ = attn_layer.o_proj(attn_output)
    return output


def patched_qwen3_next_attn_forward(
    self: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    prequant: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if prequant is None:
        qkv, _ = self.qkv_proj(hidden_states)
    else:
        proj = self.qkv_proj
        bias = None if proj.skip_bias_add else proj.bias
        qkv = proj.quant_method.apply_prequant(proj, prequant[0], prequant[1], bias)
    return _forward_qwen3_next_attn_from_qkv(self, qkv, positions)


def _extract_nk_bf16_weight(linear_layer: Any, hidden_dim: int) -> torch.Tensor:
    w = linear_layer.weight.data
    if getattr(linear_layer, WEIGHT_FLIPPED_ATTR, False) or (
        w.shape[1] != hidden_dim and w.shape[0] == hidden_dim
    ):
        w = w.transpose(0, 1)
    return w.to(torch.bfloat16)


def _fp8_gate_and_se_jax(
    hidden_states: jax.Array,
    w_gate_gu_fp8: jax.Array,
    s_gate_gu: jax.Array,
    w_se_gate_bf16: jax.Array,
    w_down_fp8: jax.Array,
    s_down: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    x_f32 = hidden_states.astype(jnp.float32)
    amax = jnp.max(jnp.abs(x_f32), axis=-1, keepdims=True)
    h_scale = amax * jnp.float32(1.0 / 448.0)
    h_scale = jnp.where(h_scale == 0.0, 1.0, h_scale)
    h_scale_inv = jnp.float32(1.0) / h_scale
    h_q = jnp.clip(x_f32 * h_scale_inv, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    seg_logit = jnp.dot(
        hidden_states,
        w_se_gate_bf16.reshape(-1),
        preferred_element_type=jnp.float32,
    )[:, None]
    seg_sig = jax.nn.sigmoid(seg_logit)
    n_inter = w_down_fp8.shape[0]
    n_gate = w_gate_gu_fp8.shape[1] - 2 * n_inter
    mm1 = jax.lax.dot_general(
        h_q,
        w_gate_gu_fp8,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    mm1_deq = mm1 * (h_scale * s_gate_gu[None, :])
    router_logits = mm1_deq[:, :n_gate].astype(jnp.bfloat16)
    gu = mm1_deq[:, n_gate:]
    g = gu[:, :n_inter]
    u = gu[:, n_inter:]
    inter = (jax.nn.silu(g) * u).astype(jnp.bfloat16).astype(jnp.float32)
    amax2 = jnp.max(jnp.abs(inter), axis=-1, keepdims=True)
    sc2 = amax2 * jnp.float32(1.0 / 448.0)
    sc2 = jnp.where(sc2 == 0.0, 1.0, sc2)
    sc2_inv = jnp.float32(1.0) / sc2
    q_inter = jnp.clip(inter * sc2_inv, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    s_inter = sc2 * seg_sig
    mm2 = jax.lax.dot_general(
        q_inter,
        w_down_fp8,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    se_out = (mm2 * (s_inter * s_down[None, :])).astype(jnp.bfloat16)
    return router_logits, se_out, h_q, h_scale


def _get_fp8_gate_and_se_op() -> Callable:
    global _fp8_gate_and_se_op
    if _fp8_gate_and_se_op is None:
        op = pallas.jax_op(
            "pallas::qwen3_next_fp8_gate_and_se",
            _fp8_gate_and_se_jax,
        )

        def _fake(
            hidden_states: torch.Tensor,
            w_gate_gu_fp8: torch.Tensor,
            s_gate_gu: torch.Tensor,
            w_se_gate_bf16: torch.Tensor,
            w_down_fp8: torch.Tensor,
            s_down: torch.Tensor,
        ):
            del s_gate_gu, w_se_gate_bf16, s_down
            n_tokens = hidden_states.shape[0]
            n_inter = int(w_down_fp8.shape[0])
            n_gate = int(w_gate_gu_fp8.shape[1]) - 2 * n_inter
            hidden_out = int(w_down_fp8.shape[1])
            return (
                hidden_states.new_empty((n_tokens, n_gate), dtype=torch.bfloat16),
                hidden_states.new_empty((n_tokens, hidden_out), dtype=torch.bfloat16),
                hidden_states.new_empty(
                    (n_tokens, hidden_out), dtype=torch.float8_e4m3fn
                ),
                hidden_states.new_empty((n_tokens, 1), dtype=torch.float32),
            )

        op.register_fake(_fake)
        _fp8_gate_and_se_op = op

    return _fp8_gate_and_se_op


_norm_fp8_gate_and_se_prequant_ops: dict[float, Callable] = {}
_norm_fp8_gate_and_se_bf16_ops: dict[float, Callable] = {}
_norm_quantize_rows_with_res_ops: dict[float, Callable] = {}
_norm_quantize_rows_no_res_ops: dict[float, Callable] = {}


def _norm_quantize_rows_with_res_jax(
    x: jax.Array,
    residual: jax.Array,
    norm_w_plus_one: jax.Array,
    *,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    res_out = (x + residual.astype(x.dtype)).astype(jnp.bfloat16)
    res_f32 = res_out.astype(jnp.float32)
    var = jnp.mean(jnp.square(res_f32), axis=-1, keepdims=True)
    h_f32 = (
        res_f32
        * jax.lax.rsqrt(var + jnp.float32(eps))
        * norm_w_plus_one.astype(jnp.float32)
    )
    h_bf16 = h_f32.astype(jnp.bfloat16)
    x_q, x_s = linear_common._quantize_rows_jax(h_bf16)
    return x_q, x_s, res_out


def _norm_quantize_rows_no_res_jax(
    x: jax.Array,
    norm_w_plus_one: jax.Array,
    *,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
    x_f32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    h_f32 = (
        x_f32
        * jax.lax.rsqrt(var + jnp.float32(eps))
        * norm_w_plus_one.astype(jnp.float32)
    )
    h_bf16 = h_f32.astype(jnp.bfloat16)
    return linear_common._quantize_rows_jax(h_bf16)


def _get_norm_quantize_rows_with_res_op(eps: float) -> Callable:
    key = float(eps)
    op = _norm_quantize_rows_with_res_ops.get(key)
    if op is None:
        tag = f"{key:.0e}".replace("-", "m").replace("+", "p").replace(".", "_")
        fn = functools.partial(_norm_quantize_rows_with_res_jax, eps=key)
        op = pallas.jax_op(f"pallas::qwen3_next_norm_quant_rows_res_{tag}", fn)

        def _fake(
            x: torch.Tensor,
            residual: torch.Tensor,
            norm_w_plus_one: torch.Tensor,
        ):
            del residual, norm_w_plus_one
            return (
                x.new_empty(x.shape, dtype=torch.float8_e4m3fn),
                x.new_empty((x.shape[0], 1), dtype=torch.float32),
                x.new_empty(x.shape, dtype=torch.bfloat16),
            )

        op.register_fake(_fake)
        _norm_quantize_rows_with_res_ops[key] = op
    return op


def _get_norm_quantize_rows_no_res_op(eps: float) -> Callable:
    key = float(eps)
    op = _norm_quantize_rows_no_res_ops.get(key)
    if op is None:
        tag = f"{key:.0e}".replace("-", "m").replace("+", "p").replace(".", "_")
        fn = functools.partial(_norm_quantize_rows_no_res_jax, eps=key)
        op = pallas.jax_op(f"pallas::qwen3_next_norm_quant_rows_nores_{tag}", fn)

        def _fake(
            x: torch.Tensor,
            norm_w_plus_one: torch.Tensor,
        ):
            del norm_w_plus_one
            return (
                x.new_empty(x.shape, dtype=torch.float8_e4m3fn),
                x.new_empty((x.shape[0], 1), dtype=torch.float32),
            )

        op.register_fake(_fake)
        _norm_quantize_rows_no_res_ops[key] = op
    return op


def _norm_fp8_gate_and_se_prequant_jax(
    x: jax.Array,
    residual: jax.Array,
    norm_w_plus_one: jax.Array,
    w_gate_gu_fp8: jax.Array,
    s_gate_gu: jax.Array,
    w_se_gate_bf16: jax.Array,
    w_down_fp8: jax.Array,
    s_down: jax.Array,
    *,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    res_out = (x + residual.astype(x.dtype)).astype(jnp.bfloat16)
    res_f32 = res_out.astype(jnp.float32)
    var = jnp.mean(jnp.square(res_f32), axis=-1, keepdims=True)
    h_f32 = res_f32 * jax.lax.rsqrt(var + jnp.float32(eps)) * norm_w_plus_one.astype(
        jnp.float32
    )
    h_bf16 = h_f32.astype(jnp.bfloat16)
    logits, se_out, h_q, h_scale = _fp8_gate_and_se_jax(
        h_bf16, w_gate_gu_fp8, s_gate_gu, w_se_gate_bf16, w_down_fp8, s_down
    )
    return logits, se_out, h_q, h_scale, res_out


def _norm_fp8_gate_and_se_bf16_jax(
    x: jax.Array,
    residual: jax.Array,
    norm_w_plus_one: jax.Array,
    w_gate_gu_fp8: jax.Array,
    s_gate_gu: jax.Array,
    w_se_gate_bf16: jax.Array,
    w_down_fp8: jax.Array,
    s_down: jax.Array,
    *,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    res_out = (x + residual.astype(x.dtype)).astype(jnp.bfloat16)
    res_f32 = res_out.astype(jnp.float32)
    var = jnp.mean(jnp.square(res_f32), axis=-1, keepdims=True)
    h_f32 = res_f32 * jax.lax.rsqrt(var + jnp.float32(eps)) * norm_w_plus_one.astype(
        jnp.float32
    )
    h_bf16 = h_f32.astype(jnp.bfloat16)
    logits, se_out, _, _ = _fp8_gate_and_se_jax(
        h_bf16, w_gate_gu_fp8, s_gate_gu, w_se_gate_bf16, w_down_fp8, s_down
    )
    return logits, se_out, h_bf16, res_out


def _get_norm_fp8_gate_and_se_prequant_op(eps: float) -> Callable:
    key = float(eps)
    op = _norm_fp8_gate_and_se_prequant_ops.get(key)
    if op is None:
        tag = f"{key:.0e}".replace("-", "m").replace("+", "p").replace(".", "_")
        fn = functools.partial(_norm_fp8_gate_and_se_prequant_jax, eps=key)
        op = pallas.jax_op(f"pallas::qwen3_next_norm_fp8_gate_se_pq_{tag}", fn)

        def _fake(
            x: torch.Tensor,
            residual: torch.Tensor,
            norm_w_plus_one: torch.Tensor,
            w_gate_gu_fp8: torch.Tensor,
            s_gate_gu: torch.Tensor,
            w_se_gate_bf16: torch.Tensor,
            w_down_fp8: torch.Tensor,
            s_down: torch.Tensor,
        ):
            n_tokens = x.shape[0]
            n_inter = int(w_down_fp8.shape[0])
            n_gate = int(w_gate_gu_fp8.shape[1]) - 2 * n_inter
            hidden_dim = int(w_down_fp8.shape[1])
            return (
                x.new_empty((n_tokens, n_gate), dtype=torch.bfloat16),
                x.new_empty((n_tokens, hidden_dim), dtype=torch.bfloat16),
                x.new_empty((n_tokens, hidden_dim), dtype=torch.float8_e4m3fn),
                x.new_empty((n_tokens, 1), dtype=torch.float32),
                x.new_empty((n_tokens, hidden_dim), dtype=torch.bfloat16),
            )

        op.register_fake(_fake)
        _norm_fp8_gate_and_se_prequant_ops[key] = op
    return op


def _get_norm_fp8_gate_and_se_bf16_op(eps: float) -> Callable:
    key = float(eps)
    op = _norm_fp8_gate_and_se_bf16_ops.get(key)
    if op is None:
        tag = f"{key:.0e}".replace("-", "m").replace("+", "p").replace(".", "_")
        fn = functools.partial(_norm_fp8_gate_and_se_bf16_jax, eps=key)
        op = pallas.jax_op(f"pallas::qwen3_next_norm_fp8_gate_se_bf16_{tag}", fn)

        def _fake(
            x: torch.Tensor,
            residual: torch.Tensor,
            norm_w_plus_one: torch.Tensor,
            w_gate_gu_fp8: torch.Tensor,
            s_gate_gu: torch.Tensor,
            w_se_gate_bf16: torch.Tensor,
            w_down_fp8: torch.Tensor,
            s_down: torch.Tensor,
        ):
            n_tokens = x.shape[0]
            n_inter = int(w_down_fp8.shape[0])
            n_gate = int(w_gate_gu_fp8.shape[1]) - 2 * n_inter
            hidden_dim = int(w_down_fp8.shape[1])
            return (
                x.new_empty((n_tokens, n_gate), dtype=torch.bfloat16),
                x.new_empty((n_tokens, hidden_dim), dtype=torch.bfloat16),
                x.new_empty((n_tokens, hidden_dim), dtype=torch.bfloat16),
                x.new_empty((n_tokens, hidden_dim), dtype=torch.bfloat16),
            )

        op.register_fake(_fake)
        _norm_fp8_gate_and_se_bf16_ops[key] = op
    return op


def _cpu_fp8_gate_and_se(
    hidden_states: torch.Tensor,
    w_gate_gu_fp8: torch.Tensor,
    s_gate_gu: torch.Tensor,
    w_se_gate_bf16: torch.Tensor,
    w_down_fp8: torch.Tensor,
    s_down: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_f32 = hidden_states.float()
    amax = x_f32.abs().amax(dim=-1, keepdim=True)
    h_scale = amax * (1.0 / 448.0)
    h_scale = torch.where(h_scale == 0.0, torch.ones_like(h_scale), h_scale)
    h_q = (x_f32 / h_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    seg_logit = torch.matmul(x_f32, w_se_gate_bf16.reshape(-1).float()).unsqueeze(-1)
    seg_sig = torch.sigmoid(seg_logit)
    n_inter = int(w_down_fp8.shape[0])
    n_gate = int(w_gate_gu_fp8.shape[1]) - 2 * n_inter
    mm1 = torch.matmul(h_q.float(), w_gate_gu_fp8.float())
    mm1_deq = mm1 * h_scale.float() * s_gate_gu.float().unsqueeze(0)
    router_logits = mm1_deq[:, :n_gate].to(torch.bfloat16)
    gu = mm1_deq[:, n_gate:]
    g = gu[:, :n_inter]
    u = gu[:, n_inter:]
    inter = (F.silu(g) * u).to(torch.bfloat16).float()
    amax2 = inter.abs().amax(dim=-1, keepdim=True)
    sc2 = amax2 * (1.0 / 448.0)
    sc2 = torch.where(sc2 == 0.0, torch.ones_like(sc2), sc2)
    q_inter = (inter / sc2).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    s_inter = sc2 * seg_sig.float()
    mm2 = torch.matmul(q_inter.float(), w_down_fp8.float())
    se_out = (mm2 * s_inter * s_down.float().unsqueeze(0)).to(torch.bfloat16)
    return router_logits, se_out


def _clear_moe_experts_internal_se_and_gate(moe: Any) -> None:
    experts = getattr(moe, "experts", None)
    if experts is not None:
        if hasattr(experts, "gate"):
            experts.gate = None
        if hasattr(experts, "shared_expert_gate"):
            experts.shared_expert_gate = None
        if hasattr(experts, "_fse_fuse_gate"):
            experts._fse_fuse_gate = False
        if hasattr(experts, "_shared_experts"):
            experts._shared_experts = None
        if hasattr(experts, "_select_forward") and callable(experts._select_forward):
            experts._forward_entry = experts._select_forward()
        moe_cfg = getattr(experts, "moe_config", None)
        if moe_cfg is not None:
            if hasattr(moe_cfg, "skip_final_all_reduce") and not getattr(
                moe, "_fused_se_is_replicated", False
            ):
                moe_cfg.skip_final_all_reduce = True
            if not (envs.TPU_MOE_SEQUENCE_PARALLEL or envs.TPU_SP_MIN_ROWS > 0):
                sp_prop = getattr(type(moe_cfg), "is_sequence_parallel", None)
                if isinstance(sp_prop, property):
                    if sp_prop.fset is not None:
                        moe_cfg.is_sequence_parallel = False
                    else:
                        p_cfg = getattr(moe_cfg, "moe_parallel_config", None)
                        if p_cfg is not None and hasattr(p_cfg, "sp_size"):
                            p_cfg.sp_size = 1
                elif hasattr(moe_cfg, "is_sequence_parallel"):
                    moe_cfg.is_sequence_parallel = False


def maybe_setup_fused_gate_se(moe: Any) -> bool:
    """Fuse router gate, shared_expert, and shared_expert_gate into 256-column-aligned
    FP8 weights and delete original separate weights so 0 extra HBM is used.

    Quantizes `gate.weight` per-channel to FP8 K-major `[K, n_gate]` and
    concatenates with `shared_expert.gate_up_proj.weight` `[K, 2*I]` into
    256-column-aligned `_fused_gate_gu_fp8_weight` (`[4096, 768]` at TP=8
    sharded or `[4096, 2560]` when `is_sequence_parallel=True`), stores
    `_fused_se_gate_weight` (`[4096]` BF16) and `_fused_se_down_fp8_weight`
    (`[I, 4096]` FP8), and folds `sigmoid(se_gate)` directly into `down_proj`'s
    per-token FP8 scale.
    """
    if getattr(moe, "_fused_gate_se_weight", None) is not None:
        return True
    if getattr(moe, "_fused_gate_se_checked", False):
        return False

    gate = getattr(moe, "gate", None)
    shared_expert = getattr(moe, "shared_expert", None)
    shared_expert_gate = getattr(moe, "shared_expert_gate", None)
    if gate is None or shared_expert is None or shared_expert_gate is None:
        moe._fused_gate_se_checked = True
        return False

    gate_up_proj = getattr(shared_expert, "gate_up_proj", None)
    down_proj = getattr(shared_expert, "down_proj", None)
    act_fn = getattr(shared_expert, "act_fn", None)
    if gate_up_proj is None or down_proj is None or act_fn is None:
        moe._fused_gate_se_checked = True
        return False

    for proj in (gate, gate_up_proj, shared_expert_gate):
        w = getattr(proj, "weight", None)
        if w is None or not isinstance(w, torch.Tensor) or w.ndim != 2:
            return False
        if getattr(proj, "bias", None) is not None:
            moe._fused_gate_se_checked = True
            return False

    qm_name = type(getattr(gate_up_proj, "quant_method", None)).__name__
    if qm_name == "VllmFp8LinearMethodTPU" and not getattr(
        gate_up_proj, WEIGHT_FLIPPED_ATTR, False
    ):
        return False

    if not (
        _is_fp8_per_channel_linear(gate_up_proj)
        and _is_fp8_per_channel_linear(down_proj)
    ):
        moe._fused_gate_se_checked = True
        return False

    hidden_dim = int(gate_up_proj.weight.shape[0])
    n_se_gu = int(gate_up_proj.weight.shape[1])
    if n_se_gu % 2 != 0:
        moe._fused_gate_se_checked = True
        return False
    n_inter = n_se_gu // 2
    if tuple(down_proj.weight.shape) != (n_inter, hidden_dim):
        moe._fused_gate_se_checked = True
        return False

    w_gate = _extract_nk_bf16_weight(gate, hidden_dim)
    w_se_gate = _extract_nk_bf16_weight(shared_expert_gate, hidden_dim)
    if (
        w_gate.shape[1] != hidden_dim
        or w_se_gate.shape[1] != hidden_dim
        or int(w_se_gate.shape[0]) != 1
    ):
        moe._fused_gate_se_checked = True
        return False

    n_gate = int(w_gate.shape[0])
    n_se_gate = 1

    w_gate_f32 = w_gate.float()
    w_gate_amax = w_gate_f32.abs().amax(dim=-1, keepdim=True)
    s_gate = torch.where(
        w_gate_amax == 0.0,
        torch.ones_like(w_gate_amax),
        w_gate_amax * (1.0 / 448.0),
    )
    w_gate_fp8_kn = (
        (w_gate_f32 / s_gate)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .transpose(0, 1)
        .contiguous()
    )
    s_gate_1d = s_gate.squeeze(-1).to(torch.float32).contiguous()

    w_gu_cat = torch.cat([w_gate_fp8_kn, gate_up_proj.weight.data], dim=1).contiguous()
    s_gu_cat = torch.cat(
        [s_gate_1d, gate_up_proj.weight_scale.data.float().reshape(-1)], dim=0
    ).contiguous()
    w_seg_1d = w_se_gate.reshape(hidden_dim).to(torch.bfloat16).contiguous()
    w_down_cat = down_proj.weight.data.contiguous()
    s_down_cat = down_proj.weight_scale.data.float().reshape(-1).contiguous()

    w_gu_buf = torch.empty_like(w_gu_cat).copy_(w_gu_cat)
    s_gu_buf = torch.empty_like(s_gu_cat).copy_(s_gu_cat)
    w_seg_buf = torch.empty_like(w_seg_1d).copy_(w_seg_1d)
    w_down_buf = torch.empty_like(w_down_cat).copy_(w_down_cat)
    s_down_buf = torch.empty_like(s_down_cat).copy_(s_down_cat)
    del (
        w_gate,
        w_se_gate,
        w_gate_f32,
        w_gate_fp8_kn,
        s_gate_1d,
        w_gu_cat,
        s_gu_cat,
        w_seg_1d,
        w_down_cat,
        s_down_cat,
    )

    p_w_gu = torch.nn.Parameter(w_gu_buf, requires_grad=False)
    p_s_gu = torch.nn.Parameter(s_gu_buf, requires_grad=False)
    p_w_seg = torch.nn.Parameter(w_seg_buf, requires_grad=False)
    p_w_down = torch.nn.Parameter(w_down_buf, requires_grad=False)
    p_s_down = torch.nn.Parameter(s_down_buf, requires_grad=False)

    if isinstance(moe, torch.nn.Module):
        moe.register_parameter("_fused_gate_gu_fp8_weight", p_w_gu)
        moe.register_parameter("_fused_gate_gu_fp8_scale", p_s_gu)
        moe.register_parameter("_fused_se_gate_weight", p_w_seg)
        moe.register_parameter("_fused_se_down_fp8_weight", p_w_down)
        moe.register_parameter("_fused_se_down_fp8_scale", p_s_down)
        moe._fused_gate_se_weight = p_w_gu
    else:
        moe._fused_gate_gu_fp8_weight = p_w_gu
        moe._fused_gate_gu_fp8_scale = p_s_gu
        moe._fused_se_gate_weight = p_w_seg
        moe._fused_se_down_fp8_weight = p_w_down
        moe._fused_se_down_fp8_scale = p_s_down
        moe._fused_gate_se_weight = p_w_gu

    moe._fused_gate_se_is_fp8 = True
    moe._fused_gate_se_sizes = (n_gate, n_se_gu, n_se_gate)
    moe._fused_se_is_replicated = bool(
        getattr(moe, "replicate_shared_expert", False)
        or getattr(moe, "is_sequence_parallel", False)
        or getattr(down_proj, "tp_size", 1) == 1
    )
    moe._fused_gate_se_checked = True

    for mod, attr in (
        (gate, "weight"),
        (gate_up_proj, "weight"),
        (gate_up_proj, "weight_scale"),
        (down_proj, "weight"),
        (down_proj, "weight_scale"),
        (shared_expert_gate, "weight"),
    ):
        if hasattr(mod, attr):
            delattr(mod, attr)

    if w_gu_buf.device.type == "tpu":
        synchronize_tensors(
            [
                moe._fused_gate_gu_fp8_weight,
                moe._fused_gate_gu_fp8_scale,
                moe._fused_se_gate_weight,
                moe._fused_se_down_fp8_weight,
                moe._fused_se_down_fp8_scale,
            ]
        )
    _get_fp8_gate_and_se_op()
    _get_norm_fp8_gate_and_se_prequant_op(1e-6)
    _get_norm_fp8_gate_and_se_bf16_op(1e-6)
    _clear_moe_experts_internal_se_and_gate(moe)
    logger.info_once(
        "Cat_FP8_256 fused [gate; shared_expert.gate_up_proj] into "
        f"_fused_gate_gu_fp8_weight={list(moe._fused_gate_gu_fp8_weight.shape)} "
        f"+ folded-scale down_proj={list(moe._fused_se_down_fp8_weight.shape)} "
        "and freed original weights."
    )
    return True


def compute_fused_gate_and_shared_expert(
    moe: Any,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute router_logits and shared_expert output from fused FP8 weights."""
    if hidden_states.device.type == "tpu":
        standalone_op = _get_fp8_gate_and_se_op()
        logits, se_out, h_q, h_scale = standalone_op(
            hidden_states,
            moe._fused_gate_gu_fp8_weight,
            moe._fused_gate_gu_fp8_scale,
            moe._fused_se_gate_weight,
            moe._fused_se_down_fp8_weight,
            moe._fused_se_down_fp8_scale,
        )
        routed_experts = getattr(
            getattr(moe, "experts", None),
            "routed_experts",
            getattr(moe, "experts", None),
        )
        if routed_experts is not None and hidden_states.shape[0] <= envs.TPU_MOE_LOWTOK_MAX_TOKENS:
            routed_experts._prequant_x = (h_q, h_scale)
        return logits, se_out
    return _cpu_fp8_gate_and_se(
        hidden_states,
        moe._fused_gate_gu_fp8_weight,
        moe._fused_gate_gu_fp8_scale,
        moe._fused_se_gate_weight,
        moe._fused_se_down_fp8_weight,
        moe._fused_se_down_fp8_scale,
    )


def patched_sparse_moe_forward(
    self: Any,
    hidden_states: torch.Tensor,
    already_sequence_parallel: bool = False,
    *,
    prenorm_outputs: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if (
        not getattr(self, "_fused_gate_se_checked", False)
        and not torch.compiler.is_compiling()
    ):
        maybe_setup_fused_gate_se(self)
    if getattr(self, "_fused_gate_se_weight", None) is None:
        return UPSTREAM_SPARSE_MOE_FORWARD(
            self,
            hidden_states,
            already_sequence_parallel=already_sequence_parallel,
        )

    orig_shape = hidden_states.shape
    hidden_dim = hidden_states.shape[-1]
    hidden_states = hidden_states.view(-1, hidden_dim)
    from vllm_torchtpu.layers.adapter import sequence_parallel

    is_sp = sequence_parallel.is_active()
    tp_size = getattr(self, "tp_size", 1)

    if prenorm_outputs is not None:
        router_logits, shared_output = prenorm_outputs
    else:
        router_logits, shared_output = compute_fused_gate_and_shared_expert(
            self, hidden_states
        )
    final_hidden_states = self.experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
    )
    # When Fused EP v3 owns dispatch (`USE_MOE_FUSED_EP_KERNEL=1`), `_apply_owning_dispatch_v3`
    # already performs `ep.combine` (all-reduce for SP=0, or in-kernel combine for SP=1).
    # When `USE_MOE_FUSED_EP_KERNEL=0` and `se_is_replicated=False`, `skip_final_all_reduce=True`
    # leaves `final_hidden_states` unreduced across EP/TP so it can be all-reduced together with `shared_output`.
    from vllm_torchtpu.layers.adapter.fused_moe_ep_v3 import (
        fused_moe_ep_supported as fused_moe_ep_v3_supported,
    )

    routed_experts = getattr(self.experts, "routed_experts", self.experts)
    v3_owns_combine = envs.USE_MOE_FUSED_EP_KERNEL and (
        fused_moe_ep_v3_supported(routed_experts)
        or fused_moe_ep_v3_supported(getattr(routed_experts, "quant_method", None))
    )
    se_is_replicated = bool(
        getattr(self, "_fused_se_is_replicated", False)
        or getattr(self, "replicate_shared_expert", False)
        or getattr(self, "is_sequence_parallel", False)
    )
    if se_is_replicated:
        final_hidden_states = final_hidden_states + shared_output
    else:
        if tp_size > 1 and v3_owns_combine:
            shared_output = tensor_model_parallel_all_reduce(shared_output)
            final_hidden_states = final_hidden_states + shared_output
        else:
            final_hidden_states = final_hidden_states + shared_output
            if tp_size > 1:
                final_hidden_states = tensor_model_parallel_all_reduce(
                    final_hidden_states
                )
    if is_sp:
        return final_hidden_states.view(-1, *orig_shape[1:])
    return final_hidden_states.view(orig_shape)


def _patched_gemma_rmsnorm_forward_native(
    self: GemmaRMSNorm,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    if residual is not None:
        x = (
            x + residual.to(orig_dtype)
            if residual.dtype != orig_dtype
            else x + residual
        )
        residual = x
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.variance_epsilon)
    w_plus_one = getattr(self, "_tpu_weight_plus_one", None)
    if w_plus_one is None:
        w_plus_one = 1.0 + self.weight.float()
    x = x * w_plus_one
    x = x.to(orig_dtype)
    return x if residual is None else (x, residual)


def setup_qwen3_next_v6_after_loading(model: Any) -> None:
    """Post-weight-load setup for fused router/shared-expert FP8 weights, Pallas sigmoid-gate o_proj, GDN weight layout, and FP8 LM head."""
    if not hasattr(model, "modules"):
        return
    is_mtp_wrapper = type(model).__name__ in (
        "Qwen3NextMTP",
        "Qwen3_5MTP",
        "Qwen3_5MoeMTP",
    )
    quant_shared_lm_head = os.environ.get(
        "TPU_QUANTIZE_SHARED_LM_HEAD", "1"
    ).strip().lower() not in ("0", "false", "no", "off")
    hoisted_norm_tensors: list[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, GemmaRMSNorm):
            w_plus_one = 1.0 + module.weight.data.float()
            module._tpu_weight_plus_one = w_plus_one
            eps = float(getattr(module, "variance_epsilon", 1e-6))
            _get_norm_quantize_rows_with_res_op(eps)
            _get_norm_quantize_rows_no_res_op(eps)
            if w_plus_one.device.type == "tpu":
                hoisted_norm_tensors.append(w_plus_one)
        if isinstance(module, qwen3_next.Qwen3NextSparseMoeBlock):
            maybe_setup_fused_gate_se(module)
        elif isinstance(module, qwen3_next.Qwen3NextAttention):
            _get_sigmoid_gate_quant_op()
            o_proj = getattr(module, "o_proj", None)
            module._pal_gate_q_eligible = bool(
                o_proj is not None
                and _is_fp8_per_channel_linear(o_proj)
                and getattr(o_proj, "input_is_parallel", True)
            )
        elif isinstance(module, qwen3_next.Qwen3NextDecoderLayer) or (
            type(module).__name__ == "Qwen3_5DecoderLayer"
        ):
            if getattr(module, "layer_scale", False):
                module._tpu_attn_layer_scale_2d = (
                    module.attn_layer_scale.data.to(torch.bfloat16)[0] + 1
                )
                module._tpu_ffn_layer_scale_2d = (
                    module.ffn_layer_scale.data.to(torch.bfloat16)[0] + 1
                )
                if module.attn_layer_scale.device.type == "tpu":
                    hoisted_norm_tensors.extend(
                        [
                            module._tpu_attn_layer_scale_2d,
                            module._tpu_ffn_layer_scale_2d,
                        ]
                    )
            if getattr(module, "layer_type", None) == "linear_attention":
                linear_attn = getattr(module, "linear_attn", None)
                if linear_attn is not None:
                    maybe_flip_gdn_in_proj_qkvz_weight(linear_attn)
                proj = getattr(linear_attn, "in_proj_qkvz", None)
            else:
                proj = getattr(getattr(module, "self_attn", None), "qkv_proj", None)
            w = getattr(proj, "weight", None)
            ws = getattr(proj, "weight_scale", None)
            module._pre_norm_quant_eligible = bool(
                isinstance(getattr(module, "input_layernorm", None), GemmaRMSNorm)
                and isinstance(w, torch.Tensor)
                and w.dtype == torch.float8_e4m3fn
                and isinstance(ws, torch.Tensor)
                and ws.ndim == 1
                and hasattr(getattr(proj, "quant_method", None), "apply_prequant")
            )
        elif hasattr(module, "in_proj_qkvz") and hasattr(
            module, "_pcp_streaming_configured"
        ):
            maybe_flip_gdn_in_proj_qkvz_weight(module)
        elif (
            type(module).__name__ == "ParallelLMHead"
            and not is_mtp_wrapper
            and quant_shared_lm_head
        ):
            from vllm_torchtpu.runner import greedy_top_tokens

            greedy_top_tokens.quantize_lm_head(module, mode="fp8", in_place=True)
    if hoisted_norm_tensors:
        synchronize_tensors(hoisted_norm_tensors)
    import gc
    gc.collect()


def install_qwen3_next_v6_patches() -> None:
    """Install post-load weight fusion hooks and Pallas kernel wrappers for Qwen3Next / Qwen3.5."""
    global _V6_PATCHES_INSTALLED
    if _V6_PATCHES_INSTALLED:
        return

    orig_attn_init = qwen3_next.Qwen3NextAttention.__init__

    @functools.wraps(orig_attn_init)
    def _patched_attn_init(self, *args, **kwargs):
        orig_attn_init(self, *args, **kwargs)
        _get_sigmoid_gate_quant_op()

    GemmaRMSNorm.forward_native = _patched_gemma_rmsnorm_forward_native
    qwen3_next.Qwen3NextAttention.__init__ = _patched_attn_init
    qwen3_next.Qwen3NextAttention.forward = patched_qwen3_next_attn_forward
    if qwen3_next.Qwen3NextSparseMoeBlock.forward is UPSTREAM_SPARSE_MOE_FORWARD:
        qwen3_next.Qwen3NextSparseMoeBlock.forward = patched_sparse_moe_forward

    if not getattr(base_loader, "_qwen3_next_v6_process_patched", False):
        orig_process = base_loader.process_weights_after_loading

        @functools.wraps(orig_process)
        def _process_weights_after_loading(model, model_config, target_device):
            res = orig_process(model, model_config, target_device)
            setup_qwen3_next_v6_after_loading(model)
            return res

        base_loader.process_weights_after_loading = _process_weights_after_loading
        base_loader._qwen3_next_v6_process_patched = True

    _V6_PATCHES_INSTALLED = True


def _sp_chunk(x: torch.Tensor, full_num_tokens: int) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return x
    tp_rank = get_tensor_model_parallel_rank()
    chunk = full_num_tokens // tp_size
    return x.narrow(0, tp_rank * chunk, chunk)


def _all_gather_hidden_and_residual(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    full_num_tokens: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if residual is None:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states.narrow(0, 0, full_num_tokens)
        return hidden_states, None

    combined_states = torch.cat([hidden_states, residual], dim=-1)
    combined_states = tensor_model_parallel_all_gather(combined_states, 0)
    combined_states = combined_states.narrow(0, 0, full_num_tokens)
    hidden_states, residual = combined_states.split([hidden_size, hidden_size], dim=-1)
    return hidden_states, residual


def patched_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor | None = None,
    **kwargs: object,
):
    from vllm_torchtpu.layers.adapter import sequence_parallel as _sp
    from vllm_torchtpu.layers.adapter import small_all_reduce_op as _sar

    sp = False
    if self.use_attn_reduce_scatter_for_moe and positions.shape[-1] >= _sp.min_rows():
        sp = True
    input_is_sequence_parallel = sp and residual is not None

    if hasattr(self, "_tpu_sp_rank"):
        positions = positions + (self._tpu_sp_rank * 0)[0]
    if _sar._RANK is not None:
        positions = positions + (_sar._RANK * 0)[0]

    prequant = None
    can_fuse_pre_norm_quant = (
        hidden_states.device.type == "tpu"
        and getattr(self, "_pre_norm_quant_eligible", False)
        and _sp.fp8_gather()
        and os.environ.get("TPU_FUSE_PRE_ATTN_NORM_QUANT", "1").strip().lower()
        not in ("0", "false", "no", "off")
    )
    if can_fuse_pre_norm_quant:
        orig_shape = hidden_states.shape
        x_2d = hidden_states.reshape(-1, orig_shape[-1])
        norm = self.input_layernorm
        w_plus_one = getattr(norm, "_tpu_weight_plus_one", None)
        if w_plus_one is None:
            w_plus_one = 1.0 + norm.weight.float()
        eps = float(getattr(norm, "variance_epsilon", 1e-6))
        if residual is None:
            residual = hidden_states
            x_q, x_s = _get_norm_quantize_rows_no_res_op(eps)(x_2d, w_plus_one)
        else:
            res_2d = residual.reshape(-1, orig_shape[-1])
            x_q, x_s, res_out = _get_norm_quantize_rows_with_res_op(eps)(
                x_2d, res_2d, w_plus_one
            )
            residual = res_out.view(orig_shape)
        if input_is_sequence_parallel:
            x_q = tensor_model_parallel_all_gather(x_q, 0)
            x_s = tensor_model_parallel_all_gather(x_s, 0)
        prequant = (x_q, x_s)
        hidden_states = x_q
    else:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if input_is_sequence_parallel and _sp.fp8_gather():
            x_q, x_s = linear_common.quantize_rows(hidden_states)
            x_q = tensor_model_parallel_all_gather(x_q, 0)
            x_s = tensor_model_parallel_all_gather(x_s, 0)
            prequant = (x_q, x_s)
            hidden_states = x_q
        elif input_is_sequence_parallel:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            prequant=prequant,
        )
    elif self.layer_type == "full_attention":
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            prequant=prequant,
        )
        gather_idx = _step0_gather_indices
        if gather_idx is not None:
            residual = torch.index_select(residual, 0, gather_idx)
            sp = False
    else:
        raise ValueError("Invalid layer_type")

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            s_attn = getattr(self, "_tpu_attn_layer_scale_2d", None)
            if s_attn is None or s_attn.dtype != hidden_states.dtype:
                s_attn = self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
            hidden_states = hidden_states * s_attn
        else:
            hidden_states = hidden_states * (
                self.attn_layer_scale.to(hidden_states.dtype) + 1
            )

    if self.use_attn_reduce_scatter_for_moe:
        if sp:
            hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
            if not input_is_sequence_parallel:
                tp = get_tensor_model_parallel_world_size()
                if hasattr(self, "_tpu_sp_rank"):
                    residual = (
                        residual.reshape(tp, -1, residual.shape[-1])
                        .index_select(0, self._tpu_sp_rank)
                        .squeeze(0)
                    )
                else:
                    residual = _sp_chunk(residual, positions.shape[-1])
        else:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    # Fully Connected
    mlp = self.mlp
    norm = self.post_attention_layernorm
    sp_active = bool(self.use_attn_reduce_scatter_for_moe)
    if (
        not getattr(mlp, "_fused_gate_se_checked", False)
        and not torch.compiler.is_compiling()
        and isinstance(mlp, qwen3_next.Qwen3NextSparseMoeBlock)
    ):
        maybe_setup_fused_gate_se(mlp)
    can_fuse_post_norm_moe = (
        hidden_states.device.type == "tpu"
        and residual is not None
        and getattr(mlp, "_fused_gate_gu_fp8_weight", None) is not None
        and isinstance(norm, GemmaRMSNorm)
        and os.environ.get("TPU_FUSE_POST_ATTN_NORM_MOE", "1").strip().lower()
        not in ("0", "false", "no", "off")
    )
    if can_fuse_post_norm_moe:
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        x_2d = hidden_states.reshape(-1, hidden_dim)
        res_2d = residual.reshape(-1, hidden_dim)
        w_plus_one = getattr(norm, "_tpu_weight_plus_one", None)
        if w_plus_one is None:
            w_plus_one = 1.0 + norm.weight.float()
        eps = float(getattr(norm, "variance_epsilon", 1e-6))
        routed_experts = getattr(
            getattr(mlp, "experts", None),
            "routed_experts",
            getattr(mlp, "experts", None),
        )
        from vllm_torchtpu.layers.adapter import moe_routing

        use_prequant_moe = (
            not sp
            and x_2d.shape[0] <= envs.TPU_MOE_LOWTOK_MAX_TOKENS
            and routed_experts is not None
            and getattr(
                getattr(routed_experts, "quant_method", None),
                "_moe_lowtok_scores_op",
                None,
            )
            is not None
            and not getattr(routed_experts, "enable_eplb", False)
            and moe_routing._SIMULATION_STRATEGY is None
        )
        if use_prequant_moe:
            op = _get_norm_fp8_gate_and_se_prequant_op(eps)
            logits, se_out, h_q, h_scale, res_out = op(
                x_2d,
                res_2d,
                w_plus_one,
                mlp._fused_gate_gu_fp8_weight,
                mlp._fused_gate_gu_fp8_scale,
                mlp._fused_se_gate_weight,
                mlp._fused_se_down_fp8_weight,
                mlp._fused_se_down_fp8_scale,
            )
            routed_experts._prequant_x = (h_q, h_scale)
            residual = res_out.view(orig_shape)
            hidden_states = patched_sparse_moe_forward(
                mlp,
                res_out.view(orig_shape),
                already_sequence_parallel=sp_active,
                prenorm_outputs=(logits, se_out),
            )
        else:
            op = _get_norm_fp8_gate_and_se_bf16_op(eps)
            logits, se_out, h_bf16, res_out = op(
                x_2d,
                res_2d,
                w_plus_one,
                mlp._fused_gate_gu_fp8_weight,
                mlp._fused_gate_gu_fp8_scale,
                mlp._fused_se_gate_weight,
                mlp._fused_se_down_fp8_weight,
                mlp._fused_se_down_fp8_scale,
            )
            residual = res_out.view(orig_shape)
            hidden_states = patched_sparse_moe_forward(
                mlp,
                h_bf16.view(orig_shape),
                already_sequence_parallel=sp_active,
                prenorm_outputs=(logits, se_out),
            )
    else:
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.use_attn_reduce_scatter_for_moe:
            hidden_states = self.mlp(
                hidden_states,
                already_sequence_parallel=True,
            )
        else:
            hidden_states = self.mlp(hidden_states)

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            s_ffn = getattr(self, "_tpu_ffn_layer_scale_2d", None)
            if s_ffn is None or s_ffn.dtype != hidden_states.dtype:
                s_ffn = self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
            hidden_states = hidden_states * s_ffn
        else:
            assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                f"shape must be the same {len(hidden_states.shape)}, "
                f"{len(self.ffn_layer_scale.shape)}"
            )
            hidden_states = hidden_states * (
                self.ffn_layer_scale.to(hidden_states.dtype) + 1
            )

    return hidden_states, residual


def patched_model_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    full_num_tokens = positions.shape[-1]
    from vllm_torchtpu.layers.adapter import sequence_parallel as _sp

    sp = positions.shape[-1] >= _sp.min_rows()

    aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
    for layer_idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer,
    ):
        if (
            sp
            and residual is not None
            and not layer.use_attn_reduce_scatter_for_moe
        ):
            hidden_states, residual = _all_gather_hidden_and_residual(
                hidden_states,
                residual,
                full_num_tokens,
                self.config.hidden_size,
            )
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        if (layer_idx + 1) in self.aux_hidden_state_layers and sp:
            hidden_states, residual = _all_gather_hidden_and_residual(
                hidden_states,
                residual,
                full_num_tokens,
                self.config.hidden_size,
            )
        self._maybe_add_hidden_state(
            aux_hidden_states, layer_idx + 1, hidden_states, residual
        )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )
    if sp and self.layers[self.end_layer - 1].use_attn_reduce_scatter_for_moe:
        hidden_states, residual = _all_gather_hidden_and_residual(
            hidden_states,
            residual,
            full_num_tokens,
            self.config.hidden_size,
        )
    hidden_states, _ = self.norm(hidden_states, residual)
    if aux_hidden_states:
        return hidden_states, aux_hidden_states
    return hidden_states


def patched_mtp_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_idx: int = 0,
) -> torch.Tensor | IntermediateTensors:
    if get_pp_group().is_first_rank:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    current_step_idx = spec_step_idx % self.num_mtp_layers
    mtp_layer = self.layers[current_step_idx]
    if _mtp_megakernel is not None and residual is None:
        # One megakernel step runs the layer and `self.norm`.
        return _mtp_megakernel.forward_layer(
            mtp_layer, hidden_states, _step0_gather_indices
        )
    from vllm_torchtpu.layers.adapter import sequence_parallel as _sp

    sp = (
        mtp_layer.use_attn_reduce_scatter_for_moe
        and _step0_gather_indices is None
        and positions.shape[-1] >= _sp.min_rows()
    )

    hidden_states, residual = mtp_layer(
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    if sp:
        full_num_tokens = positions.shape[-1]
        hidden_states, residual = _all_gather_hidden_and_residual(
            hidden_states,
            residual,
            full_num_tokens,
            self.config.hidden_size,
        )
    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


def maybe_patch_qwen3_next(*args: object, **kwargs: object) -> bool | None:
    """Install Cat_FP8_256, PalGateQ, and GDN/LM-head weight hooks for Qwen3Next / Qwen3.5."""
    install_qwen3_next_v6_patches()
    from vllm_torchtpu.layers.adapter import sequence_parallel as _sp

    if not _sp.enabled():
        return False

    if _sp.fp8_gather():
        linear_common.prebuild_prequant_ops()
        _get_norm_quantize_rows_with_res_op(1e-6)
        _get_norm_quantize_rows_no_res_op(1e-6)

    def _with_rank_buffer(original_init):
        @functools.wraps(original_init)
        def __init__(self, *a, **kw):
            original_init(self, *a, **kw)
            device = self.input_layernorm.weight.device
            self._tpu_sp_rank = torch.tensor(
                [get_tensor_model_parallel_rank()],
                dtype=torch.int32,
                device=device,
            )

        __init__._tpu_sp_rank_init = True
        return __init__

    from vllm.model_executor.models import qwen3_5 as _q35
    from vllm.model_executor.models import qwen3_5_mtp as _q35_mtp

    for layer_cls in (qwen3_next.Qwen3NextDecoderLayer, _q35.Qwen3_5DecoderLayer):
        if "__init__" in vars(layer_cls) and not getattr(
            layer_cls.__init__, "_tpu_sp_rank_init", False
        ):
            layer_cls.__init__ = _with_rank_buffer(layer_cls.__init__)

    qwen3_next.Qwen3NextDecoderLayer.forward = patched_decoder_layer_forward
    qwen3_next.Qwen3NextModel.forward = patched_model_forward
    qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward = patched_mtp_forward
    _q35_mtp.Qwen3_5MultiTokenPredictor.forward = patched_mtp_forward
    logger.info(
        "Applied TPU patch: Qwen3Next row-gated sequence-parallel forward (>= %d rows).",
        _sp.min_rows(),
    )
    return None
