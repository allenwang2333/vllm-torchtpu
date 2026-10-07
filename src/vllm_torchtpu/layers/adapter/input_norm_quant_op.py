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
"""The decoder layer's input RMS norm and its fp8 projection's activation quant
fused.

With ``TPU_INPUT_NORM_QUANT`` set, :func:`install` marks every Qwen3-Next /
Qwen3.5 decoder layer whose input projection (GDN ``in_proj_qkvz`` or attention
``qkv_proj``) is a per-channel fp8 linear. Such a layer runs the (residual add
+) ``GemmaRMSNorm`` and the per-row fp8 quantization of its output as one
kernel (``kernels/experimental/rms_norm_quant``) and hands the fp8 rows and
scales to the projection, which runs the pre-quantized fp8 matmul -- instead of
the norm's XLA fusions, the linear's separate abs-max and convert fusions and
then the matmul.
"""

from collections.abc import Callable

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental import rms_norm_quant as rnq
from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LAYER_ATTR = "_tpu_input_norm_quant"

_ops: dict[tuple[float, bool, bool, bool, str], Callable] = {}


def _build_op(
    eps: float, has_res: bool, emit_y: bool, has_rstd: bool = False, ba: bool = False
):
    gamma = envs.TPU_INPUT_NORM_QUANT_GAMMA
    key = (eps, has_res, emit_y, has_rstd, gamma, ba)
    if key in _ops:
        return _ops[key]

    A = jax.Array

    def run(x, r, w, rstd, wba=None):
        out = rnq.rms_norm_quant(
            x, r, w, eps, rstd, emit_y=emit_y, gamma=gamma, wba=wba
        )
        if wba is not None:  # (y, ro, q, s, ba) -> ba after y, as the fake
            y, ro, q, s, b = out
            return tuple(o for o in (y, b, ro, q, s) if o is not None)
        return tuple(o for o in out if o is not None)

    # pallas.jax_op reads the signature and the return annotation, so each
    # variant spells its inputs and outputs out.
    n_out = 2 + int(emit_y) + int(has_res) + int(ba)
    ret = tuple[(A,) * n_out]
    if ba and has_res and has_rstd:

        def fn(x: A, r: A, w: A, rstd: A, wba: A) -> ret:
            return run(x, r, w, rstd, wba)

    elif ba and has_res:

        def fn(x: A, r: A, w: A, wba: A) -> ret:
            return run(x, r, w, None, wba)

    elif ba and has_rstd:

        def fn(x: A, w: A, rstd: A, wba: A) -> ret:
            return run(x, None, w, rstd, wba)

    elif ba:

        def fn(x: A, w: A, wba: A) -> ret:
            return run(x, None, w, None, wba)

    elif has_res and has_rstd:

        def fn(x: A, r: A, w: A, rstd: A) -> ret:
            return run(x, r, w, rstd)

    elif has_res:

        def fn(x: A, r: A, w: A) -> ret:
            return run(x, r, w, None)

    elif has_rstd:

        def fn(x: A, w: A, rstd: A) -> ret:
            return run(x, None, w, rstd)

    else:

        def fn(x: A, w: A) -> ret:
            return run(x, None, w, None)

    op = pallas.jax_op(f"pallas::input_norm_quant_{len(_ops)}", fn)

    def fake(x, *rest):
        outs = []
        if emit_y:
            outs.append(x.new_empty(x.shape))
        if ba:
            outs.append(x.new_empty((x.shape[0], rest[-1].shape[0])))
        if has_res:
            outs.append(x.new_empty(x.shape))
        outs.append(x.new_empty(x.shape, dtype=torch.float8_e4m3fn))
        outs.append(x.new_empty((x.shape[0], 1), dtype=torch.float32))
        return tuple(outs)

    op.register_fake(fake)
    _ops[key] = op
    return op


def _projection(layer: torch.nn.Module):
    if getattr(layer, "layer_type", None) == "linear_attention":
        attn = layer.linear_attn
        return getattr(attn, "in_proj_qkvz", None), True
    return getattr(getattr(layer, "self_attn", None), "qkv_proj", None), False


def unsupported_reason(layer: torch.nn.Module) -> str | None:
    norm = getattr(layer, "input_layernorm", None)
    if type(norm).__name__ != "GemmaRMSNorm":
        return f"the input norm is {type(norm).__name__}, not GemmaRMSNorm"
    proj, _ = _projection(layer)
    if proj is None:
        return "no in_proj_qkvz / qkv_proj"
    if proj.weight.dtype != torch.float8_e4m3fn:
        return f"the projection weight is {proj.weight.dtype}, not float8_e4m3fn"
    n = proj.weight.shape[0 if getattr(proj, KEEP_VLLM_LAYOUT_ATTR, False) else 1]
    if tuple(proj.weight_scale.shape) != (n,):
        return "the projection scale is not per-channel"
    if getattr(proj, "bias", None) is not None:
        return "the projection has a bias"
    return None


def install(model: torch.nn.Module) -> int:
    """Mark the decoder layers whose input norm feeds a per-channel fp8
    projection. Call after the weights are processed."""
    count = 0
    for module in model.modules():
        if not hasattr(module, "input_layernorm") or not hasattr(module, "layer_type"):
            continue
        # The fused norm runs in the attention-MoE overlap decoder forward
        # (TPU_ATTN_MOE_OVERLAP), which the MTP drafter layers do not take.
        if not getattr(module, "_tpu_attn_moe_overlap", False):
            continue
        reason = unsupported_reason(module)
        if reason is not None:
            logger.info_once("TPU_INPUT_NORM_QUANT: skipping a layer: %s", reason)
            continue
        eps = float(module.input_layernorm.variance_epsilon)
        _, is_gdn = _projection(module)
        fold = _fold_ba(module)
        if fold:
            # [n_ba, k] copy (128 KiB) of the (k, n) in_proj_ba weight.
            module.linear_attn._tpu_in_proj_ba_t = (
                module.linear_attn.in_proj_ba.weight.detach().t().contiguous()
            )
        for has_res in (False, True):
            for has_rstd in {_exact(), _exact(EXACT_MAX_ROWS + 1)}:
                _build_op(
                    eps, has_res, emit_y=is_gdn and not fold, has_rstd=has_rstd, ba=fold
                )
        linear_common._get_prequant_matmul_op()
        setattr(module, LAYER_ATTR, True)
        count += 1
    logger.info(
        "TPU_INPUT_NORM_QUANT: %d decoder layers run the input norm and the "
        "input projection's fp8 quantization as one kernel",
        count,
    )
    return count


# Mode 3's row fork, the same bucket boundary TPU_FUSED_QK_NORM_ROPE forks at
# (the backbone is re-traced per bucket group, so the fork is legal there).
EXACT_MAX_ROWS = 64


def _exact(rows: int | None = None) -> bool:
    """Mode 1: always exact; 2: never; 3: exact on decode buckets (<= 64 rows),
    the kernel's own statistic above (one HBM pass fewer in prefill)."""
    mode = envs.TPU_INPUT_NORM_QUANT
    if mode == 3:
        return rows is None or rows <= EXACT_MAX_ROWS
    return mode != 2


def _fold_ba(layer: torch.nn.Module) -> bool:
    """TPU_INPUT_NORM_QUANT_FOLD_BA: GDN's in_proj_ba (bf16, (k, n), no bias)
    runs inside the norm kernel, so the normed rows never leave it."""
    if not envs.TPU_INPUT_NORM_QUANT_FOLD_BA:
        return False
    if getattr(layer, "layer_type", None) != "linear_attention":
        return False
    attn = layer.linear_attn
    ba = getattr(attn, "in_proj_ba", None)
    return (
        ba is not None
        and not hasattr(attn, "in_proj_qkv")
        and ba.weight.dtype == torch.bfloat16
        and getattr(ba, "bias", None) is None
        and getattr(ba, "_tpu_weight_flipped", False)
        and not getattr(attn, "_pcp_streaming_configured", False)
    )


def norm_quant(layer: torch.nn.Module, hidden_states, residual):
    """``layer.input_layernorm(hidden_states, residual)`` plus the fp8 rows of
    its output. Returns (normed or None, residual, (q, scale))."""
    norm = layer.input_layernorm
    eps = float(norm.variance_epsilon)
    _, is_gdn = _projection(layer)
    fold = _fold_ba(layer)
    exact = _exact(hidden_states.shape[0])
    extra = ()
    if exact:
        # The statistic exactly as ir.ops.(fused_add_)rms_norm computes it.
        s = hidden_states.to(torch.float32)
        if residual is not None:
            s = s + residual.to(torch.float32)
        extra = (torch.rsqrt(s.pow(2).mean(dim=-1, keepdim=True) + eps),)
    if fold:
        extra = extra + (layer.linear_attn._tpu_in_proj_ba_t,)
    emit_y = is_gdn and not fold
    gw = norm.weight
    if envs.TPU_INPUT_NORM_QUANT_GAMMA == "xla":
        # (w + 1) in f32 exactly as GemmaRMSNorm forms it; an XLA fusion output,
        # so the kernel reads it from VMEM instead of a staged parameter.
        gw = norm.weight.float() + 1.0
    if residual is None:
        out = _build_op(eps, False, emit_y, exact, fold)(
            hidden_states, gw, *extra
        )
        new_res = hidden_states
    else:
        out = _build_op(eps, True, emit_y, exact, fold)(
            hidden_states, residual, gw, *extra
        )
    out = list(out)
    y = out.pop(0) if emit_y else None
    ba = out.pop(0) if fold else None
    if residual is not None:
        new_res = out.pop(0)
    q, s = out
    if fold:
        # The GDN layer reads only the row count from its hidden states.
        return q, new_res, (q, s, ba)
    return y, new_res, (q, s)


def project(proj: torch.nn.Module, prequant) -> torch.Tensor:
    """The fp8 linear ``proj`` on pre-quantized rows (q, scale)."""
    q, s = prequant
    w = proj.weight
    if getattr(proj, KEEP_VLLM_LAYOUT_ATTR, False):
        w = w.transpose(0, 1)
    return linear_common.quantized_matmul_prequant(q, s, w, proj.weight_scale)


def attention_forward(attn: torch.nn.Module, positions, prequant) -> torch.Tensor:
    """``Qwen3NextAttention.forward`` with the qkv projection on pre-quantized
    rows."""
    qkv = project(attn.qkv_proj, prequant)
    q, k, v, gate = attn._project_qkv_gate(qkv, positions)
    attn_output = attn.attn(q, k, v)
    if gate is not None and getattr(attn, "_tpu_oproj_act_quant", False):
        from vllm_torchtpu.layers.adapter import act_quant_op

        return act_quant_op.o_proj_gated(attn, attn_output, gate)
    if gate is not None:
        attn_output = attn_output * torch.sigmoid(gate)
    output, _ = attn.o_proj(attn_output)
    return output
