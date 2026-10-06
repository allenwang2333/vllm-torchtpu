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

from vllm_torchtpu.kernels.experimental import rms_norm_quant as rnq
from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LAYER_ATTR = "_tpu_input_norm_quant"

_ops: dict[tuple[float, bool, bool], Callable] = {}


def _build_op(eps: float, has_res: bool, emit_y: bool) -> Callable:
    key = (eps, has_res, emit_y)
    if key in _ops:
        return _ops[key]

    A = jax.Array
    # pallas.jax_op reads the return annotation, so each variant spells its
    # outputs out.
    if has_res and emit_y:

        def fn(x: A, r: A, w: A) -> tuple[A, A, A, A]:
            y, ro, q, s = rnq.rms_norm_quant(x, r, w, eps, emit_y=True)
            return y, ro, q, s

    elif has_res:

        def fn(x: A, r: A, w: A) -> tuple[A, A, A]:
            _, ro, q, s = rnq.rms_norm_quant(x, r, w, eps, emit_y=False)
            return ro, q, s

    elif emit_y:

        def fn(x: A, w: A) -> tuple[A, A, A]:
            y, _, q, s = rnq.rms_norm_quant(x, None, w, eps, emit_y=True)
            return y, q, s

    else:

        def fn(x: A, w: A) -> tuple[A, A]:
            _, _, q, s = rnq.rms_norm_quant(x, None, w, eps, emit_y=False)
            return q, s

    op = pallas.jax_op(f"pallas::input_norm_quant_{len(_ops)}", fn)

    def fake(x, *rest):
        outs = []
        if emit_y:
            outs.append(x.new_empty(x.shape))
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
        for has_res in (False, True):
            _build_op(eps, has_res, emit_y=is_gdn)
        linear_common._get_prequant_matmul_op()
        setattr(module, LAYER_ATTR, True)
        count += 1
    logger.info(
        "TPU_INPUT_NORM_QUANT: %d decoder layers run the input norm and the "
        "input projection's fp8 quantization as one kernel",
        count,
    )
    return count


def norm_quant(layer: torch.nn.Module, hidden_states, residual):
    """``layer.input_layernorm(hidden_states, residual)`` plus the fp8 rows of
    its output. Returns (normed or None, residual, (q, scale))."""
    norm = layer.input_layernorm
    eps = float(norm.variance_epsilon)
    _, is_gdn = _projection(layer)
    if residual is None:
        out = _ops[(eps, False, is_gdn)](hidden_states, norm.weight)
        new_res = hidden_states
    else:
        out = _ops[(eps, True, is_gdn)](hidden_states, residual, norm.weight)
    out = list(out)
    y = out.pop(0) if is_gdn else None
    if residual is not None:
        new_res = out.pop(0)
    q, s = out
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
    if gate is not None:
        attn_output = attn_output * torch.sigmoid(gate)
    output, _ = attn.o_proj(attn_output)
    return output
