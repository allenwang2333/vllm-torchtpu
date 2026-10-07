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
"""The shared expert's SwiGLU and the attention output gate fused with the
next fp8 linear's activation quantization (TPU_ACT_QUANT).

The fused kernel (kernels/experimental/rms_norm_quant/act_quant.py) emits the
fp8 rows and scales; down_proj / o_proj run the pre-quantized fp8 matmul,
instead of the activation fusion, the linear's abs-max and its convert.
Backbone layers only (not the MTP draft).
"""

from collections.abc import Callable

import jax
import torch
from torch_tpu._internal import pallas
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.rms_norm_quant import act_quant as aq
from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

FFO_ATTR = "_tpu_ffo_act_quant"
OPROJ_ATTR = "_tpu_oproj_act_quant"
_ops: dict[tuple[str, int], Callable] = {}


def _build(kind: str) -> Callable:
    round_mid = envs.TPU_ACT_QUANT_ROUND_MID
    key = (kind, round_mid)
    if key in _ops:
        return _ops[key]
    A = jax.Array
    if kind == "silu_mul":

        def fn(x: A) -> tuple[A, A]:
            return aq.act_quant(x, kind=kind, round_mid=round_mid)

        def fake(x):
            return (
                x.new_empty((x.shape[0], x.shape[1] // 2), dtype=torch.float8_e4m3fn),
                x.new_empty((x.shape[0], 1), dtype=torch.float32),
            )

    else:

        def fn(x: A, g: A) -> tuple[A, A]:
            return aq.act_quant(x, g, kind=kind, round_mid=round_mid)

        def fake(x, g):
            return (
                x.new_empty(x.shape, dtype=torch.float8_e4m3fn),
                x.new_empty((x.shape[0], 1), dtype=torch.float32),
            )

    op = pallas.jax_op(f"pallas::act_quant_{kind}_{int(round_mid)}", fn)
    op.register_fake(fake)
    _ops[key] = op
    return op


def _fp8_per_channel(proj) -> bool:
    w = proj.weight
    if w.dtype != torch.float8_e4m3fn or getattr(proj, "bias", None) is not None:
        return False
    n = w.shape[0 if getattr(proj, KEEP_VLLM_LAYOUT_ATTR, False) else 1]
    return tuple(proj.weight_scale.shape) == (n,)


def project(proj, q, s) -> torch.Tensor:
    w = proj.weight
    if getattr(proj, KEEP_VLLM_LAYOUT_ATTR, False):
        w = w.transpose(0, 1)
    out = linear_common.quantized_matmul_prequant(q, s, w, proj.weight_scale)
    if getattr(proj, "reduce_results", False) and getattr(proj, "tp_size", 1) > 1:
        out = tensor_model_parallel_all_reduce(out)
    return out


def install(model: torch.nn.Module) -> tuple[int, int]:
    n_ffo = n_o = 0
    for module in model.modules():
        if isinstance(module, Qwen2MoeMLP):
            prefix = getattr(module.down_proj, "prefix", "")
            if "mtp" in prefix.split(".") or not _fp8_per_channel(module.down_proj):
                continue
            if type(module.act_fn).__name__ != "SiluAndMul":
                continue
            _build("silu_mul")
            setattr(module, FFO_ATTR, True)
            n_ffo += 1
        o_proj = getattr(module, "o_proj", None)
        if o_proj is not None and hasattr(module, "_project_qkv_gate"):
            if "mtp" in getattr(o_proj, "prefix", "").split("."):
                continue
            if not getattr(module, "attn_output_gate", False) or not _fp8_per_channel(o_proj):
                continue
            _build("sigmoid_gate")
            setattr(module, OPROJ_ATTR, True)
            n_o += 1
    linear_common._get_prequant_matmul_op()
    logger.info(
        "TPU_ACT_QUANT: %d shared-expert SwiGLUs and %d attention output gates "
        "quantize for their fp8 linear in one kernel",
        n_ffo,
        n_o,
    )
    return n_ffo, n_o


def o_proj_gated(attn: torch.nn.Module, attn_output, gate) -> torch.Tensor:
    """``o_proj(attn_output * sigmoid(gate))`` through the fused kernel."""
    lead = attn_output.shape[:-1]
    a2 = attn_output.reshape(-1, attn_output.shape[-1])
    q, s = _build("sigmoid_gate")(a2, gate.reshape(a2.shape))
    return project(attn.o_proj, q, s).reshape(*lead, -1)


_orig_mlp_forward = Qwen2MoeMLP.forward


def _mlp_forward(self, x):
    if not getattr(self, FFO_ATTR, False):
        return _orig_mlp_forward(self, x)
    gate_up, _ = self.gate_up_proj(x)
    q, s = _build("silu_mul")(gate_up.reshape(-1, gate_up.shape[-1]))
    out = project(self.down_proj, q, s).reshape(*gate_up.shape[:-1], -1)
    if self.expert_gate is not None:
        out = torch.sigmoid(self.expert_gate(x)[0]) * out
    return out


def apply_patch() -> None:
    Qwen2MoeMLP.forward = _mlp_forward
