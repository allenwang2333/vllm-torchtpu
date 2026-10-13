# Copyright 2025 Google LLC
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
"""Activation precision override for TPU (opt-in).

On TPU v7x the XLA:TPU lowering evaluates bf16 transcendental fusions natively in
bf16, where v6e keeps f32 internally. Measured op-level (rms error of the bf16
result against f32 of the same inputs, divided by the error of the once-rounded f32
result): bf16 SiLU 1.78x on v7x vs 1.00x on v6e, sigmoid 1.6-1.85x vs 1.00x,
gelu(tanh) 1.2-1.4x vs 1.0x (JAX jit; torch_tpu F.silu 1.47x vs 1.28x). For models
with large early-layer activations (Qwen3-4B: layer 4 reaches the hundreds) this
raises the hidden-state error of a 36-layer forward 2.6x and the logits error 5x
against an fp32 reference, and greedy decoding flips ~4.5% of argmax tokens (mean
deficit 0.24 nats). Evaluating the activation in fp32 restores the v6e profile; the
surrounding matmuls stay bf16, so the cost is one fp32 upcast of the activation
tensor per layer.

Each selected activation is overridden the way the other TPU CustomOps are: an
out-of-tree subclass registered with ``CustomOp.register_oot`` that overrides
``forward_native`` (see ``VllmDeepseekScalingRotaryEmbedding``). vLLM instantiates
the registered subclass in place of its own class, and every dispatch path on this
out-of-tree platform (``forward_oot`` with the op enabled, ``forward_native`` with
it disabled) ends in that method.

VLLM_TPU_ACTIVATION_FP32 selects which activation families run in fp32 (default:
none, behaviour unchanged):
  silu   SiluAndMul, MulAndSilu, SiluAndMulWithClamp, SituAndMul, SwigluOAIAndMul,
         SwigluStepAndMul
  gelu   GeluAndMul, GeluAndMulSparse, GELU, GELUTanh, NewGELU, FastGELU, QuickGELU
  other  every other activation CustomOp in vllm.model_executor.layers.activation
  all    everything above
e.g. VLLM_TPU_ACTIVATION_FP32=silu (validated end to end on v7x with Qwen3-4B),
=silu,gelu, =all.
"""

import inspect
from typing import cast

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENV = "VLLM_TPU_ACTIVATION_FP32"
_FAMILIES = {
    "silu": (
        "SiluAndMul",
        "MulAndSilu",
        "SiluAndMulWithClamp",
        "SituAndMul",
        "SwigluOAIAndMul",
        "SwigluStepAndMul",
    ),
    "gelu": (
        "GeluAndMul",
        "GeluAndMulSparse",
        "GELU",
        "GELUTanh",
        "NewGELU",
        "FastGELU",
        "QuickGELU",
    ),
}


def selected_families() -> set[str]:
    raw = envs.VLLM_TPU_ACTIVATION_FP32.strip().lower()
    if raw in ("", "0", "false", "off", "none"):
        return set()
    if raw in ("1", "true", "on", "all"):
        return {"silu", "gelu", "other"}
    return {f.strip() for f in raw.split(",") if f.strip()}


def _family(class_name: str) -> str:
    for fam, names in _FAMILIES.items():
        if class_name in names:
            return fam
    return "other"


def _fp32_subclass(cls: type) -> type | None:
    """An out-of-tree subclass of ``cls`` whose ``forward_native`` runs in fp32."""
    native = cls.__dict__.get("forward_native")
    if native is None:
        return None
    is_static = isinstance(native, staticmethod)
    fn = native.__func__ if is_static else native

    def forward_native(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        xf = x.float()
        y = fn(xf, *args, **kwargs) if is_static else fn(self, xf, *args, **kwargs)
        return y.to(x.dtype)

    return type(
        f"VllmTPUFp32{cls.__name__}",
        (cls,),
        {
            "__module__": __name__,
            "__doc__": f"{cls.__name__} with the activation evaluated in fp32.",
            "forward_native": forward_native,
            "_tpu_activation_fp32": True,
        },
    )


def _register_fp32_activations() -> None:
    """Register fp32 out-of-tree overrides for the selected activation CustomOps."""
    families = selected_families()
    if not families:
        return
    from vllm.model_executor import custom_op
    from vllm.model_executor.layers import activation

    registered = []
    for name, cls in inspect.getmembers(activation, inspect.isclass):
        if cls.__module__ != activation.__name__ or not issubclass(
            cls, custom_op.CustomOp
        ):
            continue
        op_cls = cast(type[custom_op.CustomOp], cls)
        if _family(name) not in families:
            continue
        existing = custom_op.op_registry_oot.get(name)
        if existing is not None:
            if not getattr(existing, "_tpu_activation_fp32", False):
                logger.warning(
                    "%s: %s already has an out-of-tree override (%s); leaving it",
                    ENV,
                    name,
                    existing.__name__,
                )
            continue
        sub = _fp32_subclass(op_cls)
        if sub is None:
            continue
        op_cls.register_oot(sub)
        registered.append(name)
    if registered:
        logger.warning(
            "%s=%s: evaluating these activations in fp32: %s",
            ENV,
            envs.VLLM_TPU_ACTIVATION_FP32,
            ", ".join(registered),
        )
