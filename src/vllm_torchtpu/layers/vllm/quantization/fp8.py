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
"""
FP8 quantization support for TPU.

This module provides TPU-compatible FP8 quantization for MoE models like
Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8. The target model uses FP8 e4m3fn
weights with block-wise scales (weight_block_size: [128, 128]).

== MoE Weights ==

The checkpoint stores FP8 weights with 2D block scales. At load time we
dequantize those weights, then requantize them into the runtime format used by
the TPU GMM kernel. By default the runtime rhs uses per-channel scales over the
contracting dimension.

== Linear Weights (attention projections) ==

The checkpoint stores FP8 weights with either block scales or tensor/channel
scales. At load time we dequantize those weights to float32, then requantize
them into the runtime FP8 layout consumed by the Pallas matmul bridge in
`vllm_torchtpu/layers/vllm/linear_common.py`. Activations are quantized
dynamically inside the bridge. By default the bridge dispatches to a pure-JAX
`dot_general`-based per-channel FP8 matmul (`xla_quantized_matmul`); setting
ENABLE_QUANTIZED_MATMUL_KERNEL=1 together with REQUANTIZE_BLOCK_SIZE selects
the blockwise Pallas kernel path instead.
"""

from typing import Optional

import torch
from torch_tpu._internal import sync
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE,
                                                        FusedMoEMethodBase)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import fp8 as vllm_fp8
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import (Fp8Config,
                                                         Fp8LinearMethod,
                                                         Fp8MoEMethod)
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped
from vllm.model_executor.utils import replace_parameter

from vllm_torchtpu import envs
from vllm_torchtpu.layers.common.quant_methods import FP8, get_tpu_quant_method
from vllm_torchtpu.layers.common.quantization import (dequantize_tensor,
                                                      quantize_tensor)
from vllm_torchtpu.layers.vllm import moe_routing
from vllm_torchtpu.layers.vllm.fused_moe import (fused_moe_gmm,
                                                 prebuild_fused_moe_kernel)
from vllm_torchtpu.layers.vllm.linear_common import quantized_matmul
from vllm_torchtpu.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_MOE_REQUANT_WEIGHT_DTYPES = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "int8": torch.int8,
    "fp4": "fp4",
}
if hasattr(torch, "float8_e5m2"):
    _MOE_REQUANT_WEIGHT_DTYPES["float8_e5m2"] = torch.float8_e5m2

LinearQuantConfig = tuple[str, torch.dtype, int | None, bool]


def _get_activation_str(activation) -> str:
    """Convert MoEActivation enum or string to plain string."""
    return activation.value if hasattr(activation,
                                       'value') else str(activation)


def _dequantize_fp8_linear(
    weight: torch.Tensor,
    *,
    weight_scale: torch.Tensor | None,
    weight_scale_inv: torch.Tensor | None,
    block_quant: bool,
    weight_block_size: tuple[int, int] | None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 linear weights."""
    weight_f = weight.to(out_dtype)

    if block_quant:
        assert weight_scale_inv is not None
        assert weight_block_size is not None
        block_h, block_w = weight_block_size

        out_dim, in_dim = weight_f.shape
        if out_dim % block_h == 0 and in_dim % block_w == 0:
            weight_f = weight_f.reshape(out_dim // block_h, block_h,
                                        in_dim // block_w, block_w)
            weight_f = weight_f * weight_scale_inv.to(out_dtype).unsqueeze(
                1).unsqueeze(3)
            weight_f = weight_f.reshape(out_dim, in_dim)
        else:
            scales = weight_scale_inv.to(out_dtype).repeat_interleave(
                block_h, dim=0).repeat_interleave(block_w, dim=1)
            weight_f = weight_f * scales[:out_dim, :in_dim]
    else:
        assert weight_scale is not None
        if weight_scale.ndim == 1:
            weight_f = weight_f * weight_scale.to(out_dtype).unsqueeze(1)
        else:
            weight_f = weight_f * weight_scale.to(out_dtype)

    return weight_f


def _get_moe_quant_config(
    log_prefix: str, ) -> tuple[str, torch.dtype, int | None]:
    """Parse MoE quantization env vars and return (dtype_name, dtype, block_size)."""
    if desired_quant_dtype_from_env := envs.MOE_REQUANTIZE_WEIGHT_DTYPE:
        desired_quant_dtype = desired_quant_dtype_from_env
    else:
        desired_quant_dtype = "float8_e4m3fn"

    requant_block_size = None
    if requant_block_size_from_env := envs.MOE_REQUANTIZE_BLOCK_SIZE:
        requant_block_size = (int(requant_block_size_from_env)
                              if requant_block_size_from_env else None)

    requant_dtype = _MOE_REQUANT_WEIGHT_DTYPES.get(desired_quant_dtype)
    if requant_dtype is None:
        supported = ", ".join(sorted(_MOE_REQUANT_WEIGHT_DTYPES))
        raise ValueError(
            "Unsupported MOE_REQUANTIZE_WEIGHT_DTYPE="
            f"{desired_quant_dtype!r}. Supported values: {supported}.")

    msg = f"{log_prefix} to {desired_quant_dtype}"
    if requant_block_size is not None:
        msg += f" with block size {requant_block_size}"
    logger.info_once(msg)

    return desired_quant_dtype, requant_dtype, requant_block_size


def _get_linear_quant_config(
    linear_config: VllmQuantLinearConfig | None, ) -> LinearQuantConfig:
    """Parse dense-linear quantization settings from env-gated overrides.

    Reads ENABLE_QUANTIZED_MATMUL_KERNEL, REQUANTIZE_BLOCK_SIZE, and
    REQUANTIZE_WEIGHT_DTYPE (either from `linear_config` or `envs`) and
    validates that the kernel-gate / block-size combination is coherent.
    """
    enable_quantized_matmul_kernel = (
        linear_config.enable_quantized_matmul_kernel
        if linear_config is not None else envs.ENABLE_QUANTIZED_MATMUL_KERNEL)
    requant_block_size = (linear_config.requant_block_size if linear_config
                          is not None else envs.REQUANTIZE_BLOCK_SIZE)
    desired_quant_dtype = (linear_config.requant_weight_dtype if linear_config
                           is not None else envs.REQUANTIZE_WEIGHT_DTYPE)

    if enable_quantized_matmul_kernel and not requant_block_size:
        raise ValueError(
            "You should set REQUANTIZE_BLOCK_SIZE to enable quantized matmul "
            "kernel. Please set the value or disable the quantized matmul "
            "kernel.")
    if not enable_quantized_matmul_kernel and requant_block_size:
        raise ValueError(
            "Blockwise quantization is supported by quantized matmul kernel. "
            "Please enable quantized_matmul_kernel or unset the quantize "
            "block size to trigger XLA per-channel quantization.")

    requant_dtype = _MOE_REQUANT_WEIGHT_DTYPES.get(desired_quant_dtype)
    if requant_dtype is None:
        supported = ", ".join(sorted(_MOE_REQUANT_WEIGHT_DTYPES))
        raise ValueError(
            "Unsupported REQUANTIZE_WEIGHT_DTYPE="
            f"{desired_quant_dtype!r}. Supported values: {supported}.")

    return (desired_quant_dtype, requant_dtype, requant_block_size,
            enable_quantized_matmul_kernel)


def _format_linear_scale_for_runtime(
    weight_scale: torch.Tensor,
    *,
    blockwise_kernel: bool,
    weight_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Reshape `weight_scale` to the layout expected by the matmul bridge.

    - Blockwise kernel path: kernel expects `[n_in_blocks, 1, n_out]`. The
      requantizer produces `[n_out, n_in_blocks]`, so transpose and add the
      singleton middle axis.
    - XLA per-channel path: kernel expects a 1-D `[n_out]` scale, so drop
      a trailing singleton dim if present.
    """
    weight_scale = weight_scale.to(torch.float32).contiguous()
    if blockwise_kernel:
        if weight_scale.ndim == 2 and weight_shape is not None and weight_scale.shape[0] != weight_shape[0]:
            repeat_factor = weight_shape[0] // weight_scale.shape[0]
            weight_scale = torch.repeat_interleave(weight_scale, repeat_factor, dim=0)
        # quantize_tensor returns [n_out, n_blocks]. The blockwise kernel
        # expects [n_blocks, 1, n_out].
        weight_scale = weight_scale.transpose(0, 1).contiguous().unsqueeze(1)
    elif weight_scale.ndim == 2 and weight_scale.shape[-1] == 1:
        weight_scale = weight_scale.squeeze(-1).contiguous()
    return weight_scale


def _process_fp8_linear_weights(
    layer: torch.nn.Module,
    *,
    block_quant: bool,
    weight_block_size: tuple[int, int] | None,
    linear_config: VllmQuantLinearConfig | None,
    linear_quant_config: LinearQuantConfig | None = None,
    is_scale_e8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, str, int | None]:
    """Convert checkpoint FP8 dense linear weights to the runtime FP8 layout.

    When the checkpoint weight is already in FP8 and block sizes match,
    directly formats the scale tensor without float32 dequantization.
    """
    if linear_quant_config is None:
        linear_quant_config = _get_linear_quant_config(linear_config)
    desired_quant_dtype, requant_dtype, requant_block_size, blockwise_kernel = (
        linear_quant_config)

    weight_scale = (layer.weight_scale.data
                    if hasattr(layer, "weight_scale") else None)
    weight_scale_inv = (layer.weight_scale_inv.data if hasattr(
        layer, "weight_scale_inv") else None)

    if is_scale_e8m0:
        if weight_scale is not None:
            if weight_scale.dtype == torch.float8_e8m0fnu:
                weight_scale = weight_scale.view(torch.uint8)
            weight_scale = torch.pow(2.0, weight_scale.to(torch.float32) - 127.0)
        if weight_scale_inv is not None:
            if weight_scale_inv.dtype == torch.float8_e8m0fnu:
                weight_scale_inv = weight_scale_inv.view(torch.uint8)
            weight_scale_inv = torch.pow(2.0, weight_scale_inv.to(torch.float32) - 127.0)

    current_scale = weight_scale_inv if weight_scale_inv is not None else weight_scale

    if layer.weight.data.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz) and current_scale is not None and (requant_block_size == weight_block_size or (requant_block_size == weight_block_size[0] if isinstance(weight_block_size, tuple) else False) or (requant_block_size is None and not block_quant)):
        weight = layer.weight.data
        weight_scale = _format_linear_scale_for_runtime(
            current_scale,
            blockwise_kernel=blockwise_kernel,
            weight_shape=weight.shape,
        )
    else:
        weight_f = _dequantize_fp8_linear(
            layer.weight.data,
            weight_scale=weight_scale,
            weight_scale_inv=weight_scale_inv,
            block_quant=block_quant,
            weight_block_size=weight_block_size,
            out_dtype=torch.float32,
        )
        weight, weight_scale = quantize_tensor(
            weight_f,
            quant_dtype=requant_dtype,
            axis=-1,
            block_size=requant_block_size,
        )
        weight_scale = _format_linear_scale_for_runtime(
            weight_scale,
            blockwise_kernel=blockwise_kernel,
            weight_shape=weight.shape,
        )
    return weight.contiguous(
    ), weight_scale, desired_quant_dtype, requant_block_size


def _quantize_and_format_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    quant_dtype: torch.dtype,
    block_size: int | None,
    activation: str,
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize float32 MoE weights to FP8 (or format existing FP8 weights) for the GMM kernel.

    Handles block alignment validation, quantization (if scales not provided),
    swigluoai deinterleaving, transpose, and scale reshaping.
    """
    if block_size is not None and (w13_scale is None or w2_scale is None):
        for name, w in (("w13", w13), ("w2", w2)):
            if w.shape[-1] % block_size != 0:
                raise ValueError(
                    f"Unsupported MoE quantization: {name} contracting "
                    f"dimension {w.shape[-1]} is not divisible by "
                    f"block_size={block_size}.")

    if w13_scale is None:
        w13, w13_scale = quantize_tensor(w13,
                                         quant_dtype=quant_dtype,
                                         axis=-1,
                                         block_size=block_size)
    if w2_scale is None:
        w2, w2_scale = quantize_tensor(w2,
                                       quant_dtype=quant_dtype,
                                       axis=-1,
                                       block_size=block_size)

    if activation == "swigluoai":
        w1 = w13[:, ::2, :]
        w3 = w13[:, 1::2, :]
        w13 = torch.cat([w1, w3], dim=1)

        if w13_scale is not None:
            s1 = w13_scale[:, ::2, :]
            s3 = w13_scale[:, 1::2, :]
            w13_scale = torch.cat([s1, s3], dim=1)

    if w13_scale is not None and w13_scale.shape[1] != w13.shape[1]:
        w13_scale = w13_scale.repeat_interleave(w13.shape[1] // w13_scale.shape[1], dim=1)
    if w2_scale is not None and w2_scale.shape[1] != w2.shape[1]:
        w2_scale = w2_scale.repeat_interleave(w2.shape[1] // w2_scale.shape[1], dim=1)

    w13 = w13.transpose(1, 2).contiguous()
    w2 = w2.transpose(1, 2).contiguous()

    # Reshape scales for GMM kernel.
    # An `unsqueeze(2)` view on a TPU tensor leaves the persistent
    # nn.Parameter carrying ambiguous strides (stride[1] == stride[2]) that
    # torch_tpu's per-op JIT re-emits as a per-step `tt_jit_as_strided`
    # program at the GMM `pallas.jax_op` boundary. Allocate a fresh 4D
    # contiguous device buffer with `torch.empty + .copy_` to break the
    # view chain so the persistent storage is a plain 4D buffer that PJRT
    # ships to the kernel with row-major layout.
    if w13_scale is not None:
        s = w13_scale.transpose(1, 2).contiguous().to(torch.float32)
        w13_scale = torch.empty(s.shape[0],
                                s.shape[1],
                                1,
                                s.shape[2],
                                dtype=s.dtype,
                                device=s.device).copy_(s.unsqueeze(2))
    if w2_scale is not None:
        s = w2_scale.transpose(1, 2).contiguous().to(torch.float32)
        w2_scale = torch.empty(s.shape[0],
                               s.shape[1],
                               1,
                               s.shape[2],
                               dtype=s.dtype,
                               device=s.device).copy_(s.unsqueeze(2))

    return w13, w13_scale, w2, w2_scale


def _process_fp8_moe_weights(
    layer: FusedMoE,
    *,
    weight_block_size: tuple[int, int],
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int
           | None]:
    """Format checkpoint FP8 MoE weights for the TPU GMM kernel without float32 dequantization.

    Returns (w13_weight, w13_scale, w2_weight, w2_scale,
             requant_dtype_name, requant_block_size).
    """
    desired_quant_dtype, requant_dtype, requant_block_size = (
        _get_moe_quant_config(
            "[MoE requantization]: re-quantizing MoE weights"))

    # Validate block alignment
    block_h, block_w = weight_block_size
    for name, weight in (("w13_weight", layer.w13_weight.data),
                         ("w2_weight", layer.w2_weight.data)):
        _, out_dim, in_dim = weight.shape
        if out_dim % block_h != 0 or in_dim % block_w != 0:
            raise ValueError(
                "FP8 block quantized MoE weights must be divisible by the checkpoint "
                f"block size, got {name}.shape={tuple(weight.shape)} and "
                f"block_size={weight_block_size}.")

    w13 = layer.w13_weight.data
    w13_scale = layer.w13_weight_scale_inv.data
    w2 = layer.w2_weight.data
    w2_scale = layer.w2_weight_scale_inv.data

    padded_intermediate = layer.moe_config.intermediate_size_per_partition
    unpadded_intermediate = (
        layer.moe_config.intermediate_size_per_partition_unpadded)
    if padded_intermediate != unpadded_intermediate:
        full_intermediate = unpadded_intermediate * layer.tp_size
        local_start = padded_intermediate * layer.tp_rank
        local_real = max(
            0, min(padded_intermediate, full_intermediate - local_start))
        if local_real < padded_intermediate:
            w13[:, local_real:padded_intermediate, :] = 0
            w13[:, padded_intermediate + local_real:2 *
                padded_intermediate, :] = 0
            w2[:, :, local_real:padded_intermediate] = 0

    # If requantization to a different dtype (e.g. FP4) or block size is requested,
    # dequantize and requantize on CPU host memory to prevent TPU HBM spikes.
    if desired_quant_dtype != "float8_e4m3fn" or (
            requant_block_size is not None and requant_block_size != weight_block_size[1]):
        from vllm_torchtpu.layers.common.quantization import (
            dequantize_tensor, quantize_tensor_to_fp4, pack_fp4_indices,
            _safe_inverse_scale)
        w13_scale_cpu = _safe_inverse_scale(w13_scale.to("cpu"))
        w2_scale_cpu = _safe_inverse_scale(w2_scale.to("cpu"))
        w13_cpu = dequantize_tensor(w13.to("cpu"), w13_scale_cpu, axis=(1, 2), out_dtype=torch.float32)
        w2_cpu = dequantize_tensor(w2.to("cpu"), w2_scale_cpu, axis=(1, 2), out_dtype=torch.float32)
        if activation == "swigluoai":
            w1 = w13_cpu[:, ::2, :]
            w3 = w13_cpu[:, 1::2, :]
            w13_cpu = torch.cat([w1, w3], dim=1)
        w13_indices, w13_scale_fp4 = quantize_tensor_to_fp4(w13_cpu, axis=2, block_size=requant_block_size or 512)
        w2_indices, w2_scale_fp4 = quantize_tensor_to_fp4(w2_cpu, axis=2, block_size=requant_block_size or 512)
        w13_packed = pack_fp4_indices(w13_indices.transpose(1, 2)).contiguous()
        w2_packed = pack_fp4_indices(w2_indices.transpose(1, 2)).contiguous()
        s13 = w13_scale_fp4.transpose(1, 2).contiguous().to(w13.device)
        w13_scale = torch.empty(s13.shape[0], s13.shape[1], 1, s13.shape[2], dtype=s13.dtype, device=s13.device).copy_(s13.unsqueeze(2))
        s2 = w2_scale_fp4.transpose(1, 2).contiguous().to(w2.device)
        w2_scale = torch.empty(s2.shape[0], s2.shape[1], 1, s2.shape[2], dtype=s2.dtype, device=s2.device).copy_(s2.unsqueeze(2))
        w13 = w13_packed.to(w13.device)
        w2 = w2_packed.to(w2.device)
    else:
        w13, w13_scale, w2, w2_scale = _quantize_and_format_moe_weights(
            w13,
            w2,
            quant_dtype=requant_dtype,
            block_size=requant_block_size,
            activation=activation,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
        )

    return w13, w13_scale, w2, w2_scale, desired_quant_dtype, requant_block_size


def _quantize_bf16_moe_weights(
    layer: FusedMoE,
    *,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int
           | None]:
    """Quantize BF16 MoE weights to FP8 for the TPU GMM kernel.

    Same output format as _process_fp8_moe_weights but starts from BF16
    weights instead of dequantizing from FP8 first.
    """
    desired_quant_dtype, requant_dtype, requant_block_size = (
        _get_moe_quant_config("[MoE online FP8]: quantizing BF16 MoE weights"))

    w13 = layer.w13_weight.data.to(torch.float32)
    w2 = layer.w2_weight.data.to(torch.float32)

    w13, w13_scale, w2, w2_scale = _quantize_and_format_moe_weights(
        w13,
        w2,
        quant_dtype=requant_dtype,
        block_size=requant_block_size,
        activation=activation,
    )

    return w13, w13_scale, w2, w2_scale, desired_quant_dtype, requant_block_size


@register_quantization_config(get_tpu_quant_method(FP8))
class VllmFp8Config(Fp8Config, VllmQuantConfig):

    @classmethod
    def get_name(cls) -> str:
        return FP8

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, FusedMoE):
            if is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                from vllm_torchtpu.layers.vllm.quantization.unquantized import \
                    VllmUnquantizedFusedMoEMethod
                moe_config = self.get_moe_config(layer)
                return VllmUnquantizedFusedMoEMethod(moe_config)
            moe_config = self.get_moe_config(layer)
            return VllmFp8MoEMethodTPU(self, moe_config)

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            return VllmFp8LinearMethodTPU(self, self.get_linear_config(layer))

        if isinstance(layer, Attention):
            return None

        return None


class VllmFp8MoEMethodTPU(Fp8MoEMethod):
    """
    TPU-native FP8 MoE method.

    Supports both FP8-serialized checkpoints (dequant → requant) and BF16
    checkpoints (online quantization to FP8 at load time).

    Uses is_monolithic=True so vLLM's DefaultMoERunner calls
    apply_monolithic(layer, x, router_logits) directly, bypassing the
    CUDA-only router.select_experts() path.
    """

    def __init__(self, quant_config: Fp8Config, layer):
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        
        is_dsv4 = (quant_config.vllm_config is not None and 
                   quant_config.vllm_config.model_config.hf_config.model_type == "deepseek_v4")
        if is_dsv4:
            self.weight_block_size = [layer.moe_config.intermediate_size_per_partition, 32]

        self.block_quant = self.weight_block_size is not None
        self.weight_scale_name = ("weight_scale_inv"
                                  if self.block_quant else "weight_scale")
        self.fp8_backend = None

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = (
            Fp8MoEMethod.maybe_roundup_sizes(
                self,
                hidden_size,
                intermediate_size_per_partition,
                act_dtype,
                moe_parallel_config,
            ))
        if self.quant_config.is_checkpoint_fp8_serialized and self.block_quant:
            assert self.weight_block_size is not None
            block_n, block_k = self.weight_block_size
            block_size = max(block_n, block_k)
            intermediate_size_per_partition = (
                (intermediate_size_per_partition + block_size - 1) //
                block_size * block_size)
        return hidden_size, intermediate_size_per_partition

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        if not hasattr(layer, "has_bias"):
            layer.has_bias = False
            
        is_dsv4 = (self.quant_config.vllm_config is not None and 
                   self.quant_config.vllm_config.model_config.hf_config.model_type == "deepseek_v4")
        
        if self.quant_config.is_checkpoint_fp8_serialized and is_dsv4:
            from vllm.model_executor.utils import set_weight_attrs
            layer.num_experts = num_experts
            layer.orig_dtype = params_dtype
            layer.weight_block_size = self.weight_block_size
            
            # 1. Weights (same as standard)
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size,
                    dtype=torch.float8_e4m3fn,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight", w13_weight)
            set_weight_attrs(w13_weight, extra_weight_attrs)

            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                    dtype=torch.float8_e4m3fn,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight", w2_weight)
            set_weight_attrs(w2_weight, extra_weight_attrs)
            
            # 2. Scales (custom shapes matching the checkpoint)
            block_k = self.weight_block_size[1]
            w13_scale_data = torch.ones(
                num_experts,
                2,
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            )
            w2_scale_data = torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // block_k,
                dtype=torch.float32,
            )
            
            w13_weight_scale = torch.nn.Parameter(w13_scale_data, requires_grad=False)
            w2_weight_scale = torch.nn.Parameter(w2_scale_data, requires_grad=False)
            
            layer.register_parameter(f"w13_{self.weight_scale_name}", w13_weight_scale)
            layer.register_parameter(f"w2_{self.weight_scale_name}", w2_weight_scale)
            
            extra_weight_attrs = dict(extra_weight_attrs)
            extra_weight_attrs.update({
                "quant_method": "block",
                "load_full_w2": False,
            })
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        else:
            if self.quant_config.is_checkpoint_fp8_serialized:
                Fp8MoEMethod.create_weights(self, layer, num_experts, hidden_size,
                                            intermediate_size_per_partition,
                                            params_dtype, **extra_weight_attrs)
            else:
                from vllm_torchtpu.layers.vllm.quantization.unquantized import \
                    VllmUnquantizedFusedMoEMethod
                VllmUnquantizedFusedMoEMethod.create_weights(
                    self, layer, num_experts, hidden_size,
                    intermediate_size_per_partition, params_dtype,
                    **extra_weight_attrs)

    @property
    def is_monolithic(self) -> bool:
        return True

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Process MoE weights into the runtime FP8 layout used by the GMM kernel.

        For FP8 checkpoints: dequantize block FP8 → float32, then requantize
        to the GMM kernel's expected per-channel FP8 format.

        For BF16 checkpoints (online quantization): quantize BF16 → FP8
        directly using the same requantization path.
        """
        assert isinstance(layer, FusedMoE)
        assert not self.moe.has_bias, "TPU FP8 MoE path does not support bias."

        activation_str = _get_activation_str(layer.activation)
        layer._tpu_activation_str = activation_str
        import os
        import torch.distributed as dist
        layer_idx = getattr(layer, "layer_idx", None)
        rank = dist.get_rank() if dist.is_initialized() else 0
        cache_dir = os.environ.get("VLLM_FP4_CACHE_DIR", "")
        cache_path = None
        if cache_dir and layer_idx is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"rank_{rank}_layer_{layer_idx}.pt")
            
        loaded_from_cache = False
        if cache_path and os.path.exists(cache_path):
            try:
                logger.info("Loading pre-quantized parameters from cache: %s", cache_path)
                checkpoint = torch.load(cache_path, map_location="cpu")
                w13 = checkpoint["w13"]
                w2 = checkpoint["w2"]
                w13_scale = checkpoint["w13_scale"]
                w2_scale = checkpoint["w2_scale"]
                requant_dtype_name = checkpoint["requant_dtype_name"]
                requant_block_size = checkpoint["requant_block_size"]
                loaded_from_cache = True
            except Exception as e:
                logger.warning("Failed to load cached parameters from %s: %s", cache_path, e)

        if not loaded_from_cache:
            is_fp8_serialized = self.quant_config.is_checkpoint_fp8_serialized
            if is_fp8_serialized:
                assert self.block_quant, \
                    "TPU FP8 MoE path expects block-quantized weights."
                (w13, w13_scale, w2, w2_scale, requant_dtype_name,
                 requant_block_size) = _process_fp8_moe_weights(
                     layer,
                     weight_block_size=tuple(self.weight_block_size),
                     activation=activation_str,
                 )
            else:
                (w13, w13_scale, w2, w2_scale, requant_dtype_name,
                 requant_block_size) = _quantize_bf16_moe_weights(
                     layer,
                     activation=activation_str,
                 )
            if cache_path:
                try:
                    logger.info("Saving pre-quantized parameters to cache: %s", cache_path)
                    torch.save({
                        "w13": w13.to("cpu"),
                        "w2": w2.to("cpu"),
                        "w13_scale": w13_scale.to("cpu"),
                        "w2_scale": w2_scale.to("cpu"),
                        "requant_dtype_name": requant_dtype_name,
                        "requant_block_size": requant_block_size,
                    }, cache_path)
                except Exception as e:
                    logger.warning("Failed to save cached parameters to %s: %s", cache_path, e)

        def to_clean_device_param(t: torch.Tensor | None, device: torch.device) -> torch.nn.Parameter | None:
            if t is None:
                return None
            clean_buf = torch.empty(t.shape, dtype=t.dtype, device=device).copy_(t)
            return torch.nn.Parameter(clean_buf, requires_grad=False)

        device = layer.w13_weight.device
        if device.type == "tpu":
            for name in ["w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv", "w13_weight_scale", "w2_weight_scale"]:
                param = getattr(layer, name, None)
                if param is not None and isinstance(param, torch.nn.Parameter):
                    param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            import gc
            gc.collect()

        layer.w13_weight = to_clean_device_param(w13, device)
        layer.w2_weight = to_clean_device_param(w2, device)
        layer.w13_weight_scale_inv = to_clean_device_param(w13_scale, device)
        layer.w2_weight_scale_inv = to_clean_device_param(w2_scale, device)

        # Eagerly materialize weights to avoid OOM during vLLM memory probing.
        # Without this, lazy tensors accumulate and the profiling run OOMs.
        if layer.w13_weight.device.type == "tpu":
            sync.synchronize(layer.w13_weight, wait=True)
            sync.synchronize(layer.w2_weight, wait=True)
            sync.synchronize(layer.w13_weight_scale_inv, wait=True)
            sync.synchronize(layer.w2_weight_scale_inv, wait=True)

        scale_desc = ("per-channel" if requant_block_size is None else
                      str(requant_block_size))
        logger.info_once(
            "FP8 weights transposed for GMM kernel: "
            f"w13={list(layer.w13_weight.shape)}, "
            f"w2={list(layer.w2_weight.shape)}, "
            f"w13_scale={list(layer.w13_weight_scale_inv.shape)}, "
            f"w2_scale={list(layer.w2_weight_scale_inv.shape)}, "
            f"requant_block_size={scale_desc}")
        if layer.moe_config.moe_parallel_config.use_ep:
            moe_routing.validate_linear_ep_placement(layer)
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            experts_start=moe_routing.get_experts_start(layer),
            has_bias=getattr(layer, "has_bias", False),
        )

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass using TPU-native GMM kernel with FP8 weights."""
        activation_str = layer._tpu_activation_str
        # Step 1: Routing
        custom_routing_fn = getattr(layer, "custom_routing_function", None)
        if custom_routing_fn is not None:
            topk_weights, topk_ids = custom_routing_fn(
                hidden_states=x,
                gating_output=router_logits,
                topk=layer.moe_config.experts_per_token,
                renormalize=layer.renormalize,
            )
        else:
            topk_weights, topk_ids = moe_routing.select_experts(
                hidden_states=x,
                router_logits=router_logits,
                topk=layer.moe_config.experts_per_token,
                renormalize=layer.renormalize,
                scoring_fn=getattr(layer, "scoring_func", "softmax"),
                layer=layer,
            )

        # Step 2: EP global->local remap happens inside fused_moe_gmm via an
        # elementwise subtract from `experts_start` (scalar).
        return fused_moe_gmm(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale_inv,
            w2_scale=layer.w2_weight_scale_inv,
            w1_bias=getattr(layer, 'w13_bias', None),
            w2_bias=getattr(layer, 'w2_bias', None),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=moe_routing.get_experts_start(layer),
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
        )


class VllmFp8LinearMethodTPU(Fp8LinearMethod):
    """
    TPU FP8 linear method that keeps weights in FP8 at runtime and quantizes
    activations dynamically inside a Pallas matmul bridge.

    Reuses vLLM's Fp8LinearMethod.create_weights() for correct FP8 weight and
    scale parameter allocation (needed by the weight loader). After loading,
    checkpoint FP8 weights are converted to the runtime FP8 layout and applied
    with dynamically quantized activations through `quantized_matmul`.

    Inherits from Fp8LinearMethod but skips GPU-specific __init__. Only
    sets attributes needed by create_weights().
    """

    def __init__(self,
                 quant_config: Fp8Config,
                 linear_config: VllmQuantLinearConfig | None = None):
        # Skip Fp8LinearMethod.__init__ which has GPU-specific code
        # (CUDA capability, Marlin, cutlass, W8A8BlockFp8LinearOp).
        # Set only the attributes that create_weights() reads.
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.act_q_static = quant_config.activation_scheme == "static"
        # These are needed by create_weights but are GPU-specific.
        # Set safe defaults so the parent's create_weights doesn't crash.
        self.use_marlin = False
        self.cutlass_block_fp8_supported = False
        self.is_scale_e8m0 = getattr(quant_config, "is_scale_e8m0", False)
        self.activation_quant_key = None
        self.weight_quant_key = None
        self.input_dtype = torch.get_default_dtype()
        self.out_dtype = torch.get_default_dtype()
        vllm_fp8.init_fp8_linear_kernel = lambda *args, **kwargs: None
        self.linear_config = linear_config
        self._linear_quant_config = _get_linear_quant_config(
            self.linear_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if self.block_quant:
            super().create_weights(
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                **extra_weight_attrs,
            )
        else:
            output_size_per_partition = sum(output_partition_sizes)
            weight_loader = extra_weight_attrs.get("weight_loader")
            
            from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                                       ModelWeightParameter)
            
            weight = ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=torch.float8_e4m3fn,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)
            
            scale = ChannelQuantScaleParameter(
                data=torch.empty(output_size_per_partition, dtype=torch.float32),
                weight_loader=weight_loader,
                output_dim=0,
            )
            layer.register_parameter("weight_scale", scale)

        if not hasattr(layer, "weight_scale_inv"):
            if hasattr(layer, "weight_scale"):
                layer.weight_scale_inv = layer.weight_scale

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert dense linear checkpoint weights to runtime FP8."""
        weight, weight_scale, requant_dtype_name, requant_block_size = (
            _process_fp8_linear_weights(
                layer,
                block_quant=self.block_quant,
                weight_block_size=tuple(self.weight_block_size)
                if self.weight_block_size is not None else None,
                linear_config=self.linear_config,
                linear_quant_config=self._linear_quant_config,
                is_scale_e8m0=self.is_scale_e8m0,
            ))

        # The dequant/requant chain may leave views backed by higher-rank PJRT
        # buffers. Allocate fresh contiguous persistent parameters so the
        # matmul boundary sees plain row-major buffers.
        weight = torch.empty(weight.shape[0],
                             weight.shape[1],
                             dtype=weight.dtype,
                             device=weight.device).copy_(weight)
        weight_scale = torch.empty(
            *weight_scale.shape,
            dtype=weight_scale.dtype,
            device=weight_scale.device,
        ).copy_(weight_scale)

        replace_parameter(layer, "weight",
                          torch.nn.Parameter(weight, requires_grad=False))
        replace_parameter(layer, "weight_scale", weight_scale)
        if hasattr(layer, "weight_scale_inv"):
            replace_parameter(layer, "weight_scale_inv", weight_scale)
        else:
            layer.weight_scale_inv = torch.nn.Parameter(weight_scale, requires_grad=False)


        scale_desc = ("per-channel" if requant_block_size is None else
                      str(requant_block_size))
        logger.info_once("FP8 linear weights requantized for runtime: "
                         f"shape={list(layer.weight.shape)}, "
                         f"weight_dtype={requant_dtype_name}, "
                         f"scale={list(layer.weight_scale.shape)}, "
                         f"requant_block_size={scale_desc}")

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = quantized_matmul(x, layer.weight, layer.weight_scale)
        if bias is not None:
            out = out + bias
        return out
