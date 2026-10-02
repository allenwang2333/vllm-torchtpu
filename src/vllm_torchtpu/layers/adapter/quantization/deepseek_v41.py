"""V4.1 model bindings for the shared DeepSeek V4 TPU quantization path."""

import functools
from math import lcm

import jax
import jax.numpy as jnp
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.models.deepseek_v4.nvidia import model as v4_upstream
from vllm.models.deepseek_v41.nvidia.model import DeepseekV4MoE

from vllm_torchtpu.layers.adapter.quantization.deepseek_v4_fp8 import (
    VllmDeepseekV4Fp8Config,
)
from vllm_torchtpu.layers.adapter.quantization.mxfp4 import (
    REQUANTIZED_BLOCK_SIZE,
    VllmDeepseekV4Mxfp4MoEMethod,
)
from vllm_torchtpu.models.vllm.deepseek_v4.layers import DeepseekV4MLP
from vllm_torchtpu.models.vllm.deepseek_v41.ops import device_hooks


@register_quantization_config("tpu-deepseek_v41_fp8")
class VllmDeepseekV41Fp8Config(VllmDeepseekV4Fp8Config):
    """Use V4 weight preparation, activation quantization and MoE dispatch."""

    @classmethod
    def get_name(cls):
        return "deepseek_v41_fp8"

    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        if type(method) is VllmDeepseekV4Mxfp4MoEMethod:
            return VllmDeepseekV41Mxfp4MoEMethod(method.moe)
        return method


def dequantize_pad_mxfp4_kmajor(
    w_packed: jax.Array, scale_u8: jax.Array, *, split: int, pad_k: int, pad_n: int
) -> jax.Array:
    """MXFP4 ``[E, N, K/2]`` uint8 + e8m0 ``[E, N, K/32]`` -> BF16 ``[E, K, N]``.

    Pads K by ``pad_k`` and N by ``pad_n``. With ``split``, N holds two halves
    (w1|w3) that are padded separately, as in the CPU path it replaces.
    """
    fp4 = jax.lax.bitcast_convert_type(w_packed, jnp.float4_e2m1fn)
    fp4 = fp4.reshape(*w_packed.shape[:-1], -1)  # [E, N, K]
    scale = jnp.ldexp(jnp.float32(1), scale_u8.astype(jnp.int32) - 127)
    blocks = fp4.astype(jnp.float32).reshape(*scale.shape, -1) * scale[..., None]
    w = jnp.swapaxes(blocks.reshape(fp4.shape).astype(jnp.bfloat16), 1, 2)
    halves = (w[..., :split], w[..., split:]) if split else (w,)
    return jnp.concatenate(
        [jnp.pad(h, ((0, 0), (0, pad_k), (0, pad_n))) for h in halves], axis=-1
    )


_dequantize_pad_ops = {}


def _dequantize_pad(w_packed, scale_u8, split, pad_k, pad_n):
    key = (split, pad_k, pad_n)
    if key not in _dequantize_pad_ops:
        from torch_tpu._internal import pallas

        _dequantize_pad_ops[key] = pallas.jax_op(
            f"pallas::deepseek_v41_mxfp4_dequant_pad_s{split}_k{pad_k}_n{pad_n}",
            functools.partial(
                dequantize_pad_mxfp4_kmajor, split=split, pad_k=pad_k, pad_n=pad_n
            ),
        )
    return _dequantize_pad_ops[key](w_packed, scale_u8)


class VllmDeepseekV41Mxfp4MoEMethod(VllmDeepseekV4Mxfp4MoEMethod):
    """Dequantize and pad on the TPU, where the staged weights already live.

    The V4 path copies them to the host and dequantizes in single-threaded
    torch (~20 s per layer on every rank). The values are identical: an FP4
    value times a power-of-two scale is exact in FP32 and in BF16.
    """

    def _dequantize_and_pad(self, layer):
        if getattr(layer, "w13_bias", None) is not None:
            return super()._dequantize_and_pad(layer)
        hidden = layer.w2_weight.shape[1]
        inter = self.intermediate_size
        # Same alignment as the V4 path: hidden to whole w13 blocks, the
        # intermediate to GMM tiles holding whole w2 blocks.
        block = REQUANTIZED_BLOCK_SIZE
        inter_align = lcm(128, min(block, inter))
        pad_hidden = -hidden % block
        pad_inter = -inter % inter_align
        w13 = _dequantize_pad(
            layer.w13_weight.data,
            layer.w13_weight_scale.data,
            inter,
            pad_hidden,
            pad_inter,
        )
        w2 = _dequantize_pad(
            layer.w2_weight.data, layer.w2_weight_scale.data, 0, pad_inter, pad_hidden
        )
        return w13, w2, None, None, inter


class DeepseekV41MoE(DeepseekV4MoE):
    def __init__(self, vllm_config, *args, **kwargs):
        if vllm_config.parallel_config.enable_eplb:
            raise NotImplementedError("V4.1 TPU MoE does not support EPLB")
        if vllm_config.kernel_config.moe_backend in v4_upstream.MEGA_MOE_BACKENDS:
            raise NotImplementedError("V4.1 TPU requires the native V4 MoE backend")
        factory = v4_upstream.FusedMoEFactory

        def v4_factory(**options):
            # Match the native TPU V4 model's scaling and shared-expert combine.
            options["apply_routed_scale_to_output"] = True
            return factory(**options)

        with device_hooks(
            v4_upstream,
            DeepseekV4MLP=DeepseekV4MLP,
            FusedMoEFactory=v4_factory,
        ):
            super().__init__(vllm_config, *args, **kwargs)
