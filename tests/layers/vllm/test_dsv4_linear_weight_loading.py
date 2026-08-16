from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_torchtpu.layers.common.quant_methods import DEEPSEEK_V4_FP8
from vllm_torchtpu.layers.vllm.quantization.fp8 import VllmFp8LinearMethodTPU


class _DummyDeepseekV4Fp8Config(Fp8Config):
    """Stands in for VllmDeepseekV4Fp8Config's get_name() without pulling in
    its real construction (which needs a live vllm_config for is_scale_e8m0)."""

    @classmethod
    def get_name(cls) -> str:
        return DEEPSEEK_V4_FP8


def test_linear_weight_loading_quant_config_resolving():
    # Setup global dummy config
    dummy_quant_config = _DummyDeepseekV4Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[1, 32],
    )

    # block_quant follows the checkpoint's weight_block_size, and the prefix
    # does not change it: dense and routed-expert layers resolve identically,
    # matching tpu-inference.
    for prefix in ("model.layers.0.self_attn.fused_wqa_wkv",
                   "model.layers.0.mlp.experts.0.w13"):
        method = VllmFp8LinearMethodTPU(
            quant_config=dummy_quant_config,
            prefix=prefix,
        )
        assert method.block_quant is True, f"Expected block_quant True for {prefix!r}"
        assert method.weight_block_size == [
            1, 32
        ], f"Expected weight_block_size [1, 32] for {prefix!r}"

    print(
        "VllmFp8LinearMethodTPU weight loading quant config resolving test passed successfully!"
    )


def test_linear_weight_loading_unquantized_fallback():
    import torch
    dummy_quant_config = _DummyDeepseekV4Fp8Config(
        is_checkpoint_fp8_serialized=False,
        activation_scheme="dynamic",
        weight_block_size=None,
    )
    dense_method = VllmFp8LinearMethodTPU(
        quant_config=dummy_quant_config,
        prefix="model.layers.0.self_attn.fused_wqa_wkv",
    )
    layer = torch.nn.Linear(64, 32, bias=False)
    dense_method.process_weights_after_loading(layer)
    assert hasattr(layer, "weight")
    assert hasattr(layer, "weight_scale")
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_block_size == (1, 64)


if __name__ == "__main__":
    test_linear_weight_loading_quant_config_resolving()
    test_linear_weight_loading_unquantized_fallback()
