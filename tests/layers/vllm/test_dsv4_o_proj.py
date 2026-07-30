from unittest.mock import MagicMock, patch

import torch

from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4.deepseek_v4_attention import \
    VllmDeepseekV4MLAAttention


class DummyRotaryEmbedding:

    def __call__(self, positions, x, inverse=False):
        # Simply return the input to bypass rotary math in shapes check
        return x, None


def test_o_proj_dequantization_and_sharding_tp1():

    class TestLayer:
        pass

    layer = TestLayer()

    # Setup parameters matching DeepSeek-V4 layout
    # o_lora_rank = 1024, head_dim = 512, n_local_heads = 8, o_groups = 8.
    # hidden_size = 4096
    layer.mla_attn = MagicMock()
    layer.mla_attn.n_local_heads = 8
    layer.mla_attn.head_dim = 512
    layer.mla_attn.o_lora_rank = 1024
    layer.mla_attn.o_groups = 8
    layer.mla_attn.rotary_emb = DummyRotaryEmbedding()

    layer.tp_size = 1  # TP=1 (single rank)
    layer.custom_prefix = "model.layers.2.attn"
    layer.n_local_groups = 8
    layer.o_lora_rank = 1024

    # 1. Mock wo_a (Grouped Projection)
    # Shape of weight: [o_lora_rank, hidden_size] = [1024, 4096]
    # Under blockwise quantization, weight_scale is [hidden_size // 128, 1, o_lora_rank]
    # For TP=1, it covers all 8 groups of size 128.
    wo_a = MagicMock()
    wo_a.weight = torch.ones(1024, 4096,
                             dtype=torch.int8)  # block-quantized weights
    wo_a.weight_scale = torch.ones(
        32, 1, 1024,
        dtype=torch.bfloat16) * 0.5  # block-scales (4096/128 = 32 blocks)
    layer.wo_a = wo_a

    # 2. Mock wo_b (RowParallelLinear)
    # Shape of weight: [hidden_size, o_lora_rank] = [4096, 1024]
    class MockWoB:

        def __init__(self, in_features, out_features):
            self.weight = torch.ones(out_features,
                                     in_features,
                                     dtype=torch.int8)
            self.weight_scale = torch.ones(in_features,
                                           dtype=torch.bfloat16) * 0.25

        def __call__(self, x):
            return torch.ones(x.shape[0],
                              self.weight.shape[0],
                              dtype=torch.bfloat16), None

    layer.wo_b = MockWoB(1024, 4096)

    # Bind the _o_proj method from VllmDeepseekV4MLAAttention
    layer._o_proj = VllmDeepseekV4MLAAttention._o_proj.__get__(
        layer, TestLayer)

    # Run _o_proj forward pass under mock rank=0
    # Inputs: activations 'o' of shape [tokens, n_local_heads * head_dim] = [3, 8 * 512] = [3, 4096]
    # and positions [3]
    o = torch.ones(3, 4096, dtype=torch.bfloat16)
    positions = torch.arange(3, dtype=torch.int64)

    # Mock torch.distributed rank lookup to return 0
    with patch("torch.distributed.get_rank", return_value=0):
        out = layer._o_proj(o, positions)

    # Verify shape of output is [tokens, hidden_size] = [3, 4096]
    assert out.shape == (3, 4096)


def test_o_proj_dequantization_and_sharding_tp8():

    class TestLayer:
        pass

    layer = TestLayer()

    layer.mla_attn = MagicMock()
    layer.mla_attn.n_local_heads = 8
    layer.mla_attn.head_dim = 512
    layer.mla_attn.o_lora_rank = 1024
    layer.mla_attn.o_groups = 8
    layer.mla_attn.rotary_emb = DummyRotaryEmbedding()

    layer.tp_size = 8  # TP=8 (sharded layout)
    layer.custom_prefix = "model.layers.2.attn"
    layer.n_local_groups = 1
    layer.o_lora_rank = 1024

    # 1. Mock wo_a (Grouped Projection)
    # Shape of weight: [o_lora_rank, hidden_size] = [1024, 4096]
    # Under blockwise quantization, weight_scale is [hidden_size // 128, 1, o_lora_rank]
    # For TP=8, we slice 1/8th of the rows: group_size = 1024 / 8 = 128 rows.
    # So each rank only uses its corresponding group slice.
    wo_a = MagicMock()
    wo_a.weight = torch.ones(1024, 4096, dtype=torch.int8)
    wo_a.weight_scale = torch.ones(32, 1, 1024, dtype=torch.bfloat16) * 0.5
    layer.wo_a = wo_a

    # 2. Mock wo_b (RowParallelLinear)
    # Shape of weight: [hidden_size, o_lora_rank] = [4096, 1024]
    # For RowParallelLinear under TP=8, the columns are sharded: output = [4096, 1024 // 8] = [4096, 128]
    class MockWoB:

        def __init__(self, in_features, out_features):
            self.weight = torch.ones(out_features,
                                     in_features,
                                     dtype=torch.int8)
            self.weight_scale = torch.ones(in_features,
                                           dtype=torch.bfloat16) * 0.25

        def __call__(self, x):
            return torch.ones(x.shape[0],
                              self.weight.shape[0],
                              dtype=torch.bfloat16), None

    layer.wo_b = MockWoB(128, 4096)

    # Bind the _o_proj method
    layer._o_proj = VllmDeepseekV4MLAAttention._o_proj.__get__(
        layer, TestLayer)

    o = torch.ones(3, 4096, dtype=torch.bfloat16)
    positions = torch.arange(3, dtype=torch.int64)

    # Mock rank=3
    with patch("torch.distributed.get_rank", return_value=3):
        out = layer._o_proj(o, positions)

    # Verify shape of output is [tokens, hidden_size] = [3, 4096]
    # Under TP=8, RowParallelLinear still outputs [3, 4096] after TP all-reduce (summing sharded outputs)
    assert out.shape == (3, 4096)
