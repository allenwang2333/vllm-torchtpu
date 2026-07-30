import torch
from types import SimpleNamespace
from unittest.mock import MagicMock

def test_deepseek_v4_weight_mapping(monkeypatch):
    from vllm.config import VllmConfig
    import vllm.models.deepseek_v4 as ds_v4
    from vllm_torchtpu.models.vllm.deepseek_v4_patch import _maybe_patch_for_deepseek_v4

    # Setup mock VllmConfig satisfying is_ds_v4
    mock_hf_config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
    mock_model_config = SimpleNamespace(hf_config=mock_hf_config)
    mock_quant_config = SimpleNamespace(expert_dtype="fp8")
    mock_vllm_config = SimpleNamespace(
        model_config=mock_model_config,
        quant_config=mock_quant_config,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
    )

    # Mock the sharding config manager so it doesn't fail when initializing context
    mock_sharding_config = SimpleNamespace(tp_size=1, expert_size=1)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.common.sharding.ShardingConfigManager.get",
        lambda *args, **kwargs: mock_sharding_config,
    )

    # Capture the mapper when the model's load_weights gets called
    captured_mapper = [None]

    # Pre-mock the AMD variant class method directly to prevent traversing named children
    import vllm.models.deepseek_v4.amd.model as amd_mod
    def mock_load_weights(self, weights):
        captured_mapper[0] = self.hf_to_vllm_mapper
        return None
    monkeypatch.setattr(amd_mod.DeepseekV4ForCausalLM, "load_weights", mock_load_weights)

    # Enter patch context
    with _maybe_patch_for_deepseek_v4(mock_vllm_config):
        # Create mock model instance with necessary fields
        model = SimpleNamespace()
        model.config = SimpleNamespace(num_hidden_layers=4)
        model.model = SimpleNamespace(layers=[SimpleNamespace() for _ in range(4)])
        model.model.named_modules = lambda: []
        model.hf_to_vllm_mapper = SimpleNamespace(
            apply=lambda w: [(w[0], torch.zeros(1))]
        )
        model.named_parameters = lambda: []
        model.parameters = lambda: []

        # Trigger patched_load_weights
        ds_v4.DeepseekV4ForCausalLM.load_weights(model, [])

    mapper = captured_mapper[0]
    assert mapper is not None

    # Test cases mapping HF keys -> Expected internal keys
    # Format: (input_hf_key, expected_vllm_key)
    test_cases = [
        # Gate / Up / Down mapping for shared experts
        (
            ("model.layers.2.shared_experts.gate_proj.weight", torch.zeros(1)),
            "model.layers.2.shared_experts.w1.weight"
        ),
        (
            ("model.layers.2.shared_experts.up_proj.weight", torch.zeros(1)),
            "model.layers.2.shared_experts.w3.weight"
        ),
        # Indexer gate_proj -> wgate
        (
            ("model.layers.2.self_attn.compressor.indexer.gate_proj.weight", torch.zeros(1)),
            "model.layers.2.attn.indexer.compressor.wgate.weight"
        ),
        # MLA attention compressor -> mla_attn
        (
            ("model.layers.2.self_attn.compressor.gate_proj.weight", torch.zeros(1)),
            "model.layers.2.attn.mla_attn.compressor.wgate.weight"
        ),
        # Input layernorm -> attn_norm
        (
            ("model.layers.2.input_layernorm.weight", torch.zeros(1)),
            "model.layers.2.attn_norm.weight"
        ),
        # MLP -> FFN
        (
            ("model.layers.2.mlp.experts.0.gate_proj.weight", torch.zeros(1)),
            "model.layers.2.ffn.experts.0.gate_proj.weight"
        ),
        # HC head replacements
        (
            ("model.layers.2.self_attn.hc_head.hc_attn.base", torch.zeros(1)),
            "model.layers.2.self_attn.hc_head_attn.base"
        ),
    ]


    for input_pair, expected_key in test_cases:
        mapped = list(mapper.apply([input_pair]))
        assert len(mapped) == 1
        assert mapped[0][0] == expected_key
