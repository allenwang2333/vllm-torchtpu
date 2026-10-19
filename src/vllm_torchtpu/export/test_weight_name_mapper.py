# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Safetensors weight name resolution and reverse mapping rules."""

from __future__ import annotations

from vllm_torchtpu.export.weight_name_mapper import (
    resolve_safetensor_name,
    resolve_source_tensor_and_path,
)


def test_exact_match_preserves_tensor_name() -> None:
    valid_keys = {
        "model.layers.0.input_layernorm.weight": "/path/to/shard0.safetensors",
        "model.embed_tokens.weight": "/path/to/shard1.safetensors",
    }
    assert (
        resolve_safetensor_name("model.embed_tokens.weight", valid_keys)
        == "model.embed_tokens.weight"
    )
    name, path = resolve_source_tensor_and_path(
        "model.embed_tokens.weight", valid_keys, default_path="/default.safetensors"
    )
    assert name == "model.embed_tokens.weight"
    assert path == "/path/to/shard1.safetensors"


def test_multimodal_vlm_language_model_prefix_remapping() -> None:
    # 397B case: PyTorch module name is model.embed_tokens.weight,
    # but HuggingFace Safetensors archive has model.language_model.embed_tokens.weight
    valid_keys = {
        "model.language_model.embed_tokens.weight": "/path/to/model-00091.safetensors",
        (
            "model.language_model.layers.0.input_layernorm.weight"
        ): "/path/to/model-00094.safetensors",
    }
    assert (
        resolve_safetensor_name("model.embed_tokens.weight", valid_keys)
        == "model.language_model.embed_tokens.weight"
    )
    assert (
        resolve_safetensor_name("model.layers.0.input_layernorm.weight", valid_keys)
        == "model.language_model.layers.0.input_layernorm.weight"
    )
    name, path = resolve_source_tensor_and_path(
        "model.embed_tokens.weight", valid_keys, default_path=""
    )
    assert name == "model.language_model.embed_tokens.weight"
    assert path == "/path/to/model-00091.safetensors"


def test_moe_router_gate_experts_infix_remapping() -> None:
    # 480B case: PyTorch module name is model.layers.0.mlp.experts.gate.weight,
    # but HuggingFace Safetensors archive has model.layers.0.mlp.gate.weight
    valid_keys = {
        "model.layers.0.mlp.gate.weight": "/path/to/model-00001.safetensors",
        "model.layers.1.mlp.gate.weight": "/path/to/model-00001.safetensors",
    }
    assert (
        resolve_safetensor_name("model.layers.0.mlp.experts.gate.weight", valid_keys)
        == "model.layers.0.mlp.gate.weight"
    )
    name, path = resolve_source_tensor_and_path(
        "model.layers.0.mlp.experts.gate.weight", valid_keys, default_path=""
    )
    assert name == "model.layers.0.mlp.gate.weight"
    assert path == "/path/to/model-00001.safetensors"


def test_tied_embeddings_lm_head_remapping() -> None:
    # Tied embeddings: lm_head.weight maps back to model.embed_tokens.weight
    # or model.language_model.embed_tokens.weight
    valid_keys_standard = {
        "model.embed_tokens.weight": "/path/to/embed.safetensors",
    }
    assert (
        resolve_safetensor_name("lm_head.weight", valid_keys_standard)
        == "model.embed_tokens.weight"
    )

    valid_keys_vlm = {
        "model.language_model.embed_tokens.weight": "/path/to/vlm_embed.safetensors",
    }
    assert (
        resolve_safetensor_name("lm_head.weight", valid_keys_vlm)
        == "model.language_model.embed_tokens.weight"
    )


def test_unmatched_tensor_falls_back_gracefully() -> None:
    valid_keys = {"other.weight": "/path/to/other.safetensors"}
    assert resolve_safetensor_name("non_existent.weight", valid_keys) is None
    name, path = resolve_source_tensor_and_path(
        "non_existent.weight", valid_keys, default_path="/fallback.safetensors"
    )
    assert name == "non_existent.weight"
    assert path == "/fallback.safetensors"


def test_gdn_linear_attention_composite_expansion() -> None:
    # 397B GDN attention: in_proj_qkvz.weight [20480, 4096] expands to
    # in_proj_qkv (12288) and in_proj_z (8192)
    valid_keys = {
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "/path/qkv.st",
        "model.language_model.layers.0.linear_attn.in_proj_z.weight": "/path/z.st",
    }
    w_name = "model.layers.0.linear_attn.in_proj_qkvz.weight"
    w_spec = {"shape": [20480, 4096]}
    from vllm_torchtpu.export.weight_name_mapper import expand_or_resolve_weight_sources

    sources = expand_or_resolve_weight_sources(w_name, w_spec, valid_keys, "")
    assert len(sources) == 2
    assert (
        sources[0]["source_tensor"]
        == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    )
    assert sources[0]["path"] == "/path/qkv.st"
    assert sources[0]["dst_slices"] == [{"dim": 0, "start": 0, "length": 12288}]

    assert (
        sources[1]["source_tensor"]
        == "model.language_model.layers.0.linear_attn.in_proj_z.weight"
    )
    assert sources[1]["path"] == "/path/z.st"
    assert sources[1]["dst_slices"] == [{"dim": 0, "start": 12288, "length": 8192}]


def test_moe_routed_experts_composite_expansion() -> None:
    # 480B / MoE: routed_experts.w13_weight [2, 6144, 5120] expands per-expert
    # into gate_proj and up_proj with rank-3 source_shape and transpose [0, 2, 1]
    valid_keys = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": "/path/e0.st",
        "model.layers.0.mlp.experts.0.up_proj.weight": "/path/e0.st",
        "model.layers.0.mlp.experts.1.gate_proj.weight": "/path/e1.st",
        "model.layers.0.mlp.experts.1.up_proj.weight": "/path/e1.st",
    }
    w_name = "model.layers.0.mlp.experts.routed_experts.w13_weight"
    w_spec = {"shape": [2, 6144, 5120]}
    from vllm_torchtpu.export.weight_name_mapper import expand_or_resolve_weight_sources

    sources = expand_or_resolve_weight_sources(w_name, w_spec, valid_keys, "")
    assert len(sources) == 4
    # expert 0 gate
    assert (
        sources[0]["source_tensor"] == "model.layers.0.mlp.experts.0.gate_proj.weight"
    )
    assert sources[0]["path"] == "/path/e0.st"
    assert sources[0]["source_shape"] == [1, 2560, 6144]
    assert sources[0]["transpose"] == [0, 2, 1]
    assert sources[0]["dst_slices"] == [
        {"dim": 0, "start": 0, "length": 1},
        {"dim": 2, "start": 0, "length": 2560},
    ]
    # expert 0 up
    assert sources[1]["source_tensor"] == "model.layers.0.mlp.experts.0.up_proj.weight"
    assert sources[1]["source_shape"] == [1, 2560, 6144]
    assert sources[1]["transpose"] == [0, 2, 1]
    assert sources[1]["dst_slices"] == [
        {"dim": 0, "start": 0, "length": 1},
        {"dim": 2, "start": 2560, "length": 2560},
    ]
    # expert 1 gate
    assert (
        sources[2]["source_tensor"] == "model.layers.0.mlp.experts.1.gate_proj.weight"
    )
    assert sources[2]["source_shape"] == [1, 2560, 6144]
    assert sources[2]["transpose"] == [0, 2, 1]
    assert sources[2]["dst_slices"] == [
        {"dim": 0, "start": 1, "length": 1},
        {"dim": 2, "start": 0, "length": 2560},
    ]


def test_gdn_in_proj_ba_expansion() -> None:
    # 397B GDN in_proj_ba: target [4096, 128], sources in_proj_b and in_proj_a
    valid_keys = {
        "model.language_model.layers.0.linear_attn.in_proj_b.weight": "/path/b.st",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": "/path/a.st",
    }
    w_name = "model.layers.0.linear_attn.in_proj_ba.weight"
    w_spec = {"shape": [4096, 128]}
    from vllm_torchtpu.export.weight_name_mapper import expand_or_resolve_weight_sources

    sources = expand_or_resolve_weight_sources(w_name, w_spec, valid_keys, "")
    assert len(sources) == 2
    assert (
        sources[0]["source_tensor"]
        == "model.language_model.layers.0.linear_attn.in_proj_b.weight"
    )
    assert sources[0]["source_shape"] == [64, 4096]
    assert sources[0]["transpose"] == [1, 0]
    assert sources[0]["dst_slices"] == [{"dim": 1, "start": 0, "length": 64}]

    assert (
        sources[1]["source_tensor"]
        == "model.language_model.layers.0.linear_attn.in_proj_a.weight"
    )
    assert sources[1]["source_shape"] == [64, 4096]
    assert sources[1]["transpose"] == [1, 0]
    assert sources[1]["dst_slices"] == [{"dim": 1, "start": 64, "length": 64}]


def test_shared_experts_gate_up_expansion() -> None:
    # 397B shared experts: target [4096, 2048], sources gate_proj and up_proj
    valid_keys = {
        (
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"
        ): "/path/gate.st",
        (
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight"
        ): "/path/up.st",
    }
    w_name = "model.layers.0.mlp.experts.shared_experts.layer.gate_up_proj.weight"
    w_spec = {"shape": [4096, 2048]}
    from vllm_torchtpu.export.weight_name_mapper import expand_or_resolve_weight_sources

    sources = expand_or_resolve_weight_sources(w_name, w_spec, valid_keys, "")
    assert len(sources) == 2
    assert (
        sources[0]["source_tensor"]
        == "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"
    )
    assert sources[0]["source_shape"] == [1024, 4096]
    assert sources[0]["transpose"] == [1, 0]
    assert sources[0]["dst_slices"] == [{"dim": 1, "start": 0, "length": 1024}]

    assert (
        sources[1]["source_tensor"]
        == "model.language_model.layers.0.mlp.shared_expert.up_proj.weight"
    )
    assert sources[1]["source_shape"] == [1024, 4096]
    assert sources[1]["transpose"] == [1, 0]
    assert sources[1]["dst_slices"] == [{"dim": 1, "start": 1024, "length": 1024}]
