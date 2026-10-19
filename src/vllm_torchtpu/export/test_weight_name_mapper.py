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
