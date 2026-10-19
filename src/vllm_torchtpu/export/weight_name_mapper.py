# SPDX-License-Identifier: Apache-2.0
"""Safetensors checkpoint tensor name resolution and reverse mapping rules."""

from __future__ import annotations

from collections.abc import Collection


def resolve_safetensor_name(
    w_name: str,
    valid_keys: Collection[str],
) -> str | None:
    """Resolves an internal PyTorch parameter name to its corresponding Safetensors key.

    Applies a prioritized rule chain to bridge PyTorch module hierarchical naming
    with upstream HuggingFace Safetensors checkpoint key naming conventions.
    """
    # Rule 0: Exact match
    if w_name in valid_keys:
        return w_name

    # Rule 1: Tied embeddings (lm_head.weight)
    if w_name == "lm_head.weight":
        for cand in (
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
            "transformer.wte.weight",
            "embed_tokens.weight",
        ):
            if cand in valid_keys:
                return cand

    # Rule 2: Multimodal / VLM decoupled language model prefix
    # Under --language-model-only, PyTorch module name strips
    # "model.language_model." to "model."
    if w_name.startswith("model."):
        vlm_cand = w_name.replace("model.", "model.language_model.", 1)
        if vlm_cand in valid_keys:
            return vlm_cand
        lm_cand = w_name.replace("model.", "language_model.", 1)
        if lm_cand in valid_keys:
            return lm_cand

    if w_name.startswith("language_model."):
        vlm_cand = f"model.{w_name}"
        if vlm_cand in valid_keys:
            return vlm_cand

    # Rule 3: MoE routing gate infix remapping
    # PyTorch modules nest gate under experts container (.mlp.experts.gate.weight),
    # while Safetensors archives store .mlp.gate.weight
    if ".mlp.experts.gate." in w_name:
        moe_cand = w_name.replace(".mlp.experts.gate.", ".mlp.gate.")
        if moe_cand in valid_keys:
            return moe_cand

    if ".block_sparse_moe.experts.gate." in w_name:
        moe_cand = w_name.replace(
            ".block_sparse_moe.experts.gate.", ".block_sparse_moe.gate."
        )
        if moe_cand in valid_keys:
            return moe_cand

    # Rule 4: General prefix normalization
    if not w_name.startswith("model."):
        model_prefixed = f"model.{w_name}"
        if model_prefixed in valid_keys:
            return model_prefixed

    return None


def resolve_source_tensor_and_path(
    w_name: str,
    tensor_to_file: dict[str, str],
    default_path: str = "",
) -> tuple[str, str]:
    """Resolves the Safetensors source tensor identifier and shard file path.

    Returns (source_tensor_name, file_path). If no mapping is found in tensor_to_file,
    returns (w_name, default_path).
    """
    resolved = resolve_safetensor_name(w_name, tensor_to_file)
    if resolved is not None:
        return resolved, tensor_to_file.get(resolved, default_path)
    return w_name, default_path
