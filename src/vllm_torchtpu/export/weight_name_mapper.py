# SPDX-License-Identifier: Apache-2.0
"""Safetensors checkpoint tensor name resolution and reverse mapping rules."""

from __future__ import annotations

import re
from collections.abc import Collection
from typing import Any


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

    # Rule 2: Special case casing (e.g. a_log -> A_log)
    if ".linear_attn.a_log" in w_name:
        casing_cand = w_name.replace(".linear_attn.a_log", ".linear_attn.A_log")
        res = resolve_safetensor_name(casing_cand, valid_keys)
        if res is not None:
            return res

    # Rule 3: MoE routing gate infix remapping
    # PyTorch modules nest gate under experts container (.mlp.experts.gate.weight),
    # while Safetensors archives store .mlp.gate.weight
    if ".mlp.experts.gate." in w_name:
        moe_cand = w_name.replace(".mlp.experts.gate.", ".mlp.gate.")
        res = resolve_safetensor_name(moe_cand, valid_keys)
        if res is not None:
            return res

    if ".block_sparse_moe.experts.gate." in w_name:
        moe_cand = w_name.replace(
            ".block_sparse_moe.experts.gate.", ".block_sparse_moe.gate."
        )
        res = resolve_safetensor_name(moe_cand, valid_keys)
        if res is not None:
            return res

    # Rule 4: Shared expert naming remapping
    # .mlp.experts.shared_experts.layer.down_proj.weight ->
    # .mlp.shared_expert.down_proj.weight
    if ".mlp.experts.shared_experts.layer." in w_name:
        shared_cand = w_name.replace(
            ".mlp.experts.shared_experts.layer.", ".mlp.shared_expert."
        )
        if ".expert_gate." in shared_cand:
            shared_cand = shared_cand.replace(
                ".mlp.shared_expert.expert_gate.", ".mlp.shared_expert_gate."
            )
        res = resolve_safetensor_name(shared_cand, valid_keys)
        if res is not None:
            return res

    # Rule 5: Multimodal / VLM decoupled language model prefix
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

    # Rule 6: General prefix normalization
    if not w_name.startswith("model."):
        model_prefixed = f"model.{w_name}"
        res = resolve_safetensor_name(model_prefixed, valid_keys)
        if res is not None:
            return res

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


def expand_or_resolve_weight_sources(
    w_name: str,
    w_spec: dict[str, Any],
    tensor_to_file: dict[str, str],
    default_path: str = "",
) -> list[dict[str, Any]]:
    """Expands composite/packed target weights or resolves 1-to-1 weight sources."""
    shape = [int(d) for d in w_spec.get("shape", []) if isinstance(d, int)]

    # 1. GDN Attention: in_proj_qkvz.weight [20480, hidden_dim]
    # Split into in_proj_qkv (12288) and in_proj_z (8192)
    if ".linear_attn.in_proj_qkvz.weight" in w_name and len(shape) == 2:
        in_dim = shape[1]
        qkv_cand_name = w_name.replace(".in_proj_qkvz.weight", ".in_proj_qkv.weight")
        z_cand_name = w_name.replace(".in_proj_qkvz.weight", ".in_proj_z.weight")
        qkv_resolved = resolve_safetensor_name(qkv_cand_name, tensor_to_file)
        z_resolved = resolve_safetensor_name(z_cand_name, tensor_to_file)
        if qkv_resolved is not None and z_resolved is not None:
            return [
                {
                    "path": tensor_to_file.get(qkv_resolved, default_path),
                    "source_tensor": qkv_resolved,
                    "source_shape": [12288, in_dim],
                    "dst_slices": [{"dim": 0, "start": 0, "length": 12288}],
                },
                {
                    "path": tensor_to_file.get(z_resolved, default_path),
                    "source_tensor": z_resolved,
                    "source_shape": [8192, in_dim],
                    "dst_slices": [{"dim": 0, "start": 12288, "length": 8192}],
                },
            ]

    # 2. GDN Attention: in_proj_ba.weight [4096, hidden_dim]
    # Split into in_proj_b (2048) and in_proj_a (2048)
    if ".linear_attn.in_proj_ba.weight" in w_name and len(shape) == 2:
        in_dim = shape[1]
        b_cand_name = w_name.replace(".in_proj_ba.weight", ".in_proj_b.weight")
        a_cand_name = w_name.replace(".in_proj_ba.weight", ".in_proj_a.weight")
        b_resolved = resolve_safetensor_name(b_cand_name, tensor_to_file)
        a_resolved = resolve_safetensor_name(a_cand_name, tensor_to_file)
        if b_resolved is not None and a_resolved is not None:
            return [
                {
                    "path": tensor_to_file.get(b_resolved, default_path),
                    "source_tensor": b_resolved,
                    "source_shape": [2048, in_dim],
                    "dst_slices": [{"dim": 0, "start": 0, "length": 2048}],
                },
                {
                    "path": tensor_to_file.get(a_resolved, default_path),
                    "source_tensor": a_resolved,
                    "source_shape": [2048, in_dim],
                    "dst_slices": [{"dim": 0, "start": 2048, "length": 2048}],
                },
            ]

    # 3. Shared Experts: gate_up_proj.weight [2 * intermediate, in_dim]
    # Split into gate_proj and up_proj
    if (
        ".mlp.experts.shared_experts.layer.gate_up_proj.weight" in w_name
        and len(shape) == 2
    ):
        intermediate = shape[0] // 2
        in_dim = shape[1]
        gate_cand = w_name.replace("gate_up_proj.weight", "gate_proj.weight")
        up_cand = w_name.replace("gate_up_proj.weight", "up_proj.weight")
        gate_resolved = resolve_safetensor_name(gate_cand, tensor_to_file)
        up_resolved = resolve_safetensor_name(up_cand, tensor_to_file)
        if gate_resolved is not None and up_resolved is not None:
            return [
                {
                    "path": tensor_to_file.get(gate_resolved, default_path),
                    "source_tensor": gate_resolved,
                    "source_shape": [intermediate, in_dim],
                    "dst_slices": [{"dim": 0, "start": 0, "length": intermediate}],
                },
                {
                    "path": tensor_to_file.get(up_resolved, default_path),
                    "source_tensor": up_resolved,
                    "source_shape": [intermediate, in_dim],
                    "dst_slices": [
                        {"dim": 0, "start": intermediate, "length": intermediate}
                    ],
                },
            ]

    # 4. MoE Routed Experts: routed_experts.w13_weight
    # [num_experts, hidden_dim, 2 * intermediate]
    if ".mlp.experts.routed_experts.w13_weight" in w_name and len(shape) == 3:
        num_local_experts, hidden_dim, total_inter = shape
        intermediate_dim = total_inter // 2
        layer_match = re.search(r"layers\.(\d+)\.", w_name)
        if layer_match:
            layer_idx = layer_match.group(1)
            sources: list[dict[str, Any]] = []
            matched_all = True
            for e in range(num_local_experts):
                gate_key_cand = (
                    f"model.layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"
                )
                up_key_cand = f"model.layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"
                gate_res = resolve_safetensor_name(gate_key_cand, tensor_to_file)
                up_res = resolve_safetensor_name(up_key_cand, tensor_to_file)
                if gate_res is None or up_res is None:
                    matched_all = False
                    break
                sources.append(
                    {
                        "path": tensor_to_file.get(gate_res, default_path),
                        "source_tensor": gate_res,
                        "source_shape": [intermediate_dim, hidden_dim],
                        "transpose": [1, 0],
                        "dst_slices": [
                            {"dim": 0, "start": e, "length": 1},
                            {"dim": 2, "start": 0, "length": intermediate_dim},
                        ],
                    }
                )
                sources.append(
                    {
                        "path": tensor_to_file.get(up_res, default_path),
                        "source_tensor": up_res,
                        "source_shape": [intermediate_dim, hidden_dim],
                        "transpose": [1, 0],
                        "dst_slices": [
                            {"dim": 0, "start": e, "length": 1},
                            {
                                "dim": 2,
                                "start": intermediate_dim,
                                "length": intermediate_dim,
                            },
                        ],
                    }
                )
            if matched_all and sources:
                return sources

    # 5. MoE Routed Experts: routed_experts.w2_weight
    # [num_experts, intermediate, hidden_dim]
    if ".mlp.experts.routed_experts.w2_weight" in w_name and len(shape) == 3:
        num_local_experts, intermediate_dim, hidden_dim = shape
        layer_match = re.search(r"layers\.(\d+)\.", w_name)
        if layer_match:
            layer_idx = layer_match.group(1)
            sources = []
            matched_all = True
            for e in range(num_local_experts):
                down_key_cand = (
                    f"model.layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"
                )
                down_res = resolve_safetensor_name(down_key_cand, tensor_to_file)
                if down_res is None:
                    matched_all = False
                    break
                sources.append(
                    {
                        "path": tensor_to_file.get(down_res, default_path),
                        "source_tensor": down_res,
                        "source_shape": [hidden_dim, intermediate_dim],
                        "transpose": [1, 0],
                        "dst_slices": [{"dim": 0, "start": e, "length": 1}],
                    }
                )
            if matched_all and sources:
                return sources

    # 6. Fallback: if w_spec already has sources, resolve each source_tensor
    existing_sources = w_spec.get("sources")
    if isinstance(existing_sources, list) and existing_sources:
        resolved_sources: list[dict[str, Any]] = []
        for src in existing_sources:
            if not isinstance(src, dict):
                continue
            entry = dict(src)
            raw = str(entry.get("source_tensor") or w_name)
            st_name, path = resolve_source_tensor_and_path(
                raw, tensor_to_file, default_path
            )
            entry["source_tensor"] = st_name
            entry["path"] = path
            if "source_shape" not in entry and shape:
                entry["source_shape"] = shape
            resolved_sources.append(entry)
        if resolved_sources:
            return resolved_sources

    # 7. Single 1-to-1 default resolution
    st_name, path = resolve_source_tensor_and_path(w_name, tensor_to_file, default_path)
    return [
        {
            "path": path,
            "source_tensor": st_name,
            "source_shape": shape,
        }
    ]
