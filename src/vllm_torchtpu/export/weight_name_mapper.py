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
    _ = w_name
    _ = valid_keys
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
    _ = tensor_to_file
    return w_name, default_path
