# SPDX-License-Identifier: Apache-2.0
"""Token-dimension gather/scatter for the pcp_size + dcp_size sparse-MLA."""

from __future__ import annotations

import torch

from vllm_torchtpu.distributed.pcp import all_gather_equal_tokens


def gather_to_request_major(
    tensor: torch.Tensor,
    request_major_to_packed_token_indices: torch.Tensor | None,
    *,
    pad_to_multiple_of: int | None = None,
) -> torch.Tensor:
    if request_major_to_packed_token_indices is None:
        return tensor
    packed = all_gather_equal_tokens(tensor, dim=0)
    idx = request_major_to_packed_token_indices[: packed.shape[0]]
    out = packed.index_select(0, idx)
    if pad_to_multiple_of and pad_to_multiple_of > 1:
        remainder = out.shape[0] % pad_to_multiple_of
        if remainder:
            pad_rows = pad_to_multiple_of - remainder
            out = torch.cat(
                [out, out.new_zeros((pad_rows, ) + out.shape[1:])], dim=0)
    return out


def scatter_to_local_chunk(
    tensor: torch.Tensor,
    local_token_slice: slice | None,
    packed_to_request_major_token_indices: torch.Tensor | None,
    *,
    local_num_tokens: int,
) -> torch.Tensor:
    del local_token_slice
    if packed_to_request_major_token_indices is None:
        return tensor
    idx = packed_to_request_major_token_indices[:local_num_tokens]
    return tensor.index_select(0, idx)
