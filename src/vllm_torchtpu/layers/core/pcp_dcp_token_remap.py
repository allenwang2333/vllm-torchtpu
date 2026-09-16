# SPDX-License-Identifier: Apache-2.0
"""Token-dimension gather/scatter for the pcp_size == dcp_size sparse-MLA
  - `gather_to_request_major`: local PCP chunk -> full request-major tensor.
    All-gathers across the PCP group, then reorders.
  - `scatter_to_local_chunk`: the inverse, applied to a DCP-merged output --
    slices back down to this rank's own chunk.

"""

from __future__ import annotations

import torch

from vllm_torchtpu.distributed.pcp import all_gather_equal_tokens


def gather_to_request_major(
    tensor: torch.Tensor,
    request_major_to_packed_token_indices: torch.Tensor | None,
    *,
    pad_to_multiple_of: int | None = None,
) -> torch.Tensor:
    """`[local_padded_tokens, ...]` -> `[real_total_tokens (+padding), ...]`,
    identical on every PCP rank. `None` index tensor returns `tensor` as-is."""
    if request_major_to_packed_token_indices is None:
        return tensor
    packed = all_gather_equal_tokens(tensor, dim=0)
    out = packed.index_select(0, request_major_to_packed_token_indices)
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
) -> torch.Tensor:
    """Inverse of `gather_to_request_major`: request-major tensor -> this
    rank's own local PCP chunk, padding rows zeroed. `None` slice/index
    returns `tensor` as-is."""
    if local_token_slice is None or packed_to_request_major_token_indices is None:
        return tensor
    local_map = packed_to_request_major_token_indices[local_token_slice]
    valid = local_map.ge(0)
    safe_map = local_map.clamp_min(0)
    out = tensor.index_select(0, safe_map)
    mask_shape = (valid.shape[0], ) + (1, ) * (out.dim() - 1)
    return torch.where(valid.reshape(mask_shape), out, torch.zeros_like(out))
