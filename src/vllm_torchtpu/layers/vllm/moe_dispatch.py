# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Route before the DP dispatch, and send the decision instead of the logits.

Today every DP rank all-gathers the raw ``[T_local, num_experts]`` router
logits and then runs the *same* softmax + top-k over all ``T_global`` rows --
``dp_size``-fold duplicated work, and 2 bytes per expert per token on the
wire. Routing is a per-row function, so each rank can decide for its own rows
and ship the decision: ``topk`` ids and ``topk`` weights, packed into one
int32 tensor that rides the all-gather the logits used to.

The pack is a plain concatenation with the weights bitcast, so the round trip
is bit-exact and the gathered blob is byte-identical to what the receiving
rank would have computed itself.

What this deliberately does *not* move is the padding mask. Only two of the six
quant methods that reach ``moe_routing.route`` zero padded rows at all, one of
them behind its own flag, so zeroing here would zero rows the flag-off path
computed on four of the six -- a semantic change from a performance flag. The
mask therefore stays exactly where it is, after the dispatch, on the gathered
rows. It costs one ``s8`` all-gather per *step* (XLA CSEs the per-layer copies),
about 10 us against a 384,000 us prefill step.
"""

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The blob is int32; router logits are always floating point, so the dtype
# alone tells the receiver which of the two it is holding.
BLOB_DTYPE = torch.int32


def pack_routing(topk_ids: torch.Tensor,
                 topk_weights: torch.Tensor) -> torch.Tensor:
    """``([T,K] ids, [T,K] weights)`` -> one flat ``[T*2K]`` int32 tensor.

    Weights are widened to f32 and bitcast rather than cast, so unpacking
    returns the exact bits that went in (bf16 -> f32 -> bf16 is exact).

    **Rank 1 is the point, not an accident.** As ``[T, 2K]`` the blob's layout
    is not ours to choose: a Mosaic consumer downstream of the unpack carries
    ``operand_layout_constraints={s32[T,K]{1,0}}``, and XLA satisfies that by
    propagating the constraint back through the slice and the bitcast into the
    all-gather itself rather than relaying out afterwards -- so the collective
    ships ``{1,0:T(8,128)}`` with 20 lanes padded to 128, 2,048 KiB/rank. A
    rank-1 tensor has exactly one layout and no consumer can flip it:
    ``{0:T(1024)}``, 320 KiB/rank, which also beats the ``[T,2K]{0,1}`` best
    case we had assumed (384 KiB) because ``T(1024)`` pads nothing.

    Row-major flatten, so rank *r*'s rows stay contiguous and the existing
    ``all_gather(dim=0)`` concatenates the ranks exactly as it did in 2-D.
    """
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(f"routing shapes disagree: {topk_ids.shape} vs "
                         f"{topk_weights.shape}")
    ids = topk_ids.to(BLOB_DTYPE)
    bits = topk_weights.float().view(BLOB_DTYPE)
    return torch.cat((ids, bits), dim=-1).reshape(-1)


def unpack_routing(
        blob: torch.Tensor, topk: int,
        weights_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`pack_routing`; returns ``(weights, ids)``.

    ``reshape`` on the contiguous gathered blob is a view, so this adds no
    copy -- pinned by the codec's no-copy test, because a materialising
    reshape here would hand back the bytes the 1-D form exists to save.
    """
    width = 2 * topk
    if blob.dim() != 1 or blob.numel() % width != 0:
        raise ValueError(
            f"routing blob has shape {tuple(blob.shape)}, expected a flat "
            f"[T*{width}]")
    rows = blob.reshape(-1, width)
    ids = rows[:, :topk]
    weights = rows[:, topk:].view(torch.float32).to(weights_dtype)
    return weights, ids


def is_routing_blob(router_logits: torch.Tensor) -> bool:
    return router_logits.dtype == BLOB_DTYPE


def unsupported_reason(layer) -> str | None:
    """Why this layer cannot route before the dispatch, or None if it can.

    Routing before the gather is only equivalent if the routing function is
    row-wise and needs nothing the dispatch has not produced yet.
    """
    if getattr(layer, "hash_indices_table", None) is not None:
        # DeepSeek-V4 hash routing keys off input_ids, which the dispatch
        # boundary does not carry.
        return "hash routing needs input_ids, which _maybe_dispatch lacks"
    if getattr(layer, "custom_routing_function", None) is not None:
        # Model-supplied; we cannot prove it is row-wise.
        return "custom_routing_function is not provably row-wise"
    return None


def route_and_pack(layer, hidden_states: torch.Tensor,
                   router_logits: torch.Tensor) -> torch.Tensor:
    """Decide this rank's rows and pack the decision for the dispatch.

    Refuses loudly rather than silently routing differently: an unsupported
    layer with the flag on is a configuration error, not a fallback, because a
    quiet fallback would leave the flag-on and flag-off paths routing
    differently with nothing to show for it.

    Returns the decision *unmasked*. Padded rows are zeroed downstream by
    whichever quant method already did that, on the gathered rows, exactly as
    without the flag -- see this module's docstring.
    """
    from vllm_torchtpu.layers.vllm import moe_routing

    reason = unsupported_reason(layer)
    if reason is not None:
        raise NotImplementedError(
            f"TPU_MOE_LOCAL_ROUTING=1 but this MoE layer cannot route before "
            f"the dispatch: {reason}. Unset the flag for this model.")
    topk_weights, topk_ids = moe_routing.route(layer, hidden_states,
                                               router_logits)
    return pack_routing(topk_ids, topk_weights)
