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

Upstream owns this idea already, on the path we do not take. vLLM's modular
MoE kernel splits the layer as ``[Router] -> [Quantize-Dispatch] -> ...``
(``fused_moe/modular_kernel.py:52``) and its *modular* prepare/finalize takes
the decision -- ``prepare(a1, topk_weights, topk_ids, ...)`` at ``:263-299``,
dispatched at ``fused_moe/prepare_finalize/naive_dp_ep.py:158``. The
*monolithic* variant our quant methods use takes ``router_logits`` and has no
decision slot at all (``:421-449``), and three separate checks refuse a
prepare/finalize object for a monolithic method
(``runner/moe_runner.py:903-910``, ``fused_moe_method_base.py:115``). So this
module implements upstream's own contract on the one path upstream left
un-modularised, rather than duplicating a seam we could have used.

TODO: the clean home for this is a monolithic prepare/finalize that dispatches
the decision, upstream. That deletes the type-pun below by construction -- the
decision would travel in its own argument instead of the logits slot -- and it
is additive, so it benefits every backend with a fused router, not just TPU.
Until then the pun is guarded by dtype (``is_routing_blob``).
(Checked against the pinned vllm 0.27.0, not the older reference checkout.)
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
    case we had assumed (384 KiB). ``T(1024)`` pads nothing only when
    ``T_local*2K`` is a multiple of 1024 -- true for the prefill buckets this
    targets (``T_local`` 4096 at dp4 and 2048 at dp8, ``K=10``), false for
    decode and for the small ``TPU_TOKEN_BUCKET_EXTRA`` buckets, where the
    last tile rounds up. The pad is at most 1023 elements (4 KiB/rank), so a
    misaligned bucket is still far under the logits it replaces; it is just
    not the quoted figure.

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

    ``reshape`` on the contiguous gathered blob is a view *in the ATen graph*,
    so it adds no copy there -- which is the property the codec's no-copy test
    measures, and a materialising reshape would hand back the bytes the 1-D
    form exists to save. It is not a claim about the device: under XLA these
    slices materialise regardless, and what the flag saves on the wire is the
    collective's layout (see :func:`pack_routing`), not an absent copy.
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
    row-wise, needs nothing the dispatch has not produced yet, and lands on a
    receiver that unpacks the decision. Every reason here is a *fallback*
    condition, not an error: the caller ships the raw logits instead and the
    receiving rank routes them exactly as upstream does.
    """
    from vllm_torchtpu.layers.vllm import moe_routing

    quant_method = getattr(layer, "quant_method", None)
    if not getattr(quant_method, "unpacks_routing_blob", False):
        # Only a method whose apply_monolithic goes through
        # `moe_routing.route` unpacks the blob. `is_monolithic` alone does not
        # imply that: a stacked install can put a monolithic method from
        # another MoE stack (tpu_inference.layers.vllm.quantization.*, or a
        # future upstream one made reachable on TPU) on the far side, and that
        # method would consume the packed int32 as logits. Declared per class,
        # cross-checked against the source in tests/layers/vllm/
        # test_moe_dispatch.py.
        return (f"{type(quant_method).__name__} does not declare "
                "unpacks_routing_blob, so the receiving side would read the "
                "packed decision as router logits")
    if moe_routing.simulation_is_active():
        # VLLM_MOE_ROUTING_SIMULATION_STRATEGY draws per-row noise shaped like
        # its input (moe_routing._uniform_random_routing), so simulating on
        # T_local rows and on the gathered T_global rows pick different --
        # equally balanced, equally meaningless -- experts. Falling back keeps
        # the profiling path bit-identical to upstream, where its whole value
        # is that a single-device profile represents the mesh.
        return ("routing simulation is enabled "
                "(VLLM_MOE_ROUTING_SIMULATION_STRATEGY), whose draw follows "
                "the row count")
    if (getattr(layer, "hash_indices_table", None) is not None
            and getattr(quant_method, "forwards_input_ids_to_route", False)):
        # DeepSeek-V4 hash routing keys off input_ids, which the dispatch
        # boundary does not carry. Carrying the table is not enough to reach
        # that branch: `select_experts` takes it only when `input_ids is not
        # None`, and only a method that declares
        # `forwards_input_ids_to_route` passes it down. The same checkpoint on
        # a method that does not (fp8) already routes from the gate today, and
        # local routing does not change that -- falling back for it would give
        # up the saving on a configuration that is fine.
        return ("hash routing needs input_ids, which _maybe_dispatch lacks "
                f"({type(quant_method).__name__} forwards it into route())")
    if getattr(layer, "custom_routing_function", None) is not None:
        # Model-supplied; we cannot prove it is row-wise.
        return "custom_routing_function is not provably row-wise"
    return None


_FALLBACKS_LOGGED: set[str] = set()


def _log_fallback_once(reason: str) -> None:
    """Say once, per distinct reason, that a layer kept the upstream path.

    Guarded on ``is_compiling`` because this runs inside the MoE forward:
    Dynamo rejects ``logging.Logger`` calls outright under ``fullgraph=True``
    ("logging.Logger method not supported for non-export cases"), which is the
    same objection torch_tpu's tracer raises. Under a compiled serving run the
    branch is therefore traced away and nothing is logged: eager callers (the
    tests, and anything running the MoE body outside a compiled region) are
    the only ones that see it.
    """
    if torch.compiler.is_compiling() or reason in _FALLBACKS_LOGGED:
        return
    _FALLBACKS_LOGGED.add(reason)
    logger.info(
        "MoE local routing: this layer keeps the upstream path (the dispatch "
        "carries raw router logits and the receiving rank routes them): %s",
        reason)


def route_and_pack(layer, hidden_states: torch.Tensor,
                   router_logits: torch.Tensor) -> torch.Tensor:
    """Decide this rank's rows and pack the decision for the dispatch.

    Falls back to ``router_logits`` unchanged for a layer whose routing this
    cannot reproduce before the gather (:func:`unsupported_reason`). The
    fallback is the upstream path, byte for byte: the receiving rank sees the
    logits it would have seen and routes them itself, so a model that cannot
    take this optimisation still runs, it just does not save anything.

    Returns the decision *unmasked*. Padded rows are zeroed downstream by
    whichever quant method already did that, on the gathered rows -- see this
    module's docstring.
    """
    from vllm_torchtpu.layers.vllm import moe_routing

    reason = unsupported_reason(layer)
    if reason is not None:
        _log_fallback_once(reason)
        return router_logits
    topk_weights, topk_ids = moe_routing.route(layer, hidden_states,
                                               router_logits)
    return pack_routing(topk_ids, topk_weights)
