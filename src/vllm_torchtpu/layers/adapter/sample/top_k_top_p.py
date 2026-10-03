# Copyright 2025 Google LLC
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
"""Binary search over float32 bits.

Includes fast algorithms for top-k masking and top-p masking on probability
distributions.
"""

from collections.abc import Callable, Sequence

import torch

# Registers the fast top-p op at import (not lazily inside a traced graph).
from vllm_torchtpu.layers.adapter.sample import fast_top_p_op

MASKED_LOGIT_VALUE = -1e12


def _int32_bsearch(
    batch_shape: Sequence[int],
    predicate: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    radix_bits: int = 1,
) -> torch.Tensor:
    """Batched binary search over int32 values.

    For each element of the batch, search for the largest int32 (closest to
    positive infinity) for which the predicate is False. If the predicate is
    always True, returns the minimum int32 value.

    Args:
      batch_shape: Shape of the search that we're batching over.
      predicate: the query we're searching for. For every batch element, this is
        required to be a monotonic function from int32 to bool. In other words,
        the predicate must return False for all numbers <= some threshold and
        then return True for all numbers > that threshold. The threshold may be
        different for different elements of the batch.
      device: device to allocate the search state on.
      radix_bits: bits resolved per predicate round. Above 1, each round
        evaluates the predicate on all 2**radix_bits - 1 candidates of the next
        digit at once (a leading candidate axis on the threshold), and the
        digit is the number of candidates that return False. Monotonicity makes
        that the same digit the 1-bit search picks, in 1/radix_bits the rounds
        (each round is one pass over the predicate's input).

    Returns:
      For each element of the batch, the largest int32 for which the predicate
      returns False. Shape: batch_shape.
    """
    current_bits = torch.zeros(batch_shape, dtype=torch.int32, device=device)
    zero = torch.zeros_like(current_bits)
    sign_bit = torch.full_like(current_bits, -(1 << 31))

    # bit 31 is special, because it compares in the opposite order of all other
    # bits.
    current_bits = current_bits | torch.where(predicate(current_bits), sign_bit, zero)

    if radix_bits > 1:
        # Bits 30..0 in digits of up to radix_bits bits, high digit first.
        # Within a fixed sign bit, int32 order is the order of the low 31 bits,
        # so candidate j of a digit at `shift` is current | (j << shift).
        hi = 31
        while hi > 0:
            width = min(radix_bits, hi)
            shift = hi - width
            num_cand = (1 << width) - 1
            cand = (
                torch.arange(1, num_cand + 1, dtype=torch.int32, device=device) << shift
            ).view((num_cand,) + (1,) * len(batch_shape))
            candidates = current_bits.unsqueeze(0) | cand
            # False for candidates 1..digit, True above (monotonic predicate).
            digit = torch.sum(
                (~predicate(candidates)).to(torch.int32), dim=0, dtype=torch.int32
            )
            current_bits = current_bits | (digit << shift)
            hi = shift
        return current_bits

    for i in range(31):
        bit = 1 << (30 - i)
        bit_t = torch.full_like(current_bits, bit)
        midpoint = current_bits | bit
        current_bits = current_bits | torch.where(predicate(midpoint), zero, bit_t)
    return current_bits


def _monotonic_int32_to_float32_bit_pattern(x: torch.Tensor) -> torch.Tensor:
    """Converts an int32 to a float32 bit pattern with consistent ordering.

    This function is the unique function that is monotonic with respect to the
    floating point total order, see
    https://en.wikipedia.org/wiki/IEEE_754#Total-ordering_predicate. Note that
    this function returns an int32, not a float32. For the function that returns
    float32, see `_monotonic_int32_to_float32`.

    Args:
      x: int32.

    Returns:
      int32 bit pattern of a float32 number.
    """
    non_sign_bits = torch.full_like(x, (1 << 31) - 1)
    zero = torch.zeros_like(x)
    # See
    # https://stackoverflow.com/questions/20097380/iee-754-total-order-in-standard-c11
    # for the relationship between int32 order and f32 total order, including
    # the "xor trick".

    # Flip the sort order for numbers where the sign bit is set. On int32,
    # the bit pattern with sign bit set and all other bits clear is the most
    # negative bit pattern (it's int32::MIN), whereas on float32 it's the least
    # negative bit pattern (it's -0.0). Flipping all the non-sign bits makes the
    # int32 sort order consistent with the float32 sort order.
    return x ^ torch.where(x < 0, non_sign_bits, zero)


def _monotonic_int32_to_float32(x: torch.Tensor) -> torch.Tensor:
    """Converts an int32 to a float32 with consistent ordering.

    This function is the unique function that is monotonic with respect to the
    floating point total order, see
    https://en.wikipedia.org/wiki/IEEE_754#Total-ordering_predicate.

    Args:
      x: int32.

    Returns:
      float32 number with consistent ordering.
    """
    return _monotonic_int32_to_float32_bit_pattern(x).view(torch.float32)


def _float32_bsearch(
    batch_shape: Sequence[int],
    predicate: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    radix_bits: int = 1,
) -> torch.Tensor:
    """Binary search on finite float32 numbers.

    For each element of the batch, this function searches for the largest finite
    non-NaN float32 for which the predicate is False.

    Args:
      batch_shape: Shape of the search that we're batching over.
      predicate: the query we're searching for. This is required to be monotonic
        with respect to the floating point order, i.e. it must be False for all
        numbers <= a threshold, and then True for all numbers > the threshold.
        The threshold may be different for different elements of the batch.
      device: device to allocate the search state on.

    Returns:
      For each element of the batch, the largest float32 for which the predicate
      returns False. Shape: f32[batch_shape].
    """

    def int32_predicate(x: torch.Tensor) -> torch.Tensor:
        exponent_bits = torch.full_like(x, (1 << 31) - (1 << 23))
        x = _monotonic_int32_to_float32_bit_pattern(x)
        is_finite = (x & exponent_bits) != exponent_bits

        # Non-finite numbers (infinity and NaN) are at the very extremes of the
        # int32 range, i.e. they include int32::MAX and int32::MIN, plus the
        # numbers adjacent to them. For the nonfinite numbers touching
        # int32::MIN, we arrange for them to return False from the predicate,
        # and for the nonfinite numbers touching int32::MAX, we arrange for them
        # to return True from the predicate. x>=0 is an easy way to achieve
        # that.
        predicate_on_nonfinite = x >= 0
        return torch.where(
            is_finite, predicate(x.view(torch.float32)), predicate_on_nonfinite
        )

    # We search over bit patterns, which requires bit shifting and ordering of
    # bit patterns. This is natively supported on int32 but not on float32.
    result = _int32_bsearch(batch_shape, int32_predicate, device, radix_bits)
    return _monotonic_int32_to_float32(result)


def topk_mask(
    logits: torch.Tensor,
    k: torch.Tensor,
    replace_val: float = MASKED_LOGIT_VALUE,
    radix_bits: int = 1,
) -> torch.Tensor:
    """Sets everything to replace_val, except the top k values per batch element.

    Sharding considerations: this function does 32 reductions over the
    vocab_size axis of the input array. To avoid excessive latency from these
    reductions, you should ensure that the vocab_size axis is unsharded on input
    to this function. Prefer to shard the batch axes instead.

    Scratchpad memory considerations: this function is most efficient if the
    entire input array can fit in a fast memory tier. To help ensure this, you
    may wish to split the batch axes into microbatches and the microbatches in a
    sequential loop.

    Args:
      logits: Values before masking. [batch..., vocab_size]
      k: Number of values to keep per batch element. In presence of ties, more
        than k values might be returned. [batch..., 1]
      replace_val: For the masked values of logits, what to overwrite them with.

    Returns:
      masked version of logits. [batch..., vocab_size]
    """
    k = k.squeeze(-1)

    def predicate(threshold: torch.Tensor) -> torch.Tensor:
        # Search negated logits so the predicate is monotonic increasing.
        neg_threshold = (-threshold).unsqueeze(-1)
        count_gt = torch.sum((logits > neg_threshold).to(torch.int32), dim=-1)
        return count_gt >= k

    cutoff = -_float32_bsearch(logits.shape[:-1], predicate, logits.device, radix_bits)
    cutoff = cutoff.unsqueeze(-1)
    replace_tensor = torch.full_like(logits, replace_val)
    return torch.where(logits >= cutoff, logits, replace_tensor)


def _parse_fast_top_p(spec: str) -> tuple[int, str, str, int]:
    """ "K[,method[,mass_accum[,window]]]" -> (K, method, mass_accum, window)."""
    parts = [x.strip() for x in spec.split(",")] + [""] * 3
    return (
        int(parts[0]),
        parts[1] or FAST_TOP_P_METHOD,
        parts[2] or FAST_TOP_P_MASS_ACCUM,
        int(parts[3] or FAST_TOP_P_WINDOW),
    )


# Defaults of a fast_top_p spec (kernels/top_p.py). "native": torch_tpu sums a
# bf16 tensor like a bf16 stablehlo reduce -- bit for bit on tpu7x
# (test_fast_top_p_matches_bisection); "f32" does not match it.
FAST_TOP_P_K = 1024
FAST_TOP_P_METHOD = "topk"
FAST_TOP_P_MASS_ACCUM = "native"
FAST_TOP_P_WINDOW = 2


def fast_topp_threshold(
    probs: torch.Tensor, p: torch.Tensor, spec: str
) -> torch.Tensor:
    """The bisection's top-p threshold [B, 1] f32, via kernels/top_p.py on TPU.

    Off TPU it is the bisection itself (the JAX op only runs on TPU)."""
    if probs.device.type != "tpu":
        return _topp_bisect_threshold(probs, p.squeeze(-1), 1).unsqueeze(-1)
    return fast_top_p_op.top_p_threshold(probs, p.float(), *_parse_fast_top_p(spec))


def _topp_bisect_threshold(probs: torch.Tensor, p: torch.Tensor, radix_bits: int):
    zero_val = torch.zeros_like(probs)

    def predicate(threshold: torch.Tensor) -> torch.Tensor:
        threshold = threshold.unsqueeze(-1)
        probability_mass = torch.sum(
            torch.where(probs >= threshold, probs, zero_val),
            dim=-1,
        )
        return probability_mass < p

    return _float32_bsearch(probs.shape[:-1], predicate, probs.device, radix_bits)


def topp_mask(
    logits: torch.Tensor,
    p: torch.Tensor,
    replace_val: float = MASKED_LOGIT_VALUE,
    radix_bits: int = 1,
    fast_top_p: str = "",
) -> torch.Tensor:
    """Applies top-p masking to logits.

    Masks logits down to the smallest set of choices, such that the total
    probability mass is >= p. Values in this set are left as they are. All other
    values are set with `replace_val`.

    Sharding considerations: this function does 33 reductions over the
    vocab_size axis of the input array. To avoid excessive latency from these
    reductions, you should ensure that the vocab_size axis is unsharded on input
    to this function. Prefer to shard the batch axes instead.

    Scratchpad memory considerations: this function is most efficient if the
    entire input array can fit in a fast memory tier. To help ensure this, you
    may wish to split the batch axes into microbatches and the microbatches in a
    sequential loop.

    Args:
      logits: Logits before masking. [batch..., vocab_size]
      p: Minimum probability mass requested. [batch..., 1]
      replace_val: For the masked values of logits, what to overwrite them with.
      fast_top_p: "" for the bisection, else a "K[,method[,mass_accum]]" spec:
        the same threshold from a K-candidate prefilter proven by one vocab
        pass, falling back to the bisection when the proof fails
        (kernels/top_p.py). [batch, vocab_size] inputs only.

    Returns:
      masked version of logits. [batch..., vocab_size]
    """
    probs = torch.softmax(logits, dim=-1)
    replace_tensor = torch.full_like(logits, replace_val)
    if fast_top_p:
        threshold = fast_topp_threshold(probs, p, fast_top_p)
    else:
        threshold = _topp_bisect_threshold(probs, p.squeeze(-1), radix_bits).unsqueeze(
            -1
        )
    return torch.where(probs >= threshold, logits, replace_tensor)


def apply_top_k_top_p(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    need_top_k: bool = True,
    need_top_p: bool = True,
    radix_bits: int = 1,
    fast_top_p: str = "",
) -> torch.Tensor:
    """Top-k then top-p masking, per row.

    need_top_k / need_top_p are static (trace-time) promises from the caller
    that some row has top_k > 0 / top_p < 1. When False, every row's
    torch.where below would select the unmasked input, so the mask is skipped
    whole: bit-identical output, minus 32-33 full-vocab reduction passes.
    fast_top_p (static, see topp_mask) computes the same top-p threshold in
    one proven vocab pass instead of 33.
    """
    masked_logits = logits
    if need_top_k:
        should_apply_topk = (top_k > 0).expand(-1, logits.shape[-1])
        topk_masked = topk_mask(logits, top_k, MASKED_LOGIT_VALUE, radix_bits)
        masked_logits = torch.where(should_apply_topk, topk_masked, logits)

    if not need_top_p:
        return masked_logits
    should_apply_topp = (top_p < 1.0).expand(-1, masked_logits.shape[-1])
    topp_masked = topp_mask(
        masked_logits, top_p, MASKED_LOGIT_VALUE, radix_bits, fast_top_p
    )
    return torch.where(should_apply_topp, topp_masked, masked_logits)
