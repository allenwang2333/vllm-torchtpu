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

MASKED_LOGIT_VALUE = -1e12


def _int32_bsearch(
    batch_shape: Sequence[int],
    predicate: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
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
    result = _int32_bsearch(batch_shape, int32_predicate, device)
    return _monotonic_int32_to_float32(result)


def topk_mask(
    logits: torch.Tensor, k: torch.Tensor, replace_val: float = MASKED_LOGIT_VALUE
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

    cutoff = -_float32_bsearch(logits.shape[:-1], predicate, logits.device)
    cutoff = cutoff.unsqueeze(-1)
    replace_tensor = torch.full_like(logits, replace_val)
    return torch.where(logits >= cutoff, logits, replace_tensor)


def topp_mask(
    logits: torch.Tensor, p: torch.Tensor, replace_val: float = MASKED_LOGIT_VALUE
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

    Returns:
      masked version of logits. [batch..., vocab_size]
    """
    p = p.squeeze(-1)
    probs = torch.softmax(logits, dim=-1)
    zero_val = torch.zeros_like(probs)
    replace_tensor = torch.full_like(logits, replace_val)

    def predicate(threshold: torch.Tensor) -> torch.Tensor:
        threshold = threshold.unsqueeze(-1)
        probability_mass = torch.sum(
            torch.where(probs >= threshold, probs, zero_val),
            dim=-1,
        )
        return probability_mass < p

    threshold = _float32_bsearch(logits.shape[:-1], predicate, logits.device)
    threshold = threshold.unsqueeze(-1)
    return torch.where(probs >= threshold, logits, replace_tensor)


def _apply_top_k_top_p_float_search(
    logits: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor
) -> torch.Tensor:
    """`apply_top_k_top_p` via 32-step float32 threshold searches."""
    should_apply_topk = (top_k > 0).expand(-1, logits.shape[-1])
    topk_masked = topk_mask(logits, top_k, MASKED_LOGIT_VALUE)
    masked_logits = torch.where(should_apply_topk, topk_masked, logits)

    should_apply_topp = (top_p < 1.0).expand(-1, masked_logits.shape[-1])
    topp_masked = topp_mask(masked_logits, top_p, MASKED_LOGIT_VALUE)
    return torch.where(should_apply_topp, topp_masked, masked_logits)


# Fast path: 16-step searches over bf16 order keys.
#
# `topk_mask` and `topp_mask` each binary-search a float32 threshold, i.e. 32
# full-vocab reductions apiece. When the logits come from a bf16 tensor, their
# order is fully described by a 16-bit key, so 16 steps suffice to find the
# boundary element; the float threshold is then read back from that element.
# The result is the same masked tensor the float32 searches return.
#
# Why it is exact: let `values` be a non-decreasing function of `order_source`
# (a bf16 tensor) within each row, e.g. `order_source / temperature` in any
# dtype. Every set the float32 searches keep is an upper set of `values`
# ({values >= cutoff}), and every such set is also an upper set of the keys.
# The search finds the smallest key upper set that satisfies the criterion
# (count >= k, or mass >= p), which is contained in the reference set; closing
# it under value ties ({values >= its smallest value}) gives back exactly the
# reference set, because the reference set is the smallest value upper set
# that satisfies the criterion.


def bf16_order_key(x: torch.Tensor) -> torch.Tensor:
    """bf16 -> int32 key in [-32768, 32767], strictly monotonic in float order.

    Same "xor trick" as `_monotonic_int32_to_float32_bit_pattern`, on 16 bits:
    negative floats have their magnitude bits flipped so larger magnitudes map
    to smaller keys.
    """
    bits = x.view(torch.int16).to(torch.int32)
    return bits ^ ((bits >> 15) & 0x7FFF)


def _int16_key_bsearch(
    batch_shape: Sequence[int],
    predicate: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Largest key t in [-32768, 32767] with predicate(t) True.

    `predicate` must be monotonic decreasing in t (True up to a threshold,
    False above it). Returns -32768 when it is False everywhere. [batch..., 1]
    """
    lo = torch.full((*batch_shape, 1), -(1 << 15), dtype=torch.int32, device=device)
    for bit in range(15, -1, -1):
        mid = lo + (1 << bit)
        lo = torch.where(predicate(mid), mid, lo)
    return lo


def _min_where(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Row-wise min of `values` over `mask`; +inf for empty rows. [batch..., 1]"""
    inf = torch.full_like(values, float("inf"))
    return torch.where(mask, values, inf).amin(dim=-1, keepdim=True)


def topk_mask_by_key(
    logits: torch.Tensor,
    key: torch.Tensor,
    k: torch.Tensor,
    replace_val: float = MASKED_LOGIT_VALUE,
) -> torch.Tensor:
    """Same result as `topk_mask(logits, k)`; `key` orders `logits` (see above)."""

    def predicate(mid: torch.Tensor) -> torch.Tensor:
        return (key >= mid).sum(dim=-1, keepdim=True, dtype=torch.int32) >= k

    boundary = _int16_key_bsearch(logits.shape[:-1], predicate, logits.device)
    cutoff = _min_where(logits, key >= boundary)
    # topk_mask only searches finite thresholds, so its cutoff never drops
    # below -max and it masks -inf logits even when fewer than k are finite.
    cutoff = cutoff.clamp(min=-torch.finfo(logits.dtype).max)
    return torch.where(logits >= cutoff, logits, torch.full_like(logits, replace_val))


def topp_mask_by_key(
    logits: torch.Tensor,
    key: torch.Tensor,
    p: torch.Tensor,
    replace_val: float = MASKED_LOGIT_VALUE,
) -> torch.Tensor:
    """Same result as `topp_mask(logits, p)`; `key` orders `logits` (see above)."""
    probs = torch.softmax(logits, dim=-1)
    zero = torch.zeros_like(probs)

    def predicate(mid: torch.Tensor) -> torch.Tensor:
        mass = torch.where(key >= mid, probs, zero).sum(dim=-1, keepdim=True)
        return mass >= p

    boundary = _int16_key_bsearch(logits.shape[:-1], predicate, logits.device)
    threshold = _min_where(probs, key >= boundary)
    return torch.where(probs >= threshold, logits, torch.full_like(logits, replace_val))


def apply_top_k_top_p(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    order_source: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masks `logits` to each row's top-k, then to its top-p nucleus.

    Args:
      logits: Values to mask (e.g. temperature-scaled). [batch, vocab_size]
      top_k: [batch, 1] int. Rows with top_k <= 0 skip top-k. Ties with the
        k-th largest value are kept.
      top_p: [batch, 1] float. Rows with top_p >= 1 skip top-p.
      order_source: Optional tensor that `logits` is a row-wise non-decreasing
        function of, e.g. the unscaled logits. Defaults to `logits`. When it is
        bf16, the thresholds are found with 16-step searches over bf16 order
        keys instead of 32-step float32 searches (about half the vocab-wide
        reductions); the result is identical either way. [batch, vocab_size]

    Returns:
      Masked logits, MASKED_LOGIT_VALUE where dropped. [batch, vocab_size]
    """
    if order_source is None:
        order_source = logits
    if order_source.dtype != torch.bfloat16:
        return _apply_top_k_top_p_float_search(logits, top_k, top_p)
    key = bf16_order_key(order_source)

    topk_masked = topk_mask_by_key(logits, key, top_k)
    masked_logits = torch.where(top_k > 0, topk_masked, logits)

    # Masked entries hold the row's smallest value, so masked_logits is still
    # non-decreasing in `key`.
    topp_masked = topp_mask_by_key(masked_logits, key, top_p)
    return torch.where(top_p < 1.0, topp_masked, masked_logits)
