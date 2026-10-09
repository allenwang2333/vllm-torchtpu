# SPDX-License-Identifier: Apache-2.0
"""Bitwise equivalence of the TD7.21 top-k/top-p rewrites against the
pre-TD7.21 formulas (embedded below verbatim as references)."""

import pytest
import torch

from vllm_torchtpu.layers.adapter.sample import top_k_top_p as sampling


def _ref_int32_bsearch(batch_shape, predicate, device):
    current_bits = torch.zeros(batch_shape, dtype=torch.int32, device=device)
    zero = torch.zeros_like(current_bits)
    sign_bit = torch.full_like(current_bits, -(1 << 31))
    current_bits = current_bits | torch.where(predicate(current_bits), sign_bit, zero)
    for i in range(31):
        bit = 1 << (30 - i)
        bit_t = torch.full_like(current_bits, bit)
        midpoint = current_bits | bit
        current_bits = current_bits | torch.where(predicate(midpoint), zero, bit_t)
    return current_bits


def _ref_bit_pattern(x):
    non_sign_bits = torch.full_like(x, (1 << 31) - 1)
    zero = torch.zeros_like(x)
    return x ^ torch.where(x < 0, non_sign_bits, zero)


def _ref_float32_bsearch(batch_shape, predicate, device):
    def int32_predicate(x):
        exponent_bits = torch.full_like(x, (1 << 31) - (1 << 23))
        x = _ref_bit_pattern(x)
        is_finite = (x & exponent_bits) != exponent_bits
        return torch.where(is_finite, predicate(x.view(torch.float32)), x >= 0)

    result = _ref_int32_bsearch(batch_shape, int32_predicate, device)
    return _ref_bit_pattern(result).view(torch.float32)


def _ref_topk_mask(logits, k, replace_val):
    k = k.squeeze(-1)

    def predicate(threshold):
        neg_threshold = (-threshold).unsqueeze(-1)
        return torch.sum((logits > neg_threshold).to(torch.int32), dim=-1) >= k

    cutoff = -_ref_float32_bsearch(logits.shape[:-1], predicate, logits.device)
    replace_tensor = torch.full_like(logits, replace_val)
    return torch.where(logits >= cutoff.unsqueeze(-1), logits, replace_tensor)


def _ref_topp_mask(logits, p, replace_val):
    p = p.squeeze(-1)
    probs = torch.softmax(logits, dim=-1)
    zero_val = torch.zeros_like(probs)
    replace_tensor = torch.full_like(logits, replace_val)

    def predicate(threshold):
        threshold = threshold.unsqueeze(-1)
        mass = torch.sum(torch.where(probs >= threshold, probs, zero_val), dim=-1)
        return mass < p

    threshold = _ref_float32_bsearch(logits.shape[:-1], predicate, logits.device)
    return torch.where(probs >= threshold.unsqueeze(-1), logits, replace_tensor)


def _bits(t):
    return t.contiguous().view(torch.int32 if t.element_size() == 4 else torch.int16)


def _assert_bitwise(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(_bits(a), _bits(b))


_INT32_EDGES = [
    -(1 << 31),
    -(1 << 31) + 1,
    -(1 << 23),
    -1,
    0,
    1,
    (1 << 23),
    (1 << 31) - 2,
    (1 << 31) - 1,
]


def test_bit_pattern_matches_reference():
    g = torch.Generator().manual_seed(0)
    x = torch.randint(
        -(1 << 31), (1 << 31) - 1, (4096,), generator=g, dtype=torch.int64
    )
    x = torch.cat([x, torch.tensor(_INT32_EDGES)]).to(torch.int32)
    out = sampling._monotonic_int32_to_float32_bit_pattern(x)
    assert out.dtype == torch.int32
    assert torch.equal(out, _ref_bit_pattern(x))


@pytest.mark.parametrize("shape", [(9,), (3, 3), (1,), (0,)])
def test_int32_bsearch_matches_reference(shape):
    g = torch.Generator().manual_seed(1)
    n = int(torch.tensor(shape).prod())
    thr = torch.tensor(_INT32_EDGES * 2)[:n]
    thr = thr + torch.randint(-3, 4, (n,), generator=g) * (thr.abs() < (1 << 30))
    thr = thr.to(torch.int32).reshape(shape)

    def pred(x):
        return x > thr

    out = sampling._int32_bsearch(shape, pred, thr.device)
    assert out.dtype == torch.int32
    assert torch.equal(out, _ref_int32_bsearch(shape, pred, thr.device))
    assert torch.equal(out, thr)


def test_int32_bsearch_all_true_and_all_false():
    shape = (2,)
    for pred in (
        lambda x: torch.ones_like(x, dtype=torch.bool),
        lambda x: torch.zeros_like(x, dtype=torch.bool),
    ):
        out = sampling._int32_bsearch(shape, pred, torch.device("cpu"))
        assert torch.equal(out, _ref_int32_bsearch(shape, pred, torch.device("cpu")))


def test_float32_bsearch_matches_reference():
    thr = torch.tensor(
        [-3.4e38, -1.0, -0.0, 0.0, 1e-45, 2.5, 3.4e38, -1e-30], dtype=torch.float32
    )

    def pred(x):
        return x > thr

    out = sampling._float32_bsearch(thr.shape, pred, thr.device)
    _assert_bitwise(out, _ref_float32_bsearch(thr.shape, pred, thr.device))


def _random_logits(seed, shape, dtype):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(shape, generator=g) * 4
    # Ties.
    logits[..., 1] = logits[..., 0]
    logits[..., 5:9] = 0.5
    return logits.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(4, 64), (2, 3, 33)])
def test_topk_mask_matches_reference(dtype, shape):
    logits = _random_logits(2, shape, dtype)
    logits[0, ..., 3] = float("inf")
    logits[-1, ..., 4] = float("-inf")
    g = torch.Generator().manual_seed(3)
    k = torch.randint(0, shape[-1] + 2, (*shape[:-1], 1), generator=g)
    for replace in (-1e12, -999.0, float("-inf")):
        _assert_bitwise(
            sampling.topk_mask(logits, k, replace), _ref_topk_mask(logits, k, replace)
        )


def test_topk_mask_nan_matches_reference():
    logits = _random_logits(4, (3, 16), torch.float32)
    logits[1, 2] = float("nan")
    k = torch.tensor([[1], [3], [16]])
    _assert_bitwise(
        sampling.topk_mask(logits, k, -5.0), _ref_topk_mask(logits, k, -5.0)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(5, 64), (2, 2, 40)])
def test_topp_mask_matches_reference(dtype, shape):
    logits = _random_logits(5, shape, dtype)
    logits[0, ..., 7] = float("-inf")
    g = torch.Generator().manual_seed(6)
    p = torch.rand((*shape[:-1], 1), generator=g)
    p.view(-1)[0] = 0.0
    p.view(-1)[-1] = 1.0
    for replace in (-1e12, float("-inf")):
        _assert_bitwise(
            sampling.topp_mask(logits, p, replace), _ref_topp_mask(logits, p, replace)
        )


def test_apply_top_k_top_p_matches_reference():
    logits = _random_logits(7, (6, 50), torch.float32)
    top_k = torch.tensor([[0], [1], [5], [-1], [50], [3]])
    top_p = torch.tensor([[1.0], [0.3], [1.0], [0.9], [0.5], [0.0]])
    ref_k = torch.where(
        (top_k > 0).expand(-1, 50),
        _ref_topk_mask(logits, top_k, sampling.MASKED_LOGIT_VALUE),
        logits,
    )
    ref = torch.where(
        (top_p < 1.0).expand(-1, 50),
        _ref_topp_mask(ref_k, top_p, sampling.MASKED_LOGIT_VALUE),
        ref_k,
    )
    _assert_bitwise(sampling.apply_top_k_top_p(logits, top_k, top_p), ref)
