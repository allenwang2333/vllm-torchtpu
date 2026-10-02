# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter.sample import top_k_top_p as sampling


def test_int32_search_boundaries():
    thresholds = torch.tensor(
        [[-2147483648, -37, -1], [0, 128, 2147483647]], dtype=torch.int32
    )
    result = sampling._int32_bsearch(
        thresholds.shape, lambda x: x > thresholds, thresholds.device
    )
    torch.testing.assert_close(result, thresholds)


def test_float32_search_boundaries():
    maximum = torch.finfo(torch.float32).max
    thresholds = torch.tensor([-maximum, -10.5, -1.0, 0.0, 3.25, maximum])
    result = sampling._float32_bsearch(
        thresholds.shape, lambda x: x > thresholds, thresholds.device
    )
    torch.testing.assert_close(
        result, thresholds, rtol=0, atol=torch.finfo(torch.float32).tiny
    )


@pytest.mark.parametrize("shape", [(5,), (2, 5), (2, 3, 5)])
@pytest.mark.parametrize("k", [1, 2, 5])
def test_topk_matches_sorted_cutoff(shape, k):
    logits = torch.tensor([-4.0, 3.0, 1.0, 3.0, -2.0]).expand(shape)
    cutoff = logits.sort(dim=-1).values[..., -k, None]
    expected = torch.where(logits >= cutoff, logits, -999.0)
    result = sampling.topk_mask(logits, torch.full((*shape[:-1], 1), k), -999.0)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "p, keep",
    [
        (0.1, [True, False, False, False, False]),
        (0.65, [True, True, True, False, False]),
        (1.0, [True, True, True, True, True]),
    ],
)
def test_topp_includes_cutoff_ties(p, keep):
    logits = torch.log(torch.tensor([0.5, 0.2, 0.2, 0.08, 0.02]))
    expected = torch.where(torch.tensor(keep), logits, -999.0)
    result = sampling.topp_mask(logits, torch.tensor([p]), -999.0)
    torch.testing.assert_close(result, expected)


def test_apply_top_k_top_p_per_request():
    probabilities = np.array(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.40, 0.30, 0.20, 0.10],
            [0.40, 0.30, 0.20, 0.10],
            [0.40, 0.30, 0.20, 0.10],
        ],
        dtype=np.float32,
    )
    logits = torch.from_numpy(np.log(probabilities))
    top_k = torch.tensor([[0], [2], [-1], [3]])
    top_p = torch.tensor([[1.0], [1.0], [0.65], [0.5]])
    # In row 3, top-k renormalizes .4/.3/.2; top-p needs two tokens to reach .5.
    keep = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [True, True, False, False],
            [True, True, False, False],
        ]
    )
    expected = torch.where(keep, logits, sampling.MASKED_LOGIT_VALUE)
    original = logits.clone()
    result = sampling.apply_top_k_top_p(logits, top_k, top_p)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(logits, original)


# The bounded (torch.topk) path must reproduce the binary-search path exactly:
# same masked positions and same values, at the served vocab size.
_VOCAB = 248320
_K_BOUND = 64


def _assert_bounded_matches(logits, top_k, top_p, device, k_bound=_K_BOUND):
    # The search path's top-p sums bf16 probabilities with torch.sum. On CPU
    # that accumulates in f32; on TPU the accumulation precision depends on
    # the fusion and can under-count the mass (all-equal row: 248,320 x 4e-6
    # sums below 0.95). The bounded path accumulates in f32 on every device,
    # so bf16 on an accelerator is checked against the CPU search path.
    reference_device = device
    if logits.dtype == torch.bfloat16 and torch.device(device).type != "cpu":
        reference_device = torch.device("cpu")
    expected = sampling.apply_top_k_top_p(
        logits.to(reference_device),
        top_k.to(reference_device),
        top_p.to(reference_device),
    )
    logits, top_k, top_p = logits.to(device), top_k.to(device), top_p.to(device)
    result = sampling.apply_top_k_top_p(logits, top_k, top_p, k_bound)
    assert result.dtype == expected.dtype
    kept_expected = expected != sampling.MASKED_LOGIT_VALUE
    kept_result = result != sampling.MASKED_LOGIT_VALUE
    assert torch.equal(kept_result.cpu(), kept_expected.cpu()), (
        kept_result.sum(-1).tolist(),
        kept_expected.sum(-1).tolist(),
    )
    assert torch.equal(result.cpu(), expected.cpu())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize("k", [1, 20, 64])
@pytest.mark.parametrize("p", [0.5, 0.95, 1.0])
@pytest.mark.parametrize("scale", [1.0, 4.0])
def test_bounded_path_matches_search(dtype, rows, k, p, scale, device):
    gen = torch.Generator().manual_seed(rows * 1009 + k * 31 + int(scale))
    # Temperature-scaled in the logits dtype, as sample_from_logits does.
    logits = (torch.randn(rows, _VOCAB, generator=gen) * scale).to(dtype)
    logits = logits / torch.tensor(0.6, dtype=dtype)
    top_k = torch.full((rows, 1), k, dtype=torch.int32)
    top_p = torch.full((rows, 1), p, dtype=torch.float32)
    _assert_bounded_matches(logits, top_k, top_p, device)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bounded_path_mixed_rows(dtype, device):
    gen = torch.Generator().manual_seed(7)
    logits = (torch.randn(8, _VOCAB, generator=gen) * 3.0).to(dtype)
    # Mixed per-row params, plus rows that apply neither mask (top_k == 0 and
    # top_p == 1, i.e. padding rows) and a top-p-only-off row.
    top_k = torch.tensor([[1], [20], [64], [5], [0], [33], [0], [64]])
    top_p = torch.tensor([[0.5], [0.95], [1.0], [0.1], [1.0], [0.99], [1.0], [0.3]])
    assert sampling.topk_fast_path_bound(top_k, top_p, _K_BOUND) == _K_BOUND
    _assert_bounded_matches(logits, top_k.to(torch.int32), top_p, device)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("k", [1, 3, 20, 64])
@pytest.mark.parametrize("p", [0.3, 0.95, 1.0])
def test_bounded_path_ties(dtype, k, p, device):
    # Heavily quantized logits: many exact ties at and around the cutoff,
    # including ties that continue past the k_bound window.
    gen = torch.Generator().manual_seed(k)
    logits = torch.randint(-20, 4, (4, _VOCAB), generator=gen).to(dtype) / 2
    # Row 1: a block of 100 tied maxima (top-k ties all past the window).
    logits[1, 1000:1100] = 10.0
    # Row 2: two levels, 10 tied above 200 tied (cutoff level overflows).
    logits[2, 5:15] = 12.0
    logits[2, 50000:50200] = 11.0
    # Row 3: all equal.
    logits[3] = 0.0
    top_k = torch.full((4, 1), k, dtype=torch.int32)
    top_p = torch.full((4, 1), p, dtype=torch.float32)
    _assert_bounded_matches(logits, top_k, top_p, device)


def test_bounded_path_neg_inf_and_edge_p(device):
    gen = torch.Generator().manual_seed(3)
    logits = torch.randn(4, _VOCAB, generator=gen)
    # Row 0: fewer finite logits than k (grammar-style -inf mask).
    logits[0, 10:] = float("-inf")
    top_k = torch.tensor([[20], [20], [20], [64]], dtype=torch.int32)
    # Row 2: p == 0 keeps nothing in the search path; row 3: p ~ 1.
    top_p = torch.tensor([[0.95], [0.95], [0.0], [0.999999]])
    _assert_bounded_matches(logits, top_k, top_p, device)


def test_bounded_path_greedy_rejection_scaling(device):
    # Rejection sampler: f32 logits / (temperature + 1e-9); greedy rows have
    # temperature 0, giving huge but finite logits.
    gen = torch.Generator().manual_seed(11)
    logits = torch.randn(4, _VOCAB, generator=gen).to(torch.bfloat16)
    temps = torch.tensor([[0.0], [0.6], [0.0], [1.0]])
    scaled = logits.to(torch.float32) / (temps + 1e-9)
    top_k = torch.full((4, 1), 20, dtype=torch.int32)
    top_p = torch.full((4, 1), 0.95)
    _assert_bounded_matches(scaled, top_k, top_p, device)


def test_topk_fast_path_bound():
    k = torch.tensor([[20], [64], [0]])
    p = torch.tensor([[0.95], [1.0], [1.0]])
    assert sampling.topk_fast_path_bound(k, p, 64) == 64
    assert sampling.topk_fast_path_bound(k, p, 0) == 0
    # top_k above the bound, top_k == 0 with live top-p, negative top_k.
    assert sampling.topk_fast_path_bound(k + 45, p, 64) == 0
    assert (
        sampling.topk_fast_path_bound(k, torch.tensor([[1.0], [1.0], [0.9]]), 64) == 0
    )
    assert (
        sampling.topk_fast_path_bound(torch.tensor([[-1]]), torch.tensor([[1.0]]), 64)
        == 0
    )
