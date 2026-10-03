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


# --- TPU_SAMPLER_VOCAB_PARALLEL: radix bisection + static mask skips ---


@pytest.mark.parametrize("radix_bits", [2, 3, 4, 8])
def test_int32_search_radix_matches_bitwise(radix_bits):
    thresholds = torch.tensor(
        [[-2147483648, -37, -1, 5], [0, 128, 2147483647, -(1 << 30)]],
        dtype=torch.int32,
    )
    result = sampling._int32_bsearch(
        thresholds.shape, lambda x: x > thresholds, thresholds.device, radix_bits
    )
    torch.testing.assert_close(result, thresholds)


@pytest.mark.parametrize("radix_bits", [2, 4])
def test_float32_search_radix_boundaries(radix_bits):
    maximum = torch.finfo(torch.float32).max
    thresholds = torch.tensor([-maximum, -10.5, -1.0, 0.0, 3.25, maximum])
    result = sampling._float32_bsearch(
        thresholds.shape, lambda x: x > thresholds, thresholds.device, radix_bits
    )
    torch.testing.assert_close(
        result, thresholds, rtol=0, atol=torch.finfo(torch.float32).tiny
    )


def _random_rows(rows=6, vocab=4099, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, vocab, generator=g) * 3.0
    logits[1, :50] = logits[1, 50]  # ties across the cutoff
    top_k = torch.tensor([[0], [7], [-1], [100], [1], [vocab - 1]])[:rows]
    top_p = torch.tensor([[1.0], [0.95], [0.5], [0.9], [1.0], [0.3]])[:rows]
    return logits, top_k, top_p


@pytest.mark.parametrize("radix_bits", [2, 3, 4])
def test_apply_top_k_top_p_radix_is_bit_identical(radix_bits):
    logits, top_k, top_p = _random_rows()
    expected = sampling.apply_top_k_top_p(logits, top_k, top_p)
    result = sampling.apply_top_k_top_p(
        logits, top_k, top_p, radix_bits=radix_bits
    )
    assert torch.equal(result, expected)


@pytest.mark.parametrize("radix_bits", [1, 2])
def test_apply_top_k_top_p_skips_are_identity(radix_bits):
    logits, _, _ = _random_rows()
    rows = logits.shape[0]
    no_k = torch.zeros((rows, 1), dtype=torch.int64)
    no_p = torch.ones((rows, 1))
    some_k = torch.tensor([[0], [7], [0], [100], [1], [0]])
    some_p = torch.tensor([[1.0], [0.95], [1.0], [0.9], [1.0], [0.3]])
    for k, p, need_k, need_p in [
        (no_k, some_p, False, True),
        (some_k, no_p, True, False),
        (no_k, no_p, False, False),
    ]:
        expected = sampling.apply_top_k_top_p(logits, k, p)
        result = sampling.apply_top_k_top_p(
            logits, k, p, need_k, need_p, radix_bits
        )
        assert torch.equal(result, expected)
