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


def test_candidate_blocks_match_full_vocab_for_agentx_sampling():
    gen = torch.Generator().manual_seed(0)
    logits = (torch.randn(6, 128 * 400, generator=gen) * 3).bfloat16().float() / 0.6
    top_k = torch.full((6, 1), 20, dtype=torch.int32)
    top_p = torch.full((6, 1), 0.95)
    expected = sampling.apply_top_k_top_p(logits, top_k, top_p)
    result = sampling.apply_top_k_top_p_candidates(logits, top_k, top_p, 64)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_candidate_blocks_keep_ties_within_the_documented_bound():
    # 19 distinct large logits plus 13 ties at the 20th value, one per block:
    # 32 logits >= the 20th largest, exactly num_blocks.
    num_blocks, k = 32, 20
    logits = torch.full((1, 128 * 100), -5.0)
    for i in range(k - 1):
        logits[0, 128 * i] = 10.0 + i
    for j in range(num_blocks - k + 1):
        logits[0, 128 * (k - 1 + j) + 7] = 9.0
    top_k = torch.tensor([[k]], dtype=torch.int32)
    for p in (1.0, 0.9):
        top_p = torch.tensor([[p]])
        expected = sampling.apply_top_k_top_p(logits, top_k, top_p)
        result = sampling.apply_top_k_top_p_candidates(logits, top_k, top_p, num_blocks)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_candidate_blocks_leave_rows_without_top_k_unmasked():
    gen = torch.Generator().manual_seed(1)
    logits = torch.randn(3, 128 * 50, generator=gen)
    top_k = torch.tensor([[0], [-1], [5]], dtype=torch.int32)
    top_p = torch.tensor([[1.0], [1.0], [0.9]])
    expected = sampling.apply_top_k_top_p(logits, top_k, top_p)
    result = sampling.apply_top_k_top_p_candidates(logits, top_k, top_p, 16)
    torch.testing.assert_close(result[:2], logits[:2], rtol=0, atol=0)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("vocab", [248_320, 151_936 + 77, 4_000])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_chunked_topk_values_match_topk(vocab, dtype):
    width = 64
    gen = torch.Generator().manual_seed(vocab)
    logits = (torch.randn(6, vocab, generator=gen) * 3).to(dtype)
    logits[1] = torch.randint(-3, 3, (vocab,), generator=gen).to(dtype)  # ties
    logits[2, :] = float("-inf")
    logits[2, :10] = 1.0  # fewer finite logits than the width
    vals, idx = sampling._chunked_topk(logits, width)
    ref_vals, _ = torch.topk(logits, width, dim=-1)
    assert torch.equal(vals, ref_vals)
    assert int(idx.max()) < vocab
    finite = torch.isfinite(vals)
    assert torch.equal(torch.gather(logits, -1, idx)[finite], vals[finite])


@pytest.mark.parametrize("vocab", [31_040, 9_024, 30_001])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shard_topk_values_match_topk(device, vocab, dtype):
    # 31040 = the Qwen3.5 TP8 slice (one-hot fetch in bf16, gather in f32);
    # 30001 does not tile into 64-wide chunks (_chunked_topk).
    width = 64
    gen = torch.Generator().manual_seed(vocab)
    logits = (torch.randn(8, vocab, generator=gen) * 3).to(dtype)
    logits[1] = torch.randint(-3, 3, (vocab,), generator=gen).to(dtype)  # ties
    logits[2] = 0.0  # all tied
    logits[3, -100:] = float("-inf")  # a padded slice tail
    logits[4, 640:704] += 40.0  # the whole window inside one chunk
    vals, idx = sampling.shard_topk(logits.to(device), width)
    vals, idx = vals.cpu(), idx.cpu()
    ref_vals, _ = torch.topk(logits.float(), width, dim=-1)
    assert idx.dtype == torch.int32
    assert torch.equal(vals.float(), ref_vals)
    assert torch.equal(torch.gather(logits, -1, idx.long()), vals)
    assert all(len(set(row.tolist())) == width for row in idx)
