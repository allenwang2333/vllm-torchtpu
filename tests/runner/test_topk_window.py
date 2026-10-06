# SPDX-License-Identifier: Apache-2.0
"""TPU_SAMPLER_VOCAB_PARALLEL_TOPK: verify windows built from the
vocabulary-parallel LM head against the full-logits window of the sparse
spec sampler, and when a chunk may take them."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams
from vllm.model_executor.layers import logits_processor as lp_module
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.sampling_params import StructuredOutputsParams

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    RejectionSampler,
    sparse_window_sample,
)
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    _chunked_topk,
    pack_value_index,
    unpack_value_index,
)
from vllm_torchtpu.runner import topk_window
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

WIDTH = 64


class _HeadShard(torch.nn.Module):
    """One rank's slice of a vocab-parallel LM head (rows [start, start+n))."""

    def __init__(self, weight, start, tp_size, num_pad):
        super().__init__()
        self.weight = weight
        self.tp_size = tp_size
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=start, num_org_vocab_padding=num_pad
        )
        self.quant_method = SimpleNamespace(
            apply=lambda layer, x, bias=None: x @ layer.weight.T
        )


class _Owner(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.lm_head = head
        with patch.object(
            lp_module,
            "get_current_vllm_config",
            return_value=SimpleNamespace(model_config=None),
        ):
            self.logits_processor = LogitsProcessor(head.weight.shape[0])


def _shards(weight, tp_size):
    """Split [vocab, H] into tp_size shards, padding the tail with rows that
    would top every window if the padding mask were missing."""
    vocab, hidden = weight.shape
    per_rank = math.ceil(vocab / tp_size)
    pad_rows = torch.full((per_rank * tp_size - vocab, hidden), 1e4, dtype=weight.dtype)
    padded = torch.cat([weight, pad_rows])
    shards = []
    for rank in range(tp_size):
        start = rank * per_rank
        num_pad = max(0, start + per_rank - vocab) if start < vocab else per_rank
        shards.append(
            _HeadShard(padded[start : start + per_rank], start, tp_size, num_pad)
        )
    return shards


def _vp_window(weight, hidden_states, tp_size, monkeypatch, width=WIDTH, fn=None):
    """window_from_hidden_states on every rank, the all-gather played back
    from the pairs each rank contributed."""
    fn = fn or topk_window.window_from_hidden_states
    shards = _shards(weight, tp_size)
    pairs = []

    def record(pair, dim):
        pairs.append(pair)
        return pair.repeat(1, tp_size)

    monkeypatch.setattr(topk_window, "tensor_model_parallel_all_gather", record)
    for shard in shards:
        fn(_Owner(shard), hidden_states, width)
    monkeypatch.setattr(
        topk_window,
        "tensor_model_parallel_all_gather",
        lambda pair, dim: torch.cat(pairs, dim=-1),
    )
    return fn(_Owner(shards[0]), hidden_states, width)


def _window_errors(logits, vals, idx, width=WIDTH):
    """Mismatches between a window and the full-logits top-``width``: the
    values position by position, the (value, id) pairs above the
    width-th value as a set, and the ids at that value carrying it."""
    logits = logits.float()
    vals = vals.float().cpu()
    idx = idx.long().cpu()
    ref = torch.sort(logits, dim=-1, descending=True, stable=True)
    ref_v, ref_i = ref.values[:, :width], ref.indices[:, :width]
    errors = []
    if not torch.equal(vals, ref_v):
        errors.append("values")
    for row in range(logits.shape[0]):
        kth = ref_v[row, -1]
        got = {(float(v), int(i)) for v, i in zip(vals[row], idx[row]) if v > kth}
        want = {(float(v), int(i)) for v, i in zip(ref_v[row], ref_i[row]) if v > kth}
        if got != want:
            errors.append(f"row {row}: pairs above the k-th value")
        if not bool(torch.all(logits[row, idx[row]] == vals[row])):
            errors.append(f"row {row}: an id does not carry its value")
        if len(set(idx[row].tolist())) != width:
            errors.append(f"row {row}: repeated id")
    return errors


def _case(kind, rows, vocab, hidden, seed, dtype):
    """Weights and hidden states whose logits have the named structure."""
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randn(vocab, hidden, generator=gen)
    hidden_states = torch.randn(rows, hidden, generator=gen)
    if kind == "one_shard":  # every top value inside one rank's slice
        hidden_states[:, 0] = 1.0
        weight[vocab // 2 : vocab // 2 + 100, 0] += 50.0
    elif kind == "ties":  # coarse logits: ties everywhere, at the max too
        weight = torch.randint(0, 3, (vocab, hidden), generator=gen).float()
        hidden_states = torch.randint(0, 2, (rows, hidden), generator=gen).float()
    elif kind == "flat":
        weight = torch.zeros(vocab, hidden)
    return weight.to(dtype), hidden_states.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tp_size", [1, 2, 8])
@pytest.mark.parametrize("kind", ["random", "one_shard", "ties", "flat"])
def test_window_holds_the_full_logits_top_k(monkeypatch, dtype, tp_size, kind):
    # 8 x 9024 rows (141 chunks of 64; bf16 takes the one-hot fetch, float32 the
    # gather; 2 ranks and 1 take _chunked_topk): wider than WIDTH chunks, so the
    # chunked top-k path runs; the last rank carries padding rows.
    weight, hidden_states = _case(kind, 6, 72190, 16, tp_size, dtype)

    vals, idx = _vp_window(weight, hidden_states, tp_size, monkeypatch)

    assert vals.dtype == torch.float32 and idx.dtype == torch.int32
    assert vals.shape == idx.shape == (6, WIDTH)
    logits = hidden_states @ weight.T
    assert _window_errors(logits, vals, idx) == []
    assert int(idx.max()) < weight.shape[0]


@pytest.mark.parametrize("tp_size", [1, 8])
def test_window_values_equal_the_sparse_sampler_window(monkeypatch, tp_size):
    weight, hidden_states = _case("random", 5, 72190, 16, 3, torch.bfloat16)
    logits = hidden_states @ weight.T

    vals, idx = _vp_window(weight, hidden_states, tp_size, monkeypatch)
    ref_vals, ref_idx = _chunked_topk(logits, WIDTH)

    assert torch.equal(vals, ref_vals.float())
    assert _window_errors(logits, ref_vals, ref_idx) == []


@pytest.mark.parametrize("kind", ["random", "max_ties"])
def test_greedy_rows_take_the_lowest_id_argmax(monkeypatch, kind):
    # Exact whenever the ids tied at a row's maximum fit in the window
    # (here: 5 ids on 3 ranks, two in one chunk); with more than WIDTH of
    # them the window holds an arbitrary WIDTH.
    weight, hidden_states = _case("random", 7, 72190, 16, 11, torch.bfloat16)
    if kind == "max_ties":
        hidden_states[:, 0] = 1.0
        for i in (70000, 30001, 9030, 9031, 50000):
            weight[i] = 0.0
            weight[i, 0] = 100.0
    vals, idx = _vp_window(weight, hidden_states, 8, monkeypatch)
    rows = hidden_states.shape[0]
    temps = torch.zeros((rows, 1))
    top_k = torch.full((rows, 1), 20, dtype=torch.int32)
    top_p = torch.ones((rows, 1))

    tokens = sparse_window_sample(
        vals, idx, temps, top_k, top_p, torch.rand((rows, WIDTH)), 1e-5
    )

    full = torch.argmax((hidden_states @ weight.T).float(), dim=-1)
    assert torch.equal(tokens.long(), full)


def test_window_sample_matches_the_full_distribution(monkeypatch):
    torch.manual_seed(5)
    n = 8192
    weight, hidden_states = _case("random", 1, 72190, 16, 5, torch.float32)
    weight = weight * 0.6
    vals, idx = _vp_window(weight, hidden_states, 8, monkeypatch)
    temps = torch.full((n, 1), 0.6)
    top_k = torch.full((n, 1), 20, dtype=torch.int32)
    top_p = torch.full((n, 1), 0.95)

    tokens = sparse_window_sample(
        vals.expand(n, WIDTH),
        idx.expand(n, WIDTH),
        temps,
        top_k,
        top_p,
        torch.rand((n, WIDTH)),
        1e-5,
    )

    logits = (hidden_states @ weight.T)[0] / 0.6
    keep = torch.sort(logits, descending=True).values[:20]
    probs = torch.softmax(keep, dim=-1)
    mass = torch.cumsum(probs, dim=-1)
    kept = int((mass < 0.95).sum()) + 1
    p = torch.zeros_like(logits)
    order = torch.sort(logits, descending=True).indices[:kept]
    p[order] = torch.softmax(logits[order], dim=-1)
    freq = torch.bincount(tokens.long(), minlength=logits.shape[0]).float() / n
    assert freq[p == 0].sum() == 0
    sigma = torch.sqrt(p * (1 - p) / n)
    assert torch.all((freq - p).abs() <= 5 * sigma + 1e-6)


def _offset_bug(owner, hidden_states, width):
    """A rank that books its candidates under the previous rank's ids."""
    start = owner.lm_head.shard_indices.org_vocab_start_index
    per_rank = owner.lm_head.weight.shape[0]
    owner.lm_head.shard_indices.org_vocab_start_index = max(0, start - per_rank)
    try:
        return topk_window.window_from_hidden_states(owner, hidden_states, width)
    finally:
        owner.lm_head.shard_indices.org_vocab_start_index = start


@pytest.mark.parametrize("bug", ["offset", "short_shard", "short_shard_keys"])
def test_window_check_catches_a_broken_window(monkeypatch, bug):
    # Positive control: the comparison above must fail on a wrong window.
    dtype = torch.bfloat16 if bug == "short_shard_keys" else torch.float32
    weight, hidden_states = _case("one_shard", 4, 72190, 16, 7, dtype)
    logits = hidden_states @ weight.T
    if bug == "short_shard_keys":
        real_keys = topk_window.shard_topk_keys

        def short_keys(x, w):
            # the packed path with one candidate too few per rank
            keys = real_keys(x, w)
            return torch.cat([keys[:, : w - 1], torch.full_like(keys[:, :1], -(2**31))], -1)

        monkeypatch.setattr(topk_window, "shard_topk_keys", short_keys)
        vals, idx = _vp_window(weight, hidden_states, 8, monkeypatch)
    elif bug == "offset":
        vals, idx = _vp_window(weight, hidden_states, 8, monkeypatch, fn=_offset_bug)
    else:
        real = topk_window.shard_topk

        def short_shard(x, w):
            # one candidate too few: the w-th is replaced by a sentinel
            vals, idx = real(x, w)
            vals = torch.cat([vals[:, : w - 1], torch.full_like(vals[:, :1], -1e30)], -1)
            return vals, idx

        monkeypatch.setattr(topk_window, "shard_topk", short_shard)
        vals, idx = _vp_window(weight, hidden_states, 8, monkeypatch)
    assert _window_errors(logits, vals, idx) != []


def _spec_md(num_reqs, k, padded_reqs, rows):
    draft = torch.full((rows,), -1, dtype=torch.int32)
    seg = torch.full((rows,), padded_reqs, dtype=torch.int64)
    grp = torch.zeros(rows, dtype=torch.int32)
    tgt = torch.zeros(rows, dtype=torch.int32)
    bonus = torch.zeros(padded_reqs, dtype=torch.int32)
    lengths = torch.zeros(padded_reqs, dtype=torch.int32)
    for q in range(num_reqs):
        for j in range(k):
            draft[q * k + j] = 1000 * q + 7 * j
            seg[q * k + j] = q
            grp[q * k + j] = j
            tgt[q * k + j] = q * (k + 1) + j
        bonus[q] = q * (k + 1) + k
        lengths[q] = k
    return SimpleNamespace(
        draft_token_ids=draft,
        draft_lengths=lengths,
        segment_ids=seg,
        group_indices=grp,
        target_logits_indices=tgt,
        bonus_logits_indices=bonus,
    )


def _runner(owner, num_reqs, top_k, temps, max_num_logprobs=None, params=None):
    runner = object.__new__(TPUModelRunner)
    runner.is_pooling_model = False
    runner._topk_window_owner = owner
    runner._sampling_topk_bound = WIDTH
    runner.vocab_size = owner.lm_head.weight.shape[0] if owner is not None else 1000
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    runner.rejection_sampler = RejectionSampler()
    params = params or [SamplingParams(temperature=0.6, top_k=20)] * num_reqs
    runner.requests = {
        f"r{i}": SimpleNamespace(sampling_params=p) for i, p in enumerate(params)
    }
    runner.input_batch = SimpleNamespace(
        all_greedy=False,
        max_num_logprobs=max_num_logprobs,
        req_ids=[*runner.requests, None],
        num_reqs=num_reqs,
        top_k_cpu_tensor=torch.tensor(top_k, dtype=torch.int32),
        temperature_cpu_tensor=torch.tensor(temps, dtype=torch.float32),
        top_p_cpu_tensor=torch.full((num_reqs,), 0.95),
    )
    return runner


def test_window_width_takes_only_sparse_verify_chunks():
    owner = _Owner(_shards(torch.zeros(1000, 4), 1)[0])
    md = object()

    def width(runner, md=md, all_greedy=False):
        return TPUModelRunner._topk_window_width(runner, md, 0, 2, all_greedy)

    assert width(_runner(owner, 2, [20, 64], [0.6, 0.6])) == WIDTH
    assert width(_runner(owner, 2, [20, 1000], [0.6, 0.0])) == WIDTH  # greedy row
    assert width(_runner(None, 2, [20, 20], [0.6, 0.6])) == 0  # flag off
    assert width(_runner(owner, 2, [20, 20], [0.6, 0.6]), md=None) == 0  # no drafts
    assert width(_runner(owner, 2, [20, 20], [0.6, 0.6]), all_greedy=True) == 0
    assert width(_runner(owner, 2, [20, 65], [0.6, 0.6])) == 0  # top_k > bound
    assert width(_runner(owner, 2, [20, 1000], [0.6, 0.6])) == 0  # top_k off
    assert width(_runner(owner, 2, [20, 20], [0.6, 0.6], max_num_logprobs=1)) == 0
    grammar = SamplingParams(
        temperature=0.6, top_k=20, structured_outputs=StructuredOutputsParams(regex="a+")
    )
    runner = _runner(
        owner, 2, [20, 20], [0.6, 0.6], params=[SamplingParams(top_k=20), grammar]
    )
    assert width(runner) == 0


def test_verify_chunk_from_window_equals_the_full_logits_sparse_path(monkeypatch):
    # Same generator state, same windows (float32 logits: no ties) -> the
    # vocab-parallel chunk selects exactly the tokens the full-logits sparse
    # path selects.
    num_reqs, k, padded_reqs, rows = 3, 3, 8, 16
    weight, hidden = _case("random", rows, 72190, 16, 13, torch.float32)
    shards = _shards(weight, 8)
    runner = _runner(_Owner(shards[0]), num_reqs, [20, 5, 64], [0.6, 1.0, 0.0])
    md = _spec_md(num_reqs, k, padded_reqs, rows)
    logits = hidden @ weight.T
    for q in range(num_reqs):  # draft = the target's top token for some rows
        md.draft_token_ids[q * k] = int(torch.argmax(logits[q * (k + 1)]))
    window = _vp_window(weight, hidden, 8, monkeypatch)

    with torch.compiler.set_stance("force_eager"):
        gen = torch.Generator().manual_seed(0)
        full = TPUModelRunner._sample_spec_verify_chunk_sparse(
            runner, logits, md, 0, num_reqs, gen
        )
        gen = torch.Generator().manual_seed(0)
        vp = TPUModelRunner._sample_spec_verify_chunk(
            runner, window, md, None, None, 0, num_reqs, False, gen
        )

    assert torch.equal(vp, full)
    assert int((vp[:num_reqs] >= 0).sum()) >= num_reqs


def test_runner_selected_window_matches_selected_logits_top_k():
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(20000, 8, generator=gen)
    hidden_states = torch.randn(12, 8, generator=gen)
    indices = torch.tensor([11, 0, 3, 3], dtype=torch.int32)
    runner = object.__new__(TPUModelRunner)
    runner.is_pooling_model = False
    runner.model = SimpleNamespace(compute_logits=lambda h: h @ weight.T)
    runner._topk_window_owner = _Owner(_shards(weight, 1)[0])

    with torch.compiler.set_stance("force_eager"):
        vals, idx = TPUModelRunner.compute_selected_topk_window(
            runner, hidden_states, indices, WIDTH
        )
        logits = TPUModelRunner.compute_selected_logits(runner, hidden_states, indices)

    assert _window_errors(logits, vals, idx) == []


def test_fused_verify_equals_the_two_programs_on_the_same_uniforms():
    # TPU_SPARSE_SPEC_FUSED_VERIFY slices one flat draw; fed the same numbers
    # the unfused bonus + rejection programs draw, it returns the same tokens.
    num_reqs, k, padded_reqs, rows = 3, 3, 8, 16
    gen = torch.Generator().manual_seed(21)
    logits = torch.randn(rows, 20000, generator=gen) * 2.0
    md = _spec_md(num_reqs, k, padded_reqs, rows)
    for q in range(num_reqs):
        md.draft_token_ids[q * k + 1] = int(torch.argmax(logits[q * (k + 1) + 1]))
    vals, idx = _chunked_topk(logits, WIDTH)
    temps = torch.tensor([[0.6], [1.0], [0.0]] + [[1.0]] * 5)
    top_k = torch.tensor([[20], [5], [64]] + [[0]] * 5, dtype=torch.int32)
    top_p = torch.tensor([[0.95], [0.5], [1.0]] + [[1.0]] * 5)
    bonus_u = torch.rand((padded_reqs, WIDTH), generator=gen)
    accept_u = torch.rand((rows,), generator=gen)
    recover_u = torch.rand((rows, WIDTH), generator=gen)

    with torch.compiler.set_stance("force_eager"):
        bonus, t_vals, t_idx = TPUModelRunner._window_bonus(
            vals,
            idx,
            md.bonus_logits_indices,
            md.target_logits_indices,
            temps,
            top_k,
            top_p,
            bonus_u,
        )
        two = RejectionSampler().forward_sparse(
            draft_token_ids=md.draft_token_ids,
            num_draft_tokens=md.draft_lengths,
            target_vals=t_vals,
            target_idx=t_idx,
            bonus_token_ids=bonus,
            segment_ids=md.segment_ids,
            group_indices=md.group_indices,
            max_draft_tokens=k,
            temperatures=temps,
            top_k=top_k,
            top_p=top_p,
            accept_u=accept_u,
            recover_u=recover_u,
        )
        fused = TPUModelRunner._sparse_verify(
            vals,
            idx,
            md.bonus_logits_indices,
            md.target_logits_indices,
            md.draft_token_ids,
            md.draft_lengths,
            md.segment_ids,
            md.group_indices,
            temps,
            top_k,
            top_p,
            torch.cat([bonus_u.flatten(), accept_u, recover_u.flatten()]),
            None,
            k,
        )

    assert torch.equal(fused, two)


def test_fused_verify_from_window_equals_from_logits(monkeypatch):
    monkeypatch.setenv("TPU_SPARSE_SPEC_FUSED_VERIFY", "1")
    num_reqs, k, padded_reqs, rows = 3, 3, 8, 16
    weight, hidden = _case("random", rows, 72190, 16, 17, torch.float32)
    runner = _runner(_Owner(_shards(weight, 8)[0]), num_reqs, [20, 5, 64], [0.6, 1.0, 0.0])
    md = _spec_md(num_reqs, k, padded_reqs, rows)
    logits = hidden @ weight.T
    window = _vp_window(weight, hidden, 8, monkeypatch)

    with torch.compiler.set_stance("force_eager"):
        gen = torch.Generator().manual_seed(1)
        full = TPUModelRunner._sample_spec_verify_chunk_sparse(
            runner, logits, md, 0, num_reqs, gen
        )
        gen = torch.Generator().manual_seed(1)
        vp = TPUModelRunner._sample_spec_verify_chunk(
            runner, window, md, None, None, 0, num_reqs, False, gen
        )

    assert torch.equal(vp, full)


def test_packed_keys_round_trip_and_order():
    vals = torch.tensor(
        [[3.5, -2.0, float("-inf"), 0.0, 1e-3, -1e-3, 3.5, -3e38, 3e38]]
    ).to(torch.bfloat16)
    idx = torch.tensor([[0, 1, 2, 3, 4, 5, 32767, 7, 8]], dtype=torch.int32)

    keys = pack_value_index(vals, idx)
    back_vals, back_idx = unpack_value_index(keys)

    assert torch.equal(back_vals.view(torch.int16), vals.view(torch.int16))
    assert torch.equal(back_idx, idx)
    order = torch.argsort(keys[0], descending=True, stable=True)
    ref = sorted(range(9), key=lambda i: (-float(vals[0, i]), int(idx[0, i])))
    assert order.tolist() == ref
