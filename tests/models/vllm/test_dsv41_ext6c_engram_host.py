# SPDX-License-Identifier: Apache-2.0
"""EXT-6c: host-side Engram hashing and raw-row staging under OFFLOAD=1.

``NgramHashState.hash_on_host`` and ``ParallelEngramEmbedding.gather_raw`` /
``Engram.prepare_embeddings`` run on the host between decode steps. These
tests pin them against independent per-token Python references over
randomized DSpark K=5 batches and cover the helper branches.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as ce

DEAD = ce.upstream.DEAD_ID


def _hash_state(
    *,
    layers=2,
    ngram=4,
    heads=8,
    depth=3,
    vocab=3000,
    cvocab=2500,
    prime_range=(120000, 140000),
    pad_id=0,
    real_multipliers=True,
    seed=0,
):
    g = torch.Generator().manual_seed(seed)
    state = object.__new__(ce.NgramHashState)
    nn.Module.__init__(state)
    state.token_map = torch.randint(0, cvocab, (vocab,), generator=g)
    if real_multipliers:
        # Upstream's bound: odd, and token * multiplier cannot overflow int64.
        bound = (np.iinfo(np.int64).max // cvocab) // 2
        state.multipliers = (
            torch.randint(0, bound, (layers, ngram), generator=g, dtype=torch.int64) * 2
            + 1
        )
    else:
        state.multipliers = torch.randint(
            -(1 << 20), 1 << 20, (layers, ngram), generator=g
        )
    state.primes = torch.randint(*prime_range, (layers, ngram - 1, heads), generator=g)
    state.offsets = torch.randint(0, 99999, (layers, (ngram - 1) * heads), generator=g)
    state.lookback_depth = depth
    state.pad_id = pad_id
    state.use_slot_cache = True
    state.block_size = 64
    state._cache = None
    state._kv_cache_ref = None
    state.swa_cache_module = SimpleNamespace()
    return state


def _hash_inputs(
    num_reqs,
    *,
    depth=3,
    vocab=3000,
    stride=6,
    ragged=False,
    pad_tokens=0,
    dead_frac=0.0,
    unknown_frac=0.0,
    extra_lb_cols=0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    lens = rng.integers(1, 9, size=num_reqs) if ragged else np.full(num_reqs, stride)
    qsl = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    t = int(qsl[-1]) + pad_tokens
    ids = rng.integers(0, vocab, size=t).astype(np.int32)
    first = rng.integers(0, 12, size=num_reqs)  # small: sources before position 0
    first[rng.random(num_reqs) < 0.5] += rng.integers(0, 4000)
    pos = np.zeros(t, dtype=np.int64)
    for r in range(num_reqs):
        pos[qsl[r] : qsl[r + 1]] = first[r] + np.arange(lens[r])
    if pad_tokens:
        pos[qsl[-1] :] = rng.integers(0, 50, size=pad_tokens)
    dead = rng.random(t) < dead_frac
    lb = rng.integers(0, vocab, size=(num_reqs, depth + extra_lb_cols)).astype(np.int32)
    lb[rng.random(lb.shape) < unknown_frac] = -1
    lb_dead = rng.random(lb.shape) < dead_frac
    return (
        torch.from_numpy(ids),
        torch.from_numpy(pos),
        torch.from_numpy(qsl),
        torch.from_numpy(dead),
        torch.from_numpy(lb),
        torch.from_numpy(lb_dead),
    )


def _reference_hash(state, ids, pos, qsl, dead, lb, lb_dead):
    """Per-token restatement of upstream's n-gram hash (no NumPy broadcasting)."""
    token_map = state.token_map.tolist()
    mult = state.multipliers.tolist()
    primes = state.primes.tolist()
    offsets = state.offsets.tolist()
    layers, ngram = state.multipliers.shape
    heads = state.primes.shape[-1]
    depth = state.lookback_depth
    pad = int(state.pad_id)
    ids, pos, qsl = ids.tolist(), pos.tolist(), qsl.tolist()
    dead, lb, lb_dead = dead.tolist(), lb.tolist(), lb_dead.tolist()
    count = len(ids)
    num_rows = len(lb)
    mapped = [DEAD if dead[t] else token_map[ids[t]] for t in range(count)]
    out = np.zeros((count, layers, (ngram - 1) * heads), dtype=np.int64)
    for t in range(count):
        req = min(sum(1 for q in qsl[1:] if t >= q), num_rows - 1)
        start = min(qsl[req], count - 1)
        chunk_pos = pos[start]
        for layer in range(layers):
            rolling = 0
            blocked = False
            for shift in range(ngram):
                past = pos[t] - shift
                if past >= chunk_pos:
                    src = mapped[max(t - shift, 0)]
                else:
                    col = chunk_pos - 1 - past
                    if 0 <= col < depth and lb[req][col] >= 0:
                        src = DEAD if lb_dead[req][col] else token_map[lb[req][col]]
                    else:
                        src = pad
                blocked = blocked or past < 0 or src == DEAD
                value = pad if blocked else src
                rolling ^= value * mult[layer][shift]
                if shift > 0:
                    for head in range(heads):
                        col_out = (shift - 1) * heads + head
                        out[t, layer, col_out] = (
                            rolling % primes[layer][shift - 1][head]
                            + offsets[layer][col_out]
                        )
    return torch.from_numpy(out.astype(np.int32))


@pytest.mark.parametrize("num_reqs", [1, 2, 7, 48, 224])
@pytest.mark.parametrize(
    "state_kw, input_kw",
    [
        ({}, {}),
        ({}, {"dead_frac": 0.15, "unknown_frac": 0.3}),
        ({}, {"ragged": True, "dead_frac": 0.05, "unknown_frac": 0.5}),
        ({}, {"pad_tokens": 7, "unknown_frac": 0.2}),
        ({"prime_range": (3, 100)}, {"dead_frac": 0.1}),
        ({"real_multipliers": False}, {"dead_frac": 0.1}),
        (
            {"ngram": 3, "depth": 4, "heads": 4, "pad_id": 7},
            {"depth": 4, "extra_lb_cols": 1},
        ),
        ({"ngram": 5, "depth": 4, "heads": 3, "layers": 1}, {"depth": 4, "stride": 1}),
        ({"ngram": 2, "depth": 1, "heads": 2}, {"depth": 1, "ragged": True}),
    ],
)
def test_hash_on_host_matches_per_token_reference(num_reqs, state_kw, input_kw):
    state = _hash_state(**state_kw)
    for seed in range(2):
        args = _hash_inputs(num_reqs, seed=seed, **input_kw)
        out = state.hash_on_host(*args)
        ref = _reference_hash(state, *args)
        assert out.dtype == torch.int32 and out.is_contiguous()
        assert torch.equal(out, ref)


def test_hash_on_host_tables_and_scratch_reuse():
    state = _hash_state()
    args_a = _hash_inputs(3)
    args_b = _hash_inputs(5, seed=1)
    out_a1 = state.hash_on_host(*args_a)
    tables = state._host_tables
    assert isinstance(tables, ce._HostHashTables)
    assert tables.token_map.dtype == np.int32
    scratch_a = tables.scratch(args_a[0].numel())
    assert scratch_a.shape == (2, 3, 8, args_a[0].numel())
    assert tables.scratch(args_a[0].numel()) is scratch_a
    # A different chunk size reallocates; the same size afterwards reuses.
    out_b = state.hash_on_host(*args_b)
    assert tables.scratch(args_b[0].numel()) is not scratch_a
    out_a2 = state.hash_on_host(*args_a)
    assert state._host_tables is tables
    assert torch.equal(out_a1, out_a2)
    assert torch.equal(out_b, _reference_hash(state, *args_b))
    # Outputs are fresh tensors, not views of the scratch buffer.
    assert out_a1.data_ptr() != out_a2.data_ptr()
    # Empty chunk short-circuits before touching the tables.
    empty = state.hash_on_host(
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int64),
        torch.tensor([0], dtype=torch.int32),
        torch.empty(0, dtype=torch.bool),
        torch.full((1, 3), -1, dtype=torch.int32),
        torch.zeros((1, 3), dtype=torch.bool),
    )
    assert empty.shape == (0, 2, 24) and empty.dtype == torch.int32


# ---------------------------------------------------------------- staging ---


def _embedding(*, cols, n_hash_cols, rows, head_start=0, dim=64, block=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    embed = object.__new__(ce.ParallelEngramEmbedding)
    nn.Module.__init__(embed)
    embed.cpu_offload = True
    embed.host_dma = False
    embed._host_rows = None
    embed.head_start = head_start
    embed.part_n_hash_cols = cols
    embed.n_hash_cols = n_hash_cols
    embed.vocab_start_idx = 10
    embed.vocab_end_idx = 10 + rows
    embed.part_num_embeddings = rows
    embed.dim = dim
    embed.block_size = block
    embed.tp_size = 1
    embed.dp_size = 1
    weight = torch.randint(0, 255, (rows, dim), generator=g, dtype=torch.uint8)
    scale = torch.randint(1, 254, (rows, dim // block), generator=g, dtype=torch.uint8)
    embed._storage = lambda: (weight, scale)
    return embed


def _reference_gather(embed, indices):
    weight, scale = embed._storage()
    t = indices.shape[0]
    cols, dim, blocks = embed.part_n_hash_cols, embed.dim, embed.dim // embed.block_size
    raw = np.zeros((t, cols, dim), dtype=np.uint8)
    sc = np.zeros((t, cols, blocks), dtype=np.uint8)
    for i in range(t):
        for c in range(cols):
            head = embed.head_start + c
            if head >= embed.n_hash_cols:
                sc[i, c] = ce._INVALID_SCALE
                row = 0
            else:
                idx = int(indices[i, head])
                if embed.vocab_start_idx <= idx < embed.vocab_end_idx:
                    row = idx - embed.vocab_start_idx
                    sc[i, c] = scale[row].numpy()
                else:
                    row = min(max(idx - embed.vocab_start_idx, 0), weight.shape[0] - 1)
                    sc[i, c] = ce._INVALID_SCALE
            raw[i, c] = weight[row].numpy()
    return raw, sc


@pytest.mark.parametrize(
    "cols, n_hash_cols, head_start, invalid_frac",
    [
        (3, 24, 0, 0.0),  # all valid: the sentinel write is skipped
        (3, 24, 3, 0.2),  # middle shard, some rows outside it
        (3, 7, 6, 0.0),  # last shard with a padded head
        (1, 1, 0, 0.5),
    ],
)
def test_gather_raw_matches_reference(cols, n_hash_cols, head_start, invalid_frac):
    rows = 500
    embed = _embedding(
        cols=cols, n_hash_cols=n_hash_cols, rows=rows, head_start=head_start
    )
    rng = np.random.default_rng(1)
    t = 37
    ids = rng.integers(
        embed.vocab_start_idx, embed.vocab_end_idx, size=(t, n_hash_cols)
    )
    ids[rng.random(ids.shape) < invalid_frac] += rows  # beyond the shard
    ids[0, : min(2, n_hash_cols)] = 0  # below the shard start
    ids_t = torch.from_numpy(ids.astype(np.int32))
    raw_out = torch.zeros((t, cols, embed.dim), dtype=torch.uint8)
    scale_out = torch.zeros((t, cols, embed.dim // embed.block_size), dtype=torch.uint8)
    embed.gather_raw(ids_t, raw_out, scale_out)
    raw_ref, scale_ref = _reference_gather(embed, ids)
    np.testing.assert_array_equal(scale_out.numpy(), scale_ref)
    # Raw bytes of valid rows must match; invalid rows are "as gathered".
    valid = scale_ref[..., 0] != ce._INVALID_SCALE
    np.testing.assert_array_equal(raw_out.numpy()[valid], raw_ref[valid])
    # Per-layer slice of a [T, layers, heads] hash tensor (non-contiguous ids).
    stacked = torch.stack([ids_t, ids_t + 1], dim=1)
    embed.gather_raw(stacked[:, 0], raw_out, scale_out)
    np.testing.assert_array_equal(scale_out.numpy(), scale_ref)


def _engram(embed, max_tokens):
    engram = object.__new__(ce.Engram)
    nn.Module.__init__(engram)
    engram.embed_tokens = embed
    ce.upstream.Engram._init_staging(engram, max_tokens, embed.dim)
    engram._init_staging(max_tokens, embed.dim)
    return engram


@pytest.mark.parametrize("cols, block", [(3, 16), (1, 64), (5, 32)])
def test_prepare_embeddings_packs_raw_then_scales(cols, block):
    rows = 300
    embed = _embedding(cols=cols, n_hash_cols=cols, rows=rows, block=block)
    engram = _engram(embed, max_tokens=64)
    rng = np.random.default_rng(2)
    blocks = embed.dim // block
    for t in (1, 7, 64, 30):
        ids = rng.integers(embed.vocab_start_idx, embed.vocab_end_idx, size=(t, cols))
        ids[rng.random(ids.shape) < 0.1] += rows
        ids_t = torch.from_numpy(ids.astype(np.int32))
        engram.prepare_embeddings(ids_t)
        raw_ref, scale_ref = _reference_gather(embed, ids)
        raw_bytes = t * cols * embed.dim
        scale_bytes = t * cols * blocks
        scale_words = t * (-(-(cols * blocks) // 4))  # rounded up per token
        staged = engram._staged_tensors(t)
        assert engram._active_raw is engram._staged_views(t)
        assert staged.numel() == raw_bytes // 4 + scale_words
        packed = staged.view(torch.uint8).numpy()
        np.testing.assert_array_equal(
            packed[:raw_bytes].reshape(t, cols, embed.dim)[scale_ref[..., 0] != 255],
            raw_ref[scale_ref[..., 0] != 255],
        )
        np.testing.assert_array_equal(
            packed[raw_bytes : raw_bytes + scale_bytes].reshape(t, cols, blocks),
            scale_ref,
        )
        # Any tail bytes up to the word boundary carry the invalid sentinel.
        assert (packed[raw_bytes + scale_bytes :] == ce._INVALID_SCALE).all()
