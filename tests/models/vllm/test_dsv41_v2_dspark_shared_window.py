# SPDX-License-Identifier: Apache-2.0
"""DSpark draft attention: one shared sliding-window gather per query block."""

from __future__ import annotations

from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    _attend,
    _attend_shared_block,
    _attention_state,
    paged_attention,
    paged_attention_jax,
)

HEADS, DIM, ROWS = 2, 64, 96


def _pool(seed=0):
    """Word pool of finite BF16 pairs: word = hi << 16 | lo."""
    rng = np.random.default_rng(seed)
    vals = rng.normal(size=(ROWS, 128, 2)).astype(jnp.bfloat16)
    bits = np.asarray(vals).view(np.uint16).astype(np.uint32)
    words = (bits[..., 1] << 16) | bits[..., 0]
    return jnp.asarray(words.view(np.int32))


def _block_records(groups, block, width, tail, seed=1, invalid=()):
    """Rows of each block share one window; ``invalid`` tokens are all -1."""
    rng = np.random.default_rng(seed)
    records = np.full((groups * block + tail, width), -1, np.int32)
    for g in range(groups):
        valid = rng.integers(1, width + 1)
        window = rng.choice(ROWS, size=valid, replace=False).astype(np.int32)
        records[g * block : (g + 1) * block, :valid] = window
    for t in range(tail):
        records[groups * block + t, :3] = rng.choice(ROWS, size=3, replace=False)
    for t in invalid:
        records[t] = -1
    return jnp.asarray(records)


def _q(tokens, seed=2):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(tokens, HEADS, DIM)), jnp.bfloat16)


@pytest.mark.parametrize("width,tail", [(64, 0), (70, 2)])
def test_shared_block_matches_per_token_attend(width, tail):
    block, groups = 5, 3
    tokens = groups * block + tail
    q, pool = _q(tokens), _pool()
    invalid = (1, 7)
    records = _block_records(groups, block, width, tail, invalid=invalid)
    state = _attention_state(q)
    want = _attend(q, pool, records, state, 0.125)
    got = _attend_shared_block(q, pool, records, state, 0.125, block)
    keep = np.ones(tokens, bool)
    keep[list(invalid)] = False
    for g, w in zip(got, want):
        assert g.shape == w.shape
        np.testing.assert_allclose(
            np.asarray(g)[keep], np.asarray(w)[keep], rtol=1e-5, atol=1e-5
        )


def test_shared_block_mismatch_is_nan():
    block, groups = 5, 2
    q, pool = _q(groups * block), _pool()
    records = np.asarray(_block_records(groups, block, 64, 0)).copy()
    records[6, 0] = (records[6, 0] + 1) % ROWS  # token 6 leaves block 1's window
    got = _attend_shared_block(
        q, pool, jnp.asarray(records), _attention_state(q), 0.125, block
    )
    for r in got:
        r = np.asarray(r)
        assert np.isfinite(r[:5]).all()
        assert np.isnan(r[5:10]).any()


def test_paged_attention_jax_block_path_matches_window_path():
    block, groups = 5, 4
    tokens = groups * block + 3
    q, pool = _q(tokens), _pool()
    invalid = (2,)
    records = _block_records(groups, block, 64, 3, invalid=invalid)
    main = jnp.zeros((tokens, 0), jnp.int32)
    sink = jnp.zeros((HEADS,), jnp.float32)
    want = paged_attention_jax(q, pool, pool, records, main, sink, 0.125)
    got = paged_attention_jax(q, pool, pool, records, main, sink, 0.125, block)
    keep = np.ones(tokens, bool)
    keep[list(invalid)] = False
    np.testing.assert_allclose(
        np.asarray(got, np.float32)[keep],
        np.asarray(want, np.float32)[keep],
        rtol=2e-2,
        atol=2e-2,
    )


def test_paged_attention_jax_block_larger_than_tokens_uses_window_path():
    q, pool = _q(3), _pool()
    records = _block_records(0, 5, 64, 3)
    main = jnp.zeros((3, 0), jnp.int32)
    sink = jnp.zeros((HEADS,), jnp.float32)
    with mock.patch.object(
        attention_kernels, "_attend_shared_block", side_effect=AssertionError
    ):
        out = paged_attention_jax(q, pool, pool, records, main, sink, 0.125, 5)
    assert out.shape == q.shape


class _Buf:
    def __init__(self, hi=False):
        self.raw = torch.zeros(4, 128, dtype=torch.int32)
        self.raw_hi = torch.zeros(4, 128, dtype=torch.int32) if hi else None
        self.total_rows = 8 if hi else 4


@pytest.mark.parametrize("hi", [False, True])
def test_paged_attention_forwards_block(monkeypatch, hi):
    op = mock.MagicMock(return_value=torch.zeros(4, 2, 64))
    name = "_attention_2bank_op" if hi else "_attention_op"
    monkeypatch.setattr(attention_kernels, name, op)
    monkeypatch.setattr(attention_kernels, "_has_hi_bank", lambda main: hi)
    rec = torch.zeros(4, 64, dtype=torch.int32)
    paged_attention(
        torch.zeros(4, 2, 64), _Buf(hi), _Buf(hi), rec, rec, torch.zeros(2), 0.1, 5
    )
    assert op.call_args.args[-1] == 5
    paged_attention(
        torch.zeros(4, 2, 64), _Buf(hi), _Buf(hi), rec, rec, torch.zeros(2), 0.1
    )
    assert op.call_args.args[-1] == 1


def test_attend_large_width_fori_loop_matches_unrolled():
    """Verify _attend fori_loop branch (steps > 8) matches static unroll."""
    tokens, width, tile = 4, 72, 8
    q, pool = _q(tokens), _pool()
    records = _block_records(tokens, 1, width, 0)
    state = _attention_state(q)
    got_fori = _attend(q, pool, records, state, 0.125, tile=tile)
    got_unrolled = _attend_shared_block(
        q, pool, records, state, 0.125, block=1, tile=tile
    )
    for g, w in zip(got_fori, got_unrolled):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-5, atol=1e-5)


def test_non_causal_swa_6page_fast_path_matches_uncached(monkeypatch):
    """Verify non_causal=True 6-page SWA fast path matches uncached _row_slots."""
    from types import SimpleNamespace

    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows

    count = 10
    rows_per_page = 32
    positions = torch.tensor(
        [150, 151, 152, 153, 154, 15, 16, 17, 18, 19], dtype=torch.int32
    )
    swa_table = torch.tensor(
        [[10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24, 25, 26]],
        dtype=torch.int32,
    )
    swa_meta = SimpleNamespace(
        seq_lens=torch.tensor([155, 20], dtype=torch.int32),
        block_tables=swa_table,
        query_start_loc=torch.tensor([0, 5, 10], dtype=torch.int32),
    )
    forward_ctx = SimpleNamespace(
        attn_metadata={"swa_pfx": swa_meta}, additional_kwargs={}
    )
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
    monkeypatch.setattr(v2_attn, "_insert", lambda *a, **kw: None)

    captured: list[torch.Tensor] = []
    monkeypatch.setattr(
        v2_attn,
        "paged_attention",
        lambda q, swa_c, main_c, recs, *args, **kw: (
            captured.append(recs.clone()) or torch.zeros_like(q)
        ),
    )

    cache_mock = mock.MagicMock(spec=PagedCacheRows)
    cache_mock.numel.return_value = 1024
    cache_mock.shape = (64, rows_per_page, 512)
    cache_mock.page_stride = rows_per_page
    cache_mock.row_offset = 1000

    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn.prefix = "swa_pfx"
    attn.window_size = 128
    attn.non_causal_multi_token_decode = True
    attn.dspark_block_size = 5
    attn.query_block_size = 5
    attn._ced_decoder = False
    attn._ced_global_from_encoder = False
    attn._local_q_attn = False
    attn.compressor = None
    attn.indexer = None
    attn.compress_ratio = 0
    attn._fp8_kv = False
    attn.n_local_heads = 2
    attn.head_dim = 8
    attn.scale = 0.25
    attn.attn_sink = torch.zeros(2)
    attn._full_attn_sink = None
    attn._tp_rank_1d = None
    attn._rotate = lambda x, p: x
    attn.wq_b = lambda qr: qr
    attn.swa_cache_layer = SimpleNamespace(prefix="swa_pfx", kv_cache=cache_mock)
    attn.kv_cache = cache_mock

    qr = torch.ones((count, 16), dtype=torch.bfloat16)
    kv = torch.ones((count, 8), dtype=torch.bfloat16)
    out = torch.zeros((count, 2, 8), dtype=torch.bfloat16)

    with v2_attn.stage_attention_cache_scope():
        attn._prepare_tpu(None, qr, kv, None, None, None, positions, out)
    attn._prepare_tpu(None, qr, kv, None, None, None, positions, out)
    assert len(captured) == 2
    assert torch.equal(captured[0], captured[1])
