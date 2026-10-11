# SPDX-License-Identifier: Apache-2.0
"""TD8.285: V4.1 adapter cache/compressor int32 addressing equals the int64 one.

``token_requests`` / ``cache_slots`` / ``row_slots`` and the ratio-2 ring in
``save_compress_norm`` now use native int32 rows/requests and flat 1D
``index_select`` gathers. The ``_old_*`` functions are the pre-TD8.285 bodies.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    deepseek_v41_cache as cache_mod,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    deepseek_v41_compressor as comp_mod,
)


def _old_token_requests(metadata, positions, *, rows=None):
    if rows is None:
        rows = torch.arange(positions.shape[0], device=positions.device)
    return torch.bucketize(rows, metadata.query_start_loc[1:], right=True)


def _old_cache_slots(
    metadata, positions, *, rows_per_page, compress_ratio=1, ring=False, requests=None
):
    if metadata.seq_lens.numel() == 0:
        return torch.full_like(positions, -1)
    table = cache_mod.block_table(metadata)
    rows = torch.arange(positions.shape[0], device=positions.device)
    if requests is None:
        requests = _old_token_requests(metadata, positions, rows=rows)
    requests = requests.clamp(max=table.shape[0] - 1)
    logical = positions if compress_ratio == 1 else positions // compress_ratio
    columns = torch.zeros_like(logical) if ring else logical // rows_per_page
    blocks = table[requests, columns.clamp(max=table.shape[1] - 1).long()].long()
    slots = blocks * rows_per_page + logical % rows_per_page
    return slots.masked_fill(rows >= metadata.query_start_loc[-1], -1)


def _old_row_slots(block_table, requests, logical, *, rows_per_page):
    safe = logical.clamp(min=0, max=block_table.shape[1] * rows_per_page - 1)
    blocks = block_table[requests[:, None], safe // rows_per_page].long()
    return blocks * rows_per_page + safe % rows_per_page


def _meta(seed, idx_dtype, lens=(3, 1, 5, 2), width=6, pad=3):
    g = torch.Generator().manual_seed(seed)
    reqs = len(lens)
    qsl = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=idx_dtype)
    starts = torch.randint(0, 20, (reqs,), generator=g)
    pos = torch.cat([torch.arange(s, s + n) for s, n in zip(starts, lens, strict=True)])
    # Padded tail tokens repeat the last real position.
    pos = torch.cat([pos, pos[-1:].expand(pad)]).to(idx_dtype)
    tables = torch.randperm(200, generator=g)[: reqs * width].to(torch.int32)
    meta = SimpleNamespace(
        seq_lens=(starts + torch.tensor(lens)).to(torch.int32),
        block_tables=tables,
        query_start_loc=qsl,
    )
    return meta, pos


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    ("ratio", "ring", "rows_per_page"), [(1, False, 4), (2, False, 2), (2, True, 8)]
)
def test_cache_slots_and_token_requests_match_int64(
    idx_dtype, seed, ratio, ring, rows_per_page
):
    meta, pos = _meta(seed, idx_dtype)
    reqs = cache_mod.token_requests(meta, pos)
    assert reqs.dtype == torch.int32
    assert torch.equal(reqs.long(), _old_token_requests(meta, pos))
    for given in (None, reqs):
        got = cache_mod.cache_slots(
            meta,
            pos,
            rows_per_page=rows_per_page,
            compress_ratio=ratio,
            ring=ring,
            requests=given,
        )
        ref = _old_cache_slots(
            meta,
            pos,
            rows_per_page=rows_per_page,
            compress_ratio=ratio,
            ring=ring,
            requests=None if given is None else given.long(),
        )
        assert got.dtype == torch.int32
        assert torch.equal(got.long(), ref.long())


def test_cache_slots_empty_metadata_is_int32():
    empty = SimpleNamespace(seq_lens=torch.zeros((0,), dtype=torch.int32))
    out = cache_mod.cache_slots(empty, torch.arange(4), rows_per_page=4)
    assert out.dtype == torch.int32 and torch.equal(out, torch.full((4,), -1).int())


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_row_slots_match_int64(idx_dtype):
    g = torch.Generator().manual_seed(3)
    table = torch.randperm(100, generator=g)[:24].view(4, 6).to(torch.int32)
    requests = torch.tensor([0, 3, 1, 2, 3], dtype=idx_dtype)
    logical = torch.randint(-5, 40, (5, 7), generator=g).to(idx_dtype)
    got = cache_mod.row_slots(table, requests, logical, rows_per_page=4)
    ref = _old_row_slots(table, requests, logical, rows_per_page=4)
    assert got.dtype == torch.int32
    assert torch.equal(got.long(), ref)


def _old_ratio2(kv_score, positions, cache, slot_mapping, qsl, token_to_req, dim):
    """Pre-TD8.285 ratio-2 previous-row selection (CPU ``read_cache``)."""
    kv = kv_score.float()
    count = kv.shape[0]
    capacity = cache.shape[1]
    rows = torch.arange(count)
    requests = token_to_req.long().clamp(max=qsl.numel() - 2)
    starts = qsl[requests]
    previous_slots = (
        slot_mapping.clamp(min=0) // capacity * capacity + (positions - 1) % capacity
    )
    return torch.where(
        (rows > starts)[:, None],
        kv[(rows - 1).clamp(min=0)],
        cache_mod.read_cache(cache, previous_slots),
    )


@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
def test_save_compress_norm_ratio2_matches_int64(monkeypatch, pos_dtype):
    g = torch.Generator().manual_seed(7)
    dim = 4
    kv_score = torch.randn(6, 2 * dim, generator=g).to(torch.bfloat16)
    positions = torch.tensor([5, 6, 7, 0, 1, 2], dtype=pos_dtype)
    cache = torch.randn(3, 4, 2 * dim, generator=g)
    slot_mapping = torch.tensor([1, 2, 3, 4, -1, 6], dtype=torch.int32)
    qsl = torch.tensor([0, 3, 6], dtype=torch.int32)
    token_to_req = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    expected_prev = _old_ratio2(
        kv_score, positions, cache.clone(), slot_mapping, qsl, token_to_req, dim
    )

    seen = {}
    real_read = comp_mod.read_cache

    def spy_read(c, slots, layout=None):
        seen["slots"] = slots
        return real_read(c, slots, layout=layout)

    monkeypatch.setattr(comp_mod, "read_cache", spy_read)
    latent = torch.empty(6, dim, dtype=torch.bfloat16)
    comp_mod.save_compress_norm(
        kv_score,
        positions,
        cache,
        slot_mapping,
        qsl,
        token_to_req,
        torch.ones(dim),
        1e-6,
        2,
        latent,
    )
    assert seen["slots"].dtype == torch.int32

    # Recompute the expected latent from the reference previous rows.
    kv = kv_score.float()
    scores = torch.stack((expected_prev[:, dim:], kv[:, dim:]), 1)
    pair = torch.stack((expected_prev[:, :dim], kv[:, :dim]), 1)
    values = (pair * scores.softmax(1)).sum(1)
    valid = (slot_mapping >= 0) & (positions % 2 == 1)
    ref = comp_mod.rms_norm(values, torch.ones(dim), 1e-6)
    ref = ref.masked_fill(~valid[:, None], 0).to(torch.bfloat16)
    assert torch.equal(latent.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_td8300_read_cache_1d_int32_index_select_and_spec_logprobs_int32(slot_dtype):
    import numpy as np

    from vllm_torchtpu.runner.tpu_runner_async_output import (
        LogprobsTensors,
        SpecLogprobsChunk,
        spec_logprobs_lists,
    )

    g = torch.Generator().manual_seed(300)
    dense = torch.randn((4, 8, 16), generator=g, dtype=torch.bfloat16)
    slots = torch.tensor([[0, 7, 9], [15, 23, 31]], dtype=slot_dtype)
    ref_dense = dense[slots // dense.shape[1], slots % dense.shape[1]]
    got_dense = cache_mod.read_cache(dense, slots, layout=None)
    assert torch.equal(got_dense.view(torch.int16), ref_dense.view(torch.int16))

    raw = torch.randint(
        0, 1000, (32, cache_mod.ROW_WORDS), generator=g, dtype=torch.int32
    )
    layout = cache_mod.RecordLayout(8, 8, 0, 256, torch.bfloat16)
    got_layout = cache_mod.read_cache(raw, slots, layout=layout)
    pool = raw.view(torch.int8).reshape(-1, 256 * 2)
    ref_layout = (
        pool[slots.long()].contiguous().view(torch.bfloat16).reshape(*slots.shape, 256)
    )
    assert torch.equal(got_layout.view(torch.int16), ref_layout.view(torch.int16))

    chunk = SpecLogprobsChunk(
        tensors=LogprobsTensors(
            logprob_token_ids=torch.tensor([[10, 11], [20, 21], [30, 31]]),
            logprobs=torch.zeros((3, 2)),
            selected_token_ranks=torch.tensor([0, 1, 0]),
        ),
        row_starts=np.array([0, 2], dtype=np.int32),
    )
    lists = spec_logprobs_lists([chunk], [2, 1])
    assert lists.cu_num_generated_tokens == [0, 2, 3]


def test_td8308_cache_slots_row_slots_and_dsv4_int32_guards(monkeypatch):
    """Verify TD8.308 int32 cast guards in cache ops, dsv4 attention, and dspark."""
    from unittest import mock

    from vllm_torchtpu.models.vllm.deepseek_v4 import attention as dsv4_attn
    from vllm_torchtpu.models.vllm.deepseek_v4 import dspark as dsv4_dspark

    # 1. read_cache non-CPU path with int32 and int64 slots
    captured_slots = []
    monkeypatch.setattr(
        cache_mod,
        "_read_values_op",
        lambda cache, slots, *args: (
            captured_slots.append(slots) or torch.zeros(slots.numel(), 4)
        ),
    )
    fake_cache = mock.MagicMock()
    fake_cache.device.type = "tpu"
    layout = SimpleNamespace(
        geometry=(4, 4, 0),
        elements=4,
        dtype=torch.bfloat16,
    )
    for dt in (torch.int32, torch.int64):
        out = cache_mod.read_cache(
            fake_cache, torch.tensor([[0, 1], [2, 3]], dtype=dt), layout=layout
        )
        assert out.shape == (2, 2, 4)
    assert len(captured_slots) == 2
    assert all(s.dtype == torch.int32 for s in captured_slots)

    # 2. cache_slots and row_slots with int32 and int64 inputs
    for dt in (torch.int32, torch.int64):
        meta = SimpleNamespace(
            seq_lens=torch.tensor([4, 4], dtype=dt),
            block_tables=torch.arange(8, dtype=dt),
            query_start_loc=torch.tensor([0, 2, 4], dtype=dt),
        )
        pos = torch.tensor([0, 1, 2, 3], dtype=dt)
        reqs = torch.tensor([0, 0, 1, 1], dtype=dt)
        slots_r1 = cache_mod.cache_slots(
            meta, pos, rows_per_page=2, compress_ratio=1, requests=reqs
        )
        slots_r2 = cache_mod.cache_slots(
            meta, pos, rows_per_page=2, compress_ratio=2, requests=reqs
        )
        assert slots_r1.dtype == torch.int32
        assert slots_r2.dtype == torch.int32

        r_slots = cache_mod.row_slots(
            torch.arange(8, dtype=dt).view(2, 4),
            reqs,
            pos[:, None],
            rows_per_page=2,
        )
        assert r_slots.dtype == torch.int32

    # 3. dsv4/dspark.py _insert_context_kv with int32 and int64 positions
    for dt in (torch.int32, torch.int64):
        captured_ops = []
        sw_cache = torch.zeros(2, 4, 8)
        attn = SimpleNamespace(
            _get_active_sw_cache=lambda c=sw_cache: c,
            custom_prefix="swa",
            kv_rope=lambda kv, p: kv,
            mla_attn=SimpleNamespace(n_local_heads=2, head_dim=4),
            compress_ratio=1,
            attn_sink=torch.zeros(2),
            attn_op=lambda *ops, cap=captured_ops: (
                cap.append(ops),
                (None, torch.ones(2, 4, 8)),
            )[1],
        )
        meta_swa = SimpleNamespace(
            seq_lens=torch.tensor([2], dtype=torch.int32),
            block_tables=torch.zeros(1, 2, dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
        )
        dsv4_dspark._insert_context_kv(
            attn, torch.zeros(2, 4), torch.tensor([1, 3], dtype=dt), meta_swa
        )
        assert len(captured_ops) == 1
        assert captured_ops[0][9].dtype == torch.int32
        assert torch.equal(captured_ops[0][9], torch.tensor([2, 4], dtype=torch.int32))

    # 4. dsv4/attention.py _run_mqa with int32 and int64 positions (is_csa=False)
    for dt in (torch.int32, torch.int64):
        captured_mqa = []
        sw_cache = torch.zeros(2, 4, 8)
        swa_layer = SimpleNamespace(
            kv_cache=sw_cache,
            prefix="swa",
        )
        dummy_attn = SimpleNamespace(
            _get_active_sw_cache=lambda c=sw_cache: c,
            attention_kind="swa",
            swa_cache_layer=swa_layer,
            main_cache_layer=swa_layer,
            compress_ratio=2,
            attn_sink=torch.zeros(2),
            attn_op=lambda *ops, cap=captured_mqa: (
                cap.append(ops),
                (torch.ones(2, 2, 4), torch.ones(2, 4, 8)),
            )[1],
        )
        meta_swa = SimpleNamespace(
            seq_lens=torch.tensor([2], dtype=torch.int32),
            block_tables=torch.zeros(1, 2, dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
        )
        monkeypatch.setattr(
            dsv4_attn,
            "get_forward_context",
            lambda meta=meta_swa: SimpleNamespace(attn_metadata={"swa": meta}),
        )
        out_mqa = dsv4_attn.DeepseekTPUAttentionMixin._run_mqa(
            dummy_attn,
            torch.zeros(2, 2, 4),
            torch.zeros(2, 4),
            torch.tensor([1, 3], dtype=dt),
            torch.zeros(2, 2, 4),
            topk_indices=None,
        )
        assert out_mqa.shape == (2, 2, 4)
        assert captured_mqa[0][9].dtype == torch.int32
        assert torch.equal(captured_mqa[0][9], torch.tensor([1, 2], dtype=torch.int32))
