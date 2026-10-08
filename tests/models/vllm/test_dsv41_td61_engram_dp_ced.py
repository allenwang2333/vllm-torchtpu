# SPDX-License-Identifier: Apache-2.0
"""TD6.1 audit: Engram DP=2 head sharding / token exchange parity on TPU.

Exercises the DP branches of
``vllm_torchtpu.models.vllm.deepseek_v41.common.engram`` end to end for every
DP rank in every staging mode (HBM, host-raw, host_dma) against a numpy
reference, the TP head order after the DP exchange, the HBM-mode staged-slice
tracking, and the ``dp_shared_memory`` guard in ``v2.model``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common_engram
from vllm_torchtpu.models.vllm.deepseek_v41.common.engram import (
    Engram,
    ParallelEngramEmbedding,
    _dp_gather_engram_rows,
    _embed_dp_size,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import DeepseekV41ForCausalLM


@pytest.fixture(autouse=True)
def setup_torch_accelerator(monkeypatch: pytest.MonkeyPatch):
    """Enable CPU execution for upstream Engram constructor checks."""
    monkeypatch.setattr(
        torch.accelerator, "current_device_index", lambda: 0, raising=False
    )


class _FakeDpGroup:
    """A node-local Engram DP group whose all_gather concatenates rank inputs.

    ``inputs`` holds every rank's tensor for the collective in rank order;
    the rank under test contributes its own live tensor so the gather sees
    exactly what the caller staged.
    """

    def __init__(self, world_size: int, rank: int, inputs=None):
        self.world_size = world_size
        self.rank_in_group = rank
        self.inputs = inputs
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def all_gather(self, tensor, dim):
        self.calls.append((tuple(tensor.shape), dim))
        parts = (
            list(self.inputs)
            if self.inputs is not None
            else [tensor] * (self.world_size)
        )
        # Peers hand the collective the same 128-lane padded width.
        parts = [
            torch.nn.functional.pad(p, (0, tensor.shape[-1] - p.shape[-1]))
            for p in parts
        ]
        parts[self.rank_in_group] = tensor
        return torch.cat(parts, dim=dim)


# ==============================================================================
# 1. ParallelEngramEmbedding: DP head sharding at DP2 x TP4
# ==============================================================================


def _patch_parallel(monkeypatch, *, tp_size, tp_rank, dp_size, dp_rank):
    monkeypatch.setattr(
        common_engram.upstream, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(
        common_engram.upstream, "get_tensor_model_parallel_rank", lambda: tp_rank
    )
    monkeypatch.setattr(common_engram, "get_engram_dp_size", lambda: dp_size)
    group = _FakeDpGroup(dp_size, dp_rank) if dp_size > 1 else None
    monkeypatch.setattr(common_engram, "get_engram_dp_group", lambda: group)
    # `engram_head_shard_rank` lives in the NVIDIA module and reads its own
    # imports of the TP rank and the engram DP group.
    nvidia = pytest.importorskip("vllm.models.deepseek_v41.nvidia.engram")
    monkeypatch.setattr(nvidia, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(nvidia, "get_engram_dp_group", lambda: group)
    return group


@pytest.mark.parametrize("tp_rank", [0, 3])
@pytest.mark.parametrize("dp_rank", [0, 1])
def test_shard_info_dp2_tp4_is_tp_major(
    monkeypatch: pytest.MonkeyPatch, tp_rank: int, dp_rank: int
):
    """24 heads at DP2 x TP4 -> 8 shards of 3 heads, shard = tp*dp + dp_rank."""
    _patch_parallel(monkeypatch, tp_size=4, tp_rank=tp_rank, dp_size=2, dp_rank=dp_rank)
    head_sizes = tuple(range(1, 25))  # distinct sizes -> distinct vocab slices
    emb = ParallelEngramEmbedding(
        sum(head_sizes), 64, head_sizes, block_size=16, cpu_offload=True
    )
    assert emb.dp_size == 2
    shard = tp_rank * 2 + dp_rank
    assert emb._get_shard_info() == (8, shard)
    assert emb.part_n_hash_cols == 3
    assert emb.head_start == shard * 3
    assert emb.vocab_start_idx == sum(head_sizes[: shard * 3])
    assert emb.vocab_end_idx == sum(head_sizes[: shard * 3 + 3])
    assert emb.weight.shape == (emb.part_num_embeddings, 64)
    # The largest of the 8 shards holds heads 22..24 (host_dma chunk sizing).
    assert emb.max_shard_rows() == 22 + 23 + 24


def test_shard_info_dp1_unchanged(monkeypatch: pytest.MonkeyPatch):
    _patch_parallel(monkeypatch, tp_size=4, tp_rank=2, dp_size=1, dp_rank=0)
    emb = ParallelEngramEmbedding(64, 32, (2,) * 24, block_size=16, cpu_offload=True)
    assert emb.dp_size == 1
    assert emb._get_shard_info() == (4, 2)
    assert emb.part_n_hash_cols == 6
    assert emb.head_start == 12


def test_embed_dp_size_ignores_non_int_attrs():
    assert _embed_dp_size(SimpleNamespace()) == 1
    assert _embed_dp_size(SimpleNamespace(embed_tokens=SimpleNamespace())) == 1
    assert _embed_dp_size(SimpleNamespace(embed_tokens=SimpleNamespace(dp_size=2))) == 2
    assert _embed_dp_size(SimpleNamespace(embed_tokens=MagicMock())) == 1


# ==============================================================================
# 2. Engram staging sized for the DP group's gathered slot
# ==============================================================================


def _bare_engram(embed_tokens) -> Engram:
    engram = object.__new__(Engram)
    nn.Module.__init__(engram)
    engram.embed_tokens = embed_tokens
    return engram


@pytest.mark.parametrize("dp_size", [1, 2, 3])
def test_init_staging_scales_every_mode_by_dp(dp_size: int):
    max_tokens, cols, head_dim, block = 8, 3, 16, 8
    # HBM table
    hbm = _bare_engram(
        SimpleNamespace(cpu_offload=False, part_n_hash_cols=cols, dp_size=dp_size)
    )
    Engram._init_staging(hbm, max_tokens=max_tokens, head_dim=head_dim)
    assert hbm.staged_rows.shape == (max_tokens * dp_size, cols, head_dim)
    assert hbm._staged_raw is None and hbm._dp_staged is None
    # Host-DRAM raw gather
    raw = _bare_engram(
        SimpleNamespace(
            cpu_offload=True,
            host_dma=False,
            part_n_hash_cols=cols,
            block_size=block,
            dp_size=dp_size,
        )
    )
    Engram._init_staging(raw, max_tokens=max_tokens, head_dim=head_dim)
    assert raw._host_raw.shape == (max_tokens * dp_size, cols, head_dim)
    assert raw._host_scale.shape == (max_tokens * dp_size, cols, head_dim // block)
    words_per_tok = (cols * head_dim) // 4 + -(-(cols * (head_dim // block)) // 4)
    assert raw._staged_raw.shape == (max_tokens * dp_size * words_per_tok,)
    assert raw._active_raw.shape == (max_tokens * dp_size, words_per_tok)
    # host_dma: device staging only, like HBM
    dma = _bare_engram(
        SimpleNamespace(
            cpu_offload=True, host_dma=True, part_n_hash_cols=cols, dp_size=dp_size
        )
    )
    Engram._init_staging(dma, max_tokens=max_tokens, head_dim=head_dim)
    assert dma.staged_rows.shape == (max_tokens * dp_size, cols, head_dim)
    assert dma._staged_raw is None


# ==============================================================================
# 3. DP token-for-heads exchange vs numpy reference, every rank
# ==============================================================================


def _reference_exchange(staged_by_rank, rank, num_tokens):
    """out[t, s*H + h] = rank s's staged row for replica ``rank``'s token t.

    This is upstream's ``_engram_select_rows`` contract (rank-major gather,
    this replica's token window from each source laid out as heads).
    """
    dp = len(staged_by_rank)
    slot = staged_by_rank[0].shape[0] // dp
    parts = [
        np.asarray(staged_by_rank[s][rank * slot : rank * slot + num_tokens].float())
        for s in range(dp)
    ]
    return np.concatenate(parts, axis=1)


@pytest.mark.parametrize("dp_size,slot,num_tokens", [(2, 5, 5), (2, 5, 3), (3, 4, 2)])
@pytest.mark.parametrize("width", [256, 64])
def test_dp_gather_engram_rows_matches_reference_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
    dp_size: int,
    slot: int,
    num_tokens: int,
    width: int,
):
    heads = 3
    torch.manual_seed(0)
    staged_by_rank = [
        torch.randn(dp_size * slot, heads, width, dtype=torch.bfloat16)
        for _ in range(dp_size)
    ]
    for rank in range(dp_size):
        group = _FakeDpGroup(dp_size, rank, inputs=staged_by_rank)
        monkeypatch.setattr(common_engram, "get_engram_dp_group", lambda g=group: g)
        out = _dp_gather_engram_rows(staged_by_rank[rank], num_tokens)
        assert out.shape == (num_tokens, dp_size * heads, width)
        assert out.dtype == torch.bfloat16 and out.is_contiguous()
        np.testing.assert_array_equal(
            np.asarray(out.float()),
            _reference_exchange(staged_by_rank, rank, num_tokens),
        )
        # One token-axis gather, 128-lane padded for narrow heads.
        assert group.calls == [((dp_size * slot, heads, max(width, 128)), 0)]


def test_dp_gather_engram_rows_rejects_bad_shapes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        common_engram, "get_engram_dp_group", lambda: _FakeDpGroup(2, 0)
    )
    with pytest.raises(AssertionError):
        _dp_gather_engram_rows(torch.zeros(3, 1, 4), 1)  # not a multiple of dp
    with pytest.raises(AssertionError):
        _dp_gather_engram_rows(torch.zeros(4, 1, 4), 3)  # more tokens than slot


# ==============================================================================
# 4. Engram._ready_rows / embed under DP in every staging mode, both ranks
# ==============================================================================


def _fake_embed(*, dp_size, cpu_offload, host_dma, cols, block, head_offset):
    """A head shard whose row for hash id ``i`` (any head) is ``i + offset``."""
    embed = SimpleNamespace(
        dp_size=dp_size,
        cpu_offload=cpu_offload,
        host_dma=host_dma,
        part_n_hash_cols=cols,
        block_size=block,
        tp_size=1,
        n_hash_cols=cols * dp_size,
        _host_rows=object() if host_dma else None,
    )

    def lookup(hash_ids, out):
        values = hash_ids[:, :, None].float() + head_offset
        out.copy_(values.expand(out.shape).to(out.dtype))

    def gather_raw(hash_ids, raw_out, scale_out):
        values = (hash_ids[:, :, None].float() + head_offset).expand(raw_out.shape)
        raw_out.copy_(values.to(torch.float8_e4m3fn).view(torch.uint8))
        scale_out.fill_(common_engram._UNIT_SCALE)

    embed.lookup = lookup
    embed.gather_raw = gather_raw
    return embed


def _hash_ids(dp_size, slot, cols):
    # Gathered ids are replica-major: replica r owns rows r*slot..(r+1)*slot.
    return (
        torch.arange(dp_size * slot * cols, dtype=torch.int32)
        .view(dp_size * slot, cols)
        .mul(2)
    )


@pytest.mark.parametrize("mode", ["hbm", "host_raw", "host_dma"])
@pytest.mark.parametrize("num_tokens", [4, 2])
def test_ready_rows_dp2_all_modes_both_ranks(
    monkeypatch: pytest.MonkeyPatch, mode: str, num_tokens: int
):
    dp_size, slot, cols, head_dim, block, max_tokens = 2, 4, 3, 16, 8, 4
    # No forward context in this test: the HBM path must take the slot from
    # the staged slice, not from `_engram_dp_slot(num_tokens)`.
    monkeypatch.setattr(
        common_engram,
        "_engram_dp_slot",
        lambda default: pytest.fail("slot re-derived from forward context"),
    )
    engrams = []
    for rank in range(dp_size):
        embed = _fake_embed(
            dp_size=dp_size,
            cpu_offload=mode != "hbm",
            host_dma=mode == "host_dma",
            cols=cols,
            block=block,
            head_offset=100 * rank,
        )
        engram = _bare_engram(embed)
        Engram._init_staging(engram, max_tokens=max_tokens, head_dim=head_dim)
        engram.use_sequence_parallel = False
        engrams.append(engram)

    hashes = _hash_ids(dp_size, slot, cols)
    for engram in engrams:
        engram.prepare_embeddings(hashes)
    staged_shape = (dp_size * slot, cols, head_dim)
    if mode == "hbm":
        assert all(e._dp_staged.shape == staged_shape for e in engrams)
        staged_by_rank = [e._dp_staged for e in engrams]
    elif mode == "host_raw":
        assert all(e._active_raw.shape[0] == dp_size * slot for e in engrams)
        # Dequantized local rows for the whole gathered slot (dp_size=1 view).
        staged_by_rank = []
        for e in engrams:
            monkeypatch.setattr(e.embed_tokens, "dp_size", 1)
            staged_by_rank.append(Engram._ready_rows(e, dp_size * slot))
            monkeypatch.setattr(e.embed_tokens, "dp_size", dp_size)
    else:
        assert all(e._active_rows.shape == staged_shape for e in engrams)
        staged_by_rank = [e._active_rows for e in engrams]

    for rank, engram in enumerate(engrams):
        # host_raw exchanges the packed int32 staging words (EXT-6d); the other
        # modes exchange bf16 rows. The reference stays the dequantized rows.
        gather_inputs = (
            [e._active_raw for e in engrams] if mode == "host_raw" else staged_by_rank
        )
        group = _FakeDpGroup(dp_size, rank, inputs=gather_inputs)
        monkeypatch.setattr(common_engram, "get_engram_dp_group", lambda g=group: g)
        rows = engram._ready_rows(num_tokens)
        if mode == "host_raw":
            words_per_tok = (cols * head_dim) // 4 + -(
                -(cols * (head_dim // block)) // 4
            )
            assert group.calls == [((dp_size * slot, words_per_tok), 0)]
        assert rows.shape == (num_tokens, dp_size * cols, head_dim)
        expected = _reference_exchange(staged_by_rank, rank, num_tokens)
        np.testing.assert_array_equal(np.asarray(rows.float()), expected)
        # Replica `rank`'s token t carries shard s's heads: id + 100*s
        # (rounded through FP8 when the host gathers raw bytes).
        own_ids = hashes[rank * slot : rank * slot + num_tokens].float()
        for s in range(dp_size):
            block_rows = rows[:, s * cols : (s + 1) * cols, 0].float()
            expected_vals = own_ids + 100 * s
            if mode == "host_raw":
                expected_vals = expected_vals.to(torch.float8_e4m3fn).float()
            torch.testing.assert_close(block_rows, expected_vals)
        # `embed` (tp=1) returns the exchanged rows with all heads.
        local_hashes = hashes[rank * slot : rank * slot + num_tokens]
        embedded = engram.embed(local_hashes)
        np.testing.assert_array_equal(np.asarray(embedded.float()), expected)


def test_embed_dp2_tp2_head_order(monkeypatch: pytest.MonkeyPatch):
    """TP gather after the DP exchange yields shard order tp*dp + dp_rank."""
    dp_size, slot, cols, head_dim, num_tokens = 2, 3, 2, 16, 3
    embed = _fake_embed(
        dp_size=dp_size,
        cpu_offload=False,
        host_dma=False,
        cols=cols,
        block=8,
        head_offset=0,
    )
    embed.tp_size = 2
    embed.n_hash_cols = cols * dp_size * 2
    engram = _bare_engram(embed)
    engram.use_sequence_parallel = False
    Engram._init_staging(engram, max_tokens=slot, head_dim=head_dim)
    hashes = _hash_ids(dp_size, slot, cols)
    engram.prepare_embeddings(hashes)
    staged = engram._dp_staged
    # DP peers hold heads offset by 64 per DP rank; TP peers by 128 (all
    # values stay below 256, exact in bf16).
    dp_group = _FakeDpGroup(dp_size, 0, inputs=[staged, staged + 64])
    monkeypatch.setattr(common_engram, "get_engram_dp_group", lambda: dp_group)
    monkeypatch.setattr(
        common_engram.upstream,
        "tensor_model_parallel_all_gather",
        lambda t, dim: torch.cat([t, t + 128], dim=dim),
    )
    out = engram.embed(hashes[:num_tokens])
    assert out.shape == (num_tokens, cols * dp_size * 2, head_dim)
    base = hashes[:num_tokens].float()
    for tp_rank in range(2):
        for dp_rank in range(dp_size):
            shard = tp_rank * dp_size + dp_rank
            block_rows = out[:, shard * cols : (shard + 1) * cols, 0].float()
            torch.testing.assert_close(block_rows, base + 128 * tp_rank + 64 * dp_rank)


def test_ready_rows_hbm_dp_falls_back_to_context_slot(
    monkeypatch: pytest.MonkeyPatch,
):
    """Without a prepared slice (direct callers), the forward-context slot is used."""
    dp_size, slot, cols, head_dim = 2, 3, 2, 16
    embed = _fake_embed(
        dp_size=dp_size,
        cpu_offload=False,
        host_dma=False,
        cols=cols,
        block=8,
        head_offset=0,
    )
    engram = _bare_engram(embed)
    Engram._init_staging(engram, max_tokens=slot, head_dim=head_dim)
    hashes = _hash_ids(dp_size, slot, cols)
    embed.lookup(hashes, engram.staged_rows)  # staged without prepare_embeddings
    assert engram._dp_staged is None
    monkeypatch.setattr(common_engram, "_engram_dp_slot", lambda default: slot)
    # This engram is DP rank 1; its peer (rank 0) holds heads offset by 64.
    staged_by_rank = [engram.staged_rows + 64, engram.staged_rows]
    group = _FakeDpGroup(dp_size, 1, inputs=staged_by_rank)
    monkeypatch.setattr(common_engram, "get_engram_dp_group", lambda: group)
    rows = engram._ready_rows(2)
    np.testing.assert_array_equal(
        np.asarray(rows.float()), _reference_exchange(staged_by_rank, 1, 2)
    )


def test_ready_rows_dp1_keeps_local_path(monkeypatch: pytest.MonkeyPatch):
    embed = _fake_embed(
        dp_size=1, cpu_offload=False, host_dma=False, cols=2, block=8, head_offset=7
    )
    engram = _bare_engram(embed)
    Engram._init_staging(engram, max_tokens=4, head_dim=16)
    monkeypatch.setattr(
        common_engram, "get_engram_dp_group", lambda: pytest.fail("DP path used")
    )
    hashes = torch.arange(6, dtype=torch.int32).view(3, 2)
    engram.prepare_embeddings(hashes)
    assert engram._dp_staged is None
    rows = engram._ready_rows(3)
    torch.testing.assert_close(rows[:, :, 0].float(), hashes.float() + 7)


# ==============================================================================
# 5. dp_shared_memory guard
# ==============================================================================


def test_causal_lm_rejects_engram_dp_shared_memory():
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, data_parallel_size=2, enable_expert_parallel=True
        ),
        engram_config=SimpleNamespace(dp_shared_memory=True),
    )
    with pytest.raises(NotImplementedError, match="dp_shared_memory"):
        DeepseekV41ForCausalLM(vllm_config=cfg)
