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
"""EXT-6d: DP token-for-heads exchange on packed int32 staging words.

Under ``OFFLOAD=1`` with ``dp_size > 1``, ``Engram._ready_rows`` used to
dequantize every staged token (``dp * slot``) to bf16 and all-gather those
rows. It now all-gathers the packed FP8 + E8M0 words and dequantizes only this
replica's token window. These tests pin bit-exactness against the old order
of operations on every rank, for even and word-straddling scale layouts,
partial windows, and the no-group fallback.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common_engram
from vllm_torchtpu.models.vllm.deepseek_v41.common.engram import (
    Engram,
    _dp_gather_engram_rows,
    _dp_gather_packed_rows,
)


class _FakeDpGroup:
    def __init__(self, world_size, rank, inputs=None):
        self.world_size = world_size
        self.rank_in_group = rank
        self.inputs = inputs
        self.calls: list[tuple[tuple[int, ...], torch.dtype, int]] = []

    def all_gather(self, tensor, dim):
        self.calls.append((tuple(tensor.shape), tensor.dtype, dim))
        parts = list(self.inputs) if self.inputs is not None else [tensor] * 2
        # Peers hand the collective the same (128-lane padded) width.
        parts = [
            torch.nn.functional.pad(p, (0, tensor.shape[-1] - p.shape[-1]))
            for p in parts
        ]
        parts[self.rank_in_group] = tensor
        return torch.cat(parts, dim=dim)


def _fake_embed(*, dp_size, cols, block, seed):
    """Host shard whose raw bytes / scales are pseudo-random per (rank, id)."""
    embed = SimpleNamespace(
        dp_size=dp_size,
        cpu_offload=True,
        host_dma=False,
        part_n_hash_cols=cols,
        block_size=block,
        tp_size=1,
        n_hash_cols=cols * dp_size,
        _host_rows=None,
    )

    def gather_raw(hash_ids, raw_out, scale_out):
        gen = torch.Generator().manual_seed(seed + int(hash_ids[0, 0]))
        raw_out.copy_(
            torch.randint(0, 256, raw_out.shape, generator=gen).to(torch.uint8)
        )
        # Mix of ordinary scales, byte 0 (-> 0) and the invalid sentinel.
        choices = torch.tensor([0, 100, 120, 127, 130, 140, 255], dtype=torch.uint8)
        pick = torch.randint(0, len(choices), scale_out.shape, generator=gen)
        scale_out.copy_(choices[pick])

    embed.gather_raw = gather_raw
    return embed


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-pattern equality (random FP8 bytes include NaN, so `torch.equal`
    on the bf16 values would be false even for identical tensors)."""
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
    )


def _bare_engram(embed) -> Engram:
    engram = object.__new__(Engram)
    nn.Module.__init__(engram)
    engram.embed_tokens = embed
    engram.use_sequence_parallel = False
    return engram


def _staged_engrams(dp_size, slot, cols, head_dim, block, max_tokens):
    engrams = []
    for rank in range(dp_size):
        e = _bare_engram(
            _fake_embed(dp_size=dp_size, cols=cols, block=block, seed=rank)
        )
        Engram._init_staging(e, max_tokens=max_tokens, head_dim=head_dim)
        engrams.append(e)
    hashes = torch.arange(dp_size * slot * cols, dtype=torch.int32).view(-1, cols)
    for e in engrams:
        e.prepare_embeddings(hashes)
    return engrams


def _trunk_rows(engram, dp_size, monkeypatch):
    """Old order of operations: dequantize every staged token (dp_size=1
    view), then exchange bf16 rows."""
    monkeypatch.setattr(engram.embed_tokens, "dp_size", 1)
    rows = Engram._ready_rows(engram, engram._active_raw.shape[0])
    monkeypatch.setattr(engram.embed_tokens, "dp_size", dp_size)
    return rows


@pytest.mark.parametrize(
    "dp_size,slot,num_tokens",
    [(2, 6, 6), (2, 6, 4), (2, 6, 1), (3, 4, 3), (4, 3, 2), (2, 2, 0)],
)
@pytest.mark.parametrize(
    "cols,head_dim,block",
    [
        (6, 256, 32),  # DP2 x TP4 production: 48 scale bytes/token, word aligned
        (3, 256, 32),  # TP8-style shard: 24 scale bytes/token
        (3, 16, 16),  # 3 scale bytes/token: tokens straddle int32 words
        (1, 64, 8),  # 8 scale bytes/token, single head
    ],
)
def test_packed_dp_gather_bit_exact_vs_dequant_then_gather(
    monkeypatch: pytest.MonkeyPatch, dp_size, slot, num_tokens, cols, head_dim, block
):
    engrams = _staged_engrams(dp_size, slot, cols, head_dim, block, max_tokens=slot)
    blocks = head_dim // block
    words_per_tok = (cols * head_dim) // 4 + -(-(cols * blocks) // 4)
    dequant_by_rank = [_trunk_rows(e, dp_size, monkeypatch) for e in engrams]
    packed_by_rank = [e._active_raw for e in engrams]

    for rank, engram in enumerate(engrams):
        # Reference: trunk's bf16 exchange of fully dequantized rows.
        ref_group = _FakeDpGroup(dp_size, rank, inputs=dequant_by_rank)
        monkeypatch.setattr(common_engram, "get_engram_dp_group", lambda g=ref_group: g)
        expected = _dp_gather_engram_rows(dequant_by_rank[rank], num_tokens)

        group = _FakeDpGroup(dp_size, rank, inputs=packed_by_rank)
        monkeypatch.setattr(common_engram, "get_engram_dp_group", lambda g=group: g)
        rows = engram._ready_rows(num_tokens)

        assert rows.shape == (num_tokens, dp_size * cols, head_dim)
        assert rows.dtype == torch.bfloat16 and rows.is_contiguous()
        assert _bits_equal(rows, expected)
        # Exactly one collective, on the packed int32 words, no padding.
        assert group.calls == [((dp_size * slot, words_per_tok), torch.int32, 0)]


def test_packed_dp_gather_moves_fewer_bytes_than_bf16_rows():
    cols, head_dim, block = 6, 256, 32
    packed = (cols * head_dim) // 4 * 4 + -(-(cols * (head_dim // block)) // 4) * 4
    bf16 = cols * head_dim * 2
    assert packed == 1584 and bf16 == 3072
    assert bf16 / packed > 1.9


def test_packed_dp_gather_without_group_keeps_single_replica_path(
    monkeypatch: pytest.MonkeyPatch,
):
    """dp_size=2 on the embed but no DP group (e.g. dp_shared_memory tests):
    `_dp_gather_packed_rows` returns None and `_ready_rows` falls back to
    dequantizing the staged rows and slicing, as before."""
    dp_size, slot, cols, head_dim, block = 2, 3, 2, 16, 8
    (engram,) = _staged_engrams(1, dp_size * slot, cols, head_dim, block, 6)
    monkeypatch.setattr(engram.embed_tokens, "dp_size", dp_size)
    monkeypatch.setattr(common_engram, "get_engram_dp_group", lambda: None)
    assert (
        _dp_gather_packed_rows(
            engram._active_raw, 2, cols=cols, head_dim=head_dim, blocks=2, block_size=8
        )
        is None
    )
    rows = engram._ready_rows(2)
    full = _trunk_rows(engram, dp_size, monkeypatch)
    assert _bits_equal(rows, full[:2])

    size_one = _FakeDpGroup(1, 0)
    monkeypatch.setattr(common_engram, "get_engram_dp_group", lambda: size_one)
    assert _bits_equal(engram._ready_rows(2), full[:2])
    assert size_one.calls == []


def test_packed_dp_gather_rejects_bad_shapes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        common_engram, "get_engram_dp_group", lambda: _FakeDpGroup(2, 0)
    )
    kw = dict(cols=1, head_dim=16, blocks=2, block_size=8)
    with pytest.raises(AssertionError):
        _dp_gather_packed_rows(torch.zeros(3, 5, dtype=torch.int32), 1, **kw)
    with pytest.raises(AssertionError):
        _dp_gather_packed_rows(torch.zeros(4, 5, dtype=torch.int32), 3, **kw)
