# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft embedding dedupe: only base ids + the noise id are looked up."""

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2.dspark import _embed_draft_blocks

MASK = 7


def _layout(bases: list[int], block: int, tokens: int) -> torch.Tensor:
    ids = torch.full((tokens,), MASK, dtype=torch.int32)
    ids[: len(bases) * block : block] = torch.tensor(bases, dtype=torch.int32)
    return ids


class _Embed:
    def __init__(self, vocab: int = 32, hidden: int = 16) -> None:
        self.weight = torch.randn(
            vocab, hidden, generator=torch.Generator().manual_seed(0)
        )
        self.calls: list[torch.Tensor] = []

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        self.calls.append(ids.clone())
        return self.weight[ids.long()]


@pytest.mark.parametrize(
    ("bases", "block", "tokens"),
    [
        ([3, 9, 11], 5, 15),  # exact blocks
        ([3, 9, 11], 5, 18),  # 3-row noise tail
        (list(range(57)), 5, 288),  # C=48 bucket layout
        ([4, 4, 4], 6, 24),  # bonus-anchor layout, repeated bases
    ],
)
def test_embed_draft_blocks_bitwise(bases, block, tokens):
    embed = _Embed()
    ids = _layout([b % 32 for b in bases], block, tokens)
    out = _embed_draft_blocks(embed, ids, block)
    ref = embed.weight[ids.long()]
    assert out.shape == ref.shape
    assert torch.equal(out, ref)
    # One lookup of the base ids plus the single noise id.
    assert len(embed.calls) == 1
    assert embed.calls[0].shape == (tokens // block + 1,)


@pytest.mark.parametrize(("block", "tokens"), [(1, 8), (5, 4)])
def test_embed_draft_blocks_falls_back_to_full_lookup(block, tokens):
    embed = _Embed()
    ids = torch.arange(tokens, dtype=torch.int32)
    out = _embed_draft_blocks(embed, ids, block)
    assert torch.equal(out, embed.weight[ids.long()])
    assert torch.equal(embed.calls[0], ids)


@pytest.mark.parametrize("bad_row", [2, 6, 16])
def test_embed_draft_blocks_poisons_layout_violations(bad_row):
    embed = _Embed()
    ids = _layout([3, 9, 11], 5, 18)
    ids[bad_row] = 1
    out = _embed_draft_blocks(embed, ids, 5)
    bad = torch.isnan(out).all(-1)
    assert bad.tolist() == [i == bad_row for i in range(18)]
    keep = ~bad
    assert torch.equal(out[keep], embed.weight[ids.long()][keep])
