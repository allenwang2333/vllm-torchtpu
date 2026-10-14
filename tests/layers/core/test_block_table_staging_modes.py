# SPDX-License-Identifier: Apache-2.0
"""The block-table staging knobs must stage the same bytes as the default."""

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.layers.core import attention_metadata
from vllm_torchtpu.layers.core.attention_metadata import stage_block_table_uploads

NUM_GROUPS = 4
MAX_REQS = 8
WIDTHS = (6, 3, 6, 2)


def _runner(tables):
    runner = MagicMock()
    runner.device = torch.device("cpu")
    runner._block_table_stage_cpu = None
    runner._block_table_stage_ring = []
    runner._block_table_stage_next = 0
    runner._unified_kv_layout = False
    groups = []
    for gid in range(NUM_GROUPS):
        bt = MagicMock()
        bt.get_cpu_tensor.return_value = tables[gid]
        builder = MagicMock()
        builder.kv_cache_group_id = gid
        builder.is_mamba_group = False
        builder._block_table_geometry.side_effect = lambda ctx, bt=bt, w=WIDTHS[gid]: (
            bt,
            w,
        )
        groups.append(SimpleNamespace(metadata_builders=[builder]))
    runner.attn_groups = [groups]
    return runner


def _ctx(num_reqs, start, dummy):
    return SimpleNamespace(
        num_reqs=num_reqs,
        start_index=start,
        position_ids_override=torch.zeros(1) if dummy else None,
        mamba_state_indices=None,
        staged_block_tables=None,
        staged_mamba_products=None,
        seq_lens=torch.ones(MAX_REQS, dtype=torch.int32),
        mamba_ckpt_window=1,
    )


def _staged(runner, ctx, padded, gids):
    stage_block_table_uploads(runner, ctx, padded, gids=gids)
    return {g: t.clone() for g, t in ctx.staged_block_tables.items()}


@pytest.mark.parametrize("ring", [2, 3])
def test_sparse_zero_stages_identical_tables(monkeypatch, ring):
    # A device upload is a copy; on CPU `.to()` would alias the host scratch.
    original_to = torch.Tensor.to

    def copying_to(self, *args, **kwargs):
        out = original_to(self, *args, **kwargs)
        return out.clone() if out.data_ptr() == self.data_ptr() else out

    monkeypatch.setattr(torch.Tensor, "to", copying_to)
    rng = random.Random(1)
    tables = [torch.randint(1, 1000, (MAX_REQS, w), dtype=torch.int32) for w in WIDTHS]
    ref_runner, new_runner = _runner(tables), _runner(tables)
    for step in range(400):
        if rng.random() < 0.3:  # rows change (append / reorder / new request)
            g = rng.randrange(NUM_GROUPS)
            tables[g][rng.randrange(MAX_REQS)] = torch.randint(
                1, 1000, (WIDTHS[g],), dtype=torch.int32
            )
        padded = rng.choice([4, 8])
        num_reqs = rng.randint(1, padded)
        start = rng.randint(0, MAX_REQS - num_reqs)
        dummy = rng.random() < 0.1
        gids = None if rng.random() < 0.7 else {rng.randrange(NUM_GROUPS), 3}
        monkeypatch.setattr(
            attention_metadata.envs, "TPU_BLOCK_TABLE_STAGE_SPARSE_ZERO", False
        )
        monkeypatch.setattr(attention_metadata.envs, "TPU_BLOCK_TABLE_STAGE_RING", 1)
        ref = _staged(ref_runner, _ctx(num_reqs, start, dummy), padded, gids)
        monkeypatch.setattr(
            attention_metadata.envs, "TPU_BLOCK_TABLE_STAGE_SPARSE_ZERO", True
        )
        monkeypatch.setattr(attention_metadata.envs, "TPU_BLOCK_TABLE_STAGE_RING", ring)
        new = _staged(new_runner, _ctx(num_reqs, start, dummy), padded, gids)
        assert ref.keys() == new.keys()
        for g in ref:
            assert torch.equal(ref[g], new[g]), (step, g)


def test_sparse_zero_requires_a_ring(monkeypatch):
    tables = [torch.ones((MAX_REQS, w), dtype=torch.int32) for w in WIDTHS]
    monkeypatch.setattr(
        attention_metadata.envs, "TPU_BLOCK_TABLE_STAGE_SPARSE_ZERO", True
    )
    monkeypatch.setattr(attention_metadata.envs, "TPU_BLOCK_TABLE_STAGE_RING", 1)
    with pytest.raises(ValueError, match="RING"):
        stage_block_table_uploads(_runner(tables), _ctx(2, 0, False), 4)
