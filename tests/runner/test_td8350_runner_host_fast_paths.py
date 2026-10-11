# SPDX-License-Identifier: Apache-2.0
"""TD8.350: runner host fast paths.

(1) `_build_attention_metadata` builds staged non-Mamba metadata once per
    kv_cache_group_id and shares the object across that group's layers.
(2) `_prepare_lookback_token_ids` skips the gather when no request has prompt
    history in its window, and skips the fill + H2D on consecutive all -1
    steps.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import vllm_torchtpu.runner.tpu_runner as tr_mod
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class _Builder:
    def __init__(self, gid, *, mamba=False):
        self.kv_cache_group_id = gid
        self.is_mamba_group = mamba
        self.calls = 0

    def build(self, common_prefix_len, common_attn_metadata):
        self.calls += 1
        return SimpleNamespace(gid=self.kv_cache_group_id, builder=self)


def _walk_runner(builders, staged):
    runner = object.__new__(TPUModelRunner)
    runner.attn_groups = [
        [
            SimpleNamespace(metadata_builders=[b], layer_names=[f"l{i}a", f"l{i}b"])
            for i, b in enumerate(builders)
        ]
    ]
    runner._attn_metadata_builder_ctx = SimpleNamespace(staged_block_tables=staged)
    return runner


def _walk(runner):
    with patch.object(tr_mod.envs, "TPU_DEEPSEEK_V41_KV_SHARD", False):
        meta, spec = TPUModelRunner._build_attention_metadata(
            runner, num_tokens=8, num_reqs=4, num_reqs_padded=4
        )
    assert spec is None
    return meta


def test_staged_walk_builds_once_per_group_id():
    b = [
        _Builder(0),
        _Builder(0),
        _Builder(1),
        _Builder(0, mamba=True),
        _Builder(0, mamba=True),
        _Builder(7),  # not staged
        _Builder(7),
        _Builder(2),  # staged entry is None
    ]
    staged = {0: torch.zeros(2), 1: torch.zeros(2), 2: None}
    meta = _walk(_walk_runner(b, staged))

    assert [x.calls for x in b] == [1, 0, 1, 1, 1, 1, 1, 1]
    assert meta["l0a"] is meta["l0b"] is meta["l1a"] is meta["l1b"]
    assert meta["l2a"] is not meta["l0a"]
    assert meta["l3a"].builder is b[3] and meta["l4a"].builder is b[4]
    assert meta["l5a"] is not meta["l6a"]
    assert meta["l7a"].builder is b[7]
    # Layer order is unchanged.
    assert list(meta) == [f"l{i}{s}" for i in range(8) for s in "ab"]


def test_unstaged_walk_builds_every_group():
    b = [_Builder(0), _Builder(0)]
    runner = _walk_runner(b, None)
    with patch.object(tr_mod, "stage_block_table_uploads") as stage:
        meta = _walk(runner)
    stage.assert_called_once()
    assert [x.calls for x in b] == [1, 1]
    assert meta["l0a"] is not meta["l1a"]


def test_shared_metadata_equals_per_builder_clone():
    """Real builders: the shared object holds exactly what the old second
    `build` (a `with_block_tables` clone of the base) returned."""
    staged = {0: torch.arange(4, dtype=torch.int32)}
    desc = SimpleNamespace(kind=SimpleNamespace(value=0), protocol=1, version=2)
    ctx = SimpleNamespace(
        staged_block_tables=staged,
        base_attn_metadata=None,
        position_ids_override=None,
        seq_lens=torch.ones(4, dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32),
        request_distribution=torch.zeros(3, dtype=torch.int32),
        sequence_layout_descriptor=desc,
        query_start_loc_cpu=(0, 1, 2, 3, 4),
        seq_lens_cpu=(1, 1, 1, 1),
        prompt_lens_cpu=(1, 1, 1, 1),
        replay_starts_cpu=None,
    )
    runner = object.__new__(TPUModelRunner)
    runner.position_ids = torch.zeros(8, dtype=torch.int32)
    runner._attn_metadata_builder_ctx = ctx

    def builder():
        bld = object.__new__(AttentionMetadataBuilder)
        bld.runner = runner
        bld.kv_cache_group_id = 0
        bld.is_mamba_group = False
        return bld

    first, second = builder(), builder()
    runner.attn_groups = [
        [
            SimpleNamespace(metadata_builders=[first], layer_names=["a"]),
            SimpleNamespace(metadata_builders=[second], layer_names=["b"]),
        ]
    ]
    meta = _walk(runner)
    assert meta["a"] is meta["b"]
    assert isinstance(meta["a"], AttentionMetadata)

    # With TD8.342, with_block_tables returns base when block_tables is identical.
    ctx.base_attn_metadata = None
    common = SimpleNamespace(num_reqs=4)
    base = first.build(common_prefix_len=0, common_attn_metadata=common)
    clone = second.build(common_prefix_len=0, common_attn_metadata=common)
    assert clone is base
    assert clone.__dict__.keys() == meta["a"].__dict__.keys()
    for k, v in clone.__dict__.items():
        got = meta["a"].__dict__[k]
        if isinstance(v, torch.Tensor):
            assert torch.equal(got, v)
        else:
            assert got == v


class _LookbackBuffer:
    def __init__(self, rows, depth):
        self.cpu = torch.zeros((rows, depth), dtype=torch.int32)
        self.np = self.cpu.numpy()
        self.gpu = torch.zeros((rows, depth), dtype=torch.int32)
        self.uploads = 0

    def copy_to_gpu(self):
        self.uploads += 1
        return self.gpu.copy_(self.cpu)


def _lookback_runner(computed, prompt, rows=4, depth=2):
    runner = object.__new__(TPUModelRunner)
    runner.lookback_token_ids = _LookbackBuffer(rows, depth)
    n = len(computed)
    runner.input_batch = SimpleNamespace(
        req_ids=[f"r{i}" for i in range(n)],
        num_computed_tokens_cpu=np.array(computed, dtype=np.int32),
        num_prompt_tokens=np.array(prompt, dtype=np.int32),
        num_tokens_no_spec=np.array(prompt, dtype=np.int32),
        token_ids_cpu=np.arange(n * 100, dtype=np.int32).reshape(n, 100),
    )
    runner._replay_starts = {}
    return runner


def _old_lookback(runner, num_reqs):
    """Pre-TD8.350 reference values."""
    buf = runner.lookback_token_ids
    depth = buf.np.shape[1]
    out = np.full(buf.np.shape, -1, dtype=np.int32)
    if num_reqs > 0:
        ib = runner.input_batch
        starts = ib.num_computed_tokens_cpu[:num_reqs, None]
        pos = starts - np.arange(1, depth + 1)
        valid = (pos >= 0) & (pos < ib.num_prompt_tokens[:num_reqs, None])
        ids = ib.token_ids_cpu[np.arange(num_reqs)[:, None], np.maximum(pos, 0)]
        out[:num_reqs] = np.where(valid, ids, -1)
    return out


def test_lookback_with_prompt_history_gathers_every_step():
    runner = _lookback_runner([3, 50], [10, 10])
    buf = runner.lookback_token_ids
    buf.np[:] = 99  # stale rows must be cleared
    for step in (1, 2):
        ids = TPUModelRunner._prepare_lookback_token_ids(runner, 2)
        assert ids is buf.gpu
        np.testing.assert_array_equal(ids.numpy(), _old_lookback(runner, 2))
        assert buf.uploads == step
    assert ids.numpy().tolist() == [[2, 1], [-1, -1], [-1, -1], [-1, -1]]
    assert runner._lookback_all_neg1_synced is False


@pytest.mark.parametrize("num_reqs", [0, 2])
def test_lookback_all_invalid_uploads_once(num_reqs):
    runner = _lookback_runner([50, 60], [10, 10])
    buf = runner.lookback_token_ids
    buf.np[:] = 99
    ids = TPUModelRunner._prepare_lookback_token_ids(runner, num_reqs)
    assert ids is buf.gpu
    assert (ids == -1).all() and (buf.np == -1).all()
    assert buf.uploads == 1
    assert runner._lookback_all_neg1_synced is True

    ids2 = TPUModelRunner._prepare_lookback_token_ids(runner, num_reqs)
    assert ids2 is buf.gpu
    assert buf.uploads == 1
    np.testing.assert_array_equal(ids2.numpy(), _old_lookback(runner, num_reqs))


def test_lookback_valid_step_resets_synced_state():
    runner = _lookback_runner([50, 60], [10, 10])
    buf = runner.lookback_token_ids
    TPUModelRunner._prepare_lookback_token_ids(runner, 2)
    assert buf.uploads == 1

    runner.input_batch.num_computed_tokens_cpu[1] = 5  # back inside its prompt
    ids = TPUModelRunner._prepare_lookback_token_ids(runner, 2)
    assert buf.uploads == 2
    assert runner._lookback_all_neg1_synced is False
    np.testing.assert_array_equal(ids.numpy(), _old_lookback(runner, 2))

    runner.input_batch.num_computed_tokens_cpu[1] = 60
    ids = TPUModelRunner._prepare_lookback_token_ids(runner, 2)
    assert buf.uploads == 3  # the -1 buffer must be re-uploaded
    assert (ids == -1).all()
