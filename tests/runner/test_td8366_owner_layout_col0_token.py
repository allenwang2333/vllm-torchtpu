# SPDX-License-Identifier: Apache-2.0
"""TD8.366: rewrite-counter proof that block-table column 0 is unchanged.

Tracked block tables bump ``_bt_rewrites`` on every row mutation but
``append_row``, which writes from column ``num_blocks_per_row[row]`` on. While
the counter matches the snapshot and every snapshot row held a block,
``step_owner_required_tokens`` and ``attach_step_owner_token_layout`` skip the
strided column-0 ``np.array_equal``; an owning read-only scheduled array that
is the cached object skips its compare too.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from vllm_torchtpu.core.owner_affine_block_pool import OwnerLayout
from vllm_torchtpu.runner import owner_token_layout as otl


def _runner(*, tracked=True, nblocks=(1, 1, 1, 1)):
    layout = OwnerLayout(num_replicated=4, num_owners=2, blocks_per_owner=10)
    bt = np.zeros((8, 4), dtype=np.int32)
    bt[:4, 0] = [4, 15, 5, 16]
    table = SimpleNamespace(
        _bt_cpu_np=bt, num_blocks_per_row=np.array([*nblocks, 0, 0, 0, 0])
    )
    if tracked:
        table._bt_rewrites = 0
    ctx = SimpleNamespace(
        num_reqs=4,
        position_ids_override=None,
        query_start_loc_cpu=(0, 1, 2, 3, 4),
        req_indices=None,
        start_index=0,
    )
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        _attn_metadata_builder_ctx=ctx,
        num_tokens_padded=8,
        num_tokens_paddings=[8, 16],
        _kv_shard_owner_layout=layout,
        input_batch=SimpleNamespace(block_table=SimpleNamespace(block_tables=[table])),
        kv_cache_config=object(),
        vllm_config=object(),
    )
    return runner, table


def _read_only(values):
    arr = np.array(values, dtype=np.int32)
    arr.flags.writeable = False
    return arr


@pytest.fixture(autouse=True)
def _kv_shard():
    with (
        mock.patch.object(otl.envs, "TPU_DEEPSEEK_V41_KV_SHARD", True),
        mock.patch.object(otl, "_sharded_group_id", lambda r: 0),
    ):
        yield


def _no_array_equal():
    return mock.patch.object(otl.np, "array_equal", side_effect=AssertionError)


def test_required_tokens_counter_hit_skips_both_compares():
    runner, table = _runner()
    sched = _read_only([1, 2, 1, 1])  # owner 0: rows 0, 2; owner 1: rows 1, 3
    assert otl.step_owner_required_tokens(runner, 0, 4, sched) == 6
    assert runner._kv_shard_required_tokens_cache[4] == (table, 0, 0, 4)
    with _no_array_equal():
        assert otl.step_owner_required_tokens(runner, 0, 4, sched) == 6


def test_required_tokens_rewrite_bump_recomputes():
    runner, table = _runner()
    sched = _read_only([1, 2, 1, 1])
    assert otl.step_owner_required_tokens(runner, 0, 4, sched) == 6
    table._bt_cpu_np[1, 0] = 5  # r1 moves to owner 0: 4 tokens vs 1
    table._bt_rewrites += 1
    assert otl.step_owner_required_tokens(runner, 0, 4, sched) == 8


def test_required_tokens_empty_row_snapshot_falls_back_to_compare():
    # r3 held no block at snapshot time, so append_row may write its column 0
    # without a rewrite bump; the cached token must not be trusted.
    runner, table = _runner(nblocks=(1, 1, 1, 0))
    sched = _read_only([1, 1, 1, 3])
    assert otl.step_owner_required_tokens(runner, 0, 4, sched) == 8
    assert runner._kv_shard_required_tokens_cache[4] is None
    table._bt_cpu_np[3, 0] = 6  # append into column 0 (owner 1 -> 0): 5 vs 1
    assert otl.step_owner_required_tokens(runner, 0, 4, sched) == 10


def test_required_tokens_untracked_table_compares():
    runner, table = _runner(tracked=False)
    sched = _read_only([1, 1, 1, 1])
    assert otl.step_owner_required_tokens(runner, 0, 4, sched) == 4
    assert runner._kv_shard_required_tokens_cache[4] is None
    table._bt_cpu_np[:4, 0] = [4, 5, 6, 7]  # all owner 0
    assert otl.step_owner_required_tokens(runner, 0, 4, sched) == 8


def test_required_tokens_writable_same_array_is_compared():
    runner, _ = _runner()
    sched = np.array([1, 1, 1, 1], dtype=np.int32)
    assert otl.step_owner_required_tokens(runner, 0, 4, sched) == 4
    sched[0] = 3  # mutated in place: identity alone must not hit
    assert otl.step_owner_required_tokens(runner, 0, 4, sched) == 8


def _per_layer():
    meta = SimpleNamespace(query_start_loc=torch.zeros(5, dtype=torch.int32))
    return {"a": meta, "b": meta}


def test_attach_counter_hit_skips_first_block_compare():
    runner, table = _runner()
    layout = otl.attach_step_owner_token_layout(runner, _per_layer())
    assert layout is not None
    assert runner._kv_shard_step_layout_cache[7] == (table, 0, 0, 4)
    per_layer = _per_layer()
    with _no_array_equal():
        assert otl.attach_step_owner_token_layout(runner, per_layer) is layout
    assert per_layer["a"].owner_token_layout is layout

    table._bt_cpu_np[1, 0] = 5
    table._bt_rewrites += 1
    assert otl.attach_step_owner_token_layout(runner, _per_layer()) is not layout


def test_attach_explicit_req_indices_compare_first_blocks():
    runner, table = _runner()
    runner._attn_metadata_builder_ctx.req_indices = np.arange(4)
    layout = otl.attach_step_owner_token_layout(runner, _per_layer())
    assert runner._kv_shard_step_layout_cache[7] is None
    assert otl.attach_step_owner_token_layout(runner, _per_layer()) is layout
    table._bt_cpu_np[1, 0] = 5  # unchanged counter, but rows are explicit
    assert otl.attach_step_owner_token_layout(runner, _per_layer()) is not layout
