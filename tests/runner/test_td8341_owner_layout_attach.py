# SPDX-License-Identifier: Apache-2.0
"""TD8.341: ``attach_step_owner_token_layout`` assigns owner fields per layer.

The id()-keyed dedup of the per-layer metadata dict was replaced by direct
(idempotent) assignment on every value: CPU timeit on the steady-state cache
hit (192 requests, 83 layers over 51 shared metadata objects) 34.0 -> 14.5 us.
Every metadata object must still carry the step's layout tensors, on both the
cache-miss and cache-hit paths and for the positional layout.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from vllm_torchtpu.core.owner_affine_block_pool import OwnerLayout
from vllm_torchtpu.runner import owner_token_layout as otl


def _runner(num_reqs=6, positional=False):
    layout = OwnerLayout(num_replicated=4, num_owners=2, blocks_per_owner=10)
    bt = np.zeros((8, 4), dtype=np.int32)
    bt[:num_reqs, 0] = [4, 15, 5, 16, 6, 17][:num_reqs]
    ctx = SimpleNamespace(
        num_reqs=num_reqs,
        position_ids_override=object() if positional else None,
        query_start_loc_cpu=np.arange(num_reqs + 1, dtype=np.int32),
        req_indices=None,
        start_index=0,
    )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        _attn_metadata_builder_ctx=ctx,
        num_tokens_padded=8,
        num_tokens_paddings=[8, 16],
        _kv_shard_owner_layout=layout,
        input_batch=SimpleNamespace(
            block_table=SimpleNamespace(block_tables=[SimpleNamespace(_bt_cpu_np=bt)])
        ),
        kv_cache_config=object(),
        vllm_config=object(),
    )


def _per_layer(num_reqs=6):
    a, b, c = (
        SimpleNamespace(query_start_loc=torch.zeros(num_reqs + 1, dtype=torch.int32))
        for _ in range(3)
    )
    # Shared objects appear non-contiguously, as when a later group overrides.
    return {"l0": a, "l1": a, "l2": b, "l3": c, "l4": a, "l5": b}, (a, b, c)


def _check(metas, layout):
    for m in metas:
        assert m.owner_token_layout is layout
        assert m.owner_perm is metas[0].owner_perm
        assert m.owner_inv is metas[0].owner_inv
        assert m.owner_q_loc is metas[0].owner_q_loc


def test_attach_sets_fields_on_every_metadata_miss_and_hit():
    runner = _runner()
    with (
        mock.patch.object(otl.envs, "TPU_DEEPSEEK_V41_KV_SHARD", True),
        mock.patch.object(otl, "_sharded_group_id", lambda r: 0),
    ):
        per_layer, metas = _per_layer()
        layout = otl.attach_step_owner_token_layout(runner, per_layer)
        assert layout is not None
        _check(metas, layout)
        # Next step: new metadata objects, same layout key -> cache hit.
        per_layer2, metas2 = _per_layer()
        assert otl.attach_step_owner_token_layout(runner, per_layer2) is layout
        _check(metas2, layout)
        assert metas2[0].owner_perm is metas[0].owner_perm

        # Third step: padded_tokens changes -> step_key misses, but owners_cache hits.
        runner.num_tokens_padded = 16
        per_layer3, metas3 = _per_layer()
        layout3 = otl.attach_step_owner_token_layout(runner, per_layer3)
        assert layout3 is not None
        _check(metas3, layout3)


def test_attach_positional_layout_sets_every_metadata():
    runner = _runner(positional=True)
    with (
        mock.patch.object(otl.envs, "TPU_DEEPSEEK_V41_KV_SHARD", True),
        mock.patch.object(otl, "_sharded_group_id", lambda r: 0),
    ):
        per_layer, metas = _per_layer()
        assert otl.attach_step_owner_token_layout(runner, per_layer) is None
        for m in metas:
            assert m.owner_perm is metas[0].owner_perm is not None
            assert m.owner_q_loc is metas[0].owner_q_loc


def test_td8348_required_tokens_and_step_layout_tuple_and_array_equal_caches():
    runner = _runner(num_reqs=4)
    bt = runner.input_batch.block_table.block_tables[0]._bt_cpu_np
    sched = np.array([1, 1, 1, 1], dtype=np.int32)
    with (
        mock.patch.object(otl.envs, "TPU_DEEPSEEK_V41_KV_SHARD", True),
        mock.patch.object(otl, "_sharded_group_id", lambda r: 0),
    ):
        # 1. step_owner_required_tokens caches copied arrays & _kv_shard_owners_cache
        req1 = otl.step_owner_required_tokens(runner, 0, 4, sched)
        assert req1 == 4
        assert len(runner._kv_shard_required_tokens_cache) == 6
        assert isinstance(runner._kv_shard_required_tokens_cache[1], np.ndarray)
        assert isinstance(runner._kv_shard_required_tokens_cache[2], np.ndarray)
        assert isinstance(runner._kv_shard_owners_cache[1], np.ndarray)
        assert otl.step_owner_required_tokens(runner, 0, 4, sched) == 4

        # 2. attach_step_owner_token_layout with tuple reuses _kv_shard_owners_cache
        q_tuple = (0, 1, 2, 3, 4)
        runner._attn_metadata_builder_ctx.query_start_loc_cpu = q_tuple
        per_layer1, _ = _per_layer(num_reqs=4)
        with mock.patch.object(
            otl, "_owners_from_first_blocks", wraps=otl._owners_from_first_blocks
        ) as spy_owners:
            layout1 = otl.attach_step_owner_token_layout(runner, per_layer1)
            assert layout1 is not None
            assert spy_owners.call_count == 0
        assert runner._kv_shard_step_layout_cache[2] is q_tuple

        # 3. Cache hit with padded tuple slices to q_tuple without build
        runner._attn_metadata_builder_ctx.query_start_loc_cpu = (0, 1, 2, 3, 4, 4, 4)
        per_layer2, _ = _per_layer(num_reqs=4)
        with mock.patch.object(
            otl, "build_owner_token_layout", wraps=otl.build_owner_token_layout
        ) as spy_build:
            layout2 = otl.attach_step_owner_token_layout(runner, per_layer2)
            assert layout2 is layout1
            assert spy_build.call_count == 0

        # 4. Out-of-range first_blocks caches 0 and hits cache on repeat
        bt[0, 0] = 0  # < num_replicated (4)
        assert otl.step_owner_required_tokens(runner, 0, 4, sched) == 0
        assert runner._kv_shard_required_tokens_cache[3] == 0
        assert otl.step_owner_required_tokens(runner, 0, 4, sched) == 0
