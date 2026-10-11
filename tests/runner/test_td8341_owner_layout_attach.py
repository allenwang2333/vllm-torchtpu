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
