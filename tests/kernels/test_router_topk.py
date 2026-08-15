# Copyright 2026 Google LLC
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
"""Pallas router top-k kernel vs its torch reference, on the host.

``interpret=True`` runs the kernel body in JAX on CPU, so the kernel's own
logic -- block splitting, the sentinel, the tie rule -- is gated without a
TPU. What it does *not* gate is Mosaic lowering; that is the AOT compile in
``impl/w6-topk.md``.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax import lax

from vllm_torchtpu.kernels.router_topk import (MAX_BLOCK_ROWS, block_rows_for,
                                               select)
from vllm_torchtpu.layers.common.rowmax_topk import rowmax_topk

EXPERTS = 512
TOPK = 10


def _both(scores_np):
    kw, ki = select(jnp.asarray(scores_np), TOPK, interpret=True)
    rw, ri = rowmax_topk(torch.from_numpy(scores_np), TOPK)
    return (np.asarray(kw), np.asarray(ki)), (rw.numpy(), ri.numpy())


@pytest.mark.parametrize("rows", [8, 64, 256, 512, 1024, 1536])
def test_kernel_matches_torch_reference(rows):
    rng = np.random.default_rng(rows)
    scores = rng.random((rows, EXPERTS), dtype=np.float32)
    (kw, ki), (rw, ri) = _both(scores)
    np.testing.assert_array_equal(kw, rw)
    np.testing.assert_array_equal(ki, ri)


@pytest.mark.parametrize("rows", [17, 33, 250, 1000, 1040, 4384])
def test_partial_last_block_does_not_leak(rows):
    """Row counts that are not a multiple of the block stay exact.

    The final grid step reads a partial block, so the kernel computes on rows
    that do not exist. Mosaic discards the stores past ``rows``; this pins that
    the rows it keeps are untouched by them, at every row, not just in bulk.
    """
    rng = np.random.default_rng(rows)
    scores = rng.random((rows, EXPERTS), dtype=np.float32)
    (kw, ki), (rw, ri) = _both(scores)
    assert kw.shape == ki.shape == (rows, TOPK)
    np.testing.assert_array_equal(kw, rw)
    np.testing.assert_array_equal(ki, ri)
    # A leak from the surplus region shows up as an impossible expert or a
    # sentinel weight long before it shows up as a wrong-but-plausible one.
    assert ((ki >= 0) & (ki < EXPERTS)).all()
    assert np.isfinite(kw).all()


def test_kernel_matches_torch_topk_on_real_routing_distribution():
    rng = np.random.default_rng(99)
    logits = torch.from_numpy((rng.standard_normal(
        (1024, EXPERTS), dtype=np.float32) * 2.0)).to(torch.bfloat16)
    scores = logits.float().softmax(dim=-1)
    kw, ki = select(jnp.asarray(scores.numpy()), TOPK, interpret=True)
    ref_v, _ = torch.topk(scores, TOPK, dim=-1)
    np.testing.assert_array_equal(np.asarray(kw), ref_v.numpy())
    gathered = torch.gather(scores, 1, torch.from_numpy(np.asarray(ki)).long())
    np.testing.assert_array_equal(gathered.numpy(), ref_v.numpy())


def test_all_nan_row_selects_k_distinct_experts():
    """The kernel half of the same property (see the torch reference test).

    An all-NaN row is the flush sentinel in every column; masking with that
    same sentinel keeps every column tied with the row max, so every pass
    returns column 0 and the row selects one expert ``TOPK`` times. Measured
    against the sort path at 512 experts / top-k 10, that is 1 distinct expert
    where ``lax.sort`` takes 10.
    """
    scores = np.full((3, EXPERTS), np.nan, dtype=np.float32)
    kw, ki = select(jnp.asarray(scores), TOPK, interpret=True)
    ki = np.asarray(ki)
    assert np.isnan(np.asarray(kw)).all()
    for row in ki:
        assert sorted(row.tolist()) == list(range(TOPK))
    iota = lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    _, sort_i = lax.sort((-jnp.asarray(scores), iota),
                         dimension=1,
                         is_stable=False,
                         num_keys=1)
    for got, ref in zip(ki, np.asarray(sort_i)[:, :TOPK]):
        assert len(set(got.tolist())) == len(set(ref.tolist())) == TOPK


def test_partial_nan_row_selects_k_distinct_finite_experts():
    rng = np.random.default_rng(5)
    scores = rng.random((2, EXPERTS), dtype=np.float32)
    scores[:, :5] = np.nan
    kw, ki = select(jnp.asarray(scores), TOPK, interpret=True)
    assert np.isfinite(np.asarray(kw)).all()
    for row in np.asarray(ki):
        assert len(set(row.tolist())) == TOPK
        assert min(row.tolist()) >= 5


def test_kernel_ties_and_nan_match_the_reference():
    scores = np.zeros((4, EXPERTS), dtype=np.float32)
    scores[0] = 1.0 / EXPERTS  # padded token: all tied
    scores[1, 200:212] = 4.0  # tie across the boundary
    scores[2] = np.nan  # fully poisoned row
    scores[3, :3] = np.nan  # partially poisoned
    scores[3, 300:310] = np.linspace(1.0, 0.1, 10)
    (kw, ki), (rw, ri) = _both(scores)
    np.testing.assert_array_equal(ki, ri)
    assert np.isnan(kw[2]).all() and np.isnan(rw[2]).all()
    keep = [0, 1, 3]
    np.testing.assert_array_equal(kw[keep], rw[keep])


# Row counts that are not multiples of the block, with the block the old
# "halve until it divides" rule produced. 4352/4384 are the compile_sizes and
# --max-num-batched-tokens of a Qwen3.5 config that ships today
# (.buildkite/kubernetes/manifests/v7x/jobset_disagg.yaml:354); the last token
# bucket is max_num_batched_tokens verbatim (tpu_platform.py:1119), so any
# integer is reachable, and a prime one used to give a block of 1.
RAGGED = [
    (1000, 8),
    (1009, 1),
    (4352, 256),
    (4384, 32),
    (8191, 1),
]


@pytest.mark.parametrize("rows,block,grid", [
    (4, 4, 1),
    (8, 8, 1),
    (17, 17, 1),
    (33, 33, 1),
    (48, 48, 1),
    (64, 64, 1),
    (250, 250, 1),
    (512, 512, 1),
    (1000, 512, 2),
    (1009, 512, 2),
    (1024, 512, 2),
    (1536, 512, 3),
    (2048, 512, 4),
    (4352, 512, 9),
    (4384, 512, 9),
    (8191, 512, 16),
    (8192, 512, 16),
    (16384, 512, 32),
])
def test_block_is_tuned_size_and_grid_rounds_up(rows, block, grid):
    assert block_rows_for(rows) == block == min(MAX_BLOCK_ROWS, rows)
    assert -(-rows // block) == grid


@pytest.mark.parametrize("rows,old_block", RAGGED)
def test_ragged_row_counts_keep_the_tuned_block(rows, old_block):
    """Regression: requiring the block to divide the row count shrank it.

    The old rule halved until it divided, which turned the 512 the kernel was
    tuned at into ``old_block`` -- 32 on a shipped config, 1 on a prime.
    """
    assert old_block < MAX_BLOCK_ROWS, "pick a row count the old rule shrank"
    assert block_rows_for(rows) == MAX_BLOCK_ROWS
