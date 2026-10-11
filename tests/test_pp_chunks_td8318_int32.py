# SPDX-License-Identifier: Apache-2.0
"""TD8.318: ``schedule_pairs`` int32 temporaries equal the int64 computation.

The per-step attention SMEM capacity check computes (query block, KV block)
pairs with int32 arrays (was int64); CPU timeit: 22.0-22.3 -> 20.9-21.4 us for
192 decode requests and 39.4 -> 33.1-34.0 us for 8 mixed prefills (bq=16).
"""

from __future__ import annotations

import numpy as np
import pytest

from vllm_torchtpu.core.pp_chunks import schedule_pairs


def _ref(q_lens, kv_lens, bq, bkv):
    q = np.asarray(q_lens, dtype=np.int64)
    kv = np.asarray(kv_lens, dtype=np.int64)
    keep = q > 0
    q, kv = q[keep], kv[keep]
    if q.size == 0:
        return 0
    blocks = -(-q // bq)
    i = np.arange(int(blocks.max()))
    q_end = np.minimum((i[None, :] + 1) * bq, q[:, None])
    k_end = -(-(kv[:, None] - q[:, None] + q_end) // bkv)
    return int(np.where(i[None, :] < blocks[:, None], k_end, 0).sum())


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(("bq", "bkv"), [(16, 128), (128, 512), (8, 64)])
def test_schedule_pairs_int32_matches_int64(seed, bq, bkv):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 200))
    q = rng.integers(0, 4096, n)
    q[rng.random(n) < 0.2] = 0
    kv = q + rng.integers(0, 1_000_000, n)
    got = schedule_pairs(q.tolist(), kv, bq, bkv)
    assert isinstance(got, int)
    assert got == _ref(q, kv, bq, bkv)


def test_schedule_pairs_large_total_does_not_overflow():
    # 4096 requests x 1M-token contexts: the pair total exceeds 2**31 while
    # every per-element value stays in int32 range.
    q = np.full(4096, 2048)
    kv = np.full(4096, 1_000_000)
    got = schedule_pairs(q, kv, 16, 16)
    assert got == _ref(q, kv, 16, 16) and got > 2**31
