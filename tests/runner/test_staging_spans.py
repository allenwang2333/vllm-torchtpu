# SPDX-License-Identifier: Apache-2.0
import random

import torch

from vllm_torchtpu.runner.staging_spans import zero_stale_spans


def test_matches_full_zero_then_fresh_write():
    rng = random.Random(0)
    for _ in range(500):
        n = rng.randint(1, 64)

        def spans(n=n):
            out = []
            for _ in range(rng.randint(0, 4)):
                a = rng.randint(0, n)
                out.append((a, rng.randint(a, n)))
            return out

        stale, fresh = spans(), spans()
        buf = torch.zeros(n, dtype=torch.int32)
        for a, b in stale:
            buf[a:b] = 7
        zero_stale_spans(buf, stale, fresh)
        # Outside the fresh spans everything is zero; inside, the stale value
        # may survive because the caller overwrites it.
        mask = torch.ones(n, dtype=torch.bool)
        for a, b in fresh:
            mask[a:b] = False
        assert not buf[mask].any()


def test_fresh_covering_stale_skips_work():
    buf = torch.full((10,), 5, dtype=torch.int32)
    zero_stale_spans(buf, [(2, 6)], [(0, 8)])
    assert buf.tolist() == [5] * 10
    zero_stale_spans(buf, [(2, 6)], [(3, 4)])
    assert buf.tolist() == [5, 5, 0, 5, 0, 0, 5, 5, 5, 5]
