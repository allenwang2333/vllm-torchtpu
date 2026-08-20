# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The computed block schedule for the MLA mixed kernel.

`TuningKey` is keyed partly on deployment settings -- `max_num_tokens`,
`max_num_seqs`, `pages_per_seq` -- so a hand-written table row matches only
the exact configuration it was measured on. The miss path therefore has to
produce a sane schedule on its own; returning the minimum legal value put the
kernel at 80.3% of device time and 2.76% FLOPS utilization on Kimi-Linear.

The anchor test is that the computed value reproduces the measured optimum
for the shape we swept end to end.
"""

from __future__ import annotations

import pytest

from vllm_torchtpu.kernels.mla.v2.tuned_params import (
    TuningKey, mixed_kv_pages_per_block)


def _key(**overrides) -> TuningKey:
    """Kimi-Linear-48B-A3B at TP=4, from the profile's XLA expression."""
    fields = dict(
        case="mixed",
        max_num_tokens=2048,
        actual_num_q_heads=8,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="bfloat16",
        page_size_per_kv_packing=16,
        kv_packing=2,
        max_num_seqs=64,
        pages_per_seq=512,
    )
    fields.update(overrides)
    return TuningKey(**fields)


def test_reproduces_the_measured_optimum() -> None:
    """Kimi-Linear's shape must land on 32.

    Swept end to end at 8k/1k, concurrency 16: 1 -> 581.94 tok/s, 4 -> 829.70,
    16 -> 945.33, 32 -> 964.71. The curve flattens past 16 and nothing OOMed,
    so 32 is just past the knee.
    """
    assert mixed_kv_pages_per_block(_key()) == 32


def test_never_exceeds_the_pages_a_sequence_has() -> None:
    """A block cannot span more pages than the sequence owns."""
    assert mixed_kv_pages_per_block(_key(pages_per_seq=8)) == 8
    assert mixed_kv_pages_per_block(_key(pages_per_seq=1)) == 1


def test_backs_off_when_pages_are_large() -> None:
    """Wider pages mean fewer fit in VMEM, so the block count must drop."""
    big = _key(page_size_per_kv_packing=256, kv_packing=4)
    assert mixed_kv_pages_per_block(big) < 32
    assert mixed_kv_pages_per_block(big) >= 1


def test_result_is_always_a_positive_power_of_two() -> None:
    """The field's documented step is powers of two, and 0 is not legal."""
    for pages_per_seq in (1, 3, 7, 9, 100, 512, 4096):
        for page_size in (16, 64, 256):
            value = mixed_kv_pages_per_block(
                _key(pages_per_seq=pages_per_seq,
                     page_size_per_kv_packing=page_size))
            assert value >= 1
            assert value & (value - 1) == 0, f"{value} is not a power of two"


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit setting bypasses the computation entirely."""
    import vllm_torchtpu.envs as tpu_envs
    monkeypatch.setattr(tpu_envs, "TPU_MLA_MIXED_KV_PAGES_PER_BLOCK", 7)
    assert mixed_kv_pages_per_block(_key()) == 7
