# SPDX-License-Identifier: Apache-2.0
"""DSpark under attention DP: verify-row placement, drafting a wide verify in
parts, and the schedules each layout accepts."""

import types

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.megakernel.kimi_k3.decode import KimiDecoder
from vllm_torchtpu.megakernel.kimi_k3.decode_ops import Control, dp_group_rows
from vllm_torchtpu.megakernel.kimi_k3.draft import DsparkDraft, _batched_fn
from vllm_torchtpu.megakernel.kimi_k3.kernels import dspark
from vllm_torchtpu.megakernel.kimi_k3.program import spec_tokens_per_batch

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("batch", [8, 16, 32, 64, 96, 128])
def test_verify_rows_are_the_rows_each_group_attends(batch):
    rows = dp_group_rows(batch, 2)
    decoder = types.SimpleNamespace(groups=2)
    expected = [KimiDecoder._group_rows(decoder, batch, g) for g in range(2)]
    assert rows.tolist() == expected[0] + expected[1]


def _stub_fused(weights, caches, positions, aux, logits, vectors, bases, count, anchor):
    out = jnp.zeros((bases.shape[0], 128), jnp.int32)
    out = out.at[:, dspark.OUT_COUNT].set(count).at[:, dspark.OUT_BONUS].set(anchor)
    # Outputs that depend on each request's own verify rows.
    out = out.at[:, 40].set(aux[0, :: dspark.ROWS, 0].astype(jnp.int32))
    out = out.at[:, 41].set(logits[0, :, 0, 0].astype(jnp.int32))
    return out, caches + 1, positions + bases[None, :, None]


@pytest.mark.parametrize("verified", [24, 32])
def test_drafting_in_parts_matches_one_call(verified):
    rng = np.random.default_rng(verified)
    rows_per, pages = 4, 3
    control = Control(slots=65, rows=64, pages=pages)
    slots = rng.permutation(np.arange(1, 65))[:verified]
    slots[[3, verified - 2]] = 0  # padding requests share the null slot
    entries = [
        (
            int(s),
            i % 3 == 0,
            int(rng.integers(5, 999)),
            int(rng.integers(9, 9000)),
            (0.0, 1.0, 0, 0),
        )
        for i, s in enumerate(slots)
    ]
    plan = control.plan(entries, [np.arange(pages, dtype=np.int32)] * verified)
    state = np.concatenate([rng.integers(0, 50, 3 * control.slots), plan]).astype(
        np.int32
    )
    total = verified * rows_per
    ids = jnp.asarray(rng.integers(0, 1000, 4 * total + verified * pages), jnp.int32)
    aux = jnp.asarray(rng.integers(0, 99, (5, total, 16)), jnp.bfloat16)
    logits = jnp.asarray(rng.integers(0, 99, (32, total, 32)), jnp.float32)
    decisions = jnp.asarray(
        np.stack(
            [rng.integers(0, rows_per, verified), rng.integers(0, 999, verified)], 1
        ),
        jnp.int32,
    )

    def draft(requests, first, state):
        fn = _batched_fn(
            dspark, _stub_fused, control, (), requests, rows_per, first, verified
        )
        return fn(
            jnp.zeros((1, requests, 2, 4, 8), jnp.bfloat16),
            jnp.zeros((1, requests, 4), jnp.int32),
            jnp.zeros((1, requests, 128), jnp.int32),
            aux,
            logits,
            ids,
            state,
            decisions,
        )

    whole = draft(verified, 0, jnp.asarray(state))
    parts, part_state = [], jnp.asarray(state)
    for first, requests in DsparkDraft.parts(verified, rows_per):
        out = draft(requests, first, part_state)
        parts.append(out)
        part_state = out[4]
    np.testing.assert_array_equal(
        np.concatenate([p[0] for p in parts], axis=1), whole[0]
    )
    np.testing.assert_array_equal(np.concatenate([p[3] for p in parts]), whole[3])
    np.testing.assert_array_equal(part_state, whole[4])


def _config(attention_dp, schedule, max_num_seqs=32):
    return MegakernelEngineConfig(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=32 // attention_dp,
        block_size=512,
        kv_cache_dtype="fp8",
        max_num_seqs=max_num_seqs,
        max_model_len=1 << 17,
        max_num_batched_tokens=512,
        num_speculative_tokens=7,
        num_speculative_tokens_per_batch_size=schedule,
        attention_dp=attention_dp,
    )


def test_attention_dp2_speculates_up_to_32_requests():
    schedule = ((1, 2, 7), (3, 32, 3))
    dense = spec_tokens_per_batch(_config(2, schedule))
    assert dense[1:3] == (7, 7) and set(dense[3:]) == {3}
    with pytest.raises(NotImplementedError):
        spec_tokens_per_batch(_config(1, schedule))
    with pytest.raises(NotImplementedError):
        spec_tokens_per_batch(_config(2, ((1, 16, 7),)))
