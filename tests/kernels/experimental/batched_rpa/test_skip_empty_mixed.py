# SPDX-License-Identifier: Apache-2.0
"""The MIXED pass is skipped on decode-only steps, bit-exactly.

`skip_empty_mixed` puts the MIXED schedule and kernel under a `lax.cond` on
the runtime distribution. A decode-only step must not launch them and must
return exactly what the empty MIXED kernel returned; a step with a MIXED
sequence, including a short prefill chunk in a decode-sized bucket, must take
the MIXED path unchanged.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jex_core

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

LAYOUT = configs.KVLayout.SEQ_ALONG_LANE
PAGE_SIZE = 256
HEAD_DIM = 256
NUM_Q_HEADS = 16
MAX_SEQS = 8
DECODE_QUERY_SIZE = 4  # MTP: 1 + 3 rows per decode request.

# (q_lens, kv_lens, distribution). Rows are 1-4 per decode request.
DECODE_ONLY = ((4, 4, 3, 4), (1025, 4100, 7, 300), (4, 4, 4))
# Two MTP decodes plus a 7-token prefill chunk: 15 rows, a decode-sized bucket.
SMALL_BUCKET_PREFILL = ((4, 4, 7), (1025, 4100, 520), (2, 2, 3))
# Prefill only: no decode sequence, every sequence MIXED.
PREFILL_ONLY = ((33, 120), (600, 120), (0, 0, 2))


def _inputs(q_lens, kv_lengths, distribution, *, num_tokens=None):
    rng = np.random.default_rng(8228)
    total = sum(q_lens)
    num_tokens = num_tokens or max(16, total)
    pages_per_seq = (max(kv_lengths) + PAGE_SIZE - 1) // PAGE_SIZE + 1
    cache_shape = wrapper.get_kv_cache_shape(
        MAX_SEQS * pages_per_seq + 2,
        PAGE_SIZE,
        1,
        HEAD_DIM,
        jnp.float8_e4m3fn,
        kv_layout=LAYOUT,
    )

    def rand(shape, dtype):
        return jnp.asarray((rng.standard_normal(shape) * 0.5).astype(dtype))

    q = rand((num_tokens, NUM_Q_HEADS, HEAD_DIM), jnp.bfloat16)
    k = rand((num_tokens, 1, HEAD_DIM), jnp.float8_e4m3fn)
    v = rand((num_tokens, 1, HEAD_DIM), jnp.float8_e4m3fn)
    cache = rand(cache_shape, jnp.float8_e4m3fn)
    page_table = rng.permutation(cache_shape[0])[: MAX_SEQS * pages_per_seq]
    kv_lens = np.zeros(MAX_SEQS, np.int32)
    kv_lens[: len(kv_lengths)] = kv_lengths
    cu_q_lens = np.full(MAX_SEQS + 1, total, np.int32)
    cu_q_lens[: len(q_lens) + 1] = np.cumsum((0, *q_lens))
    return (
        q,
        k,
        v,
        cache,
        jnp.asarray(kv_lens),
        jnp.asarray(page_table.astype(np.int32)),
        jnp.asarray(cu_q_lens),
        jnp.asarray(distribution, jnp.int32),
    )


def _rpa(skip_empty_mixed):
    return functools.partial(
        wrapper.ragged_paged_attention,
        sm_scale=HEAD_DIM**-0.5,
        kv_layout=LAYOUT,
        decode_query_size=DECODE_QUERY_SIZE,
        skip_empty_mixed=skip_empty_mixed,
    )


def _count_pallas_calls(jaxpr, under_cond=False, counts=None):
    """Pallas calls outside and inside `cond` branches, recursively."""
    if counts is None:
        counts = {False: 0, True: 0}
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if name == "pallas_call":
            counts[under_cond] += 1
            continue
        if name == "cond":
            for branch in eqn.params["branches"]:
                _count_pallas_calls(branch.jaxpr, True, counts)
            continue
        for param in eqn.params.values():
            if isinstance(param, jex_core.ClosedJaxpr):
                _count_pallas_calls(param.jaxpr, under_cond, counts)
            elif isinstance(param, jex_core.Jaxpr):
                _count_pallas_calls(param, under_cond, counts)
    return counts


def _copy_args(args):
    return jax.tree.map(lambda x: x.copy(), args)


@pytest.mark.parametrize("skip", [True, False], ids=["skip", "always"])
def test_mixed_pass_sits_under_cond(skip):
    """Decode's schedule + kernel run unconditionally, MIXED's under the cond."""
    args = _inputs(*DECODE_ONLY)
    jaxpr = jax.make_jaxpr(_rpa(skip))(*args).jaxpr
    counts = _count_pallas_calls(jaxpr)
    if skip:
        assert counts == {False: 2, True: 2}, counts
    else:
        assert counts == {False: 4, True: 0}, counts


@pytest.mark.parametrize(
    "case",
    [DECODE_ONLY, SMALL_BUCKET_PREFILL, PREFILL_ONLY],
    ids=["decode_only", "small_bucket_prefill", "prefill_only"],
)
def test_skip_is_bit_exact(case):
    """Output and cache match the unconditional MIXED pass exactly."""
    args = _inputs(*case)
    out_skip, cache_skip = jax.block_until_ready(_rpa(True)(*_copy_args(args)))
    out_ref, cache_ref = jax.block_until_ready(_rpa(False)(*_copy_args(args)))
    total = sum(case[0])
    out_skip = np.asarray(out_skip[:total].astype(jnp.float32))
    out_ref = np.asarray(out_ref[:total].astype(jnp.float32))
    assert np.isfinite(out_ref).all()
    np.testing.assert_array_equal(out_skip, out_ref)
    np.testing.assert_array_equal(
        np.asarray(cache_skip).view(np.uint8), np.asarray(cache_ref).view(np.uint8)
    )


def test_small_bucket_prefill_takes_mixed_path():
    """Positive control: the prefill chunk's rows need the MIXED pass.

    Lying that the step is decode-only (MIXED range empty) must change the
    prefill rows; otherwise the bit-exact test above could not tell a skipped
    MIXED pass from a taken one.
    """
    q_lens, kv_lens, _ = SMALL_BUCKET_PREFILL
    args = _inputs(*SMALL_BUCKET_PREFILL)
    lie = args[:-1] + (jnp.asarray((2, 2, 2), jnp.int32),)
    out_true, _ = _rpa(True)(*_copy_args(args))
    out_lie, _ = _rpa(True)(*_copy_args(lie))
    start, end = sum(q_lens[:2]), sum(q_lens)
    decode_true = np.asarray(out_true[:start].astype(jnp.float32))
    decode_lie = np.asarray(out_lie[:start].astype(jnp.float32))
    np.testing.assert_array_equal(decode_true, decode_lie)
    prefill_true = np.asarray(out_true[start:end].astype(jnp.float32))
    prefill_lie = np.asarray(out_lie[start:end].astype(jnp.float32))
    assert not np.array_equal(prefill_true, prefill_lie)


def test_cond_does_not_copy_kv_cache():
    """The cond passes the donated cache through without a full-pool copy."""
    args = _inputs(*DECODE_ONLY)
    hlo = jax.jit(_rpa(True), donate_argnums=(3,)).lower(*args).compile().as_text()
    assert "conditional" in hlo
    cache_dims = "[" + ",".join(str(d) for d in args[3].shape) + "]"
    copies = [
        line
        for line in hlo.splitlines()
        if cache_dims in line and (" copy(" in line or " copy-start(" in line)
    ]
    assert not copies, "\n".join(copies)
