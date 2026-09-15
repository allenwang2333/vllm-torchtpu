# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the v3 RPA single-query-block mask collapse.

When a compute block holds exactly one query token (`actual_bq_csz == 1`), every
row of the score matrix is a different head of the *same* token, so the causal
bound and the kv-length bound both reduce to conditions on the kv index alone
and collapse into one comparison against one scalar:

    causal   q_span >= k_span          <=>  i < processed_q_len - processed_kv_len + 1
    kv len   k_span < effective_kv_len <=>  i < effective_kv_len - processed_kv_len

Why this file exists
--------------------
Nothing else in the repo cross-checks the v3 Pallas kernel's *numerics* against
`ref_ragged_paged_attention`. `test_rpa_bundled.py` compares
`ragged_paged_attention_bundled` against `ragged_paged_attention`, but the
bundled entry point calls `ragged_paged_attention` internally, so the attention
math -- masks included -- cancels on both sides of every assertion: a wrong
bound yields the same wrong answer twice and the suite passes. It is a test of
the page-index remap and buffer donation, not of the mask.

The shapes below are chosen to make a wrong bound *visible*, which the obvious
decode shape does not:

  * `processed_kv_len > 0`. The collapsed path is the first code here to
    *subtract*, and `processed_kv_len` is the term it subtracts. Any shape with
    `bkv_csz >= kv_len` runs a single trip with `processed_kv_len == 0`, where
    dropping the subtraction entirely is undetectable. Every case here forces
    at least two trips.
  * `q_len > 1` with `bq_csz == 1`. In pure decode `processed_q_len + 1` and
    `effective_kv_len` are *equal* at every `bkv_start`, so `jnp.minimum` is a
    no-op and dropping either term -- or the `minimum` itself -- changes
    nothing. Only a multi-token query block separates them.
  * A partial final block (`kv_len % bkv_csz != 0`) and ragged per-sequence
    `kv_len`, so the tail is exercised at an offset rather than on a boundary.

`test_parity_resolves_a_one_column_bound_error` is the negative control: it
asserts that this comparison actually *fails* when the kv bound is off by one.
Without it, a green suite here would be as uninformative as the bundled one.

If you inject a fault into the kernel to re-check this suite's teeth, do not
inject it into `v_bound`. `v` is deliberately bounded by `effective_kv_len`
rather than by the causal bound, because positions the causal mask kills
already carry `p == 0` and zeroing their `v` would change nothing -- the
kernel says so in its own comment. Loosening `v_bound` is therefore a no-op by
construction, and job j-8f5b7415 confirmed it empirically: a `+ 1` there left
the output bit-identical in every one of twelve cases. Inject into the
`s_bound` causal term (`processed_q_len - processed_kv_len + 1`) or delete the
`jnp.minimum` clamp instead; those are load-bearing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as K

HEAD_DIM = 128
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
PAGE_SIZE = 16

# Job j-8f5b7415 measured this suite's resolving power and found it had none:
# a deliberately broken kernel produced output bit-identical to the correct one
# to eight significant figures. Ordinary random data cannot expose a bound that
# is one column too loose, because one column out of 256 carries ~0.4% of the
# softmax mass -- well under the fp8 rounding floor, which a pre-existing
# 1-in-12288 element already pushes past atol=2e-2. The tolerance cannot be
# tightened past that floor, so the *signal* is made larger instead. Two
# planted signals, because the two bounds fail into different data:
#
#   POISON_V  cache slots at or past kv_len. A kv-length bound one column too
#             loose reads one of these. K is set to zero there rather than to
#             something large: a zero key gives a score of exactly 0 and hence
#             an ordinary softmax weight exp(0), whereas a large random key
#             gives a score of random sign that is exponentially suppressed
#             half the time -- which would make detection a coin flip.
#   NEW_V     the v of the q_len newly-written tokens. A *causal* bound one
#             column too loose does not read poison at all; it reads a future
#             token of the same sequence, which is legitimate data the row
#             below it attends to. Scaling those tokens' v is the only way to
#             make that error visible.
#
# Both move the output by O(NEW_V / kv_len) ~ 0.1-0.3 against a baseline output
# of ~0.1, i.e. tens of percent, against tolerances of a few percent.
POISON_V = 64.0
NEW_V = 16.0

RTOL = 3e-2
ATOL = 5e-2


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


# `seqs` is a list of (kv_len, q_len). kv_len counts the q_len new tokens, so
# the cache carries positions [0, kv_len - q_len) and the kernel writes the rest.
CASES = {
    # Pure decode, but with bkv_csz well below kv_len so the kv loop runs four
    # trips and `processed_kv_len` takes the values 0, 64, 128, 192.
    "decode_multi_trip":
    dict(seqs=[(256, 1)] * 4, distribution=(4, 4, 4), bkv=64),
    # kv_len is not a multiple of bkv_csz, so the final trip is partial and the
    # bound lands mid-block rather than on a block boundary.
    "decode_partial_block":
    dict(seqs=[(200, 1)] * 4, distribution=(4, 4, 4), bkv=64),
    # Per-sequence kv_lens differ, so a bound computed from the wrong sequence's
    # length -- or hoisted out of the sequence loop -- shows up here.
    "decode_ragged_kv_lens":
    dict(seqs=[(71, 1), (256, 1), (129, 1), (200, 1)],
         distribution=(4, 4, 4),
         bkv=64),
    # THE discriminating case. q_len > 1 with bq_csz == 1 gives one query token
    # per compute block at several distinct `processed_q_len` values, so the
    # causal bound is strictly tighter than the kv bound for every token but the
    # last. This is the only shape here where `jnp.minimum` does real work.
    "specdecode_causal_vs_kv":
    dict(seqs=[(200, 5), (256, 8)], distribution=(0, 0, 2), bkv=64),
    # Same, mixed against true decode rows in one call.
    "mixed_specdecode_and_decode":
    dict(seqs=[(128, 1), (128, 1), (200, 4), (256, 6)],
         distribution=(2, 2, 4),
         bkv=64),
}


def _build(seqs, distribution, kv_dtype, seed=0):
    """Build the workload as *host* numpy arrays.

    Deliberately not device arrays: `ragged_paged_attention` donates its inputs,
    so every device buffer it is handed is deleted on return and a second call
    against the same buffers dies with `RuntimeError: Array has been deleted`.
    Every test here calls twice (kernel vs reference, or collapsed vs general),
    so the device arrays are materialized per call by `_call` instead.
    """
    rng = np.random.default_rng(seed)
    kv_lens = [kv for kv, _ in seqs]
    q_lens = [q for _, q in seqs]
    num_seqs = len(seqs)
    # One page of slack per sequence: the negative control bumps a kv_len by 1,
    # and without the slack a kv_len that is an exact multiple of PAGE_SIZE
    # overflows the page table (`Require page_cnt=17 <= pages_per_seq=16`)
    # before any numerics run.
    pages_per_seq = int(np.ceil((max(kv_lens) + 1) / PAGE_SIZE))
    num_blocks = num_seqs * pages_per_seq

    cu_q_lens = np.cumsum([0] + q_lens, dtype=np.int32)
    total_q = int(cu_q_lens[-1])

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    q = r(total_q, NUM_Q_HEADS, HEAD_DIM)
    k = r(total_q, NUM_KV_HEADS, HEAD_DIM)
    # The new tokens' v, varied per token so that attending to the wrong one is
    # unambiguous rather than merely doubling an identical contribution.
    v = np.broadcast_to(
        (NEW_V * (1 + np.arange(total_q) % 4)).astype(np.float32)[:, None,
                                                                  None],
        (total_q, NUM_KV_HEADS, HEAD_DIM)).copy()

    pool = np.arange(num_blocks, dtype=np.int32)
    rng.shuffle(pool)

    cache_shape = K.get_kv_cache_shape(num_blocks, PAGE_SIZE, NUM_KV_HEADS,
                                       HEAD_DIM, kv_dtype)
    # Along the num_kv_heads_x2 axis the cache interleaves (K, V) per head:
    # index 2h is head h's key, 2h+1 its value. `ref_ragged_paged_attention`
    # reshapes (num_kv_heads_x2, head_dim) -> (num_kv_heads, head_dim * 2) and
    # splits at head_dim, which is what fixes that order.
    flat = np.empty((num_blocks, PAGE_SIZE, NUM_KV_HEADS * 2, cache_shape[-1]),
                    np.float32)
    # Poison first, everywhere -- including pages no sequence maps to.
    flat[:, :, 0::2, :] = 0.0
    flat[:, :, 1::2, :] = POISON_V
    # Then write real history back over the live range of each sequence. The
    # slots left poisoned are exactly those a correct mask must never read.
    for s, kv_len in enumerate(kv_lens):
        for p in range(kv_len):
            pg = pool[s * pages_per_seq + p // PAGE_SIZE]
            off = p % PAGE_SIZE
            flat[pg, off, 0::2, :] = r(NUM_KV_HEADS, HEAD_DIM)
            flat[pg, off, 1::2, :] = r(NUM_KV_HEADS, HEAD_DIM)
    cache = np.asarray(
        jnp.asarray(flat.reshape(cache_shape), dtype=kv_dtype))

    def host(x, dtype):
        return np.asarray(jnp.asarray(x, dtype))

    return dict(
        queries=host(q, jnp.bfloat16),
        keys=host(k, kv_dtype),
        values=host(v, kv_dtype),
        kv_cache=cache,
        kv_lens=np.asarray(kv_lens, np.int32),
        page_indices=pool,
        cu_q_lens=cu_q_lens,
        distribution=np.asarray(distribution, np.int32),
    )


def _call(fn, wl, bkv, **kw):
    """Invoke `fn` with bq_csz pinned to 1 and bkv_csz pinned to `bkv`.

    All three case block-size tuples are set identically so that whichever of
    decode/prefill/mixed the `distribution` selects, the collapsed path is the
    one under test. `ref_ragged_paged_attention` takes no block sizes -- it is
    a plain dense implementation -- so they are dropped for it.

    `wl` holds host arrays and is uploaded fresh here on every call, because the
    callee donates (and therefore deletes) what it is given. See `_build`.
    """
    d = {k: jnp.asarray(v) for k, v in wl.items()}
    blocks = (1, bkv, 1, bkv)
    extra = {} if fn is K.ref_ragged_paged_attention else dict(
        d_block_sizes=blocks, p_block_sizes=blocks, m_block_sizes=blocks)
    out, cache = fn(d["queries"],
                    d["keys"],
                    d["values"],
                    d["kv_cache"],
                    d["kv_lens"],
                    d["page_indices"],
                    d["cu_q_lens"],
                    d["distribution"],
                    sm_scale=HEAD_DIM**-0.5,
                    **extra,
                    **kw)
    return np.asarray(out.astype(jnp.float32)), np.asarray(
        cache.astype(jnp.float32))


def _valid_rows(wl, seqs):
    """Row mask over the q axis; padding rows past cu_q_lens[-1] are undefined."""
    total_q = int(wl["cu_q_lens"][-1])
    return slice(0, total_q)


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("kv_dtype", [jnp.bfloat16, jnp.float8_e4m3fn],
                         ids=["bf16", "fp8"])
def test_pallas_matches_reference(case, kv_dtype):
    """The Pallas kernel must agree with the dense reference on every shape.

    This is the assertion `test_rpa_bundled.py` cannot make, because both of its
    sides route through `ragged_paged_attention`.
    """
    _require_tpu()
    spec = CASES[case]
    wl = _build(spec["seqs"], spec["distribution"], kv_dtype)
    scales = dict(k_scale=1.0,
                  v_scale=1.0) if kv_dtype == jnp.float8_e4m3fn else {}

    got, got_cache = _call(K.ragged_paged_attention, wl, spec["bkv"], **scales)
    want, want_cache = _call(K.ref_ragged_paged_attention, wl, spec["bkv"],
                             **scales)

    rows = _valid_rows(wl, spec["seqs"])
    # Tolerance absorbs fp8 quantization and the accumulation-order difference
    # between an online-softmax kernel and a dense reference; it is not what
    # gives the suite its teeth. The planted signals do that, and
    # `test_parity_resolves_a_one_column_bound_error` is what holds these two
    # numbers honest -- if they are ever loosened far enough to hide a
    # one-column error, that test goes red.
    np.testing.assert_allclose(got[rows],
                               want[rows],
                               rtol=RTOL,
                               atol=ATOL,
                               err_msg=f"output mismatch for {case}/{kv_dtype}")
    np.testing.assert_allclose(got_cache,
                               want_cache,
                               rtol=RTOL,
                               atol=ATOL,
                               err_msg=f"cache mismatch for {case}/{kv_dtype}")


def test_parity_resolves_a_one_column_bound_error():
    """Negative control: the comparison above must catch an off-by-one bound.

    A green parity suite proves nothing unless the comparison can resolve the
    error class it is meant to guard. Here the *reference* is given a kv_len one
    larger than the kernel's for a single sequence, which is exactly the
    observable effect of a `+ 1` slipping into (or out of) the collapsed bound.
    If this test starts passing -- i.e. the mismatch stops being detected -- the
    tolerances above have gone slack and the suite is no longer a gate.
    """
    _require_tpu()
    spec = CASES["decode_multi_trip"]
    wl = _build(spec["seqs"], spec["distribution"], jnp.bfloat16)

    got, _ = _call(K.ragged_paged_attention, wl, spec["bkv"])

    perturbed = dict(wl)
    bumped = wl["kv_lens"].copy()
    bumped[0] += 1
    perturbed["kv_lens"] = bumped
    want, _ = _call(K.ref_ragged_paged_attention, perturbed, spec["bkv"])

    rows = _valid_rows(wl, spec["seqs"])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(got[rows], want[rows], rtol=RTOL, atol=ATOL)


def test_collapsed_and_general_paths_agree():
    """The collapsed path and the general path must produce the same answer.

    `sliding_window` is excluded from the collapse, so passing one routes the
    identical workload down the general path. Agreement between the two is a
    direct A/B of the collapse itself, independent of the reference: the window
    is set wider than any kv_len here, so it masks nothing and the two calls are
    mathematically the same attention.
    """
    _require_tpu()
    spec = CASES["specdecode_causal_vs_kv"]
    wl = _build(spec["seqs"], spec["distribution"], jnp.bfloat16)
    wide = int(max(kv for kv, _ in spec["seqs"])) + 1

    collapsed, collapsed_cache = _call(K.ragged_paged_attention, wl,
                                       spec["bkv"])
    general, general_cache = _call(K.ragged_paged_attention,
                                   wl,
                                   spec["bkv"],
                                   sliding_window=wide)

    rows = _valid_rows(wl, spec["seqs"])
    # Both sides are the same kernel on the same inputs, differing only in
    # which branch computes the bounds, so this is far tighter than the
    # reference comparison above.
    np.testing.assert_allclose(collapsed[rows], general[rows], rtol=1e-3,
                               atol=1e-3)
    np.testing.assert_allclose(collapsed_cache, general_cache, rtol=1e-3,
                               atol=1e-3)
