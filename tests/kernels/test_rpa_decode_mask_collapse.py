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
  * `q_len > 1` with `bq_csz == 1` *and `bq_sz > 1`*. The last of those three
    is easy to miss and voids the other two: see BQ_SZ. In pure decode
    `processed_q_len + 1` and
    `effective_kv_len` are *equal* at every `bkv_start`, so `jnp.minimum` is a
    no-op and dropping either term -- or the `minimum` itself -- changes
    nothing. Only a multi-token query block separates them.
  * A partial final block (`kv_len % bkv_csz != 0`) and ragged per-sequence
    `kv_len`, so the tail is exercised at an offset rather than on a boundary.

Two checks, and the order matters. `test_collapsed_and_general_paths_agree` is
the PRIMARY gate: it runs the same kernel twice on the same inputs, once down
the collapsed path and once down the general path (a `sliding_window` wider
than any kv_len excludes the collapse while masking nothing), so quantization
and accumulation order are identical on both sides and it can assert at 1e-3
on fp8 as well as bf16. `test_pallas_matches_reference` is the secondary
check: it is the only one independent of the kernel, but it pays for that with
a tolerance 30x looser, and on fp8 the price is more than the signal is worth
-- see its docstring.

`test_parity_resolves_a_kv_bound_error` is the negative control: it asserts
that the reference comparison actually *fails* when the bound is loose.
Without it, a green suite here would be as uninformative as the bundled one.

Where to inject a fault, if you are re-checking this suite's teeth
-----------------------------------------------------------------
Two of the three lines in the collapsed path are *not* load-bearing, each for
a different reason, and both have burned a TPU job here already.

1. Not `v_bound`. `v` is deliberately bounded by `effective_kv_len` rather than
   by the causal bound, because positions the causal mask kills already carry
   `p == 0` and zeroing their `v` would change nothing -- the kernel says so in
   its own comment. Job j-8f5b7415 confirmed it: a `+ 1` there left the output
   bit-identical in all twelve cases.

2. Not the causal clamp, *if what you care about is decode*. This is the one
   that matters and it is invisible from the collapsed path itself:

       kernel.py:346-347      if case == RpaCase.DECODE:
                                  use_causal_mask = False

   `case` is a static Python argument, so in the DECODE dispatch
   (kernel.py:1983-1990, `static_q_len=1`) the whole `if use_causal_mask:`
   block is a trace-time-false branch that emits *nothing*. The production op
   `RPAd-p_256-bq_1_1-bkv_9216_9216` is that dispatch -- the `d` is
   `RpaCase.DECODE.symbol`. Mutating the clamp therefore cannot perturb the
   shipping compile at all, and job j-6e0fe029 demonstrated exactly that: two
   clamp mutations were caught only by the `specdecode`/`mixed` cases, which
   run the MIXED dispatch, while every pure-decode case passed both. That was
   read at the time as "the clamp cannot bind in decode"; the truth is
   stronger and worse -- the clamp is not there.

That leaves the one line that *is* live in the decode compile, and which every
mutation before j-<pending> had missed:

       v_bound_i32 = effective_kv_len - processed_kv_len
       s_bound_i32 = v_bound_i32

In DECODE this single subtraction is the entire mask. Inject there. Note that
the causal clamp *masks* a loose kv bound wherever the clamp is live, so the
`specdecode`/`mixed` cases -- the ones with teeth against the clamp -- are the
ones least able to catch a kv-term error. The two regimes need different cases,
which is why both are present below.
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

# The q *fetch* block size. It must be > 1 while bq_csz stays 1, and job
# j-329615f6 is the reason: with both pinned to 1 the collapsed bounds cannot
# be wrong, so the suite passed two deliberately broken kernels.
#
# `effective_kv_len` is computed once per bq block (kernel.py:996-998) as
#   min(kv_len, processed_q_len + actual_bq_sz)
# while the causal clamp inside the collapsed path uses the *chunk's* offset,
#   processed_q_len + bq_start                    (kernel.py:1093)
# with bq_start stepping by actual_bq_csz. When actual_bq_sz == 1 there is one
# chunk at bq_start == 0, so the block bound and the chunk bound coincide and
# `jnp.minimum` returns its first argument unchanged -- deleting the clamp or
# loosening it to +2 are then no-ops by construction, in the same way the
# earlier v_bound mutation was. Pinning bq_sz to 1, as this file previously
# did, put every case in exactly that regime.
#
# With bq_sz > 1 and q_len > 1 the early chunks of a block have a causal bound
# strictly tighter than the block's effective_kv_len, the clamp binds, and a
# wrong bound reads future tokens of the same sequence -- which is what NEW_V
# is scaled to expose. Note this regime is not reachable in pure decode
# (q_len == 1 forces actual_bq_sz == 1 via `min(bq_sz, static_q_len)`), so the
# cases that exercise it are the speculative-decode and mixed ones.
BQ_SZ = 8

# Reference comparison (bf16 only): absorbs the online-softmax vs dense
# accumulation-order difference. Loose, and held honest by
# `test_parity_resolves_a_kv_bound_error`.
RTOL = 3e-2
ATOL = 5e-2
# Same-kernel A/B: no dtype or accumulation asymmetry on either side, so this
# is where the real resolving power lives.
AB_RTOL = 1e-3
AB_ATOL = 1e-3


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
    # THE SHIPPING COMPILE. Everything above runs at toy page and block sizes;
    # this one reproduces the production decode op end to end --
    # `RPAd-p_256-bq_1_1-bkv_9216_9216` at kv_len 8192 -- so that the collapsed
    # path is gated in the exact configuration the benchmark measures:
    #
    #   * DECODE dispatch (q_len == 1, distribution routes all four seqs to it),
    #     hence `use_causal_mask = False` and the causal clamp is not emitted.
    #     The kv term is the whole mask here; see the module docstring.
    #   * page_size 256, not 16.
    #   * bkv_csz 9216 >= kv_len, so this runs a SINGLE trip with
    #     processed_kv_len == 0 throughout, exactly as production does.
    #   * max_kv_len = 33 * 256 = 8448 <= 32767, with a packed q dtype on
    #     TPU >= 6, so `int_ty` narrows to i16 -- the same narrowing production
    #     takes, and the one the `.astype(int_ty)` in the collapsed path exists
    #     to serve.
    #
    # The single trip means this case cannot detect a *dropped* subtraction
    # (processed_kv_len is 0, so subtracting it is the identity). That is what
    # the multi-trip cases above are for. This one detects a wrong *bound*: at
    # kv_len 8192 the new token sits at position 8191 and position 8192 is
    # poisoned, so +1 reads poison and -1 drops the new token. The two kinds of
    # coverage are complementary and neither substitutes for the other.
    "decode_production_shape":
    dict(seqs=[(8192, 1)] * 4,
         distribution=(4, 4, 4),
         bkv=9216,
         page_size=256),
}


def _build(seqs, distribution, kv_dtype, seed=0, page_size=PAGE_SIZE):
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
    pages_per_seq = int(np.ceil((max(kv_lens) + 1) / page_size))
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

    cache_shape = K.get_kv_cache_shape(num_blocks, page_size, NUM_KV_HEADS,
                                       HEAD_DIM, kv_dtype)
    # Along the num_kv_heads_x2 axis the cache interleaves (K, V) per head:
    # index 2h is head h's key, 2h+1 its value. `ref_ragged_paged_attention`
    # reshapes (num_kv_heads_x2, head_dim) -> (num_kv_heads, head_dim * 2) and
    # splits at head_dim, which is what fixes that order.
    flat = np.empty((num_blocks, page_size, NUM_KV_HEADS * 2, cache_shape[-1]),
                    np.float32)
    # Poison first, everywhere -- including pages no sequence maps to.
    flat[:, :, 0::2, :] = 0.0
    flat[:, :, 1::2, :] = POISON_V
    # Then write real history back over the live range of each sequence. The
    # slots left poisoned are exactly those a correct mask must never read.
    #
    # Vectorized per sequence rather than per position: the production-shape
    # case has 4 x 8192 positions and a Python loop over them costs more than
    # the TPU work it sets up. `both` is drawn as (p, {k,v}, head, dim) in C
    # order, which is the same draw sequence the per-position loop made (K_p
    # then V_p), so the data is bit-identical to what the earlier jobs ran.
    for s, kv_len in enumerate(kv_lens):
        pos = np.arange(kv_len)
        pgs = pool[s * pages_per_seq + pos // page_size]
        offs = pos % page_size
        both = r(kv_len, 2, NUM_KV_HEADS, HEAD_DIM)
        flat[pgs, offs, 0::2, :] = both[:, 0]
        flat[pgs, offs, 1::2, :] = both[:, 1]
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
    """Invoke `fn` with bq_csz pinned to 1, bq_sz at BQ_SZ, bkv_csz at `bkv`.

    bq_csz == 1 is what selects the collapsed path; bq_sz > 1 is what makes the
    causal clamp inside it able to bind. See BQ_SZ.

    All three case block-size tuples are set identically so that whichever of
    decode/prefill/mixed the `distribution` selects, the collapsed path is the
    one under test. `ref_ragged_paged_attention` takes no block sizes -- it is
    a plain dense implementation -- so they are dropped for it.

    `wl` holds host arrays and is uploaded fresh here on every call, because the
    callee donates (and therefore deletes) what it is given. See `_build`.
    """
    d = {k: jnp.asarray(v) for k, v in wl.items()}
    blocks = (BQ_SZ, bkv, 1, bkv)
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
def test_collapsed_and_general_paths_agree(case, kv_dtype):
    """PRIMARY GATE. The collapsed path must match the general path exactly.

    `sliding_window` is excluded from the collapse, so passing one routes the
    identical workload down the general path. The window is set wider than any
    kv_len here, so it masks nothing and the two calls are mathematically the
    same attention -- the only difference is which branch computes the bounds,
    which is precisely the code under test.

    This is the primary gate rather than the reference comparison below, for
    one reason: both sides are the *same kernel* on the *same inputs* in the
    *same dtype*, so there is no quantization asymmetry and no online-vs-dense
    accumulation-order difference to absorb. The tolerance can therefore be
    ~30x tighter than the reference comparison can afford, and -- unlike that
    comparison -- it stays tight on fp8, which is the configuration the
    optimization actually ships on. Any error in the collapsed bounds shows up
    here at full amplitude.
    """
    _require_tpu()
    spec = CASES[case]
    wl = _build(spec["seqs"],
                spec["distribution"],
                kv_dtype,
                page_size=spec.get("page_size", PAGE_SIZE))
    scales = dict(k_scale=1.0,
                  v_scale=1.0) if kv_dtype == jnp.float8_e4m3fn else {}
    wide = int(max(kv for kv, _ in spec["seqs"])) + 1

    collapsed, collapsed_cache = _call(K.ragged_paged_attention, wl,
                                       spec["bkv"], **scales)
    general, general_cache = _call(K.ragged_paged_attention,
                                   wl,
                                   spec["bkv"],
                                   sliding_window=wide,
                                   **scales)

    rows = _valid_rows(wl, spec["seqs"])

    # Finiteness first, and as its own assertion rather than a flag on the
    # comparison below. `np.testing.assert_allclose` defaults to
    # `equal_nan=True`, so a NaN at the same index in both arms compares *equal*
    # and this A/B would pass while comparing garbage to garbage. That is not
    # hypothetical here: under the `kv_bound_plus1` mutation the fp8
    # `decode_production_shape` case fails with a bare "nan location mismatch"
    # and no element statistics at all, which means NaNs are reachable in this
    # kernel one column past `kv_len`. Whether the *pristine* output is already
    # carrying them was untested until this line existed -- and if it is, every
    # NaN element is one the gate above was silently excusing.
    for name, arr in (("collapsed", collapsed[rows]), ("general",
                                                       general[rows])):
        bad = int((~np.isfinite(np.asarray(arr, np.float32))).sum())
        assert bad == 0, (f"{name} output has {bad} non-finite element(s) for "
                          f"{case}/{kv_dtype}; assert_allclose would have "
                          f"compared them as equal")

    np.testing.assert_allclose(collapsed[rows],
                               general[rows],
                               rtol=AB_RTOL,
                               atol=AB_ATOL,
                               err_msg=f"output mismatch for {case}/{kv_dtype}")
    np.testing.assert_allclose(collapsed_cache,
                               general_cache,
                               rtol=AB_RTOL,
                               atol=AB_ATOL,
                               err_msg=f"cache mismatch for {case}/{kv_dtype}")


@pytest.mark.parametrize("case", sorted(CASES))
def test_pallas_matches_reference(case):
    """The Pallas kernel must agree with the dense reference on every shape.

    This is the assertion `test_rpa_bundled.py` cannot make, because both of
    its sides route through `ragged_paged_attention`. It is the only check here
    that is independent of the kernel entirely, which is why it is kept even
    though the A/B above is the sharper instrument.

    bf16 only, and that is a real limitation rather than an oversight. Job
    j-18bdb2f9 ran this against fp8 with the planted signal and got up to 13.9%
    of elements out of tolerance on the *pristine* kernel: `float8_e4m3fn` has
    three mantissa bits, so a cache holding both POISON_V-scale values and
    N(0, 0.5) history quantizes coarsely at the top of that range, and the
    online-softmax and dense accumulations then diverge at the scale of the
    large term. Shrinking the signal to fit fp8 shrinks it below what a
    one-column error plants, so the two requirements are in direct conflict and
    there is no scale that satisfies both. fp8 is covered by the A/B above,
    where quantization is symmetric and cancels.
    """
    _require_tpu()
    spec = CASES[case]
    wl = _build(spec["seqs"],
                spec["distribution"],
                jnp.bfloat16,
                page_size=spec.get("page_size", PAGE_SIZE))

    got, got_cache = _call(K.ragged_paged_attention, wl, spec["bkv"])
    want, want_cache = _call(K.ref_ragged_paged_attention, wl, spec["bkv"])

    rows = _valid_rows(wl, spec["seqs"])
    np.testing.assert_allclose(got[rows],
                               want[rows],
                               rtol=RTOL,
                               atol=ATOL,
                               err_msg=f"output mismatch for {case}")
    np.testing.assert_allclose(got_cache,
                               want_cache,
                               rtol=RTOL,
                               atol=ATOL,
                               err_msg=f"cache mismatch for {case}")


def test_parity_resolves_a_kv_bound_error():
    """Negative control: the reference comparison must catch a loose kv bound.

    A green parity suite proves nothing unless the comparison can resolve the
    error class it is meant to guard. The *reference* is given a larger kv_len
    than the kernel for one sequence, so it reads a slot the kernel's bound
    correctly excludes. That slot is poisoned, so the deviation is ~POISON_V /
    kv_len rather than ~0.4%.

    The bump is 2, not 1, and it has to be. The new tokens are written at
    `kv_len - q_len`, so bumping kv_len by one advances the write position onto
    the very slot the extra read would have covered: the perturbed reference
    overwrites the poison with its own token before attending to it, and the
    only residue is one ordinary history column at weight ~1/kv_len, which is
    a deviation of order 0.002. Job j-18bdb2f9 measured exactly that and the
    control reported DID NOT RAISE. A bump of 2 leaves one poisoned slot inside
    the read extent and outside the write position. Decoupling the read extent
    from the write position to get a true one-column control is not expressible
    through `kv_lens` alone; the one-column case is covered instead by injecting
    the fault into the kernel and running the A/B above.
    """
    _require_tpu()
    spec = CASES["decode_multi_trip"]
    wl = _build(spec["seqs"], spec["distribution"], jnp.bfloat16)

    got, _ = _call(K.ragged_paged_attention, wl, spec["bkv"])

    perturbed = dict(wl)
    bumped = wl["kv_lens"].copy()
    bumped[0] += 2
    perturbed["kv_lens"] = bumped
    want, _ = _call(K.ref_ragged_paged_attention, perturbed, spec["bkv"])

    rows = _valid_rows(wl, spec["seqs"])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(got[rows], want[rows], rtol=RTOL, atol=ATOL)


