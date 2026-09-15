"""What RPA v3 requires of the KV-cache rows past ``effective_kv_len``.

A page holds more rows than the sequence occupying it, and the kernel reads
the whole page. ``RPA_V_MASK_MODE=full`` zeroes those rows explicitly; the
default ``none`` drops that and leans on the score mask having already driven
``p = exp(s - m_curr)`` to exactly zero there. Eliding it measured -8.35%
median / -8.49% min on the decode kernel (``j-30133ae0``).

**The two are not equivalent, and this file draws the line.** ``p == 0``
annihilates a *finite* V and nothing else: ``0 * NaN`` and ``0 * inf`` are
both ``NaN``. So ``none`` is safe for any finite padding no matter how large
-- ``0 * 3e38`` is 0 -- and unsafe only for a non-finite one. That is a much
weaker requirement than "the padding is zeroed", and it is the requirement
production actually meets, at the allocator: see
``tests/test_kv_cache_zero_init.py``.

The dependency is pinned in both directions on purpose. Asserting only that
``none`` works on good data would let someone widen the padding's provenance
without noticing what they had broken, which is precisely what happened once
already: ``torch.empty`` on TPU fp8 returns 100% non-finite bytes (measured,
``j-761bc2df``), so for two commits the default was producing NaN.

**These tests fail if the elision is wrong.** The suite that shipped with it
could not. ``test_rpa_skip_kv_update`` builds its cache with ``np.zeros`` and
writes only ``[:kv_len]``, so the rows the mask targets are already 0.0 and
``where(..., v, 0.0)`` is a literal no-op on that data; and it compares Pallas
against *itself* rather than against the reference. Here the padding is
hostile and the oracle is ``ref_ragged_paged_attention``, which slices
``kv[:kv_len]`` and so cannot see the junk by construction.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as K

# kv_len 33 against page_size 16 leaves the sequence ending mid-page, so rows
# 33..127 are padding and the last kv block runs off the end of the sequence.
PAGE_SIZE, TOTAL_PAGES = 16, 8
KV_LEN, Q_LEN = 33, 1
NUM_KV_HEADS, NUM_Q_HEADS, HEAD_DIM = 1, 4, 128

# Queries are bf16 even against an fp8 cache: that is what production does,
# and the kernel asserts k/v match the *cache* dtype (v3/kernel.py:1425).
Q_DTYPE = jnp.bfloat16
KV_DTYPES = [jnp.bfloat16, jnp.float8_e4m3fn]
KV_DTYPE_IDS = ["bf16", "fp8"]

# Finite, and as far from zero as the dtype goes -- `p == 0` must annihilate
# these exactly, however large. Resolved against the dtype rather than fixed,
# because "large" is not a dtype-independent notion: bf16 tops out near
# 3.39e38 and f8e4m3fn at 448, and 3e38 written into an fp8 cache is not a big
# finite number, it is NaN (f8e4m3fn has no infinity and does not saturate).
# Measured the hard way in j-f636426b, where a fixed 3e38 turned this into a
# non-finite case and failed.
FINITE_POISONS = {
    "zero": lambda dt: 0.0,
    "dtype_max": lambda dt: float(jnp.finfo(dt).max),
    "neg_dtype_max": lambda dt: -float(jnp.finfo(dt).max),
}
# Not finite in any dtype. `p == 0` cannot annihilate these -- 0 * NaN is NaN.
NONFINITE_POISONS = {"nan": np.nan, "posinf": np.inf, "neginf": -np.inf}


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.skip(f"Requires TPU backend, found {backend!r}.")


def _poisoned_cache(target_k, target_v, kv_dtype, poison):
    """Pack [0, KV_LEN) normally and fill every later row with `poison`."""
    shape = K.get_kv_cache_shape(TOTAL_PAGES, PAGE_SIZE, NUM_KV_HEADS,
                                 HEAD_DIM, kv_dtype)
    merged = np.asarray(
        K.merge_kv(jnp.asarray(target_k, kv_dtype),
                   jnp.asarray(target_v, kv_dtype)).astype(jnp.float32))
    cache = np.full(shape, poison, np.float32)
    flat = cache.reshape(TOTAL_PAGES * PAGE_SIZE, *shape[2:])
    flat[:KV_LEN] = merged
    return flat.reshape(shape)


def _attend(mode, poison, kv_dtype):
    """Run Pallas and the reference against the same poisoned cache."""
    rng = np.random.default_rng(0)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    target_k = r(KV_LEN, NUM_KV_HEADS, HEAD_DIM)
    target_v = r(KV_LEN, NUM_KV_HEADS, HEAD_DIM)
    query = r(Q_LEN, NUM_Q_HEADS, HEAD_DIM)
    cache_np = _poisoned_cache(target_k, target_v, kv_dtype, poison)

    def run(fn):
        # Both entry points donate their inputs, so each call needs its own.
        out, _ = fn(
            jnp.asarray(query, Q_DTYPE),
            jnp.asarray(target_k[KV_LEN - Q_LEN:KV_LEN], kv_dtype),
            jnp.asarray(target_v[KV_LEN - Q_LEN:KV_LEN], kv_dtype),
            jnp.asarray(cache_np, kv_dtype),
            jnp.array([KV_LEN], jnp.int32),
            jnp.arange(TOTAL_PAGES, dtype=jnp.int32),
            jnp.array([0, Q_LEN], jnp.int32),
            jnp.asarray((1, 1, 1), jnp.int32),
            sm_scale=HEAD_DIM**-0.5,
            # Leave the poisoned cache exactly as built.
            skip_kv_update=True,
        )
        return np.asarray(out.astype(jnp.float32))

    got = run(K.ragged_paged_attention)
    # The reference slices kv[:kv_len], so the poison is unreachable for it by
    # construction -- it is an oracle, not a second opinion.
    want = run(K.ref_ragged_paged_attention)
    return got, want


def _set_mode(monkeypatch, mode):
    monkeypatch.setenv("RPA_V_MASK_MODE", mode)
    # The mode is read at trace time, so a cached executable would silently
    # re-measure the previous parametrization.
    jax.clear_caches()
    assert envs.RPA_V_MASK_MODE == mode


@pytest.mark.parametrize("kv_dtype", KV_DTYPES, ids=KV_DTYPE_IDS)
@pytest.mark.parametrize("poison", list(FINITE_POISONS),
                         ids=list(FINITE_POISONS))
@pytest.mark.parametrize("mode", ["none", "full"])
def test_finite_padding_is_annihilated_under_both_modes(
        monkeypatch, mode, poison, kv_dtype):
    """The production invariant: finite padding cannot reach the output.

    Magnitude is irrelevant, which is the whole reason `none` is viable --
    `p` is exactly 0.0 past `effective_kv_len`, and `0 * 3e38` is 0.
    """
    _require_tpu()
    _set_mode(monkeypatch, mode)
    value = FINITE_POISONS[poison](kv_dtype)
    # If the value is not finite once it is in the cache, this test is
    # measuring the wrong thing and would read as a kernel bug.
    assert np.isfinite(
        np.asarray(jnp.asarray(value, kv_dtype).astype(jnp.float32))), (
            f"{poison} resolved to {value!r}, which is not finite in "
            f"{kv_dtype.__name__}; fix the parametrization, not the kernel")
    got, want = _attend(mode, value, kv_dtype)

    assert np.isfinite(got).all(), (
        f"finite padding ({poison}={value:g}) produced a non-finite output "
        f"under RPA_V_MASK_MODE={mode}: "
        f"{np.count_nonzero(~np.isfinite(got))}/{got.size} non-finite")
    np.testing.assert_allclose(got, want, atol=2e-2, rtol=0)


@pytest.mark.parametrize("kv_dtype", KV_DTYPES, ids=KV_DTYPE_IDS)
@pytest.mark.parametrize("poison", list(NONFINITE_POISONS),
                         ids=list(NONFINITE_POISONS))
def test_full_absorbs_nonfinite_padding(monkeypatch, poison, kv_dtype):
    """The control. `full` is robust to anything, because it overwrites V.

    If this ever fails, the test below proves nothing: it would mean the
    poison is not reaching the kernel rather than that `none` is fragile.
    """
    _require_tpu()
    _set_mode(monkeypatch, "full")
    got, want = _attend("full", NONFINITE_POISONS[poison], kv_dtype)

    assert np.isfinite(got).all(), (
        f"RPA_V_MASK_MODE=full failed to absorb {poison}: "
        f"{np.count_nonzero(~np.isfinite(got))}/{got.size} non-finite")
    np.testing.assert_allclose(got, want, atol=2e-2, rtol=0)


@pytest.mark.parametrize("kv_dtype", KV_DTYPES, ids=KV_DTYPE_IDS)
@pytest.mark.parametrize("poison", list(NONFINITE_POISONS),
                         ids=list(NONFINITE_POISONS))
def test_none_requires_finite_padding_and_says_so_loudly(
        monkeypatch, poison, kv_dtype):
    """`none` does NOT tolerate non-finite padding. Pinned deliberately.

    This is an assertion about a limitation, which is unusual, and it is here
    because the limitation is the price of the -8.5%. Two things it buys:

    * It documents the contract at the boundary that depends on it, so a
      reader cannot mistake `none` for unconditionally safe.
    * It fails if someone makes `none` robust by accident -- at which point
      the zeroing has crept back in and the speedup is gone. A green run here
      is also a statement that we are still paying nothing for this.

    The failure mode is loud, not subtle: the whole output row goes NaN, so a
    violated invariant surfaces immediately rather than as quiet drift.
    """
    _require_tpu()
    _set_mode(monkeypatch, "none")
    got, _ = _attend("none", NONFINITE_POISONS[poison], kv_dtype)

    assert not np.isfinite(got).all(), (
        f"RPA_V_MASK_MODE=none absorbed {poison} padding. Either the V "
        f"zeroing is back (check mask_v; the -8.5% is gone) or the kernel "
        f"stopped reading the rows past effective_kv_len. Both make this "
        f"file's premise stale -- re-derive it before deleting the test.")


@pytest.mark.parametrize("kv_dtype", KV_DTYPES, ids=KV_DTYPE_IDS)
def test_the_poison_survives_the_cast_into_the_cache(kv_dtype):
    """Guards every test above from passing vacuously.

    If the cast to the cache dtype quietly turned the poison finite -- fp8
    e4m3fn has no infinity, so a saturating cast is a real possibility -- the
    assertions above would hold no matter what the kernel did. Measured: it
    does not saturate, it produces NaN, which is equally non-finite and
    equally fatal to `p == 0`. Runs on CPU; no TPU needed.
    """
    zero = np.zeros((KV_LEN, NUM_KV_HEADS, HEAD_DIM), np.float32)
    for name, poison in NONFINITE_POISONS.items():
        cache = np.asarray(
            jnp.asarray(_poisoned_cache(zero, zero, kv_dtype, poison),
                        kv_dtype).astype(jnp.float32))
        flat = cache.reshape(TOTAL_PAGES * PAGE_SIZE, *cache.shape[2:])
        assert np.isfinite(flat[:KV_LEN]).all(), f"{name}: real rows dirty"
        assert not np.isfinite(flat[KV_LEN:]).any(), (
            f"{name} became finite in {kv_dtype.__name__}; every "
            f"non-finite test above is vacuous")
