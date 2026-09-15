"""The V rows past ``effective_kv_len`` must not reach the output.

A page holds more rows than the sequence occupying it, and RPA v3 reads the
whole page. It used to zero those rows explicitly; it now relies on the score
mask having already driven ``p`` to exactly zero there, which is ~8.5% faster
on the decode kernel (``j-30133ae0``). The two are equivalent only while the
junk is finite -- ``0 * NaN`` is ``NaN``, and ``0xFF`` is a NaN in
``f8e4m3fn`` -- so this file pins the property the speedup depends on.

**These tests fail if the elision is wrong, which is the point.** The suite
that shipped alongside the change could not: ``test_rpa_skip_kv_update``
builds its cache with ``np.zeros`` and then writes only ``[:kv_len]``, so the
rows the mask targets are already 0.0 and ``where(..., v, 0.0)`` is a literal
no-op on that data; and it compares the Pallas kernel against *itself* under a
different layout rather than against the reference. Here the padding is
hostile and the oracle is ``ref_ragged_paged_attention``, which slices
``kv[:kv_len]`` and so cannot see the junk by construction.

Both mask modes are asserted. ``full`` passing is the control: it says the
poison is reaching the kernel at all, so a green ``none`` means something.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as K

# Enough pages that the sequence ends mid-page and leaves a poisoned tail.
PAGE_SIZE, TOTAL_PAGES = 16, 8
KV_LEN, Q_LEN = 33, 1
NUM_KV_HEADS, NUM_Q_HEADS, HEAD_DIM = 1, 4, 128

# Each must survive `p == 0`. Measured round-trip through the cache dtype:
# bf16 keeps all four as written (`huge` stays 3.004e38, finite); f8e4m3fn has
# no infinity and does not saturate, so inf/-inf/3e38 all arrive as NaN. So
# `huge` is the only one that probes finite overflow, and it does so in bf16
# only -- on fp8 the four cases collapse to one. That is fine; it is the fp8
# path that motivated this file, and NaN is the encoding that path produces.
POISONS = {
    "nan": np.nan,
    "posinf": np.inf,
    "neginf": -np.inf,
    "huge": 3.0e38,
}


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.skip(f"Requires TPU backend, found {backend!r}.")


def _poisoned_cache(target_k, target_v, dtype, poison):
    """Pack [0, KV_LEN) normally and fill every later row with `poison`."""
    shape = K.get_kv_cache_shape(TOTAL_PAGES, PAGE_SIZE, NUM_KV_HEADS,
                                 HEAD_DIM, dtype)
    merged = np.asarray(
        K.merge_kv(jnp.asarray(target_k, dtype),
                   jnp.asarray(target_v, dtype)).astype(jnp.float32))
    cache = np.full(shape, poison, np.float32)
    flat = cache.reshape(TOTAL_PAGES * PAGE_SIZE, *shape[2:])
    flat[:KV_LEN] = merged
    return flat.reshape(shape)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn],
                         ids=["bf16", "fp8"])
@pytest.mark.parametrize("poison", list(POISONS), ids=list(POISONS))
@pytest.mark.parametrize("mode", ["none", "full"])
def test_poisoned_padding_cannot_reach_the_output(monkeypatch, mode, poison,
                                                  dtype):
    _require_tpu()
    monkeypatch.setenv("RPA_V_MASK_MODE", mode)
    # The mode is read at trace time, so a cached executable would silently
    # measure the previous parametrization.
    jax.clear_caches()
    assert envs.RPA_V_MASK_MODE == mode

    rng = np.random.default_rng(0)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    target_k = r(KV_LEN, NUM_KV_HEADS, HEAD_DIM)
    target_v = r(KV_LEN, NUM_KV_HEADS, HEAD_DIM)
    query = r(Q_LEN, NUM_Q_HEADS, HEAD_DIM)
    cache_np = _poisoned_cache(target_k, target_v, dtype, POISONS[poison])

    def run(fn):
        # Both entry points donate their inputs, so each call needs its own.
        out, _ = fn(
            jnp.asarray(query, dtype),
            jnp.asarray(target_k[KV_LEN - Q_LEN:KV_LEN], dtype),
            jnp.asarray(target_v[KV_LEN - Q_LEN:KV_LEN], dtype),
            jnp.asarray(cache_np, dtype),
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
    # The reference slices kv[:kv_len], so the poison is unreachable for it
    # by construction -- it is an oracle, not a second opinion.
    want = run(K.ref_ragged_paged_attention)

    assert np.isfinite(got).all(), (
        f"{poison} in the rows past kv_len reached the output under "
        f"RPA_V_MASK_MODE={mode}: "
        f"{np.count_nonzero(~np.isfinite(got))}/{got.size} non-finite")
    np.testing.assert_allclose(got, want, atol=2e-2, rtol=0)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn],
                         ids=["bf16", "fp8"])
def test_the_poison_survives_the_cast_into_the_cache(dtype):
    """Guards every test above from passing vacuously.

    If the cast to the cache dtype quietly turned the poison finite -- fp8
    e4m3fn has no infinity, so a saturating cast is a real possibility -- the
    assertions above would hold no matter what the kernel did. Runs on CPU.
    """
    zero = np.zeros((KV_LEN, NUM_KV_HEADS, HEAD_DIM), np.float32)
    cache = np.asarray(
        jnp.asarray(_poisoned_cache(zero, zero, dtype, np.nan),
                    dtype).astype(jnp.float32))
    flat = cache.reshape(TOTAL_PAGES * PAGE_SIZE, *cache.shape[2:])
    assert np.isfinite(flat[:KV_LEN]).all(), "the real rows must be clean"
    assert np.isnan(flat[KV_LEN:]).all(), "the padding must actually be NaN"
