"""moe_tp_prefill against a same-math XLA reference and a float64 oracle.

Runs in the Pallas TPU interpreter on CPU (small shapes) and compiled on TPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental import moe_tp_prefill as mtp
from vllm_torchtpu.kernels.fused_moe.v2.kernel import rowquant_fp8

F8 = jnp.float8_e4m3fn
def _on_tpu():
    return jax.devices()[0].platform == "tpu"


def _q(w, axis):
    s = jnp.max(jnp.abs(w), axis=axis, keepdims=True) / 448.0
    return (w / s).astype(F8), s


def make_case(T, E, H, I, K, seed=0, skew=False):
    ks = jax.random.split(jax.random.key(seed), 8)
    x = (jax.random.normal(ks[0], (T, H), jnp.float32)).astype(jnp.bfloat16)
    w1, s1 = _q(jax.random.normal(ks[1], (E, H, 2 * I)) * 0.05, 1)
    w2, s2 = _q(jax.random.normal(ks[2], (E, I, H)) * 0.05, 1)
    w1s, s1s = _q(jax.random.normal(ks[3], (H, 2 * I)) * 0.05, 0)
    w2s, s2s = _q(jax.random.normal(ks[4], (I, H)) * 0.05, 0)
    wg = (jax.random.normal(ks[5], (H, 1)) * 0.05).astype(jnp.bfloat16)
    logits = jax.random.normal(ks[6], (T, E))
    if skew:
        logits = logits + jnp.linspace(3.0, 0.0, E)[None]
    tw, ti = jax.lax.top_k(jax.nn.softmax(logits, -1), K)
    tw = tw / tw.sum(-1, keepdims=True)
    return dict(x=x, w1=w1, w2=w2, s1=s1.reshape(E, 2 * I), s2=s2.reshape(E, H),
                w1s=w1s, w2s=w2s, s1s=s1s.reshape(-1), s2s=s2s.reshape(-1), wg=wg,
                tw=tw.astype(jnp.float32), ti=ti.astype(jnp.int32))


def run_kernel(c, n_valid=None, **kw):
    return mtp.moe_tp_prefill(c["x"], c["w1"], c["w2"], c["s1"], c["s2"], c["w1s"], c["w2s"],
                              c["s1s"], c["s2s"], c["wg"], c["tw"], c["ti"], n_valid=n_valid,
                              interpret=not _on_tpu(), **kw)


def same_math(c, n_valid=None):
    """The kernel's arithmetic, in XLA: per-row fp8 activations and mid."""
    x = c["x"]
    T = x.shape[0]
    xq, xs = rowquant_fp8(x)

    def ffn(w1, w2, s1, s2):
        h = jnp.dot(xq, w1, preferred_element_type=jnp.float32) * xs * s1[None]
        i = w1.shape[-1] // 2
        mid = (jax.nn.silu(h[:, :i]) * h[:, i:]).astype(jnp.bfloat16)
        mq, ms = rowquant_fp8(mid)
        y = jnp.dot(mq, w2, preferred_element_type=jnp.float32) * ms * s2[None]
        return y.astype(jnp.bfloat16).astype(jnp.float32)

    out = jnp.zeros((T, x.shape[1]), jnp.float32)
    E = c["w1"].shape[0]
    for e in range(E):
        y = ffn(c["w1"][e], c["w2"][e], c["s1"][e], c["s2"][e])
        wsel = jnp.sum(jnp.where(c["ti"] == e, c["tw"], 0.0), -1, keepdims=True)
        out = out + wsel * y
    g = jax.nn.sigmoid(jnp.dot(x, c["wg"], preferred_element_type=jnp.float32))
    out = out + g * ffn(c["w1s"], c["w2s"], c["s1s"], c["s2s"])
    if n_valid is not None:
        out = jnp.where((jnp.arange(T) < n_valid)[:, None], out, 0.0)
    return np.asarray(out)


def oracle(c, n_valid=None):
    f = lambda a: np.asarray(jnp.asarray(a).astype(jnp.float32), np.float64)
    x = f(c["x"])
    T = x.shape[0]
    E = c["w1"].shape[0]

    def ffn(xx, w1, w2):
        h = xx @ w1
        i = w1.shape[1] // 2
        return (h[:, :i] / (1 + np.exp(-h[:, :i])) * h[:, i:]) @ w2

    out = np.zeros_like(x)
    tw, ti = f(c["tw"]), np.asarray(c["ti"])
    for e in range(E):
        w1 = f(c["w1"][e]) * f(c["s1"][e])[None]
        w2 = f(c["w2"][e]) * f(c["s2"][e])[None]
        wsel = np.sum(np.where(ti == e, tw, 0.0), -1, keepdims=True)
        out += wsel * ffn(x, w1, w2)
    g = 1 / (1 + np.exp(-(x @ f(c["wg"]))))
    out += g * ffn(x, f(c["w1s"]) * f(c["s1s"])[None], f(c["w2s"]) * f(c["s2s"])[None])
    if n_valid is not None:
        out[n_valid:] = 0
    return out


def row_err(y, ref):
    return float(np.max(np.linalg.norm(y - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)))


SMALL = [  # (T, E, H, I, K, skew, n_valid)
    (40, 8, 256, 128, 3, False, None),
    (200, 8, 256, 128, 3, False, None),
    (300, 16, 256, 128, 4, True, None),
    (256, 8, 256, 128, 3, False, 131),
    (129, 4, 256, 128, 2, True, None),
]


@pytest.mark.parametrize("T,E,H,I,K,skew,n_valid", SMALL)
@pytest.mark.parametrize("nbuf", [2, 3])
def test_matches_same_math_and_oracle(T, E, H, I, K, skew, n_valid, nbuf):
    c = make_case(T, E, H, I, K, skew=skew)
    y = np.asarray(run_kernel(c, n_valid=n_valid, nbuf=nbuf).astype(jnp.float32))
    sm = same_math(c, n_valid)
    orc = oracle(c, n_valid)
    nv = T if n_valid is None else n_valid
    # Same arithmetic up to f32 summation order and one bf16 rounding.
    assert row_err(y[:nv], sm[:nv]) < 1e-2
    # fp8 activations: a few percent per row against exact math.
    assert row_err(y[:nv], orc[:nv]) < 8e-2
    if n_valid is not None:
        assert np.all(y[nv:] == 0)


def test_fault_dropped_expert_is_caught():
    """The gate must see a single expert's rows going missing."""
    c = make_case(200, 8, 256, 128, 3)
    y = np.asarray(run_kernel(c).astype(jnp.float32))
    c2 = dict(c)
    c2["w2"] = c["w2"].at[3].set(jnp.zeros_like(c["w2"][3]))
    sm = same_math(c2)
    assert row_err(y, sm) > 1e-2


def test_combine_chunks_equal():
    """Chunking the combine changes nothing but the f32 summation grouping."""
    rng = np.random.default_rng(0)
    T, H, K, n_rows = 256, 256, 4, 1500
    part = jnp.asarray(rng.standard_normal((n_rows, H // 128, 128)), jnp.bfloat16)
    pos = jnp.asarray(rng.integers(0, n_rows, (T, K)), jnp.int32)
    coef = jnp.asarray(rng.random((T, K)), jnp.float32)
    on_tpu = _on_tpu()
    a = np.asarray(mtp.combine(part, pos, coef, T, H, 1, None, not on_tpu).astype(jnp.float32))
    b = np.asarray(mtp.combine(part, pos, coef, T, H, 4, None, not on_tpu).astype(jnp.float32))
    ref = np.einsum("tk,tkh->th", np.asarray(coef, np.float64),
                    np.asarray(part.astype(jnp.float32), np.float64).reshape(n_rows, H)[np.asarray(pos)])
    np.testing.assert_allclose(a, b, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(a, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("ablate", [("gather",), ("compute",), ("write",), ("weights",),
                                    ("relayout_in",), ("relayout_out",),
                                    ("gather", "compute", "write", "weights")])
def test_ablation_arms_build_and_run(ablate):
    """Timing-only arms: they must build and terminate (values are wrong by design)."""
    c = make_case(200, 8, 256, 128, 3)
    y = run_kernel(c, ablate=frozenset(ablate))
    assert y.shape == (200, 256)


@pytest.mark.parametrize("stop", ["prep", "kernel"])
def test_stop_after_arms(stop):
    c = make_case(200, 8, 256, 128, 3)
    assert run_kernel(c, stop_after=stop).shape == (200, 256)


@pytest.mark.parametrize("n_valid", [None, 131])
def test_routing_plans_agree(n_valid):
    c = make_case(256, 8, 256, 128, 3, skew=True)
    xs = jnp.abs(c["x"][:, 0].astype(jnp.float32)) + 1.0
    a = mtp.routing_plan(c["ti"], 8, n_valid, row_scale=xs, plan="scatter")
    for plan in ("sort2", "sort2_cmp"):
        b = mtp.routing_plan(c["ti"], 8, n_valid, row_scale=xs, plan=plan)
        for u, v in zip(a, b):
            np.testing.assert_array_equal(np.asarray(u), np.asarray(v))


@pytest.mark.parametrize("unroll", [1, 2, 8])
def test_gather_unroll_bit_identical(unroll):
    c = make_case(300, 16, 256, 128, 4, skew=True)
    ref = np.asarray(run_kernel(c, gather_unroll=1, plan="scatter").astype(jnp.float32))
    y = np.asarray(run_kernel(c, gather_unroll=unroll).astype(jnp.float32))
    np.testing.assert_array_equal(y, ref)


def test_zero_weight_routes_are_skipped():
    """Padded bucket rows carry zero routing weight anywhere in the batch."""
    c = make_case(200, 8, 256, 128, 3)
    tw = c["tw"].at[17].set(0.0).at[150:].set(0.0).at[40, 1].set(0.0)
    c = dict(c, tw=tw)
    y = np.asarray(run_kernel(c).astype(jnp.float32))
    sm = same_math(c)
    assert row_err(y, sm) < 1e-2


def test_default_combine_chunks():
    assert mtp.default_combine_chunks(16384) == 4
    assert mtp.default_combine_chunks(3072) == 1
    assert mtp.default_combine_chunks(4096) == 1
    assert mtp.default_combine_chunks(8192) == 2


@pytest.mark.parametrize("tile_m,n_rungs", [(128, 4), (256, 1), (256, 4)])
def test_tile_height_and_rungs(tile_m, n_rungs):
    c = make_case(300, 16, 256, 128, 4, skew=True)
    y = np.asarray(run_kernel(c, tile_m=tile_m, n_rungs=n_rungs).astype(jnp.float32))
    assert row_err(y, same_math(c)) < 1e-2


@pytest.mark.parametrize("relayout", [("out_f32",), ("in_f32",), ("out_strided",), ("in_strided",)])
def test_relayout_variants_bit_identical(relayout):
    c = make_case(300, 16, 256, 128, 4, skew=True)
    y = np.asarray(run_kernel(c, relayout=frozenset(relayout)).astype(jnp.float32))
    assert row_err(y, same_math(c)) < 1e-2
