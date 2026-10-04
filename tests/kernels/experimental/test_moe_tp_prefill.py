# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""moe_tp_prefill against a float64 oracle.

Runs in the Pallas TPU interpreter on CPU (small shapes) and compiled on TPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental import moe_tp_prefill as mtp

F8 = jnp.float8_e4m3fn


def _on_tpu():
    return jax.devices()[0].platform == "tpu"


def _q(w, axis):
    s = jnp.max(jnp.abs(w), axis=axis, keepdims=True) / 448.0
    return (w / s).astype(F8), s


def make_case(T, E, H, INTER, K, seed=0, skew=False):
    ks = jax.random.split(jax.random.key(seed), 8)
    x = (jax.random.normal(ks[0], (T, H), jnp.float32)).astype(jnp.bfloat16)
    w1, s1 = _q(jax.random.normal(ks[1], (E, H, 2 * INTER)) * 0.05, 1)
    w2, s2 = _q(jax.random.normal(ks[2], (E, INTER, H)) * 0.05, 1)
    w1s, s1s = _q(jax.random.normal(ks[3], (H, 2 * INTER)) * 0.05, 0)
    w2s, s2s = _q(jax.random.normal(ks[4], (INTER, H)) * 0.05, 0)
    wg = (jax.random.normal(ks[5], (H, 1)) * 0.05).astype(jnp.bfloat16)
    logits = jax.random.normal(ks[6], (T, E))
    if skew:
        logits = logits + jnp.linspace(3.0, 0.0, E)[None]
    tw, ti = jax.lax.top_k(jax.nn.softmax(logits, -1), K)
    tw = tw / tw.sum(-1, keepdims=True)
    return dict(
        x=x,
        w1=w1,
        w2=w2,
        s1=s1.reshape(E, 2 * INTER),
        s2=s2.reshape(E, H),
        w1s=w1s,
        w2s=w2s,
        s1s=s1s.reshape(-1),
        s2s=s2s.reshape(-1),
        wg=wg,
        tw=tw.astype(jnp.float32),
        ti=ti.astype(jnp.int32),
    )


def run_kernel(c, n_valid=None, **kw):
    return mtp.moe_tp_prefill(
        c["x"],
        c["w1"],
        c["w2"],
        c["s1"],
        c["s2"],
        c["w1s"],
        c["w2s"],
        c["s1s"],
        c["s2s"],
        c["wg"],
        c["tw"],
        c["ti"],
        n_valid=n_valid,
        interpret=not _on_tpu(),
        **kw,
    )


def _f64(a):
    return np.asarray(jnp.asarray(a).astype(jnp.float32), np.float64)


def oracle(c, n_valid=None):
    f = _f64
    x = f(c["x"])
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
    return float(
        np.max(
            np.linalg.norm(y - ref, axis=1)
            / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
        )
    )


CASES = [  # (T, E, H, INTER, K, skew, n_valid)
    (200, 8, 256, 128, 3, False, None),
    (300, 16, 256, 128, 4, True, None),
    (256, 8, 256, 128, 3, False, 131),
]


@pytest.mark.parametrize("T,E,H,INTER,K,skew,n_valid", CASES)
def test_matches_oracle(T, E, H, INTER, K, skew, n_valid):
    c = make_case(T, E, H, INTER, K, skew=skew)
    y = np.asarray(run_kernel(c, n_valid=n_valid).astype(jnp.float32))
    nv = T if n_valid is None else n_valid
    # fp8 activations: a few percent per row against exact math.
    assert row_err(y[:nv], oracle(c, n_valid)[:nv]) < 8e-2
    assert np.all(y[nv:] == 0)


def test_fault_dropped_expert_is_caught():
    """The bound must see a single expert's rows going missing."""
    c = make_case(200, 8, 256, 128, 3)
    y = np.asarray(run_kernel(c).astype(jnp.float32))
    broken = dict(c, w2=c["w2"].at[3].set(jnp.zeros_like(c["w2"][3])))
    assert row_err(y, oracle(broken)) > 8e-2
