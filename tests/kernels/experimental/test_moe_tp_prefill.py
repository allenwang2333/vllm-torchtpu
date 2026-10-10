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


def test_expert_parallel_ranks_sum_to_oracle():
    """Each rank holds half the experts whole and half the shared expert's
    intermediate; the per-rank partials must sum to the full layer."""
    T, E, H, INTER, K, ranks = 200, 8, 256, 256, 3, 2
    c = make_case(T, E, H, INTER, K)
    e_loc, i_sh = E // ranks, INTER // ranks
    total = 0
    for r in range(ranks):
        ex, sh = slice(r * e_loc, (r + 1) * e_loc), slice(r * i_sh, (r + 1) * i_sh)
        gate_up = jnp.concatenate(
            [c["w1s"][:, :INTER][:, sh], c["w1s"][:, INTER:][:, sh]], axis=1
        )
        scale_up = jnp.concatenate([c["s1s"][:INTER][sh], c["s1s"][INTER:][sh]])
        rank_case = dict(
            c,
            w1=c["w1"][ex],
            w2=c["w2"][ex],
            s1=c["s1"][ex],
            s2=c["s2"][ex],
            w1s=gate_up,
            s1s=scale_up,
            w2s=c["w2s"][sh],
        )
        y = run_kernel(rank_case, expert_start=r * e_loc)
        total = total + np.asarray(y.astype(jnp.float32))
    assert row_err(total, oracle(c)) < 8e-2


BLOCKED_CASES = [  # (T, E, H, INTER, K, skew, n_valid, i_block, chunk_rows)
    (200, 8, 256, 512, 3, False, None, 128, 128),
    # skewed: the hot experts span several chunks
    (600, 8, 256, 512, 3, True, None, 256, 256),
    (256, 8, 256, 256, 3, False, 131, 128, 128),
    # 64-token multiples: the sparse combine kernel, not its XLA form
    (512, 8, 256, 512, 4, True, None, 256, 256),
    (768, 8, 256, 256, 3, False, 500, 128, 128),
    # hidden 1024: the down projection in two column chunks
    (256, 8, 1024, 512, 3, False, None, 256, 128),
]


@pytest.mark.parametrize("T,E,H,INTER,K,skew,n_valid,i_block,chunk_rows", BLOCKED_CASES)
def test_blocked_matches_oracle(T, E, H, INTER, K, skew, n_valid, i_block, chunk_rows):
    """Experts streamed in intermediate blocks over row chunks (the mode
    GLM-5.2's EP experts take), forced by i_block at a small shape."""
    c = make_case(T, E, H, INTER, K, skew=skew)
    y = run_kernel(c, n_valid=n_valid, i_block=i_block, chunk_rows=chunk_rows)
    y = np.asarray(y.astype(jnp.float32))
    nv = T if n_valid is None else n_valid
    assert row_err(y[:nv], oracle(c, n_valid)[:nv]) < 8e-2
    assert np.all(y[nv:] == 0)


def test_blocked_fault_dropped_expert_is_caught():
    c = make_case(200, 8, 256, 512, 3)
    y = np.asarray(run_kernel(c, i_block=128, chunk_rows=128).astype(jnp.float32))
    broken = dict(c, w2=c["w2"].at[3].set(jnp.zeros_like(c["w2"][3])))
    assert row_err(y, oracle(broken)) > 8e-2


def test_blocked_expert_parallel_ranks_sum_to_oracle():
    T, E, H, INTER, K, ranks = 512, 8, 256, 512, 3, 2
    c = make_case(T, E, H, INTER, K, skew=True)
    e_loc, i_sh = E // ranks, INTER // ranks
    total = 0
    for r in range(ranks):
        ex, sh = slice(r * e_loc, (r + 1) * e_loc), slice(r * i_sh, (r + 1) * i_sh)
        gate_up = jnp.concatenate(
            [c["w1s"][:, :INTER][:, sh], c["w1s"][:, INTER:][:, sh]], axis=1
        )
        scale_up = jnp.concatenate([c["s1s"][:INTER][sh], c["s1s"][INTER:][sh]])
        rank_case = dict(
            c,
            w1=c["w1"][ex],
            w2=c["w2"][ex],
            s1=c["s1"][ex],
            s2=c["s2"][ex],
            w1s=gate_up,
            s1s=scale_up,
            w2s=c["w2s"][sh],
        )
        y = run_kernel(rank_case, expert_start=r * e_loc, i_block=128, chunk_rows=128)
        total = total + np.asarray(y.astype(jnp.float32))
    assert row_err(total, oracle(c)) < 8e-2


def test_plan_blocks_only_oversized_experts():
    """Whole experts for Qwen3.5-397B (TP8 i 128, EP8 i 1024) and GLM-5.2 at
    TP16 (i 128) at both tile heights; blocks for GLM-5.2's EP experts."""
    for tile_m in (128, 256):
        args = (3, tile_m, 2, None, None, None)
        assert mtp.plan_blocked(4096, 128, 128, *args) is None
        assert mtp.plan_blocked(4096, 1024, 128, *args) is None
        assert mtp.plan_blocked(6144, 128, 128, *args) is None
        ib, nb, rows = mtp.plan_blocked(6144, 2048, 128, *args)
        assert 2048 % ib == 0 and nb >= 2 and rows % tile_m == 0 and rows >= tile_m


GLM_SCALE = 2.5


def oracle_glm(c, n_valid=None, scale_shared=False):
    """GLM-5.2 with the kernel off: routed * routed_scaling_factor + ungated
    shared expert. scale_shared=True is the wrong placement (scaled sum)."""
    no_shared = dict(c, w2s=jnp.zeros_like(c["w2s"]))
    routed = oracle(no_shared, n_valid)  # the gate only multiplies a zero row
    f = _f64
    x = f(c["x"])

    def ffn(xx, w1, w2):
        h = xx @ w1
        i = w1.shape[1] // 2
        return (h[:, :i] / (1 + np.exp(-h[:, :i])) * h[:, i:]) @ w2

    shared = ffn(x, f(c["w1s"]) * f(c["s1s"])[None], f(c["w2s"]) * f(c["s2s"])[None])
    if n_valid is not None:
        shared[n_valid:] = 0
    out = GLM_SCALE * routed + (GLM_SCALE if scale_shared else 1.0) * shared
    return out


GLM_CASES = [  # (T, INTER, i_block, chunk_rows, n_valid)
    (256, 512, 128, 128, None),  # 64-token multiple: sparse combine kernel
    (200, 512, 256, 128, None),  # XLA combine
    (512, 256, 128, 128, 300),
]


@pytest.mark.parametrize("T,INTER,i_block,chunk_rows,n_valid", GLM_CASES)
def test_glm_semantics_ungated_shared_routed_scale(
    T, INTER, i_block, chunk_rows, n_valid
):
    """GLM-5.2 semantics in the blocked mode: no shared-expert gate, routed
    scale 2.5 on the routed routes only, against the kernel-off composition,
    with the wrong-placement and gated-shared must-fail controls."""
    c = make_case(T, 8, 256, INTER, 3, skew=True)
    y = run_kernel(
        dict(c, wg=None),
        n_valid=n_valid,
        i_block=i_block,
        chunk_rows=chunk_rows,
        routed_scale=GLM_SCALE,
    )
    y = np.asarray(y.astype(jnp.float32))
    nv = T if n_valid is None else n_valid
    assert row_err(y[:nv], oracle_glm(c, n_valid)[:nv]) < 8e-2
    assert np.all(y[nv:] == 0)
    assert row_err(y[:nv], oracle_glm(c, n_valid, scale_shared=True)[:nv]) > 8e-2
    assert row_err(y[:nv], oracle(c, n_valid)[:nv]) > 8e-2  # gated, unscaled


def test_glm_semantics_expert_parallel_ranks_sum():
    """Two EP ranks, ungated shared expert TP-sliced, routed scale 2.5: the
    per-rank partials sum to the kernel-off layer."""
    T, E, H, INTER, K, ranks = 512, 8, 256, 512, 3, 2
    c = make_case(T, E, H, INTER, K, skew=True)
    e_loc, i_sh = E // ranks, INTER // ranks
    total = 0
    for r in range(ranks):
        ex, sh = slice(r * e_loc, (r + 1) * e_loc), slice(r * i_sh, (r + 1) * i_sh)
        gate_up = jnp.concatenate(
            [c["w1s"][:, :INTER][:, sh], c["w1s"][:, INTER:][:, sh]], axis=1
        )
        scale_up = jnp.concatenate([c["s1s"][:INTER][sh], c["s1s"][INTER:][sh]])
        rank_case = dict(
            c,
            w1=c["w1"][ex],
            w2=c["w2"][ex],
            s1=c["s1"][ex],
            s2=c["s2"][ex],
            w1s=gate_up,
            s1s=scale_up,
            w2s=c["w2s"][sh],
            wg=None,
        )
        y = run_kernel(
            rank_case,
            expert_start=r * e_loc,
            i_block=128,
            chunk_rows=128,
            routed_scale=GLM_SCALE,
        )
        total = total + np.asarray(y.astype(jnp.float32))
    assert row_err(total, oracle_glm(c)) < 8e-2
    assert row_err(total, oracle_glm(c, scale_shared=True)) > 8e-2
