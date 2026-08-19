# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""On-device gate for the fused EP MoE kernel: same answer, and how much faster.

The CPU gates cover compilation and arithmetic; only a chip can say whether the
in-kernel all-to-all actually delivers each row to the rank that owns it, and
what it costs. Both tests run the two MoE paths in one process against one
clock, so the comparison is not a cross-job artifact.

Run on an 8-chip node:
    pytest tests/kernels/fused_moe/test_fused_ep_device.py -m multichip -s
"""

import functools
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v1.kernel import fused_ep_moe, ref_moe
from vllm_torchtpu.layers.common.fused_moe_gmm import fused_moe_func

pytestmark = [
    pytest.mark.multichip,
    # Pallas on a CPU host only runs in interpret mode, which this kernel trips
    # (see the numerics test module). Skip rather than fail when the suite is
    # run without `-m "not multichip"`.
    pytest.mark.skipif(jax.default_backend() != "tpu",
                       reason="fused EP kernel needs a TPU backend"),
]

EP_AXIS = "model"
FP8 = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8).max)

# Qwen3.5-397B-A17B-FP8 on one v7x 2x2x1 node. `local_tokens` 2048 is the
# 16384-token full-prefill point of every InferenceX sharding.
HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K = 4096, 1024, 512, 10


def _mesh(ep_size):
    devices = np.asarray(jax.devices()[:ep_size]).reshape(1, ep_size)
    return Mesh(devices,
                axis_names=("data", EP_AXIS),
                axis_types=(AxisType.Explicit, AxisType.Explicit))


def _weights(seed, num_experts, hidden, inter):
    """Per-output-channel fp8, in the layout the FP8 loader produces."""
    rng = np.random.default_rng(seed)
    w13 = rng.normal(0, 0.05, (num_experts, hidden, 2 * inter))
    w2 = rng.normal(0, 0.05, (num_experts, inter, hidden))
    s13 = np.abs(w13).max(axis=1, keepdims=True) / FP8_MAX
    s2 = np.abs(w2).max(axis=1, keepdims=True) / FP8_MAX
    return (jnp.asarray(w13 / s13, FP8),
            jnp.asarray(s13[:, None], jnp.float32).reshape(
                num_experts, 1, 1, 2 * inter), jnp.asarray(w2 / s2, FP8),
            jnp.asarray(s2, jnp.float32).reshape(num_experts, 1, 1, hidden))


def _shard(x, mesh, spec=P(EP_AXIS)):
    return jax.device_put(x, NamedSharding(mesh, spec))


@functools.lru_cache(maxsize=None)
def _gmm_reference_fn(mesh, ep_size, local_experts):
    """Build the jitted reference ONCE per (mesh, sharding).

    `@jax.jit` on a closure defined inside the caller mints a fresh function
    object per call, so JAX's compilation cache misses every time and a timing
    loop measures the compiler. That is not hypothetical: it is why this file
    once reported `gmm=11102.307 ms` and "182x / 2911x" speedups, all of which
    were compile time (`rh0syw4w/w4/CORRECTION.md` §4). The v2 device ladder
    imports this function for its `gmm_ms` column, so the defect propagates.
    """

    @jax.jit
    @jax.shard_map(mesh=mesh,
                   in_specs=(P(EP_AXIS), ) * 6,
                   out_specs=P(EP_AXIS),
                   check_vma=False)
    def run(tokens, logits, w13, s13, w2, s2):
        gathered_tokens = jax.lax.all_gather(tokens, EP_AXIS, tiled=True)
        gathered_logits = jax.lax.all_gather(logits, EP_AXIS, tiled=True)
        probs = jax.nn.softmax(gathered_logits.astype(jnp.float32), axis=-1)
        weights, ids = jax.lax.top_k(probs, TOP_K)
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
        experts_start = jax.lax.axis_index(EP_AXIS).astype(
            jnp.int32) * local_experts
        out = fused_moe_func(
            hidden_states=gathered_tokens,
            w1=w13,
            w2=w2,
            w1_scale=s13,
            w2_scale=s2,
            w1_bias=None,
            w2_bias=None,
            topk_weights=weights.astype(tokens.dtype),
            topk_ids=ids.astype(jnp.int32),
            experts_start=experts_start,
            topk=TOP_K,
            activation="silu",
            use_ep=True,
        )
        return jax.lax.psum_scatter(out, EP_AXIS, tiled=True)

    return run


def _gmm_reference(tokens, logits, w13, s13, w2, s2, mesh, ep_size):
    """What the general path computes: all-gather, mask, GMM, reduce-scatter.

    Modelled with plain collectives so the comparison is against the *semantics*
    the fused kernel replaces, not against another Pallas kernel's tiling.
    """
    run = _gmm_reference_fn(mesh, ep_size, w2.shape[0] // ep_size)
    return run(tokens, logits, w13, s13, w2, s2)


def _fused(tokens, logits, w13, s13, w2, s2, mesh, quant_act):
    return fused_ep_moe(mesh=mesh,
                        tokens=tokens,
                        w1=w13,
                        w2=w2,
                        gating_output=logits,
                        top_k=TOP_K,
                        renormalize_topk_logits=True,
                        act_fn="silu",
                        scoring_fn="softmax",
                        subc_quant_w1_sz=HIDDEN,
                        subc_quant_w2_sz=INTERMEDIATE,
                        quant_act=quant_act,
                        w13_concat_layout=True,
                        w1_scale=s13,
                        w2_scale=s2,
                        ep_axis_name=EP_AXIS)


def _inputs(mesh, ep_size, local_tokens, seed=0):
    rng = np.random.default_rng(seed)
    tokens = jnp.asarray(
        rng.normal(0, 1, (local_tokens * ep_size, HIDDEN)) / 4, jnp.bfloat16)
    logits = jnp.asarray(rng.normal(0, 1, (local_tokens * ep_size,
                                           NUM_EXPERTS)), jnp.bfloat16)
    w13, s13, w2, s2 = _weights(seed + 1, NUM_EXPERTS, HIDDEN, INTERMEDIATE)
    return tuple(
        _shard(x, mesh) for x in (tokens, logits, w13, s13, w2, s2))


@pytest.mark.parametrize("local_tokens", [2048])
def test_fused_ep_matches_the_gmm_path_on_device(local_tokens):
    ep_size = len(jax.devices())
    if ep_size < 8:
        pytest.skip(f"needs 8 devices, saw {ep_size}")
    mesh = _mesh(ep_size)
    tokens, logits, w13, s13, w2, s2 = _inputs(mesh, ep_size, local_tokens)

    fused = np.asarray(jax.device_get(
        _fused(tokens, logits, w13, s13, w2, s2, mesh, True)), np.float32)
    ref = np.asarray(
        jax.device_get(
            _gmm_reference(tokens, logits, w13, s13, w2, s2, mesh, ep_size)),
        np.float32)

    diff = np.abs(fused - ref)
    rel_l2 = np.linalg.norm(fused - ref) / np.linalg.norm(ref)
    print(f"[device local_tokens={local_tokens}] max_abs={diff.max():.4g} "
          f"rel_l2={rel_l2:.4g} ref_absmax={np.abs(ref).max():.4g}")
    # Reported, not asserted: the two paths make *different* rounding choices,
    # so agreement between them is not the right bar. gmm_v2 quantizes the lhs
    # to fp8 and then accumulates in **bf16** (gmm_v2.py:1245-1252) with the
    # gmm1 intermediate round-tripping HBM in bf16, while the fused kernel
    # accumulates in f32 and keeps the intermediate in registers. Which one is
    # closer to the truth is settled against `ref_moe` by
    # test_which_mechanism_disagrees / test_the_gmm_reference_arm_agrees_with_ref_moe;
    # this number only has to be small enough that neither is broken.
    assert rel_l2 < 0.2, "the two MoE paths disagree beyond fp8 rounding"


@pytest.mark.parametrize("local_tokens", [2048])
def test_report_layer_time_against_the_gmm_path(local_tokens):
    """One clock, one process: per-layer device time of both paths."""
    ep_size = len(jax.devices())
    if ep_size < 8:
        pytest.skip(f"needs 8 devices, saw {ep_size}")
    mesh = _mesh(ep_size)
    args = _inputs(mesh, ep_size, local_tokens)

    def bench(fn, iters=20):
        out = fn()
        jax.block_until_ready(out)
        start = time.perf_counter()
        for _ in range(iters):
            out = fn()
        jax.block_until_ready(out)
        return (time.perf_counter() - start) / iters * 1e3

    fused_ms = bench(lambda: _fused(*args, mesh, True))
    gmm_ms = bench(lambda: _gmm_reference(*args, mesh, ep_size))
    print(f"[device local_tokens={local_tokens} node={local_tokens * ep_size}] "
          f"fused={fused_ms:.3f} ms  gmm={gmm_ms:.3f} ms  "
          f"speedup={gmm_ms / fused_ms:.3f}x")


# --------------------------------------------------------------------------
# Diagnostics. The serving-shape check above compares two Pallas programs, so a
# disagreement does not say which one is wrong. These run a third,
# Pallas-free arm (`ref_moe`, the kernel file's own fp32 reference) at a shape
# small enough for its per-token Python loop, and toggle one mechanism at a
# time. Small shapes, one compile each -- the whole block is minutes.
# --------------------------------------------------------------------------
SMALL = dict(hidden=512, inter=256, experts=16, top_k=2, local_tokens=8)


def _small_inputs(ep_size, seed=0):
    """Both weight layouts from one source, plus tokens and logits."""
    rng = np.random.default_rng(seed)
    h, f, e = SMALL["hidden"], SMALL["inter"], SMALL["experts"]
    tokens = jnp.asarray(
        rng.normal(0, 1, (SMALL["local_tokens"] * ep_size, h)) / 4,
        jnp.bfloat16)
    logits = jnp.asarray(rng.normal(0, 1, (SMALL["local_tokens"] * ep_size, e)),
                         jnp.bfloat16)
    native = rng.normal(0, 0.05, (e, 2, h, f))
    w2 = rng.normal(0, 0.05, (e, f, h))
    ns = np.abs(native).max(axis=2, keepdims=True) / FP8_MAX      # (e,2,1,f)
    s2 = np.abs(w2).max(axis=1, keepdims=True) / FP8_MAX          # (e,1,h)
    q_native = jnp.asarray(native / ns, FP8)
    q_concat = jnp.asarray(np.concatenate([np.asarray(q_native[:, 0]),
                                           np.asarray(q_native[:, 1])], -1),
                           FP8)
    s_native = jnp.asarray(ns[:, :, :, None, :], jnp.float32)     # (e,2,1,1,f)
    s_concat = jnp.asarray(np.concatenate([ns[:, 0], ns[:, 1]], -1)[:, :, None],
                           jnp.float32).reshape(e, 1, 1, 2 * f)
    return (tokens, logits, q_native, s_native, q_concat, s_concat,
            jnp.asarray(w2 / s2, FP8),
            jnp.asarray(s2, jnp.float32).reshape(e, 1, 1, h))


def _small_ref(tokens, logits, q_native, s_native, q2, s2):
    return np.asarray(jax.device_get(
        ref_moe(tokens,
                q_native,
                q2,
                logits,
                SMALL["top_k"],
                renormalize_topk_logits=True,
                act_fn="silu",
                scoring_fn="softmax",
                subc_quant_w1_sz=SMALL["hidden"],
                subc_quant_w2_sz=SMALL["inter"],
                w1_scale=s_native,
                w2_scale=s2)), np.float32)


def _small_fused(mesh, tokens, logits, w1, s1, q2, s2, *, concat, quant_act):
    out = fused_ep_moe(mesh=mesh,
                       tokens=_shard(tokens, mesh),
                       w1=_shard(w1, mesh),
                       w2=_shard(q2, mesh),
                       gating_output=_shard(logits, mesh),
                       top_k=SMALL["top_k"],
                       renormalize_topk_logits=True,
                       act_fn="silu",
                       scoring_fn="softmax",
                       subc_quant_w1_sz=SMALL["hidden"],
                       subc_quant_w2_sz=SMALL["inter"],
                       quant_act=quant_act,
                       w13_concat_layout=concat,
                       w1_scale=_shard(s1, mesh),
                       w2_scale=_shard(s2, mesh),
                       ep_axis_name=EP_AXIS)
    return np.asarray(jax.device_get(out), np.float32)


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("ep_size,concat,quant_act", [
    (1, True, True),    # no all-to-all at all: is the local compute right?
    (8, True, True),    # the shipping configuration
    (8, False, True),   # the kernel's own w13 layout
    (8, True, False),   # no in-kernel activation quantization
    (8, False, False),  # closest to the vendored kernel: neither local change
])
def test_which_mechanism_disagrees(ep_size, concat, quant_act):
    if len(jax.devices()) < ep_size:
        pytest.skip(f"needs {ep_size} devices")
    mesh = _mesh(ep_size)
    (tokens, logits, q_native, s_native, q_concat, s_concat, q2,
     s2) = _small_inputs(ep_size)
    reference = _small_ref(tokens, logits, q_native, s_native, q2, s2)
    fused = _small_fused(mesh,
                         tokens,
                         logits,
                         q_concat if concat else q_native,
                         s_concat if concat else s_native,
                         q2,
                         s2,
                         concat=concat,
                         quant_act=quant_act)
    rel = _rel_l2(fused, reference)
    print(f"[diag ep={ep_size} concat={concat} quant_act={quant_act}] "
          f"vs ref_moe rel_l2={rel:.5g}")
    # ref_moe dequantizes and computes in fp32; the kernel keeps fp8 weights and
    # (optionally) fp8 activations, so this is a loose bound on purpose.
    # Upstream's own test for this kernel uses atol=rtol=2e-1 against the same
    # reference.
    assert rel < 0.06


def test_the_gmm_reference_arm_agrees_with_ref_moe():
    """Is the *reference* in this file the broken one? Same triangle, other leg."""
    ep_size = len(jax.devices())
    if ep_size < 8:
        pytest.skip("needs 8 devices")
    mesh = _mesh(ep_size)
    (tokens, logits, q_native, s_native, q_concat, s_concat, q2,
     s2) = _small_inputs(ep_size)
    global HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K
    saved = (HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K)
    HIDDEN, INTERMEDIATE = SMALL["hidden"], SMALL["inter"]
    NUM_EXPERTS, TOP_K = SMALL["experts"], SMALL["top_k"]
    try:
        gmm = np.asarray(
            jax.device_get(
                _gmm_reference(_shard(tokens, mesh), _shard(logits, mesh),
                               _shard(q_concat, mesh), _shard(s_concat, mesh),
                               _shard(q2, mesh), _shard(s2, mesh), mesh,
                               ep_size)), np.float32)
    finally:
        HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K = saved
    rel = _rel_l2(gmm, _small_ref(tokens, logits, q_native, s_native, q2, s2))
    print(f"[diag gmm-reference vs ref_moe] rel_l2={rel:.5g}")
    assert rel < 0.06


def test_the_fused_kernel_is_deterministic():
    """A barrier/DMA race would show up here and nowhere else."""
    ep_size = len(jax.devices())
    if ep_size < 8:
        pytest.skip("needs 8 devices")
    mesh = _mesh(ep_size)
    args = _inputs(mesh, ep_size, 64)
    first = np.asarray(jax.device_get(_fused(*args, mesh, True)), np.float32)
    for _ in range(3):
        again = np.asarray(jax.device_get(_fused(*args, mesh, True)),
                           np.float32)
        assert np.array_equal(first, again), "fused EP kernel is not deterministic"


# --------------------------------------------------------------------------
# W4 fault localizer. `test_which_mechanism_disagrees` says the kernel is 14%
# from `ref_moe` but not *how* the error is shaped, and a CPU model of every
# documented approximation only reaches 0.7% (bf16 activations) / 4.4% (fp8).
# Two families survive that on CPU and they need different fixes:
#   discrete  -- ~1% of (token,k) rows dropped / mis-routed / overwritten,
#                which at 128 slots is 1-2 rows: per-row error is bimodal;
#   distributed -- every element carries ~14% extra from a relayout, a bitcast
#                or a rounding mode: per-row error is unimodal.
# One decomposition of the same arrays separates them, so this adds no kernel
# change and no new Mosaic program -- the five arms below are exactly the five
# `test_which_mechanism_disagrees` already compiles.
# --------------------------------------------------------------------------
_W4_ARMS = [(1, True, True), (8, True, True), (8, False, True),
            (8, True, False), (8, False, False)]


def _w4_decompose(fused, ref, idx, tag):
    """Row / column / scale / quantisation structure of `fused - ref`."""
    import base64
    import gzip
    err = fused - ref
    n = np.linalg.norm(ref)
    row = np.linalg.norm(err, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-30)
    col = np.linalg.norm(err, axis=0) / (np.linalg.norm(ref, axis=0) + 1e-30)
    # best scalar rescale: is the whole thing just shrunk?
    a = float((fused * ref).sum() / (ref * ref).sum())
    resid = float(np.linalg.norm(fused - a * ref) / n)
    big = np.abs(ref) > np.median(np.abs(ref))
    relerr = (err[big] / ref[big])
    print(f"[w4 {tag}] rel_l2={np.linalg.norm(err) / n!r}")
    print(f"[w4 {tag}] rows: n={len(row)} min={row.min():.5f} p50={np.median(row):.5f} "
          f"p90={np.percentile(row, 90):.5f} max={row.max():.5f} "
          f"n_above_3x_median={int((row > 3 * np.median(row)).sum())}")
    # A count that is neither 1 nor 0 is a *pattern*, and the indices name it:
    # local = position inside the rank's bt block, dev = which rank, e0/e1 = the
    # token's experts. All-same-local => a block boundary; 0/1 or bt-2/bt-1 =>
    # the software-pipeline prologue/epilogue; all-same-e => a per-expert fault.
    worst = np.argsort(-row)[:8]
    bt_local = SMALL["local_tokens"]
    print(f"[w4 {tag}] worst rows {worst.tolist()} -> "
          f"{row[worst].round(4).tolist()}")
    print(f"[w4 {tag}] worst rows local={(worst % bt_local).tolist()} "
          f"dev={(worst // bt_local).tolist()}"
          + (f" e0={idx[worst, 0].tolist()} e1={idx[worst, 1].tolist()}"
             if idx is not None else ""))
    print(f"[w4 {tag}] cols: n={len(col)} min={col.min():.5f} p50={np.median(col):.5f} "
          f"p90={np.percentile(col, 90):.5f} max={col.max():.5f} "
          f"n_above_3x_median={int((col > 3 * np.median(col)).sum())}")
    wcol = np.argsort(-col)[:8]
    print(f"[w4 {tag}] worst cols {wcol.tolist()} -> {col[wcol].round(4).tolist()}")
    print(f"[w4 {tag}] worst cols lane={(wcol % 128).tolist()} "
          f"half={(wcol // (ref.shape[1] // 2)).tolist()}")
    print(f"[w4 {tag}] best_fit_scale={a:.6f} residual_after_rescale={resid:.6f}")
    # Truncation (`res >> 16`) is one-sided with a hard edge at 2^-8; a garbled
    # row is symmetric and heavy-tailed; iid rounding is symmetric and light.
    # skew and the sign split separate them without any extra data.
    c = relerr - relerr.mean()
    skew = float((c ** 3).mean() / (c.std() ** 3 + 1e-30))
    print(f"[w4 {tag}] elementwise rel err on |ref|>median: "
          f"mean={relerr.mean():.5f} std={relerr.std():.5f} "
          f"absmax={np.abs(relerr).max():.5f} skew={skew:.3f} "
          f"frac_negative={float((relerr < 0).mean()):.5f} "
          f"frac_above_0.5={float((np.abs(relerr) > 0.5).mean()):.5f}")
    print(f"[w4 {tag}] rel err quantiles "
          f"{np.percentile(relerr, [0.1, 1, 25, 50, 75, 99, 99.9]).round(6).tolist()}")
    if idx is not None:
        for e in np.unique(idx[:, 0]):
            m = idx[:, 0] == e
            print(f"[w4 {tag}] top1-expert {int(e):3d}: {int(m.sum()):3d} rows "
                  f"rel_l2={np.linalg.norm(err[m]) / np.linalg.norm(ref[m]):.5f}")
    # Opt-in: the raw arrays are ~80 lines of base64 and would push the
    # summary out of a truncated log. Everything above is the diagnosis; this
    # is only for redoing the analysis offline.
    if os.environ.get("W4_DUMP_BLOB") == "1":
        blob = base64.b64encode(gzip.compress(
            np.concatenate([ref.astype(np.float32).ravel(),
                            err.astype(np.float32).ravel()]).tobytes())).decode()
        print(f"[w4 {tag}] blob_shape={ref.shape} blob_len={len(blob)}")
        for i in range(0, len(blob), 4000):
            print(f"[w4 {tag}] blob {i // 4000:03d} {blob[i:i + 4000]}")


def test_w4_error_shape_of_the_disagreement():
    """Every arm's rel_l2 on one line each, plus the error shape of the pristine arm."""
    ndev = len(jax.devices())
    lines = []
    for ep_size, concat, quant_act in _W4_ARMS:
        if ndev < ep_size:
            lines.append(f"[w4 arm] ep={ep_size} concat={concat} "
                         f"quant_act={quant_act} SKIPPED needs {ep_size} devices")
            continue
        mesh = _mesh(ep_size)
        (tokens, logits, q_native, s_native, q_concat, s_concat, q2,
         s2) = _small_inputs(ep_size)
        ref = _small_ref(tokens, logits, q_native, s_native, q2, s2)
        fused = _small_fused(mesh, tokens, logits,
                             q_concat if concat else q_native,
                             s_concat if concat else s_native, q2, s2,
                             concat=concat, quant_act=quant_act)
        lines.append(f"[w4 arm] ep={ep_size} concat={concat} "
                     f"quant_act={quant_act} rel_l2={_rel_l2(fused, ref)!r}")
        # The pristine arm is the one to explain: no local kernel change is
        # active on it, and its documented approximations are worth 0.7%.
        if (ep_size, concat, quant_act) == (8, False, False):
            probs = jax.nn.softmax(jnp.asarray(logits, jnp.float32), axis=-1)
            _, ids = jax.lax.top_k(probs, SMALL["top_k"])
            _w4_decompose(fused, ref, np.asarray(ids), "pristine")
        if (ep_size, concat, quant_act) == (1, True, True):
            _w4_decompose(fused, ref, None, "ep1")
    # Last, so a truncated log still keeps them.
    for line in lines:
        print(line)


# --------------------------------------------------------------------------
# W4 bisection. Upstream's only two *quantized* v1 tests
# (tpu-inference tests/kernels/fused_moe_v1_test.py:367,402) are also the only
# two in that file with num_bd1 == num_bf == num_bd2 == 1: every multi-block
# configuration it covers is unquantized bf16, and float8_e4m3fn appears
# nowhere in it. Our failing shape is quantized *and* split three ways
# (num_bd1 = num_bf = num_bd2 = 2), i.e. a combination no upstream test makes.
# Splitting the contraction is what forces the kernel to apply a weight scale
# to a *partial* product (kernel.py:870-901, 996-1006) and to accumulate the
# per-bf partials in bf16 (:1022-1029), so it is a mechanism, not a coincidence.
# Five block choices turn that into a bisection; the bf16-weight arm is the
# control that says whether quantization is involved at all.
# --------------------------------------------------------------------------
_W4_BLOCKS = {
    #        bt  bf  bd1  bd2  btc bfc bd1c bd2c        num_bd1/bf/bd2
    "1x1x1": (8, 256, 512, 512, 8, 256, 512, 512),   # upstream's tested shape
    "bd1":   (8, 256, 256, 512, 8, 256, 256, 512),   # split hidden in gmm1 only
    "bd2":   (8, 256, 512, 256, 8, 256, 512, 256),   # split hidden in gmm2 only
    "bf":    (8, 128, 512, 512, 8, 128, 512, 512),   # split intermediate only
    "2x2x2": (8, 128, 256, 256, 8, 128, 256, 256),   # the default = the failure
}


def _w4_brief(fused, ref):
    """Sign and shape of the error in one line: enough to name the mechanism.

    A weight scale applied to each *partial* product is a biased error -- it
    moves `scale` away from 1 and `frac_neg` away from the no-fault value; a
    bf16 accumulate of the partials is symmetric and leaves `scale` at ~1. Both
    grow with the number of blocks, linearly and as sqrt() respectively, which
    is what the growth-law sweep measures.
    """
    err = fused - ref
    row = np.linalg.norm(err, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-30)
    a = float((fused * ref).sum() / (ref * ref).sum())
    big = np.abs(ref) > np.median(np.abs(ref))
    rel = err[big] / ref[big]
    return (f"scale={a:.6f} resid={np.linalg.norm(fused - a * ref) / np.linalg.norm(ref):.6f} "
            f"row_p50={np.median(row):.5f} rows_gt3x={int((row > 3 * np.median(row)).sum())} "
            f"frac_neg={float((rel < 0).mean()):.4f} relerr_mean={rel.mean():+.5f}")


def _small_fused_blocks(mesh, tokens, logits, w1, s1, q2, s2, blocks):
    names = ("bt", "bf", "bd1", "bd2", "btc", "bfc", "bd1c", "bd2c")
    out = fused_ep_moe(mesh=mesh,
                       tokens=_shard(tokens, mesh),
                       w1=_shard(w1, mesh),
                       w2=_shard(q2, mesh),
                       gating_output=_shard(logits, mesh),
                       top_k=SMALL["top_k"],
                       renormalize_topk_logits=True,
                       act_fn="silu",
                       scoring_fn="softmax",
                       subc_quant_w1_sz=None if s1 is None else SMALL["hidden"],
                       subc_quant_w2_sz=None if s2 is None else SMALL["inter"],
                       quant_act=False,
                       w13_concat_layout=False,
                       w1_scale=None if s1 is None else _shard(s1, mesh),
                       w2_scale=None if s2 is None else _shard(s2, mesh),
                       ep_axis_name=EP_AXIS,
                       **dict(zip(names, blocks)))
    return np.asarray(jax.device_get(out), np.float32)


def test_w4_block_bisection():
    """Does the 14% survive when the contraction is not split? And without fp8?"""
    ep_size = len(jax.devices())
    if ep_size < 8:
        pytest.skip("needs 8 devices")
    mesh = _mesh(ep_size)
    (tokens, logits, q_native, s_native, _, _, q2, s2) = _small_inputs(ep_size)
    ref = _small_ref(tokens, logits, q_native, s_native, q2, s2)

    for name, blocks in _W4_BLOCKS.items():
        try:
            fused = _small_fused_blocks(mesh, tokens, logits, q_native,
                                        s_native, q2, s2, blocks)
            print(f"[w4 blocks] {name:6s} {blocks} fp8-per-channel "
                  f"rel_l2={_rel_l2(fused, ref)!r} {_w4_brief(fused, ref)}")
        except Exception as exc:  # a rejected block choice must not hide the rest
            print(f"[w4 blocks] {name:6s} {blocks} REJECTED {type(exc).__name__}: {exc}")

    # Control: identical kernel, identical blocks, weights dequantized to bf16.
    # ref_moe with w1_scale=None consumes the same numbers, so any residual is
    # the kernel's own bf16 arithmetic and nothing to do with quantization.
    w1_bf16 = jnp.asarray(np.asarray(q_native, np.float32) *
                          np.asarray(s_native, np.float32).reshape(
                              SMALL["experts"], 2, 1, SMALL["inter"]),
                          jnp.bfloat16)
    w2_bf16 = jnp.asarray(np.asarray(q2, np.float32) *
                          np.asarray(s2, np.float32).reshape(
                              SMALL["experts"], 1, SMALL["hidden"]),
                          jnp.bfloat16)
    ref_bf16 = np.asarray(jax.device_get(
        ref_moe(tokens, w1_bf16, w2_bf16, logits, SMALL["top_k"],
                renormalize_topk_logits=True, act_fn="silu",
                scoring_fn="softmax")), np.float32)
    for name in ("1x1x1", "2x2x2"):
        fused = _small_fused_blocks(mesh, tokens, logits, w1_bf16, None,
                                    w2_bf16, None, _W4_BLOCKS[name])
        print(f"[w4 blocks] {name:6s} bf16-weights control "
              f"rel_l2={_rel_l2(fused, ref_bf16)!r} {_w4_brief(fused, ref_bf16)}")


# --------------------------------------------------------------------------
# W4 growth law. `test_w4_block_bisection` says *which axis* the split matters
# on; the number of blocks says *which mechanism*. A weight scale applied to
# each partial product before summation is a biased error that grows ~linearly
# in the block count; a bf16 accumulate of the partials is symmetric and grows
# ~sqrt(). At H=512 the contraction only admits 1 or 2 blocks (bd1c must be a
# multiple of t_packing*128), so this uses H=2048/F=1024, which reaches 8.
# --------------------------------------------------------------------------
GROW = dict(hidden=2048, inter=1024, experts=16, top_k=2, local_tokens=8)


def test_w4_growth_law():
    """Does the error grow with the number of contraction blocks, and how?"""
    ep_size = len(jax.devices())
    if ep_size < 8:
        pytest.skip("needs 8 devices")
    mesh = _mesh(ep_size)
    h, f, e, k = GROW["hidden"], GROW["inter"], GROW["experts"], GROW["top_k"]
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.normal(0, 1, (GROW["local_tokens"] * ep_size, h)) / 4,
                         jnp.bfloat16)
    logits = jnp.asarray(rng.normal(0, 1, (GROW["local_tokens"] * ep_size, e)),
                         jnp.bfloat16)
    native = rng.normal(0, 0.05, (e, 2, h, f))
    w2 = rng.normal(0, 0.05, (e, f, h))
    ns = np.abs(native).max(axis=2, keepdims=True) / FP8_MAX
    s2 = np.abs(w2).max(axis=1, keepdims=True) / FP8_MAX
    q1 = jnp.asarray(native / ns, FP8)
    s1 = jnp.asarray(ns[:, :, :, None, :], jnp.float32)
    q2 = jnp.asarray(w2 / s2, FP8)
    s2k = jnp.asarray(s2, jnp.float32).reshape(e, 1, 1, h)
    ref = np.asarray(jax.device_get(
        ref_moe(tokens, q1, q2, logits, k, renormalize_topk_logits=True,
                act_fn="silu", scoring_fn="softmax", subc_quant_w1_sz=h,
                subc_quant_w2_sz=f, w1_scale=s1, w2_scale=s2k)), np.float32)

    def run(bf_, bd1_, tag):
        blocks = (8, bf_, bd1_, h, 8, bf_, bd1_, h)
        try:
            out = fused_ep_moe(mesh=mesh, tokens=_shard(tokens, mesh),
                               w1=_shard(q1, mesh), w2=_shard(q2, mesh),
                               gating_output=_shard(logits, mesh), top_k=k,
                               renormalize_topk_logits=True, act_fn="silu",
                               scoring_fn="softmax", subc_quant_w1_sz=h,
                               subc_quant_w2_sz=f, quant_act=False,
                               w13_concat_layout=False,
                               w1_scale=_shard(s1, mesh), w2_scale=_shard(s2k, mesh),
                               ep_axis_name=EP_AXIS,
                               bt=8, bf=bf_, bd1=bd1_, bd2=h,
                               btc=8, bfc=bf_, bd1c=bd1_, bd2c=h)
            out = np.asarray(jax.device_get(out), np.float32)
            print(f"[w4 grow] {tag} num_bd1={h // bd1_} num_bf={f // bf_} "
                  f"rel_l2={_rel_l2(out, ref)!r} {_w4_brief(out, ref)}")
        except Exception as exc:
            print(f"[w4 grow] {tag} REJECTED {type(exc).__name__}: {exc}")

    for bd1_ in (h, h // 2, h // 4, h // 8):        # num_bd1 = 1, 2, 4, 8
        run(f, bd1_, f"bd1={bd1_}")
    for bf_ in (f, f // 2, f // 4, f // 8):         # num_bf  = 1, 2, 4, 8
        run(bf_, h, f"bf={bf_}")
