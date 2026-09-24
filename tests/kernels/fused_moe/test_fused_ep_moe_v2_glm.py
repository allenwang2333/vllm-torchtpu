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
"""fused_ep_moe_v2 at GLM-5.2-FP8's per-core MoE geometry.

GLM-5.2 serves EP16 over 256 experts, so each core owns 16 experts of
hidden 6144 / intermediate 2048 in fp8 e4m3 with per-output-channel scales,
routes with sigmoid scores plus the `e_score_correction_bias` (top-8,
renormalized, routed scaling applied outside the layer), and runs silu at a
tile of 128. On the eight-core slice the same per-core work is EP8 over 128
experts: g_local is 16 either way, and prefill at 1024 local tokens puts
~512 rows on each expert as production does.

Decode is padded the way the serving path pads it: 16 local rows of which
one is real. The padded rows are given -inf logits. Under SIGMOID scoring
that does not stop them routing -- sigmoid(-inf) is 0, a finite score, so the
row is selected on the bias alone and dispatched with weight exactly zero.
Every padded row therefore lands on the same eight highest-bias experts; the
test checks those rows come back exactly zero.

Three references, each on its own band:

(a) `_dense_reference_sigmoid`: the existing suite's dense f32 reference
    (tests/kernels/fused_moe/test_fused_ep_moe_v2.py::_dense_reference) with
    its router swapped for sigmoid + score bias, matmuls at HIGHEST
    precision. Nothing is quantized except the weights, so the band against
    it is the fp8 transport's.
(b) `_oracle_f64`: a float64 numpy oracle of the SAME quantized math the
    layer does -- per-row fp8 activation quant (reduced and applied in bf16),
    dequantized fp8 weights, the bf16 intermediate requantized per row, the
    down row requantized for the wire, then the weighted f64 combine. It sits
    ~3.7% (relative L2) from the device, and that is the DEVICE, not the
    kernel: an XLA program of the same jnp math on the same chip sits at the
    same distance (tpu7x, libtpu 0.0.47). Two device properties, measured
    stage by stage: the TPU fp8 matmul with f32 output is not an exact f32
    accumulation (4.2e-3 relative error against f64 on exactly representable
    operands, at every precision setting), which moves values across fp8
    rounding boundaries in the two requantizations after it; and the wire
    quantization rounds differently from bf16-then-e4m3 on 3.3% of elements
    even given the device's own accumulator. On CPU the same jnp math matches
    this oracle to 1.5e-5, so the oracle is the math as written.
(c) `_xla_twin`: the kernel's own jnp expert math (kernel.expert_ffn_fp8 +
    kernel.wire_row_and_scale, the layer's rowquant_fp8) run as plain XLA on
    the same TPU, per expert over the f64 routing, combined in f64. It shares
    the device's arithmetic, so what is left against it is the Mosaic
    kernel's own drift -- 1.7e-3 on the incumbent -- and it is the gate that
    sees a small fault.

Tolerances are calibrated from the incumbent kernel (see the constants), and
a positive control -- one expert's gate/up slab replaced by its neighbour's --
has to break them.
"""

import functools
import inspect

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2 import AXIS, WeightFormat, fused_ep_moe_v2
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    expert_ffn_fp8,
    rowquant_fp8,
    wire_row_and_scale,
)

from .test_fused_ep_moe_v2 import (
    CAPACITY,
    EP,
    _make_weights,
    _mesh,
    _relative_l2,
    _worst_token_relative_l2,
)

pytestmark = pytest.mark.multichip


@pytest.fixture(scope="module", autouse=True)
def _require_eight_devices():
    """See test_fused_ep_moe_v2._require_eight_devices: a fixture, not a
    skipif, so collection never builds a PJRT client."""
    if jax.device_count() < 8:
        pytest.skip("the GLM fused EP MoE case needs the eight-device slice")


GLM = dict(hidden=6144, inter=2048, e_total=16 * EP, topk=8)
PREFILL_T_LOCAL = 1024
DECODE_T_LOCAL = 16
DECODE_REAL_PER_SHARD = 1
# The references are evaluated on every ORACLE_STRIDE-th prefill token (1024
# of 8192, spread over all eight shards); decode is checked whole.
ORACLE_STRIDE = 8
# A token whose 8th and 9th biased scores sit closer than this may select
# differently in f32 than in f64; it is excluded from the oracle comparison
# and counted.
NEAR_TIE = 1e-6

WEIGHT_SEED = 11
PRIOR_SEED = 12
PREFILL_SEED = 13
DECODE_SEED = 14

FP8_MAX = 448.0

# (batch relative L2, worst-token relative L2), calibrated from the incumbent
# (fp8_stream=False) on tpu7x 2x2x1, job j-a30ed105 (ref, oracle) and
# j-890122f1 (twin). The inputs are seeded and the kernel deterministic, so
# the observed error is a fixed number, and the bounds are 1.25x it rather
# than 2x: at 2x the reference band would be 0.108, twice the existing
# suite's 0.06 fp8 band. Observed, prefill / decode:
#   ref    0.05387/0.05675   0.05342/0.05596
#   oracle 0.03684/0.04209   0.03693/0.03852
#   twin   rel 0.00166       rel 0.00167 (worst token printed, not gated)
REF_BOUNDS = {"prefill": (0.068, 0.071), "decode": (0.067, 0.070)}
ORACLE_BOUNDS = {"prefill": (0.046, 0.053), "decode": (0.047, 0.049)}
# The twin gates the batch norm only, at 2.4x observed; its worst-token
# figure was not in the calibration run.
TWIN_REL_BOUND = 0.004

HAS_FP8_STREAM = "fp8_stream" in inspect.signature(fused_ep_moe_v2).parameters
HAS_VALID_ROWS = "valid_rows" in inspect.signature(fused_ep_moe_v2).parameters
needs_valid_rows = pytest.mark.skipif(
    not HAS_VALID_ROWS, reason="this tree has no valid_rows padding drop"
)
FP8_STREAM_PARAMS = [
    pytest.param(False, id="incumbent"),
    pytest.param(
        True,
        id="fp8_stream",
        marks=pytest.mark.skipif(
            not HAS_FP8_STREAM, reason="this tree has no fp8_stream kernel arm"
        ),
    ),
]


# --------------------------------------------------------------------------- #
# Inputs.
# --------------------------------------------------------------------------- #


def _expert_prior():
    """(popularity offset, score bias) per global expert: model constants,
    shared by the prefill and decode batches."""
    rng = np.random.default_rng(PRIOR_SEED)
    popularity = rng.normal(0.0, 0.2, GLM["e_total"])
    score_bias = rng.normal(0.0, 0.02, GLM["e_total"]).astype(np.float32)
    return popularity, score_bias


def _host_batch(seed, t_local, real_per_shard):
    """(x bf16, gating f32) on the host, [EP * t_local, ...], shard-major.

    Real rows get N(0,1) hidden states and N(0, 2) + popularity logits, so
    routing is spread but not uniform. Padded rows (local index >=
    real_per_shard) get zero hidden states and -inf logits.
    """
    popularity, _ = _expert_prior()
    tokens = EP * t_local
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (tokens, GLM["hidden"])).astype(np.float32)
    gating = (
        rng.normal(0.0, 2.0, (tokens, GLM["e_total"])) + popularity[None, :]
    ).astype(np.float32)
    pad = (np.arange(tokens) % t_local) >= real_per_shard
    x[pad] = 0.0
    gating[pad] = -np.inf
    return x.astype(ml_dtypes.bfloat16), gating, pad


@functools.cache
def _batch(kind):
    if kind == "prefill":
        x, g, pad = _host_batch(PREFILL_SEED, PREFILL_T_LOCAL, PREFILL_T_LOCAL)
        rows = np.arange(0, x.shape[0], ORACLE_STRIDE)
    else:
        x, g, pad = _host_batch(DECODE_SEED, DECODE_T_LOCAL, DECODE_REAL_PER_SHARD)
        rows = np.arange(x.shape[0])
    return x, g, pad, rows


def _device_batch(mesh, kind):
    x, g, pad, rows = _batch(kind)
    shard = NamedSharding(mesh, P(AXIS))
    return (
        jax.device_put(x, shard),
        jax.device_put(g, shard),
        pad,
        rows,
    )


# --------------------------------------------------------------------------- #
# The layer under test.
# --------------------------------------------------------------------------- #


def run_glm_layer(
    x,
    gating,
    score_bias,
    weights,
    mesh,
    *,
    sharded_plan=False,
    fp8_stream=False,
    valid_rows=None,
):
    w1, w2, w1s, w2s = weights
    kw = {"fp8_stream": bool(fp8_stream)} if HAS_FP8_STREAM else {}
    if valid_rows is not None:
        kw["valid_rows"] = valid_rows
    return fused_ep_moe_v2(
        x,
        w1,
        w2,
        w1s,
        w2s,
        gating,
        topk=GLM["topk"],
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP8,
        scoring_fn="sigmoid",
        score_bias=score_bias,
        sharded_plan=sharded_plan,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Reference (a): the existing dense f32 reference, sigmoid + bias router.
# --------------------------------------------------------------------------- #


def _dense_reference_sigmoid(mesh, x, gating, score_bias, w1, w2, w1s, w2s, *, topk):
    (axis,) = mesh.axis_names
    g_local = w1.shape[0] // mesh.shape[axis]
    inter = w1.shape[2] // 2
    hi = lax.Precision.HIGHEST

    def local(x_g, gating_g, bias_g, w1_l, w2_l, s1, s2):
        scores = jax.nn.sigmoid(gating_g.astype(jnp.float32))
        _, topk_idx = lax.top_k(scores + bias_g[None, :], topk)
        topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1)
        topk_weights = topk_weights / jnp.maximum(
            topk_weights.sum(axis=-1, keepdims=True), 1e-20
        )
        first = lax.axis_index(axis) * g_local
        x32 = x_g.astype(jnp.float32)

        def expert(e, acc):
            w1e = lax.dynamic_index_in_dim(w1_l, e, 0, keepdims=False).astype(
                jnp.float32
            ) * lax.dynamic_index_in_dim(s1, e, 0, keepdims=True)
            w2e = lax.dynamic_index_in_dim(w2_l, e, 0, keepdims=False).astype(
                jnp.float32
            ) * lax.dynamic_index_in_dim(s2, e, 0, keepdims=True)
            weight = jnp.sum(
                jnp.where(topk_idx == first + e, topk_weights, 0.0), axis=-1
            )[:, None]
            acc1 = jnp.dot(x32, w1e, precision=hi)
            row = jnp.dot(
                jax.nn.silu(acc1[:, :inter]) * acc1[:, inter:], w2e, precision=hi
            )
            return acc + weight * row

        mine = lax.fori_loop(0, g_local, expert, jnp.zeros_like(x32))
        return lax.psum(mine, axis)

    fn = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(), P(), P(), P(axis), P(axis), P(axis), P(axis)),
            out_specs=P(),
            check_vma=False,
        )
    )
    return np.asarray(fn(x, gating, score_bias, w1, w2, w1s, w2s), np.float64)


# --------------------------------------------------------------------------- #
# Reference (b): float64 oracle of the layer's quantized math.
# --------------------------------------------------------------------------- #


def _rowquant_emul(v_bf16):
    """kernel.rowquant_fp8 on the host: amax in bf16, scale = amax / 448 in
    f32, the inverse rounded to bf16, the product rounded to bf16, then e4m3.
    Returns (fp8 values as f64, scale f64 [rows, 1])."""
    v = np.asarray(v_bf16, ml_dtypes.bfloat16).astype(np.float32)
    amax = np.max(np.abs(v), axis=-1, keepdims=True)
    scale = (amax / np.float32(FP8_MAX)).astype(np.float32)
    usable = np.isfinite(scale) & (scale > 0)
    scale = np.where(usable, scale, np.float32(0))
    sinv = np.where(usable, np.float32(1) / np.where(usable, scale, 1), 0).astype(
        np.float32
    )
    sinv_bf = sinv.astype(ml_dtypes.bfloat16).astype(np.float32)
    prod = (v * sinv_bf).astype(ml_dtypes.bfloat16)
    q = np.where(usable, prod, 0).astype(ml_dtypes.float8_e4m3fn)
    return q.astype(np.float64), scale.astype(np.float64)


def oracle_routing(gating, score_bias, topk):
    """(topk_idx, topk_weights, near_tie mask) in f64, the layer's rule:
    select on sigmoid + bias, weigh by sigmoid, renormalize."""
    g = gating.astype(np.float64)
    with np.errstate(over="ignore"):
        scores = 1.0 / (1.0 + np.exp(-g))
    sel = scores + score_bias.astype(np.float64)[None, :]
    order = np.argsort(-sel, axis=-1, kind="stable")[:, : topk + 1]
    idx = order[:, :topk]
    r = np.arange(g.shape[0])
    gap = sel[r, order[:, topk - 1]] - sel[r, order[:, topk]]
    w = np.take_along_axis(scores, idx, axis=-1)
    w = w / np.maximum(w.sum(axis=-1, keepdims=True), 1e-20)
    return idx, w, gap < NEAR_TIE


def _oracle_f64(x_bf16, gating, score_bias, host_weights, *, topk, inter):
    w1, w2, w1s, w2s = host_weights
    idx, w, near_tie = oracle_routing(gating, score_bias, topk)
    q, qs = _rowquant_emul(x_bf16)
    np.seterr(over="ignore")
    out = np.zeros((x_bf16.shape[0], w2.shape[2]), np.float64)
    for e in np.unique(idx):
        rows, slots = np.nonzero(idx == e)
        h = (q[rows] @ w1[e].astype(np.float64)) * qs[rows] * w1s[e][None, :]
        gate, up = h[:, :inter], h[:, inter:]
        mid = (gate / (1.0 + np.exp(-gate))) * up
        mq, ms = _rowquant_emul(mid.astype(np.float32).astype(ml_dtypes.bfloat16))
        down = (mq @ w2[e].astype(np.float64)) * ms * w2s[e][None, :]
        wq, wsc = _rowquant_emul(down.astype(np.float32).astype(ml_dtypes.bfloat16))
        np.add.at(out, rows, wq * wsc * w[rows, slots][:, None])
    return out, near_tie


def _xla_twin(x_bf16, gating, score_bias, weights, *, topk):
    """(c): the kernel's jnp expert math as plain XLA on one TPU core."""
    w1, w2, w1s, w2s = weights
    idx, w, _ = oracle_routing(gating, score_bias, topk)
    dev = jax.devices()[0]
    q, qs = jax.jit(rowquant_fp8)(jax.device_put(np.asarray(x_bf16), dev))

    @jax.jit
    def ffn(qr, sr, w1e, w2e, s1, s2):
        acc2, ms = expert_ffn_fp8(qr, sr, w1e, w2e, s1[None, :])
        return wire_row_and_scale(acc2, ms, w2s=s2[None, :])

    out = np.zeros((x_bf16.shape[0], w2.shape[2]), np.float64)
    for e in np.unique(idx):
        rows, slots = np.nonzero(idx == e)
        n = len(rows)
        padded = np.concatenate([rows, np.full(-(-n // 64) * 64 - n, rows[0])])
        wr, ws = ffn(
            q[padded],
            qs[padded],
            *(jax.device_put(a[e], dev) for a in (w1, w2, w1s, w2s)),
        )
        v = np.asarray(wr).astype(np.float64)[:n] * np.asarray(ws, np.float64)[:n]
        np.add.at(out, rows, v * w[rows, slots][:, None])
    return out


# --------------------------------------------------------------------------- #
# Metrics, gates, fixtures.
# --------------------------------------------------------------------------- #


def error_metrics(actual, want):
    a = np.asarray(actual, np.float64)
    w = np.asarray(want, np.float64)
    d = np.abs(a - w)
    rms = np.sqrt(np.mean(w * w))
    big = np.abs(w) >= 0.01 * rms
    return dict(
        rel_l2=_relative_l2(a, w),
        worst_token=_worst_token_relative_l2(a, w),
        max_abs=float(d.max()),
        max_rel=float((d[big] / np.abs(w[big])).max()) if big.any() else 0.0,
        ref_rms=float(rms),
    )


def gate_violations(metrics, bounds, label):
    batch, token = bounds
    bad = []
    if not metrics["rel_l2"] < batch:
        bad.append(f"{label}: rel_l2 {metrics['rel_l2']:.5f} >= {batch}")
    if not metrics["worst_token"] < token:
        bad.append(f"{label}: worst_token {metrics['worst_token']:.5f} >= {token}")
    return bad


def routing_stats(topk_idx, pad, g_local):
    """Per-core non-empty experts and rows/expert, from the f64 routing."""
    e_total = g_local * EP
    counts = np.bincount(topk_idx.ravel(), minlength=e_total).reshape(EP, g_local)
    real = np.bincount(topk_idx[~pad].ravel(), minlength=e_total).reshape(EP, g_local)
    nonempty = (counts > 0).sum(axis=1)
    live = counts[counts > 0]
    return dict(
        nonempty_per_core=nonempty.tolist(),
        nonempty_real_only_per_core=(real > 0).sum(axis=1).tolist(),
        rows_per_nonempty_expert=dict(
            min=int(live.min()),
            median=float(np.median(live)),
            max=int(live.max()),
            mean=float(live.mean()),
        ),
        max_rows_per_core=int(counts.sum(axis=1).max()),
    )


@pytest.fixture(scope="module")
def glm_weights():
    mesh = _mesh()
    weights = _make_weights(
        mesh,
        WEIGHT_SEED,
        hidden=GLM["hidden"],
        inter=GLM["inter"],
        e_total=GLM["e_total"],
        weight_format=WeightFormat.FP8,
    )
    jax.block_until_ready(weights)
    host = tuple(np.asarray(a) for a in weights)
    host = (host[0], host[1], host[2].astype(np.float64), host[3].astype(np.float64))
    yield weights, host
    del weights, host


@functools.cache
def _score_bias_host():
    return _expert_prior()[1]


_REF_CACHE = {}


def references(kind, glm_weights):
    """(rows, ref_a, ref_b, ref_c, near_tie, stats) for a batch kind, computed once."""
    if kind in _REF_CACHE:
        return _REF_CACHE[kind]
    weights, host = glm_weights
    mesh = _mesh()
    x, g, pad, rows = _batch(kind)
    bias = _score_bias_host()
    idx_all, _, _ = oracle_routing(g, bias, GLM["topk"])
    stats = routing_stats(idx_all, pad, GLM["e_total"] // EP)
    ref_a = _dense_reference_sigmoid(
        mesh,
        jnp.asarray(x[rows]),
        jnp.asarray(g[rows]),
        jnp.asarray(bias),
        *weights,
        topk=GLM["topk"],
    )
    ref_b, near_tie = _oracle_f64(
        x[rows], g[rows], bias, host, topk=GLM["topk"], inter=GLM["inter"]
    )
    ref_c = _xla_twin(x[rows], g[rows], bias, weights, topk=GLM["topk"])
    _REF_CACHE[kind] = (rows, ref_a, ref_b, ref_c, near_tie, stats)
    return _REF_CACHE[kind]


def _check(kind, out, glm_weights, *, label):
    """All violations of `out` against the three references, plus metrics."""
    rows, ref_a, ref_b, ref_c, near_tie, stats = references(kind, glm_weights)
    got = np.asarray(out, np.float64)[rows]
    keep = ~near_tie
    m_ref = error_metrics(got[keep], ref_a[keep])
    m_orc = error_metrics(got[keep], ref_b[keep])
    m_twin = error_metrics(got[keep], ref_c[keep])
    print(
        f"\n[GLM {kind} {label}] routing {stats}\n"
        f"  near-tie tokens excluded: {int(near_tie.sum())} of {len(rows)}\n"
        f"  vs ref (a) f32 dense: {m_ref}\n"
        f"  vs oracle (b) f64 quantized: {m_orc}\n"
        f"  vs twin (c) XLA on TPU: {m_twin}\n"
        f"  ref(a) vs oracle(b): {error_metrics(ref_a[keep], ref_b[keep])}\n"
        f"  twin(c) vs oracle(b): {error_metrics(ref_c[keep], ref_b[keep])}"
    )
    bad = gate_violations(m_ref, REF_BOUNDS[kind], "ref") + gate_violations(
        m_orc, ORACLE_BOUNDS[kind], "oracle"
    )
    if not m_twin["rel_l2"] < TWIN_REL_BOUND:
        bad.append(f"twin: rel_l2 {m_twin['rel_l2']:.5f} >= {TWIN_REL_BOUND}")
    return bad, m_ref, m_orc


def _kernel_name_in_lowering(mesh, x, g, bias, weights, fp8_stream):
    """Which fused_ep_moe_v2_* kernel names appear in the lowered program."""
    import re

    fn = jax.jit(
        lambda x_, g_, b_, *w: run_glm_layer(x_, g_, b_, w, mesh, fp8_stream=fp8_stream)
    )
    text = fn.lower(x, g, bias, *weights).as_text()
    return sorted(set(re.findall(r"fused_ep_moe_v2\w*", text)))


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["prefill", "decode"])
@pytest.mark.parametrize("fp8_stream", FP8_STREAM_PARAMS)
def test_glm_shape_tracks_reference_and_f64_oracle(glm_weights, kind, fp8_stream):
    mesh = _mesh()
    weights, _ = glm_weights
    x, g, pad, _ = _device_batch(mesh, kind)
    bias = jnp.asarray(_score_bias_host())
    names = _kernel_name_in_lowering(mesh, x, g, bias, weights, fp8_stream)
    print(f"\n[GLM {kind} fp8_stream={fp8_stream}] kernel names in lowering: {names}")
    out = run_glm_layer(x, g, bias, weights, mesh, fp8_stream=fp8_stream)
    assert out.shape == x.shape and out.dtype == x.dtype
    host_out = np.asarray(out, np.float64)
    # A padded row is dispatched with weight exactly zero, so it must come
    # back exactly zero, not merely small.
    if pad.any():
        assert np.all(host_out[pad] == 0.0), "a padded row came back non-zero"
    bad, _, _ = _check(kind, out, glm_weights, label=f"fp8_stream={fp8_stream}")
    assert not bad, bad


@pytest.mark.parametrize("kind", ["prefill", "decode"])
def test_glm_sharded_plan_is_bit_identical(glm_weights, kind):
    """sharded_plan changes routing-plan construction only."""
    mesh = _mesh()
    weights, _ = glm_weights
    x, g, _, _ = _device_batch(mesh, kind)
    bias = jnp.asarray(_score_bias_host())
    a = np.asarray(run_glm_layer(x, g, bias, weights, mesh, sharded_plan=False))
    b = np.asarray(run_glm_layer(x, g, bias, weights, mesh, sharded_plan=True))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("fp8_stream", FP8_STREAM_PARAMS)
def test_glm_positive_control_wrong_expert_turns_red(glm_weights, fp8_stream):
    """One expert's gate/up slab replaced by its neighbour's must break the
    calibrated gates on both references. A mild fault (one expert's down
    scale x1.01) is reported, not asserted, as the gate's sensitivity."""
    mesh = _mesh()
    weights, _ = glm_weights
    w1, w2, w1s, w2s = weights
    x, g, _, rows = _device_batch(mesh, "prefill")
    bias = jnp.asarray(_score_bias_host())
    idx, _, _ = oracle_routing(_batch("prefill")[1][rows], _score_bias_host(), 8)
    load = np.bincount(idx.ravel(), minlength=GLM["e_total"])[:15]
    victim = int(np.argmax(load))  # busiest sampled expert on shard 0
    shard = NamedSharding(mesh, P(AXIS))
    swap = jax.jit(lambda w: w.at[victim].set(w[victim + 1]), out_shardings=shard)
    nudge = jax.jit(lambda s: s.at[victim].multiply(1.01), out_shardings=shard)

    bad_w1 = run_glm_layer(
        x, g, bias, (swap(w1), w2, swap(w1s), w2s), mesh, fp8_stream=fp8_stream
    )
    bad, m_ref, m_orc = _check(
        "prefill", bad_w1, glm_weights, label=f"FAULT swap e{victim}"
    )
    assert any(v.startswith("ref") for v in bad), (m_ref, bad)
    assert any(v.startswith("oracle") for v in bad), (m_orc, bad)
    assert any(v.startswith("twin") for v in bad), bad

    mild = run_glm_layer(
        x, g, bias, (w1, w2, w1s, nudge(w2s)), mesh, fp8_stream=fp8_stream
    )
    mild_bad, _, _ = _check(
        "prefill", mild, glm_weights, label=f"MILD FAULT w2s[e{victim}]*1.01"
    )
    print(f"  mild fault trips the gate: {bool(mild_bad)} {mild_bad}")


# --------------------------------------------------------------------------- #
# valid_rows: padding rows kept out of dispatch.
# --------------------------------------------------------------------------- #

# The serving path's pad rows are not -inf rows: they are token 0 run through
# the model, so every pad row on every core carries the SAME finite hidden
# state and the SAME finite router logits, and all of them select the same
# eight experts (~120 rows on each at EP8). This seed draws that one shared
# row; the real rows are the "decode" batch's, untouched.
FINITE_PAD_SEED = 15


@functools.cache
def _finite_pad_batch():
    """(x, gating, pad) for decode with production-like pad rows: row 0 of
    each core is the "decode" batch's real row, rows 1..15 share one finite
    x row and one finite logit vector."""
    x, g, pad, _ = _batch("decode")
    x, g = x.copy(), g.copy()
    popularity, _ = _expert_prior()
    rng = np.random.default_rng(FINITE_PAD_SEED)
    x_pad = rng.normal(0.0, 1.0, GLM["hidden"]).astype(ml_dtypes.bfloat16)
    g_pad = (rng.normal(0.0, 2.0, GLM["e_total"]) + popularity).astype(np.float32)
    x[pad] = x_pad[None, :]
    g[pad] = g_pad[None, :]
    return x, g, pad


def _valid_rows(mesh, pad):
    return jax.device_put((~pad).astype(np.int32), NamedSharding(mesh, P(AXIS)))


@needs_valid_rows
@pytest.mark.parametrize("fp8_stream", FP8_STREAM_PARAMS)
def test_glm_decode_valid_rows_drops_finite_pads(glm_weights, fp8_stream):
    """Production-like decode pads, with and without valid_rows.

    With valid_rows the real rows must be BIT-IDENTICAL to the same batch
    without it (dropping a pad row cannot change any real row's arithmetic),
    the pad rows must be exactly zero, and the whole output must pass the
    decode references -- which are the "decode" batch's, whose pad rows are
    exactly zero too and whose real rows are these. Without valid_rows the
    finite pads DO route, so they come back non-zero: that is what makes the
    zero check discriminating rather than inherited from the -inf batch.
    An all-padding step (the runner's warmup) must come back all zero.
    """
    mesh = _mesh()
    weights, _ = glm_weights
    x_h, g_h, pad = _finite_pad_batch()
    x_dec, g_dec, pad_dec, _ = _batch("decode")
    np.testing.assert_array_equal(pad, pad_dec)
    np.testing.assert_array_equal(x_h[~pad], x_dec[~pad])
    np.testing.assert_array_equal(g_h[~pad], g_dec[~pad])
    assert np.all(np.isfinite(g_h)), "production pads carry finite logits"

    shard = NamedSharding(mesh, P(AXIS))
    x = jax.device_put(x_h, shard)
    g = jax.device_put(g_h, shard)
    bias = jnp.asarray(_score_bias_host())
    idx, _, _ = oracle_routing(g_h, _score_bias_host(), GLM["topk"])
    print(
        f"\n[GLM decode finite pads fp8_stream={fp8_stream}] routing "
        f"{routing_stats(idx, pad, GLM['e_total'] // EP)}"
    )
    run = functools.partial(
        run_glm_layer,
        x,
        g,
        bias,
        weights,
        mesh,
        sharded_plan=True,
        fp8_stream=fp8_stream,
    )
    without = np.asarray(run())
    with_valid = np.asarray(run(valid_rows=_valid_rows(mesh, pad)))
    assert with_valid.shape == x_h.shape

    pads_live = int(np.any(without[pad] != 0, axis=1).sum())
    assert pads_live == int(pad.sum()), (
        f"only {pads_live} of {int(pad.sum())} finite pad rows came back "
        "non-zero without valid_rows, so the pads are not being dispatched "
        "and the drop below would be vacuous"
    )
    np.testing.assert_array_equal(
        with_valid[~pad],
        without[~pad],
        err_msg="valid_rows changed a real row",
    )
    assert np.all(with_valid[pad] == 0.0), (
        f"{int(np.any(with_valid[pad] != 0, axis=1).sum())} dropped pad rows "
        "came back non-zero"
    )
    assert np.all(np.isfinite(with_valid))
    bad, _, _ = _check(
        "decode", with_valid, glm_weights, label=f"valid_rows fp8_stream={fp8_stream}"
    )
    assert not bad, bad

    everything_pad = np.ones_like(pad)
    idle = np.asarray(run(valid_rows=_valid_rows(mesh, everything_pad)))
    assert np.all(idle == 0.0), "an all-padding step came back non-zero"


@needs_valid_rows
@pytest.mark.parametrize("fp8_stream", FP8_STREAM_PARAMS)
def test_glm_prefill_all_valid_rows_is_bit_identical(glm_weights, fp8_stream):
    """A full prefill chunk has no padding: an all-ones mask must not change
    a single bit of the output."""
    mesh = _mesh()
    weights, _ = glm_weights
    x, g, pad, _ = _device_batch(mesh, "prefill")
    assert not pad.any()
    bias = jnp.asarray(_score_bias_host())
    run = functools.partial(
        run_glm_layer,
        x,
        g,
        bias,
        weights,
        mesh,
        sharded_plan=True,
        fp8_stream=fp8_stream,
    )
    a = np.asarray(run())
    b = np.asarray(run(valid_rows=_valid_rows(mesh, pad)))
    np.testing.assert_array_equal(a, b)


@needs_valid_rows
def test_glm_valid_rows_requires_sharded_plan(glm_weights):
    """Only the sharded plan maps the drop id off every shard; the replicated
    plan would fold it into shard zero's slab, so it must refuse."""
    mesh = _mesh()
    weights, _ = glm_weights
    x, g, pad, _ = _device_batch(mesh, "decode")
    bias = jnp.asarray(_score_bias_host())
    with pytest.raises(ValueError, match="sharded_plan"):
        run_glm_layer(
            x,
            g,
            bias,
            weights,
            mesh,
            sharded_plan=False,
            valid_rows=_valid_rows(mesh, pad),
        )
