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
"""Device coverage for the fused EP MoE v2 kernel's K-BLOCKED weight residency.

`fused_moe/v2` buffers a WHOLE EXPERT per weight slot. At a wide enough
expert that does not fit VMEM and `_build_fused_ep_moe_kernel` refuses, so
`host.resolve_weight_kblock` gives whole-expert residency up and buffers a
CONTRACTION BLOCK per slot instead -- `(KBLOCK_HIDDEN, KBLOCK_INTER)` =
(2048, 1024) on the eight-bit float path.

WHY THIS FILE EXISTS. `test_fused_ep_moe_v2.py` and
`test_fused_ep_moe_v2_tables.py` pass at every shape they carry, and every
one of those shapes FITS whole-expert: `resolve_weight_kblock` answers None
for all of them, the blocked buffers are never declared and the blocked
reader is never traced. Those passes are a no-regression result for the
incumbent residency and say nothing at all about the blocked one. This file
is the blocked arm's only in-tree coverage.

THE SHAPES are GLM-5.2-FP8's MoE block, which is the shape that forced the
fallback: hidden 6144, `moe_intermediate_size` 2048, 256 routed experts,
top-8, fp8 e4m3, `noaux_tc` routing -- a sigmoid score with an
`e_score_correction_bias` added for SELECTION only
(`knowledge/models/GLM-5.2-FP8/refs/model.md`, U-annyan-260909-01 and -06,
`config.json` @ GCS generation `1785565055944226`).

THE EP DEGREE DIFFERS FROM PRODUCTION, DELIBERATELY. GLM-5.2 serves at EP16
on a 2x2x2 slice: 256 global experts over 16 shards is `g_local` 16. The
probe pod is one 2x2x1 node, 8 cores, so this file runs EP8 with 128 GLOBAL
experts -- which is the same `g_local` 16, and `g_local` is what the weight
buffers, the scale tables and therefore the VMEM budget are a function of.
The residency decision this file gates is byte-identical to production's;
only the number of shards the routed rows travel between is halved.

WHAT IS NOT MODELLED. The checkpoint's fp8 weights are block-scaled
([128, 128], U-annyan-260909-08); the kernel's FP8 weight format takes ONE
scale per OUTPUT CHANNEL, so the weights here are quantized that way, as
the sibling suite's `_make_weights` does. The contraction heights, the
buffer sizes and the VMEM budget do not depend on the scale layout.
`routed_scaling_factor` is left at 1.0.

NO INTERPRET MODE. `pltpu.InterpretParams` deadlocks inside its own
cross-device vector-clock barrier at ep > 1 -- the UNMODIFIED whole-expert
path hangs identically, so it is the harness and not this change. Every
test here runs the real kernel on real chips.
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2 import (AXIS, WeightFormat,
                                                fused_ep_moe_v2, host)
from vllm_torchtpu.kernels.fused_moe.v2 import kernel as kernel_mod

pytestmark = pytest.mark.multichip


@pytest.fixture(scope="module", autouse=True)
def _require_eight_devices():
    """Skip the module unless this host really has the eight-way slice.

    A fixture rather than a `skipif`, for the reason the sibling suite
    states: a `skipif` is evaluated while pytest IMPORTS the module, and
    `jax.device_count()` builds the PJRT client to answer it, which locks
    every spawned vLLM worker in the same pytest run out of libtpu.
    """
    if jax.device_count() < 8:
        pytest.skip("the fused EP MoE kernel is an eight-way expert-parallel "
                    "collective and this host reports fewer than 8 JAX "
                    "devices")


EP = 8
CAPACITY = 128

FP8 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16
FP8_MAX = float(jnp.finfo(FP8).max)

# GLM-5.2-FP8's MoE block at EP8, holding production's 16 local experts.
GLM = dict(hidden=6144, inter=2048, e_total=128, topk=8, tokens=512)
GLM_G_LOCAL = GLM["e_total"] // EP

# Qwen3.5-397B's MoE block, the shape that must KEEP whole-expert residency.
QWEN = dict(hidden=4096, inter=1024, g_local=64)

# What `resolve_weight_kblock` has to answer at GLM's shapes, and the scope
# name the build then carries. The tag is the arm-activation evidence: it is
# emitted by `_build_fused_ep_moe_kernel` only on the blocked branch, and it
# is the one place a profile or a test can be told which residency ran.
EXPECTED_KBLOCK = (host.KBLOCK_HIDDEN, host.KBLOCK_INTER)
GLM_KERNEL_NAME = (f"fused_ep_moe_v2_kb{host.KBLOCK_HIDDEN}"
                   f"x{host.KBLOCK_INTER}_g{GLM_G_LOCAL}_c{CAPACITY}"
                   f"_nb{host.NBUF}")

# The small structurally-complete shape from the sibling suite, carried here
# as the negative control on the NAME itself: it fits whole-expert, so its
# build must not carry a `_kb` tag. Without it, "the name contains _kb2048x1024"
# is consistent with a tag that is always on.
SMALL = dict(hidden=512, inter=256, e_total=32, topk=4, tokens=256)
SMALL_KERNEL_NAME = (f"fused_ep_moe_v2_g{SMALL['e_total'] // EP}"
                     f"_c{CAPACITY}_nb{host.NBUF}")

# The layer's bands against the dense reference. These are the SIBLING
# SUITE's fp8 bands, and they have to be: the routed rows cross the mesh
# quantized to e4m3 with one scale a row and the intermediate is requantized
# the same way, so what the layer output can be held to is the transport's
# error, not the contraction's. The contraction's own bands are three orders
# tighter and are gated separately below, where there is no wire in the way.
FP8_RELATIVE_BOUND = 0.06
FP8_TOKEN_BOUND = 0.075
ROTATED_CONTROL_FACTOR = 10

# The contraction's bands, earned by the float64 oracle in
# `kernel_work/instruments/unit_test.py` over a KB sweep of 128..whole-K at
# these exact shapes: worst measured acc1 1.28e-7 (vs the oracle), and
# 1.36e-4 batch relative L2 / 3.04e-3 worst single row on the output.
#
# THE OUTPUT BANDS ARE STATED AGAINST THE WHOLE-K CONTRACTION, not against
# the oracle, and that is the instrument's own framing -- "what an on-device
# A/B would compare". The accumulator band is gated both ways, because the
# algebraic claim is a statement about the sum and the oracle is the only
# thing in this file with no block structure of its own.
#
# The output bands are three orders wider than the accumulator band because
# the FFN requantizes its intermediate to e4m3 per row and a sub-ulp fp32
# move in acc1 that straddles a quantization boundary flips one code by a
# full 12%.
ACC1_RTOL = 1e-6
OUT_RTOL = 1e-3
OUT_ROW_RTOL = 2e-2
# A control has to miss by orders of magnitude, not by a factor.
CONTRACTION_CONTROL_FACTOR = 1e3
# Rows the contraction study runs at: a decode-sized tile and a full one.
CONTRACTION_ROWS = (32, 128)

# The kernel's own spelling of a contraction.
DN = (((1, ), (0, )), ((), ()))


# --------------------------------------------------------------------------
# arm-activation recording
# --------------------------------------------------------------------------
class _RecordingPallas:
    """A stand-in for `kernel.pl` that records every pallas_call name.

    The name is built inside `_build_fused_ep_moe_kernel.make_call` and
    never leaves it, so there is no return value to inspect and no attribute
    on the built callable to read. Proxying the module the builder calls
    through is the only way to see the string it actually passed, and the
    string is the only artifact that distinguishes the two residencies: they
    have the same operands, the same output shape and the same dtypes.

    Everything except `pallas_call` is delegated untouched, so the builder
    still gets the real `pl.run_scoped`, `pl.BlockSpec` and the rest.
    """

    def __init__(self, real, names):
        self._real = real
        self.names = names

    def __getattr__(self, attr):
        return getattr(self._real, attr)

    def pallas_call(self, *args, **kwargs):
        self.names.append(kwargs.get("name"))
        return self._real.pallas_call(*args, **kwargs)


def _moe_build_names(names):
    """The fused-EP builds among the recorded names.

    `kernel.py` also builds the arrival combine, whose name starts
    `moe_v2_combine_`; the residency tag lives on the FFN program only.
    """
    return [n for n in names if n and n.startswith("fused_ep_moe_v2")]


# --------------------------------------------------------------------------
# operands, in the sibling suite's idiom
# --------------------------------------------------------------------------
def _mesh():
    return Mesh(np.asarray(jax.devices()[:EP]), axis_names=(AXIS, ))


def _make_weights(mesh, seed, *, hidden, inter, e_total):
    """(w1, w2, w1_scale, w2_scale) as fp8 global arrays sharded on experts.

    Built a shard at a time rather than on one device and handed out: GLM's
    128-expert draw is 4.8GB in f32 where its fp8 form is 1.2GB, and only
    the latter has to exist at once. Scales are per output channel, taken
    along each matmul's contraction axis, which is the layout the kernel
    takes: w1_scale [E, 2 * inter], w2_scale [E, hidden].
    """
    (axis, ) = mesh.axis_names
    g_local = e_total // mesh.shape[axis]

    def local(_):
        me = lax.axis_index(axis)
        k1, k2 = jax.random.split(jax.random.fold_in(jax.random.key(seed), me))
        out = []
        for key, shape in ((k1, (g_local, hidden, 2 * inter)),
                           (k2, (g_local, inter, hidden))):
            w = jax.random.normal(key, shape, jnp.float32) / 10
            amax = jnp.max(jnp.abs(w), axis=1, keepdims=True)
            scale = jnp.where(amax == 0, 1.0, amax / FP8_MAX)
            out.append(((w / scale).astype(FP8),
                        scale.astype(jnp.float32).reshape(shape[0], shape[2])))
        (w1, w1s), (w2, w2s) = out
        return w1, w2, w1s, w2s

    return jax.jit(
        jax.shard_map(local,
                      mesh=mesh,
                      in_specs=(P(axis), ),
                      out_specs=(P(axis), ) * 4,
                      check_vma=False))(jnp.zeros((EP, ), jnp.float32))


def _make_inputs(mesh, seed, *, tokens, hidden, e_total):
    """(x, gating) as global arrays sharded on the tokens."""
    (axis, ) = mesh.axis_names
    shard = NamedSharding(mesh, P(axis))

    @jax.jit
    def build():
        kx, kg = jax.random.split(jax.random.key(seed))
        x = (jax.random.normal(kx, (tokens, hidden), jnp.float32) / 10).astype(
            jnp.bfloat16)
        gating = jax.random.normal(kg, (tokens, e_total), jnp.float32)
        return (lax.with_sharding_constraint(x, shard),
                lax.with_sharding_constraint(gating, shard))

    return build()


def _make_score_bias(seed, e_total):
    """GLM's `e_score_correction_bias`: one value per GLOBAL expert.

    Replicated, not sharded -- every shard scores the full expert set, and
    the layer takes it under `P()` for that reason. Drawn at the same scale
    as the sigmoid scores it is added to, so it actually changes which
    experts win rather than being lost in the rounding.
    """
    return jax.random.normal(jax.random.key(seed),
                             (e_total, ), jnp.float32) * 0.1


def _dense_reference(mesh, x, w1, w2, w1_scale, w2_scale, gating, score_bias,
                     *, topk):
    """The plain MoE under GLM's router: sigmoid, biased top-k, dense FFN.

    Every expert is applied to every token in f32 and weighted by the router
    weight of the slots that chose it, which is zero for a token that did
    not. Nothing is quantized -- the weights are dequantized once and the
    whole body runs in f32 -- so the difference from the kernel is the
    kernel's transport rather than a second implementation of the same
    shortcuts.

    The router mirrors `layer.local_fn`: the scores are sigmoid, SELECTION
    is on `scores + score_bias` (`noaux_tc`), the WEIGHTS come from the
    unbiased scores, and they are renormalized to sum to one.

    The loop is spread over the same expert axis the layer shards on, which
    is a distribution of the reference and not a change of it: the summands
    and their order are the same.
    """
    (axis, ) = mesh.axis_names
    g_local = w1.shape[0] // mesh.shape[axis]
    inter = w1.shape[2] // 2

    def local(x_g, gating_g, bias_g, w1_l, w2_l, s1, s2):
        scores = jax.nn.sigmoid(gating_g.astype(jnp.float32))
        _, topk_idx = lax.top_k(scores + bias_g[None, :], topk)
        topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1)
        topk_weights = topk_weights / jnp.maximum(
            topk_weights.sum(axis=-1, keepdims=True), 1e-20)
        first = lax.axis_index(axis) * g_local
        x32 = x_g.astype(jnp.float32)

        def expert(e, acc):
            w1e = lax.dynamic_index_in_dim(w1_l, e, 0,
                                           keepdims=False).astype(jnp.float32)
            w2e = lax.dynamic_index_in_dim(w2_l, e, 0,
                                           keepdims=False).astype(jnp.float32)
            w1e = w1e * lax.dynamic_index_in_dim(s1, e, 0, keepdims=True)
            w2e = w2e * lax.dynamic_index_in_dim(s2, e, 0, keepdims=True)
            weight = jnp.sum(jnp.where(topk_idx == first + e, topk_weights,
                                       0.0),
                             axis=-1)[:, None]
            acc1 = x32 @ w1e
            row = (jax.nn.silu(acc1[:, :inter]) * acc1[:, inter:]) @ w2e
            return acc + weight * row

        mine = lax.fori_loop(0, g_local, expert, jnp.zeros_like(x32))
        return lax.psum(mine, axis)

    in_specs = (P(), P(), P()) + (P(axis), ) * 4
    return jax.jit(
        jax.shard_map(local,
                      mesh=mesh,
                      in_specs=in_specs,
                      out_specs=P(),
                      check_vma=False))(x, gating, score_bias, w1, w2,
                                        w1_scale, w2_scale)


def _relative_l2(actual, want):
    a = np.asarray(actual, np.float64)
    w = np.asarray(want, np.float64)
    denom = np.linalg.norm(w)
    return float(np.linalg.norm(a - w) / (denom if denom else 1.0))


def _worst_row_relative_l2(actual, want):
    """The largest per-row relative error. A routing failure is per token,
    and a batch-wide norm divides one bad row by every other row's size."""
    a = np.asarray(actual, np.float64)
    w = np.asarray(want, np.float64)
    per = np.linalg.norm(a - w, axis=-1)
    scale = np.linalg.norm(w, axis=-1)
    return float(np.max(per / np.where(scale == 0, 1.0, scale)))


# --------------------------------------------------------------------------
# the one GLM run, and the name it was built under
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def glm_run():
    """Run the layer once at GLM's shapes, recording what got built.

    One fixture rather than one run per assertion: the weights are 1.2GB and
    the build is a Mosaic compile, and splitting the run would also make the
    activation evidence order-dependent -- `kernel._BUILD_CACHE` and
    `layer._LAYER_SM_CACHE` both memoize, so only the FIRST call at a given
    config traces `make_call` at all. Recording around that first call is
    what makes `names` non-empty, and every test below asserts it is.
    """
    mesh = _mesh()
    w1, w2, w1s, w2s = _make_weights(mesh,
                                     11,
                                     hidden=GLM["hidden"],
                                     inter=GLM["inter"],
                                     e_total=GLM["e_total"])
    x, gating = _make_inputs(mesh,
                             12,
                             tokens=GLM["tokens"],
                             hidden=GLM["hidden"],
                             e_total=GLM["e_total"])
    score_bias = _make_score_bias(13, GLM["e_total"])
    jax.block_until_ready((w1, w2, w1s, w2s, x, gating, score_bias))

    names = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kernel_mod, "pl", _RecordingPallas(kernel_mod.pl, names))
        out = fused_ep_moe_v2(x,
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
                              score_bias=score_bias)
        out = jax.block_until_ready(out)

    run = types.SimpleNamespace(mesh=mesh,
                                out=out,
                                names=names,
                                x=x,
                                gating=gating,
                                score_bias=score_bias,
                                weights=(w1, w2, w1s, w2s))
    yield run
    del run, out, w1, w2, w1s, w2s, x, gating


# --------------------------------------------------------------------------
# 1. the blocked path actually ran
# --------------------------------------------------------------------------
def test_the_policy_asks_for_the_blocked_residency_at_glm_shapes():
    """`resolve_weight_kblock` is the whole policy; this is its GLM answer."""
    got = host.resolve_weight_kblock(GLM_G_LOCAL,
                                     CAPACITY,
                                     GLM["hidden"],
                                     GLM["inter"],
                                     weight_format=WeightFormat.FP8)
    assert got == EXPECTED_KBLOCK, (
        f"GLM-5.2-FP8's expert ({GLM['hidden']}x{GLM['inter']}, g_local "
        f"{GLM_G_LOCAL}) resolved to {got!r}, not {EXPECTED_KBLOCK!r}; the "
        "blocked residency is not what this shape would build")


def test_the_built_glm_kernel_is_the_blocked_one(glm_run):
    """The program the layer actually built carries the blocked scope tag.

    This is the assertion the whole file exists for, and it is deliberately
    NOT "the build did not raise". A build can succeed for reasons that have
    nothing to do with the arm under test, and an in-process A/B of this
    kernel has already once produced the BLOCKED program for both arms and
    reported a flawless bit-identical result: jit and closure reuse survived
    a `_BUILD_CACHE.clear()`, so both arms ran the same Mosaic program.
    The scope name is emitted inside `make_call`, once per build, on the
    blocked branch only -- so an equality against it cannot be satisfied by
    a cached program from the other arm.
    """
    built = _moe_build_names(glm_run.names)
    assert built, (
        "no fused_ep_moe_v2 pallas_call was recorded during the GLM run, so "
        "this test observed a CACHED program and proves nothing about which "
        "residency was built")
    expected = {GLM_KERNEL_NAME}
    assert set(built) == expected, (
        f"the GLM build(s) were named {sorted(set(built))}, expected exactly "
        f"[{GLM_KERNEL_NAME!r}]; a name without the "
        f"_kb{host.KBLOCK_HIDDEN}x{host.KBLOCK_INTER} tag is the "
        "whole-expert residency")


def test_a_fitting_shape_builds_without_the_blocked_tag():
    """The negative control on the evidence above.

    "The name contains `_kb2048x1024`" is only evidence if a name CAN come
    back without it. The small shape fits whole-expert, so its build must
    carry no residency tag at all -- which also re-states, on the build side,
    that the fallback is a fallback.
    """
    mesh = _mesh()
    w1, w2, w1s, w2s = _make_weights(mesh,
                                     14,
                                     hidden=SMALL["hidden"],
                                     inter=SMALL["inter"],
                                     e_total=SMALL["e_total"])
    x, gating = _make_inputs(mesh,
                             15,
                             tokens=SMALL["tokens"],
                             hidden=SMALL["hidden"],
                             e_total=SMALL["e_total"])
    names = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kernel_mod, "pl", _RecordingPallas(kernel_mod.pl, names))
        jax.block_until_ready(
            fused_ep_moe_v2(x,
                            w1,
                            w2,
                            w1s,
                            w2s,
                            gating,
                            topk=SMALL["topk"],
                            renormalize=True,
                            mesh=mesh,
                            capacity=CAPACITY,
                            weight_format=WeightFormat.FP8))
    built = _moe_build_names(names)
    assert built, ("no fused_ep_moe_v2 pallas_call was recorded for the "
                   "small shape, so this control observed a cached program")
    expected = {SMALL_KERNEL_NAME}
    assert set(built) == expected, (
        f"the small shape built {sorted(set(built))}, expected exactly "
        f"[{SMALL_KERNEL_NAME!r}]; a `_kb` tag here would mean the tag does "
        "not track the arm")


# --------------------------------------------------------------------------
# 2. numerics -- the layer, against a dense reference
# --------------------------------------------------------------------------
def test_the_blocked_layer_tracks_a_dense_reference_at_glm_shapes(glm_run):
    """The whole layer on the blocked program, against the dense f32 MoE.

    The bands are the fp8 transport's, not the contraction's: the routed
    rows cross the mesh as e4m3 and the intermediate is requantized, which
    is ~5e-2 of relative L2 before the contraction is even reached. The
    contraction's own 1e-6/1e-3/2e-2 bands are gated in the next section,
    where nothing is on a wire.

    The rotated-expert control is what makes a band this wide mean
    something: it is wide enough to cover an fp8 wire, so without a control
    a layer that sent every token to the wrong expert could still clear it.
    """
    w1, w2, w1s, w2s = glm_run.weights
    want = _dense_reference(glm_run.mesh,
                            glm_run.x,
                            w1,
                            w2,
                            w1s,
                            w2s,
                            glm_run.gating,
                            glm_run.score_bias,
                            topk=GLM["topk"])
    out = glm_run.out
    assert out.shape == glm_run.x.shape
    assert out.dtype == glm_run.x.dtype
    assert out.sharding.spec == glm_run.x.sharding.spec

    error = _relative_l2(out, want)
    worst = _worst_row_relative_l2(out, want)
    print(f"GLM blocked layer: relative_l2={error:.6f} "
          f"worst_token={worst:.6f} (bands {FP8_RELATIVE_BOUND}/"
          f"{FP8_TOKEN_BOUND})")
    assert error < FP8_RELATIVE_BOUND, (
        f"relative L2 {error:.4f} past the {FP8_RELATIVE_BOUND} band")
    assert worst < FP8_TOKEN_BOUND, (
        f"worst token's relative error {worst:.4f} past the "
        f"{FP8_TOKEN_BOUND} per-token band; the batch norm was {error:.4f}")

    rotated = _dense_reference(glm_run.mesh,
                               glm_run.x,
                               jnp.roll(w1, 1, axis=0),
                               jnp.roll(w2, 1, axis=0),
                               jnp.roll(w1s, 1, axis=0),
                               jnp.roll(w2s, 1, axis=0),
                               glm_run.gating,
                               glm_run.score_bias,
                               topk=GLM["topk"])
    control = _relative_l2(out, rotated)
    print(f"GLM blocked layer: rotated-expert control {control:.6f}, "
          f"{control / error:.1f}x the reference error")
    assert control > ROTATED_CONTROL_FACTOR * error, (
        f"a reference computed over rotated experts sits {control:.4f} from "
        f"the kernel against the true reference's {error:.4f}, under the "
        f"{ROTATED_CONTROL_FACTOR}x a band this wide needs to mean anything")


# --------------------------------------------------------------------------
# 3. numerics -- the contraction, against a float64 oracle
# --------------------------------------------------------------------------
def _quant_weight(w):
    """Per-output-channel fp8 quantization, the contraction axis reduced."""
    amax = jnp.max(jnp.abs(w), axis=0, keepdims=True)
    scale = jnp.where(amax == 0, 1.0, amax / FP8_MAX)
    return (w / scale).astype(FP8), scale.astype(jnp.float32)


def _rowquant_fp8(x_bf16):
    """`kernel.rowquant_fp8`, including its bf16 reciprocal."""
    amax = jnp.max(jnp.abs(x_bf16), axis=-1, keepdims=True).astype(jnp.float32)
    scale = amax / FP8_MAX
    usable = jnp.isfinite(scale) & (scale > 0)
    scale = jnp.where(usable, scale, 0.0)
    sinv = jnp.where(usable, 1.0 / jnp.where(usable, scale, 1.0), 0.0)
    q = jnp.where(usable, x_bf16 * sinv.astype(x_bf16.dtype),
                  jnp.zeros_like(x_bf16)).astype(FP8)
    return q, scale


def _rowquant_fp8_np(x_bf16):
    """The same, in numpy, for the oracle. Same bf16 arithmetic."""
    x32 = np.asarray(x_bf16, np.float32)
    amax = np.max(np.abs(x32), axis=-1, keepdims=True)
    scale = amax / FP8_MAX
    usable = np.isfinite(scale) & (scale > 0)
    scale = np.where(usable, scale, 0.0).astype(np.float32)
    sinv = np.where(usable, 1.0 / np.where(usable, scale, 1.0),
                    0.0).astype(np.float32)
    q = np.where(usable, x_bf16 * sinv.astype(BF16), np.zeros_like(x_bf16))
    return q.astype(FP8), scale


def _expert_operands(seed, m):
    """One expert's fp8 operands at GLM's shapes, heavy-tailed activations.

    2% of the hidden channels get a 20x standard deviation, a stand-in for
    the outlier channels real LLM hidden states carry: summation error is
    driven by cancellation, so the conditioning of the contraction is what a
    reassociation study is sensitive to, and the bands above were taken from
    the heavy-tailed case.
    """
    hidden, inter = GLM["hidden"], GLM["inter"]
    kx, k1, k2 = jax.random.split(jax.random.key(seed), 3)
    x = jax.random.normal(kx, (m, hidden), jnp.float32) / 10.0
    outlier = jax.random.uniform(jax.random.fold_in(kx, 1), (1, hidden)) < 0.02
    x = x * jnp.where(outlier, 20.0, 1.0)
    act_q, act_scale = _rowquant_fp8(x.astype(jnp.bfloat16))
    w1_q, w1s = _quant_weight(
        jax.random.normal(k1, (hidden, 2 * inter), jnp.float32) / 10.0)
    w2_q, w2s = _quant_weight(
        jax.random.normal(k2, (inter, hidden), jnp.float32) / 10.0)
    return jax.block_until_ready((act_q, act_scale, w1_q, w1s, w2_q, w2s))


def _dot(a, b):
    """One contraction: bf16 operands, fp32 accumulator.

    THE CAST IS EXACT AND IT IS LOAD-BEARING. An e4m3 code has a four-bit
    significand and a bf16 one has eight, so every fp8 operand and every
    pairwise product of two of them is representable in bf16 without
    rounding; what the MXU then accumulates in fp32 is the kernel's
    arithmetic, value for value.

    Handing the fp8 arrays to `dot_general` directly instead does NOT give
    that, on this hardware, and the difference is not subtle (measured at
    GLM's shapes, m=32, in j-9d8fe4cc):

        spelling        vs float64     blocked vs whole-K
        fp8 DEFAULT      4.9235e-03     0.0000e+00  (bit-identical)
        fp8 HIGHEST      4.9235e-03     0.0000e+00  (bit-identical)
        bf16 DEFAULT     6.3375e-08     7.6936e-08
        f32  HIGHEST     7.4055e-08     7.6666e-08

    XLA's f8e4m3 lowering on tpu7x sits 4.9e-3 from a float64 evaluation of
    the same codes -- five orders wider than the fp32 reassociation this
    section measures -- and it returns a BIT-IDENTICAL result whether the
    caller blocks the contraction or not, because it is already blocked
    internally on boundaries of its own. Both facts make it useless here:
    the first drowns the signal and the second removes it, and a test whose
    two arms are bit-identical for a reason that has nothing to do with the
    change is a test that cannot fail. `precision=HIGHEST` does not move
    either number. The bf16 spelling reproduces the CPU instrument's
    7.62e-08 to two figures, which is what says it is measuring the same
    thing the bands were taken from.
    """
    return lax.dot_general(a.astype(BF16),
                           b.astype(BF16),
                           DN,
                           preferred_element_type=jnp.float32)


def _contract(a, b, kb, *, permute_block=None):
    """The contraction, blocked exactly as the kernel's blocked body blocks it.

    `kb is None` is the single `dot_general` the whole-expert residency runs.
    `permute_block` reverses the ROW ORDER of one weight block, which leaves
    every shape and every block boundary intact -- the kind of wrongness a
    shape check cannot see.
    """
    if kb is None:
        assert permute_block is None
        return _dot(a, b)
    acc = None
    for i, b0 in enumerate(range(0, a.shape[1], kb)):
        wb = b[b0:b0 + kb]
        if permute_block == i:
            wb = wb[::-1]
        part = _dot(lax.slice_in_dim(a, b0, b0 + kb, axis=1), wb)
        acc = part if acc is None else acc + part
    return acc


def _ffn_fp32(ops, kb1, kb2, *, permute_block=None):
    """The fp8 expert FFN with a (possibly blocked) contraction, on device.

    The per-OUTPUT-CHANNEL `w1s` still lands once, in the epilogue, exactly
    where `expert_ffn_fp8` puts it -- copying `expert_ffn_blockscale`'s
    `w1s=None` instead would be uniformly wrong, and is the mutation the
    output band below is sized against. Returns (acc1, out).
    """
    act_q, act_scale, w1_q, w1s, w2_q, w2s = ops
    inter = GLM["inter"]
    acc1 = _contract(act_q, w1_q, kb1, permute_block=permute_block)
    assert acc1.dtype == jnp.float32, acc1.dtype
    scaled = acc1 * act_scale * w1s
    mid = (jax.nn.silu(scaled[:, :inter]) * scaled[:, inter:]).astype(BF16)
    mid_q, mid_scale = _rowquant_fp8(mid)
    out = _contract(mid_q, w2_q, kb2) * mid_scale * w2s
    return jax.block_until_ready((acc1, out))


def _ffn_f64(ops):
    """The same FFN in numpy float64. Ground truth, block-structure free.

    Every fp8 code and every f32 scale is exact in float64, so the only
    approximations left are the ones that are part of the ALGORITHM rather
    than of its precision -- the bf16 cast of the intermediate and its
    requantization to e4m3 -- and those are kept, because both device
    spellings do them identically and an oracle that skipped them would be
    measuring a different function.
    """
    act_q, act_scale, w1_q, w1s, w2_q, w2s = ops
    inter = GLM["inter"]
    a = np.asarray(act_q, np.float64)
    acc1 = a @ np.asarray(w1_q, np.float64)
    scaled = acc1 * np.asarray(act_scale, np.float64) * np.asarray(
        w1s, np.float64)
    gate, up = scaled[:, :inter], scaled[:, inter:]
    mid64 = (gate / (1.0 + np.exp(-gate))) * up
    mid_q, mid_scale = _rowquant_fp8_np(mid64.astype(BF16))
    acc2 = np.asarray(mid_q, np.float64) @ np.asarray(w2_q, np.float64)
    return acc1, acc2 * np.asarray(mid_scale, np.float64) * np.asarray(
        w2s, np.float64)


@pytest.mark.parametrize("m", CONTRACTION_ROWS)
def test_the_blocked_contraction_matches_the_whole_k_one(m):
    """The blocked contraction itself, at GLM's expert shapes, on device.

    The layer test above cannot see this: an fp8 wire sits between the
    kernel's output and any reference, two orders of magnitude wider than
    the reassociation being checked here. This runs the blocked loop the
    kernel runs -- (2048, 1024) over hidden 6144 and inter 2048 -- against
    the whole-K contraction the incumbent residency runs, in the same
    spelling on the same chip, and holds the difference to the bands.

    The float64 oracle -- the same fp8 codes evaluated in numpy float64,
    where they are exact and where there is no block structure at all -- is
    the second reference, and the accumulator is held to it directly. It is
    what distinguishes "the two spellings agree" from "the two spellings are
    both right": two arms can agree because neither of them is doing
    anything.
    """
    ops = _expert_operands(0, m)
    oracle_acc1, oracle_out = _ffn_f64(ops)
    blocked_acc1, blocked_out = _ffn_fp32(ops, *EXPECTED_KBLOCK)
    whole_acc1, whole_out = _ffn_fp32(ops, None, None)

    acc1_err = _relative_l2(blocked_acc1, whole_acc1)
    out_err = _relative_l2(blocked_out, whole_out)
    row_err = _worst_row_relative_l2(blocked_out, whole_out)
    blocked_f64 = _relative_l2(blocked_acc1, oracle_acc1)
    whole_f64 = _relative_l2(whole_acc1, oracle_acc1)
    out_f64 = _relative_l2(blocked_out, oracle_out)
    print(f"GLM contraction m={m}: blocked vs whole-K acc1={acc1_err:.3e} "
          f"out={out_err:.3e} worst_row={row_err:.3e} (bands {ACC1_RTOL}/"
          f"{OUT_RTOL}/{OUT_ROW_RTOL}) | vs float64 acc1 blocked="
          f"{blocked_f64:.3e} whole-K={whole_f64:.3e} out={out_f64:.3e}")

    # The two arms must not be bit-identical: if they were, the bands would
    # be measuring nothing. They differ because the blocked loop regroups
    # 6144 fp32 additions into three, which is the only thing this section
    # can see and the only thing it is here to bound.
    assert acc1_err > 0.0, (
        "the blocked and whole-K contractions came back bit-identical, so "
        "no reassociation happened and the bands below are vacuous; the "
        "loop is being folded away or both arms are the same program")
    assert acc1_err < ACC1_RTOL, (
        f"the blocked stage-1 accumulator sits {acc1_err:.3e} from the "
        f"whole-K one, past the {ACC1_RTOL} band; this is the band the "
        "algebraic claim sum_b (A_b . W_b) . s == (A . W) . s lives on")
    assert blocked_f64 < ACC1_RTOL, (
        f"the blocked stage-1 accumulator sits {blocked_f64:.3e} from the "
        f"float64 oracle, past the {ACC1_RTOL} band")
    assert out_err < OUT_RTOL, (
        f"the blocked FFN output sits {out_err:.3e} from the whole-K one, "
        f"past the {OUT_RTOL} band")
    assert row_err < OUT_ROW_RTOL, (
        f"the worst single row sits {row_err:.3e} from the whole-K one, "
        f"past the {OUT_ROW_RTOL} band; the batch norm was {out_err:.3e}")
    # Blocked partial sums are a pairwise reduction, so they should be at
    # least as accurate as one long chain. A factor rather than the CPU
    # instrument's 1.05x, because the MXU regroups a "whole-K" dot
    # internally too. What must not happen is a drift.
    assert blocked_f64 <= whole_f64 * 2.0, (
        f"blocked acc1 sits {blocked_f64:.3e} from float64 against whole-K's "
        f"{whole_f64:.3e}; a blocked contraction that is materially further "
        "from a block-structure-free ground truth is drifting, not "
        "reassociating")


@pytest.mark.parametrize("m", CONTRACTION_ROWS)
def test_the_degenerate_one_block_loop_is_bit_identical_to_whole_k(m):
    """A loop with one block must be the unblocked dot, bit for bit.

    This separates the two things a blocked candidate can get wrong. If it
    fails, the LOOP HARNESS above is wrong -- a slicing or an
    accumulator-dtype mistake -- and the numbers the previous test reports
    mean nothing. If it passes, every difference that test shows is
    summation order and nothing else.
    """
    ops = _expert_operands(0, m)
    whole = _ffn_fp32(ops, None, None)
    one = _ffn_fp32(ops, GLM["hidden"], GLM["inter"])
    np.testing.assert_array_equal(np.asarray(one[0]), np.asarray(whole[0]))
    np.testing.assert_array_equal(np.asarray(one[1]), np.asarray(whole[1]))


@pytest.mark.parametrize("m", CONTRACTION_ROWS)
def test_a_rotated_weight_block_misses_by_orders_of_magnitude(m):
    """The negative control on the contraction bands.

    Reversing one weight block's rows keeps every shape, every block
    boundary and every element of the weight. If a band this tight cannot
    see that, it cannot see a real blocking bug -- a wrong block offset, a
    short loop bound -- either.
    """
    ops = _expert_operands(0, m)
    whole_acc1 = _ffn_fp32(ops, None, None)[0]
    good = _relative_l2(_ffn_fp32(ops, *EXPECTED_KBLOCK)[0], whole_acc1)
    bad = _relative_l2(
        _ffn_fp32(ops, *EXPECTED_KBLOCK, permute_block=1)[0], whole_acc1)
    print(f"GLM contraction m={m}: rotated-block control {bad:.3e} vs "
          f"{good:.3e} against whole-K, {bad / max(good, 1e-30):.3e}x")
    assert bad > good * CONTRACTION_CONTROL_FACTOR, (
        f"reversing one weight block's rows moved acc1 to {bad:.3e} from "
        f"the good {good:.3e}, under the {CONTRACTION_CONTROL_FACTOR:.0e}x "
        "a control has to miss by")


@pytest.mark.parametrize("m", CONTRACTION_ROWS)
def test_a_bf16_partial_accumulator_is_caught_by_the_acc1_band(m):
    """The second control, and the one that sets the floor.

    Reversing a block's rows misses by 1e7x, which proves the comparison
    works and proves nothing about where its FLOOR is. The realistic way to
    get a blocked contraction wrong is not to scramble it but to let the
    partial sums land in bf16 -- a blocked loop has a named intermediate
    where an unblocked `dot_general` has none, so it is a slip the
    whole-expert spelling cannot make. It misses by ~3e3x, and that is the
    margin ACC1_RTOL is actually chosen against.

    The mutation is the ACCUMULATOR, not the operands: `_dot` already hands
    the MXU bf16 operands, which is exact for fp8 codes, and what this
    breaks is the fp32 running sum the blocked loop needs.
    """
    ops = _expert_operands(0, m)
    act_q, w1_q = ops[0], ops[2]
    whole_acc1 = _ffn_fp32(ops, None, None)[0]
    kb1 = EXPECTED_KBLOCK[0]
    acc = None
    for b0 in range(0, GLM["hidden"], kb1):
        part = _dot(lax.slice_in_dim(act_q, b0, b0 + kb1, axis=1),
                    w1_q[b0:b0 + kb1]).astype(BF16)
        acc = part if acc is None else (acc + part).astype(BF16)
    err = _relative_l2(acc.astype(jnp.float32), whole_acc1)
    print(f"GLM contraction m={m}: bf16-accumulator control {err:.3e} "
          f"against whole-K, band {ACC1_RTOL}")
    assert err > ACC1_RTOL * CONTRACTION_CONTROL_FACTOR, (
        f"a bf16 partial accumulator sits {err:.3e} from whole-K, which "
        f"the {ACC1_RTOL} band would not catch at the "
        f"{CONTRACTION_CONTROL_FACTOR:.0e}x a control needs")


# --------------------------------------------------------------------------
# 4. the guard discriminates, and the VMEM claim
# --------------------------------------------------------------------------
def test_a_shape_that_fits_whole_expert_keeps_it():
    """Qwen3.5-397B is the red control on the budget half of the policy.

    The blocked residency re-reads every weight byte once per ROW TILE where
    the whole-expert one reads it once per EXPERT, so a shape that fits must
    not pay for room it does not need. The premise is asserted too: if
    Qwen's whole-expert estimate ever stopped fitting, "returns None" would
    be right for the wrong reason and this test would stop discriminating.
    """
    whole = host.vmem_estimate_bytes(QWEN["g_local"],
                                     CAPACITY,
                                     QWEN["hidden"],
                                     QWEN["inter"],
                                     weight_format=WeightFormat.FP8,
                                     kblock=None)
    limit = host.vmem_limit()
    print(f"Qwen3.5-397B whole-expert VMEM {whole / 2**20:.3f} MiB of "
          f"{limit / 2**20:.3f} MiB")
    assert whole <= limit, (
        "the premise of this control is gone: Qwen3.5-397B no longer fits "
        f"whole-expert ({whole / 2**20:.3f} MiB of {limit / 2**20:.3f} MiB)")
    assert host.resolve_weight_kblock(QWEN["g_local"],
                                      CAPACITY,
                                      QWEN["hidden"],
                                      QWEN["inter"],
                                      weight_format=WeightFormat.FP8) is None


@pytest.mark.parametrize(
    "weight_format",
    [f for f in WeightFormat if f not in host.KBLOCKABLE_FORMATS])
def test_a_format_with_no_blocked_body_is_never_blocked(weight_format):
    """The format half of the policy, at the shape that would otherwise fire.

    Four-bit weights already block their contraction at the scale block and
    integer weights already chunk theirs at the widening chunk, so neither
    has a whole slab to give up; bf16 has no blocked body written for it at
    all, and `vmem_scratch_arrays` raises outright if asked for one on fp4.
    Asked at GLM's shapes, so the answer is the FORMAT's and not the
    budget's -- these shapes are over budget in every format.
    """
    assert host.resolve_weight_kblock(GLM_G_LOCAL,
                                      CAPACITY,
                                      GLM["hidden"],
                                      GLM["inter"],
                                      weight_format=weight_format) is None


def test_block_heights_that_do_not_divide_the_contraction_are_refused():
    """The third `None`, and the one with no diagnostic of its own.

    An expert 6144 x 1536 does not fit whole-expert either, but 1536 is not
    a whole number of KBLOCK_INTER, so there is no blocked layout to fall
    back to and the caller's own budget check has to raise instead. The
    premise -- that this shape really is over budget -- is asserted, so the
    test cannot pass by the shape quietly starting to fit.
    """
    over = host.vmem_estimate_bytes(GLM_G_LOCAL,
                                    CAPACITY,
                                    GLM["hidden"],
                                    1536,
                                    weight_format=WeightFormat.FP8,
                                    kblock=None)
    assert over > host.vmem_limit(), (
        f"6144x1536 now fits whole-expert at {over / 2**20:.3f} MiB, so this "
        "test no longer exercises the divisibility gate")
    assert 1536 % host.KBLOCK_INTER
    assert host.resolve_weight_kblock(GLM_G_LOCAL,
                                      CAPACITY,
                                      GLM["hidden"],
                                      1536,
                                      weight_format=WeightFormat.FP8) is None


def test_the_blocked_residency_is_what_makes_glm_fit_vmem():
    """The claim the change was made for, as an assertion rather than a note.

    Both directions, because only the pair is informative: whole-expert must
    be OVER the budget (otherwise the fallback would never fire and the
    blocked code is dead) and blocked must be UNDER it (otherwise the build
    raises and the kernel cannot serve the model at all).
    """
    limit = host.vmem_limit()
    common = dict(weight_format=WeightFormat.FP8)
    whole = host.vmem_estimate_bytes(GLM_G_LOCAL,
                                     CAPACITY,
                                     GLM["hidden"],
                                     GLM["inter"],
                                     kblock=None,
                                     **common)
    blocked = host.vmem_estimate_bytes(GLM_G_LOCAL,
                                       CAPACITY,
                                       GLM["hidden"],
                                       GLM["inter"],
                                       kblock=EXPECTED_KBLOCK,
                                       **common)
    print(f"GLM VMEM: whole-expert {whole / 2**20:.3f} MiB, blocked "
          f"{blocked / 2**20:.3f} MiB, limit {limit / 2**20:.3f} MiB")
    assert whole > limit, (
        f"GLM's whole-expert residency needs {whole / 2**20:.3f} MiB and the "
        f"budget is {limit / 2**20:.3f} MiB; if this ever stops being true "
        "the fallback is dead code")
    assert blocked <= limit, (
        f"GLM's blocked residency needs {blocked / 2**20:.3f} MiB, over the "
        f"{limit / 2**20:.3f} MiB budget")
