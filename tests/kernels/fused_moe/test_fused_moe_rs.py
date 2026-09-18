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
"""Device tests for the fused reduce-scatter MoE kernel, in pure JAX.

``fused_moe_func_rs`` is the whole EP MoE minus the router: gather, GMM1,
activation, GMM2, combine, and the ICI reduce-scatter, as one Pallas program.
The adapter (`layers/adapter/fused_moe_rs.py`) hands it vLLM's own
``topk_weights``/``topk_indices``, so routing is equal to the grouped-matmul
path by construction and what is left to check is everything after routing.

Two things are checked, and they are not the same question.

**Does it compile and fit VMEM at the served shape.** The sibling
``fused_ep_moe_v2`` kernel provably could not fit GLM-5.2 in VMEM (120.64 MiB
wanted against 62.72 MiB available, and tile size was inert), so "does the
Mosaic compiler accept these dims at all" outranks any numeric band. That is
``test_the_served_shape_compiles_and_runs``, which needs no reference: a
VMEM-exceeded shape fails at trace/compile time.

**Is it numerically the grouped-matmul path.** That is
``test_small_shape_tracks_a_dense_reference``, held to a dense f32 reference
over the same block-dequantized weights, with a rotated-expert control so a
band wide enough to cover fp8 still means something.

**Shapes are GLM-5.2-FP8's**, from `knowledge/models/GLM-5.2-FP8/refs/model.md`
(`config.json` @ GCS generation 1785565055944226): hidden 6144,
``moe_intermediate_size`` 2048, top-8, fp8 e4m3 with ``weight_block_size``
[128, 128]. Production shards those 256 routed experts over EP16 on a 2x2x2
slice, i.e. **16 local experts a core**. This pod is 2x2x1 -- eight cores --
so EP16 cannot be built here. The substitution is deliberate and is stated in
``EP``/``E_TOTAL`` below: **128 global experts over EP8**, which reproduces
production's 16 local experts a core (the number that sets the per-core weight
footprint and the kernel's group loop) at half the mesh width. ``SERVED_TOKENS``
is then chosen to hit production's ``size_m`` exactly rather than EP8's own.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.distributed.ep_mesh import (RS_DATA_AXIS_NAME,
                                               RS_EXPERT_AXIS_NAME)
from vllm_torchtpu.kernels.fused_moe_rs import fused_moe_func_rs

pytestmark = pytest.mark.multichip

FP8 = jnp.float8_e4m3fn


@pytest.fixture(scope="module", autouse=True)
def _require_eight_devices():
    """Skip the module unless this host really has the eight-way slice.

    A fixture rather than a `skipif`, for the reason
    `test_fused_ep_moe_v2.py` gives at length: a `skipif` is evaluated during
    collection and `jax.device_count()` builds the PJRT client to answer it,
    which locks every spawned vLLM worker in the same pytest run out of
    libtpu.
    """
    if jax.device_count() < 8:
        pytest.skip("the fused RS MoE kernel is an expert-parallel collective "
                    f"and this host reports {jax.device_count()} JAX devices, "
                    "fewer than the 8 it needs")


# The expert-parallel width this pod can build. Production is EP16 on a 2x2x2
# slice; see the module docstring for why 8 is what a 2x2x1 pod offers and
# what is held fixed across the substitution.
EP = 8

# GLM-5.2-FP8, config.json @ gen 1785565055944226.
HIDDEN = 6144
INTER = 2048
TOPK = 8
BLOCK = 128  # weight_block_size [128, 128]

# 256 routed experts over production's EP16 is 16 local a core. 128 over this
# pod's EP8 is also 16 local a core -- the per-core count is what is held.
E_TOTAL = 128
E_LOCAL = E_TOTAL // EP

# Node-wide tokens the kernel is handed. Production is --max-num-batched-tokens
# 1024 per engine x DP16 = 16384, so size_m = 16384 x topk 8 = 131072. Held at
# the production size_m rather than at EP8's own 8192 x 8 = 65536, because
# size_m is what sets the m-tiling, the scalar-prefetch tables and the
# `pack_indices` branch (which trips above 120k). 65536 is run too, as the
# count an EP8 node would really present.
SERVED_TOKENS = 16384
SERVED_SIZE_M = SERVED_TOKENS * TOPK  # 131072
HALF_TOKENS = 8192

# Structurally complete and cheap enough for a dense reference: same top-k,
# same [128, 128] block scales, same 16-local-experts-a-core geometry, with
# hidden and intermediate cut to one and two blocks.
SMALL = dict(hidden=512, inter=256, tokens=512)

# How far the kernel may sit from a dense f32 reference over the same
# block-dequantized weights. The kernel quantizes the activations online
# (`lhs_quant_block_size`) and accumulates in f32, so the band is an fp8
# activation wire's, in the same shape as the v2 suite's two bounds: a
# batch-wide relative L2 and the worst single token's.
RELATIVE_BOUND = 0.06
TOKEN_BOUND = 0.075

# How much further a reference computed over rotated experts has to sit than
# the reference itself, so a band this wide still discriminates. Same control
# and same factor as the v2 suite.
ROTATED_CONTROL_FACTOR = 10


def _mesh():
    """The ``(data=1, expert=EP)`` mesh `build_ep_rs_mesh` builds in serving.

    Built here from `jax.devices()` rather than through `build_ep_rs_mesh`,
    which reads vLLM's EP group and so needs a served model; the axis NAMES
    come from the same module, because `get_moe_expert_axis` searches a fixed
    tuple for the expert axis and a differently-named mesh resolves to a
    literal "expert" that is then unbound.
    """
    return Mesh(
        np.asarray(jax.devices()[:EP]).reshape(1, EP),
        axis_names=(RS_DATA_AXIS_NAME, RS_EXPERT_AXIS_NAME),
    )


def _kernel_scale_layout(block_scale):
    """[E, K/BLOCK, N/BLOCK] -> the [E, K/BLOCK, 1, N] the kernel indexes.

    `gmm_fused_rs_nodedup` DMAs a scale tile with
    ``w1_scale_ref.at[expert, pl.ds(k_id * nqb, nqb), :, pl.ds(n_off, tile_n)]``
    -- four indices, and the last slice is in N *elements*, width ``tile_n1``.
    Its unquantized placeholder is `jnp.zeros((1, 1, 1, 1))`, which pins the
    rank. So a [128, 128]-block checkpoint reaches this kernel by repeating
    each N block across its 128 columns; that is value-identical to the block
    scale and is what the reference dequantizes with.
    """
    return jnp.repeat(block_scale, BLOCK, axis=-1)[:, :, None, :]


def _quantize_blockwise(w_f32):
    """Block-quantize [K, N] f32 to (fp8 [K, N], f32 [K/BLOCK, N/BLOCK]).

    One scale per [BLOCK, BLOCK] tile, which is the checkpoint's own scheme.
    """
    size_k, size_n = w_f32.shape
    tiles = w_f32.reshape(size_k // BLOCK, BLOCK, size_n // BLOCK, BLOCK)
    amax = jnp.max(jnp.abs(tiles), axis=(1, 3), keepdims=True)
    scale = jnp.where(amax == 0, 1.0, amax / float(jnp.finfo(FP8).max))
    q = (tiles / scale).astype(FP8).reshape(size_k, size_n)
    return q, scale.reshape(size_k // BLOCK, size_n // BLOCK).astype(
        jnp.float32)


def _make_weights(mesh, seed, *, hidden, inter):
    """(w1, w2, w1_scale, w2_scale) as global arrays sharded on the experts.

    K-major, which is the layout the kernel's GMM reads and what the adapter's
    `relayout_weights_for_rs` transposes vLLM's storage into: w1 is
    [E, hidden, 2*inter] and w2 is [E, inter, hidden]. Scales come back in the
    kernel's own 4D form.

    Built a shard at a time rather than on one device and handed out, so the
    served shape's f32 draw never has to exist whole.
    """
    axis = RS_EXPERT_AXIS_NAME

    def local(_):
        me = lax.axis_index(axis)
        k1, k2 = jax.random.split(jax.random.fold_in(jax.random.key(seed), me))
        out = []
        for key, shape in ((k1, (E_LOCAL, hidden, 2 * inter)),
                           (k2, (E_LOCAL, inter, hidden))):
            w = jax.random.normal(key, shape, jnp.float32) / 10
            q, s = jax.vmap(_quantize_blockwise)(w)
            out.append((q, _kernel_scale_layout(s)))
        (w1, w1s), (w2, w2s) = out
        return w1, w2, w1s, w2s

    return jax.jit(
        jax.shard_map(local,
                      mesh=mesh,
                      in_specs=(P(axis), ),
                      out_specs=(P(axis), ) * 4,
                      check_vma=False))(jnp.zeros((EP, ), jnp.float32))


def _make_cheap_weights(mesh, *, hidden, inter):
    """The served shape's weights, without ever materializing an f32 draw.

    A [128, 6144, 4096] f32 draw is 12.9 GB; the fp8 form is 3.2 GB, and only
    the latter has to exist. Values are a cheap deterministic ramp over the
    last axis broadcast across the rest, because this pair of tests asks
    whether Mosaic accepts the shape, not what number comes out.
    """
    axis = RS_EXPERT_AXIS_NAME

    def local(_):
        me = lax.axis_index(axis)
        out = []
        for off, shape in ((0, (E_LOCAL, hidden, 2 * inter)),
                           (3, (E_LOCAL, inter, hidden))):
            ramp = 0.25 + 0.125 * (
                (jnp.arange(shape[-1], dtype=jnp.int32) + off + me) % 8)
            w = jnp.broadcast_to(ramp.astype(FP8), shape)
            scale_shape = (shape[0], shape[1] // BLOCK, 1, shape[2])
            out.append((w, jnp.full(scale_shape, 0.5, jnp.float32)))
        (w1, w1s), (w2, w2s) = out
        return w1, w2, w1s, w2s

    return jax.jit(
        jax.shard_map(local,
                      mesh=mesh,
                      in_specs=(P(axis), ),
                      out_specs=(P(axis), ) * 4,
                      check_vma=False))(jnp.zeros((EP, ), jnp.float32))


def _make_inputs(mesh, seed, *, tokens, hidden):
    """(x, topk_weights, topk_indices, scores) the adapter's contract supplies.

    The adapter runs vLLM's `moe_routing.route` and passes the result in, so
    the routing here is computed outside the kernel exactly as it is in
    serving: a softmax over the router scores, top-k, renormalized. The scores
    are returned as well because the dense reference has to weight by the same
    numbers.
    """

    @jax.jit
    def build():
        kx, kg = jax.random.split(jax.random.key(seed))
        x = (jax.random.normal(kx, (tokens, hidden), jnp.float32) / 10).astype(
            jnp.bfloat16)
        scores = jax.nn.softmax(jax.random.normal(kg, (tokens, E_TOTAL),
                                                  jnp.float32),
                                axis=-1)
        w, idx = lax.top_k(scores, TOPK)
        w = (w / w.sum(axis=-1, keepdims=True)).astype(jnp.bfloat16)
        rep = NamedSharding(mesh, P())
        # The f32 copy the reference weights by is the bf16 one widened, not
        # the pre-rounding value: the kernel never sees the latter, so
        # referencing against it would charge the kernel for a rounding the
        # caller did.
        return (lax.with_sharding_constraint(x, rep),
                lax.with_sharding_constraint(w, rep),
                lax.with_sharding_constraint(idx.astype(jnp.int32), rep),
                lax.with_sharding_constraint(w.astype(jnp.float32), rep))

    return build()


def _run(mesh, x, w1, w2, w1_scale, w2_scale, topk_weights, topk_indices):
    """Call the kernel the way `layers/adapter/fused_moe_rs.py` calls it.

    Same argument order, same dead `renormalize=False` / `"softmax"` pair --
    both are only read when the kernel computes its own top-k, which supplying
    both routing tensors skips.
    """
    return fused_moe_func_rs(
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        None,  # w1_bias
        None,  # w2_bias
        None,  # gating_output, unused when routing is supplied
        TOPK,
        False,  # renormalize
        mesh,
        "silu",
        "softmax",
        topk_weights=topk_weights,
        topk_indices=topk_indices,
    )


def _dense_reference(mesh, x, w1, w2, w1_scale, w2_scale, topk_weights,
                     topk_indices):
    """Every expert applied to every token in f32, weighted by its routing.

    The weights are dequantized from the same [128, 128] blocks the kernel
    reads -- the 4D scale's N axis is already per column, so undoing it is a
    plain multiply -- and the whole body runs in f32. What is left between
    this and the kernel is the kernel's online activation quantization and its
    transport, not a second implementation of the arithmetic.

    Spread over the same expert axis the kernel shards on: each shard applies
    its own experts to the whole batch and the shards are summed. That is a
    distribution of the reference, not a change of it.
    """
    axis = RS_EXPERT_AXIS_NAME
    inter = w2.shape[1]

    def local(x_g, tw, ti, w1_l, w2_l, s1, s2):
        first = lax.axis_index(axis) * E_LOCAL
        x32 = x_g.astype(jnp.float32)

        def expert(e, acc):
            w1e = lax.dynamic_index_in_dim(w1_l, e, 0,
                                           keepdims=False).astype(jnp.float32)
            w2e = lax.dynamic_index_in_dim(w2_l, e, 0,
                                           keepdims=False).astype(jnp.float32)
            s1e = lax.dynamic_index_in_dim(s1, e, 0, keepdims=False)[:, 0, :]
            s2e = lax.dynamic_index_in_dim(s2, e, 0, keepdims=False)[:, 0, :]
            w1e = w1e * jnp.repeat(s1e, BLOCK, axis=0)
            w2e = w2e * jnp.repeat(s2e, BLOCK, axis=0)
            weight = jnp.sum(jnp.where(ti == first + e, tw, 0.0),
                             axis=-1)[:, None]
            acc1 = x32 @ w1e
            row = (jax.nn.silu(acc1[:, :inter]) * acc1[:, inter:]) @ w2e
            return acc + weight * row

        mine = lax.fori_loop(0, E_LOCAL, expert, jnp.zeros_like(x32))
        return lax.psum(mine, axis)

    return jax.jit(
        jax.shard_map(local,
                      mesh=mesh,
                      in_specs=(P(), P(), P(), P(axis), P(axis), P(axis),
                                P(axis)),
                      out_specs=P(),
                      check_vma=False))(x, topk_weights, topk_indices, w1, w2,
                                        w1_scale, w2_scale)


def _relative_l2(actual, want):
    a = np.asarray(actual, np.float64)
    w = np.asarray(want, np.float64)
    return float(np.linalg.norm(a - w) / np.linalg.norm(w))


def _worst_token_relative_l2(actual, want):
    """The largest per-token relative error. A routing or gather failure is
    per token, and a batch-wide norm divides one bad row by every other."""
    a = np.asarray(actual, np.float64)
    w = np.asarray(want, np.float64)
    per_token = np.linalg.norm(a - w, axis=-1)
    scale = np.linalg.norm(w, axis=-1)
    return float(np.max(per_token / np.where(scale == 0, 1.0, scale)))


def test_the_expert_axis_name_is_the_one_the_kernel_resolves():
    """`get_moe_expert_axis` searches a fixed tuple, so the mesh cannot be
    named freely -- `build_ep_rs_mesh`'s choice has to be in that tuple."""
    from vllm_torchtpu.kernels.fused_moe_rs.fused_moe_rs import (
        get_moe_expert_axis)
    mesh = _mesh()
    assert get_moe_expert_axis(mesh) == RS_EXPERT_AXIS_NAME
    assert mesh.shape[RS_DATA_AXIS_NAME] == 1
    assert mesh.shape[RS_EXPERT_AXIS_NAME] == EP


def test_the_adapter_relayout_produces_the_scale_rank_the_kernel_indexes():
    """What `relayout_weights_for_rs` hands the kernel, against what the
    kernel's scale DMA indexes.

    The kernel slices ``w1_scale_ref.at[e, pl.ds(...), :, pl.ds(...)]`` -- four
    indices, the last in N elements -- and its unquantized placeholder is
    `jnp.zeros((1, 1, 1, 1))`. So the scale it takes is
    [E, size_k // block, 1, size_n]. This test asks the adapter for the same
    thing, on one layer's worth of GLM-5.2-shaped weights.
    """
    import torch

    from vllm_torchtpu.layers.adapter.fused_moe_rs import (
        RS_W1_SCALE_ATTR, RS_W2_SCALE_ATTR, relayout_weights_for_rs)

    class _Layer:
        pass

    def _param(*shape, dtype=torch.float32):
        return torch.nn.Parameter(torch.zeros(shape, dtype=dtype),
                                  requires_grad=False)

    # Two experts rather than the layer's 16: the assertion is about rank and
    # the last axis's units, and the served expert count would cost 1.2 GB of
    # host copies to say the same thing.
    e = 2
    layer = _Layer()
    # vLLM's storage layout for a block-scaled fp8 MoE layer.
    layer.w13_weight = _param(e, 2 * INTER, HIDDEN, dtype=torch.float8_e4m3fn)
    layer.w2_weight = _param(e, HIDDEN, INTER, dtype=torch.float8_e4m3fn)
    layer.w13_weight_scale_inv = _param(e, (2 * INTER) // BLOCK,
                                        HIDDEN // BLOCK)
    layer.w2_weight_scale_inv = _param(e, HIDDEN // BLOCK, INTER // BLOCK)

    relayout_weights_for_rs(layer)
    w1_scale = getattr(layer, RS_W1_SCALE_ATTR)
    w2_scale = getattr(layer, RS_W2_SCALE_ATTR)

    want1 = (e, HIDDEN // BLOCK, 1, 2 * INTER)
    want2 = (e, INTER // BLOCK, 1, HIDDEN)
    assert tuple(w1_scale.shape) == want1, (
        f"relayout_weights_for_rs gave w1_scale {tuple(w1_scale.shape)}; the "
        f"kernel's scale DMA indexes {want1} -- four dims, the last in N "
        "elements")
    assert tuple(w2_scale.shape) == want2, (
        f"relayout_weights_for_rs gave w2_scale {tuple(w2_scale.shape)}; the "
        f"kernel's scale DMA indexes {want2}")


def test_small_shape_tracks_a_dense_reference():
    """The kernel end to end at a small but structurally complete shape.

    The rotated-expert control is what makes the band mean something: it is
    wide enough to cover an fp8 activation wire, so without the control a
    kernel that gathered the wrong rows could still clear it.
    """
    mesh = _mesh()
    hidden, inter, tokens = SMALL["hidden"], SMALL["inter"], SMALL["tokens"]
    w1, w2, w1_scale, w2_scale = _make_weights(mesh, 0, hidden=hidden,
                                               inter=inter)
    x, tw, ti, tw32 = _make_inputs(mesh, 1, tokens=tokens, hidden=hidden)

    out = jax.block_until_ready(
        _run(mesh, x, w1, w2, w1_scale, w2_scale, tw, ti))
    assert out.shape == x.shape
    assert out.dtype == x.dtype

    want = _dense_reference(mesh, x, w1, w2, w1_scale, w2_scale, tw32, ti)
    error = _relative_l2(out, want)
    worst = _worst_token_relative_l2(out, want)
    print(f"RS_MOE_SMALL relative_l2={error:.5f} worst_token={worst:.5f}")
    assert error < RELATIVE_BOUND, (
        f"relative L2 {error:.4f} past the {RELATIVE_BOUND} band")
    assert worst < TOKEN_BOUND, (
        f"worst token's relative error {worst:.4f} past the {TOKEN_BOUND} "
        f"per-token band; the batch norm was {error:.4f}")

    rotated = _dense_reference(mesh, x, jnp.roll(w1, 1, axis=0),
                               jnp.roll(w2, 1, axis=0),
                               jnp.roll(w1_scale, 1, axis=0),
                               jnp.roll(w2_scale, 1, axis=0), tw32, ti)
    control = _relative_l2(out, rotated)
    print(f"RS_MOE_SMALL rotated_control={control:.5f}")
    assert control > ROTATED_CONTROL_FACTOR * error, (
        f"a reference over rotated experts sits {control:.4f} away against "
        f"the real reference's {error:.4f}; the band does not discriminate")


@pytest.mark.parametrize("tokens", [HALF_TOKENS, SERVED_TOKENS])
def test_the_served_shape_compiles_and_runs(tokens):
    """GLM-5.2's MoE block at the served ``size_m``: does Mosaic take it.

    No reference and no band. The question is the one that killed
    ``fused_ep_moe_v2`` on this model -- whether the tiles the kernel selects
    fit VMEM at hidden 6144 / intermediate 2048 / 16 local experts -- and a
    shape that does not fit raises at trace or compile time, before a single
    number is produced.

    ``tokens=16384`` is production's node-wide count (1024 per engine x DP16),
    so ``size_m`` is 131072, which is also the only one of the two that crosses
    the kernel's ``pack_indices`` threshold at 120k. ``tokens=8192`` is what an
    EP8 node would present at the same per-engine cap.
    """
    mesh = _mesh()
    size_m = tokens * TOPK
    w1, w2, w1_scale, w2_scale = _make_cheap_weights(mesh,
                                                     hidden=HIDDEN,
                                                     inter=INTER)

    @jax.jit
    def build_inputs():
        kx, kg = jax.random.split(jax.random.key(2))
        x = (jax.random.normal(kx, (tokens, HIDDEN), jnp.float32) /
             10).astype(jnp.bfloat16)
        scores = jax.random.normal(kg, (tokens, E_TOTAL), jnp.float32)
        w, idx = lax.top_k(scores, TOPK)
        w = jax.nn.softmax(w, axis=-1).astype(jnp.bfloat16)
        rep = NamedSharding(mesh, P())
        return (lax.with_sharding_constraint(x, rep),
                lax.with_sharding_constraint(w, rep),
                lax.with_sharding_constraint(idx.astype(jnp.int32), rep))

    x, tw, ti = build_inputs()
    print(f"RS_MOE_SERVED tokens={tokens} size_m={size_m} ep={EP} "
          f"e_total={E_TOTAL} e_local={E_LOCAL} hidden={HIDDEN} "
          f"inter={INTER} topk={TOPK} block={BLOCK}")
    out = jax.block_until_ready(
        _run(mesh, x, w1, w2, w1_scale, w2_scale, tw, ti))
    assert out.shape == (tokens, HIDDEN)
    assert out.dtype == jnp.bfloat16
    finite = bool(np.isfinite(np.asarray(out, np.float32)).all())
    print(f"RS_MOE_SERVED tokens={tokens} ran; all_finite={finite}")
