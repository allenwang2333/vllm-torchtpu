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

Three things are checked, and they are not the same question.

**Does it compile and fit VMEM at the served shape.** The sibling
``fused_ep_moe_v2`` kernel provably could not fit GLM-5.2 in VMEM (120.64 MiB
wanted against 62.72 MiB available, and tile size was inert), so "does the
Mosaic compiler accept these dims at all" outranks any numeric band. That is
``test_the_served_shape_compiles_and_runs``, which needs no reference: a
VMEM-exceeded shape fails at trace/compile time.

**Is it numerically the grouped-matmul path.** That is
``test_small_shape_tracks_a_dense_reference``, held to a dense f32 reference
over the same dequantized weights, with a rotated-expert control so a band
wide enough to cover fp8 still means something.

**Does the adapter hand it a layout it accepts.** That is
``test_bind_weights_passes_the_loaders_layout_through_unchanged``,
``test_the_kernel_accepts_what_the_adapter_hands_it`` and
``test_the_contract_check_refuses_a_scale_in_n_blocks``. This is where the
first version of this file was wrong, so it is worth stating precisely what
the layout is and where it is known from.

**THE WEIGHT LAYOUT IS THE LOADER'S, NOT THE CHECKPOINT'S.** This is READ FROM
A SERVER LOG -- job ``j-3c919167-e3fa-42ad-a8a6``, GLM-5.2-FP8 on DP16/EP16 --
not derived here. `process_weights_after_loading` runs
`_process_fp8_moe_weights` before either MoE adapter sees the layer: it
dequantizes the checkpoint, **requantizes** it (per-channel by default, since
`MOE_REQUANTIZE_BLOCK_SIZE` is unset) and transposes. That job logged

    FP8 weights transposed for GMM kernel: w13=[16, 6144, 4096],
    w2=[16, 2048, 6144], w13_scale=[16, 1, 1, 4096],
    w2_scale=[16, 1, 1, 6144], requant_block_size=per-channel

so what the adapter is handed is [E, hidden, 2*inter] / [E, inter, hidden] and
four-dimensional scales [E, K_blocks, 1, N] with **K_blocks 1** and the last
axis in N **elements**. The checkpoint's own [128, 128] block scaling is gone
by then; it never reaches this kernel. Per-channel is the K_blocks == 1 case
of the same contract -- `_recover_quant_block_size(size_k, 1)` short-circuits
to `size_k`, one quant block spanning all of K.

That is why the numeric and served-shape arms are parametrized over
``per_channel`` and ``block128``: **per_channel is what production runs**, and
block128 is only reachable by setting `MOE_REQUANTIZE_BLOCK_SIZE`. An earlier
version of this file ran block128 alone and reported it as the served shape.

**Shapes are GLM-5.2-FP8's**, from `knowledge/models/GLM-5.2-FP8/refs/model.md`
(`config.json` @ GCS generation 1785565055944226): hidden 6144,
``moe_intermediate_size`` 2048, top-8. Production shards 256 routed experts
over EP16 on a 2x2x2 slice, i.e. **16 local experts a core**. This pod is
2x2x1 -- eight cores -- so EP16 cannot be built here. The substitution is
deliberate and is stated in ``EP``/``E_TOTAL`` below: **128 global experts over
EP8**, which reproduces production's 16 local experts a core (the number that
sets the per-core weight footprint and the kernel's group loop) at half the
mesh width. ``SERVED_TOKENS`` is then chosen to hit production's ``size_m``
exactly rather than EP8's own.
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

# The requantization block size `MOE_REQUANTIZE_BLOCK_SIZE` would select. Not
# the checkpoint's [128, 128] -- that is dequantized away before the adapter
# runs -- but the same number, and the only non-per-channel case reachable.
BLOCK = 128

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
# same scale schemes, same 16-local-experts-a-core geometry, with hidden and
# intermediate cut to one and two 128-blocks.
SMALL = dict(hidden=512, inter=256, tokens=512)

# How far the kernel may sit from a dense f32 reference over the same
# weights, as a batch-wide relative L2 and the worst single token's -- the
# v2 suite's two bounds, because one unlucky row hides inside a batch norm.
#
# The bf16 band is the one that means "the arithmetic is right": with
# unquantized weights the kernel quantizes nothing, so what is left between it
# and the reference is bf16 rounding in the two matmuls.
BF16_RELATIVE_BOUND = 0.01
BF16_TOKEN_BOUND = 0.01

# The fp8 band is wider because it is a *transport* band, not an arithmetic
# one: the kernel online-quantizes each activation row to e4m3 in blocks of
# `lhs_quant_block_size` (256 at this hidden size) on the way into both GMMs,
# and e4m3's three mantissa bits are ~6% at full scale. Measured on this host
# at the small shape: 0.0652 batch, 0.0793 worst token for block128. That
# number is only admissible as a band because two controls in the same test
# bound it from both sides -- the bf16 arm above, an order of magnitude
# tighter over the same code path, and the rotated-expert reference below, an
# order of magnitude wider.
FP8_RELATIVE_BOUND = 0.09
FP8_TOKEN_BOUND = 0.11

# How much further a reference computed over rotated experts has to sit than
# the reference itself, so a band this wide still discriminates. Same control
# and same factor as the v2 suite.
ROTATED_CONTROL_FACTOR = 10

# The two scale schemes the kernel can be handed. `per_channel` is what
# `_process_fp8_moe_weights` produces by default and therefore what production
# runs; `block128` needs `MOE_REQUANTIZE_BLOCK_SIZE=128`.
SCHEMES = ["per_channel", "block128"]


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


def _quantize(w_f32, scheme):
    """[K, N] f32 -> (fp8 [K, N], f32 scale [K_blocks, N]).

    The scale comes back in the kernel's own units: K in **blocks**, N in
    **elements**, which is what `w1_scale_ref.at[e, pl.ds(...), :, pl.ds(...)]`
    indexes. `per_channel` is the K_blocks == 1 case, one scale per output
    column -- exactly what `_process_fp8_moe_weights` emits by default.
    """
    size_k, size_n = w_f32.shape
    k_block = size_k if scheme == "per_channel" else BLOCK
    tiles = w_f32.reshape(size_k // k_block, k_block, size_n)
    amax = jnp.max(jnp.abs(tiles), axis=1, keepdims=True)
    scale = jnp.where(amax == 0, 1.0, amax / float(jnp.finfo(FP8).max))
    q = (tiles / scale).astype(FP8).reshape(size_k, size_n)
    return q, scale.reshape(size_k // k_block, size_n).astype(jnp.float32)


def _make_weights(mesh, seed, *, hidden, inter, scheme):
    """(w1, w2, w1_scale, w2_scale) as global arrays sharded on the experts.

    K-major with four-dimensional scales, which is what the loader produces
    and what `bind_weights_for_rs` passes straight through: w1 is
    [E, hidden, 2*inter], w2 is [E, inter, hidden], scales
    [E, K_blocks, 1, N]. `scheme="bf16"` quantizes nothing and returns no
    scales.

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
            if scheme == "bf16":
                out.append((w.astype(jnp.bfloat16), ))
                continue
            q, s = jax.vmap(lambda a: _quantize(a, scheme))(w)
            out.append((q, s[:, :, None, :]))
        return tuple(a for pair in out for a in pair)

    quantized = scheme != "bf16"
    built = jax.jit(
        jax.shard_map(local,
                      mesh=mesh,
                      in_specs=(P(axis), ),
                      out_specs=(P(axis), ) * (4 if quantized else 2),
                      check_vma=False))(jnp.zeros((EP, ), jnp.float32))
    if quantized:
        w1, w1s, w2, w2s = built
        return w1, w2, w1s, w2s
    w1, w2 = built
    return w1, w2, None, None


def _make_cheap_weights(mesh, *, hidden, inter, scheme):
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
            k_blocks = 1 if scheme == "per_channel" else shape[1] // BLOCK
            out.append(
                (w, jnp.full((shape[0], k_blocks, 1, shape[2]), 0.5,
                             jnp.float32)))
        (w1, w1s), (w2, w2s) = out
        return w1, w2, w1s, w2s

    return jax.jit(
        jax.shard_map(local,
                      mesh=mesh,
                      in_specs=(P(axis), ),
                      out_specs=(P(axis), ) * 4,
                      check_vma=False))(jnp.zeros((EP, ), jnp.float32))


def _make_inputs(mesh, seed, *, tokens, hidden):
    """(x, topk_weights, topk_indices, weights_as_f32) the adapter supplies.

    The adapter runs vLLM's `moe_routing.route` and passes the result in, so
    the routing here is computed outside the kernel exactly as it is in
    serving: a softmax over the router scores, top-k, renormalized.
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


def _rank(mesh):
    """Each chip's mesh index as ``int32[EP, 1]`` sharded on the expert axis.

    The kernel's `_shard_body` reads `rank_l[0, 0]` and derives both its
    `group_offset` and the `shard_index` it hands the local body from it, so
    this operand is what decides which global experts this chip owns. The
    adapter builds the same thing per rank as a torch tensor --
    `layers/adapter/fused_moe_rs.py:357`, `torch.tensor([[ep_mesh_index()]])`
    sharded `P(RS_EXPERT_AXIS_NAME)` -- which is the global
    `arange(EP).reshape(EP, 1)` below, one row a chip, in mesh order.

    It is not optional: `_shard_body` passes `shard_index=rank_l[0, 0]`
    unconditionally, so the `lax.axis_index` fallback kept in
    `moe_gmm_local_rs_nodedup` for JAX-native callers is not reachable through
    `expert_parallel_gmm_rs`, and leaving `rank` at its `None` default raises
    `TypeError: 'NoneType' object is not subscriptable` during tracing.
    """
    return jax.device_put(
        jnp.arange(EP, dtype=jnp.int32).reshape(EP, 1),
        NamedSharding(mesh, P(RS_EXPERT_AXIS_NAME)),
    )


def _run(mesh, x, w1, w2, w1_scale, w2_scale, topk_weights, topk_indices):
    """Call the kernel the way `layers/adapter/fused_moe_rs.py` calls it.

    Same argument order, same dead `renormalize=False` / `"softmax"` pair --
    both are only read when the kernel computes its own top-k, which supplying
    both routing tensors skips. `rank` is the adapter's `_RS_RANK_BUFFER`; see
    `_rank`.
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
        rank=_rank(mesh),
    )


def _dense_reference(mesh, x, w1, w2, w1_scale, w2_scale, topk_weights,
                     topk_indices):
    """Every expert applied to every token in f32, weighted by its routing.

    The weights are dequantized from the same scales the kernel reads. The
    repeat factor is derived from the scale's own K extent rather than from
    `BLOCK`, so the same reference serves per-channel (K_blocks 1, repeat K)
    and block128 (K_blocks K/128, repeat 128); hardcoding 128 would silently
    compare against the wrong dequantization on the per-channel arm, which is
    the one production runs.

    Spread over the same expert axis the kernel shards on: each shard applies
    its own experts to the whole batch and the shards are summed. That is a
    distribution of the reference, not a change of it.
    """
    axis = RS_EXPERT_AXIS_NAME
    inter = w2.shape[1]
    scaled = w1_scale is not None

    def local(x_g, tw, ti, w1_l, w2_l, *scales):
        s1, s2 = scales if scaled else (None, None)
        first = lax.axis_index(axis) * E_LOCAL
        x32 = x_g.astype(jnp.float32)

        def expert(e, acc):
            w1e = lax.dynamic_index_in_dim(w1_l, e, 0,
                                           keepdims=False).astype(jnp.float32)
            w2e = lax.dynamic_index_in_dim(w2_l, e, 0,
                                           keepdims=False).astype(jnp.float32)
            if scaled:
                s1e = lax.dynamic_index_in_dim(s1, e, 0,
                                               keepdims=False)[:, 0, :]
                s2e = lax.dynamic_index_in_dim(s2, e, 0,
                                               keepdims=False)[:, 0, :]
                w1e = w1e * jnp.repeat(
                    s1e, w1e.shape[0] // s1e.shape[0], axis=0)
                w2e = w2e * jnp.repeat(
                    s2e, w2e.shape[0] // s2e.shape[0], axis=0)
            weight = jnp.sum(jnp.where(ti == first + e, tw, 0.0),
                             axis=-1)[:, None]
            acc1 = x32 @ w1e
            row = (jax.nn.silu(acc1[:, :inter]) * acc1[:, inter:]) @ w2e
            return acc + weight * row

        mine = lax.fori_loop(0, E_LOCAL, expert, jnp.zeros_like(x32))
        return lax.psum(mine, axis)

    in_specs = (P(), P(), P(), P(axis), P(axis)) + ((P(axis), ) * 2 if scaled
                                                   else ())
    args = (x, topk_weights, topk_indices, w1, w2) + (
        (w1_scale, w2_scale) if scaled else ())
    return jax.jit(
        jax.shard_map(local,
                      mesh=mesh,
                      in_specs=in_specs,
                      out_specs=P(),
                      check_vma=False))(*args)


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


def _check_against_reference(mesh, tag, x, w1, w2, w1_scale, w2_scale, tw32,
                             ti, out, batch_bound, token_bound):
    """Hold `out` to the dense reference, and rotate the experts as a control.

    Every number is printed before anything is asserted, so a red arm still
    reports what the others measured.
    """
    want = _dense_reference(mesh, x, w1, w2, w1_scale, w2_scale, tw32, ti)
    error = _relative_l2(out, want)
    worst = _worst_token_relative_l2(out, want)

    def roll(a):
        return None if a is None else jnp.roll(a, 1, axis=0)

    rotated = _dense_reference(mesh, x, roll(w1), roll(w2), roll(w1_scale),
                               roll(w2_scale), tw32, ti)
    control = _relative_l2(out, rotated)
    print(f"{tag} relative_l2={error:.5f} worst_token={worst:.5f} "
          f"rotated_control={control:.5f} "
          f"control_ratio={control / max(error, 1e-12):.1f}x")
    assert control > ROTATED_CONTROL_FACTOR * error, (
        f"a reference over rotated experts sits {control:.4f} away against "
        f"the real reference's {error:.4f}; the band does not discriminate")
    assert error < batch_bound, (
        f"relative L2 {error:.4f} past the {batch_bound} band")
    assert worst < token_bound, (
        f"worst token's relative error {worst:.4f} past the {token_bound} "
        f"per-token band; the batch norm was {error:.4f}")


def _served_layer(e=E_LOCAL):
    """A stub layer in the layout job j-3c919167 logged, for the adapter.

    Not a shape this file derived: `_process_fp8_moe_weights` logged
    ``w13=[16, 6144, 4096], w2=[16, 2048, 6144], w13_scale=[16, 1, 1, 4096],
    w2_scale=[16, 1, 1, 6144], requant_block_size=per-channel`` on
    GLM-5.2-FP8 at DP16/EP16, and these are those tensors.
    """
    import torch

    class _Layer:
        pass

    def _p(*shape, dtype=torch.float32):
        return torch.nn.Parameter(torch.empty(shape, dtype=dtype),
                                  requires_grad=False)

    layer = _Layer()
    layer.w13_weight = _p(e, HIDDEN, 2 * INTER, dtype=torch.float8_e4m3fn)
    layer.w2_weight = _p(e, INTER, HIDDEN, dtype=torch.float8_e4m3fn)
    layer.w13_weight_scale_inv = _p(e, 1, 1, 2 * INTER)
    layer.w2_weight_scale_inv = _p(e, 1, 1, HIDDEN)
    return layer


def test_the_expert_axis_name_is_the_one_the_kernel_resolves():
    """`get_moe_expert_axis` searches a fixed tuple, so the mesh cannot be
    named freely -- `build_ep_rs_mesh`'s choice has to be in that tuple."""
    from vllm_torchtpu.kernels.fused_moe_rs.fused_moe_rs import (
        get_moe_expert_axis)
    mesh = _mesh()
    assert get_moe_expert_axis(mesh) == RS_EXPERT_AXIS_NAME
    assert mesh.shape[RS_DATA_AXIS_NAME] == 1
    assert mesh.shape[RS_EXPERT_AXIS_NAME] == EP


def test_bind_weights_passes_the_loaders_layout_through_unchanged():
    """`bind_weights_for_rs` must alias, not convert.

    The layer is built in the post-requantization layout job j-3c919167
    logged -- see `_served_layer` -- at the served expert count, so the shapes
    asserted here are literally the logged ones. Two claims:

    - the contract check accepts it, which is the claim the e2e run falsified
      for the previous version of this adapter;
    - all four RS attributes share storage with the parameters. Aliasing is
      the assertion rather than equal contents, because a copy of a GLM-5.2
      layer's experts is 45.3 GB a rank and would be invisible to any
      value-based check.
    """
    from vllm_torchtpu.layers.adapter.fused_moe_rs import (
        RS_W1_ATTR, RS_W1_SCALE_ATTR, RS_W2_ATTR, RS_W2_SCALE_ATTR,
        _weights_unsupported_reason, bind_weights_for_rs)

    layer = _served_layer()
    assert tuple(layer.w13_weight.shape) == (E_LOCAL, 6144, 4096)
    assert tuple(layer.w2_weight.shape) == (E_LOCAL, 2048, 6144)
    assert tuple(layer.w13_weight_scale_inv.shape) == (E_LOCAL, 1, 1, 4096)
    assert tuple(layer.w2_weight_scale_inv.shape) == (E_LOCAL, 1, 1, 6144)

    reason = _weights_unsupported_reason(layer)
    assert reason is None, (
        f"the contract check refuses the layout the loader actually produces: "
        f"{reason}")

    before = {
        name: (tuple(getattr(layer, name).shape), getattr(layer, name).dtype,
               getattr(layer, name).data.data_ptr())
        for name in ("w13_weight", "w2_weight", "w13_weight_scale_inv",
                     "w2_weight_scale_inv")
    }
    bind_weights_for_rs(layer)
    after = {
        name: (tuple(getattr(layer, name).shape), getattr(layer, name).dtype,
               getattr(layer, name).data.data_ptr())
        for name in before
    }
    assert after == before, (
        "bind_weights_for_rs changed the parameters it was supposed to leave "
        f"alone: {before} -> {after}")

    for attr, param in ((RS_W1_ATTR, "w13_weight"), (RS_W2_ATTR, "w2_weight"),
                        (RS_W1_SCALE_ATTR, "w13_weight_scale_inv"),
                        (RS_W2_SCALE_ATTR, "w2_weight_scale_inv")):
        bound = getattr(layer, attr)
        want = getattr(layer, param)
        assert bound.data_ptr() == want.data.data_ptr(), (
            f"{attr} is a second buffer, not {param}; on GLM-5.2 that is a "
            "duplicate of the layer's experts")
        assert tuple(bound.shape) == tuple(want.shape)
        assert bound.dtype == want.dtype


def test_the_contract_check_refuses_a_scale_in_n_blocks():
    """The red negative control for `_weights_unsupported_reason`.

    It is now the only thing between a wrong layout and a Pallas error on the
    first batch, so it has to refuse a scale that a rank-only check would wave
    through: four dimensions, unit axis 2, and a last axis that is N in
    **blocks** rather than N in elements. That shape is what a plain transpose
    of vLLM's on-disk scale produces, which is exactly the mistake the
    previous adapter made.

    The positive control is `test_bind_weights_...` above, which asserts the
    same function returns None on the loader's real output -- without it, a
    checker that refused everything would pass this test.
    """
    import torch

    from vllm_torchtpu.layers.adapter.fused_moe_rs import (
        _weights_unsupported_reason)

    layer = _served_layer()
    layer.w13_weight_scale_inv = torch.nn.Parameter(torch.empty(
        E_LOCAL, 1, 1, (2 * INTER) // BLOCK),
                                                    requires_grad=False)
    reason = _weights_unsupported_reason(layer)
    print(f"RS_MOE_REFUSAL n_blocks -> {reason!r}")
    assert reason is not None, (
        "a scale whose last axis is N in blocks was accepted; every scale "
        "would land on the wrong columns and the kernel would not complain")
    assert "N elements" in reason, (
        f"the refusal does not say why: {reason!r}")

    # The other half of the same mistake: three dimensions, which is what the
    # Pallas ref rejects with `indices must not be longer than shape`.
    layer = _served_layer()
    layer.w13_weight_scale_inv = torch.nn.Parameter(torch.empty(
        E_LOCAL, 1, 2 * INTER),
                                                    requires_grad=False)
    reason_3d = _weights_unsupported_reason(layer)
    print(f"RS_MOE_REFUSAL 3d -> {reason_3d!r}")
    assert reason_3d is not None and "K_blocks" in reason_3d


@pytest.mark.parametrize("scheme", ["bf16"] + SCHEMES)
def test_small_shape_tracks_a_dense_reference(scheme):
    """The kernel end to end at a small but structurally complete shape.

    Three arms, because a band on its own would not be evidence. The ``bf16``
    arm runs the same kernel with nothing quantized, so its band is the
    arithmetic's; the two fp8 arms add the scales and the kernel's online
    activation quantization, and their band is the transport's. An order of
    magnitude between them is what says the wider one is the wire rather than
    a mistake. ``per_channel`` is the scheme production runs.
    """
    mesh = _mesh()
    hidden, inter, tokens = SMALL["hidden"], SMALL["inter"], SMALL["tokens"]
    w1, w2, w1_scale, w2_scale = _make_weights(mesh,
                                               0,
                                               hidden=hidden,
                                               inter=inter,
                                               scheme=scheme)
    if scheme != "bf16":
        k_blocks = 1 if scheme == "per_channel" else hidden // BLOCK
        assert w1_scale.shape == (E_TOTAL, k_blocks, 1, 2 * inter)
    x, tw, ti, tw32 = _make_inputs(mesh, 1, tokens=tokens, hidden=hidden)

    out = jax.block_until_ready(
        _run(mesh, x, w1, w2, w1_scale, w2_scale, tw, ti))
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    quantized = scheme != "bf16"
    _check_against_reference(
        mesh, f"RS_MOE_SMALL[{scheme}]", x, w1, w2, w1_scale, w2_scale, tw32,
        ti, out, FP8_RELATIVE_BOUND if quantized else BF16_RELATIVE_BOUND,
        FP8_TOKEN_BOUND if quantized else BF16_TOKEN_BOUND)


def test_the_kernel_accepts_what_the_adapter_hands_it():
    """The whole chain: the loader's layout -> `bind_weights_for_rs` -> kernel.

    Every other arm hands the kernel tensors this file built in the kernel's
    own layout, which cannot catch a binding that is wrong on both sides of
    itself. Here the weights are per-channel fp8 in the layout j-3c919167
    logged, they go through the real `bind_weights_for_rs` as a torch layer,
    and what comes off the RS attributes is what the kernel and the reference
    both get.

    Per-channel is the arm that matters: `_recover_quant_block_size(size_k, 1)`
    short-circuits to `size_k`, so this is the one quant block over all of K
    that production takes, exercised rather than reasoned about.

    The round trip to torch and back is affordable only at the small shape:
    128 experts of [512, 512] is 34 MB where the served layer would be 3.2 GB.
    """
    import torch

    from vllm_torchtpu.layers.adapter.fused_moe_rs import (
        RS_W1_ATTR, RS_W1_SCALE_ATTR, RS_W2_ATTR, RS_W2_SCALE_ATTR,
        _weights_unsupported_reason, bind_weights_for_rs)

    def to_torch_fp8(a):
        """jax fp8 -> torch fp8, through the uint8 neither library refuses."""
        u8 = np.asarray(jax.device_get(lax.bitcast_convert_type(a,
                                                                jnp.uint8)))
        return torch.from_numpy(u8.copy()).view(torch.float8_e4m3fn)

    def to_jax_fp8(t):
        u8 = jnp.asarray(t.contiguous().view(torch.uint8).numpy())
        return lax.bitcast_convert_type(u8, FP8)

    mesh = _mesh()
    hidden, inter, tokens = SMALL["hidden"], SMALL["inter"], SMALL["tokens"]

    k1, k2 = jax.random.split(jax.random.key(3))
    q13, s13 = jax.vmap(lambda a: _quantize(a, "per_channel"))(
        jax.random.normal(k1, (E_TOTAL, hidden, 2 * inter), jnp.float32) / 10)
    q2, s2 = jax.vmap(lambda a: _quantize(a, "per_channel"))(
        jax.random.normal(k2, (E_TOTAL, inter, hidden), jnp.float32) / 10)
    s13, s2 = s13[:, :, None, :], s2[:, :, None, :]
    assert s13.shape == (E_TOTAL, 1, 1, 2 * inter)
    assert s2.shape == (E_TOTAL, 1, 1, hidden)

    class _Layer:
        pass

    def _np_param(a):
        return torch.nn.Parameter(torch.from_numpy(
            np.asarray(jax.device_get(a)).copy()),
                                  requires_grad=False)

    layer = _Layer()
    layer.w13_weight = torch.nn.Parameter(to_torch_fp8(q13),
                                          requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(to_torch_fp8(q2), requires_grad=False)
    layer.w13_weight_scale_inv = _np_param(s13)
    layer.w2_weight_scale_inv = _np_param(s2)

    reason = _weights_unsupported_reason(layer)
    assert reason is None, f"the contract check refuses this layout: {reason}"
    bind_weights_for_rs(layer)

    w1 = to_jax_fp8(getattr(layer, RS_W1_ATTR))
    w2 = to_jax_fp8(getattr(layer, RS_W2_ATTR))
    w1_scale = jnp.asarray(getattr(layer, RS_W1_SCALE_ATTR).numpy())
    w2_scale = jnp.asarray(getattr(layer, RS_W2_SCALE_ATTR).numpy())
    print(f"RS_MOE_ADAPTER w1={w1.shape} w1_scale={w1_scale.shape} "
          f"w2={w2.shape} w2_scale={w2_scale.shape}")

    x, tw, ti, tw32 = _make_inputs(mesh, 1, tokens=tokens, hidden=hidden)
    out = jax.block_until_ready(
        _run(mesh, x, w1, w2, w1_scale, w2_scale, tw, ti))
    _check_against_reference(mesh, "RS_MOE_ADAPTER", x, w1, w2, w1_scale,
                             w2_scale, tw32, ti, out, FP8_RELATIVE_BOUND,
                             FP8_TOKEN_BOUND)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("tokens", [HALF_TOKENS, SERVED_TOKENS])
def test_the_served_shape_compiles_and_runs(tokens, scheme):
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

    Parametrized over the scale scheme because the scheme changes the tiling,
    not just the values: `_select_fused_rs_block_sizes` takes
    `quant_block_size` and the scale VMEM buffers are sized
    ``ceil(tile_k / quant_block_size)`` deep. **per_channel is production**;
    an earlier version of this file ran block128 alone.
    """
    mesh = _mesh()
    size_m = tokens * TOPK
    w1, w2, w1_scale, w2_scale = _make_cheap_weights(mesh,
                                                     hidden=HIDDEN,
                                                     inter=INTER,
                                                     scheme=scheme)

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
    print(f"RS_MOE_SERVED scheme={scheme} tokens={tokens} size_m={size_m} "
          f"ep={EP} e_total={E_TOTAL} e_local={E_LOCAL} hidden={HIDDEN} "
          f"inter={INTER} topk={TOPK} w1_scale={w1_scale.shape}")
    out = jax.block_until_ready(
        _run(mesh, x, w1, w2, w1_scale, w2_scale, tw, ti))
    assert out.shape == (tokens, HIDDEN)
    assert out.dtype == jnp.bfloat16
    finite = bool(np.isfinite(np.asarray(out, np.float32)).all())
    print(f"RS_MOE_SERVED scheme={scheme} tokens={tokens} ran; "
          f"all_finite={finite}")
