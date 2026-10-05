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
"""Grouped expert FFN in one kernel: gate/up, activation and down per gm tile.

The arithmetic of every tile is gmm_v2's (`tile_matmul`, `apply_act_fn`,
`store_tile_output`) under the tiling gmm_v2 picks for each leg on its own, so
the output equals `gmm_v2(gmm_v2(x, w1, fuse_act), w2)`. What changes is the
schedule: one pipeline streams both legs' weights for an expert, the activated
intermediate stays in VMEM, and the group metadata is filled once.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import (
    FusedWeightsRef,
    GmmConfigs,
    MetadataRef,
    WeightsRef,
    _get_lhs_sublane_size,
    align_to,
    apply_act_fn,
    calculate_tiling,
    fill_metadata,
    generate_block_specs,
    get_cost_estimate,
    get_metadata,
    get_packing_factor,
    make_gmm_configs,
    store_tile_output,
    tile_matmul,
)

# gmm_v2.generate_block_specs streams each rhs with this many buffers.
_RHS_BUFFERS = 3


def _default_vmem_limit() -> int:
    return int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)


def _ffn_vmem_estimate(up: GmmConfigs, down: GmmConfigs) -> int:
    """VMEM the fused pipeline holds: both legs' rhs buffers plus one tile's values."""

    def nbytes(dtype) -> float:
        return jax.dtypes.itemsize_bits(dtype) / 8

    def rhs_bytes(c: GmmConfigs, parts: int) -> float:
        weight = (
            c.tiles.tile_k
            * c.tiles.tile_n
            * nbytes(c.rhs_cfgs.quant_dtype or c.rhs_cfgs.dtype)
        )
        scale = (
            c.num_quant_blocks_per_tile_k * c.tiles.tile_n * 4
            if c.rhs_cfgs.has_scale
            else 0
        )
        bias = c.tiles.tile_n * 4 if c.rhs_cfgs.has_bias else 0
        return parts * (_RHS_BUFFERS * weight + 2 * (scale + bias))

    def lhs_values(c: GmmConfigs, tm: int) -> float:
        quant = c.lhs_cfgs.quant_dtype
        return tm * c.tiles.tile_k * (nbytes(quant) if quant is not None else 0)

    tm = up.tiles.tile_m
    hidden = up.out_size_n
    total = rhs_bytes(up, 2) + rhs_bytes(down, 1)
    total += 2 * tm * up.tiles.tile_k * nbytes(up.lhs_cfgs.dtype)  # lhs blocks
    total += 2 * tm * down.tiles.tile_n * nbytes(down.out_dtype)  # out blocks
    total += lhs_values(up, tm) + tm * 2 * hidden * nbytes(up.acc_dtype)
    total += tm * hidden * nbytes(up.out_dtype) + lhs_values(down, tm)
    total += tm * down.tiles.tile_n * nbytes(down.acc_dtype)
    return int(total)


def make_ffn_configs(
    lhs: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    *,
    fuse_act: str,
    maybe_quantize_lhs: bool,
    rhs_quant_dtype: jnp.dtype | None,
    out_dtype: jnp.dtype | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[GmmConfigs, GmmConfigs] | None:
    """Per-leg gmm_v2 configs for the fused kernel, or None if it cannot run them.

    The fused kernel runs one gm tile through both legs per grid step, so it
    needs w2 to consume exactly the activated width, both legs to tile m
    identically and to cover k and n in one tile, and both legs' rhs pipelines
    to fit in VMEM together.
    """
    if vmem_limit_bytes is None:
        vmem_limit_bytes = _default_vmem_limit()
    size_m = align_to(lhs.shape[0], _get_lhs_sublane_size(lhs.dtype, lhs.shape[0]))
    lhs = jax.ShapeDtypeStruct((size_m, lhs.shape[1]), lhs.dtype)
    offset = jax.ShapeDtypeStruct((1,), jnp.int32)
    common = dict(
        vmem_limit_bytes=vmem_limit_bytes,
        acc_dtype=None,
        maybe_quantize_lhs=maybe_quantize_lhs,
        zero_initialize=False,
        rhs_quant_dtype=rhs_quant_dtype,
        tile_info=calculate_tiling,
    )
    up = make_gmm_configs(
        lhs,
        w1,
        w1_scale,
        w1_bias,
        group_sizes,
        offset,
        out_dtype=None,
        fuse_act=fuse_act,
        **common,
    )
    w2_size_k = w2.shape[1] * get_packing_factor(w2.dtype, rhs_quant_dtype)
    if w2_size_k != up.out_size_n:
        return None
    hidden = jax.ShapeDtypeStruct((size_m, up.out_size_n), up.out_dtype)
    down = make_gmm_configs(
        hidden,
        w2,
        w2_scale,
        w2_bias,
        group_sizes,
        offset,
        out_dtype=out_dtype or lhs.dtype,
        fuse_act=None,
        **common,
    )

    def single_tile(c: GmmConfigs) -> bool:
        return c.tiles.tile_k >= c.dims.size_k and c.tiles.tile_n >= c.out_size_n

    same_m_tiling = (up.tiles.tile_m, up.tiles.bucket_base) == (
        down.tiles.tile_m,
        down.tiles.bucket_base,
    ) and up.dims.size_lhs_sublane == down.dims.size_lhs_sublane
    if not (same_m_tiling and single_tile(up) and single_tile(down)):
        return None
    if _ffn_vmem_estimate(up, down) > vmem_limit_bytes:
        return None
    return up, down


def ffn_inner_kernel(
    # In
    tiled_lhs_ref: jax.Array,  # [tile_m // size_lhs_sublane, size_lhs_sublane, k]
    w13_ref: FusedWeightsRef,  # gate, up: [k, i]
    w2_ref: WeightsRef,  # [i, n]
    # Out
    tiled_out_ref: jax.Array,  # [tile_m // size_lhs_sublane, size_lhs_sublane, n]
    # Scratch
    partial_out_ref: jax.Array,  # [size_lhs_sublane, n]
    metadata_ref: MetadataRef,
    *,
    up: GmmConfigs,
    down: GmmConfigs,
):
    """One gm tile through gate/up, activation and down; see gmm_v2.inner_kernel."""
    gm_id = pl.program_id(1)
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = m_start - m_start % up.dims.size_lhs_sublane
    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset

    def run_tile(bucket_m: int):
        tiled_lhs = tiled_lhs_ref.reshape(-1, up.tiles.tile_k)[:bucket_m]
        acc = tile_matmul(
            tiled_lhs, w13_ref, cfgs=up, bucket_m=bucket_m, is_last_k_step=True
        )
        if up.rhs_cfgs.has_bias:
            acc += w13_ref.get_bias().astype(acc.dtype)
        hidden = apply_act_fn(acc, up.fuse_act).astype(up.out_dtype)

        acc = tile_matmul(
            hidden, w2_ref, cfgs=down, bucket_m=bucket_m, is_last_k_step=True
        )
        if down.rhs_cfgs.has_bias:
            acc += w2_ref.get_bias().astype(acc.dtype)
        store_tile_output(
            acc,
            tiled_out_ref,
            partial_out_ref,
            gm_id=gm_id,
            m_start_local=m_start_local,
            m_end_local=m_end_local,
            cfgs=down,
        )

    branches = []
    for bucket_idx in range(up.tiles.tile_m // up.tiles.bucket_base):
        bucket_m = up.tiles.bucket_base * (bucket_idx + 1)
        branches.append(
            jax.named_scope(f"bm{bucket_m}")(functools.partial(run_tile, bucket_m))
        )
    jax.lax.switch(m_end_local // up.tiles.bucket_base, branches)


def _unpack(rhs_ref: WeightsRef, cfgs: GmmConfigs) -> WeightsRef:
    if not cfgs.rhs_cfgs.should_unpack:
        return rhs_ref
    return dataclasses.replace(
        rhs_ref, weight=rhs_ref.weight.bitcast(cfgs.rhs_cfgs.quant_dtype)
    )


def ffn_kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    # In
    lhs_ref: jax.Array,  # [size_m, k]
    w13_ref: WeightsRef,  # [size_group, k, 2 * i]
    w2_ref: WeightsRef,  # [size_group, i, n]
    # Out
    out_ref: jax.Array,  # [size_m, n]
    # Scratch
    partial_out_ref: jax.Array,  # [size_lhs_sublane, n]
    metadata_ref: MetadataRef,
    *,
    up: GmmConfigs,
    down: GmmConfigs,
):
    """Fills the gm tile metadata once and pipelines both legs over it."""
    w13_ref = _unpack(w13_ref, up)
    w2_ref = _unpack(w2_ref, down)

    num_gm = fill_metadata(lhs_group_sizes_ref, group_offset_ref, metadata_ref, cfgs=up)

    (lhs_spec, w13_spec), _ = generate_block_specs(metadata_ref, up)
    (_, w2_spec), out_spec = generate_block_specs(metadata_ref, down)
    w13_up_ref = jax.tree.map(lambda x: x.at[..., up.out_size_n :], w13_ref)
    w13_ref = FusedWeightsRef(gate=w13_ref, up=w13_up_ref)
    w13_spec = FusedWeightsRef(gate=w13_spec, up=w13_spec)

    # The (n, gm, k) grid shape is the one gmm_v2's index maps are written for.
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(ffn_inner_kernel, up=up, down=down),
        grid=(1, num_gm, 1),
        in_specs=(lhs_spec, w13_spec, w2_spec),
        out_specs=out_spec,
    )
    sublane = up.dims.size_lhs_sublane
    lhs_in = lhs_ref.reshape(-1, sublane, lhs_ref.shape[-1])
    out_in = out_ref.reshape(-1, sublane, out_ref.shape[-1])
    pipeline_fn(
        lhs_in, w13_ref, w2_ref, out_in, scratches=[partial_out_ref, metadata_ref]
    )


def _hbm_f32(x: jax.Array | None) -> tuple[jax.Array | None, pl.BlockSpec | None]:
    if x is None:
        return None, None
    return x.astype(jnp.float32), pl.BlockSpec(memory_space=pltpu.HBM)


@jax.jit(
    static_argnames=[
        "fuse_act",
        "maybe_quantize_lhs",
        "rhs_quant_dtype",
        "preferred_element_type",
        "vmem_limit_bytes",
    ]
)
def gmm_v2_ffn(
    lhs: jax.Array,  # [size_m, k]
    w1: jax.Array,  # [size_group, k, 2 * i]: gate then up along the last dim
    w2: jax.Array,  # [size_group, i, n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    w1_scale: jax.Array | None = None,  # [size_group, num_blocks, 1, 2 * i]
    w1_bias: jax.Array | None = None,  # [size_group, 1, 2 * i]
    w2_scale: jax.Array | None = None,  # [size_group, num_blocks, 1, n]
    w2_bias: jax.Array | None = None,  # [size_group, 1, n]
    group_offset: jax.Array | None = None,  # int32[1]
    *,
    fuse_act: str,
    maybe_quantize_lhs: bool = True,
    rhs_quant_dtype: jnp.dtype | None = None,
    preferred_element_type: jnp.dtype | None = None,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
    """`gmm_v2(act(gmm_v2(lhs, w1)), w2)` as one kernel; rows past the groups are unset.

    Only for shapes `make_ffn_configs` accepts.
    """
    if vmem_limit_bytes is None:
        vmem_limit_bytes = _default_vmem_limit()
    cfgs = make_ffn_configs(
        lhs,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        group_sizes,
        fuse_act=fuse_act,
        maybe_quantize_lhs=maybe_quantize_lhs,
        rhs_quant_dtype=rhs_quant_dtype,
        out_dtype=preferred_element_type,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    if cfgs is None:
        raise ValueError(
            f"gmm_v2_ffn cannot run {lhs.shape=} {w1.shape=} {w2.shape=}; "
            "check make_ffn_configs before calling it."
        )
    up, down = cfgs

    size_m = lhs.shape[0]
    padded_size_m = up.dims.size_m
    if padded_size_m != size_m:
        lhs = jnp.pad(lhs, ((0, padded_size_m - size_m), (0, 0)))
    if group_offset is None:
        group_offset = jnp.zeros((1,), jnp.int32)

    w1_scale, w1_scale_spec = _hbm_f32(w1_scale)
    w1_bias, w1_bias_spec = _hbm_f32(w1_bias)
    w2_scale, w2_scale_spec = _hbm_f32(w2_scale)
    w2_bias, w2_bias_spec = _hbm_f32(w2_bias)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    max_num_gm = up.dims.size_group + pl.cdiv(padded_size_m, up.tiles.tile_m) - 1
    scratch_shapes = [
        pltpu.VMEM((down.dims.size_lhs_sublane, down.tiles.tile_n), down.out_dtype),
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
        ),
    ]
    num_lanes = pltpu.get_tpu_info().num_lanes
    out_shape = jax.ShapeDtypeStruct(
        (padded_size_m, align_to(down.out_size_n, num_lanes)), down.out_dtype
    )
    up_cost, down_cost = get_cost_estimate(up), get_cost_estimate(down)
    name = (
        f"gmm_v2_ffn-g_{up.dims.size_group}-m_{padded_size_m}-k_{up.dims.size_k}"
        f"-i_{up.out_size_n}-n_{down.out_size_n}-act_{fuse_act}"
        f"-tm_{up.tiles.tile_m}"
    )
    metadata = {f"up.{k}": v for k, v in get_metadata(up).items()}
    metadata |= {f"down.{k}": v for k, v in get_metadata(down).items()}

    return pl.pallas_call(
        functools.partial(ffn_kernel_main, up=up, down=down),
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                hbm,
                WeightsRef(weight=hbm, scale=w1_scale_spec, bias=w1_bias_spec),
                WeightsRef(weight=hbm, scale=w2_scale_spec, bias=w2_bias_spec),
            ],
            out_specs=hbm,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=name,
        cost_estimate=pl.CostEstimate(
            flops=up_cost.flops + down_cost.flops,
            # The intermediate never reaches HBM: the up leg's output and the
            # down leg's input drop out of the byte count.
            bytes_accessed=up_cost.bytes_accessed
            + down_cost.bytes_accessed
            - 2 * padded_size_m * up.out_size_n * jnp.dtype(up.out_dtype).itemsize,
            transcendentals=0,
        ),
        metadata=metadata,
    )(
        group_sizes,
        group_offset,
        lhs,
        WeightsRef(weight=w1, scale=w1_scale, bias=w1_bias),
        WeightsRef(weight=w2, scale=w2_scale, bias=w2_bias),
    )[:size_m, : down.out_size_n]
