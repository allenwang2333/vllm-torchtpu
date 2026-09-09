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
"""Fused permute + grouped matmul.

Reads the LHS per-row from an un-permuted pool inside the kernel, so the MoE
dispatch permute never materialises. Equal to
`gmm_v2(lhs[gather_indices], ...)` over the valid prefix, bit for bit up to the
sign of zero -- a revisited gm tile accumulates onto its own output, which turns
-0.0 into +0.0.

Design, relative to a contiguous `gmm_v2`:

  * The gather tax is per-gm-tile, not per launch, so `tile_m` is sized from
    the live rows per group rather than to fill VMEM -- see
    `config.gather_tiling`.
  * The index table is a kernel-invariant SMEM scalar-prefetch operand, so
    `tile_m` carries no window-copy alignment constraint.
  * A bf16 pool is read lane-blocked, as `bf16[size_src, size_k // 128, 128]`,
    which puts the (2, 1) sub-word packing on the block axis instead of on
    rows, so a row is whole 32-bit words and a plain per-row DMA is legal.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox import gmm_v2
from vllm_torchtpu.kernels.megablox.fused_permute_gmm import (config, kernel,
                                                              memory_ref)

# gmm_v2's default share of VMEM handed to the tile fit.
VMEM_LIMIT_FRACTION = 0.9
# Headroom the compiler gets over this call's estimated footprint. Above the
# floor the limit is a codegen knob, not a budget.
VMEM_SLACK_BYTES = 12 << 20


def _blocked_pool(lhs: jax.Array, lanes: int) -> jax.Array:
    """The pool as the per-row DMA reads it: [size_src, size_k // lanes, lanes]."""
    if lhs.dtype.itemsize != 2:
        raise NotImplementedError(
            f"pool dtype {lhs.dtype} unsupported: the LHS build pairs two "
            "elements per 32-bit word, so only 2-byte pools are implemented")
    if lhs.ndim == 3:
        if lhs.shape[-1] != lanes:
            raise ValueError(
                f"a blocked pool must be [size_src, size_k // {lanes}, "
                f"{lanes}], got {lhs.shape}")
        return lhs
    if lhs.ndim == 2:
        if lhs.shape[1] % lanes:
            raise NotImplementedError(
                f"size_k {lhs.shape[1]} must be a multiple of {lanes}")
        return lhs.reshape(lhs.shape[0], lhs.shape[1] // lanes, lanes)
    raise ValueError(f"pool must be 2-D or 3-D, got {lhs.shape}")


@jax.jit(static_argnames=[
    "vmem_limit_bytes",
    "preferred_element_type",
    "acc_dtype",
    "maybe_quantize_lhs",
    "zero_initialize",
    "fuse_act",
    "ep_size",
    "tuning",
])
def fused_permute_gmm(
    lhs: jax.Array,  # [size_src, size_k] un-permuted pool
    rhs: jax.Array,  # [size_group, size_k, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    gather_indices: jax.Array,  # int32[size_m]
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    *,
    vmem_limit_bytes: int | None = None,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    zero_initialize: bool = True,
    fuse_act: str | None = None,
    ep_size: int = 1,
    tuning: config.Tuning | None = None,
) -> jax.Array:
    """Grouped matmul over `lhs[gather_indices]` without materialising it.

    Row `i` of the effective LHS is `lhs[gather_indices[i]]`; the output is
    [size_m, size_n] with size_m = gather_indices.shape[0]. `ep_size` is the
    shard count the indices span; it feeds `config.gather_tiling`. `tuning`
    defaults to `Tuning()`, whose SMEM capacity is resolved from the device.

    `lhs` is the un-permuted pool, either `[size_src, size_k]` or the blocked
    `[size_src, size_k // 128, 128]` the rows are actually read in. The LHS
    build pairs lane-blocks and packs two elements per word, so size_k must be
    a multiple of 256 and the pool 2-byte; a caller that cannot meet that
    should materialise the permute and use `gmm_v2`.
    """
    if gather_indices.ndim != 1:
        raise ValueError("gather_indices must be 1-D")
    if tuning is None:
        tuning = config.Tuning()

    info = pltpu.get_tpu_info()
    lanes = info.num_lanes
    pool = _blocked_pool(lhs, lanes)
    size_k = pool.shape[1] * lanes

    size_m = gather_indices.shape[0]
    # The kernel reshapes the output by its sublane block, so pad the rows up to
    # it, as gmm_v2 pads its lhs. The extra rows sit past every group, so they
    # gather row 0 and add no work, and they are sliced back off below.
    sublane = gmm_v2._get_lhs_sublane_size(lhs.dtype, size_m)
    padded_size_m = gmm_v2.align_to(size_m, sublane)
    if padded_size_m != size_m:
        gather_indices = jnp.pad(gather_indices, (0, padded_size_m - size_m))
    if padded_size_m > tuning.max_index_rows:
        raise NotImplementedError(
            f"{size_m} indices exceed the SMEM scalar-operand budget "
            f"({tuning.smem_index_limit_bytes} B); materialise the permute "
            "and use gmm_v2")

    if group_offset is None:
        group_offset = jnp.zeros((1, ), jnp.int32)
    elif group_offset.ndim == 0:
        group_offset = group_offset[None]
    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(info.vmem_capacity_bytes * VMEM_LIMIT_FRACTION)
    gmm_cfgs = gmm_v2.make_gmm_configs(jax.ShapeDtypeStruct(
        (padded_size_m, size_k), lhs.dtype),
                                       rhs,
                                       rhs_scale,
                                       rhs_bias,
                                       group_sizes,
                                       group_offset,
                                       tile_info=config.gather_tile_fn(
                                           ep_size, tuning),
                                       vmem_limit_bytes=vmem_limit_bytes,
                                       out_dtype=preferred_element_type,
                                       acc_dtype=acc_dtype,
                                       maybe_quantize_lhs=maybe_quantize_lhs,
                                       zero_initialize=zero_initialize,
                                       fuse_act=fuse_act)
    cfg = config.Config(gmm=gmm_cfgs,
                        tuning=tuning,
                        num_lanes=lanes,
                        num_sublanes=info.num_sublanes,
                        mxu_columns=info.mxu_column_size)

    if cfg.tiles.tile_k != cfg.dims.size_k:
        raise NotImplementedError(
            "fused_permute_gmm gathers whole rows, so tile_k must be size_k "
            f"({cfg.tiles.tile_k} != {cfg.dims.size_k})")
    # Pad the index table so a tile that runs past the last live row still
    # names a valid pool row (bounds checks are off).
    idx = jnp.pad(gather_indices.astype(config.INDEX_DTYPE),
                  (0, cfg.index_rows - padded_size_m),
                  mode="edge")

    rhs_scale_spec = rhs_bias_spec = None
    if rhs_scale is not None:
        rhs_scale = rhs_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if rhs_bias is not None:
        rhs_bias = rhs_bias.astype(jnp.float32)
        rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    rhs_weights = gmm_v2.WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)

    # The fit already chose the tile against `vmem_limit_bytes`, so the
    # compiler only needs to cover what that tile costs.
    compiler_limit = min(cfg.estimated_vmem_bytes + VMEM_SLACK_BYTES,
                         vmem_limit_bytes)
    out = pl.pallas_call(
        functools.partial(kernel.kernel_main, cfg=cfg),
        out_shape=cfg.out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                gmm_v2.WeightsRef(weight=pl.BlockSpec(memory_space=pltpu.HBM),
                                  scale=rhs_scale_spec,
                                  bias=rhs_bias_spec),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[memory_ref.ScratchRef.create_scratch_types(cfg)],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=compiler_limit,
            disable_bounds_checks=True,
        ),
        name=cfg.kernel_name,
        cost_estimate=cfg.cost_estimate,
        metadata=cfg.kernel_metadata,
    )(group_sizes, group_offset, idx, pool, rhs_weights)
    return out[:size_m, :cfg.gmm.out_size_n]
