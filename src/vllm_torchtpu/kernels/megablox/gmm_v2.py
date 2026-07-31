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

import dataclasses
import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Util.


def swigluoai(gate: jax.Array,
              up: jax.Array,
              *,
              alpha: float = 1.702,
              limit: float = 7.0) -> jax.Array:
    """Activation used in some models such as GPT-OSS."""

    gate = jnp.clip(gate, max=limit)
    up = jnp.clip(up, min=-limit, max=limit)
    glu = gate * jax.nn.sigmoid(alpha * gate)
    return (up + 1.0) * glu


def silu_and_mul_with_clamp(gate: jax.Array,
                            up: jax.Array,
                            limit: float = 10.0) -> jax.Array:
    """Activation used in some models DeepSeek V4."""
    # The limit value is from DSV4's config.
    # TODO: pass limit from model config, instead of hardcoding here.
    gate = jnp.clip(gate, max=limit)
    up = jnp.clip(up, min=-limit, max=limit)
    return jax.nn.silu(gate) * up


def interleave_lane(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """Interleaves two arrays along lane dim at zero-cost."""
    assert lhs.shape == rhs.shape
    chunk_size = pltpu.get_tpu_info().num_lanes
    num_chunks = lhs.shape[-1] // chunk_size
    lhs_chunks = jnp.split(lhs, num_chunks, axis=-1)
    rhs_chunks = jnp.split(rhs, num_chunks, axis=-1)
    interleaved = []
    for i in range(num_chunks):
        interleaved += [lhs_chunks[i], rhs_chunks[i]]
    return jnp.concat(interleaved, axis=-1)


def deinterleave_lane(val: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Deinterleaves an array along lane dim at zero-cost."""
    chunk_size = pltpu.get_tpu_info().num_lanes
    num_chunks = val.shape[-1] // chunk_size
    chunks = jnp.split(val, num_chunks, axis=-1)
    lhs = jnp.concat(chunks[0::2], axis=-1)
    rhs = jnp.concat(chunks[1::2], axis=-1)
    return lhs, rhs


def apply_act_fn(acc: jax.Array, fuse_act: str | None):
    """Applies a fused activation function to the accumulator.

    This function is used when an activation function is fused with the matrix
    multiplication. The input accumulator `acc` is expected to contain
    concatenated results for both the 'gate' and 'up' projections.

    Args:
        acc: The accumulator array, with the last dimension being 2 * tile_n.
        fuse_act: The name of the activation function to apply.

    Returns:
        The result of applying the activation function.

    Raises:
        NotImplementedError: If an unsupported `fuse_act` is provided.
    """

    if fuse_act is None:
        return acc

    acc_gate, acc_up = deinterleave_lane(acc)
    match fuse_act:
        case "silu":
            return jax.nn.silu(acc_gate) * acc_up
        case "gelu":
            return jax.nn.gelu(acc_gate) * acc_up
        case "gelu_tanh":
            return jax.nn.gelu(acc_gate, approximate=True) * acc_up
        case "swigluoai":
            return swigluoai(acc_gate, acc_up)
        case "silu_and_mul_with_clamp":
            return silu_and_mul_with_clamp(acc_gate, acc_up)
        case _:
            raise NotImplementedError(
                f"Unsupported activation function: {fuse_act}")


def align_to(x, a):
    return pl.cdiv(x, a) * a


def get_packing_factor(storage_dtype: jnp.dtype,
                       quant_dtype: jnp.dtype | None) -> int:
    """Return the logical elements stored in one physical RHS element.

    The production path supported here is signed INT4 packed K-major in an
    INT32/UINT32 carrier. Native logical dtypes return one.
    """
    storage_dtype = jnp.dtype(storage_dtype)
    quant_dtype = None if quant_dtype is None else jnp.dtype(quant_dtype)

    if quant_dtype is None or quant_dtype == storage_dtype:
        return 1

    if (quant_dtype != jnp.dtype(jnp.int4)
            or storage_dtype not in (jnp.dtype(jnp.int32),
                                     jnp.dtype(jnp.uint32))):
        raise ValueError(
            "Packed GMM currently supports only signed INT4 packed in "
            "INT32/UINT32 carriers; received "
            f"storage_dtype={storage_dtype}, quant_dtype={quant_dtype}.")

    storage_bits = jax.dtypes.itemsize_bits(storage_dtype)
    quant_bits = jax.dtypes.itemsize_bits(quant_dtype)
    if storage_bits % quant_bits != 0:
        raise ValueError(
            f"{storage_bits}-bit storage cannot exactly contain "
            f"{quant_bits}-bit values.")

    packing = storage_bits // quant_bits
    if packing != 8:
        raise ValueError(
            f"Expected eight INT4 values per 32-bit carrier, got {packing}.")
    return packing


@functools.partial(
    jax.jit,
    static_argnames=("quant_dtype", "scale_dtype"),
)
def prequantize_lhs_with_a32(
    lhs: jax.Array,
    row_group_ids: jax.Array,
    lhs_scale_factor: jax.Array,
    *,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
    scale_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[jax.Array, jax.Array]:
    """Fuse normalized expert A32 into LHS and quantize once per row/K512.

    Args:
        lhs: BF16/FP32 routed rows [M, K].
        row_group_ids: Local expert id for every routed row [M].
        lhs_scale_factor: Padded normalized A32 [G, K/512, 128]. Only
            [..., :16] is used and must satisfy abs(A32) <= 1. The removed
            maximum must already be folded into B512.
        quant_dtype: FP8 compute dtype.
        scale_dtype: Storage dtype for the dynamic per-row K512 scales.

    Returns:
        lhs_q: [M, K] quantized activations.
        lhs_q_scale: [M, K/512] scales such that
            lhs * A32 ~= lhs_q * lhs_q_scale.
    """
    quant_dtype = jnp.dtype(quant_dtype)
    scale_dtype = jnp.dtype(scale_dtype)
    if lhs.ndim != 2:
        raise ValueError(f"lhs must be rank two, got {lhs.shape}.")
    size_m, size_k = lhs.shape
    if size_k % 512 != 0:
        raise ValueError(f"K={size_k} must be divisible by 512.")
    if row_group_ids.shape != (size_m,):
        raise ValueError(
            f"row_group_ids must have shape {(size_m,)}, got "
            f"{row_group_ids.shape}.")
    expected_tail = (size_k // 512, 128)
    if lhs_scale_factor.shape[1:] != expected_tail:
        raise ValueError(
            f"lhs_scale_factor tail must be {expected_tail}, got "
            f"{lhs_scale_factor.shape[1:]}.")

    dtype_max = float(jnp.finfo(quant_dtype).max)
    grouped = lhs.reshape(size_m, size_k // 512, 16, 32)
    a32 = lhs_scale_factor[row_group_ids, :, :16].astype(lhs.dtype)

    # Exact absmax(lhs*A32), without materializing a second BF16 [M,K] tensor.
    group_absmax = jnp.max(jnp.abs(grouped), axis=-1)
    block_absmax = jnp.max(group_absmax * jnp.abs(a32), axis=-1)
    block_scale = block_absmax / dtype_max
    block_scale_inv = jnp.where(block_scale == 0, 0, 1 / block_scale)

    effective_multiplier = a32[..., None] * block_scale_inv[..., None, None]
    lhs_q = (grouped * effective_multiplier).astype(quant_dtype).reshape(
        size_m, size_k)
    pad_w = 128 - block_scale.shape[-1]
    block_scale_pad = jnp.concatenate([
        block_scale,
        jnp.zeros((size_m, pad_w), dtype=block_scale.dtype)
    ], axis=-1)
    return lhs_q, block_scale_pad.astype(scale_dtype)


def group_sizes_to_row_ids(group_sizes: jax.Array,
                           size_m: int) -> jax.Array:
    """Expand contiguous routed group sizes into one local group id per row."""
    return jnp.repeat(
        jnp.arange(group_sizes.shape[0], dtype=jnp.int32),
        group_sizes,
        total_repeat_length=size_m,
    )


# Define data classes.


class RhsRef(ABC):
    """Abstract class that defines interfaces for RHS values."""

    @abstractmethod
    def get_weight(self) -> jax.Array:
        ...

    @abstractmethod
    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        ...

    @abstractmethod
    def get_scale_1d(self, b_id: int, start_n: int,
                     col_size: int) -> jax.Array:
        ...

    @abstractmethod
    def get_bias(self) -> jax.Array:
        ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LhsRef:
    """LHS value plus optional expert-specific normalized A32 factors."""

    value: Any
    # Physical layout: [G, K / 512, 128]. Only lanes [:16] are used.
    scale_factor: Any | None
    # Optional prequantized activation scale [M, K/512].
    quant_scale: Any | None

    def get_value(self) -> jax.Array:
        return self.value[...]

    def get_scale_factor(self) -> jax.Array:
        assert self.scale_factor is not None
        return self.scale_factor[...]

    def get_quant_scale(self) -> jax.Array:
        assert self.quant_scale is not None
        return self.quant_scale[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightsRef(RhsRef):
    """Dataclass for a single weight tensor."""

    weight: Any
    scale: Any | None
    bias: Any | None

    def get_weight(self) -> jax.Array:
        return self.weight[...]

    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        assert self.scale is not None
        del replicate_size
        return self.scale[...]

    def get_scale_1d(self, b_id: int, start_n: int,
                     col_size: int) -> jax.Array:
        assert self.scale is not None
        return self.scale[b_id, 0, start_n:start_n + col_size][...]

    def get_bias(self) -> jax.Array:
        assert self.bias is not None
        return self.bias[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FusedWeightsRef(RhsRef):
    """Dataclass for gate and up weights used in a fused activation."""

    gate: WeightsRef
    up: WeightsRef

    def get_weight(self) -> jax.Array:
        return interleave_lane(self.gate.get_weight(), self.up.get_weight())

    def get_scale(self, replicate_size: int | None = None) -> jax.Array:
        return interleave_lane(self.gate.get_scale(replicate_size),
                               self.up.get_scale(replicate_size))

    def get_scale_1d(self, b_id: int, start_n: int,
                     col_size: int) -> jax.Array:
        # Slice after lane interleaving. A single MXU-width microtile can be
        # entirely gate or entirely up; splitting start_n/col_size in half is
        # therefore incorrect.
        scale = self.get_scale()
        return scale[b_id, 0, start_n:start_n + col_size]

    def get_bias(self) -> jax.Array:
        return interleave_lane(self.gate.get_bias(), self.up.get_bias())

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
    gm_id_to_group_id: jax.Array
    gm_id_to_m_offset: jax.Array


@dataclasses.dataclass(frozen=True)
class TileSizes:
    tile_m: int
    tile_k: int
    tile_n: int
    bucket_base: int


@dataclasses.dataclass(frozen=True)
class Dimensions:
    size_m: int
    size_k: int
    size_n: int
    size_group: int
    size_lhs_group: int
    size_lhs_sublane: int


@dataclasses.dataclass(frozen=True)
class InputConfigs:
    # dtype is the physical storage dtype. quant_dtype is the logical compute
    # dtype and differs from dtype only for a packed RHS.
    quant_dtype: jnp.dtype | None
    quant_block_size: int | None
    dtype: jnp.dtype
    has_bias: bool = False
    has_scale: bool = False

    # LHS-only A32 metadata.
    scale_dtype: jnp.dtype | None = None
    scale_block_size: int | None = None
    scale_storage_width: int = 128
    is_prequantized: bool = False

    @property
    def should_unpack(self) -> bool:
        if self.quant_dtype is None:
            return False
        return (
            jnp.dtype(self.quant_dtype) == jnp.dtype(jnp.int4)
            and jnp.dtype(self.dtype)
            in (jnp.dtype(jnp.int32), jnp.dtype(jnp.uint32))
        )

    @property
    def should_dequantize_before_matmul(self) -> bool:
        if not self.has_scale:
            return False
        assert self.quant_block_size is not None
        return self.quant_block_size < pltpu.get_tpu_info().mxu_column_size

    @property
    def should_dequantize_after_matmul(self) -> bool:
        return self.has_scale and not self.should_dequantize_before_matmul

    @property
    def should_quantize(self) -> bool:
        if self.is_prequantized:
            return True
        if self.quant_dtype is None:
            return False
        # Packed RHS is already quantized and only needs a zero-copy bitcast.
        return (not self.should_unpack
                and jnp.dtype(self.quant_dtype) != jnp.dtype(self.dtype))

@dataclasses.dataclass(frozen=True)
class GmmConfigs:
    tiles: TileSizes
    dims: Dimensions
    lhs_cfgs: InputConfigs
    rhs_cfgs: InputConfigs
    out_dtype: jnp.dtype
    acc_dtype: jnp.dtype
    zero_init: bool
    fuse_act: str | None

    @property
    def num_quant_blocks_per_tile_k(self) -> int:
        return pl.cdiv(self.tiles.tile_k, self.rhs_cfgs.quant_block_size)

    @property
    def num_lhs_scale_blocks_per_tile_k(self) -> int:
        assert self.lhs_cfgs.scale_block_size is not None
        if self.tiles.tile_k % self.lhs_cfgs.scale_block_size != 0:
            raise ValueError(
                f"tile_k={self.tiles.tile_k} must be divisible by the LHS "
                f"factor block {self.lhs_cfgs.scale_block_size}.")
        return self.tiles.tile_k // self.lhs_cfgs.scale_block_size

    @property
    def out_size_n(self) -> int:
        if self.fuse_act is None:
            return self.dims.size_n
        else:
            return self.dims.size_n // 2


TileFn = Callable[[Dimensions, InputConfigs, InputConfigs, int, str | None],
                  TileSizes]


class IndexMaps:
    """Index maps for GMM kernel."""

    def __init__(self, metadata_ref: MetadataRef, cfgs: GmmConfigs):
        self.metadata_ref = metadata_ref
        self.cfgs = cfgs

    def lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_size = row_end - row_start

        return (pl.ds(row_start, row_size), 0, k_id)

    def lhs_scale_index_map(self, _: jax.Array, gm_id: jax.Array,
                            k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        # The physical factor layout is [G, K/512, 128]. BlockSpec indexing
        # advances by tile_k/512 along its second dimension.
        return (group_id, k_id, 0)

    def lhs_quant_scale_index_map(self, _: jax.Array, gm_id: jax.Array,
                                  k_id: jax.Array):
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]
        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_size = row_end - row_start
        return (pl.ds(row_start, row_size), 0, 0)

    def rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array,
                             k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, k_id, n_id)

    def rhs_bias_index_map(self, n_id: jax.Array, gm_id: jax.Array,
                           _: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, 0, n_id)

    def rhs_scale_index_map(self, n_id: jax.Array, gm_id: jax.Array,
                            k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        # Simply multiplying k_id by num_quant_blocks_per_tile_k will not work
        # since a single quant block could be shared along multiple k tile.
        k_row = k_id * self.cfgs.tiles.tile_k
        b_row = k_row // self.cfgs.rhs_cfgs.quant_block_size
        b_tile_id = b_row // self.cfgs.num_quant_blocks_per_tile_k
        return (group_id, b_tile_id, 0, n_id)

    def out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
        is_last_gm = gm_id == (pl.num_programs(1) - 1)
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
        last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
        row_size = row_end - row_start

        return (pl.ds(row_start, row_size), 0, n_id)


def generate_block_specs(
        metadata_ref: MetadataRef, cfgs: GmmConfigs
) -> Tuple[Tuple[LhsRef, WeightsRef], pl.BlockSpec]:
    """Generate pipeline BlockSpecs for logical LHS/RHS tiles."""

    index_map = IndexMaps(metadata_ref, cfgs)
    bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m //
                                       cfgs.dims.size_lhs_sublane)

    lhs_value_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
        index_map.lhs_index_map,
    )

    lhs_scale_spec = None
    if cfgs.lhs_cfgs.has_scale:
        lhs_scale_spec = pl.BlockSpec(
            (None, cfgs.num_lhs_scale_blocks_per_tile_k,
             cfgs.lhs_cfgs.scale_storage_width),
            index_map.lhs_scale_index_map,
        )

    lhs_quant_scale_spec = None
    if cfgs.lhs_cfgs.is_prequantized:
        lhs_quant_scale_spec = pl.BlockSpec(
            (bounded_slice_gm, cfgs.dims.size_lhs_sublane,
             cfgs.lhs_cfgs.scale_storage_width),
            index_map.lhs_quant_scale_index_map,
        )

    lhs_block_spec = LhsRef(value=lhs_value_spec,
                            scale_factor=lhs_scale_spec,
                            quant_scale=lhs_quant_scale_spec)

    # kernel_main bitcasts a packed RHS Ref to its logical [G, K, N] view
    # before these specs are constructed, so tile_k is always logical here.
    rhs_weight_spec = pl.BlockSpec(
        (None, cfgs.tiles.tile_k, cfgs.tiles.tile_n),
        index_map.rhs_weight_index_map,
        pipeline_mode=pl.Buffered(buffer_count=3),
    )

    rhs_scale_block_spec = rhs_bias_block_spec = None
    if cfgs.rhs_cfgs.has_bias:
        rhs_bias_block_spec = pl.BlockSpec(
            (None, 1, cfgs.tiles.tile_n),
            index_map.rhs_bias_index_map,
        )
    if cfgs.rhs_cfgs.has_scale:
        rhs_scale_block_spec = pl.BlockSpec(
            (None, cfgs.num_quant_blocks_per_tile_k, 1, cfgs.tiles.tile_n),
            index_map.rhs_scale_index_map,
        )

    rhs_block_spec = WeightsRef(weight=rhs_weight_spec,
                                scale=rhs_scale_block_spec,
                                bias=rhs_bias_block_spec)

    out_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
        index_map.out_index_map,
    )
    return (lhs_block_spec, rhs_block_spec), out_block_spec

# Define kernels.


def inner_kernel(
    # In
    tiled_lhs_ref: LhsRef,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_k]
    tiled_rhs_ref: RhsRef,  # [tile_k, tile_n]
    # Out
    tiled_out_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_n]
    # Scratch
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
):
    """Inner kernel invoked by emit_pipeline to perform matmul.

    tiled_lhs_ref and tiled_out_ref points to rows [m_start:m_end] of lhs and out.
    Additionally, m_start and m_end does not have to align with tile boundaries
    [m_offset:m_offset+tile_m]. Therefore, rows [m_offset:m_start] and
    [m_end:m_offset+tile_m] of tiled_lhs_ref and tiled_out_ref will contain
    invalid data and needs to be masked out.

    Args:
        tiled_lhs_ref: Contains value lhs[m_start:m_end, k_start:k_end]
        tiled_rhs_ref: Contains value rhs[g_id, k_start:k_end, n_start:n_end]. where
            g_id is the group associated with lhs[m_start:m_end, :]
        tiled_out_ref: Contains value out[m_start:m_end, n_start:n_end]
        partial_out_ref: Contains last size_lhs_sublane rows of the previous output.
            Will be initialized to zero if this is first tile for grid[n_id, :, :].
        acc_ref: Reference to the accumulator.
        metadata_ref: Reference to the metadata.
        cfgs: GmmConfigs.
    """

    gm_id = pl.program_id(1)
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane

    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset

    def _matmul(is_first_k_step: bool, is_last_k_step: bool, bucket_m: int, k_id: jax.Array | None = None):
        tpu_info = pltpu.get_tpu_info()
        mxu_size = tpu_info.mxu_column_size

        # Step 1: Input pre-processing.
        tiled_lhs = tiled_lhs_ref.get_value().reshape(
            -1, cfgs.tiles.tile_k)[:bucket_m]
        tiled_lhs_scale_factor = None
        if cfgs.lhs_cfgs.has_scale:
            tiled_lhs_scale_factor = tiled_lhs_ref.get_scale_factor()
        tiled_lhs_quant_scale = None
        if cfgs.lhs_cfgs.is_prequantized:
            tiled_lhs_quant_scale = tiled_lhs_ref.get_quant_scale().reshape(
                -1, cfgs.lhs_cfgs.scale_storage_width)[:bucket_m]
        tiled_rhs = tiled_rhs_ref.get_weight()

        # This should only be taken in the case where we don't requantize
        # the scales and thus we need to dequantize inside VMEM to avoid small
        # contracting dimmensions
        rhs_tile_n = tiled_rhs.shape[1]
        rhs_qbs = cfgs.rhs_cfgs.quant_block_size
        if cfgs.rhs_cfgs.should_dequantize_before_matmul:
            tiled_rhs_scale = tiled_rhs_ref.get_scale(
                replicate_size=rhs_qbs).astype(cfgs.lhs_cfgs.dtype)
            num_blocks = cfgs.num_quant_blocks_per_tile_k
            tiled_rhs_dequant = tiled_rhs.astype(cfgs.lhs_cfgs.dtype).reshape(
                num_blocks, rhs_qbs, rhs_tile_n)
            tiled_rhs_dequant = tiled_rhs_dequant * tiled_rhs_scale
            tiled_rhs = tiled_rhs_dequant.reshape(cfgs.tiles.tile_k,
                                                  rhs_tile_n)
            rhs_qbs = cfgs.tiles.tile_k

        valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
        if is_last_k_step and valid_k != 0:
            mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape,
                                            0) < valid_k
            tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)

        # Step 2: Matmul.
        acc_list = []
        if not cfgs.lhs_cfgs.should_quantize:
            # Unquantized matmul path.
            for start_n in range(0, rhs_tile_n, mxu_size):
                end_n = min(rhs_tile_n, start_n + mxu_size)
                col_size = end_n - start_n

                acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_ref.dtype)
                for start_k in range(0, cfgs.tiles.tile_k, rhs_qbs):
                    end_k = min(cfgs.tiles.tile_k, start_k + rhs_qbs)

                    block_acc = jnp.matmul(
                        tiled_lhs[:, start_k:end_k],
                        tiled_rhs[start_k:end_k, start_n:end_n],
                        preferred_element_type=jnp.float32,
                    ).astype(acc_ref.dtype)

                    if cfgs.rhs_cfgs.should_dequantize_after_matmul:
                        b_id = start_k // rhs_qbs
                        rhs_scale_1d = tiled_rhs_ref.get_scale_1d(
                            b_id, start_n, col_size)
                        block_acc *= rhs_scale_1d[None, :].astype(acc_ref.dtype)

                    acc_n += block_acc
                acc_list.append(acc_n)
        else:
            # Quantized matmul path.
            lhs_q_dtype = cfgs.lhs_cfgs.quant_dtype
            q_block_size = cfgs.lhs_cfgs.quant_block_size
            assert lhs_q_dtype is not None
            assert q_block_size is not None

            if jnp.issubdtype(lhs_q_dtype, jnp.floating):
                dtype_max = float(jnp.finfo(lhs_q_dtype).max)
                preferred_element_type = jnp.float32
            else:
                dtype_max = float(jnp.iinfo(lhs_q_dtype).max)
                preferred_element_type = jnp.int32

            # A32 path: quantize each [M, 512] LHS block exactly once per
            # pipeline invocation and reuse it across all N microtiles. A32 is
            # normalized offline so abs(A32) <= 1. The exact scaled absmax is
            # computed from sixteen group maxima without materializing lhs*A32.
            lhs_q_blocks = None
            lhs_q_scales = None
            if cfgs.lhs_cfgs.is_prequantized:
                assert tiled_lhs_quant_scale is not None
                if cfgs.tiles.tile_k % q_block_size != 0:
                    raise ValueError(
                        "Prequantized LHS requires tile_k divisible by 512.")
                num_q_blocks = cfgs.tiles.tile_k // q_block_size
                lhs_q_blocks = tuple(
                    tiled_lhs[:, block_id * q_block_size:
                              (block_id + 1) * q_block_size]
                    for block_id in range(num_q_blocks)
                )
            elif cfgs.lhs_cfgs.has_scale:
                assert tiled_lhs_scale_factor is not None
                factor_block = cfgs.lhs_cfgs.scale_block_size
                assert factor_block == 512
                if q_block_size != factor_block:
                    raise ValueError(
                        "The factorized path currently requires one dynamic "
                        "LHS scale per 512 values.")
                if cfgs.tiles.tile_k % q_block_size != 0:
                    raise ValueError(
                        "tile_k must be divisible by 512 for A32 fusion.")

                num_q_blocks = cfgs.tiles.tile_k // q_block_size
                lhs_q_blocks_list = []
                lhs_q_scales_list = []
                for block_id in range(num_q_blocks):
                    start_k = block_id * q_block_size
                    end_k = start_k + q_block_size
                    block_lhs = tiled_lhs[:, start_k:end_k]

                    # Physical row is padded to 128. The first sixteen values
                    # are the sixteen source block-32 factors in this K=512
                    # region. Keep all per-element arithmetic in LHS dtype.
                    a32 = tiled_lhs_scale_factor[block_id, :16].astype(
                        block_lhs.dtype)
                    block_grouped = block_lhs.reshape(bucket_m, 16, 32)

                    group_abs_max = jnp.max(
                        jnp.abs(block_grouped), axis=2)
                    block_abs_max = jnp.max(
                        group_abs_max * jnp.abs(a32)[None, :],
                        axis=1,
                        keepdims=True,
                    )
                    block_scale = block_abs_max / dtype_max
                    block_scale_inv = jnp.where(block_scale == 0, 0,
                                                1 / block_scale)

                    effective_multiplier = (
                        a32[None, :, None]
                        * block_scale_inv[:, None, :]
                    )
                    block_lhs_q = (
                        block_grouped * effective_multiplier
                    ).astype(lhs_q_dtype).reshape(bucket_m, q_block_size)

                    lhs_q_blocks_list.append(block_lhs_q)
                    lhs_q_scales_list.append(block_scale)

                lhs_q_blocks = tuple(lhs_q_blocks_list)
                lhs_q_scales = tuple(lhs_q_scales_list)

            # N remains outermost so each completed [M, MXU_N] result can feed
            # the output pipeline while MXU work for the next columns proceeds.
            for start_n in range(0, rhs_tile_n, mxu_size):
                end_n = min(rhs_tile_n, start_n + mxu_size)
                col_size = end_n - start_n
                acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_ref.dtype)

                for block_id, start_k in enumerate(
                        range(0, cfgs.tiles.tile_k, q_block_size)):
                    end_k = min(cfgs.tiles.tile_k, start_k + q_block_size)
                    block_rhs = tiled_rhs[start_k:end_k, start_n:end_n]

                    if cfgs.lhs_cfgs.is_prequantized:
                        assert lhs_q_blocks is not None
                        block_lhs_q = lhs_q_blocks[block_id]
                        global_block_id = k_id * num_q_blocks + block_id
                        block_scale = tiled_lhs_quant_scale[:, global_block_id:global_block_id + 1]
                    elif cfgs.lhs_cfgs.has_scale:
                        assert lhs_q_blocks is not None
                        assert lhs_q_scales is not None
                        block_lhs_q = lhs_q_blocks[block_id]
                        block_scale = lhs_q_scales[block_id]
                    else:
                        block_lhs = tiled_lhs[:, start_k:end_k]
                        block_abs_max = jnp.max(
                            jnp.abs(block_lhs), axis=1, keepdims=True)
                        block_scale = block_abs_max / dtype_max
                        block_scale_inv = jnp.where(block_scale == 0, 0,
                                                    1 / block_scale)
                        block_lhs_q = (
                            block_lhs * block_scale_inv
                        ).astype(lhs_q_dtype)

                    if not tpu_info.is_matmul_supported(
                            lhs_q_dtype, block_rhs.dtype):
                        block_rhs = block_rhs.astype(lhs_q_dtype)

                    block_acc = jnp.matmul(
                        block_lhs_q,
                        block_rhs,
                        preferred_element_type=preferred_element_type,
                    ).astype(acc_ref.dtype)

                    block_acc *= block_scale.astype(acc_ref.dtype)

                    if cfgs.rhs_cfgs.should_dequantize_after_matmul:
                        b_id = start_k // rhs_qbs
                        rhs_scale_1d = tiled_rhs_ref.get_scale_1d(
                            b_id, start_n, col_size)
                        block_acc *= rhs_scale_1d[None, :].astype(
                            acc_ref.dtype)

                    acc_n += block_acc
                acc_list.append(acc_n)
        acc = jnp.concatenate(acc_list, axis=1)

        # Step 3: Output post-processing.
        if not is_first_k_step:
            acc = jnp.pad(acc, ((cfgs.tiles.tile_m - bucket_m, 0), (0, 0)))
            acc += acc_ref[...]
        acc_m = acc.shape[0]

        if is_last_k_step:
            if cfgs.rhs_cfgs.has_bias:
                tiled_rhs_bias = tiled_rhs_ref.get_bias()
                acc += tiled_rhs_bias.astype(acc.dtype)

            acc = apply_act_fn(acc, cfgs.fuse_act)

            # Mask out rows that does not belong to the current group.
            iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
            mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
            acc_masked = jnp.where(mask, acc, 0)

            # Write the final output to the output ref.
            tiled_out_2d_ref = tiled_out_ref.reshape(-1, cfgs.tiles.tile_n)
            tiled_out_2d_ref[:acc_m] = acc_masked.astype(tiled_out_ref.dtype)

            # If this is the first tile for grid[n_id, :, :], we initialize the
            # partial out to zeros. Otherwise, partial out from last tile of
            # grid[n_id-1, :, :] can be used and cause numeric issues.
            partial_out_zeros = jnp.zeros_like(partial_out_ref)

            # Accumulate the partial output from the previous step.
            tiled_out_ref[0] += jnp.where(gm_id == 0, partial_out_zeros,
                                          partial_out_ref[...])

            # Consider following case where size_lhs_sublane = 4, number denotes group
            # id and | denotes boundaries between sublanes:
            # | 0 0 1 2 | 2 2 2 2 | 3 3 4 4 |
            #
            # Assuming group id of current step is 1, current step will not completely
            # fill size_lhs_sublane rows and will be revisited at the next step. By
            # storing the partial rows into the partial_out_ref, the next step can
            # read them and accumulate to them.  Additionally, for group id of 2,
            # since it completely fills the size_lhs_sublane rows, we need to zero out
            # partial_out_ref to avoid numeric error for group 3.
            last_row = m_end_local // cfgs.dims.size_lhs_sublane
            partial_out_ref[...] = jnp.where(
                m_end_local % cfgs.dims.size_lhs_sublane == 0,
                partial_out_zeros,
                tiled_out_ref[last_row],
            )
        else:
            acc_ref[:acc_m] = acc

    def run_matmul_step(bucket_idx: int):
        bucket_m = cfgs.tiles.bucket_base * (bucket_idx + 1)
        num_k = pl.num_programs(2)
        k_id = pl.program_id(2)

        @jax.named_scope(f"bm{bucket_m}_first_last")
        def matmul_first_last():
            _matmul(is_first_k_step=True,
                    is_last_k_step=True,
                    bucket_m=bucket_m,
                    k_id=k_id)

        @jax.named_scope(f"bm{bucket_m}_first")
        def matmul_first():
            _matmul(is_first_k_step=True,
                    is_last_k_step=False,
                    bucket_m=bucket_m,
                    k_id=k_id)

        @jax.named_scope(f"bm{bucket_m}_mid")
        def matmul_mid():
            _matmul(is_first_k_step=False,
                    is_last_k_step=False,
                    bucket_m=bucket_m,
                    k_id=k_id)

        @jax.named_scope(f"bm{bucket_m}_last")
        def matmul_last():
            _matmul(is_first_k_step=False,
                    is_last_k_step=True,
                    bucket_m=bucket_m,
                    k_id=k_id)

        is_first_k_step = k_id == 0
        is_last_k_step = k_id == (num_k - 1)

        if bucket_m == cfgs.tiles.tile_m:
            lax.cond(
                is_first_k_step,
                lambda: lax.cond(is_last_k_step, matmul_first_last,
                                 matmul_first),
                lambda: lax.cond(is_last_k_step, matmul_last, matmul_mid),
            )
        else:
            # partial m is only invoked at last matmul.
            lax.cond(is_first_k_step, matmul_first_last, matmul_last)

    branches = []
    num_buckets = max(1, cfgs.tiles.tile_m // cfgs.tiles.bucket_base)
    for bucket_idx in range(num_buckets):
        branches.append(
            functools.partial(run_matmul_step, bucket_idx=bucket_idx))
    bucket_idx = m_end_local // cfgs.tiles.bucket_base
    bucket_idx = jnp.minimum(bucket_idx, num_buckets - 1)
    lax.switch(bucket_idx, branches)


def fill_metadata(
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
) -> jax.Array:
    """Fills the metadata for the given lhs group sizes and group offset.

    Iterates over the lhs group sizes and if the group id is valid, determines
    the number of gm tiles that are needed to process the current group. Then,
    it fills starting and ending offset (gm_id_to_m_offset), and the group id
    (gm_id_to_group_id) for each gm tile.

    Args:
        lhs_group_sizes_ref: The group sizes of lhs.
        group_offset_ref: Offset of the first group to process.
        metadata_ref: Metadata that is used to determine the group id and m offsets
            for each gmm tile.
        cfgs: GmmConfigs.

    Returns:
        The number of gm tiles to process lhs with given group offset.
    """

    group_offset = group_offset_ref[0]
    max_num_group = group_offset + cfgs.dims.size_group
    metadata_ref.gm_id_to_m_offset[0] = 0

    @jax.named_scope("inner_tm_loop")
    def inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id):
        local_offset = curr_m_offset % cfgs.dims.size_lhs_sublane
        tm_size = jnp.minimum(cfgs.tiles.tile_m - local_offset,
                              end_m_offset - curr_m_offset)

        metadata_ref.gm_id_to_group_id[tm_id] = group_id

        next_m_offset = curr_m_offset + tm_size
        metadata_ref.gm_id_to_m_offset[tm_id] = curr_m_offset
        metadata_ref.gm_id_to_m_offset[tm_id + 1] = next_m_offset

        return next_m_offset

    @jax.named_scope("outer_group_loop")
    def outer_group_loop(lhs_group_id, carry):
        num_gm, start_m_offset = carry

        group_id = lhs_group_id - group_offset
        group_size = lhs_group_sizes_ref[lhs_group_id]
        end_m_offset = start_m_offset + group_size

        # Assume following arguments:
        # - size_lhs_sublane & tile_m = 4
        # - group_size = 3
        # - start_m_offset = 7
        #
        # If we visualize it, it will look like this where:
        # - |: denotes boundaries between sublanes
        # - 0: denotes values for other groups
        # - 1: denotes values for the current group
        # | 0 0 0 0 | 0 0 0 1 | 1 1 0 0 |
        #
        # In this example, we see that we require processing 2 m tiles.
        # But, performing a naive cdiv(group_size, tile_m) will return 1.
        # Instead, adding local_offset will give us the correct value.
        local_offset = start_m_offset % cfgs.dims.size_lhs_sublane
        aligned_group_size = group_size + local_offset
        curr_num_gm = pl.cdiv(aligned_group_size, cfgs.tiles.tile_m)

        # We need to handle cases where we should not process the group.
        # 1. Even if group_size is 0, if local_offset is not 0, cdiv will return 1.
        # 2. If group comes before the group_offset, we should not process it.
        should_process = jnp.logical_and(group_size > 0, group_id >= 0)
        curr_num_gm = jnp.where(should_process, curr_num_gm, 0)
        next_num_gm = num_gm + curr_num_gm

        tm_loop_fn = functools.partial(
            inner_tm_loop,
            end_m_offset=end_m_offset,
            group_id=group_id,
        )
        lax.fori_loop(num_gm, next_num_gm, tm_loop_fn, start_m_offset)

        return next_num_gm, end_m_offset

    num_gm, _ = lax.fori_loop(0, max_num_group, outer_group_loop, (0, 0))
    return num_gm


def zero_out_start(
    out_ref: jax.Array,  # [size_m, size_n]
    zero_ref: jax.Array,  # [tile_zero_m, num_lanes]
    semaphore_ref: jax.Array,  # [1]
    metadata_ref: MetadataRef,
    num_gm: jax.Array,
    *,
    dims: Dimensions,
):
    """Zero out output rows that are not used in the computation."""

    num_lanes = pltpu.get_tpu_info().num_lanes
    assert num_lanes == zero_ref.shape[-1]
    zero_ref[...] = jnp.zeros_like(zero_ref)

    zero_dma = zero_ref.reshape(-1, dims.size_lhs_sublane, num_lanes)
    out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, out_ref.shape[-1])
    row_size = zero_dma.shape[0]

    compute_start = metadata_ref.gm_id_to_m_offset[0]
    compute_end = metadata_ref.gm_id_to_m_offset[num_gm]

    left_zero_start = 0
    left_zero_end = compute_start // dims.size_lhs_sublane
    left_zero_size = left_zero_end - left_zero_start
    left_num_loops = pl.cdiv(left_zero_size, row_size)

    right_zero_start = pl.cdiv(compute_end, dims.size_lhs_sublane)
    right_zero_end = out_dma.shape[0]
    right_zero_size = right_zero_end - right_zero_start
    right_num_loops = pl.cdiv(right_zero_size, row_size)

    def fill_zero(i, zero_size, *, start, end):
        dma_start = start + i * row_size
        dma_end = jnp.minimum(dma_start + row_size, end)
        dma_size = dma_end - dma_start

        # Static loop. Will be unrolled during compile time.
        for n_start in range(0, out_dma.shape[-1], num_lanes):
            n_end = n_start + num_lanes
            pltpu.make_async_copy(
                src_ref=zero_dma.at[pl.ds(0, dma_size)],
                dst_ref=out_dma.at[pl.ds(dma_start, dma_size), :,
                                   n_start:n_end],
                sem=semaphore_ref.at[0],
            ).start(priority=1)

        return zero_size + dma_size

    @jax.named_scope("left_fill_zero")
    def left_fill_zero(i, zero_size):
        return fill_zero(i,
                         zero_size,
                         start=left_zero_start,
                         end=left_zero_end)

    @jax.named_scope("right_fill_zero")
    def right_fill_zero(i, zero_size):
        return fill_zero(i,
                         zero_size,
                         start=right_zero_start,
                         end=right_zero_end)

    zero_size = lax.fori_loop(0, left_num_loops, left_fill_zero, 0)
    zero_size = lax.fori_loop(0, right_num_loops, right_fill_zero, zero_size)
    return zero_size


def zero_out_end(
    out_ref: jax.Array,  # [size_m, size_n]
    semaphore_ref: jax.Array,  # [1]
    zero_size: jax.Array,
    *,
    dims: Dimensions,
):
    out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, out_ref.shape[-1])
    pltpu.make_async_copy(
        src_ref=out_dma.at[pl.ds(0, zero_size)],
        dst_ref=out_dma.at[pl.ds(0, zero_size)],
        sem=semaphore_ref.at[0],
    ).wait()


def kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    # In
    lhs_ref: LhsRef,  # value=[M,K], factor=[G,K/512,128]
    rhs_ref: WeightsRef,  # [size_group, size_k, size_n]
    # Out
    out_ref: jax.Array,  # [size_m, size_n]
    # Scratch memory
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    zero_ref: jax.Array | None,  # [tile_zero_m, num_lanes]
    semaphore_ref: jax.Array | None,  # [1]
    *,
    cfgs: GmmConfigs,
):
    """Entry point for GMM kernel.

    Computes metadata to determine which rows of lhs needs processing and how
    they will be tiled. And then, invoke inner kernel using metadata.

    Uses the following notation:
    - g: rhs group dimension
    - m: Batch dimension
    - gm: Batch tiling dimension. Aligned to size_lhs_sublane and has tile size
        of tile_m. Skips over empty groups and accounts for revisited tiles.
    - k: in dimension
    - n: out dimension

    Args:
        lhs_group_sizes_ref: Reference to the group sizes of lhs.
        group_offset_ref: Reference to the group offset.
        lhs_ref: Reference to the lhs.
        rhs_ref: Reference to the rhs.
        out_ref: Reference to the out.
        partial_out_ref: Reference to the partial output.
        acc_ref: Reference to the accumulator.
        metadata_ref: Reference to the metadata.
        zero_ref: Scratch memory for storing zero values used in initialization.
        semaphore_ref: Semaphore for zero initialization DMAs.
        cfgs: GmmConfigs.
    """

    num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

    if cfgs.rhs_cfgs.should_unpack:
        rhs_weight = rhs_ref.weight.bitcast(cfgs.rhs_cfgs.quant_dtype)
        expected_shape = (cfgs.dims.size_group, cfgs.dims.size_k,
                          cfgs.dims.size_n)
        if rhs_weight.shape != expected_shape:
            raise ValueError(
                f"Packed RHS bitcast produced {rhs_weight.shape}; expected "
                f"{expected_shape}. Packing must be along K.")
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

    # Fill metadata buffer and return number of group & m interations.
    num_gm = fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=cfgs,
    )

    if cfgs.zero_init:
        zero_size = zero_out_start(
            out_ref,
            zero_ref,
            semaphore_ref,
            metadata_ref,
            num_gm,
            dims=cfgs.dims,
        )

    (lhs_spec, rhs_spec), out_spec = generate_block_specs(metadata_ref, cfgs)

    if cfgs.fuse_act is not None:
        rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n:],
                                  rhs_ref)
        rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)

        rhs_spec = FusedWeightsRef(
            gate=rhs_spec,
            up=rhs_spec,
        )

    # Execute the inner kernel.
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(inner_kernel, cfgs=cfgs),
        grid=(num_n, num_gm, num_k),
        in_specs=(lhs_spec, rhs_spec),
        out_specs=out_spec,
    )

    # Bounded slice requires second last dim to be aligned to the sublane size.
    # rhs_ref uses static tiling thus reshape is not needed.
    lhs_in = LhsRef(
        value=lhs_ref.value.reshape(
            -1, cfgs.dims.size_lhs_sublane, lhs_ref.value.shape[-1]),
        scale_factor=lhs_ref.scale_factor,
        quant_scale=(
            None if lhs_ref.quant_scale is None else
            lhs_ref.quant_scale.reshape(
                -1, cfgs.dims.size_lhs_sublane,
                lhs_ref.quant_scale.shape[-1])
        ),
    )
    out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
    scratches = [partial_out_ref, acc_ref, metadata_ref]
    pipeline_fn(lhs_in, rhs_ref, out_in, scratches=scratches)

    if cfgs.zero_init:
        zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs.dims)


def calculate_tiling(
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    fuse_act: str | None = None,
) -> TileSizes:
    """Calculate optimal tile sizes for GMM kernel."""

    lhs_dtype = lhs_cfgs.dtype
    rhs_dtype = rhs_cfgs.quant_dtype or rhs_cfgs.dtype

    lhs_bits = jax.dtypes.itemsize_bits(lhs_dtype)
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)

    # When using bf16 for lhs and rhs, 128 is the largest tile_m value that is
    # safe to use for most scenarios. But if lower bitwidth is used, we need
    # to tweak tile_m to account for using faster hardware unit.
    # TODO(kyuyeunk): Account for different TPU hardware specs.
    bf16_bf16_tile_m = 128
    rhs_mod = min(pl.cdiv(16, rhs_bits), 2)
    tile_m = bf16_bf16_tile_m // rhs_mod
    if lhs_cfgs.should_quantize:
        tile_m *= 2
    tile_m = min(tile_m, dims.size_m)

    # To avoid stalling MXU, we add some buffer room where tile_n cannot go
    # smaller than 2x of mxu_column_size.
    tile_n_limit = pltpu.get_tpu_info().mxu_column_size * 2
    tile_n_limit = min(tile_n_limit, dims.size_n)

    size_n_per_rhs = dims.size_n
    fuse_act_factor = 1
    if fuse_act is not None:
        # When computing activation function, rhs is concatenated along dim n.
        fuse_act_factor = 2
        size_n_per_rhs //= fuse_act_factor
        tile_n_limit //= fuse_act_factor

    def _is_tile_k_quant_block_compatible(tk: int) -> bool:
        if (tk % rhs_cfgs.quant_block_size != 0
                and rhs_cfgs.quant_block_size % tk != 0):
            return False
        if ((lhs_cfgs.has_scale or lhs_cfgs.is_prequantized)
                and tk % 512 != 0):
            return False
        return True

    # Initialize tile_k and tile_n to their maximum valid values.
    num_k_tiles = num_n_tiles = 1
    num_lanes = pltpu.get_tpu_info().num_lanes
    tile_k = align_to(dims.size_k, num_lanes)
    tile_n = align_to(size_n_per_rhs, num_lanes)

    def _gmm_vmem_estimate(tm: int, tn: int, tk: int) -> int:
        # 1. LHS tile (double-buffered HBM load)
        lhs_tile_bytes = lhs_bits // 8
        lhs_vmem = 2 * tm * tk * lhs_tile_bytes
        # If LHS is quantized on-the-fly, we need an extra single-buffered cast
        # buffer in VMEM.
        if lhs_cfgs.should_quantize and not lhs_cfgs.is_prequantized:
            lhs_quant_bits = jax.dtypes.itemsize_bits(lhs_cfgs.quant_dtype)
            lhs_vmem += tm * tk * (lhs_quant_bits // 8)

        if lhs_cfgs.has_scale:
            assert lhs_cfgs.scale_dtype is not None
            assert lhs_cfgs.scale_block_size is not None
            scale_bytes = jax.dtypes.itemsize_bits(
                lhs_cfgs.scale_dtype) // 8
            factor_rows = pl.cdiv(tk, lhs_cfgs.scale_block_size)
            lhs_vmem += (factor_rows * lhs_cfgs.scale_storage_width
                         * scale_bytes)
            # Dynamic output scale for every [M, 512] block.
            lhs_vmem += tm * factor_rows * 4
        if lhs_cfgs.is_prequantized:
            lhs_vmem += tm * pl.cdiv(tk, 512) * 2

        # 2. RHS tile (triple-buffered, includes scale and bias if present)
        # If fuse_act is enabled, we have both gate and up weights,
        # so RHS memory is doubled.
        rhs_weight_vmem = tk * tn * rhs_bits // 8
        rhs_scale_vmem = 0
        if rhs_cfgs.has_scale and rhs_cfgs.quant_block_size is not None:
            num_quant_blocks_per_tile_k = pl.cdiv(tk,
                                                  rhs_cfgs.quant_block_size)
            rhs_scale_vmem = num_quant_blocks_per_tile_k * tn * 4
        rhs_bias_vmem = 0
        if rhs_cfgs.has_bias:
            rhs_bias_vmem = tn * 4
        rhs_vmem = fuse_act_factor * (3 * rhs_weight_vmem +
                                      2 * rhs_scale_vmem + 2 * rhs_bias_vmem)

        # 3. Accumulator
        acc_cols = fuse_act_factor * tn
        acc_dtype_bytes = 2 if lhs_cfgs.quant_dtype is not None else 4
        acc_vmem = tm * acc_cols * acc_dtype_bytes

        # 4. Output tile (double-buffered)
        out_dtype_bytes = jax.dtypes.itemsize_bits(lhs_cfgs.dtype) // 8
        out_vmem = 2 * tm * tn * out_dtype_bytes

        return lhs_vmem + rhs_vmem + acc_vmem + out_vmem

    # Multiple k tiles will introduce accumulation overhead. Thus, we first try
    # to fit the tensors into vmem by only adjusting tile_n.

    # Decrease tile_n until total memory fits in vmem limit.
    while (_gmm_vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes
           and tile_n > tile_n_limit):
        num_n_tiles += 1
        tile_n = align_to(size_n_per_rhs,
                          num_n_tiles * num_lanes) // num_n_tiles

    # If decreasing tile_n is no longer possible, we decrease tile_k instead.
    if tile_n < tile_n_limit:
        num_n_tiles -= 1
        tile_n = align_to(size_n_per_rhs,
                          num_n_tiles * num_lanes) // num_n_tiles

        # Decrease tile_k until total memory fits in vmem limit and tile_k is valid.
        while (_gmm_vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes
               or not _is_tile_k_quant_block_compatible(tile_k)
               ) and tile_k > num_lanes:
            num_k_tiles += 1
            tile_k = align_to(dims.size_k,
                              num_k_tiles * num_lanes) // num_k_tiles

    if tile_n == 0 or tile_k == 0:
        final_estimate = _gmm_vmem_estimate(tile_m, tile_n, tile_k)
        raise ValueError(f"Could not find valid tile sizes for {dims=} and"
                         f" {final_estimate=} (limit: {vmem_limit_bytes}).")

    # TODO(alynie, kyuyeunk): max number of bucket was choosen empiirically.
    # Revisit the value to be based on number of instruction memory size.
    max_num_buckets = 4
    bucket_base = tile_m
    for _ in range(1, max_num_buckets):
        new_tile_m = tile_m + bucket_base
        if new_tile_m > dims.size_m:
            break
        if _gmm_vmem_estimate(new_tile_m, tile_n, tile_k) > vmem_limit_bytes:
            break
        tile_m = new_tile_m

    return TileSizes(tile_m=tile_m,
                     tile_k=tile_k,
                     tile_n=tile_n,
                     bucket_base=bucket_base)


def validate_inputs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    lhs_scale_factor: jax.Array | None,
    lhs_quant_scale: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    fuse_act: str | None = None,
    packing_factor: int = 1,
) -> Dimensions:
    """Validate logical dimensions and the fixed A32 storage layout."""

    size_m = lhs.shape[0]
    size_group, physical_k, size_n = rhs.shape
    size_k = physical_k * packing_factor
    size_lhs_group = group_sizes.shape[0]

    if size_group > size_lhs_group:
        raise ValueError(
            f"RHS groups {size_group} exceed LHS groups {size_lhs_group}.")
    if lhs.shape != (size_m, size_k):
        raise ValueError(
            f"lhs shape {lhs.shape} must be {(size_m, size_k)}.")
    if rhs_bias is not None and rhs_bias.shape != (size_group, 1, size_n):
        raise ValueError(f"Invalid rhs_bias shape {rhs_bias.shape}.")

    if rhs_scale is not None:
        num_quant_blocks = rhs_scale.shape[1]
        expected = (size_group, num_quant_blocks, 1, size_n)
        if rhs_scale.shape != expected:
            raise ValueError(
                f"rhs_scale shape {rhs_scale.shape} must be {expected}.")
        if size_k % num_quant_blocks != 0:
            raise ValueError(
                f"Logical K={size_k} must be divisible by "
                f"rhs scale blocks={num_quant_blocks}.")

    if lhs_scale_factor is not None and lhs_quant_scale is not None:
        raise ValueError(
            "Supply either lhs_scale_factor for in-kernel fusion or "
            "lhs_quant_scale for a prequantized LHS, not both.")

    if lhs_scale_factor is not None:
        if size_k % 512 != 0:
            raise ValueError(
                f"A32 fusion requires logical K divisible by 512; got {size_k}.")
        expected = (size_group, size_k // 512, 128)
        if lhs_scale_factor.shape != expected:
            raise ValueError(
                "lhs_scale_factor must use the stable padded layout "
                f"[G, K/512, 128]; got {lhs_scale_factor.shape}, expected "
                f"{expected}. Only [:, :, :16] contains A32 values.")
        if rhs_scale is None:
            raise ValueError(
                "A32 fusion requires the factorized outer B512 rhs_scale.")
        rhs_block = size_k // rhs_scale.shape[1]
        if rhs_block != 512:
            raise ValueError(
                f"A32 fusion requires B512 scales; got block {rhs_block}.")

    if lhs_quant_scale is not None:
        if lhs.dtype != jnp.dtype(jnp.float8_e4m3fn):
            raise ValueError(
                "The prequantized path currently expects FP8 E4M3FN lhs; "
                f"got {lhs.dtype}.")
        expected = (size_m, 128)
        if lhs_quant_scale.shape != expected:
            raise ValueError(
                f"lhs_quant_scale shape {lhs_quant_scale.shape} must be "
                f"{expected} (padded to 128 elements).")
        if rhs_scale is None or size_k // rhs_scale.shape[1] != 512:
            raise ValueError(
                "Prequantized LHS requires matching B512 rhs scales.")

    if group_offset.shape != (1,):
        raise ValueError(
            f"group_offset must have shape (1,), got {group_offset.shape}.")

    size_lhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(lhs.dtype)
    size_lhs_sublane = min(size_lhs_sublane, size_m)
    if fuse_act is not None:
        num_lanes = pltpu.get_tpu_info().num_lanes
        if size_n % (2 * num_lanes) != 0:
            raise ValueError(
                f"size_n={size_n} must be divisible by 2*num_lanes when "
                "fuse_act is enabled.")

    return Dimensions(size_m=size_m,
                      size_k=size_k,
                      size_n=size_n,
                      size_group=size_group,
                      size_lhs_group=size_lhs_group,
                      size_lhs_sublane=size_lhs_sublane)

def get_cost_estimate(cfgs: GmmConfigs):
    """Return logical HBM traffic; packed RHS uses logical element bits."""

    dims = cfgs.dims
    lhs_dtype = cfgs.lhs_cfgs.dtype
    rhs_dtype = cfgs.rhs_cfgs.quant_dtype or cfgs.rhs_cfgs.dtype
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)
    fp32_bytes = jnp.dtype(jnp.float32).itemsize

    flops = 2 * dims.size_m * dims.size_k * dims.size_n
    lhs_bytes = dims.size_m * dims.size_k * lhs_dtype.itemsize
    rhs_bytes = dims.size_group * dims.size_k * dims.size_n * rhs_bits // 8

    if cfgs.rhs_cfgs.has_scale:
        num_blocks = pl.cdiv(dims.size_k,
                              cfgs.rhs_cfgs.quant_block_size)
        rhs_bytes += dims.size_group * num_blocks * dims.size_n * fp32_bytes
    if cfgs.rhs_cfgs.has_bias:
        rhs_bytes += dims.size_group * dims.size_n * fp32_bytes

    if cfgs.lhs_cfgs.has_scale:
        scale_bytes = jax.dtypes.itemsize_bits(
            cfgs.lhs_cfgs.scale_dtype) // 8
        lhs_bytes += (dims.size_group * (dims.size_k // 512)
                      * cfgs.lhs_cfgs.scale_storage_width * scale_bytes)
    if cfgs.lhs_cfgs.is_prequantized:
        lhs_bytes += dims.size_m * (dims.size_k // 512) * 2

    out_bytes = dims.size_m * cfgs.out_size_n * cfgs.out_dtype.itemsize
    return pl.CostEstimate(flops=flops,
                           bytes_accessed=lhs_bytes + rhs_bytes + out_bytes,
                           transcendentals=0)

def get_scope_name(cfgs: GmmConfigs) -> str:
    dims = cfgs.dims
    tiles = cfgs.tiles
    return (
        f"gmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}-act_{cfgs.fuse_act}"
        f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
    )


def make_gmm_configs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    lhs_scale_factor: jax.Array | None,
    lhs_quant_scale: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    *,
    tile_info: TileSizes | TileFn,
    vmem_limit_bytes: int | None,
    out_dtype: jnp.dtype | None,
    acc_dtype: jnp.dtype | None,
    maybe_quantize_lhs: bool,
    zero_initialize: bool,
    fuse_act: str | None = None,
    rhs_logical_dtype: jnp.dtype | None = None,
):
    """Fill GMM config for packed W4 and optional A32/B512 factors."""

    rhs_logical_dtype = (None if rhs_logical_dtype is None else
                         jnp.dtype(rhs_logical_dtype))
    packing = get_packing_factor(rhs.dtype, rhs_logical_dtype)
    dims = validate_inputs(lhs, rhs, rhs_scale, rhs_bias, lhs_scale_factor,
                           lhs_quant_scale, group_sizes, group_offset,
                           fuse_act, packing)

    if rhs_scale is not None:
        rhs_quant_dtype = rhs_logical_dtype or jnp.dtype(rhs.dtype)
        rhs_block_size = dims.size_k // rhs_scale.shape[1]
    else:
        rhs_quant_dtype = None
        rhs_block_size = dims.size_k

    rhs_cfgs = InputConfigs(
        quant_dtype=rhs_quant_dtype,
        quant_block_size=rhs_block_size,
        dtype=jnp.dtype(rhs.dtype),
        has_bias=rhs_bias is not None,
        has_scale=rhs_scale is not None,
    )

    lhs_q_dtype = None
    lhs_is_prequantized = lhs_quant_scale is not None
    if lhs_is_prequantized:
        lhs_q_dtype = jnp.dtype(lhs.dtype)
    elif maybe_quantize_lhs and rhs_cfgs.should_dequantize_after_matmul:
        tpu_info = pltpu.get_tpu_info()
        is_rhs_float = jnp.issubdtype(rhs_quant_dtype, jnp.floating)
        is_rhs_4bit = jax.dtypes.itemsize_bits(rhs_quant_dtype) == 4

        if lhs_scale_factor is not None:
            if tpu_info.fp8_ops_per_second <= 0:
                raise NotImplementedError(
                    "A32 fusion requires FP8 MXU support.")
            lhs_q_dtype = jnp.dtype(jnp.float8_e4m3fn)
        else:
            if tpu_info.fp8_ops_per_second > 0 and (is_rhs_float
                                                    or is_rhs_4bit):
                lhs_q_dtype = jnp.dtype(jnp.float8_e4m3fn)
            if tpu_info.int8_ops_per_second > 0 and not is_rhs_float:
                lhs_q_dtype = jnp.dtype(jnp.int8)

    lhs_cfgs = InputConfigs(
        quant_dtype=lhs_q_dtype,
        quant_block_size=512,
        dtype=jnp.dtype(lhs.dtype),
        has_scale=lhs_scale_factor is not None,
        scale_dtype=(
            jnp.dtype(lhs_scale_factor.dtype)
            if lhs_scale_factor is not None
            else (jnp.dtype(jnp.bfloat16) if lhs_is_prequantized else None)
        ),
        scale_block_size=(
            512 if (lhs_scale_factor is not None or lhs_is_prequantized)
            else None
        ),
        scale_storage_width=128,
        is_prequantized=lhs_is_prequantized,
    )

    if lhs_scale_factor is not None and lhs_q_dtype is None:
        raise ValueError(
            "lhs_scale_factor was supplied but LHS quantization is disabled.")

    if out_dtype is None:
        out_dtype = lhs.dtype
    if acc_dtype is None:
        acc_dtype = (jnp.float32.dtype if lhs_q_dtype is None else
                     jnp.bfloat16.dtype)

    if isinstance(tile_info, TileSizes):
        tiles = tile_info
    else:
        tiles = tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)

    if ((lhs_scale_factor is not None or lhs_quant_scale is not None)
            and tiles.tile_k % 512 != 0):
        raise ValueError(
            f"A32 fusion requires tile_k divisible by 512; got {tiles.tile_k}.")

    return GmmConfigs(dims=dims,
                      tiles=tiles,
                      lhs_cfgs=lhs_cfgs,
                      rhs_cfgs=rhs_cfgs,
                      out_dtype=jnp.dtype(out_dtype),
                      acc_dtype=jnp.dtype(acc_dtype),
                      zero_init=zero_initialize,
                      fuse_act=fuse_act)

def get_metadata(cfgs: GmmConfigs) -> dict[str, str | int | float]:
    cfgs_dict = dataclasses.asdict(cfgs)
    ret = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
        key = jax.tree_util.keystr(path, simple=True, separator=".")
        if not isinstance(val, str | int | float):
            val = str(val)
        ret[key] = val
    return ret


@jax.jit(static_argnames=[
    "tile_info",
    "vmem_limit_bytes",
    "precision",
    "preferred_element_type",
    "acc_dtype",
    "maybe_quantize_lhs",
    "zero_initialize",
    "fuse_act",
    "rhs_logical_dtype",
])
def gmm_v2(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    lhs_scale_factor: jax.Array | None = None,
    lhs_quant_scale: jax.Array | None = None,
    *,
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    zero_initialize: bool = True,
    fuse_act: str | None = None,
    rhs_logical_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """Grouped matmul with optional packed W4 and A32/B512 factorization.

    lhs_scale_factor uses physical shape [G, K/512, 128]. The first sixteen
    values in each row are normalized nonnegative A32 factors with max <= 1.
    The removed maximum must already be folded into rhs_scale (B512).
    """

    del precision

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    elif jnp.isscalar(group_offset):
        group_offset = group_offset[None]

    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

    # Prevent accidental FP32 promotion in the elementwise A32 path.
    if lhs_scale_factor is not None:
        lhs_scale_factor = lhs_scale_factor.astype(lhs.dtype)

    cfgs = make_gmm_configs(
        lhs,
        rhs,
        rhs_scale,
        rhs_bias,
        lhs_scale_factor,
        lhs_quant_scale,
        group_sizes,
        group_offset,
        tile_info=tile_info,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=preferred_element_type,
        acc_dtype=acc_dtype,
        maybe_quantize_lhs=maybe_quantize_lhs,
        zero_initialize=zero_initialize,
        fuse_act=fuse_act,
        rhs_logical_dtype=rhs_logical_dtype,
    )
    dims = cfgs.dims
    tiles = cfgs.tiles

    rhs_scale_spec = rhs_bias_spec = lhs_factor_spec = lhs_qscale_spec = None
    if rhs_scale is not None:
        rhs_scale = rhs_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if rhs_bias is not None:
        rhs_bias = rhs_bias.astype(jnp.float32)
        rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if lhs_scale_factor is not None:
        lhs_factor_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if lhs_quant_scale is not None:
        lhs_quant_scale = lhs_quant_scale.astype(jnp.bfloat16)
        lhs_qscale_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
    acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n
    scratch_shapes = [
        pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
        pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
        ),
    ]

    num_lanes = pltpu.get_tpu_info().num_lanes
    if cfgs.zero_init:
        target_zero_ref_bytes = 2 * 1024 * 1024
        out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
        tile_zero_m = target_zero_ref_bytes // num_lanes // out_bytes
        tile_zero_m = min(tile_zero_m, dims.size_m)
        scratch_shapes += [
            pltpu.VMEM((tile_zero_m, num_lanes), cfgs.out_dtype),
            pltpu.SemaphoreType.DMA((1,)),
        ]
    else:
        scratch_shapes += [None, None]

    aligned_n = align_to(cfgs.out_size_n, num_lanes)
    out_init = jax.ShapeDtypeStruct((dims.size_m, aligned_n), cfgs.out_dtype)

    lhs_inputs = LhsRef(value=lhs,
                        scale_factor=lhs_scale_factor,
                        quant_scale=lhs_quant_scale)
    lhs_specs = LhsRef(
        value=pl.BlockSpec(memory_space=pltpu.HBM),
        scale_factor=lhs_factor_spec,
        quant_scale=lhs_qscale_spec,
    )
    rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)

    return pl.pallas_call(
        functools.partial(kernel_main, cfgs=cfgs),
        out_shape=out_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                lhs_specs,
                WeightsRef(
                    weight=pl.BlockSpec(memory_space=pltpu.HBM),
                    scale=rhs_scale_spec,
                    bias=rhs_bias_spec,
                ),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=get_scope_name(cfgs),
        cost_estimate=get_cost_estimate(cfgs),
        metadata=get_metadata(cfgs),
    )(group_sizes, group_offset, lhs_inputs, rhs_weights)[:, :cfgs.out_size_n]



