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
# ==============================================================================

"""Module for static config object."""

import dataclasses
import enum

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from vllm_torchtpu.kernels.routing_moe.common import layout_util


class RoutingMode(enum.StrEnum):
  """Instructs how DMA for input and output should be done."""

  DIRECT = enum.auto()  # DMA is performed on continuous memory region.
  LOCAL = enum.auto()  # Local DMA is performed in per-row basis.
  # TODO(amandaliang): Support remove DMA.
  REMOTE = enum.auto()  # Remote DMA is performed in per-row basis.


def get_packing_factor(
    storage_dtype: jnp.dtype, quant_dtype: jnp.dtype | None
) -> int:
  """Number of quant_dtype values packed in one storage_dtype element."""
  if quant_dtype is None or quant_dtype == storage_dtype:
    return 1
  storage_bits = jax.dtypes.itemsize_bits(storage_dtype)
  quant_bits = jax.dtypes.itemsize_bits(quant_dtype)
  packing, remainder = divmod(storage_bits, quant_bits)
  if packing < 1 or remainder:
    raise ValueError(
        f"Storage dtype {storage_dtype} cannot pack quant dtype {quant_dtype}."
    )
  return packing


@dataclasses.dataclass(frozen=True)
class WeightConfig:
  """Contains how weights are stored and dequantized.

  dtype is the storage dtype. Sub-byte weights (e.g. float4_e2m1fn) may be
  stored packed along k in a wider carrier such as uint8: with b the quant
  dtype's bit width, the value at row k sits in bits
  [b * (k % packing), b * (k % packing + 1)) of storage row k // packing, so
  the lowest bits hold the even row.
  """

  quant_dtype: jnp.dtype | None
  quant_block_size: int | None
  dtype: jnp.dtype
  has_bias: bool = False
  has_scale: bool = False

  @property
  def packing(self) -> int:
    return get_packing_factor(self.dtype, self.quant_dtype)

  @property
  def should_unpack(self) -> bool:
    return self.packing > 1


@dataclasses.dataclass(frozen=True)
class ActivationConfig:
  """Contains how activation should be processed."""

  rows: int
  cols: int
  dtype: jnp.dtype
  quant_dtype: jnp.dtype | None
  quant_block_size: int | None
  mode: RoutingMode

  @property
  def should_quantize(self) -> bool:
    if self.quant_dtype is None:
      return False
    return self.quant_dtype != self.dtype

  @property
  def compact_shape(self) -> tuple[int, int, int, int]:
    return layout_util.to_compact_shape((self.rows, self.cols), self.dtype)


@dataclasses.dataclass(frozen=True)
class Dimension:
  size_m: int
  size_k: int
  size_n: int
  size_out_m: int
  size_out_n: int
  size_local_groups: int
  size_global_groups: int


@dataclasses.dataclass(frozen=True)
class TileConfig:
  """Contains how input and output should be tiled."""

  tile_m: int
  tile_k: int
  tile_n: int
  bucket_base: int
  num_xo_buffers: int = 3
  num_w_buffers: int = 2

  @property
  def bucket_rows(self):
    bucket_base = self.bucket_base
    return [b for b in range(bucket_base, self.tile_m + 1, bucket_base)]

  @property
  def num_max_buffers(self):
    return max(self.num_xo_buffers, self.num_w_buffers)


@dataclasses.dataclass(frozen=True)
class RoutingGmmConfig:
  """Stores compile time static information of routing gmm."""

  dim: Dimension
  tile: TileConfig
  x: ActivationConfig
  w: WeightConfig
  out: ActivationConfig
  tpu_info: pltpu.TpuInfo
  fuse_act: str | None
  acc_dtype: jnp.dtype

  @classmethod
  def create(
      cls,
      x: jax.Array,
      w: jax.Array,
      w_scale: jax.Array | None,
      group_sizes: jax.Array,
      group_offset: jax.Array | None,
      gather_indices: jax.Array | None,
      scatter_indices: jax.Array | None,
      tile: TileConfig | None,
      fuse_act: str | None,
      w_quant_dtype: jnp.dtype | None = None,
  ):
    """Create routing gmm config object based on arguments."""

    del group_offset

    if tile is not None:
      raise NotImplementedError("Overriding tile config is not supported.")

    size_local_groups, size_k, size_n = w.shape
    size_global_groups = group_sizes.size
    if w_quant_dtype is None and w_scale is not None:
      w_quant_dtype = w.dtype
    size_k *= get_packing_factor(w.dtype, w_quant_dtype)

    size_m = size_out_m = x.shape[0]

    # Step 1: Initialize static config objects.
    x_mode = out_mode = RoutingMode.DIRECT
    if gather_indices is not None and scatter_indices is not None:
      raise NotImplementedError()
    elif gather_indices is not None:
      x_mode = RoutingMode.LOCAL
      size_out_m = gather_indices.shape[0]
    elif scatter_indices is not None:
      out_mode = RoutingMode.LOCAL
      size_out_m = scatter_indices.shape[0]

    w_quant_block_size = None
    x_quant_dtype = None
    # TODO(kyuyeunk): Choose different block size based on HW.
    x_quant_block_size = 512
    has_scale = False
    if w_scale is not None:
      num_scales = w_scale.shape[1]
      if size_k % num_scales:
        raise ValueError(
            f"k={size_k} must be divisible by the {num_scales} scale blocks."
        )
      w_quant_block_size = size_k // num_scales
      has_scale = True
      assert w_quant_dtype is not None

      # Activations are quantized to 8 bits whatever the weight width, as in
      # gmm_v2: e4m3 for floating weights (A8W4 for float4_e2m1fn, which the
      # MXU only takes widened), int8 for integer ones.
      if jnp.issubdtype(w_quant_dtype, jnp.floating):
        x_quant_dtype = jnp.dtype(jnp.float8_e4m3fn)
      else:
        x_quant_dtype = jnp.dtype(jnp.int8)

      # The matmul steps over k in min(x, w) block chunks, so each chunk must
      # see a single activation and a single weight scale. A block spanning
      # all of k (e.g. per-channel weights) has no inner boundary to respect.
      k_step = min(x_quant_block_size, w_quant_block_size)
      for block_size in (x_quant_block_size, w_quant_block_size):
        if block_size < size_k and block_size % k_step:
          raise ValueError(
              f"Weight block size {w_quant_block_size} and activation block"
              f" size {x_quant_block_size} must divide each other."
          )
    elif w_quant_dtype is not None:
      raise ValueError(f"Weights quantized to {w_quant_dtype} need a w_scale.")

    w_cfg = WeightConfig(
        quant_dtype=w_quant_dtype,
        dtype=w.dtype,
        quant_block_size=w_quant_block_size,
        has_scale=has_scale,
    )

    x_cfg = ActivationConfig(
        rows=size_m,
        cols=size_k,
        mode=x_mode,
        quant_dtype=x_quant_dtype,
        dtype=x.dtype,
        quant_block_size=x_quant_block_size,
    )

    size_out_n = size_n
    if fuse_act is not None:
      size_out_n //= 2
    out_cfg = ActivationConfig(
        rows=size_out_m,
        cols=size_out_n,
        mode=out_mode,
        dtype=x.dtype,
        quant_dtype=None,
        quant_block_size=None,
    )

    dim = Dimension(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        size_out_m=size_out_m,
        size_out_n=size_out_n,
        size_local_groups=size_local_groups,
        size_global_groups=size_global_groups,
    )

    tpu_info = pltpu.get_tpu_info()
    vmem_limit_bytes = int(VMEM_FRACTION * tpu_info.vmem_capacity_bytes)
    tile = calculate_tile_size(x_cfg, w_cfg, out_cfg, dim, vmem_limit_bytes)

    return cls(
        tpu_info=tpu_info,
        fuse_act=fuse_act,
        dim=dim,
        tile=tile,
        x=x_cfg,
        w=w_cfg,
        out=out_cfg,
        acc_dtype=jnp.bfloat16.dtype,
    )

  @property
  def max_num_gm(self) -> int:
    """Returns maximun number of possible tiles including OOB safety padding."""
    max_num_single_group = pl.cdiv(self.dim.size_out_m, self.tile.tile_m)
    # Padd left and right by number of buffers to avoid OOB access.
    safety_padding = self.tile.num_max_buffers * 2
    return self.dim.size_local_groups + max_num_single_group + safety_padding

  @property
  def max_routed_rows(self) -> int:
    """Returns maximum number of supported rows based on SMEM capacity."""
    return int(0.45 * self.tpu_info.smem_capacity_bytes) // 4

  @property
  def kernel_name(self):
    name = f"routing_gmm-{self.x.mode}-{self.out.mode}"
    name += f"-g{self.dim.size_local_groups}-m{self.dim.size_out_m}"
    name += f"-k{self.dim.size_k}-n{self.dim.size_out_n}"
    return name

  def validate(self, x, w, w_scale, group_sizes):
    """Validate if arguments are correct."""

    # TODO(kyuyeunk): Maybe perform this while config creation phase?

    if x.shape != self.x.compact_shape:
      raise ValueError(
          f"Input x must have shae of {self.x.compact_shape}. Got {x.shape}."
      )
    w_shape = (
        self.dim.size_local_groups,
        self.dim.size_k // self.w.packing,
        self.dim.size_n,
    )
    if w.shape != w_shape:
      raise ValueError(f"Weight shape must be {w_shape}. Got {w.shape}.")

    if group_sizes.size < self.dim.size_local_groups:
      raise ValueError(
          f"group_sizes must be {self.dim.size_local_groups}. Got"
          f" {group_sizes.shape}."
      )
    if w_scale is not None:
      assert self.w.quant_block_size is not None
      num_blocks = self.dim.size_k // self.w.quant_block_size
      scale_shape = (self.dim.size_local_groups, num_blocks, 1, self.dim.size_n)
      if w_scale.shape != scale_shape:
        raise ValueError(
            f"w_scale must have shape {scale_shape}. Got {w_scale.shape}."
        )
    if not 0 < self.dim.size_out_m <= self.max_routed_rows:
      raise ValueError(
          f"Output rows must be less than {self.max_routed_rows} to avoid SMEM "
          f"OOM. Got {self.dim.size_out_m}."
      )


# Share of VMEM the estimate below may fill; the rest covers what it does not
# count (Mosaic scratch, spills, the widened fp4/fp8 blocks).
VMEM_FRACTION = 0.95
# x/out row buffers: three let the gather run two steps ahead of compute, two
# only one. Three are used unless the resident weights leave no room (e.g.
# DeepSeek-V4-Pro's 7168 x 6144 fp4 gate/up slab at tile_m=256).
MAX_NUM_XO_BUFFERS = 3
MIN_NUM_XO_BUFFERS = 2


def estimate_vmem_bytes(
    *,
    tile_m: int,
    size_k: int,
    size_n: int,
    size_out_n: int,
    w_bits: int,
    num_scale_blocks: int,
    num_xo_buffers: int,
    num_w_buffers: int = TileConfig.num_w_buffers,
) -> int:
  """Upper bound on one routing gmm call's VMEM.

  With no N/K tiling the whole k x n expert slab stays resident in
  num_w_buffers, next to num_xo_buffers bf16 row tiles of x and out. Block
  scales stay resident as f32 [num_blocks, n] (sublane padded), and the matmul
  intermediates are bounded by an f32 tile_m x n accumulator.
  """
  w_bytes = num_w_buffers * size_k * size_n * w_bits // 8
  scale_bytes = num_w_buffers * pl.cdiv(num_scale_blocks, 8) * 8 * size_n * 4
  xo_bytes = num_xo_buffers * tile_m * (size_k + size_out_n) * 2
  acc_bytes = tile_m * size_n * 4
  return w_bytes + scale_bytes + xo_bytes + acc_bytes


def calculate_tile_size(
    x: ActivationConfig,
    w: WeightConfig,
    out: ActivationConfig,
    dim: Dimension,
    vmem_limit_bytes: int,
) -> TileConfig:
  """Utilize heuristic to calculate tile size for given inputs."""

  del x, out

  # TODO(kyuyeunk): Add more sophisticated tile selection logic.
  tile_m = min(256, dim.size_out_m)
  bucket_base = min(128, tile_m)

  # TODO(kyuyeunk): Support N/K dim tiling.
  tile_n = dim.size_out_n
  tile_k = dim.size_k

  num_scale_blocks = 0
  if w.has_scale:
    assert w.quant_block_size is not None
    num_scale_blocks = dim.size_k // w.quant_block_size
  max_xo_bytes = estimate_vmem_bytes(
      tile_m=tile_m,
      size_k=dim.size_k,
      size_n=dim.size_n,
      size_out_n=dim.size_out_n,
      w_bits=jax.dtypes.itemsize_bits(w.quant_dtype or w.dtype),
      num_scale_blocks=num_scale_blocks,
      num_xo_buffers=MAX_NUM_XO_BUFFERS,
  )
  num_xo_buffers = MAX_NUM_XO_BUFFERS
  if max_xo_bytes > vmem_limit_bytes:
    num_xo_buffers = MIN_NUM_XO_BUFFERS

  return TileConfig(
      tile_m=tile_m,
      tile_k=tile_k,
      tile_n=tile_n,
      bucket_base=bucket_base,
      num_xo_buffers=num_xo_buffers,
  )
