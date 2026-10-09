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
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.routing_moe.common import layout_util


class RoutingMode(enum.StrEnum):
  """Instructs how DMA for input and output should be done."""

  DIRECT = enum.auto()  # DMA is performed on continuous memory region.
  LOCAL = enum.auto()  # Local DMA is performed in per-row basis.
  # TODO(amandaliang): Support remove DMA.
  REMOTE = enum.auto()  # Remote DMA is performed in per-row basis.


@dataclasses.dataclass(frozen=True)
class WeightConfig:
  quant_dtype: jnp.dtype | None
  quant_block_size: int | None
  dtype: jnp.dtype
  has_bias: bool = False
  has_scale: bool = False


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
  size_out_m: int
  size_w_k: int
  size_w_n: int
  size_w2_k: int | None
  size_w2_n: int | None
  size_out_n: int
  size_local_groups: int
  size_global_groups: int


@dataclasses.dataclass(frozen=True)
class TileConfig:
  """Contains how input and output should be tiled."""

  tile_m: int
  tile_w_k: int
  tile_w_n: int
  tile_w2_k: int | None
  tile_w2_n: int | None
  tile_out_n: int
  bucket_base: int
  num_xo_buffers: int = 3
  num_w_buffers: int = 2

  @property
  def bucket_rows(self):
    bucket_base = self.bucket_base
    # Explicitly include tile_m in case tile_m % bucket_base != 0.
    return [*range(bucket_base, self.tile_m, bucket_base), self.tile_m]

  @property
  def idx_block_size(self) -> int:
    # Align idx block size to be multiple of lane size (128). Additionally,
    # add padding to avoid OOB access.
    lane_size = 128
    return (pl.cdiv(self.tile_m, lane_size) + 1) * lane_size

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
  w2: WeightConfig | None
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
      acc_dtype: jnp.dtype,
      w2: jax.Array | None = None,
      w2_scale: jax.Array | None = None,
  ):
    """Create routing gmm config object based on arguments."""

    del group_offset

    if tile is not None:
      raise NotImplementedError("Overriding tile config is not supported.")

    size_local_groups, size_w_k, size_w_n = w.shape
    if fuse_act is not None:
      size_w_n //= 2
    size_global_groups = group_sizes.size

    size_m = size_out_m = x.shape[0]

    # Step 1: Initialize static config objects.
    x_mode = out_mode = RoutingMode.DIRECT
    if gather_indices is not None and scatter_indices is not None:
      assert gather_indices.shape[0] == scatter_indices.shape[0]
    if gather_indices is not None:
      x_mode = RoutingMode.LOCAL
      size_out_m = gather_indices.shape[0]
    if scatter_indices is not None:
      out_mode = RoutingMode.LOCAL
      size_out_m = scatter_indices.shape[0]

    w_quant_block_size = None
    w_quant_dtype = None
    x_quant_dtype = None
    w_has_scale = False
    if w_scale is not None:
      w_quant_block_size = size_w_k
      w_quant_dtype = w.dtype
      w_has_scale = True

      # TODO(kyuyeunk): Support mixed precision.
      x_quant_dtype = w.dtype

    w_cfg = WeightConfig(
        quant_dtype=w_quant_dtype,
        dtype=w.dtype,
        quant_block_size=w_quant_block_size,
        has_scale=w_has_scale,
    )

    x_cfg = ActivationConfig(
        rows=size_m,
        cols=size_w_k,
        mode=x_mode,
        quant_dtype=x_quant_dtype,
        dtype=x.dtype,
        # TODO(kyuyeunk): Choose different block size based on HW.
        quant_block_size=512,
    )

    w2_cfg = size_w2_k = size_w2_n = None
    if w2 is None:
      size_out_n = size_w_n
    else:
      _, size_w2_k, size_w2_n = w2.shape
      size_out_n = size_w2_n

      w2_quant_block_size = None
      w2_quant_dtype = None
      w2_has_scale = False
      if w2_scale is not None:
        w2_has_scale = True
        w2_num_scales = w2_scale.shape[1]
        assert w2_num_scales == 1
        # Only support case where both up and down has same dtype.
        w2_quant_dtype = w2.dtype
        assert w2_quant_dtype == w_quant_dtype
      w2_cfg = WeightConfig(
          quant_dtype=w2_quant_dtype,
          dtype=w2.dtype,
          quant_block_size=w2_quant_block_size,
          has_scale=w2_has_scale,
      )

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
        size_w_k=size_w_k,
        size_w_n=size_w_n,
        size_w2_k=size_w2_k,
        size_w2_n=size_w2_n,
        size_out_m=size_out_m,
        size_out_n=size_out_n,
        size_local_groups=size_local_groups,
        size_global_groups=size_global_groups,
    )

    tile = calculate_tile_size(x_cfg, w_cfg, w2_cfg, out_cfg, dim)

    cfg = cls(
        tpu_info=pltpu.get_tpu_info(),
        fuse_act=fuse_act,
        dim=dim,
        tile=tile,
        x=x_cfg,
        w=w_cfg,
        w2=w2_cfg,
        out=out_cfg,
        acc_dtype=acc_dtype,
    )

    # Check few things before exiting.
    if x.shape != cfg.x.compact_shape:
      raise ValueError(
          f"Input x must have shae of {cfg.x.compact_shape}. Got {x.shape}."
      )

    if group_sizes.size < cfg.dim.size_local_groups:
      raise ValueError(
          f"group_sizes must be {cfg.dim.size_local_groups}. Got"
          f" {group_sizes.shape}."
      )

    return cfg

  @property
  def max_num_gm(self) -> int:
    """Returns maximum number of possible tiles including OOB safety padding."""
    max_num_single_group = pl.cdiv(self.dim.size_out_m, self.tile.tile_m)
    # Padd left and right by number of buffers to avoid OOB access.
    safety_padding = self.tile.num_max_buffers * 3
    return self.dim.size_local_groups + max_num_single_group + safety_padding

  @property
  def kernel_name(self):
    name = f"routing_gmm-{self.x.mode}-{self.out.mode}"
    if self.w2 is not None:
      # Denote Up & Down.
      name += "-ud"
    name += f"-g{self.dim.size_local_groups}-m{self.dim.size_out_m}"
    name += f"-k{self.dim.size_w_k}-n{self.dim.size_out_n}"
    return name


def calculate_tile_size(
    x: ActivationConfig,
    w: WeightConfig,
    w2: WeightConfig | None,
    out: ActivationConfig,
    dim: Dimension,
) -> TileConfig:
  """Utilize heuristic to calculate tile size for given inputs."""

  del x, w, w2, out

  # TODO(kyuyeunk): Add more sophisticated tile selection logic.
  tile_m = min(256, dim.size_out_m)
  bucket_base = min(128, tile_m)

  # TODO(kyuyeunk): Support N/K dim tiling.
  tile_w_k = dim.size_w_k

  tile_w_n = dim.size_w_n
  tile_w2_k = dim.size_w2_k
  tile_w2_n = dim.size_w2_n
  tile_out_n = dim.size_out_n

  return TileConfig(
      tile_m=tile_m,
      tile_w_k=tile_w_k,
      tile_w_n=tile_w_n,
      tile_w2_k=tile_w2_k,
      tile_w2_n=tile_w2_n,
      tile_out_n=tile_out_n,
      bucket_base=bucket_base,
  )
