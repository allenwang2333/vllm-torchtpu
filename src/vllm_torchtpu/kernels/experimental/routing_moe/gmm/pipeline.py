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

"""Main kernel module for orchestrating DMA and computation pipeline."""

import functools

import jax.numpy as jnp
from jax import lax
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.routing_moe.common import layout_util
from vllm_torchtpu.kernels.experimental.routing_moe.gmm import (
  compute,
  config,
  memory_ref,
)


def inner_pipeline(
    x_ref: hijax.TransformedRef,
    w_ref: memory_ref.AbstractWeightRef,
    w2_ref: memory_ref.SingleWeightRef,
    gather_idx_ref: hijax.TransformedRef | None,
    scatter_idx_ref: hijax.TransformedRef | None,
    out_ref: hijax.TransformedRef,
    *,
    x_bref: memory_ref.CustomBufferedRef,
    out_bref: memory_ref.CustomBufferedRef,
    metadata_ref: memory_ref.MetadataRef,
    cfg: config.RoutingGmmConfig,
):
  """Inner pipeline that get executed per pipeline step."""

  del gather_idx_ref, scatter_idx_ref
  gm_id = pl.program_id(1)

  x_ref = pltpu.annotate(x_ref, no_hazard_no_deps=True)
  out_ref = pltpu.annotate(out_ref, no_hazard_no_deps=True)

  def _bucket_fn(bucket_m: int) -> None:
    # Step 1: Wait DMA for the current step slot.
    x_bref.branchless_wait_in(gm_id)
    out_bref.branchless_wait_out(gm_id - cfg.tile.num_xo_buffers)

    # Step 2: Start DMA for previous step slot.
    # NOTE: While we typically issue DMA for the other slots before waiting for
    # current slot, that will prevent DMA instructions to be pipelined with
    # compute.  Therefore, DMA is issued after the waiting.
    x_bref.branchless_copy_in(gm_id + cfg.tile.num_xo_buffers - 1)
    out_bref.branchless_copy_out(gm_id - 1)

    # Step 3: Load data from VMEM.
    x = layout_util.load_compact_to_large(
        x_ref.at[:bucket_m], cfg.tile.tile_w_k
    )
    w = w_ref.get_weight()

    # TODO(kyuyeunk): Potentially move this branching logic into matmul.
    w_scale = None
    if cfg.w.has_scale:
      w_scale = w_ref.get_scale()

    # Step 4: Perform computation.
    out = compute.matmul(cfg, x, w, w_scale, None)
    out = compute.apply_act_fn(out, cfg.fuse_act).astype(out_ref.dtype)

    if cfg.w2 is not None:
      # Step 4.1: Perform optional gate down compute.
      w2 = w2_ref.get_weight()
      w2_scale = None
      if cfg.w2.has_scale:
        w2_scale = w2_ref.get_scale()
      out = compute.matmul(cfg, out, w2, w2_scale, None)

    # Step 5: Store output into VMEM.
    layout_util.store_large_to_compact(out_ref.at[:bucket_m], out)

  # Select which bucket to choose based on number of active rows of given tile.
  num_rows = metadata_ref.get_num_rows(gm_id)
  bucket_idx = pl.cdiv(
      jnp.maximum(num_rows - cfg.tile.bucket_base, 0), cfg.tile.bucket_base
  )

  lax.switch(
      bucket_idx,
      [
          functools.partial(_bucket_fn, bucket_m)
          for bucket_m in cfg.tile.bucket_rows
      ],
  )


def outer_pipeline(
    metadata_ref: memory_ref.MetadataRef,
    x_hbm_ref: hijax.TransformedRef,
    w_hbm_ref: memory_ref.SingleWeightRef,
    w2_hbm_ref: memory_ref.SingleWeightRef,
    gather_idx_hbm_ref: hijax.TransformedRef | None,
    scatter_idx_hbm_ref: hijax.TransformedRef | None,
    out_hbm_ref: hijax.TransformedRef,
    *,
    cfg: config.RoutingGmmConfig,
):
  """Outer pipeline that handles memory and scheduling for inner pipeline."""

  num_gm = metadata_ref.fill_metadata_and_return_num_gm()

  x_buf, w_buf, w2_buf, gather_idx_buf, scatter_idx_buf, out_buf = (
      memory_ref.create_allocs(
          x_hbm_ref,
          w_hbm_ref,
          w2_hbm_ref,
          gather_idx_hbm_ref,
          scatter_idx_hbm_ref,
          out_hbm_ref,
          metadata_ref,
          cfg,
      )
  )
  x_spec, w_spec, w2_spec, gather_idx_spec, scatter_idx_spec, out_spec = (
      memory_ref.get_specs(
          x_buf, w_buf, w2_buf, gather_idx_buf, scatter_idx_buf, out_buf
      )
  )
  wrapped_w_hbm_ref, wrapped_w2_hbm_ref = memory_ref.transform_weight_ref(
      w_hbm_ref, w2_hbm_ref, cfg
  )

  @functools.partial(
      pl.run_scoped,
      final_allocs=(
          x_buf,
          w_buf,
          w2_buf,
          gather_idx_buf,
          scatter_idx_buf,
          out_buf,
      ),
  )
  def _run(final_allocs):
    (
        x_alloc,
        w_alloc,
        w2_alloc,
        gather_idx_alloc,
        scatter_idx_alloc,
        out_alloc,
    ) = final_allocs

    if gather_idx_alloc is not None:
      x_alloc = x_alloc.with_idx_ref(gather_idx_alloc.window_ref)
    if scatter_idx_alloc is not None:
      out_alloc = out_alloc.with_idx_ref(scatter_idx_alloc.window_ref)

    final_allocs = (
        x_alloc,
        w_alloc,
        w2_alloc,
        gather_idx_alloc,
        scatter_idx_alloc,
        out_alloc,
    )

    # TODO(kyuyeunk): Add support for N/K dim tiling.
    num_n = num_k = 1
    pipeline = pltpu.emit_pipeline(
        functools.partial(
            inner_pipeline,
            cfg=cfg,
            metadata_ref=metadata_ref,
            x_bref=x_alloc,
            out_bref=out_alloc,
        ),
        in_specs=(x_spec, w_spec, w2_spec, gather_idx_spec, scatter_idx_spec),
        out_specs=out_spec,
        grid=(num_n, num_gm, num_k),
    )

    # Prologue: Issue DMAs for first num_buffers - 1 number of steps.
    if cfg.x.mode != config.RoutingMode.DIRECT:
      for gm_id in range(cfg.tile.num_xo_buffers - 1):
        step = gm_id - (cfg.tile.num_xo_buffers - 1)
        slot = jnp.uint32(gm_id + 1)
        bref = gather_idx_alloc.with_slot_index(copy_in_slot=slot)
        bref.copy_in(gather_idx_hbm_ref, (0, step, 0))

      for gm_id in range(cfg.tile.num_xo_buffers - 1):
        step = gm_id - (cfg.tile.num_xo_buffers - 1)
        slot = jnp.uint32(gm_id + 1)
        bref = gather_idx_alloc.with_slot_index(wait_in_slot=slot)
        bref.wait_in(gather_idx_hbm_ref, (0, step, 0))
        x_alloc.branchless_copy_in(gm_id)

    pipeline(
        x_hbm_ref,
        wrapped_w_hbm_ref,
        wrapped_w2_hbm_ref,
        gather_idx_hbm_ref,
        scatter_idx_hbm_ref,
        out_hbm_ref,
        allocations=final_allocs,
    )

    # Epilogue: Copy out last executed tile.
    # NOTE: If output is written into VMEM at step i, output DMA is performed
    # at step i + 1. Therefore, output from last compute tile will not have been
    # issued output DMA. This copy_out ensures that the output from the last
    # step is correctly issued.
    if cfg.out.mode != config.RoutingMode.DIRECT:
      step = num_gm
      slot = jnp.mod(step, cfg.tile.num_xo_buffers).astype(jnp.uint32)
      bref = scatter_idx_alloc.with_slot_index(
          copy_in_slot=slot, wait_in_slot=slot
      )
      bref.copy_in(scatter_idx_hbm_ref, (0, step, 0))
      bref.wait_in(scatter_idx_hbm_ref, (0, step, 0))
    out_alloc.branchless_copy_out(num_gm - 1)

    # Perform DMA wiat from last num_buffers steps.
    for b in range(cfg.tile.num_xo_buffers):
      out_alloc.branchless_wait_out(num_gm - cfg.tile.num_xo_buffers + b)
