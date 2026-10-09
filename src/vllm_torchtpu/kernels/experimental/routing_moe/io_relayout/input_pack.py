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

"""Input relayout helper from large to compact layout."""

import jax
import jax.numpy as jnp
from jax.experimental import hijax, layout
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.routing_moe.common import layout_util


@jax.jit(static_argnames=("tile_size",))
def input_pack(x: jax.Array, tile_size: int = 256) -> jax.Array:
  tpu_info = pltpu.get_tpu_info()
  num_lanes = tpu_info.num_lanes
  x_dtype = x.dtype
  packing = 4 // x_dtype.itemsize
  batch_size, dim_size = x.shape
  num_chunks = dim_size // num_lanes
  num_packed_chunks = num_chunks // packing
  num_buffers = 2

  tile_size = min(batch_size, tile_size)
  num_tiles = batch_size // tile_size
  num_buffers = min(num_tiles, num_buffers)

  def outer_kernel(
      in_hbm_ref: hijax.TransformedRef,
      out_hbm_ref: hijax.TransformedRef,
      in_scratch_ref: hijax.TransformedRef,
      out_scratch_ref: hijax.TransformedRef,
      send_sem_ref: hijax.TransformedRef,
      recv_sem_ref: hijax.TransformedRef,
  ):

    for b in range(num_buffers):
      start = b * tile_size
      end = start + tile_size
      pltpu.make_async_copy(
          in_hbm_ref.at[start:end],
          in_scratch_ref.at[b],
          recv_sem_ref.at[b],
      ).start()

    @pl.loop(0, num_tiles)
    def inner_kernel(p_id: jax.Array):
      slot = p_id % num_buffers

      # Step 1: Wait in
      pltpu.make_async_copy(
          in_scratch_ref.at[0],
          in_scratch_ref.at[0],
          recv_sem_ref.at[slot],
      ).wait()

      # Step 2: Wait out
      @pl.when(num_buffers <= p_id)
      def b():
        pltpu.make_async_copy(
            out_scratch_ref.at[0, :, :num_packed_chunks],
            out_scratch_ref.at[0, :, :num_packed_chunks],
            send_sem_ref.at[slot],
        ).wait()

      # Step 3: relayout
      in_slot_ref = in_scratch_ref.at[slot]
      out_slot_ref = out_scratch_ref.at[slot]
      out_slot_u32_ref = out_slot_ref.bitcast(jnp.uint32)
      num_padded_chunks = out_slot_u32_ref.shape[1]
      out_slot_u32_ref = out_slot_u32_ref.reshape(-1, num_lanes)

      val = in_slot_ref[...]
      if packing != 1:
        assert x_dtype == jnp.bfloat16
        unpacked = val.astype(jnp.float32)
        sliced_unpacked = jnp.split(unpacked, num_chunks, axis=-1)
        even_chunks = sliced_unpacked[0::2]
        odd_chunks = sliced_unpacked[1::2]

        for c in range(num_packed_chunks):
          chunk = pltpu.pack_elementwise(
              [even_chunks[c], odd_chunks[c]], packed_dtype=jnp.bfloat16
          )
          chunk = pltpu.bitcast(chunk, jnp.uint32)
          out_slot_u32_ref[c::num_padded_chunks] = chunk
      else:
        chunks = jnp.split(val, num_chunks, axis=-1)
        for c in range(num_packed_chunks):
          chunk = pltpu.bitcast(chunks[c], jnp.uint32)
          out_slot_u32_ref[c::num_padded_chunks] = chunk
      start = p_id * tile_size

      # Step 4: Copy out
      pltpu.make_async_copy(
          out_scratch_ref.at[slot, :, :num_packed_chunks],
          out_hbm_ref.at[pl.ds(start, tile_size)],
          send_sem_ref.at[slot],
      ).start(priority=1)

      # Step 5: Copy in
      next_p_id = p_id + num_buffers

      @pl.when(next_p_id < num_tiles)
      def a():
        next_start = next_p_id * tile_size
        pltpu.make_async_copy(
            in_hbm_ref.at[pl.ds(next_start, tile_size)],
            in_scratch_ref.at[slot],
            recv_sem_ref.at[slot],
        ).start()

    for i in range(num_tiles - num_buffers, num_tiles):
      b = i % num_buffers
      pltpu.make_async_copy(
          out_scratch_ref.at[0, :, :num_packed_chunks],
          out_scratch_ref.at[0, :, :num_packed_chunks],
          send_sem_ref.at[b],
      ).wait()

  # Probably can be removed?
  sublane_size = tpu_info.get_sublane_tiling(x_dtype)
  custom_dll = layout.Layout(
      major_to_minor=(0, 1), tiling=((sublane_size, 128), (packing, 1))
  )
  x = layout.with_layout_constraint(x, custom_dll)
  x_shape = x.shape
  assert len(x_shape) == 2
  out_shape = layout_util.to_compact_shape(x_shape, x.dtype)
  vmem_chunks = layout_util.to_bank_conflict_aware_stride(out_shape[1])

  return pl.kernel(
      outer_kernel,
      out_type=jax.ShapeDtypeStruct(out_shape, x_dtype),
      mesh=layout_util.create_tensorcore_mesh(axis_name="core", num_cores=1),
      scratch_types=(
          pltpu.VMEM((num_buffers, tile_size, dim_size), x_dtype),
          pltpu.VMEM(
              (num_buffers, tile_size, vmem_chunks, packing, num_lanes),
              x_dtype,
          ),
          pltpu.SemaphoreType.DMA((num_buffers,)),
          pltpu.SemaphoreType.DMA((num_buffers,)),
      ),
      name="reshape_kernel",
  )(x)
