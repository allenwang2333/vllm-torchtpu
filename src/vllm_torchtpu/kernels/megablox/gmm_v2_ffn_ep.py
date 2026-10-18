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
"""gmm_v2_ffn with the expert-parallel combine's transport inside the kernel.

Under DP attention with EP, every rank runs its local experts over the whole
gathered batch. Instead of writing expert-sorted rows for a gather-reduce and
a reduce-scatter that run after the kernel, each finished output row is sent,
while the next experts' weights stream, to the rank that owns its token: into
that token's top-k slot of the owner's ``[tokens_per_rank * topk, n / 2]``
receive buffer, two bf16 values per u32 word (`pack_rows`). The owner sums a
token's slots after the kernel.

Must run under ``shard_map`` over a 1-D mesh axis whose index is the device id
the destination table uses. Every device of the axis must call it the same
number of times: it starts with a barrier over the axis.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import (
    FusedWeightsRef,
    MetadataRef,
    WeightsRef,
    fill_metadata,
    generate_block_specs,
    get_cost_estimate,
    get_metadata,
)
from vllm_torchtpu.kernels.megablox.gmm_v2_ffn import (
    _default_vmem_limit,
    _hbm_f32,
    _unpack,
    ffn_tile,
    make_ffn_configs,
)

COLLECTIVE_ID = 0
# A destination-table entry is `device << DEST_DEVICE_SHIFT | receive row`.
DEST_DEVICE_SHIFT = 24
_DEST_ROW_MASK = (1 << DEST_DEVICE_SHIFT) - 1
# Rows per semaphore wait when draining a count; the remainder waits singly.
_WAIT_CHUNK = 8
# pending_ref slots: remote sends per staging slot, local copies per staging
# slot, and whether this call has passed its barrier.
_PENDING_REMOTE, _PENDING_LOCAL, _BARRIER_DONE = 0, 2, 4

_HIGH_HALF = 0xFFFF0000


def pack_rows(y: jax.Array) -> jax.Array:
    """bf16 [m, n] -> u32 [m, n / 2]: word j holds column j low, j + n / 2 high.

    A bf16 row is half a sublane word, so a single row cannot be addressed by
    a DMA; a u32 row can. Contiguous halves keep this to shifts and masks.
    """
    half = y.shape[-1] // 2
    bits = lax.bitcast_convert_type(y.astype(jnp.float32), jnp.uint32)
    return (bits[:, :half] >> 16) | (bits[:, half:] & jnp.uint32(_HIGH_HALF))


def unpack_rows(p: jax.Array) -> jax.Array:
    """Inverse of `pack_rows`, as f32 holding the bf16 values exactly."""
    lo = lax.bitcast_convert_type(p << 16, jnp.float32)
    hi = lax.bitcast_convert_type(p & jnp.uint32(_HIGH_HALF), jnp.float32)
    return jnp.concatenate([lo, hi], axis=-1)


def _wait_rows(sem, rows_ref, num_rows):
    """Block until `num_rows` rows' worth of DMAs on `sem` have landed."""

    def wait(n: int):
        def body(_, carry):
            pltpu.make_async_copy(
                rows_ref.at[pl.ds(0, n)], rows_ref.at[pl.ds(0, n)], sem
            ).wait()
            return carry

        return body

    lax.fori_loop(0, num_rows // _WAIT_CHUNK, wait(_WAIT_CHUNK), 0)
    lax.fori_loop(0, num_rows % _WAIT_CHUNK, wait(1), 0)


def _barrier_signal(num_devices: int):
    barrier = pltpu.get_barrier_semaphore()
    for d in range(num_devices):
        pl.semaphore_signal(
            barrier, 1, device_id=(d,), device_id_type=pl.DeviceIdType.MESH
        )


def _barrier_wait(num_devices: int, second):
    """Completes the barrier `_barrier_signal` opened at entry.

    The second round, on the kernel-scratch semaphore `second`, stops a peer
    that already re-entered on the next call from standing in, with its next
    signal, for one still behind. It is scratch rather than run_scoped because
    a device with no rows reaches it after the pipeline, where a scoped
    semaphore need not have the address its peers signal from inside it.
    """
    pl.semaphore_wait(pltpu.get_barrier_semaphore(), num_devices)
    for d in range(num_devices):
        pl.semaphore_signal(
            second, 1, device_id=(d,), device_id_type=pl.DeviceIdType.MESH
        )
    pl.semaphore_wait(second, num_devices)


def ep_inner_kernel(
    # In
    tiled_lhs_ref,  # [tile_m // size_lhs_sublane, size_lhs_sublane, k]
    w13_ref,
    w2_ref,
    # Passed through
    dest_ref,  # SMEM int32[size_m]
    recv_ref,  # HBM u32[recv_rows, n / 2]
    staging_ref,  # VMEM u32[2, tile_m, n / 2]
    metadata_ref,
    pending_ref,  # SMEM int32[5]
    send_sems,  # DMA[2]
    local_sems,  # DMA[2]
    recv_sem,  # DMA
    barrier_sem,  # REGULAR: `_barrier_wait`'s second round
    *,
    up,
    down,
    num_devices: int,
    axis_name: str,
):
    """One gm tile through the FFN, then each of its group's rows to its slot."""
    gm_id = pl.program_id(1)
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = m_start - m_start % up.dims.size_lhs_sublane
    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset
    slot = gm_id % 2
    staging = staging_ref.at[slot]

    # The sends of two tiles ago read this staging slot; let them finish.
    _wait_rows(send_sems.at[slot], staging, pending_ref[_PENDING_REMOTE + slot])
    _wait_rows(local_sems.at[slot], staging, pending_ref[_PENDING_LOCAL + slot])

    def run_tile(bucket_m: int):
        acc = ffn_tile(
            tiled_lhs_ref, w13_ref, w2_ref, bucket_m=bucket_m, up=up, down=down
        )
        staging[:bucket_m] = pack_rows(acc.astype(down.out_dtype))

    branches = []
    for bucket_idx in range(up.tiles.tile_m // up.tiles.bucket_base):
        bucket_m = up.tiles.bucket_base * (bucket_idx + 1)
        branches.append(
            jax.named_scope(f"bm{bucket_m}")(functools.partial(run_tile, bucket_m))
        )
    jax.lax.switch(m_end_local // up.tiles.bucket_base, branches)

    # Peers' receive buffers are only theirs to write once they have entered.
    @pl.when(pending_ref[_BARRIER_DONE] == 0)
    def _():
        _barrier_wait(num_devices, barrier_sem)
        pending_ref[_BARRIER_DONE] = 1

    me = lax.axis_index(axis_name)

    def send_row(i, num_local):
        entry = dest_ref[m_offset + i]
        device = entry >> DEST_DEVICE_SHIFT
        dst = recv_ref.at[pl.ds(entry & _DEST_ROW_MASK, 1)]
        src = staging.at[pl.ds(i, 1)]
        is_local = device == me

        @pl.when(is_local)
        def _():
            pltpu.make_async_copy(src, dst, local_sems.at[slot]).start()

        @pl.when(jnp.logical_not(is_local))
        def _():
            pltpu.make_async_remote_copy(
                src,
                dst,
                send_sems.at[slot],
                recv_sem,
                device_id=(device,),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()

        return num_local + is_local.astype(jnp.int32)

    num_local = lax.fori_loop(m_start_local, m_end_local, send_row, jnp.int32(0))
    pending_ref[_PENDING_LOCAL + slot] = num_local
    pending_ref[_PENDING_REMOTE + slot] = m_end_local - m_start_local - num_local


def ep_kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref,  # int32[size_lhs_group]
    group_offset_ref,  # int32[1]
    dest_ref,  # int32[size_m]
    num_recv_ref,  # int32[1]: rows peers will send here
    # In
    lhs_ref,  # [size_m // size_lhs_sublane, size_lhs_sublane, k]
    w13_ref: WeightsRef,  # [size_group, k, 2 * i]
    w2_ref: WeightsRef,  # [size_group, i, n]
    # Out
    recv_ref,  # u32[recv_rows, n / 2]
    # Scratch
    staging_ref,
    metadata_ref: MetadataRef,
    pending_ref,
    send_sems,
    local_sems,
    recv_sem,
    barrier_sem,
    *,
    up,
    down,
    num_devices: int,
    axis_name: str,
):
    """Opens the barrier, pipelines both legs over the gm tiles, then drains."""
    _barrier_signal(num_devices)
    for i in range(pending_ref.shape[0]):
        pending_ref[i] = 0

    w13_ref = _unpack(w13_ref, up)
    w2_ref = _unpack(w2_ref, down)
    num_gm = fill_metadata(lhs_group_sizes_ref, group_offset_ref, metadata_ref, cfgs=up)

    (lhs_spec, w13_spec), _ = generate_block_specs(metadata_ref, up)
    (_, w2_spec), _ = generate_block_specs(metadata_ref, down)
    w13_up_ref = jax.tree.map(lambda x: x.at[..., up.out_size_n :], w13_ref)
    w13_ref = FusedWeightsRef(gate=w13_ref, up=w13_up_ref)
    w13_spec = FusedWeightsRef(gate=w13_spec, up=w13_spec)

    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(
            ep_inner_kernel,
            up=up,
            down=down,
            num_devices=num_devices,
            axis_name=axis_name,
        ),
        grid=(1, num_gm, 1),
        in_specs=(lhs_spec, w13_spec, w2_spec),
        out_specs=(),
    )
    pipeline_fn(
        lhs_ref,
        w13_ref,
        w2_ref,
        scratches=[
            dest_ref,
            recv_ref,
            staging_ref,
            metadata_ref,
            pending_ref,
            send_sems,
            local_sems,
            recv_sem,
            barrier_sem,
        ],
    )

    # A device with no local rows still owes its peers the barrier.
    @pl.when(pending_ref[_BARRIER_DONE] == 0)
    def _():
        _barrier_wait(num_devices, barrier_sem)

    for slot in range(2):
        staging = staging_ref.at[slot]
        _wait_rows(send_sems.at[slot], staging, pending_ref[_PENDING_REMOTE + slot])
        _wait_rows(local_sems.at[slot], staging, pending_ref[_PENDING_LOCAL + slot])
    _wait_rows(recv_sem, recv_ref, num_recv_ref[0])


@jax.jit(
    static_argnames=[
        "recv_rows",
        "num_devices",
        "axis_name",
        "fuse_act",
        "maybe_quantize_lhs",
        "rhs_quant_dtype",
        "preferred_element_type",
        "vmem_limit_bytes",
    ]
)
def gmm_v2_ffn_ep(
    lhs: jax.Array,  # [size_m, k]
    w1: jax.Array,  # [size_group, k, 2 * i]: gate then up along the last dim
    w2: jax.Array,  # [size_group, i, n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    dest: jax.Array,  # int32[size_m], see DEST_DEVICE_SHIFT; read for grouped rows
    num_recv: jax.Array,  # int32[1]: rows the other devices send to this one
    w1_scale: jax.Array | None = None,  # [size_group, num_blocks, 1, 2 * i]
    w2_scale: jax.Array | None = None,  # [size_group, num_blocks, 1, n]
    *,
    recv_rows: int,
    num_devices: int,
    axis_name: str,
    fuse_act: str,
    maybe_quantize_lhs: bool = True,
    rhs_quant_dtype: jnp.dtype | None = None,
    preferred_element_type: jnp.dtype | None = None,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
    """Each grouped row of `gmm_v2(act(gmm_v2(lhs, w1)), w2)`, packed, at its slot.

    Returns this device's u32[recv_rows, n / 2] receive buffer. Row r holds the
    `pack_rows` of the output row some device's `dest` addressed to (this
    device, r); rows nobody addressed are unset. `num_recv` must equal the
    number of rows addressed here by other devices, or the call never returns.
    """
    if vmem_limit_bytes is None:
        vmem_limit_bytes = _default_vmem_limit()
    cfgs = make_ffn_configs(
        lhs,
        w1,
        w1_scale,
        None,
        w2,
        w2_scale,
        None,
        group_sizes,
        fuse_act=fuse_act,
        maybe_quantize_lhs=maybe_quantize_lhs,
        rhs_quant_dtype=rhs_quant_dtype,
        out_dtype=preferred_element_type,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    if cfgs is None:
        raise ValueError(
            f"gmm_v2_ffn_ep cannot run {lhs.shape=} {w1.shape=} {w2.shape=}; "
            "check make_ffn_configs before calling it."
        )
    up, down = cfgs
    num_lanes = pltpu.get_tpu_info().num_lanes
    if down.out_dtype != jnp.bfloat16 or down.out_size_n % (2 * num_lanes):
        raise ValueError(
            f"gmm_v2_ffn_ep packs bf16 rows of a multiple of {2 * num_lanes} "
            f"lanes, not {down.out_dtype} x {down.out_size_n}"
        )
    if recv_rows > _DEST_ROW_MASK + 1:
        raise ValueError(f"{recv_rows=} does not fit the destination table")

    size_m = lhs.shape[0]
    padded_size_m = up.dims.size_m
    if padded_size_m != size_m:
        lhs = jnp.pad(lhs, ((0, padded_size_m - size_m), (0, 0)))
        dest = jnp.pad(dest, (0, padded_size_m - size_m))
    group_offset = jnp.zeros((1,), jnp.int32)

    w1_scale, w1_scale_spec = _hbm_f32(w1_scale)
    w2_scale, w2_scale_spec = _hbm_f32(w2_scale)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    half_n = down.out_size_n // 2
    max_num_gm = up.dims.size_group + pl.cdiv(padded_size_m, up.tiles.tile_m) - 1
    scratch_shapes = [
        pltpu.VMEM((2, up.tiles.tile_m, half_n), jnp.uint32),
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
        ),
        pltpu.SMEM((5,), jnp.int32),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.REGULAR,
    ]
    up_cost, down_cost = get_cost_estimate(up), get_cost_estimate(down)
    name = (
        f"gmm_v2_ffn_ep-g_{up.dims.size_group}-m_{padded_size_m}-k_{up.dims.size_k}"
        f"-i_{up.out_size_n}-n_{down.out_size_n}-act_{fuse_act}"
        f"-tm_{up.tiles.tile_m}-r_{recv_rows}"
    )
    metadata = {f"up.{k}": v for k, v in get_metadata(up).items()}
    metadata |= {f"down.{k}": v for k, v in get_metadata(down).items()}

    return pl.pallas_call(
        functools.partial(
            ep_kernel_main,
            up=up,
            down=down,
            num_devices=num_devices,
            axis_name=axis_name,
        ),
        out_shape=jax.ShapeDtypeStruct((recv_rows, half_n), jnp.uint32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[
                hbm,
                WeightsRef(weight=hbm, scale=w1_scale_spec, bias=None),
                WeightsRef(weight=hbm, scale=w2_scale_spec, bias=None),
            ],
            out_specs=hbm,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
            collective_id=COLLECTIVE_ID,
        ),
        name=name,
        cost_estimate=pl.CostEstimate(
            flops=up_cost.flops + down_cost.flops,
            bytes_accessed=up_cost.bytes_accessed
            + down_cost.bytes_accessed
            - 2 * padded_size_m * up.out_size_n * jnp.dtype(up.out_dtype).itemsize,
            transcendentals=0,
        ),
        metadata=metadata,
    )(
        group_sizes,
        group_offset,
        dest,
        num_recv,
        # Sublane-split outside the kernel: the same bytes, and no ref reshape.
        lhs.reshape(-1, up.dims.size_lhs_sublane, lhs.shape[-1]),
        WeightsRef(weight=w1, scale=w1_scale, bias=None),
        WeightsRef(weight=w2, scale=w2_scale, bias=None),
    )
