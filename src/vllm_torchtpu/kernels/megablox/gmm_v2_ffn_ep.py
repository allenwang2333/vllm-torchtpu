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
token's slots after the kernel. The input side is pushed the same way: each
device sends its own `pack_rows`-packed token rows straight into the
expert-sorted lhs rows of the devices whose experts they route to, one local
expert index at a time, and a gm tile waits only on its own expert's receive
semaphore, so no gathered copy of the batch exists before the kernel.

Must run under ``shard_map`` over a 1-D mesh axis whose index is the device id
the destination table uses. Every device of the axis must call it the same
number of times: it starts with a barrier over the axis.
"""

import dataclasses
import functools
import math

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
# Local experts past the next tile's whose token rows a device pushes before
# waiting on that tile's own rows.
DISPATCH_LOOKAHEAD = 2
# Rows per DMA when copying a gm tile's received rows into VMEM (a tile_m
# that is not a multiple takes one tile_m-row DMA).
FETCH_CHUNK_ROWS = 16
# pending_ref slots: remote sends per staging slot, local copies per staging
# slot, the number of gm tiles, the local experts whose rows this device has
# pushed, and how many of those pushes went to other devices.
_PENDING_REMOTE, _PENDING_LOCAL, _NUM_GM, _PUSHED_GROUPS, _PUSHED_REMOTE = (
    0,
    2,
    4,
    5,
    6,
)
_NUM_PENDING = 7

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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DispatchRefs:
    """What the dispatch side of the kernel reads and signals."""

    x: jax.Array  # HBM u32[num_tokens, k / 2]: `pack_rows` of this device's tokens
    lhs: jax.Array  # HBM u32[size_m + chunk rows, k / 2]: rows pushed here
    group_sizes: jax.Array  # SMEM int32[size_group]: rows pushed to each expert
    send_dest: jax.Array  # SMEM int32[num_routes]: device << SHIFT | lhs row
    send_src: jax.Array  # SMEM int32[num_routes]: row of x each route sends
    send_bounds: jax.Array  # SMEM int32[size_group + 1]
    send_sem: jax.Array  # DMA: this device's remote pushes
    recv_sems: jax.Array  # DMA[size_group]: rows landed per local expert


def _push_groups(group_end, refs: DispatchRefs, pending_ref, me):
    """Pushes this device's rows for local expert indices up to `group_end`.

    Routes `send_bounds[g]:send_bounds[g + 1]` go to local expert g of their
    device, and they land on that device's `recv_sems[g]` (local copies too).
    """
    size_group = refs.recv_sems.shape[0]
    start = pending_ref[_PUSHED_GROUPS]
    end = jnp.maximum(start, jnp.minimum(group_end, size_group))

    def push_group(g, num_remote):
        def push_row(i, num_remote):
            entry = refs.send_dest[i]
            device = entry >> DEST_DEVICE_SHIFT
            src = refs.x.at[pl.ds(refs.send_src[i], 1)]
            dst = refs.lhs.at[pl.ds(entry & _DEST_ROW_MASK, 1)]
            is_local = device == me

            @pl.when(is_local)
            def _():
                pltpu.make_async_copy(src, dst, refs.recv_sems.at[g]).start()

            @pl.when(jnp.logical_not(is_local))
            def _():
                pltpu.make_async_remote_copy(
                    src,
                    dst,
                    refs.send_sem,
                    refs.recv_sems.at[g],
                    device_id=(device,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

            return num_remote + jnp.logical_not(is_local).astype(jnp.int32)

        return lax.fori_loop(
            refs.send_bounds[g], refs.send_bounds[g + 1], push_row, num_remote
        )

    pending_ref[_PUSHED_REMOTE] = lax.fori_loop(
        start, end, push_group, pending_ref[_PUSHED_REMOTE]
    )
    pending_ref[_PUSHED_GROUPS] = end


def _prepare_tile(
    gm_id, refs: DispatchRefs, lhs_staging, sem, metadata_ref, pending_ref, me, sublane
):
    """Pushes ahead of gm tile `gm_id`, waits for its expert's rows if it is
    the expert's first tile, and starts copying its rows into `lhs_staging`.

    Every device pushes an expert index's rows before it waits on that index,
    so the lowest index anyone waits on has been pushed by all.
    """
    group = metadata_ref.gm_id_to_group_id[gm_id]
    _push_groups(group + 1 + DISPATCH_LOOKAHEAD, refs, pending_ref, me)
    previous = metadata_ref.gm_id_to_group_id[jnp.maximum(gm_id - 1, 0)]

    @pl.when(jnp.logical_or(gm_id == 0, previous != group))
    def _():
        _wait_rows(refs.recv_sems.at[group], refs.lhs, refs.group_sizes[group])

    _tile_row_copies(gm_id, refs.lhs, lhs_staging, sem, metadata_ref, sublane, True)


def _fetch_chunk_rows(tile_m: int) -> int:
    return FETCH_CHUNK_ROWS if tile_m % FETCH_CHUNK_ROWS == 0 else tile_m


def _tile_row_copies(gm_id, lhs_ref, lhs_staging, sem, metadata_ref, sublane, start):
    """Starts (or waits on) the copies of gm tile `gm_id`'s rows into VMEM.

    The last chunk may run past the tile into the next expert's rows while
    they are still landing; the tile never uses those rows.
    """
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = m_start - m_start % sublane
    rows = _fetch_chunk_rows(lhs_staging.shape[0])

    def chunk(c, carry):
        row = pl.multiple_of(m_offset + c * rows, math.gcd(sublane, rows))
        copy = pltpu.make_async_copy(
            lhs_ref.at[pl.ds(row, rows)],
            lhs_staging.at[pl.ds(c * rows, rows)],
            sem,
        )
        if start:
            copy.start()
        else:
            copy.wait()
        return carry

    lax.fori_loop(0, pl.cdiv(m_end - m_offset, rows), chunk, 0)


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
    signal, for one still behind. Peers signal it remotely, so it is kernel
    scratch, which has the same address on every device, not run_scoped.
    """
    pl.semaphore_wait(pltpu.get_barrier_semaphore(), num_devices)
    for d in range(num_devices):
        pl.semaphore_signal(
            second, 1, device_id=(d,), device_id_type=pl.DeviceIdType.MESH
        )
    pl.semaphore_wait(second, num_devices)


def ep_inner_kernel(
    # In
    w13_ref,
    w2_ref,
    # Passed through
    dispatch: DispatchRefs,
    lhs_staging_ref,  # VMEM u32[2, tile_m, k / 2]
    fetch_sems,  # DMA[2]
    dest_ref,  # SMEM int32[size_m]
    recv_ref,  # HBM u32[recv_rows, n / 2]
    staging_ref,  # VMEM u32[2, tile_m, n / 2]
    metadata_ref,
    pending_ref,  # SMEM int32[_NUM_PENDING]
    send_sems,  # DMA[2]
    local_sems,  # DMA[2]
    recv_sem,  # DMA
    *,
    up,
    down,
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
    lhs_staging = lhs_staging_ref.at[slot]
    me = lax.axis_index(axis_name)
    prepare = functools.partial(
        _prepare_tile,
        refs=dispatch,
        metadata_ref=metadata_ref,
        pending_ref=pending_ref,
        me=me,
        sublane=up.dims.size_lhs_sublane,
    )

    @pl.when(gm_id == 0)
    def _():
        prepare(0, lhs_staging=lhs_staging, sem=fetch_sems.at[slot])

    # The next tile's rows land while this one computes; its slot was read by
    # the tile before this one, which has finished.
    @pl.when(gm_id + 1 < pending_ref[_NUM_GM])
    def _():
        prepare(
            gm_id + 1,
            lhs_staging=lhs_staging_ref.at[1 - slot],
            sem=fetch_sems.at[1 - slot],
        )

    _tile_row_copies(
        gm_id,
        dispatch.lhs,
        lhs_staging,
        fetch_sems.at[slot],
        metadata_ref,
        up.dims.size_lhs_sublane,
        False,
    )

    # The sends of two tiles ago read this staging slot; let them finish.
    _wait_rows(send_sems.at[slot], staging, pending_ref[_PENDING_REMOTE + slot])
    _wait_rows(local_sems.at[slot], staging, pending_ref[_PENDING_LOCAL + slot])

    def run_tile(bucket_m: int):
        lhs = unpack_rows(lhs_staging[:bucket_m]).astype(up.lhs_cfgs.dtype)
        acc = ffn_tile(lhs, w13_ref, w2_ref, bucket_m=bucket_m, up=up, down=down)
        staging[:bucket_m] = pack_rows(acc.astype(down.out_dtype))

    branches = []
    for bucket_idx in range(up.tiles.tile_m // up.tiles.bucket_base):
        bucket_m = up.tiles.bucket_base * (bucket_idx + 1)
        branches.append(
            jax.named_scope(f"bm{bucket_m}")(functools.partial(run_tile, bucket_m))
        )
    jax.lax.switch(m_end_local // up.tiles.bucket_base, branches)

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
    lhs_group_sizes_ref,  # int32[size_lhs_group]: rows pushed to each local expert
    group_offset_ref,  # int32[1]
    dest_ref,  # int32[size_m]
    num_recv_ref,  # int32[1]: rows peers will send here
    send_dest_ref,  # int32[num_routes]
    send_src_ref,  # int32[num_routes]
    send_bounds_ref,  # int32[size_lhs_group + 1]
    # In
    x_ref,  # HBM u32[num_tokens, k / 2]: `pack_rows` of this device's tokens
    w13_ref: WeightsRef,  # [size_group, k, 2 * i]
    w2_ref: WeightsRef,  # [size_group, i, n]
    # Out
    recv_ref,  # u32[recv_rows, n / 2]
    lhs_ref,  # u32[size_m + chunk rows, k / 2]: the expert-sorted rows
    # Scratch
    lhs_staging_ref,
    fetch_sems,
    staging_ref,
    metadata_ref: MetadataRef,
    pending_ref,
    send_sems,
    local_sems,
    recv_sem,
    barrier_sem,
    push_send_sem,
    push_recv_sems,
    *,
    up,
    down,
    num_devices: int,
    axis_name: str,
):
    """Opens the barrier, pushes, pipelines both legs over the gm tiles, then drains."""
    _barrier_signal(num_devices)
    for i in range(pending_ref.shape[0]):
        pending_ref[i] = 0

    w13_ref = _unpack(w13_ref, up)
    w2_ref = _unpack(w2_ref, down)
    num_gm = fill_metadata(lhs_group_sizes_ref, group_offset_ref, metadata_ref, cfgs=up)
    pending_ref[_NUM_GM] = num_gm
    dispatch = DispatchRefs(
        x=x_ref,
        lhs=lhs_ref,
        group_sizes=lhs_group_sizes_ref,
        send_dest=send_dest_ref,
        send_src=send_src_ref,
        send_bounds=send_bounds_ref,
        send_sem=push_send_sem,
        recv_sems=push_recv_sems,
    )
    me = lax.axis_index(axis_name)

    # Peers' buffers are only theirs to write once they have entered.
    _barrier_wait(num_devices, barrier_sem)
    first_group = jnp.where(num_gm > 0, metadata_ref.gm_id_to_group_id[0], 0)
    _push_groups(first_group + 1 + DISPATCH_LOOKAHEAD, dispatch, pending_ref, me)

    (_, w13_spec), _ = generate_block_specs(metadata_ref, up)
    (_, w2_spec), _ = generate_block_specs(metadata_ref, down)
    w13_up_ref = jax.tree.map(lambda x: x.at[..., up.out_size_n :], w13_ref)
    w13_ref = FusedWeightsRef(gate=w13_ref, up=w13_up_ref)
    w13_spec = FusedWeightsRef(gate=w13_spec, up=w13_spec)

    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(ep_inner_kernel, up=up, down=down, axis_name=axis_name),
        grid=(1, num_gm, 1),
        in_specs=(w13_spec, w2_spec),
        out_specs=(),
    )
    pipeline_fn(
        w13_ref,
        w2_ref,
        scratches=[
            dispatch,
            lhs_staging_ref,
            fetch_sems,
            dest_ref,
            recv_ref,
            staging_ref,
            metadata_ref,
            pending_ref,
            send_sems,
            local_sems,
            recv_sem,
        ],
    )

    # Peers may still wait on rows for experts this device has no tiles past.
    _push_groups(up.dims.size_group, dispatch, pending_ref, me)
    for slot in range(2):
        staging = staging_ref.at[slot]
        _wait_rows(send_sems.at[slot], staging, pending_ref[_PENDING_REMOTE + slot])
        _wait_rows(local_sems.at[slot], staging, pending_ref[_PENDING_LOCAL + slot])
    _wait_rows(push_send_sem, x_ref, pending_ref[_PUSHED_REMOTE])
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
    x: jax.Array,  # u32[num_tokens, k / 2]: `pack_rows` of this device's bf16 tokens
    w1: jax.Array,  # [size_group, k, 2 * i]: gate then up along the last dim
    w2: jax.Array,  # [size_group, i, n]
    group_sizes: jax.Array,  # int32[size_lhs_group]: rows all devices push here
    dest: jax.Array,  # int32[size_m], see DEST_DEVICE_SHIFT; read for grouped rows
    num_recv: jax.Array,  # int32[1]: rows the other devices send to this one
    send_dest: jax.Array,  # int32[num_routes]: device << SHIFT | its grouped row
    send_src: jax.Array,  # int32[num_routes]: the row of x each route pushes
    send_bounds: jax.Array,  # int32[size_lhs_group + 1], see below
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
    """Each grouped row of `gmm_v2(act(gmm_v2(lhs, w1)), w2)`, packed, at its slot,
    where `lhs` is the bf16 rows every device's `send_*` tables push here.

    This device pushes row `send_src[i]` of `x` to grouped row
    `send_dest[i] & mask` of device `send_dest[i] >> DEST_DEVICE_SHIFT`, for
    `i` in `[send_bounds[g], send_bounds[g + 1])` to that device's local
    expert g; the pushes to all devices must fill each device's grouped rows
    `[0, sum(group_sizes))` exactly once.

    Returns this device's u32[recv_rows, n / 2] receive buffer. Row r holds the
    `pack_rows` of the output row some device's `dest` addressed to (this
    device, r); rows nobody addressed are unset. `num_recv` must equal the
    number of rows addressed here by other devices, or the call never returns.
    """
    if vmem_limit_bytes is None:
        vmem_limit_bytes = _default_vmem_limit()
    size_m = dest.shape[0]
    lhs = jax.ShapeDtypeStruct((size_m, 2 * x.shape[1]), jnp.bfloat16)
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

    if x.dtype != jnp.uint32:
        raise ValueError(f"gmm_v2_ffn_ep pushes pack_rows u32 rows, not {x.dtype}")
    if send_bounds.shape != (up.dims.size_group + 1,):
        raise ValueError(
            f"{send_bounds.shape=} does not bound {up.dims.size_group} experts"
        )
    padded_size_m = up.dims.size_m
    if padded_size_m > _DEST_ROW_MASK + 1:
        raise ValueError(f"{padded_size_m=} does not fit the destination table")
    if padded_size_m != size_m:
        dest = jnp.pad(dest, (0, padded_size_m - size_m))
    group_offset = jnp.zeros((1,), jnp.int32)

    w1_scale, w1_scale_spec = _hbm_f32(w1_scale)
    w2_scale, w2_scale_spec = _hbm_f32(w2_scale)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    half_n = down.out_size_n // 2
    max_num_gm = up.dims.size_group + pl.cdiv(padded_size_m, up.tiles.tile_m) - 1
    scratch_shapes = [
        pltpu.VMEM((2, up.tiles.tile_m, x.shape[1]), jnp.uint32),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.VMEM((2, up.tiles.tile_m, half_n), jnp.uint32),
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
        ),
        pltpu.SMEM((_NUM_PENDING,), jnp.int32),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.REGULAR,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((up.dims.size_group,)),
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
        out_shape=(
            jax.ShapeDtypeStruct((recv_rows, half_n), jnp.uint32),
            jax.ShapeDtypeStruct(
                (padded_size_m + _fetch_chunk_rows(up.tiles.tile_m), x.shape[1]),
                jnp.uint32,
            ),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=7,
            in_specs=[
                hbm,
                WeightsRef(weight=hbm, scale=w1_scale_spec, bias=None),
                WeightsRef(weight=hbm, scale=w2_scale_spec, bias=None),
            ],
            out_specs=(hbm, hbm),
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
        send_dest,
        send_src,
        send_bounds,
        x,
        WeightsRef(weight=w1, scale=w1_scale, bias=None),
        WeightsRef(weight=w2, scale=w2_scale, bias=None),
    )[0]
