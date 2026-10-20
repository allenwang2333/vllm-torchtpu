# ruff: noqa: E501 -- anchors below must match kernel source text verbatim
"""Diagnostic variants of the fused-EP MoE v2 kernel, built from the tree's own source.

Each arm is a list of exact text substitutions applied to private copies of
``vllm_torchtpu/kernels/fused_moe/v2/kernel.py`` (and ``layer.py`` for the
layer pieces); a substitution whose anchor
does not occur exactly once refuses, so an arm can never silently measure the
unmodified kernel. The copy is imported as a fresh module (its own build
cache) and its ``build_fused_ep_moe_kernel`` is installed into ``layer``.

Every arm is a TIMING probe only: its outputs are garbage by construction.
"""

import importlib.util
import pathlib
import sys

from vllm_torchtpu.kernels.fused_moe.v2 import kernel as _kernel_mod
from vllm_torchtpu.kernels.fused_moe.v2 import layer as _layer_mod

# Weight slabs never stream: slot contents are whatever VMEM held. The small
# per-expert scale tables still stream so the scale reads are unchanged.
_NO_WEIGHTS = [
    (
        """            w1_copy(expert, slot).start(priority=WEIGHT_DMA_PRIORITY)
            w2_copy(expert, slot).start(priority=WEIGHT_DMA_PRIORITY)
""",
        """            pass
""",
    ),
    (
        """                @pl.when(first_tile)
                def _():
                    copy(expert, slot).wait()
""",
        """                return
""",
    ),
]

# The fp4 -> fp8 widening is replaced by a plain bitcast of twice the packed
# rows: the matmul sees an fp8 [qb, N] block of the same slab bytes, so MXU
# work and VMEM reads stay, and only the unpack/convert goes.
_NO_DEQUANT = [
    (
        """                block = pltpu.bitcast(
                    w1_vm[slot, pl.ds(b * packed_rows, packed_rows), :], FP4
                ).astype(FP8)
""",
        """                block = pltpu.bitcast(
                    w1_vm[
                        slot,
                        pl.ds(
                            min(b * packed_rows, w1_vm.shape[1] - 2 * packed_rows),
                            2 * packed_rows,
                        ),
                        :,
                    ],
                    FP8,
                )
""",
    ),
    (
        """                return pltpu.bitcast(
                    w2_vm[slot, pl.ds(b * packed_rows, packed_rows), :], FP4
                ).astype(FP8)
""",
        """                return pltpu.bitcast(
                    w2_vm[
                        slot,
                        pl.ds(
                            min(b * packed_rows, w2_vm.shape[1] - 2 * packed_rows),
                            2 * packed_rows,
                        ),
                        :,
                    ],
                    FP8,
                )
""",
    ),
]

# The whole expert FFN is replaced by a zero tile: weights still stream (and
# are still waited), rows are still gathered, tiles are still staged and
# shipped.
_NO_FFN = [
    (
        """                        acc2, mid_scale = expert_ffn_blockscale(
                            act_rows,
                            act_scales,
                            w1_block,
                            w2_block,
                            w1s_blocks,
                            w2s_blocks,
                            qb=rhs_qb,
                            act_fn=act_fn,
                            w1b=w1b_row,
                        )
""",
        """                        del w1_block, w2_block, w1s_blocks, w2s_blocks
                        # The slab waits normally sit inside the block readers.
                        _wait = _slab_waiter(e, slot, first_tile)
                        _wait(w1_copy, 0)
                        _wait(w2_copy, 0)
                        # Rows-dependent rather than a constant: Mosaic refuses
                        # the epilogue's bf16 row reduction on a splat.
                        acc2 = act_rows.astype(jnp.float32)
                        mid_scale = act_scales
""",
    ),
]

# Remote pushes become local copies into this shard's own arrival buffer at
# the same offsets (same DMA engine and HBM write traffic, no fabric); the
# arrival waits for peers' pushes go.
_LOOPBACK = [
    (
        """                        pltpu.make_async_remote_copy(
                            src_ref=out_vm.at[parity, pl.ds(src_row, true_here)],
                            dst_ref=recv_hbm.at[pl.ds(dst_row, true_here)],
                            send_sem=send_sems.at[parity],
                            recv_sem=recv_sem,
                            device_id=(jnp.int32(d),),
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start()
""",
        """                        pltpu.make_async_copy(
                            out_vm.at[parity, pl.ds(src_row, true_here)],
                            recv_hbm.at[pl.ds(dst_row, true_here)],
                            send_sems.at[parity],
                        ).start()
""",
    ),
    (
        """                        pltpu.make_async_remote_copy(
                            src_ref=oscl_vm.at[parity],
                            dst_ref=rscl_hbm.at[pl.ds(dst_sub, 1)],
                            send_sem=send_scl_sems.at[parity],
                            recv_sem=recv_scl_sem,
                            device_id=(jnp.int32(d),),
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start()
""",
        """                        pltpu.make_async_copy(
                            oscl_vm.at[parity],
                            rscl_hbm.at[pl.ds(dst_sub, 1)],
                            send_scl_sems.at[parity],
                        ).start()
""",
    ),
    (
        """                _rows_wait(recv_sem, recv_hbm, count(host.COUNT_RECV_ROWS))
                _rows_wait(recv_scl_sem, rscl_hbm, count(host.COUNT_RECV_MIRROR))
                return
""",
        """                return
""",
    ),
]

# Every tile computes at the full tile height (M = 128) instead of the
# smallest rung covering its live rows.
_FULL_HEIGHT = [
    (
        """    height_rungs = tile_height_rungs(tile_m, ragged_rows_alloc, g_local, ep)
""",
        """    height_rungs = (tile_m,)
""",
    ),
]

# Every tile computes at the lowest rung (M = 32) whatever its live rows.
_LOW_HEIGHT = [
    (
        """                        @pl.when(jnp.logical_and(live_rows > below, live_rows <= rung))
                        def _(rung=rung):
                            compute_tile(rung)
""",
        """                        @pl.when(jnp.logical_and(live_rows > below, live_rows <= rung))
                        def _(rung=rung):
                            compute_tile(height_rungs[0])
""",
    ),
]

# The per-row token gather is not issued and not waited: activation rows are
# whatever the staging slot held. The row-scale window still moves.
_NO_GATHER = [
    (
        """                    for r in range(ROWBLK):
                        token = gather_smem[gather_offset + i * ROWBLK + r]
                        pltpu.make_async_copy(
                            lhs_hbm.at[token],
                            lhs_vm.at[slot, i * ROWBLK + r],
                            lhs_sems.at[slot],
                        ).start()
""",
        """                    pass
""",
    ),
    (
        """            pltpu.make_async_copy(
                lhs_vm.at[slot, pl.ds(0, live_rows)],
                lhs_vm.at[slot, pl.ds(0, live_rows)],
                lhs_sems.at[slot],
            ).wait()
""",
        """            del live_rows
""",
    ),
]

# LAYER piece: the dispatch all-gather is replaced by tiling this shard's own
# rows ep times (a local 11.8 MB copy instead of the collective).
_NO_ALLGATHER = [
    (
        """        q_g = lax.all_gather(
            q_l.reshape(t_local, hidden // HIDDEN_LANE_BLOCK, HIDDEN_LANE_BLOCK),
            ax,
            axis=0,
            tiled=True,
        )
""",
        """        q_g = jnp.tile(
            q_l.reshape(t_local, hidden // HIDDEN_LANE_BLOCK, HIDDEN_LANE_BLOCK),
            (ep, 1, 1),
        )
""",
    ),
]

_LAYER_PIECES = {"noag": _NO_ALLGATHER}

_PIECES = {
    "nogather": _NO_GATHER,
    "now": _NO_WEIGHTS,
    "nodq": _NO_DEQUANT,
    "noffn": _NO_FFN,
    "loop": _LOOPBACK,
    "m128": _FULL_HEIGHT,
    "m32": _LOW_HEIGHT,
}


def arm_pieces(arm):
    """'base' or a '+'-joined set of piece names, e.g. 'now+nodq'.

    Returns (kernel substitutions, layer substitutions)."""
    if arm == "base":
        return [], []
    names = arm.split("+")
    unknown = [n for n in names if n not in _PIECES and n not in _LAYER_PIECES]
    if unknown:
        raise ValueError(
            f"unknown ablation pieces {unknown}; have "
            f"{sorted(_PIECES) + sorted(_LAYER_PIECES)}"
        )
    return (
        [sub for n in names for sub in _PIECES.get(n, [])],
        [sub for n in names for sub in _LAYER_PIECES.get(n, [])],
    )


def _patched(path, subs, arm):
    src = pathlib.Path(path).read_text()
    for old, new in subs:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(
                f"arm {arm!r}: anchor occurs {n} times, expected 1:\n{old}"
            )
        src = src.replace(old, new)
    return src


def _load(name, source):
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = f"<{name}>"
    sys.modules[name] = mod
    exec(compile(source, mod.__file__, "exec"), mod.__dict__)
    return mod


def install(arm):
    """Build the arm's kernel (and, if it has layer pieces, layer) modules.

    Returns (kernel module, layer module); the layer module's
    `build_fused_ep_moe_kernel` is the arm's. The shard_map cache of the layer
    module is cleared because its key does not include the kernel variant.
    """
    kernel_subs, layer_subs = arm_pieces(arm)
    tag = arm.replace("+", "_")
    kmod = _load(
        f"_moe_abl_kernel_{tag}", _patched(_kernel_mod.__file__, kernel_subs, arm)
    )
    lmod = (
        _load(f"_moe_abl_layer_{tag}", _patched(_layer_mod.__file__, layer_subs, arm))
        if layer_subs
        else _layer_mod
    )
    lmod.build_fused_ep_moe_kernel = kmod.build_fused_ep_moe_kernel
    lmod._LAYER_SM_CACHE.clear()
    return kmod, lmod
