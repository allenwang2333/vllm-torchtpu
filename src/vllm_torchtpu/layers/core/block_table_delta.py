"""Device-resident staged block tables with delta uploads.

`stage_block_table_uploads` packs every KV cache group's block-table rows into
one flat int32 buffer and uploads it on each metadata walk. At long context
that buffer is large (DeepSeek-V4.1 at 100k context, 51 groups, 16 requests:
`s32[2166208]`, 8.67 MB per rank per step). Pin memory is unsupported on TPU,
so the copy is synchronous on the host and it sits on the decode critical
path. Yet between two decode steps only a handful of entries change: one new
block id each time a request crosses a block boundary in some group.

`BlockTableDeltaStager` keeps the last uploaded packed buffer on device, plus a
host mirror of its contents, for each staging layout. On the next walk with the
same layout it diffs only the rows that may have changed against the mirror.
It uploads the changed `(flat_index, value)` pairs as one fixed-size
`int32[2, DELTA_CAPACITY]` tensor and scatters them into the device copy.
A walk with no changes uploads nothing. A walk with more changes than
`DELTA_CAPACITY`, or with a new layout, falls back to the full upload.

Rows that may have changed are found through per-row version counters. They
are installed on each upstream `BlockTable` by swapping its class for a
subclass whose row mutators (`append_row` / `add_row` / `clear_row` /
`move_row` / `swap_row` / `clear`) bump the counters. Those methods are the
only writers of the host table: no `vllm_torchtpu` code writes it directly.
`TPU_VERIFY_INCREMENTAL_BLOCK_TABLE=1` rechecks the whole table every step.

The device tensor is only ever read by `_staged_walk_products_compiled`, whose
outputs are fresh buffers, so reusing it across steps is safe. The scatter
program is functional (clone + index_put), so the previous buffer is never
written in place.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.v1.worker.block_table import BlockTable

# Changed entries one delta upload carries. Larger walks take the full path.
# 128 requests crossing a block boundary together in every one of 51 groups is
# 6528 entries.
DELTA_CAPACITY = 8192
# Staging layouts kept on device (target walk, drafter walk, chunk offsets).
MAX_LAYOUTS = 4

_token_counter = itertools.count(1)
_tracked_classes: dict[type, type] = {}


def _versioned_subclass(base: type) -> type:
    """A subclass of `base` whose row mutators record what they changed.

    `_row_rewrite[row]` bumps on every mutation other than `append_row`, which
    only writes columns `[num_blocks_per_row[row], ...)` and increments
    `num_blocks_per_row[row]`. So a row whose `_row_rewrite` matches the
    snapshot changed only in the columns between the snapshot's and the current
    `num_blocks_per_row`.
    """
    cls = _tracked_classes.get(base)
    if cls is not None:
        return cls

    class _Versioned(base):  # type: ignore[valid-type, misc]
        @property
        def _row_version(self) -> np.ndarray:
            return self._row_rewrite + self.num_blocks_per_row

        def _rewrote(self, rows):
            self._bt_mutations += 1
            self._bt_rewrites += 1
            self._row_rewrite[rows] += 1

        def append_row(self, block_ids, row_idx):
            if not block_ids:
                return
            if len(block_ids) == 1 and self._bt_fast_append:
                nblocks = self.num_blocks_per_row
                start = int(nblocks[row_idx])
                nblocks[row_idx] = start + 1
                self._bt_cpu_np[row_idx, start] = block_ids[0]
            else:
                super().append_row(block_ids, row_idx)
            self._bt_mutations += 1

        def add_row(self, block_ids, row_idx):
            super().add_row(block_ids, row_idx)
            self._rewrote(row_idx)

        def clear_row(self, row_idx):
            super().clear_row(row_idx)
            self._rewrote(row_idx)

        def move_row(self, src, tgt):
            super().move_row(src, tgt)
            self._rewrote([src, tgt])

        def swap_row(self, src, tgt):
            super().swap_row(src, tgt)
            self._rewrote([src, tgt])

        def clear(self):
            super().clear()
            self._rewrote(slice(None))

    _Versioned.__name__ = f"Versioned{base.__name__}"
    _Versioned.__qualname__ = _Versioned.__name__
    _tracked_classes[base] = _Versioned
    return _Versioned


def ensure_tracked(block_table: Any) -> int:
    """Install row-version tracking on `block_table`; return its identity token.

    The token, not `id()`, keys staging layouts, so a re-created block table
    (KV cache re-initialization) can never alias a stale device copy.
    """
    token = getattr(block_table, "_bt_delta_token", None)
    if token is None:
        cpu_tensor = block_table.get_cpu_tensor()
        num_rows = cpu_tensor.shape[0]
        block_table._bt_cpu_np = cpu_tensor.numpy()
        block_table._row_rewrite = np.zeros(num_rows, dtype=np.int32)
        block_table._bt_mutations = 0
        block_table._bt_rewrites = 0
        block_table._bt_fast_append = type(
            block_table
        ).append_row is BlockTable.append_row and not getattr(
            block_table, "use_hybrid_blocks", False
        )
        block_table.__class__ = _versioned_subclass(type(block_table))
        token = next(_token_counter)
        block_table._bt_delta_token = token
    return token


def _apply_block_table_delta(
    staged: torch.Tensor, update: torch.Tensor
) -> torch.Tensor:
    """`staged` with `staged[update[0]] = update[1]`, as a new tensor.

    Padding slots repeat a real `(index, value)` pair, so duplicate indices
    always carry identical values and the scatter result is order-independent.
    """
    out = staged.clone()
    out[update[0].to(torch.int64)] = update[1]
    return out


_apply_block_table_delta_compiled = torch.compile(
    _apply_block_table_delta, backend="tpu", fullgraph=True, dynamic=False
)


def apply_block_table_delta(staged: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    if staged.device.type == "tpu":
        return _apply_block_table_delta_compiled(staged, update)
    return _apply_block_table_delta(staged, update)


@dataclass
class _EntryState:
    """What the device copy holds for one group's region."""

    offset: int
    num_blocks: int  # row width (target_num_blocks)
    active_cols: int  # columns copied from the host table; zeros beyond
    num_reqs: int  # live rows; rows past this are zeros
    rewrites: np.ndarray  # source-row _row_rewrite at upload, length num_reqs
    nblocks: np.ndarray  # source-row num_blocks_per_row at upload
    mutations: int  # source table's total mutation count at upload
    total_rewrites: int = 0  # source table's total rewrite count at upload
    bt: Any = None
    ac_key: tuple[int, bool, int] = (0, False, 0)


@dataclass
class _Layout:
    dev: torch.Tensor
    mirror: np.ndarray  # int32, equal to `dev` element for element
    entries: list[_EntryState]
    unique_geoms: tuple[tuple[Any, int, _EntryState], ...] = ()
    tables: dict[int, torch.Tensor] | None = None


class BlockTableDeltaStager:
    """Per-runner cache of device-resident staged block tables."""

    def __init__(self) -> None:
        self._layouts: OrderedDict[tuple, _Layout] = OrderedDict()
        self._zero_update: dict[torch.device, torch.Tensor] = {}
        self._update_buf: np.ndarray | None = None
        self._update_tensor: torch.Tensor | None = None
        self.num_full = 0
        self.num_delta = 0
        self.num_reuse = 0

    @staticmethod
    def layout_key(
        entries: list[tuple[int, Any, int]], num_reqs_padded: int, start_index: int
    ) -> tuple:
        return (
            tuple((gid, ensure_tracked(bt), tnb) for gid, bt, tnb in entries),
            num_reqs_padded,
            start_index,
        )

    def get_cached_tables(self, key: tuple) -> dict[int, torch.Tensor] | None:
        layout = self._layouts.get(key)
        return layout.tables if layout is not None else None

    def set_cached_tables(self, key: tuple, tables: dict[int, torch.Tensor]) -> None:
        layout = self._layouts.get(key)
        if layout is not None:
            layout.tables = tables

    def try_stage(
        self,
        key: tuple,
        ctx: Any,
        entries: list[tuple[int, Any, int]],
        builders: list[Any],
        num_reqs_padded: int,
    ) -> torch.Tensor | None:
        """The up-to-date device copy for `key`, or None for a full upload."""
        layout = self._layouts.get(key)
        if layout is None:
            return None
        start, num_reqs = ctx.start_index, ctx.num_reqs
        if (
            layout.entries
            and num_reqs == layout.entries[0].num_reqs
            and all(st.bt._bt_mutations == st.mutations for st in layout.entries)
            and all(
                b._active_block_table_columns(ctx, tnb) == st.active_cols
                for b, tnb, st in layout.unique_geoms
            )
        ):
            self._layouts.move_to_end(key)
            self.num_reuse += 1
            return layout.dev
        end = start + num_reqs
        cap = DELTA_CAPACITY
        update_buf = self._update_buf
        if update_buf is None or update_buf.shape[1] != cap:
            update_buf = np.empty((2, cap), dtype=np.int32)
            self._update_buf = update_buf
            self._update_tensor = torch.from_numpy(update_buf)
        update_tensor = self._update_tensor
        assert update_tensor is not None
        idx_out = update_buf[0]
        val_out = update_buf[1]
        pos = 0
        active_cols_cache: dict[tuple[int, bool, int], int] = {}
        prev_plan: (
            tuple[
                int,
                int,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]
            | None
        ) = None

        for builder, (_, bt, tnb), st in zip(
            builders, entries, layout.entries, strict=True
        ):
            ac_key = st.ac_key
            active_cols = active_cols_cache.get(ac_key)
            if active_cols is None:
                active_cols = builder._active_block_table_columns(ctx, tnb)
                active_cols_cache[ac_key] = active_cols
            mutations = bt._bt_mutations
            if (
                mutations == st.mutations
                and active_cols == st.active_cols
                and num_reqs == st.num_reqs
            ):
                continue
            nblocks = bt.num_blocks_per_row[start:end]
            source = bt._bt_cpu_np[start:end]
            rewrites = bt._bt_rewrites

            if (
                active_cols == st.active_cols
                and num_reqs == st.num_reqs
                and rewrites == st.total_rewrites
            ):
                cur_plan: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
                if (
                    prev_plan is not None
                    and prev_plan[0] == tnb
                    and prev_plan[1] == active_cols
                    and np.array_equal(nblocks, prev_plan[2])
                    and np.array_equal(st.nblocks, prev_plan[3])
                ):
                    cur_plan = (prev_plan[4], prev_plan[5], prev_plan[6])
                else:
                    changed_rows = np.flatnonzero(nblocks != st.nblocks).astype(
                        np.int32, copy=False
                    )
                    lo_all = np.minimum(st.nblocks[changed_rows], np.int32(active_cols))
                    hi_all = np.minimum(nblocks[changed_rows], np.int32(active_cols))
                    counts = hi_all - lo_all
                    if (counts <= 1).all():
                        if (counts == 1).all():
                            app_rows, lo = changed_rows, lo_all
                        else:
                            valid = counts == 1
                            app_rows, lo = changed_rows[valid], lo_all[valid]
                        row_offsets = app_rows * np.int32(tnb) + lo
                        cur_plan = (app_rows, lo, row_offsets)
                        nblocks_snap = nblocks.copy()
                        st_nblocks_snap = st.nblocks.copy()
                        prev_plan = (
                            tnb,
                            active_cols,
                            nblocks_snap,
                            st_nblocks_snap,
                            app_rows,
                            lo,
                            row_offsets,
                        )
                if cur_plan is not None:
                    app_rows, lo, row_offsets = cur_plan
                    k = app_rows.size
                    if k:
                        if pos + k > cap:
                            del self._layouts[key]
                            return None
                        flat_idx = np.int32(st.offset) + row_offsets
                        new_vals = source[app_rows, lo]
                        layout.mirror[flat_idx] = new_vals
                        idx_out[pos : pos + k] = flat_idx
                        val_out[pos : pos + k] = new_vals
                        pos += k
                    st.nblocks[:] = nblocks
                    st.mutations = mutations
                    continue

            region = layout.mirror[st.offset : st.offset + num_reqs_padded * tnb]
            mirror = region.reshape(num_reqs_padded, tnb)
            row_rewrites = bt._row_rewrite[start:end]
            span = max(num_reqs, st.num_reqs)
            if active_cols != st.active_cols:
                full_rows = np.arange(span, dtype=np.int32)
                app_rows = np.empty(0, dtype=np.int32)
            else:
                common = min(num_reqs, st.num_reqs)
                rewritten_mask = row_rewrites[:common] != st.rewrites[:common]
                rewritten_rows = np.flatnonzero(rewritten_mask).astype(
                    np.int32, copy=False
                )
                full_rows = (
                    np.concatenate(
                        [rewritten_rows, np.arange(common, span, dtype=np.int32)]
                    )
                    if common < span
                    else rewritten_rows
                )
                app_rows = np.flatnonzero(
                    (~rewritten_mask) & (nblocks[:common] != st.nblocks[:common])
                ).astype(np.int32, copy=False)
            if full_rows.size:
                # Rows rewritten in place, or entering / leaving the live range:
                # compare their whole active width against the mirror.
                width = max(active_cols, st.active_cols)
                expected = np.zeros((full_rows.size, width), dtype=np.int32)
                live = full_rows < num_reqs
                expected[live, :active_cols] = source[full_rows[live], :active_cols]
                r, c = np.nonzero(expected != mirror[full_rows, :width])
                k = r.size
                if pos + k > cap:
                    del self._layouts[key]
                    return None
                rows = full_rows[r]
                new = expected[r, c]
                mirror[rows, c] = new
                idx_out[pos : pos + k] = st.offset + rows * tnb + c
                val_out[pos : pos + k] = new
                pos += k
            if app_rows.size:
                # Append-only since the snapshot: columns below the snapshot's
                # block count, and at or past the current one, are unchanged.
                lo_arr = np.minimum(st.nblocks[app_rows], active_cols)
                hi_arr = np.minimum(nblocks[app_rows], active_cols)
                valid = hi_arr > lo_arr
                for row, l_i, h_i in zip(
                    app_rows[valid].tolist(),
                    lo_arr[valid].tolist(),
                    hi_arr[valid].tolist(),
                ):
                    k = h_i - l_i
                    if pos + k > cap:
                        del self._layouts[key]
                        return None
                    new_row = source[row, l_i:h_i]
                    mirror[row, l_i:h_i] = new_row
                    base = st.offset + row * tnb
                    idx_out[pos : pos + k] = np.arange(
                        base + l_i, base + h_i, dtype=np.int32
                    )
                    val_out[pos : pos + k] = new_row
                    pos += k
            st.active_cols = active_cols
            if st.num_reqs == num_reqs:
                st.rewrites[:] = row_rewrites
                st.nblocks[:] = nblocks
            else:
                st.num_reqs = num_reqs
                st.rewrites = row_rewrites.copy()
                st.nblocks = nblocks.copy()
            st.mutations = mutations
            st.total_rewrites = rewrites
        self._layouts.move_to_end(key)
        if pos:
            idx_out[pos:] = idx_out[0]
            val_out[pos:] = val_out[0]
            update_dev = update_tensor.to(layout.dev.device)
            layout.dev = apply_block_table_delta(layout.dev, update_dev)
            layout.tables = None
            self.num_delta += 1
        else:
            self.num_reuse += 1
        return layout.dev

    def record_full(
        self,
        key: tuple,
        ctx: Any,
        entries: list[tuple[int, Any, int]],
        builders: list[Any],
        num_reqs_padded: int,
        flat: torch.Tensor,
        staged_dev: torch.Tensor,
    ) -> None:
        """Remember a full upload so the next walk can send a delta."""
        start, num_reqs = ctx.start_index, ctx.num_reqs
        end = start + num_reqs
        states: list[_EntryState] = []
        unique_geoms: list[tuple[Any, int, _EntryState]] = []
        seen_geoms: dict[tuple[int, bool, int], int] = {}
        offset = 0
        for builder, (_, bt, tnb) in zip(builders, entries, strict=True):
            ac_key = (builder.target_block_size, builder.is_mamba_group, tnb)
            is_new_geom = ac_key not in seen_geoms
            active_cols = (
                builder._active_block_table_columns(ctx, tnb)
                if is_new_geom
                else seen_geoms[ac_key]
            )
            st = _EntryState(
                offset,
                tnb,
                active_cols,
                num_reqs,
                bt._row_rewrite[start:end].copy(),
                bt.num_blocks_per_row[start:end].copy(),
                bt._bt_mutations,
                bt._bt_rewrites,
                bt,
                ac_key,
            )
            states.append(st)
            if is_new_geom:
                seen_geoms[ac_key] = active_cols
                unique_geoms.append((builder, tnb, st))
            offset += num_reqs_padded * tnb
        if staged_dev.device.type == "cpu":
            # `.to()` on a CPU device returns the shared host scratch itself,
            # which the next full upload overwrites.
            staged_dev = staged_dev.clone()
        self._layouts[key] = _Layout(
            staged_dev, flat.numpy().copy(), states, tuple(unique_geoms)
        )
        self._layouts.move_to_end(key)
        while len(self._layouts) > MAX_LAYOUTS:
            self._layouts.popitem(last=False)
        self.num_full += 1

    def warm(self, staged_dev: torch.Tensor) -> None:
        """Compile the scatter for this layout during dummy runs (no-op update)."""
        device = staged_dev.device
        zero = self._zero_update.get(device)
        if zero is None:
            zero = torch.zeros((2, DELTA_CAPACITY), dtype=torch.int32).to(device)
            self._zero_update[device] = zero
        apply_block_table_delta(staged_dev, zero)

    def verify(self, key: tuple, expected_flat: torch.Tensor) -> None:
        """Raise unless the mirror and the device copy equal `expected_flat`."""
        layout = self._layouts[key]
        expected = expected_flat.numpy()
        if not np.array_equal(layout.mirror, expected):
            bad = np.flatnonzero(layout.mirror != expected)
            raise RuntimeError(
                f"incremental block table mirror diverged at {bad.size} entries "
                f"(first {bad[:8].tolist()})"
            )
        device_copy = layout.dev.cpu().numpy()
        if not np.array_equal(device_copy, expected):
            bad = np.flatnonzero(device_copy != expected)
            raise RuntimeError(
                f"incremental block table device copy diverged at {bad.size} "
                f"entries (first {bad[:8].tolist()})"
            )


def get_stager(runner: Any) -> BlockTableDeltaStager:
    stager = getattr(runner, "_block_table_delta_stager", None)
    if not isinstance(stager, BlockTableDeltaStager):
        stager = BlockTableDeltaStager()
        runner._block_table_delta_stager = stager
    return stager
