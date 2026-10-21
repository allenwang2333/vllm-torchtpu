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
"""Shapes, tile plans and the global tile schedule of the dense weight ring.

The dense chain of every decoder layer reads its weights through one VMEM ring
in consumption order. A *stream* is one weight kind (e.g. ``wq_b``); each
stream is pre-tiled in HBM as ``[layers_with_it, n_tiles, tile_rows,
tile_cols]`` so every tile is one contiguous DMA. The schedule is the
flattened sequence of (stream, layer ordinal, tile) over all layers.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np

FP8 = "float8_e4m3fn"
BF16 = "bfloat16"
_ITEMSIZE = {FP8: 1, BF16: 2}
# Rows of one (8, 128) 32-bit VMEM tile once sub-32-bit types are packed.
_SUBLANES = {FP8: 32, BF16: 16}

DSV4_PRO_RATIOS = (128, 128) + (4, 128) * 29 + (4,)

HCA, CSA, SWA = 0, 1, 2  # SWA: sliding-window-only layers (compress ratio <= 1)


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def storage_bytes(rows: int, cols: int, dtype: str) -> int:
    """VMEM bytes of a 2-D tile in its native tiled layout."""
    assert cols % 128 == 0, cols
    return _round_up(rows, _SUBLANES[dtype]) * cols * _ITEMSIZE[dtype]


@dataclasses.dataclass(frozen=True)
class Stream:
    """One weight kind streamed through the ring."""

    name: str
    dtype: str
    k: int  # contraction rows of the stored [k, n] matrix (cols for "nt")
    n: int
    layer_type: int | None  # None: every layer; else HCA, CSA or SWA only
    tile_rows: int
    tile_cols: int
    nt: bool = False  # stored [n, k]; contracted over its last dim

    @property
    def n_k(self) -> int:
        """Tiles along the contraction."""
        return self.k // (self.tile_cols if self.nt else self.tile_rows)

    @property
    def n_n(self) -> int:
        return 1 if self.nt else self.n // self.tile_cols

    @property
    def n_tiles(self) -> int:
        return self.n_k * self.n_n

    @property
    def tile_bytes(self) -> int:
        return storage_bytes(self.tile_rows, self.tile_cols, self.dtype)

    @property
    def hbm_bytes(self) -> int:
        """Bytes streamed per layer (the tiles exactly; no padding in HBM)."""
        return self.tile_rows * self.tile_cols * _ITEMSIZE[self.dtype] * self.n_tiles

    def tile_coords(self, t: int) -> tuple[int, int]:
        """(k block, n block) of tile t: n-block outer, k-block inner."""
        return t % self.n_k, t // self.n_k


def _plan_tile(k: int, n: int, dtype: str, bank: int, min_rows: int) -> tuple[int, int]:
    """Largest contiguous [rows, cols] tile under ``bank`` bytes.

    Prefers full-width row slabs (one contiguous HBM range); splits the width
    in halves only when a slab of ``min_rows`` would not fit.
    """
    sub = _SUBLANES[dtype]
    cols = n
    while True:
        cands = [
            r
            for r in range(sub, k + 1, sub)
            if k % r == 0
            and (r % 128 == 0 or r == k)  # activation K-slices stay lane-aligned
            and storage_bytes(r, cols, dtype) <= bank
        ]
        if cands and (max(cands) >= min(min_rows, k) or cols % 256):
            return max(cands), cols
        assert cols % 256 == 0, (k, n, dtype, cols)
        cols //= 2


@dataclasses.dataclass(frozen=True)
class DenseChainConfig:
    """DSv4 dense-chain geometry for one TP rank (``tp`` = shard factor S)."""

    hidden: int = 7168
    hc_mult: int = 4
    q_lora: int = 1536
    head_dim: int = 512
    rope_dim: int = 64
    n_heads: int = 128
    heads_per_group: int = 8
    o_lora: int = 1024
    moe_inter: int = 3072
    n_experts: int = 384
    idx_heads: int = 64
    idx_head_dim: int = 128
    ratios: tuple[int, ...] = DSV4_PRO_RATIOS
    tp: int = 1
    rows: int = 16
    rms_eps: float = 1e-6
    hc_eps: float = 1e-6
    hc_post_mult: float = 2.0
    sinkhorn_iters: int = 20
    swiglu_limit: float = 10.0
    hc_chunks: int = 3
    # Fewer, larger tiles: each tile costs a fixed ~0.5 us of issue/drain
    # (S=16 on tpu7x: 2 MiB x 12 banks 2.64 ms, 4 MiB x 10 2.02, 5 MiB x 10
    # 1.94).
    bank_bytes: int = 5 * 1024 * 1024
    n_banks: int = 10
    min_tile_rows: int = 256

    def __post_init__(self):
        assert self.rows % 16 == 0, "pad rows to a multiple of 16"
        assert self.n_heads % (self.tp * self.heads_per_group) == 0
        assert self.moe_inter % self.tp == 0
        assert all(r in (0, 1, 4, 128) for r in self.ratios), self.ratios

    # ---- derived geometry -------------------------------------------------
    @property
    def n_layers(self) -> int:
        return len(self.ratios)

    @property
    def layer_types(self) -> tuple[int, ...]:
        return tuple(CSA if r == 4 else HCA if r == 128 else SWA for r in self.ratios)

    @property
    def heads(self) -> int:
        return self.n_heads // self.tp

    @property
    def groups(self) -> int:
        return self.heads // self.heads_per_group

    @property
    def inter(self) -> int:
        return self.moe_inter // self.tp

    @property
    def hc_hidden(self) -> int:
        return self.hc_mult * self.hidden

    @property
    def n_mix(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    @property
    def hc_rows(self) -> int:
        """Stacked bf16 chunk rows of one hc_fn, padded to a bf16 tile."""
        return _round_up(self.n_mix * self.hc_chunks, 16)

    @property
    def wp_cols(self) -> int:
        return _round_up(self.idx_heads, 128)

    @property
    def xproj_cols(self) -> dict[int, int]:
        """bf16 projections of the normed hidden: [wp | idx compressor |
        compressor] for CSA, the compressor alone for HCA."""
        return {
            CSA: self.wp_cols + 4 * self.idx_head_dim + 4 * self.head_dim,
            HCA: 2 * self.head_dim,
        }

    @property
    def qkv_cols(self) -> int:
        return self.q_lora + self.head_dim

    # ---- streams ----------------------------------------------------------
    def _stream(self, name, dtype, k, n, layer_type=None, min_rows=None):
        rows, cols = _plan_tile(
            k, n, dtype, self.bank_bytes, min_rows or self.min_tile_rows
        )
        return Stream(name, dtype, k, n, layer_type, rows, cols)

    @functools.cached_property
    def streams(self) -> dict[str, Stream]:
        h = self.hidden
        hc = Stream(
            "hc", BF16, self.hc_hidden, self.hc_rows, None, self.hc_rows, h, nt=True
        )
        s = [
            # SWA (DSpark draft) layers: the kv columns of fused_wqa_wkv applied
            # to the draft context (main_x), before the layer's own ops
            self._stream("wqa_ctx", FP8, h, self.head_dim, SWA),
            dataclasses.replace(hc, name="hc_attn"),
            self._stream("wqa", FP8, h, self.qkv_cols),
            # wq_b before the compressor / indexer projections: the whole step
            # sends q to the owners while those stream (consumption order)
            self._stream("wq_b", FP8, self.q_lora, self.heads * self.head_dim),
            self._stream("xproj_csa", BF16, h, self.xproj_cols[CSA], CSA),
            self._stream("xproj_hca", BF16, h, self.xproj_cols[HCA], HCA),
            self._stream(
                "idx_wq_b", FP8, self.q_lora, self.idx_heads * self.idx_head_dim, CSA
            ),
            self._wo_a_stream(),
            self._stream("wo_b", FP8, self.groups * self.o_lora, h),
            dataclasses.replace(hc, name="hc_ffn"),
            self._stream("gate", BF16, h, self.n_experts),
            self._stream("w13", FP8, h, 2 * self.inter),
            self._stream("w2", FP8, self.inter, h),
        ]
        # a target chain (no SWA layers) has exactly the streams it always had
        s = [x for x in s if x.layer_type != SWA or SWA in self.layer_types]
        return {x.name: x for x in s}

    def _wo_a_stream(self) -> Stream:
        k = self.heads_per_group * self.head_dim
        n = self.groups * self.o_lora
        rows, cols = _plan_tile(k, self.o_lora, FP8, self.bank_bytes, 256)
        assert self.o_lora % cols == 0  # a tile never straddles two groups
        return Stream("wo_a", FP8, k, n, None, rows, cols)

    @property
    def stream_names(self) -> tuple[str, ...]:
        return tuple(self.streams)

    def layer_streams(self, layer_type: int) -> tuple[str, ...]:
        """Streams of one layer of the given type, in consumption order."""
        return tuple(
            n
            for n, s in self.streams.items()
            if s.layer_type is None or s.layer_type == layer_type
        )

    def stream_layers(self, name: str) -> tuple[int, ...]:
        """Layers that carry stream ``name`` (its HBM stack order)."""
        lt = self.streams[name].layer_type
        return tuple(
            li for li, t in enumerate(self.layer_types) if lt is None or t == lt
        )

    @functools.cached_property
    def stream_offsets(self) -> dict[int, dict[str, int]]:
        """First tile of each stream within a layer of each type."""
        out = {}
        for lt in (HCA, CSA, SWA):
            off, d = 0, {}
            for n in self.layer_streams(lt):
                d[n] = off
                off += self.streams[n].n_tiles
            d["_total"] = off
            out[lt] = d
        return out

    # ---- global schedule --------------------------------------------------
    @functools.cached_property
    def schedule(self) -> tuple[np.ndarray, np.ndarray]:
        """(packed per-tile entries [T], first tile of each layer [L]).

        Entry = stream_id | ordinal << 4 | tile << 12.
        """
        ids = {n: i for i, n in enumerate(self.stream_names)}
        assert len(ids) <= 16
        ords = {n: {li: i for i, li in enumerate(self.stream_layers(n))} for n in ids}
        entries, base = [], []
        for li, lt in enumerate(self.layer_types):
            base.append(len(entries))
            for n in self.layer_streams(lt):
                assert ords[n][li] < 256
                for t in range(self.streams[n].n_tiles):
                    entries.append(ids[n] | ords[n][li] << 4 | t << 12)
        return np.asarray(entries, np.int32), np.asarray(base, np.int32)

    @property
    def n_tiles_total(self) -> int:
        return int(self.schedule[0].shape[0])

    # ---- per-layer small vectors ------------------------------------------
    @functools.cached_property
    def smalls_layout(self) -> dict[str, tuple[int, int]]:
        """name -> (first row, length) in the per-layer f32 ``[rows, 128]``
        block; element e of a vector sits at (row + e // 128, e % 128), so the
        block is dense in HBM and a 128-lane chunk is one row."""
        s = self.streams
        items = [
            ("attn_norm", self.hidden),
            ("ffn_norm", self.hidden),
            ("q_norm", self.q_lora),
            ("kv_norm", self.head_dim),
            ("hc_attn_base", self.n_mix),
            ("hc_attn_scale", 3),
            ("hc_ffn_base", self.n_mix),
            ("hc_ffn_scale", 3),
            ("wqa_scale", s["wqa"].n),
            ("wq_b_scale", s["wq_b"].n),
            ("idx_wq_b_scale", s["idx_wq_b"].n),
            ("wo_a_scale", s["wo_a"].n),
            ("wo_b_scale", s["wo_b"].n),
            ("w13_scale", s["w13"].n),
            ("w2_scale", s["w2"].n),
        ]
        if SWA in self.layer_types:
            items.append(("wqa_ctx_scale", s["wqa_ctx"].n))
        out, row = {}, 0
        for name, n in items:
            out[name] = (row, n)
            row += -(-n // 128)
        out["_total"] = (_round_up(row, 8), 0)
        return out

    @property
    def smalls_rows(self) -> int:
        return self.smalls_layout["_total"][0]

    # ---- probes (per-layer side outputs checked by the tests) --------------
    @functools.cached_property
    def probe_layout(self) -> dict[str, tuple[int, int]]:
        items = [
            ("qkv", self.qkv_cols),
            ("xproj", max(self.xproj_cols.values())),
            ("gate", self.n_experts),
        ]
        out, off = {}, 0
        for name, n in items:
            out[name] = (off, n)
            off += _round_up(n, 128)
        out["_total"] = (off, 0)
        return out

    # ---- bytes floor ------------------------------------------------------
    def streamed_bytes(self) -> dict[str, int]:
        """HBM bytes the kernel reads per step, by stream (+ smalls)."""
        out = {}
        for n, s in self.streams.items():
            out[n] = s.hbm_bytes * len(self.stream_layers(n))
        out["smalls"] = self.smalls_rows * 128 * 4 * self.n_layers
        return out

    def floor_us(self, hbm_bytes_per_s: float = 3.10e12) -> float:
        return sum(self.streamed_bytes().values()) / hbm_bytes_per_s * 1e6

    @property
    def max_tile_bytes(self) -> int:
        return max(s.tile_bytes for s in self.streams.values())


def small_config(**kw) -> DenseChainConfig:
    """Reduced widths for CPU interpret runs; same structure as DSv4-Pro."""
    base = dict(
        hidden=512,
        q_lora=256,
        head_dim=256,
        n_heads=32,
        o_lora=256,
        moe_inter=512,
        n_experts=128,
        idx_heads=8,
        idx_head_dim=128,
        ratios=(128, 128, 4),
        bank_bytes=512 * 1024,
        n_banks=4,
        min_tile_rows=128,
    )
    base.update(kw)
    return DenseChainConfig(**base)


def describe(cfg: DenseChainConfig) -> str:
    lines = []
    for n, s in cfg.streams.items():
        lines.append(
            f"{n:10s} {s.dtype:14s} k={s.k:6d} n={s.n:6d} tile=[{s.tile_rows},"
            f"{s.tile_cols}] x{s.n_tiles:3d} {s.tile_bytes / 2**20:.2f}MiB "
            f"layers={len(cfg.stream_layers(n))}"
        )
    tot = sum(cfg.streamed_bytes().values())
    lines.append(
        f"tiles={cfg.n_tiles_total} bytes={tot / 1e9:.3f}GB "
        f"floor={cfg.floor_us():.1f}us bank={cfg.max_tile_bytes / 2**20:.2f}MiB"
    )
    return "\n".join(lines)
