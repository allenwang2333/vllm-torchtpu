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
"""CSA Configs.

Modes:
  HCA:
    - State Bytes: 1024 (dim) x 4 (bytes per fp32) = 4096 bytes
    - Head Dim:    512 (bf16, no quantization)
    - Cache:       [num_pages, _, 4, 128] uint8 (where _ = state_block_size * 8
    or kv_block_size * 2)

  CSA:
    - State Bytes: 2048 (dim) x 4 (bytes per fp32) = 8192 bytes
    - Head Dim:    448 fp8 + 64 lane-periodic scales = 512 bytes
    - Cache:       [num_pages, _, 4, 128] uint8 (where _ = state_block_size * 16
    or kv_block_size), allocated as [num_pages, _, 128] int32
    - RoPE Cache:  64 bf16 = 128 bytes, [num_pages, kv_block_size // 4, 4, 128]
    uint8, allocated as [num_pages, kv_block_size // 4, 128] int32 (see
    `csa_cache_layout`)

  CSA-Indexer:
    - State Bytes: 512 (dim) x 4 (bytes per fp32) = 2048 bytes
    - Head Dim:    128 fp8 + 1 scale = 129 bytes -> 256 bytes
    - Cache:       [num_pages, _, 4, 256] uint8 (where _ = state_block_size * 4
    or kv_block_size // 4).
"""

import dataclasses
import enum
import functools
from typing import Any

import jax
import jax.numpy as jnp

# --- physical layout constants ------------------------------------------------
LANE = 128  # bytes per sub-slot / TPU lane width
SLOT_PACK = 4  # sub-slots packed into one physical HBM slot row
N_FIELDS = 2  # values stored per token: kv + score
FP32_BYTES = 4
CSA_COMPRESS_RATIO = 4  # the compress ratio that selects Mode.CSA


class Mode(enum.Enum):
    HCA = "hca"
    CSA = "csa"
    CSA_INDEXER = "csa_indexer"


_MODE_HCA: Mode = Mode.HCA
_MODE_CSA: Mode = Mode.CSA
_MODE_CSA_INDEXER: Mode = Mode.CSA_INDEXER
_CACHE_LAST_DIMS_INDEXER: tuple[int, int] = (SLOT_PACK, 2 * LANE)
_CACHE_LAST_DIMS_DEFAULT: tuple[int, int] = (SLOT_PACK, LANE)

_MODE_DEFAULTS: dict[Mode, dict[str, Any]] = {
    Mode.HCA: dict(
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=128,
        quant_block=0,
        overlap=False,
        has_rope_cache=False,
    ),
    Mode.CSA: dict(
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=4,
        quant_block=64,
        overlap=True,
        has_rope_cache=True,
    ),
    Mode.CSA_INDEXER: dict(
        head_dim=128,
        rope_head_dim=64,
        compress_ratio=4,
        quant_block=128,
        overlap=True,
        has_rope_cache=False,
    ),
}
_MODE_DEFAULT_TUPLES: dict[Mode, tuple[int, int, int, int, bool, bool]] = {
    m: (
        d["head_dim"],
        d["rope_head_dim"],
        d["compress_ratio"],
        d["quant_block"],
        d["overlap"],
        d["has_rope_cache"],
    )
    for m, d in _MODE_DEFAULTS.items()
}


def select_mode(head_dim: int, overlap: bool) -> Mode:
    """Picks the storage mode from the compressor's head_dim / overlap flag."""
    if head_dim == LANE:
        return _MODE_CSA_INDEXER
    return _MODE_CSA if overlap else _MODE_HCA


def physical_page_size(
    mode: Mode, kv_cache_block_size: int, compress_ratio: int
) -> int:
    """Rows per page of the uint8 cache array allocated for `mode`.

    This must mirror the shapes ``KVCacheManager._create_dsv4_kv_caches``
    allocates, because ``compress_norm_rope_store`` re-derives the whole page
    geometry from ``cache.shape[1]``. Anything that computes slot indices for
    that array (the compressor wrapper's state-cache ``block_size``, and
    ``derive_metadata``) has to agree with it or writes land on the wrong page.
    """
    # Compressed KV tokens per page (vLLM's ``spec.storage_block_size``).
    storage_block_size = kv_cache_block_size // compress_ratio
    if mode is _MODE_CSA_INDEXER:
        # (N, T // 4, 4, 256) u8
        return storage_block_size >> 2
    if mode is _MODE_CSA:
        # (N, T, 4, 128) u8
        return storage_block_size
    # HCA (N, T * 2, 4, 128) u8
    return storage_block_size << 1


def state_host_page_size(
    mode: Mode, kv_cache_block_size: int, compress_ratio: int
) -> int:
    """Rows per page of the array that *hosts* ``mode``'s f32 compressor state."""
    if mode is _MODE_HCA:
        return kv_cache_block_size >> 2
    return physical_page_size(mode, kv_cache_block_size, compress_ratio)


def last_dim_size(mode: Mode) -> int:
    """Size of the last HBM cache dimension in bytes."""
    return 256 if mode is _MODE_CSA_INDEXER else 128


def row_size_bytes(mode: Mode) -> int:
    """Size of one physical HBM row in bytes."""
    return 1024 if mode is _MODE_CSA_INDEXER else 512


def state_width(head_dim: int, overlap: bool) -> int:
    """Width of one stored field: head_dim, doubled when windows overlap."""
    # Overlapping windows keep two copies per token: prev + curr.
    return head_dim << 1 if overlap else head_dim


def state_rows_per_token(mode: Mode, head_dim: int, overlap: bool) -> int:
    """Physical HBM rows one token's full state (kv + score) occupies."""
    state_bytes = (head_dim << 4) if overlap else (head_dim << 3)
    return state_bytes >> (10 if mode is _MODE_CSA_INDEXER else 9)


def _state_page_capacity(
    mode: Mode,
    kv_cache_block_size: int,
    compress_ratio: int | None = None,
    head_dim: int | None = None,
) -> int:
    """Token states one page of `mode`'s host array physically holds."""
    defaults = _MODE_DEFAULTS[mode]
    compress_ratio = (
        defaults["compress_ratio"] if compress_ratio is None else compress_ratio
    )
    head_dim = defaults["head_dim"] if head_dim is None else head_dim
    page_rows = (
        kv_cache_block_size >> 2
        if mode is _MODE_HCA
        else physical_page_size(mode, kv_cache_block_size, compress_ratio)
    )
    state_bytes = (head_dim << 4) if defaults["overlap"] else (head_dim << 3)
    return page_rows // (state_bytes >> (10 if mode is _MODE_CSA_INDEXER else 9))


_PEER_MODE: dict[Mode, Mode] = {
    Mode.CSA: Mode.CSA_INDEXER,
    Mode.CSA_INDEXER: Mode.CSA,
}


@functools.lru_cache(maxsize=64)
def state_block_size(
    mode: Mode,
    kv_cache_block_size: int,
    compress_ratio: int | None = None,
    head_dim: int | None = None,
) -> int:
    """Token states vLLM should page `mode`'s state cache at.

    vLLM buckets DeepSeek-V4 cache groups by ``(block_size, sliding_window)``,
    we let CSA attention and the indexer's compressor state to have same block
    size to land in the kv cache group.
    """
    own = _state_page_capacity(mode, kv_cache_block_size, compress_ratio, head_dim)
    # CSA and the indexer are each other's peer; HCA has none and pages alone.
    peer_mode = _PEER_MODE.get(mode)
    if peer_mode is None:
        return own
    return min(own, _state_page_capacity(peer_mode, kv_cache_block_size))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TileSizes:
    """Tile sizes for the kernel."""

    tile_n: int


_DEFAULT_TILE_SIZES = TileSizes(tile_n=4)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dimensions:
    """True inputs only. Anything derivable is a property below."""

    mode: Mode = dataclasses.field(metadata=dict(static=True))
    size_n: int  # num_tokens
    head_dim: int
    rope_head_dim: int
    compress_ratio: int
    physical_page_size: int
    state_physical_page_size: int
    state_block_size: int = dataclasses.field(metadata=dict(static=True))
    quant_block: int
    overlap: bool
    has_rope_cache: bool
    rms_eps: float = 1e-6

    cos_sin_dtype: jax.typing.DTypeLike = jnp.float32
    rope_width: int = 128

    @property
    def is_quantized(self) -> bool:
        return self.quant_block > 0

    @property
    def has_rope(self) -> bool:
        return self.rope_head_dim > 0

    @property
    def nope_dtype(self) -> jax.typing.DTypeLike:
        return jnp.uint8 if self.is_quantized else jnp.bfloat16

    @property
    def rope_dtype(self) -> jax.typing.DTypeLike:
        return self.nope_dtype


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Configs:
    """Configuration for the kernel."""

    tile_sizes: TileSizes
    dims: Dimensions

    # --- factory ---------------------------------------------------------------
    @classmethod
    def make(
        cls,
        mode: Mode,
        *,
        size_n,
        physical_page_size,
        state_block_size,
        state_physical_page_size=None,
        rms_eps=1e-6,
        tile_n=4,
        **overrides,
    ) -> "Configs":
        """Build a config for `mode`; `overrides` replace per-mode defaults.

        `state_physical_page_size` defaults to `physical_page_size`, i.e. the
        state and the compressed KV share one buffer.
        """
        tile_sizes = _DEFAULT_TILE_SIZES if tile_n == 4 else TileSizes(tile_n=tile_n)
        state_phys = (
            physical_page_size
            if state_physical_page_size is None
            else state_physical_page_size
        )
        if not overrides:
            hd, rhd, cr, qb, ovl, hrc = _MODE_DEFAULT_TUPLES[mode]
            dims = Dimensions(
                mode,
                size_n,
                hd,
                rhd,
                cr,
                physical_page_size,
                state_phys,
                state_block_size,
                qb,
                ovl,
                hrc,
                rms_eps,
            )
            return cls(tile_sizes=tile_sizes, dims=dims)

        actual_overrides = {**_MODE_DEFAULTS[mode], **overrides}
        if mode is _MODE_HCA:
            actual_overrides["quant_block"] = 0

        dims = Dimensions(
            mode=mode,
            size_n=size_n,
            physical_page_size=physical_page_size,
            state_block_size=state_block_size,
            state_physical_page_size=state_phys,
            rms_eps=rms_eps,
            **actual_overrides,
        )
        return cls(tile_sizes=tile_sizes, dims=dims)

    # --- compute tiling (logical, in LANE-wide tiles) --------------------------
    @property
    def overlap_factor(self) -> int:
        """State copies kept per step: 1, or 2 when windows overlap (prev+curr)."""
        return 2 if self.dims.overlap else 1

    @property
    def state_width(self) -> int:
        """Width of one stored field (head_dim, doubled when overlapping)."""
        dims = self.dims
        return dims.head_dim << 1 if dims.overlap else dims.head_dim

    @property
    def head_tiles(self) -> int:
        """head_dim split into LANE-wide compute tiles (old: slots_per_part)."""
        return self.dims.head_dim >> 7

    @property
    def window(self) -> int:
        """Timesteps compressed together (x overlap_factor when overlapping)."""
        dims = self.dims
        return dims.compress_ratio << 1 if dims.overlap else dims.compress_ratio

    # --- rope sub-layout -------------------------------------------------------
    @property
    def nope_dim(self) -> int:
        """Width of the non-rope (nope) part of a head."""
        return self.dims.head_dim - self.dims.rope_head_dim

    @property
    def nope_store_dim(self) -> int:
        """Dimension of the nope storage (contains rope if no separate rope cache)."""
        return (
            self.dims.head_dim - self.dims.rope_head_dim
            if self.dims.has_rope_cache
            else self.dims.head_dim
        )

    @property
    def half_rope(self) -> int:
        """cos/sin split point (half the rope dim)."""
        return self.dims.rope_head_dim >> 1

    @property
    def rope_slot(self) -> int:
        """Head-tile holding the rope channels (the last one)."""
        return (self.dims.head_dim >> 7) - 1

    # --- output record size ----------------------------------------------------
    @property
    def record_bytes(self) -> int:
        """Bytes in one packed output record (old: total_bytes_out)."""
        if not self.dims.is_quantized:
            return self.dims.head_dim << 1  # bf16
        # fp8 payload + scale + padding, capped to a fixed record width.
        return 256 if self.dims.head_dim == LANE else 512

    @property
    def record_rows(self) -> int:
        """Packed physical rows per output record."""
        row_bytes = 1024 if self.dims.mode is _MODE_CSA_INDEXER else 512
        return (self.record_bytes + row_bytes - 1) // row_bytes

    # --- HBM storage packing ---------------------------------------------------
    @property
    def row_size_bytes(self) -> int:
        """Size of one physical HBM row in bytes."""
        return 1024 if self.dims.mode is _MODE_CSA_INDEXER else 512

    @property
    def hbm_pack(self) -> int:
        """Sub-slots physically packed per slot row, <= SLOT_PACK."""
        return SLOT_PACK

    @property
    def last_dim_size(self) -> int:
        """Size of the last HBM cache dimension in bytes."""
        return 256 if self.dims.mode is _MODE_CSA_INDEXER else 128

    @property
    def cache_last_dims(self) -> tuple[int, ...]:
        return (
            _CACHE_LAST_DIMS_INDEXER
            if self.dims.mode is _MODE_CSA_INDEXER
            else _CACHE_LAST_DIMS_DEFAULT
        )

    # --- slot translation helpers -----------------------------------------
    @property
    def tokens_in_second_minor(self) -> int:
        """Divisor to map wrapper row offset to new physical row offset."""
        tokens_per_row = self.row_size_bytes // self.record_bytes
        return tokens_per_row if tokens_per_row > 1 else 1

    @property
    def kv_stride(self) -> int:
        """Token stride in logical slot units."""
        if self.record_bytes <= self.row_size_bytes:
            return 1
        return self.record_rows

    # --- block sizes and physical page size ------------------------------------
    @property
    def physical_page_size(self) -> int:
        """Number of physical HBM rows per page of the compressed-KV array."""
        return self.dims.physical_page_size

    @property
    def state_physical_page_size(self) -> int:
        """Number of physical HBM rows per page of the *state* array."""
        return self.dims.state_physical_page_size

    @property
    def state_rows_per_token(self) -> int:
        """Number of physical HBM rows occupied by one token's full state
        (kv + score)."""
        dims = self.dims
        state_bytes = (dims.head_dim << 4) if dims.overlap else (dims.head_dim << 3)
        return state_bytes >> (10 if dims.mode is _MODE_CSA_INDEXER else 9)

    @property
    def field_rows(self) -> int:
        """Number of physical HBM rows spanned by one field during gather."""
        row_bytes = 1024 if self.dims.mode is _MODE_CSA_INDEXER else 512
        return ((self.dims.head_dim << 2) + row_bytes - 1) // row_bytes

    @property
    def state_page_capacity(self) -> int:
        """Number of token states one page of the state array physically holds."""
        return self.dims.state_physical_page_size // self.state_rows_per_token

    @property
    def state_block_size(self) -> int:
        """Number of state tokens per page of the state array."""
        return self.dims.state_block_size

    @property
    def kv_block_size(self) -> int:
        """Number of compressed KV tokens per page."""
        page_bytes = self.dims.physical_page_size * self.row_size_bytes
        return page_bytes // self.record_bytes

    @property
    def rope_page_size(self) -> int:
        """Number of physical HBM rows per page in RoPE cache."""
        return self.dims.physical_page_size >> 2

    @property
    def pages_to_buffer_per_token(self) -> int:
        """Pages that must be resident to cover one token's window (+1 guard)."""
        sbs = self.dims.state_block_size
        return (self.window + sbs - 1) // sbs + 1

    # --- shapes (single source of truth for every reshape / BlockSpec) ---------
    @property
    def _tile_n(self) -> int:
        return self.tile_sizes.tile_n

    def window_shape(self) -> tuple[int, ...]:
        """f32 window scratch: (fields, tile, window, head_tiles, lane)."""
        dims = self.dims
        w = dims.compress_ratio << 1 if dims.overlap else dims.compress_ratio
        return (N_FIELDS, self.tile_sizes.tile_n, w, dims.head_dim >> 7, LANE)

    def window_bytes_shape(self) -> tuple[int, ...]:
        """uint8 view of the window scratch; each f32 lane -> FP32_BYTES rows.

        Note: that trailing FP32_BYTES (4) is bytes-per-f32, NOT the HBM SLOT_PACK
        (also 4) -- they're numerically equal but mean different things.

        Returns:
          The shape of the window bytes scratch.
        """
        dims = self.dims
        w = dims.compress_ratio << 1 if dims.overlap else dims.compress_ratio
        return (
            N_FIELDS,
            self.tile_sizes.tile_n,
            w,
            dims.head_dim >> 7,
            FP32_BYTES,
            LANE,
        )

    def output_shape(self) -> tuple[int, ...]:
        """Packed output tile / VMEM block: (tile, record_rows) + cache_last_dims."""
        last_dim = 256 if self.dims.mode is _MODE_CSA_INDEXER else 128
        return (self.tile_sizes.tile_n, self.record_rows, SLOT_PACK, last_dim)

    def page_buffer_shape(self) -> tuple[int, ...]:
        """Page-buffer VMEM block: (tile, pages, state page rows) + cache_last_dims.

        Pages are DMA'd whole out of the *state* array, so this is sized by
        `state_physical_page_size`, not by the compressed-KV page.
        """
        dims = self.dims
        last_dim = 256 if dims.mode is _MODE_CSA_INDEXER else 128
        return (
            self.tile_sizes.tile_n,
            self.pages_to_buffer_per_token,
            dims.state_physical_page_size,
            SLOT_PACK,
            last_dim,
        )

    def rope_output_shape(self) -> tuple[int, ...]:
        """RoPE output VMEM block: (tile, 4, lane)."""
        assert self.dims.has_rope_cache
        return (self.tile_sizes.tile_n, 4, LANE)

    def cos_sin_shape(self) -> tuple[int, ...]:
        """cos/sin VMEM block: (tile, rope_head_dim)."""
        return (self.tile_sizes.tile_n, self.dims.rope_head_dim)

    def cache_shape(self, num_pages: int) -> tuple[int, ...]:
        """Shape of the global HBM KV cache, as uint8 slabs.

        CSA's arrays are allocated as ``(num_pages, rows, 128)`` int32
        (`csa_cache_layout`), the same bytes as this shape.
        """
        dims = self.dims
        last_dim = 256 if dims.mode is _MODE_CSA_INDEXER else 128
        return (num_pages, dims.physical_page_size, SLOT_PACK, last_dim)

    def state_cache_shape(self, num_pages: int) -> tuple[int, ...]:
        """Shape of the global HBM array hosting the f32 compressor state."""
        dims = self.dims
        last_dim = 256 if dims.mode is _MODE_CSA_INDEXER else 128
        return (num_pages, dims.state_physical_page_size, SLOT_PACK, last_dim)

    def rope_cache_shape(self, num_pages: int) -> tuple[int, ...]:
        """Shape of the global HBM RoPE cache, as uint8 slabs."""
        assert self.dims.has_rope_cache
        return (num_pages, self.dims.physical_page_size >> 2, 4, 128)
