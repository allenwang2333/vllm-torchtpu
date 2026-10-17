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

import dataclasses
import enum
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Rows in one HBM tile of a 16-bit [tokens, dim] activation (XLA lays it out
# in (8, 128) tiles of row pairs), so viewing it as
# [tokens / QKV_TILE_ROWS, dim / lanes, QKV_TILE_ROWS, lanes] is a bitcast.
QKV_TILE_ROWS = 8

# Output rows one DMA clears when a launch zeroes the rows past the last
# real token itself (`GDNConfig.zero_out_tail`).
OUT_ZERO_ROWS = 16

# Fraction of VMEM a launch without verify windows may scope.
DEFAULT_VMEM_FRACTION = 0.7

# Fewest tokens a launch pair applies the gated output norm for (`gate_out`).
# Below it the norm runs after the launches, in XLA or in the output
# epilogue. 0 gates in the kernel at every bucket: the z lane groups add
# little to the decode body's DMA, which the per-sequence states dominate.
GATE_OUT_MIN_TOKENS = 0

# Run the PER_SEQ launch before the BATCHED one. On 1024-token steps XLA
# overlaps an HBM-bound SparseCore relayout of the qkvz projection with
# whichever launch comes first; PER_SEQ is compute-bound, BATCHED streams
# decode states and slows by the bandwidth it loses.
PER_SEQ_FIRST = True


class AttentionMode(enum.StrEnum):
    # Gated Delta Net: one scalar gate per value head.
    GDN = enum.auto()
    # Kimi Delta Attention: the gate is per-channel, so it carries
    # `kq_head_dim` values per value head instead of one.
    KDA = enum.auto()


class GDNMode(enum.StrEnum):
    # Multiple sequences per tile, one tile per sequence. Each sequence
    # carries `window_size` tokens: a single decoded token, or a speculative
    # verify window. See `GDNConfig.window_size`.
    BATCHED = enum.auto()
    PER_SEQ = enum.auto()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dtypes:
    act_in: jnp.dtype
    act_out: jnp.dtype
    compute: jnp.dtype
    recurrent_state: jnp.dtype
    conv_state: jnp.dtype


@dataclasses.dataclass(frozen=True)
class StateRegion:
    """Copy-plan for one state region of an indexed external state source.

    A slot's region spans ``nblocks`` consecutive source blocks starting
    ``kb0`` blocks into the slot's window, rows ``[row0, row0 + nrows)``
    of each — one contiguous DMA per slot. The raw bytes are accessed
    through a typed view (``view_dtype`` elements, ``lane_split``-way
    128-lane split); the leading ``rows_used`` typed rows carry the
    state, the rest is zero padding.
    """

    kb0: int
    nblocks: int
    row0: int
    nrows: int
    view_dtype: jnp.dtype
    lane_split: int
    rows_used: int
    # Optional static typed-row permutation between the pool's stored row
    # order and the kernel's logical row order: logical_rows[i] is stored
    # at typed row rows_perm[i].  None or the identity means the stored
    # order IS the logical order.  Used by the QK pair-blocked conv layout,
    # where full-width states interleave Q and K row-pairs per tap so that
    # one whole pool token holds one head-pair's Q and K rows.
    rows_perm: tuple[int, ...] | None = None


@dataclasses.dataclass(frozen=True)
class StateSourcePlan:
    """States live in an indexed external source instead of dense per-slot
    tensors: each state index owns a window of ``stride`` consecutive
    source blocks holding both state regions.

    The plan describes one checkpoint's geometry. With speculative
    decoding every checkpoint is an independent source block named by
    ``MetadataRef.s_idx_to_ckpt_indices`` and reuses this same geometry,
    so nothing here varies with the verify window."""

    stride: int
    conv: StateRegion
    recurrent: StateRegion
    whole_block_dma: bool = False


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNConfig:
    mode: GDNMode
    dtypes: Dtypes
    batch_size: int
    dim_size: int
    kernel_size: int
    tile_size: int
    num_kq_heads: int
    num_v_heads: int
    kq_head_dim: int
    v_head_dim: int
    attention_mode: AttentionMode = AttentionMode.GDN
    # KDA only, and must be negative. Selects the bounded gate form over the
    # unbounded one; see `fused_conv1d_gdn` for the two and their validation.
    gate_lower_bound: float | None = None
    num_buffers: int = 2
    state_plan: StateSourcePlan | None = None
    # Max tokens per speculative verify window (= num_speculative_tokens + 1),
    # which is also the number of state checkpoints kept per sequence. The
    # kernel reads a sequence's initial state from
    # `state_indices[s] + read_offset[s]` and writes one checkpoint per window
    # position to `state_indices[s] + t`, which is how rejected draft tokens
    # are rolled back (by checkpoint selection). It is 1 without speculative
    # decoding, where only the state after the last real token is kept, so
    # shapes and loops can be sized off it unconditionally and the extra axis
    # / iterations fold away in the non-speculative paths.
    window_size: int = 1
    # qkv arrives in the activation dtype as its HBM row tiles (see
    # QKV_TILE_ROWS) and is widened to the compact f32 layout in VMEM.
    qkv_row_tiled: bool = False
    # The first launch creates the output; this launch zeroes its rows from
    # `MetadataRef.out_zero_row` on (none when that is the buffer's end). The
    # rows before it are all computed by the BATCHED or the PER_SEQ launch.
    zero_out_tail: bool = False
    # The qkv row tiles carry the z gate (`v_dim_size` columns) right after
    # q, k and v; the launch writes RMSNormGated(out, z) instead of out (see
    # `fused_conv1d_gdn`'s `out_norm_weight`), with this epsilon.
    gate_out: bool = False
    out_norm_eps: float = 0.0

    @property
    def chunk_size(self) -> int:
        if self.mode == GDNMode.PER_SEQ:
            return self.tile_size
        # One tile per sequence, holding its whole verify window. BATCHED is
        # the single-token case of that (window_size == 1).
        return self.window_size

    @property
    def seq_tile_size(self) -> int:
        if self.mode == GDNMode.PER_SEQ:
            return 1
        return self.tile_size

    @property
    def qkv_tile_groups(self) -> int:
        """Row tiles a grid step's qkv rows can span, the first one partial."""
        rows = self.seq_tile_size * self.chunk_size
        return pl.cdiv(QKV_TILE_ROWS - 1 + rows, QKV_TILE_ROWS)

    @property
    def qkv_lane_groups(self) -> int:
        """Lane groups of a token row the conv reads: q, k and v."""
        return self.dim_size // pltpu.get_tpu_info().num_lanes

    @property
    def qkv_src_lane_groups(self) -> int:
        """Lane groups of a token row a grid step copies in."""
        z_groups = self.v_dim_size // pltpu.get_tpu_info().num_lanes
        return self.qkv_lane_groups + (z_groups if self.gate_out else 0)

    @property
    def z_stage_groups(self) -> int:
        """z lane groups a PER_SEQ step widens at a time: as many as the qkv
        scratch, free until the conv writes it, can stage at
        `qkv_stage_tile_stride` (halving from all of them)."""
        groups = self.qkv_src_lane_groups - self.qkv_lane_groups
        free_rows = self.chunk_size * self.qkv_row_stride
        while self.qkv_tile_groups * groups * self.qkv_stage_tile_stride > free_rows:
            assert groups % 2 == 0, self
            groups //= 2
        return groups

    @property
    def qkv_row_stride(self) -> int:
        """Lane rows between consecutive token rows of the 2D qkv scratch the
        chunked step reads heads from. Odd, so one head's lane group over 8
        token rows is a single strided load: at the even unpadded stride
        each such load issues as 8 one-sublane loads merged by 7 ORs."""
        return (self.dim_size // pltpu.get_tpu_info().num_lanes) | 1

    @property
    def qkv_stage_tile_stride(self) -> int:
        """Lane rows between consecutive widened row tiles in the PER_SEQ
        stage. Odd, so one token row's lane groups, a tile apart, are a
        single strided load, and no tile is transposed on the way in."""
        return QKV_TILE_ROWS | 1

    @property
    def out_stage_row_stride(self) -> int:
        """Rows between consecutive heads in the chunked step's output stage.
        Odd, so 8 heads' rows of one token are a single strided load and the
        [heads, chunk] -> [chunk, heads] swap moves no data between
        sublanes."""
        return self.chunk_size | 1

    @property
    def prev_kernel_size(self) -> int:
        return self.kernel_size - 1

    @property
    def is_kda(self) -> bool:
        return self.attention_mode == AttentionMode.KDA

    @property
    def gate_dim(self) -> int:
        """Width of the gate activation `a`, in elements per token."""
        if self.is_kda:
            return self.num_v_heads * self.kq_head_dim
        return self.num_v_heads

    @property
    def aligned_gate_dim(self) -> int:
        num_lanes = pltpu.get_tpu_info().num_lanes
        return pl.cdiv(self.gate_dim, num_lanes) * num_lanes

    @property
    def triangular_block_size(self) -> int:
        """Sub-block size of the chunked KDA intra-chunk solve.

        The diagonal sub-blocks are evaluated on the VPU/XLU with exact
        pairwise gate differences and the off-diagonal ones as MXU GEMMs,
        so this trades the two units off against each other: raising it
        moves work from MXU to XLU. More value heads already means more
        XLU work, hence the smaller blocks there.
        """
        # TODO(alynie): create a better heuristic to tune this value.
        if self.num_v_heads == 64:
            return 2
        if self.num_v_heads == 32:
            return 8
        return 16

    @property
    def use_recurrent(self) -> bool:
        """Whether GDN runs the token-recurrent scan instead of the chunked one.

        Keeping more than one state checkpoint mandates the recurrent scan,
        since the chunked path only ever produces the final state.
        """
        return self.chunk_size == 1 or self.window_size > 1

    @property
    def v_dim_size(self) -> int:
        return self.num_v_heads * self.v_head_dim

    @property
    def kq_dim_size(self) -> int:
        return self.num_kq_heads * self.kq_head_dim

    @property
    def v_per_kq_head(self) -> int:
        return self.num_v_heads // self.num_kq_heads

    @property
    def aligned_num_v_heads(self) -> int:
        tpu_info = pltpu.get_tpu_info()
        num_lanes = tpu_info.num_lanes
        return pl.cdiv(self.num_v_heads, num_lanes) * num_lanes

    def get_kernel_name(self) -> str:
        # Windows of different sizes compile to different kernels; keep them
        # distinguishable in profiles.
        suffix = f"_w{self.window_size}" if self.window_size > 1 else ""
        if self.is_kda and self.gate_lower_bound is not None:
            suffix += "_bounded"
        return f"fused_conv1d_{self.attention_mode.value}_{self.mode.value}{suffix}"

    def get_metadata(self) -> dict[str, str | int | float]:
        cfgs_dict = dataclasses.asdict(self)
        ret = {}
        for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
            key = jax.tree_util.keystr(path, simple=True, separator=".")
            if not isinstance(val, str | int | float):
                val = str(val)
            ret[key] = val
        return ret

    def get_out_shape(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            (self.batch_size, self.num_v_heads, self.v_head_dim),
            self.dtypes.act_out,
        )

    # Fraction of VMEM the kernel may use. A multi-token window holds one
    # state checkpoint per position in VMEM, so for large-head models the
    # default 0.7 budget is not enough; those get a higher limit and the
    # wrapper sizes the tile against the same factor.
    WINDOWED_VMEM_FRACTION = 0.9

    def get_vmem_limit_bytes(self) -> int:
        tpu_info = pltpu.get_tpu_info()
        fraction = (
            self.WINDOWED_VMEM_FRACTION
            if self.window_size > 1
            else DEFAULT_VMEM_FRACTION
        )
        return int(fraction * tpu_info.vmem_capacity_bytes)

    def get_scratch_shape_dict(self) -> dict[str, Any]:
        conv_shape = (self.seq_tile_size, self.prev_kernel_size, 1, self.dim_size)
        recurrent_shape = (
            self.seq_tile_size,
            self.num_v_heads,
            self.kq_head_dim,
            self.v_head_dim,
        )

        carry_conv_scratch = carry_recurrent_scratch = None
        # NOTE: In batched mode 1 seq = 1 tile, so inter-tile carry is not
        # needed.
        if self.mode == GDNMode.PER_SEQ:
            carry_conv_scratch = pltpu.VMEM(conv_shape, jnp.float32)
            carry_recurrent_scratch = pltpu.VMEM(recurrent_shape, jnp.float32)

        qkv_stage_scratch = qkv_scratch = None
        if self.qkv_row_tiled:
            num_lanes = pltpu.get_tpu_info().num_lanes
            stage_rows = self.qkv_tile_groups * QKV_TILE_ROWS
            if self.seq_tile_size == 1:
                # Each widened row tile `qkv_stage_tile_stride` lane rows
                # apart; the tile's rows start at a dynamic offset into it.
                stage_shape = (
                    self.qkv_tile_groups
                    * self.qkv_lane_groups
                    * self.qkv_stage_tile_stride,
                    num_lanes,
                )
            else:
                stage_shape = (stage_rows * self.qkv_src_lane_groups, num_lanes)
            qkv_stage_scratch = pltpu.VMEM(stage_shape, jnp.float32)
            if self.seq_tile_size == 1 and not self.use_recurrent:
                # Token rows `qkv_row_stride` lane rows apart; see there.
                qkv_scratch_shape = (self.chunk_size * self.qkv_row_stride, num_lanes)
            else:
                qkv_scratch_shape = (
                    self.seq_tile_size,
                    self.chunk_size,
                    1,
                    self.dim_size,
                )
            qkv_scratch = pltpu.VMEM(qkv_scratch_shape, jnp.float32)

        out_stage_scratch = None
        if self.mode == GDNMode.PER_SEQ and not self.use_recurrent and not self.is_kda:
            out_stage_scratch = pltpu.VMEM(
                (self.num_v_heads * self.out_stage_row_stride, self.v_head_dim),
                jnp.float32,
            )

        out_zero_scratch = out_zero_sem = None
        if self.zero_out_tail:
            out_zero_scratch = pltpu.VMEM(
                (OUT_ZERO_ROWS, self.num_v_heads, self.v_head_dim),
                self.dtypes.act_out,
            )
            out_zero_sem = pltpu.SemaphoreType.DMA

        return dict(
            carry_conv_scratch_ref=carry_conv_scratch,
            carry_recurrent_scratch_ref=carry_recurrent_scratch,
            qkv_stage_scratch_ref=qkv_stage_scratch,
            qkv_scratch_ref=qkv_scratch,
            out_stage_scratch_ref=out_stage_scratch,
            out_zero_scratch_ref=out_zero_scratch,
            out_zero_sem_ref=out_zero_sem,
        )
