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
"""Core specifications for the config-driven TPU megakernel library."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp

from . import pool_alias
from .ops import rup


@dataclass(frozen=True)
class Spec:
    """Shape + dtype descriptor for nested/tupled per-layer HBM tensors."""

    shape: tuple[int, ...]
    dtype: Any


@dataclass(frozen=True)
class WeightDecl:
    """Declarative specification for a block weight tensor.

    The planner derives the leading layer count and layer-to-index map from the
    slot/branch in `ModelSpec.layer_type` where the owning `Block` appears.
    """

    name: str
    K: int  # Reduction dimension (or row dim for non-streamed HBM weight)
    N: int  # Output column dimension per rank
    dtype: Any
    stream_ring: Literal["static", "dynamic", "hbm"] = "static"
    layout: Literal["stacked", "per_layer"] = "stacked"
    lead_dims: tuple[
        int, ...
    ] = ()  # Extra per-layer leading dims (e.g. `(EW,)` for experts)


@dataclass(frozen=True)
class StateDecl:
    """Declarative specification for a state, cache, or per-layer output tensor.

    `shape` is the per-layer, per-rank shape; when `per_layer=True`, the planner
    prefixes the owning block's active layer count `L_block`.
    """

    name: str
    shape: tuple[int, ...]
    dtype: Any
    access: Literal["r", "w", "rw"] = "r"
    per_layer: bool = True
    memory_space: Literal["hbm", "smem", "vmem"] = "hbm"
    tupled: bool = False


@dataclass(frozen=True)
class SmallField:
    """Per-layer 1D vector packed into the double-buffered `smalls` table."""

    name: str
    width: int  # Local elements on each rank


@dataclass(frozen=True)
class ScratchReq:
    """VMEM / SMEM / Semaphore buffer request declared by a Block or Core."""

    name: str
    spec: Any  # pltpu.VMEM, pltpu.SMEM, or pltpu.SemaphoreType.DMA
    phase: str = "persistent"  # "persistent" or phase tag for non-overlapping overlay


@dataclass(frozen=True)
class Tiles:
    """Execution tiling and pipeline depth parameters.

    Every field declares its numerics class in `TILE_NUMERICS` (below).
    """

    bank: int = 2 << 20  # Bytes per dense ring slot
    nbd: int = 14  # Dense ring slots
    nbe: int = 10  # Expert ring slots
    ebank: int = 1 << 20  # Bytes per expert ring slot
    kv_blk: int = 2048  # KV tokens per block
    nkv: int = 3  # KV ring depth
    vmem_mib: int = 62  # VMEM limit in MiB
    smem_mib: int = 1  # SMEM limit in MiB
    bounds_checks: bool = False
    lazy: bool = True  # Refill attention/MoE dense slots after their streams
    pipe: bool = True  # Software-pipeline single-tile experts
    topk: bool = True  # One-reduction top-k picks on packed (value, lane) keys
    inplace: bool = False  # Alias state inputs to outputs where supported
    split: bool = True  # Sequence-split attention across G ranks sharing a KV head
    rsag: bool = False  # Two-round reduce-scatter + all-gather
    hier: int = 0  # Hierarchical all-reduce on v7x-8 (0, 1, 2, 3)
    war: int = -1  # Unfolded [MR, H] all-reduce payload (-1: auto, 0: off, 1: on)
    pack: int = -1  # Pack PB = 8 // nh rows per 8-row tile (-1: auto)
    mb8: int = 0  # Internal activation row count MB (0: M, 1: max(16, MR), 2: MR)
    gdn_rows: int = 8  # GDN state row chunk size above 16 rows
    gdn_bufs: int = 3  # GDN state ring depth
    kv_sub: int = 0  # KV tokens per online-softmax update inside a block (0: full)
    amask: int = 1  # Attention mask mode (0..4)
    ecopy_opt: int = 0  # Redundant expert dedup optimization (0 or 1)
    # Per-bucket schedule thresholds (row counts). Blocks never compare `B` to a
    # literal; they read these, and tile rules (`profiles/`) may tune them.
    wide_rows: int = 16  # `B > wide_rows`: WIDE schedule (chunked GDN, auto pack/war)
    # Unrolled / PIPE_EP experts: paired ring releases + unrolled refills up to here
    moe_group_rows: int = 40
    kv_scatter_vec_ns: int = 16  # Non-paged host KV update: one scatter above this NS
    # Opt-in performance flags below default to the frozen-parity path. Named
    # profiles (`profile.py`, `profiles/`) turn them on; defaults never do.
    pipe_ep: bool = False  # Software-pipeline EP experts with T13 == 2 * T2
    r2_direct_store: bool = False  # Direct VMEM store on MoE r2 tile 0
    # Vectorized MXU expert route packing (a distinct routing/Po2C policy that
    # supersedes `ecopy_opt` whenever its static gate in `MoEBlock.route` holds)
    route_vec: bool = False
    dense_acc_pair: bool = False  # Unrolled / paired dense matmul accumulation
    kv_ering: bool = False  # Alias KV DMA ring onto epool when unused in GQA
    pair_kv: bool = False  # Pair KV attention items to overlap KV DMA + softmax
    dmm_unroll: int = 0  # Max tiles to fully unroll in dmatmul (0: never)
    gqa_qv: bool = False  # Head-batched Q/K norm+RoPE and 128-lane Q/O staging
    gq_pack: bool = False  # Pack G*nh Q heads into 16-row MXU tiles (needs gqa_qv)
    topk_lanes: bool = False  # Batcher odd-even top-k on lanes for E >= 128
    conv_stage: bool = False  # Stage fused-path GDN conv taps through VMEM `sb`
    # Head-major GDN recurrence (per-head state update; reassociates vs frozen)
    gdn_head_major: bool = False
    moe_unroll: bool = False  # Unrolled expert tile loops + grouped expert refills
    rel_fetch: bool = False  # Trace-time-decoded dense refills (fetch_rel/flush_rel)

    def tk(self, K: int, N: int, dtype: Any, bank: int) -> int:
        """Largest 128-multiple divisor of K whose `[tk, N]` tile fits `bank`."""
        best = None
        for tk in range(128, K + 1, 128):
            if (
                K % tk == 0
                and pool_alias.storage_size((tk, N), jnp.dtype(dtype)) <= bank
            ):
                best = tk
        assert best is not None, ("no 128-row tile fits a bank", K, N, dtype, bank)
        return best


NumericsClass = Literal["exact", "approx"]

# Numerics class of every `Tiles` field (`profile.approx_features` reads this).
# `exact`: moving the field off the `default` profile's value keeps hidden
# states and pools bit-identical, because it only changes DMA / ring
# scheduling, budgets or debug checks. `approx`: changes the
# arithmetic (matmul K-tiling, reduction / softmax order, rounding, routing) or
# is not proven exact; profiles that move any of these are `approx`.
TILE_NUMERICS: dict[str, NumericsClass] = {
    "bank": "approx",  # dense K-tiling (`Tiles.tk`)
    "nbd": "exact",  # dense ring depth (scheduling)
    "nbe": "approx",  # expert ring depth also gates PIPE / PIPE_EP and KVE paths
    "ebank": "approx",  # expert K-tiling
    "kv_blk": "approx",  # online-softmax block size
    "nkv": "exact",  # KV ring depth (scheduling)
    "vmem_mib": "exact",  # budget only
    "smem_mib": "exact",  # budget only
    "bounds_checks": "exact",  # debug checks only
    "lazy": "exact",  # dense refill placement (scheduling)
    "pipe": "approx",  # single-tile expert pipelining
    "topk": "approx",  # packed-key top-k
    "inplace": "approx",  # state input/output aliasing (not proven exact)
    "split": "approx",  # sequence-split attention (partial-softmax merge)
    "rsag": "approx",  # rounds all-reduce partial sums to bf16
    "hier": "approx",  # all-reduce reduction order
    "war": "approx",  # all-reduce payload layout
    "pack": "approx",  # split-attention row packing
    "mb8": "approx",  # activation row count
    "gdn_rows": "approx",  # chunked GDN state rows (GCH)
    "gdn_bufs": "exact",  # GDN state ring depth (scheduling)
    "kv_sub": "approx",  # online-softmax sub-block updates
    "amask": "approx",  # attention mask mode
    "ecopy_opt": "approx",  # redundant-expert dispatch policy
    "wide_rows": "approx",  # WIDE schedule threshold (GCH, auto pack / war)
    "moe_group_rows": "exact",  # paired expert-ring releases (scheduling)
    "kv_scatter_vec_ns": "exact",  # host KV update: one scatter vs per-sequence
    "pipe_ep": "approx",  # EP expert software pipeline
    "r2_direct_store": "approx",  # `=` instead of zero-fill + `+=` (signed zero)
    "route_vec": "approx",  # vectorized routing policy
    "dense_acc_pair": "approx",  # paired dense accumulation
    "kv_ering": "exact",  # KV ring staged in the expert pool (TPU bisect, CPU test)
    "pair_kv": "exact",  # paired KV items (TPU bisect, CPU test)
    "dmm_unroll": "approx",  # unrolled dense matmul accumulation
    "gqa_qv": "exact",  # head-batched Q/K norm + RoPE (TPU bisect, CPU test)
    "gq_pack": "approx",  # packed G*nh query tiles
    "topk_lanes": "approx",  # Batcher odd-even lane top-k
    "conv_stage": "approx",  # fused-path conv staging
    "gdn_head_major": "approx",  # head-major GDN state update (reassociates)
    "moe_unroll": "approx",  # unrolled expert tiles (not exact on TPU, bisect)
    "rel_fetch": "exact",  # trace-time-decoded dense refills (scheduling)
}

# Numerics class of the pool-I/O modes selected by profiles.
POOL_IO_NUMERICS: dict[str, NumericsClass] = {
    "fused_pool_io": "approx",  # always runs the head-major GDN recurrence
    # GPR, auto (`Profile.pool_gather=None`): WIDE verify buckets only, where
    # GCH is already on (bitwise vs split at T=4 B=24).
    "pool_gather": "exact",
    # GPR on every eligible bucket (`Profile.pool_gather=True`): may reassociate
    # non-WIDE verify buckets under tuned tile schedules.
    "pool_gather_eligible": "approx",
}

DEFAULT_TILES = Tiles(hier=1, pack=1, mb8=2)
VERIFY_TILES = dataclasses.replace(DEFAULT_TILES, nkv=4, rsag=True, hier=2, ecopy_opt=1)
WIDE_TILES = dataclasses.replace(VERIFY_TILES, nbd=10, nkv=3, vmem_mib=62, hier=0)
TINY_TILES = Tiles(
    bank=128 << 10, nbd=10, nbe=6, ebank=256 << 10, kv_blk=256, nkv=3, vmem_mib=32
)


def default_tiles(B: int, *, moe_tp: bool = False, R: int = 8) -> Tiles:
    """General, frozen-parity default tiles for batch row count `B`.

    The per-bucket schedule is the `default` profile's tile rule
    (`profiles/default.py:frozen_parity_tiles`); this wrapper keeps the public
    name. Every opt-in performance flag stays at its `Tiles` default (off).
    """
    from .profiles.default import frozen_parity_tiles as _rule

    return _rule(B, moe_tp=moe_tp, R=R)


def frozen_parity_tiles(B: int, *, moe_tp: bool = False, R: int = 8) -> Tiles:
    """Default per-bucket tile schedule (`== default_tiles`)."""
    return default_tiles(B, moe_tp=moe_tp, R=R)


# Valid `ExecSpec.pool_gather` values (see the field comment).
POOL_GATHER_VALUES: tuple[Any, ...] = (False, "split", None, "eligible", True)


@dataclass(frozen=True)
class DebugOptions:
    """Debug and ablation flags kept separate from model geometry and execution."""

    forced_routing: bool = False  # Use external rbias HBM input for MoE routing
    row_mask: bool = False  # Use external moe_rows SMEM mask for MoE routing
    skip: frozenset[str] = (
        frozenset()
    )  # Ablation skip set (e.g. {"dense", "comm", "kv", "experts"})
    scopes: int | bool = False  # Emit per-phase jax.named_scope annotations
    # Host-side ckpt_indices bounds validation on every eager pooled-step call
    # (device-to-host sync + Python loop). In-kernel clamps always apply.
    check_ckpt_bounds: bool = False


@dataclass(frozen=True)
class ExecSpec:
    """Execution-time configuration (independent of model architecture)."""

    B: int = 4  # Active batch rows (= NS * T)
    T: int = 1  # Tokens per sequence (>1 for MTP verify)
    R: int = 8  # Tensor-parallel / expert-parallel ranks
    ctx: int = 2048  # Max KV sequence length
    stack_e: bool = False  # True: 4D stacked expert weights; False: per-layer 3D
    paged_kv: bool = False  # True: stream KV from paged FP8 pools via page_table
    page_size: int = 128  # Tokens per KV page
    num_pages: int = 4096  # Total pages per attention pool per rank
    max_pages_per_seq: int = 0  # Page-table width per sequence (0 -> ctx // page_size)
    fused_pool_io: bool = False  # True: fuse KV RMW + GDN gather/scatter in-kernel
    # In-kernel GDN gather (GPR): False (default) = split gather kernel, which
    # a profile may override; "split" = split, explicitly (wins over a profile);
    # None = auto (WIDE verify buckets where eligible); "eligible" = every bucket
    # the pool backend can gather on; True = required (`MegakernelConfigError`
    # if the bucket cannot).
    pool_gather: bool | None | Literal["eligible", "split"] = False
    act_mode: str = "from_ckpt"  # Sequence activity mode ("from_ckpt", "all", "ref")
    ckpt_2d: bool = False  # True: 2D ckpt_indices (NS, K); False: 3D (G_PER_ATT, NS, K)
    ckpt_k: int = 0  # Last-axis width of ckpt_indices (0 -> T)
    roff_1d: bool = False  # True: 1D read_offsets (NS,); False: 2D (G_PER_ATT, NS)
    interpret: Any = False  # Run Pallas in CPU interpret mode (bool or InterpretParams)
    tiles: Tiles | None = None  # None -> default_tiles(B, R=R)
    debug: DebugOptions = DebugOptions()

    def __post_init__(self) -> None:
        if self.pool_gather not in POOL_GATHER_VALUES:
            raise ValueError(
                f"ExecSpec.pool_gather must be one of {POOL_GATHER_VALUES}, "
                f"got {self.pool_gather!r}"
            )

    @property
    def FUSED(self) -> bool:
        return bool(self.paged_kv and self.fused_pool_io)

    @property
    def GPR(self) -> bool:
        """In-kernel GDN gather on (only after `pool_gather` is resolved to `True`)."""
        return bool(
            self.paged_kv and not self.fused_pool_io and self.pool_gather is True
        )

    @property
    def resolved_tiles(self) -> Tiles:
        return self.tiles if self.tiles is not None else default_tiles(self.B, R=self.R)

    def resolve_tiles(self, g: Any = None) -> Tiles:
        """Same as `resolved_tiles` (`g` kept for callers; defaults ignore `moe_tp`)."""
        del g
        return self.resolved_tiles

    @property
    def resolved_max_pages_per_seq(self) -> int:
        return (
            self.max_pages_per_seq
            if self.max_pages_per_seq > 0
            else max(1, self.ctx // self.page_size)
        )

    @property
    def NS(self) -> int:
        assert self.B % self.T == 0, (self.B, self.T)
        return self.B // self.T

    @property
    def MR(self) -> int:
        return rup(self.B, 8)

    @property
    def M(self) -> int:
        return max(16, rup(self.B, 16))

    @property
    def MB(self) -> int:
        t = self.resolved_tiles
        return self.MR if t.mb8 == 2 else (max(16, self.MR) if t.mb8 else self.M)

    @property
    def VER(self) -> bool:
        return self.T > 1

    @property
    def WIDE(self) -> bool:
        return self.resolved_tiles.wide_rows < self.B

    @property
    def forced_routing(self) -> bool:
        return self.debug.forced_routing

    @property
    def row_mask(self) -> bool:
        return self.debug.row_mask

    @property
    def skip(self) -> frozenset[str]:
        return frozenset(self.debug.skip)

    @property
    def scopes(self) -> int | bool:
        return self.debug.scopes


class Block:
    """Composable megakernel building block (attention, GDN, MoE, MLP, etc.)."""

    name: str = "block"
    reduction: Literal["allreduce", "none", "custom"] = "allreduce"
    input_norm: str | None = None

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        return []

    def states(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        return []

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        return []

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        return []

    def shared_widths(self, g: Any, ex: ExecSpec) -> dict[str, int]:
        """Minimum column widths required on shared core VMEM buffers."""
        return {}

    def plan_params(self, g: Any, ex: ExecSpec, plan: Any) -> Any:
        """Optional block-specific derived parameters computed at build time."""
        return None

    def scratch(
        self, g: Any, ex: ExecSpec, plan: Any
    ) -> dict[str, Any] | list[ScratchReq]:
        return {}

    def prologue(self, ctx: Any) -> None:
        """Executed once before the inter-rank barrier and main layer loop."""

    def init_buffers(self, ctx: Any) -> None:
        """Executed once after the inter-rank barrier and before the main layer loop."""

    def on_layer_start(self, ctx: Any, lyr: Any, slot: Any) -> None:
        """Executed at the start of layer `lyr` alongside `smalls` double-buffering."""

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        """Execute block for layer `l` and return partial `[MB, H]` output."""
        raise NotImplementedError

    def epilogue(self, ctx: Any) -> None:
        """Executed once after the main layer loop."""

    def host_update_states(
        self,
        g: Any,
        ex: ExecSpec,
        pos: Any,
        states: dict[str, Any],
        raw_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Host-side state update inside `shard_map` after `pallas_call` returns."""
        return {}


@dataclass(frozen=True)
class SwitchSlot:
    """Conditional slot inside a single layer loop body (e.g. GDN vs GQA).

    In each repeating cycle of `period = sum(branch_counts)` layers, `branches[i]`
    executes for `branch_counts[i]` consecutive layers.
    """

    name: str
    branches: tuple[Block, ...]
    branch_counts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.branch_counts:
            object.__setattr__(self, "branch_counts", (1,) * len(self.branches))
        assert len(self.branches) == len(self.branch_counts), (
            self.branches,
            self.branch_counts,
        )
        assert all(c >= 0 for c in self.branch_counts) and sum(self.branch_counts) > 0

    @property
    def period(self) -> int:
        return sum(self.branch_counts)

    @property
    def active_pairs(self) -> tuple[tuple[Block, int], ...]:
        """Return `(branch, count)` pairs with `count > 0`."""
        return tuple((b, c) for b, c in zip(self.branches, self.branch_counts) if c > 0)

    @property
    def active_branches(self) -> tuple[Block, ...]:
        return tuple(b for b, _ in self.active_pairs)

    @property
    def active_counts(self) -> tuple[int, ...]:
        return tuple(c for _, c in self.active_pairs)


@dataclass(frozen=True)
class LayerType:
    """Sequence of slots executed in a single transformer layer."""

    name: str
    slots: tuple[Block | SwitchSlot, ...]


@dataclass(frozen=True)
class PoolBackend:
    """Model-injected unified pool layout and gather/scatter callbacks."""

    num_pools: int
    has_recurrent_pool: bool
    recurrent_slots_per_pool: int = 1
    ckpt_pages_per_slot: int = 5
    pool_state_name: str = "raw"
    page_table_state_name: str = "page_table"
    kernel_names: tuple[str, str] = ("gather_gdn_states", "scatter_pool_states")
    gather_states: Any = None
    scatter_states: Any = None
    can_pool_gather: Any = None
    supports_fused_io: bool = False
    supports_pool_gather: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """Complete declarative model architecture specification."""

    name: str
    config: Any
    layer_type: LayerType
    pool_backend: PoolBackend | None = None

    def all_blocks(self) -> list[Block]:
        """Return unique active blocks in deterministic declaration order."""
        seen: set[str] = set()
        out: list[Block] = []
        for s in self.layer_type.slots:
            bs = s.active_branches if isinstance(s, SwitchSlot) else (s,)
            for b in bs:
                if b.name not in seen:
                    seen.add(b.name)
                    out.append(b)
        return out
