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

from vllm_torchtpu.megakernel import pool_alias
from vllm_torchtpu.megakernel.ops import rup


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
    """Execution tiling and pipeline depth parameters."""

    bank: int = 2 << 20  # Bytes per dense ring slot
    nbd: int = 14  # Dense ring slots
    nbe: int = 10  # Expert ring slots
    ebank: int = 1 << 20  # Bytes per expert ring slot
    kv_blk: int = 2048  # KV tokens per block
    nkv: int = 3  # KV ring depth
    vmem_mib: int = 62  # VMEM limit in MiB
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


DEFAULT_TILES = Tiles(hier=1, pack=1, mb8=2)
VERIFY_TILES = dataclasses.replace(DEFAULT_TILES, nkv=4, rsag=True, hier=2, ecopy_opt=1)
WIDE_TILES = dataclasses.replace(VERIFY_TILES, nbd=10, nkv=3, vmem_mib=62, hier=0)
TINY_TILES = Tiles(
    bank=128 << 10, nbd=10, nbe=6, ebank=256 << 10, kv_blk=256, nkv=3, vmem_mib=32
)


def default_tiles(B: int) -> Tiles:
    """Tuned default tiles for batch row count `B`."""
    if B > 32:
        return WIDE_TILES
    if B > 24:
        return dataclasses.replace(WIDE_TILES, hier=2)
    if B > 16:
        return dataclasses.replace(WIDE_TILES, nbd=12, hier=2)
    if B > 8:
        return VERIFY_TILES
    if B > 4:
        return dataclasses.replace(DEFAULT_TILES, rsag=True, hier=2, ecopy_opt=1)
    return DEFAULT_TILES


@dataclass(frozen=True)
class DebugOptions:
    """Debug and ablation flags kept separate from model geometry and execution."""

    forced_routing: bool = False  # Use external rbias HBM input for MoE routing
    row_mask: bool = False  # Use external moe_rows SMEM mask for MoE routing
    skip: frozenset[str] = (
        frozenset()
    )  # Ablation skip set (e.g. {"dense", "comm", "kv", "experts"})
    scopes: int | bool = False  # Emit per-phase jax.named_scope annotations


@dataclass(frozen=True)
class ExecSpec:
    """Execution-time configuration (independent of model architecture)."""

    B: int = 4  # Active batch rows (= NS * T)
    T: int = 1  # Tokens per sequence (>1 for MTP verify)
    R: int = 8  # Tensor-parallel / expert-parallel ranks
    ctx: int = 2048  # Max KV sequence length
    stack_e: bool = False  # True: 4D stacked expert weights; False: per-layer 3D
    interpret: Any = False  # Run Pallas in CPU interpret mode (bool or InterpretParams)
    tiles: Tiles | None = None  # None -> default_tiles(B)
    debug: DebugOptions = DebugOptions()

    @property
    def resolved_tiles(self) -> Tiles:
        return self.tiles if self.tiles is not None else default_tiles(self.B)

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
        return self.B > 16

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
class ModelSpec:
    """Complete declarative model architecture specification."""

    name: str
    config: Any
    layer_type: LayerType

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
