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
"""Core specifications for the config-driven TPU megakernel library.

Only core execution knobs live here. A block with its own knobs declares a
frozen `Tunables` dataclass (`Block.Tunables`, with a per-field `NUMERICS`
class); `Tiles.blocks` holds one instance per block name, and blocks read theirs
through `Block.tunables(ex)`. Execution extensions owned by other packages ride
on `ExecSpec.cache` / `ModelSpec.cache_backend`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeVar

from .ops import rup

NumericsClass = Literal["exact", "approx"]
_T = TypeVar("_T")


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
class ExtraInput:
    """Positional kernel input after the states, declared by a block.

    Used for optional debug inputs (e.g. externally forced routing). The kernel
    passes them in block declaration order after the state inputs; the block
    reads the ref as `ctx.extra[name]`. HBM inputs are sharded across ranks,
    SMEM inputs replicated and counted against the SMEM budget.
    """

    name: str
    memory_space: Literal["hbm", "smem"]
    shape: tuple[int, ...] = ()  # Per-rank shape (SMEM budget accounting)
    dtype: Any = None


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


# ---------------------------------------------------------------------------
# Block tunables
# ---------------------------------------------------------------------------
# Block name -> its `Tunables` class (registered by `Block` subclasses).
_TUNABLES: dict[str, type] = {}


def register_tunables(name: str, cls: type) -> None:
    """Register `cls` as the `Tunables` of blocks named `name`.

    `cls` must be a frozen dataclass whose `NUMERICS` class attribute maps
    every field to `"exact"` or `"approx"`. Re-registering the same class is a
    no-op; a different class for a registered name is an error.
    """
    if not (dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen):  # type: ignore[attr-defined]
        raise TypeError(f"{cls.__name__} must be a frozen dataclass")
    fields = {f.name for f in dataclasses.fields(cls)}
    numerics = dict(getattr(cls, "NUMERICS", {}))
    if set(numerics) != fields:
        raise TypeError(
            f"{cls.__name__}.NUMERICS must cover exactly its fields: missing "
            f"{sorted(fields - set(numerics))}, "
            f"unknown {sorted(set(numerics) - fields)}"
        )
    bad = {k: v for k, v in numerics.items() if v not in ("exact", "approx")}
    if bad:
        raise TypeError(f"{cls.__name__}.NUMERICS has invalid classes {bad}")
    prev = _TUNABLES.get(name)
    if prev is not None and prev is not cls:
        raise ValueError(
            f"block {name!r} already registered tunables {prev.__name__}, "
            f"not {cls.__name__}"
        )
    _TUNABLES[name] = cls


def tunables_class(name: str) -> type:
    """The registered `Tunables` class of blocks named `name`."""
    try:
        return _TUNABLES[name]
    except KeyError:
        raise KeyError(
            f"no tunables registered for block {name!r}; known: {sorted(_TUNABLES)}"
        ) from None


def registered_tunables() -> dict[str, type]:
    """All registered `{block name: Tunables class}` (a copy)."""
    return dict(_TUNABLES)


class TunablesMap(Mapping[str, Any]):
    """Immutable, hashable `{block name: Tunables}` mapping (`Tiles.blocks`).

    Entries equal to their class defaults are dropped, so tiles that leave a
    block at its defaults compare equal however they were built.
    """

    __slots__ = ("_items",)

    def __init__(self, items: Mapping[str, Any] | None = None) -> None:
        pairs = []
        for name, val in dict(items or {}).items():
            if not dataclasses.is_dataclass(val) or isinstance(val, type):
                raise TypeError(
                    f"Tiles.blocks[{name!r}] must be a Tunables instance, got "
                    f"{type(val).__name__} (use Tiles.with_blocks for field dicts)"
                )
            if val != type(val)():
                pairs.append((name, val))
        self._items: tuple[tuple[str, Any], ...] = tuple(sorted(pairs))

    def __getitem__(self, key: str) -> Any:
        for name, val in self._items:
            if name == key:
                return val
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(name for name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TunablesMap):
            return self._items == other._items
        return isinstance(other, Mapping) and dict(self._items) == dict(other)

    def __repr__(self) -> str:
        return f"TunablesMap({dict(self._items)!r})"


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tiles:
    """Core execution tiling, ring depths, budgets and collectives.

    Every core field declares its numerics class in `TILE_NUMERICS` (below);
    block knobs live in `blocks` (`{block name: Tunables}`, see `block` and
    `with_blocks`).
    """

    bank: int = 2 << 20  # Bytes per dense ring slot
    nbd: int = 14  # Dense ring slots
    nbe: int = 10  # Expert ring slots
    ebank: int = 1 << 20  # Bytes per expert ring slot
    vmem_mib: int = 62  # VMEM limit in MiB
    smem_mib: int = 1  # SMEM limit in MiB
    bounds_checks: bool = False
    lazy: bool = True  # Refill block dense slots after their streams
    inplace: bool = False  # Alias state inputs to outputs where supported
    rsag: bool = False  # Two-round reduce-scatter + all-gather
    hier: int = 0  # Hierarchical all-reduce on v7x-8 (0, 1, 2, 3)
    war: int = -1  # Unfolded [MR, H] all-reduce payload (-1: auto, 0: off, 1: on)
    mb8: int = 0  # Internal activation row count MB (0: M, 1: max(16, MR), 2: MR)
    # Per-bucket schedule threshold (rows). Blocks never compare `B` to a
    # literal; they read this, and tile rules (`profiles/`) may tune it.
    wide_rows: int = 16  # `B > wide_rows`: WIDE schedule (`ExecSpec.WIDE`)
    # Opt-in performance flags below default to off. Named profiles
    # (`profile.py`, `profiles/`) turn them on; defaults never do.
    dense_acc_pair: bool = False  # Unrolled / paired dense matmul accumulation
    dmm_unroll: int = 0  # Max tiles to fully unroll in dmatmul (0: never)
    rel_fetch: bool = False  # Trace-time-decoded dense refills (fetch_rel/flush_rel)
    # Per-block tunables (`Block.Tunables` instances keyed by block name).
    blocks: Mapping[str, Any] = TunablesMap()

    def __post_init__(self) -> None:
        if not isinstance(self.blocks, TunablesMap):
            object.__setattr__(self, "blocks", TunablesMap(self.blocks))

    def block(self, name: str, cls: type[_T] | None = None) -> _T:
        """Tunables of block `name` (class defaults when unset)."""
        val = self.blocks.get(name)
        if val is not None:
            return val
        return (cls if cls is not None else tunables_class(name))()

    def with_blocks(self, updates: Mapping[str, Any]) -> Tiles:
        """Copy with block tunables replaced: `{block name: {field: value}}`.

        A value may also be a `Tunables` instance (replaces the block's whole
        set). Unknown blocks or fields raise.
        """
        blocks = dict(self.blocks)
        for name, upd in updates.items():
            if isinstance(upd, Mapping):
                blocks[name] = dataclasses.replace(self.block(name), **upd)
            else:
                blocks[name] = upd
        return dataclasses.replace(self, blocks=TunablesMap(blocks))


# Numerics class of every core `Tiles` field (`profile.approx_features` reads
# this together with each block's `Tunables.NUMERICS`, see `tile_numerics`).
# `exact`: moving the field off the reference profile's value keeps hidden
# states and caches bit-identical, because it only changes DMA / ring
# scheduling, budgets or debug checks. `approx`: changes the
# arithmetic (matmul K-tiling, reduction order, rounding, routing) or
# is not proven exact; profiles that move any of these are `approx`.
TILE_NUMERICS: dict[str, NumericsClass] = {
    "bank": "approx",  # dense K-tiling (`planner.k_tile`)
    "nbd": "exact",  # dense ring depth (scheduling)
    "nbe": "approx",  # expert ring depth also gates block pipelining paths
    "ebank": "approx",  # expert K-tiling
    "vmem_mib": "exact",  # budget only
    "smem_mib": "exact",  # budget only
    "bounds_checks": "exact",  # debug checks only
    "lazy": "exact",  # dense refill placement (scheduling)
    "inplace": "approx",  # state input/output aliasing (not proven exact)
    "rsag": "approx",  # rounds all-reduce partial sums to bf16
    "hier": "approx",  # all-reduce reduction order
    "war": "approx",  # all-reduce payload layout
    "mb8": "approx",  # activation row count
    "wide_rows": "approx",  # WIDE schedule threshold
    "dense_acc_pair": "approx",  # paired dense accumulation
    "dmm_unroll": "approx",  # unrolled dense matmul accumulation
    "rel_fetch": "exact",  # trace-time-decoded dense refills (scheduling)
}


def tile_numerics() -> dict[str, NumericsClass]:
    """Numerics class of every tile knob: core fields plus `<block>.<field>`."""
    out: dict[str, NumericsClass] = dict(TILE_NUMERICS)
    for name, cls in sorted(_TUNABLES.items()):
        out.update({f"{name}.{k}": v for k, v in cls.NUMERICS.items()})  # type: ignore[attr-defined]
    return out


def default_tiles(B: int, **kw: Any) -> Tiles:
    """Model-agnostic default tiles (`profiles/default.py:default_tiles`).

    Used for `ExecSpec.tiles=None`; the kernel builder then fits their ring
    depths to the model's VMEM budget (`vmem.fit_default_tiles`). Keyword
    arguments follow the tile-rule signature (`profile.TileRule`).
    """
    from .profiles.default import default_tiles as _rule

    return _rule(B, **kw)


@dataclass(frozen=True)
class DebugOptions:
    """Debug and ablation flags kept separate from model geometry and execution."""

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
    # None -> default_tiles(B, R=R), ring depths fitted to VMEM at build
    tiles: Tiles | None = None
    debug: DebugOptions = DebugOptions()
    # Cache extension (a frozen `CacheExec` from the cache package; None: plain
    # states without page tables). Core code never reads it.
    cache: Any = None

    @property
    def resolved_tiles(self) -> Tiles:
        return self.tiles if self.tiles is not None else default_tiles(self.B, R=self.R)

    def resolve_tiles(self, g: Any = None) -> Tiles:
        """Same as `resolved_tiles` (`g` kept for callers; the defaults ignore it)."""
        del g
        return self.resolved_tiles

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
    def skip(self) -> frozenset[str]:
        return frozenset(self.debug.skip)

    @property
    def scopes(self) -> int | bool:
        return self.debug.scopes


class Block:
    """Composable megakernel building block (attention, recurrent, FFN, etc.).

    A block with its own knobs sets `Tunables` to a frozen dataclass (fields
    with defaults plus a `NUMERICS` class attribute); defining the subclass
    registers it under `name`, and the block reads its values with
    `self.tunables(ex)`.
    """

    name: str = "block"
    reduction: Literal["allreduce", "none", "custom"] = "allreduce"
    input_norm: str | None = None
    Tunables: ClassVar[type | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.Tunables is not None and (
            "Tunables" in cls.__dict__ or "name" in cls.__dict__
        ):
            register_tunables(cls.name, cls.Tunables)

    def tunables(self, ex: ExecSpec) -> Any:
        """This block's `Tunables` in `ex`'s tiles (class defaults when unset)."""
        assert self.Tunables is not None, f"block {self.name!r} declares no Tunables"
        return ex.resolved_tiles.block(self.name, self.Tunables)

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        return []

    def states(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        return []

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        return []

    def extra_inputs(self, g: Any, ex: ExecSpec) -> list[ExtraInput]:
        """Positional inputs after the states (see `ExtraInput`)."""
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
    """Conditional slot inside a single layer loop body (e.g. two mixer kinds).

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
    # Model-injected cache backend (layout + gather / scatter callbacks) from
    # the cache package; None for models without a page-based cache.
    cache_backend: Any = None

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
