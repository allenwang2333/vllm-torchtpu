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
from typing import Any, Callable, Literal

import jax.numpy as jnp
from vllm_torchtpu.kernels.megakernel import pool_alias


@dataclass(frozen=True)
class Spec:
    """Shape + dtype descriptor for nested/tupled per-layer HBM tensors."""

    shape: tuple[int, ...]
    dtype: Any


@dataclass(frozen=True)
class WeightDecl:
    """Declarative specification for a model weight tensor."""

    name: str
    K: int                              # Reduction dimension (or leading dim if non-streamed)
    N: int                              # Output column dimension per rank
    dtype: Any
    layers: Literal["all", "attn", "gdn", "none"] = "all"
    stream_ring: Literal["static", "dynamic", "smalls", "hbm"] = "static"
    layout: Literal["stacked", "per_layer"] = "stacked"
    shape_override: tuple[int, ...] | None = None


@dataclass(frozen=True)
class StateDecl:
    """Declarative specification for a state / cache / auxiliary tensor."""

    name: str
    shape: tuple[int, ...]              # Per-rank local shape
    dtype: Any
    access: Literal["r", "w", "rw"] = "r"
    tupled_count: int = 0               # >0 when passed as a tuple of `tupled_count` Spec refs


@dataclass(frozen=True)
class SmallField:
    """Per-layer 1D vector packed into the double-buffered `smalls` table."""

    name: str
    width: int                          # Local elements on each rank


@dataclass(frozen=True)
class ScratchReq:
    """VMEM / SMEM / Semaphore buffer request declared by a Block or Core."""

    name: str
    spec: Any                           # pltpu.VMEM, pltpu.SMEM, or pltpu.SemaphoreType.DMA
    phase: str = "persistent"           # "persistent" or slot phase for overlay sharing


@dataclass(frozen=True)
class Tiles:
    """Execution tiling and pipeline depth parameters."""

    bank: int = 2 << 20                 # Bytes per dense ring slot
    nbd: int = 14                       # Dense ring slots
    nbe: int = 10                       # Expert ring slots
    ebank: int = 1 << 20                # Bytes per expert ring slot
    kv_blk: int = 2048                  # KV tokens per block
    nkv: int = 3                        # KV ring depth
    vmem_mib: int = 62                  # VMEM limit in MiB
    bounds_checks: bool = False
    lazy: bool = True                   # Refill attention/MoE dense slots after their streams
    pipe: bool = True                   # Software-pipeline single-tile experts
    topk: bool = True                   # One-reduction top-k picks on packed (value, lane) keys
    inplace: bool = False               # Alias GDN state inputs to outputs
    split: bool = True                  # Sequence-split attention across G ranks sharing a KV head
    rsag: bool = False                  # Two-round reduce-scatter + all-gather
    hier: int = 0                       # Hierarchical all-reduce on v7x-8 (0, 1, 2, 3)
    war: int = -1                       # Unfolded [MR, H] all-reduce payload (-1: auto, 0: off, 1: on)
    pack: int = -1                      # Pack PB = 8 // nh rows per 8-row tile (-1: auto)
    mb8: int = 0                        # Internal activation row count MB (0: M, 1: max(16, MR), 2: MR)
    gdn_rows: int = 8                   # GDN state row chunk size above 16 rows
    gdn_bufs: int = 3                   # GDN state ring depth
    kv_sub: int = 0                     # KV tokens per online-softmax update inside a block (0: full)
    amask: int = 1                      # Attention mask mode (0..4)
    ecopy_opt: int = 0                  # Redundant expert dedup optimization (0 or 1)

    def tk(self, K: int, N: int, dtype: Any, bank: int) -> int:
        """Largest 128-multiple divisor of K whose [tk, N] tile fits a `bank`-byte slot."""
        best = None
        for tk in range(128, K + 1, 128):
            if K % tk == 0 and pool_alias.storage_size((tk, N), jnp.dtype(dtype)) <= bank:
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
class ExecSpec:
    """Execution-time configuration (independent of model architecture)."""

    B: int = 4                          # Active batch rows (= NS * T)
    T: int = 1                          # Tokens per sequence (>1 for MTP verify)
    R: int = 8                          # Tensor-parallel / expert-parallel ranks
    ctx: int = 2048                     # Max KV sequence length
    paged_kv: bool = False              # Paged FP8 KV cache vs dense BF16 KV cache
    page_size: int = 128                # KV cache page size (tokens)
    num_pages: int = 64                 # Total pages per attention layer
    max_pages_per_seq: int = 16         # Max pages per sequence in page_table
    fused_pool_io: bool = False         # Fuse P1+P2 GDN+KV pool I/O into the megakernel
    act_mode: str = "none"              # Active sequence guard ("none", "ref", "from_ckpt", "pos")
    ckpt_2d: bool = False               # 2D vs 3D ckpt_indices
    roff_1d: bool = False               # 1D vs 2D read_offsets
    ckpt_k: int = 0                     # Checkpoint K dimension override
    stack_e: bool = False               # True: 4D stacked expert weights; False: per-layer 3D
    forced_routing: bool = False        # Use external rbias for routing
    row_mask: bool = False              # Use external moe_rows mask for routing
    interpret: bool = False             # Run Pallas in CPU interpret mode
    skip: frozenset[str] = frozenset()  # Debug/ablation skip flags
    scopes: int | bool = False          # Emit per-phase named_scope annotations
    tiles: Tiles | None = None          # None -> default_tiles(B)


@dataclass
class ResidualState:
    """Live VMEM residual stream references carried across layer slots."""

    res: Any                            # scr["res"]: (MB, H) f32 residual stream
    xb: Any                             # scr["xb"]: (MB, H) bf16 normalized residual input
    mix: Any                            # scr["mix"]: (MB, NMIX) f32 mixer output


class Block:
    """Composable megakernel building block (attention, GDN, MoE, MLP, etc.)."""

    name: str = "block"

    def weights(self, c: Any) -> list[WeightDecl]:
        return []

    def states(self, c: Any) -> list[StateDecl]:
        return []

    def smalls(self, c: Any) -> list[SmallField]:
        return []

    def scratch(self, c: Any, tiles: Tiles, plan_info: dict[str, Any]) -> dict[str, Any]:
        return {}

    def prologue(self, ctx: Any) -> None:
        """Executed once before the main layer loop."""

    def emit(self, ctx: Any, l: Any, slot: Any, g0: Any) -> Any:
        """Executed inside the layer loop for layer `l`. Returns updated stream cursor `g`."""
        return g0

    def epilogue(self, ctx: Any) -> None:
        """Executed once after the main layer loop."""


@dataclass(frozen=True)
class SwitchSlot:
    """Conditional slot inside a single layer loop body (e.g. GDN vs GQA).

    Dispatches between sibling `Block`s inside one `lax.fori_loop(0, L)` step
    without unrolling multi-layer periods.
    """

    name: str
    branches: tuple[Block, ...]
    # period: number of layers in one repeating cycle (e.g. AE=4: 3 GDN + 1 GQA)
    period: int = 1


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
        """Return unique blocks in deterministic declaration order."""
        seen: set[str] = set()
        out: list[Block] = []
        for s in self.layer_type.slots:
            bs = s.branches if isinstance(s, SwitchSlot) else (s,)
            for b in bs:
                if b.name not in seen:
                    seen.add(b.name)
                    out.append(b)
        return out

