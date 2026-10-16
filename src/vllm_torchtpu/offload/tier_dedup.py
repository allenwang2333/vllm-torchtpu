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
"""Deduplicated host-tier layout for KV heads replicated across TP ranks.

With fewer KV heads than TP ranks, every rank of a head group holds the same
full-attention KV. The replicated layout stores each such block once per rank;
this layout stores it once per head group. It runs two Raiden stores, each
with its own controller, ledger, LRU and capacity:

- the full-attention store: every rank registers its KV tensors with a host
  pool that lives in one shared-memory segment per head group, so the ranks of
  a group map the same host bytes. Saves (device to host) go to one leader per
  group; loads go to every rank, each reading its group's segment.
- the state store: Mamba/GDN state blocks, which differ on every rank, in a
  private host pool per rank as before.

`ClassRoutedStore` presents the two as one store to `RaidenOffloadingManager`,
routing every sub-hash by the KV cache group in its key.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

FA_CLASS = "fa"
STATE_CLASS = "state"

# Offset of the full-attention store's controller port from the state store's.
FA_PORT_OFFSET = 100
# Prefix of the full-attention host segments under /dev/shm.
_FA_SEGMENT_PREFIX = "raidenfa"


def tier_dedup_requested() -> bool:
    return os.environ.get("TPU_RAIDEN_TIER_DEDUP", "0") == "1"


def fa_fraction() -> float:
    """Share of the host budget the full-attention store gets."""
    value = float(os.environ.get("TPU_RAIDEN_TIER_FA_FRACTION", "0.7"))
    if not 0.0 < value < 1.0:
        raise ValueError(f"TPU_RAIDEN_TIER_FA_FRACTION must be in (0, 1), got {value}")
    return value


def group_idx_of_sub_hash(sub_hash: bytes) -> int:
    """KV cache group of a store sub-hash: namespace ‖ block hash ‖ group ‖ sub_idx."""
    return int.from_bytes(sub_hash[-8:-4], "big")


@dataclass(frozen=True)
class TierDedupLayout:
    fa_group_ids: frozenset[int]
    # Ranks holding each KV head; a head group is `replicas` consecutive ranks.
    replicas: int
    world_size: int
    fa_capacity_keys: int
    state_capacity_keys: int
    state_port: int
    fa_port: int

    @property
    def num_head_groups(self) -> int:
        return self.world_size // self.replicas

    def head_group(self, rank: int) -> int:
        return rank // self.replicas

    def leader_ranks(self) -> list[int]:
        return [g * self.replicas for g in range(self.num_head_groups)]

    def class_of(self, sub_hash: bytes) -> str:
        if group_idx_of_sub_hash(sub_hash) in self.fa_group_ids:
            return FA_CLASS
        return STATE_CLASS


def resolve_tier_dedup_layout(
    *,
    fa_group_ids: Sequence[int],
    total_num_kv_heads: int,
    tp_size: int,
    world_size: int,
    pcp_size: int,
    dcp_size: int,
    cpu_bytes: int,
    worker_bytes_per_block: int,
    state_port: int,
) -> TierDedupLayout | None:
    """The deduplicated layout for this engine, or None when it does not apply.

    It applies when TPU_RAIDEN_TIER_DEDUP=1, the engine is plain TP (one rank
    per TP shard, no context parallelism) and KV heads are replicated, that
    is the model has fewer KV heads than TP ranks. vLLM then gives rank r the
    head r // (tp_size // total_num_kv_heads), so consecutive ranks form the
    head groups.
    """
    if not tier_dedup_requested():
        return None
    reason = None
    if pcp_size > 1 or dcp_size > 1 or world_size != tp_size:
        reason = (
            f"needs plain TP (world {world_size}, TP {tp_size}, PCP {pcp_size}, "
            f"DCP {dcp_size})"
        )
    elif total_num_kv_heads >= tp_size or tp_size % total_num_kv_heads:
        reason = f"{total_num_kv_heads} KV heads over TP {tp_size} are not replicated"
    elif not fa_group_ids:
        reason = "no full-attention KV cache group"
    if reason is not None:
        logger.warning(
            "TPURaidenOffloadingConnector: TPU_RAIDEN_TIER_DEDUP=1 ignored: %s; "
            "keeping the replicated layout",
            reason,
        )
        return None
    replicas = tp_size // total_num_kv_heads
    fraction = fa_fraction()
    num_groups = world_size // replicas
    fa_capacity = int(cpu_bytes * fraction) // (worker_bytes_per_block * num_groups)
    state_capacity = int(cpu_bytes * (1.0 - fraction)) // (
        worker_bytes_per_block * world_size
    )
    if fa_capacity <= 0 or state_capacity <= 0:
        raise ValueError(
            "TPURaidenOffloadingConnector: host budget too small for the "
            f"deduplicated layout (full attention {fa_capacity} blocks, state "
            f"{state_capacity} blocks)"
        )
    return TierDedupLayout(
        fa_group_ids=frozenset(fa_group_ids),
        replicas=replicas,
        world_size=world_size,
        fa_capacity_keys=fa_capacity,
        state_capacity_keys=state_capacity,
        state_port=state_port,
        fa_port=state_port + FA_PORT_OFFSET,
    )


def fa_segment_key(fa_port: int, run_token: str, head_group: int) -> str:
    """Shared-memory segment name of one head group's full-attention pool.

    `run_token` is the content of the full-attention controller's ready marker,
    which the scheduler makes unique per run, so a segment left in /dev/shm by
    an earlier container of the same pod is never joined.
    """
    return f"{_FA_SEGMENT_PREFIX}-{fa_port}-{run_token}-g{head_group}"


def remove_stale_fa_segments(fa_port: int, run_token: str) -> list[str]:
    """Unlinks this port's full-attention segments from earlier runs.

    Their pages are freed once no process maps them; a restarted container's
    predecessor no longer does. Returns the names removed.
    """
    prefix = f"{_FA_SEGMENT_PREFIX}-{fa_port}-"
    current = f"{prefix}{run_token}-"
    removed: list[str] = []
    try:
        names = os.listdir("/dev/shm")
    except OSError:
        return removed
    for name in names:
        if name.startswith(prefix) and not name.startswith(current):
            try:
                os.unlink(os.path.join("/dev/shm", name))
                removed.append(name)
            except OSError:
                pass
    return removed


class ClassRoutedStore:
    """One store API over the full-attention and state stores.

    Every call splits its sub-hashes by class, preserving their relative order
    within each class, and merges the answers. Pins stay per store, so the
    manager's pin bookkeeping is unchanged. Calls that launch transfers have
    `*_parts` variants reporting which sub-hashes launched, because one class
    can launch while the other is refused.
    """

    def __init__(
        self,
        stores: dict[str, Any],
        layout: TierDedupLayout,
        block_id_cls: Any,
    ):
        self._stores = stores
        self._layout = layout
        self._block_id_cls = block_id_cls
        self.evicted_sub_hashes = {cls: 0 for cls in stores}
        # Sub-hashes whose save or load launched, per class.
        self.saved_sub_hashes = {cls: 0 for cls in stores}
        self.loaded_sub_hashes = {cls: 0 for cls in stores}

    # -- routing helpers --------------------------------------------------

    def _split(self, hashes: Sequence[bytes]) -> dict[str, list[int]]:
        positions: dict[str, list[int]] = {}
        for i, sub_hash in enumerate(hashes):
            positions.setdefault(self._layout.class_of(sub_hash), []).append(i)
        return positions

    @property
    def raiden_id(self) -> Any:
        return self._stores[STATE_CLASS].raiden_id

    @property
    def raiden_controller_address(self) -> str:
        return ", ".join(
            f"{cls} {store.raiden_controller_address}"
            for cls, store in self._stores.items()
        )

    @property
    def store_server_address(self) -> str:
        return self._stores[STATE_CLASS].store_server_address

    def capacity(self) -> int:
        return sum(store.capacity() for store in self._stores.values())

    def class_capacity(self, cls: str) -> int:
        return self._stores[cls].capacity()

    # -- directory --------------------------------------------------------

    def lookup(
        self,
        block_hashes: Sequence[bytes],
        enable_global: bool = False,
        pin_found: bool = True,
    ) -> list[tuple[bytes, Any]]:
        """The resident prefix of `block_hashes`, as one store would answer.

        Each class store answers the resident prefix of its own subsequence;
        the merged prefix ends at the first position some class did not
        answer. Pins a class store took past that point are given back.
        """
        block_hashes = list(block_hashes)
        positions = self._split(block_hashes)
        answers: dict[int, tuple[bytes, Any]] = {}
        found: dict[str, list[bytes]] = {}
        for cls, idx in positions.items():
            matched = self._stores[cls].lookup(
                [block_hashes[i] for i in idx],
                enable_global=enable_global,
                pin_found=pin_found,
            )
            found[cls] = [sub_hash for sub_hash, _ in matched]
            for i, entry in zip(idx, matched):
                answers[i] = entry
        prefix = 0
        while prefix in answers:
            prefix += 1
        if pin_found:
            beyond = {block_hashes[i] for i in answers if i >= prefix}
            for cls, hashes in found.items():
                extra = [h for h in hashes if h in beyond]
                if extra:
                    self._stores[cls].release(extra)
        return [answers[i] for i in range(prefix)]

    def insert(
        self, block_hashes: Sequence[bytes], slices: Sequence[Any], on_host: bool
    ) -> bool:
        """All-or-nothing across classes, like one store's insert.

        A class store that admitted its part before another refused is
        released again; any HBM entry that leaves behind is unbound residue,
        which the manager already treats as a miss and a failed store.
        """
        block_hashes = list(block_hashes)
        inserted: list[tuple[str, list[bytes]]] = []
        for cls, idx in self._split(block_hashes).items():
            store = self._stores[cls]
            hashes = [block_hashes[i] for i in idx]
            class_slices = [
                self._block_id_cls(
                    raiden_id=store.raiden_id,
                    host_block_id=slices[i].host_block_id,
                    device_block_id=slices[i].device_block_id,
                    status=slices[i].status,
                )
                for i in idx
            ]
            if not store.insert(hashes, class_slices, on_host=on_host):
                for done_cls, done_hashes in inserted:
                    self._stores[done_cls].release(done_hashes)
                return False
            inserted.append((cls, hashes))
        return True

    def release(self, block_hashes: Sequence[bytes]) -> None:
        block_hashes = list(block_hashes)
        for cls, idx in self._split(block_hashes).items():
            self._stores[cls].release([block_hashes[i] for i in idx])

    def demote(self, block_hashes: Sequence[bytes]) -> int:
        block_hashes = list(block_hashes)
        demoted = 0
        for cls, idx in self._split(block_hashes).items():
            store = self._stores[cls]
            demoted += store.demote([block_hashes[i] for i in idx])
        return demoted

    # -- transfers --------------------------------------------------------

    def save_parts(
        self, block_hashes: Sequence[bytes]
    ) -> tuple[list[bytes], list[bytes]]:
        """Launches each class's save; returns (launched, refused) sub-hashes."""
        block_hashes = list(block_hashes)
        launched: list[bytes] = []
        refused: list[bytes] = []
        for cls, idx in self._split(block_hashes).items():
            hashes = [block_hashes[i] for i in idx]
            if self._stores[cls].save(hashes):
                launched.extend(hashes)
                self.saved_sub_hashes[cls] += len(hashes)
            else:
                refused.extend(hashes)
        return launched, refused

    def save(self, block_hashes: Sequence[bytes]) -> bool:
        _, refused = self.save_parts(block_hashes)
        return not refused

    def load_parts(
        self, block_hashes: Sequence[bytes], device_block_ids: Sequence[int]
    ) -> tuple[list[bytes], list[bytes]]:
        """Launches each class's load; returns (launched, refused) sub-hashes."""
        block_hashes = list(block_hashes)
        launched: list[bytes] = []
        refused: list[bytes] = []
        for cls, idx in self._split(block_hashes).items():
            hashes = [block_hashes[i] for i in idx]
            if self._stores[cls].load(hashes, [device_block_ids[i] for i in idx]):
                launched.extend(hashes)
                self.loaded_sub_hashes[cls] += len(hashes)
            else:
                refused.extend(hashes)
        return launched, refused

    def load(
        self, block_hashes: Sequence[bytes], device_block_ids: Sequence[int]
    ) -> bool:
        _, refused = self.load_parts(block_hashes, device_block_ids)
        return not refused

    def read_remote(
        self,
        block_hashes: Sequence[bytes],
        slices: Sequence[Any],
        device_block_ids: Sequence[int],
    ) -> bool:
        # The deduplicated layout runs without a registry or storage tier, so
        # nothing resolves to a peer or storage slice.
        return False

    def _merge_polls(self, poll: Callable[[Any], tuple]) -> tuple:
        merged: list[list[bytes]] | None = None
        for store in self._stores.values():
            result = poll(store)
            if merged is None:
                merged = [list(part) for part in result]
            else:
                for acc, part in zip(merged, result):
                    acc.extend(part)
        return tuple(merged or ())

    def poll_save_status(self) -> tuple:
        return self._merge_polls(lambda store: store.poll_save_status())

    def poll_load_status(self) -> tuple:
        return self._merge_polls(lambda store: store.poll_load_status())

    def poll_remote_read_status(self) -> tuple:
        return ([], [], [])

    def set_eviction_callback(self, callback: Callable | None) -> None:
        for cls, store in self._stores.items():
            if callback is None:
                store.set_eviction_callback(None)
                continue

            def _counted(hashes: list[bytes], cls: str = cls) -> None:
                self.evicted_sub_hashes[cls] += len(hashes)
                callback(hashes)

            store.set_eviction_callback(_counted)
