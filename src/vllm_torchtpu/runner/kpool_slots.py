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
"""The stride-1 ``req_id -> slot`` pool that fills ``kpool_state_indices``.

The design, and the alternatives rejected on the way to it, are in
``docs/glm5-3-flash-kpool-pending-state.md``. It mirrors
``TpuModelRunner._init_mamba_slot_pool`` / ``._build_mamba_state_indices`` --
same req_id keying, same null-slot-0 sentinel, same DP-dummy tolerance -- at
**stride 1**, because the kpool pending ring is one row-window per request and
not a recurrence with per-draft checkpoints.

Three things this file exists to keep apart, each of which has already burned
somebody:

1. **It is not ``mamba_state_indices``.** Under unified-align mode that field
   is derived from the device block table and is an *attention-pool block id*,
   which at ``max_model_len`` 4096 and ``block_size`` 16 already exceeds
   ``max_num_reqs + 1`` by orders of magnitude. Indexing a
   ``[max_num_reqs + 1, depth, 2, 128]`` ring with it is a silent
   out-of-bounds read -- rejected option (vii) of the pending-state doc.
2. **It lives outside ``tpu_runner.py``.** That module cannot be imported on
   the CPU rung (a pre-existing ``ModelOptNvFp4LinearMethod`` rebase drift), so
   every line of slot logic that a test could catch a bug in lives here and the
   runner keeps only the three-line call. ``torch`` is the single dependency
   and it is CPU-importable.
3. **A wrong mapping is silent.** Two live requests sharing a slot read each
   other's ``(k, gate)`` rows with *matching* absolute-position tags, because
   absolute positions collide across requests -- the per-row tag cannot see it.
   ``DIAG_SLOT_OUT_OF_RANGE`` and ``DIAG_SLOT_DUPLICATES``
   (``kernels/glm5_kpool/pool_write.py``) are the only detectors, and
   :func:`assert_slot_indices_sane` is their host-side twin so a bad build is
   caught before it reaches the device.
"""

from __future__ import annotations

import torch

#: Slot 0 is the null sentinel: never handed to a request, written by every
#: padded persistent-batch position and every non-carried token, and its tag is
#: always ``PENDING_TAG_EMPTY`` so it can never satisfy a tag check. Same role
#: slot 0 plays in the compact mamba pool.
KPOOL_NULL_SLOT = 0


def kpool_num_slots(max_num_reqs: int) -> int:
    """``max_num_reqs + 1`` -- the pool size, and the ring's major dimension.

    Kept as a named function rather than an inline ``+ 1`` because the same
    number is derived independently by
    ``derive_pending_state_geometry(...).num_slots``; the two agreeing is what
    makes an out-of-range slot impossible rather than merely unlikely, and
    :func:`assert_pool_matches_ring` ties them.
    """
    if max_num_reqs < 1:
        raise ValueError(f"max_num_reqs must be >= 1; got {max_num_reqs}")
    return max_num_reqs + 1


class KpoolSlotPool:
    """``req_id -> ring slot``, stride 1, slot 0 reserved.

    Keyed on the stable ``req_id`` rather than the persistent-batch position,
    which is what makes the mapping immune to upstream vLLM's
    ``InputBatch.condense`` / swap reordering (we do not subclass it). Slots
    are popped from the high end so the first request gets slot 1, matching
    ``_init_mamba_slot_pool``'s ordering purely so a debug print reads the
    same way.
    """

    def __init__(self, max_num_reqs: int) -> None:
        self.max_num_reqs = int(max_num_reqs)
        self.num_slots = kpool_num_slots(self.max_num_reqs)
        self._slot_by_req_id: dict[str, int] = {}
        self._free_slots: list[int] = []
        #: Slots handed to a request *this build*. Exposed so a later step can
        #: reset those ring rows' tags to ``PENDING_TAG_EMPTY`` on admission.
        #: Not required for correctness -- a fresh request's first step starts
        #: at absolute position 0 and reads nothing from the ring (invariant R
        #: with ``s = 0`` is empty), and the step-11 spike's ``wipe=False``
        #: rows measure exactly that -- but it is what would turn "safe by an
        #: argument" into "detected if the argument is ever violated".
        self.newly_allocated: list[int] = []
        self.reset()

    def reset(self) -> None:
        """Drop every mapping and refill the free list."""
        self._slot_by_req_id = {}
        self._free_slots = list(reversed(range(1, self.num_slots)))
        self.newly_allocated = []

    # -- introspection, for tests and for the device probe -----------------

    @property
    def num_free(self) -> int:
        return len(self._free_slots)

    @property
    def mapping(self) -> dict[str, int]:
        return dict(self._slot_by_req_id)

    def slot_for(self, req_id: str) -> int | None:
        return self._slot_by_req_id.get(req_id)

    # -- the two operations the runner performs ----------------------------

    def release_absent(self, live_req_ids) -> list[int]:
        """Return the slots of requests no longer in the persistent batch.

        Called before assignment, exactly as ``_build_mamba_state_indices``
        does, so a finished request's slot is reusable in the same step a new
        request takes one. Without this the pool is exhausted after
        ``max_num_reqs`` *cumulative* requests instead of ``max_num_reqs``
        *concurrent* ones -- which on a long soak looks like a mid-run crash
        rather than a leak.
        """
        live = set(live_req_ids)
        freed = []
        for req_id in list(self._slot_by_req_id.keys()):
            if req_id not in live:
                slot = self._slot_by_req_id.pop(req_id)
                self._free_slots.append(slot)
                freed.append(slot)
        return freed

    def slots_for_batch(
        self,
        req_ids,
        *,
        start_index: int,
        num_reqs: int,
        target_num_reqs: int,
    ) -> list[int]:
        """The ``target_num_reqs``-long slot vector for this chunk.

        Positions ``[0, num_live)`` hold the slot of
        ``req_ids[start_index + i]``; everything after holds
        :data:`KPOOL_NULL_SLOT`.

        ``num_reqs`` may exceed what the persistent batch actually holds: an
        idle DP rank replaying a peer's collective trace passes the
        coordinated shape (``num_reqs_max_model_len``) while its own batch is
        empty or shorter, which is how ``run_dp_dummy_draft`` reaches here.
        Those positions carry no request, so they keep the null slot the
        padded tail already uses -- the ``min(num_reqs, len(req_ids) -
        start_index)`` clamp is the whole fix, and without it the compact
        layout raises ``IndexError`` and kills the engine.
        """
        if target_num_reqs < 0:
            raise ValueError(f"target_num_reqs must be >= 0; got "
                             f"{target_num_reqs}")
        slots = [KPOOL_NULL_SLOT] * target_num_reqs
        num_live = max(0, min(int(num_reqs), len(req_ids) - int(start_index)))
        num_live = min(num_live, target_num_reqs)
        self.newly_allocated = []
        for i in range(num_live):
            req_id = req_ids[start_index + i]
            # Mirrors `_build_mamba_state_indices`: a `None` here means the
            # persistent batch handed us a hole inside its own live prefix,
            # which would map to the null slot and silently share it with the
            # padded tail.
            assert req_id is not None, (
                f"kpool slot pool: req_ids[{start_index + i}] is None inside "
                f"the live prefix [{start_index}, {start_index + num_live}).")
            slot = self._slot_by_req_id.get(req_id)
            if slot is None:
                if not self._free_slots:
                    raise RuntimeError(
                        "kpool slot pool exhausted: "
                        f"{len(self._slot_by_req_id)} live mappings against "
                        f"{self.num_slots} slots (max_num_reqs="
                        f"{self.max_num_reqs}). The persistent batch holds at "
                        "most max_num_reqs requests, so this means "
                        "release_absent() was not called with the live "
                        "req_ids before assignment.")
                slot = self._free_slots.pop()
                self._slot_by_req_id[req_id] = slot
                self.newly_allocated.append(slot)
            slots[i] = slot
        return slots

    def build_state_indices(
        self,
        req_ids,
        *,
        start_index: int,
        num_reqs: int,
        target_num_reqs: int,
        out: torch.Tensor,
        live_req_ids=None,
    ) -> torch.Tensor:
        """Fill ``out[:target_num_reqs]`` with this chunk's slots.

        Args:
            out: an int32 CPU staging tensor of length at least
                ``target_num_reqs``; the runner keeps one for the engine's
                lifetime and H2Ds the slice, exactly as
                ``mamba_state_indices_cpu`` does.
            live_req_ids: the persistent batch's live keys. Defaults to
                ``req_ids`` minus ``None``s, which is right for the standalone
                case; the runner passes ``input_batch.req_id_to_index.keys()``
                because ``req_ids`` is only this *chunk's* window.

        Returns:
            the filled ``out[:target_num_reqs]`` view (CPU). The caller does
            the H2D.
        """
        if out.dtype != torch.int32:
            raise ValueError(
                f"kpool state-index staging tensor must be int32, got "
                f"{out.dtype}; the op reads it as i32[num_seqs].")
        if out.numel() < target_num_reqs:
            raise ValueError(
                f"kpool state-index staging tensor holds {out.numel()} "
                f"entries, need {target_num_reqs}.")
        if live_req_ids is None:
            live_req_ids = [r for r in req_ids if r is not None]
        self.release_absent(live_req_ids)
        slots = self.slots_for_batch(req_ids,
                                     start_index=start_index,
                                     num_reqs=num_reqs,
                                     target_num_reqs=target_num_reqs)
        view = out[:target_num_reqs]
        view.zero_()
        if slots:
            view.copy_(torch.tensor(slots, dtype=torch.int32))
        return view


# --- the two host-side checks that mirror the device diagnostics ----------


def assert_pool_matches_ring(pool: KpoolSlotPool, *, ring_num_slots: int,
                             where: str) -> None:
    """The pool's slot space and the ring's major dimension must be one number.

    ``KpoolSlotPool`` derives ``max_num_reqs + 1`` from the runner's
    ``max_num_reqs``; ``derive_pending_state_geometry`` derives the same
    quantity from the indexer's view of the config. They are two expressions
    of one fact, and the failure mode of them disagreeing by even one is a
    scatter into the last row of a ring that has no last row -- which XLA
    silently drops on the write and silently clamps on the read.
    """
    if pool.num_slots != ring_num_slots:
        raise ValueError(
            f"{where}: the kpool slot pool has {pool.num_slots} slots "
            f"(max_num_reqs={pool.max_num_reqs}) but the pending ring has "
            f"{ring_num_slots}. Both must be max_num_reqs + 1. A pool larger "
            "than the ring scatters out of bounds; a pool smaller than the "
            "ring wastes a row and means one of the two read a different "
            "max_num_reqs.")


def assert_slot_indices_sane(slots, *, num_slots: int, num_live: int,
                             where: str) -> None:
    """Host-side twin of ``DIAG_SLOT_OUT_OF_RANGE`` / ``DIAG_SLOT_DUPLICATES``.

    Checks exactly what the device counters count, on exactly the vector that
    is about to be uploaded:

    * every entry in ``[0, num_slots)``, and every *live* entry in
      ``[1, num_slots)`` -- a live request on slot 0 aliases the null sentinel
      that every padded token also writes;
    * no two live positions sharing a slot.

    The padded tail is excluded from the duplicate check on purpose: it is
    *supposed* to be all slot 0.
    """
    values = [int(v) for v in slots]
    bad_range = [(i, v) for i, v in enumerate(values)
                 if not 0 <= v < num_slots]
    if bad_range:
        raise ValueError(
            f"{where}: {len(bad_range)} kpool slot ids outside "
            f"[0, {num_slots}): {bad_range[:8]}. This is what "
            "DIAG_SLOT_OUT_OF_RANGE counts on device, and the classic cause "
            "is mamba_state_indices being passed through -- those are "
            "attention-pool block ids under unified-align mode.")
    live = values[:num_live]
    on_null = [i for i, v in enumerate(live) if v == KPOOL_NULL_SLOT]
    if on_null:
        raise ValueError(
            f"{where}: live persistent-batch positions {on_null[:8]} carry "
            "the null slot 0, which every padded token also writes. Their "
            "pending rows would be clobbered every step.")
    seen: dict[int, int] = {}
    dups = []
    for i, v in enumerate(live):
        if v in seen:
            dups.append((seen[v], i, v))
        else:
            seen[v] = i
    if dups:
        raise ValueError(
            f"{where}: {len(dups)} pairs of live requests share a kpool ring "
            f"slot: {dups[:8]}. This is DIAG_SLOT_DUPLICATES, and it is the "
            "one failure the per-row position tag cannot see -- absolute "
            "positions collide across requests, so both sequences read each "
            "other's rows with tags that match.")
