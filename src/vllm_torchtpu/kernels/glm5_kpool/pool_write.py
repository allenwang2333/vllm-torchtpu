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
"""``kpool_write``: pending-state carry, pooling, fp8 pack, pooled scatter.

The pending-pool state's design, and the alternatives that were rejected,
are in ``docs/glm5-3-flash-kpool-pending-state.md``. Pure ``jnp`` -- no torch, no
``torch_tpu`` -- so every claim below has a CPU test rather than a device
claim.

WHAT THIS DOES, IN ORDER
------------------------

One step covers, per request, the absolute positions ``[step_start, seq_len)``.
For every position ``p`` with ``(p + 1) % index_kpool == 0`` the pool
``j = p // index_kpool`` becomes complete, and its pooled record must be
written to the pooled indexer K cache at pool slot ``j``:

1.  Gather the pool's ``index_kpool`` ``(k, gate)`` rows. **Split by source**:
    a position ``q >= step_start`` comes from this step's own ``k``/``gate``
    tensors; a position ``q < step_start`` comes from the pending ring at
    ``(slot, q % pending_depth)``. Every ring read is checked against the
    ring's per-row absolute-position tag.
2.  ``pool_window`` -- the per-channel softmax over the pool's slots, with
    ``ape`` indexed by the token's slot *within its pool*.
3.  ``pack_fp8_ue8m0`` -- ``index_head_dim`` fp8_e4m3 bytes plus one e8m0
    scale byte, zero-padded to the record width.
4.  Scatter that record at the pool's ``(page, slot)``. Non-boundary and
    padded tokens are routed to ``OOB_PAGE`` and dropped, the idiom
    ``mla_attention_op.py:98-101`` already uses.
5.  Carry the **trailing** ``pending_depth`` positions of each request's
    in-step range into the ring, keyed by absolute position.

THE IDEMPOTENCE INVARIANT (why rollback needs no checkpointing)
---------------------------------------------------------------

Every ring row is addressed by an **absolute position** and every write
**overwrites** rather than accumulates, so the row for position ``p`` is a pure
function of the token at ``p``. A row written by a rejected draft at ``p`` is
simply rewritten by the real token at ``p`` on the next step, and it is
rewritten *before* any pool containing ``p`` can complete, because pool ``j``
completes only when ``index_kpool*j + index_kpool - 1 >= p`` is written. That
is the entire reason this state is **not** a recurrence and must not reuse
KDA's checkpointed recurrent-state machinery. Do not "optimise"
this into a running ``(numerator, denominator)`` accumulator: it saves zero
bytes (2x128 f32 is exactly one raw ``(k, gate)`` row) and destroys precisely
this property -- see rejected option (vi) in
``docs/glm5-3-flash-kpool-pending-state.md``.

WHY THE READS AND WRITES ARE SPLIT BY SOURCE
--------------------------------------------

The obvious design -- write *every* in-step token into the ring first, then
gather all four rows out of it -- is wrong at any chunk longer than
``pending_depth``: an 8192-token prefill chunk would alias itself in a
depth-7 ring. Splitting by source means positions belonging to the current step
never round-trip the buffer, so the ring only ever has to hold the at most
``index_kpool - 1`` positions carried across a step boundary, plus the
speculative rollback slack. The invariants (W and R in
``docs/glm5-3-flash-kpool-pending-state.md``) are what fix ``pending_depth``.

FLOAT32, AND IT IS LOAD-BEARING
-------------------------------

The ring is f32 and the in-step rows are f32, so the two gather sources are
**bit-identical** for the same position and "chunked prefill at every residue
== single-shot" is an *exact array equality*, not a tolerance. bf16 would save
2.7 MiB of HBM and make the same prompt produce a different pooled key
depending on where the scheduler happened to cut the chunk.
:func:`kpool_write` asserts the ring dtype for that reason, and
``tests/kernels/glm5_kpool/test_pool_write.py`` is the test that fails if
somebody switches it.

WHAT IS NOT HERE
----------------

Allocating the two ring tensors, and the ``req_id -> slot`` pool that fills
``kpool_state_indices``. Both are runner/module-construction concerns (the
decision requires the tensors to exist before ``determine_available_memory()``
profiles). :func:`derive_pending_state_geometry` is the single home for their
shapes so the allocator and this kernel cannot disagree.

That slot pool lives outside this file, so this file cannot make it correct --
but it does *check* it: ``DIAG_SLOT_OUT_OF_RANGE`` and
``DIAG_SLOT_DUPLICATES`` count a slot that is not in ``[1, num_slots)`` and a
slot two live sequences hold at once, before the clip that would otherwise
fold either onto a real request's rows. The tag cannot cover that case; see
``DIAG_SLOT_DUPLICATES``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from vllm_torchtpu.kernels.glm5_kpool.pool_math import (pack_fp8_ue8m0,
                                                        pool_window)

#: ``pending_tag`` value meaning "this ring row has not been written since the
#: slot was handed out". Never a legal absolute position.
PENDING_TAG_EMPTY = -1

#: Scatter page index for a write that must be dropped. XLA drops
#: out-of-bounds scatter writes; ``mla_attention_op.py:98-101`` and
#: ``kernels/mla/kv_cache_utils.py:151`` use the same constant.
OOB_PAGE = 2**30

#: Index into the ``diagnostics`` vector :func:`kpool_write` returns.
#:
#: These are **counters, not asserts**, because there is no way to raise from
#: inside a jitted XLA computation without a host sync on every step. The
#: compare itself is unconditional -- it is one int32 compare per carried row
#: and there are at most ``index_kpool - 1`` of those per completing pool -- so
#: a violation is always *detected*; what the caller chooses to do with the
#: count (sync every step, sync once at the end, log) is the caller's.
#: Nothing about this is a debug flag: the counts are always computed and
#: always returned.
DIAG_TAG_VIOLATIONS = 0
DIAG_POSITION_MISMATCHES = 1

#: Live sequences whose raw ``kpool_state_indices`` entry is not a legal slot
#: (``< 1`` -- slot 0 is the null sentinel and is never handed to a request --
#: or ``>= num_slots``).
DIAG_SLOT_OUT_OF_RANGE = 2

#: Excess live sequences sharing a slot: ``sum_s max(count[s] - 1, 0)`` over
#: the one-hot of the in-range live slots.
#:
#: THIS IS THE COUNTER THE TAG CANNOT REPLACE. :func:`kpool_write` clips
#: ``kpool_state_indices`` into range so a bad slot cannot scatter out of
#: bounds, and the clip folds an illegal slot onto a *live request's* slot. The
#: per-row tag cannot see that, because the tag is an **absolute position**
#: and absolute positions collide across requests: two sequences sharing a
#: slot read each other's ``(k, gate)`` rows with tags that match. So
#: The ``silent == 0`` column of that table is a statement
#: about *ring depth*, not about slot assignment, and this pair of counters is
#: what covers the other half.
DIAG_SLOT_DUPLICATES = 3

KPOOL_WRITE_DIAG_SIZE = 4

DIAG_NAMES = ("tag_violations", "position_mismatches", "slot_out_of_range",
              "slot_duplicates")


def describe_diagnostics(diag) -> str:
    """Human-readable form of the counter vector, for a test or a log line."""
    values = [int(v) for v in jnp.asarray(diag).reshape(-1)]
    assert len(values) == KPOOL_WRITE_DIAG_SIZE, values
    return ", ".join(f"{n}={v}" for n, v in zip(DIAG_NAMES, values))


# ---------------------------------------------------------------------------
# Pending-state geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PendingStateGeometry:
    """Shapes of the two runner-lifetime ring tensors, for one DSA layer.

    Built only by :func:`derive_pending_state_geometry`, which refuses rather
    than returning a depth that cannot survive an MTP rollback.
    """

    index_kpool: int
    index_head_dim: int
    num_slots: int
    """``max_num_reqs + 1``. Slot 0 is the null sentinel and is never handed to
    a request; padded batch tails and non-carried tokens write there."""

    pending_depth: int
    max_rollback_tokens: int
    max_num_reqs: int

    @property
    def kv_shape(self) -> tuple[int, int, int, int]:
        """``pending_kv``: ``f32[num_slots, depth, 2, index_head_dim]``.

        The ``2`` is ``(k, gate)`` **in that order**; swapping them produces a
        finite pooled key of the right shape and fluent text.
        """
        return (self.num_slots, self.pending_depth, 2, self.index_head_dim)

    @property
    def tag_shape(self) -> tuple[int, int]:
        """``pending_tag``: ``i32[num_slots, depth]``, absolute position."""
        return (self.num_slots, self.pending_depth)

    @property
    def bytes_per_slot_row(self) -> int:
        return 2 * self.index_head_dim * 4 + 4

    @property
    def total_bytes(self) -> int:
        """Both tensors, one layer. Flat in ``max_model_len`` -- identical at
        2048, 128 K and 1 M, which is the whole point of staying out of the
        KV-cache allocator."""
        return self.num_slots * self.pending_depth * self.bytes_per_slot_row

    def describe(self) -> str:
        return ("glm5_next kpool pending state: "
                f"index_kpool={self.index_kpool}, "
                f"index_head_dim={self.index_head_dim}, "
                f"max_num_reqs={self.max_num_reqs}, "
                f"num_slots={self.num_slots} (slot 0 = null sentinel), "
                f"max_rollback_tokens={self.max_rollback_tokens}, "
                f"pending_depth={self.pending_depth}, "
                f"pending_kv{self.kv_shape} f32 + "
                f"pending_tag{self.tag_shape} i32 = "
                f"{self.total_bytes} B/layer "
                f"({self.bytes_per_slot_row} B per (slot, row))")


def assert_pending_depth_sufficient(*, pending_depth: int, index_kpool: int,
                                    max_rollback_tokens: int) -> None:
    """The two depth asserts, both carrying the numbers that decide them.

    Split out of :func:`derive_pending_state_geometry` for two reasons: the
    derivation there satisfies them by construction, so an assertion buried
    inside it can never be *shown* to fire; and the caller has to validate a
    buffer allocated somewhere else against the same rule.

    The failure these guard is a silently-wrong pooled key, not a crash. An
    under-deep ring produces a finite pooled key built from whatever position
    happened to alias the slot, which degrades long-context retrieval and
    nothing else. Enumerated over 270 schedules, depth ``index_kpool`` with
    two speculative tokens produces 228 wrong pooled records; see
    ``docs/glm5-3-flash-kpool-pending-state.md`` and
    ``tests/kernels/glm5_kpool/test_pool_write.py``.
    """
    floor = index_kpool - 1 + max_rollback_tokens + 1
    assert pending_depth >= floor, (
        f"pending_depth={pending_depth} is below the rollback floor {floor} = "
        f"(index_kpool={index_kpool} - 1) + "
        f"(max_rollback_tokens={max_rollback_tokens}) + 1. A step over "
        "[s, e) leaves the ring holding [e - depth, e); the next step can "
        f"restart at s' = e - {max_rollback_tokens} and reads back to "
        f"s' - {index_kpool - 1}, so the ring must cover "
        f"{max_rollback_tokens + index_kpool - 1} positions plus the one the "
        "step itself writes.")
    assert pending_depth >= index_kpool, (
        f"pending_depth={pending_depth} < index_kpool={index_kpool}: a single "
        "pool would alias itself in the ring.")


def derive_pending_state_geometry(
    *,
    index_kpool: int,
    index_head_dim: int,
    max_num_reqs: int,
    max_concurrent_batches: int = 1,
    num_speculative_tokens: int = 0,
) -> PendingStateGeometry:
    """The ring depth, **derived** from the config -- never 4, 5 or 8 chosen.

    ``pending_depth = index_kpool + max_concurrent_batches *
    num_speculative_tokens``.

    Both factors are first-class ``VllmConfig`` properties at v0.27.0:
    ``max_concurrent_batches`` (``vllm/config/vllm.py:540``) is ``pp_size``, or
    2 when async scheduling is on at ``pp_size <= 1``;
    ``num_speculative_tokens``
    (``:564``) is 0 without a speculative config. Pass them straight through;
    do not reconstruct them from a speculative config here, or the two
    expressions get to drift.

    Why that depth, in one line: after a step over ``[s, e)`` the ring holds
    every position in ``[e - depth, e)`` (invariant W), and the next step reads
    only ``[s' - (index_kpool - 1), s' - 1]`` (invariant R), where
    ``s' >= e - max_rollback_tokens``. Those two close iff
    ``depth >= max_rollback_tokens + index_kpool - 1``. The chosen depth clears
    that floor by exactly one slot. The tests measure the floor directly at
    ``k_spec in {1, 2, 3, 4, 7}`` rather than trusting the derivation.
    """
    if index_kpool < 1:
        raise ValueError(f"index_kpool must be >= 1; got {index_kpool}")
    if index_head_dim < 1:
        raise ValueError(f"index_head_dim must be >= 1; got {index_head_dim}")
    if max_num_reqs < 1:
        raise ValueError(f"max_num_reqs must be >= 1; got {max_num_reqs}")
    if max_concurrent_batches < 1:
        raise ValueError("max_concurrent_batches must be >= 1; got "
                         f"{max_concurrent_batches}")
    if num_speculative_tokens < 0:
        raise ValueError("num_speculative_tokens must be >= 0; got "
                         f"{num_speculative_tokens}")

    max_rollback_tokens = max_concurrent_batches * num_speculative_tokens
    pending_depth = index_kpool + max_rollback_tokens
    assert_pending_depth_sufficient(pending_depth=pending_depth,
                                    index_kpool=index_kpool,
                                    max_rollback_tokens=max_rollback_tokens)

    return PendingStateGeometry(
        index_kpool=index_kpool,
        index_head_dim=index_head_dim,
        num_slots=max_num_reqs + 1,
        pending_depth=pending_depth,
        max_rollback_tokens=max_rollback_tokens,
        max_num_reqs=max_num_reqs,
    )


def empty_pending_state(
        geometry: PendingStateGeometry) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(pending_kv, pending_tag)`` in the never-written state.

    The jnp form, for tests and for the reference. The production tensors are
    torch and are allocated in the indexer module's construction path so the
    HBM profile sees them.
    """
    return (jnp.zeros(geometry.kv_shape, jnp.float32),
            jnp.full(geometry.tag_shape, PENDING_TAG_EMPTY, jnp.int32))


def clear_pending_slots(pending_tag: jnp.ndarray,
                        slots: jnp.ndarray) -> jnp.ndarray:
    """Reset ``pending_tag`` to ``PENDING_TAG_EMPTY`` for the given slots.

    Called when a slot is handed to a new request. **Not required for
    correctness** -- the whole preemption sweep stays green with the slot
    deliberately not cleared, because the first step of a
    re-prefill has ``s == 0`` and by invariant R reads nothing. It is here
    because it converts "safe by an argument about ``s``" into "detected by the
    tag if that argument is ever violated", which is the difference between a
    slot-pool bug that shows up as a counter and one that shows up as a quiet
    retrieval regression.

    ``pending_kv`` is deliberately left alone: a tagged row is unreadable
    whatever it contains, and zeroing 1 KiB per slot per layer buys nothing.
    """
    slots = jnp.asarray(slots, jnp.int32).reshape(-1)
    return pending_tag.at[slots].set(
        jnp.int32(PENDING_TAG_EMPTY),
        mode="drop",
    )


# ---------------------------------------------------------------------------
# Token bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenIndex:
    """Per-token derived quantities, all ``[num_tokens]`` int32/bool."""

    seq_id: jnp.ndarray
    pos: jnp.ndarray
    valid: jnp.ndarray
    step_start: jnp.ndarray
    """Absolute position of the first token of *this step* for the token's
    request. Positions below it are carried; positions at or above it are in
    this step's ragged tensors."""
    seq_end: jnp.ndarray
    """``seq_lens[seq_id]`` -- one past this step's last position."""
    local: jnp.ndarray
    """Row of this token inside its request's slice of the ragged batch."""


def index_tokens(seq_lens: jnp.ndarray, cu_q_lens: jnp.ndarray,
                 num_tokens: int) -> TokenIndex:
    """Exactly ``update_sparse_mla_kv_cache``'s derivation
    (``kernels/mla/kv_cache_utils.py:139-148``), reused rather than restated so
    the pooled cache's paging cannot drift from the MLA cache's."""
    num_seqs = seq_lens.shape[0]
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_id = jnp.searchsorted(cu_q_lens[1:], tok,
                              side="right").astype(jnp.int32)
    seq_id = jnp.clip(seq_id, 0, num_seqs - 1)
    q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
    local = tok - cu_q_lens[seq_id]
    seq_end = seq_lens[seq_id].astype(jnp.int32)
    step_start = (seq_end - q_len).astype(jnp.int32)
    pos = step_start + local
    valid = tok < cu_q_lens[-1]
    return TokenIndex(seq_id=seq_id,
                      pos=pos.astype(jnp.int32),
                      valid=valid,
                      step_start=step_start,
                      seq_end=seq_end,
                      local=local.astype(jnp.int32))


# ---------------------------------------------------------------------------
# The pooling half
# ---------------------------------------------------------------------------


def read_carried_rows(
    pending_kv: jnp.ndarray,
    pending_tag: jnp.ndarray,
    seq_lens: jnp.ndarray,
    cu_q_lens: jnp.ndarray,
    slot_seq: jnp.ndarray,
    *,
    index_kpool: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The **only** ring read: ``[num_seqs, index_kpool - 1]`` rows.

    Invariant R, made structural instead of commented. A pool completing at
    position ``p`` spans ``index_kpool*floor(p/index_kpool) .. p``, whose first
    element is ``>= p - (index_kpool - 1) >= s - (index_kpool - 1)`` for a step
    starting at ``s``; every position at or above ``s`` takes the in-step
    branch. So a step can read the ring at exactly the positions
    ``[s - (index_kpool - 1), s - 1]`` of each request -- three rows per
    request at GLM's ``index_kpool = 4``, and *nothing else*.

    Gathering them once, per sequence, is not only ~100x less gather work than
    the obvious ``pending_kv[slot[t], src_pos[t, m] % depth]`` over
    ``[num_tokens, index_kpool]`` (which at an 8192-token chunk reads 33 MB out
    of a 5 MiB buffer); it also means no expression in this file *can* address
    the ring anywhere else.

    Returns:
        ``(k [S, W, D], gate [S, W, D], tag [S, W])`` where
        ``W = max(index_kpool - 1, 1)`` and column ``i`` holds position
        ``step_start - (index_kpool - 1) + i``.
    """
    num_seqs = seq_lens.shape[0]
    pending_depth = pending_kv.shape[1]
    width = max(index_kpool - 1, 1)

    q_len_seq = cu_q_lens[1:num_seqs + 1] - cu_q_lens[:num_seqs]
    step_start_seq = seq_lens - q_len_seq  # [S]
    carry_pos = (step_start_seq[:, None] - (index_kpool - 1) +
                 jnp.arange(width, dtype=jnp.int32)[None, :])  # [S, W]
    rows = jnp.remainder(carry_pos, pending_depth)
    slots = jnp.broadcast_to(slot_seq[:, None], rows.shape)
    kv = pending_kv[slots, rows]  # [S, W, 2, D]
    tag = pending_tag[slots, rows]  # [S, W]
    return kv[:, :, 0, :], kv[:, :, 1, :], tag


def kpool_pooled_rows(
    pending_kv: jnp.ndarray,
    pending_tag: jnp.ndarray,
    k: jnp.ndarray,
    gate: jnp.ndarray,
    ape: jnp.ndarray,
    idx: TokenIndex,
    slot_seq: jnp.ndarray,
    seq_lens: jnp.ndarray,
    cu_q_lens: jnp.ndarray,
    *,
    index_kpool: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(pooled f32[T, D], boundary[T], pool_id i32[T], tag_violations)``.

    Exposed separately from :func:`kpool_write` so the pooling can be diffed
    against the CPU oracle's ``pooled_keys`` **before** fp8 quantization, which
    is the only way to tell a pooling bug from a quantization one.

    Row ``t`` is computed for **every** token as if it were a pool boundary --
    the pool is ``[pos - index_kpool + 1, pos]`` -- and ``boundary`` says which
    rows are real. That is deliberate: it keeps the whole thing a fixed-shape
    elementwise/gather computation with no dynamic compaction, and the
    non-boundary rows are dropped by the scatter's ``OOB_PAGE`` routing.

    ``slot_seq`` is per **sequence**, not per token: the only ring read is
    :func:`read_carried_rows`, which is per sequence by construction.
    """
    num_tokens = k.shape[0]

    # Slot `m` of the pool holds absolute position `pool_id*kpool + m`, and for
    # a boundary token that is `pos - (index_kpool - 1) + m`. `ape[m]` is
    # therefore indexed by the token's slot within its pool by construction --
    # which is the reference's `(idx - first_key) % index_kpool` only because
    # `first_key == 0` under vLLM's unpadded ragged layout. `kpool_write`
    # counts every disagreement between `positions` and the position derived
    # here; see DIAG_POSITION_MISMATCHES.
    m = jnp.arange(index_kpool, dtype=jnp.int32)[None, :]  # [1, kpool]
    src_pos = idx.pos[:, None] - (index_kpool - 1) + m  # [T, kpool]
    in_step = src_pos >= idx.step_start[:, None]

    # --- source A: this step's own ragged rows ---------------------------
    in_step_row = cu_q_lens[idx.seq_id][:, None] + (src_pos -
                                                    idx.step_start[:, None])
    in_step_row = jnp.clip(in_step_row, 0, num_tokens - 1)
    k_instep = k[in_step_row]  # [T, kpool, D]
    g_instep = gate[in_step_row]

    # --- source B: the at most `index_kpool - 1` carried rows -------------
    carry_k, carry_g, carry_tag = read_carried_rows(pending_kv,
                                                    pending_tag,
                                                    seq_lens,
                                                    cu_q_lens,
                                                    slot_seq,
                                                    index_kpool=index_kpool)
    width = carry_tag.shape[1]
    # Column of the carry window: `src_pos - (step_start - (index_kpool - 1))`,
    # which lies in [0, index_kpool - 2] on the ring branch. The clip only
    # keeps the in-step columns in range; they are discarded by `where`.
    carry_col = jnp.clip(src_pos - idx.step_start[:, None] + (index_kpool - 1),
                         0, width - 1)
    seq_2d = jnp.broadcast_to(idx.seq_id[:, None], src_pos.shape)
    ring_k = carry_k[seq_2d, carry_col]  # [T, kpool, D]
    ring_g = carry_g[seq_2d, carry_col]
    ring_tag = carry_tag[seq_2d, carry_col]  # [T, kpool]

    k_rows = jnp.where(in_step[..., None], k_instep, ring_k)
    g_rows = jnp.where(in_step[..., None], g_instep, ring_g)

    boundary = (jnp.remainder(idx.pos + 1, index_kpool) == 0) & idx.valid
    pool_id = idx.pos // index_kpool

    # The tag check. Always on: one int32 compare per carried row, at most
    # `index_kpool - 1` carried rows per completing pool. It is what makes an
    # under-sized ring, a wrong slot, or a scheduler rollback larger than
    # `max_rollback_tokens` fail *loudly*: across the under-sized depths in
    # the schedule sweep, zero of the 12 474 wrong pooled records belonged
    # to a pool this compare did not flag.
    tag_bad = boundary[:, None] & (~in_step) & (ring_tag != src_pos)
    tag_violations = jnp.sum(tag_bad.astype(jnp.int32))

    pooled = pool_window(k_rows, g_rows, ape)
    return pooled, boundary, pool_id.astype(jnp.int32), tag_violations


# ---------------------------------------------------------------------------
# The write half
# ---------------------------------------------------------------------------


def _scatter_pooled_records(pooled_cache: jnp.ndarray, records: jnp.ndarray,
                            page: jnp.ndarray, slot: jnp.ndarray,
                            kv_packing: int) -> jnp.ndarray:
    """``mla_attention_op.py:103-105`` / ``_scatter_rows``, one record per row.

    ``records`` is ``uint8[T, index_head_dim + 1]``; the cache row is
    ``record_bytes`` wide (128-aligned), so the tail is zero-padded here. The
    padding is *not* done in ``pack_fp8_ue8m0``: the record layout the kernel
    bitcasts and the page layout the cache is allocated from are two different
    geometries and merging them is how a derivation gets to disagree with
    itself.
    """
    record_bytes = pooled_cache.shape[-1]
    pad = record_bytes - records.shape[-1]
    assert pad >= 0, (
        f"pooled record is {records.shape[-1]} B but the cache row is only "
        f"{record_bytes} B; pooled_record_bytes() and this cache disagree")
    if pad:
        records = jnp.pad(records, ((0, 0), (0, pad)))
    return pooled_cache.at[page, slot // kv_packing,
                           slot % kv_packing].set(records)


def _write_pending_state(
    pending_kv: jnp.ndarray,
    pending_tag: jnp.ndarray,
    k: jnp.ndarray,
    gate: jnp.ndarray,
    idx: TokenIndex,
    slot_id: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Carry the trailing ``pending_depth`` positions of each in-step range.

    Only the trailing window, not every token: a request whose chunk is longer
    than the ring would otherwise alias itself, and the earlier positions are
    provably never read again (invariant R -- the next step reads at most
    ``index_kpool - 1`` positions back from its own start).

    Everything not carried -- earlier in-step positions, padded batch tail --
    is routed to **slot 0**, the null sentinel that is never handed to a
    request. Unlike the pooled scatter's ``OOB_PAGE`` this is a real allocated
    row, so the write lands somewhere harmless rather than relying on XLA's
    out-of-bounds drop; and its tag is written as ``PENDING_TAG_EMPTY``, so
    slot 0 can never satisfy a tag check either.
    """
    pending_depth = pending_kv.shape[1]
    carried = idx.valid & (idx.pos + pending_depth >= idx.seq_end)

    write_slot = jnp.where(carried, slot_id, jnp.int32(0))
    write_row = jnp.remainder(idx.pos, pending_depth)
    rows = jnp.stack([k, gate], axis=1)  # [T, 2, D]
    pending_kv = pending_kv.at[write_slot, write_row].set(rows)
    pending_tag = pending_tag.at[write_slot, write_row].set(
        jnp.where(carried, idx.pos, jnp.int32(PENDING_TAG_EMPTY)))
    return pending_kv, pending_tag


def kpool_write(
    pooled_cache: jnp.ndarray,
    pending_kv: jnp.ndarray,
    pending_tag: jnp.ndarray,
    k: jnp.ndarray,
    gate: jnp.ndarray,
    ape: jnp.ndarray,
    positions: jnp.ndarray,
    seq_lens: jnp.ndarray,
    cu_q_lens: jnp.ndarray,
    pooled_block_tables: jnp.ndarray,
    kpool_state_indices: jnp.ndarray,
    *,
    index_kpool: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Write this step's completed pooled records and carry the tail.

    Args:
        pooled_cache: ``uint8[num_blocks, page_size // kv_packing, kv_packing,
            record_bytes]`` -- the packed pooled indexer K cache, the same
            4-D layout ``streamindex_topk`` unpacks at ``:745``. **All page
            geometry is recovered from this shape**, never passed in, so the
            writer and the kernel cannot disagree about what ``page_size``
            means.
        pending_kv: ``f32[num_slots, pending_depth, 2, index_head_dim]``.
        pending_tag: ``i32[num_slots, pending_depth]``.
        k: ``[num_tokens, index_head_dim]`` -- already ``LayerNorm(wk(h))``.
        gate: ``[num_tokens, index_head_dim]`` -- ``compress_gate @ h``.
        ape: ``[index_kpool, index_head_dim]``.
        positions: ``i32[num_tokens]`` absolute query positions. Used **only**
            as a cross-check; the addressing is derived from ``seq_lens`` /
            ``cu_q_lens``. See ``DIAG_POSITION_MISMATCHES``.
        seq_lens: ``i32[num_seqs]`` total KV length *including* this step's
            tokens.
        cu_q_lens: ``i32[num_seqs + 1]`` cumulative new-token counts.
        pooled_block_tables: ``i32[num_seqs * pages_per_seq]``, flattened --
            the same array handed to ``streamindex_topk`` as ``page_indices``.
        kpool_state_indices: ``i32[num_seqs]`` ring slot per persistent-batch
            position, from the runner's ``req_id``-keyed pool. **Not**
            ``mamba_state_indices``: under unified-align mode those are
            attention-pool block ids and would index this buffer far out of
            bounds (rejected option (vii) in the pending-state doc). Clipped
            into range before use so a bad entry cannot scatter out of bounds,
            but counted first -- see ``DIAG_SLOT_OUT_OF_RANGE`` and
            ``DIAG_SLOT_DUPLICATES``, which are the only detectors of a folded
            or shared slot.
        index_kpool: 4 at the real checkpoint.

    Returns:
        ``(pooled_cache, pending_kv, pending_tag, diagnostics)`` where
        ``diagnostics`` is ``i32[KPOOL_WRITE_DIAG_SIZE]``.
    """
    # --- static contract --------------------------------------------------
    assert index_kpool >= 1, index_kpool
    assert pooled_cache.ndim == 4, pooled_cache.shape
    assert pooled_cache.dtype == jnp.uint8, pooled_cache.dtype
    assert k.ndim == 2 and gate.shape == k.shape, (k.shape, gate.shape)
    num_tokens, head_dim = k.shape
    assert ape.shape == (index_kpool, head_dim), ape.shape

    assert pending_kv.ndim == 4, pending_kv.shape
    num_slots, pending_depth, two, pend_dim = pending_kv.shape
    assert two == 2 and pend_dim == head_dim, pending_kv.shape
    assert pending_tag.shape == (num_slots, pending_depth), pending_tag.shape
    assert pending_tag.dtype == jnp.int32, pending_tag.dtype
    # f32, and load-bearing: it is what makes the in-step and carried gather
    # sources bit-identical, which is what makes the chunked-prefill test an
    # exact equality instead of a tolerance. See the module docstring.
    assert pending_kv.dtype == jnp.float32, (
        f"pending_kv must be float32, got {pending_kv.dtype}. A bf16 ring "
        "saves ~2.7 MiB and makes the same prompt produce a different pooled "
        "key depending on where the scheduler cut the chunk.")
    assert pending_depth >= index_kpool, (
        f"pending_depth={pending_depth} < index_kpool={index_kpool}; a single "
        "pool would alias itself in the ring. Build the depth with "
        "derive_pending_state_geometry().")

    # `positions` must be 1-D absolute positions. If GLM's vision path ever
    # makes them mrope/3-D this fires at trace time instead of silently
    # addressing the ring with a rope section index.
    assert positions.ndim == 1 and positions.shape[0] == num_tokens, (
        f"positions must be 1-D absolute positions of shape ({num_tokens},); "
        f"got {positions.shape}. A 3-D mrope position tensor would address "
        "the pending ring and the pooled pages with a rope-section index.")

    num_seqs = seq_lens.shape[0]
    assert cu_q_lens.shape[0] == num_seqs + 1, (cu_q_lens.shape, num_seqs)
    assert kpool_state_indices.shape[0] == num_seqs, (
        f"kpool_state_indices has {kpool_state_indices.shape[0]} entries for "
        f"{num_seqs} sequences")
    assert pooled_block_tables.ndim == 1, pooled_block_tables.shape
    assert pooled_block_tables.shape[0] % num_seqs == 0, (
        f"pooled_block_tables ({pooled_block_tables.shape[0]}) is not a whole "
        f"number of rows for {num_seqs} sequences")

    _, page_size_per_kv_packing, kv_packing, record_bytes = pooled_cache.shape
    page_size = page_size_per_kv_packing * kv_packing
    assert record_bytes >= head_dim + 1, (
        f"pooled cache row is {record_bytes} B but a record needs "
        f"{head_dim} value bytes + 1 e8m0 scale byte")

    # f32 throughout: the ring is f32, so upcasting here makes the two gather
    # sources bit-identical for the same position.
    k = k.astype(jnp.float32)
    gate = gate.astype(jnp.float32)
    ape = ape.astype(jnp.float32)
    seq_lens = seq_lens.astype(jnp.int32)
    cu_q_lens = cu_q_lens.astype(jnp.int32)

    idx = index_tokens(seq_lens, cu_q_lens, num_tokens)

    # THE `first_key == 0` ASSERTION (see also pool_math's docstring
    # names this function as the place it belongs). `ape` is indexed by the
    # token's slot within its pool, which this code computes as
    # `pos % index_kpool` with `pos` derived from seq_lens/cu_q_lens. The
    # reference indexes it by `(idx - first_key) % index_kpool`, and the two
    # agree only because vLLM's unpadded ragged layout puts `first_key` at 0 --
    # i.e. only because the runner's own `input_positions` equal the derived
    # positions. Counting the disagreement is the check; a Python `assert` on
    # a traced array is not available inside a jit.
    position_mismatches = jnp.sum(
        (idx.valid & (positions.astype(jnp.int32) != idx.pos)).astype(
            jnp.int32))

    # --- slot sanity, BEFORE the clip -------------------------------------
    #
    # The clip below is what keeps a bad slot from scattering out of bounds,
    # and it is also what makes a bad slot *unobservable* afterwards: an
    # out-of-range index folds onto `num_slots - 1`, which is a live request's
    # slot, and a duplicated index was already one. The tag check cannot see
    # either, because a tag is an absolute position and absolute positions
    # collide across requests -- two sequences sharing a slot read each other's
    # rows with tags that match perfectly. These two counters are the only
    # thing in this file that can. See DIAG_SLOT_DUPLICATES.
    raw_slots = kpool_state_indices.astype(jnp.int32)
    q_len_seq = cu_q_lens[1:num_seqs + 1] - cu_q_lens[:num_seqs]
    live_seq = q_len_seq > 0
    slot_in_range = (raw_slots >= 1) & (raw_slots < num_slots)
    slot_out_of_range = jnp.sum((live_seq & ~slot_in_range).astype(jnp.int32))
    # One-hot over slots, live and in-range sequences only: an out-of-range
    # entry is already counted above and must not be double-counted here.
    countable = live_seq & slot_in_range
    one_hot = jnp.where(countable[:, None], (raw_slots[:, None] == jnp.arange(
        num_slots, dtype=jnp.int32)[None, :]), False).astype(jnp.int32)
    slot_duplicates = jnp.sum(jnp.maximum(jnp.sum(one_hot, axis=0) - 1, 0))

    # Per sequence for the ring read, per token for the ring write. A padded
    # batch tail has no sequence of its own, so its token-level slot is the
    # null sentinel 0.
    slot_seq = jnp.clip(raw_slots, 0, num_slots - 1)
    slot_id = jnp.where(idx.valid, slot_seq[idx.seq_id], jnp.int32(0))

    pooled, boundary, pool_id, tag_violations = kpool_pooled_rows(
        pending_kv,
        pending_tag,
        k,
        gate,
        ape,
        idx,
        slot_seq,
        seq_lens,
        cu_q_lens,
        index_kpool=index_kpool)

    records = pack_fp8_ue8m0(pooled)  # uint8[T, D + 1]

    block_tables_2d = pooled_block_tables.astype(jnp.int32).reshape(
        num_seqs, -1)
    page = jnp.where(boundary, block_tables_2d[idx.seq_id,
                                               pool_id // page_size],
                     jnp.int32(OOB_PAGE))
    slot = jnp.remainder(pool_id, page_size)
    pooled_cache = _scatter_pooled_records(pooled_cache, records, page, slot,
                                           kv_packing)

    pending_kv, pending_tag = _write_pending_state(pending_kv, pending_tag, k,
                                                   gate, idx, slot_id)

    diagnostics = jnp.zeros((KPOOL_WRITE_DIAG_SIZE, ), jnp.int32)
    diagnostics = diagnostics.at[DIAG_TAG_VIOLATIONS].set(tag_violations)
    diagnostics = diagnostics.at[DIAG_POSITION_MISMATCHES].set(
        position_mismatches)
    diagnostics = diagnostics.at[DIAG_SLOT_OUT_OF_RANGE].set(slot_out_of_range)
    diagnostics = diagnostics.at[DIAG_SLOT_DUPLICATES].set(slot_duplicates)
    return pooled_cache, pending_kv, pending_tag, diagnostics


def read_pooled_records(pooled_cache: jnp.ndarray, pool_ids: jnp.ndarray,
                        block_table_row: jnp.ndarray) -> jnp.ndarray:
    """Read pooled records back out of the packed cache: ``uint8[N, R]``.

    The inverse addressing of :func:`kpool_write`'s scatter, so a test can
    check what landed without restating the page arithmetic in the test
    itself. ``block_table_row`` is one sequence's row of the block table.
    """
    _, page_size_per_kv_packing, kv_packing, _ = pooled_cache.shape
    page_size = page_size_per_kv_packing * kv_packing
    pool_ids = jnp.asarray(pool_ids, jnp.int32)
    page = jnp.asarray(block_table_row, jnp.int32)[pool_ids // page_size]
    slot = jnp.remainder(pool_ids, page_size)
    return pooled_cache[page, slot // kv_packing, slot % kv_packing]


__all__ = [
    "DIAG_NAMES",
    "DIAG_POSITION_MISMATCHES",
    "DIAG_SLOT_DUPLICATES",
    "DIAG_SLOT_OUT_OF_RANGE",
    "DIAG_TAG_VIOLATIONS",
    "KPOOL_WRITE_DIAG_SIZE",
    "TokenIndex",
    "OOB_PAGE",
    "PENDING_TAG_EMPTY",
    "PendingStateGeometry",
    "assert_pending_depth_sufficient",
    "clear_pending_slots",
    "derive_pending_state_geometry",
    "describe_diagnostics",
    "empty_pending_state",
    "index_tokens",
    "kpool_pooled_rows",
    "kpool_write",
    "read_carried_rows",
    "read_pooled_records",
]
