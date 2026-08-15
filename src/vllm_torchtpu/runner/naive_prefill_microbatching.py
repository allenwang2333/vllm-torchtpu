# SPDX-License-Identifier: Apache-2.0
"""Request-safe planning for two-microbatch prefill steps.

The feature splits one step's work into two microbatches so their
expert-parallel collectives can be overlapped. This module owns the part of
that decision which needs no device state: which local work is eligible, how
the step is partitioned, what every rank must agree on, and the padded length
the staged forwards are required to use.

The planner answers one question: **can this step be cut into two independent
halves that together cost no more device time than running it serially?**

Three rules come out of that, and they are the whole design.

1. *Two real groups, never a dummy partner.* A slot with no work still runs a
   full padded forward, so one real request paired with a dummy costs
   ``2 x pad(T)`` against a serial ``pad(T)``. That is strictly more work, so
   the cost rule below rejects it -- see ``allow_dummy_partner`` for the
   measurement-only override.
2. *Cost rule:* ``2 * pad(max(tokens_a, tokens_b)) <= pad(total_tokens)``.
   Both sides are padding buckets, so this is exact, not an estimate.
3. *Independence:* the two groups must share no request. vLLM schedules each
   request at most once per step, so partitioning the step's request list is
   enough to guarantee it -- see ``partition_is_independent``.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

# The legacy fixed shape. Still the default microbatch bucket, but no longer
# the only one a staged step may use.
PREFILL_MICROBATCH_TOKENS = 1024
NUM_PREFILL_MICROBATCHES = 2

# Microbatch lengths the staged graph is precompiled for. Every entry costs a
# full graph compile at startup (~54 s on the first touch, record #26), so this
# is deliberately short rather than "every token bucket". A microbatch far
# below 512 tokens has too little expert compute to hide a collective behind
# anyway, which is the only reason to stage at all.
DEFAULT_STAGED_MICROBATCH_BUCKETS: tuple[int, ...] = (512, 1024)


class PrefillStepMode(IntEnum):
    """What graph a step runs, agreed DP-wide before any forward launches.

    Ordered so a MAX reduction over the ranks' local modes yields the
    coordinated mode: an idle rank must not veto an eligible one, and any rank
    that cannot stage forces every rank onto the serial path.
    """
    IDLE = 0
    TWO_MICROBATCH = 1
    SERIAL = 2


@dataclass(frozen=True)
class PrefillMicrobatch:
    """One slot: a contiguous run of the step's scheduled requests.

    Contiguity is not cosmetic. ``TPUModelRunner._prepare_inputs`` builds a
    forward's tensors from a ``[start_index, end_index)`` range of the input
    batch, so a slot that needed a scattered set of requests could not be
    prepared without a gather that does not exist today. See
    ``plan_two_microbatch_step`` for what this costs.
    """
    request_ids: tuple[str, ...]
    start_index: int
    end_index: int
    num_tokens: int
    output_index: int

    def __post_init__(self) -> None:
        if self.end_index - self.start_index != len(self.request_ids):
            raise ValueError("microbatch index range must match its requests")

    @property
    def num_reqs(self) -> int:
        return len(self.request_ids)


@dataclass(frozen=True)
class TwoMicrobatchPlan:
    """A step's partition into two slots, plus the shape they must launch at.

    ``microbatch_tokens`` is the padded length **both** slots use, real or
    dummy. It is carried on the plan rather than recomputed at the launch site
    because it is the value the DP ranks have to agree on, and one field that
    every consumer reads is easier to keep honest than three derivations.
    """
    microbatches: tuple[PrefillMicrobatch, ...]
    microbatch_tokens: int
    serial_tokens: int

    def __post_init__(self) -> None:
        if not 1 <= len(self.microbatches) <= NUM_PREFILL_MICROBATCHES:
            raise ValueError("a plan requires one or two real microbatches")
        if self.microbatch_tokens <= 0:
            raise ValueError("microbatch_tokens must be positive")

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(request_id for microbatch in self.microbatches
                     for request_id in microbatch.request_ids)

    @property
    def num_real_microbatches(self) -> int:
        return len(self.microbatches)

    @property
    def num_reqs_per_microbatch(self) -> list[int]:
        return [microbatch.num_reqs for microbatch in self.microbatches]

    @property
    def staged_cost_tokens(self) -> int:
        """Padded token-work the staged step will do, dummies included."""
        return NUM_PREFILL_MICROBATCHES * self.microbatch_tokens

    @property
    def imbalance(self) -> int:
        """How much more real work one slot carries than the other.

        The tiebreaker between equal-cost plans. Both slots run for the same
        padded length, so the slot with less real work spends the difference
        computing padding -- and padding has no expert routing to overlap the
        other slot's collectives against.

        A one-real-microbatch plan (the ``allow_dummy_partner`` measurement
        mode) reports the whole slot length: its partner is entirely padding.
        """
        if len(self.microbatches) < NUM_PREFILL_MICROBATCHES:
            return self.microbatch_tokens
        return abs(self.microbatches[0].num_tokens -
                   self.microbatches[1].num_tokens)

    @property
    def cost_key(self) -> tuple[int, int]:
        """Order plans by device cost, then by how much work can overlap."""
        return (self.staged_cost_tokens, self.imbalance)

    def restore_output_order(self, outputs: Sequence[Any]) -> list[Any]:
        """Map slot-ordered outputs back onto the scheduler's request order."""
        if len(outputs) != self.num_real_microbatches:
            raise ValueError("one output is required per real microbatch")
        restored: list[Any] = [None] * self.num_real_microbatches
        for output, microbatch in zip(outputs, self.microbatches):
            restored[microbatch.output_index] = output
        return restored


def padded_token_len(paddings: Sequence[int], num_tokens: int) -> int | None:
    """Smallest bucket that fits ``num_tokens``, or None if none does."""
    for bucket in paddings:
        if bucket >= num_tokens:
            return int(bucket)
    return None


def partition_is_independent(
        left: Sequence[str], right: Sequence[str]) -> bool:
    """Whether two slots can run in one graph without a data dependency.

    The staged graph interleaves the two slots, so slot 1 may read KV cache
    lines while slot 0's attention is still writing its own. That is safe
    exactly when no request appears on both sides: distinct requests own
    disjoint KV blocks, and a request's dependency on its *earlier* chunks was
    satisfied by earlier steps, not by this one.

    Restating the case that motivates it, because it is the one that looks
    dangerous and is not. An 8k prompt chunked into eight 1024-token steps has
    a strict chain -- chunk 3 attends over the KV that chunks 0-2 wrote. Those
    chunks are in *different steps*, so they are never two slots of one staged
    step. vLLM schedules a request at most once per step, so the chain can only
    ever be split across steps, which is what makes chunked prefill admissible
    at all.
    """
    return not (set(left) & set(right))


def plan_two_microbatch_step(
    request_ids: Sequence[str | None],
    num_scheduled_tokens: Mapping[str, int],
    token_buckets: Sequence[int],
    *,
    microbatch_buckets: Sequence[int] | None = None,
    max_reqs_per_microbatch: int | None = None,
    allow_dummy_partner: bool = False,
) -> TwoMicrobatchPlan | None:
    """Partition one step into two independent, no-more-costly microbatches.

    Returns ``None`` for any step that should run serially, which is the
    caller's signal to vote ``SERIAL`` in the DP reduction.

    ``token_buckets`` are the runner's token paddings and define what
    "cost" means: a forward of *n* tokens costs the bucket it pads up to, not
    *n*. Both the serial and the staged cost are measured in those buckets, so
    the comparison is exact.

    ``microbatch_buckets`` restricts the per-slot padded length to shapes the
    staged graph was precompiled for. A length outside it would compile
    mid-step while eight peers waited inside the collectives they had already
    entered -- so a split that needs an uncompiled shape is not admitted.

    Search is over contiguous split points only, for the reason in
    ``PrefillMicrobatch``. That does forfeit real cases: a step of
    ``[512, 1024, 512]`` has a perfect balanced partition, but not a contiguous
    one, so it runs serial. Reordering the input batch would fix it and is a
    much larger change.
    """
    live_ids = [request_id for request_id in request_ids
                if request_id is not None]
    if not live_ids:
        return None
    if len(set(live_ids)) != len(live_ids):
        return None

    tokens = [int(num_scheduled_tokens.get(request_id, -1))
              for request_id in live_ids]
    if any(count <= 0 for count in tokens):
        return None

    buckets = sorted(int(bucket) for bucket in token_buckets)
    allowed = (sorted(int(bucket) for bucket in microbatch_buckets)
               if microbatch_buckets is not None else buckets)
    if not buckets or not allowed:
        return None

    total_tokens = sum(tokens)
    serial_tokens = padded_token_len(buckets, total_tokens)
    if serial_tokens is None:
        return None

    def _microbatch(start: int, end: int, output_index: int
                    ) -> PrefillMicrobatch:
        return PrefillMicrobatch(
            request_ids=tuple(live_ids[start:end]),
            start_index=start, end_index=end,
            num_tokens=sum(tokens[start:end]), output_index=output_index)

    def _fits(num_reqs: int) -> bool:
        return (max_reqs_per_microbatch is None
                or num_reqs <= max_reqs_per_microbatch)

    # A dummy partner is never cheaper than serial -- 2*pad(T) against pad(T)
    # -- so it exists only for measurement, where a single request has to be
    # forced down the staged path to be observed at all.
    if allow_dummy_partner and len(live_ids) == 1:
        bucket = padded_token_len(allowed, total_tokens)
        if bucket is None or not _fits(1):
            return None
        return TwoMicrobatchPlan((_microbatch(0, 1, 0),), bucket,
                                 serial_tokens)

    best: TwoMicrobatchPlan | None = None
    best_key: tuple[int, int] | None = None
    for split in range(1, len(live_ids)):
        left_tokens = sum(tokens[:split])
        right_tokens = sum(tokens[split:])
        if not _fits(split) or not _fits(len(live_ids) - split):
            continue
        # Redundant with the duplicate-id rejection above, deliberately: a
        # partition of a duplicate-free list cannot share a request, and a
        # list with a duplicate has no independent partition. Either check
        # alone rejects every unsafe plan, so deleting one is invisible to a
        # test -- which is exactly why both stay. This is the invariant the
        # staged graph's correctness rests on, and it is worth two spellings.
        if not partition_is_independent(live_ids[:split], live_ids[split:]):
            continue
        bucket = padded_token_len(allowed, max(left_tokens, right_tokens))
        if bucket is None:
            continue
        staged_cost = NUM_PREFILL_MICROBATCHES * bucket
        if staged_cost > serial_tokens:
            continue
        # Cheapest first; among equals prefer the most balanced split, which
        # leaves the least padding inside each slot and so the most real
        # expert work available to overlap.
        key = (staged_cost, abs(left_tokens - right_tokens))
        if best_key is None or key < best_key:
            best_key = key
            best = TwoMicrobatchPlan(
                (_microbatch(0, split, 0),
                 _microbatch(split, len(live_ids), 1)),
                bucket, serial_tokens)
    return best


@dataclass(frozen=True)
class ReorderedPartition:
    """A two-slot partition that the current request order cannot express.

    ``plan_two_microbatch_step`` may only cut the input batch at one point, so
    a step like ``[500, 1000, 512]`` has no admissible plan even though
    ``{500, 512} | {1000}`` fits two 1024-token slots exactly. This is that
    partition, named by *indices into the current order* plus the permutation
    that makes it contiguous. Applying the permutation is the caller's job --
    it owns the input batch -- after which the ordinary planner finds this
    split on its own.

    ``left_indices`` and ``right_indices`` keep their relative order from the
    current batch, so a partition that is already contiguous permutes nothing.
    """
    left_indices: tuple[int, ...]
    right_indices: tuple[int, ...]
    left_tokens: int
    right_tokens: int
    microbatch_tokens: int
    serial_tokens: int

    @property
    def new_order(self) -> tuple[int, ...]:
        """Current-order indices in the order the batch must be permuted to."""
        return self.left_indices + self.right_indices

    @property
    def staged_cost_tokens(self) -> int:
        return NUM_PREFILL_MICROBATCHES * self.microbatch_tokens

    @property
    def imbalance(self) -> int:
        return abs(self.left_tokens - self.right_tokens)

    @property
    def cost_key(self) -> tuple[int, int]:
        return (self.staged_cost_tokens, self.imbalance)

    def is_already_contiguous(self) -> bool:
        """Whether the current order already expresses this partition."""
        order = self.new_order
        return all(later > earlier
                   for earlier, later in zip(order, order[1:]))


def _reachable_subset_sums(tokens: Sequence[int]) -> list[list[int]]:
    """Layered subset-sum reachability, one bitmask per (prefix, count).

    ``layers[i][k]`` has bit *s* set exactly when some ``k`` of the first ``i``
    items sum to ``s``. The whole table is built with one shift-or per
    (item, count) pair by carrying each row as a Python big integer -- bit *s*
    of the integer stands for "sum *s* is reachable", so ``mask << t`` adds
    item ``t`` to every reachable sum at once.

    Exact rather than greedy on purpose. A greedy longest-first packer solves
    ``[500, 1000, 512]`` but misses ``[600, 600, 500, 500]`` into two 1100s,
    and a partitioner that is *usually* right produces a staged step whose
    slots do not fit their bucket -- which is not a slower step, it is a
    refused one, so the cost of being wrong is silent lost yield rather than a
    visible error.

    The count dimension exists because a slot is one ``_prepare_inputs`` call
    and so holds at most ``max_reqs_per_microbatch`` requests. Sums are bounded
    by ``max_num_batched_tokens``, which keeps every mask a couple of thousand
    bits wide.
    """
    num_items = len(tokens)
    row = [0] * (num_items + 1)
    row[0] = 1
    layers = [list(row)]
    for index, count in enumerate(tokens, start=1):
        for taken in range(min(index, num_items), 0, -1):
            row[taken] |= row[taken - 1] << count
        layers.append(list(row))
    return layers


def _closest_reachable_sum(mask: int, target: int, low: int,
                           high: int) -> int | None:
    """The reachable sum in ``[low, high]`` nearest ``target``, or None.

    Nearest to the balance point, because among partitions that fit the same
    bucket the most balanced one leaves the least padding in either slot.
    """
    if high < low:
        return None
    window = (mask >> low) & ((1 << (high - low + 1)) - 1)
    if not window:
        return None
    start = min(max(target, low), high)
    for offset in range(0, high - low + 1):
        for candidate in (start - offset, start + offset):
            if low <= candidate <= high and (mask >> candidate) & 1:
                return candidate
    return None


def plan_reordered_two_microbatch_step(
    request_ids: Sequence[str | None],
    num_scheduled_tokens: Mapping[str, int],
    token_buckets: Sequence[int],
    *,
    microbatch_buckets: Sequence[int] | None = None,
    max_reqs_per_microbatch: int | None = None,
    num_frozen_prefix: int = 0,
) -> ReorderedPartition | None:
    """Find the cheapest two-slot partition, ignoring the current order.

    The ragged case this exists for: vLLM hands the step ``[500, 1000, 512]``,
    which has no admissible *contiguous* split -- every cut leaves one side
    over 1024 or forces a 2048-token slot -- while the partition
    ``{500, 512} | {1000}`` fits two 1024-token slots and costs exactly what
    the serial step costs. Reordering the input batch to ``[500, 512, 1000]``
    is what turns the second one into a split the runner can prepare.

    ``num_frozen_prefix`` is the number of leading requests that may not move:
    ``TPUModelRunner._reorder_batch_for_rpa`` puts 1-token decodes at the front
    of the batch and the ragged-paged-attention kernel is told where that
    segment ends, so permuting across the boundary would describe a decode to
    the kernel as prefill. Those requests stay where they are and land in the
    left slot, which is the batch prefix either way.

    Returns None when no permutation admits a staged step, which leaves the
    caller on whatever the ordinary planner decided.
    """
    live_ids = [request_id for request_id in request_ids
                if request_id is not None]
    if not live_ids:
        return None
    if len(set(live_ids)) != len(live_ids):
        return None

    tokens = [int(num_scheduled_tokens.get(request_id, -1))
              for request_id in live_ids]
    if any(count <= 0 for count in tokens):
        return None

    buckets = sorted(int(bucket) for bucket in token_buckets)
    allowed = (sorted(int(bucket) for bucket in microbatch_buckets)
               if microbatch_buckets is not None else buckets)
    if not buckets or not allowed:
        return None

    total_tokens = sum(tokens)
    serial_tokens = padded_token_len(buckets, total_tokens)
    if serial_tokens is None:
        return None

    frozen = max(0, min(int(num_frozen_prefix), len(live_ids)))
    movable = tokens[frozen:]
    if not movable:
        return None
    frozen_tokens = sum(tokens[:frozen])
    num_movable = len(movable)
    cap = max_reqs_per_microbatch

    layers = _reachable_subset_sums(movable)
    best: tuple[tuple[int, int], int, int, int] | None = None
    for bucket in allowed:
        if NUM_PREFILL_MICROBATCHES * bucket > serial_tokens:
            continue
        # Left slot carries the frozen decodes plus a subset summing to s;
        # right slot carries the rest. Both must fit this bucket.
        low = max(0, total_tokens - frozen_tokens - bucket)
        high = bucket - frozen_tokens
        # Balanced means the two slots hold equal real work, which puts the
        # subset at half of what is left once the frozen prefix is placed.
        target = (total_tokens - 2 * frozen_tokens) // 2
        for taken in range(0, num_movable + 1):
            left_reqs = frozen + taken
            right_reqs = num_movable - taken
            if left_reqs == 0 or right_reqs == 0:
                continue
            if cap is not None and (left_reqs > cap or right_reqs > cap):
                continue
            subset_sum = _closest_reachable_sum(layers[num_movable][taken],
                                                target, low, high)
            if subset_sum is None:
                continue
            left_tokens = frozen_tokens + subset_sum
            right_tokens = total_tokens - left_tokens
            key = (NUM_PREFILL_MICROBATCHES * bucket,
                   abs(left_tokens - right_tokens))
            if best is None or key < best[0]:
                best = (key, bucket, taken, subset_sum)
        if best is not None:
            # Buckets ascend and cost is 2*bucket, so nothing later can be
            # cheaper. Only the imbalance tiebreak is still open, and it is
            # already minimised within this bucket.
            break
    if best is None:
        return None

    _, bucket, taken, subset_sum = best
    chosen = _recover_subset(layers, movable, taken, subset_sum)
    left_movable = sorted(index + frozen for index in chosen)
    right_movable = [index for index in range(frozen, len(live_ids))
                     if index - frozen not in chosen]
    left_indices = tuple(range(frozen)) + tuple(left_movable)
    right_indices = tuple(right_movable)
    if frozen == 0 and right_indices and left_indices and (
            right_indices[0] < left_indices[0]):
        # Which group is slot 0 is free when nothing is frozen, and the two
        # choices are not equally good: slot 0 is the batch *prefix*, so the
        # group holding the earliest request belongs there. Without this the
        # search can return the mirror image of a partition the current order
        # already expresses, and report a permutation where none is needed.
        left_indices, right_indices = right_indices, left_indices
    left_tokens = sum(tokens[index] for index in left_indices)
    right_tokens = total_tokens - left_tokens
    if max(left_tokens, right_tokens) > bucket:
        # Internal consistency, not input validation: a slot over its bucket
        # means the reachability window and the reconstruction disagree. The
        # downstream re-plan would refuse the permutation and the step would
        # merely run serial, so the damage is invisible yield rather than a
        # wrong result -- which is exactly why it is worth raising on.
        raise RuntimeError(
            f"partition put {max(left_tokens, right_tokens)} tokens in a "
            f"{bucket}-token slot")
    return ReorderedPartition(left_indices=left_indices,
                              right_indices=right_indices,
                              left_tokens=left_tokens,
                              right_tokens=right_tokens,
                              microbatch_tokens=bucket,
                              serial_tokens=serial_tokens)


def _recover_subset(layers: Sequence[Sequence[int]], tokens: Sequence[int],
                    count: int, total: int) -> set[int]:
    """Which items the reachability table used to reach ``(count, total)``.

    Walks the layers backwards: item ``i-1`` was taken exactly when the sum
    left over is reachable one layer down with one fewer item. Ties are broken
    arbitrarily -- several subsets often reach the same sum, and they are
    interchangeable, because what the caller does with the result is decided
    by sums and counts alone.
    """
    chosen: set[int] = set()
    remaining_count = count
    remaining_total = total
    for index in range(len(tokens), 0, -1):
        value = tokens[index - 1]
        if (remaining_count > 0 and remaining_total >= value
                and (layers[index - 1][remaining_count - 1] >>
                     (remaining_total - value)) & 1):
            chosen.add(index - 1)
            remaining_count -= 1
            remaining_total -= value
    if remaining_count != 0 or remaining_total != 0:
        raise RuntimeError("subset reconstruction disagreed with the "
                           "reachability table it was built from")
    return chosen


def build_two_microbatch_plan(
    request_ids: Sequence[str | None],
    num_scheduled_tokens: Mapping[str, int],
    num_computed_tokens: Mapping[str, int] | Sequence[int],
) -> TwoMicrobatchPlan | None:
    """The original fixed envelope: one or two fresh 1024-token requests.

    Retained so the fixed-shape experiment stays reproducible bit for bit
    against the numbers already recorded for it (records #25, #26, #30).
    ``plan_two_microbatch_step`` is what the runner uses.
    """
    live_ids = [request_id for request_id in request_ids
                if request_id is not None]
    if not 1 <= len(live_ids) <= NUM_PREFILL_MICROBATCHES:
        return None
    if len(set(live_ids)) != len(live_ids):
        return None
    if isinstance(num_computed_tokens, Mapping):
        computed = [int(num_computed_tokens.get(req_id, -1))
                    for req_id in live_ids]
    else:
        if len(num_computed_tokens) < len(live_ids):
            return None
        computed = [int(num_computed_tokens[i])
                    for i in range(len(live_ids))]
    if any(value != 0 for value in computed):
        return None
    if any(int(num_scheduled_tokens.get(req_id, -1)) !=
           PREFILL_MICROBATCH_TOKENS
           for req_id in live_ids):
        return None
    batches = tuple(
        PrefillMicrobatch(request_ids=(req_id,), start_index=i, end_index=i + 1,
                          num_tokens=PREFILL_MICROBATCH_TOKENS, output_index=i)
        for i, req_id in enumerate(live_ids))
    return TwoMicrobatchPlan(batches, PREFILL_MICROBATCH_TOKENS,
                             PREFILL_MICROBATCH_TOKENS * len(live_ids))


def resolve_staged_microbatch_buckets(
        token_buckets: Sequence[int],
        configured: Sequence[int] | None = None) -> tuple[int, ...]:
    """Staged microbatch lengths to precompile, validated against the paddings.

    A staged bucket must itself be a token padding: the slot's tensors are
    built by the ordinary preparation path, which only ever produces padded
    lengths, and the warmup dummy run has to reach the same shape.
    """
    requested = (tuple(DEFAULT_STAGED_MICROBATCH_BUCKETS)
                 if configured is None else tuple(configured))
    known = {int(bucket) for bucket in token_buckets}
    resolved = tuple(sorted({int(bucket) for bucket in requested
                             if int(bucket) in known}))
    if not resolved:
        raise ValueError(
            f"no staged microbatch bucket in {sorted(requested)} is a token "
            f"padding bucket {sorted(known)}; the staged graph can only be "
            "traced at shapes the ordinary preparation path produces")
    return resolved


def staged_token_bucket(mode: PrefillStepMode, step_bucket: int | None,
                        microbatch_tokens: int | None = None) -> int | None:
    """Resolve the padded token length a step's forwards must all use.

    ``step_bucket`` is the DP max-reduction over each rank's *whole-step*
    scheduled token count, which is the right padding target only when a rank's
    step is one forward. ``TWO_MICROBATCH`` decomposes the step into
    ``NUM_PREFILL_MICROBATCHES`` forwards of ``microbatch_tokens``, so the step
    total must not reach a per-microbatch padding decision.

    This is not a preference. The staged graph's EP collectives are in-graph,
    so a rank that pads its microbatches differently from its peers compiles a
    structurally different program, the collective operand sizes disagree, and
    every core blocks forever with no error. The fixed-shape version of this
    feature hid the problem entirely: with one 1024-token request per rank the
    step total and the microbatch length were the same number by coincidence.
    Once slots may hold several requests of unequal length, they are not, and
    ``microbatch_tokens`` has to come from the DP reduction rather than from
    anything this rank measured on its own.
    """
    if mode is PrefillStepMode.TWO_MICROBATCH:
        return (PREFILL_MICROBATCH_TOKENS if microbatch_tokens is None
                else int(microbatch_tokens))
    return step_bucket


def coordinated_microbatch_tokens(
        local_microbatch_tokens: Sequence[int],
        fallback: int = PREFILL_MICROBATCH_TOKENS) -> int:
    """Reduce the ranks' desired slot lengths to the one they all launch at.

    MAX, matching the mode reduction it rides along with, and for the same
    reason: a rank must never launch a *smaller* program than a peer, because
    the peer's in-graph collective would then wait on operands that were never
    sent. Padding up is wasteful; padding down hangs.

    ``fallback`` covers the case where no rank asked for a length -- which
    cannot coincide with a staged step, since a rank votes ``TWO_MICROBATCH``
    only when it holds a plan and a plan always carries a length.
    """
    lengths = [int(value) for value in local_microbatch_tokens if value > 0]
    if not lengths:
        return int(fallback)
    return max(lengths)


def staged_step_is_cheaper(microbatch_tokens: int,
                           serial_tokens: int) -> bool:
    """The cost rule, evaluated on DP-coordinated values.

    Both inputs are MAX-reductions over the ranks, and both are monotonic in
    the same per-rank quantities, so a group in which every busy rank passed
    its own local check passes this one too. It is asserted globally anyway:
    the check is a few host-side integers, and the failure it guards against
    -- a whole DP group doing double work while reporting parity -- is one that
    a benchmark cannot distinguish from the feature simply not helping.
    """
    return NUM_PREFILL_MICROBATCHES * int(microbatch_tokens) <= int(
        serial_tokens)


def validate_staged_microbatch_tokens(num_tokens: int, *, role: str,
                                      index: int, field: str,
                                      expected: int) -> None:
    """Reject a staged bundle whose padded token length is not the agreed one.

    Raised locally at bundle-construction time so a divergence surfaces as one
    rank's traceback instead of a silent DP-wide hang. See
    :func:`staged_token_bucket` for why the lengths must agree.
    """
    if int(num_tokens) != int(expected):
        raise RuntimeError(
            f"staged prefill bundle {index} ({role}) has {field} padded to "
            f"{int(num_tokens)} tokens; every rank must pad every staged "
            f"microbatch to exactly {int(expected)} for this step, or the "
            "in-graph EP collectives desync and halt every core")


def validate_staged_microbatch_bundle(bundle: Any, *, role: str,
                                      index: int, expected: int) -> None:
    """Check every length the staged graph is traced against.

    Both roles build their bundles independently -- the busy rank through the
    real ``_prepare_inputs`` path and the idle rank through ``prepare_dummy``
    -- so nothing structurally forces them to agree. This is the checkpoint
    that does, and it runs before the launch that would otherwise hang.
    """
    validate_staged_microbatch_tokens(bundle.input_ids.shape[0], role=role,
                                      index=index, field="input_ids",
                                      expected=expected)
    validate_staged_microbatch_tokens(bundle.position_ids.shape[-1], role=role,
                                      index=index, field="position_ids",
                                      expected=expected)
    layout = bundle.layout_plan
    if layout is not None:
        validate_staged_microbatch_tokens(
            layout.global_padded_num_tokens, role=role, index=index,
            field="layout global_padded_num_tokens", expected=expected)
        validate_staged_microbatch_tokens(
            layout.local_padded_num_tokens, role=role, index=index,
            field="layout local_padded_num_tokens", expected=expected)


def local_step_mode(total_num_scheduled_tokens: int,
                    plan: TwoMicrobatchPlan | None) -> PrefillStepMode:
    """Classify local scheduler work before the DP-wide reduction."""
    if total_num_scheduled_tokens == 0:
        return PrefillStepMode.IDLE
    if plan is not None:
        return PrefillStepMode.TWO_MICROBATCH
    return PrefillStepMode.SERIAL


def coordinated_step_mode(modes: Sequence[PrefillStepMode]) -> PrefillStepMode:
    """Choose one graph globally; idle ranks do not veto eligible ranks."""
    busy = [mode for mode in modes if mode != PrefillStepMode.IDLE]
    if not busy:
        return PrefillStepMode.IDLE
    if all(mode == PrefillStepMode.TWO_MICROBATCH for mode in busy):
        return PrefillStepMode.TWO_MICROBATCH
    return PrefillStepMode.SERIAL


def validate_feature_config(vllm_config: Any) -> None:
    """Fail fast when opt-in configuration is outside the supported envelope.

    Relaxed from the fixed-shape experiment's exact equalities to bounds: the
    planner now handles unequal lengths, several requests per slot, and
    partially computed (chunked) requests, so pinning ``max_num_seqs`` to 2
    would rule out the workloads the planner exists to serve. What stays is
    what the *device* path still assumes: one Qwen3.5-MoE replica per rank
    across eight expert-parallel ranks.
    """
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    scheduler = vllm_config.scheduler_config
    architectures = getattr(model.hf_config, "architectures", ()) or ()
    errors: list[str] = []
    if not any("Qwen3_5Moe" in architecture for architecture in architectures):
        errors.append("model must be Qwen3.5-MoE")
    if int(parallel.data_parallel_size) != 8 or not parallel.enable_expert_parallel:
        errors.append("expert parallelism must be enabled with EP/DP size 8")
    if int(parallel.tensor_parallel_size) != 1:
        errors.append("tensor parallel size must be 1")
    if int(scheduler.max_num_batched_tokens) < 2 * min(
            DEFAULT_STAGED_MICROBATCH_BUCKETS):
        errors.append(
            "max_num_batched_tokens must be at least "
            f"{2 * min(DEFAULT_STAGED_MICROBATCH_BUCKETS)}, or no step can "
            "hold two microbatches")
    if int(scheduler.max_num_seqs) < 2:
        errors.append("max_num_seqs must be at least 2, or a step can never "
                      "hold two independent requests")
    if int(model.max_model_len) < 2048:
        errors.append("max_model_len must be at least 2048")
    if getattr(vllm_config, "speculative_config", None) is not None:
        errors.append("speculative decoding must be disabled")
    if bool(getattr(vllm_config.cache_config, "enable_prefix_caching", False)):
        errors.append("prefix caching must be disabled")
    if errors:
        raise ValueError("VLLM_TPU_NAIVE_PREFILL_MICROBATCHING: " +
                         "; ".join(errors))


def validate_dummy_partner_config(microbatching_enabled: bool) -> None:
    """The dummy-partner override is a measurement mode, not a serving one."""
    if not microbatching_enabled:
        raise ValueError(
            "VLLM_TPU_NAIVE_PREFILL_ALLOW_DUMMY_PARTNER requires "
            "VLLM_TPU_NAIVE_PREFILL_MICROBATCHING=1")
