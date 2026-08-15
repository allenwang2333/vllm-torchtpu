# SPDX-License-Identifier: Apache-2.0
"""Reordering the input batch so a ragged step has a staged partition.

A microbatch slot is one contiguous ``[start, end)`` range of the input batch,
so the cost-aware planner may only cut a step at a single point. vLLM does not
hand out steps shaped for that. The case this file exists for is

    [500, 1024, 512]     no cut balances it: 500|1536, 1524|512
    [1024, 500, 512]     cut after the first: 1024 | 1012, two 1024 slots

-- the same three requests, admissible only in the second order. Everything
here is about finding that order and applying it safely.

Two invariants are load-bearing and are asserted separately from the yield:

* the decode prefix ``_reorder_batch_for_rpa`` established never moves, because
  the ragged-paged-attention kernel is told where that segment ends;
* a partition is applied only when it is strictly better than what the current
  order already gives, so steps that already stage pay nothing.
"""
import random
from itertools import combinations
from types import SimpleNamespace

import pytest

from vllm_torchtpu.runner.naive_prefill_microbatching import (
    DEFAULT_STAGED_MICROBATCH_BUCKETS, NUM_PREFILL_MICROBATCHES,
    plan_reordered_two_microbatch_step, plan_two_microbatch_step)

BUCKETS = [16, 32, 64, 128, 256, 512, 1024, 2048]
STAGED = DEFAULT_STAGED_MICROBATCH_BUCKETS


def _ids(count):
    return [f"r{index}" for index in range(count)]


def partition(lengths, *, frozen=0, max_reqs=None, buckets=BUCKETS,
              microbatch_buckets=STAGED):
    """The best partition of a step, ignoring the order it arrived in."""
    request_ids = _ids(len(lengths))
    return plan_reordered_two_microbatch_step(
        request_ids, dict(zip(request_ids, lengths)), buckets,
        microbatch_buckets=microbatch_buckets,
        max_reqs_per_microbatch=max_reqs, num_frozen_prefix=frozen)


def contiguous(lengths, *, max_reqs=None, buckets=BUCKETS,
               microbatch_buckets=STAGED):
    """What the ordinary planner makes of the step as it stands."""
    request_ids = _ids(len(lengths))
    return plan_two_microbatch_step(
        request_ids, dict(zip(request_ids, lengths)), buckets,
        microbatch_buckets=microbatch_buckets,
        max_reqs_per_microbatch=max_reqs)


def reordered(lengths, order):
    """The step's lengths after applying a partition's permutation."""
    return [lengths[index] for index in order]


# ----- The ragged case: an admissible partition no single cut can express -----


def test_the_motivating_ragged_step_stages_only_after_reordering():
    """[500, 1024, 512]: 2036 tokens, so serial pads to one 2048 forward.

    Every contiguous cut fails. 500 | 1536 needs a 2048-token slot, and two of
    those cost 4096. 1524 | 512 needs the same. Regroup as {1024} | {500, 512}
    and both slots fit 1024, for a staged cost of 2048 -- exactly the serial
    cost, which is what the cost rule requires.
    """
    lengths = [500, 1024, 512]
    assert contiguous(lengths) is None

    found = partition(lengths)
    assert found is not None
    assert found.microbatch_tokens == 1024
    assert found.staged_cost_tokens == 2048 == found.serial_tokens
    assert sorted(found.left_indices + found.right_indices) == [0, 1, 2]
    assert {found.left_tokens, found.right_tokens} == {1024, 1012}
    # And the reordered step is one the ordinary planner admits, which is the
    # only thing that matters: the runner re-plans after permuting.
    assert contiguous(reordered(lengths, found.new_order)) is not None


def test_the_reordered_step_is_what_the_ordinary_planner_then_sees():
    """The permutation is the whole interface. Nothing else is handed on.

    The runner does not consume the partition directly -- it permutes the
    input batch and lets ``plan_two_microbatch_step`` decide again. So the
    contract is exactly: after applying ``new_order``, the ordinary planner
    admits a plan no more expensive than the partition promised.
    """
    lengths = [500, 1000, 512]
    found = partition(lengths)
    assert found is not None
    plan = contiguous(reordered(lengths, found.new_order))
    assert plan is not None
    assert plan.cost_key <= found.cost_key


@pytest.mark.parametrize("lengths", [
    [500, 1024, 512],
    [700, 400, 300, 600],
    [768, 768, 256, 256],
    [565, 339, 452, 339, 339],
])
def test_ragged_steps_with_no_contiguous_split_become_stageable(lengths):
    assert contiguous(lengths) is None
    found = partition(lengths)
    assert found is not None
    assert contiguous(reordered(lengths, found.new_order)) is not None


def test_some_ragged_steps_have_no_partition_in_any_order():
    """Reordering is not a way around the cost rule, only around the layout.

    [900, 128, 900, 100] must put one 900 in each slot -- both together are
    1800 -- and whichever slot takes the 128 lands on 1028, over the 1024
    bucket. No permutation changes that, so the step stays serial.
    """
    lengths = [900, 128, 900, 100]
    assert contiguous(lengths) is None
    assert partition(lengths) is None


# ----- Exactness: the partition search is not a greedy packer -----


def test_the_search_solves_a_case_greedy_longest_first_would_miss():
    """Longest-first bin packing is the obvious cheap answer, and it is wrong.

    Greedy on [565, 452, 339, 339, 339], always filling the lighter slot:
    565 | 452, then 339 joins the 452 (791), then 339 joins the 565 (904),
    then the last 339 joins the 791 (1130). That last slot is over the 1024
    bucket, so greedy would refuse the step. The exact partition is
    {565, 452} | {339, 339, 339} -- 1017 against 1017.

    This is why the search is an exact subset-sum rather than a heuristic:
    being wrong here does not produce a slower step, it produces a refused
    one, and refusals are invisible except as yield that never arrived.
    """
    lengths = [565, 339, 452, 339, 339]
    found = partition(lengths)
    assert found is not None
    assert found.microbatch_tokens == 1024
    assert found.left_tokens == found.right_tokens == 1017


def test_the_cheapest_bucket_wins_before_balance_does():
    """Two 512-slots beat two 1024-slots even when the 1024s balance better.

    Cost is what the cost rule compares, so a partition is chosen by the slot
    length it needs first, and only then by how evenly the real work is spread
    inside those slots.
    """
    found = partition([512, 500, 12])
    assert found is not None
    assert found.microbatch_tokens == 512
    assert found.staged_cost_tokens == 1024


def test_balance_breaks_ties_within_one_bucket():
    """Among partitions needing the same slot length, take the even one.

    Both slots run for the same padded length, so the lighter slot spends the
    difference computing padding -- and padding has no expert routing behind
    which to hide the other slot's collectives.
    """
    found = partition([600, 500, 400, 300])
    assert found is not None
    assert found.imbalance == 0
    assert found.left_tokens == found.right_tokens == 900


def test_a_partition_is_never_admitted_against_the_cost_rule():
    """The rule the whole feature rests on, restated for the new search."""
    for lengths in ([1024], [1024, 1024, 1024], [2048], [1500, 600]):
        found = partition(lengths)
        if found is None:
            continue
        assert (NUM_PREFILL_MICROBATCHES * found.microbatch_tokens
                <= found.serial_tokens)


def test_a_slot_may_fill_its_bucket_exactly():
    """[512, 1024, 512] regroups to {1024} | {512, 512}: 1024 against 1024.

    The boundary belongs in the admitted set. Padding is what makes the cost
    rule survivable at all, so a partition that uses a bucket to the last
    token is the best case, not a marginal one.
    """
    found = partition([512, 1024, 512])
    assert found is not None
    assert found.microbatch_tokens == 1024
    assert found.left_tokens == found.right_tokens == 1024


def test_a_slot_one_token_over_its_bucket_is_refused():
    """The other side of that boundary, and the reason it is tested directly.

    [1025, 500, 500] has exactly one interesting partition, {1025} | {1000},
    and its heavy slot misses the 1024 bucket by one token. The next bucket up
    is 2048, and two of those cost 4096 against a serial 2048.

    Random cases do not find this: the search prefers balanced partitions, so
    it only lands on a bucket boundary when forced to. A budget off by one
    survived the whole fuzz suite until this case was written for it.
    """
    assert partition([1025, 500, 500]) is None
    assert partition([500, 1025, 500]) is None


def test_a_lone_request_has_no_partition_to_find():
    """One request cannot fill two slots, and a dummy partner doubles cost."""
    assert partition([1024]) is None
    assert partition([512]) is None


def test_a_step_too_big_for_any_bucket_is_refused():
    """2049 tokens exceeds every padding, so there is no serial cost to beat
    and nothing to compare a partition against."""
    assert partition([2048, 1]) is None


# ----- The decode prefix is frozen -----


def test_the_decode_prefix_never_moves():
    """_reorder_batch_for_rpa puts 1-token decodes at the front and tells the
    ragged-paged-attention kernel where that segment ends. A permutation that
    crossed the boundary would describe a decode to the kernel as prefill.
    """
    lengths = [1, 1, 500, 1000, 512]
    found = partition(lengths, frozen=2)
    assert found is not None
    # Frozen rows keep their positions, which is stronger than "stay left":
    # they are the first entries of the new order, in their original sequence.
    assert found.new_order[:2] == (0, 1)
    assert set(found.left_indices) >= {0, 1}
    assert 0 not in found.right_indices and 1 not in found.right_indices


def test_frozen_decode_tokens_count_against_the_left_slot_budget():
    """The decodes are real work in slot 0, not free.

    Eight decodes plus a 1020-token prefill would overflow a 1024 slot at
    1028, so the partition has to put the prefill on the right instead.
    """
    lengths = [1] * 8 + [1020, 1000]
    found = partition(lengths, frozen=8)
    assert found is not None
    assert found.left_tokens <= found.microbatch_tokens
    assert found.right_tokens <= found.microbatch_tokens


def test_a_step_that_is_all_frozen_decodes_has_nothing_to_permute():
    assert partition([1, 1, 1, 1], frozen=4) is None


# ----- Slot capacity -----


def test_a_slot_never_takes_more_requests_than_one_prepare_inputs_can():
    """A slot is one _prepare_inputs call, so its request count is capped.

    With the cap at 1 the {500, 512} grouping the ragged case needs is
    unavailable, and the step stays serial however cheap the token maths is.
    """
    assert partition([500, 1024, 512], max_reqs=1) is None
    found = partition([500, 1024, 512], max_reqs=2)
    assert found is not None
    assert len(found.left_indices) <= 2
    assert len(found.right_indices) <= 2


def test_the_cap_counts_frozen_decodes_in_the_left_slot():
    """Frozen decodes occupy rows in slot 0, so they spend its request
    budget -- three decodes plus a prefill is four requests, not one."""
    lengths = [1, 1, 1, 900, 900]
    assert partition(lengths, frozen=3, max_reqs=3) is None
    assert partition(lengths, frozen=3, max_reqs=4) is not None


# ----- No churn: an order that already works is left alone -----


@pytest.mark.parametrize("lengths", [
    [1024, 1024],
    [1000, 24, 1000],
    [300, 300, 300, 300, 300, 300],
    [1, 1, 1000, 500, 512],
])
def test_a_partition_the_current_order_already_expresses_is_contiguous(
        lengths):
    """Many steps have several equally cheap partitions. The search has to
    return the one already in front of it, or the runner would permute the
    batch every step for nothing -- and every swap copies a request's whole
    token row.
    """
    found = partition(lengths)
    assert found is not None
    assert found.is_already_contiguous()
    assert found.new_order == tuple(range(len(lengths)))


def test_is_already_contiguous_is_false_when_a_move_is_required():
    found = partition([500, 1024, 512])
    assert found is not None
    assert not found.is_already_contiguous()


# ----- Against brute force -----


def _random_step(rng):
    """A step the scheduler could actually emit: decodes first, then prefills.

    Lengths are drawn inside the 2048-token budget rather than drawn freely
    and discarded, so every generated case exercises the search instead of
    being skipped for being a step no scheduler would hand out.
    """
    num_decodes = rng.choice([0, 0, 0, 1, 2, 3])
    num_prefills = rng.randint(2, 6)
    budget = (2048 - num_decodes) // num_prefills
    prefills = [rng.randint(1, budget) for _ in range(num_prefills)]
    return [1] * num_decodes + prefills, num_decodes


def _brute_force_best(lengths, frozen, max_reqs, buckets, microbatch_buckets):
    """Every partition, scored the way the planner scores them.

    Exponential and unashamed: it exists to prove the bitset search returns an
    optimal partition, so it may not share any of the search's reasoning.
    """
    total = sum(lengths)
    serial = next((b for b in sorted(buckets) if b >= total), None)
    if serial is None:
        return None
    movable = list(range(frozen, len(lengths)))
    frozen_tokens = sum(lengths[:frozen])
    best = None
    for size in range(len(movable) + 1):
        for chosen in combinations(movable, size):
            left = frozen + len(chosen)
            right = len(movable) - len(chosen)
            if left == 0 or right == 0:
                continue
            if max_reqs is not None and (left > max_reqs or right > max_reqs):
                continue
            left_tokens = frozen_tokens + sum(lengths[i] for i in chosen)
            right_tokens = total - left_tokens
            bucket = next((b for b in sorted(microbatch_buckets)
                           if b >= max(left_tokens, right_tokens)), None)
            if bucket is None:
                continue
            if NUM_PREFILL_MICROBATCHES * bucket > serial:
                continue
            key = (NUM_PREFILL_MICROBATCHES * bucket,
                   abs(left_tokens - right_tokens))
            if best is None or key < best:
                best = key
    return best


@pytest.mark.parametrize("seed", range(40))
def test_the_search_matches_brute_force_on_random_steps(seed):
    """Fuzz. The interesting failure is not a crash, it is a partition that
    exists and is not found: that shows up only as yield the feature quietly
    never collected.
    """
    rng = random.Random(seed)
    lengths, frozen = _random_step(rng)
    max_reqs = rng.choice([None, 2, 4, 8])

    expected = _brute_force_best(lengths, frozen, max_reqs, BUCKETS, STAGED)
    found = partition(lengths, frozen=frozen, max_reqs=max_reqs)
    if expected is None:
        assert found is None
        return
    assert found is not None
    assert found.cost_key == expected


@pytest.mark.parametrize("seed", range(40))
def test_every_partition_found_is_a_valid_partition(seed):
    """Structural invariants, checked on whatever the fuzz produces.

    Independence is the one that matters for correctness: the staged graph
    interleaves the slots, so a request appearing in both would have slot 1
    reading KV that slot 0 is still writing.
    """
    rng = random.Random(1000 + seed)
    lengths, _ = _random_step(rng)
    found = partition(lengths)
    if found is None:
        return
    left, right = set(found.left_indices), set(found.right_indices)
    assert not left & right
    assert left | right == set(range(len(lengths)))
    assert len(found.new_order) == len(lengths)
    assert found.left_tokens == sum(lengths[i] for i in left)
    assert found.right_tokens == sum(lengths[i] for i in right)
    assert max(found.left_tokens,
               found.right_tokens) <= found.microbatch_tokens


# ----- Applying the permutation to the input batch -----


class FakeInputBatch:
    """Just enough of vLLM's InputBatch to observe the permutation.

    ``swap_states`` on the real one moves a dozen per-request arrays and the
    block table; what the runner depends on is that it keeps
    ``req_id_to_index`` consistent, which is what this models.
    """

    def __init__(self, req_ids):
        self._req_ids = list(req_ids)
        self.swaps = []

    @property
    def req_ids(self):
        return self._req_ids

    @property
    def num_reqs(self):
        return len(self._req_ids)

    @property
    def req_id_to_index(self):
        return {req_id: index for index, req_id in enumerate(self._req_ids)}

    def swap_states(self, i1, i2):
        self.swaps.append((i1, i2))
        self._req_ids[i1], self._req_ids[i2] = (self._req_ids[i2],
                                                self._req_ids[i1])


def make_runner(lengths, *, monkeypatch, reorder=True, legacy=False,
                max_reqs=8):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    for name, value in (("VLLM_TPU_NAIVE_PREFILL_MICROBATCHING", True),
                        ("VLLM_TPU_NAIVE_PREFILL_REORDER", reorder),
                        ("VLLM_TPU_NAIVE_PREFILL_LEGACY_ADMISSION", legacy),
                        ("VLLM_TPU_NAIVE_PREFILL_ALLOW_DUMMY_PARTNER",
                         False)):
        monkeypatch.setattr(
            f"vllm_torchtpu.runner.tpu_runner.envs.{name}", value)

    request_ids = _ids(len(lengths))

    class FakeRunner:
        num_tokens_paddings = BUCKETS
        num_reqs_max_model_len = max_reqs
        _staged_microbatch_buckets = STAGED
        _two_microbatch_plan = None
        _naive_prefill_step_mode = None
        _naive_prefill_reorders = 0
        _naive_prefill_planner_limits = (
            TPUModelRunner._naive_prefill_planner_limits)
        _maybe_reorder_batch_for_microbatching = (
            TPUModelRunner._maybe_reorder_batch_for_microbatching)
        _build_local_naive_prefill_plan = (
            TPUModelRunner._build_local_naive_prefill_plan)

    runner = FakeRunner()
    runner.input_batch = FakeInputBatch(request_ids)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens=dict(zip(request_ids, lengths)),
        total_num_scheduled_tokens=sum(lengths))
    return runner, scheduler_output


def test_the_runner_permutes_the_batch_and_then_admits_the_step(monkeypatch):
    """End to end through the runner's own methods: a ragged step that plans
    to nothing before the permutation and to a staged step after it."""
    runner, scheduler_output = make_runner([500, 1024, 512],
                                           monkeypatch=monkeypatch)

    assert runner._build_local_naive_prefill_plan(scheduler_output).name == \
        "SERIAL"
    moved = runner._maybe_reorder_batch_for_microbatching(scheduler_output, 0)

    assert moved is True
    assert runner._naive_prefill_reorders == 1
    mode = runner._build_local_naive_prefill_plan(scheduler_output)
    assert mode.name == "TWO_MICROBATCH"
    plan = runner._two_microbatch_plan
    assert plan.num_real_microbatches == 2
    assert plan.microbatch_tokens == 1024
    # The plan's slots are the batch's new front and back, in that order.
    assert plan.request_ids == tuple(runner.input_batch.req_ids)


def test_a_step_that_already_stages_is_not_permuted(monkeypatch):
    """No swap for a step whose order already works. Every swap copies a
    request's full token row, so churn is not free."""
    runner, scheduler_output = make_runner([1024, 1024],
                                           monkeypatch=monkeypatch)
    assert runner._maybe_reorder_batch_for_microbatching(scheduler_output,
                                                         0) is False
    assert runner.input_batch.swaps == []
    assert runner._naive_prefill_reorders == 0


def test_reordering_is_skipped_when_the_flag_is_off(monkeypatch):
    """The control arm. Without it there is no way to measure what the ragged
    cases were costing, because the planner cannot be asked twice on hardware
    -- the flag is read once at startup."""
    runner, scheduler_output = make_runner([500, 1024, 512],
                                           monkeypatch=monkeypatch,
                                           reorder=False)
    assert runner._maybe_reorder_batch_for_microbatching(scheduler_output,
                                                         0) is False
    assert runner.input_batch.req_ids == ["r0", "r1", "r2"]


def test_reordering_is_skipped_under_legacy_admission(monkeypatch):
    """Legacy admission reproduces the fixed-shape experiment bit for bit.
    Permuting the batch under it would change what those records measured."""
    runner, scheduler_output = make_runner([500, 1024, 512],
                                           monkeypatch=monkeypatch,
                                           legacy=True)
    assert runner._maybe_reorder_batch_for_microbatching(scheduler_output,
                                                         0) is False
    assert runner.input_batch.swaps == []


def test_the_runner_leaves_the_decode_prefix_in_place(monkeypatch):
    """Two decodes at the front, and a ragged prefill tail behind them."""
    runner, scheduler_output = make_runner([1, 1, 500, 1000, 512],
                                           monkeypatch=monkeypatch)
    moved = runner._maybe_reorder_batch_for_microbatching(scheduler_output, 2)

    assert moved is True
    assert runner.input_batch.req_ids[:2] == ["r0", "r1"]
    assert all(i1 >= 2 and i2 >= 2 for i1, i2 in runner.input_batch.swaps)
    assert runner._build_local_naive_prefill_plan(
        scheduler_output).name == "TWO_MICROBATCH"


def test_a_step_with_one_request_is_left_alone(monkeypatch):
    runner, scheduler_output = make_runner([1024], monkeypatch=monkeypatch)
    assert runner._maybe_reorder_batch_for_microbatching(scheduler_output,
                                                         0) is False


@pytest.mark.parametrize("seed", range(25))
def test_the_permutation_moves_every_request_exactly_where_planned(
        seed, monkeypatch):
    """The selection-into-place loop is easy to get subtly wrong: swapping
    changes the positions it is still reading. This asserts the batch ends in
    the planned order, and that no request is lost or duplicated on the way.
    """
    rng = random.Random(2000 + seed)
    lengths, _ = _random_step(rng)
    runner, scheduler_output = make_runner(lengths, monkeypatch=monkeypatch)
    before = list(runner.input_batch.req_ids)

    moved = runner._maybe_reorder_batch_for_microbatching(scheduler_output, 0)

    assert sorted(runner.input_batch.req_ids) == sorted(before)
    if not moved:
        assert runner.input_batch.req_ids == before
        return
    found = partition(lengths)
    assert runner.input_batch.req_ids == [before[i] for i in found.new_order]
    # A permutation of n items needs at most n-1 swaps; more means the loop is
    # undoing its own work.
    assert len(runner.input_batch.swaps) <= len(lengths) - 1
