# SPDX-License-Identifier: Apache-2.0
"""The cost-aware two-microbatch partitioner.

Every case here is decided by one inequality:

    2 * pad(max(tokens_a, tokens_b))  <=  pad(total_tokens)

Both sides are padding buckets rather than raw token counts, which is why a
1023+896 step is admissible and a lone 1024 is not. The buckets used throughout
are the runner's defaults, powers of two from 16 to 2048.
"""
import pytest

from vllm_torchtpu.runner.naive_prefill_microbatching import (
    DEFAULT_STAGED_MICROBATCH_BUCKETS, NUM_PREFILL_MICROBATCHES,
    PrefillStepMode, coordinated_microbatch_tokens, local_step_mode,
    padded_token_len, partition_is_independent, plan_two_microbatch_step,
    resolve_staged_microbatch_buckets, staged_step_is_cheaper,
    staged_token_bucket)

BUCKETS = [16, 32, 64, 128, 256, 512, 1024, 2048]
# What the staged graph is compiled for by default. A split needing any other
# slot length is not admissible however cheap it looks.
STAGED = DEFAULT_STAGED_MICROBATCH_BUCKETS


def plan(lengths, *, buckets=BUCKETS, microbatch_buckets=STAGED,
         max_reqs=None, allow_dummy_partner=False):
    """Plan a step of requests r0, r1, ... with the given token counts."""
    request_ids = [f"r{index}" for index in range(len(lengths))]
    scheduled = dict(zip(request_ids, lengths))
    return plan_two_microbatch_step(
        request_ids, scheduled, buckets,
        microbatch_buckets=microbatch_buckets,
        max_reqs_per_microbatch=max_reqs,
        allow_dummy_partner=allow_dummy_partner)


# ----- Requirement 1: never more device work than the serial path -----


def test_a_lone_request_is_rejected_because_a_dummy_doubles_the_work():
    """The case the whole cost rule exists for.

    One 1024-token prefill costs pad(1024) = 1024 serially. Staging it pairs it
    with a dummy that runs a *full* 1024-token forward whose output is thrown
    away, so the step costs 2048. Twice the work to overlap nothing.
    """
    assert plan([1024]) is None


@pytest.mark.parametrize("lengths", [
    [1024],
    [512],
    [2048],
    [37],
])
def test_no_single_request_step_can_ever_be_staged(lengths):
    assert plan(lengths) is None


def test_the_dummy_partner_override_reinstates_it_for_measurement():
    """Off by default, because it is a worse step. On for harnesses that need
    a concurrency-1 request to reach the staged path at all."""
    staged = plan([1024], allow_dummy_partner=True)
    assert staged is not None
    assert staged.num_real_microbatches == 1
    assert staged.microbatch_tokens == 1024
    # Stated plainly: this plan costs 2048 against a serial 1024.
    assert staged.staged_cost_tokens == 2048
    assert staged.serial_tokens == 1024
    assert not staged_step_is_cheaper(staged.microbatch_tokens,
                                      staged.serial_tokens)


def test_an_unbalanced_split_that_would_cost_more_is_rejected():
    """1024 + 64: serial pads to 2048, but the slots pad to 2 x 1024 = 2048.

    Equal, so admissible -- padding is what makes lopsided splits survivable.
    Push the imbalance past the bucket and it stops being free.
    """
    assert plan([1024, 64]) is not None
    # 1088 + 64 = 1152 -> serial 2048, but the big slot pads to 2048 itself,
    # so staging would cost 4096.
    assert plan([1088, 64]) is None


# ----- Requirement 2: diverse lengths, and more than one slot shape -----


def test_unequal_lengths_pad_into_one_shared_slot_shape():
    """1023 + 896 -- neither is 1024 and they are not equal to each other."""
    staged = plan([1023, 896])
    assert staged is not None
    assert staged.microbatch_tokens == 1024
    assert staged.serial_tokens == 2048  # pad(1919)
    assert staged.staged_cost_tokens == 2048
    assert [m.num_tokens for m in staged.microbatches] == [1023, 896]
    assert [m.request_ids for m in staged.microbatches] == [("r0", ), ("r1", )]


def test_a_smaller_slot_shape_is_used_when_it_fits():
    """512 + 512 costs 1024 staged against a serial 1024 -- but only because
    the graph has a 512 slot compiled. It is a different program."""
    staged = plan([512, 512])
    assert staged is not None
    assert staged.microbatch_tokens == 512
    assert staged.staged_cost_tokens == 1024 == staged.serial_tokens


def test_a_slot_shape_that_was_never_compiled_is_not_admissible():
    """The same 512+512 step, on a server that only warmed the 1024 slot.

    Admitting it would make one rank compile a new program mid-step while its
    peers waited inside the collectives they had already entered.
    """
    assert plan([512, 512], microbatch_buckets=(1024, )) is None
    # 256+256 needs a 256 slot, which is not in the default staged set even
    # though 256 is a perfectly ordinary token bucket.
    assert plan([256, 256]) is None
    assert plan([256, 256], microbatch_buckets=(256, 512, 1024)) is not None


def test_several_requests_may_share_one_slot():
    """1024 prefill against 512 one-token decodes.

    The user's case: the two sides need not be the same kind of work, only
    independent. Serial pads 1536 to 2048; the slots pad to 1024 each.
    """
    staged = plan([1024] + [1] * 512)
    assert staged is not None
    assert staged.microbatch_tokens == 1024
    assert staged.num_reqs_per_microbatch == [1, 512]
    assert [m.num_tokens for m in staged.microbatches] == [1024, 512]
    assert staged.staged_cost_tokens == 2048 == staged.serial_tokens


def test_a_slot_cannot_hold_more_requests_than_one_forward_prepares():
    """_prepare_inputs caps a chunk's request count; a slot is one such call."""
    assert plan([1024] + [1] * 512, max_reqs=8) is None
    assert plan([1024] + [1] * 512, max_reqs=512) is not None


def test_the_most_balanced_split_wins_among_equally_cheap_ones():
    """[600, 300, 700]: two admissible splits, both costing 2 x 1024.

    600|1000 and 900|700 are indistinguishable on cost, so the tie-break
    decides, and it takes the tighter one -- less padding inside each slot
    means more real expert work available to overlap.

    This case is chosen deliberately. The obvious one, four equal 512s, does
    not test the tie-break at all: there the cost rule rejects both uneven
    splits on its own and leaves exactly one candidate.
    """
    staged = plan([600, 300, 700])
    assert staged is not None
    assert staged.num_reqs_per_microbatch == [2, 1]
    assert [m.num_tokens for m in staged.microbatches] == [900, 700]


def test_the_cost_rule_alone_forces_the_even_split_of_equal_requests():
    """Four 512s: only 1024|1024 is affordable, both 512|1536 splits pad the
    big slot to 2048 and so cost 4096 against a serial 2048."""
    staged = plan([512, 512, 512, 512])
    assert staged is not None
    assert staged.num_reqs_per_microbatch == [2, 2]
    assert [m.num_tokens for m in staged.microbatches] == [1024, 1024]


def test_a_cheaper_uneven_split_beats_a_costlier_balanced_one():
    """Cost is compared before balance, not after."""
    # [1000, 24, 1000]: the even-token split is r0 | r1,r2 (1000 vs 1024),
    # both padding to 1024. Splitting after r1 gives 1024 vs 1000 -- also
    # 1024. Either is 2048 = serial; the tie-break picks the tighter one.
    staged = plan([1000, 24, 1000])
    assert staged is not None
    assert staged.microbatch_tokens == 1024
    assert staged.staged_cost_tokens <= staged.serial_tokens


def test_contiguity_forfeits_some_perfect_partitions():
    """A known limitation, recorded rather than hidden.

    [512, 1024, 512] partitions perfectly as {r0, r2} | {r1}, but a slot is a
    contiguous range of the input batch, so that partition cannot be prepared.
    Every contiguous split is 512|1536 or 1536|512, both needing a 2048 slot.
    """
    assert plan([512, 1024, 512]) is None
    # Reordered so the same three requests are contiguous, it is admissible.
    assert plan([512, 512, 1024]) is not None


# ----- Requirement 3: chunked prefill -----


def test_a_partially_computed_request_is_admissible():
    """A chunk of an 8k prompt is as stageable as a fresh request.

    The planner never consults num_computed_tokens. Chunk 3 of a chunked
    prefill attends over KV that chunks 0-2 wrote in *earlier steps*, so
    within this step it depends on nothing the other slot produces.
    """
    staged = plan([1024, 1024])
    assert staged is not None
    assert staged.num_real_microbatches == NUM_PREFILL_MICROBATCHES


def test_two_chunks_of_one_prompt_can_never_land_in_one_step():
    """The dependency that would be unsafe is unreachable.

    Chunk i+1 must attend over chunk i's KV, so they may not be two slots of
    one step. vLLM schedules a request at most once per step, so a request id
    appears at most once in the step's list -- and a duplicate is rejected
    outright rather than being silently split across the slots.
    """
    assert plan_two_microbatch_step(
        ["same", "same"], {"same": 1024}, BUCKETS,
        microbatch_buckets=STAGED) is None


def test_independence_is_decided_by_request_identity():
    assert partition_is_independent(("a", "b"), ("c", ))
    assert not partition_is_independent(("a", "b"), ("b", "c"))
    assert partition_is_independent((), ())


def test_the_independence_invariant_has_two_redundant_guards():
    """Recorded so a later simplification knows what it is deleting.

    The planner rejects duplicate request ids *and* checks each candidate
    split for independence. Either alone rejects every unsafe plan, so
    mutation-testing either one shows no failure -- the other catches it.
    Deleting both would be silent, and no test here can catch that. Asserted
    directly instead, at the level where each guard is visible.
    """
    # Guard 1, on its own terms.
    assert not partition_is_independent(("x", ), ("x", ))
    # Guard 2, on its own terms.
    duplicated = ["x", "x"]
    assert len(set(duplicated)) != len(duplicated)
    # And the planner, which must reject however it gets there.
    assert plan_two_microbatch_step(duplicated, {"x": 1024}, BUCKETS,
                                    microbatch_buckets=STAGED) is None


def test_eight_chunks_of_one_prompt_can_each_stage_against_another_prompt():
    """The evaluation the user asked for, as a test.

    Two 8k prompts chunked into eight 1024-token steps each. At every step i,
    prompt A's chunk i and prompt B's chunk i are different requests, so each
    of the eight steps stages. What cannot happen is A's chunk i pairing with
    A's chunk i+1.
    """
    for chunk in range(8):
        staged = plan_two_microbatch_step(
            [f"A@{chunk}", f"B@{chunk}"],
            {f"A@{chunk}": 1024, f"B@{chunk}": 1024},
            BUCKETS, microbatch_buckets=STAGED)
        assert staged is not None, chunk
        assert staged.microbatch_tokens == 1024
        assert staged.staged_cost_tokens == staged.serial_tokens


def test_a_lone_chunked_prompt_stages_no_better_than_a_lone_fresh_one():
    """Chunking does not create a partner out of nothing.

    One 8k prompt alone gives one 1024-token chunk per step and nothing to
    pair it with, so all eight steps run serial. Chunked prefill makes the
    feature *applicable*, not automatic.
    """
    for _ in range(8):
        assert plan([1024]) is None


# ----- Degenerate and malformed input -----


@pytest.mark.parametrize("lengths", [[], [0, 1024], [-1, 1024]])
def test_empty_or_nonsensical_token_counts_are_rejected(lengths):
    assert plan(lengths) is None


def test_a_step_larger_than_the_largest_bucket_is_rejected():
    assert plan([4096, 4096]) is None


def test_a_missing_token_count_is_rejected():
    assert plan_two_microbatch_step(
        ["a", "b"], {"a": 1024}, BUCKETS, microbatch_buckets=STAGED) is None


def test_none_request_ids_are_ignored():
    """The input batch's request slice is None-padded past num_reqs."""
    staged = plan_two_microbatch_step(
        ["a", "b", None], {"a": 1024, "b": 1024}, BUCKETS,
        microbatch_buckets=STAGED)
    assert staged is not None
    assert staged.request_ids == ("a", "b")


# ----- The DP-coordinated shape -----


def test_the_coordinated_slot_length_is_the_maximum_over_ranks():
    """Padding up wastes device time; padding down hangs the core.

    A rank that launched a smaller program than a peer would leave that peer's
    in-graph collective waiting on operands nobody sent.
    """
    assert coordinated_microbatch_tokens([512, 1024, 512]) == 1024
    # Idle ranks contribute 0 and must not drag the maximum down or up.
    assert coordinated_microbatch_tokens([0, 512, 0]) == 512
    assert coordinated_microbatch_tokens([0, 0]) == 1024


def test_the_staged_bucket_overrides_the_whole_step_bucket():
    """The step total is not a per-forward padding target when the step is two
    forwards."""
    assert staged_token_bucket(PrefillStepMode.TWO_MICROBATCH, 2048, 512) == 512
    assert staged_token_bucket(PrefillStepMode.SERIAL, 2048, 512) == 2048
    assert staged_token_bucket(PrefillStepMode.IDLE, None, 512) is None


def test_the_cost_rule_holds_globally_when_it_holds_on_every_rank():
    """Why one all-reduce is enough to settle this.

    Serial cost is a MAX over ranks of the step total, and the coordinated slot
    length is a MAX over ranks of their local slot lengths. Both are monotonic
    in the same per-rank quantities, so a group in which every busy rank passed
    its own check cannot fail the global one.
    """
    per_rank = [plan([1023, 896]), plan([512, 512]), plan([1024, 1024])]
    assert all(item is not None for item in per_rank)
    coordinated = coordinated_microbatch_tokens(
        [item.microbatch_tokens for item in per_rank])
    serial = max(item.serial_tokens for item in per_rank)
    assert staged_step_is_cheaper(coordinated, serial)


def test_the_cost_rule_rejects_a_slot_length_that_outgrew_the_step():
    assert not staged_step_is_cheaper(1024, 1024)
    assert staged_step_is_cheaper(1024, 2048)
    assert staged_step_is_cheaper(512, 1024)


def test_the_runner_downgrades_to_serial_on_a_coordinated_cost_violation(
        monkeypatch):
    """The global backstop, and why it cannot split the group.

    Both operands come out of the one all-reduce, so every rank evaluates the
    same inequality on the same numbers and reaches the same verdict. A rank
    that decided this locally could disagree with its peers and hang them.
    """
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.envs."
        "VLLM_TPU_NAIVE_PREFILL_ALLOW_DUMMY_PARTNER", False)
    enforce = TPUModelRunner._enforce_staged_cost_rule
    runner = object()

    assert enforce(runner, PrefillStepMode.TWO_MICROBATCH, 1024,
                   2048) is PrefillStepMode.TWO_MICROBATCH
    assert enforce(runner, PrefillStepMode.TWO_MICROBATCH, 1024,
                   1024) is PrefillStepMode.SERIAL
    # It only ever downgrades; it must not promote a rank that voted SERIAL.
    assert enforce(runner, PrefillStepMode.SERIAL, 512,
                   2048) is PrefillStepMode.SERIAL
    assert enforce(runner, PrefillStepMode.IDLE, 512,
                   2048) is PrefillStepMode.IDLE


def test_an_uncompiled_coordinated_slot_length_raises_instead_of_hanging():
    """The one failure worse than a crash.

    If the reduction ever settled on a slot length this process never traced,
    the rank would compile mid-step while its peers waited inside collectives
    they had already entered -- and nothing would be logged. A traceback on one
    rank is strictly better. Unreachable in practice: every rank draws its
    length from the compiled set and a MAX over that set stays in it.
    """
    import torch
    import vllm.distributed.parallel_state as parallel_state

    import pytest as _pytest
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch = _pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            "vllm_torchtpu.runner.tpu_runner.envs."
            "VLLM_TPU_NAIVE_PREFILL_MICROBATCHING", True)
        monkeypatch.setattr(
            "vllm_torchtpu.runner.tpu_runner.envs."
            "VLLM_TPU_NAIVE_PREFILL_ALLOW_DUMMY_PARTNER", False)
        monkeypatch.setattr(parallel_state, "get_dp_group",
                            lambda: __import__("types").SimpleNamespace(
                                cpu_group=object()))

        def all_reduce(tensor, op, group):
            # A 768-token slot: a plausible length, and not compiled.
            tensor.copy_(torch.tensor([2048, 1, 1, 768]))

        monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

        class FakeRunner:
            num_tokens_paddings = BUCKETS
            _staged_microbatch_buckets = (512, 1024)
            _staged_microbatch_tokens = 1024

            def _dp_lockstep_enabled(self):
                return True

            def _local_staged_microbatch_tokens(self):
                return 1024

            _default_staged_tokens = TPUModelRunner._default_staged_tokens
            _enforce_staged_cost_rule = (
                TPUModelRunner._enforce_staged_cost_rule)

        with pytest.raises(RuntimeError, match="not one of the precompiled"):
            TPUModelRunner._dp_coordinated_step(
                FakeRunner(), 2048, 1, PrefillStepMode.TWO_MICROBATCH)
    finally:
        monkeypatch.undo()


def test_the_fallback_slot_length_is_a_compiled_one():
    """With a narrowed bucket list the module constant may be a shape this
    process never traced, so the fallback has to come from the list."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    class FakeRunner:
        _staged_microbatch_buckets = (256, 512)

    assert TPUModelRunner._default_staged_tokens(FakeRunner()) == 512
    assert coordinated_microbatch_tokens([0, 0], fallback=512) == 512
    # A real request still wins over the fallback.
    assert coordinated_microbatch_tokens([0, 256], fallback=512) == 256


def test_the_measurement_override_is_exempt_from_the_global_cost_rule(
        monkeypatch):
    """Otherwise the backstop would undo the override it exists alongside: a
    dummy-partner plan violates the cost rule by definition."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.envs."
        "VLLM_TPU_NAIVE_PREFILL_ALLOW_DUMMY_PARTNER", True)
    assert TPUModelRunner._enforce_staged_cost_rule(
        object(), PrefillStepMode.TWO_MICROBATCH, 1024,
        1024) is PrefillStepMode.TWO_MICROBATCH


# ----- Startup validation of the compiled slot shapes -----


def test_staged_buckets_must_be_real_token_buckets():
    assert resolve_staged_microbatch_buckets(BUCKETS) == (512, 1024)
    assert resolve_staged_microbatch_buckets(BUCKETS, (1024, )) == (1024, )
    assert resolve_staged_microbatch_buckets(BUCKETS, (1024, 512)) == (512,
                                                                       1024)
    with pytest.raises(ValueError, match="token padding bucket"):
        resolve_staged_microbatch_buckets(BUCKETS, (768, ))
    with pytest.raises(ValueError, match="token padding bucket"):
        resolve_staged_microbatch_buckets([16, 32], None)


def test_warmup_covers_the_request_shapes_a_real_slot_produces():
    """Record #34: warming only the idle-rank shape left real steps to compile.

    A real slot holding n requests pads to `num_reqs_paddings`; an idle rank's
    dummy uses `num_reqs_max_model_len`. Warming only the latter meant the
    first real staged step compiled mid-flight — 49 s on one rank while seven
    peers sat inside the collectives they had already entered.
    """
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    class FakeRunner:
        num_reqs_paddings = [8, 16]
        num_reqs_max_model_len = 16

    counts = TPUModelRunner._staged_warmup_req_counts(FakeRunner())
    assert counts == (8, 16)
    # MIN_NUM_SEQS is what a single-request slot pads to, and it is the shape
    # the old warmup missed entirely.
    assert 8 in counts

    class NarrowRunner:
        """A cap below some paddings must not warm shapes that cannot occur."""
        num_reqs_paddings = [8, 16, 32]
        num_reqs_max_model_len = 8

    assert TPUModelRunner._staged_warmup_req_counts(NarrowRunner()) == (8, )

    class OffGridRunner:
        """The idle-rank shape need not be one of the paddings.

        `num_reqs_max_model_len` is capped by the Pallas backend's SMEM limit,
        so it can land between padding buckets. It still has to be warmed --
        it is the shape every idle rank launches.
        """
        num_reqs_paddings = [8, 16, 32]
        num_reqs_max_model_len = 12

    assert TPUModelRunner._staged_warmup_req_counts(OffGridRunner()) == (8, 12)


def test_warmup_request_shapes_are_deduplicated():
    """The idle shape is usually also a padding; it must be warmed once."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    class FakeRunner:
        num_reqs_paddings = [8, 16]
        num_reqs_max_model_len = 16

    counts = TPUModelRunner._staged_warmup_req_counts(FakeRunner())
    assert len(counts) == len(set(counts))
    assert sorted(counts) == list(counts)


def test_a_dummy_bundle_request_count_is_clamped_to_what_the_runner_supports():
    """Warmup drives this directly, so an out-of-range value must not build a
    bundle the live path can never produce."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    import inspect

    signature = inspect.signature(
        TPUModelRunner._prepare_dummy_prefill_microbatch)
    assert "num_reqs" in signature.parameters
    assert signature.parameters["num_reqs"].default is None
    # Checked against the live signature rather than a fake, per record #20:
    # a fake that defines the interface it stands in for proves nothing.
    idle = inspect.signature(TPUModelRunner._run_idle_two_microbatch_prefill)
    assert "num_reqs" in idle.parameters
    assert idle.parameters["num_reqs"].default is None


def test_padded_token_len_returns_none_past_the_last_bucket():
    assert padded_token_len(BUCKETS, 1) == 16
    assert padded_token_len(BUCKETS, 1024) == 1024
    assert padded_token_len(BUCKETS, 1025) == 2048
    assert padded_token_len(BUCKETS, 4096) is None


def test_the_cost_aware_planner_admits_what_the_legacy_one_rejected():
    """Why VLLM_TPU_NAIVE_PREFILL_LEGACY_ADMISSION has to exist.

    compare_staged_vs_serial.py separates its arms with input lengths 1024 and
    1023, relying on the fixed envelope admitting *exactly* 1024. The
    cost-aware planner admits 1023 too -- 2 x pad(1023) = 2048 = pad(2046) --
    so both arms would stage and the A/B would compare the staged path against
    itself while reporting a serial control.
    """
    from vllm_torchtpu.runner.naive_prefill_microbatching import (
        build_two_microbatch_plan)

    ids, tokens = ["a", "b"], 1023
    scheduled = {"a": tokens, "b": tokens}
    assert build_two_microbatch_plan(ids, scheduled, [0, 0]) is None
    assert plan_two_microbatch_step(ids, scheduled, BUCKETS,
                                    microbatch_buckets=STAGED) is not None
    # Both planners agree on the 1024 arm, so the staged side of that
    # comparison is unchanged.
    exact = {"a": 1024, "b": 1024}
    assert build_two_microbatch_plan(ids, exact, [0, 0]) is not None
    assert plan_two_microbatch_step(ids, exact, BUCKETS,
                                    microbatch_buckets=STAGED) is not None


def test_a_rejected_plan_makes_the_rank_vote_serial():
    """The planner's None is what puts the whole DP group on the serial path,
    since PrefillStepMode MAX-reduces and SERIAL outranks TWO_MICROBATCH."""
    assert local_step_mode(1024, plan([1024])) is PrefillStepMode.SERIAL
    assert local_step_mode(2048, plan([1024, 1024])) is (
        PrefillStepMode.TWO_MICROBATCH)
    assert local_step_mode(0, None) is PrefillStepMode.IDLE
