from types import SimpleNamespace

import pytest

from vllm_torchtpu.runner.naive_prefill_microbatching import (
    PREFILL_MICROBATCH_TOKENS, PrefillStepMode, build_two_microbatch_plan,
    coordinated_step_mode, local_step_mode,
    staged_token_bucket, validate_feature_config,
    validate_staged_microbatch_tokens)

# ----- Planning: which local work is eligible for the staged path -----


def test_plan_has_two_independent_fixed_request_ranges():
    plan = build_two_microbatch_plan(["b", "a"], {"a": 1024, "b": 1024},
                                     {"a": 0, "b": 0})
    assert plan is not None
    assert plan.request_ids == ("b", "a")
    assert [(m.start_index, m.end_index) for m in plan.microbatches] == [
        (0, 1), (1, 2)]
    assert [m.num_tokens for m in plan.microbatches] == [1024, 1024]
    assert plan.restore_output_order(["b-logits", "a-logits"]) == [
        "b-logits", "a-logits"]


@pytest.mark.parametrize(("request_ids", "scheduled", "computed"), [
    (["a"], {"a": 512}, [0]),
    (["a", "b"], {"a": 4096, "b": 1024}, [0, 0]),
    (["a", "b"], {"a": 1024, "b": 1024}, [1, 0]),
    (["a", "a"], {"a": 1024}, [0, 0]),
])
def test_plan_rejects_mixed_chunked_resumed_or_duplicate_requests(
        request_ids, scheduled, computed):
    assert build_two_microbatch_plan(request_ids, scheduled, computed) is None


def test_plan_accepts_one_real_request_for_dummy_padding():
    plan = build_two_microbatch_plan(["a"], {"a": 1024}, [0])
    assert plan is not None
    assert plan.request_ids == ("a",)
    assert plan.num_real_microbatches == 1
    assert plan.restore_output_order(["a-logits"]) == ["a-logits"]


def test_local_step_modes_follow_scheduler_work():
    plan = build_two_microbatch_plan(["a", "b"],
                                     {"a": 1024, "b": 1024}, [0, 0])
    assert local_step_mode(0, None) == PrefillStepMode.IDLE
    assert local_step_mode(2048, plan) == PrefillStepMode.TWO_MICROBATCH
    assert local_step_mode(1, None) == PrefillStepMode.SERIAL


# ----- Coordination: every rank must choose the same graph -----


def test_dp_coordination_idle_rank_supplies_dummies():
    assert coordinated_step_mode([
        PrefillStepMode.TWO_MICROBATCH, PrefillStepMode.IDLE,
        PrefillStepMode.TWO_MICROBATCH
    ]) == PrefillStepMode.TWO_MICROBATCH


def test_dp_coordination_ineligible_busy_rank_forces_serial():
    assert coordinated_step_mode([
        PrefillStepMode.TWO_MICROBATCH, PrefillStepMode.SERIAL,
        PrefillStepMode.IDLE
    ]) == PrefillStepMode.SERIAL


def test_busy_and_idle_paths_use_one_unified_coordination_payload(monkeypatch):
    """Both entry points must issue the same payload at the same position.

    execute_model and execute_dummy_batch reach this collective from different
    call stacks; if their payloads differed in length or order the reduction
    would silently mean different things on different ranks.
    """
    import torch
    import vllm.distributed.parallel_state as parallel_state

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.envs."
        "VLLM_TPU_NAIVE_PREFILL_MICROBATCHING", True)

    group = object()
    monkeypatch.setattr(parallel_state, "get_dp_group",
                        lambda: SimpleNamespace(cpu_group=group))
    payloads = []

    def all_reduce(tensor, op, group):
        payloads.append(tensor.tolist())
        tensor.copy_(torch.tensor([2048, 1, 1, 1024]))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    class FakeRunner:
        num_tokens_paddings = [16, 32, 64, 128, 256, 512, 1024, 2048]
        _staged_microbatch_tokens = 1024

        def _dp_lockstep_enabled(self):
            return True

        _staged_microbatch_buckets = (512, 1024)

        def _local_staged_microbatch_tokens(self):
            return self._local_tokens

        _default_staged_tokens = TPUModelRunner._default_staged_tokens
        _enforce_staged_cost_rule = (
            TPUModelRunner._enforce_staged_cost_rule)

    busy = FakeRunner()
    busy._local_tokens = 1024
    idle = FakeRunner()
    idle._local_tokens = 0
    busy_result = TPUModelRunner._dp_coordinated_step(
        busy, 2048, 1, PrefillStepMode.TWO_MICROBATCH)
    idle_result = TPUModelRunner._dp_coordinated_step(
        idle, 0, 0, PrefillStepMode.IDLE)

    # An idle rank contributes 0 to the slot-length MAX, so it cannot drag the
    # group to a length nobody asked for -- but it still reads back the busy
    # rank's 1024 and runs its dummies at that shape.
    assert payloads == [[2048, 1, 1, 1024], [0, 0, 0, 0]]
    assert busy_result == (1024, 1, PrefillStepMode.TWO_MICROBATCH)
    assert idle_result == (1024, 1, PrefillStepMode.TWO_MICROBATCH)
    assert idle._staged_microbatch_tokens == 1024


def test_coordination_payload_omits_the_mode_when_the_feature_is_off(
        monkeypatch):
    """The reduction must keep its original shape for unrelated runs."""
    import torch
    import vllm.distributed.parallel_state as parallel_state

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.envs."
        "VLLM_TPU_NAIVE_PREFILL_MICROBATCHING", False)

    group = object()
    monkeypatch.setattr(parallel_state, "get_dp_group",
                        lambda: SimpleNamespace(cpu_group=group))
    payloads = []

    def all_reduce(tensor, op, group):
        payloads.append(tensor.tolist())
        tensor.copy_(torch.tensor([2048, 2]))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    class FakeRunner:
        num_tokens_paddings = [16, 32, 64, 128, 256, 512, 1024, 2048]

        def _dp_lockstep_enabled(self):
            return True

    result = TPUModelRunner._dp_coordinated_step(FakeRunner(), 2048, 2)

    assert payloads == [[2048, 2]]
    assert result == (2048, 2, PrefillStepMode.SERIAL)


def test_zero_token_step_issues_no_dp_coordination(monkeypatch):
    """A zero-token step must stay off the DP group, feature on or off.

    gloo pairs reductions in issue order, not by step, so one extra reduction
    here offsets this rank against its peers permanently: from then on its Nth
    reduction meets their (N+1)th, every later step resolves its token bucket
    and chunk count from mismatched operands, and the ranks eventually launch
    differently shaped programs whose in-graph EP collectives never rendezvous.

    Idle pairing belongs in execute_dummy_batch, which the engine calls after a
    step that executed nothing -- that path coordinates and runs the paired
    dummy bundles.

    Regression test for the 2026-08-07 hardware wedge: with the reduction
    hoisted above this early return, isl1024/osl1024/c16 completed 14/16 and
    stalled 300s; with it removed, 16/16 in 40s.
    """
    import numpy as np

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.envs."
        "VLLM_TPU_NAIVE_PREFILL_MICROBATCHING", True)
    monkeypatch.setattr("vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group",
                        lambda: False)

    calls = []

    class FakeRunner:
        execute_model_state = None
        input_batch = SimpleNamespace(num_reqs=0,
                                      req_ids=[],
                                      num_computed_tokens_cpu=np.zeros(0))
        scheduler_config = SimpleNamespace(async_scheduling=True)
        num_tokens_paddings = [16, 32, 64, 128, 256, 512, 1024, 2048]
        num_reqs_max_model_len = 8
        _staged_microbatch_buckets = (512, 1024)
        _staged_microbatch_tokens = 1024
        _two_microbatch_plan = None
        _naive_prefill_step_mode = PrefillStepMode.IDLE
        _last_logged_naive_prefill_mode = PrefillStepMode.IDLE

        _build_local_naive_prefill_plan = (
            TPUModelRunner._build_local_naive_prefill_plan)
        _naive_prefill_planner_limits = (
            TPUModelRunner._naive_prefill_planner_limits)

        def _update_states(self, scheduler_output):
            calls.append("update_states")

        def _flush_disjoint_async_results(self):
            calls.append("flush")

        def _dp_coordinated_step(self, *args, **kwargs):
            calls.append("coordinate")
            raise AssertionError(
                "a zero-token step must not join the DP reduction")

        def _count_input_chunks(self, scheduler_output):
            calls.append("count_chunks")
            return 0

        def _run_idle_two_microbatch_prefill(self):
            calls.append("idle_staged")

    runner = FakeRunner()
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=0,
                                       num_scheduled_tokens={})

    TPUModelRunner.execute_model(runner, scheduler_output)

    assert calls == ["update_states", "flush"]
    assert runner._naive_prefill_step_mode is PrefillStepMode.IDLE


# ----- Padding: the lengths every rank's staged forwards must use -----


def test_staged_bucket_is_per_microbatch_not_per_step():
    assert staged_token_bucket(PrefillStepMode.TWO_MICROBATCH,
                               2048) == PREFILL_MICROBATCH_TOKENS
    # One real microbatch used to make this agree by coincidence.
    assert staged_token_bucket(PrefillStepMode.TWO_MICROBATCH,
                               1024) == PREFILL_MICROBATCH_TOKENS


def test_serial_modes_keep_the_reduced_step_bucket():
    assert staged_token_bucket(PrefillStepMode.SERIAL, 2048) == 2048
    assert staged_token_bucket(PrefillStepMode.IDLE, 512) == 512
    assert staged_token_bucket(PrefillStepMode.SERIAL, None) is None


def test_staged_bundle_length_is_validated():
    validate_staged_microbatch_tokens(PREFILL_MICROBATCH_TOKENS, role="real",
                                      index=0, field="input_ids",
                                      expected=PREFILL_MICROBATCH_TOKENS)
    # 512 is a legal slot length now, but only for a step that agreed on 512.
    validate_staged_microbatch_tokens(512, role="real", index=0,
                                      field="input_ids", expected=512)
    with pytest.raises(RuntimeError, match="in-graph EP collectives desync"):
        validate_staged_microbatch_tokens(2048, role="real", index=1,
                                          field="input_ids",
                                          expected=PREFILL_MICROBATCH_TOKENS)
    with pytest.raises(RuntimeError, match="in-graph EP collectives desync"):
        validate_staged_microbatch_tokens(1024, role="real", index=1,
                                          field="input_ids", expected=512)


def test_staged_forward_rejects_equal_but_wrong_lengths():
    """Equal shapes are necessary but not sufficient.

    Peers pad to the fixed staged constant, not to whatever this rank chose,
    so two microbatches that agree with each other at the wrong length still
    disagree with every other rank.
    """
    import torch

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    def bundle(num_tokens):
        return SimpleNamespace(input_ids=torch.zeros(num_tokens))

    runner = SimpleNamespace(_staged_microbatch_tokens=1024)
    assert TPUModelRunner._staged_forward_num_tokens(
        runner, (bundle(1024), bundle(1024))) == 1024
    with pytest.raises(RuntimeError, match="in-graph EP collectives desync"):
        TPUModelRunner._staged_forward_num_tokens(
            runner, (bundle(2048), bundle(2048)))
    with pytest.raises(RuntimeError, match="equal shapes"):
        TPUModelRunner._staged_forward_num_tokens(
            runner, (bundle(1024), bundle(2048)))
    # The agreed length is per step, not a constant: 512 is right for a step
    # that settled on 512 and wrong for one that settled on 1024.
    runner_512 = SimpleNamespace(_staged_microbatch_tokens=512)
    assert TPUModelRunner._staged_forward_num_tokens(
        runner_512, (bundle(512), bundle(512))) == 512
    with pytest.raises(RuntimeError, match="in-graph EP collectives desync"):
        TPUModelRunner._staged_forward_num_tokens(
            runner, (bundle(512), bundle(512)))


# ----- Configuration envelope -----


def _config(**overrides):
    values = dict(dp=8, tp=1, ep=True, tokens=2048, seqs=2,
                  model_len=2048, prefix=False, speculative=None)
    values.update(overrides)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Qwen3_5MoeForConditionalGeneration"]),
            max_model_len=values["model_len"]),
        parallel_config=SimpleNamespace(
            data_parallel_size=values["dp"],
            tensor_parallel_size=values["tp"],
            enable_expert_parallel=values["ep"]),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=values["tokens"],
            max_num_seqs=values["seqs"]),
        cache_config=SimpleNamespace(enable_prefix_caching=values["prefix"]),
        speculative_config=values["speculative"])


def test_fixed_feature_config_is_accepted():
    validate_feature_config(_config())


@pytest.mark.parametrize("override", [
    {"dp": 4}, {"tp": 2}, {"ep": False}, {"tokens": 512}, {"seqs": 1},
    {"model_len": 1024}, {"prefix": True}, {"speculative": object()},
])
def test_unsupported_feature_config_is_rejected(override):
    with pytest.raises(ValueError, match="NAIVE_PREFILL_MICROBATCHING"):
        validate_feature_config(_config(**override))


# ----- Bundle preparation -----


def test_staged_prepare_call_matches_the_real_prepare_inputs_signature():
    """The bundle test's fake must not diverge from the method it stands in for.

    ``_prepare_two_microbatch_inputs`` passes ``max_chunk_reqs=1`` so each
    microbatch consumes exactly one request. The fake below accepted that
    keyword before the real ``_prepare_inputs`` did, so every unit test passed
    while the first live staged step died with a TypeError. Bind the fake to
    the real signature so that gap cannot reopen.
    """
    import inspect

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    signature = inspect.signature(TPUModelRunner._prepare_inputs)
    assert "max_chunk_reqs" in signature.parameters
    # Optional, so unrelated callers keep working unchanged.
    assert signature.parameters["max_chunk_reqs"].default is None


def test_prepared_bundles_do_not_alias_runner_staging():
    """Two bundles are live at once, so each needs its own copy.

    _prepare_inputs writes into shared staging buffers that the next call
    overwrites; without snapshots the first microbatch would silently execute
    the second one's inputs.
    """
    import torch

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    plan = build_two_microbatch_plan(["a", "b"],
                                     {"a": 1024, "b": 1024}, [0, 0])
    assert plan is not None

    class FakeRunner:
        _two_microbatch_plan = plan
        _staged_microbatch_tokens = 1024
        input_batch = SimpleNamespace(req_ids=["a", "b"])
        input_ids = torch.zeros(1024, dtype=torch.int32)
        position_ids = torch.zeros(1024, dtype=torch.int32)
        logits = torch.zeros(1, dtype=torch.int32)

        def _prepare_inputs(self, scheduler_output, start_index,
                            num_decode_reqs, max_chunk_reqs=None,
                            force_use_max_model_len=False):
            assert max_chunk_reqs == 1
            # Both slots must take the same branch, or they disagree on
            # padded_num_reqs and compile different programs.
            assert force_use_max_model_len is True
            self.input_ids.fill_(start_index + 1)
            self.position_ids.fill_(start_index + 10)
            self.logits.fill_(start_index + 20)
            self._last_sequence_layout_plan = SimpleNamespace(
                request=torch.tensor(start_index),
                global_padded_num_tokens=1024,
                local_padded_num_tokens=1024)
            self._attn_metadata_builder_ctx = {
                "state": torch.tensor(start_index)
            }
            metadata = {"blocks": torch.tensor([start_index])}
            return (metadata, self.logits, 1, 1, start_index + 1,
                    torch.tensor([start_index]), torch.tensor([start_index]),
                    None)

        def _apply_async_token_substitution(self, input_ids, *unused):
            return input_ids

    bundles = TPUModelRunner._prepare_two_microbatch_inputs(
        FakeRunner(), SimpleNamespace())
    assert [bundle.request_id for bundle in bundles] == ["a", "b"]
    assert [bundle.output_index for bundle in bundles] == [0, 1]
    assert bundles[0].input_ids[0].item() == 1
    assert bundles[1].input_ids[0].item() == 2
    assert bundles[0].position_ids[0].item() == 10
    assert bundles[0].attention_metadata["blocks"].item() == 0
    assert bundles[0].logits_indices.item() == 20
    assert bundles[0].layout_plan.request.item() == 0


def test_snapshot_preserves_absent_init_false_metadata_fields():
    import torch

    from vllm_torchtpu.layers.common.attention_metadata import AttentionMetadata
    from vllm_torchtpu.runner.tpu_runner import _snapshot_microbatch_value

    metadata = AttentionMetadata(input_positions=torch.tensor([1, 2]))
    assert not hasattr(metadata, "query_start_loc_cpu")
    assert not hasattr(metadata, "seq_lens_cpu")

    snapshot = _snapshot_microbatch_value(metadata)

    assert snapshot is not metadata
    assert snapshot.input_positions is not metadata.input_positions
    assert torch.equal(snapshot.input_positions, metadata.input_positions)
    assert not hasattr(snapshot, "query_start_loc_cpu")
    assert not hasattr(snapshot, "seq_lens_cpu")


# ----- Dispatch -----


def _dispatch_runner(bundles, plan, forwarded=None):
    import torch

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    class Planner:
        def maybe_select_logits_hidden_states(self, hidden, layout, indices):
            return hidden + (10 if layout == "real" else 100)

    class FakeRunner:
        _two_microbatch_plan = plan
        sequence_layout_planner = Planner()
        _token_padding_state = SimpleNamespace(update=lambda *args: None)
        # Exercise the real logits selection against the fake planner rather
        # than stubbing it out: which hidden state each microbatch selects
        # from is exactly what this dispatch has to get right.
        _staged_microbatch_logits = TPUModelRunner._staged_microbatch_logits

        def _prepare_two_microbatch_inputs(self, scheduler_output,
                                           num_decode_reqs=0):
            return bundles

        def _flush_mamba_state_seed_copies(self):
            pass

        def _run_two_microbatch_forward(self, prepared,
                                        expected_tokens=None):
            if forwarded is not None:
                forwarded.extend(prepared)
            return tuple(torch.tensor([index + 1])
                         for index in range(len(prepared)))

        def compute_logits_from_hidden_states(self, hidden):
            return hidden

    return FakeRunner()


def test_one_real_microbatch_discards_dummy_logits():
    import torch

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    plan = build_two_microbatch_plan(["a"], {"a": 1024}, [0])
    assert plan is not None
    bundles = (
        SimpleNamespace(layout_plan="real", logits_indices=torch.tensor([0])),
        SimpleNamespace(layout_plan="dummy", logits_indices=torch.tensor([0])),
    )

    logits, counts, spec = TPUModelRunner._execute_two_microbatch_prefill(
        _dispatch_runner(bundles, plan), SimpleNamespace())

    assert [item.item() for item in logits] == [11]
    assert counts == [1]
    assert spec == [None]


def test_two_microbatch_execute_restores_scheduler_order():
    """Slot order is the plan's; the scheduler's order is what must come back."""
    import torch

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    # "b" is scheduled first, so it occupies slot 0 and output_index 0.
    plan = build_two_microbatch_plan(["b", "a"], {"a": 1024, "b": 1024},
                                     [0, 0])
    assert plan is not None
    bundles = (
        SimpleNamespace(layout_plan="real", logits_indices=torch.tensor([0])),
        SimpleNamespace(layout_plan="real", logits_indices=torch.tensor([0])),
    )

    logits, counts, spec = TPUModelRunner._execute_two_microbatch_prefill(
        _dispatch_runner(bundles, plan), SimpleNamespace())

    assert [item.item() for item in logits] == [11, 12]
    assert counts == [1, 1]
    assert spec == [None, None]


def test_staged_step_runs_every_microbatch_through_the_forward():
    """Both physical slots execute even when only one holds a real request."""
    import torch

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    plan = build_two_microbatch_plan(["a"], {"a": 1024}, [0])
    assert plan is not None
    bundles = (
        SimpleNamespace(layout_plan="real", logits_indices=torch.tensor([0])),
        SimpleNamespace(layout_plan="dummy", logits_indices=torch.tensor([0])),
    )
    forwarded: list = []

    TPUModelRunner._execute_two_microbatch_prefill(
        _dispatch_runner(bundles, plan, forwarded), SimpleNamespace())

    assert forwarded == list(bundles)


def test_staged_forward_dispatches_one_graph_with_both_metadata_sets(
        monkeypatch):
    """Both microbatches enter a single graph carrying their own metadata.

    Overlap depends on both being visible to one trace, and correctness
    depends on each slot still reaching its own block tables and slot
    mappings -- which is what TwoMicrobatchAttentionMetadata carries in.
    """
    import contextlib

    import torch
    import vllm.distributed.parallel_state as parallel_state

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    communicator = object()
    monkeypatch.setattr(
        parallel_state, "get_dp_group",
        lambda: SimpleNamespace(device_communicator=communicator))
    observed = {}

    @contextlib.contextmanager
    def fake_forward_context(metadata, config, num_tokens,
                             num_tokens_across_dp):
        observed["metadata"] = metadata
        observed["num_tokens"] = num_tokens
        yield

    @contextlib.contextmanager
    def fake_wrapper_context(**kwargs):
        yield

    def fake_forward(model, ids0, ids1, pos0, pos1, comm):
        observed["dispatch"] = (ids0, ids1, pos0, pos1, comm)
        return SimpleNamespace(hidden_states0=torch.tensor([1]),
                               hidden_states1=torch.tensor([2]))

    monkeypatch.setattr("vllm_torchtpu.runner.tpu_runner.set_forward_context",
                        fake_forward_context)
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context",
        fake_wrapper_context)
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner."
        "_forward_qwen3_5_two_microbatches", fake_forward)

    bundles = tuple(
        SimpleNamespace(attention_metadata={"layer": f"mb{index}"},
                        input_ids=torch.zeros(PREFILL_MICROBATCH_TOKENS),
                        position_ids=torch.full((PREFILL_MICROBATCH_TOKENS, ),
                                                index))
        for index in range(2))

    class FakeRunner:
        model = object()
        mesh = None
        vllm_config = None
        _two_microbatch_plan = None
        _staged_microbatch_tokens = PREFILL_MICROBATCH_TOKENS
        _naive_prefill_timed_forwards = set()
        _staged_forward_num_tokens = (
            TPUModelRunner._staged_forward_num_tokens)

        def _dp_num_tokens_across_dp(self, num_tokens):
            return None

    hidden = TPUModelRunner._run_two_microbatch_forward(FakeRunner(), bundles)

    assert observed["num_tokens"] == PREFILL_MICROBATCH_TOKENS
    # One graph, but each slot still resolves to its own metadata.
    assert observed["metadata"]["layer"] == "mb0"
    assert observed["dispatch"][4] is communicator
    assert observed["dispatch"][2][0].item() == 0
    assert observed["dispatch"][3][0].item() == 1
    assert [item.item() for item in hidden] == [1, 2]


def test_staged_forward_requires_an_initialized_communicator():
    """The staged graph cannot launch EP collectives without one."""
    import torch
    import vllm.distributed.parallel_state as parallel_state

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    original = parallel_state.get_dp_group
    parallel_state.get_dp_group = (
        lambda: SimpleNamespace(device_communicator=None))
    try:
        bundles = (SimpleNamespace(
            attention_metadata={},
            input_ids=torch.zeros(PREFILL_MICROBATCH_TOKENS)), ) * 2
        with pytest.raises(RuntimeError, match="device communicator"):
            TPUModelRunner._run_two_microbatch_forward(SimpleNamespace(),
                                                       bundles)
    finally:
        parallel_state.get_dp_group = original


# ----- Capture harness: one profiled staged step per server start -----


class _NoReadTensor:
    """Tensor stub whose contents cannot be read without raising.

    The fingerprint runs on the launch path of the very step we are debugging,
    so it must stay host-side. Reading a device tensor's data there would force
    a synchronization and change the timing it exists to measure.
    """

    def __init__(self, shape, dtype="int32"):
        self.shape = shape
        self.dtype = dtype

    def __iter__(self):
        raise AssertionError("fingerprint read tensor contents")

    def tolist(self):
        raise AssertionError("fingerprint read tensor contents")

    def item(self):
        raise AssertionError("fingerprint read tensor contents")


def _fake_bundle(*, start_index, mamba=True):
    context = SimpleNamespace(
        num_reqs=4, start_index=start_index, use_max_model_len=True,
        seq_lens=_NoReadTensor((4, )),
        query_start_loc=_NoReadTensor((5, )),
        request_distribution=_NoReadTensor((3, )),
        position_ids_override=_NoReadTensor((1024, )),
        mamba_state_indices=_NoReadTensor((4, )) if mamba else None,
        mamba_slot_read_offsets=None, mamba_request_distribution=None,
        sequence_layout_descriptor=SimpleNamespace(cache_key=("all", "v", 1)))
    return SimpleNamespace(
        num_reqs=1, start_index=start_index, end_index=start_index + 1,
        padded_num_reqs=4, input_ids=_NoReadTensor((1024, )),
        position_ids=_NoReadTensor((1024, )),
        logits_indices=_NoReadTensor((4, )), spec_decode_metadata=None,
        attention_builder_context=context,
        layout_plan=SimpleNamespace(
            token_slice=slice(0, 1024), global_num_tokens=1024,
            global_padded_num_tokens=1024, local_num_tokens=1024,
            local_padded_num_tokens=1024))


def test_step_accounting_counts_every_step_not_just_transitions(monkeypatch,
                                                                caplog):
    """The burst summary must count steps, not mode changes.

    The admission line fires only when the coordinated mode changes, so it
    cannot say how many steps staged nor how often a rank held two real
    microbatches. Three separate conclusions in the debugging record were
    weakened by reading transition counts as step counts; this asserts the
    summary does not repeat that.
    """
    import logging

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.envs."
        "VLLM_TPU_NAIVE_PREFILL_MICROBATCHING", True)

    class FakeRunner:
        _naive_prefill_step_mode = PrefillStepMode.IDLE
        _last_logged_naive_prefill_mode = None
        _two_microbatch_plan = None
        _count_naive_prefill_step = TPUModelRunner._count_naive_prefill_step
        _set_coordinated_naive_prefill_mode = (
            TPUModelRunner._set_coordinated_naive_prefill_mode)

        def __init__(self):
            self._naive_prefill_mode_steps = {}
            self._naive_prefill_real_steps = {}
            self._naive_prefill_total_staged_steps = 0
            self._naive_prefill_reorders = 0

    runner = FakeRunner()
    # Ten staged steps in a row: two real microbatches for six of them, one
    # for the rest. The admission line would report this as a single
    # transition; the summary has to report ten.
    for i in range(10):
        runner._two_microbatch_plan = SimpleNamespace(
            num_real_microbatches=2 if i < 6 else 1)
        runner._set_coordinated_naive_prefill_mode(
            PrefillStepMode.TWO_MICROBATCH)
    runner._two_microbatch_plan = None
    for _ in range(3):
        runner._set_coordinated_naive_prefill_mode(PrefillStepMode.SERIAL)

    assert runner._naive_prefill_total_staged_steps == 10
    assert runner._naive_prefill_real_steps == {2: 6, 1: 4}

    with caplog.at_level(logging.INFO):
        runner._set_coordinated_naive_prefill_mode(PrefillStepMode.IDLE)
    summary = [r.getMessage() for r in caplog.records
               if "burst summary" in r.getMessage()]
    assert len(summary) == 1, summary
    assert "staged_steps=10" in summary[0]
    assert "real2=6 real1=4 real0=0" in summary[0]
    assert "serial_steps=3" in summary[0]

    # Counters reset, so a second idle step does not re-emit the same burst.
    with caplog.at_level(logging.INFO):
        caplog.clear()
        runner._set_coordinated_naive_prefill_mode(PrefillStepMode.IDLE)
    assert not [r for r in caplog.records if "burst summary" in r.getMessage()]
    # The lifetime total survives the reset.
    assert runner._naive_prefill_total_staged_steps == 10
