from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from vllm_torchtpu.core.dp_cadence_scheduler import DpCadenceScheduler

BUDGET = 1024


def _req(rid, prefilling, computed=0, prompt=0):
    return SimpleNamespace(
        request_id=rid,
        is_prefill_chunk=prefilling,
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
    )


def _make_scheduler(
    running=(),
    waiting=(),
    filler=32,
    attn_budget=40.0,
    min_chunk=64,
    bound_escape=False,
    log_steps=0,
    lpt=0,
):
    s = DpCadenceScheduler.__new__(DpCadenceScheduler)
    s.running = list(running)
    s.waiting = list(waiting)
    s.prefill_capacity_bound = False
    s.max_num_scheduled_tokens = BUDGET
    s.max_num_running_reqs = 8
    s.scheduler_config = SimpleNamespace(long_prefill_token_threshold=lpt)
    s._cad_filler_tokens = filler
    s._cad_attn_budget = attn_budget * 1e6
    s._cad_min_chunk = min_chunk
    s._cad_bound_escape = bound_escape
    s._cad_log_steps = log_steps
    s._cad_init_stats()
    return s


def _step(s, throttle, tokens, seen=None):
    """One schedule() with upstream replaced by a fake that records the budget,
    chunk limit and capacity-bound flag it was called with, and schedules
    ``tokens`` (request id -> count)."""
    output = SimpleNamespace(
        num_scheduled_tokens=tokens, total_num_scheduled_tokens=sum(tokens.values())
    )

    def fake(*args, **kwargs):
        if seen is not None:
            seen["budget"] = s.max_num_scheduled_tokens
            seen["lpt"] = s.scheduler_config.long_prefill_token_threshold
            seen["bound"] = s.prefill_capacity_bound
        return output

    with patch.object(AsyncScheduler, "schedule", side_effect=fake) as schedule:
        result = s.schedule(throttle_prefills=throttle)
    schedule.assert_called_once_with(throttle)
    assert result is output
    return result


def test_throttled_step_clears_capacity_bound_escape():
    s = _make_scheduler(
        running=[_req("d0", False)], waiting=[_req("w0", False, prompt=900)]
    )
    s.prefill_capacity_bound = True
    seen = {}

    _step(s, throttle=True, tokens={"d0": 7}, seen=seen)

    assert seen == {"budget": BUDGET, "lpt": 0, "bound": False}
    assert s._cad_stats["decode_steps"] == 1
    assert s._cad_stats["throttled_escape"] == 0


def test_bound_escape_opt_in_keeps_upstream_flag():
    s = _make_scheduler(running=[_req("d0", False)], bound_escape=True)
    s.prefill_capacity_bound = True
    seen = {}

    _step(s, throttle=True, tokens={"d0": 7, "p0": 300}, seen=seen)

    assert seen["bound"] is True
    # Prefill on a throttled step with decoders is an escape and is counted.
    assert s._cad_stats["throttled_escape"] == 1


def test_decoderless_rank_gets_only_filler_tokens():
    s = _make_scheduler(running=[_req("p0", True, computed=5000)], filler=32)
    seen = {}

    _step(s, throttle=True, tokens={"p0": 32}, seen=seen)

    assert seen["budget"] == 32
    assert s.max_num_scheduled_tokens == BUDGET
    assert s._cad_stats["filler_steps"] == 1
    assert s._cad_stats["filler_prefill_tokens"] == 32
    assert s._cad_stats["throttled_escape"] == 0


def test_budget_restored_when_upstream_raises():
    s = _make_scheduler(running=[_req("p0", True)], filler=0)
    with (
        patch.object(AsyncScheduler, "schedule", side_effect=RuntimeError),
        pytest.raises(RuntimeError),
    ):
        s.schedule(throttle_prefills=True)
    assert s.max_num_scheduled_tokens == BUDGET


@pytest.mark.parametrize(
    "running,waiting,lpt,expected",
    [
        # 40M / 400k context = 100 tokens per chunk.
        ([_req("p0", True, computed=400_000)], [], 0, 100),
        # A waiting request counts at its prompt length (upper bound on context).
        ([], [_req("w0", False, prompt=200_000)], 0, 200),
        # The floor keeps a 1M-context prompt moving.
        ([_req("p0", True, computed=1_000_000)], [], 0, 64),
        # Short contexts: the cap is above the budget, so no cap at all.
        ([_req("p0", True, computed=10_000)], [], 0, 0),
        # An explicit --long-prefill-token-threshold below the cap still wins.
        ([_req("p0", True, computed=400_000)], [], 72, 72),
        # Decoders' contexts don't count; nothing to prefill means no cap.
        ([_req("d0", False, computed=900_000)], [], 0, 0),
    ],
)
def test_cadence_step_caps_chunk_by_attention(running, waiting, lpt, expected):
    s = _make_scheduler(running=running, waiting=waiting, lpt=lpt)
    seen = {}

    _step(s, throttle=False, tokens={}, seen=seen)

    assert seen["lpt"] == expected
    assert seen["budget"] == BUDGET
    assert s.scheduler_config.long_prefill_token_threshold == lpt


def test_attention_cap_disabled():
    s = _make_scheduler(running=[_req("p0", True, computed=400_000)], attn_budget=0)
    seen = {}
    _step(s, throttle=False, tokens={}, seen=seen)
    assert seen["lpt"] == 0


def test_step_periods_recorded_by_kind_and_logged():
    s = _make_scheduler(running=[_req("d0", False)], log_steps=3)
    clock = iter([10.0, 10.05, 10.35, 10.40])
    with (
        patch(
            "vllm_torchtpu.core.dp_cadence_scheduler.time.monotonic",
            lambda: next(clock),
        ),
        patch.object(AsyncScheduler, "update_from_output", return_value={}),
    ):
        for throttle, tokens in ((True, {"d0": 7}), (False, {"d0": 7, "w0": 900})):
            out = _step(s, throttle=throttle, tokens=tokens)
            s.update_from_output(out, None)
        assert s._cad_periods == {"cadence": [pytest.approx(0.05)]}
        out = _step(s, throttle=True, tokens={"d0": 7})
        pending = _step(s, throttle=True, tokens={"d0": 7})
        s.update_from_output(out, None)
    # The third step triggered the log and reset; the in-flight output survives.
    assert s._cad_stats["steps"] == 0
    assert id(pending) in s._cad_pending


# --------------------------------------------------------------------------
# Integration: the overrides act through the real vLLM 0.29 scheduler.


def _real_scheduler(monkeypatch, max_num_batched_tokens=BUDGET, **env):
    from vllm.config import (
        CacheConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.structured_output import StructuredOutputManager

    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    max_len = 32768
    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        dtype="float16",
        seed=0,
        skip_tokenizer_init=True,
        max_model_len=max_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=8,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_len,
        enable_chunked_prefill=True,
        async_scheduling=True,
        is_encoder_decoder=False,
        watermark=0.0,
    )
    cache_config = CacheConfig(
        block_size=16, cache_dtype="auto", enable_prefix_caching=False
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=20000,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    cache_config.num_gpu_blocks = 20000
    register_all_kvcache_specs(vllm_config)
    return DpCadenceScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=16,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def _request(rid, n):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    return Request(rid, list(range(n)), SamplingParams(max_tokens=16), None)


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def test_real_decoderless_throttled_step_schedules_filler_only(monkeypatch, offline):
    s = _real_scheduler(monkeypatch, TPU_DP_CADENCE_FILLER_TOKENS=32)
    s.add_request(_request("a", 900))

    out = s.schedule(throttle_prefills=True)

    assert out.total_num_scheduled_tokens == 32
    assert s.max_num_scheduled_tokens == BUDGET


def test_real_decoderless_throttled_step_idles_without_filler(monkeypatch, offline):
    s = _real_scheduler(monkeypatch, TPU_DP_CADENCE_FILLER_TOKENS=0)
    s.add_request(_request("a", 900))

    out = s.schedule(throttle_prefills=True)

    assert out.total_num_scheduled_tokens == 0


def test_real_cadence_step_applies_attention_cap(monkeypatch, offline):
    # 1M token x context / 20k-token prompt = 50, floored to the 64-token minimum.
    s = _real_scheduler(
        monkeypatch, TPU_DP_CADENCE_ATTN_BUDGET=1.0, TPU_DP_CADENCE_MIN_CHUNK=64
    )
    s.add_request(_request("long", 20000))
    s.add_request(_request("short", 300))

    out = s.schedule(throttle_prefills=False)

    # The step's cap comes from the longest context, so both chunks are 64.
    assert out.num_scheduled_tokens == {"long": 64, "short": 64}
    assert s.scheduler_config.long_prefill_token_threshold == 0


def test_real_cadence_step_without_long_context_fills_budget(monkeypatch, offline):
    s = _real_scheduler(monkeypatch, TPU_DP_CADENCE_ATTN_BUDGET=40.0)
    s.add_request(_request("a", 700))
    s.add_request(_request("b", 700))

    out = s.schedule(throttle_prefills=False)

    assert out.total_num_scheduled_tokens == BUDGET


def _runner_output(out):
    from vllm.v1.outputs import ModelRunnerOutput

    # Only requests whose prompt is done sample a token.
    req_ids = list(out.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[[7] if r == "d" else [] for r in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _decoder_and_backlog(s):
    """Cadence step that finishes "d"'s prompt, starts p0's, and leaves the rest
    waiting on the token budget, so upstream marks the rank capacity-bound."""
    s.add_request(_request("d", 100))
    for i in range(8):
        s.add_request(_request(f"p{i}", 4000))
    out = s.schedule(throttle_prefills=False)
    assert out.num_scheduled_tokens["d"] == 100 and len(s.waiting) > 0
    assert s.prefill_capacity_bound is True
    s.update_from_output(out, _runner_output(out))
    assert not s.requests["d"].is_prefill_chunk


@pytest.mark.parametrize("bound_escape", [False, True])
def test_real_throttled_step_with_decoder_is_decode_only(
    monkeypatch, offline, bound_escape
):
    s = _real_scheduler(monkeypatch, TPU_DP_CADENCE_BOUND_ESCAPE=int(bound_escape))
    _decoder_and_backlog(s)

    out = s.schedule(throttle_prefills=True)

    if bound_escape:
        # Upstream's escape: the capacity-bound rank prefills off-cadence too.
        assert set(out.num_scheduled_tokens) > {"d"}
        assert s._cad_stats["throttled_escape"] == 1
    else:
        assert out.num_scheduled_tokens == {"d": 1}
        assert s._cad_stats["throttled_escape"] == 0
