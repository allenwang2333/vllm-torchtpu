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

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.request import RequestStatus

from vllm_torchtpu.core.hybrid_retention import HybridStateRetention, _ConvLRUEntry
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler


def _make_request(
    request_id: str,
    num_tokens: int = 1000,
    arrival_time: float = 0.0,
    num_preemptions: int = 0,
    num_computed_tokens: int = 0,
    skip_reading_prefix_cache: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=num_tokens,
        arrival_time=arrival_time,
        num_preemptions=num_preemptions,
        num_computed_tokens=num_computed_tokens,
        skip_reading_prefix_cache=skip_reading_prefix_cache,
        block_hashes=[object()],
    )


def _make_scheduler(
    requests: list[SimpleNamespace],
    warmth_map: dict[str, int],
    window: int = 64,
    enable_caching: bool = True,
    raise_on_lookup: bool = False,
) -> PrefixAwareScheduler:
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    q = FCFSRequestQueue()
    for r in requests:
        q.add_request(r)
    s.waiting = q
    s.kv_holding_waiting = FCFSRequestQueue()
    s.running = []
    s.policy = SchedulingPolicy.FCFS
    s.WINDOW = window
    s._paw_n = 0
    s._paw_cold_tokens = 0
    s._paw_prompt_tokens = 0
    s._paw_scoring_broken = False

    coordinator = MagicMock()

    def find_hit(block_hashes, max_len):
        if raise_on_lookup:
            raise RuntimeError("simulated coordinator failure")
        for r in requests:
            if r.block_hashes is block_hashes:
                return ({}, warmth_map.get(r.request_id, 0), 0)
        return ({}, 0, 0)

    coordinator.find_longest_cache_hit.side_effect = find_hit

    kvm = MagicMock()
    kvm.enable_caching = enable_caching
    kvm.coordinator = coordinator
    s.kv_cache_manager = kvm
    return s


def test_default_knobs():
    assert PrefixAwareScheduler.WINDOW == 64


def test_promotes_warmest_within_window():
    now = time.time()
    r_cold = _make_request("cold", num_tokens=1000, arrival_time=now - 10)
    r_warm = _make_request("warm", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r_cold, r_warm], {"cold": 0, "warm": 800})

    expected = object()

    def schedule(throttle_prefills):
        assert throttle_prefills is True
        assert [r.request_id for r in s.waiting] == ["warm", "cold"]
        return expected

    with patch.object(AsyncScheduler, "schedule", side_effect=schedule):
        assert s.schedule(throttle_prefills=True) is expected


def test_promotes_waiting_while_kv_holding_waiting_is_not_empty():
    now = time.time()
    r_cold = _make_request("cold", num_tokens=1000, arrival_time=now - 10)
    r_warm = _make_request("warm", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r_cold, r_warm], {"cold": 0, "warm": 800})
    loading = _make_request("loading", num_computed_tokens=900)
    s.kv_holding_waiting.add_request(loading)

    q = s._select_waiting_queue_for_scheduling()
    assert q is s.kv_holding_waiting
    assert [r.request_id for r in s.waiting] == ["warm", "cold"]


def test_ranks_by_uncached_tokens_not_hit_ratio():
    now = time.time()
    # 100k tokens, 90% cached -> 10k uncached tokens
    r_long = _make_request("long", num_tokens=100_000, arrival_time=now - 10)
    # 20k tokens, 80% cached -> 4k uncached tokens (fewer uncached tokens wins)
    r_short = _make_request("short", num_tokens=20_000, arrival_time=now - 5)
    s = _make_scheduler([r_long, r_short], {"long": 90_000, "short": 16_000})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["short", "long"]


def test_fcfs_order_preserved_when_all_cold():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 0})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    order = [r.request_id for r in q]
    assert order == ["r1", "r2"]


def test_preempted_tier_outranks_normal():
    now = time.time()
    r_preempted = _make_request(
        "preempted", num_tokens=1000, arrival_time=now - 5, num_preemptions=1
    )
    r_warm = _make_request("warm", num_tokens=1000, arrival_time=now - 10)
    s = _make_scheduler([r_preempted, r_warm], {"preempted": 0, "warm": 900})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    order = [r.request_id for r in q]
    assert order == ["preempted", "warm"]


def test_window_bounds_evaluation():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 15)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 10)
    r_outside = _make_request("outside", num_tokens=1000, arrival_time=now - 5)
    # window = 2, so r_outside is beyond the inspection window
    s = _make_scheduler(
        [r1, r2, r_outside], {"r1": 100, "r2": 200, "outside": 999}, window=2
    )

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    order = [r.request_id for r in q]
    assert order[0] == "r2"  # r2 was warmest in window [r1, r2]


def test_single_or_empty_queue_is_noop():
    s = _make_scheduler([_make_request("single")], {"single": 500})
    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["single"]


def test_uses_num_computed_tokens_when_present():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request(
        "r2", num_tokens=1000, arrival_time=now - 5, num_computed_tokens=900
    )
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 0})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["r2", "r1"]


def test_skip_reading_prefix_cache_or_caching_disabled():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request(
        "r2", num_tokens=1000, arrival_time=now - 5, skip_reading_prefix_cache=True
    )
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 900})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["r1", "r2"]

    s_disabled = _make_scheduler(
        [r1, _make_request("r3", num_tokens=1000, arrival_time=now - 5)],
        {"r1": 0, "r3": 900},
        enable_caching=False,
    )
    q_disabled = s_disabled._select_waiting_queue_for_scheduling()
    assert q_disabled is not None
    assert [r.request_id for r in q_disabled] == ["r1", "r3"]


def test_coordinator_failure_falls_back_to_fcfs():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 500}, raise_on_lookup=True)

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["r1", "r2"]
    assert s._paw_scoring_broken is True


def test_epoch_cache_reuses_longest_cache_hit_and_invalidates_on_epoch_change():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r1, r2], {"r1": 128, "r2": 640})

    class DummyBlockPool:
        def __init__(self):
            self.cached_block_hash_to_block = {}
            self._tpu_prefix_cache_epoch = 1

    pool = DummyBlockPool()
    s.kv_cache_manager.block_pool = pool
    s.kv_cache_manager.enable_kv_cache_events = False
    s.kv_cache_manager.prefix_cache_lookup_enabled = lambda req: True
    s.kv_cache_manager.create_kv_cache_blocks = lambda blocks: blocks
    s.connector = None

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["r2", "r1"]
    assert s.kv_cache_manager.coordinator.find_longest_cache_hit.call_count == 2

    # _get_local_prefix_cache_hit on the promoted request reuses _paw_hit_cache
    blocks, num_local, boundary, diverged = s._get_local_prefix_cache_hit(r2)
    assert num_local == 640
    assert diverged is False
    assert s.kv_cache_manager.coordinator.find_longest_cache_hit.call_count == 2

    # Repeated scheduling step with unchanged epoch does not re-invoke coordinator
    s._select_waiting_queue_for_scheduling()
    assert s.kv_cache_manager.coordinator.find_longest_cache_hit.call_count == 2

    # Advancing _tpu_prefix_cache_epoch invalidates _paw_hit_cache
    pool._tpu_prefix_cache_epoch += 1
    s._select_waiting_queue_for_scheduling()
    assert s.kv_cache_manager.coordinator.find_longest_cache_hit.call_count == 4


def test_free_block_fit_tier_promotes_fitting_warm_over_stalling_head():
    now = time.time()
    # waiting[0] is preempted but needs ceil(1000 / 16) = 63 blocks > 10 free blocks
    r_preempted = _make_request(
        "preempted_big",
        num_tokens=1000,
        arrival_time=now - 20,
        num_preemptions=1,
    )
    # waiting[1] is warm (cold_tokens=320 -> 20 blocks), still > 10 free blocks
    r_warm_big = _make_request("warm_big", num_tokens=1000, arrival_time=now - 15)
    # waiting[2] is warm (cold_tokens=120 -> ceil(120/16)=8 blocks <= 10 free blocks)
    r_warm_fit = _make_request("warm_fit", num_tokens=1000, arrival_time=now - 10)

    s = _make_scheduler(
        [r_preempted, r_warm_big, r_warm_fit],
        {"preempted_big": 0, "warm_big": 680, "warm_fit": 880},
    )
    s.block_size = 16
    s.kv_cache_manager.block_pool.get_num_free_blocks.return_value = 10

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == [
        "warm_fit",
        "preempted_big",
        "warm_big",
    ]


def test_free_block_fit_tier_picks_warmest_among_multiple_fitting():
    now = time.time()
    r_head = _make_request(
        "head_big", num_tokens=1000, arrival_time=now - 20, num_preemptions=1
    )
    # 128 cold tokens -> 8 blocks <= 10 free blocks
    r_fit_8 = _make_request("fit_8", num_tokens=1000, arrival_time=now - 15)
    # 48 cold tokens -> 3 blocks <= 10 free blocks (fewer cold tokens wins)
    r_fit_3 = _make_request("fit_3", num_tokens=1000, arrival_time=now - 10)

    s = _make_scheduler(
        [r_head, r_fit_8, r_fit_3],
        {"head_big": 200, "fit_8": 872, "fit_3": 952},
    )
    s.block_size = 16
    s.kv_cache_manager.block_pool.get_num_free_blocks.return_value = 10

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["fit_3", "head_big", "fit_8"]


def test_free_block_fit_tier_preserves_order_when_no_warm_request_fits():
    now = time.time()
    r_preempted = _make_request(
        "preempted_big",
        num_tokens=1000,
        arrival_time=now - 20,
        num_preemptions=1,
    )
    # 160 cold tokens -> 10 blocks > 5 free blocks
    r_warm_nonfit = _make_request(
        "warm_nonfit", num_tokens=1000, arrival_time=now - 15
    )
    # 32 cold tokens -> 2 blocks <= 5 free blocks, but warmth == 0 (cold, not warm)
    r_cold_small = _make_request(
        "cold_small", num_tokens=32, arrival_time=now - 10
    )

    s = _make_scheduler(
        [r_preempted, r_warm_nonfit, r_cold_small],
        {"preempted_big": 0, "warm_nonfit": 840, "cold_small": 0},
    )
    s.block_size = 16
    s.kv_cache_manager.block_pool.get_num_free_blocks.return_value = 5

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == [
        "preempted_big",
        "warm_nonfit",
        "cold_small",
    ]


def test_free_block_fit_tier_inactive_when_head_fits_in_free_blocks():
    now = time.time()
    # 160 cold tokens -> 10 blocks <= 10 free blocks (head fits!)
    r_preempted = _make_request(
        "preempted_fits",
        num_tokens=1000,
        arrival_time=now - 10,
        num_preemptions=1,
    )
    # 32 cold tokens -> 2 blocks <= 10 free blocks
    r_warm = _make_request("warm", num_tokens=1000, arrival_time=now - 5)

    s = _make_scheduler(
        [r_preempted, r_warm],
        {"preempted_fits": 840, "warm": 968},
    )
    s.block_size = 16
    s.kv_cache_manager.block_pool.get_num_free_blocks.return_value = 10

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["preempted_fits", "warm"]



def _running_request(request_id: str, prompt: int, computed: int) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=prompt,
        num_computed_tokens=computed,
    )


def _make_step_scheduler(running, monkeypatch, budget=8192):
    """A PrefixAwareScheduler whose vLLM schedule() records what it saw."""
    seen = {}

    def fake_schedule(self, *args, **kwargs):
        seen["order"] = [r.request_id for r in self.running]
        seen["budget"] = self.max_num_scheduled_tokens
        seen["args"] = (args, kwargs)
        return "out"

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_schedule, raising=False)
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.running = list(running)
    s.max_num_scheduled_tokens = budget
    return s, seen


def test_decoders_first_moves_decoders_ahead_of_partial_prefills(monkeypatch):
    running = [
        _running_request("p1", prompt=100_000, computed=7680),
        _running_request("d1", prompt=500, computed=501),
        _running_request("new", prompt=2000, computed=0),
        _running_request("d2", prompt=800, computed=800),
        _running_request("p2", prompt=50_000, computed=640),
    ]
    s, seen = _make_step_scheduler(running, monkeypatch)
    running_list = s.running

    assert s.schedule(True) == "out"

    # With SRPF enabled by default, decoders come first in stable order and
    # prefilling requests are ordered by fewest remaining prompt tokens.
    assert seen["order"] == ["d1", "d2", "new", "p2", "p1"]
    assert s.running is running_list
    assert seen["args"] == ((True,), {})

    # With TPU_SCHED_SRPF_RUNNING=0, prefilling requests keep FCFS order.
    monkeypatch.setenv("TPU_SCHED_SRPF_RUNNING", "0")
    s_fcfs, seen_fcfs = _make_step_scheduler(running, monkeypatch)
    assert s_fcfs.schedule(True) == "out"
    assert seen_fcfs["order"] == ["d1", "d2", "p1", "new", "p2"]


def test_decoders_first_is_noop_without_a_mix(monkeypatch):
    for running in (
        [_running_request("d1", 10, 10), _running_request("d2", 10, 12)],
        [_running_request("p1", 10, 5), _running_request("p2", 10, 0)],
        [],
    ):
        s, seen = _make_step_scheduler(running, monkeypatch)
        s.schedule()
        assert seen["order"] == [r.request_id for r in running]


def test_no_step_token_cap_with_many_decoders(monkeypatch):
    """The validated policy has no decode-protect cap: the step budget stays
    max_num_batched_tokens however many requests decode."""
    running = [_running_request(f"d{i}", 100, 100) for i in range(24)]
    running.append(_running_request("p", 100_000, 0))
    s, seen = _make_step_scheduler(running, monkeypatch, budget=8192)
    s.schedule()
    assert seen["budget"] == 8192
    assert s.max_num_scheduled_tokens == 8192


def test_cold_prefill_chunk_cap_only_caps_cold_requests_at_high_concurrency():
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s._step_num_decoding = 16
    s._step_decode_tokens = 64
    s.waiting = []
    s.kv_holding_waiting = []

    # Cold request (remaining = 90,000 > 8192) is capped to 2048 - 64 = 1984
    # so decode_tokens + cold_chunk <= 2048 XLA compile bucket.
    cold = _running_request("cold", prompt=90_000, computed=0)
    assert s._cap_cold_prefill_tokens(cold, 0, 8128) == 2048 - 64

    # At 32 decoders (128 decode tokens), cap is 2048 - 128 = 1920 (exact 3 * 640 Mamba blocks).
    s._step_num_decoding = 32
    s._step_decode_tokens = 128
    assert s._cap_cold_prefill_tokens(cold, 0, 8064) == 1920

    # Warm delta turn (remaining = 4,000 <= 8192) is NEVER capped, even at 30 decoders.
    s._step_num_decoding = 30
    s._step_decode_tokens = 120
    warm = _running_request("warm", prompt=50_000, computed=0)
    assert s._cap_cold_prefill_tokens(warm, 46_000, 4000) == 4000

    # At c=8 / c=12 (num_decoding < 12 and waiting empty), cold request is NOT capped.
    s._step_num_decoding = 11
    s._step_decode_tokens = 44
    assert s._cap_cold_prefill_tokens(cold, 0, 8128) == 8128

    # When waiting queue is non-empty, cold request in running IS capped to unlock waiting.
    s._step_num_decoding = 4
    s._step_decode_tokens = 16
    s.waiting = [warm]
    s._step_running_prefill_ids = {id(cold)}
    assert s._cap_cold_prefill_tokens(cold, 2048, 8128) == 2048 - 16

    # First-step admission from waiting only triggers on decoder count (not self-triggering).
    s._step_running_prefill_ids = set()
    assert s._cap_cold_prefill_tokens(cold, 0, 8128) == 8128


def test_step_total_cap_prevents_multi_prefill_8192_bucket_spill():
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.block_size = 640
    s._step_num_decoding = 28
    s._step_decode_tokens = 112
    s._step_prefill_tokens_by_req = {}
    s.waiting = []
    s.kv_holding_waiting = []

    c1 = _running_request("c1", prompt=90_000, computed=0)
    c2 = _running_request("c2", prompt=90_000, computed=0)
    c3 = _running_request("c3", prompt=90_000, computed=0)

    # First cold chunk: 2048 - 112 = 1936 (-> 1920 after Mamba 640-block align)
    assert s._cap_cold_prefill_tokens(c1, 0, 8080) == 1936
    s._step_prefill_tokens_by_req[id(c1)] = 1920

    # Second cold chunk: 112 + 1920 = 2032 already scheduled; fits within 4096 cap (1936 -> 1920)
    assert s._cap_cold_prefill_tokens(c2, 0, 6160) == 1936
    s._step_prefill_tokens_by_req[id(c2)] = 1920

    # Third prefill request: 112 + 1920 + 1920 = 3952 already scheduled ->
    # remaining under 4096 cap is 144 < 640 block_size, so it is deferred (0 tokens)
    # rather than spilling the step into the 8192 XLA bucket.
    assert s._cap_cold_prefill_tokens(c3, 0, 4240) == 0

    # Strict 2048 step-total cap (TPU_SCHED_STEP_TOTAL_CAP=2048) defers 2nd prefill immediately:
    s._step_total_cap = 2048
    s._step_prefill_tokens_by_req = {}
    assert s._cap_cold_prefill_tokens(c1, 0, 8080) == 1936
    s._step_prefill_tokens_by_req[id(c1)] = 1920
    assert s._cap_cold_prefill_tokens(c2, 0, 6160) == 0


def test_warm_streak_cadence_gating_defers_prefills_until_k_decode_steps():
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.block_size = 640
    s._step_num_decoding = 28
    s._step_decode_tokens = 112
    s._warm_streak = 2
    s._warm_min_decoders = 24
    s._pure_decode_streak = 0
    s.waiting = []
    s.kv_holding_waiting = []

    warm = _running_request("warm", prompt=20_000, computed=18_000)
    # Streak = 0 < 2 -> prefill deferred
    assert s._cap_cold_prefill_tokens(warm, 18_000, 2000) == 0
    # Streak = 1 < 2 -> prefill deferred
    s._pure_decode_streak = 1
    assert s._cap_cold_prefill_tokens(warm, 18_000, 2000) == 0
    # Streak = 2 >= 2 -> prefill admitted
    s._pure_decode_streak = 2
    assert s._cap_cold_prefill_tokens(warm, 18_000, 2000) == 2000


def test_mk_max_running_caps_running_seqs_when_megakernel_active(monkeypatch):
    seen = {}

    def fake_schedule(self, *args, **kwargs):
        seen["max_running"] = self.max_num_active_reqs
        seen["runner_slots"] = self.max_num_running_reqs
        return "out"

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_schedule, raising=False)
    monkeypatch.setenv("VLLM_TPU_QWEN35_MEGAKERNEL", "1")
    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_SPLIT_WIDE", "0")
    # Auto mode is opt-in (the default 0 leaves max_num_active_reqs alone).
    monkeypatch.setenv("TPU_SCHED_MK_MAX_RUNNING", "-1")

    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.running = [_running_request(f"d{i}", 100, 100) for i in range(16)]
    s.max_num_scheduled_tokens = 8192
    s.max_num_running_reqs = 32
    s.max_num_active_reqs = 32
    s.dcp_world_size = 1
    s.num_spec_tokens = 3

    assert s._resolve_mk_max_running() == 16
    assert s.schedule() == "out"
    assert seen["max_running"] == 16
    assert seen["runner_slots"] == 32
    # Restored after schedule() returns
    assert s.max_num_active_reqs == 32

    # TPU_SCHED_MK_MAX_RUNNING=0 (also the default) preserves max_num_active_reqs=32
    monkeypatch.setenv("TPU_SCHED_MK_MAX_RUNNING", "0")
    s._mk_max_running = None
    assert s.schedule() == "out"
    assert seen["max_running"] == 32
    monkeypatch.delenv("TPU_SCHED_MK_MAX_RUNNING")
    s._mk_max_running = None
    assert s._resolve_mk_max_running() == 0


def test_near_finish_decode_protection_defers_large_prefill_with_starvation_cap():
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.block_size = 640
    s.num_spec_tokens = 3  # tokens_per_dec = 4
    s._step_num_decoding = 4
    s._step_decode_tokens = 16
    s._finish_decode_steps = 2  # threshold = 2 * 4 = 8 remaining output tokens
    s._finish_decode_max_deferral_steps = 2
    s._finish_decode_prefill_min_tokens = 2048
    s._finish_decode_defer_streak = 0
    s.waiting = []
    s.kv_holding_waiting = []

    near_done = _running_request("d_near", prompt=1000, computed=1094)
    near_done.max_tokens = 100
    near_done.num_output_tokens = 94  # remaining = 6 <= 8
    far_from_done = _running_request("d_far", prompt=1000, computed=1050)
    far_from_done.max_tokens = 100
    far_from_done.num_output_tokens = 50  # remaining = 50 > 8
    s.running = [near_done, far_from_done]

    cold = _running_request("cold", prompt=20_000, computed=0)
    warm_small = _running_request("warm_small", prompt=2000, computed=1500)

    # Small warm prefill (500 < 2048) is NOT deferred even when near_done is active:
    assert s._cap_cold_prefill_tokens(warm_small, 1500, 500) == 500
    assert s._finish_decode_defer_streak == 0

    # Large prefill (4096 >= 2048) is deferred for up to max_deferral_steps (2) steps:
    assert s._cap_cold_prefill_tokens(cold, 0, 4096) == 0
    assert s._finish_decode_defer_streak == 1
    assert s._cap_cold_prefill_tokens(cold, 0, 4096) == 0
    assert s._finish_decode_defer_streak == 2

    # On the 3rd consecutive attempt (streak == 2 == max_deferral_steps), starvation cap
    # allows the prefill through and resets the streak to 0:
    assert s._cap_cold_prefill_tokens(cold, 0, 4096) == 4096
    assert s._finish_decode_defer_streak == 0

    # When no decoder is within threshold, prefill is not deferred:
    s.running = [far_from_done]
    assert s._cap_cold_prefill_tokens(cold, 0, 4096) == 4096
    assert s._finish_decode_defer_streak == 0


def test_near_finish_decode_protection_in_schedule_loop(monkeypatch):
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.block_size = 640
    s.num_spec_tokens = 3
    s._finish_decode_steps = 2
    s._finish_decode_max_deferral_steps = 2
    s._finish_decode_prefill_min_tokens = 2048
    s._finish_decode_defer_streak = 0
    s.waiting = []
    s.kv_holding_waiting = []

    near_done = _running_request("d_near", prompt=1000, computed=1095)
    near_done.max_tokens = 100
    near_done.num_output_tokens = 95  # remaining = 5 <= 8
    cold = _running_request("cold", prompt=20_000, computed=0)
    s.running = [near_done, cold]

    scheduled_chunks = []

    def fake_schedule(self, *args, **kwargs):
        toks = self._cap_cold_prefill_tokens(cold, 0, 4096)
        scheduled_chunks.append(toks)
        return "out"

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_schedule, raising=False)

    # Step 1: deferred (streak becomes 1)
    s.schedule()
    assert scheduled_chunks[-1] == 0
    assert s._finish_decode_defer_streak == 1

    # Step 2: deferred (streak becomes 2)
    s.schedule()
    assert scheduled_chunks[-1] == 0
    assert s._finish_decode_defer_streak == 2

    # Step 3: starvation cap reached -> allowed through, streak resets to 0
    s.schedule()
    assert scheduled_chunks[-1] == 4096
    assert s._finish_decode_defer_streak == 0


def test_hbm_warm_outranks_cpu_warm_and_cpu_warm_outranks_cold():
    now = time.time()
    r_cold = _make_request("cold", num_tokens=90_000, arrival_time=now - 30)
    r_cpu = _make_request("cpu_warm", num_tokens=90_000, arrival_time=now - 20)
    r_hbm = _make_request("hbm_warm", num_tokens=90_000, arrival_time=now - 10)
    s = _make_scheduler(
        [r_cold, r_cpu, r_hbm],
        {"cold": 0, "cpu_warm": 0, "hbm_warm": 80_000},
    )
    s.block_size = 640
    connector = MagicMock()
    connector.estimate_external_hit_tokens.side_effect = (
        lambda req, local_toks: 88_000 if req.request_id == "cpu_warm" else 0
    )
    s.connector = connector

    # Even though cpu_warm has 88,000 external tokens (2,000 uncached) vs
    # hbm_warm's 80,000 HBM tokens (10,000 uncached), hbm_warm is strictly
    # preferred over cpu_warm on the first selection, and cpu_warm is promoted
    # over cold on the next selection.
    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert q.pop_request().request_id == "hbm_warm"
    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["cpu_warm", "cold"]


def test_kv_producer_cold_cap_reserves_budget_for_ready_warm_and_multi_cold():
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s._is_disagg = True
    s._is_kv_producer = True
    s._is_kv_consumer = False
    s.block_size = 640
    s.max_num_scheduled_tokens = 8192
    s._cold_chunk_cap = 2048
    s._cold_threshold = 8192
    s._cold_min_decoders = 12
    s._step_num_decoding = 0
    s._step_decode_tokens = 0
    s._step_prefill_tokens_by_req = {}

    c1 = _running_request("c1", prompt=90_000, computed=0)
    c2 = _running_request("c2", prompt=90_000, computed=0)
    s._step_running_prefill_ids = {id(c1), id(c2)}

    # Step t: CPU-warm request in waiting (needs 0 TPU tokens on Step t, only
    # needs token_budget > 0 in Phase 2 to start async CPU load).
    r_cpu = _make_request("cpu_warm", num_tokens=86_200, arrival_time=1.0)
    s.waiting = [r_cpu]
    s.kv_holding_waiting = []
    s.finished_recving_kv_req_ids = set()
    s._prefix_warmth = lambda req: 0
    s._external_warmth = lambda req, loc: 85_000 if req is r_cpu else 0

    assert s._compute_warm_waiting_tokens(8192) == 0
    # First cold request gets full 8191 (-> 7680 aligned); second gets 0 so
    # token_budget > 0 is preserved for Phase 2 async load initiation.
    assert s._cap_cold_prefill_tokens(c1, 0, 8192) == 8191
    s._step_prefill_tokens_by_req[id(c1)] = 7680
    assert s._cap_cold_prefill_tokens(c2, 0, 512) == 0

    # Step t+1: CPU load finished -> r_cpu is in kv_holding_waiting with
    # finished_recving_kv_req_ids and 1,200 remaining tokens.
    s._step_prefill_tokens_by_req = {}
    r_cpu.num_computed_tokens = 85_000
    s.waiting = []
    s.kv_holding_waiting = [r_cpu]
    s.finished_recving_kv_req_ids = {"cpu_warm"}

    warm_toks = s._compute_warm_waiting_tokens(8192)
    assert warm_toks == 3 * 640  # ceil(1200/640) + 1 = 3 blocks = 1920
    # First cold request is capped to 2048; second cold request gets 0!
    assert s._cap_cold_prefill_tokens(c1, 7680, 8192) == 2048
    s._step_prefill_tokens_by_req[id(c1)] = 1920
    assert s._cap_cold_prefill_tokens(c2, 0, 6272) == 0






# --- PAW aging, cold-cap release, retention guard, admission telemetry ---


def _aged(request_id: str, prompt: int, computed: int, waited_s: float, now: float):
    r = _running_request(request_id, prompt=prompt, computed=computed)
    r.arrival_time = now - waited_s
    return r


def _producer(now: float, budget: int = 8192) -> PrefixAwareScheduler:
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s._is_disagg = True
    s._is_kv_producer = True
    s._is_kv_consumer = False
    s.block_size = 640
    s.max_num_scheduled_tokens = budget
    s._cold_chunk_cap = 2560
    s._cold_threshold = 4096
    s._cold_min_decoders = 12
    s._step_num_decoding = 0
    s._step_decode_tokens = 0
    s._step_prefill_tokens_by_req = {}
    s._step_wall_time = now
    s._step_cold_exempt_ids = set()
    s.kv_holding_waiting = []
    return s


def test_paw_aging_lets_long_waiting_cold_prefill_overtake_srpf(monkeypatch):
    now = time.time()
    running = [
        _aged("cold_old", 240_000, 0, waited_s=90, now=now),
        _aged("cold_new", 150_000, 0, waited_s=1, now=now),
        _aged("warm", 3_000, 0, waited_s=0.5, now=now),
        _running_request("d1", prompt=500, computed=501),
    ]
    monkeypatch.setenv("TPU_SCHED_PAW_AGING_TOK_PER_S", "0")
    s, seen = _make_step_scheduler(running, monkeypatch)
    s.schedule()
    assert seen["order"] == ["d1", "warm", "cold_new", "cold_old"]

    # 240,000 - 3,000 x 90 s < 0 puts the 90 s old cold prefill first;
    # 150,000 - 3,000 x 1 s keeps the fresh one behind the warm turn.
    monkeypatch.setenv("TPU_SCHED_PAW_AGING_TOK_PER_S", "3000")
    s, seen = _make_step_scheduler(running, monkeypatch)
    s.schedule()
    assert seen["order"] == ["d1", "cold_old", "warm", "cold_new"]


def test_paw_aging_reorders_waiting_queue_within_a_tier(monkeypatch):
    now = time.time()
    small = _make_request("small_cold", num_tokens=30_000, arrival_time=now - 1)
    big = _make_request("big_cold", num_tokens=150_000, arrival_time=now - 61)

    monkeypatch.setenv("TPU_SCHED_PAW_AGING_TOK_PER_S", "0")
    s = _make_scheduler([small, big], {"small_cold": 0, "big_cold": 0})
    q = s._select_waiting_queue_for_scheduling()
    assert [r.request_id for r in q] == ["small_cold", "big_cold"]

    # 150,000 - 3,000 x 61 s is below 30,000 - 3,000 x 1 s.
    monkeypatch.setenv("TPU_SCHED_PAW_AGING_TOK_PER_S", "3000")
    warm = _make_request("warm", num_tokens=50_000, arrival_time=now)
    s = _make_scheduler(
        [small, big, warm], {"small_cold": 0, "big_cold": 0, "warm": 47_000}
    )
    q = s._select_waiting_queue_for_scheduling()
    # Aging works within a tier: the locally warm request still goes first.
    assert q.pop_request().request_id == "warm"
    q = s._select_waiting_queue_for_scheduling()
    assert [r.request_id for r in q] == ["big_cold", "small_cold"]


def test_cold_cap_exempts_running_cold_request_past_max_wait(monkeypatch):
    monkeypatch.setenv("TPU_SCHED_PAW_MAX_WAIT_S", "10")
    now = time.time()
    s = _producer(now)
    fresh = _aged("fresh", 200_000, 0, waited_s=2, now=now)
    old = _aged("old", 200_000, 0, waited_s=30, now=now)
    s._step_running_prefill_ids = {id(fresh), id(old)}
    s.waiting = [object()]
    # Two warm turns waiting need 7,552 tokens: the cold group gets one block.
    s._step_warm_waiting_tokens = 7552

    assert s._cap_cold_prefill_tokens(fresh, 0, 8192) == 640
    assert s._step_warm_reserve_applied is True
    s._step_prefill_tokens_by_req[id(fresh)] = 640
    # The old request's group budget is 8,191 - min(7,552, 0.5 x 8,192) =
    # 4,095, so the waiting warm turns keep 4,096 tokens of the step.
    assert s._cap_cold_prefill_tokens(old, 0, 8192) == 4095 - 640
    assert s._step_cold_exempt_ids == {id(old)}

    # Alone on the step, the old request gets the same 4,095.
    s = _producer(now)
    s._step_running_prefill_ids = {id(old)}
    s.waiting = [object()]
    s._step_warm_waiting_tokens = 7552
    assert s._cap_cold_prefill_tokens(old, 0, 8192) == 4095
    # A small warm backlog leaves the exempt request the rest of the step.
    s._step_prefill_tokens_by_req = {}
    s._step_warm_waiting_tokens = 1280
    assert s._cap_cold_prefill_tokens(old, 0, 8192) == 8191 - 1280

    # Reserve 0: the exempt request keeps only the phase-1 one-token gap.
    monkeypatch.setenv("TPU_SCHED_PAW_EXEMPT_WARM_RESERVE", "0")
    s = _producer(now)
    s._step_running_prefill_ids = {id(old)}
    s.waiting = [object()]
    s._step_warm_waiting_tokens = 7552
    assert s._cap_cold_prefill_tokens(old, 0, 8192) == 8191

    # MAX_WAIT_S = 0: no exemption, the old request is capped like the fresh one.
    monkeypatch.setenv("TPU_SCHED_PAW_MAX_WAIT_S", "0")
    s = _producer(now)
    s._step_running_prefill_ids = {id(old)}
    s.waiting = [object()]
    s._step_warm_waiting_tokens = 7552
    assert s._cap_cold_prefill_tokens(old, 0, 8192) == 640


def test_cold_cap_released_after_a_step_that_admitted_no_warm_request(monkeypatch):
    monkeypatch.setenv("TPU_SCHED_PAW_MAX_WAIT_S", "10")
    now = time.time()
    cold = _aged("cold", 200_000, 6400, waited_s=1, now=now)
    warm = _make_request("warm", num_tokens=50_000, arrival_time=now)
    admit_warm = {"on": False}
    chunks = []

    def fake_schedule(self, *args, **kwargs):
        n = self._cap_cold_prefill_tokens(cold, cold.num_computed_tokens, 8192)
        chunks.append(n)
        scheduled = {"cold": n}
        if admit_warm["on"]:
            scheduled["warm"] = 4000
        return SimpleNamespace(
            num_scheduled_tokens=scheduled,
            total_num_scheduled_tokens=sum(scheduled.values()),
        )

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_schedule, raising=False)
    s = _producer(now)
    s.running = [cold]
    s.waiting = [warm]
    s._prefix_warmth = lambda r: 46_000 if r is warm else 0
    s._external_warmth = lambda r, local: 0

    # (ceil(4000 / 640) + 1) x 640 = 5,120 warm tokens: cold cap 2,560.
    s.schedule()
    assert chunks[-1] == 2560 and s._warm_reserve_wasted
    # The warm turn was not admitted: the next step does not cap the cold group.
    s.schedule()
    assert chunks[-1] == 8191 and not s._warm_reserve_wasted
    # The release step reserved nothing, so the cap comes back.
    admit_warm["on"] = True
    s.schedule()
    assert chunks[-1] == 2560 and not s._warm_reserve_wasted
    s.schedule()
    assert chunks[-1] == 2560

    # With TPU_SCHED_PAW_MAX_WAIT_S=0 there is no release: the S0 cold cap.
    monkeypatch.setenv("TPU_SCHED_PAW_MAX_WAIT_S", "0")
    admit_warm["on"] = False
    s = _producer(now)
    s.running = [cold]
    s.waiting = [warm]
    s._prefix_warmth = lambda r: 46_000 if r is warm else 0
    s._external_warmth = lambda r, local: 0
    s.schedule()
    s.schedule()
    assert chunks[-2:] == [2560, 2560]


class _MambaSpec:
    pass


_MambaSpec.__name__ = "MambaSpec"


class _FullAttentionSpec:
    pass


_FullAttentionSpec.__name__ = "FullAttentionSpec"


class _GuardPool:
    num_gpu_blocks = 8000

    def __init__(self, free: int):
        self.free = free

    def get_num_free_blocks(self) -> int:
        return self.free

    def free_blocks(self, blocks) -> None:
        for b in blocks:
            b.ref_cnt -= 1
            if b.ref_cnt == 0:
                self.free += 1


class _GuardKVM:
    """KVCacheManager stand-in with v0.31.0's two allocate_slots refusals."""

    def __init__(self, free: int, need: int):
        self.pool = _GuardPool(free)
        self.need = need
        self.mamba_mgr = SimpleNamespace(block_size=640, cached_blocks_this_step=set())
        groups = [
            SimpleNamespace(spec=_FullAttentionSpec(), group_ids=[0]),
            SimpleNamespace(spec=_MambaSpec(), group_ids=[1]),
        ]
        self.coordinator = SimpleNamespace(
            attention_groups=groups,
            single_type_managers=[SimpleNamespace(block_size=640), self.mamba_mgr],
            block_pool=self.pool,
            get_num_blocks_to_allocate=self._need,
        )
        self.enable_caching = True
        self.watermark_blocks = 0
        self.calls = 0

    def _need(self, **kwargs) -> int:
        blocks = kwargs.get("new_computed_blocks") or ((), ())
        mamba = blocks[1] if len(blocks) > 1 else ()
        if mamba and mamba[-1].block_hash in self.mamba_mgr.cached_blocks_this_step:
            return self.pool.num_gpu_blocks + 1
        return self.need

    def allocate_slots(
        self,
        request,
        num_new_tokens,
        new_computed_blocks=None,
        delay_cache_blocks=False,
        reserved_blocks=0,
        has_scheduled_reqs=True,
        **kwargs,
    ):
        self.calls += 1
        need = self.coordinator.get_num_blocks_to_allocate(
            new_computed_blocks=getattr(new_computed_blocks, "blocks", None)
        )
        if need > self.pool.get_num_free_blocks() - reserved_blocks:
            return None
        self.pool.free -= need
        return "blocks"


def _guarded(free: int, need: int, n_convs: int = 10, blocks_per_conv: int = 5):
    kvm = _GuardKVM(free, need)
    ret = HybridStateRetention(kvm, max_conversations=256)
    assert ret.enabled
    for i in range(n_convs):
        blocks = [
            SimpleNamespace(ref_cnt=1, is_null=False) for _ in range(blocks_per_conv)
        ]
        ret._conv_lru[("c", i)] = _ConvLRUEntry(
            conv_key=("c", i),
            req_id=f"r{i}",
            first_hash=None,
            boundary_idx=0,
            boundary_hash=None,
            blocks=blocks,
            hit=640,
        )
    return kvm, ret


def _waiting_req(request_id: str = "w") -> SimpleNamespace:
    return SimpleNamespace(request_id=request_id, status=RequestStatus.WAITING)


def test_retention_guard_keeps_lru_when_refused_by_inflight_reservation():
    kvm, ret = _guarded(free=7000, need=350)
    ret.begin_step()
    res = kvm.allocate_slots(
        _waiting_req(), 0, delay_cache_blocks=True, reserved_blocks=6900
    )
    assert res is None
    assert len(ret._conv_lru) == 10
    assert ret.alloc_none["inflight_reserve"] == 1
    assert ret.alloc_none_evicted["no_blocks"] == 0
    assert ret.step_waiting_none == ["inflight_reserve"]
    assert kvm.calls == 1


def test_retention_guard_evicts_only_the_shortfall():
    # 112 needed, 100 free: three 5-block conversations cover the 12 missing.
    kvm, ret = _guarded(free=100, need=112)
    ret.begin_step()
    assert kvm.allocate_slots(_waiting_req(), 640) == "blocks"
    assert len(ret._conv_lru) == 7
    assert ret.alloc_none_evicted["no_blocks"] == 3
    assert sum(ret.alloc_none.values()) == 0
    assert ret.step_admits == 1 and ret.step_async_admits == 0


def test_retention_guard_shortfall_then_reservation_stops_after_shortfall():
    kvm, ret = _guarded(free=100, need=112)
    ret.begin_step()
    res = kvm.allocate_slots(
        _waiting_req(), 0, delay_cache_blocks=True, reserved_blocks=50
    )
    assert res is None
    assert len(ret._conv_lru) == 7
    assert ret.alloc_none_evicted["no_blocks"] == 3
    assert ret.alloc_none["inflight_reserve"] == 1


def test_retention_guard_ignores_same_step_mamba_refusal():
    kvm, ret = _guarded(free=7000, need=10)
    mamba_block = SimpleNamespace(block_hash="h-mamba")
    kvm.mamba_mgr.cached_blocks_this_step.add("h-mamba")
    attn_block = SimpleNamespace(block_hash="h-attn")
    hit = SimpleNamespace(blocks=((attn_block,), (mamba_block,)))
    ret.begin_step()
    assert kvm.allocate_slots(_waiting_req(), 640, new_computed_blocks=hit) is None
    assert len(ret._conv_lru) == 10
    assert ret.alloc_none["same_step"] == 1
    assert ret.step_waiting_none == ["same_step"]


def test_retention_guard_counts_async_admission_and_running_refusals():
    kvm, ret = _guarded(free=5, need=3, n_convs=0)
    ret.begin_step()
    assert kvm.allocate_slots(_waiting_req(), 0, delay_cache_blocks=True) == "blocks"
    assert ret.step_admits == 1 and ret.step_async_admits == 1
    running = SimpleNamespace(request_id="r", status=RequestStatus.RUNNING)
    assert kvm.allocate_slots(running, 640) is None
    assert ret.alloc_none["no_blocks"] == 1
    # A running request's refusal does not end the waiting loop.
    assert ret.step_waiting_none == []


def test_waiting_loop_stop_reason_and_admission_telemetry():
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.running = []
    s.waiting = [object()]
    s.kv_holding_waiting = []
    s.max_num_scheduled_tokens = 8192
    s.max_num_active_reqs = 32
    s.block_size = 640
    out = SimpleNamespace(total_num_scheduled_tokens=640, num_scheduled_tokens={})

    s._step_wait_event = None
    s._retention = SimpleNamespace(
        step_waiting_none=["inflight_reserve"],
        step_admits=0,
        step_async_admits=0,
        alloc_none={"inflight_reserve": 7, "no_blocks": 0},
    )
    assert s._waiting_loop_stop_reason(out, 1) == "alloc:inflight_reserve"

    # The first stop of a waiting request wins; running requests are ignored.
    s._note_waiting_stop(SimpleNamespace(status=RequestStatus.RUNNING), "align0")
    assert s._step_wait_event is None
    s._note_waiting_stop(SimpleNamespace(status=RequestStatus.WAITING), "cap0:cold_cap")
    s._note_waiting_stop(SimpleNamespace(status=RequestStatus.WAITING), "align0")
    assert s._waiting_loop_stop_reason(out, 1) == "cap0:cold_cap"

    s._step_wait_event = None
    s._retention.step_waiting_none = []
    assert s._waiting_loop_stop_reason(out, 1) == "other"
    full = SimpleNamespace(total_num_scheduled_tokens=8192, num_scheduled_tokens={})
    assert s._waiting_loop_stop_reason(full, 1) == "budget"
    # A skipped request back in `waiting` is deferred, not unreached.
    s.deferred_waiting = set(s.waiting)
    assert s._waiting_loop_stop_reason(out, 1) == "drained"
    s.deferred_waiting = set()
    s.running = [object()] * 32
    assert s._waiting_loop_stop_reason(out, 1) == "max_running"
    s.running = []
    s.waiting = []
    assert s._waiting_loop_stop_reason(out, 0) == "idle"
    s.kv_holding_waiting = [object()]
    assert s._waiting_loop_stop_reason(out, 0) == "drained"
    s.kv_holding_waiting = []

    s.waiting = [object()]
    s._step_cold_exempt_ids = {1}
    s._record_step_telemetry(out, 3, "alloc:inflight_reserve")
    s._step_cold_exempt_ids = set()
    s._record_step_telemetry(out, 3, "alloc:inflight_reserve")
    s._record_step_telemetry(full, 0, "idle")
    # Every waiting request got in: a short step is not a small step then.
    s.waiting = []
    s._record_step_telemetry(out, 2, "drained")
    s._step_warm_waiting_tokens = 7552
    s._step_max_cold_budget = 640
    s._step_cold_release = False
    s._step_cold_exempt_ids = set()
    text = s._admission_telemetry("alloc:inflight_reserve")
    assert "warm_waiting_toks=7552 max_cold_budget=640 cold_release=0" in text
    assert "stop=alloc:inflight_reserve" in text
    assert "min_steps=4 min_small_steps_waiting=2 min_exempt_steps=1" in text
    assert "min_alloc_none=inflight_reserve:7,no_blocks:0" in text
    assert "min_stops=alloc:inflight_reserve:2,drained:1" in text
    # The per-minute counters restart after each telemetry line.
    text = s._admission_telemetry("idle")
    assert "min_steps=0" in text and "min_alloc_none=inflight_reserve:0" in text
