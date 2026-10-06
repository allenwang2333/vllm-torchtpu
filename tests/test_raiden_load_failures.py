# SPDX-License-Identifier: Apache-2.0
"""Decode-side Raiden KV load error output contracts."""

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import FinishReason

from vllm_torchtpu.core.raiden_load_failures import _DecodeLoadFailures


@pytest.fixture(params=[Scheduler, AsyncScheduler])
def scheduler_cls(request):
    return request.param


def _lifecycle_scheduler(scheduler_cls):
    """Keep scheduling/finish/free code real; replace only cache/device services."""
    from unittest.mock import MagicMock

    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue
    from vllm.v1.request import Request, RequestStatus

    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.recompute_kv_load_failures = False
    scheduler.requests = {}
    for req_id in ("bad", "good"):
        req = Request(req_id, [1, 2, 3], SamplingParams(max_tokens=8), None)
        req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        scheduler.requests[req_id] = req
    scheduler.running = []
    scheduler.waiting = FCFSRequestQueue()
    scheduler.skipped_waiting = FCFSRequestQueue(scheduler.requests.values())
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler._inflight_prefills = set()
    scheduler.grammar_compile_error_reqs = []
    scheduler.defer_block_free = False
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.encoder_cache_manager = MagicMock()
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.take_events.return_value = None
    # Equal block IDs must not turn an identity-scoped error into two failures.
    scheduler.kv_cache_manager.get_block_ids.return_value = ([41],)
    scheduler.perf_metrics = None
    scheduler.make_stats = lambda *args: None
    _DecodeLoadFailures(scheduler)
    return scheduler


def _lifecycle_step(
    scheduler, failures=(), finished=(), *, wrapped=False, scheduled=None
):
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiKVConnectorWorkerMetadata,
    )
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

    from vllm_torchtpu.distributed.kv_transfer.raiden_load_failures import (
        RaidenLoadFailures,
    )

    metadata = RaidenLoadFailures(set(failures)) if failures else None
    if wrapped:
        metadata = MultiKVConnectorWorkerMetadata([None, metadata])
    output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=KVConnectorOutput(
            finished_recving=set(finished), kv_connector_worker_meta=metadata
        ),
    )
    return scheduler.update_from_output(
        SimpleNamespace(
            num_scheduled_tokens=scheduled or {},
            total_num_scheduled_tokens=sum((scheduled or {}).values()),
            scheduled_spec_decode_tokens={},
        ),
        output,
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_load_error_finishes_request_but_waits_for_receive_barrier(
    scheduler_cls, wrapped
):
    from vllm.v1.request import RequestStatus

    scheduler = _lifecycle_scheduler(scheduler_cls)
    bad = scheduler.requests["bad"]
    result = _lifecycle_step(scheduler, ["bad"], wrapped=wrapped)
    (terminal,) = result[bad.client_index].outputs
    assert terminal.new_token_ids == []
    assert terminal.finish_reason is FinishReason.ERROR
    assert (
        terminal.kv_transfer_params["kv_transfer_error"]["status_name"]
        == "model.KVTDecodeLoadFailure"
    )
    assert bad.status == RequestStatus.FINISHED_ERROR
    assert set(scheduler.requests) == {"bad", "good"}
    scheduler.kv_cache_manager.free.assert_not_called()
    assert [r.request_id for r in scheduler.skipped_waiting] == ["good"]

    # Another rank may repeat a failure before all receive completions arrive.
    assert _lifecycle_step(scheduler, ["bad"]) == {}
    assert _lifecycle_step(scheduler, finished=["bad", "good"]) == {}
    scheduler.kv_cache_manager.free.assert_called_once_with(bad)
    assert set(scheduler.requests) == {"good"}
    assert scheduler.finished_recving_kv_req_ids == {"good"}
    assert scheduler.requests["good"].status == RequestStatus.WAITING_FOR_REMOTE_KVS
    # Later metadata cannot resurrect a completed request or emit another error.
    assert _lifecycle_step(scheduler, ["bad"]) == {}


def test_cancelled_load_failure_only_releases_buffers(scheduler_cls):
    from vllm.v1.request import RequestStatus

    scheduler = _lifecycle_scheduler(scheduler_cls)
    bad = scheduler.requests["bad"]
    scheduler.finish_requests("bad", RequestStatus.FINISHED_ABORTED)
    scheduler.kv_cache_manager.free.assert_not_called()
    assert _lifecycle_step(scheduler, ["bad"], ["bad"]) == {}
    scheduler.kv_cache_manager.free.assert_called_once_with(bad)
    assert bad.status == RequestStatus.FINISHED_ABORTED


def test_failed_load_discards_inflight_tokens(scheduler_cls):
    scheduler = _lifecycle_scheduler(scheduler_cls)
    bad = scheduler.requests["bad"]
    bad.num_in_flight_tokens = 1
    result = _lifecycle_step(scheduler, ["bad"], ["bad"], scheduled={"bad": 1})
    (terminal,) = result[bad.client_index].outputs
    assert terminal.new_token_ids == []
    assert list(bad.output_token_ids) == []
    assert bad.num_in_flight_tokens == 0
    scheduler.kv_cache_manager.cache_blocks.assert_not_called()


@pytest.mark.parametrize(
    "role,policy,transport,inline,connector,enabled",
    [
        ("kv_consumer", "fail", "raiden", False, "TPURaidenConnector", True),
        ("kv_consumer", "fail", "raiden", False, "TPUConnector", True),
        ("kv_consumer", "fail", "raiden", False, "TPUMultiConnector", True),
        ("kv_consumer", "recompute", "raiden", False, "TPURaidenConnector", False),
        ("kv_consumer", "fail", "raiden", True, "TPURaidenConnector", False),
        ("kv_consumer", "fail", "zmq", False, "TPURaidenConnector", False),
        ("kv_producer", "fail", "raiden", False, "TPURaidenConnector", False),
        ("kv_consumer", "fail", "raiden", False, "ExampleConnector", False),
    ],
)
def test_feedback_is_limited_to_non_inline_raiden_fail_policy(
    role, policy, transport, inline, connector, enabled, monkeypatch
):
    from vllm_torchtpu.distributed.kv_transfer import raiden_load_failures as rlf

    monkeypatch.setattr(rlf.envs, "TPU_KV_RESHARD_TRANSPORT", transport)
    monkeypatch.setattr(rlf.dist_utils, "get_raiden_inline_load", lambda: inline)
    cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role=role,
            kv_load_failure_policy=policy,
            kv_connector=connector,
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "TPURaidenConnector"},
                    {"kv_connector": "TPURaidenOffloadingConnector"},
                ]
            },
        )
    )
    assert rlf.raiden_load_failure_feedback_enabled(cfg) is enabled


def test_failure_metadata_aggregates_request_identity():
    from vllm_torchtpu.distributed.kv_transfer.raiden_load_failures import (
        RaidenLoadFailures,
    )

    failures = RaidenLoadFailures({"bad"})
    assert failures.aggregate(RaidenLoadFailures({"bad", "other"})) is failures
    assert failures.failed_req_ids == {"bad", "other"}


@pytest.mark.parametrize("enabled", [False, True])
def test_scheduler_lifecycle_installs_only_selected_feedback(enabled, monkeypatch):
    from vllm_torchtpu.core import raiden_load_failures as rlf

    def base_update(*_args):
        return {}

    def base_init(self, config):
        self.update_from_output = base_update

    monkeypatch.setattr(Scheduler, "__init__", base_init)
    monkeypatch.setattr(
        Scheduler, "_tpu_raiden_load_failures_patch", False, raising=False
    )
    monkeypatch.setattr(rlf, "raiden_load_failure_feedback_enabled", lambda _: enabled)
    rlf.patch_scheduler_init()
    first_init = Scheduler.__init__
    rlf.patch_scheduler_init()
    assert Scheduler.__init__ is first_init
    scheduler = Scheduler.__new__(Scheduler)
    Scheduler.__init__(scheduler, object())
    assert (scheduler.update_from_output is base_update) is (not enabled)
