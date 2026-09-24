# SPDX-License-Identifier: Apache-2.0
"""Stage 6.5: shipping a chunked prefill as one transfer per chunk.

A 64K prompt does not arrive at the producer all at once. vLLM computes it a
scheduler step at a time, and until stage 6.5 the KV for all of it waited for
the last step before any of it moved. Here each chunk ships as soon as its
forward has run, so chunk one is crossing the wire while chunk two computes.

The tests are grouped by the thing that can go wrong. Decode has to know how
many transfers to wait for before the first one lands, so the admission
announcement has to be right. The producer must not ship a chunk before the
forward that wrote its pages, so the emission has to be one step late. Raiden
places bytes by destination page, so the page window a chunk names has to be
sliced and its skip page-aligned. And vLLM must not be told a request's send
is finished until every one of its chunks is, or it frees pages still being
read.

Stage 7 cuts the same request along the other axis, by layer instead of by
token, and its tests are here too. They reuse this harness, and the two splits
have to compose: a chunked prompt with the layer split on sends chunks, then
the early layers, then the rest.

Helpers come from the sibling module rather than being duplicated; this file
is separate only because that one is already five thousand lines.
"""
import json
import os
import time
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.request import RequestStatus

from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    SendMeta, TPUConnectorMetadata)

from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (  # isort: skip
    _chunk_base_req_id, _chunk_req_id, _DestinationCollector,
    _early_group_bells, _early_layer_split_enabled, _fa_group_cuts,
    _fa_group_ordinals, _flatten_block_ids, _send_meta_fa_group,
    _send_meta_int, _Stage3Admission, _STAGE7_ARM_DEADLINE_S,
    _maybe_strand_stage7_watch_for_test)

from .test_tpu_connector import (  # isort: skip
    _FakeRaidenControllerFacade, _FakeRaidenEngine, _MOD,
    _make_raiden_scheduler, _make_raiden_worker, _RecordingNotifier,
    _stage3_producer_env)

# The deployed 397B prefill: 8192-token pages, PCP8, a 32768-token per-step
# budget. One scheduler block is 65536 tokens, so a 64K prompt is a single
# block per rank and the chunk boundary falls inside it.
PAGE = 8192
PCP = 8
BUDGET = 32768
# The producer transfers every token but the last, so a prompt of PROMPT
# tokens ships SENT of them and every boundary below is measured against
# SENT. Picking a prompt one past a round number keeps the arithmetic
# readable: 65537 tokens in, 65536 on the wire, two chunks of 32768.
SENT = 65536
PROMPT = SENT + 1


@contextmanager
def _chunk_split_on(enabled: bool = True):
    """Both flags, because the split only works producer-initiated."""
    with patch(f"{_MOD}.tpu_envs.TPU_KV_CHUNK_SPLIT", enabled, create=True), \
         patch(f"{_MOD}.tpu_envs.TPU_KV_PREFILL_SUBMIT", True, create=True):
        yield


def _producer(*, budget: int = BUDGET, page: int = PAGE):
    producer = _make_raiden_scheduler(is_producer=True,
                                      block_size=page,
                                      pcp_size=PCP)
    producer.vllm_config.scheduler_config.max_num_batched_tokens = budget
    notifier = _RecordingNotifier()
    producer._admission_notifier = notifier
    return producer, notifier


def _request(req_id: str, prompt_tokens: int, dst_batch_url: str = ""):
    req = MagicMock()
    req.request_id = req_id
    req.prompt_token_ids = [0] * prompt_tokens
    req.num_prompt_tokens = prompt_tokens
    req.num_computed_tokens = prompt_tokens
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {
        "admit_callback_url": f"http://proxy.test:8000/kv_admit/{req_id}"
    }
    # Stage 7d. Present in a real deployment; left out by default so the
    # tests that predate the worker-side collector still describe a producer
    # with nowhere to wait for a late destination.
    if dst_batch_url:
        req.kv_transfer_params["dst_batch_url"] = dst_batch_url
    return req


def _step(*, new=(), cached=(), scheduled=None):
    """A scheduler output carrying only what _collect_chunk_sends reads.

    ``new`` is (req_id, block_ids) for requests scheduled for the first time,
    ``cached`` is (req_id, newly_allocated_block_ids) for the ones continuing.
    """
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=r, block_ids=([list(b)])) for r, b in new
        ],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[r for r, _ in cached],
                                              new_block_ids=[[list(b)]
                                                             for _, b in cached
                                                             ]),
        num_scheduled_tokens=dict(scheduled or {}),
    )


def _published_destination(uuid: int, pages: list[int], *, skip: int = 0):
    """What decode's announcement looks like once the collector has it."""
    return {
        "uuid": uuid,
        "dst_controller_address": "decode-controller.test:27000",
        "dst_job_name": "decode-job",
        "dst_engine_id": "consumer-engine-2",
        "dst_data_replica_idx": 0,
        "dst_block_ids": list(pages),
        "dst_mamba_state_block_ids": [],
        "dst_skip_tokens": skip,
    }


def _give_destination(producer, req_id: str, entry: dict):
    collector = MagicMock()
    collector.take.side_effect = lambda r: entry if r == req_id else None
    collector.peek.side_effect = lambda r: entry if r == req_id else None
    producer._destination_collector = collector
    return collector


# ---------------------------------------------------------------------------
# Derived request ids
# ---------------------------------------------------------------------------


class TestChunkRequestIds:
    """Each chunk travels under an id vLLM has never heard of, so that every
    req_id-keyed structure keeps treating one transfer as one request."""

    def test_round_trips(self):
        assert _chunk_req_id("abc", 0) == "abc#c0"
        assert _chunk_base_req_id("abc#c0") == "abc"
        assert _chunk_base_req_id(_chunk_req_id("abc", 13)) == "abc"

    def test_leaves_an_ordinary_id_alone(self):
        for req_id in ("abc", "a#cb", "a#c", "a#c1x", ""):
            assert _chunk_base_req_id(req_id) == req_id

    def test_survives_a_request_id_that_already_looks_derived(self):
        """vLLM ids are arbitrary strings, so one can collide with the suffix
        shape. The rpartition takes the last separator, so the base is what
        this connector appended to, not what the caller happened to name."""
        assert _chunk_base_req_id(_chunk_req_id("job#c7", 2)) == "job#c7"


class TestFlattenBlockIds:
    """Scheduler output gives block ids one list per KV cache group, and the
    two paths that carry them are shaped differently."""

    def test_flattens_the_grouped_form(self):
        assert _flatten_block_ids([[1, 2], [3]]) == [1, 2, 3]

    def test_accepts_a_bare_list(self):
        assert _flatten_block_ids([4, 5]) == [4, 5]

    def test_none_and_empty_are_empty(self):
        assert _flatten_block_ids(None) == []
        assert _flatten_block_ids([]) == []


class TestSendMetaInt:
    """The legacy send paths pass metadata types that never heard of these
    fields, and tests pass MagicMocks whose int() is 1. Either one read as a
    token offset would put a chunk boundary one token into the prompt."""

    def test_reads_a_real_int(self):
        assert _send_meta_int(SimpleNamespace(start_token=4096),
                              "start_token") == 4096

    def test_a_missing_field_is_zero(self):
        assert _send_meta_int(SimpleNamespace(), "start_token") == 0

    def test_a_mock_is_zero_rather_than_one(self):
        assert _send_meta_int(MagicMock(), "start_token") == 0


# ---------------------------------------------------------------------------
# The admission announcement
# ---------------------------------------------------------------------------


class TestChunkTokens:

    def test_off_by_default(self):
        producer, _ = _producer()
        assert producer._chunk_tokens() == 0

    def test_is_the_budget_rounded_down_to_a_page(self):
        """The budget is what makes a boundary free, because an uncontended
        prefill chunk already ends there. The rounding is what makes it
        expressible, because the destination clip moves in whole pages."""
        producer, _ = _producer(budget=30000, page=PAGE)
        with _chunk_split_on():
            assert producer._chunk_tokens() == 24576

    def test_a_budget_below_one_page_disables_the_split(self):
        producer, _ = _producer(budget=4096, page=PAGE)
        with _chunk_split_on():
            assert producer._chunk_tokens() == 0

    def test_needs_the_producer_submit_path_too(self):
        """The consumer starts its transfer when it is ready, which for a
        chunk is before the pages exist."""
        producer, _ = _producer()
        with patch(f"{_MOD}.tpu_envs.TPU_KV_CHUNK_SPLIT", True, create=True), \
             patch(f"{_MOD}.tpu_envs.TPU_KV_PREFILL_SUBMIT", False,
                   create=True):
            assert producer._chunk_tokens() == 0


class TestAnnouncement:

    def _announce(self, producer, req, uuids):
        with ExitStack() as stack:
            for ctx in _stage3_producer_env(uuids[0]):
                stack.enter_context(ctx)
            stack.enter_context(
                patch(f"{_MOD}.get_uuid", side_effect=list(uuids)))
            producer.update_state_after_alloc(req, MagicMock(), 0)

    def test_a_prompt_inside_one_chunk_announces_exactly_what_stage_6_did(
            self):
        """The wire format has to be unchanged for the common case, or every
        short request pays for a feature it does not use."""
        producer, notifier = _producer()
        req = _request("short", 8192)
        with _chunk_split_on():
            self._announce(producer, req, [900])

        _url, announced = notifier.posts[0]
        assert "chunk_uuids" not in announced
        assert "chunk_tokens" not in announced
        assert producer._chunk_progress == {}

    def test_a_two_chunk_prompt_announces_one_leading_uuid(self):
        producer, notifier = _producer()
        req = _request("long", PROMPT)
        with _chunk_split_on():
            self._announce(producer, req, [900, 901])

        _url, announced = notifier.posts[0]
        assert announced["uuid"] == 900
        assert announced["chunk_uuids"] == [901]
        assert announced["chunk_tokens"] == BUDGET
        assert announced["num_tokens"] == SENT

    def test_the_count_is_ceiling_not_floor(self):
        """A prompt past a boundary is two chunks, and decode sizing its wait
        by the floor would stop one transfer early. A page of tail is the
        smallest one that survives the sliver rule below."""
        producer, notifier = _producer()
        req = _request("edge", BUDGET + PAGE + 1)
        with _chunk_split_on():
            self._announce(producer, req, [900, 901])

        assert notifier.posts[0][1]["chunk_uuids"] == [901]

    def test_a_tail_shorter_than_a_page_is_not_worth_a_boundary(self):
        """A chunk hides in the compute that follows it, so one with a sliver
        of a tail hides nothing, and it still commits decode to a second
        arrival the producer cannot be sure it can send. One of these took
        down the prefill engine in long_32768_c8."""
        producer, notifier = _producer()
        req = _request("sliver", BUDGET + 2)
        with _chunk_split_on():
            self._announce(producer, req, [900])

        assert "chunk_uuids" not in notifier.posts[0][1]
        assert producer._chunk_progress == {}

    def test_the_sliver_rule_gives_up_one_boundary_and_no_more(self):
        """Folding the sliver into the chunk before it leaves that chunk over
        a page long, so there is never a second boundary to drop."""
        producer, notifier = _producer()
        req = _request("nearly_four", BUDGET * 3 + 2)
        with _chunk_split_on():
            self._announce(producer, req, [900, 901, 902])

        assert notifier.posts[0][1]["chunk_uuids"] == [901, 902]

    def test_a_prompt_exactly_one_chunk_long_is_not_chunked(self):
        producer, notifier = _producer()
        req = _request("exact", BUDGET + 1)
        with _chunk_split_on():
            self._announce(producer, req, [900])

        assert "chunk_uuids" not in notifier.posts[0][1]

    def test_four_chunks_get_three_leading_uuids(self):
        producer, notifier = _producer()
        req = _request("huge", BUDGET * 4 + 1)
        with _chunk_split_on():
            self._announce(producer, req, [900, 901, 902, 903])

        assert notifier.posts[0][1]["chunk_uuids"] == [901, 902, 903]

    def test_with_the_split_off_nothing_is_announced_about_chunks(self):
        producer, notifier = _producer()
        req = _request("off", PROMPT)
        with _chunk_split_on(False):
            self._announce(producer, req, [900])

        assert "chunk_uuids" not in notifier.posts[0][1]
        assert producer._chunk_progress == {}


class TestAdmissionBoundaries:
    """_Stage3Admission is the only place that says where a chunk starts and
    ends, and both sides derive everything else from it."""

    def test_the_chunks_tile_the_prompt_without_gap_or_overlap(self):
        admission = _Stage3Admission(uuid=900,
                                     num_tokens=70000,
                                     chunk_tokens=BUDGET,
                                     chunk_uuids=(901, 902))
        bounds = [(admission.chunk_start(i), admission.chunk_end(i))
                  for i in range(3)]
        assert bounds == [(0, 32768), (32768, 65536), (65536, 70000)]

    def test_the_final_chunk_is_clamped_to_the_prompt(self):
        admission = _Stage3Admission(uuid=900,
                                     num_tokens=40000,
                                     chunk_tokens=BUDGET,
                                     chunk_uuids=(901, ))
        assert admission.chunk_end(1) == 40000

    def test_an_unchunked_admission_is_one_whole_chunk(self):
        admission = _Stage3Admission(uuid=900, num_tokens=8192)
        assert admission.chunk_start(0) == 0
        assert admission.chunk_end(0) == 8192


# ---------------------------------------------------------------------------
# Emitting the chunks
# ---------------------------------------------------------------------------


class TestChunkSends:

    def _admitted(self, prompt_tokens=PROMPT, budget=BUDGET):
        producer, _notifier = _producer(budget=budget)
        req = _request("long", prompt_tokens)
        with _chunk_split_on():
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                stack.enter_context(
                    patch(f"{_MOD}.get_uuid", side_effect=[900, 901, 902,
                                                           903]))
                producer.update_state_after_alloc(req, MagicMock(), 0)
        _give_destination(producer, "long",
                          _published_destination(900, list(range(200, 208))))
        return producer, req

    def _build(self, producer, step):
        with _chunk_split_on():
            with patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
                       return_value=30.0):
                return producer.build_connector_meta(step)

    def test_a_chunk_ships_the_step_after_the_one_that_computed_it(self):
        """The worker reads this metadata before the step's forward, so a
        chunk announced in the step that computed it would have its transfer
        start against pages that forward had not written."""
        producer, _req = self._admitted()

        meta = self._build(
            producer, _step(new=[("long", [100])], scheduled={"long": BUDGET}))
        assert meta.reqs_to_send == {}

        meta = self._build(
            producer,
            _step(cached=[("long", [101])], scheduled={"long": BUDGET}))
        assert set(meta.reqs_to_send) == {"long#c0"}

    def test_the_chunk_covers_only_its_own_window(self):
        producer, _req = self._admitted()
        self._build(producer,
                    _step(new=[("long", [100])], scheduled={"long": BUDGET}))
        meta = self._build(
            producer,
            _step(cached=[("long", [101])], scheduled={"long": BUDGET}))

        send = meta.reqs_to_send["long#c0"]
        assert send.uuid == 901
        assert send.start_token == 0
        assert send.num_tokens == BUDGET
        assert send.chunk_index == 0
        assert send.is_final_chunk is False
        # One 65536-token scheduler block covers the 32768-token window, so
        # the chunk names the one block it reaches and not the second.
        assert send.local_block_ids == [100]

    def test_the_final_send_picks_up_where_the_chunks_stopped(self):
        producer, req = self._admitted()
        self._build(producer,
                    _step(new=[("long", [100])], scheduled={"long": BUDGET}))
        self._build(
            producer,
            _step(cached=[("long", [101])], scheduled={"long": BUDGET}))

        with _chunk_split_on():
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                delay, _params = producer.request_finished(req, [100, 101])

        assert delay
        final = producer.reqs_to_send["long"]
        assert final.uuid == 900
        assert final.start_token == BUDGET
        assert final.num_tokens == SENT
        assert final.is_final_chunk is True
        # A 65536-token prompt is exactly one 65536-token scheduler block at
        # PCP8, so both sends name the same single block and it is the source
        # byte spans, not the block list, that tells them apart.
        assert final.local_block_ids == [100]

    def test_the_chunk_and_the_final_send_partition_the_prompt(self):
        """This is the whole correctness claim in one assertion: every token
        is covered once. Overlap wastes bandwidth, a gap is corruption.

        The sends are gathered from each step's metadata rather than read off
        the scheduler at the end, because building the metadata is what
        drains them.
        """
        producer, req = self._admitted()
        sends = []
        for step in (_step(new=[("long", [100])], scheduled={"long": BUDGET}),
                     _step(cached=[("long", [101])],
                           scheduled={"long": BUDGET})):
            sends += self._build(producer, step).reqs_to_send.values()
        with _chunk_split_on():
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                producer.request_finished(req, [100, 101])
        sends += producer.reqs_to_send.values()

        windows = sorted((s.start_token, s.num_tokens) for s in sends)
        assert windows == [(0, BUDGET), (BUDGET, SENT)]

    def test_a_chunk_reads_the_destination_without_consuming_it(self):
        """Decode publishes one announcement for the whole request, so a
        chunk that took it would leave the final send unable to start."""
        producer, _req = self._admitted()
        collector = producer._destination_collector
        self._build(producer,
                    _step(new=[("long", [100])], scheduled={"long": BUDGET}))
        self._build(
            producer,
            _step(cached=[("long", [101])], scheduled={"long": BUDGET}))

        collector.take.assert_not_called()
        assert collector.peek.call_count == 1

    def test_a_chunk_matches_the_admission_uuid_not_its_own(self):
        """Decode's announcement names the uuid it was admitted under. A
        chunk checking its own uuid would reject every announcement and fall
        back to a transfer nobody starts."""
        producer, _req = self._admitted()
        self._build(producer,
                    _step(new=[("long", [100])], scheduled={"long": BUDGET}))
        meta = self._build(
            producer,
            _step(cached=[("long", [101])], scheduled={"long": BUDGET}))

        assert meta.reqs_to_send["long#c0"].dst_params is not None

    def test_a_chunk_waits_for_pages_decode_has_not_published_yet(self):
        producer, _req = self._admitted()
        producer._destination_collector = None

        self._build(producer,
                    _step(new=[("long", [100])], scheduled={"long": BUDGET}))
        meta = self._build(producer, _step(scheduled={}))
        assert meta.reqs_to_send == {}
        assert producer._chunk_progress["long"].pending is not None

    def test_a_chunk_that_waits_too_long_fails_loudly(self):
        """Decode counts arrivals against the number announced, so a chunk
        that silently never ships hangs the request instead."""
        producer, _req = self._admitted()
        producer._destination_collector = None

        self._build(producer,
                    _step(new=[("long", [100])], scheduled={"long": BUDGET}))
        with pytest.raises(ValueError, match="has waited"):
            for _ in range(producer._CHUNK_DST_WAIT_STEPS + 2):
                self._build(producer, _step(scheduled={}))

    def test_a_step_that_overshoots_a_boundary_still_ships_one_chunk(self):
        """The boundaries are fixed at admission, not read off what the
        scheduler did, because decode was told the count before the first
        step ran. Overshooting costs overlap, never a transfer."""
        producer, _req = self._admitted()

        self._build(
            producer,
            _step(new=[("long", [100])], scheduled={"long": BUDGET + 8192}))
        meta = self._build(
            producer, _step(cached=[("long", [101])],
                            scheduled={"long": 24576}))
        assert set(meta.reqs_to_send) == {"long#c0"}
        assert meta.reqs_to_send["long#c0"].num_tokens == BUDGET

    def test_a_chunk_pending_when_the_prefill_ends_is_flushed(self):
        """A chunk held back because decode had not published yet is still
        pending when the prompt ends, and there is no next step to carry it.
        request_finished has to emit it, or decode waits forever for an
        arrival that was announced and never sent.

        Since the tail rule, this is the only way to reach the flush. A step
        can no longer both cross a boundary and finish, because a boundary
        leaves at least a page behind it and a step computes at most the
        budget, which is the boundary itself.
        """
        producer, _notifier = _producer()
        req = _request("long", PROMPT)
        with _chunk_split_on():
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                stack.enter_context(
                    patch(f"{_MOD}.get_uuid", side_effect=[900, 901]))
                producer.update_state_after_alloc(req, MagicMock(), 0)

        self._build(producer,
                    _step(new=[("long", [100])], scheduled={"long": BUDGET}))
        # The step that would ship it finds no destination and shelves it.
        self._build(
            producer,
            _step(cached=[("long", [101])], scheduled={"long": SENT - BUDGET}))
        assert producer._chunk_progress["long"].pending is not None

        _give_destination(producer, "long",
                          _published_destination(900, list(range(200, 208))))
        with _chunk_split_on():
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                producer.request_finished(req, [100, 101])

        assert set(producer.reqs_to_send) == {"long#c0", "long"}
        assert producer.reqs_to_send["long#c0"].num_tokens == BUDGET
        assert producer.reqs_to_send["long"].start_token == BUDGET

    def test_an_abort_drops_the_progress(self):
        producer, req = self._admitted()
        self._build(producer,
                    _step(new=[("long", [100])], scheduled={"long": BUDGET}))
        req.status = RequestStatus.FINISHED_ABORTED

        with _chunk_split_on():
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                delay, _params = producer.request_finished(req, [100, 101])

        assert not delay
        assert producer._chunk_progress == {}

    def test_a_block_list_short_of_the_chunk_is_an_error(self):
        """vLLM allocates a chunk's blocks before computing it, so reaching
        here means this connector's block accounting has drifted from the
        scheduler's. Shipping anyway would read whatever is in the pages."""
        producer, _req = self._admitted()
        self._build(producer,
                    _step(new=[("long", [])], scheduled={"long": BUDGET}))
        with pytest.raises(ValueError, match="scheduler blocks"):
            self._build(producer, _step(scheduled={}))

    def test_the_split_off_leaves_build_connector_meta_alone(self):
        """The stage gate. With the flag unset no request is ever tracked, so
        the scheduler output is not even looked at."""
        producer, _notifier = _producer()
        req = _request("long", 65536)
        with _chunk_split_on(False):
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                producer.update_state_after_alloc(req, MagicMock(), 0)
                meta = producer.build_connector_meta(
                    _step(new=[("long", [100])], scheduled={"long": BUDGET}))

        assert meta.reqs_to_send == {}
        assert producer._chunk_progress == {}


# ---------------------------------------------------------------------------
# Placing a chunk's bytes at the destination
# ---------------------------------------------------------------------------


class TestDestinationWindow:
    """Raiden's planner insists every destination block a transfer names is
    covered from its first byte and that only the last may be partial. A
    chunk covers a window in the middle of a request, so it has to name only
    the pages that window touches and declare the rest as skipped."""

    def _worker(self):
        return _make_raiden_worker(is_producer=True, block_size=PAGE)

    def test_the_first_chunk_names_only_the_pages_it_reaches(self):
        worker = self._worker()
        pages = list(range(200, 208))
        got, skip = worker._chunk_dst_window(req_id="r",
                                             dst_block_ids=pages,
                                             start_token=0,
                                             end_token=BUDGET,
                                             local_skip_tokens=0)
        assert got == [200, 201, 202, 203]
        assert skip == 0

    def test_a_later_chunk_skips_the_pages_before_it(self):
        worker = self._worker()
        pages = list(range(200, 208))
        got, skip = worker._chunk_dst_window(req_id="r",
                                             dst_block_ids=pages,
                                             start_token=BUDGET,
                                             end_token=65536,
                                             local_skip_tokens=0)
        assert got == [204, 205, 206, 207]
        assert skip == BUDGET

    def test_a_partial_final_page_is_included(self):
        worker = self._worker()
        pages = list(range(200, 208))
        got, _skip = worker._chunk_dst_window(req_id="r",
                                              dst_block_ids=pages,
                                              start_token=BUDGET,
                                              end_token=BUDGET + 100,
                                              local_skip_tokens=0)
        assert got == [204]

    def test_a_boundary_off_a_page_is_refused(self):
        """The planner takes the skip in whole destination pages, so a
        boundary it cannot express has to be caught here rather than
        producing a plan that writes the bytes somewhere else."""
        worker = self._worker()
        with pytest.raises(ValueError, match="whole destination page"):
            worker._chunk_dst_window(req_id="r",
                                     dst_block_ids=list(range(200, 208)),
                                     start_token=1000,
                                     end_token=65536,
                                     local_skip_tokens=0)

    def test_a_prefix_hit_past_the_chunk_is_refused(self):
        worker = self._worker()
        with pytest.raises(ValueError, match="cached prefix"):
            worker._chunk_dst_window(req_id="r",
                                     dst_block_ids=list(range(200, 208)),
                                     start_token=0,
                                     end_token=BUDGET,
                                     local_skip_tokens=65536)

    def test_too_few_published_pages_is_refused(self):
        """Decode publishing fewer pages than the window needs means the two
        engines disagree about the page size, and the write would land
        outside the request's allocation."""
        worker = self._worker()
        with pytest.raises(ValueError, match="published"):
            worker._chunk_dst_window(req_id="r",
                                     dst_block_ids=[200, 201],
                                     start_token=0,
                                     end_token=65536,
                                     local_skip_tokens=0)


class TestTransferLayout:
    """Recurrent state is not a prefix of itself. Only its value at the end
    of the prompt is the one decode needs, so it rides the final transfer and
    the leading chunks carry attention pages alone."""

    def _worker(self):
        worker = _make_raiden_worker(is_producer=True, block_size=PAGE)
        worker._stage3_state_group_count = 2
        return worker

    def test_a_leading_chunk_carries_no_state_tags(self):
        worker = self._worker()
        fa_tags, *_ = worker._stage3_transfer_layout(
            dst_block_ids=[200, 201],
            mamba_state_block_ids=None,
            skip_tokens=0,
            fa_only=True)
        both_tags, *_ = worker._stage3_transfer_layout(
            dst_block_ids=[200, 201],
            mamba_state_block_ids=[300, 301],
            skip_tokens=0,
            fa_only=False)
        assert len(fa_tags) < len(both_tags)

    def test_a_leading_chunk_needs_no_state_blocks_to_be_laid_out(self):
        """The final send is the one that carries recurrent state, so a
        leading chunk has no state slots to offer and must not be asked."""
        worker = self._worker()
        worker._stage3_transfer_layout(dst_block_ids=[200],
                                       mamba_state_block_ids=None,
                                       skip_tokens=0,
                                       fa_only=True)

    def test_every_attention_group_takes_the_prefix_clip(self):
        """Stage 7 turns one attention tag into two, and both address the same
        destination byte space. Clipping only the first would leave the second
        group writing over the prefix the destination already holds."""
        worker = self._worker()
        worker._stage3_fa_tags = ("fa.g0", "fa.g1")
        worker._stage3_fa_token_bytes = lambda page_tokens: 1024
        tags, _blocks, _counts, clip, skip = worker._stage3_transfer_layout(
            dst_block_ids=[200],
            mamba_state_block_ids=[300, 301],
            skip_tokens=8,
            fa_only=False)
        assert tags[:2] == ["fa.g0", "fa.g1"]
        assert skip == 8 * 1024
        assert clip["dst_skip_bytes"] == [skip, skip] + [0] * (len(tags) - 2)

    def test_one_attention_group_clips_exactly_as_before(self):
        """All-off has to reproduce stage 0 byte for byte, and the clip list is
        part of the wire arguments."""
        worker = self._worker()
        worker._stage3_fa_token_bytes = lambda page_tokens: 1024
        tags, _blocks, _counts, clip, _skip = worker._stage3_transfer_layout(
            dst_block_ids=[200],
            mamba_state_block_ids=[300, 301],
            skip_tokens=8,
            fa_only=False)
        assert tags[0] == "fa"
        assert clip["dst_skip_bytes"] == [8 * 1024] + [0] * (len(tags) - 1)


# ---------------------------------------------------------------------------
# Reporting a chunked send as finished
# ---------------------------------------------------------------------------


class TestProducerTerminals:
    """vLLM frees a request's blocks when it is told the send finished, so
    the base id is withheld until every chunk of it is terminal. Reporting on
    the first chunk would free pages the rest are still reading."""

    def _worker(self):
        return _make_raiden_worker(is_producer=True, block_size=PAGE)

    def test_an_unchunked_request_is_reported_immediately(self):
        worker = self._worker()
        worker._stage3_send_outcomes = {"plain": "done"}
        done, failed, cancelled = (
            worker._stage3_ready_producer_send_terminals())
        assert done == {"plain"}
        assert not failed and not cancelled

    def test_the_base_is_withheld_until_every_chunk_is_terminal(self):
        worker = self._worker()
        worker._chunk_send_parts["long"] = {"long#c0", "long"}

        worker._stage3_send_outcomes = {"long#c0": "done"}
        done, _f, _c = worker._stage3_ready_producer_send_terminals()
        assert done == set()

        worker._stage3_send_outcomes = {"long": "done"}
        done, _f, _c = worker._stage3_ready_producer_send_terminals()
        assert done == {"long"}

    def test_the_derived_ids_are_never_reported_to_vllm(self):
        """vLLM has never heard of them, and an unknown finished_sending id
        trips its bookkeeping."""
        worker = self._worker()
        worker._chunk_send_parts["long"] = {"long#c0", "long"}
        worker._stage3_send_outcomes = {"long#c0": "done", "long": "done"}
        done, failed, cancelled = (
            worker._stage3_ready_producer_send_terminals())
        assert done == {"long"}
        assert not any("#c" in r for r in done | failed | cancelled)

    def test_one_failed_chunk_fails_the_request(self):
        worker = self._worker()
        worker._chunk_send_parts["long"] = {"long#c0", "long"}
        worker._stage3_send_outcomes = {"long#c0": "failed", "long": "done"}
        done, failed, _c = worker._stage3_ready_producer_send_terminals()
        assert failed == {"long"}
        assert done == set()

    def test_all_cancelled_is_cancelled(self):
        worker = self._worker()
        worker._chunk_send_parts["long"] = {"long#c0", "long"}
        worker._stage3_send_outcomes = {
            "long#c0": "cancelled",
            "long": "cancelled"
        }
        _d, _f, cancelled = worker._stage3_ready_producer_send_terminals()
        assert cancelled == {"long"}

    def test_outcomes_arriving_in_separate_passes_are_accumulated(self):
        """The chunks finish at different times, so their outcomes land in
        different polls and the earlier ones have to survive."""
        worker = self._worker()
        worker._chunk_send_parts["long"] = {"long#c0", "long#c1", "long"}
        for outcome in ({"long#c0": "done"}, {"long": "done"}):
            worker._stage3_send_outcomes = outcome
            done, _f, _c = worker._stage3_ready_producer_send_terminals()
            assert done == set()
        worker._stage3_send_outcomes = {"long#c1": "done"}
        done, _f, _c = worker._stage3_ready_producer_send_terminals()
        assert done == {"long"}

    def test_the_bookkeeping_is_dropped_once_reported(self):
        worker = self._worker()
        worker._chunk_send_parts["long"] = {"long#c0", "long"}
        worker._stage3_send_outcomes = {"long#c0": "done", "long": "done"}
        worker._stage3_ready_producer_send_terminals()
        assert worker._chunk_send_parts == {}
        assert worker._chunk_part_outcomes == {}


class TestPartRegistrationRelease:
    """A part's registration is released on that part's own terminal.

    get_finished releases registrations by walking the buckets this function
    returns, and those hold base ids only. So before this, a chunk's row was
    never released: it sat in the source store until the expiry probe
    cancelled it ten minutes later. On the first 397B run all eight ranks
    reached that point in the same second, at the start of conc_8192_c32,
    and every transfer after it timed out.
    """

    def _worker(self):
        return _make_raiden_worker(is_producer=True, block_size=PAGE)

    def test_a_finished_part_is_marked_for_release(self):
        worker = self._worker()
        worker._chunk_send_parts["long"] = {"long#c0", "long"}
        worker._stage3_send_outcomes = {"long#c0": "done"}
        worker._stage3_ready_producer_send_terminals()
        assert worker._stage3_terminal_cleanup["long#c0"] is True

    def test_every_part_including_the_base_is_marked(self):
        worker = self._worker()
        worker._chunk_send_parts["long"] = {"long#c0", "long#c1", "long"}
        worker._stage3_send_outcomes = {
            "long#c0": "done",
            "long#c1": "done",
            "long": "done",
        }
        worker._stage3_ready_producer_send_terminals()
        assert set(
            worker._stage3_terminal_cleanup) == {"long#c0", "long#c1", "long"}

    def test_a_cancelled_part_is_released_without_forcing(self):
        """A force-release retires the controller's D5 row, and an expiry is
        terminal only once the controller has confirmed nobody claimed it.
        The base path makes the same distinction."""
        worker = self._worker()
        worker._chunk_send_parts["long"] = {"long#c0", "long"}
        worker._stage3_send_outcomes = {"long#c0": "cancelled"}
        worker._stage3_ready_producer_send_terminals()
        assert worker._stage3_terminal_cleanup["long#c0"] is False

    def test_a_failed_part_is_still_released(self):
        worker = self._worker()
        worker._chunk_send_parts["long"] = {"long#c0", "long"}
        worker._stage3_send_outcomes = {"long#c0": "failed"}
        worker._stage3_ready_producer_send_terminals()
        assert worker._stage3_terminal_cleanup["long#c0"] is True

    def test_an_unchunked_request_is_left_to_the_caller(self):
        """Its id reaches the buckets, so get_finished releases it the way it
        always did, and marking it here too would be a second owner."""
        worker = self._worker()
        worker._stage3_send_outcomes = {"plain": "done"}
        worker._stage3_ready_producer_send_terminals()
        assert worker._stage3_terminal_cleanup == {}

    def test_a_late_part_terminal_surfaces_under_its_derived_id(self):
        """Pins the precondition that makes get_finished map buckets back to
        base ids. Once the request has aggregated, _chunk_send_parts has
        forgotten the part, so a terminal arriving after that is bucketed
        raw and would otherwise be reported to vLLM under an id it has never
        heard of."""
        worker = self._worker()
        worker._stage3_send_outcomes = {"long#c0": "cancelled"}
        _done, _failed, cancelled = (
            worker._stage3_ready_producer_send_terminals())
        assert cancelled == {"long#c0"}
        assert _chunk_base_req_id("long#c0") == "long"


# ---------------------------------------------------------------------------
# What decode expects
# ---------------------------------------------------------------------------


class TestDecodeExpectations:

    def test_an_unchunked_load_expects_one_arrival(self):
        worker = _make_raiden_worker(is_producer=False, block_size=PAGE)
        assert _send_meta_int(SimpleNamespace(), "num_chunks") == 0
        assert max(1,
                   _send_meta_int(SimpleNamespace(num_chunks=1),
                                  "num_chunks")) == 1
        del worker

    def test_a_chunked_load_expects_one_arrival_per_chunk(self):
        assert max(1,
                   _send_meta_int(SimpleNamespace(num_chunks=3),
                                  "num_chunks")) == 3

    def test_send_meta_defaults_reproduce_the_unchunked_send(self):
        """Every legacy caller builds a SendMeta without these fields, so the
        defaults are what keeps the split invisible when it is off."""
        meta = SendMeta(uuid=1, local_block_ids=[2], expiration_time=0.0)
        assert meta.start_token == 0
        assert meta.chunk_index == 0
        assert meta.is_final_chunk is True


class TestConsumerIdBinding:
    """Decode has one destination per request but several producer ids.

    The producer announces the base id at admission and then ships each chunk
    under a derived one, so the reverse binding is many-to-one. The first
    hardware run died here: binding a chunk through the symmetric helper saw
    the base already bound to the destination and raised, killing the decode
    engine in the same second the producer died on GDN state.
    """

    @staticmethod
    def _consumer():
        return _make_raiden_worker(tp_rank=0,
                                   tp_size=1,
                                   is_producer=False,
                                   dp_size=1,
                                   pcp_size=1)

    def test_chunk_ids_join_the_reverse_map_without_displacing_the_base(self):
        worker = self._consumer()
        worker._bind_stage3_request_ids("dst", "src")
        worker._bind_stage3_chunk_source_id("dst", "src#c0")
        worker._bind_stage3_chunk_source_id("dst", "src#c1")

        # Asked who sent this request, the consumer still answers with the
        # base id; the chunks are an implementation detail of the transfer.
        assert worker._stage3_source_request_id("dst") == "src"
        # Every id an arrival can carry resolves back to the one destination.
        for source in ("src", "src#c0", "src#c1"):
            assert worker._stage3_destination_request_id(source) == "dst"

    def test_rebinding_the_same_chunk_is_idempotent(self):
        """Load metadata is replayed across steps, so this happens normally."""
        worker = self._consumer()
        worker._bind_stage3_request_ids("dst", "src")
        worker._bind_stage3_chunk_source_id("dst", "src#c0")
        worker._bind_stage3_chunk_source_id("dst", "src#c0")
        assert worker._stage3_chunk_source_req_ids["dst"] == {"src#c0"}

    def test_a_chunk_id_claimed_by_another_destination_still_raises(self):
        """The guard is looser than before, not absent."""
        worker = self._consumer()
        worker._bind_stage3_chunk_source_id("dst-a", "src#c0")
        with pytest.raises(ValueError, match="already bound to a different"):
            worker._bind_stage3_chunk_source_id("dst-b", "src#c0")


class TestGdnStateRegistration:
    """A leading chunk must not register GDN state, only full attention.

    This is where stage 6.5 first went wrong on hardware. GDN keeps one
    recurrent state per group that is not correct until the prompt's last
    token has run, so vLLM hands a mid-prompt chunk no state blocks to name.
    The registration path asked for them anyway and raised, which on a
    397B prefill killed all eight workers on the first prompt long enough to
    chunk, and every leg after that failed. The submit path had the rule
    right already; only registration was missing it.
    """

    @staticmethod
    def _gdn_producer():
        """A producer whose model has one GDN state group."""
        worker = _make_raiden_worker(tp_rank=0,
                                     tp_size=1,
                                     is_producer=True,
                                     dp_size=1,
                                     pcp_size=8,
                                     interleave_size=256,
                                     block_size=4096)
        engine = _FakeRaidenEngine()
        engine.poll_results = [([], [], [])]
        facade = _FakeRaidenControllerFacade()
        worker._raiden_transfer_engine = engine
        worker._raiden_controller_facade = facade
        worker._raiden_controller_address = "prefill-controller.test:27000"
        worker._raiden_work_unit = SimpleNamespace(
            job_name="prefill",
            job_replica_id="producer-engine-rank0",
            data_name="kv.fa",
            data_replica_idx=0)
        worker._local_raiden_transfer_rank = MagicMock(return_value=3)
        # FA lowering measures bytes per token from the admitted manifest.
        # 4096 tokens a page at 1024 bytes a token is the geometry the sibling
        # module's rank-stripe test uses, so the spans below are comparable.
        worker._raiden_manifest = SimpleNamespace(
            geometry_by_tag=lambda:
            {"fa": {
                "live_bytes_per_block": 4096 * 1024
            }})
        # What makes this a hybrid model as far as registration is concerned.
        worker._stage3_state_group_count = 1
        return worker, facade

    @staticmethod
    def _meta(*, is_final):
        meta = TPUConnectorMetadata()
        # num_tokens spans two 4096-token pages at PCP8, matching the two
        # scheduler blocks named below; the geometry check runs before the
        # state registration this test is about.
        meta.reqs_to_send["chunked"] = MagicMock(
            uuid=1234,
            local_block_ids=[100, 101],
            num_tokens=65_023,
            expiration_time=1e20,
            mamba_state_block_ids=None,
            is_final_chunk=is_final,
            start_token=0,
            chunk_index=0,
        )
        return meta

    def test_leading_chunk_registers_full_attention_only(self):
        worker, facade = self._gdn_producer()
        with ExitStack() as stack:
            for ctx in _stage3_producer_env(1234):
                stack.enter_context(ctx)
            worker.process_send_load(self._meta(is_final=False))
        assert len(facade.register_request_blocks_calls) == 1
        spans = facade.register_request_blocks_calls[0]["pool_spans"]
        # One entry, the FA stripe. Two would mean a GDN class came along.
        assert len(spans) == 1
        assert spans[0].tag != "gdn.conv.g0"

    def test_final_send_still_demands_its_state_blocks(self):
        """The guard must not have quietly disabled GDN for everyone.

        A final send with no state blocks is a real inconsistency on a hybrid
        model, and it should still raise rather than ship a request missing
        forty-five layers of state.
        """
        worker, _ = self._gdn_producer()
        with ExitStack() as stack:
            for ctx in _stage3_producer_env(1234):
                stack.enter_context(ctx)
            with pytest.raises(RuntimeError, match="mamba state block ids"):
                worker.process_send_load(self._meta(is_final=True))


# ---------------------------------------------------------------------------
# Stage 7: the early attention group as a transfer of its own
# ---------------------------------------------------------------------------


@contextmanager
def _early_layers_on(count: int = 1):
    """Both flags, because only the producer can start the extra transfer."""
    with patch(f"{_MOD}.tpu_envs.TPU_KV_EARLY_LAYERS", count, create=True), \
         patch(f"{_MOD}.tpu_envs.TPU_KV_PREFILL_SUBMIT", True, create=True):
        yield


class TestEarlyLayerGroupSend:
    """One request, two transfers: the early attention layers and the rest.

    Stage 7 splits a request by layer the way stage 6.5 splits it by token.
    Decode is told at admission how many transfers to expect and derives each
    one's id from its position, so the early group rides the announcement as
    one more leading transfer and decode needs no change at all. Nothing goes
    early yet; both transfers are built in the same step.
    """

    def _admitted(self, *, early: int = 1, uuids=(900, 901)):
        producer, notifier = _producer()
        req = _request("short", PAGE + 1)
        with _early_layers_on(early):
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(uuids[0]):
                    stack.enter_context(ctx)
                stack.enter_context(
                    patch(f"{_MOD}.get_uuid", side_effect=list(uuids)))
                producer.update_state_after_alloc(req, MagicMock(), 0)
        return producer, req, notifier

    def _finish(self, producer, req, *, early: int = 1):
        with _early_layers_on(early):
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                return producer.request_finished(req, [100])

    def test_the_announcement_asks_decode_for_one_more_arrival(self):
        """Decode counts arrivals against len(chunk_uuids) + 1, so the early
        group has to be in that list or decode stops one transfer early and
        reads fifteen layers of KV that never landed."""
        _producer_, _req, notifier = self._admitted()
        payload = notifier.posts[0][1]
        assert payload["uuid"] == 900
        assert payload["chunk_uuids"] == [901]

    def test_with_the_split_off_the_announcement_is_unchanged(self):
        producer, notifier = _producer()
        req = _request("short", PAGE + 1)
        with ExitStack() as stack:
            for ctx in _stage3_producer_env(900):
                stack.enter_context(ctx)
            stack.enter_context(patch(f"{_MOD}.get_uuid", return_value=900))
            producer.update_state_after_alloc(req, MagicMock(), 0)
        assert "chunk_uuids" not in notifier.posts[0][1]

    def test_the_early_group_ships_as_its_own_send(self):
        producer, req = self._admitted()[:2]
        _give_destination(producer, "short",
                          _published_destination(900, [200]))
        self._finish(producer, req)

        early = producer.reqs_to_send["short#c0"]
        assert early.uuid == 901
        assert early.fa_group == 0
        assert early.is_final_chunk is False
        # The whole prompt, one group of layers. That is the axis this send is
        # split on, and it is why the token window is the same as the final
        # send's.
        assert early.start_token == 0
        assert early.num_tokens == PAGE

    def test_the_final_send_keeps_the_rest(self):
        producer, req = self._admitted()[:2]
        _give_destination(producer, "short",
                          _published_destination(900, [200]))
        self._finish(producer, req)

        final = producer.reqs_to_send["short"]
        assert final.uuid == 900
        assert final.fa_group == 1
        assert final.is_final_chunk is True
        assert final.num_tokens == PAGE

    def test_with_the_split_off_there_is_one_send_carrying_everything(self):
        """All-off has to reproduce stage 0, and fa_group=None is what tells
        the worker to name every attention tag on the one transfer."""
        producer, notifier = _producer()
        req = _request("short", PAGE + 1)
        with ExitStack() as stack:
            for ctx in _stage3_producer_env(900):
                stack.enter_context(ctx)
            stack.enter_context(patch(f"{_MOD}.get_uuid", return_value=900))
            producer.update_state_after_alloc(req, MagicMock(), 0)
        _give_destination(producer, "short",
                          _published_destination(900, [200]))
        with ExitStack() as stack:
            for ctx in _stage3_producer_env(900):
                stack.enter_context(ctx)
            producer.request_finished(req, [100])

        assert set(producer.reqs_to_send) == {"short"}
        assert producer.reqs_to_send["short"].fa_group is None

    def test_pages_decode_never_published_fail_loudly(self):
        """The announced arrival can only come from the producer, and by the
        time the prompt is over there is nothing left to wait for. A silent
        drop here would hang the request until decode's timeout."""
        producer, req = self._admitted()[:2]
        with pytest.raises(ValueError, match="never published"):
            self._finish(producer, req)

    def test_a_chunk_and_the_early_group_share_the_id_space(self):
        """Both splits number their sends off the same counter, so the early
        group takes the index after the last chunk. A collision would have one
        send overwrite the other in reqs_to_send and lose it."""
        producer, _notifier = _producer()
        req = _request("long", PROMPT)
        with _chunk_split_on(), _early_layers_on():
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                stack.enter_context(
                    patch(f"{_MOD}.get_uuid", side_effect=[900, 901, 902]))
                producer.update_state_after_alloc(req, MagicMock(), 0)
        _give_destination(producer, "long",
                          _published_destination(900, list(range(200, 208))))
        with _chunk_split_on(), _early_layers_on():
            with patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
                       return_value=30.0):
                first = _step(new=[("long", [100])],
                              scheduled={"long": BUDGET})
                second = _step(cached=[("long", [101])],
                               scheduled={"long": BUDGET})
                producer.build_connector_meta(first)
                producer.build_connector_meta(second)
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                producer.request_finished(req, [100, 101])

        # Chunk zero went out in an earlier step and was drained with its
        # metadata, so what is left is the early group at index one and the
        # final send.
        early = producer.reqs_to_send["long#c1"]
        assert early.uuid == 902
        assert early.fa_group == 0
        # The chunk already carried both groups over [0, BUDGET), so the early
        # group only owes the tokens behind it.
        assert (early.start_token, early.num_tokens) == (BUDGET, SENT)
        assert producer.reqs_to_send["long"].fa_group == 1


class TestEarlyLayerGroupLayout:
    """What the worker puts on the wire for a send that names one group."""

    def _worker(self):
        worker = _make_raiden_worker(is_producer=True, block_size=PAGE)
        worker._stage3_state_group_count = 0
        worker._stage3_fa_tags = ("fa.g0", "fa.g1")
        return worker

    def test_a_group_send_names_exactly_its_own_tag(self):
        worker = self._worker()
        tags, blocks, counts, _clip, _skip = worker._stage3_transfer_layout(
            dst_block_ids=[200, 201],
            mamba_state_block_ids=None,
            skip_tokens=0,
            fa_only=True,
            fa_group=0)
        assert tags == ["fa.g0"]
        # One tag replays over the destination pages once, so the block list
        # is not repeated the way it is when both groups ride together.
        assert blocks == [200, 201]
        assert counts == [2]

    def test_the_late_group_names_the_other_tag(self):
        worker = self._worker()
        tags, _blocks, _counts, _clip, _skip = worker._stage3_transfer_layout(
            dst_block_ids=[200],
            mamba_state_block_ids=None,
            skip_tokens=0,
            fa_only=True,
            fa_group=1)
        assert tags == ["fa.g1"]

    def test_no_group_still_names_both(self):
        worker = self._worker()
        tags, blocks, _counts, _clip, _skip = worker._stage3_transfer_layout(
            dst_block_ids=[200],
            mamba_state_block_ids=None,
            skip_tokens=0,
            fa_only=True)
        assert tags == ["fa.g0", "fa.g1"]
        assert blocks == [200, 200]

    def test_a_group_index_the_worker_does_not_have_is_refused(self):
        """A send naming group 1 on a worker that admitted one attention tag
        means the peers read the layer cuts differently. Silently clamping
        would ship one group's bytes under the other's name."""
        worker = self._worker()
        worker._stage3_fa_tags = ("fa", )
        with pytest.raises(RuntimeError, match="disagree about the attention"):
            worker._stage3_transfer_layout(dst_block_ids=[200],
                                           mamba_state_block_ids=None,
                                           skip_tokens=0,
                                           fa_only=True,
                                           fa_group=1)

    def test_a_group_send_still_takes_the_prefix_clip(self):
        worker = self._worker()
        worker._stage3_fa_token_bytes = lambda page_tokens: 1024
        tags, _blocks, _counts, clip, skip = worker._stage3_transfer_layout(
            dst_block_ids=[200],
            mamba_state_block_ids=None,
            skip_tokens=8,
            fa_only=True,
            fa_group=0)
        assert tags == ["fa.g0"]
        assert clip["dst_skip_bytes"] == [skip]

    def test_a_test_double_is_not_a_group_index(self):
        """Legacy send metadata has never heard of fa_group, and a mock
        answers every attribute. Either one reading as group zero would drop
        the late layers off a transfer that is meant to carry everything."""
        assert _send_meta_fa_group(MagicMock()) is None
        assert _send_meta_fa_group(SimpleNamespace()) is None
        assert _send_meta_fa_group(SimpleNamespace(fa_group=1)) == 1


# ---------------------------------------------------------------------------
# Stage 7c: the early group emitted in the step that computes it
# ---------------------------------------------------------------------------


@contextmanager
def _arming_on(on: bool = True, layers: int = 1):
    """Everything _doorbell_arm_enabled checks, in one place.

    All four have to agree before a send is built ahead of its pages, so a
    test that flips one of them off is testing the gate rather than the
    feature.
    """
    with _early_layers_on(layers), \
         patch(f"{_MOD}.tpu_envs.TPU_KV_DOORBELL_ARM", on, create=True), \
         patch(f"{_MOD}.tpu_envs.TPU_KV_DOORBELL", True, create=True), \
         patch(f"{_MOD}.tpu_envs.TPU_KV_DOORBELL_POLL_US", 200, create=True), \
         patch(f"{_MOD}.tpu_envs.TPU_KV_STAGE6_PREPARE", True, create=True), \
         patch(f"{_MOD}._stage6_prepare_fire_supported", return_value=True):
        yield


class TestEarlyGroupEmittedBeforeItsForward:
    """The send is built during the step whose forward will write its pages.

    Stage 7b built it afterwards, in request_finished, which is where every
    other send here is built and is one step too late to use a bell: by then
    all fifteen groups have rung, and on 397B group 0's rang about 1.3 s
    earlier. Building it beforehand is only safe because of the arm_on_bell
    flag it carries, which is the worker's instruction not to fire it.
    """

    def _admitted(self, prompt=PAGE + 1, uuids=(900, 901)):
        producer, notifier = _producer()
        req = _request("short", prompt)
        with _arming_on():
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(uuids[0]):
                    stack.enter_context(ctx)
                stack.enter_context(
                    patch(f"{_MOD}.get_uuid", side_effect=list(uuids)))
                producer.update_state_after_alloc(req, MagicMock(), 0)
        _give_destination(producer, "short",
                          _published_destination(900, [200]))
        return producer, req, notifier

    def _build(self, producer, step, *, on=True):
        with _arming_on(on):
            with patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
                       return_value=30.0):
                return producer.build_connector_meta(step)

    def _finish(self, producer, req):
        with _arming_on():
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                return producer.request_finished(req, [100])

    def test_the_early_group_is_in_the_metadata_of_the_last_step(self):
        producer, _req, _n = self._admitted()
        meta = self._build(
            producer,
            _step(new=[("short", [100])], scheduled={"short": PAGE + 1}))
        early = meta.reqs_to_send["short#c0"]
        assert early.uuid == 901
        assert early.fa_group == 0
        # The instruction that makes building it here legitimate.
        assert early.arm_on_bell == 0
        assert (early.start_token, early.num_tokens) == (0, PAGE)

    def test_a_step_that_does_not_finish_the_prompt_emits_nothing(self):
        """The test that says this is not simply 'emit earlier'. Group 0's
        bell rings in every forward, so a send armed in a step that is not
        the last would fire against a prompt that is only part written."""
        producer, _req, _n = self._admitted(prompt=BUDGET * 2)
        meta = self._build(
            producer, _step(new=[("short", [100])],
                            scheduled={"short": BUDGET}))
        assert meta.reqs_to_send == {}

    def test_request_finished_does_not_send_it_a_second_time(self):
        producer, req, _n = self._admitted()
        self._build(
            producer,
            _step(new=[("short", [100])], scheduled={"short": PAGE + 1}))
        producer.reqs_to_send.clear()
        self._finish(producer, req)
        # Only the final send, and it still knows the early group went
        # without it, because what it carries depends on that.
        assert set(producer.reqs_to_send) == {"short"}
        assert producer.reqs_to_send["short"].fa_group == 1

    def test_with_arming_off_it_waits_for_request_finished(self):
        """The stage gate. Off, this is stage 7b to the byte: nothing is built
        before its pages exist, so nothing depends on a bell to be correct."""
        producer, req, _n = self._admitted()
        meta = self._build(producer,
                           _step(new=[("short", [100])],
                                 scheduled={"short": PAGE + 1}),
                           on=False)
        assert meta.reqs_to_send == {}
        self._finish(producer, req)
        early = producer.reqs_to_send["short#c0"]
        assert early.arm_on_bell is None

    def test_with_nowhere_to_wait_it_falls_through_to_the_late_path(self):
        """A producer that was never told where the proxy is cannot hold the
        send, so it leaves it alone and request_finished picks it up in
        exactly the state stage 7b expects. That is a working fallback rather
        than a lost send."""
        producer, notifier = _producer()
        req = _request("short", PAGE + 1)
        with _arming_on():
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                stack.enter_context(
                    patch(f"{_MOD}.get_uuid", side_effect=[900, 901]))
                producer.update_state_after_alloc(req, MagicMock(), 0)
        meta = self._build(
            producer,
            _step(new=[("short", [100])], scheduled={"short": PAGE + 1}))
        assert meta.reqs_to_send == {}

        _give_destination(producer, "short",
                          _published_destination(900, [200]))
        self._finish(producer, req)
        assert producer.reqs_to_send["short#c0"].arm_on_bell is None

    def test_the_bell_and_the_chunks_compose(self):
        """The two splits are on different axes and they meet cleanly here.

        By the last step the leading chunks have shipped tokens [0, BUDGET)
        with every layer, so the early group owes only the tokens behind them
        and only its own layers. The forward this step is about to run is what
        writes those tokens, and its group-0 bell is exactly the signal that
        the part the early send names is finished."""
        producer, notifier = _producer()
        req = _request("long", BUDGET * 2)
        with _arming_on(), _chunk_split_on():
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                stack.enter_context(
                    patch(f"{_MOD}.get_uuid", side_effect=[900, 901, 902]))
                producer.update_state_after_alloc(req, MagicMock(), 0)
        _give_destination(producer, "long",
                          _published_destination(900, list(range(200, 208))))
        with _arming_on(), _chunk_split_on():
            with patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
                       return_value=30.0):
                producer.build_connector_meta(
                    _step(new=[("long", list(range(100, 108)))],
                          scheduled={"long": BUDGET}))
                meta = producer.build_connector_meta(
                    _step(cached=[("long", [])], scheduled={"long": BUDGET}))
        # Chunk 0, held one step and emitted now, plus the early group at the
        # next index. Decode was told to expect both and derives each id from
        # its position, so this is the announcement coming true.
        assert set(meta.reqs_to_send) == {"long#c0", "long#c1"}
        early = meta.reqs_to_send["long#c1"]
        assert early.arm_on_bell == 0
        assert early.fa_group == 0
        assert early.start_token == BUDGET
        # Chunk 0 carries every layer and waits on a device event, because the
        # forward that wrote it has already been issued. Only the send whose
        # forward has not run yet waits on a bell.
        assert meta.reqs_to_send["long#c0"].arm_on_bell is None


class TestArmGeneration:
    """Which forward's bell a send waits for.

    The whole risk of stage 7c is in this number. Naming a generation the
    request's KV is not finished in ships pages that still hold the previous
    request's data, which is the failure the first attempt at doorbells hit
    and never explained. Naming one too late costs a pass and nothing else,
    so every case here resolves towards later.
    """

    def _worker(self, *, generation=7, chunks=1, reader=True):
        worker = _make_raiden_worker(is_producer=True, block_size=PAGE)
        bells = SimpleNamespace(coming_generation=lambda: generation + 1)
        worker.runner = SimpleNamespace(
            _doorbells=bells,
            _doorbell_watcher=object() if reader else None,
            _dp_step_num_chunks=chunks)
        return worker

    def test_a_single_chunk_step_waits_for_the_next_forward(self):
        assert self._worker(generation=7)._arm_generation() == 8

    def test_a_multi_chunk_step_waits_for_the_last_of_them(self):
        """Each chunk is a forward of its own and rings the bells again. A
        request's tokens can be in any of them, so the only generation that
        is certainly after all of this request's KV is the last."""
        assert self._worker(generation=7, chunks=4)._arm_generation() == 11

    def test_no_reader_answers_zero_rather_than_guessing(self):
        """Zero is what tells the caller not to send at all. Firing without a
        bell would ship pages the forward has not written, and there is no
        third option that both ships and waits."""
        assert self._worker(reader=False)._arm_generation() == 0

    def test_no_runner_at_all_answers_zero(self):
        worker = _make_raiden_worker(is_producer=True, block_size=PAGE)
        assert worker._arm_generation() == 0

    def test_a_zero_chunk_count_is_read_as_one(self):
        """The runner leaves it at zero until the first step sets it, and a
        count of zero would name the generation before the coming one."""
        assert self._worker(generation=7, chunks=0)._arm_generation() == 8


# ---------------------------------------------------------------------------
# Stage 7d: the destination arrives after the send does
# ---------------------------------------------------------------------------

_DST_URL = "http://proxy.test:8000/kv_dst_batch"


class TestEarlyGroupShipsBeforeTheDestination:
    """The scheduler stops requiring a destination it usually cannot have.

    Stage 7c checked for decode's pages in the step that finishes the prompt
    and gave up if they were missing. On 397B decode takes about 1.4 s to
    allocate and answer, and a prompt of 16384 tokens or fewer prefills in one
    scheduler step, so the check ran roughly 50 ms after decode was first told
    the request existed and failed every time. Coverage came out at 26%, and
    the requests it missed were the short ones, which are the majority.

    So the send goes out naming the proxy instead. The worker holds it, and
    the transfer starts when the bell and the destination have both arrived.
    """

    def _admitted(self, *, url=_DST_URL, destination=True, prompt=PAGE + 1):
        producer, notifier = _producer()
        req = _request("short", prompt, dst_batch_url=url)
        with _arming_on():
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                stack.enter_context(
                    patch(f"{_MOD}.get_uuid", side_effect=[900, 901]))
                producer.update_state_after_alloc(req, MagicMock(), 0)
        if destination:
            _give_destination(producer, "short",
                              _published_destination(900, [200]))
        return producer, req, notifier

    def _build(self, producer, prompt=PAGE + 1):
        with _arming_on():
            with patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
                       return_value=30.0):
                return producer.build_connector_meta(
                    _step(new=[("short", [100])], scheduled={"short": prompt}))

    def test_a_missing_destination_no_longer_cancels_the_early_group(self):
        producer, _req, _n = self._admitted(destination=False)
        early = self._build(producer).reqs_to_send["short#c0"]
        assert early.arm_on_bell == 0
        assert early.dst_params is None
        # The whole content of the send's promise: there is somewhere to wait.
        assert early.dst_batch_url == _DST_URL

    def test_a_destination_already_in_hand_is_still_carried(self):
        """The long-prompt case, which stage 7c already handled. Nothing
        defers when there is nothing to wait for."""
        producer, _req, _n = self._admitted()
        early = self._build(producer).reqs_to_send["short#c0"]
        assert early.dst_params is not None
        assert early.dst_batch_url is None

    def test_the_url_is_learned_from_admission_not_from_the_send(self):
        """It is one address per engine, not per request, so a producer that
        has admitted anything at all knows where to point the worker."""
        producer, _req, _n = self._admitted(destination=False)
        assert producer._dst_batch_url == _DST_URL

    def test_the_late_path_still_refuses_to_ship_without_a_destination(self):
        """request_finished has no worker holding anything for it, so a
        missing destination there is still a request decode will wait out.
        Raising names the cause; shipping nothing would not."""
        producer, req, _n = self._admitted(destination=False)
        with pytest.raises(ValueError, match="never published"):
            producer._emit_early_group_send(
                "short",
                producer._chunk_progress["short"].admission,
                (100, ),
                PAGE,
                0,
                30.0,
                None,
            )


class TestWorkerHoldsASendForItsDestination:
    """The worker's half: register now, build when the answer lands.

    Only the generation can be read on the model runner thread, because it
    counts steps and reading it later would name a bell that has not rung.
    Everything else waits, which is safe because the doorbell reader compares
    each bell against the armed generation on every poll rather than only
    when a bell moves. A watch armed after its bell has rung fires on the next
    poll, so the send goes out when the later of the two arrives.
    """

    def _worker(self, *, generation=7, reader=True):
        worker = _make_raiden_worker(is_producer=True, block_size=PAGE)
        worker.runner = SimpleNamespace(
            _doorbells=SimpleNamespace(
                coming_generation=lambda: generation + 1),
            _doorbell_watcher=object() if reader else None,
            _dp_step_num_chunks=1)
        worker._ensure_stage3_submit_workers = MagicMock()
        return worker

    def _meta(self, *, url=_DST_URL, arm=0):
        return SendMeta(uuid=901,
                        local_block_ids=[100],
                        expiration_time=30.0,
                        num_tokens=PAGE,
                        dst_params=None,
                        fa_group=0,
                        arm_on_bell=arm,
                        dst_batch_url=url)

    def _defer(self, worker, meta, req_id="short#c0"):
        with patch.object(_DestinationCollector, "ensure_started"):
            worker._defer_until_destination(req_id=req_id,
                                            req_meta=meta,
                                            uuid=901,
                                            num_tokens=PAGE,
                                            parallelism=PCP)
        return worker._worker_destination_collector

    def test_a_held_send_records_the_generation_read_at_defer_time(self):
        worker = self._worker(generation=7)
        self._defer(worker, self._meta())
        held = worker._stage7_pending_dst["short#c0"]
        # Eight, not whatever the counter says when the answer arrives several
        # steps later. This is the only field that cannot be recomputed.
        assert held.arm_generation == 8
        assert held.uuid == 901

    def test_an_unarmed_send_is_not_held(self):
        """Stage 6.5's chunks reach the same branch with no destination, and
        they have always been the consumer's problem to start. Holding them
        would change a path this stage is not touching."""
        worker = self._worker()
        meta = self._meta(arm=None)
        meta.is_final_chunk = False
        self._defer(worker, meta)
        assert worker._stage7_pending_dst == {}
        assert worker._worker_destination_collector is None

    def test_a_final_send_is_held_without_a_bell(self):
        """Its early groups count as arrivals and cancel decode's fallback, so
        a final send that is not held here is never started by anyone. That
        was the concurrency-32 hang: the destination landed after the prompt
        finished."""
        worker = self._worker(generation=7)
        meta = self._meta(arm=None)
        meta.fa_group = 14
        collector = self._defer(worker, meta, req_id="short")
        assert worker._stage7_pending_dst["short"].arm_generation == 0
        collector._entries["short"] = _published_destination(901, [200])
        worker._maybe_start_producer_transfer = MagicMock()
        worker._release_pending_destinations(["short"])
        worker._stage3_submit_queue.get()()
        kwargs = worker._maybe_start_producer_transfer.call_args.kwargs
        assert kwargs["req_id"] == "short"
        assert kwargs["req_meta"].dst_params is not None

    def test_a_send_with_no_url_is_not_held(self):
        worker = self._worker()
        self._defer(worker, self._meta(url=None))
        assert worker._stage7_pending_dst == {}

    def test_a_worker_without_a_doorbell_reader_refuses_to_hold(self):
        """Nothing would ever release it, and firing it unreleased would ship
        pages the forward has not written. Dropping it costs decode its
        timeout, which is the only safe answer."""
        worker = self._worker(reader=False)
        self._defer(worker, self._meta())
        assert worker._stage7_pending_dst == {}

    def test_the_arriving_destination_starts_the_transfer(self):
        worker = self._worker(generation=7)
        collector = self._defer(worker, self._meta())
        dst = _published_destination(901, [200])
        collector._entries["short"] = dst

        worker._maybe_start_producer_transfer = MagicMock()
        worker._release_pending_destinations(["short"])
        assert worker._stage3_submit_queue.qsize() == 1
        worker._stage3_submit_queue.get()()

        kwargs = worker._maybe_start_producer_transfer.call_args.kwargs
        assert kwargs["req_id"] == "short#c0"
        assert kwargs["arm_generation_override"] == 8
        assert kwargs["req_meta"].dst_params == dst
        # Taken out of the table by the release, so a second batch naming the
        # same request cannot start it again.
        assert worker._stage7_pending_dst == {}

    def test_the_held_send_is_matched_on_the_base_request_id(self):
        """Decode publishes against the id it was admitted under. The send
        travels under a derived one, because every chunk does."""
        worker = self._worker()
        collector = self._defer(worker, self._meta(), req_id="short#c3")
        collector._entries["short"] = _published_destination(901, [200])
        worker._maybe_start_producer_transfer = MagicMock()
        worker._release_pending_destinations(["short"])
        assert worker._stage7_pending_dst == {}
        assert worker._stage3_submit_queue.qsize() == 1

    def test_an_answer_that_beat_the_send_is_picked_up_at_defer_time(self):
        """The 1.4 s wait is a measurement, not a guarantee, and a long prompt
        finishes well after it. Without this check that request waits for the
        next unrelated batch to wake the callback."""
        worker = self._worker()
        collector = _DestinationCollector(name="worker")
        collector._entries["short"] = _published_destination(901, [200])
        worker._worker_destination_collector = collector
        worker._maybe_start_producer_transfer = MagicMock()
        with patch.object(_DestinationCollector, "ensure_started"):
            worker._defer_until_destination(req_id="short#c0",
                                            req_meta=self._meta(),
                                            uuid=901,
                                            num_tokens=PAGE,
                                            parallelism=PCP)
        assert worker._stage7_pending_dst == {}
        assert worker._stage3_submit_queue.qsize() == 1

    def test_a_destination_for_someone_else_leaves_the_send_held(self):
        worker = self._worker()
        collector = self._defer(worker, self._meta())
        collector._entries["other"] = _published_destination(901, [200])
        worker._release_pending_destinations(["other"])
        assert set(worker._stage7_pending_dst) == {"short#c0"}
        assert worker._stage3_submit_queue.qsize() == 0

    def test_a_send_held_past_its_deadline_is_dropped(self):
        """A consumer that dies between allocating and publishing would
        otherwise pin the entry for the life of the process. Dropping it
        leaves the request to decode's own timeout, which is what stage 7c
        gave every late destination anyway."""
        worker = self._worker()
        self._defer(worker, self._meta())
        held = worker._stage7_pending_dst["short#c0"]
        worker._stage7_pending_dst["short#c0"] = replace(held, deadline=0.0)
        worker._release_pending_destinations(["nobody"])
        assert worker._stage7_pending_dst == {}
        assert worker._stage3_submit_queue.qsize() == 0

    def test_only_transfer_rank_zero_ever_holds_anything(self):
        """One arm covers the whole eight-way transfer, so one worker polls.
        Measured on 397B: 443 of 444 dispatches came from rank 0."""
        worker = self._worker()
        worker._maybe_start_producer_transfer(req_id="short#c0",
                                              req_meta=self._meta(),
                                              uuid=901,
                                              num_tokens=PAGE,
                                              transfer_rank=3,
                                              parallelism=PCP)
        assert worker._stage7_pending_dst == {}


class TestCollectorSharesEntriesWithASecondReader:
    """Two collectors now read the same proxy, so neither may consume.

    The engine core needs a destination to build an ordinary send and the
    worker needs one to release a held send, and they run in different
    processes. The proxy therefore keeps entries and each reader asks for what
    is newer than the cursor it last saw.
    """

    def _poll(self, collector, replies):
        """Runs _run against a canned sequence, stopping after the last."""
        seen = []

        class _Resp:

            def __init__(self, body):
                self._body = body

            def read(self):
                return json.dumps(self._body).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def _open(url, timeout=None):
            seen.append(url)
            if len(seen) > len(replies):
                raise _Stop()
            return _Resp(replies[len(seen) - 1])

        class _Stop(Exception):
            pass

        with patch(f"{_MOD}.urllib.request.urlopen", side_effect=_open):
            try:
                collector._run()
            except _Stop:
                pass
        return seen

    def test_the_poll_carries_the_cursor_the_proxy_last_returned(self):
        collector = _DestinationCollector(name="worker")
        collector._url = _DST_URL
        urls = self._poll(collector, [
            {
                "entries": {
                    "a": {
                        "x": 1
                    }
                },
                "cursor": 7
            },
            {
                "entries": {},
                "cursor": 7
            },
        ])
        assert urls[0].endswith("after=0")
        assert urls[1].endswith("after=7")

    def test_a_proxy_that_returns_no_cursor_leaves_it_alone(self):
        """An older proxy consumes on read and says nothing about cursors.
        One collector against it still works, which is what makes the two
        sides independently deployable."""
        collector = _DestinationCollector(name="engine")
        collector._url = _DST_URL
        self._poll(collector, [{"entries": {"a": {"x": 1}}}])
        assert collector._cursor == 0
        assert collector.peek("a") == {"x": 1}

    def test_arrivals_are_announced_once_the_entries_are_readable(self):
        """The callback starts transfers that immediately peek, so the merge
        has to be finished before it runs."""
        seen = {}
        collector = _DestinationCollector(
            name="worker",
            on_arrival=lambda ids: seen.update(
                {i: collector.peek(i)
                 for i in ids}))
        collector._url = _DST_URL
        self._poll(collector, [{"entries": {"a": {"x": 1}}, "cursor": 1}])
        assert seen == {"a": {"x": 1}}

    def test_a_failing_callback_does_not_kill_the_poll_thread(self):
        """Losing a release costs one request its timeout. Losing the thread
        costs every request after it."""
        collector = _DestinationCollector(
            name="worker", on_arrival=MagicMock(side_effect=RuntimeError))
        collector._url = _DST_URL
        urls = self._poll(collector, [{"entries": {"a": {}}, "cursor": 1}])
        # It polled again, and with the cursor advanced, so the failure cost
        # one release and not the entries behind it.
        assert urls[1].endswith("after=1")


class TestTheEarlyGroupWaitsOnItsLastLayer:
    """A bell is a layer, a transfer group is a span of them.

    The two indices are equal only while the early group is one layer wide,
    which is the only width stage 7 runs, so the literal zero stage 7c passed
    has always been right and has always been right by accident. Widen the
    split and it fires on the first layer of the group while the rest is still
    being written, shipping whatever those pages held before. That is a wrong
    answer rather than a slow one, and it is the failure mode the first
    attempt at this hit and never explained.
    """

    def _early(self, layers):
        producer, _notifier = _producer()
        req = _request("short", PAGE + 1, dst_batch_url=_DST_URL)
        with _arming_on(layers=layers):
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                stack.enter_context(
                    patch(f"{_MOD}.get_uuid", side_effect=[900, 901]))
                producer.update_state_after_alloc(req, MagicMock(), 0)
            with patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
                       return_value=30.0):
                meta = producer.build_connector_meta(
                    _step(new=[("short", [100])],
                          scheduled={"short": PAGE + 1}))
        return meta.reqs_to_send["short#c0"]

    def test_one_layer_still_waits_on_bell_zero(self):
        """The width stage 7 shipped and measured. This has to be unchanged,
        or the fix is a behaviour change dressed as a correctness fix."""
        assert self._early(1).arm_on_bell == 0

    @pytest.mark.parametrize("layers,bell", [(2, 1), (3, 2), (8, 7)])
    def test_a_wider_group_waits_on_its_last_layer(self, layers, bell):
        assert self._early(layers).arm_on_bell == bell


class TestAWatchThatNeverFiresIsSweptAndSent:
    """The request that hung a 397B run, and why it hung forever.

    _arm_generation predicts the generation the step's last forward will ring.
    It can overshoot, and then the watch names a generation no forward reaches.
    With traffic behind it a later request's forward rings that generation and
    the watch fires late but fires. On the last prefill of a run there is no
    traffic behind it. Before stage 7e nothing noticed: an armed watch was
    never retried, never swept and never handed back, so decode sat in
    WAITING_FOR_REMOTE_KVS for as long as anyone let it.

    The sweep makes the worst case today's behaviour again, which is what the
    design promised. Cancel is the handoff that keeps it exactly once, because
    the reader refuses to cancel a watch it has already fired.
    """

    def _worker(self, *, cancels=True):
        worker = _make_raiden_worker(is_producer=True, block_size=PAGE)
        worker.runner = SimpleNamespace(_doorbell_watcher=SimpleNamespace(
            cancel=MagicMock(return_value=cancels)))
        facade = MagicMock()
        worker._require_stage3_controller = MagicMock(return_value=(facade,
                                                                    None))
        return worker, facade

    def _arm(self, worker, *, overdue, req_id="short#c0", uuid=901):
        """Puts one watch in the table, already past its deadline or not."""
        offset = -1.0 if overdue else _STAGE7_ARM_DEADLINE_S
        worker._stage7_watches[(req_id, uuid)] = (77,
                                                  time.monotonic() + offset)

    def test_an_expired_watch_is_cancelled_and_sent_from_here(self):
        worker, facade = self._worker()
        self._arm(worker, overdue=True)
        worker._sweep_expired_watches()
        worker.runner._doorbell_watcher.cancel.assert_called_once_with(77)
        facade.fire_transfer.assert_called_once_with("short#c0", 901)
        assert worker._stage7_watches == {}

    def test_a_watch_still_within_its_deadline_is_left_alone(self):
        worker, facade = self._worker()
        self._arm(worker, overdue=False)
        worker._sweep_expired_watches()
        worker.runner._doorbell_watcher.cancel.assert_not_called()
        facade.fire_transfer.assert_not_called()
        assert ("short#c0", 901) in worker._stage7_watches

    def test_a_watch_that_fired_first_is_not_sent_twice(self):
        """The reader refuses the cancel, which is how the sweep learns the
        transfer is already on the wire. Sending it again would be a duplicate
        the controller rejects and, worse, a second write into pages the
        consumer may have started reading."""
        worker, facade = self._worker(cancels=False)
        self._arm(worker, overdue=True)
        worker._sweep_expired_watches()
        facade.fire_transfer.assert_not_called()
        assert worker._stage7_watches == {}

    def test_a_send_that_fails_falls_back_to_the_consumer(self):
        """Discarding the plan releases its source registration claim. Without
        that the consumer's own fallback submit comes back as a duplicate and
        the request is no better off than it was hanging."""
        worker, facade = self._worker()
        facade.fire_transfer.side_effect = RuntimeError("controller is gone")
        self._arm(worker, overdue=True)
        worker._sweep_expired_watches()
        facade.discard_prepared_transfer.assert_called_once_with(
            "short#c0", 901)

    def test_the_sweep_does_nothing_without_a_reader(self):
        worker, facade = self._worker()
        worker.runner = SimpleNamespace(_doorbell_watcher=None)
        self._arm(worker, overdue=True)
        worker._sweep_expired_watches()
        facade.fire_transfer.assert_not_called()

    def test_sweeping_an_empty_table_touches_no_controller(self):
        """It runs once a second forever, so the common case has to be free."""
        worker, _facade = self._worker()
        worker._sweep_expired_watches()
        worker._require_stage3_controller.assert_not_called()

    def test_only_the_expired_entry_leaves_the_table(self):
        worker, facade = self._worker()
        self._arm(worker, overdue=True, req_id="a#c0", uuid=1)
        self._arm(worker, overdue=False, req_id="b#c0", uuid=2)
        worker._sweep_expired_watches()
        assert list(worker._stage7_watches) == [("b#c0", 2)]
        facade.fire_transfer.assert_called_once_with("a#c0", 1)

    def test_an_abort_still_takes_a_watch_back_before_the_deadline(self):
        """_cancel_bell_watch reads the same table the sweep now writes, so
        the pair it stores has to stay readable by both."""
        worker, facade = self._worker()
        self._arm(worker, overdue=False)
        worker._cancel_bell_watch("short#c0", 901)
        worker.runner._doorbell_watcher.cancel.assert_called_once_with(77)
        facade.discard_prepared_transfer.assert_called_once_with(
            "short#c0", 901)
        assert worker._stage7_watches == {}

    def test_the_sweeper_thread_starts_once_and_is_a_daemon(self):
        """It has to outlive the step loop, because the case it exists for is
        a watch armed by the last prefill of a run with no step after it."""
        worker, _facade = self._worker()
        with patch(f"{_MOD}.threading.Thread") as thread:
            worker._ensure_stage7_watch_sweeper()
            worker._ensure_stage7_watch_sweeper()
        thread.assert_called_once()
        assert thread.call_args.kwargs["daemon"] is True
        thread.return_value.start.assert_called_once()

    def test_the_strand_injection_is_inert_unless_asked_for(self):
        """It rewrites the generation a watch arms on, so a stray default
        would silently route every transfer through the sweep."""
        assert _maybe_strand_stage7_watch_for_test(8) == 8

    def test_the_strand_injection_moves_the_bell_out_of_reach(self):
        """Every second watch, so a leg run with it set has both paths in it
        and the ordinary one is still the one being compared against."""
        with patch.dict(os.environ, {"TPU_KV_STAGE7_STRAND_EVERY": "2"}):
            armed = [_maybe_strand_stage7_watch_for_test(8) for _ in range(4)]
        assert armed.count(8) == 2
        assert all(g > 8 for g in armed if g != 8)


# ---------------------------------------------------------------------------
# Stage 8: every attention group ships on its own bell
# ---------------------------------------------------------------------------


@contextmanager
def _cuts_on(spec: str):
    """The stage 8 knob, plus the producer-submit flag it needs.

    TPU_KV_GROUP_BOUNDS names the cuts directly, so unlike _early_layers_on
    this does not have to say how wide each group is. "1,4,8" is four groups
    and three of them ship early.
    """
    with patch(f"{_MOD}.tpu_envs.TPU_KV_GROUP_BOUNDS", spec, create=True), \
         patch(f"{_MOD}.tpu_envs.TPU_KV_PREFILL_SUBMIT", True, create=True):
        yield


@contextmanager
def _cuts_arming_on(spec: str):
    """_arming_on, but with the cuts spelled as stage 8 spells them."""
    with _cuts_on(spec), \
         patch(f"{_MOD}.tpu_envs.TPU_KV_DOORBELL_ARM", True, create=True), \
         patch(f"{_MOD}.tpu_envs.TPU_KV_DOORBELL", True, create=True), \
         patch(f"{_MOD}.tpu_envs.TPU_KV_DOORBELL_POLL_US", 200, create=True), \
         patch(f"{_MOD}.tpu_envs.TPU_KV_STAGE6_PREPARE", True, create=True), \
         patch(f"{_MOD}._stage6_prepare_fire_supported", return_value=True):
        yield


class TestTheCutsAreReadTheWayTheyAreWritten:
    """Where the layers are cut, before anything is sent anywhere.

    The cuts decide three separate things that all have to agree: how many
    uuids admission mints, which bell each send waits for, and how many pool
    tags the worker registers. Getting this wrong is not a slow run. Too few
    uuids and decode waits forever for an arrival nobody sends; the wrong bell
    and a transfer ships pages the forward has not written yet.
    """

    def test_nothing_set_is_no_split_at_all(self):
        """The floor the whole design promised: every knob off reproduces
        stage 6, one transfer carrying every layer."""
        with patch(f"{_MOD}.tpu_envs.TPU_KV_GROUP_BOUNDS", "", create=True), \
             patch(f"{_MOD}.tpu_envs.TPU_KV_EARLY_LAYERS", 0, create=True):
            assert _fa_group_cuts() == ()
            assert _early_group_bells() == ()

    def test_early_layers_is_the_one_cut_spelling(self):
        """Stage 7's knob still means what it meant, because stage 7's
        measurements were taken with it and they have to stay comparable."""
        with patch(f"{_MOD}.tpu_envs.TPU_KV_GROUP_BOUNDS", "", create=True), \
             patch(f"{_MOD}.tpu_envs.TPU_KV_EARLY_LAYERS", 3, create=True):
            assert _fa_group_cuts() == (3, )

    def test_the_cut_list_wins_over_the_old_knob(self):
        with patch(f"{_MOD}.tpu_envs.TPU_KV_GROUP_BOUNDS", "2,5",
                   create=True), \
             patch(f"{_MOD}.tpu_envs.TPU_KV_EARLY_LAYERS", 9, create=True):
            assert _fa_group_cuts() == (2, 5)

    @pytest.mark.parametrize("spec,cuts", [
        ("1", (1, )),
        ("1,4,8", (1, 4, 8)),
        (" 2 , 6 ", (2, 6)),
        ("1,2,3,4,5,6,7,8,9,10,11,12,13,14", tuple(range(1, 15))),
    ])
    def test_a_well_formed_list_parses(self, spec, cuts):
        with _cuts_on(spec):
            assert _fa_group_cuts() == cuts

    @pytest.mark.parametrize("spec", ["0", "0,3", "4,2", "2,2", "-1", "3,3,5"])
    def test_a_malformed_list_is_refused_rather_than_repaired(self, spec):
        """Sorting it here would be guessing at an intent the operator did not
        express, and the guess is invisible until a transfer ships the wrong
        layers under the right tag."""
        with _cuts_on(spec):
            with pytest.raises(ValueError, match="strictly increasing"):
                _fa_group_cuts()

    @pytest.mark.parametrize("spec,bells", [
        ("1", (0, )),
        ("1,4,8", (0, 3, 7)),
        ("14", (13, )),
    ])
    def test_a_group_waits_on_the_layer_that_finishes_it(self, spec, bells):
        """A group ending just before cut C is finished by layer C-1. Waiting
        on the group's first layer instead fires while the rest is still being
        written, which is stale bytes rather than a slow run."""
        with _cuts_on(spec):
            assert _early_group_bells() == bells

    def test_the_split_still_needs_the_producer_to_be_the_one_sending(self):
        """Decode's fallback timer only ever starts the base uuid, so under
        consumer-initiated transfers an early group is a transfer nobody would
        start and the request would never finish."""
        with patch(f"{_MOD}.tpu_envs.TPU_KV_GROUP_BOUNDS", "1,4",
                   create=True), \
             patch(f"{_MOD}.tpu_envs.TPU_KV_PREFILL_SUBMIT", False,
                   create=True):
            assert _fa_group_cuts() == (1, 4)
            assert not _early_layer_split_enabled()


class TestManyGroupsShipAsManyTransfers:
    """One request, N+1 transfers: N early groups and the rest.

    Stage 7 proved one group can leave before the forward finishes. Stage 8 is
    the same mechanism at every width, and the thing it has to get right is
    bookkeeping rather than timing. Decode derives each arrival's id from its
    position in the announced list, so the uuids, the send ids and the group
    indices have to be three views of one ordering.
    """

    CUTS = "1,4,8"
    UUIDS = (900, 901, 902, 903)

    def _admitted(self, spec=CUTS, uuids=UUIDS, prompt=PAGE + 1):
        producer, notifier = _producer()
        req = _request("short", prompt, dst_batch_url=_DST_URL)
        with _cuts_arming_on(spec):
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(uuids[0]):
                    stack.enter_context(ctx)
                stack.enter_context(
                    patch(f"{_MOD}.get_uuid", side_effect=list(uuids)))
                producer.update_state_after_alloc(req, MagicMock(), 0)
        _give_destination(producer, "short",
                          _published_destination(uuids[0], [200]))
        return producer, req, notifier

    def _build(self, producer, spec=CUTS, prompt=PAGE + 1):
        with _cuts_arming_on(spec):
            with patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout",
                       return_value=30.0):
                return producer.build_connector_meta(
                    _step(new=[("short", [100])], scheduled={"short": prompt}))

    def test_admission_mints_one_uuid_per_early_group(self):
        _p, _req, notifier = self._admitted()
        announced = notifier.posts[-1][1]
        # Three early groups, no chunks: decode is told to expect four
        # arrivals, which is the three leading uuids plus the final one.
        assert announced["chunk_uuids"] == [901, 902, 903]

    def test_the_admission_remembers_which_bell_each_one_waits_on(self):
        """Frozen at admission rather than re-read when the send is built, so
        the count decode was promised and the count actually sent come from
        one place and cannot drift apart."""
        producer, _req, _n = self._admitted()
        admission = producer._chunk_progress["short"].admission
        assert admission.early_uuids == (901, 902, 903)
        assert admission.early_bells == (0, 3, 7)

    def test_each_group_ships_under_its_own_id_group_and_bell(self):
        producer, _req, _n = self._admitted()
        meta = self._build(producer)
        for index, (uuid, group, bell) in enumerate([(901, 0, 0), (902, 1, 3),
                                                     (903, 2, 7)]):
            send = meta.reqs_to_send[_chunk_req_id("short", index)]
            assert (send.uuid, send.fa_group,
                    send.arm_on_bell) == (uuid, group, bell)
            # None of them retires the request or carries recurrent state;
            # the final send behind them does both.
            assert send.is_final_chunk is False

    def test_the_final_send_carries_the_one_group_nobody_cut_away(self):
        """Three cuts make four groups, so the last is group 3. Naming any
        other index would ship a group twice and leave one on the floor."""
        producer, req, _n = self._admitted()
        self._build(producer)
        with _cuts_arming_on(self.CUTS):
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                producer.request_finished(req, [100])
        assert producer.reqs_to_send["short"].fa_group == 3

    def test_fifteen_groups_is_fourteen_early_sends(self):
        """The width the design is aiming at, where every attention layer is
        its own transfer and only the last waits for the forward to end."""
        spec = ",".join(str(cut) for cut in range(1, 15))
        uuids = tuple(range(900, 915))
        producer, _req, _n = self._admitted(spec=spec, uuids=uuids)
        meta = self._build(producer, spec=spec)
        assert len(meta.reqs_to_send) == 14
        sends = [
            meta.reqs_to_send[_chunk_req_id("short", i)] for i in range(14)
        ]
        assert [s.fa_group for s in sends] == list(range(14))
        assert [s.arm_on_bell for s in sends] == list(range(14))
        assert [s.uuid for s in sends] == list(range(901, 915))

    def test_every_transfer_gets_a_uuid_of_its_own(self):
        """Two sends sharing a uuid is one transfer as far as raiden is
        concerned, and the second registration comes back a duplicate."""
        producer, _req, _n = self._admitted()
        meta = self._build(producer)
        uuids = [
            meta.reqs_to_send[_chunk_req_id("short", i)].uuid for i in range(3)
        ]
        assert len(set(uuids)) == 3

    def test_the_late_path_ships_the_same_groups_without_the_bells(self):
        """request_finished emits after the prompt is over, so there is
        nothing left to wait for and every send goes out unarmed. It is the
        fallback for a step that crosses a chunk boundary and finishes the
        prompt together."""
        producer, req, _n = self._admitted()
        with _cuts_on(self.CUTS):
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                producer.request_finished(req, [100])
        sends = [
            producer.reqs_to_send[_chunk_req_id("short", i)] for i in range(3)
        ]
        assert [s.fa_group for s in sends] == [0, 1, 2]
        assert all(s.arm_on_bell is None for s in sends)
        assert producer.reqs_to_send["short"].fa_group == 3


class TestTheTwoSplitsStillCompose:
    """Chunks cut by token, groups cut by layer, and the ids stay in order.

    Decode walks one list. The chunks come first and the groups after, so a
    request with two chunks and three groups announces five leading uuids and
    the group that the worker puts on the wire for #c3 has to be group 1, not
    chunk 3.
    """

    def test_the_groups_are_numbered_after_the_chunks(self):
        # A budget of two pages cuts a four-page prompt into two chunks, so
        # this request sends two chunk transfers and two group transfers.
        producer, notifier = _producer(budget=2 * PAGE)
        req = _request("long", 4 * PAGE, dst_batch_url=_DST_URL)
        uuids = list(range(900, 910))
        with _cuts_arming_on("1,4"), _chunk_split_on():
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                stack.enter_context(
                    patch(f"{_MOD}.get_uuid", side_effect=uuids))
                producer.update_state_after_alloc(req, MagicMock(), 0)
        admission = producer._stage3_admitted_sends["long"]
        announced = notifier.posts[-1][1]["chunk_uuids"]
        # The chunk uuids lead and the group uuids follow, which is the order
        # the derived ids are numbered in and the only thing decode knows.
        assert announced == list(admission.chunk_uuids) + list(
            admission.early_uuids)
        assert len(admission.chunk_uuids) == 1
        assert len(admission.early_uuids) == 2
        # Two chunks means one leading chunk transfer, so the groups start at
        # #c1 and the final send is still the base id.
        assert announced == [900 + i for i in range(1, 4)]


class TestTheWorkerRegistersOneTagPerGroup:
    """What the worker puts on the wire for a send naming one of N groups."""

    def _worker(self, groups: int):
        worker = _make_raiden_worker(is_producer=True, block_size=PAGE)
        worker._stage3_state_group_count = 0
        worker._stage3_fa_tags = tuple(f"fa.g{i}" for i in range(groups))
        return worker

    @pytest.mark.parametrize("group", [0, 2, 3])
    def test_a_group_send_names_exactly_its_own_tag(self, group):
        worker = self._worker(4)
        tags, blocks, counts, _clip, _skip = worker._stage3_transfer_layout(
            dst_block_ids=[200, 201],
            mamba_state_block_ids=None,
            skip_tokens=0,
            fa_only=True,
            fa_group=group)
        assert tags == [f"fa.g{group}"]
        # One tag replays over the destination pages once, however many other
        # groups exist, because the bytes it carries are its own layers only.
        assert blocks == [200, 201]
        assert counts == [2]

    def test_a_group_past_the_end_is_refused(self):
        """Group 4 on a worker holding four tags means the peers read the cut
        list differently, and clamping would ship one group's bytes under
        another's name."""
        worker = self._worker(4)
        with pytest.raises(RuntimeError, match="disagree about the attention"):
            worker._stage3_transfer_layout(dst_block_ids=[200],
                                           mamba_state_block_ids=None,
                                           skip_tokens=0,
                                           fa_only=True,
                                           fa_group=4)

    def test_no_group_named_still_names_all_of_them(self):
        """The unsplit send, which is what the final transfer is when the
        layer split is off entirely."""
        worker = self._worker(4)
        tags, blocks, _counts, _clip, _skip = worker._stage3_transfer_layout(
            dst_block_ids=[200],
            mamba_state_block_ids=None,
            skip_tokens=0,
            fa_only=True)
        assert tags == ["fa.g0", "fa.g1", "fa.g2", "fa.g3"]
        assert blocks == [200] * 4


class TestEveryLayerLandsInExactlyOneGroup:
    """The partition, which is where a cut list becomes pool tags.

    The layers arrive in pool order, so a layer's position is also the bell it
    rings, and the whole scheme rests on that. A layer in the wrong group
    means its bytes go out under another group's tag: the transfer succeeds,
    decode is happy, and the KV is wrong.
    """

    LAYERS = tuple(f"layer{i}" for i in range(15))

    def test_one_cut_is_the_stage_seven_split(self):
        ordinals = _fa_group_ordinals(self.LAYERS, (1, ))
        assert ordinals["layer0"] == 0
        assert all(ordinals[f"layer{i}"] == 1 for i in range(1, 15))

    def test_the_boundaries_land_where_the_cuts_say(self):
        """(1, 4, 8) reads as four groups holding layers 0, 1-3, 4-7 and
        8 onward, so the layer at each cut starts the next group rather than
        ending the previous one."""
        ordinals = _fa_group_ordinals(self.LAYERS, (1, 4, 8))
        assert [ordinals[f"layer{i}"]
                for i in range(15)] == ([0] + [1] * 3 + [2] * 4 + [3] * 7)

    def test_every_layer_its_own_group(self):
        ordinals = _fa_group_ordinals(self.LAYERS, tuple(range(1, 15)))
        assert [ordinals[f"layer{i}"] for i in range(15)] == list(range(15))

    def test_a_cut_past_the_last_layer_is_refused(self):
        """An empty final group is a request that never finishes: that group
        is the one carrying the recurrent state and retiring the request."""
        with pytest.raises(ValueError, match="leave a late group"):
            _fa_group_ordinals(self.LAYERS, (1, 15))

    def test_a_cut_on_the_last_layer_is_fine(self):
        """Fourteen cuts of fifteen layers leaves the fifteenth alone, which
        is the widest split the model allows."""
        ordinals = _fa_group_ordinals(self.LAYERS, (14, ))
        assert ordinals["layer14"] == 1
        assert ordinals["layer13"] == 0
