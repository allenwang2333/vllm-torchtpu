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

Helpers come from the sibling module rather than being duplicated; this file
is separate only because that one is already five thousand lines.
"""
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.request import RequestStatus

from vllm_torchtpu.distributed.kv_transfer.connector_metadata import SendMeta

from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (  # isort: skip
    _chunk_base_req_id, _chunk_req_id, _flatten_block_ids, _send_meta_int,
    _Stage3Admission)

from .test_tpu_connector import (  # isort: skip
    _MOD, _make_raiden_scheduler, _make_raiden_worker, _RecordingNotifier,
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


def _request(req_id: str, prompt_tokens: int):
    req = MagicMock()
    req.request_id = req_id
    req.prompt_token_ids = [0] * prompt_tokens
    req.num_prompt_tokens = prompt_tokens
    req.num_computed_tokens = prompt_tokens
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {
        "admit_callback_url": f"http://proxy.test:8000/kv_admit/{req_id}"
    }
    return req


def _step(*, new=(), cached=(), scheduled=None):
    """A scheduler output carrying only what _collect_chunk_sends reads.

    ``new`` is (req_id, block_ids) for requests scheduled for the first time,
    ``cached`` is (req_id, newly_allocated_block_ids) for the ones continuing.
    """
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=r, block_ids=([list(b)]))
            for r, b in new
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[r for r, _ in cached],
            new_block_ids=[[list(b)] for _, b in cached]),
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
        """A prompt one token past a boundary is still two chunks, and decode
        sizing its wait by the floor would stop one transfer early."""
        producer, notifier = _producer()
        req = _request("edge", BUDGET + 2)
        with _chunk_split_on():
            self._announce(producer, req, [900, 901])

        assert notifier.posts[0][1]["chunk_uuids"] == [901]

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
                    patch(f"{_MOD}.get_uuid",
                          side_effect=[900, 901, 902, 903]))
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
            producer,
            _step(new=[("long", [100])], scheduled={"long": BUDGET}))
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
        self._build(producer,
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
        self._build(producer,
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
            producer,
            _step(cached=[("long", [101])], scheduled={"long": 24576}))
        assert set(meta.reqs_to_send) == {"long#c0"}
        assert meta.reqs_to_send["long#c0"].num_tokens == BUDGET

    def test_a_chunk_pending_when_the_prefill_ends_is_flushed(self):
        """A step can cross a boundary and finish the prompt together when
        the budget is not a whole number of pages. There is no next step to
        carry the chunk, so request_finished has to emit it."""
        producer, _notifier = _producer(budget=30000)
        req = _request("long", 28000)
        with _chunk_split_on():
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                stack.enter_context(
                    patch(f"{_MOD}.get_uuid", side_effect=[900, 901]))
                producer.update_state_after_alloc(req, MagicMock(), 0)
        _give_destination(producer, "long",
                          _published_destination(900, list(range(200, 208))))

        self._build(producer,
                    _step(new=[("long", [100])], scheduled={"long": 28000}))
        assert producer._chunk_progress["long"].pending is not None

        with _chunk_split_on():
            with ExitStack() as stack:
                for ctx in _stage3_producer_env(900):
                    stack.enter_context(ctx)
                producer.request_finished(req, [100])

        assert set(producer.reqs_to_send) == {"long#c0", "long"}
        assert producer.reqs_to_send["long#c0"].num_tokens == 24576
        assert producer.reqs_to_send["long"].start_token == 24576

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


# ---------------------------------------------------------------------------
# What decode expects
# ---------------------------------------------------------------------------


class TestDecodeExpectations:

    def test_an_unchunked_load_expects_one_arrival(self):
        worker = _make_raiden_worker(is_producer=False, block_size=PAGE)
        assert _send_meta_int(SimpleNamespace(), "num_chunks") == 0
        assert max(1, _send_meta_int(SimpleNamespace(num_chunks=1),
                                     "num_chunks")) == 1
        del worker

    def test_a_chunked_load_expects_one_arrival_per_chunk(self):
        assert max(1, _send_meta_int(SimpleNamespace(num_chunks=3),
                                     "num_chunks")) == 3

    def test_send_meta_defaults_reproduce_the_unchunked_send(self):
        """Every legacy caller builds a SendMeta without these fields, so the
        defaults are what keeps the split invisible when it is off."""
        meta = SendMeta(uuid=1, local_block_ids=[2], expiration_time=0.0)
        assert meta.start_token == 0
        assert meta.chunk_index == 0
        assert meta.is_final_chunk is True
