# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _load_toy_proxy_server():
    path = (Path(__file__).resolve().parents[2] / "examples" / "disagg" /
            "toy_proxy_server.py")
    spec = importlib.util.spec_from_file_location("toy_proxy_server", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_toy_proxy_server = _load_toy_proxy_server()
_AsciiSafeStreamEncoder = _toy_proxy_server._AsciiSafeStreamEncoder
_replace_prompt_with_rendered_token_ids = (
    _toy_proxy_server._replace_prompt_with_rendered_token_ids)


def test_ascii_safe_stream_encoder_handles_split_utf8() -> None:
    encoder = _AsciiSafeStreamEncoder()
    text = "\u044f\U0001f600"
    message = f'data: {{"choices": [{{"text": "{text}"}}]}}\n\n'
    encoded = message.encode("utf-8")
    split_at = encoded.index(text.encode("utf-8")) + 1

    output = encoder.encode(encoded[:split_at])
    output += encoder.encode(encoded[split_at:])
    output += encoder.encode(final=True)

    assert output.isascii()
    assert b"\\u044f" in output
    assert b"\\ud83d\\ude00" in output

    payload = output.decode("ascii").removeprefix("data: ").strip()
    assert json.loads(payload)["choices"][0]["text"] == text


def test_ascii_safe_stream_encoder_leaves_ascii_unchanged() -> None:
    encoder = _AsciiSafeStreamEncoder()
    message = b'data: {"choices": [{"text": "ok"}]}\n\n'

    assert encoder.encode(message) == message
    assert encoder.encode(final=True) == b""


def test_replace_prompt_with_rendered_token_ids_single_prompt() -> None:
    request = {
        "model": "test-model",
        "prompt": "hello",
        "prompt_embeds": b"unused-after-render",
    }
    rendered = [{"token_ids": [1, 2, 3]}]

    converted = _replace_prompt_with_rendered_token_ids(request, rendered)

    assert converted["prompt"] == [1, 2, 3]
    assert "prompt_embeds" not in converted
    assert request["prompt"] == "hello"


def test_replace_prompt_with_rendered_token_ids_multi_prompt() -> None:
    request = {
        "model": "test-model",
        "prompt": ["hello", "world"],
    }
    rendered = [{"token_ids": [1, 2, 3]}, {"token_ids": [4, 5]}]

    converted = _replace_prompt_with_rendered_token_ids(request, rendered)

    assert converted["prompt"] == [[1, 2, 3], [4, 5]]


def test_replace_prompt_with_rendered_token_ids_rejects_bad_response() -> None:
    request = {
        "model": "test-model",
        "prompt": "hello",
    }

    try:
        _replace_prompt_with_rendered_token_ids(request, [{
            "token_ids": ["1"]
        }])
    except ValueError as exc:
        assert "token_ids" in str(exc)
    else:
        raise AssertionError("malformed render response should fail")


# ---- admission reorder ----------------------------------------------------

_MOD = _toy_proxy_server

_ADMIT_PARAMS = {
    "req_id": "engine-side-id",
    "uuid": 4242,
    "num_tokens": 65_023,
    "src_controller_address": "prefill-controller.test:27000",
    "src_job_name": "prefill",
    "src_engine_id": "producer-engine-9",
    "src_data_replica_idx": 0,
    "src_parallelism": 8,
}


class _FakeRequest:
    """The parts of a starlette Request that _handle_completions touches."""

    def __init__(self, payload):
        self._payload = payload
        self.app = SimpleNamespace()

    async def json(self):
        return self._payload


class _ProxyHarness:
    """Runs _handle_completions against stub prefill and decode services.

    Prefill blocks until released, which is what makes the reordering
    observable: decode either starts while prefill is still outstanding or it
    does not.
    """

    def __init__(self):
        self.prefill_seen = None
        self.decode_seen = None
        self.release_prefill = None
        self.prefill_started = None
        self.prefill_response = {"kv_transfer_params": dict(_ADMIT_PARAMS)}

    async def _prefill(self, client_info, endpoint, req_data, request_id):
        self.prefill_seen = req_data
        self.prefill_started.set()
        await self.release_prefill.wait()
        return SimpleNamespace(json=lambda: self.prefill_response)

    async def _decode(self, client_info, endpoint, req_data, request_id):
        self.decode_seen = dict(req_data)
        yield b'data: {"choices": [{"text": "hi"}]}\n\n'

    def patches(self):
        return [
            patch.object(_MOD, "send_request_to_prefill", self._prefill),
            patch.object(_MOD, "stream_from_decode", self._decode),
            patch.object(_MOD, "get_next_client",
                         lambda app, kind: {"client": None}),
            patch.object(
                _MOD,
                "global_args",
                # Deliberately different hosts. The proxy is a sidecar in the
                # prefill pod, so prefill dials it on localhost and decode
                # must be handed something it can actually reach.
                SimpleNamespace(
                    admit_callback_url="http://127.0.0.1:10000",
                    dst_callback_url="http://prefill-pod.test:10000"),
                create=True),
        ]

    async def run(self,
                  *,
                  reorder: bool,
                  admit: bool,
                  timeout_s: str = "30",
                  prefill_submit: bool = False):
        self.release_prefill = asyncio.Event()
        self.prefill_started = asyncio.Event()
        stack = []
        for ctx in self.patches():
            ctx.start()
            stack.append(ctx)
        stack.append(
            patch.object(_MOD, "_admission_reorder_enabled", lambda: reorder))
        stack[-1].start()
        stack.append(
            patch.dict(
                "os.environ", {
                    "PROXY_ADMIT_TIMEOUT_S": timeout_s,
                    "TPU_KV_PREFILL_SUBMIT": "1" if prefill_submit else "0",
                }))
        stack[-1].start()
        try:
            # The chat API rather than /v1/completions: that one renders the
            # prompt against prefill first, which is a second round trip this
            # test is not about.
            request = _FakeRequest({
                "model":
                "m",
                "messages": [{
                    "role": "user",
                    "content": "hello"
                }]
            })
            handler = asyncio.ensure_future(
                _MOD._handle_completions("/v1/chat/completions", request))
            await asyncio.wait_for(self.prefill_started.wait(), 5)

            if admit:
                token = self.prefill_seen["kv_transfer_params"][
                    "admit_callback_url"].rsplit("/", 1)[1]
                await _MOD.handle_kv_admit(token,
                                           _FakeRequest(dict(_ADMIT_PARAMS)))

            response = await asyncio.wait_for(handler, 10)
            chunks = [c async for c in response.body_iterator]
            # Decode has now been dispatched. Whether prefill had finished
            # first is the whole question, so release it only here.
            decode_before_prefill = not self.release_prefill.is_set()
            self.release_prefill.set()
            await asyncio.sleep(0)
            return chunks, decode_before_prefill
        finally:
            self.release_prefill.set()
            for ctx in reversed(stack):
                ctx.stop()


def test_admission_reorder_starts_decode_before_prefill_returns() -> None:
    harness = _ProxyHarness()
    chunks, decode_before_prefill = asyncio.run(
        harness.run(reorder=True, admit=True))

    assert chunks and b"hi" in chunks[0]
    assert decode_before_prefill
    # Decode must receive the announced parameters, and must not receive the
    # callback URL: that field is between the proxy and prefill.
    assert harness.decode_seen["kv_transfer_params"] == _ADMIT_PARAMS
    assert "admit_callback_url" not in harness.decode_seen[
        "kv_transfer_params"]
    assert _MOD._ADMIT_WAITERS == {}


def test_each_side_is_told_an_address_it_can_actually_reach() -> None:
    """The proxy is a sidecar in the prefill pod, so the URL prefill polls
    and the URL decode posts to are different addresses. Handing decode
    prefill's localhost would point it at its own pod, where nothing is
    listening, and every request would fall back."""
    harness = _ProxyHarness()
    asyncio.run(harness.run(reorder=True, admit=True, prefill_submit=True))

    prefill_params = harness.prefill_seen["kv_transfer_params"]
    assert prefill_params["dst_batch_url"] == (
        "http://127.0.0.1:10000/kv_dst_batch")
    assert prefill_params["admit_callback_url"].startswith(
        "http://127.0.0.1:10000/kv_admit/")

    decode_params = harness.decode_seen["kv_transfer_params"]
    assert decode_params["dst_callback_url"] == (
        f"http://prefill-pod.test:10000/kv_dst/{_ADMIT_PARAMS['req_id']}")


def test_the_destination_relay_is_absent_when_the_feature_is_off() -> None:
    harness = _ProxyHarness()
    asyncio.run(harness.run(reorder=True, admit=True, prefill_submit=False))

    assert "dst_batch_url" not in harness.prefill_seen["kv_transfer_params"]
    assert "dst_callback_url" not in harness.decode_seen["kv_transfer_params"]


def test_admission_reorder_off_waits_for_prefill() -> None:
    """The stage gate on the proxy side: with the feature off, decode is not
    dispatched until prefill's response is in hand, exactly as today."""
    harness = _ProxyHarness()

    async def drive():
        task = asyncio.ensure_future(harness.run(reorder=False, admit=False))
        await asyncio.sleep(0.05)
        # Nothing can proceed while prefill is blocked.
        assert not task.done()
        assert harness.decode_seen is None
        harness.release_prefill.set()
        return await task

    chunks, decode_before_prefill = asyncio.run(drive())
    assert chunks and b"hi" in chunks[0]
    assert not decode_before_prefill
    assert "admit_callback_url" not in harness.prefill_seen.get(
        "kv_transfer_params", {})
    assert harness.decode_seen["kv_transfer_params"] == _ADMIT_PARAMS


def test_a_lost_admission_callback_falls_back_to_prefills_response() -> None:
    """A dropped announcement must cost latency, not the request."""
    harness = _ProxyHarness()

    async def drive():
        task = asyncio.ensure_future(
            harness.run(reorder=True, admit=False, timeout_s="0.05"))
        await asyncio.sleep(0.2)
        harness.release_prefill.set()
        return await task

    chunks, decode_before_prefill = asyncio.run(drive())
    assert chunks and b"hi" in chunks[0]
    assert not decode_before_prefill
    assert harness.decode_seen["kv_transfer_params"] == _ADMIT_PARAMS
    assert _MOD._ADMIT_WAITERS == {}


def test_a_late_admission_callback_is_ignored_rather_than_failing() -> None:

    async def drive():
        return await _MOD.handle_kv_admit("no-such-request",
                                          _FakeRequest(dict(_ADMIT_PARAMS)))

    assert asyncio.run(drive()) == {"status": "ignored"}


# ---------------------------------------------------------------------------
# Stage 5 — the destination relay: decode publishes where it wants its KV
# written, and prefill collects those in bulk over one long poll.
# ---------------------------------------------------------------------------

_DST_PARAMS = {
    "req_id": "engine-side-id",
    "uuid": 4242,
    "dst_block_ids": [7, 8, 9],
    "dst_controller_address": "decode-controller.test:28000",
}


def _reset_dst_state() -> None:
    _MOD._DST_PENDING.clear()
    _MOD._DST_SEQ = 0
    _MOD._DST_EVENT = None


def test_a_published_destination_reaches_the_next_poll_once() -> None:
    """Once per collector, not once in total.

    A collector reports the cursor it last saw and gets what is newer, so a
    second poll from the same one returns nothing while the entry itself
    stays. That is what lets prefill's scheduler and prefill's transfer rank
    read the same relay without stealing from each other.
    """
    _reset_dst_state()

    async def drive():
        await _MOD.handle_kv_dst("r-1", _FakeRequest(dict(_DST_PARAMS)))
        first = await _MOD.handle_kv_dst_batch(timeout=0.05)
        second = await _MOD.handle_kv_dst_batch(timeout=0.05,
                                                after=first["cursor"])
        return first, second

    first, second = asyncio.run(drive())
    assert first == {"entries": {"r-1": _DST_PARAMS}, "cursor": 1}
    assert second == {"entries": {}, "cursor": 1}
    assert "r-1" in _MOD._DST_PENDING


def test_a_second_collector_gets_everything_the_first_did() -> None:
    """The stage 7d requirement. The scheduler needs a destination to build
    an ordinary send and the worker needs one to release a held send, and
    neither may consume what the other is waiting for."""
    _reset_dst_state()

    async def drive():
        await _MOD.handle_kv_dst("r-1", _FakeRequest(dict(_DST_PARAMS)))
        engine = await _MOD.handle_kv_dst_batch(timeout=0.05)
        worker = await _MOD.handle_kv_dst_batch(timeout=0.05)
        return engine, worker

    engine, worker = asyncio.run(drive())
    assert engine == worker == {"entries": {"r-1": _DST_PARAMS}, "cursor": 1}


def test_one_poll_drains_every_announcement_since_the_last() -> None:
    """This is what keeps the relay at one connection per collector rather
    than one per in-flight request."""
    _reset_dst_state()

    async def drive():
        for index in range(3):
            await _MOD.handle_kv_dst(f"r-{index}",
                                     _FakeRequest({"uuid": index}))
        return await _MOD.handle_kv_dst_batch(timeout=0.05)

    assert asyncio.run(drive()) == {
        "entries": {
            "r-0": {
                "uuid": 0
            },
            "r-1": {
                "uuid": 1
            },
            "r-2": {
                "uuid": 2
            },
        },
        "cursor": 3,
    }


def test_a_poll_past_the_cursor_skips_what_it_already_has() -> None:
    _reset_dst_state()

    async def drive():
        for index in range(3):
            await _MOD.handle_kv_dst(f"r-{index}",
                                     _FakeRequest({"uuid": index}))
        return await _MOD.handle_kv_dst_batch(timeout=0.05, after=2)

    assert asyncio.run(drive()) == {
        "entries": {
            "r-2": {
                "uuid": 2
            }
        },
        "cursor": 3,
    }


def test_a_waiting_poll_wakes_on_the_next_publish() -> None:
    """Prefill must not learn about a destination only on its next reconnect;
    that announcement is on the critical path for its request."""
    _reset_dst_state()

    async def drive():
        poll = asyncio.ensure_future(_MOD.handle_kv_dst_batch(timeout=5.0))
        await asyncio.sleep(0.05)
        assert not poll.done()
        await _MOD.handle_kv_dst("r-late", _FakeRequest({"uuid": 9}))
        return await asyncio.wait_for(poll, timeout=1.0)

    assert asyncio.run(drive()) == {
        "entries": {
            "r-late": {
                "uuid": 9
            }
        },
        "cursor": 1,
    }


def test_two_waiting_polls_both_wake_on_one_publish() -> None:
    """With a single event that each collector cleared before waiting, one
    collector's clear could erase a set the other had not seen and leave it
    sitting out its whole timeout. Replacing the event instead of clearing it
    is what fixes that, and this is the test that would catch a regression."""
    _reset_dst_state()

    async def drive():
        engine = asyncio.ensure_future(_MOD.handle_kv_dst_batch(timeout=5.0))
        worker = asyncio.ensure_future(_MOD.handle_kv_dst_batch(timeout=5.0))
        await asyncio.sleep(0.05)
        await _MOD.handle_kv_dst("r-both", _FakeRequest({"uuid": 12}))
        return await asyncio.wait_for(asyncio.gather(engine, worker),
                                      timeout=1.0)

    engine, worker = asyncio.run(drive())
    assert engine == worker == {
        "entries": {
            "r-both": {
                "uuid": 12
            }
        },
        "cursor": 1,
    }


def test_an_idle_poll_returns_empty_rather_than_hanging() -> None:
    _reset_dst_state()
    assert asyncio.run(_MOD.handle_kv_dst_batch(timeout=0.05)) == {
        "entries": {},
        "cursor": 0,
    }


def test_a_publish_between_polls_is_still_there_on_reconnect() -> None:
    _reset_dst_state()

    async def drive():
        # No poll is outstanding at this moment.
        await _MOD.handle_kv_dst("r-gap", _FakeRequest({"uuid": 11}))
        return await _MOD.handle_kv_dst_batch(timeout=0.05)

    assert asyncio.run(drive()) == {
        "entries": {
            "r-gap": {
                "uuid": 11
            }
        },
        "cursor": 1,
    }


def test_an_entry_nobody_claimed_ages_out() -> None:
    """Nothing acknowledges an entry now, so the clock is what bounds the
    table when traffic is too light for the count cap to bite."""
    _reset_dst_state()

    async def drive():
        await _MOD.handle_kv_dst("r-stale", _FakeRequest({"uuid": 13}))
        seq, published, payload = _MOD._DST_PENDING["r-stale"]
        _MOD._DST_PENDING["r-stale"] = (seq, published - _MOD._DST_TTL_S - 1.0,
                                        payload)
        await _MOD.handle_kv_dst("r-fresh", _FakeRequest({"uuid": 14}))

    asyncio.run(drive())
    assert "r-stale" not in _MOD._DST_PENDING
    assert "r-fresh" in _MOD._DST_PENDING
    _reset_dst_state()


def test_an_uncollected_backlog_is_bounded() -> None:
    """If prefill stops collecting, the proxy sheds rather than growing. A
    request whose parameters are dropped falls back to decode's own submit."""
    _reset_dst_state()

    async def drive():
        for index in range(_MOD._DST_MAX_PENDING + 5):
            await _MOD.handle_kv_dst(f"r-{index}",
                                     _FakeRequest({"uuid": index}))

    asyncio.run(drive())
    assert len(_MOD._DST_PENDING) == _MOD._DST_MAX_PENDING
    # The oldest go first, so the requests still likely to be in flight stay.
    assert "r-0" not in _MOD._DST_PENDING
    assert f"r-{_MOD._DST_MAX_PENDING + 4}" in _MOD._DST_PENDING
    _reset_dst_state()
