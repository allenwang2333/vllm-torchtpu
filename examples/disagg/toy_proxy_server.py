# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import codecs
import itertools
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class _ProxyLatencyTracker:
    """Small running latency summary for the proxy path."""

    def __init__(self, log_interval_s: float):
        self._log_interval_s = log_interval_s
        self._lock = threading.Lock()
        self._stats: dict[str, list[float]] = {}
        self._next_log = time.monotonic() + log_interval_s

    def record(self, phase: str, ms: float) -> None:
        if self._log_interval_s <= 0:
            return
        msg: str | None = None
        with self._lock:
            stat = self._stats.setdefault(phase, [0.0, 0.0, ms, ms])
            stat[0] += 1.0
            stat[1] += ms
            stat[2] = min(stat[2], ms)
            stat[3] = max(stat[3], ms)

            now = time.monotonic()
            if now >= self._next_log:
                self._next_log = now + self._log_interval_s
                parts = []
                for key in sorted(self._stats):
                    n, total, min_ms, max_ms = self._stats[key]
                    parts.append(f"{key}: n={int(n)} avg={total / n:.2f}ms "
                                 f"min={min_ms:.2f}ms max={max_ms:.2f}ms")
                msg = f"PERF PROXY latency summary | {' | '.join(parts)}"
        if msg is not None:
            print(msg, flush=True)
            logger.info(msg)


def _proxy_latency_interval() -> float:
    val = os.getenv("PROXY_LATENCY_LOG_INTERVAL", "30")
    try:
        return float(val)
    except ValueError:
        return 30.0


_PROXY_LATENCY = _ProxyLatencyTracker(_proxy_latency_interval())

# Requests waiting to hear that prefill has admitted them. The value is
# resolved by the /kv_admit callback with the transfer parameters decode needs,
# which today arrive only in prefill's final response.
_ADMIT_WAITERS: dict[str, "asyncio.Future[dict]"] = {}

# Destination parameters decode has published. Prefill long-polls
# /kv_dst_batch and drains this in bulk, which keeps the relay at one
# connection per collector rather than one per in-flight request.
#
# Two collectors read this: prefill's scheduler, which needs a destination to
# build a send, and prefill's transfer rank, which needs one to release a send
# the doorbell has already fired. So an entry is not consumed on delivery.
# Each collector reports the sequence number it last saw and gets whatever is
# newer, and entries age out by count and by clock rather than by
# acknowledgement. The value is (sequence, publish time, payload).
_DST_PENDING: "OrderedDict[str, tuple[int, float, dict]]" = OrderedDict()
_DST_SEQ = 0
_DST_EVENT: "asyncio.Event | None" = None
# The cap and the age limit are both generous. A destination that nobody has
# claimed within five minutes belongs to a request that has long since timed
# out on both sides.
_DST_MAX_PENDING = 4096
_DST_TTL_S = 300.0


def _admission_reorder_enabled() -> bool:
    return os.getenv("TPU_KV_ADMISSION_REORDER",
                     "0").strip().lower() in ("1", "true", "yes", "on")


def _prefill_submit_enabled() -> bool:
    return os.getenv("TPU_KV_PREFILL_SUBMIT",
                     "0").strip().lower() in ("1", "true", "yes", "on")


def _admit_timeout_s() -> float:
    """How long to wait for the admission callback before falling back.

    The fallback is the ordinary path, so this bounds the damage from a lost
    callback to the latency the feature was meant to save. It has to exceed a
    prefill queueing delay, though, or a busy prefill never gets to announce.
    """
    try:
        return float(os.getenv("PROXY_ADMIT_TIMEOUT_S", "30"))
    except ValueError:
        return 30.0


def _render_endpoint_for_api(api: str) -> str | None:
    if api == "/v1/completions":
        return "/v1/completions/render"
    return None


def _token_ids_from_completion_render(
        rendered: Any) -> list[int] | list[list[int]]:
    if not isinstance(rendered, list) or not rendered:
        raise ValueError("Completion render response must be a non-empty list")

    prompts: list[list[int]] = []
    for item in rendered:
        if not isinstance(item, dict):
            raise ValueError("Completion render item must be an object")
        token_ids = item.get("token_ids")
        if (not isinstance(token_ids, list) or not all(
                type(token_id) is int and token_id >= 0
                for token_id in token_ids)):
            raise ValueError("Completion render item must contain token_ids")
        prompts.append(token_ids)

    return prompts[0] if len(prompts) == 1 else prompts


def _replace_prompt_with_rendered_token_ids(req_data: dict,
                                            rendered: Any) -> dict:
    req_data = req_data.copy()
    req_data["prompt"] = _token_ids_from_completion_render(rendered)
    req_data.pop("prompt_embeds", None)
    return req_data


def _json_escape_non_ascii(text: str) -> bytes:
    """Encode text as ASCII bytes while preserving JSON string semantics."""
    if text.isascii():
        return text.encode("ascii")

    parts = []
    for char in text:
        codepoint = ord(char)
        if codepoint < 128:
            parts.append(char)
        elif codepoint <= 0xFFFF:
            parts.append(f"\\u{codepoint:04x}")
        else:
            codepoint -= 0x10000
            high_surrogate = 0xD800 + (codepoint >> 10)
            low_surrogate = 0xDC00 + (codepoint & 0x3FF)
            parts.append(f"\\u{high_surrogate:04x}\\u{low_surrogate:04x}")
    return "".join(parts).encode("ascii")


class _AsciiSafeStreamEncoder:
    """Converts UTF-8 stream chunks to ASCII-only JSON-compatible chunks."""

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")()

    def encode(self, chunk: bytes = b"", *, final: bool = False) -> bytes:
        text = self._decoder.decode(chunk, final=final)
        if not text:
            return b""
        return _json_escape_non_ascii(text)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f'http://{host}:{port}'
        app.state.prefill_clients.append({
            'client':
            httpx.AsyncClient(timeout=None, base_url=prefiller_base_url),
            'host':
            host,
            'port':
            port,
            'id':
            i
        })

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f'http://{host}:{port}'
        app.state.decode_clients.append({
            'client':
            httpx.AsyncClient(timeout=None, base_url=decoder_base_url),
            'host':
            host,
            'port':
            port,
            'id':
            i
        })

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(
        range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(
        range(len(app.state.decode_clients)))

    print(f"Initialized {len(app.state.prefill_clients)} prefill clients "
          f"and {len(app.state.decode_clients)} decode clients.")

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info['client'].aclose()

    for client_info in app.state.decode_clients:
        await client_info['client'].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")

    # Where prefill should POST its admission announcements. It has to be an
    # address the prefill process can reach, which --host often is not: the
    # usual value there is a bind address, and localhost on the prefill host
    # is the prefill host.
    parser.add_argument("--admit-callback-url",
                        type=str,
                        default=os.getenv("PROXY_ADMIT_CALLBACK_URL", ""))

    # Where decode should POST the pages it wants written. This is a separate
    # address because the two sides sit in different places: the proxy is
    # commonly a sidecar in the prefill pod, so prefill reaches it on
    # localhost and decode cannot. Defaults to the admission URL, which is
    # correct whenever both servers see the proxy at the same address.
    parser.add_argument("--dst-callback-url",
                        type=str,
                        default=os.getenv("PROXY_DST_CALLBACK_URL", ""))

    # For prefiller instances
    parser.add_argument("--prefiller-hosts",
                        "--prefiller-host",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--prefiller-ports",
                        "--prefiller-port",
                        type=int,
                        nargs="+",
                        default=[8400])

    # For decoder instances
    parser.add_argument("--decoder-hosts",
                        "--decoder-host",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--decoder-ports",
                        "--decoder-port",
                        type=int,
                        nargs="+",
                        default=[9400])

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports")

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError(
            "Number of decoder hosts must match number of decoder ports")

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(
        zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    if not args.admit_callback_url:
        args.admit_callback_url = f"http://{args.host}:{args.port}"
    args.admit_callback_url = args.admit_callback_url.rstrip("/")
    if not args.dst_callback_url:
        args.dst_callback_url = args.admit_callback_url
    args.dst_callback_url = args.dst_callback_url.rstrip("/")

    return args


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == 'prefill':
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == 'decode':
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_prefill(client_info: dict, endpoint: str,
                                  req_data: dict, request_id: str):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    # Must overwrite these for prefill workers.
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }

    response = await client_info['client'].post(endpoint,
                                                json=req_data,
                                                headers=headers)
    response.raise_for_status()

    return response


async def render_completion_prompt(client_info: dict, endpoint: str,
                                   req_data: dict, request_id: str):
    """
    Render/tokenize a completion request once so downstream P/D requests can
    use prompt token IDs instead of retokenizing raw text independently.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }

    response = await client_info['client'].post(endpoint,
                                                json=req_data,
                                                headers=headers)
    response.raise_for_status()

    return response


async def stream_from_decode(client_info: dict, endpoint: str, req_data: dict,
                             request_id: str):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }

    async with client_info['client'].stream("POST",
                                            endpoint,
                                            json=req_data,
                                            headers=headers) as response:
        response.raise_for_status()
        encoder = _AsciiSafeStreamEncoder()
        async for chunk in response.aiter_bytes():
            chunk = encoder.encode(chunk)
            if chunk:
                yield chunk
        chunk = encoder.encode(final=True)
        if chunk:
            yield chunk


def _watch_prefill(task: "asyncio.Future", request_id: str, t_send_perf: float,
                   t_recv_perf: float) -> None:
    """Record prefill's outcome once decode has already been started.

    Nothing downstream is waiting on it, but an un-retrieved task exception is
    invisible, and prefill failing after decode has been dispatched is the one
    failure this reordering introduces.
    """

    def _done(fut: "asyncio.Future") -> None:
        prefill_ms = (time.perf_counter() - t_send_perf) * 1000.0
        _PROXY_LATENCY.record("prefill", prefill_ms)
        _PROXY_LATENCY.record("prefill_from_req",
                              (time.perf_counter() - t_recv_perf) * 1000.0)
        exc = None if fut.cancelled() else fut.exception()
        if exc is not None:
            print(
                f"PERF PROXY prefill_failed req_id={request_id} "
                f"dur_ms={prefill_ms:.2f} err={exc!r}",
                flush=True)
            logger.error("Prefill failed after decode was dispatched for %s",
                         request_id,
                         exc_info=exc)
        else:
            print(
                f"PERF PROXY prefill_recv req_id={request_id} "
                f"ts={time.time():.6f} dur_ms={prefill_ms:.2f}",
                flush=True)

    task.add_done_callback(_done)


@app.post("/kv_admit/{token}")
async def handle_kv_admit(token: str, request: Request):
    """Prefill's announcement that it has admitted a request.

    The body is the same parameter dict prefill would otherwise return at the
    end of its response. Arriving late or twice is harmless: the waiter is
    gone or already resolved, and the request has taken the ordinary path.
    """
    payload = await request.json()
    future = _ADMIT_WAITERS.get(token)
    if future is not None and not future.done():
        future.set_result(payload)
        return {"status": "ok"}
    return {"status": "ignored"}


def _dst_event() -> "asyncio.Event":
    """The event a collector's long poll waits on, created on the loop."""
    global _DST_EVENT
    if _DST_EVENT is None:
        _DST_EVENT = asyncio.Event()
    return _DST_EVENT


def _dst_publish() -> None:
    """Wake every waiting collector and install a fresh event behind them.

    Replacing the event is what makes a second collector safe. If they shared
    one event and each cleared it before waiting, one collector's clear could
    erase a set the other had not yet observed, and that collector would then
    sit out its whole timeout with an entry already waiting for it here.
    """
    global _DST_EVENT
    event = _DST_EVENT
    _DST_EVENT = None
    if event is not None:
        event.set()


def _dst_prune(now: float) -> None:
    """Drop entries that are too old or too many, oldest sequence first."""
    while _DST_PENDING:
        token = next(iter(_DST_PENDING))
        if (len(_DST_PENDING) <= _DST_MAX_PENDING
                and now - _DST_PENDING[token][1] <= _DST_TTL_S):
            return
        _DST_PENDING.pop(token, None)


def _dst_since(after: int) -> tuple[dict[str, dict], int]:
    """Entries published after the caller's cursor, and the new cursor.

    Insertion order is sequence order, so walking backwards stops at the
    first entry the caller has already seen and the steady-state cost is the
    number of new entries rather than the size of the table.
    """
    fresh: dict[str, dict] = {}
    for token in reversed(_DST_PENDING):
        seq, _, payload = _DST_PENDING[token]
        if seq <= after:
            break
        fresh[token] = payload
    return fresh, _DST_SEQ


@app.post("/kv_dst/{token}")
async def handle_kv_dst(token: str, request: Request):
    """Decode's announcement of where it wants this request's KV written.

    Raiden requires the coordination call to name the destination pages, so
    prefill cannot start a transfer until it has these. Measured on 397B,
    decode takes about 1.4 seconds to answer, which is longer than an 8192
    token prompt spends in prefill. So this is not slack the critical path
    can spend, and prefill's transfer rank waits on the arrival instead of
    requiring it up front.
    """
    global _DST_SEQ
    payload = await request.json()
    now = time.time()
    _DST_SEQ += 1
    _DST_PENDING.pop(token, None)
    _DST_PENDING[token] = (_DST_SEQ, now, payload)
    _dst_prune(now)
    _dst_publish()
    return {"status": "ok"}


@app.get("/kv_dst_batch")
async def handle_kv_dst_batch(timeout: float = 25.0, after: int = 0):
    """A collector's long poll for every announcement newer than its cursor.

    Returning the whole new set rather than one entry is what keeps this to a
    single connection per collector. The reply carries the cursor to send
    back next time; a collector that reconnects with a stale or zero cursor
    is re-sent whatever is still in the window, which is harmless because
    entries are keyed by request and applying one twice is a no-op.
    """
    entries, cursor = _dst_since(after)
    if entries:
        return {"entries": entries, "cursor": cursor}
    event = _dst_event()
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        return {"entries": {}, "cursor": cursor}
    entries, cursor = _dst_since(after)
    return {"entries": entries, "cursor": cursor}


async def _handle_completions(api: str, request: Request):
    try:
        t_request_recv_perf = time.perf_counter()
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        print(
            f"PERF PROXY req_received req_id={request_id} ts={time.time():.6f}",
            flush=True)

        # Get the next prefill client in round-robin fashion
        prefill_client_info = get_next_client(request.app, 'prefill')
        render_endpoint = _render_endpoint_for_api(api)
        if render_endpoint is not None:
            render_response = await render_completion_prompt(
                prefill_client_info, render_endpoint, req_data, request_id)
            req_data = _replace_prompt_with_rendered_token_ids(
                req_data, render_response.json())

        # Ask prefill to announce this request the moment it is admitted, so
        # decode can allocate and register while the prefill is still running
        # instead of afterwards.
        admit_future: "asyncio.Future[dict] | None" = None
        if _admission_reorder_enabled():
            admit_future = asyncio.get_running_loop().create_future()
            _ADMIT_WAITERS[request_id] = admit_future
            kv_params_out = dict(req_data.get("kv_transfer_params") or {})
            kv_params_out["admit_callback_url"] = (
                f"{global_args.admit_callback_url}/kv_admit/{request_id}")
            if _prefill_submit_enabled():
                # Where prefill collects decode's destination pages. One URL
                # for the engine, not one per request: prefill holds a single
                # long poll and drains whatever has arrived.
                kv_params_out["dst_batch_url"] = (
                    f"{global_args.admit_callback_url}/kv_dst_batch")
            req_data["kv_transfer_params"] = kv_params_out

        # Send request to prefill service
        t_prefill_send = time.time()
        t_prefill_send_perf = time.perf_counter()
        print(
            f"PERF PROXY prefill_send req_id={request_id} ts={t_prefill_send:.6f}",
            flush=True)
        # Prefill gets its own copy: req_data is rewritten below with the
        # parameters decode needs, and with the reorder on that happens while
        # the prefill request is still in flight.
        prefill_task = asyncio.ensure_future(
            send_request_to_prefill(prefill_client_info, api, dict(req_data),
                                    request_id))

        kv_transfer_params: dict | None = None
        try:
            if admit_future is not None:
                await asyncio.wait({admit_future, prefill_task},
                                   return_when=asyncio.FIRST_COMPLETED,
                                   timeout=_admit_timeout_s())
                if not prefill_task.done() and admit_future.done():
                    kv_transfer_params = admit_future.result()
                    t_admit = time.time()
                    admit_ms = (time.perf_counter() -
                                t_prefill_send_perf) * 1000.0
                    _PROXY_LATENCY.record("prefill_admit", admit_ms)
                    print(
                        f"PERF PROXY prefill_admit req_id={request_id} "
                        f"ts={t_admit:.6f} dur_ms={admit_ms:.2f} "
                        f"kv_uuid={kv_transfer_params.get('uuid')}",
                        flush=True)
                elif not admit_future.done():
                    print(
                        f"PERF PROXY admit_fallback req_id={request_id} "
                        "reason=prefill_first_or_timeout",
                        flush=True)

            if kv_transfer_params is None:
                # Either the reorder is off or its callback did not arrive in
                # time. Both land on the original path: read the parameters
                # out of prefill's response.
                response = await prefill_task
                t_prefill_recv = time.time()
                t_prefill_recv_perf = time.perf_counter()
                prefill_ms = (t_prefill_recv_perf -
                              t_prefill_send_perf) * 1000.0
                _PROXY_LATENCY.record("prefill", prefill_ms)
                _PROXY_LATENCY.record(
                    "prefill_from_req",
                    (t_prefill_recv_perf - t_request_recv_perf) * 1000.0)
                kv_transfer_params = response.json().get(
                    'kv_transfer_params', {})
                kv_uuid = (kv_transfer_params.get("uuid")
                           if kv_transfer_params else None)
                kv_uuid_log = (f" kv_uuid={kv_uuid}"
                               if kv_uuid is not None else "")
                print(
                    f"PERF PROXY prefill_recv req_id={request_id} "
                    f"ts={t_prefill_recv:.6f} "
                    f"dur_ms={prefill_ms:.2f}{kv_uuid_log}",
                    flush=True)
            else:
                # Decode goes now; prefill's response is still worth reading,
                # but only to log a failure that decode would otherwise report
                # as an expired registration wait.
                _watch_prefill(prefill_task, request_id, t_prefill_send_perf,
                               t_request_recv_perf)
        finally:
            _ADMIT_WAITERS.pop(request_id, None)

        kv_uuid = kv_transfer_params.get(
            "uuid") if kv_transfer_params else None
        kv_uuid_log = f" kv_uuid={kv_uuid}" if kv_uuid is not None else ""

        if kv_transfer_params:
            if _prefill_submit_enabled():
                # Decode posts its destination pages back here, keyed by the
                # producer's own request id so prefill can match them to the
                # registration it is about to make.
                kv_transfer_params = dict(kv_transfer_params)
                kv_transfer_params["dst_callback_url"] = (
                    f"{global_args.dst_callback_url}/kv_dst/"
                    f"{kv_transfer_params.get('req_id', request_id)}")
            req_data["kv_transfer_params"] = kv_transfer_params

        # Get the next decode client in round-robin fashion
        decode_client_info = get_next_client(request.app, 'decode')

        logger.debug("Using %s %s", prefill_client_info, decode_client_info)

        # Stream response from decode service
        async def generate_stream():
            t_decode_send = time.time()
            t_decode_send_perf = time.perf_counter()
            print(
                f"PERF PROXY decode_send req_id={request_id} "
                f"ts={t_decode_send:.6f}{kv_uuid_log}",
                flush=True)
            first = True
            async for chunk in stream_from_decode(decode_client_info,
                                                  api,
                                                  req_data,
                                                  request_id=request_id):
                if first:
                    t_first = time.time()
                    t_first_perf = time.perf_counter()
                    decode_first_ms = (t_first_perf -
                                       t_decode_send_perf) * 1000.0
                    ttft_ms = (t_first_perf - t_request_recv_perf) * 1000.0
                    _PROXY_LATENCY.record("decode_first", decode_first_ms)
                    _PROXY_LATENCY.record("ttft", ttft_ms)
                    print(
                        f"PERF PROXY first_chunk req_id={request_id} "
                        f"ts={t_first:.6f} "
                        f"dur_from_decode_send_ms={decode_first_ms:.2f} "
                        f"ttft_ms={ttft_ms:.2f}{kv_uuid_log}",
                        flush=True)
                    first = False
                yield chunk

        return StreamingResponse(generate_stream(),
                                 media_type="application/json")

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server"
              f" - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/v1/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/v1/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients)
    }


if __name__ == '__main__':
    global global_args
    global_args = parse_args()

    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
