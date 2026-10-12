# SPDX-License-Identifier: Apache-2.0
"""Session/prefix-affine HTTP router for N independent vLLM servers.

Serves one model as several independent single-host replicas behind one endpoint
(e.g. Qwen3.5-397B AgentX on 8 tpu7x chips as 2 x TP8 servers) while keeping each
conversation on one replica so its prefix cache stays warm. Streaming responses are
passed through unchanged.

Policies (--policy / ROUTER_POLICY):
  session      Sticky by a per-conversation header (--session-header, default
               X-Correlation-ID, which aiperf sends per conversation instance). A
               new session goes to the backend with the fewest in-flight requests
               (ties: fewest sessions), then stays there. Requests without the
               header fall back to `prefix`.
  prefix       Longest-prefix match on chained hashes of the request's
               messages/prompt, with a load guard: if the best-matching backend has
               more than --balance-abs in-flight requests above the least-loaded
               one, the least-loaded one wins.
  round_robin  No affinity. Only for measuring what affinity is worth.

DP-rank mode (--dp-rank-header, e.g. X-data-parallel-rank): for one vLLM API server
fronting several DP engines, list that server once per rank in --backends; every
POST then carries the chosen index in that header, which vLLM uses to pin the
request to that engine.

Endpoints:
  POST /v1/*          proxied (streaming) to the chosen backend
  GET  /health        200 only when every distinct backend answers /health
  GET  /router/stats  JSON counters (per-backend requests, sessions, in-flight)
  anything else       proxied to backend 0

Example:
  python3 session_router.py --port 8000 --stats-file /tmp/router.jsonl \\
      --backends http://server-0:8000 http://server-1:8000

Every flag also reads a ROUTER_* environment variable as its default (see --help).
"""

import argparse
import asyncio
import collections
import hashlib
import itertools
import json
import os
import sys
from dataclasses import dataclass

import aiohttp
from aiohttp import web

POLICIES = ("session", "prefix", "round_robin")

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


@dataclass
class RouterConfig:
    backends: list[str]
    policy: str = "session"
    session_header: str = "X-Correlation-ID"
    chunk_chars: int = 4096
    balance_abs: int = 8
    lru_entries: int = 2_000_000
    dp_rank_header: str = ""

    def __post_init__(self):
        self.backends = [u.rstrip("/") for u in self.backends]
        if not self.backends:
            raise ValueError("at least one backend is required")
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {self.policy!r}")


class Router:
    """Routing state and decisions; no I/O, so it can be unit tested directly."""

    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        n = len(cfg.backends)
        self.inflight = [0] * n
        self.requests = [0] * n
        self.errors = [0] * n
        self.sessions: dict[str, int] = {}
        self.sessions_per_backend = [0] * n
        self.prefix_lru = [collections.OrderedDict() for _ in range(n)]
        self.prefix_chunks_total = 0
        self.prefix_chunks_matched = 0
        self.fallback_no_header = 0
        self._rr = itertools.cycle(range(n))

    def least_loaded(self) -> int:
        return min(
            range(len(self.cfg.backends)),
            key=lambda i: (self.inflight[i], self.sessions_per_backend[i], i),
        )

    def chunk_hashes(self, body: bytes) -> list[bytes]:
        """Chained hashes over the prompt text, chunk_chars at a time."""
        try:
            obj = json.loads(body)
            text = json.dumps(
                obj.get("messages", obj.get("prompt", "")),
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except Exception:  # noqa: BLE001 - non-JSON bodies are hashed as raw text
            text = body.decode("utf-8", "replace")
        size = self.cfg.chunk_chars
        out, h = [], hashlib.blake2b(digest_size=16)
        for i in range(0, len(text) - len(text) % size, size):
            h.update(text[i : i + size].encode("utf-8", "replace"))
            out.append(h.copy().digest())
        return out

    def pick_by_prefix(self, hashes: list[bytes]) -> int:
        best, best_len = None, -1
        for i, lru in enumerate(self.prefix_lru):
            n = 0
            for d in hashes:
                if d in lru:
                    n += 1
                else:
                    break
            if n > best_len or (
                n == best_len and self.inflight[i] < self.inflight[best]
            ):
                best, best_len = i, n
        least = self.least_loaded()
        if (
            best_len <= 0
            or self.inflight[best] - self.inflight[least] > self.cfg.balance_abs
        ):
            best, best_len = least, 0
        self.prefix_chunks_total += len(hashes)
        self.prefix_chunks_matched += max(best_len, 0)
        return best

    def remember_prefix(self, idx: int, hashes: list[bytes]) -> None:
        lru = self.prefix_lru[idx]
        for d in hashes:
            lru[d] = None
            lru.move_to_end(d)
        while len(lru) > self.cfg.lru_entries:
            lru.popitem(last=False)

    def choose(self, headers, body: bytes) -> int:
        """Backend index for one POST request."""
        if self.cfg.policy == "round_robin":
            return next(self._rr)
        if self.cfg.policy == "session":
            sid = headers.get(self.cfg.session_header)
            if sid:
                idx = self.sessions.get(sid)
                if idx is None:
                    idx = self.least_loaded()
                    self.sessions[sid] = idx
                    self.sessions_per_backend[idx] += 1
                return idx
            self.fallback_no_header += 1
        hashes = self.chunk_hashes(body)
        idx = self.pick_by_prefix(hashes)
        self.remember_prefix(idx, hashes)
        return idx

    def upstream_headers(self, headers, idx: int, method: str) -> dict[str, str]:
        out = {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}
        if self.cfg.dp_rank_header and method == "POST":
            out[self.cfg.dp_rank_header] = str(idx)
        return out

    def snapshot(self) -> dict:
        return {
            "policy": self.cfg.policy,
            "backends": self.cfg.backends,
            "requests": self.requests,
            "inflight": self.inflight,
            "errors": self.errors,
            "sessions_per_backend": self.sessions_per_backend,
            "sessions_total": len(self.sessions),
            "fallback_no_header": self.fallback_no_header,
            "prefix_chunk_match_rate": (
                self.prefix_chunks_matched / self.prefix_chunks_total
                if self.prefix_chunks_total
                else None
            ),
        }


def build_app(
    router: Router, stats_file: str = "", stats_interval: float = 30.0
) -> web.Application:
    cfg = router.cfg

    async def proxy(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        idx = router.choose(request.headers, body) if request.method == "POST" else 0
        url = cfg.backends[idx] + request.rel_url.path_qs
        headers = router.upstream_headers(request.headers, idx, request.method)
        router.inflight[idx] += 1
        router.requests[idx] += 1
        resp = None
        try:
            async with request.app["client"].request(
                request.method, url, data=body, headers=headers
            ) as upstream:
                resp = web.StreamResponse(
                    status=upstream.status,
                    headers={
                        k: v
                        for k, v in upstream.headers.items()
                        if k.lower() not in HOP_BY_HOP
                    },
                )
                await resp.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        except Exception as e:  # noqa: BLE001 - report any backend failure as 502
            router.errors[idx] += 1
            print(
                f"[session_router] backend {idx} error: {e!r}",
                file=sys.stderr,
                flush=True,
            )
            if resp is not None and resp.prepared:
                raise
            return web.Response(
                status=502, text=f"session_router: backend {idx} failed: {e!r}"
            )
        finally:
            router.inflight[idx] -= 1

    async def health(request: web.Request) -> web.Response:
        client = request.app["client"]
        for u in dict.fromkeys(cfg.backends):
            try:
                async with client.get(
                    u + "/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    if r.status != 200:
                        return web.Response(status=503, text=f"{u} -> {r.status}")
            except Exception as e:  # noqa: BLE001
                return web.Response(status=503, text=f"{u} -> {e!r}")
        return web.Response(text="ok")

    async def stats(_request: web.Request) -> web.Response:
        return web.json_response(router.snapshot())

    async def stats_logger(_app):
        while True:
            await asyncio.sleep(stats_interval)
            line = json.dumps(router.snapshot())
            print(f"[session_router] stats {line}", flush=True)
            if stats_file:
                with open(stats_file, "a") as f:
                    f.write(line + "\n")

    async def on_startup(app):
        app["client"] = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30),
            connector=aiohttp.TCPConnector(limit=0, ttl_dns_cache=300),
            auto_decompress=False,
        )
        app["stats_task"] = asyncio.create_task(stats_logger(app))

    async def on_cleanup(app):
        app["stats_task"].cancel()
        await app["client"].close()

    app = web.Application(client_max_size=1 << 30)
    app.router.add_get("/health", health)
    app.router.add_get("/router/stats", stats)
    app.router.add_route("*", "/{tail:.*}", proxy)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def parse_args(argv=None) -> argparse.Namespace:
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--backends",
        nargs="+",
        default=env("ROUTER_BACKENDS", "").split() or None,
        help="backend base URLs (env ROUTER_BACKENDS, space separated)",
    )
    p.add_argument(
        "--policy", choices=POLICIES, default=env("ROUTER_POLICY", "session")
    )
    p.add_argument("--host", default=env("ROUTER_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(env("ROUTER_PORT", "8000")))
    p.add_argument(
        "--session-header", default=env("ROUTER_SESSION_HEADER", "X-Correlation-ID")
    )
    p.add_argument(
        "--chunk-chars", type=int, default=int(env("ROUTER_PREFIX_CHUNK_CHARS", "4096"))
    )
    p.add_argument(
        "--balance-abs", type=int, default=int(env("ROUTER_BALANCE_ABS", "8"))
    )
    p.add_argument(
        "--lru-entries",
        type=int,
        default=int(env("ROUTER_PREFIX_LRU_ENTRIES", "2000000")),
    )
    p.add_argument(
        "--dp-rank-header",
        default=env("ROUTER_DP_RANK_HEADER", ""),
        help="e.g. X-data-parallel-rank: send the chosen backend index to vLLM",
    )
    p.add_argument("--stats-file", default=env("ROUTER_STATS_FILE", ""))
    p.add_argument(
        "--stats-interval",
        type=float,
        default=float(env("ROUTER_STATS_INTERVAL", "30")),
    )
    args = p.parse_args(argv)
    if not args.backends:
        p.error("--backends (or ROUTER_BACKENDS) is required")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = RouterConfig(
        backends=args.backends,
        policy=args.policy,
        session_header=args.session_header,
        chunk_chars=args.chunk_chars,
        balance_abs=args.balance_abs,
        lru_entries=args.lru_entries,
        dp_rank_header=args.dp_rank_header,
    )
    try:
        import uvloop  # noqa: PLC0415 - optional speedup

        uvloop.install()
    except ImportError:
        pass
    print(
        f"[session_router] policy={cfg.policy} listen={args.host}:{args.port} "
        f"backends={cfg.backends} dp_rank_header={cfg.dp_rank_header!r}",
        flush=True,
    )
    web.run_app(
        build_app(Router(cfg), args.stats_file, args.stats_interval),
        host=args.host,
        port=args.port,
        access_log=None,
    )


if __name__ == "__main__":
    main()
