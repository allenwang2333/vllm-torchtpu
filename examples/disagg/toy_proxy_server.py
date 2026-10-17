# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import codecs
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import itertools
import logging
import os
import threading
import time
from typing import Any
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class _PodMetricsSnapshot:
    running_reqs: float = 0.0
    waiting_reqs: float = 0.0
    gpu_cache_usage: float = 0.0
    updated_at: float = 0.0


class _BackendMetricsScraper:
    """Continuously polls vLLM Prometheus /metrics on Prefill and Decode pods.

    Provides live vllm:num_requests_running, vllm:num_requests_waiting, and
    vllm:gpu_cache_usage_perc for llm-d active-request-scorer and
    kv-cache-utilization-scorer.
    """

    def __init__(
        self,
        num_prefill: int,
        num_decode: int,
        poll_interval_s: float = 0.5,
        stale_after_s: float = 3.0,
    ) -> None:
        self.num_prefill = num_prefill
        self.num_decode = num_decode
        self.poll_interval_s = max(0.2, poll_interval_s)
        self.stale_after_s = max(1.0, stale_after_s)
        self._lock = threading.Lock()
        self._prefill: list[_PodMetricsSnapshot] = [
            _PodMetricsSnapshot() for _ in range(num_prefill)
        ]
        self._decode: list[_PodMetricsSnapshot] = [
            _PodMetricsSnapshot() for _ in range(num_decode)
        ]
        self._stop_event: asyncio.Event | None = None
        self._task: asyncio.Task | None = None

    @staticmethod
    def parse_prometheus_text(text: str) -> tuple[float, float, float]:
        running = 0.0
        waiting = 0.0
        kv_usage = 0.0
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            if line.startswith("vllm:num_requests_running"):
                parts = line.rsplit(None, 1)
                if len(parts) == 2:
                    try:
                        running += float(parts[1])
                    except ValueError:
                        pass
            elif line.startswith("vllm:num_requests_waiting"):
                parts = line.rsplit(None, 1)
                if len(parts) == 2:
                    try:
                        waiting += float(parts[1])
                    except ValueError:
                        pass
            elif line.startswith("vllm:gpu_cache_usage_perc") or line.startswith(
                "vllm:kv_cache_usage_perc"
            ):
                parts = line.rsplit(None, 1)
                if len(parts) == 2:
                    try:
                        val = float(parts[1])
                        if val > 1.0:
                            val = val / 100.0
                        kv_usage = max(kv_usage, min(1.0, max(0.0, val)))
                    except ValueError:
                        pass
        return running, waiting, kv_usage

    def update_prefill(
        self, rank: int, running: float, waiting: float, kv_usage: float
    ) -> None:
        now = time.monotonic()
        with self._lock:
            if 0 <= rank < self.num_prefill:
                self._prefill[rank] = _PodMetricsSnapshot(
                    running_reqs=running,
                    waiting_reqs=waiting,
                    gpu_cache_usage=kv_usage,
                    updated_at=now,
                )

    def update_decode(
        self, rank: int, running: float, waiting: float, kv_usage: float
    ) -> None:
        now = time.monotonic()
        with self._lock:
            if 0 <= rank < self.num_decode:
                self._decode[rank] = _PodMetricsSnapshot(
                    running_reqs=running,
                    waiting_reqs=waiting,
                    gpu_cache_usage=kv_usage,
                    updated_at=now,
                )

    def get_prefill(self, rank: int) -> _PodMetricsSnapshot | None:
        now = time.monotonic()
        with self._lock:
            if 0 <= rank < self.num_prefill:
                snap = self._prefill[rank]
                if (
                    snap.updated_at > 0
                    and (now - snap.updated_at) <= self.stale_after_s
                ):
                    return snap
        return None

    def get_decode(self, rank: int) -> _PodMetricsSnapshot | None:
        now = time.monotonic()
        with self._lock:
            if 0 <= rank < self.num_decode:
                snap = self._decode[rank]
                if (
                    snap.updated_at > 0
                    and (now - snap.updated_at) <= self.stale_after_s
                ):
                    return snap
        return None

    async def _scrape_one(
        self, client_info: dict[str, Any], is_prefill: bool, rank: int
    ) -> None:
        try:
            resp = await client_info["client"].get("/metrics", timeout=0.35)
            if resp.status_code == 200:
                running, waiting, kv_usage = self.parse_prometheus_text(resp.text)
                if is_prefill:
                    self.update_prefill(rank, running, waiting, kv_usage)
                else:
                    self.update_decode(rank, running, waiting, kv_usage)
        except Exception:
            pass

    async def _poll_loop(
        self,
        prefill_clients: list[dict[str, Any]],
        decode_clients: list[dict[str, Any]],
    ) -> None:
        while self._stop_event is not None and not self._stop_event.is_set():
            tasks = []
            for i, c in enumerate(prefill_clients):
                tasks.append(self._scrape_one(c, True, i))
            for i, c in enumerate(decode_clients):
                tasks.append(self._scrape_one(c, False, i))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.poll_interval_s
                )
            except asyncio.TimeoutError:
                pass

    def start(
        self,
        prefill_clients: list[dict[str, Any]],
        decode_clients: list[dict[str, Any]],
    ) -> None:
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._poll_loop(prefill_clients, decode_clients)
        )

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            try:
                await self._task
            except Exception:
                pass


class _PrefixAffinityRouter:
    """llm-d EndpointPicker (EPP) Router adapted for TPU v7x xPyD Disaggregation.

    Implements two distinct routing profiles matching llm-d:
    1. Prefill Profile (route_prefill):
       - 640-token block-aligned 2-tier per-pod bounded LRUs:
         - Tier 1 (HBM): 4,884 blocks (3,125,760 tokens = 41.4 GB per P pod)
         - Tier 2 (DRAM): 47,500 blocks (30,400,000 tokens = 760 GB per P pod)
       - Tracks both in-flight requests AND in-flight uncached/DRAM tokens
         (inflight-load-producer + token-load-scorer).
       - prefix-cache-affinity-filter TTFT cost model (peakPrefillThroughput
         R_P = 42,000 tok/s, R_DRAM = 280,000 tok/s): stays sticky to the pod
         holding the conversation prefix in HBM/DRAM unless waiting for its
         queued tokens is slower than recomputing on an idle pod.
    2. Decode Profile (route_decode, evaluated AFTER prefill completes):
       - Per-pod bounded HBM LRU: 4,300 blocks (2,752,000 tokens = 36.5 GB per D pod).
       - Combines decode-prefix-cache-scorer (w=2.0), active-request-scorer
         around the 32-slot Megakernel ceiling (w=3.0), and live /metrics
         kv-cache-utilization-scorer (w=2.0).
       - Preserves multi-turn HBM locality on Decode (preventing 36.5 GB HBM
         duplication across Decode pods and avoiding full-prefix Raiden pulls)
         while balancing cold conversations across Decode pods by live MK slot
         and KV cache availability.
    """

    def __init__(
        self,
        n: int,
        chunk: int = 2560,
        min_frac: float = 0.40,
        slack: int = 10,
        role: str = "prefill",
        block_tokens: int = 640,
        hbm_max_blocks: int = 7739,
        dram_max_blocks: int = 15735,
        peak_prefill_tok_s: float = 42000.0,
        dram_load_tok_s: float = 280000.0,
        mk_ceiling_reqs: float = 40.0,
    ) -> None:
        self.n = n
        self.chunk = max(256, chunk)
        self.min_frac = min_frac
        self.slack = slack
        self.role = role
        self.block_tokens = max(1, block_tokens)
        self.hbm_max_blocks = max(1, hbm_max_blocks)
        self.dram_max_blocks = max(0, dram_max_blocks)
        self.peak_prefill_tok_s = max(1000.0, peak_prefill_tok_s)
        self.dram_load_tok_s = max(10000.0, dram_load_tok_s)
        self.mk_ceiling_reqs = max(1.0, mk_ceiling_reqs)

        self._lock = threading.Lock()
        self.hbm_lru: list[OrderedDict[bytes, None]] = [
            OrderedDict() for _ in range(n)
        ]
        self.dram_lru: list[OrderedDict[bytes, None]] = [
            OrderedDict() for _ in range(n)
        ]
        self.inflight = [0] * n
        self.inflight_uncached_tokens = [0] * n
        self.inflight_dram_tokens = [0] * n
        self.placed = [0] * n
        self.sticky = 0
        self.cold = 0
        self.spilled = 0

    @staticmethod
    def extract_text(body: dict[str, Any]) -> str:
        if msgs := body.get("messages"):
            parts: list[str] = []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "")
                c = m.get("content", "")
                if isinstance(c, str):
                    parts.append(f"{role}:{c}")
                elif isinstance(c, list):
                    sub = "".join(p.get("text", "") for p in c if isinstance(p, dict))
                    parts.append(f"{role}:{sub}")
            return "\n".join(parts)
        p = body.get("prompt", "")
        if isinstance(p, str):
            return p
        if isinstance(p, list):
            if p and isinstance(p[0], int):
                return ",".join(map(str, p))
            return "\n".join(str(x) for x in p)
        return ""

    def extract_blocks_and_tokens(
        self, body: dict[str, Any]
    ) -> tuple[list[bytes], int]:
        """Extract 640-token-aligned cumulative block hashes and estimated token count."""
        p = body.get("prompt")
        if isinstance(p, list) and p and isinstance(p[0], int):
            total_tokens = max(1, len(p))
            out: list[bytes] = []
            h = hashlib.blake2b(digest_size=8)
            step = self.block_tokens
            for i in range(0, len(p) - step + 1, step):
                chunk_bytes = ",".join(map(str, p[i : i + step])).encode("ascii")
                h.update(chunk_bytes)
                out.append(h.digest())
            if not out and p:
                h.update(",".join(map(str, p[: min(len(p), 160)])).encode("ascii"))
                out.append(h.digest())
            return out, total_tokens

        text = self.extract_text(body)
        raw = text.encode("utf-8", errors="ignore")
        total_tokens = max(64, len(raw) // 4)
        out = []
        h = hashlib.blake2b(digest_size=8)
        for i in range(0, len(raw) - self.chunk + 1, self.chunk):
            h.update(raw[i : i + self.chunk])
            out.append(h.digest())
        if not out and len(raw) >= 64:
            h.update(raw[: min(len(raw), 640)])
            out.append(h.digest())
        return out, total_tokens

    def hashes(self, text: str) -> list[bytes]:
        raw = text.encode("utf-8", errors="ignore")
        out: list[bytes] = []
        h = hashlib.blake2b(digest_size=8)
        for i in range(0, len(raw) - self.chunk + 1, self.chunk):
            h.update(raw[i : i + self.chunk])
            out.append(h.digest())
        if not out and len(raw) >= 64:
            h.update(raw[: min(len(raw), 640)])
            out.append(h.digest())
        return out

    def _match_pod_prefix(self, rank: int, hs: list[bytes]) -> tuple[int, int]:
        """Count contiguous prefix blocks in pod rank HBM LRU and DRAM LRU."""
        if not hs:
            return 0, 0
        hbm = self.hbm_lru[rank]
        dram = self.dram_lru[rank]
        hbm_hits = 0
        dram_hits = 0
        for d in hs:
            if d in hbm:
                hbm_hits += 1
            elif self.dram_max_blocks > 0 and d in dram:
                dram_hits += 1
            else:
                break
        return hbm_hits, dram_hits

    def _record_pod_blocks(self, rank: int, hs: list[bytes]) -> None:
        """Insert/touch prefix blocks in pod rank bounded HBM + DRAM LRUs."""
        if not hs:
            return
        hbm = self.hbm_lru[rank]
        dram = self.dram_lru[rank]
        for d in hs:
            if d in hbm:
                hbm.move_to_end(d)
            else:
                hbm[d] = None
                if len(hbm) > self.hbm_max_blocks:
                    evicted, _ = hbm.popitem(last=False)
                    if self.dram_max_blocks > 0:
                        if evicted in dram:
                            dram.move_to_end(evicted)
                        else:
                            dram[evicted] = None
            if self.dram_max_blocks > 0:
                if d in dram:
                    dram.move_to_end(d)
                else:
                    dram[d] = None
                while len(dram) > self.dram_max_blocks:
                    dram.popitem(last=False)

    def route_prefill(
        self,
        hs: list[bytes] | None,
        total_tokens: int = 4096,
        scraper: _BackendMetricsScraper | None = None,
    ) -> tuple[int, bool, int, int]:
        """llm-d Prefill Profile: 2-tier cache affinity + token-load TTFT cost model."""
        hs = hs or []
        total_tokens = max(64, total_tokens)
        if self.n <= 0:
            return 0, True, total_tokens, 0

        with self._lock:
            num_blocks = max(1, len(hs))
            est_ttft_s: list[float] = [0.0] * self.n
            eff_cache_frac: list[float] = [0.0] * self.n
            matched_toks: list[int] = [0] * self.n
            uncached_toks: list[int] = [0] * self.n
            dram_toks: list[int] = [0] * self.n
            q_reqs: list[int] = [0] * self.n

            for r in range(self.n):
                hbm_hits, dram_hits = self._match_pod_prefix(r, hs)
                hbm_frac = hbm_hits / num_blocks if hs else 0.0
                dram_frac = dram_hits / num_blocks if hs else 0.0
                eff_cache_frac[r] = hbm_frac + 0.85 * dram_frac

                h_tok = int(total_tokens * hbm_frac)
                d_tok = int(total_tokens * dram_frac)
                u_tok = max(640, total_tokens - h_tok - d_tok)

                matched_toks[r] = h_tok + d_tok
                uncached_toks[r] = u_tok
                dram_toks[r] = d_tok

                reqs_r = self.inflight[r]
                if scraper is not None:
                    snap = scraper.get_prefill(r)
                    if snap is not None:
                        reqs_r = max(
                            reqs_r, int(snap.running_reqs + snap.waiting_reqs)
                        )
                q_reqs[r] = reqs_r

                queue_wait_s = (
                    (self.inflight_uncached_tokens[r] / self.peak_prefill_tok_s)
                    + (self.inflight_dram_tokens[r] / self.dram_load_tok_s)
                    + (reqs_r * 0.015)
                )
                exec_s = (u_tok / self.peak_prefill_tok_s) + (
                    d_tok / self.dram_load_tok_s
                )
                est_ttft_s[r] = queue_wait_s + exec_s

            best_cache_rank = max(
                range(self.n),
                key=lambda r: (
                    round(eff_cache_frac[r], 4),
                    -round(est_ttft_s[r], 3),
                    -q_reqs[r],
                    -self.placed[r],
                    -r,
                ),
            )
            min_ttft = min(est_ttft_s)
            min_reqs = min(q_reqs)

            has_warm_prefix = (
                bool(hs)
                and matched_toks[best_cache_rank] >= 2560
                and eff_cache_frac[best_cache_rank] >= self.min_frac
            )
            if (
                has_warm_prefix
                and q_reqs[best_cache_rank] <= min_reqs + self.slack
                and est_ttft_s[best_cache_rank] <= min_ttft + _env_float("PROXY_PREFILL_STICKY_TTFT_SLACK_S", 0.65)
            ):
                rank = best_cache_rank
                is_sticky = True
                self.sticky += 1
            else:
                if has_warm_prefix:
                    self.spilled += 1
                rank = min(
                    range(self.n),
                    key=lambda r: (
                        round(est_ttft_s[r], 3),
                        self.inflight_uncached_tokens[r],
                        q_reqs[r],
                        self.placed[r],
                        r,
                    ),
                )
                is_sticky = False
                self.cold += 1

            u_chosen = uncached_toks[rank]
            d_chosen = dram_toks[rank]
            self.inflight[rank] += 1
            self.inflight_uncached_tokens[rank] += u_chosen
            self.inflight_dram_tokens[rank] += d_chosen
            self.placed[rank] += 1
            self._record_pod_blocks(rank, hs)

            total = self.sticky + self.cold
            if total % 64 == 0:
                print(
                    f"PERF PROXY llmd_prefill_router | total={total} "
                    f"sticky={self.sticky} ({100.0 * self.sticky / total:.1f}%) "
                    f"cold={self.cold} spilled={self.spilled} placed={self.placed} "
                    f"inflight_reqs={self.inflight} "
                    f"inflight_uncached_toks={self.inflight_uncached_tokens} "
                    f"hbm_blocks={[len(x) for x in self.hbm_lru]} "
                    f"dram_blocks={[len(x) for x in self.dram_lru]}",
                    flush=True,
                )
            return rank, is_sticky, u_chosen, d_chosen

    def done_prefill(
        self, rank: int, uncached_tokens: int = 0, dram_tokens: int = 0
    ) -> None:
        with self._lock:
            self.inflight[rank] = max(0, self.inflight[rank] - 1)
            self.inflight_uncached_tokens[rank] = max(
                0, self.inflight_uncached_tokens[rank] - uncached_tokens
            )
            self.inflight_dram_tokens[rank] = max(
                0, self.inflight_dram_tokens[rank] - dram_tokens
            )

    def route_decode(
        self,
        hs: list[bytes] | None,
        total_tokens: int = 4096,
        scraper: _BackendMetricsScraper | None = None,
    ) -> tuple[int, bool, int]:
        """llm-d Decode Profile (evaluated post-prefill):
        Combines decode-prefix-cache-scorer (w=2), active-request-scorer around
        32-slot MK ceiling (w=3), and kv-cache-utilization-scorer (w=2).
        """
        hs = hs or []
        total_tokens = max(64, total_tokens)
        if self.n <= 0:
            return 0, True, total_tokens

        with self._lock:
            num_blocks = max(1, len(hs))
            hbm_frac: list[float] = [0.0] * self.n
            hbm_toks: list[int] = [0] * self.n
            eff_reqs: list[float] = [0.0] * self.n
            kv_usage: list[float] = [0.0] * self.n
            scores: list[float] = [0.0] * self.n

            for d in range(self.n):
                hbm_hits, _ = self._match_pod_prefix(d, hs)
                frac = hbm_hits / num_blocks if hs else 0.0
                hbm_frac[d] = frac
                hbm_toks[d] = int(total_tokens * frac)

                reqs_d = float(self.inflight[d])
                kv_d = min(
                    1.0,
                    self.inflight_uncached_tokens[d]
                    / float(self.hbm_max_blocks * self.block_tokens),
                )
                if scraper is not None:
                    snap = scraper.get_decode(d)
                    if snap is not None:
                        reqs_d = max(reqs_d, snap.running_reqs + snap.waiting_reqs)
                        kv_d = snap.gpu_cache_usage
                eff_reqs[d] = reqs_d
                kv_usage[d] = kv_d

                # 1. active-request-scorer (w=3.0) around 32-slot MK ceiling
                if reqs_d < self.mk_ceiling_reqs:
                    load_score = (self.mk_ceiling_reqs - reqs_d) / self.mk_ceiling_reqs
                else:
                    load_score = -0.15 * (reqs_d - (self.mk_ceiling_reqs - 1.0))

                # 2. kv-cache-utilization-scorer (w=2.0)
                kv_score = 1.0 - kv_d
                if kv_d > 0.90:
                    kv_score -= 2.0

                # 3. decode-prefix-cache-scorer (w=2.0)
                cache_score = frac

                scores[d] = 3.0 * load_score + 2.0 * cache_score + 2.0 * kv_score

            best_cache_rank = max(
                range(self.n),
                key=lambda d: (
                    hbm_toks[d],
                    round(scores[d], 4),
                    -eff_reqs[d],
                    -self.placed[d],
                    -d,
                ),
            )
            min_eff_reqs = min(eff_reqs)
            has_warm_prefix = (
                bool(hs)
                and hbm_toks[best_cache_rank] >= 4096
                and hbm_frac[best_cache_rank] >= self.min_frac
            )
            if (
                has_warm_prefix
                and eff_reqs[best_cache_rank] <= min_eff_reqs + self.slack
                and kv_usage[best_cache_rank] < 0.92
            ):
                rank = best_cache_rank
                is_sticky = True
                self.sticky += 1
            else:
                if has_warm_prefix:
                    self.spilled += 1
                rank = max(
                    range(self.n),
                    key=lambda d: (
                        round(scores[d], 4),
                        -eff_reqs[d],
                        -self.inflight_uncached_tokens[d],
                        -self.placed[d],
                        -d,
                    ),
                )
                is_sticky = False
                self.cold += 1

            self.inflight[rank] += 1
            self.inflight_uncached_tokens[rank] += total_tokens
            self.placed[rank] += 1
            self._record_pod_blocks(rank, hs)

            total = self.sticky + self.cold
            if total % 64 == 0:
                print(
                    f"PERF PROXY llmd_decode_router | total={total} "
                    f"sticky={self.sticky} ({100.0 * self.sticky / total:.1f}%) "
                    f"cold={self.cold} spilled={self.spilled} placed={self.placed} "
                    f"inflight_reqs={self.inflight} eff_reqs={[round(x, 1) for x in eff_reqs]} "
                    f"kv_usage={[round(x, 3) for x in kv_usage]} "
                    f"hbm_blocks={[len(x) for x in self.hbm_lru]}",
                    flush=True,
                )
            return rank, is_sticky, total_tokens

    def done_decode(self, rank: int, total_tokens: int = 0) -> None:
        with self._lock:
            self.inflight[rank] = max(0, self.inflight[rank] - 1)
            self.inflight_uncached_tokens[rank] = max(
                0, self.inflight_uncached_tokens[rank] - total_tokens
            )

    def route(self, hs: list[bytes]) -> tuple[int, bool]:
        if self.role == "decode":
            rank, is_sticky, _ = self.route_decode(
                hs, len(hs) * self.block_tokens, None
            )
            return rank, is_sticky
        rank, is_sticky, _, _ = self.route_prefill(
            hs, len(hs) * self.block_tokens, None
        )
        return rank, is_sticky

    def done(self, rank: int) -> None:
        with self._lock:
            self.inflight[rank] = max(0, self.inflight[rank] - 1)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "role": self.role,
                "placed": list(self.placed),
                "inflight": list(self.inflight),
                "inflight_uncached_tokens": list(self.inflight_uncached_tokens),
                "inflight_dram_tokens": list(self.inflight_dram_tokens),
                "sticky": self.sticky,
                "cold": self.cold,
                "spilled": self.spilled,
                "hbm_blocks": [len(x) for x in self.hbm_lru],
                "dram_blocks": [len(x) for x in self.dram_lru],
            }


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
                    parts.append(
                        f"{key}: n={int(n)} avg={total / n:.2f}ms "
                        f"min={min_ms:.2f}ms max={max_ms:.2f}ms"
                    )
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


_PRE_SEND_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
_MAX_ATTEMPTS = max(1, _env_int("PROXY_MAX_ATTEMPTS", 5))
_RETRY_BASE_DELAY_S = _env_float("PROXY_RETRY_BASE_DELAY_S", 0.5)


def _log_unsafe_transport_failure(
    phase: str, request_id: str, exc: Exception, detail: str = ""
) -> None:
    """Flag a failure that cannot be retried safely at this layer."""
    msg = (
        f"[UNSAFE-{phase}] req_id={request_id} failed after the request "
        f"was already sent: {exc!r}. Not retrying: the server may have "
        f"admitted this request, and replaying it with the same "
        f"X-Request-Id would collide with the in-flight vLLM request id "
        f"(and the Raiden KV registration on the decode hop). Recovery "
        f"needs a fresh request id + new KV uuid and must be handled at "
        f"the request-flow level, not in this per-hop retry wrapper."
    )
    if detail:
        msg = f"{msg} {detail}"
    print(msg, flush=True)
    logger.error(msg)


async def _backoff_before_retry(
    attempt: int, phase: str, request_id: str, exc: Exception
) -> None:
    delay = _RETRY_BASE_DELAY_S * (2**attempt)
    msg = (
        f"[RETRY-{phase}] attempt {attempt + 1}/{_MAX_ATTEMPTS} "
        f"failed for req_id={request_id}: {exc!r}, "
        f"retrying in {delay:.1f}s"
    )
    print(msg, flush=True)
    logger.warning(msg)
    await asyncio.sleep(delay)


async def _post_with_retries(
    client_info: dict,
    endpoint: str,
    req_data: dict,
    headers: dict,
    request_id: str,
    phase: str,
):
    """POST to a backend, retrying only connection-establishment failures."""
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = await client_info["client"].post(
                endpoint, json=req_data, headers=headers
            )
            response.raise_for_status()
            return response
        except _PRE_SEND_ERRORS as e:
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            await _backoff_before_retry(attempt, phase, request_id, e)
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                _log_unsafe_transport_failure(phase, request_id, e)
            raise
        except httpx.TransportError as e:
            _log_unsafe_transport_failure(phase, request_id, e)
            raise


def _render_endpoint_for_api(api: str) -> str | None:
    if api == "/v1/completions":
        return "/v1/completions/render"
    return None


def _token_ids_from_completion_render(rendered: Any) -> list[int] | list[list[int]]:
    if not isinstance(rendered, list) or not rendered:
        raise ValueError("Completion render response must be a non-empty list")

    prompts: list[list[int]] = []
    for item in rendered:
        if not isinstance(item, dict):
            raise ValueError("Completion render item must be an object")
        token_ids = item.get("token_ids")
        if not isinstance(token_ids, list) or not all(
            type(token_id) is int and token_id >= 0 for token_id in token_ids
        ):
            raise ValueError("Completion render item must contain token_ids")
        prompts.append(token_ids)

    return prompts[0] if len(prompts) == 1 else prompts


def _replace_prompt_with_rendered_token_ids(req_data: dict, rendered: Any) -> dict:
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
    app.state.prefill_clients = []
    app.state.decode_clients = []

    keepalive_expiry = _env_float("PROXY_KEEPALIVE_EXPIRY_S", 4.0)
    limits = httpx.Limits(
        max_connections=None,
        max_keepalive_connections=None,
        keepalive_expiry=keepalive_expiry,
    )

    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}"
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None, base_url=prefiller_base_url, limits=limits
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}"
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None, base_url=decoder_base_url, limits=limits
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    base_block_tokens = _env_int("PROXY_BLOCK_SIZE_TOKENS", _env_int("BLOCK_SIZE", 640))
    prefill_pp_size = max(1, _env_int("PIPELINE_PARALLEL_SIZE_PREFILL", 1))
    prefill_block_tokens = _env_int("BLOCK_SIZE_PREFILL", 2176 if prefill_pp_size >= 8 else base_block_tokens)
    decode_block_tokens = _env_int("BLOCK_SIZE_DECODE", base_block_tokens)
    prefill_chunk_bytes = _env_int("PROXY_PREFILL_PREFIX_HASH_CHUNK", prefill_block_tokens * 4)
    decode_chunk_bytes = _env_int("PROXY_DECODE_PREFIX_HASH_CHUNK", decode_block_tokens * 4)
    chunk_bytes = prefill_chunk_bytes
    block_tokens = prefill_block_tokens

    kv_offload_bytes = _env_int("KV_OFFLOAD_BYTES_PREFILL", 770000000000)
    if prefill_pp_size >= 8 and kv_offload_bytes == 420000000000:
        kv_offload_bytes = 550000000000
    bytes_per_prefill_block = 35651584 if prefill_pp_size >= 8 else 41944704
    default_prefill_hbm_blocks = 1616 if prefill_pp_size >= 8 else 7739
    default_prefill_dram_blocks = max(0, int(kv_offload_bytes // bytes_per_prefill_block)) if kv_offload_bytes > 0 else 18358
    prefill_dram_blocks = _env_int("PROXY_PREFILL_DRAM_MAX_BLOCKS", default_prefill_dram_blocks)
    if prefill_pp_size >= 8 and prefill_dram_blocks == 11780:
        prefill_dram_blocks = 15427
    default_peak_prefill_tok_s = 62000.0 if prefill_pp_size >= 8 else 42000.0

    routing_mode = os.getenv("PROXY_PREFILL_ROUTING", "prefix_hash").strip().lower()
    if (len(app.state.prefill_clients) > 1 and routing_mode in ("prefix_hash", "llmd")) or (
        len(app.state.prefill_clients) >= 1 and routing_mode == "llmd"
    ):
        app.state.prefill_router = _PrefixAffinityRouter(
            n=len(app.state.prefill_clients),
            chunk=prefill_chunk_bytes,
            min_frac=_env_float("PROXY_PREFIX_HASH_MIN_FRAC", 0.40),
            slack=_env_int("PROXY_PREFIX_HASH_SLACK", 10),
            role="prefill",
            block_tokens=prefill_block_tokens,
            hbm_max_blocks=_env_int("PROXY_PREFILL_HBM_MAX_BLOCKS", default_prefill_hbm_blocks),
            dram_max_blocks=prefill_dram_blocks,
            peak_prefill_tok_s=_env_float("PROXY_PEAK_PREFILL_TOK_S", default_peak_prefill_tok_s),
            dram_load_tok_s=_env_float("PROXY_DRAM_LOAD_TOK_S", 280000.0),
        )
    else:
        app.state.prefill_router = None

    decode_routing = (
        os.getenv(
            "PROXY_DECODE_ROUTING", os.getenv("PROXY_PREFILL_ROUTING", "prefix_hash")
        )
        .strip()
        .lower()
    )
    decode_dcp_size = max(1, _env_int("DECODE_CONTEXT_PARALLEL_SIZE_DECODE", _env_int("PROXY_DECODE_DCP_SIZE", 4)))
    default_decode_hbm_blocks = 27481 if decode_dcp_size >= 4 else (7903 * decode_dcp_size)
    default_mk_ceiling = _env_float("MAX_NUM_SEQS_DECODE", 40.0 if decode_dcp_size >= 4 else 32.0)
    if (len(app.state.decode_clients) > 1 and decode_routing in ("prefix_hash", "llmd")) or (
        len(app.state.decode_clients) >= 1 and decode_routing == "llmd"
    ):
        app.state.decode_router = _PrefixAffinityRouter(
            n=len(app.state.decode_clients),
            chunk=decode_chunk_bytes,
            min_frac=_env_float("PROXY_DECODE_HASH_MIN_FRAC", 0.40),
            slack=_env_int("PROXY_DECODE_HASH_SLACK", 12),
            role="decode",
            block_tokens=decode_block_tokens,
            hbm_max_blocks=_env_int("PROXY_DECODE_HBM_MAX_BLOCKS", default_decode_hbm_blocks),
            dram_max_blocks=0,
            mk_ceiling_reqs=_env_float("PROXY_DECODE_MK_CEILING", default_mk_ceiling),
        )
    else:
        app.state.decode_router = None

    # Under DCP=4, each Decode pod holds 27,481 HBM blocks (17.59M tokens, ~175 concurrent
    # trajectories); default to 160 inflight per Decode pod so the proxy never throttles
    # finished-prefill requests when Decode has >50% free HBM.
    default_decode_max_inflight = 160 if decode_dcp_size >= 4 else 48
    decode_cap = _env_int("PROXY_DECODE_MAX_INFLIGHT", default_decode_max_inflight)
    app.state.decode_slots = (
        [asyncio.Semaphore(decode_cap) for _ in app.state.decode_clients]
        if decode_cap > 0
        else None
    )

    if app.state.prefill_router is not None or app.state.decode_router is not None:
        app.state.metrics_scraper = _BackendMetricsScraper(
            num_prefill=len(app.state.prefill_clients),
            num_decode=len(app.state.decode_clients),
            poll_interval_s=_env_float("PROXY_METRICS_POLL_INTERVAL_S", 0.5),
        )
        app.state.metrics_scraper.start(
            app.state.prefill_clients, app.state.decode_clients
        )
    else:
        app.state.metrics_scraper = None

    print(
        f"Initialized {len(app.state.prefill_clients)} prefill clients "
        f"(routing={'llmd_2tier_prefix_token_load' if app.state.prefill_router else 'round_robin'}) "
        f"and {len(app.state.decode_clients)} decode clients.",
        flush=True,
    )
    print(
        "Decode routing="
        f"{'llmd_post_prefill_mk32_kv_prefix' if app.state.decode_router else 'round_robin'} "
        f"max_inflight_per_decode={decode_cap or 'unlimited'} "
        f"block_tokens={block_tokens} chunk_bytes={chunk_bytes}",
        flush=True,
    )

    yield

    if getattr(app.state, "metrics_scraper", None) is not None:
        await app.state.metrics_scraper.stop()

    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()

    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")

    parser.add_argument(
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8400]
    )

    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"]
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[9400]
    )

    args = parser.parse_args()

    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports"
        )

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


def get_next_client(app, service_type: str):
    if service_type == "prefill":
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_prefill(
    client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
    forwarded: dict | None = None,
):
    req_data = req_data.copy()
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        **(forwarded or {}),
    }

    return await _post_with_retries(
        client_info, endpoint, req_data, headers, request_id, phase="PREFILL"
    )


async def render_completion_prompt(
    client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
    forwarded: dict | None = None,
):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        **(forwarded or {}),
    }

    return await _post_with_retries(
        client_info, endpoint, req_data, headers, request_id, phase="RENDER"
    )


async def stream_from_decode(
    client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
    forwarded: dict | None = None,
):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        **(forwarded or {}),
    }

    for attempt in range(_MAX_ATTEMPTS):
        yielded_any = False
        try:
            async with client_info["client"].stream(
                "POST", endpoint, json=req_data, headers=headers
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                response.raise_for_status()
                encoder = _AsciiSafeStreamEncoder()
                async for chunk in response.aiter_bytes():
                    chunk = encoder.encode(chunk)
                    if chunk:
                        yielded_any = True
                        yield chunk
                chunk = encoder.encode(final=True)
                if chunk:
                    yielded_any = True
                    yield chunk
            return
        except _PRE_SEND_ERRORS as e:
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            await _backoff_before_retry(attempt, "DECODE", request_id, e)
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if yielded_any:
                detail = (
                    "Stream had already emitted bytes to the client, "
                    "so the response is truncated."
                )
            else:
                detail = (
                    "Failed before the first byte; the client will "
                    "still see HTTP 200 followed by an empty/short "
                    "body because StreamingResponse has already "
                    "flushed headers."
                )
            _log_unsafe_transport_failure("DECODE", request_id, e, detail)
            raise


async def _handle_completions(api: str, request: Request):
    try:
        t_request_recv_perf = time.perf_counter()
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        forwarded = {}
        if (dp_rank := request.headers.get("X-data-parallel-rank")) is not None:
            forwarded["X-data-parallel-rank"] = dp_rank
        print(
            f"PERF PROXY req_received req_id={request_id} ts={time.time():.6f}",
            flush=True,
        )

        router: _PrefixAffinityRouter | None = getattr(
            request.app.state, "prefill_router", None
        )
        decode_router: _PrefixAffinityRouter | None = getattr(
            request.app.state, "decode_router", None
        )
        scraper: _BackendMetricsScraper | None = getattr(
            request.app.state, "metrics_scraper", None
        )
        prefill_idx: int | None = None
        prefill_uncached_toks: int = 0
        prefill_dram_toks: int = 0
        decode_idx: int | None = None
        decode_toks: int = 0
        prefill_hs: list[bytes] | None = None
        decode_hs: list[bytes] | None = None
        est_tokens: int = 4096

        if router is not None:
            prefill_hs, est_tokens = router.extract_blocks_and_tokens(req_data)
        if decode_router is not None:
            if router is not None and router.chunk == decode_router.chunk:
                decode_hs = prefill_hs
            else:
                decode_hs, est_tokens = decode_router.extract_blocks_and_tokens(req_data)

        decode_slots = getattr(request.app.state, "decode_slots", None)
        slot: asyncio.Semaphore | None = None
        slot_acquired = False
        released = False

        def release_decode() -> None:
            nonlocal released
            if released:
                return
            released = True
            if slot is not None and slot_acquired:
                slot.release()
            if decode_router is not None and decode_idx is not None:
                decode_router.done_decode(decode_idx, decode_toks)

        try:
            # 1. Route to Prefill using llm-d Prefill Profile
            if router is not None:
                (
                    prefill_idx,
                    _,
                    prefill_uncached_toks,
                    prefill_dram_toks,
                ) = router.route_prefill(prefill_hs, est_tokens, scraper)
                prefill_client_info = request.app.state.prefill_clients[prefill_idx]
            else:
                prefill_client_info = get_next_client(request.app, "prefill")

            try:
                render_endpoint = _render_endpoint_for_api(api)
                if render_endpoint is not None:
                    render_response = await render_completion_prompt(
                        prefill_client_info,
                        render_endpoint,
                        req_data,
                        request_id,
                        forwarded,
                    )
                    req_data = _replace_prompt_with_rendered_token_ids(
                        req_data, render_response.json()
                    )

                t_prefill_send = time.time()
                t_prefill_send_perf = time.perf_counter()
                print(
                    f"PERF PROXY prefill_send req_id={request_id} "
                    f"prefill_id={prefill_client_info.get('id', 0)} "
                    f"ts={t_prefill_send:.6f}",
                    flush=True,
                )
                response = await send_request_to_prefill(
                    prefill_client_info, api, req_data, request_id, forwarded
                )
            finally:
                if router is not None and prefill_idx is not None:
                    router.done_prefill(
                        prefill_idx, prefill_uncached_toks, prefill_dram_toks
                    )

            t_prefill_recv = time.time()
            t_prefill_recv_perf = time.perf_counter()
            prefill_ms = (t_prefill_recv_perf - t_prefill_send_perf) * 1000.0
            _PROXY_LATENCY.record("prefill", prefill_ms)
            _PROXY_LATENCY.record(
                "prefill_from_req", (t_prefill_recv_perf - t_request_recv_perf) * 1000.0
            )

            response_json = response.json()
            usage = response_json.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
                est_tokens = max(64, usage["prompt_tokens"])

            kv_transfer_params = response_json.get("kv_transfer_params", {})
            kv_uuid = kv_transfer_params.get("uuid") if kv_transfer_params else None
            kv_uuid_log = f" kv_uuid={kv_uuid}" if kv_uuid is not None else ""
            print(
                f"PERF PROXY prefill_recv req_id={request_id} ts={t_prefill_recv:.6f} "
                f"dur_ms={prefill_ms:.2f}{kv_uuid_log}",
                flush=True,
            )

            if kv_transfer_params:
                req_data["kv_transfer_params"] = kv_transfer_params

            # 2. Route to Decode AFTER Prefill completes using llm-d Decode Profile
            if decode_router is not None:
                decode_idx, _, decode_toks = decode_router.route_decode(
                    decode_hs, est_tokens, scraper
                )
            else:
                decode_idx = next(request.app.state.decode_iterator)
                decode_toks = est_tokens
            decode_client_info = request.app.state.decode_clients[decode_idx]
            slot = decode_slots[decode_idx] if decode_slots is not None else None

            if slot is not None:
                t_admit_perf = time.perf_counter()
                await slot.acquire()
                slot_acquired = True
                admit_wait_ms = (time.perf_counter() - t_admit_perf) * 1000.0
                _PROXY_LATENCY.record("decode_admit", admit_wait_ms)
                print(
                    f"PERF PROXY decode_admit req_id={request_id} "
                    f"decode_id={decode_client_info.get('id', decode_idx)} "
                    f"wait_ms={admit_wait_ms:.2f} ts={time.time():.6f}",
                    flush=True,
                )

            logger.debug("Using %s %s", prefill_client_info, decode_client_info)

            async def generate_stream():
                try:
                    t_decode_send = time.time()
                    t_decode_send_perf = time.perf_counter()
                    print(
                        f"PERF PROXY decode_send req_id={request_id} "
                        f"ts={t_decode_send:.6f}{kv_uuid_log}",
                        flush=True,
                    )
                    first = True
                    async for chunk in stream_from_decode(
                        decode_client_info, api, req_data, request_id, forwarded
                    ):
                        if first:
                            t_first = time.time()
                            t_first_perf = time.perf_counter()
                            decode_first_ms = (
                                t_first_perf - t_decode_send_perf
                            ) * 1000.0
                            ttft_ms = (t_first_perf - t_request_recv_perf) * 1000.0
                            _PROXY_LATENCY.record("decode_first", decode_first_ms)
                            _PROXY_LATENCY.record("ttft", ttft_ms)
                            print(
                                f"PERF PROXY first_chunk req_id={request_id} "
                                f"ts={t_first:.6f} "
                                f"dur_from_decode_send_ms={decode_first_ms:.2f} "
                                f"ttft_ms={ttft_ms:.2f}{kv_uuid_log}",
                                flush=True,
                            )
                            first = False
                        yield chunk
                finally:
                    release_decode()

            media_type = (
                "text/event-stream" if req_data.get("stream") else "application/json"
            )
            return StreamingResponse(generate_stream(), media_type=media_type)
        except BaseException:
            release_decode()
            raise

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/v1/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/v1/chat/completions", request)


@app.get("/v1/models")
async def models():
    client = app.state.decode_clients[0]["client"]
    response = await client.get("/v1/models")
    return JSONResponse(status_code=response.status_code, content=response.json())


@app.api_route("/metrics", methods=["GET", "HEAD"])
async def metrics(request: Request):
    """Forward /metrics from decode-0 so aiperf telemetry scraping succeeds."""
    if request.method == "HEAD":
        return Response(status_code=200, media_type="text/plain; version=0.0.4")
    try:
        client = app.state.decode_clients[0]["client"]
        resp = await client.get("/metrics", timeout=2.0)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "text/plain; version=0.0.4"),
        )
    except Exception:
        return Response(content=b"", status_code=200, media_type="text/plain")


@app.get("/healthcheck")
async def healthcheck():
    router: _PrefixAffinityRouter | None = getattr(app.state, "prefill_router", None)
    res: dict[str, Any] = {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }
    if router is not None:
        res["prefill_router"] = router.stats()
    decode_router: _PrefixAffinityRouter | None = getattr(
        app.state, "decode_router", None
    )
    if decode_router is not None:
        res["decode_router"] = decode_router.stats()
    return res


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port, timeout_keep_alive=300, backlog=4096)
