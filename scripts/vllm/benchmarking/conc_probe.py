#!/usr/bin/env python3
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
"""Concurrency probe for a live vLLM server (issue #845).

For each N in --conc-list, fire N identical chat requests at once, wait for
all of them, and classify every reply: ok (the expected count-up is present),
wrong (complete but not the expected text), garbage (CJK / HTML entities /
word repeats in an English answer), empty, or error (HTTP / transport).
While the burst runs, /metrics is sampled so the peak number of running
requests is known. One line per N on stdout; a JSON summary in --out.

    conc_probe.py --base-url http://127.0.0.1:8000 --model M \
        --conc-list 128,160,192,256 [--max-tokens 300] [--out summary.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time

import aiohttp

PROMPT = ("Count from 1 to 40, one number per line, nothing else. "
          "After 40 write the single word DONE.")
GARBAGE_RE = re.compile(r"[一-鿿぀-ヿ]|&nbsp|<span|(\b\w+\b)(?:\s+\1\b){3,}")
NUMBERS_RE = re.compile(r"\b(\d{1,2})\b")


def classify(text: str) -> str:
    if not text or not text.strip():
        return "empty"
    if GARBAGE_RE.search(text):
        return "garbage"
    nums = [int(x) for x in NUMBERS_RE.findall(text)]
    if "DONE" in text and nums[:40] == list(range(1, 41)):
        return "ok"
    # counted correctly but ran out of tokens before DONE: still a sane reply
    if nums[:20] == list(range(1, 21)):
        return "ok"
    return "wrong"


async def one_request(session, url, model, max_tokens, idx):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{PROMPT} (request {idx})"}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    }
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=1800)) as r:
            body = await r.json()
            if r.status != 200:
                return {"idx": idx, "cls": "error", "detail": f"HTTP {r.status}: {json.dumps(body)[:200]}", "s": time.perf_counter() - t0}
            msg = body["choices"][0]["message"]
            text = msg.get("content") or ""
            usage = body.get("usage") or {}
            return {"idx": idx, "cls": classify(text), "text": text[:300], "finish": body["choices"][0].get("finish_reason"),
                    "completion_tokens": usage.get("completion_tokens"), "s": time.perf_counter() - t0}
    except Exception as e:  # noqa: BLE001 - any transport failure is a data point
        return {"idx": idx, "cls": "error", "detail": f"{type(e).__name__}: {e}"[:200], "s": time.perf_counter() - t0}


async def sample_metrics(session, metrics_url, stop, out):
    peak_running = 0
    peak_waiting = 0
    while not stop.is_set():
        try:
            async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                txt = await r.text()
            for line in txt.splitlines():
                if line.startswith("vllm:num_requests_running"):
                    peak_running = max(peak_running, int(float(line.rsplit(" ", 1)[1])))
                elif line.startswith("vllm:num_requests_waiting") and not line.startswith("vllm:num_requests_waiting_by"):
                    peak_waiting = max(peak_waiting, int(float(line.rsplit(" ", 1)[1])))
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(1)
    out["peak_running"] = peak_running
    out["peak_waiting"] = peak_waiting


async def probe(base_url, model, n, max_tokens):
    url = f"{base_url}/v1/chat/completions"
    metrics_url = f"{base_url}/metrics"
    stop = asyncio.Event()
    peaks = {}
    async with aiohttp.ClientSession() as session:
        sampler = asyncio.create_task(sample_metrics(session, metrics_url, stop, peaks))
        t0 = time.perf_counter()
        results = await asyncio.gather(*(one_request(session, url, model, max_tokens, i) for i in range(n)))
        wall = time.perf_counter() - t0
        stop.set()
        await sampler
    counts = {}
    for r in results:
        counts[r["cls"]] = counts.get(r["cls"], 0) + 1
    return {"conc": n, "wall_s": round(wall, 1), "counts": counts, **peaks, "samples": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--conc-list", default="128,160,192,256")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    summary = []
    rc = 0
    for n in [int(x) for x in a.conc_list.split(",") if x]:
        try:
            res = asyncio.run(probe(a.base_url, a.model, n, a.max_tokens))
        except Exception as e:  # noqa: BLE001
            print(f"[conc-probe] N={n}: probe itself failed: {type(e).__name__}: {e}", flush=True)
            rc = 1
            break
        c = res["counts"]
        bad = sum(v for k, v in c.items() if k != "ok")
        print(f"[conc-probe] N={n}: wall={res['wall_s']}s peak_running={res.get('peak_running')} "
              f"peak_waiting={res.get('peak_waiting')} ok={c.get('ok',0)} wrong={c.get('wrong',0)} "
              f"garbage={c.get('garbage',0)} empty={c.get('empty',0)} error={c.get('error',0)}", flush=True)
        for s in res["samples"]:
            if s["cls"] in ("garbage", "wrong", "empty"):
                print(f"[conc-probe]   example {s['cls']} (finish={s.get('finish')}, tokens={s.get('completion_tokens')}): {s.get('text','')[:160]!r}", flush=True)
                break
        for s in res["samples"]:
            if s["cls"] == "error":
                print(f"[conc-probe]   example error: {s['detail']}", flush=True)
                break
        summary.append(res)
        if bad:
            rc = 1
        if c.get("error", 0) == n:
            print("[conc-probe] every request failed; server is probably dead, stopping", flush=True)
            break
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
