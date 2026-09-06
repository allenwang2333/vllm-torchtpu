#!/usr/bin/env python3
"""Capture a synchronized request wave; serving owns decode phase gating.

Run separately from measured benchmarks. The server must use the phased
profiler, max_iterations=15 and an appropriate decode-only KV threshold.
"""
import argparse
import concurrent.futures
import json
import threading
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    payload = dict(model=args.model,
                   prompt=[1000],
                   max_tokens=1000,
                   temperature=0,
                   seed=42,
                   ignore_eos=True,
                   stream=False)

    def post(route, data):
        request = urllib.request.Request(
            args.base_url.rstrip("/") + route,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=1800) as response:
            raw = response.read().decode()
        return json.loads(raw) if raw else {}

    def wave(label):
        barrier = threading.Barrier(args.concurrency)

        def request_one(index):
            barrier.wait(timeout=60)
            response = post("/v1/completions", payload)
            (args.output / f"{label}-{index}.json"
             ).write_text(json.dumps(response, indent=2) + "\n")
            usage = response["usage"]
            if usage["prompt_tokens"] != 1 or usage[
                    "completion_tokens"] != 1000:
                raise RuntimeError(f"Unexpected token counts: {usage}")
            return usage

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency) as pool:
            return list(pool.map(request_one, range(args.concurrency)))

    wave("warmup-1")
    wave("warmup-2")
    post("/start_profile", {})
    try:
        usages = wave("capture")
    finally:
        post("/stop_profile", {})
    # Request completion is not proof of trace length: verify server-side
    # phase stats and the resulting XPlane before accepting the artifact.
    summary = dict(concurrency=args.concurrency,
                   requested_active_decode_steps=15,
                   profile_steps_verified=False,
                   measurement_kind="profile_request_wave_not_benchmark",
                   payload=payload,
                   usages=usages)
    (args.output /
     "requests-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PROFILE_REQUEST_WAVE_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
