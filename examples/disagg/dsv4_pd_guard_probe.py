# SPDX-License-Identifier: Apache-2.0
"""Send the request that used to kill the DSv4 prefill replica, straight at it.

The proxy pins the prefill leg to ``max_tokens=1``. That clamp is what keeps the
producer halted at the handoff token, and it is the only reason the recycled
sliding-window page path stays out of reach on the ordinary path. So this probe
bypasses the proxy and talks to the producer's own port.

A prompt of a hundred-odd tokens at ``max_tokens=48`` is the shape that took the
DP4/DP4 engine down: the producer ran to 161 computed tokens while the transfer
prefix was 113, the allocator had freed nine of its eleven compressor-state
pages, and every page left in the prefix was the null block. Raiden refuses the
duplicate destination by raising out of ``execute_model``, which kills the
engine. On the DP4/DP4 pair this prompt measures 131 tokens and drives the
producer to 178 against a 130-token prefix, leaving two dead pages each in the
indexer and CSA state groups, which is comfortably past the threshold.

With the guard in place the connector declines that send, the request prefills
on the decode side, and the producer stays up. This exits 0 when the completion
comes back and the producer is still healthy afterwards, so a green run means
the guard held rather than that nothing interesting happened. Check
``Stage-3 skipping the send`` in the producer log to confirm it fired.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# 131 tokens under the DSv4 tokenizer, which puts the compressor-state groups
# at nine 16-token pages under an 8-token window, the tightest geometry the
# model has. Any prompt in the low hundreds does the job; this one is the one
# that was measured.
_PROMPT = ("The duty officer code is 42662735. " +
           "A slow rain settled over the terraced fields. " * 12)


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",
                    required=True,
                    help="the producer's own base URL, not the proxy's")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    base = args.base.rstrip("/")
    try:
        out = _post(f"{base}/v1/completions", {
            "model": args.model,
            "prompt": _PROMPT,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
        }, args.timeout)
    except (urllib.error.URLError, OSError) as exc:
        print(json.dumps({"ok": False, "error": repr(exc)}))
        return 1

    usage = out.get("usage", {})
    # The completion succeeding is not enough on its own. Raiden raises on the
    # producer worker, so the engine can die a moment after the response is
    # written; ask the producer whether it is still there.
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=30) as resp:
            health = resp.status
    except (urllib.error.URLError, OSError) as exc:
        print(
            json.dumps({
                "ok": False,
                "usage": usage,
                "error": f"producer unhealthy after the probe: {exc!r}",
            }))
        return 1

    ok = health == 200 and usage.get("completion_tokens", 0) > 0
    print(
        json.dumps({
            "ok": ok,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "producer_health": health,
        }))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
