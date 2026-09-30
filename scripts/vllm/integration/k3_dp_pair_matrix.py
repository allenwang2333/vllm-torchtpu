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
"""Greedy output of one prompt over every prefill-DP x decode-DP pair.

Diagnostic for a P/D pair whose roles both run data parallelism. Plays the
proxy's part itself so it can pin each phase to a DP rank with vLLM's
X-data-parallel-rank header: prefill on rank i (max_tokens 1), then decode on
rank j with the kv_transfer_params prefill returned. Also runs the prompt
locally on each decode rank, without a transfer, as the aggregated reference.

With --reset, both servers' prefix caches are reset before every request
(needs VLLM_SERVER_DEV_MODE=1), so every run is cold on both sides and every
difference is the pair's, not a cache hit's.

Prints one JSON line per run and, per prompt, the classes of identical token
sequences. Exit 0 when every run of a prompt agrees, 1 otherwise.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from smoke_mamba_state_contamination import build_prefix  # noqa: E402


def post(url: str, body: dict | None, headers: dict, timeout: int) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **headers}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode(errors="replace")[:300]}


def reset(url: str, timeout: int) -> bool:
    for _ in range(20):
        status, _ = post(f"{url}/reset_prefix_cache", None, {}, timeout)
        if status == 200:
            return True
        if status == 404:
            return False
        time.sleep(0.5)
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prefill-url", required=True)
    p.add_argument("--decode-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prefill-dp", type=int, default=4)
    p.add_argument("--decode-dp", type=int, default=4)
    p.add_argument("--tags", default="ALPHA,BRAVO")
    p.add_argument("--lines", type=int, default=340)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--timeout", type=int, default=600)
    a = p.parse_args()

    can_reset = a.reset
    all_agree = True
    for tag in a.tags.split(","):
        prompt = (
            build_prefix(tag, a.lines)
            + "\nSummarise the ledger above in one sentence, then state the final "
            "record's quantity.\n"
        )
        body = {
            "model": a.model,
            "prompt": prompt,
            "max_tokens": a.max_tokens,
            "temperature": 0,
            "return_token_ids": True,
        }
        runs = [("local", None, j) for j in range(a.decode_dp)]
        runs += [("pd", i, j) for i in range(a.prefill_dp) for j in range(a.decode_dp)]
        outputs: dict[str, list[int]] = {}
        texts: dict[str, str] = {}
        for kind, i, j in runs:
            if can_reset:
                can_reset = reset(a.prefill_url, a.timeout) and reset(
                    a.decode_url, a.timeout
                )
                if not can_reset:
                    print("PAIR_MATRIX_RESET_UNAVAILABLE 1", flush=True)
            rid = str(uuid.uuid4())
            name = f"local@D{j}" if kind == "local" else f"P{i}->D{j}"
            params = None
            if kind == "pd":
                status, pre = post(
                    f"{a.prefill_url}/v1/completions",
                    {**body, "max_tokens": 1},
                    {"X-data-parallel-rank": str(i), "X-Request-Id": rid},
                    a.timeout,
                )
                params = pre.get("kv_transfer_params")
                if status != 200 or not params:
                    print(
                        json.dumps(
                            {
                                "tag": tag,
                                "run": name,
                                "prefill_status": status,
                                "prefill": str(pre)[:300],
                            }
                        ),
                        flush=True,
                    )
                    all_agree = False
                    continue
            dec_body = dict(body)
            if params:
                dec_body["kv_transfer_params"] = params
            status, out = post(
                f"{a.decode_url}/v1/completions",
                dec_body,
                {"X-data-parallel-rank": str(j), "X-Request-Id": rid},
                a.timeout,
            )
            choice = (out.get("choices") or [{}])[0]
            usage = out.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            outputs[name] = choice.get("token_ids") or []
            texts[name] = choice.get("text", "")
            print(
                json.dumps(
                    {
                        "tag": tag,
                        "run": name,
                        "status": status,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "cached_tokens": details.get("cached_tokens"),
                        "reset": can_reset,
                        "text": texts[name][:100],
                    }
                ),
                flush=True,
            )
        classes: dict[tuple, list[str]] = {}
        for name, toks in outputs.items():
            classes.setdefault(tuple(toks), []).append(name)
        print(f"PAIR_MATRIX {tag} runs={len(outputs)} classes={len(classes)}")
        for n, (toks, names) in enumerate(
            sorted(classes.items(), key=lambda kv: -len(kv[1]))
        ):
            print(
                f"PAIR_MATRIX_CLASS {tag} #{n} size={len(names)} "
                f"text={texts[names[0]][:100]!r} members={' '.join(names)}"
            )
        if len(classes) != 1:
            all_agree = False
    print(f"PAIR_MATRIX_OK {int(all_agree)}")
    return 0 if all_agree else 1


if __name__ == "__main__":
    sys.exit(main())
