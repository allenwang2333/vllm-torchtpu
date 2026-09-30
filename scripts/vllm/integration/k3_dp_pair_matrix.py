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


def make_prompt(tag: str, lines: int) -> str:
    return (
        build_prefix(tag, lines)
        + "\nSummarise the ledger above in one sentence, then state the final "
        "record's quantity.\n"
    )


def run_one(a, body: dict, i: int | None, j: int) -> dict:
    """Prefill on rank i then decode on rank j, or local on decode rank j."""
    rid = str(uuid.uuid4())
    dec_body = dict(body)
    if i is not None:
        status, pre = post(
            f"{a.prefill_url}/v1/completions",
            {**body, "max_tokens": 1},
            {"X-data-parallel-rank": str(i), "X-Request-Id": rid},
            a.timeout,
        )
        params = pre.get("kv_transfer_params")
        if status != 200 or not params:
            return {"ok": False, "prefill_status": status, "error": str(pre)[:300]}
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
    return {
        "ok": status == 200,
        "status": status,
        "tokens": choice.get("token_ids") or [],
        "text": choice.get("text", ""),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
    }


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
    p.add_argument("--skip-matrix", action="store_true")
    # The state-contamination stage pinned to one pair: the tags interleaved,
    # references cold, then --rounds more rounds resuming from both engines'
    # prefix caches. "i:j,..." pairs of prefill rank and decode rank.
    p.add_argument("--pinned-pairs", default="")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--timeout", type=int, default=600)
    a = p.parse_args()

    state = {"can_reset": a.reset}

    def maybe_reset() -> None:
        if state["can_reset"]:
            state["can_reset"] = reset(a.prefill_url, a.timeout) and reset(
                a.decode_url, a.timeout
            )
            if not state["can_reset"]:
                print("PAIR_MATRIX_RESET_UNAVAILABLE 1", flush=True)

    tags = a.tags.split(",")
    bodies = {
        tag: {
            "model": a.model,
            "prompt": make_prompt(tag, a.lines),
            "max_tokens": a.max_tokens,
            "temperature": 0,
            "return_token_ids": True,
        }
        for tag in tags
    }
    all_agree = True

    for tag in [] if a.skip_matrix else tags:
        runs = [(None, j) for j in range(a.decode_dp)]
        runs += [(i, j) for i in range(a.prefill_dp) for j in range(a.decode_dp)]
        outputs: dict[str, list[int]] = {}
        texts: dict[str, str] = {}
        for i, j in runs:
            maybe_reset()
            name = f"local@D{j}" if i is None else f"P{i}->D{j}"
            r = run_one(a, bodies[tag], i, j)
            line = {"tag": tag, "run": name, "reset": state["can_reset"]}
            line.update({k: v for k, v in r.items() if k != "tokens"})
            line["text"] = r.get("text", "")[:100]
            print(json.dumps(line), flush=True)
            if not r["ok"]:
                all_agree = False
                continue
            outputs[name] = r["tokens"]
            texts[name] = r["text"]
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
    if not a.skip_matrix:
        print(f"PAIR_MATRIX_OK {int(all_agree)}")

    pinned_ok = True
    for pair in filter(None, a.pinned_pairs.split(",")):
        i, j = (int(x) for x in pair.split(":"))
        maybe_reset()
        reference: dict[str, list[int]] = {}
        hits = divergences = 0
        for rnd in range(1, a.rounds + 2):
            for tag in tags:  # interleaved: the slot last held the other prefix
                r = run_one(a, bodies[tag], i, j)
                if not r["ok"]:
                    print(f"PINNED P{i}->D{j} {tag} r{rnd} FAILED {r}")
                    pinned_ok = False
                    continue
                if tag not in reference:
                    reference[tag] = r["tokens"]
                    verdict = "reference"
                else:
                    hits += 1
                    same = r["tokens"] == reference[tag]
                    divergences += not same
                    verdict = "match" if same else "DIVERGED"
                print(
                    json.dumps(
                        {
                            "pinned": f"P{i}->D{j}",
                            "round": rnd,
                            "tag": tag,
                            "cached_tokens": r["cached_tokens"],
                            "verdict": verdict,
                            "text": r["text"][:80],
                        }
                    ),
                    flush=True,
                )
        print(
            f"PINNED_CONTAMINATION P{i}->D{j} rounds={hits} divergences={divergences}"
        )
        pinned_ok = pinned_ok and hits > 0 and divergences == 0
    if a.pinned_pairs:
        print(f"PINNED_CONTAMINATION_OK {int(pinned_ok)}")
    return 0 if all_agree and pinned_ok else 1


if __name__ == "__main__":
    sys.exit(main())
