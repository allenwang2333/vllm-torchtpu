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
"""KV offload multi-turn recall: does turn 2 resume from the host store at
turn 1's prompt boundary, and answer exactly as an uncached run would?

Runs against run_dsv4_offload_correctness.sh's server (max_num_seqs=1,
max_num_batched_tokens = the hit block, requests pinned to one DP rank), so
cached and uncached passes prefill in the same block-aligned chunks.

Per prompt length L:

  turn1  a fresh ledger prompt. Gate: nothing cached.
  reset  POST /reset_prefix_cache empties HBM; the Raiden store keeps its
         entries.
  turn2  turn 1's prompt + its answer + a follow-up question. Gate: no local
         hits, an external hit reaching at least the last whole block before
         turn 1's final prompt token, less one block of slack for a
         speculative-decoding peek chunk.
  ref    turn 2 under a different cache_salt (nothing can hit). Gate: turn 2's
         token ids equal ref's.

Writes every pass to --out as JSON so two server configurations can be
compared token for token.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smoke_basic_offload_correctness import (  # noqa: E402
    COUNTERS,
    Client,
    _metrics,
    _post,
    manifest_code,
    size_prompt,
)

FOLLOW_UP = "Repeat the manifest code, then quote entries 0002 and 0003 word for word."


def ask_messages(client, messages, cache_salt=None):
    payload = {
        "model": client.model,
        "messages": messages,
        "add_generation_prompt": True,
        "add_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": client.max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "return_token_ids": True,
    }
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    before = _metrics(client.base)
    status, body = _post(
        client.base, "/v1/chat/completions", payload, client.timeout, client.headers
    )
    after = _metrics(client.base)
    delta = {k: int(after.get(k, 0) - before.get(k, 0)) for k in COUNTERS}
    if status != 200:
        raise SystemExit(f"request failed (HTTP {status}): {body}")
    choice = body["choices"][0]
    details = (body.get("usage") or {}).get("prompt_tokens_details") or {}
    return {
        "local_hits": delta["prefix_cache_hits_total"],
        "ext_hits": delta["external_prefix_cache_hits_total"],
        "cached_tokens": details.get("cached_tokens"),
        "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens"),
        "token_ids": choice.get("token_ids"),
        "text": (choice.get("message") or {}).get("content"),
    }


def run_length(client, salt, aim, hit_block, settle_s):
    namespace = f"{salt}-L{aim}"
    code = manifest_code(namespace)
    prompt, sized = size_prompt(client, namespace, code, aim, hit_block)
    turn1_msgs = [{"role": "user", "content": prompt}]
    turn1 = ask_messages(client, turn1_msgs)
    failures = []
    if turn1["cached_tokens"]:
        failures.append(f"L={sized}: turn 1 already hit ({turn1['cached_tokens']})")
    time.sleep(settle_s)
    if not client.reset_prefix_cache():
        return failures + [f"L={sized}: /reset_prefix_cache never succeeded"], {}
    turn2_msgs = turn1_msgs + [
        {"role": "assistant", "content": turn1["text"]},
        {"role": "user", "content": FOLLOW_UP},
    ]
    turn2 = ask_messages(client, turn2_msgs)
    ref = ask_messages(client, turn2_msgs, cache_salt=f"ref-{namespace}")
    p1 = turn1["prompt_tokens"]
    floor = (p1 - 1) // hit_block * hit_block - hit_block
    record = {"L": p1, "turn1": turn1, "turn2": turn2, "ref": ref}
    print(
        json.dumps(
            {
                "L": p1,
                "turn2_prompt": turn2["prompt_tokens"],
                "turn2_local": turn2["local_hits"],
                "turn2_ext": turn2["ext_hits"],
                "turn2_cached": turn2["cached_tokens"],
                "ref_cached": ref["cached_tokens"],
                "tokens_equal": turn2["token_ids"] == ref["token_ids"],
            }
        ),
        flush=True,
    )
    if turn2["local_hits"]:
        failures.append(f"L={p1}: turn 2 hit HBM after the reset")
    if turn2["ext_hits"] < max(floor, hit_block):
        failures.append(
            f"L={p1}: turn 2 recalled {turn2['ext_hits']} tokens from the store, "
            f"want >= {max(floor, hit_block)}"
        )
    if ref["cached_tokens"]:
        failures.append(f"L={p1}: the salted reference hit ({ref['cached_tokens']})")
    if turn2["token_ids"] is None or turn2["token_ids"] != ref["token_ids"]:
        failures.append(
            f"L={p1}: turn 2 differs from the uncached reference "
            f"(turn2={turn2['text']!r} ref={ref['text']!r})"
        )
    return failures, record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8123")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--dp-rank", type=int, default=0)
    ap.add_argument("--salt", default="multiturn-1788410251")
    ap.add_argument("--settle-s", type=float, default=10.0)
    ap.add_argument("--hit-block-size", type=int, default=1024)
    ap.add_argument(
        "--lengths",
        default="2200,9300,34000",
        help="comma-separated turn-1 prompt lengths to aim for",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    client = Client(
        args.base,
        args.model,
        args.timeout,
        args.max_tokens,
        dp_rank=None if args.dp_rank < 0 else args.dp_rank,
    )
    failures, records = [], []
    for aim in (int(x) for x in args.lengths.split(",")):
        f, rec = run_length(client, args.salt, aim, args.hit_block_size, args.settle_s)
        failures += f
        records.append(rec)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(records, fh)
    for f in failures:
        print("FAIL: " + f)
    print("MULTITURN_RECALL_OK" if not failures else "MULTITURN_RECALL_FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
