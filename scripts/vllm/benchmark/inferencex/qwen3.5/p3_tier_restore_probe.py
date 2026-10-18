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
"""Known-answer probe for KV restored from the prefill leg's host tier.

Runs against a live disaggregated stack (proxy, prefill and decode legs):

1. Turn 1 of N conversations: a long context (default about 90k tokens)
   holding K planted records ("the access code for the vault named X is
   NNNNNN") at spread-out depths.
2. Control: turn 2 of every conversation right away, asking for the codes.
   The prefill leg serves the context from HBM.
3. Churn: unrelated long single-turn prompts, enough to turn over every
   leg's KV pool (read from their /metrics), so no leg holds the
   conversations in HBM afterwards. The proxy spreads the churn over the legs
   of each side, so a side with several legs takes a multiple of its largest
   pool (churn_plan).
4. Restore: the same turn-2 requests again. The prefill leg now has to load
   the context from its host tier.

The run passes when the restore phase answers at least as many records as the
control minus a tolerance (2 by default; outputs are not compared byte for
byte), and the prefill leg's counters show the restore phase was served by
external (tier) hits rather than local HBM hits, with the decode leg not
holding the prefixes either. Otherwise it is FAIL, or INCONCLUSIVE when the
control itself is weak or the restore phase was not actually served from the
tier.

Standard library only, so it runs in the benchmark image. Every request's
prompt length is recorded, which lets the prefill leg's HIT_TRACE lines
(prompt=, local=, ext=, computed_before=) be matched offline.

Phase 3 harness port of agentx-p1-tier scripts/vllm/benchmark/inferencex/qwen3.5/
tier_restore_probe.py (unchanged logic). Added: an environment id line
(TIER_PROBE ENVIRONMENT_ID ...: P3_IMAGE / SERVER_IMAGE, the servers' /version
answers, P3_SOURCE_SHA or the P3_SNAPSHOT.json sha, job id, layout knobs from
the environment), a per-run output directory under --out (default
/workspace/tmp/p3-harness/jobs/<job>/tier_probe-<utc>/ when that root exists),
per-phase wall windows (turn1, control, churn, restore) in the summary, and
--context-tokens / --convs / --facts / --churn-factor / --concurrency kept as
the knobs phase 4 sweeps (FA/GDN joint hit, replication, load latency and
capacity pressure are read from the legs' metrics as before).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

_WORDS = (  # noqa: SIM905
    "amber anchor apple arch ash autumn badge bamboo barrel basin beacon bell birch "
    "blade bloom bolt bramble brass breeze brick bridge brook buckle cabin candle "
    "canyon cargo cedar chalk chapel cinder cliff clover cobalt comet coral cotton "
    "crane crater crest crystal current dagger delta desert dial dune ember engine "
    "falcon feather fern field flint forge fountain fox frost garnet glacier glade "
    "granite harbor hazel hearth helm heron hollow horizon iris island ivory jasper "
    "juniper kettle lagoon lantern lark ledger lily linen maple marble meadow mesa "
    "mill mint mirror moss nectar needle oak ochre olive onyx orchard otter paddle "
    "pearl pebble pepper pine plume prairie quarry quartz quill raven reed ridge "
    "river robin rope saddle sage sail sand scroll shale shell silk slate spire "
    "spruce stone summit swan thistle thorn timber topaz torch tower trail tulip "
    "tundra valley velvet violet walnut willow wren yarrow zephyr"
).split()
_VERBS = (  # noqa: SIM905
    "follows crosses guards carries lifts circles watches mends shapes gathers "
    "marks holds bends turns reaches trims"
).split()
# Tokens per word of the filler, measured on the first turn and used only to
# size the context; the report carries the real prompt lengths.
_TOKENS_PER_WORD = 1.45

_METRIC_RE = re.compile(r"^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$")


def _metrics(url: str) -> dict[str, float]:
    """Sums each vllm: series of one /metrics endpoint over its labels."""
    text = urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=30).read()
    out: dict[str, float] = {}
    for line in text.decode().splitlines():
        m = _METRIC_RE.match(line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def _pool_tokens(url: str) -> int:
    """KV pool size in tokens, from vllm:cache_config_info's labels."""
    text = urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=30).read()
    for line in text.decode().splitlines():
        if line.startswith("vllm:cache_config_info{"):
            labels = dict(re.findall(r'(\w+)="([^"]*)"', line))
            blocks = labels.get("num_gpu_blocks") or labels.get(
                "num_gpu_blocks_override"
            )
            if blocks and blocks != "None":
                return int(blocks) * int(labels.get("block_size", "16"))
    return 0


def _proxy_leg_counts(proxy: str) -> tuple[int, int] | None:
    """Prefill and decode leg counts the proxy routes over, from /healthcheck."""
    try:
        status = json.loads(
            urllib.request.urlopen(
                proxy.rstrip("/") + "/healthcheck", timeout=30
            ).read()
        )
        return int(status["prefill_instances"]), int(status["decode_instances"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


# Headroom on a side with several legs: prefix_hash and running-balanced
# routing do not split the churn evenly over them.
_LEG_MARGIN = 1.1


def churn_plan(
    prefill_pools: list[int],
    decode_pools: list[int],
    prefill_legs: int,
    decode_legs: int,
    churn_factor: float,
    context_tokens: int,
) -> dict:
    """Sizes the churn phase so every leg's KV pool turns over.

    The churn of one side is spread over that side's legs, so each side needs
    its largest pool times a leg factor: 1 for a single leg, legs *
    _LEG_MARGIN otherwise. The churn budget is churn_factor times the larger
    of the two needs; leg_factor reports that need over the largest pool of
    either side.
    """

    def leg_factor(legs: int) -> float:
        return 1.0 if legs <= 1 else legs * _LEG_MARGIN

    pool_tokens = max(prefill_pools + decode_pools + [0])
    need = max(
        max(prefill_pools + [0]) * leg_factor(prefill_legs),
        max(decode_pools + [0]) * leg_factor(decode_legs),
    )
    return {
        "pool_tokens": pool_tokens,
        "leg_factor": need / pool_tokens if pool_tokens else 1.0,
        "prompts": max(1, math.ceil(churn_factor * need / max(context_tokens, 1))),
    }


def _counters(urls: list[str]) -> dict[str, float]:
    total: dict[str, float] = {}
    for url in urls:
        for key, value in _metrics(url).items():
            total[key] = total.get(key, 0.0) + value
    return total


def _delta(after: dict[str, float], before: dict[str, float], *names: str) -> float:
    for name in names:
        if name in after:
            return after[name] - before.get(name, 0.0)
    return 0.0


class Client:
    def __init__(self, proxy: str, model: str, timeout: float):
        self.url = proxy.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict], max_tokens: int) -> dict:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            }
        ).encode()
        last_error: Exception | None = None
        for attempt in range(3):
            request = urllib.request.Request(
                self.url, data=body, headers={"Content-Type": "application/json"}
            )
            start = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read())
                return {
                    "text": payload["choices"][0]["message"].get("content") or "",
                    "prompt_tokens": payload.get("usage", {}).get("prompt_tokens"),
                    "seconds": time.monotonic() - start,
                }
            except (urllib.error.URLError, ConnectionError, TimeoutError) as err:
                last_error = err
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"request failed after retries: {last_error}")


def _filler(rng: random.Random, words: int) -> list[str]:
    sentences = []
    count = 0
    n = 0
    while count < words:
        n += 1
        sentence = (
            f"Entry {n}: the {rng.choice(_WORDS)} {rng.choice(_WORDS)} "
            f"{rng.choice(_VERBS)} the {rng.choice(_WORDS)} near the "
            f"{rng.choice(_WORDS)} {rng.choice(_WORDS)}."
        )
        sentences.append(sentence)
        count += len(sentence.split())
    return sentences


def build_conversation(seed: int, context_tokens: int, facts: int) -> dict:
    """A long context with `facts` planted records at spread-out depths."""
    rng = random.Random(seed)
    sentences = _filler(rng, int(context_tokens / _TOKENS_PER_WORD))
    records = []
    used: set[str] = set()
    for i in range(facts):
        while True:
            name = f"{rng.choice(_WORDS).title()} {rng.choice(_WORDS).title()}"
            if name not in used:
                used.add(name)
                break
        code = f"{rng.randrange(100000, 1000000)}"
        depth = (i + 0.5) / facts
        records.append({"name": name, "code": code, "depth": round(depth, 3)})
    for record in sorted(records, key=lambda r: -r["depth"]):
        at = int(record["depth"] * len(sentences))
        sentences.insert(
            at,
            f"Important record: the access code for the vault named "
            f"{record['name']} is {record['code']}.",
        )
    context = " ".join(sentences)
    turn1 = (
        f"{context}\n\nRead the text above carefully. "
        "Reply with exactly one word: READY."
    )
    names = [r["name"] for r in records]
    rng.shuffle(names)
    turn2 = (
        "From the text above, give the access code of each of these vaults, one "
        "per line in the form 'Name: code': " + "; ".join(names) + "."
    )
    return {"seed": seed, "records": records, "turn1": turn1, "turn2": turn2}


def score(text: str, records: list[dict]) -> int:
    return sum(1 for record in records if record["code"] in text)


def _environment_id() -> dict:
    """Image, server versions, source sha, job and layout knobs, from the
    environment and the servers; standard library only."""
    env = os.environ
    out = {
        "image": env.get("P3_IMAGE") or env.get("SERVER_IMAGE") or env.get("IMAGE") or "",
        "client_image": env.get("CLIENT_IMAGE", ""),
        "source_sha": env.get("P3_SOURCE_SHA", ""),
        "job": env.get("CDK_JOB_ID") or env.get("JS_NAME") or "",
        "layout": {k: env[k] for k in sorted(env) if k.endswith((
            "TENSOR_PARALLEL_SIZE_PREFILL", "TENSOR_PARALLEL_SIZE_DECODE",
            "PREFILL_CONTEXT_PARALLEL_SIZE_PREFILL", "DECODE_CONTEXT_PARALLEL_SIZE_DECODE",
            "DATA_PARALLEL_SIZE_PREFILL", "DATA_PARALLEL_SIZE_DECODE",
            "PIPELINE_PARALLEL_SIZE_PREFILL", "PIPELINE_PARALLEL_SIZE_DECODE",
            "KV_OFFLOAD_BYTES_PREFILL", "TPU_RAIDEN_TIER_DEDUP", "BLOCK_SIZE"))},
        "servers": {},
    }
    snap = "/workspace/vllm-torchtpu/P3_SNAPSHOT.json"
    if not out["source_sha"] and os.path.exists(snap):
        try:
            out["source_sha"] = json.load(open(snap)).get("source_sha", "")
        except (OSError, ValueError):
            pass
    for key in ("VLLM_URLS", "DECODE_URLS"):
        for url in env.get(key, "").split():
            try:
                out["servers"][url] = json.loads(
                    urllib.request.urlopen(url.rstrip("/") + "/version", timeout=10).read()
                )
            except Exception as err:  # noqa: BLE001
                out["servers"][url] = f"unavailable: {type(err).__name__}"
    return out


def _run_all(fn, items, concurrency):
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(fn, items))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--proxy", default=os.environ.get("PROBE_PROXY_URL", ""))
    ap.add_argument(
        "--prefill-urls",
        default=os.environ.get("VLLM_URLS", ""),
        help="space-separated prefill server base URLs (for /metrics)",
    )
    ap.add_argument(
        "--decode-urls",
        default=os.environ.get("DECODE_URLS", ""),
        help="space-separated decode server base URLs (for /metrics)",
    )
    ap.add_argument("--model", default=os.environ.get("PROBE_MODEL", ""))
    ap.add_argument(
        "--convs", type=int, default=int(os.environ.get("PROBE_CONVS", "8"))
    )
    ap.add_argument(
        "--facts", type=int, default=int(os.environ.get("PROBE_FACTS", "4"))
    )
    ap.add_argument(
        "--context-tokens",
        type=int,
        default=int(os.environ.get("PROBE_CONTEXT_TOKENS", "90000")),
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("PROBE_CONCURRENCY", "4")),
    )
    ap.add_argument(
        "--churn-factor",
        type=float,
        default=float(os.environ.get("PROBE_CHURN_FACTOR", "1.3")),
        help="churn tokens as a multiple of the larger leg's KV pool, times "
        "the leg factor of a side with several legs (churn_plan)",
    )
    ap.add_argument(
        "--tolerance",
        type=int,
        default=int(os.environ.get("PROBE_TOLERANCE", "2")),
        help="records the restore phase may answer fewer than the control",
    )
    ap.add_argument(
        "--seed", type=int, default=int(os.environ.get("PROBE_SEED", "20261016"))
    )
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument(
        "--out",
        default=os.environ.get("PROBE_OUT", ""),
        help="output directory; empty: a fresh per-run directory under "
        "/workspace/tmp/p3-harness/jobs/<job>/ (or ./tier_probe-<utc>)",
    )
    args = ap.parse_args()
    if not args.out:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        job = os.environ.get("CDK_JOB_ID") or os.environ.get("JS_NAME") or "local"
        root = "/workspace/tmp/p3-harness/jobs"
        args.out = (
            os.path.join(root, job, f"tier_probe-{stamp}")
            if os.path.isdir(root) else f"tier_probe-{stamp}"
        )
    env_id = _environment_id()
    print("TIER_PROBE ENVIRONMENT_ID " + json.dumps(env_id), flush=True)
    windows: dict[str, dict] = {}

    prefill_urls = args.prefill_urls.split()
    decode_urls = args.decode_urls.split()
    if not args.proxy or not prefill_urls:
        print("TIER_PROBE: --proxy and --prefill-urls are required", flush=True)
        return 2
    model = args.model
    if not model and os.environ.get("SERVED_MODEL"):
        # The agentx recipe's servers answer to Qwen/<SERVED_MODEL>.
        model = f"Qwen/{os.environ['SERVED_MODEL']}"
    if not model:
        listing = json.loads(
            urllib.request.urlopen(
                args.proxy.rstrip("/") + "/v1/models", timeout=30
            ).read()
        )
        model = listing["data"][0]["id"]
    client = Client(args.proxy, model, args.timeout)
    os.makedirs(args.out, exist_ok=True)
    log: list[dict] = []

    def record(phase: str, conv: int, result: dict, matches: int | None) -> None:
        entry = {"phase": phase, "conv": conv, **result, "matches": matches}
        log.append(entry)
        print(
            f"TIER_PROBE {phase} conv={conv} prompt_tokens={result['prompt_tokens']} "
            f"seconds={result['seconds']:.1f} matches={matches}",
            flush=True,
        )

    convs = [
        build_conversation(args.seed + i, args.context_tokens, args.facts)
        for i in range(args.convs)
    ]
    total_records = args.convs * args.facts

    # 1. Turn 1.
    def turn1(i: int) -> dict:
        result = client.chat([{"role": "user", "content": convs[i]["turn1"]}], 8)
        convs[i]["assistant"] = result["text"]
        record("turn1", i, result, None)
        return result

    _run_all(turn1, range(args.convs), args.concurrency)

    def turn2_messages(i: int) -> list[dict]:
        c = convs[i]
        return [
            {"role": "user", "content": c["turn1"]},
            {"role": "assistant", "content": c["assistant"]},
            {"role": "user", "content": c["turn2"]},
        ]

    def turn2(phase: str):
        def run(i: int) -> int:
            result = client.chat(turn2_messages(i), 160)
            matches = score(result["text"], convs[i]["records"])
            record(phase, i, result, matches)
            return matches

        return run

    phases: dict[str, dict] = {}

    def run_phase(name: str, fn, items) -> list:
        windows[name] = {"t0": time.time()}
        p0, d0 = _counters(prefill_urls), _counters(decode_urls)
        start = time.monotonic()
        results = _run_all(fn, items, args.concurrency)
        p1, d1 = _counters(prefill_urls), _counters(decode_urls)
        prompt_tokens = sum(e["prompt_tokens"] or 0 for e in log if e["phase"] == name)
        phases[name] = {
            "seconds": round(time.monotonic() - start, 1),
            "prompt_tokens": prompt_tokens,
            "prefill_local_hit_tokens": _delta(
                p1, p0, "vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"
            ),
            "prefill_external_hit_tokens": _delta(
                p1,
                p0,
                "vllm:external_prefix_cache_hits_total",
                "vllm:external_prefix_cache_hits",
            ),
            "prefill_prompt_tokens": _delta(
                p1, p0, "vllm:prompt_tokens_total", "vllm:prompt_tokens"
            ),
            "decode_local_hit_tokens": _delta(
                d1, d0, "vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"
            ),
        }
        windows[name]["t1"] = time.time()
        windows[name]["wall_s"] = round(windows[name]["t1"] - windows[name]["t0"], 1)
        print(f"TIER_PROBE phase {name}: {json.dumps(phases[name])}", flush=True)
        return results

    # 2. Control, served from HBM.
    control = sum(run_phase("control", turn2("control"), range(args.convs)))

    # 3. Churn past every leg's KV pool.
    legs = _proxy_leg_counts(args.proxy)
    legs_source = "proxy"
    if legs is None:
        legs, legs_source = (len(prefill_urls), len(decode_urls)), "urls"
    plan = churn_plan(
        [_pool_tokens(u) for u in prefill_urls],
        [_pool_tokens(u) for u in decode_urls],
        prefill_legs=legs[0],
        decode_legs=legs[1],
        churn_factor=args.churn_factor,
        context_tokens=args.context_tokens,
    )
    pool_tokens, churn_count = plan["pool_tokens"], plan["prompts"]
    print(
        f"TIER_PROBE churn: larger KV pool {pool_tokens} tokens, "
        f"{legs[0]}P{legs[1]}D ({legs_source}), leg factor {plan['leg_factor']:.2f} "
        f"-> {churn_count} prompts of ~{args.context_tokens} tokens",
        flush=True,
    )

    def churn(j: int) -> None:
        c = build_conversation(args.seed + 100000 + j, args.context_tokens, 1)
        result = client.chat([{"role": "user", "content": c["turn1"]}], 1)
        record("churn", j, result, None)

    run_phase("churn", churn, range(churn_count))

    # 4. Restore: the same turn-2 requests, now from the host tier.
    restore = sum(run_phase("restore", turn2("restore"), range(args.convs)))

    r = phases["restore"]
    tokens = max(r["prompt_tokens"], 1)
    external_fraction = r["prefill_external_hit_tokens"] / tokens
    local_fraction = r["prefill_local_hit_tokens"] / tokens
    decode_fraction = r["decode_local_hit_tokens"] / tokens
    reasons = []
    if control < 0.75 * total_records:
        reasons.append(f"control answered only {control}/{total_records}")
    if external_fraction < 0.5:
        reasons.append(
            f"restore phase took {external_fraction:.2f} of its prompt from the tier"
        )
    if decode_fraction > 0.5:
        reasons.append(
            f"decode leg still held {decode_fraction:.2f} of the restore prompts"
        )
    if reasons:
        verdict = "INCONCLUSIVE"
    elif restore >= control - args.tolerance:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    summary = {
        "verdict": verdict,
        "reasons": reasons,
        "records": total_records,
        "control_matches": control,
        "restore_matches": restore,
        "tolerance": args.tolerance,
        "restore_external_fraction": round(external_fraction, 3),
        "restore_local_fraction": round(local_fraction, 3),
        "restore_decode_local_fraction": round(decode_fraction, 3),
        "churn_prompts": churn_count,
        "pool_tokens": pool_tokens,
        "churn_leg_factor": round(plan["leg_factor"], 3),
        "legs": {"prefill": legs[0], "decode": legs[1], "source": legs_source},
        "phases": phases,
        "windows": windows,
        "model": model,
        "environment_id": env_id,
        "args": vars(args),
    }
    with open(os.path.join(args.out, "tier_probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "tier_probe_requests.jsonl"), "w") as f:
        for entry in log:
            f.write(json.dumps(entry) + "\n")
    print(
        f"TIER_PROBE RESULT: {verdict} control={control}/{total_records} "
        f"restore={restore}/{total_records} tolerance={args.tolerance} "
        f"restore_external={external_fraction:.2f} restore_local={local_fraction:.2f} "
        f"decode_local={decode_fraction:.2f}"
        + (f" ({'; '.join(reasons)})" if reasons else ""),
        flush=True,
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
