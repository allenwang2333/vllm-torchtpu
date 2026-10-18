# SPDX-License-Identifier: Apache-2.0
"""Greedy A/B of host-path flags: same tokens and same MTP acceptance, flags off vs on.

  python host_flags_greedy_ab.py run <out.json>      # one arm; flags come from the env
  python host_flags_greedy_ab.py compare <a.json> <b.json>

Greedy output alone cannot see a drafter regression (rejection keeps the target's
tokens), so the per-run draft/accepted counters are compared too: identical drafts
give identical counters on a deterministic greedy run.
"""

import json
import os
import sys

MODEL = os.environ.get("AB_MODEL", "Qwen/Qwen3.5-35B-A3B-FP8")
TP = int(os.environ.get("AB_TP", "8"))
K = int(os.environ.get("AB_K", "3"))

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number, with a docstring.",
    "List the planets of the solar system in order from the sun, one per line.",
    "Explain in three sentences why the sky is blue.",
    "Translate to French: The quick brown fox jumps over the lazy dog.",
    "Count from 1 to 40, separated by commas.",
    "What is 17 times 23? Show your working step by step.",
    "Give a short recipe for pancakes.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
]
MAX_TOKENS = [96, 64, 80, 48, 128, 72, 112, 56]


def run(out_path: str) -> None:
    from vllm import LLM, SamplingParams

    flags = {k: v for k, v in os.environ.items() if k.startswith(("TPU_", "VLLM_TPU"))}
    llm = LLM(
        model=MODEL,
        speculative_config={"method": "mtp", "num_speculative_tokens": K},
        tensor_parallel_size=TP,
        max_model_len=1024,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        async_scheduling=True,
        language_model_only=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        enable_prefix_caching=False,
        disable_log_stats=False,
    )
    params = [SamplingParams(temperature=0, max_tokens=m) for m in MAX_TOKENS]
    outs = llm.generate(PROMPTS, params)
    counters = {}
    for metric in llm.get_metrics():
        if metric.name.startswith("vllm:spec_decode_num_"):
            counters[metric.name] = counters.get(metric.name, 0) + getattr(
                metric, "value", 0
            )
    res = {
        "flags": flags,
        "token_ids": [list(o.outputs[0].token_ids) for o in outs],
        "texts": [o.outputs[0].text for o in outs],
        "counters": counters,
    }
    with open(out_path, "w") as f:
        json.dump(res, f)
    print("AB_RUN", out_path, json.dumps(counters), [len(t) for t in res["token_ids"]])
    llm.llm_engine.engine_core.shutdown()


def compare(a_path: str, b_path: str) -> int:
    with open(a_path) as fa, open(b_path) as fb:
        a, b = json.load(fa), json.load(fb)
    bad = 0
    for i, (ta, tb) in enumerate(zip(a["token_ids"], b["token_ids"], strict=True)):
        if ta != tb:
            bad += 1
            j = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
            print(f"AB_DIFF prompt {i}: first diff at {j}, lens {len(ta)}/{len(tb)}")
    print("AB_TOKENS", "identical" if not bad else f"{bad} prompt(s) differ")
    print("AB_COUNTERS a", json.dumps(a["counters"]), "b", json.dumps(b["counters"]))
    same_counters = a["counters"] == b["counters"] and bool(a["counters"])
    print("AB_COUNTERS", "identical" if same_counters else "DIFFER")
    print("AB_FLAGS_B_ONLY", sorted(set(b["flags"].items()) - set(a["flags"].items())))
    return 0 if (not bad and same_counters) else 1


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        sys.exit(compare(sys.argv[2], sys.argv[3]))
