# SPDX-License-Identifier: Apache-2.0
"""Probe align-mode state restore with exact-length token-id prompts.

E2: producer ends exactly on a block boundary (state materialized there).
E1: producer runs past the boundary mid-block (boundary state unwritten).
Consumer outputs are compared across two runs (second run is a cache hit).
"""
import json
import sys
import urllib.request

from transformers import AutoTokenizer

PORT = sys.argv[1]
MODEL = "/mnt/lustre/hf-models/hub/models--moonshotai--Kimi-Linear-48B-A3B-Instruct/snapshots/e1df551a447157d4658b573f9a695d57658590e9"
BLOCK = 214

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa "
         "lambda mu nu xi omicron pi rho sigma tau upsilon").split()
WORD_IDS = [tok.encode(w, add_special_tokens=False)[0] for w in WORDS]


def exact_prefix(n):
    return [WORD_IDS[i % len(WORD_IDS)] for i in range(n)]


def complete_ids(ids, max_tokens=24):
    body = {
        "model": MODEL,
        "prompt": ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "return_token_ids": True,
    }
    req = urllib.request.Request(f"http://localhost:{PORT}/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    ch = resp["choices"][0]
    return (ch.get("token_ids")
            or [], (resp["usage"].get("prompt_tokens_details")
                    or {}).get("cached_tokens"))


def run_case(name, producer_len, consumer_extra):
    producer = exact_prefix(producer_len)
    consumer = producer + tok.encode("\nQuestion: name three Greek letters.",
                                     add_special_tokens=False)
    complete_ids(producer)  # populate the cache
    c1 = complete_ids(consumer)
    c2 = complete_ids(consumer)
    print(f"{name}: producer_len={producer_len} "
          f"consumer_cached_run1={c1[1]} run2={c2[1]} "
          f"run1==run2: {c1[0] == c2[0]}")
    print(f"  run1: {tok.decode(c1[0])[:60]!r}")
    print(f"  run2: {tok.decode(c2[0])[:60]!r}")


# E2: producer ends exactly on the 9-block boundary -> state@1926 in block 8.
run_case("E2-boundary-aligned", 9 * BLOCK, None)
# E1: producer ends 37 tokens past the boundary -> state@1926 unwritten.
run_case("E1-midblock", 9 * BLOCK + 37, None)
