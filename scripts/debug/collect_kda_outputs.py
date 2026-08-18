# SPDX-License-Identifier: Apache-2.0
"""Collect deterministic outputs for a fixed prompt set (A/B between serves)."""
import json
import sys
import urllib.request

MODEL = "/mnt/lustre/hf-models/hub/models--moonshotai--Kimi-Linear-48B-A3B-Instruct/snapshots/e1df551a447157d4658b573f9a695d57658590e9"


def build_prompts():
    prompts = {}
    prompts["short-5tok"] = "The capital of France is"
    for n in (10, 24, 60, 120, 200):
        filler = "\n".join(f"Filler line {i:04d}: quantity {(i * 7) % 97}."
                           for i in range(n))
        prompts[f"ledger-{n}"] = (
            "Inventory ledger. Read every record before answering.\n" +
            filler + "\nThe secret code is 424242.\n" +
            "Question: what is the secret code? Answer with just the number.")
    return prompts


def main(port: str, out_path: str):
    results = {}
    for name, prompt in build_prompts().items():
        body = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 24,
            "temperature": 0,
            "return_token_ids": True,
        }
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        results[name] = resp["choices"][0].get("token_ids") or []
        print(name, resp["usage"]["prompt_tokens"], "->", results[name][:12])
    with open(out_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
