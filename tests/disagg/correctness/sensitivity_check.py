"""Prove the correctness harness can fail, by making it fail on purpose.

A check that only ever passes tells you nothing, and this one has a specific
way of being vacuous: if the model's greedy answer did not actually depend on
the buried contents of the prompt, a corrupted KV transfer would produce the
same tokens and check_tokens.py would report a clean run.

So flip one digit of the access code planted at 8% depth, thousands of tokens
back, and see whether the generation moves. It does, and early: against the
Qwen3.5-35B reference the token streams diverge at position 23 of 128 for the
8k prompt and position 61 of 128 for the 64k one. The answer therefore depends
on context from the very start of the prompt, which is the region that early
per-layer shipping puts at risk.

Run this once per model, after generating a reference and against the same
server that generated it.
"""

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import prompt_set  # noqa: E402


def ask(url, model, text, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens, "temperature": 0.0, "seed": 42,
        "return_token_ids": True}).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["token_ids"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8077")
    p.add_argument("--model", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--lengths", nargs="*", default=["8k", "64k"])
    p.add_argument("--timeout", type=float, default=1800)
    args = p.parse_args()

    ref = json.load(open(args.ref))
    by_name = {r["name"]: r for r in ref["rows"]}
    insensitive = 0

    print(f"{'prompt':<8}{'prefix':>8}{'of':>6}  verdict")
    for name in args.lengths:
        _, text, sha, codes = prompt_set([name])[0]
        row = by_name[name]
        if row["prompt_sha"] != sha:
            print(f"{name}: reference predates the current prompts.py, "
                  f"regenerate it")
            return 2
        old = f"is {codes[0]}."
        new = f"is {(int(codes[0]) + 1) % 1000000:06d}."
        assert text.count(old) == 1, f"{name}: needle not found once"
        ids = ask(args.url, args.model, text.replace(old, new),
                  ref["max_tokens"], args.timeout)
        pre = 0
        for a, b in zip(row["token_ids"], ids):
            if a != b:
                break
            pre += 1
        same = pre == len(row["token_ids"]) == len(ids)
        insensitive += same
        print(f"{name:<8}{pre:>8}{len(row['token_ids']):>6}  "
              f"{'INSENSITIVE' if same else 'sensitive'}")

    if insensitive:
        print(f"\n{insensitive} prompt(s) ignored a changed access code. The "
              f"harness cannot detect a corrupt transfer on those.")
        return 1
    print("\nevery prompt's answer moved when the context moved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
