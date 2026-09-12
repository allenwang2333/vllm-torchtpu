"""Does a prefill/decode split produce the same tokens as a single server?

This is the only check in the KV-transfer work that can catch a wrong answer.
Everything else we measure is timing, or is a checksum taken on the sending
side at one instant. If an early-layer transfer ships layer 4's bytes under
layer 3's address, the request still completes, the latency still looks
ordinary, and no counter moves. The only symptom is different tokens.

Two modes. `--save` runs the prompts against one ordinary server with no KV
transfer at all and writes what it saw. `--check` runs the same prompts
against a disaggregated deployment and compares. Sampling is off in both, so
identical inputs must give identical outputs and any difference is a bug.

Requests go one at a time on purpose. TPU attention kernels can pick different
code paths for different batch shapes, and those paths are not always bitwise
equal, so batching the reference differently from the comparison would produce
token differences that have nothing to do with the KV transfer.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from prompts import prompt_set  # noqa: E402


def complete(url, model, text, max_tokens, timeout, endpoint):
    """One greedy generation. Returns (text, token_ids, prompt_tokens).

    The chat endpoint is the default because this is an instruct model and the
    raw completion endpoint does not apply its chat template. Without the
    template it ignores the question and emits a run of a single repeated
    token, which is worse than useless here: a degenerate answer is the same
    whatever the KV cache holds, so a corrupt transfer would sail through.
    """
    common = {"model": model, "max_tokens": max_tokens, "temperature": 0.0,
              "seed": 42, "return_token_ids": True}
    if endpoint == "chat":
        path, body = "/v1/chat/completions", dict(
            common, messages=[{"role": "user", "content": text}])
    else:
        path, body = "/v1/completions", dict(common, prompt=text)
    req = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    choice = out["choices"][0]
    got = (choice["message"]["content"] if endpoint == "chat"
           else choice["text"])
    return got, choice.get("token_ids"), out["usage"]["prompt_tokens"]


def run(args):
    rows = []
    for name, text, sha, codes in prompt_set():
        if args.lengths and name not in args.lengths:
            continue
        t0 = time.perf_counter()
        try:
            got, ids, ptok = complete(args.url, args.model, text,
                                      args.max_tokens, args.timeout,
                                      args.endpoint)
        except urllib.error.HTTPError as exc:
            print(f"{name}: HTTP {exc.code} {exc.read()[:400]!r}")
            raise
        # return_token_ids is the point of this harness, but if the server is
        # older than the field we still learn most of what we need from the
        # text. Say so loudly rather than silently comparing something weaker.
        if ids is None:
            print(f"{name}: WARNING server returned no token_ids, "
                  f"falling back to text comparison")
        rows.append({
            "name": name,
            "prompt_sha": sha,
            "prompt_tokens": ptok,
            "token_ids": ids,
            "text": got,
            "expected_codes": codes,
            "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
        })
        found = sum(c in got for c in codes)
        print(f"{name}: {ptok} prompt tokens, {len(ids) if ids else '?'} out, "
              f"{found}/{len(codes)} codes recalled, "
              f"{rows[-1]['wall_ms']:.0f} ms")
    return rows


def common_prefix(a, b):
    n = 0
    for x, y in zip(a or [], b or []):
        if x != y:
            break
        n += 1
    return n


def compare(ref_rows, new_rows):
    """Report per-prompt agreement and return the number of mismatches.

    The common-prefix length is printed alongside the verdict because the two
    failure modes look completely different. A KV transfer that ships the
    wrong bytes diverges within the first few tokens, since the planted codes
    are the first thing the answer emits. A one-token difference near the end
    is far more likely to be ordinary kernel nondeterminism, and is worth
    seeing rather than hiding behind a bare FAIL.
    """
    by_name = {r["name"]: r for r in ref_rows}
    bad = 0
    print(f"\n{'prompt':<8}{'ref tok':>9}{'new tok':>9}{'prefix':>8}"
          f"{'codes':>8}  verdict")
    for new in new_rows:
        ref = by_name.get(new["name"])
        if ref is None:
            print(f"{new['name']:<8}{'-':>9}{'-':>9}{'-':>8}{'-':>8}  "
                  f"NOT IN REFERENCE")
            bad += 1
            continue
        if ref["prompt_sha"] != new["prompt_sha"]:
            print(f"{new['name']:<8}  prompt sha changed "
                  f"({ref['prompt_sha']} -> {new['prompt_sha']}); "
                  f"prompts.py was edited, regenerate the reference")
            bad += 1
            continue
        if ref["token_ids"] and new["token_ids"]:
            same = ref["token_ids"] == new["token_ids"]
            pre = common_prefix(ref["token_ids"], new["token_ids"])
            nref, nnew = len(ref["token_ids"]), len(new["token_ids"])
        else:
            same = ref["text"] == new["text"]
            pre = -1
            nref = nnew = -1
        found = sum(c in new["text"] for c in new["expected_codes"])
        bad += not same
        print(f"{new['name']:<8}{nref:>9}{nnew:>9}{pre:>8}"
              f"{found:>5}/{len(new['expected_codes'])}  "
              f"{'match' if same else 'MISMATCH'}")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8077")
    p.add_argument("--model", required=True)
    p.add_argument("--save", metavar="PATH",
                   help="write a reference file from this deployment")
    p.add_argument("--check", metavar="PATH",
                   help="compare this deployment against a reference file")
    p.add_argument("--lengths", nargs="*", default=None,
                   help="subset of prompt names, e.g. 2k 8k")
    p.add_argument("--endpoint", choices=("chat", "completions"),
                   default="chat")
    # 128 rather than a handful. Every generated token is another chance for a
    # damaged KV cache to show itself, and the model's default template opens
    # with a reasoning trace that quotes the prompt back, so a long budget buys
    # a long content-derived answer for the price of a few decode steps.
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--timeout", type=float, default=1800)
    args = p.parse_args()
    if bool(args.save) == bool(args.check):
        p.error("pass exactly one of --save or --check")

    rows = run(args)

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"model": args.model, "max_tokens": args.max_tokens,
                       "rows": rows}, f, indent=1, sort_keys=True)
        print(f"\nwrote reference for {len(rows)} prompts to {args.save}")
        return 0

    with open(args.check) as f:
        ref = json.load(f)
    if ref["max_tokens"] != args.max_tokens:
        print(f"reference used max_tokens={ref['max_tokens']}, "
              f"this run used {args.max_tokens}")
        return 2
    bad = compare(ref["rows"], rows)
    print(f"\n{len(rows) - bad}/{len(rows)} prompts match the reference")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
