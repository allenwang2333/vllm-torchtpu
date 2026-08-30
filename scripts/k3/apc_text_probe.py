#!/usr/bin/env python3
"""Prefix-cache correctness probe scored on retrieved facts, not byte equality.

The token-ID probe (apc_probe.py) feeds the model random IDs, which is maximally
sensitive to state corruption but also maximally sensitive to noise: measured on
a K3 server with prefix caching switched off entirely, two identical runs
differed on 11 of 33 responses. Garbage context leaves the top-2 logits nearly
tied, so any run-to-run numeric variation flips a token. Byte equality on those
prompts is not a test, it is a coin.

So score something the noise cannot move. Each prompt is a document of numbered
facts followed by a question about one line, and the check is whether the answer
contains that line's number. A prefix-cache hit that restores attention KV
without the matching KDA recurrent state does not perturb the last digit -- it
loses the document, and the answer stops being retrievable at all.

Structure, per document:
  cold         nothing can hit (the cache is reset first)
  warm         same document, different question -- the shared prefix hits
  interleaved  documents round-robin, so each request resumes into a slot whose
               previous occupant held a different document. This is the case
               that catches contamination; a same-document repeat would not.

  ENDPOINT=http://... python3 scripts/k3/apc_text_probe.py --out probe.json
"""
import argparse
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request

MODEL = os.environ.get("MODEL", "moonshotai/Kimi-K3")

NOUNS = ["harbor", "lantern", "quarry", "meadow", "bridge", "cellar", "orchard",
         "beacon", "furnace", "trellis", "cistern", "granary", "windmill",
         "aqueduct", "belfry", "cloister"]
ADJS = ["copper", "silent", "narrow", "frozen", "amber", "hollow", "crimson",
        "distant", "gilded", "rugged", "sunken", "velvet"]


def build_documents(n_docs, n_lines):
    """Documents long enough to span several pool blocks, with unique facts.

    Every document draws from the same vocabulary but a different value stream,
    so a contaminated answer is a value that exists -- just in the wrong
    document. An answer drawn from thin air and an answer stolen from a
    neighbour are different failures and worth telling apart.
    """
    docs = []
    for d in range(n_docs):
        rng = random.Random(90210 + d)
        lines, facts = [], {}
        for i in range(1, n_lines + 1):
            value = rng.randrange(1000, 9999)
            facts[i] = value
            lines.append(f"Record {i}: the {rng.choice(ADJS)} "
                         f"{rng.choice(NOUNS)} holds {value} crates.")
        docs.append({"id": f"doc{d}", "text": "\n".join(lines), "facts": facts})
    return docs


def make_prompt(doc, line):
    return (f"{doc['text']}\n\n"
            f"Question: According to the records above, how many crates does "
            f"record {line} hold?\nAnswer: Record {line} holds")


def post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ask(endpoint, doc, line, phase, max_tokens, timeout):
    payload = {"model": MODEL, "prompt": make_prompt(doc, line),
               "max_tokens": max_tokens, "temperature": 0}
    row = {"doc": doc["id"], "line": line, "phase": phase,
           "expected": doc["facts"][line]}
    try:
        body = post(f"{endpoint}/v1/completions", payload, timeout)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        row["error"] = repr(exc)
        return row
    usage = body.get("usage", {})
    details = usage.get("prompt_tokens_details") or {}
    text = body["choices"][0]["text"]
    row.update({
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "text": text,
        # The answer is correct if the expected value appears; "which other
        # document's value did it give" is the contamination signature.
        "correct": str(doc["facts"][line]) in text,
        # The graded quantity. Two identical runs against a cache-free server
        # agreed on all 24 answers and disagreed on 13 of 24 full completions:
        # once the question is answered the model free-runs, and that tail is
        # noise. Cross-arm comparison uses this field for the same reason.
        "answer": (re.findall(r"\d{4}", text) or [None])[0],
        "numbers": re.findall(r"\d{4}", text)[:4],
    })
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=4)
    ap.add_argument("--lines", type=int, default=60)
    ap.add_argument("--queries", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    if not args.endpoint:
        ap.error("set --endpoint or ENDPOINT")
    endpoint = args.endpoint.rstrip("/")

    docs = build_documents(args.docs, args.lines)
    # Spread the queried lines across the document so at least one sits well
    # past the first pool block, where a lost prefix shows up.
    step = max(1, args.lines // (args.queries + 1))
    queried = [min(args.lines, step * (k + 1)) for k in range(args.queries)]

    try:
        post(f"{endpoint}/reset_prefix_cache", {}, 60)
        print("reset_prefix_cache: ok")
    except Exception as exc:  # noqa: BLE001 - informational
        print(f"reset_prefix_cache: {exc}")

    rows = []
    print(f"\n{'phase':<12}{'doc':<7}{'line':>5}{'prompt':>8}{'cached':>8}"
          f"{'ok':>4}  answer")
    def emit(row):
        rows.append(row)
        if "error" in row:
            print(f"{row['phase']:<12}{row['doc']:<7}{row['line']:>5}"
                  f"       ERROR {row['error'][:60]}")
            return
        print(f"{row['phase']:<12}{row['doc']:<7}{row['line']:>5}"
              f"{row['prompt_tokens']:>8}{row['cached_tokens']!s:>8}"
              f"{'y' if row['correct'] else 'N':>4}  {row['text'][:40]!r}")

    # cold: every document once, nothing can hit
    for doc in docs:
        emit(ask(endpoint, doc, queried[0], "cold", args.max_tokens, args.timeout))
    # warm: same documents, later questions -- the document prefix hits
    for doc in docs:
        for line in queried[1:]:
            emit(ask(endpoint, doc, line, "warm", args.max_tokens, args.timeout))
    # interleaved: round-robin, so consecutive requests never share a document
    for line in queried:
        for doc in docs:
            emit(ask(endpoint, doc, line, "interleaved", args.max_tokens,
                     args.timeout))

    ok = sum(1 for r in rows if r.get("correct"))
    total = len(rows)
    by_phase = {}
    for r in rows:
        p = by_phase.setdefault(r["phase"], [0, 0])
        p[1] += 1
        p[0] += 1 if r.get("correct") else 0
    # Contamination: a wrong answer that is some *other* document's value for
    # the same line is the pooled failure mode, and looks nothing like a
    # numerically flipped digit.
    all_values = {d["id"]: d["facts"] for d in docs}
    contaminated = []
    for r in rows:
        if r.get("correct") or "text" not in r:
            continue
        for did, facts in all_values.items():
            if did != r["doc"] and str(facts.get(r["line"])) in r["text"]:
                contaminated.append((r["doc"], r["line"], r["phase"], did))
    out = {"endpoint": endpoint, "docs": args.docs, "lines": args.lines,
           "rows": rows, "correct": ok, "total": total,
           "by_phase": by_phase, "contaminated": contaminated,
           "total_cached_tokens": sum(r.get("cached_tokens") or 0 for r in rows)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"\nretrieval accuracy {ok}/{total}")
    for phase, (c, t) in by_phase.items():
        print(f"  {phase:<12} {c}/{t}")
    print(f"cross-document contamination: {len(contaminated)}")
    for row in contaminated:
        print(f"  doc={row[0]} line={row[1]} phase={row[2]} answered with {row[3]}'s value")
    print(f"total cached_tokens: {out['total_cached_tokens']}")
    print(f"-> {args.out}")
    return 0 if ok == total and not contaminated else 1


if __name__ == "__main__":
    sys.exit(main())
