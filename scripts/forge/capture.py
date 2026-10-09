"""Read a running vLLM server for a forge scorer arm on a Buildkite agent.

    capture.py counters --port P
        One reading of the server's step counters: a `# <unix time>` line, then
        its `vllm:iteration_tokens_total` buckets, `vllm:prompt_tokens_total`,
        and `vllm:cache_config_info`, which names each engine's KV blocks and
        block size.
    capture.py capture --port P --suite S.json --out C.json [--no-logprobs] [--batch N]
        An output fingerprint, as `perfkit numerics capture` takes one: the same
        requests, the same capture file, the same digest, and the same line on
        stdout. Serial by default, one request of N rows with --batch.

The agent has no perfkit, so this file stands alone on the standard library.
`perfkit/numerics.py` is the reference, and tests/test_buildkite.py holds the
two to the same requests and the same digest.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

DEFAULT_MAX_TOKENS = 64


def _get(port, path, timeout=30):
    with urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=timeout) as r:
        return r.read().decode()


def _post(port, path, payload, timeout=900):
    req = urllib.request.Request(f"http://localhost:{port}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode().strip()
    return json.loads(body) if body else {}


def counters(port):
    text = _get(port, "/metrics")
    keep = [line for line in text.splitlines()
            if line.startswith(("vllm:iteration_tokens_total_bucket", "vllm:prompt_tokens_total",
                                "vllm:cache_config_info"))]
    if keep:
        print(f"# {int(time.time())}")
        print("\n".join(keep))


def token_ids(choice, lp):
    """`perfkit.numerics._token_ids`."""
    if isinstance(choice.get("token_ids"), list):
        return choice["token_ids"]
    toks = lp.get("tokens")
    if isinstance(toks, list):
        return [int(hashlib.sha1(str(t).encode()).hexdigest()[:8], 16) for t in toks]
    text = choice.get("text", "")
    return [int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)]


def item(ident, choice):
    lp = choice.get("logprobs") or {}
    return {
        "id": ident,
        "token_ids": token_ids(choice, lp),
        "text": choice.get("text", ""),
        "logprobs": [None if v is None else round(float(v), 6)
                     for v in (lp.get("token_logprobs") or [])],
        "finish_reason": choice.get("finish_reason"),
    }


def digest(items):
    """`perfkit.numerics.Capture.digest`."""
    h = hashlib.sha256()
    for it in sorted(items, key=lambda i: str(i.get("id", ""))):
        h.update(json.dumps([str(it.get("id", "")), it.get("token_ids") or []],
                            separators=(",", ":")).encode())
    return h.hexdigest()[:16]


def serial(port, model, prompts, max_tokens, logprobs):
    """`perfkit.numerics.capture`: one request at a time."""
    items = []
    for i, p in enumerate(prompts):
        want = min(p.get("max_tokens", max_tokens), max_tokens) if max_tokens else \
            p.get("max_tokens", DEFAULT_MAX_TOKENS)
        payload = {"model": model or p.get("model", ""), "prompt": p["prompt"],
                   "max_tokens": want, "temperature": 0.0, "top_p": 1.0,
                   "seed": p.get("seed", 0), "stream": False}
        if logprobs is not None:
            payload["logprobs"] = logprobs
        resp = _post(port, "/v1/completions", payload)
        items.append(item(p.get("id", f"p{i}"), resp["choices"][0]))
    return items, max_tokens


def batched(port, model, prompts, max_tokens, logprobs, rows):
    """`perfkit.numerics.capture_batched`: every row in one request."""
    picked = [prompts[i % len(prompts)] for i in range(rows)]
    want = min(min(p.get("max_tokens", max_tokens) for p in picked), max_tokens) \
        if max_tokens else min(p.get("max_tokens", DEFAULT_MAX_TOKENS) for p in picked)
    payload = {"model": model or picked[0].get("model", ""),
               "prompt": [p["prompt"] for p in picked], "max_tokens": want,
               "temperature": 0.0, "top_p": 1.0, "seed": 0, "stream": False}
    if logprobs:
        payload["logprobs"] = logprobs
    resp = _post(port, "/v1/completions", payload)
    choices = sorted(resp["choices"], key=lambda c: c.get("index", 0))
    if len(choices) != rows:
        raise SystemExit(f"a batched capture of {rows} rows came back with "
                         f"{len(choices)} completions")
    items = [item(f"{p.get('id', f'p{i % len(prompts)}')}#{i}", c)
             for i, (p, c) in enumerate(zip(picked, choices))]
    return items, want


def capture(a):
    prompts = json.load(open(a.suite))["prompts"]
    model = json.loads(_get(a.port, "/v1/models"))["data"][0]["id"]
    logprobs = None if a.no_logprobs else 5
    if a.batch:
        items, max_tokens = batched(a.port, model, prompts, a.max_tokens, logprobs, a.batch)
        suite = f"{a.name}@batch{a.batch}" + ("" if logprobs else "-nologprobs")
    else:
        items, max_tokens = serial(a.port, model, prompts, a.max_tokens, logprobs)
        suite = a.name
    cap = {"target": os.environ.get("FORGE_TARGET", ""), "config": os.environ.get("FORGE_CONFIG", ""),
           "suite": suite, "model": model, "commit": os.environ.get("BUILDKITE_COMMIT", ""),
           "branch": os.environ.get("BUILDKITE_BRANCH", ""),
           "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "max_tokens": max_tokens, "items": items, "env_overrides": {}}
    with open(a.out, "w") as fh:
        json.dump(cap, fh, indent=2)
    # The path is the file's name: forge downloads it beside this output and
    # reads it from there.
    print(json.dumps({"saved": os.path.basename(a.out), "digest": digest(items),
                      "prompts": len(items), "commit": cap["commit"], "branch": cap["branch"]},
                     indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="capture.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("counters")
    c.add_argument("--port", type=int, default=8000)
    f = sub.add_parser("capture")
    f.add_argument("--port", type=int, default=8000)
    f.add_argument("--suite", required=True, help="the prompt suite's JSON file")
    f.add_argument("--name", default="default", help="the suite's name, as perfkit records it")
    f.add_argument("--out", required=True)
    f.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    f.add_argument("--batch", type=int, default=0)
    f.add_argument("--no-logprobs", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "counters":
        counters(a.port)
    else:
        capture(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
