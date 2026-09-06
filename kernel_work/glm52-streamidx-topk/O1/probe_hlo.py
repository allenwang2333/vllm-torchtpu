"""Does `lax.empty` actually skip the write, and does `lax.cond` add a copy?

  python3 probe_hlo.py --rows A-D16-8k,B-D1-1M

Static compile-time analysis, NOT a measurement. Nothing this prints is a
benchmark result and none of it belongs in an experiment log. It exists because
two of O1's three changes rest on assumptions about what XLA does with them,
and both assumptions are cheap to check and expensive to be wrong about:

  1. `elide_score_prefill` replaces `jnp.full(-inf)` with `lax.empty`, whose
     docstring says outright that the initialization is "backend dependent".
     On the CPU backend it lowers to `broadcast(0)` -- a full write of the
     buffer, which would make the change a no-op that merely swaps -inf for 0.
     The change is only worth anything if TPU lowers it to an allocation with
     no producer. So: count the ops that WRITE something the size of the score
     buffer, in each arm, and compare.

  2. `skip_empty_launches` wraps a `pallas_call` that carries
     `input_output_aliases` in a `lax.cond`. A conditional whose branches
     disagree about aliasing is exactly the situation where XLA inserts a
     defensive buffer copy -- and at B-P1024-256k the score buffer is 4 GiB, so
     a copy would swamp every saving in this variant. So: count `copy`s of that
     size too.

Read this before believing a `fill` or `cond` timing delta, and read it again
if one of them comes out at zero.
"""
import argparse
import os
import re
import sys

import jax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, "instruments"))

import shapes  # noqa: E402  shared, read-only

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    streamindex_topk as shipping_streamindex_topk  # noqa: E402

import api  # noqa: E402
from run_bench import ARMS  # noqa: E402  one definition of the arm list

# An HLO line's opcode, and the byte size of its result shape.
_SHAPE = re.compile(r"=\s*(\w+)\[([\d,]*)\]")


def _result_bytes(line):
    m = _SHAPE.search(line)
    if not m:
        return 0
    dtype, dims = m.group(1), m.group(2)
    width = {"pred": 1, "s8": 1, "u8": 1, "bf16": 2, "f16": 2, "s16": 2,
             "f32": 4, "s32": 4, "u32": 4, "f64": 8, "s64": 8}.get(dtype)
    if width is None or not dims:
        return 0
    n = 1
    for d in dims.split(","):
        n *= int(d)
    return n * width


def summarize(hlo, score_bytes):
    """Ops whose RESULT is about the size of the score buffer, by opcode.

    "About" is >= half: the buffer may be split across the three launches or
    padded, and an exact match would miss a copy of a slightly different shape
    while still catching nothing useful.
    """
    big = {}
    custom_calls = {}
    for line in hlo.splitlines():
        line = line.strip()
        if " = " not in line:
            continue
        opcode_m = re.search(r"=\s*\S+\s+(\w[\w-]*)\(", line)
        opcode = opcode_m.group(1) if opcode_m else "?"
        if opcode == "custom-call":
            name = re.search(r'custom_call_target="([^"]+)"', line)
            key = name.group(1) if name else "custom-call"
            custom_calls[key] = custom_calls.get(key, 0) + 1
        nbytes = _result_bytes(line)
        if nbytes >= score_bytes // 2:
            big.setdefault(opcode, []).append(nbytes)
    return big, custom_calls


def run_row(name):
    print(f"\n{'=' * 74}\n{name}\n{'=' * 74}", flush=True)
    args, kwargs, meta = shapes.build(name)
    score_bytes = meta["score_bytes"]
    print(f"  score buffer {score_bytes / 2**20:.0f} MiB "
          f"(T={meta['T']} n={meta['n']})", flush=True)

    for nm, sw in ARMS:
        if sw is None:
            fn = jax.jit(lambda *x: shipping_streamindex_topk(*x, **kwargs))
        else:
            fn = jax.jit(lambda *x, _s=sw: api.streamindex_topk(
                *x, **kwargs, **_s))
        hlo = fn.lower(*args).compile().as_text()
        big, ccs = summarize(hlo, score_bytes)
        tot = sum(sum(v) for v in big.values())
        parts = ", ".join(
            f"{op} x{len(v)} ({sum(v) / 2**20:.0f}MiB)"
            for op, v in sorted(big.items(), key=lambda kv: -sum(kv[1])))
        print(f"  {nm:9} big-result ops: {tot / 2**20:8.0f} MiB total   "
              f"{parts or '(none)'}", flush=True)
        alloc = sum(n for k, n in ccs.items() if "AllocateBuffer" in k)
        print(f"  {'':9}   custom calls: "
              f"{dict(sorted(ccs.items()))}   AllocateBuffer={alloc}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="A-D16-8k,B-D1-1M")
    a = ap.parse_args()
    for name in [r.strip() for r in a.rows.split(",") if r.strip()]:
        run_row(name)
    print("\nReminder: compile-time analysis. Not a result, not loggable.")


if __name__ == "__main__":
    main()
