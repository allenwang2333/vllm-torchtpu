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
"""Confirm the variant's MXU operand types FROM THE LOWERING, not the source.

Step 2's P1 established that the incumbent's `compute_scores` einsum reaches
the MXU as native fp8 (`f8E4M3FN x f8E4M3FN -> f32`, zero `bf16` matmul ops),
and that any bf16-operand formulation is a 1.86x regression. Variant A changes
where the scale byte comes from and deletes a transpose on the scale path, both
of which sit next to the operand that feeds the matmul. If the split caused the
operands to arrive as bf16, A would be dead on arrival however few bytes it
read -- and the JAX-level dtypes would still say fp8, because the upcast would
be inserted below JAX.

So this reads the answer out of the compiler, at two independent levels, the
same two P1 used:

  mosaic  `*post-infer-vector-layout.txt` -- the `tpu.matmul` operand types
  jf      `*-71-final_bundles.txt`        -- the `vmatmul` opcode mnemonics

and prints them to stdout, so the answer is in the job log and no
`sync-outputs` of a 1 GiB dump is needed. `*-06-post-rematerialize-
allocations.txt` is summarised too, because the variant adds a VMEM scratch
buffer for the scale plane and its size should be read rather than assumed.

EVERYTHING HERE IS STATIC COMPILE-TIME ANALYSIS. The wall-clock printed at the
end proves the kernel ran; it is LLO-instrumented and host-contended and is not
a performance number.

Usage:
    python3 .../llo_narrow.py --regime mixed_prod --arm variant \
        --out $CDK_OUTPUT_DIR/mixed_prod_variant
"""

import argparse
import os
import re
import sys

# --- must precede `import jax` -------------------------------------------
_ap = argparse.ArgumentParser()
_ap.add_argument("--regime", required=True)
_ap.add_argument("--arm", required=True, choices=["variant", "incumbent"])
_ap.add_argument("--out", required=True)
_args = _ap.parse_args()

_JF = os.path.join(_args.out, "jf")
_MOSAIC = os.path.join(_args.out, "mosaic")
os.makedirs(_JF, exist_ok=True)
os.makedirs(_MOSAIC, exist_ok=True)

os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={_JF} "
    f"--xla_jf_dump_llo_text=true "
    f"--xla_mosaic_dump_to={_MOSAIC}")
print(f"LIBTPU_INIT_ARGS={os.environ['LIBTPU_INIT_ARGS']}", flush=True)
# -------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _load  # noqa: E402,F401  -- puts the repo root on sys.path

import time  # noqa: E402
from collections import Counter  # noqa: E402

import jax  # noqa: E402

from kernel_work.streamindex_topk.instruments import inputs as I  # noqa: E402
from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402

# The Mosaic dump files are named `...-mosaic-dump-anonymous-<stage>.txt` --
# the program name is NOT in the filename, it is in the `func.func @"..."`
# inside. The jf dump files DO carry it. Both are handled accordingly; a
# filename filter on the mosaic side would silently match nothing, which is the
# failure mode where an absent answer reads like a negative one.
MATMUL_RE = re.compile(r"tpu\.matmul\b.*")
VMATMUL_RE = re.compile(r"\b(vmatmul[\w.]*)")
FUNC_RE = re.compile(r'func\.func @"([^"]+)"')
HBM_RE = re.compile(r"memref<([^,]+),.{0,200}?memory_space<hbm>>")
ALLOC_RE = re.compile(
    r"shape = '([^']*)'[^]]*?space=(\w+), size = (0x[0-9a-fA-F]+|\d+)")


def report_mosaic(root):
    print(f"\n--- mosaic: tpu.matmul operand types, under {root} ---",
          flush=True)
    hits = 0
    for dp, _, fs in os.walk(root):
        for f in sorted(fs):
            if "post-infer-vector-layout" not in f:
                continue
            path = os.path.join(dp, f)
            seen = Counter()
            funcs = []
            hbm = []
            with open(path, errors="replace") as fh:
                for line in fh:
                    fm = FUNC_RE.search(line)
                    if fm:
                        funcs.append(fm.group(1))
                        hbm = HBM_RE.findall(line)
                    m = MATMUL_RE.search(line)
                    if m:
                        sig = m.group(0)
                        sig = sig[sig.rfind("} :") + 2:] if "} :" in sig else sig
                        seen[" ".join(sig.split())] += 1
            if not seen:
                continue
            hits += 1
            print(f"[{f}] func={funcs}", flush=True)
            for a in hbm:
                print(f"    hbm operand: memref<{a}...>", flush=True)
            for sig, n in seen.items():
                print(f"    x{n}  tpu.matmul {sig}", flush=True)
    if not hits:
        print("    NO post-infer-vector-layout FILE WITH A tpu.matmul FOUND "
              "-- the operand types were NOT confirmed. Do not claim fp8.",
              flush=True)
    return hits


def report_bundles(root):
    print(f"\n--- jf: vmatmul opcodes in final bundles, under {root} ---",
          flush=True)
    hits = 0
    for dp, _, fs in os.walk(root):
        for f in sorted(fs):
            if not f.endswith("-final_bundles.txt") or "StreamIdx" not in f:
                continue
            path = os.path.join(dp, f)
            ops = Counter()
            with open(path, errors="replace") as fh:
                for line in fh:
                    for m in VMATMUL_RE.finditer(line):
                        ops[m.group(1)] += 1
            hits += 1
            total = sum(ops.values())
            n_bf16 = sum(v for k, v in ops.items() if "bf16" in k)
            n_f8 = sum(v for k, v in ops.items() if "f8" in k)
            n_f32 = sum(v for k, v in ops.items()
                        if "f32" in k and "f8" not in k)
            print(f"[{f}] vmatmul total={total} bf16={n_bf16} f8={n_f8} "
                  f"f32={n_f32}", flush=True)
            for k, v in sorted(ops.items(), key=lambda kv: -kv[1]):
                print(f"    {v:>6}  {k}", flush=True)
    if not hits:
        print("    NO StreamIdx final_bundles FILE FOUND.", flush=True)
    return hits


def report_allocs(root):
    print(f"\n--- jf: VMEM allocations, under {root} ---", flush=True)
    for dp, _, fs in os.walk(root):
        for f in sorted(fs):
            if "post-rematerialize-allocations" not in f:
                continue
            if "StreamIdx" not in f:
                continue
            path = os.path.join(dp, f)
            by_space = Counter()
            counts = Counter()
            lines = []
            with open(path, errors="replace") as fh:
                for line in fh:
                    m = ALLOC_RE.search(line)
                    if not m:
                        continue
                    shape, space, size = m.group(1), m.group(2), m.group(3)
                    nbytes = int(size, 0)
                    by_space[space] += nbytes
                    counts[space] += 1
                    if space == "vmem":
                        lines.append(f"{shape}  {nbytes} B "
                                     f"({nbytes/2**20:.3f} MiB)")
            if not by_space:
                continue
            print(f"[{f}]", flush=True)
            for sp, tot in sorted(by_space.items()):
                print(f"    {sp}: {counts[sp]} allocs, {tot} B "
                      f"({tot/2**20:.3f} MiB)", flush=True)
            for ln in lines:
                print(f"      vmem {ln}", flush=True)


def main():
    print("jax", jax.__version__, "devices", jax.devices(), flush=True)
    regime = S.REGIMES_BY_NAME[_args.regime]
    print(f"arm={_args.arm} regime={regime.name} phase={regime.phase} "
          f"kv_len={regime.kv_len} num_tokens={regime.num_tokens}", flush=True)

    if _args.arm == "variant":
        impl = _load.load("streamindex_topk_narrow")
        narrow_inputs = _load.load("narrow_inputs")
        d = narrow_inputs.build(regime)
        cv, cs = d["cache_kv"]
        print(f"cache_val {cv.shape} {cv.dtype}; cache_scale {cs.shape} "
              f"{cs.dtype}", flush=True)
    else:
        from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as impl
        d = I.build(regime)
        print(f"cache_kv {d['cache_kv'].shape} {d['cache_kv'].dtype}",
              flush=True)
    print(f"impl module: {impl.__file__}", flush=True)

    call = jax.jit(lambda q, w, c, s, p, cu, dist: impl.streamindex_topk(
        q, w, c, s, p, cu, dist,
        k=S.INDEX_TOPK,
        compression_ratio=S.COMPRESSION_RATIO,
        num_kv_pages_per_block=S.NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=S.NUM_QUERIES_PER_BLOCK,
        decode_req_batch_size=S.DECODE_REQ_BATCH_SIZE,
        enable_early_exit=False,
    ))
    args = (d["q"], d["indexer_weights"], d["cache_kv"], d["seq_lens"],
            d["page_indices"], d["cu_q_lens"], d["distribution"])

    t0 = time.perf_counter()
    out = jax.block_until_ready(call(*args))
    print(f"out shape={out.shape} dtype={out.dtype}", flush=True)
    print(f"[not a perf number] compile+first "
          f"{(time.perf_counter()-t0)*1e3:.1f} ms", flush=True)

    for label, root in (("jf", _JF), ("mosaic", _MOSAIC)):
        n = sum(len(f) for _, _, f in os.walk(root))
        b = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(root) for f in fs)
        print(f"dump {label}: {n} files, {b/2**20:.1f} MiB at {root}",
              flush=True)

    report_mosaic(_MOSAIC)
    report_bundles(_JF)
    report_allocs(_JF)
    return 0


if __name__ == "__main__":
    sys.exit(main())
