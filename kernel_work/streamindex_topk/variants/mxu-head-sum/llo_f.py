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
"""Variant F -- what operand types does the head-sum matmul actually reach the
MXU with?

This is the question F lives or dies on and it CANNOT be answered from source.
The JAX-level dtype is what the kernel asks for; Mosaic decides what the
hardware gets, and P1 [measured (j-689b87cd-7460-4b90-bd87)] exists precisely
because those two disagreed as a live possibility on the scoring einsum. Its
answer there:

    tpu.matmul : vector<4096x128xf8E4M3FN>, vector<2048x128xf8E4M3FN>,
                 vector<4096x2048xf32> -> vector<4096x2048xf32>

    StreamIdxTC-m-bq_128-bkvp_2.1 : 1024 vmatmul, 0 bf16, 1024 f8, 0 f32
    StreamIdxTC-d-bq_1-bkvp_2.2   :   32 vmatmul, 0 bf16,   32 f8, 0 f32
    StreamIdxTC-d-bq_1-bkvp_2.3   :    8 vmatmul, 0 bf16,    8 f8, 0 f32

so any bf16 `vmatmul` this file finds in the variant is a bf16 matmul F
introduced, and `measurement_and_design.md` Sec 2.5 prices a bf16-operand move on
this path as a 1.86x regression.

EVERYTHING THIS FILE PRODUCES IS STATIC COMPILE-TIME ANALYSIS. No number here
is a timing. The wall-clock is printed only as proof the kernel executed; a
fanned-out capture's wall-clock is contended and `xprof-llo/SKILL.md` is
explicit that it is not a performance number. It is not quoted and must not be.

The `xprof-cli` readers `get_llo_fit_summary` / `get_custom_call_mlir` are
unavailable -- their pinned repo 404s under an authenticated `gh`, and
`setup.sh` exits 0 on that failure. `probes/llo_capture_RESULT.md` records the
substitution this file reuses: read the raw dumps, which is what the readers
wrap.

Dumps go to /tmp, NOT to $CDK_OUTPUT_DIR: one capture is ~1 GiB and there is
nothing in it worth syncing back once the counts below are printed.
"""

import argparse
import os
import re
import sys

# --- must precede `import jax` -------------------------------------------
_ap = argparse.ArgumentParser()
_ap.add_argument("--impl", required=True, choices=["incumbent", "variant"])
_ap.add_argument("--regime", default="mixed_prod")
_ap.add_argument("--dtype", default="bf16")
_ap.add_argument("--min-bq", default="0")
_ap.add_argument("--dump-root", default="/tmp/llo_f")
_args = _ap.parse_args()

_OUT = os.path.join(_args.dump_root, f"{_args.impl}-{_args.regime}")
_JF = os.path.join(_OUT, "jf")
_MOSAIC = os.path.join(_OUT, "mosaic")
os.makedirs(_JF, exist_ok=True)
os.makedirs(_MOSAIC, exist_ok=True)

os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={_JF} "
    f"--xla_jf_dump_llo_text=true "
    f"--xla_mosaic_dump_to={_MOSAIC}")
os.environ["F_MXU_HEAD_SUM"] = "1"
os.environ["F_DTYPE"] = _args.dtype
os.environ["F_MIN_BQ"] = str(_args.min_bq)
print(f"LIBTPU_INIT_ARGS={os.environ['LIBTPU_INIT_ARGS']}", flush=True)
# -------------------------------------------------------------------------

sys.path.insert(0, __file__.rsplit("/kernel_work/", 1)[0])

import time  # noqa: E402

import jax  # noqa: E402

from kernel_work.streamindex_topk.instruments import inputs as I  # noqa: E402
from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402

VMATMUL = re.compile(r"\bvmatmul[\w.]*")
MATPUSH = re.compile(r"\bvmatpush[\w.]*")


def _walk(root, suffix):
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(suffix):
                yield os.path.join(dp, f)


def _prog(path):
    """Best-effort program name from a dump filename."""
    b = os.path.basename(path)
    m = re.search(r"(StreamIdxTC[\w.\-]*?)[-.]\d\d-", b)
    if m:
        return m.group(1)
    m = re.search(r"(StreamIdxTC[\w.\-]*)", b)
    return m.group(1) if m else b


def report_mosaic():
    print("\n### Mosaic: every `tpu.matmul` at post-infer-vector-layout "
          "(types only)")
    seen = set()
    for path in sorted(_walk(_MOSAIC, "post-infer-vector-layout.txt")):
        if "StreamIdxTC" not in path:
            continue
        with open(path, errors="replace") as fh:
            for line in fh:
                if "tpu.matmul" not in line:
                    continue
                t = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
                key = (_prog(path), t)
                if key in seen:
                    continue
                seen.add(key)
                print(f"  [{_prog(path)}] {t}")
    if not seen:
        print("  (no StreamIdxTC post-infer-vector-layout dump found)")

    print("\n### Mosaic: every non-matmul bf16/f8 vector type mentioned "
          "(so a cast can be told from a matmul)")
    kinds = {}
    for path in sorted(_walk(_MOSAIC, "post-infer-vector-layout.txt")):
        if "StreamIdxTC" not in path:
            continue
        with open(path, errors="replace") as fh:
            for line in fh:
                if "tpu.matmul" in line:
                    continue
                for m in re.finditer(r"vector<[\dx]*x(bf16|f8E4M3FN|f8E5M2)>",
                                     line):
                    kinds.setdefault((_prog(path), m.group(0)), 0)
                    kinds[(_prog(path), m.group(0))] += 1
    for (p, t), n in sorted(kinds.items()):
        print(f"  [{p}] {t}  x{n} lines")
    if not kinds:
        print("  (none)")


def report_bundles():
    print("\n### LLO: `vmatmul` opcode counts, per program "
          "(from *-71-final_bundles.txt)")
    print("Raw counts. The denominator is the program's own total, named "
          "inline; no percentage is printed for a number nobody can "
          "re-derive.")
    for path in sorted(_walk(_JF, "-71-final_bundles.txt")):
        if "StreamIdxTC" not in path:
            continue
        ops, pushes = {}, {}
        with open(path, errors="replace") as fh:
            for line in fh:
                for m in VMATMUL.finditer(line):
                    ops[m.group(0)] = ops.get(m.group(0), 0) + 1
                for m in MATPUSH.finditer(line):
                    pushes[m.group(0)] = pushes.get(m.group(0), 0) + 1
        if not ops and not pushes:
            continue
        total = sum(ops.values())
        nbf16 = sum(v for k, v in ops.items() if "bf16" in k)
        nf8 = sum(v for k, v in ops.items() if "f8" in k)
        nf32 = sum(v for k, v in ops.items() if "f32" in k)
        print(f"\n  {_prog(path)}   vmatmul total {total}  "
              f"| bf16 {nbf16} | f8 {nf8} | f32 {nf32} "
              f"| other {total - nbf16 - nf8 - nf32}")
        for k, v in sorted(ops.items(), key=lambda kv: -kv[1]):
            print(f"      {v:6d}  {k}")
        for k, v in sorted(pushes.items(), key=lambda kv: -kv[1]):
            print(f"      {v:6d}  {k}   (weight push)")


def report_vmem():
    print("\n### LLO: VMEM allocations "
          "(from *-06-post-rematerialize-allocations.txt)")
    for path in sorted(_walk(_JF, "-06-post-rematerialize-allocations.txt")):
        if "StreamIdxTC" not in path:
            continue
        tot = 0
        lines = []
        with open(path, errors="replace") as fh:
            for line in fh:
                if "space=vmem" not in line:
                    continue
                m = re.search(r"size=(\d+)", line)
                if m:
                    tot += int(m.group(1))
                lines.append(line.rstrip()[:160])
        print(f"\n  {_prog(path)}   vmem total {tot / 2**20:.3f} MiB over "
              f"{len(lines)} allocations")
        for ln in lines[:24]:
            print(f"      {ln}")
        if len(lines) > 24:
            print(f"      ... {len(lines) - 24} more")


def main():
    print("jax", jax.__version__, "devices", jax.devices(), flush=True)
    regime = S.REGIMES_BY_NAME[_args.regime]

    if _args.impl == "incumbent":
        from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as impl
    else:
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            "streamindex_topk_mxu",
            os.path.join(here, "streamindex_topk_mxu.py"))
        impl = importlib.util.module_from_spec(spec)
        sys.modules["streamindex_topk_mxu"] = impl
        spec.loader.exec_module(impl)
        print(f"variant knobs: F_MXU_HEAD_SUM={impl.F_MXU_HEAD_SUM} "
              f"F_DTYPE={impl._F_DTYPE_NAME} -> {impl.F_DTYPE.__name__} "
              f"precision={impl.F_PRECISION} F_MIN_BQ={impl.F_MIN_BQ}",
              flush=True)

    print(f"impl={_args.impl} regime={regime.name} phase={regime.phase} "
          f"kv_len={regime.kv_len} num_tokens={regime.num_tokens} "
          f"num_bq={regime.num_bq} num_bkv={regime.num_bkv} "
          f"bkv_p={S.NUM_KV_PAGES_PER_BLOCK}", flush=True)

    d = I.build(regime)
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
    out = call(*args)
    jax.block_until_ready(out)
    # NOT A PERFORMANCE NUMBER -- LLO-instrumented and host-contended.
    print(f"[not a perf number] compile+first "
          f"{(time.perf_counter() - t0) * 1e3:.1f} ms, out {out.shape} "
          f"{out.dtype}", flush=True)

    for label, root in (("jf", _JF), ("mosaic", _MOSAIC)):
        n = sum(len(f) for _, _, f in os.walk(root))
        b = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(root) for f in fs)
        print(f"dump {label}: {n} files, {b / 2**20:.1f} MiB at {root}",
              flush=True)

    print(f"\n================ STATIC ANALYSIS, impl={_args.impl}, "
          f"regime={regime.name} ================")
    report_mosaic()
    report_bundles()
    report_vmem()
    return 0


if __name__ == "__main__":
    sys.exit(main())
