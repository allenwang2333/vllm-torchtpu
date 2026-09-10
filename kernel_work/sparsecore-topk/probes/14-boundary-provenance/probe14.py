#!/usr/bin/env python3
"""probe 14 -- boundary provenance. Compile only. No timing anywhere.

Q1  Is production's `%bitcast_convert_type.3` (f32[1024,80,128]{2,1,0:T(8,128)}
    -> s32, SAME shape, SAME layout) materialised, or aliased away?
    Route: XLA's own buffer assignment. Two HloValues in ONE allocation with
    the same offset == aliased; two allocations (or two offsets) == two live
    40 MiB buffers == materialised.

Q2  Does the 10.007 us output retiling
    `ROOT reshape s32[1024,2048]{1,0:T(8,128)} <- s32[2097152]{0:T(1024)}`
    survive in-model, or is it an artifact of compiling `streamindex_topk`
    standalone?
    Route: compile the ENCLOSING entry. `mla_attention_op.py:72`
    `_streamidx_topk_jax` is the outermost JAX jit -- above it is torch
    (`pallas.jax_op`, mla_attention_op.py:126 -> pallas.py:804 jax.jit(...,
    donate_argnums=(2,), keep_unused=True) -> pallas.py:514 jax.export.export).
    So compiling it is not "a bigger scope", it IS production's entry.

Everything reported here is `.lower().compile()` on ShapeDtypeStructs and a
read of the resulting optimized HLO / buffer assignment. Nothing executes.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MIB = 1024.0 * 1024.0


def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


# ---------------------------------------------------------------------------
# The shipping geometry. Copied VERBATIM from probes/11 probe.py:347-376 so
# that this job's inner-entry module is comparable to probe 11's; the copy is
# md5-gated below against probe 11's own dumped HLO line.
# ---------------------------------------------------------------------------
PROD = {
    "num_tokens": 1024, "max_num_seqs": 64, "max_model_len": 9216,
    "page_size": 1024, "kv_packing": 4, "lkv_dim": 256,
    "num_index_heads": 64, "head_dim": 128, "k": 2048,
    "compression_ratio": 1, "num_kv_pages_per_block": (2, 2, 2),
    "num_queries_per_block": (1, 128, 128),
}
B, N, K = 1024, 10240, 2048
KERNEL = f"sc_topk_direct_b{B}_n{N}_k{K}"


def _derived():
    p = PROD
    pps = -(-p["max_model_len"] // p["page_size"])
    return {
        "pages_per_seq": pps,
        "num_pages": p["max_num_seqs"] * pps,
        "num_sublanes_total": max((-(-pps // bp) * bp) * p["page_size"] // 128
                                  for bp in p["num_kv_pages_per_block"]),
    }


def _inner_avals(jnp):
    """probe 11's `_prod_avals` -- the INNER entry `streamindex_topk`."""
    p, d = PROD, _derived()
    nt, hi, hd, ms = (p["num_tokens"], p["num_index_heads"], p["head_dim"],
                      p["max_num_seqs"])
    npages = d["num_pages"]
    import jax
    sds = jax.ShapeDtypeStruct
    return (
        sds((nt, hi, hd), jnp.float8_e4m3fn),
        sds((nt, hi), jnp.float32),
        sds((npages, p["page_size"] // p["kv_packing"], p["kv_packing"],
             p["lkv_dim"]), jnp.uint8),
        sds((ms, ), jnp.int32),
        sds((npages, ), jnp.int32),
        sds((ms + 1, ), jnp.int32),
        sds((3, ), jnp.int32),
    )


def _outer_avals(jnp):
    """The ENCLOSING entry `_streamidx_topk_jax` (mla_attention_op.py:72-83).

    Eight arrays, no static args (`k`/blocks/`enable_early_exit` are closed
    over at mla_attention_op.py:70,115-121). `q_bytes` is the uint8 view of
    the fp8 q (forward_oot :186, bitcast back at :106); `k_packed` is
    uint8[num_tokens, head_dim+1] (:165, :77-78).
    """
    import jax
    p, d = PROD, _derived()
    nt, hi, hd, ms = (p["num_tokens"], p["num_index_heads"], p["head_dim"],
                      p["max_num_seqs"])
    npages = d["num_pages"]
    sds = jax.ShapeDtypeStruct
    return (
        sds((nt, hi, hd), jnp.uint8),                      # q_bytes
        sds((nt, hi), jnp.bfloat16),                       # weights
        sds((npages, p["page_size"] // p["kv_packing"], p["kv_packing"],
             p["lkv_dim"]), jnp.uint8),                    # cache_kv (donated)
        sds((nt, hd + 1), jnp.uint8),                      # k_packed
        sds((ms, ), jnp.int32),                            # seq_lens
        sds((npages, ), jnp.int32),                        # page_indices
        sds((ms + 1, ), jnp.int32),                        # cu_q_lens
        sds((3, ), jnp.int32),                             # distribution
    )


# ---------------------------------------------------------------------------
# HLO reading. Operands are resolved BY NAME (hlo_tools). An HLO line prints
# its own output shape and names its operands, so grepping a line for an
# operand's shape can never match.
# ---------------------------------------------------------------------------
_BC = re.compile(r'backend_config=(\{.*\})\s*$')


def backend_config(line: str):
    m = _BC.search(line.rstrip().rstrip(","))
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def users_of(h, name):
    return [i for i in h.order if name in i["operands"] and i["name"] != name]


def report_bitcast_converts(h, tag, out):
    print(f"\n  --- [{tag}] every bitcast-convert in the module ---",
          flush=True)
    rows = []
    for i in h.order:
        if i["opcode"] != "bitcast-convert":
            continue
        src = i["operands"][0] if i["operands"] else None
        src_shape = h.shape_of(src, i["comp"]) if src else "<none>"
        cfg = backend_config(i["line"]) or {}
        w = cfg.get("window_config") or {}
        used = cfg.get("used_scoped_memory_configs") or []
        row = {
            "name": i["name"], "comp": i["comp"], "out": i["shape"],
            "src": src, "src_shape": src_shape,
            "MiB": round(i["bytes"] / MIB, 3),
            "same_shape_and_layout": (i["shape"].split("[", 1)[1:] and
                                      _strip_dtype(i["shape"]) ==
                                      _strip_dtype(src_shape)),
            "estimated_cycles": w.get("estimated_cycles"),
            "iteration_bounds": w.get("iteration_bounds"),
            "output_window_bounds": w.get("output_window_bounds"),
            "buffering_level": w.get("buffering_level"),
            "used_scoped_bytes": [u.get("size") for u in used],
            "aliasing_operands": cfg.get("aliasing_operands"),
            "n_users": len(users_of(h, i["name"])),
            "users": [u["name"] for u in users_of(h, i["name"])][:6],
            "src_n_users": len(users_of(h, src)) if src else None,
        }
        rows.append(row)
        print(f"    %{row['name']}  in %{row['comp']}  {row['MiB']:.3f} MiB",
              flush=True)
        print(f"      out {row['out']}", flush=True)
        print(f"      src %{row['src']} = {row['src_shape']}", flush=True)
        print(f"      SAME shape+layout (dtype aside): "
              f"{row['same_shape_and_layout']}", flush=True)
        print(f"      window: cycles={row['estimated_cycles']} "
              f"iter={row['iteration_bounds']} "
              f"win={row['output_window_bounds']} "
              f"buf={row['buffering_level']} "
              f"used_scoped={row['used_scoped_bytes']}", flush=True)
        print(f"      aliasing_operands={row['aliasing_operands']}  "
              f"users={row['users']}  src_users={row['src_n_users']}",
              flush=True)
    if not rows:
        print("    (none)", flush=True)
    out[f"{tag}_bitcast_converts"] = rows
    return rows


def _strip_dtype(shape: str) -> str:
    return re.sub(r"^[a-z][a-z0-9]*", "", shape.strip())


def report_topk_output(h, tag, out):
    """Q2: what consumes the kernel's flat s32[b*k] output, down to ROOT."""
    print(f"\n  --- [{tag}] kernel output chain, walked by NAME ---",
          flush=True)
    seeds = [i for i in h.order
             if KERNEL in i["name"] and i["comp"] == _entry_name(h)]
    if not seeds:
        seeds = [i for i in h.order if KERNEL in i["name"]]
    chain, rec = [], []
    seen = set()
    cur = [s for s in seeds if "call-done" in s["name"]] or seeds[-1:]
    for s in seeds:
        print(f"    seed %{s['name']}  {s['opcode']}  {s['shape'][:110]}",
              flush=True)
    frontier = [c["name"] for c in cur]
    for _ in range(12):
        nxt = []
        for nm in frontier:
            for u in users_of(h, nm):
                if u["name"] in seen:
                    continue
                seen.add(u["name"])
                src = u["operands"][0] if u["operands"] else None
                e = {"name": u["name"], "opcode": u["opcode"],
                     "shape": u["shape"], "root": u["root"], "comp": u["comp"],
                     "MiB": round(u["bytes"] / MIB, 3),
                     "src": src,
                     "src_shape": h.shape_of(src, u["comp"]) if src else ""}
                rec.append(e)
                print(f"      user of %{nm}: "
                      f"{'ROOT ' if u['root'] else ''}%{u['name']} "
                      f"{u['opcode']:14s} {u['shape'][:90]}", flush=True)
                if u["opcode"] in ("reshape", "copy", "transpose", "bitcast"):
                    print(f"          <- %{src} = {e['src_shape'][:90]}",
                          flush=True)
                nxt.append(u["name"])
        frontier = nxt
        if not frontier:
            break
    retile = [e for e in rec
              if e["opcode"] in ("reshape", "copy", "transpose")
              and e["MiB"] >= 1.0]
    out[f"{tag}_topk_out_chain"] = rec
    out[f"{tag}_topk_out_retiling"] = retile
    print(f"    >>> retiling ops (reshape/copy/transpose >=1 MiB) on the "
          f"kernel output: {[e['name'] for e in retile] or 'NONE'}",
          flush=True)
    return retile


def _entry_name(h):
    for c in h.comps.values():
        if c["is_entry"]:
            return c["name"]
    return "<toplevel>"


def report_entry(h, tag, out):
    for c in h.comps.values():
        if c["is_entry"]:
            print(f"\n  [{tag}] ENTRY {c['line'][:400]}", flush=True)
            out[f"{tag}_entry_line"] = c["line"]


# ---------------------------------------------------------------------------
# Buffer assignment
# ---------------------------------------------------------------------------
_ALLOC = re.compile(r"^allocation (\d+):(.*)$")


def parse_buffer_assignment(path, names):
    """Return, per requested instruction name, the allocations that hold it."""
    txt = open(path, errors="replace").read()
    lines = txt.splitlines()
    cur = None
    hits = {n: [] for n in names}
    allocs = {}
    for ln in lines:
        m = _ALLOC.match(ln.strip())
        if m:
            cur = int(m.group(1))
            allocs[cur] = {"header": ln.strip(), "values": []}
            continue
        if cur is None:
            continue
        s = ln.strip()
        if s.startswith("value:"):
            allocs[cur]["values"].append(s)
            for n in names:
                # `value: <7 name @0> (size=..,offset=..): shape`
                if re.search(r"<\d+\s+" + re.escape(n) + r"\s", s):
                    off = re.search(r"offset=(\d+)", s)
                    sz = re.search(r"size=(\d+)", s)
                    hits[n].append({
                        "allocation": cur, "header": allocs[cur]["header"],
                        "value_line": s,
                        "offset": int(off.group(1)) if off else None,
                        "size": int(sz.group(1)) if sz else None,
                    })
    return hits, allocs


# ---------------------------------------------------------------------------
def compile_and_report(label, fn, avals, kwargs, results, outdir):
    import jax
    print("\n" + "=" * 74, flush=True)
    print(f"COMPILE  {label}", flush=True)
    print("=" * 74, flush=True)
    rec = {"label": label}
    results["modules"][label] = rec
    t0 = time.perf_counter()
    try:
        compiled = fn.lower(*avals, **kwargs).compile()
    except Exception as exc:
        traceback.print_exc()
        rec["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  !! COMPILE FAILED: {rec['error']}", file=sys.stderr,
              flush=True)
        return None, None
    rec["compile_s"] = round(time.perf_counter() - t0, 1)
    txt = compiled.as_text()
    with open(os.path.join(outdir, f"hlo-{label}.txt"), "w") as fh:
        fh.write(txt)
    rec["hlo_md5"] = md5(txt.encode())
    rec["hlo_lines"] = len(txt.splitlines())
    print(f"  compiled in {rec['compile_s']}s; optimized HLO "
          f"{rec['hlo_lines']} lines, md5 {rec['hlo_md5']}", flush=True)

    hit = KERNEL in txt
    rec["shipping_kernel_present"] = hit
    print(f"  SHAPE GATE: `{KERNEL}` present: {hit}", flush=True)
    if not hit:
        print(f"  !! wrong shape -- every statement from this module is "
              f"about a DIFFERENT kernel.", file=sys.stderr, flush=True)

    try:
        rec["memory_analysis"] = {
            k: getattr(compiled.memory_analysis(), k)
            for k in ("argument_size_in_bytes", "output_size_in_bytes",
                      "temp_size_in_bytes", "alias_size_in_bytes",
                      "generated_code_size_in_bytes",
                      "host_temp_size_in_bytes")
            if hasattr(compiled.memory_analysis(), k)
        }
        print(f"  memory_analysis: "
              f"{json.dumps(rec['memory_analysis'])}", flush=True)
        for k, v in rec["memory_analysis"].items():
            print(f"      {k:32s} {v:>14,d}  ({v / MIB:9.3f} MiB)", flush=True)
    except Exception as exc:
        rec["memory_analysis_error"] = repr(exc)
        print(f"  !! memory_analysis: {exc!r}", file=sys.stderr, flush=True)

    from hlo_tools import Hlo, summarize
    h = Hlo(txt)
    rec["summary"] = summarize(txt, KERNEL)
    print(f"  data-format-call lines: "
          f"{rec['summary']['data_format_line_count']}; "
          f"big copies: {rec['summary']['big_copy_count']} "
          f"({rec['summary']['big_copy_MiB']:.3f} MiB); "
          f"big fusions: {rec['summary']['big_fusion_count']} "
          f"({rec['summary']['big_fusion_MiB']:.3f} MiB)", flush=True)
    report_entry(h, label, rec)
    report_bitcast_converts(h, label, rec)
    report_topk_output(h, label, rec)
    print(f"\n  --- [{label}] every >=1 MiB non-bitcast instruction ---",
          flush=True)
    for i in rec["summary"]["inventory"]:
        print(f"      {i['MiB']:9.3f} MiB  {i['opcode']:16s} "
              f"%{i['name'][:50]:50s} {i['shape'][:70]}", flush=True)
    return compiled, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dumpdir", default="/tmp/xdump")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dump = args.dumpdir
    os.makedirs(dump, exist_ok=True)

    prev = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = (
        prev + f" --xla_dump_to={dump} --xla_dump_hlo_as_text"
        " --xla_dump_hlo_as_proto").strip()
    print(f"XLA_FLAGS={os.environ['XLA_FLAGS']}", flush=True)

    import jax
    import jax.numpy as jnp
    print(f"jax {jax.__version__}", flush=True)
    if not args.dry_run:
        print(f"  devices={jax.devices()}", flush=True)

    results = {"_outdir": args.outdir, "jax": jax.__version__,
                "PROD": {k: list(v) if isinstance(v, tuple) else v
                         for k, v in PROD.items()},
                "derived": _derived(), "modules": {}}

    import vllm_torchtpu.kernels.deepseek_v4.streamindex_topk as sit
    import vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk as sct
    import vllm_torchtpu.layers.vllm.custom_ops.mla_attention_op as mao
    for nm, mod in (("streamindex_topk", sit), ("sparsecore_topk", sct),
                     ("mla_attention_op", mao)):
        p = mod.__file__
        results.setdefault("subject_md5", {})[nm] = {
            "path": p, "md5": md5(open(p, "rb").read())}
        print(f"  subject {nm:20s} {p}  md5 "
              f"{results['subject_md5'][nm]['md5']}", flush=True)
    print(f"  STREAMIDX_ENABLE_EARLY_EXIT = "
          f"{mao.STREAMIDX_ENABLE_EARLY_EXIT}  (mla_attention_op.py:57)",
          flush=True)

    for nm, av in (("inner", _inner_avals(jnp)), ("outer", _outer_avals(jnp))):
        print(f"  {nm} avals:", flush=True)
        for a in av:
            print(f"      {str(a.dtype):18s} {tuple(a.shape)}", flush=True)
    assert _derived()["num_sublanes_total"] * 128 == N, _derived()

    if args.dry_run:
        print("DRY RUN OK", flush=True)
        return 0

    # ---------------------------------------------------------------- inner
    # probe 11's module, recompiled here so this job's numbers are anchored to
    # a module whose HLO probe 11 already dumped.
    static = dict(k=PROD["k"], compression_ratio=PROD["compression_ratio"],
                  num_kv_pages_per_block=PROD["num_kv_pages_per_block"],
                  num_queries_per_block=PROD["num_queries_per_block"],
                  enable_early_exit=False)
    compile_and_report("inner_streamindex_topk", sit.streamindex_topk,
                       _inner_avals(jnp), static, results, args.outdir)

    # ---------------------------------------------------------------- outer
    # The REAL production entry. Rebuilt to match pallas.py:804-809 exactly:
    #   jax.jit(fn, static_argnums=(), donate_argnums=(2,), keep_unused=True)
    topk = PROD["k"]

    def _streamidx_topk_jax(q_bytes, weights, cache_kv, k_packed, seq_lens,
                            page_indices, cu_q_lens, distribution):
        # verbatim from mla_attention_op.py:84-123
        _, block_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape
        block_size = block_size_per_kv_packing * kv_packing
        k_packed = jnp.pad(k_packed,
                           ((0, 0), (0, lkv_dim - k_packed.shape[-1])))
        tok = jnp.arange(k_packed.shape[0], dtype=jnp.int32)
        seq_id = jnp.searchsorted(cu_q_lens[1:], tok,
                                  side="right").astype(jnp.int32)
        q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
        pos = seq_lens[seq_id] - q_len + (tok - cu_q_lens[seq_id])
        page = jnp.where(
            tok < cu_q_lens[-1],
            page_indices.reshape(seq_lens.shape[0], -1)[seq_id,
                                                        pos // block_size],
            jnp.int32(2**30))
        slot = pos % block_size
        cache_kv = cache_kv.at[page, slot // kv_packing,
                               slot % kv_packing].set(k_packed)
        q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
        topk_indices = sit.streamindex_topk(
            q, weights, cache_kv, seq_lens, page_indices, cu_q_lens,
            distribution,
            k=topk, compression_ratio=1,
            num_kv_pages_per_block=(2, 2, 2),
            num_queries_per_block=(1, 128, 128),
            enable_early_exit=mao.STREAMIDX_ENABLE_EARLY_EXIT)
        return cache_kv, topk_indices

    outer = jax.jit(_streamidx_topk_jax, static_argnums=(),
                    donate_argnums=(2, ), keep_unused=True)
    compile_and_report("outer_streamidx_topk_jax", outer, _outer_avals(jnp),
                       {}, results, args.outdir)

    # ------------------------------------------------- synthetic control A/B
    # A fusion-blocked, same-shape, same-layout f32->s32 bitcast-convert of a
    # 40 MiB intermediate, against the identical graph that never needs one.
    # If a same-shape bitcast-convert is free, temp_size is equal.
    print("\n" + "=" * 74, flush=True)
    print("SYNTHETIC CONTROL -- does a same-shape bitcast-convert of a "
          "40 MiB\nfusion-blocked intermediate cost an allocation?",
          flush=True)
    print("=" * 74, flush=True)
    shp = (1024, 80, 128)
    sds = jax.ShapeDtypeStruct

    @jax.jit
    def syn_f32(a):
        y = jax.lax.optimization_barrier(a)          # blocks fusion
        w = jax.lax.bitcast_convert_type(y, jnp.int32)
        return jax.lax.optimization_barrier(w.reshape(-1))

    @jax.jit
    def syn_s32(a):
        y = jax.lax.optimization_barrier(a)
        return jax.lax.optimization_barrier(y.reshape(-1))

    syn = {}
    for lbl, fnc, dt in (("syn_f32_bcvt", syn_f32, jnp.float32),
                          ("syn_s32_none", syn_s32, jnp.int32)):
        try:
            c = fnc.lower(sds(shp, dt)).compile()
            t = c.as_text()
            with open(os.path.join(args.outdir, f"hlo-{lbl}.txt"), "w") as fh:
                fh.write(t)
            ma = c.memory_analysis()
            syn[lbl] = {"temp": ma.temp_size_in_bytes,
                        "arg": ma.argument_size_in_bytes,
                        "out": ma.output_size_in_bytes,
                        "n_bitcast_convert": t.count("bitcast-convert(")}
            print(f"  {lbl}: temp={ma.temp_size_in_bytes:,} "
                  f"({ma.temp_size_in_bytes / MIB:.3f} MiB)  "
                  f"arg={ma.argument_size_in_bytes:,}  "
                  f"out={ma.output_size_in_bytes:,}  "
                  f"bitcast-converts={syn[lbl]['n_bitcast_convert']}",
                  flush=True)
            from hlo_tools import Hlo
            report_bitcast_converts(Hlo(t), lbl, syn)
        except Exception as exc:
            traceback.print_exc()
            syn[lbl] = {"error": repr(exc)}
    if "temp" in syn.get("syn_f32_bcvt", {}) and \
            "temp" in syn.get("syn_s32_none", {}):
        d = syn["syn_f32_bcvt"]["temp"] - syn["syn_s32_none"]["temp"]
        syn["delta_temp_bytes"] = d
        print(f"  >>> delta temp = {d:,} bytes ({d / MIB:.3f} MiB). "
              f"40 MiB == the bitcast-convert got its own buffer; "
              f"0 == aliased.", flush=True)
    results["synthetic"] = syn

    # -------------------------------------------------- buffer assignment
    print("\n" + "=" * 74, flush=True)
    print("BUFFER ASSIGNMENT", flush=True)
    print("=" * 74, flush=True)
    files = sorted(glob.glob(os.path.join(dump, "*")))
    print(f"  {len(files)} files under {dump}", flush=True)
    ba = [f for f in files if "buffer-assignment" in os.path.basename(f)]
    for f in files[:400]:
        print(f"    {os.path.getsize(f):>12,d}  {os.path.basename(f)}",
              flush=True)
    # carry only the small, decisive files out to $CDK_OUTPUT_DIR
    import shutil
    keep = os.path.join(args.outdir, "xdump_keep")
    os.makedirs(keep, exist_ok=True)
    for f in files:
        bn = os.path.basename(f)
        want = ("buffer-assignment" in bn or "memory-usage-report" in bn
                or "live-range" in bn
                or ("streamidx" in bn.lower() and bn.endswith(".txt")))
        if want and os.path.getsize(f) < 200 * 1024 * 1024:
            try:
                shutil.copy2(f, os.path.join(keep, bn))
            except Exception as exc:
                print(f"    !! copy {bn}: {exc!r}", file=sys.stderr,
                      flush=True)
    print(f"  carried {len(os.listdir(keep))} files to {keep}", flush=True)
    results["dump_files"] = [os.path.basename(f) for f in files]
    results["buffer_assignment_files"] = [os.path.basename(f) for f in ba]
    if not ba:
        print("  !! no buffer-assignment file was dumped. Q1 falls back to "
              "the synthetic control and the window_config read.",
              file=sys.stderr, flush=True)
    results["buffer_assignment"] = {}
    for f in ba:
        names = set()
        for lbl, rec in results["modules"].items():
            for r in rec.get(f"{lbl}_bitcast_converts", []) or []:
                names.add(r["name"])
                if r["src"]:
                    names.add(r["src"])
        names = sorted(n for n in names if n)
        if not names:
            continue
        hits, allocs = parse_buffer_assignment(f, names)
        got = {n: v for n, v in hits.items() if v}
        if not got:
            continue
        print(f"\n  === {os.path.basename(f)} "
              f"({len(allocs)} allocations) ===", flush=True)
        for n, v in got.items():
            for e in v:
                print(f"    %{n}", flush=True)
                print(f"      {e['header'][:200]}", flush=True)
                print(f"      {e['value_line'][:220]}", flush=True)
        results["buffer_assignment"][os.path.basename(f)] = got
        # verdict for every bitcast-convert whose src is also placed
        for lbl, rec in results["modules"].items():
            for r in rec.get(f"{lbl}_bitcast_converts", []) or []:
                a = got.get(r["name"]) or []
                b = got.get(r["src"]) or []
                if not (a and b):
                    continue
                same = (a[0]["allocation"] == b[0]["allocation"]
                        and a[0]["offset"] == b[0]["offset"])
                verdict = ("ALIASED (free)" if same
                           else "DISTINCT BUFFERS (materialised)")
                print(f"    >>> [{lbl}] %{r['name']} alloc "
                      f"{a[0]['allocation']}@{a[0]['offset']} vs src "
                      f"%{r['src']} alloc {b[0]['allocation']}@"
                      f"{b[0]['offset']}  ==> {verdict}", flush=True)
                r["buffer_verdict"] = ("aliased" if same
                                       else "distinct_buffers")
                r["alloc_self"] = a[0]
                r["alloc_src"] = b[0]

    # the per-value `uses:` blocks, which say what each buffer is for
    print("\n  --- buffer-assignment VALUE blocks of interest ---", flush=True)
    keys = ("bitcast_convert_type", "StreamIdxTC", KERNEL)
    for f in files:
        if "buffer-assignment-values" not in os.path.basename(f):
            continue
        try:
            lines = open(f, errors="replace").read().splitlines()
        except Exception:
            continue
        blocks, cur = [], None
        for ln in lines:
            if ln.startswith("<"):
                if cur:
                    blocks.append(cur)
                cur = [ln]
            elif cur is not None:
                cur.append(ln)
        if cur:
            blocks.append(cur)
        hits = [b for b in blocks if any(k in b[0] for k in keys)]
        if not hits:
            continue
        print(f"\n  === {os.path.basename(f)} ===", flush=True)
        for b in hits[:40]:
            for ln in b[:10]:
                print(f"      {ln[:220]}", flush=True)

    for f in files:
        bn = os.path.basename(f)
        if "memory-usage-report" in bn and bn.endswith(".txt") \
                and ("streamidx" in bn.lower() or "syn_" in bn.lower()) \
                and os.path.getsize(f) < 200000:
            print(f"\n  === {bn} ===", flush=True)
            for ln in open(f, errors="replace").read().splitlines()[:60]:
                print(f"      {ln[:200]}", flush=True)

    with open(os.path.join(args.outdir, "boundary_provenance.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
