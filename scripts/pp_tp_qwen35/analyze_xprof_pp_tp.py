#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fast zero-dependency XPlane (.xplane.pb) analyzer for Qwen3.5 PP x TP Big Shark Hunting.

Uses a direct memoryview protobuf wire parser that skips /host:CPU and /host:metadata
planes in <1ms and parses /device:TPU:* and SparseCore planes in <0.5s per 300MB trace,
with zero dependency on torch, protobuf, or vllm_torchtpu.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import statistics
import struct
import sys
from typing import Any


def read_varint(buf: memoryview, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def skip_field(buf: memoryview, pos: int, wire_type: int) -> int:
    if wire_type == 0:
        while buf[pos] & 0x80:
            pos += 1
        return pos + 1
    elif wire_type == 1:
        return pos + 8
    elif wire_type == 2:
        length, pos = read_varint(buf, pos)
        return pos + length
    elif wire_type == 5:
        return pos + 4
    raise ValueError(f"Unsupported wire_type={wire_type} at pos={pos}")


def peek_plane_name(buf: memoryview, start: int, end: int) -> str:
    pos = start
    limit = min(end, start + 512)
    while pos < limit:
        tag, pos = read_varint(buf, pos)
        fn, wt = tag >> 3, tag & 7
        if fn == 2 and wt == 2:
            length, pos = read_varint(buf, pos)
            return bytes(buf[pos : pos + length]).decode("utf-8", errors="replace")
        pos = skip_field(buf, pos, wt)
    return ""


def parse_metadata_entry(buf: memoryview, start: int, end: int) -> tuple[int, str, str]:
    pos = start
    key = 0
    name = ""
    disp = ""
    while pos < end:
        tag, pos = read_varint(buf, pos)
        fn, wt = tag >> 3, tag & 7
        if fn == 1 and wt == 0:
            key, pos = read_varint(buf, pos)
        elif fn == 2 and wt == 2:
            vlen, pos = read_varint(buf, pos)
            vend = pos + vlen
            while pos < vend:
                vtag, pos = read_varint(buf, pos)
                vfn, vwt = vtag >> 3, vtag & 7
                if vfn == 1 and vwt == 0:
                    _, pos = read_varint(buf, pos)
                elif vfn == 2 and vwt == 2:
                    slen, pos = read_varint(buf, pos)
                    name = bytes(buf[pos : pos + slen]).decode("utf-8", errors="replace")
                    pos += slen
                elif vfn in (3, 4) and vwt == 2:
                    slen, pos = read_varint(buf, pos)
                    if vfn == 4 or not disp:
                        disp = bytes(buf[pos : pos + slen]).decode("utf-8", errors="replace")
                    pos += slen
                else:
                    pos = skip_field(buf, pos, vwt)
        else:
            pos = skip_field(buf, pos, wt)
    return key, name, disp


def parse_line_header(buf: memoryview, start: int, end: int) -> tuple[str, list[tuple[int, int]]]:
    pos = start
    line_name = ""
    disp_name = ""
    event_spans = []
    while pos < end:
        tag, pos = read_varint(buf, pos)
        fn, wt = tag >> 3, tag & 7
        if fn == 2 and wt == 2:
            slen, pos = read_varint(buf, pos)
            line_name = bytes(buf[pos : pos + slen]).decode("utf-8", errors="replace")
            pos += slen
        elif fn == 11 and wt == 2:
            slen, pos = read_varint(buf, pos)
            disp_name = bytes(buf[pos : pos + slen]).decode("utf-8", errors="replace")
            pos += slen
        elif fn == 4 and wt == 2:
            elen, pos = read_varint(buf, pos)
            event_spans.append((pos, pos + elen))
            pos += elen
        else:
            pos = skip_field(buf, pos, wt)
    return (line_name or disp_name), event_spans


def parse_event_fast(buf: memoryview, start: int, end: int) -> tuple[int, int, int]:
    pos = start
    meta_id = 0
    offset_ps = 0
    dur_ps = 0
    while pos < end:
        tag = buf[pos]
        if tag == 0x08:
            pos += 1
            meta_id, pos = read_varint(buf, pos)
        elif tag == 0x10:
            pos += 1
            offset_ps, pos = read_varint(buf, pos)
        elif tag == 0x18:
            pos += 1
            dur_ps, pos = read_varint(buf, pos)
        else:
            tag, pos = read_varint(buf, pos)
            pos = skip_field(buf, pos, tag & 7)
    return meta_id, offset_ps, dur_ps


def parse_event_stats(
    buf: memoryview,
    start: int,
    end: int,
    stat_meta: dict[int, str],
    ev_meta: dict[int, tuple[str, str]],
) -> dict[str, Any]:
    pos = start
    out: dict[str, Any] = {}
    while pos < end:
        tag, pos = read_varint(buf, pos)
        fn, wt = tag >> 3, tag & 7
        if fn == 4 and wt == 2:
            slen, pos = read_varint(buf, pos)
            send = pos + slen
            s_id = 0
            s_val: Any = None
            while pos < send:
                stag, pos = read_varint(buf, pos)
                sfn, swt = stag >> 3, stag & 7
                if sfn == 1 and swt == 0:
                    s_id, pos = read_varint(buf, pos)
                elif sfn == 2 and swt == 1:
                    s_val = struct.unpack_from("<d", buf, pos)[0]
                    pos += 8
                elif sfn in (3, 4) and swt == 0:
                    s_val, pos = read_varint(buf, pos)
                elif sfn == 5 and swt == 2:
                    vlen, pos = read_varint(buf, pos)
                    s_val = bytes(buf[pos : pos + vlen]).decode("utf-8", errors="replace")
                    pos += vlen
                elif sfn == 7 and swt == 0:
                    ref_id, pos = read_varint(buf, pos)
                    s_val = ev_meta.get(ref_id, (str(ref_id), ""))[0]
                else:
                    pos = skip_field(buf, pos, swt)
            sname = stat_meta.get(s_id, "")
            if sname and s_val is not None:
                out[sname] = s_val
        else:
            pos = skip_field(buf, pos, wt)
    return out


def classify_qwen35_hlo(name: str, display_name: str, stats_str: str) -> str:
    nd = f"{name} {display_name}".lower()
    full = f"{nd} {stats_str}".lower()
    if "collective-permute" in full or "ppermute" in full:
        return "1_PP_CollectivePermute"
    if any(k in full for k in ("all-reduce", "all_reduce")):
        return "2_TP_AllReduce"
    if any(
        k in full
        for k in (
            "all-to-all",
            "all_to_all",
            "ragged-all-to-all",
            "ragged_all_to_all",
            "reduce-scatter",
            "all-gather",
        )
    ):
        return "3_EP_AllToAll_OtherCollectives"
    if any(
        k in full
        for k in (
            "fused_moe",
            "moe_ep",
            "moe_gmm",
            "gmm",
            "ragged_dot",
            "ragged-dot",
            "megablox",
            "expert",
            "router",
            "top_k",
            "topk",
            "radix_sort",
        )
    ):
        return "4_MoE_Kernels_GMM_EP"
    if any(
        k in full
        for k in (
            "rpa_kernel",
            "rpa_",
            "rpam",
            "paged_attention",
            "flash_attention",
            "qk_norm_rope",
        )
    ):
        return "5_FullAttn_RPA_QKNormRoPE"
    if any(
        k in full
        for k in (
            "gdn",
            "delta_rule",
            "chunk_gated",
            "sigmoid_gate_quant",
            "conv1d",
            "causal_conv",
            "mamba",
        )
    ):
        return "6_GDN_LinearAttn_Conv_GateQuant"
    if any(k in full for k in ("custom-call", "tpu_custom_call", "mosaic", "pallas")):
        return "7_Other_Pallas_CustomCall"
    if any(k in nd for k in ("dot", "convolution", "matmul")):
        return "8_Dense_Matmul_Dot"
    if any(k in nd for k in ("copy", "relayout", "bitcast", "reshape", "transpose", "slice", "gather")):
        return "9_Copy_Relayout_Gather"
    return "10_Fusion_Elementwise_Norm"


def normalize_op_name(name: str, display_name: str) -> str:
    head = name.split("=")[0].strip() if "=" in name else name.strip()
    head_base = re.sub(r"\.\d+(\.cloned\.\d+)?$", "", head)
    if display_name and display_name != name:
        disp_clean = re.sub(r"\.\d+\b", "", display_name)[:90]
        return f"{head_base} [{disp_clean}]"
    if "=" in name:
        rhs = name.split("=", 1)[1].strip()[:90]
        rhs = re.sub(r"\.\d+\b", "", rhs)
        return f"{head_base} = {rhs}"
    return head_base[:100]


def analyze_module_window(
    buf: memoryview,
    mname: str,
    spans: list[tuple[int, int]],
    ops_parsed: list[tuple[int, int, int, int, int]],
    ev_meta: dict[int, tuple[str, str]],
    st_meta: dict[int, str],
) -> dict[str, Any]:
    sorted_spans = sorted(spans, key=lambda x: x[1])
    med_start, med_dur = sorted_spans[len(sorted_spans) // 2]
    med_end = med_start + med_dur
    med_dur_ms = med_dur / 1e9

    cat_totals_ps: dict[str, int] = collections.defaultdict(int)
    cat_counts: dict[str, int] = collections.defaultdict(int)
    op_totals_ps: dict[str, int] = collections.defaultdict(int)
    op_counts: dict[str, int] = collections.defaultdict(int)
    op_cat: dict[str, str] = {}
    op_sample_stats: dict[str, str] = {}

    num_ops_in_window = 0
    for mid, off_ps, dur_ps, estart, eend in ops_parsed:
        if (med_start - 1_000_000) <= off_ps < med_end:
            num_ops_in_window += 1
            raw_name, disp_name = ev_meta.get(mid, (f"id_{mid}", ""))
            norm_key = normalize_op_name(raw_name, disp_name)
            if norm_key not in op_sample_stats:
                st_dict = parse_event_stats(buf, estart, eend, st_meta, ev_meta)
                s_parts = [
                    f"{k}={str(v)[:60]}"
                    for k, v in st_dict.items()
                    if k in ("hlo_category", "tf_op", "long_name", "source_info")
                ]
                op_sample_stats[norm_key] = " ".join(s_parts)
            s_str = op_sample_stats.get(norm_key, "")
            cat = classify_qwen35_hlo(raw_name, disp_name, s_str)
            cat_totals_ps[cat] += dur_ps
            cat_counts[cat] += 1
            op_totals_ps[norm_key] += dur_ps
            op_counts[norm_key] += 1
            op_cat[norm_key] = cat

    total_ops_ms = sum(cat_totals_ps.values()) / 1e9
    cats_out = {}
    for cat, tot_ps in sorted(cat_totals_ps.items()):
        ms = tot_ps / 1e9
        cats_out[cat] = {
            "count": cat_counts[cat],
            "total_ms": round(ms, 3),
            "pct_of_step": round(100.0 * ms / max(med_dur_ms, 1e-6), 2),
        }

    top_ops_out = []
    for norm_key, tot_ps in sorted(op_totals_ps.items(), key=lambda kv: kv[1], reverse=True)[:35]:
        ms = tot_ps / 1e9
        cnt = op_counts[norm_key]
        top_ops_out.append(
            {
                "op": norm_key,
                "category": op_cat[norm_key],
                "count": cnt,
                "total_ms": round(ms, 3),
                "mean_us": round((tot_ps / 1e6) / max(cnt, 1), 1),
                "pct_of_step": round(100.0 * ms / max(med_dur_ms, 1e-6), 2),
                "info": op_sample_stats.get(norm_key, ""),
            }
        )

    return {
        "module": mname,
        "num_ops": num_ops_in_window,
        "step_duration_ms": round(med_dur_ms, 3),
        "sum_xla_ops_ms": round(total_ops_ms, 3),
        "categories": cats_out,
        "top_ops": top_ops_out,
    }


def analyze_rank_xplane(path: str) -> dict[str, Any]:
    raw_bytes = open(path, "rb").read()
    buf = memoryview(raw_bytes)
    end = len(buf)
    pos = 0

    planes_found: list[str] = []
    result: dict[str, Any] = {
        "path": path,
        "planes": planes_found,
        "modules": {},
        "step_breakdown": {},
        "all_module_breakdowns": [],
        "async_ops_summary": [],
        "sparse_core_summary": {},
    }

    while pos < end:
        tag, pos = read_varint(buf, pos)
        fn, wt = tag >> 3, tag & 7
        if fn == 1 and wt == 2:
            plen, pos = read_varint(buf, pos)
            pstart, pend = pos, pos + plen
            pos = pend
            pname = peek_plane_name(buf, pstart, pend)
            planes_found.append(pname)

            if not pname.startswith("/device:TPU:"):
                continue

            ev_meta: dict[int, tuple[str, str]] = {}
            st_meta: dict[int, str] = {}
            lines: list[tuple[str, list[tuple[int, int]]]] = []
            ppos = pstart
            while ppos < pend:
                ptag, ppos = read_varint(buf, ppos)
                pfn, pwt = ptag >> 3, ptag & 7
                if pfn == 3 and pwt == 2:
                    llen, ppos = read_varint(buf, ppos)
                    lines.append(parse_line_header(buf, ppos, ppos + llen))
                    ppos += llen
                elif pfn == 4 and pwt == 2:
                    mlen, ppos = read_varint(buf, ppos)
                    k, n, d = parse_metadata_entry(buf, ppos, ppos + mlen)
                    ev_meta[k] = (n, d)
                    ppos += mlen
                elif pfn == 5 and pwt == 2:
                    slen, ppos = read_varint(buf, ppos)
                    k, n, _ = parse_metadata_entry(buf, ppos, ppos + slen)
                    st_meta[k] = n
                    ppos += slen
                else:
                    ppos = skip_field(buf, ppos, pwt)

            if "SparseCore" in pname:
                sc_totals_ps: dict[str, int] = collections.defaultdict(int)
                sc_counts: dict[str, int] = collections.defaultdict(int)
                for lname, espans in lines:
                    if "ops" not in lname.lower() and "step" not in lname.lower():
                        continue
                    for estart, eend in espans:
                        mid, _, dur_ps = parse_event_fast(buf, estart, eend)
                        mname, mdisp = ev_meta.get(mid, (f"id_{mid}", ""))
                        label = mdisp or mname
                        sc_totals_ps[label] += dur_ps
                        sc_counts[label] += 1
                top_sc = []
                for label, tot_ps in sorted(sc_totals_ps.items(), key=lambda kv: kv[1], reverse=True)[:15]:
                    top_sc.append(
                        {
                            "op": label[:90],
                            "count": sc_counts[label],
                            "total_ms": round(tot_ps / 1e9, 3),
                        }
                    )
                result["sparse_core_summary"][pname] = top_sc
                continue

            if pname != "/device:TPU:0":
                continue

            mod_spans = None
            ops_spans = None
            async_ops_spans = None
            for lname, espans in lines:
                ll = lname.strip().lower()
                if ll == "xla modules":
                    mod_spans = espans
                elif ll == "xla ops":
                    ops_spans = espans
                elif ll == "async xla ops":
                    async_ops_spans = espans

            if async_ops_spans:
                async_totals_ps: dict[str, int] = collections.defaultdict(int)
                async_counts: dict[str, int] = collections.defaultdict(int)
                for estart, eend in async_ops_spans:
                    mid, _, dur_ps = parse_event_fast(buf, estart, eend)
                    mname, mdisp = ev_meta.get(mid, (f"id_{mid}", ""))
                    norm_key = normalize_op_name(mname, mdisp)
                    async_totals_ps[norm_key] += dur_ps
                    async_counts[norm_key] += 1
                result["async_ops_summary"] = [
                    {
                        "op": k,
                        "count": async_counts[k],
                        "total_ms": round(v / 1e9, 3),
                        "mean_us": round((v / 1e6) / max(async_counts[k], 1), 1),
                    }
                    for k, v in sorted(async_totals_ps.items(), key=lambda kv: kv[1], reverse=True)[:15]
                ]

            if mod_spans:
                by_mod: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
                for estart, eend in mod_spans:
                    mid, off_ps, dur_ps = parse_event_fast(buf, estart, eend)
                    mname, mdisp = ev_meta.get(mid, (f"id_{mid}", ""))
                    label = mdisp or mname
                    by_mod[label].append((off_ps, dur_ps))

                mod_summary = {}
                for mname, spans in sorted(
                    by_mod.items(), key=lambda kv: sum(d for _, d in kv[1]), reverse=True
                ):
                    spans.sort()
                    durs_ms = [d / 1e9 for _, d in spans]
                    gaps_ms = [
                        (spans[i][0] - (spans[i - 1][0] + spans[i - 1][1])) / 1e9
                        for i in range(1, len(spans))
                    ]
                    mod_summary[mname] = {
                        "count": len(durs_ms),
                        "total_ms": round(sum(durs_ms), 3),
                        "mean_ms": round(statistics.mean(durs_ms), 3),
                        "median_ms": round(statistics.median(durs_ms), 3),
                        "min_ms": round(min(durs_ms), 3),
                        "max_ms": round(max(durs_ms), 3),
                        "median_gap_ms": round(statistics.median(gaps_ms), 3) if gaps_ms else 0.0,
                    }
                result["modules"] = mod_summary

                if by_mod and ops_spans:
                    ops_parsed = []
                    for estart, eend in ops_spans:
                        mid, off_ps, dur_ps = parse_event_fast(buf, estart, eend)
                        ops_parsed.append((mid, off_ps, dur_ps, estart, eend))

                    breakdowns = []
                    for mname, spans in sorted(
                        by_mod.items(), key=lambda kv: sum(d for _, d in kv[1]), reverse=True
                    )[:5]:
                        bd = analyze_module_window(buf, mname, spans, ops_parsed, ev_meta, st_meta)
                        breakdowns.append(bd)

                    result["all_module_breakdowns"] = breakdowns
                    if breakdowns:
                        result["step_breakdown"] = max(
                            breakdowns, key=lambda b: b["num_ops"]
                        )
        else:
            pos = skip_field(buf, pos, wt)

    return result


def run_http_server(port: int) -> int:
    import http.server
    import subprocess
    import time

    script_path = os.path.abspath(__file__)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            time.sleep(3)
            out_dir = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
            js_name = os.environ.get("JS_NAME", "unknown-job")
            node_idx = os.environ.get("JOBSET_JOB_INDEX", "0")
            proc = subprocess.run(
                [sys.executable, script_path, out_dir],
                capture_output=True,
                text=True,
            )
            report = (
                f"=== XPROF ANALYSIS (node={node_idx}, rc={proc.returncode}) ===\n"
                f"{proc.stdout}\n{proc.stderr}\n"
            )
            print(report, flush=True)
            gcs_dst = f"gs://yinlin-tpu-bucket-us/shriharim-pptp/xprof/{js_name}/node-{node_idx}/"
            up = subprocess.run(
                ["gcloud", "storage", "cp", "-r", out_dir, gcs_dst],
                capture_output=True,
                text=True,
            )
            report += (
                f"\n=== GCS UPLOAD to {gcs_dst} (rc={up.returncode}) ===\n"
                f"{up.stdout}\n{up.stderr}\n"
            )
            print(f"Uploaded traces to {gcs_dst} (rc={up.returncode})", flush=True)
            body = report.encode("utf-8", errors="replace")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"Starting XProf analysis HTTP server on 0.0.0.0:{port}...", flush=True)
    http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--serve":
        port = int(sys.argv[2]) if len(sys.argv) >= 3 else 8099
        return run_http_server(port)

    root_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/xprof_traces"
    if not os.path.exists(root_dir):
        print(f"Trace root directory {root_dir} does not exist.", file=sys.stderr)
        return 1

    session_dirs = set()
    for path in glob.glob(os.path.join(root_dir, "**", "*.xplane.pb"), recursive=True):
        session_dirs.add(os.path.dirname(path))

    if not session_dirs:
        print(f"No .xplane.pb files found under {root_dir}.", file=sys.stderr)
        return 1

    full_report: dict[str, Any] = {"root_dir": root_dir, "sessions": {}}

    for sdir in sorted(session_dirs):
        rel_sdir = os.path.relpath(sdir, root_dir)
        xplane_files = sorted(glob.glob(os.path.join(sdir, "*.xplane.pb")))
        phase_dir = os.path.dirname(os.path.dirname(os.path.dirname(sdir)))
        stats_files = sorted(
            glob.glob(os.path.join(phase_dir, "**", "batch_composition_stats_*.json"), recursive=True)
        )
        batch_stats = []
        for sf in stats_files[:20]:
            try:
                with open(sf) as f:
                    batch_stats.append(json.load(f))
            except Exception:
                pass

        print("=" * 100)
        print(f"XPROF SESSION: {rel_sdir} ({len(xplane_files)} rank traces)")
        if batch_stats:
            print(f"  Batch Composition Stats ({len(batch_stats)} samples, showing first 5):")
            for bs in batch_stats[:5]:
                print(f"    {json.dumps(bs)}")

        session_data: dict[str, Any] = {"batch_stats": batch_stats, "ranks": {}}
        for xp_path in xplane_files:
            bname = os.path.basename(xp_path)
            m = re.match(r"^(rank\d+)_", bname)
            rlabel = m.group(1) if m else bname
            rdata = analyze_rank_xplane(xp_path)
            session_data["ranks"][rlabel] = rdata

            sb = rdata.get("step_breakdown", {})
            mods = rdata.get("modules", {})
            print(f"\n  --- {rlabel} ({bname}) ---")
            for mname, mstat in list(mods.items())[:5]:
                print(
                    f"    Module {mname}: count={mstat['count']} "
                    f"median={mstat['median_ms']}ms mean={mstat['mean_ms']}ms "
                    f"min={mstat['min_ms']}ms max={mstat['max_ms']}ms "
                    f"median_gap={mstat['median_gap_ms']}ms"
                )
            if sb:
                print(
                    f"    Main Forward Breakdown (module={sb['module']}, "
                    f"num_ops={sb['num_ops']}, step={sb['step_duration_ms']}ms, "
                    f"sum_ops={sb['sum_xla_ops_ms']}ms):"
                )
                for cat, cstat in sb["categories"].items():
                    print(
                        f"      {cat:35s}: {cstat['total_ms']:8.3f} ms "
                        f"({cstat['pct_of_step']:5.1f}%, count={cstat['count']})"
                    )
                if rlabel in ("rank0", "rank2", "rank4", "rank6") or len(xplane_files) == 1:
                    print(f"    Top 25 HLO Ops on {rlabel}:")
                    for idx, op in enumerate(sb["top_ops"][:25], 1):
                        print(
                            f"      {idx:2d}. [{op['category']}] {op['total_ms']:7.3f}ms "
                            f"({op['pct_of_step']:4.1f}%) | n={op['count']:4d} | "
                            f"mean={op['mean_us']:7.1f}us | {op['op']} {op['info']}"
                        )
            sc = rdata.get("sparse_core_summary", {})
            for sc_plane, sc_ops in sc.items():
                if sc_ops and (rlabel in ("rank0", "rank2") or len(xplane_files) == 1):
                    print(f"    SparseCore ({sc_plane}) Top Ops:")
                    for op in sc_ops[:6]:
                        print(f"      - {op['total_ms']:7.3f}ms (n={op['count']}): {op['op']}")

        full_report["sessions"][rel_sdir] = session_data

    out_json = os.path.join(root_dir, "xprof_pp_tp_analysis.json")
    with open(out_json, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved full JSON analysis to {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
