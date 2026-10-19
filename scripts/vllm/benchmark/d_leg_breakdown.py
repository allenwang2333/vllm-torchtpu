#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Component breakdown of one decode step from a d_leg_bench xprof capture (rank 0, TPU:0).

    python3 scripts/vllm/benchmark/d_leg_breakdown.py <rank0 .xplane.pb> [--steps N]

Steps are counted from the backbone program's executions. Device op time (XLA Ops line) is
grouped by program (target backbone / MTP draft step 0 / MTP draft loop / other) and by
component (attention, GDN, MoE experts, MoE dispatch, collectives, dense matmul, other).
"""

import argparse
import collections
import os  # noqa: E402
import re
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src")
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from vllm_torchtpu.tools import xplane_pb2  # noqa: E402
except Exception:  # standalone: proto module next to this file
    import xplane_pb2  # type: ignore  # noqa: E402

COMP = [
    ("attention", r"^RPAd|rpa_metadata|qk_norm_rope|sigmoid_gate_quant|^RPAm|^RPAp"),
    ("gdn", r"fused_conv1d_gdn|fused_out_proj|gdn"),
    ("moe_experts", r"moe_lowtok|^gmm_v2|fused_ep_moe|^moe$|^moe\b|dense_expert"),
    ("moe_dispatch", r"fused_ep_|sc_ragged|scatter_offload|^sort|gather_rows|ragged"),
    (
        "collective",
        r"all-reduce|all_reduce|all-gather|all_gather|reduce-scatter|reduce_scatter|"
        r"collective-permute|psum|small_all_reduce",
    ),
    ("dense_matmul", r"^convolution|^fusion\.\d+\.refs|convolution"),
]


def comp_of(name, cat):
    for c, pat in COMP:
        if re.search(pat, name):
            return c
    if cat and "convolution" in cat:
        return "dense_matmul"
    if cat in ("all-reduce", "all-gather", "reduce-scatter"):
        return "collective"
    if cat == "data formatting":
        return "copies"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--steps", type=float, default=0)
    a = ap.parse_args()
    xs = xplane_pb2.XSpace()
    xs.ParseFromString(open(a.path, "rb").read())
    dev = next(p for p in xs.planes if p.name == "/device:TPU:0")
    smd = {k: v.name for k, v in dev.stat_metadata.items()}
    # program id -> group, from the XLA Modules line
    prog_group, prog_runs = {}, collections.Counter()
    mod_ms = collections.Counter()
    for ln in dev.lines:
        if ln.name != "XLA Modules":
            continue
        for e in ln.events:
            md = dev.event_metadata.get(e.metadata_id)
            name = md.name if md else ""
            m = re.search(r"\((\d+)\)$", name)
            pid = m.group(1) if m else name
            prog_runs[pid] += 1
            mod_ms[pid] += e.duration_ps / 1e9
            if "mtp_fused_loop" in name:
                prog_group[pid] = "mtp_loop"
            elif "patched_gemma_rmsnorm" in name or "all_gather_into_tensor" in name:
                prog_group.setdefault(pid, "gemma")
            else:
                prog_group.setdefault(pid, "other")
    ops = collections.defaultdict(lambda: collections.Counter())
    for ln in dev.lines:
        if ln.name != "XLA Ops":
            continue
        for e in ln.events:
            md = dev.event_metadata.get(e.metadata_id)
            if not md:
                continue
            name = md.display_name or md.name
            st = {smd.get(s.metadata_id): s for s in md.stats}
            cat = st["hlo_category"].str_value if "hlo_category" in st else ""
            pid = (
                str(st["program_id"].uint64_value or st["program_id"].int64_value)
                if "program_id" in st
                else "?"
            )
            ops[pid][comp_of(name, cat)] += e.duration_ps / 1e9
    tot_by_pid = {pid: sum(c.values()) for pid, c in ops.items()}
    gem = sorted(
        (p for p, g in prog_group.items() if g == "gemma"),
        key=lambda p: -tot_by_pid.get(p, 0),
    )
    names = {}
    if gem:
        names[gem[0]] = "target"
        for p in gem[1:]:
            names[p] = "mtp_step0"
    for p, g in prog_group.items():
        if g == "mtp_loop":
            names[p] = "mtp_loop"
    steps = a.steps or (prog_runs[gem[0]] if gem else 1)
    groups = collections.defaultdict(collections.Counter)
    for pid, c in ops.items():
        groups[names.get(pid, "other")].update(c)
    comps = [c for c, _ in COMP] + ["copies", "other"]
    total = sum(sum(c.values()) for c in groups.values()) / steps
    print(f"steps={steps:g} device_op_ms_per_step={total:.2f}")
    print("group," + ",".join(comps) + ",total")
    for g in ("target", "mtp_step0", "mtp_loop", "other"):
        c = groups.get(g, collections.Counter())
        print(
            g
            + ","
            + ",".join(f"{c[k] / steps:.2f}" for k in comps)
            + f",{sum(c.values()) / steps:.2f}"
        )
    allc = collections.Counter()
    for c in groups.values():
        allc.update(c)
    print("all," + ",".join(f"{allc[k] / steps:.2f}" for k in comps) + f",{total:.2f}")
    print(
        "pct,"
        + ",".join(f"{100 * allc[k] / steps / total:.1f}" for k in comps)
        + ",100"
    )


if __name__ == "__main__":
    main()
