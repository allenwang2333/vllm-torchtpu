# SPDX-License-Identifier: Apache-2.0
"""Per-step, per-layer fused-EP MoE kernel durations from a run.py --profile capture.

    XP_DIR=<run>/profile/plugins/profile/<ts> python profile_moe_us.py OUT.json
    python profile_moe_us.py OUT.json \
        --ops ~/dsv41-work/analysis/decode-<rev>-ops-r0.json

Target layers are the g48 kernels (40 per step), draft layers the g16 ones. Writes
{"per_step": [[us]*40]*S, "per_layer": [...], "draft_per_step": [[us]*3]*S,
"device_spread_us": max-min over the eight devices of rank 0..7 (xplane mode)}.
"""

import argparse
import json
import os
import sys

import numpy as np


def from_ops(path):
    with open(path) as f:
        ops = json.load(f)["ops"]
    tgt = [
        o["dur"]
        for o in ops
        if o["op"].startswith("fused_ep_moe_v2") and "_g48_" in o["op"]
    ]
    dr = [
        o["dur"]
        for o in ops
        if o["op"].startswith("fused_ep_moe_v2") and "_g16_" in o["op"]
    ]
    return {"rank0": (tgt, dr)}


def from_xplane():
    sys.path.insert(0, os.path.expanduser("~/dsv41-work/analysis"))
    from xp import cap  # noqa: E402

    out = {}
    for rank in range(8):
        c = cap(rank)
        for p in c.planes:
            if not p.name.startswith("/device:TPU:") or "XLA Ops" not in p.lines:
                continue
            ln = p.lines["XLA Ops"]
            names = [p.names[i] for i in ln.name_idx]
            dur = ln.dur_ps / 1e6
            tgt = [
                float(dur[i])
                for i, n in enumerate(names)
                if n.startswith("%fused_ep_moe_v2") and "_g48_" in n
            ]
            dr = [
                float(dur[i])
                for i, n in enumerate(names)
                if n.startswith("%fused_ep_moe_v2") and "_g16_" in n
            ]
            out[f"r{rank}{p.name}"] = (tgt, dr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--ops")
    a = ap.parse_args()
    planes = from_ops(a.ops) if a.ops else from_xplane()
    mats = {}
    for k, (tgt, dr) in planes.items():
        S = len(tgt) // 40
        mats[k] = (
            np.array(tgt[: S * 40]).reshape(S, 40),
            np.array(dr[: S * 3]).reshape(-1, 3),
        )
    first = next(iter(mats.values()))
    res = {
        "per_step": first[0].tolist(),
        "per_layer": first[0].mean(0).tolist(),
        "draft_per_step": first[1].tolist(),
        "planes": list(mats),
    }
    if len(mats) > 1:
        stack = np.stack([m[0] for m in mats.values() if m[0].shape == first[0].shape])
        res["device_spread_us"] = float((stack.max(0) - stack.min(0)).max())
    with open(a.out, "w") as f:
        json.dump(res, f)
    print(
        f"{len(mats)} planes, {first[0].shape[0]} steps, "
        f"mean {first[0].mean():.1f} us/layer, "
        f"draft {first[1].mean() if first[1].size else float('nan'):.1f}"
    )


if __name__ == "__main__":
    main()
