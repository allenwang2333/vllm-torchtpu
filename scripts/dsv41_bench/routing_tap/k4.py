# SPDX-License-Identifier: Apache-2.0
"""Decide K4 (static expert placement) from served routing.

    python k4.py <routing dir> [--profile-us prof.json] [--baseline-us base.json]
                 [--replay-dir DIR]

<routing dir> holds convert.py's routing_<label>.npz files: --cal (placement is
solved on these steps) and --held (evaluated on these). Per layer it prints the
tile model's kernel time as routed (slowest device), at the device mean, and
with a placement balanced on the calibration steps, plus the measured kernel
time where given (json: {"per_step": [[us]*40]*S} or {"per_layer": [us]*40}).
With --replay-dir it writes harness tables (topk_ids [1, 2304, 6]) for the
chosen layers, as routed and relabelled by the balanced placement.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tile_model as tm  # noqa: E402


def per_step_predictions(counts, perms=None):
    """counts [S, L, EP, E] -> (routed, mean, balanced) [S, L] kernel us."""
    S, L = counts.shape[:2]
    out = np.zeros((3, S, L))
    for s in range(S):
        for li in range(L):
            rows = tm.rows_from_counts(counts[s, li])
            out[0, s, li], out[1, s, li], _ = tm.predict(rows)
            if perms is not None:
                out[2, s, li] = tm.predict(rows, perms[li])[0]
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("routing", type=Path)
    ap.add_argument("--profile-us", type=Path)
    ap.add_argument("--baseline-us", type=Path)
    ap.add_argument("--replay-dir", type=Path)
    ap.add_argument("--replay-layers", default="10,11,12,13,22,23,24,27,29,32")
    ap.add_argument("--replay-step", type=int, default=0)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--cal", default="round1")
    ap.add_argument("--held", default="round0")
    a = ap.parse_args()

    cal = np.load(a.routing / f"routing_{a.cal}.npz")
    held = np.load(a.routing / f"routing_{a.held}.npz")
    layers = cal["layers"].tolist()
    assert layers == held["layers"].tolist()
    mean_rows = np.stack(
        [
            np.mean([tm.rows_from_counts(c) for c in cal["counts"][:, li]], 0)
            for li in range(len(layers))
        ]
    )
    perms = [tm.balanced_perm(np.round(r).astype(int)) for r in mean_rows]
    p_cal = per_step_predictions(cal["counts"], perms)
    p_held = per_step_predictions(held["counts"], perms)

    meas = None
    if a.profile_us:
        meas = np.array(json.loads(a.profile_us.read_text())["per_step"])
    base = None
    if a.baseline_us:
        base = np.array(json.loads(a.baseline_us.read_text())["per_layer"])

    pr, pm, pb = p_held.mean(1)
    print(
        f"steps: calibration ({a.cal}) {cal['counts'].shape[0]}, held-out ({a.held}) "
        f"{held['counts'].shape[0]}"
    )
    hdr = "layer  routed  dev-mean  slow-mean  balanced(held)  balanced(cal)"
    hdr += "  measured(prof)" if meas is not None else ""
    hdr += "  baseline-d186106" if base is not None else ""
    print(hdr)
    for i, li in enumerate(layers):
        line = (
            f"{li:5d}  {pr[i]:6.1f}  {pm[i]:8.1f}  {pr[i] - pm[i]:9.1f}  "
            f"{pb[i]:14.1f}  "
            f"{p_cal[2][:, i].mean():13.1f}"
        )
        if meas is not None:
            line += f"  {meas[:, i].mean():14.1f}"
        if base is not None:
            line += f"  {base[i]:16.1f}"
        print(line)
    res = dict(
        layers=layers,
        routed=pr.tolist(),
        dev_mean=pm.tolist(),
        balanced_held=pb.tolist(),
        balanced_cal=p_cal[2].mean(0).tolist(),
        routed_cal=p_cal[0].mean(0).tolist(),
    )
    print(
        f"mean over layers: routed {pr.mean():.1f}  device-mean {pm.mean():.1f}  "
        f"slowest-minus-mean {np.mean(pr - pm):.1f}  balanced(held) {pb.mean():.1f}"
    )
    if meas is not None:
        # Per-layer kernel time is stable across steps and starts, so the model's
        # layer means (routing of other steps) are compared with the profile's.
        mm = meas.mean(0)
        err = pr - mm
        rms = float(np.sqrt(np.mean(err**2)) / mm.mean())
        corr = float(np.corrcoef(pr, mm)[0, 1])
        slow = [layers.index(li) for li in range(22, 36)]
        fast = [layers.index(li) for li in (10, 11, 12, 13)]
        print(
            f"model vs profile, per layer: RMS {100 * rms:.1f}% of mean, mean bias "
            f"{err.mean():+.1f} us; layer corr {corr:.3f}; measured slow(22-35) "
            f"{mm[slow].mean():.1f} fast(10-13) {mm[fast].mean():.1f}; "
            f"model slow {pr[slow].mean():.1f} fast {pr[fast].mean():.1f}"
        )
        res.update(
            measured=mm.tolist(), rms_frac=rms, corr=corr, bias=float(err.mean())
        )
    if a.json_out:
        a.json_out.write_text(json.dumps(res, indent=1))
    if a.replay_dir:
        a.replay_dir.mkdir(parents=True, exist_ok=True)
        s = a.replay_step
        for li in (int(x) for x in a.replay_layers.split(",")):
            i = layers.index(li)
            ids = held["topk_ids"][i, s]
            np.savez(
                a.replay_dir / f"L{li:02d}_s{s}_routed.npz",
                topk_ids=ids[None].astype(np.int32),
            )
            np.savez(
                a.replay_dir / f"L{li:02d}_s{s}_balanced.npz",
                topk_ids=perms[i][ids][None].astype(np.int32),
            )
            r = tm.expert_rows(ids)
            print(
                f"replay L{li:02d} step {s}: model routed {tm.predict(r)[0]:.1f} "
                f"balanced {tm.predict(r, perms[i])[0]:.1f}"
            )


if __name__ == "__main__":
    main()
