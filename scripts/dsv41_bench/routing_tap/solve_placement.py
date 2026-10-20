# SPDX-License-Identifier: Apache-2.0
"""Solve static per-layer expert placements for TPU_MOE_EXPERT_PERM_PATH.

    python solve_placement.py <routing dir> <out.json> [--cal round1] [--eval round0]

<routing dir> holds convert.py's routing_<label>.npz. Per layer, the placement
balances the tile model's per-expert cost (tile_model.expert_cost) of the mean
calibration-step rows over the EP devices, 48 experts each (LPT + swaps). The
file maps checkpoint expert -> new id; it also records, per layer, the tile
model's kernel time as routed and as placed on the calibration and the
evaluation steps.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tile_model as tm  # noqa: E402
from k4 import per_step_predictions  # noqa: E402


def solve(counts):
    """Per layer: perm[g] = new id, from counts [S, L, EP, E]."""
    perms = []
    for li in range(counts.shape[1]):
        rows = np.mean([tm.rows_from_counts(c) for c in counts[:, li]], 0)
        perms.append(tm.balanced_perm(np.round(rows).astype(int)))
    return perms


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("routing", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cal", default="round1")
    ap.add_argument("--eval", default="round0")
    a = ap.parse_args()
    cal = np.load(a.routing / f"routing_{a.cal}.npz")
    ev = np.load(a.routing / f"routing_{a.eval}.npz")
    layers = cal["layers"].tolist()
    if ev["layers"].tolist() != layers:
        raise ValueError("calibration and evaluation tables cover different layers")
    perms = solve(cal["counts"])
    p_cal = per_step_predictions(cal["counts"], perms).mean(1)
    p_ev = per_step_predictions(ev["counts"], perms).mean(1)
    report = {
        name: {
            "routed_us": p[0].round(1).tolist(),
            "device_mean_us": p[1].round(1).tolist(),
            "placed_us": p[2].round(1).tolist(),
        }
        for name, p in ((a.cal, p_cal), (a.eval, p_ev))
    }
    for name, steps in (
        (a.cal, cal["counts"].shape[0]),
        (a.eval, ev["counts"].shape[0]),
    ):
        report[name]["steps"] = int(steps)
    doc = {
        "perm": {str(li): perms[i].tolist() for i, li in enumerate(layers)},
        "meta": {
            "source": str(a.routing.resolve()),
            "calibration": a.cal,
            "evaluation": a.eval,
            "cost_model": "routing_tap/tile_model.py PARAMS",
            "layers": layers,
            "predicted": report,
        },
    }
    a.out.write_text(json.dumps(doc))
    for name, p in ((a.cal, p_cal), (a.eval, p_ev)):
        print(
            f"{name}: tile-model kernel us/layer, mean over {len(layers)} layers: "
            f"routed {p[0].mean():.1f}, placed {p[2].mean():.1f} "
            f"(device mean {p[1].mean():.1f}); layers 22-35 routed "
            f"{p[0][22:36].mean():.1f} placed {p[2][22:36].mean():.1f}"
        )


if __name__ == "__main__":
    main()
