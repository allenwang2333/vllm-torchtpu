# SPDX-License-Identifier: Apache-2.0
"""Turn the tap's per-rank dumps into step-aligned tables.

    python convert.py <run dir> <out dir> [--label round1|profile]

Reads <run dir>/tap/rank-<r>-<label>.npz (r = 0..7) and, for a benchmark round,
<run dir>/fixed/round-<n>-steps.json to mark which recorded forwards had all
48 requests on every rank. Writes <out dir>/routing_<label>.npz:

  topk_ids   int32 [L, S, 8*rows, topk]  global expert ids; token block r = DP/EP rank r
  counts     int32 [S, L, 8, experts]    rows from source rank r routed to each expert
  layers     int32 [L]                   decoder layer index of each table
  draft_*    the same for the DSpark draft MoE layers, when recorded
  step_full  bool  [S_all]               (round labels) recorded forwards that were full
and csa_<label>.npz: csa_topk int32 [Lc, S, 8, rows, width] (-1 = no index),
csa_layers [Lc] -- the first steady steps of the window. Only steps that were
full on every rank are kept.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

EP = 8
_LAYER = re.compile(r"layers\.(\d+)\.")


def _order(name):
    m = _LAYER.search(name)
    return (int(m.group(1)) if m else 1 << 20, name)


def load(run: Path, label: str):
    dumps = []
    for r in range(EP):
        z = np.load(run / "tap" / f"rank-{r}-{label}.npz")
        meta = json.loads(bytes(z["meta_json"]).decode())
        dumps.append((z, meta))
    return dumps


def window(meta, kind, name):
    if "profile_counters" in meta:
        pc = meta["profile_counters"]
        return pc["start"][kind][name], pc["stop"][kind][name]
    return meta["counters_begin"][kind][name], meta["counters_end"][kind][name]


def full_mask(run: Path, label: str, rank: int, n: int, draft: bool = False):
    """Per recorded forward of this rank (288-row verify, or 240-row draft): did it
    carry all 48 decode requests?"""
    if not label.startswith("round"):
        return np.ones(n, bool)
    rnd = int(label[len("round") :])
    name = "warmup-steps.json" if rnd == 0 else f"round-{rnd}-steps.json"
    steps = json.loads((run / "fixed" / name).read_text())["results"]
    while steps and isinstance(steps[0], list):  # gathered per DP worker, nested
        steps = [x for item in steps for x in item]
    recs = next(x for x in steps if x["rank"] == rank)
    flags = []
    for s in recs["steps"]:
        if draft:
            st = next((g for g in s["stages"] if g["stage"] == "draft_inputs"), None)
            if st is None or st["padded_rows"] != 240:
                continue
            flags.append(st["requests"] == 48)
            continue
        enc = next((g for g in s["stages"] if g["stage"] == "encoder"), None)
        if enc is None or enc["padded_rows"] != 288:
            continue
        flags.append(enc["phase"] == "decode" and enc["requests"] == 48)
    if len(flags) > n:
        raise ValueError(
            f"rank {rank}: {len(flags)} 288-row forwards in steps.json, "
            f"only {n} recorded by the tap"
        )
    # Forwards the stage hook did not see (an idle rank's DP dummy runs) can only
    # come after this rank's last request, so they pad the tail as not-full.
    return np.array(flags + [False] * (n - len(flags)))


def ring_steps(z, m, c0, c1, rows_per_step, ring):
    """[c1-c0, rows_per_step * width/rows] in step order, from a ring dump."""
    n = c1 - c0
    if n > ring:
        raise ValueError(f"window of {n} steps overran the {ring}-step ring")
    buf = z[m["key"]]
    return np.stack(
        [
            buf[
                ((c0 + s) % ring) * rows_per_step : ((c0 + s) % ring + 1)
                * rows_per_step
            ]
            for s in range(n)
        ]
    )


def convert_moe(dumps, run, label, which):
    names = sorted((n for n, m in dumps[0][1]["moe"].items() if which(m)), key=_order)
    if not names:
        return None
    m0 = dumps[0][1]["moe"][names[0]]
    rows, topk, experts, ring = m0["rows"], m0["topk"], m0["experts"], m0["ring"]
    per_rank, masks = [], []
    for r, (z, meta) in enumerate(dumps):
        # Layers' counters differ by the warmup forwards each saw; the windows
        # must still be the same length.
        wins = {n: window(meta, "moe", n) for n in names}
        lengths = {c1 - c0 for c0, c1 in wins.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"rank {r}: layers disagree on the window length {lengths}"
            )
        n_all = lengths.pop()
        for name, (c0, c1) in wins.items():
            end = meta["counters_end"]["moe"][name]
            if end - max(c0, c1 - ring) > ring:
                raise ValueError(
                    f"rank {r} {name}: the ring was overwritten after the window "
                    f"[{c0}, {c1}) (dumped at {end}, ring {ring})"
                )
        kept = min(n_all, ring)  # a window longer than the ring keeps its last steps
        masks.append(full_mask(run, label, r, n_all, draft=rows != 288)[n_all - kept :])
        per_rank.append(
            np.stack(
                [
                    ring_steps(
                        z, meta["moe"][n], wins[n][1] - kept, wins[n][1], 1, ring
                    )[:, 0]
                    for n in names
                ]
            )
        )  # [L, kept, rows*topk]
    n = {p.shape[1] for p in per_rank}
    if len(n) != 1:
        raise ValueError(f"ranks recorded different step counts {n}")
    keep = np.logical_and.reduce(masks)
    if any(m.size < n_all for m in masks):
        print(
            f"note: window of {n_all} steps overran the {ring}-step ring; "
            f"kept the last {ring}"
        )
    ids = np.stack(per_rank, 2)[:, keep]  # [L, S, EP, rows*topk]
    L, S = ids.shape[:2]
    ids = ids.reshape(L, S, EP * rows, topk)
    counts = np.zeros((S, L, EP, experts), np.int32)
    src = np.repeat(np.arange(EP), rows * topk)
    for li in range(L):
        for s in range(S):
            np.add.at(counts[s, li], (src, ids[li, s].reshape(-1)), 1)
    layers = np.array(
        [int(_LAYER.search(n).group(1)) if _LAYER.search(n) else -1 for n in names]
    )
    return dict(
        topk_ids=ids.astype(np.int32),
        counts=counts,
        layers=layers,
        names=np.array(names),
        step_full=keep,
    )


def convert_csa(dumps, run, label):
    """The CSA rings fill once from the window start: the first `ring` 288-row steps."""
    names = sorted(dumps[0][1]["csa"], key=_order)
    m0 = dumps[0][1]["csa"][names[0]]
    rows, width, ring = m0["rows"], m0["width"], m0["ring"]
    out, masks = [], []
    for r, (z, meta) in enumerate(dumps):
        ends = {meta["counters_end"]["csa"][n] for n in names}
        n = min(min(ends), ring)
        mw = window(meta, "moe", next(iter(meta["moe"])))
        masks.append(
            full_mask(run, label, r, mw[1] - mw[0])[:n]
        )  # CSA slots are the first steps
        out.append(
            np.stack(
                [
                    z[meta["csa"][nm]["key"]][: n * rows].reshape(n, rows, width)
                    for nm in names
                ]
            )
        )  # [Lc, n, rows, width]
    S = min(len(m) for m in masks)
    keep = np.logical_and.reduce([m[:S] for m in masks])
    arr = np.stack([o[:, :S] for o in out], 2)[:, keep]  # [Lc, S, EP, rows, width]
    layers = np.array(
        [int(_LAYER.search(n).group(1)) if _LAYER.search(n) else -1 for n in names]
    )
    return dict(
        csa_topk=arr.astype(np.int32), csa_layers=layers, csa_names=np.array(names)
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--label", default="round1")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dumps = load(a.run, a.label)
    target = convert_moe(dumps, a.run, a.label, lambda m: m["experts"] == 384)
    draft = convert_moe(dumps, a.run, a.label, lambda m: m["experts"] != 384)
    out = dict(target)
    if draft is not None:
        out.update({f"draft_{k}": v for k, v in draft.items()})
    np.savez_compressed(a.out / f"routing_{a.label}.npz", **out)
    print(
        f"routing_{a.label}: topk_ids {target['topk_ids'].shape} "
        f"layers {target['layers'].tolist()} "
        f"full steps kept {int(target['step_full'].sum())}/{target['step_full'].size}"
        + (f"; draft {draft['topk_ids'].shape}" if draft else "")
    )
    if dumps[0][1]["csa"]:
        csa = convert_csa(dumps, a.run, a.label)
        np.savez_compressed(a.out / f"csa_{a.label}.npz", **csa)
        print(
            f"csa_{a.label}: {csa['csa_topk'].shape} "
            f"layers {csa['csa_layers'].tolist()}"
        )


if __name__ == "__main__":
    main()
