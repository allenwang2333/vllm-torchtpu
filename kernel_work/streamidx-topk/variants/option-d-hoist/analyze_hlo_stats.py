"""Bucket an `hlo_stats` dump into R2's partition and check it is one.

R2's table is a *verified partition*: every TensorCore row of the `hlo_stats`
table is assigned to exactly one bucket and `sum(buckets) - TC total` is
asserted to be 0. That is what makes it a decomposition of the call rather than
a selection of the interesting rows, and it is reproduced here so an Option D
before/after table can be read the same way.

Two rules carried over from R2 and enforced, not assumed:

  * `rows_kept == rows_total`. R2's first dump silently truncated at
    `--top-n 60`; a truncated table looks like a complete one.
  * The SparseCore-plane rows are **excluded, not added**. `sc_topk_direct…
    call-done` appears on both planes and is the same asynchronous operation;
    only the TensorCore charge is what the critical path pays.

Run:

    python analyze_hlo_stats.py --dump pre21=/path/a.json --dump d21=/path/b.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# Ordered; first match wins. Matched against "<HLO op name> || <framework op name>".
BUCKETS: tuple[tuple[str, re.Pattern], ...] = (
    ("stage_a", re.compile(r"StreamIdxTC-")),
    ("stage_b_kernel", re.compile(r"sc_topk_direct|sparsecore_topk/sc_topk")),
    ("cond", re.compile(r"^cond[.\d]*\.clone|\bconditional\b")),
    ("predicate", re.compile(r"reduce_max")),
    ("stage_b_shaping",
     re.compile(r"bitcast_convert_type|sparsecore_topk\)?/reshape|^reshape")),
    # `-inf` fill of the donated scores buffer. The incumbent emits it as
    # `broadcast.13.scores_in_hbm_ref`; with the `cond` gone XLA names the same
    # op `broadcast_in_dim.<n>.scores_in_hbm_ref[.<n>.clone.<n>]`. Both spellings
    # must land in this bucket or the D arms show a phantom `init_neg_inf`
    # saving and an equal phantom `metadata` regression.
    ("init_neg_inf",
     re.compile(r"broadcast(_in_dim)?[.\d]*\.scores_in_hbm_ref")),
    ("prefetch_c3",
     re.compile(r"start_end_seq_idx_ref|sem_ids_ref|bo_sz_ref")),
    ("weights_layout",
     re.compile(r"\|\|\s*indexer_weights|/convert_element_type")),
    ("boundary_copy", re.compile(r"^copy-(start|done)|^copy[.\d]*\s*\|\|\s*$")),
)
RESIDUAL_BUCKETS = tuple(b for b, _ in BUCKETS if b
                         not in ("stage_a", "stage_b_kernel")) + ("metadata", )


def classify(name: str, fw: str, category: str) -> str:
    blob = f"{name} || {fw}"
    if category == "conditional":
        return "cond"
    if category in ("copy-start", "copy-done"):
        # A copy is only boundary plumbing if it is not something else's copy.
        for bucket, pat in BUCKETS:
            if bucket in ("stage_a", "stage_b_kernel") and pat.search(blob):
                return bucket
        return "boundary_copy"
    for bucket, pat in BUCKETS:
        if pat.search(blob):
            return bucket
    return "metadata"


def load(path: str) -> dict:
    with open(path) as fh:
        doc = json.load(fh)
    tool = doc["tools"]["hlo_stats"]
    if tool.get("status") != "ok":
        raise SystemExit(f"{path}: hlo_stats status={tool.get('status')}")
    data = tool["data"]
    if data["rows_kept"] != data["rows_total"]:
        raise SystemExit(f"{path}: TRUNCATED, "
                         f"rows_kept={data['rows_kept']} "
                         f"rows_total={data['rows_total']} -- re-dump with a "
                         f"larger --top-n; a truncated table is not a "
                         f"partition")
    return data


def summarize(label: str, path: str, layers: int) -> dict:
    data = load(path)
    tc, sc = [], []
    for r in data["rows"]:
        (tc if r["TPU core type"] == "TensorCore" else sc).append(r)

    per_bucket: dict[str, float] = {}
    items: dict[str, list] = {}
    for r in tc:
        b = classify(r["HLO op name"], r["Framework op name"] or "",
                     r["HLO op category"])
        per_bucket[b] = per_bucket.get(b, 0.0) + r["Avg. self time (us)"]
        items.setdefault(b, []).append(
            (r["Avg. self time (us)"], r["HLO op name"],
             (r["Framework op name"] or "")[:80]))

    tc_total = sum(r["Avg. self time (us)"] for r in tc)
    bucket_sum = sum(per_bucket.values())
    check = bucket_sum - tc_total
    residual = sum(v for k, v in per_bucket.items() if k in RESIDUAL_BUCKETS)
    return {
        "label": label,
        "path": path,
        "layers": layers,
        "rows_total": data["rows_total"],
        "n_tc_rows": len(tc),
        "n_sc_rows": len(sc),
        "sc_excluded_us": sum(r["Avg. self time (us)"] for r in sc),
        "tc_total_us_per_prog": tc_total,
        "tc_total_us_per_layer": tc_total / layers,
        "bucket_sum_minus_tc_total": check,
        "buckets_us_per_prog": per_bucket,
        "buckets_us_per_layer": {k: v / layers for k, v in per_bucket.items()},
        "residual_us_per_prog": residual,
        "residual_us_per_layer": residual / layers,
        "items": {k: sorted(v, reverse=True) for k, v in items.items()},
    }


ORDER = ("stage_a", "stage_b_kernel", "cond", "predicate", "metadata",
         "prefetch_c3", "stage_b_shaping", "init_neg_inf", "weights_layout",
         "boundary_copy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True,
                    metavar="LABEL=PATH[:LAYERS]",
                    help="LABEL=PATH; layers inferred from a trailing 21/1 in "
                         "LABEL unless given as LABEL=PATH:LAYERS")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--items", action="store_true",
                    help="print every row under its bucket")
    args = ap.parse_args()

    summaries = []
    for spec in args.dump:
        label, _, rest = spec.partition("=")
        path, _, lay = rest.partition(":")
        layers = int(lay) if lay else (21 if label.endswith("21") else 1)
        summaries.append(summarize(label, path, layers))

    labels = [s["label"] for s in summaries]
    w = max(14, max(len(l) for l in labels) + 2)

    print("\n### Device self time per bucket, us per LAYER "
          "(hlo_stats, TensorCore plane)\n")
    print("| bucket | " + " | ".join(f"{l}" for l in labels) + " |")
    print("|---|" + "---:|" * len(labels))
    keys = list(ORDER) + sorted(
        {k for s in summaries for k in s["buckets_us_per_layer"]}
        - set(ORDER))
    for k in keys:
        vals = [s["buckets_us_per_layer"].get(k, 0.0) for s in summaries]
        if not any(abs(v) > 1e-9 for v in vals):
            continue
        print(f"| `{k}` | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    print("| **TC total** | " +
          " | ".join(f"**{s['tc_total_us_per_layer']:.3f}**"
                     for s in summaries) + " |")
    print("| _residual (TC total - A - B)_ | " +
          " | ".join(f"{s['residual_us_per_layer']:.3f}" for s in summaries) +
          " |")

    print("\n### Partition check  (bucket sum - TC total, must be 0)\n")
    for s in summaries:
        print(f"  {s['label']:<{w}} {s['bucket_sum_minus_tc_total']:+.6f} us"
              f"   rows={s['rows_total']} (TC {s['n_tc_rows']}, "
              f"SC {s['n_sc_rows']} excluded, {s['sc_excluded_us']:.3f} us)"
              f"   layers={s['layers']}"
              f"   prog total={s['tc_total_us_per_prog']:.3f} us")

    if args.items:
        for s in summaries:
            print(f"\n### {s['label']} -- every TensorCore row, by bucket\n")
            for k in keys:
                if k not in s["items"]:
                    continue
                tot = s["buckets_us_per_prog"][k]
                print(f"  [{k}] {tot:.3f} us/prog")
                for us, name, fw in s["items"][k]:
                    print(f"      {us:9.3f}  {name[:64]:64} | {fw}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(summaries, fh, indent=1)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
