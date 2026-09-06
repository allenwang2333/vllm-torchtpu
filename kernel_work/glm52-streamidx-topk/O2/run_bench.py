"""O2 A/B: incumbent vs O2 tiles, same job, same instruments, same inputs.

    python3 run_bench.py --rows B-D1-1M
    python3 run_bench.py --rows A-D64-8k,A-D16-8k

Every arm is timed by `instruments/bench.interleaved` -- adjacently, five
repeats, differenced INSIDE each repeat -- because run-order drift at these
shapes is 3-9%, larger than the effect several of these arms are meant to
resolve. The incumbent is one of the arms, not a number from another job.

Arms at regime (B), page_size 512, all with the design's (1024, 4) MIXED tile:
    incumbent        bkv_p=(2,2,2)    bq_sz=(1,128,128)   -- ref
    O2-heur          whatever `pallas_mosaic_tpu` picks after VMEM screening
    O2-dec16         bkv_p=(16,2,4)   -- where P-ITER's sweep stopped
    O2-dec32         bkv_p=(32,2,4)
    O2-dec64         bkv_p=(64,2,4)   -- expected over budget; the compiler decides

An earlier revision of this file swept the MIXED tile over (1024,4), (512,2)
and (256,2) instead. `measured (j-980e8710-2ee6-4441-a9f3)` those three landed
at -0.8%, -0.8% and -0.7% at B-P1024-256k against 0.4 pt spreads, i.e.
indistinguishable, so the MIXED tile is now pinned and `bkv_p` is the swept
knob. That job also settled the VMEM question: `tiling.einsum_temp_bytes`
prices the (1024,4) tile at 289.25 MiB against 64.0 MiB of VMEM and it
compiled and ran correctly anyway, so the einsum-temp term is an upper bound
on something Mosaic tiles rather than an occupancy.

Arms at regime (A), page_size 1024:
    incumbent        bkv_p=(2,2,2)                        -- ref
    O2-heur          must resolve to the incumbent; that IS the gating test
    O2-WRONG-bkv16   bkv_p=(16,2,2) forced past the heuristic

The last one is deliberate. Section 9.2 prices a globally-applied bkv_p=16 at
+1.7% (slower) at (A) and tags it `modelled`; it is the whole reason O2 is
regime-conditional. Forcing it here turns that prediction into a measurement
instead of leaving the (A) row as an assertion -- and the measurement is much
worse than the model: `measured (j-980e8710-2ee6-4441-a9f3)` +19.1% at
A-D64-8k and +15.2% at A-D16-8k, against a 7.1 pt / 5.9 pt spread. The gating
is load-bearing by an order of magnitude more than the design credited.

Nothing in `instruments/` is edited or copied. `shapes.build` supplies the
inputs, which are built ONCE per row and shared by every arm.
"""
import argparse
import functools
import json
import os
import sys

import jax
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.environ.get(
    "INST_DIR", os.path.join(_HERE, "..", "instruments")))

import api  # noqa: E402
import pallas_mosaic_tpu as policy  # noqa: E402
import tiling  # noqa: E402
from bench import interleaved  # noqa: E402
from shapes import K, ROWS, build  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    streamindex_topk as incumbent_topk  # noqa: E402

HEUR = "heuristic"


def _arm_specs(page_size):
    if page_size == 512:
        # The MIXED tile is pinned at the design's (1024, 4) in every O2 arm
        # here, because j-980e8710 settled it: all three MIXED tiles landed
        # within 0.2 pt of each other at B-P1024-256k against 0.4 pt spreads,
        # so it is not the variable. `bkv_p` is, and this is its sweep.
        mixed = dict(bq=1024, bkv=4)
        return [
            ("incumbent", None),
            ("O2-heur", HEUR),
            ("O2-dec16", policy.Config(bkv_p=(16, 2, mixed["bkv"]),
                                       bq_sz=(1, 128, mixed["bq"]))),
            ("O2-dec32", policy.Config(bkv_p=(32, 2, mixed["bkv"]),
                                       bq_sz=(1, 128, mixed["bq"]))),
            ("O2-dec64", policy.Config(bkv_p=(64, 2, mixed["bkv"]),
                                       bq_sz=(1, 128, mixed["bq"]))),
        ]
    if page_size == 1024:
        return [
            ("incumbent", None),
            ("O2-heur", HEUR),
            ("O2-WRONG-bkv16",
             policy.Config(bkv_p=(16, 2, 2), bq_sz=(1, 128, 128))),
        ]
    raise SystemExit(f"no arm set defined for page_size={page_size}")


def _describe(cfg, meta):
    """What this tile costs, before the compiler is asked."""
    psz, pps = meta["psz"], meta["pages_per_seq"]
    n = tiling.padded_n(psz, pps, cfg.bkv_p)
    rep = policy.vmem_report(cfg, page_size=psz, kv_packing=4, lkv_dim=256,
                             num_q_heads=32, head_dim=128, q_itemsize=1,
                             decode_req_batch_size=4)
    peak = max(p["total"] for p in rep.values())
    qlen = meta.get("qlen", 1)
    slot = policy.MIXED if meta["D"] == 0 else policy.DECODE
    iters = tiling.kv_iterations(meta["seqlen"], psz, cfg.bkv_p[slot],
                                 qlen, cfg.bq_sz[slot])
    return dict(n=n, peak_vmem_MiB=peak / 2**20, in_body_iters=iters,
                per_launch_MiB={k: v["total"] / 2**20 for k, v in rep.items()})


def _selected(idx, rows=4):
    """Per-row selected index sets, for cross-arm agreement.

    Padded token rows come back all -1 and must be dropped, not scored: a row
    with nothing in it agrees with nothing, and averaging that in reads as a
    correctness failure. At B-D1-1M only 1 of the first 4 rows is live, which
    is how j-5136b90a reported 0.2500 agreement for arms that in fact selected
    identically.
    """
    a = np.asarray(idx[:rows])
    out = [set(int(v) for v in r if v >= 0) for r in a]
    return [s for s in out if s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    results = []
    for name in [s.strip() for s in a.rows.split(",")]:
        if name not in ROWS:
            raise SystemExit(f"unknown row {name!r}")
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}", flush=True)
        args, kwargs, meta = build(name)
        jax.block_until_ready(args[2])
        psz = meta["psz"]
        print(f"  psz={psz} pps={meta['pages_per_seq']} T={meta['T']} "
              f"R={meta['R']} D={meta['D']} qlen={meta.get('qlen', 1)} "
              f"seqlen={meta['seqlen']}  n(incumbent)={meta['n']}  "
              f"cache={meta['cache_bytes'] / 2**20:.0f}MiB", flush=True)

        # Strip the tile knobs the O2 arms are supposed to override; keep
        # everything else the instrument passes, verbatim.
        o2_kwargs = {k: v for k, v in kwargs.items()
                     if k not in ("num_kv_pages_per_block",
                                  "num_queries_per_block")}

        arms, meta_by_arm, ref_sel = [], {}, None
        for arm_name, spec in _arm_specs(psz):
            if spec is None:
                fn = jax.jit(lambda *x: incumbent_topk(*x, **kwargs))
                cfg = policy.Config(bkv_p=kwargs["num_kv_pages_per_block"],
                                    bq_sz=kwargs["num_queries_per_block"])
            elif spec is HEUR:
                fn = jax.jit(lambda *x: api.streamindex_topk(*x, **o2_kwargs))
                cfg = None
            else:
                cfg = spec
                fn = jax.jit(
                    lambda *x, _c=spec: api.streamindex_topk(
                        *x, **o2_kwargs, force_config=_c))

            if cfg is not None:
                d = _describe(cfg, meta)
                print(f"\n  -- {arm_name}: bkv_p={cfg.bkv_p} bq_sz={cfg.bq_sz}"
                      f"  n={d['n']}  peak VMEM {d['peak_vmem_MiB']:.2f} MiB"
                      f"  in-body iters {d['in_body_iters']}", flush=True)
                print(f"     per launch MiB: " + "  ".join(
                    f"{k}={v:.2f}" for k, v in d["per_launch_MiB"].items()),
                    flush=True)
            else:
                d = None
                print(f"\n  -- {arm_name}: (heuristic)", flush=True)

            try:
                idx = fn(*args)
                jax.block_until_ready(idx)
            except Exception as e:  # noqa: BLE001 -- a refused tile is a result
                head = str(e).strip().splitlines()
                print(f"     REFUSED: {type(e).__name__}: "
                      f"{' | '.join(head[:4])}", flush=True)
                meta_by_arm[arm_name] = dict(cfg=None if cfg is None
                                             else dict(bkv_p=list(cfg.bkv_p),
                                                       bq_sz=list(cfg.bq_sz)),
                                             describe=d, refused=str(e)[:2000])
                continue

            sel = _selected(idx)
            if bool((np.asarray(idx[0]) == np.arange(K)).all()):
                raise SystemExit(f"{name}/{arm_name}: took _fast_path "
                                 "-- measures nothing")
            if not sel:
                raise SystemExit(f"{name}/{arm_name}: no live token rows in "
                                 "the first 4 -- the agreement check is blind")
            if ref_sel is None:
                ref_sel = sel
                agree = 1.0
            else:
                agree = float(np.mean([len(x & y) / len(y)
                                       for x, y in zip(sel, ref_sel)]))
            print(f"     ok. {len(sel)} live rows checked, filled row0 "
                  f"{len(sel[0])}/{K}, set agreement vs incumbent "
                  f"{agree:.4f}", flush=True)

            # `bench.interleaved` calls arms as zero-argument thunks
            # (bench.py:71-72), so the inputs are bound HERE, not passed there.
            arms.append((arm_name, functools.partial(fn, *args)))
            meta_by_arm[arm_name] = dict(
                cfg=None if cfg is None else dict(bkv_p=list(cfg.bkv_p),
                                                  bq_sz=list(cfg.bq_sz)),
                describe=d, agreement=agree)

        if len(arms) < 2:
            print("  only one arm survived compilation; nothing to difference.",
                  flush=True)
            results.append(dict(row=name, arms=meta_by_arm, timing=None))
            continue

        print(f"\n  timing {len(arms)} arms, {a.repeats} repeats x {a.iters} "
              f"iters, paired vs incumbent", flush=True)
        timing = interleaved(arms, repeats=a.repeats, iters=a.iters,
                             ref="incumbent")
        results.append(dict(row=name, arms=meta_by_arm,
                            timing={k: v for k, v in timing.items()}))

    print(f"\n{'=' * 78}")
    print(f"{'row':16} {'arm':18} {'med us':>10} {'d time %':>10} "
          f"{'d spread':>10}")
    for r in results:
        for arm, t in (r["timing"] or {}).items():
            dp = "--" if t["delta_pct"] is None else f"{t['delta_pct']:+.1f}"
            ds = ("--" if t["delta_spread_pct"] is None
                  else f"{t['delta_spread_pct']:.1f}pt")
            print(f"{r['row']:16} {arm:18} {t['median']:10.1f} {dp:>10} "
                  f"{ds:>10}")
        for arm, m in r["arms"].items():
            if m.get("refused"):
                print(f"{r['row']:16} {arm:18} {'REFUSED':>10}")
    print("  (negative d time % is faster)")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
