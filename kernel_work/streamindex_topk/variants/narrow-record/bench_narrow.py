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
"""A/B driver for variant A ("Narrow Record"): incumbent vs variant, one job.

Both arms are timed by the READ-ONLY `instruments/microbench.py`, in the same
process, on the same shapes, back to back. The variant arm differs only in
which `streamindex_topk` is called and in `inputs.build` being rebound to the
splitting builder -- the timing code, the warmup, the rep count and the
in-flight budget are the instrument's, unmodified.

Arm order per regime: incumbent, variant, incumbent again. The third arm is a
DRIFT CONTROL, not a result: if the two incumbent measurements of the same
regime disagree by more than their own spread, the machine moved under the
experiment and the A/B delta for that regime cannot be read.

Sign convention (binding): `delta_time_pct = (variant - incumbent) /
incumbent * 100`. Negative is faster.

Usage:
    python3 .../bench_narrow.py --regimes decode_prod,decode_64k
"""

import argparse
import json
import sys

import _load  # noqa: F401  -- puts the repo root on sys.path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    import jax

    from kernel_work.streamindex_topk.instruments import microbench as MB
    from kernel_work.streamindex_topk.instruments import shapes as S
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as incumbent

    narrow_inputs = _load.load("narrow_inputs")
    variant = _load.load("streamindex_topk_narrow")

    import hashlib
    print("jax", jax.__version__, "devices", jax.devices(), flush=True)
    for label, mod in (("incumbent", incumbent), ("variant", variant)):
        h = hashlib.md5(open(mod.__file__, "rb").read()).hexdigest()
        print(f"{label:<10} module: {mod.__file__}  md5={h}", flush=True)

    rows = []
    for name in [n for n in a.regimes.split(",") if n]:
        regime = S.REGIMES_BY_NAME[name]

        def timed(label, impl, patched, measure_topk):
            try:
                if patched:
                    with narrow_inputs.patched_inputs():
                        r = MB.run_regime(impl, regime, early_exit=False,
                                          reps=a.reps,
                                          measure_topk=measure_topk)
                else:
                    r = MB.run_regime(impl, regime, early_exit=False,
                                      reps=a.reps, measure_topk=measure_topk)
            except Exception as exc:  # noqa: BLE001
                r = dict(regime=name, error=repr(exc)[:2000])
            r["arm"] = label
            rows.append(r)
            print(json.dumps(r), flush=True)
            return r

        # Report the shapes the split arm will actually see, once, so the
        # RESULTS.md byte accounting can be checked against the job log rather
        # than against this file.
        d = narrow_inputs.build(regime)
        cv, cs = d["cache_kv"]
        print(f"[{name}] split cache: val {cv.shape} {cv.dtype} "
              f"({cv.size/2**20:.1f} MiB), scale {cs.shape} {cs.dtype} "
              f"({cs.size/2**20:.1f} MiB)", flush=True)
        del d, cv, cs

        base = timed("incumbent", incumbent, False, True)
        var = timed("variant", variant, True, True)
        base2 = timed("incumbent_repeat", incumbent, False, False)

        if "error" in base or "error" in var:
            continue
        delta = (var["full_ms"] - base["full_ms"]) / base["full_ms"] * 100.0
        speedup = base["full_ms"] / var["full_ms"]
        drift = (float("nan") if "error" in base2 else
                 (base2["full_ms"] - base["full_ms"]) / base["full_ms"] * 100.0)
        summary = dict(kind="ab", regime=name,
                       incumbent_ms=base["full_ms"],
                       incumbent_spread_pct=base["full_spread_pct"],
                       variant_ms=var["full_ms"],
                       variant_spread_pct=var["full_spread_pct"],
                       delta_time_pct=delta, speedup_x=speedup,
                       incumbent_repeat_ms=base2.get("full_ms"),
                       drift_pct=drift,
                       incumbent_scoring_ms=base.get("scoring_ms_by_difference"),
                       variant_scoring_ms=var.get("scoring_ms_by_difference"),
                       incumbent_topk_ms=base.get("_control_topk_only_ms"),
                       variant_topk_ms=var.get("_control_topk_only_ms"),
                       inner=base.get("inner"), reps=a.reps)
        rows.append(summary)
        print(json.dumps(summary), flush=True)

    print("\n=== A/B: full call, median ms, single TensorCore ===")
    print("delta_time% = (variant - incumbent)/incumbent; NEGATIVE IS FASTER")
    hdr = (f"{'regime':<13}{'incumb_ms':>11}{'sprd%':>8}{'var_ms':>11}"
           f"{'sprd%':>8}{'dtime%':>9}{'speedup':>9}{'drift%':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("kind") != "ab":
            continue
        print(f"{r['regime']:<13}{r['incumbent_ms']:>11.3f}"
              f"{r['incumbent_spread_pct']:>8.2f}{r['variant_ms']:>11.3f}"
              f"{r['variant_spread_pct']:>8.2f}{r['delta_time_pct']:>9.2f}"
              f"{r['speedup_x']:>9.3f}{r['drift_pct']:>8.2f}")

    for r in rows:
        if "error" in r:
            print(f"\nERROR arm={r.get('arm')} regime={r['regime']}:\n"
                  f"{r['error']}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
