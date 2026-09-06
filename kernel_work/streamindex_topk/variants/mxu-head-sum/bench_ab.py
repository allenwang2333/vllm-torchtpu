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
"""Variant F -- incumbent vs MXU head sum, ONE job, same instruments.

Three arms per regime, in this order:

    incumbent          `vllm_torchtpu.kernels.deepseek_v4.streamindex_topk`,
                       the shipped module -- the denominator of every delta
    variant            `streamindex_topk_mxu`, F_MXU_HEAD_SUM=1
    incumbent-repeat   the DRIFT CONTROL: the shipped module again, last

The drift control is not optional and is not padding. Variant E's job
[measured (j-ce9a19a4-4639-4a8f-9059)] reversed a conclusion on exactly this:
within-arm spread said "free at mixed", and a repeated incumbent turned it into
a real ~2% cost. Spread inside one arm does not bound drift ACROSS arms in one
process -- compile order, allocator state and host temperature all move between
them. Any variant delta smaller than the incumbent-to-incumbent-repeat gap is
not readable as an effect.

The incumbent arm is the SHIPPED module, not `streamindex_topk_mxu` with
F_MXU_HEAD_SUM=0. The two should be identical and the diff says so, but "should
be" is not a denominator.

`instruments/` is READ-ONLY. This file imports `microbench.run_regime` and
executes it exactly as written; the only thing it varies is which module is
passed as `impl`.

`early_exit=False`, matching step 1's baseline table and every arm measured so
far. Production sets it True (`mla_attention_op.py:57`); it is held constant
across arms either way.

The `topk` arm is measured ONCE per regime, on the first incumbent pass only.
It is `sparsecore_topk` on a synthetic scores buffer of the same width and does
not depend on which scoring implementation ran, so measuring it three times
would triple the job for no information. It is a CONTROL and is reported only
as the subtrahend of `full - topk`.

Sign convention: every percentage is a change in TIME, (new - old)/old.
Negative is faster.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("/kernel_work/", 1)[0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes",
                    default="decode_prod,decode_64k,mixed_prod,mixed_64k")
    ap.add_argument("--dtype", default="bf16",
                    help="F_DTYPE for the variant: bf16|f32|f32_highest|fp8")
    ap.add_argument("--min-bq", default="0", help="F_MIN_BQ gate")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    # Must precede the variant import -- the knobs are read at import time.
    os.environ["F_MXU_HEAD_SUM"] = "1"
    os.environ["F_DTYPE"] = a.dtype
    os.environ["F_MIN_BQ"] = str(a.min_bq)

    import jax  # noqa: E402
    from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402
    from kernel_work.streamindex_topk.instruments import microbench as MB  # noqa: E402
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as INCUMBENT  # noqa: E402

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    import importlib.util  # noqa: E402
    spec = importlib.util.spec_from_file_location(
        "streamindex_topk_mxu", os.path.join(HERE, "streamindex_topk_mxu.py"))
    VARIANT = importlib.util.module_from_spec(spec)
    sys.modules["streamindex_topk_mxu"] = VARIANT
    spec.loader.exec_module(VARIANT)

    print(f"jax {jax.__version__} devices {jax.devices()}", flush=True)
    print(f"variant knobs: F_MXU_HEAD_SUM={VARIANT.F_MXU_HEAD_SUM} "
          f"F_DTYPE={VARIANT._F_DTYPE_NAME} -> {VARIANT.F_DTYPE.__name__} "
          f"precision={VARIANT.F_PRECISION} F_MIN_BQ={VARIANT.F_MIN_BQ}",
          flush=True)
    print(f"NUM_KV_PAGES_PER_BLOCK={S.NUM_KV_PAGES_PER_BLOCK} "
          f"NUM_QUERIES_PER_BLOCK={S.NUM_QUERIES_PER_BLOCK} "
          f"DECODE_REQ_BATCH_SIZE={S.DECODE_REQ_BATCH_SIZE}", flush=True)

    arms = [("incumbent", INCUMBENT, True),
            ("variant", VARIANT, False),
            ("incumbent-repeat", INCUMBENT, False)]

    rows = []
    for name in [n for n in a.regimes.split(",") if n]:
        regime = S.REGIMES_BY_NAME[name]
        for label, impl, want_topk in arms:
            rec = dict(regime=name, arm=label)
            try:
                row = MB.run_regime(impl, regime, early_exit=False,
                                    reps=a.reps, measure_topk=want_topk)
                rec.update(row)
            except Exception as exc:  # noqa: BLE001
                rec["error"] = repr(exc)[:900]
            rec["arm"] = label
            rec["regime"] = name
            rows.append(rec)
            print(json.dumps(rec), flush=True)

    summarise(rows)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 0


def summarise(rows):
    print("\n=== four-regime A/B: median ms per `streamindex_topk` call, "
          "single TensorCore ===")
    print("`Delta time %` = (new - old)/old, and NEGATIVE IS FASTER. Every "
          "delta sits in the same row as the two absolute times it came from.")
    hdr = (f"{'regime':<12}{'arm':<18}{'full_ms':>14}{'spread%':>9}"
           f"{'baseline_ms':>14}{'Delta time %':>14}{'speedup x':>11}")
    print(hdr)
    print("-" * len(hdr))
    base = {}
    for r in rows:
        if "error" in r:
            print(f"{r['regime']:<12}{r['arm']:<18}   ERROR {r['error'][:100]}")
            continue
        if r["arm"] == "incumbent":
            base[r["regime"]] = r["full_ms"]
        b = base.get(r["regime"])
        if b:
            d = (r["full_ms"] - b) / b * 100.0
            ds, ss = f"{d:+.2f}", f"{b / r['full_ms']:.3f}"
        else:
            ds, ss = "-", "-"
        print(f"{r['regime']:<12}{r['arm']:<18}{r['full_ms']:>14.8f}"
              f"{r['full_spread_pct']:>9.2f}{(b if b else float('nan')):>14.8f}"
              f"{ds:>14}{ss:>11}")

    print("\n=== drift control: is any of that readable? ===")
    for reg, b in base.items():
        rep = next((r for r in rows
                    if r["regime"] == reg and r["arm"] == "incumbent-repeat"
                    and "error" not in r), None)
        var = next((r for r in rows
                    if r["regime"] == reg and r["arm"] == "variant"
                    and "error" not in r), None)
        if not rep:
            print(f"   {reg:<12} drift control missing")
            continue
        drift = (rep["full_ms"] - b) / b * 100.0
        line = (f"   {reg:<12} incumbent {b:.8f} ms -> incumbent-repeat "
                f"{rep['full_ms']:.8f} ms, Delta time {drift:+.2f}%")
        if var:
            eff = (var["full_ms"] - b) / b * 100.0
            verdict = ("READABLE" if abs(eff) > abs(drift)
                       else "NOT READABLE -- within drift")
            line += (f"   | variant {eff:+.2f}% -> {verdict}")
        print(line)

    print("\n=== control arm: `sparsecore_topk` alone (subtrahend, not a "
          "result) ===")
    for r in rows:
        if "error" in r or "_control_topk_only_ms" not in r:
            continue
        print(f"   {r['regime']:<12} topk-only CONTROL "
              f"{r['_control_topk_only_ms']:.8f} ms "
              f"(spread {r['_control_topk_spread_pct']:.2f}%), so the "
              f"incumbent's scoring-by-difference is "
              f"{r['scoring_ms_by_difference']:.8f} ms")
    print("   `report.md` marks the long-KV scoring-by-difference figures "
          "UNUSABLE (133.1% spread on the control they came from). Printed for "
          "re-derivation only.")


if __name__ == "__main__":
    sys.exit(main())
