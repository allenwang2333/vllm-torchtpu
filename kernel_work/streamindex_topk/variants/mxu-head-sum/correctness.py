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
"""Variant F -- the standard correctness gate, run on the variant AND on the
incumbent in the same job.

`instruments/test_correctness.py` is READ-ONLY and is imported and executed
exactly as written: float64 oracle, `RANK_SLACK=256`, `MIN_RECALL=0.99`, and a
negative control that mutates the input and requires the check to go red.

Why the incumbent is re-run here rather than quoted
---------------------------------------------------
The exactness claim F rests on is a claim about a DIFFERENCE. Area 6 records
that pure reassociation of the 32-term head sum is bounded by 2.46e-06 relative
[verified j-ed517191], four decades below the 61-rank churn already present, so
an exact-by-reassociation F must land on margins BIT-IDENTICAL to the
incumbent's -- `worst_rank=2109 / min_recall=0.99561` at `decode_prod` and
`2098 / 0.99512` at `mixed_prod`. Anything else is a bug, not a tolerance. That
comparison is only meaningful against an incumbent measured on the same inputs
in the same process, so both are run.

Regimes are ordered cheapest-first and each is wrapped, because the float64
oracle at `mixed_64k` builds a [1024, 65536] f64 array (537 MB) plus a
[64, 32, 65536] f64 chunk (1.07 GB), on a host shared with three other shards.
If it dies, the regimes before it still report.

Sign convention does not apply here -- there are no times in this file.
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, __file__.rsplit("/kernel_work/", 1)[0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes",
                    default="decode_prod,mixed_prod,decode_64k,mixed_32k,"
                            "mixed_64k")
    ap.add_argument("--control-regimes", default="decode_prod,mixed_prod",
                    help="where to run the negative control. It exists to "
                         "prove the check is not vacuous, which one short and "
                         "one long regime establish; the float64 oracle costs "
                         "~550 GFLOP at mixed_64k and running it four times "
                         "there buys nothing.")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--min-bq", default="0")
    a = ap.parse_args(argv)

    os.environ["F_MXU_HEAD_SUM"] = "1"
    os.environ["F_DTYPE"] = a.dtype
    os.environ["F_MIN_BQ"] = str(a.min_bq)

    import jax  # noqa: E402
    from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402
    from kernel_work.streamindex_topk.instruments import test_correctness as TC  # noqa: E402
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as INCUMBENT  # noqa: E402

    HERE = os.path.dirname(os.path.abspath(__file__))
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
    print(f"gate: RANK_SLACK={TC.RANK_SLACK} MIN_RECALL={TC.MIN_RECALL} "
          f"k={S.INDEX_TOPK}, allowed worst_rank={S.INDEX_TOPK + TC.RANK_SLACK}",
          flush=True)
    print(f"NUM_KV_PAGES_PER_BLOCK={S.NUM_KV_PAGES_PER_BLOCK}", flush=True)

    rc = 0
    margins = {}
    ctrl_regimes = {n for n in a.control_regimes.split(",") if n}
    for name in [n for n in a.regimes.split(",") if n]:
        regime = S.REGIMES_BY_NAME[name]
        for label, impl in (("incumbent", INCUMBENT), ("variant", VARIANT)):
            for control in ((False, True) if name in ctrl_regimes
                            else (False, )):
                tag = f"[{name} {label}{' CONTROL' if control else ''}]"
                try:
                    n, failures, st = TC.run(impl, regime,
                                             negative_control=control)
                except Exception as exc:  # noqa: BLE001
                    print(f"{tag} ERROR {exc!r}", flush=True)
                    traceback.print_exc()
                    rc = 1
                    continue
                margin = (f"worst_rank={st['worst_rank_seen']} of allowed "
                          f"{S.INDEX_TOPK + TC.RANK_SLACK}, min_recall="
                          f"{st['min_recall']:.5f} of required {TC.MIN_RECALL}")
                if control:
                    if failures:
                        print(f"{tag} NEGATIVE CONTROL OK -- fired on "
                              f"{len(failures)} of {n} rows. {margin}",
                              flush=True)
                    else:
                        print(f"{tag} NEGATIVE CONTROL DID NOT FIRE over {n} "
                              f"rows ({margin}). The check is vacuous and must "
                              f"not be quoted as evidence.", flush=True)
                        rc = 1
                else:
                    margins[(name, label)] = (st["worst_rank_seen"],
                                              st["min_recall"])
                    if failures:
                        print(f"{tag} FAIL -- {len(failures)} of {n} rows bad "
                              f"({margin})", flush=True)
                        for f in failures[:5]:
                            print("    ", f, flush=True)
                        rc = 1
                    else:
                        print(f"{tag} PASS -- all {n} rows a valid top-"
                              f"{S.INDEX_TOPK} vs the float64 oracle. {margin}",
                              flush=True)

    print("\n=== bit-identical margins? ===")
    print("An EXACT reassociation must land on the incumbent's margins "
          "exactly. A difference is a bug, not a tolerance.")
    for name in [n for n in a.regimes.split(",") if n]:
        i, v = margins.get((name, "incumbent")), margins.get((name, "variant"))
        if not i or not v:
            print(f"   {name:<12} incomplete")
            continue
        same = (i == v)
        print(f"   {name:<12} incumbent worst_rank={i[0]} min_recall="
              f"{i[1]:.5f}   variant worst_rank={v[0]} min_recall={v[1]:.5f}"
              f"   {'IDENTICAL' if same else 'DIFFERENT -- F is not exact'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
