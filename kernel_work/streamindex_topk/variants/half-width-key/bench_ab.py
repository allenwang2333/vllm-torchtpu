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
"""A/B driver for the half-width-key arms. Incumbent and variant, same job.

The yardstick is `instruments/microbench.py`, used as a library
(`run_regime`) and not forked -- every number below comes out of the same
timing loop, the same `inputs.build`, the same `_inner_for` in-flight budget
that the step-1 baselines came out of. This file only decides which impls to
call and in what order.

Ordering: for each regime, the arms run back to back before moving on, so a
slow drift in the machine shows up in every arm of a row rather than in one
arm of the sweep.

Sign convention, binding: every percentage is a change in TIME,
`(new - old) / old`. Negative is faster.

The `cast` row is not an implementation. It times, in isolation, the
`bf16[T, S, 128] -> f32` widen that the `bf16out` arm pays and that option C
would not pay, so the arm's gap can be decomposed:

    t(bf16out) - t(incumbent) - t(cast) = t(TC half write) - t(TC full write)

i.e. the left side, negated, is C's TensorCore-side prize, measured rather
than inferred from a bytes-are-linear assumption.
"""

import argparse
import json
import os
import statistics
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, "/workspace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kernel_work.streamindex_topk.instruments import microbench as M  # noqa: E402,E501
from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402

import vllm_torchtpu.kernels.deepseek_v4.streamindex_topk as INCUMBENT  # noqa: E402,E501

REGIMES = ["decode_prod", "decode_64k", "mixed_prod", "mixed_64k"]


def _impls(names):
    out = []
    for n in names:
        if n == "incumbent":
            out.append(("incumbent", INCUMBENT))
        elif n == "bf16out":
            import impl_bf16out
            out.append(("bf16out", impl_bf16out))
        else:
            raise SystemExit(f"unknown impl {n!r}")
    return out


def time_cast(regime, reps=M.REPS):
    """Isolate the bf16 -> f32 widen at this regime's scores shape."""
    T = regime.num_tokens
    st = S.num_sublanes_total(regime.max_model_len)
    x = jnp.zeros((T, st, 128), jnp.bfloat16)
    fn = jax.jit(lambda a: a.astype(jnp.float32))
    inner = M._inner_for(regime)
    ms, spread = M._median_spread(M._time(fn, x, reps=reps, inner=inner))
    return ms, spread


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--impls", default="incumbent,bf16out")
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--reps", type=int, default=M.REPS)
    ap.add_argument("--no-cast-arm", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    impls = _impls([x for x in a.impls.split(",") if x])
    names = [x for x in a.regimes.split(",") if x]

    rows = []
    for name in names:
        regime = S.REGIMES_BY_NAME[name]
        for label, mod in impls:
            t0 = time.perf_counter()
            try:
                r = M.run_regime(mod, regime, early_exit=False, reps=a.reps,
                                 measure_topk=False)
            except Exception as exc:  # noqa: BLE001
                r = dict(regime=name, error=repr(exc)[:600])
            r["impl"] = label
            r["wall_s"] = round(time.perf_counter() - t0, 1)
            rows.append(r)
            print(json.dumps(r), flush=True)
        if not a.no_cast_arm:
            ms, spread = time_cast(regime, reps=a.reps)
            r = dict(regime=name, impl="cast_bf16_to_f32", full_ms=ms,
                     full_spread_pct=spread,
                     scores_MB=S.scores_bytes(regime.num_tokens,
                                              regime.max_model_len) / 1e6)
            rows.append(r)
            print(json.dumps(r), flush=True)

    print("\n=== half-width-key A/B (median ms/call, single TensorCore) ===")
    print("negative Delta time % is faster; every delta sits in the same row "
          "as the two absolute times it came from")
    hdr = (f"{'regime':<13}{'arm':<19}{'scoresMB':>9}{'ms':>10}{'spread%':>9}"
           f"{'base_ms':>10}{'D time %':>10}{'speedup x':>11}")
    print(hdr)
    print("-" * len(hdr))
    base = {}
    for r in rows:
        if r.get("impl") == "incumbent" and "full_ms" in r:
            base[r["regime"]] = r["full_ms"]
    for r in rows:
        if "error" in r:
            print(f"{r['regime']:<13}{r.get('impl',''):<19}  ERROR "
                  f"{r['error'][:70]}")
            continue
        b = base.get(r["regime"])
        if b and r["impl"] != "incumbent" and r["impl"] != "cast_bf16_to_f32":
            d = (r["full_ms"] - b) / b * 100.0
            sp = b / r["full_ms"]
            dcol, scol = f"{d:>10.2f}", f"{sp:>11.3f}"
        else:
            dcol, scol = f"{'-':>10}", f"{'-':>11}"
        print(f"{r['regime']:<13}{r['impl']:<19}{r.get('scores_MB',0):>9.1f}"
              f"{r['full_ms']:>10.5f}{r['full_spread_pct']:>9.2f}"
              f"{(b if b else 0):>10.5f}{dcol}{scol}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
