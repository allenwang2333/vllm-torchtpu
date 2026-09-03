#!/usr/bin/env python3
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
"""Correctness: do the v128 arms put the same bytes in the same places?

Three checks, in increasing strength, run at both configs.

1. **`bytes`** -- scatter only.  `scat256` writes the production cache;
   `scat128` writes the 128-minor one; `scat128rt` writes the 128-minor view of
   a production-shaped buffer.  Every result is read back **through the
   original `[415,256,4,256]` view** -- the layout the consumer uses -- and
   compared with `np.array_equal`.  Nothing here involves the kernel, so a
   failure is unambiguously the scatter's.

2. **`crosscheck`** -- the production `StreamIdxTC` kernel run twice: once on a
   `base`-written cache and once on the `v128`-written cache reshaped back.  If
   check 1 passes this is redundant by construction, and that is the point:
   it is the evidence that survives if the v128 *kernel* fails to compile.

3. **`kernel`** -- the whole `v128` arm (128-minor cache, 128-minor kernel)
   against the whole `base` arm, comparing the returned top-k indices.  This is
   the check that the v128 kernel reads the same values out.

Top-k indices are compared exactly.  The scores feeding `sparsecore_topk` are
computed from identical bytes by arithmetic that differs only in which sub-row
a byte is fetched from, so bit-identical scores and therefore identical indices
are the expectation; a per-row *set* comparison is reported alongside so that a
pure tie-break difference would be distinguishable from a real one.

No donation here: `donate_argnums=(2,)` invalidates the input buffer, and every
check needs to hand the same input to more than one arm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import numpy as np  # noqa: E402

import kvs_common as KC  # noqa: E402
import kvs_op as KO  # noqa: E402


def log(msg):
    print(f"[kvs-correct] {msg}", flush=True)


def _as_base(arr) -> np.ndarray:
    """Read a cache back through the production `[415,256,4,256]` view.

    A base-shaped array passes through.  A 128-minor one goes through
    `KC.v128_to_base`, the exact inverse of the placement `build_inputs` used
    -- a plain reshape in the interleaved modes, the un-permutation under
    `split`.  Doing it this way keeps every byte-equality check below valid in
    all three modes instead of having to skip them.
    """
    a = np.asarray(arr)
    if tuple(a.shape) == tuple(KC.BASE_CACHE_SHAPE):
        return a
    return np.asarray(KC.v128_to_base(a))


def _cmp_cache(name: str, ref: np.ndarray, got: np.ndarray) -> dict:
    eq = bool(np.array_equal(ref, got))
    diff = int(np.count_nonzero(ref != got)) if not eq else 0
    rec = {"array_equal": eq, "differing_bytes": diff, "total_bytes": ref.size}
    log(f"CACHE {name}: np.array_equal={eq} differing_bytes={diff}/{ref.size}")
    if not eq:
        idx = np.argwhere(ref != got)[:5]
        rec["first_diffs"] = [{
            "index": i.tolist(),
            "ref": int(ref[tuple(i)]),
            "got": int(got[tuple(i)])
        } for i in idx]
        for d in rec["first_diffs"]:
            log(f"  DIFF {d}")
    return rec


def _cmp_topk(name: str, ref, got) -> dict:
    r = np.asarray(ref)
    g = np.asarray(got)
    eq = bool(r.shape == g.shape and np.array_equal(r, g))
    rec = {"array_equal": eq, "shape": list(r.shape)}
    if not eq and r.shape == g.shape:
        rows = np.any(r != g, axis=1)
        rec["differing_rows"] = int(rows.sum())
        rec["total_rows"] = int(r.shape[0])
        same_set = sum(
            1 for i in range(r.shape[0]) if set(r[i].tolist()) == set(
                g[i].tolist()))
        rec["rows_with_identical_set"] = same_set
        rec["first_differing_row"] = int(np.argmax(rows))
    log(f"TOPK {name}: np.array_equal={eq} {rec}")
    return rec


def run_cfg(mods, cfg, results: dict) -> None:
    base_in = KC.build_inputs(cfg, layout="base", seed=0)
    v128_in = KC.build_inputs(cfg, layout="v128", seed=0)

    # Sanity: the two input sets must start byte-identical, or nothing below
    # means anything.
    pre = _cmp_cache(f"{cfg.name}/inputs-identical",
                     _as_base(base_in["cache_kv"]),
                     _as_base(v128_in["cache_kv"]))
    results[f"{cfg.name}/inputs_identical"] = pre

    base_args = KC.as_args(base_in)
    v128_args = KC.as_args(v128_in)

    # The `*rt` arms hold the cache in the production `[...,4,256]` shape and
    # reach the 128-minor form by a *reshape* across the scatter.  Under
    # `split` the 128-minor form is a permutation, not a reshape, so those arms
    # would write this step's tokens in one placement into a buffer whose KV
    # history is in the other.  That is incoherent by construction rather than
    # wrong, so they are dropped from the run and recorded as dropped.
    rt_ok = KO.DEINTERLEAVE != "split"
    results[f"{cfg.name}/deinterleave"] = KO.DEINTERLEAVE
    results[f"{cfg.name}/round_trip_arms_meaningful"] = rt_ok
    if not rt_ok:
        log(f"{cfg.name}: KVS_DEINTERLEAVE={KO.DEINTERLEAVE}; the round-trip "
            f"arms (`scat128rt`, `v128rt`) reach the 128-minor form by a "
            f"reshape, which is not the split placement.  Dropped.")
    byte_arms = ("scat256", "scat128", "scat128rt") if rt_ok else ("scat256",
                                                                   "scat128")
    kernel_arms = ("v128", "v128rt") if rt_ok else ("v128", )

    # ---- 1. bytes ---------------------------------------------------------
    ref = None
    for arm in byte_arms:
        key = f"{cfg.name}/bytes/{arm}"
        try:
            fn = KO.make_arm(mods, arm, donate=False)
            args = v128_args if KC.CACHE_LAYOUT[arm] == "v128" else base_args
            (out, ) = jax.block_until_ready(fn(*args))
            got = _as_base(out)
            if ref is None:
                ref = got
                results[key] = {"role": "reference", "shape": list(out.shape)}
                log(f"CACHE {key}: reference, device shape {out.shape}")
            else:
                results[key] = _cmp_cache(key, ref, got)
                results[key]["device_shape"] = list(out.shape)
            del out
        except Exception as err:  # noqa: BLE001
            results[key] = {"error": repr(err)[:1200]}
            log(f"{key}: FAILED {err!r}")
            traceback.print_exc()

    # ---- 2. crosscheck: production kernel on both written caches ----------
    # `base_written` and `v128_written_as_base` are byte-identical if check 1
    # passed; running the production kernel on each proves the kernel reads the
    # same values out without depending on the v128 kernel compiling.
    topk_ref = None
    fn_base = KO.make_arm(mods, "base", donate=False)
    try:
        cache_b, topk_b = jax.block_until_ready(fn_base(*base_args))
        topk_ref = np.asarray(topk_b)
        results[f"{cfg.name}/base_topk_shape"] = list(topk_ref.shape)
        base_written = _as_base(cache_b)
        del cache_b, topk_b
    except Exception as err:  # noqa: BLE001
        results[f"{cfg.name}/crosscheck/base"] = {"error": repr(err)[:1200]}
        log(f"{cfg.name}/crosscheck/base: FAILED {err!r}")
        traceback.print_exc()
        base_written = None

    if base_written is not None:
        key = f"{cfg.name}/crosscheck/v128_written_then_prod_kernel"
        try:
            fn128 = KO.make_arm(mods, "scat128", donate=False)
            (written128, ) = jax.block_until_ready(fn128(*v128_args))
            as_base = _as_base(written128)
            results[f"{cfg.name}/crosscheck/bytes"] = _cmp_cache(
                f"{cfg.name}/crosscheck/bytes", base_written, as_base)
            del written128

            # Feed the v128-written bytes to the production kernel by handing
            # the caller a cache that is already written; the caller's own
            # scatter then re-writes the same rows with the same values, so
            # the kernel sees exactly the v128 result.
            alt = dict(base_in)
            alt["cache_kv"] = jax.device_put(as_base)
            _c, topk_cross = jax.block_until_ready(fn_base(*KC.as_args(alt)))
            results[key] = _cmp_topk(key, topk_ref, topk_cross)
            del _c, topk_cross, alt
        except Exception as err:  # noqa: BLE001
            results[key] = {"error": repr(err)[:1200]}
            log(f"{key}: FAILED {err!r}")
            traceback.print_exc()

    # ---- 3. kernel: the whole v128 arm ------------------------------------
    for arm in kernel_arms:
        key = f"{cfg.name}/kernel/{arm}"
        if topk_ref is None:
            results[key] = {"error": "base arm failed; nothing to compare to"}
            continue
        try:
            fn = KO.make_arm(mods, arm, donate=False)
            args = v128_args if KC.CACHE_LAYOUT[arm] == "v128" else base_args
            cache_x, topk_x = jax.block_until_ready(fn(*args))
            results[key] = _cmp_topk(key, topk_ref, topk_x)
            if base_written is not None:
                results[key + "/cache"] = _cmp_cache(
                    key + "/cache", base_written, _as_base(cache_x))
            del cache_x, topk_x
        except Exception as err:  # noqa: BLE001
            results[key] = {"error": repr(err)[:1200]}
            log(f"{key}: FAILED {err!r}")
            traceback.print_exc()

    for v in list(base_in.values()) + list(v128_in.values()):
        try:
            v.delete()
        except (AttributeError, RuntimeError):
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="all")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    log(f"jax={jax.__version__} devices={jax.devices()}")
    log(f"KVS_DEINTERLEAVE={os.environ.get('KVS_DEINTERLEAVE', 'index')}")
    mods = KC.load_modules()
    if not KC.report_md5s(mods, log):
        log("FATAL: kernel md5 read-back mismatch. Refusing to produce a "
            "result.")
        return 3
    KO.check_fidelity(log)

    names = ([c.name for c in KC.CONFIGS]
             if a.configs == "all" else a.configs.split(","))
    results: dict = {
        "deinterleave": os.environ.get("KVS_DEINTERLEAVE", "index")
    }
    for name in names:
        log(f"===== config {name} =====")
        run_cfg(mods, KC.CONFIG_BY_NAME[name], results)
        path = a.json or os.path.join(
            os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "kvs", "correct.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(results, fh, indent=1, default=str)

    fails = [
        k for k, v in results.items() if isinstance(v, dict) and (
            "error" in v or v.get("array_equal") is False)
    ]
    log(f"SUMMARY failed_checks={len(fails)} {fails}")
    log(f"wrote {path}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
