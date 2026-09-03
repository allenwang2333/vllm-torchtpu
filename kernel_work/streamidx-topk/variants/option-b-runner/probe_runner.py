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
"""Option B through the real caller: plan, correctness, host cost.

Arms
----
`plan`      Static and traced plan inspection.  No kernel runs.  Checks the
            source-fidelity of the copied caller, the `plan_chunks` verdict at
            every (config, G), the launch count, the 32-row wave quantum, and
            -- the load-bearing one -- that `chunk_metadata` reproduces the
            pseudo-sequence plan `variants/bxc/` validated by hand
            (`seq_lens = 8448, 8704, 8960, 9216`, `cu_q_lens = [0, 256]`).

`correct`   Per-row top-k INDEX SET equality against the production caller
            calling the production kernel, order-insensitive, ignoring -1.
            Stage-A scores are not bitwise identical under Option C and are
            not expected to be; the selection is.

`hostcost`  The one measurement a device-time method cannot make.  Reports
            per-call wall clock as `max(device, host)` -- never a subtraction
            -- alongside the dispatch-only floor, the traced HLO instruction
            count, and the compile time.  Wall clock here is NOT a device-time
            result and is labelled so.

Device time does not live here: it comes from `bench_runner_trace.py` +
`read_runner_hlo_stats.py` (`hlo_stats` self time), for the reason
`variants/merged/RESULTS.md` gives -- per-call wall clock is
`max(device, host)` and the host floor is a clamp, not a term.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import runner_common as RC  # noqa: E402
import runner_op  # noqa: E402

# The plan `variants/bxc/RESULTS.md` section 1.4 validated by hand, at
# q_len 1024 / ctx 9216 / G=4.  If `chunk_metadata` does not reproduce it, the
# general formula is not the thing that was measured.
BXC_G4_SEQ_LENS = [8448, 8704, 8960, 9216]
BXC_G4_CU_Q_LENS = [0, 256]


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def arm_plan(mods, log=print) -> dict:
    out = {"fidelity": runner_op.check_fidelity(log)}
    ok = out["fidelity"]["prologue_identical"]

    rows = []
    for cfg in RC.CONFIGS:
        inputs = RC.build_inputs(cfg)
        for g in RC.G_SWEEP:
            plan = mods["b"].plan_chunks(cfg.num_tokens, cfg.slots, g)
            policy_g = runner_op.chunk_policy(cfg.num_tokens, cfg.slots, g)
            row = {
                "config": cfg.name,
                "G": g,
                "plan_ok": plan["ok"],
                "reason": plan["reason"],
                "policy_g": policy_g,
                "num_launches": plan.get("num_launches"),
            }
            if plan["ok"] and g > 1:
                meta = []
                for (a, b) in plan["ranges"]:
                    sl, cq, ms, me = mods["b"].chunk_metadata(
                        inputs["seq_lens"], inputs["cu_q_lens"],
                        inputs["distribution"], a, b)
                    ns = cfg.num_seqs
                    meta.append({
                        "range": [a, b],
                        "seq_lens": np.asarray(sl)[:ns].tolist(),
                        "cu_q_lens": np.asarray(cq)[:ns + 1].tolist(),
                        "mixed_start": int(ms),
                        "mixed_end": int(me),
                    })
                row["chunks"] = meta
                # The 32-row wave quantum `_sc_topk_direct` launches on.
                t = plan["chunk_size"]
                row["wave_guard"] = bool(t >= 32 and t % 32 == 0)
                row["sc_partition"] = int(
                    mods["sparsecore"]._pick_partition(t, RC.NUM_POSITIONS,
                                                       RC.K_IDX))
                ok = ok and row["wave_guard"] and row["sc_partition"] == 1
            rows.append(row)
            log(f"PLAN {cfg.name:<24} G={g} ok={plan['ok']!s:<5} "
                f"policy_g={policy_g} launches={plan.get('num_launches')} "
                f"wave_guard={row.get('wave_guard')} "
                f"p={row.get('sc_partition')} :: {plan['reason']}")
            if "chunks" in row:
                for c in row["chunks"]:
                    log(f"     chunk {c['range']} seq_lens={c['seq_lens']} "
                        f"cu_q_lens={c['cu_q_lens']} "
                        f"mixed=[{c['mixed_start']},{c['mixed_end']})")
    out["rows"] = rows

    # The bxc cross-check.
    cfg = RC.CONFIG_BY_NAME["pre_q1024_ctx9216_s1"]
    inputs = RC.build_inputs(cfg)
    plan = mods["b"].plan_chunks(cfg.num_tokens, cfg.slots, 4)
    got_sl, got_cq = [], []
    for (a, b) in plan["ranges"]:
        sl, cq, _, _ = mods["b"].chunk_metadata(inputs["seq_lens"],
                                                inputs["cu_q_lens"],
                                                inputs["distribution"], a, b)
        got_sl.append(int(np.asarray(sl)[0]))
        got_cq.append(np.asarray(cq)[:2].tolist())
    match = (got_sl == BXC_G4_SEQ_LENS
             and all(c == BXC_G4_CU_Q_LENS for c in got_cq))
    ok = ok and match
    out["bxc_crosscheck"] = {
        "expected_seq_lens": BXC_G4_SEQ_LENS,
        "got_seq_lens": got_sl,
        "expected_cu_q_lens": BXC_G4_CU_Q_LENS,
        "got_cu_q_lens": got_cq,
        "match": match,
    }
    log(f"BXC-CROSSCHECK G=4 seq_lens expected={BXC_G4_SEQ_LENS} "
        f"got={got_sl} cu_q_lens={got_cq} match={match}")

    # The decode gate: `chunk_policy` must refuse to chunk a decode program.
    dec = RC.CONFIG_BY_NAME["dec_b64_ctx8704_s64"]
    gate = runner_op.chunk_policy(dec.num_tokens, dec.slots)
    ok = ok and gate == 1
    out["decode_gate_g"] = gate
    log(f"DECODE-GATE chunk_policy(num_tokens={dec.num_tokens}, "
        f"max_num_seqs={dec.slots}) = {gate} (must be 1)")

    out["ok"] = bool(ok)
    log(f"PLAN_ARM_OK={out['ok']}")
    return out


# ---------------------------------------------------------------------------
# correct
# ---------------------------------------------------------------------------


def _index_sets_equal(ref: np.ndarray, got: np.ndarray) -> dict:
    """Per-row set equality, ignoring -1 (the empty-slot fill)."""
    assert ref.shape == got.shape, (ref.shape, got.shape)
    bad_rows, first = 0, None
    for r in range(ref.shape[0]):
        a = set(int(x) for x in ref[r] if x >= 0)
        b = set(int(x) for x in got[r] if x >= 0)
        if a != b:
            bad_rows += 1
            if first is None:
                first = {
                    "row": r,
                    "n_ref": len(a),
                    "n_got": len(b),
                    "missing": sorted(a - b)[:8],
                    "extra": sorted(b - a)[:8],
                }
    return {
        "rows": int(ref.shape[0]),
        "bad_rows": bad_rows,
        "first_bad": first,
        "ref_neg1": int((ref < 0).sum()),
        "got_neg1": int((got < 0).sum()),
    }


def arm_correct(mods, configs, log=print) -> dict:
    results = []
    all_ok = True
    for name in configs:
        cfg = RC.CONFIG_BY_NAME[name]
        inputs = RC.build_inputs(cfg)
        args = RC.as_args(inputs)

        ref_fn = RC.make_arm(mods, "inc")
        t0 = time.perf_counter()
        ref_cache, ref_idx = jax.block_until_ready(ref_fn(*args))
        log(f"REF {cfg.name} compile+run {time.perf_counter()-t0:.1f}s "
            f"shape={tuple(ref_idx.shape)}")
        ref_np = np.asarray(ref_idx)
        ref_cache_np = np.asarray(ref_cache)

        for g in RC.G_SWEEP:
            if g > 1 and not mods["b"].plan_chunks(cfg.num_tokens, cfg.slots,
                                                   g)["ok"]:
                log(f"SKIP {cfg.name} G={g}: plan rejects it")
                continue
            fn = RC.make_arm(mods, "b", g)
            t0 = time.perf_counter()
            try:
                got_cache, got_idx = jax.block_until_ready(fn(*args))
            except Exception as err:  # noqa: BLE001
                all_ok = False
                results.append({
                    "config": cfg.name,
                    "G": g,
                    "error": repr(err)[:600]
                })
                log(f"CORRECT {cfg.name} G={g} RAISED {repr(err)[:300]}")
                continue
            dt = time.perf_counter() - t0
            cmp = _index_sets_equal(ref_np, np.asarray(got_idx))
            cache_same = bool(
                np.array_equal(ref_cache_np, np.asarray(got_cache)))
            ok = cmp["bad_rows"] == 0 and cache_same
            all_ok = all_ok and ok
            rec = {
                "config": cfg.name,
                "G": g,
                "compile_s": round(dt, 1),
                "cache_identical": cache_same,
                **cmp, "ok": ok
            }
            results.append(rec)
            log(f"CORRECT {cfg.name:<24} G={g} bad_rows={cmp['bad_rows']}"
                f"/{cmp['rows']} cache_identical={cache_same} "
                f"ref_neg1={cmp['ref_neg1']} got_neg1={cmp['got_neg1']} "
                f"compile={dt:.1f}s OK={ok}")
            if cmp["first_bad"]:
                log(f"   first_bad={cmp['first_bad']}")
    log(f"CORRECT_ARM_OK={all_ok}")
    return {"ok": all_ok, "results": results}


# ---------------------------------------------------------------------------
# hostcost
# ---------------------------------------------------------------------------


def _wall_us(step, trials: int) -> float:
    """Best-of-`trials` per-call wall clock, blocked.

    This is `max(device, host)`, and it is reported as such.  It is NOT a
    device time and nothing is subtracted from it.
    """
    jax.block_until_ready(step())
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        out = step()
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
        del out
    return best * 1e6


def _dispatch_us(step, trials: int) -> float:
    """Time to *enqueue* a call, without waiting for it.

    The host term, and the whole reason this arm exists: a runner change adds
    host work that no device-time method can see.  Measured by not blocking:
    the loop returns as soon as the call is dispatched, so it bounds the
    host-side cost of the extra launches Option B adds.  A queue that fills
    makes this converge to device time, so it is a floor on the host cost and
    a ceiling only when device time is larger, which is exactly why it must
    never be subtracted from anything.
    """
    jax.block_until_ready(step())
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        out = step()
        best = min(best, time.perf_counter() - t0)
        jax.block_until_ready(out)
        del out
    return best * 1e6


def arm_hostcost(mods, configs, trials: int, log=print) -> dict:
    rows = []
    for name in configs:
        cfg = RC.CONFIG_BY_NAME[name]
        for arm in RC.perf_arms(mods, cfg):
            # Per arm, not per config: these arms donate `cache_kv`.
            inputs = RC.build_inputs(cfg)
            args = RC.as_args(inputs)
            g = 1 if arm == "inc" else int(arm[2:])
            fn = RC.make_arm(mods,
                             "inc" if arm == "inc" else "b",
                             g,
                             donate=True)
            t0 = time.perf_counter()
            lowered = fn.lower(*args)
            compiled = lowered.compile()
            compile_s = time.perf_counter() - t0
            # StableHLO, pre-optimization: an indicative size, not a schedule.
            txt = lowered.as_text()
            n_custom = txt.count("tpu_custom_call")
            n_lines = txt.count("\n")
            step, _state = RC.chained_caller(fn, args)
            wall = _wall_us(step, trials)
            disp = _dispatch_us(step, trials)
            rec = {
                "config": cfg.name,
                "arm": arm,
                "donate_cache_kv": True,
                "compile_s": round(compile_s, 2),
                "hlo_lines": n_lines,
                "hlo_custom_calls": n_custom,
                "wall_us_max_device_host": round(wall, 3),
                "dispatch_us_host_floor": round(disp, 3),
            }
            del compiled
            rows.append(rec)
            log(f"HOST {cfg.name:<24} {arm:<5} "
                f"wall_us(max(dev,host))={wall:9.3f} "
                f"dispatch_us={disp:8.3f} compile_s={compile_s:6.1f} "
                f"hlo_lines={n_lines} custom_calls={n_custom}")
    return {"rows": rows}


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=("plan", "correct", "hostcost"))
    ap.add_argument("--configs", default="")
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    mods = RC.load_modules()
    md5_ok = RC.report_md5s(mods)
    if not md5_ok:
        print("FATAL: kernel module md5 read-back does not match the "
              "checkout; the pod is running the image's stale kernels.")
        return 3
    print(f"JAX {jax.__version__} devices={jax.devices()}")

    if a.arm == "plan":
        res = arm_plan(mods)
    else:
        names = ([n for n in a.configs.split(",") if n]
                 or list(RC.PERF_CONFIGS if a.arm == "hostcost" else
                         [c.name for c in RC.CONFIGS]))
        res = (arm_correct(mods, names) if a.arm == "correct" else
               arm_hostcost(mods, names, a.trials))

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        print(f"wrote {a.json}")
    return 0 if res.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
