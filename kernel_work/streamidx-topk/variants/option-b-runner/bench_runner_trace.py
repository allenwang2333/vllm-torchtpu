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
"""Capture one `*.xplane.pb` per (config, arm) for the Option B runner A/B.

Method: `hlo_stats` **device self time**, as `variants/merged/` and
`variants/bxc/` used.  Per-instruction, no host dispatch component, immune to
pod contention.

The wall clock printed per arm is **contended and is not a result** -- it is
there so a hung arm is visible.  The host-side cost that Option B really does
add is measured separately and deliberately, by `probe_runner.py --arm
hostcost`, which is the one place in this exercise where host time is the
point rather than an artifact.

One trace per arm, never one per config: `hlo_stats` keys on the HLO op name
and two arms in one trace would be summed.

The traced unit is the **caller** (`_streamidx_topk_jax`: KV-cache scatter +
bitcast + `streamindex_topk`), not the kernel.  That is deliberate -- a runner
change has to be judged on what the runner runs, and the scatter is real work
the kernel-level harnesses never charged anyone for.

Needs jax/jaxlib 0.11.0; `runner_shard.sh` builds the side venv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runner_common as RC  # noqa: E402


def log(msg):
    print(f"[runner-trace] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="perf")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    out_root = args.outdir or os.path.join(
        os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "obrunner", "trace")
    os.makedirs(out_root, exist_ok=True)

    import jax

    mods = RC.load_modules()
    if not RC.report_md5s(mods, log):
        log("FATAL: kernel md5 read-back mismatch -- the pod is running the "
            "image's stale kernels.  Refusing to produce numbers.")
        return 3

    meta = {
        "jax": jax.__version__,
        "device_kind": jax.devices()[0].device_kind,
        "num_devices": len(jax.devices()),
        "iters": args.iters,
        "modules": {
            os.path.basename(m.__file__): {
                "path": m.__file__,
                "md5": RC.md5(m.__file__)
            }
            for m in (mods["incumbent"], mods["sparsecore"], mods["merged"],
                      mods["b"])
        },
        "runs": [],
    }

    if args.configs == "perf":
        names = list(RC.PERF_CONFIGS)
    elif args.configs == "all":
        names = [c.name for c in RC.CONFIGS]
    else:
        names = [s for s in args.configs.split(",") if s]

    meta_path = os.path.join(out_root, "trace_meta.json")

    def flush_meta():
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=1, default=str)

    flush_meta()

    for name in names:
        cfg = RC.CONFIG_BY_NAME[name]
        arms = (RC.perf_arms(mods, cfg) if args.arms == "all" else
                [a for a in args.arms.split(",") if a])
        log(f"=== {cfg.name}: slots={cfg.slots} real_seqs={cfg.num_seqs} "
            f"tokens={cfg.num_tokens} decode={cfg.num_decode} arms={arms}")

        for arm in arms:
            # Rebuilt per arm, not per config: the arms donate `cache_kv`
            # (as the real op does), so the buffer does not survive an arm.
            # `build_inputs` is seeded, so every arm gets identical bytes.
            inputs = RC.build_inputs(cfg, seed=args.seed)
            call_args = RC.as_args(inputs)
            g = 1 if arm == "inc" else int(arm[2:])
            plan = mods["b"].plan_chunks(cfg.num_tokens, cfg.slots, g)
            rec = {
                "config": cfg.name,
                "arm": arm,
                "G": g,
                "slots": cfg.slots,
                "num_seqs": cfg.num_seqs,
                "num_tokens": cfg.num_tokens,
                "num_decode": cfg.num_decode,
                "num_positions": RC.NUM_POSITIONS,
                "plan": plan,
                "policy_g": RC.runner_op.chunk_policy(cfg.num_tokens,
                                                      cfg.slots, g),
            }
            try:
                fn = RC.make_arm(mods,
                                 "inc" if arm == "inc" else "b",
                                 g,
                                 donate=True)
                step, _state = RC.chained_caller(fn, call_args)
                t0 = time.perf_counter()
                out = step()
                jax.block_until_ready(out)
                rec["compile_s"] = time.perf_counter() - t0
                rec["out_shape"] = [list(o.shape) for o in out]
                del out
                for _ in range(args.warmup):
                    jax.block_until_ready(step())

                logdir = os.path.join(out_root, f"{cfg.name}__{arm}")
                n = 0
                t0 = time.perf_counter()
                with jax.profiler.trace(logdir):
                    while n < args.iters:
                        k = min(args.chunk, args.iters - n)
                        last = None
                        for _ in range(k):
                            last = step()
                        jax.block_until_ready(last)
                        n += k
                        del last
                wall = time.perf_counter() - t0
                rec["traced_iters"] = n
                rec["wall_us_per_call_CONTENDED_NOT_A_RESULT"] = (wall / n *
                                                                  1e6)
                pbs = []
                for root, _d, files in os.walk(logdir):
                    for f in files:
                        if f.endswith(".xplane.pb"):
                            pbs.append(os.path.join(root, f))
                rec["xplane_files"] = pbs
                rec["ok"] = bool(pbs)
                log(f"  {cfg.name}/{arm}: G={g} launches="
                    f"{plan.get('num_launches')} "
                    f"compile={rec['compile_s']:.1f}s wall="
                    f"{rec['wall_us_per_call_CONTENDED_NOT_A_RESULT']:.1f}us "
                    f"(CONTENDED) xplanes={len(pbs)}")
                if not pbs:
                    log(f"  {cfg.name}/{arm}: NO XPLANE -- trace did not take")
            except Exception as err:  # noqa: BLE001
                rec["ok"] = False
                rec["error"] = repr(err)[:600]
                log(f"  {cfg.name}/{arm}: FAILED {err!r}")
                traceback.print_exc()
            meta["runs"].append(rec)
            flush_meta()

            for value in inputs.values():
                try:
                    value.delete()
                except (AttributeError, RuntimeError):
                    pass  # donated buffers are already gone
            del inputs, call_args

    n_ok = sum(1 for r in meta["runs"] if r.get("ok"))
    log(f"done: {n_ok}/{len(meta['runs'])} arms produced an xplane")
    return 0 if n_ok == len(meta["runs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
