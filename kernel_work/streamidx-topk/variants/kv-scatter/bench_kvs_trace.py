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
"""Capture one `*.xplane.pb` per (config, arm) for the KV-scatter A/B.

Method: `hlo_stats` **device self time**, TensorCore rows only, as every other
variant in this exercise used.  Per-instruction, no host-dispatch component,
and immune to pod contention -- which is why these traces may be fanned out
across the pod's four chips.

The wall clock printed per arm is **contended and is not a result**.  It is
there so a hung arm is visible in the log.  Nothing in `RESULTS.md` quotes it.

One trace per arm, never one per config: `hlo_stats` keys on the HLO op name,
so two arms sharing a trace would have their reshapes summed.

Every arm donates `cache_kv` (`donate_argnums=(2,)`), which is how
`pallas.jax_op` builds the real op.  Without donation XLA materialises a second
108.8 MB cache and the copy swamps the triad this exercise is measuring; the
Option B exercise measured that copy at ~170 µs and showed the triad itself
unchanged to 0.2% either way.

Needs jax/jaxlib 0.11.0: the image's pinned 0.10.2 SIGSEGVs on the first
`jax.profiler.trace` call against libtpu 0.0.44.1 (`PLUGIN_Profiler_Api size:
expected 80, got 104`).  `kvs_shard.sh` builds the side venv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kvs_common as KC  # noqa: E402
import kvs_op as KO  # noqa: E402


def log(msg):
    print(f"[kvs-trace] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="all")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    out_root = args.outdir or os.path.join(
        os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "kvs", "trace")
    os.makedirs(out_root, exist_ok=True)

    import jax

    mods = KC.load_modules()
    if not KC.report_md5s(mods, log):
        log("FATAL: kernel md5 read-back mismatch -- the pod is running the "
            "image's stale kernels.  Refusing to produce numbers.")
        return 3

    meta = {
        "jax": jax.__version__,
        "device_kind": jax.devices()[0].device_kind,
        "num_devices": len(jax.devices()),
        "iters": args.iters,
        "deinterleave": os.environ.get("KVS_DEINTERLEAVE", "index"),
        "cache_shape_base": list(KC.BASE_CACHE_SHAPE),
        "cache_shape_v128": list(KC.V128_CACHE_SHAPE),
        "cache_bytes": KC.CACHE_BYTES,
        "modules": {
            os.path.basename(m.__file__): {
                "path": m.__file__,
                "md5": KC.md5(m.__file__)
            }
            for m in (mods["incumbent"], mods["sparsecore"], mods["v128"])
        },
        "runs": [],
    }

    names = ([c.name for c in KC.CONFIGS]
             if args.configs == "all" else
             [s for s in args.configs.split(",") if s])
    arms_arg = (list(KC.ALL_ARMS) if args.arms == "all" else
                [s for s in args.arms.split(",") if s])

    meta_path = os.path.join(out_root, "trace_meta.json")

    def flush_meta():
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=1, default=str)

    flush_meta()

    for name in names:
        cfg = KC.CONFIG_BY_NAME[name]
        log(f"=== {cfg.name}: slots={cfg.slots} seqs={cfg.num_seqs} "
            f"tokens={cfg.num_tokens} decode={cfg.num_decode} arms={arms_arg}")
        for arm in arms_arg:
            # Rebuilt per arm: the arms donate `cache_kv`, so the buffer does
            # not survive one.  `build_inputs` is seeded, so every arm still
            # gets byte-identical inputs.
            inputs = KC.build_inputs(cfg,
                                     layout=KC.CACHE_LAYOUT[arm],
                                     seed=args.seed)
            call_args = KC.as_args(inputs)
            rec = {
                "config": cfg.name,
                "arm": arm,
                "cache_layout": KC.CACHE_LAYOUT[arm],
                "cache_shape": list(inputs["cache_kv"].shape),
                "slots": cfg.slots,
                "num_seqs": cfg.num_seqs,
                "num_tokens": cfg.num_tokens,
                "num_decode": cfg.num_decode,
            }
            try:
                fn = KO.make_arm(mods, arm, donate=True)
                step, _state = KC.chained_caller(fn, call_args)
                t0 = time.perf_counter()
                out = step()
                jax.block_until_ready(out)
                rec["compile_s"] = time.perf_counter() - t0
                rec["out_shapes"] = [list(o.shape) for o in out]
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
                log(f"  {cfg.name}/{arm}: compile={rec['compile_s']:.1f}s "
                    f"wall={rec['wall_us_per_call_CONTENDED_NOT_A_RESULT']:.1f}"
                    f"us (CONTENDED) xplanes={len(pbs)}")
                if not pbs:
                    log(f"  {cfg.name}/{arm}: NO XPLANE -- trace did not take")
            except Exception as err:  # noqa: BLE001
                rec["ok"] = False
                rec["error"] = repr(err)[:1500]
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
