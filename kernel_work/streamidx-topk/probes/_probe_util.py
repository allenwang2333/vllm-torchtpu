"""Timing helper shared by the step-2 probes.

`time_arm` is the same methodology the step-1 instrument uses
(`instruments/bench_streamidx_topk.py:681`, which is read-only and is not
edited): warm up, size an iteration count so the host clock is not measuring
its own resolution, then take the **best** of N batched trials.

Best-of-N is optimistic by construction.  On a fanned-out pod the upper tail is
contention from the other three shards, which is why the best is the right
statistic here -- but every number it produces is a lower bound on the op's
cost, and a *fanned-out* wall clock is contended regardless.  Both caveats are
repeated in each probe's report.
"""

from __future__ import annotations

import json
import statistics
import time


def time_arm(fn, args, warmup: int = 3, trials: int = 7,
             target_s: float = 0.05):
    """Returns dict(best_s, median_s, iters, compile_s, trials_s)."""
    import jax

    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    single = time.perf_counter() - t0
    iters = max(1, min(500, int(target_s / max(single, 1e-9))))

    per_trial = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(*args)
        jax.block_until_ready(out)
        per_trial.append((time.perf_counter() - t0) / iters)
    del out
    return dict(
        best_s=min(per_trial),
        median_s=statistics.median(per_trial),
        iters=iters,
        compile_s=compile_s,
        trials_s=per_trial,
    )


def p(tag, *rest):
    print(f"[probe] {tag}:", *rest, flush=True)


def emit_json(name, payload, outdir=None):
    """Print the machine-readable payload, and drop it under $CDK_OUTPUT_DIR
    if one exists so it survives the job."""
    import os
    blob = json.dumps(payload, indent=2, default=str)
    print(f"===== {name} JSON BEGIN =====", flush=True)
    print(blob, flush=True)
    print(f"===== {name} JSON END =====", flush=True)
    outdir = outdir or os.environ.get("CDK_OUTPUT_DIR")
    if outdir:
        try:
            shard = os.environ.get("TPU_FANOUT_SHARD", "x")
            d = os.path.join(outdir, "probes")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{name}.shard{shard}.json"), "w") as f:
                f.write(blob)
        except Exception as e:  # never let bookkeeping kill a probe
            print(f"[probe] emit_json write failed: {e!r}", flush=True)


def tpu_banner():
    import jax
    from jax.experimental.pallas import tpu as pltpu
    p("jax", jax.__version__)
    p("device_kind", jax.devices()[0].device_kind)
    p("num_devices", len(jax.devices()))
    try:
        info = pltpu.get_tpu_info()
        sc = getattr(info, "sparse_core", None)
        p("sparse_core", sc)
        p("mem_bw_bytes_per_second",
          getattr(info, "mem_bw_bytes_per_second", "<absent>"))
        p("vmem_capacity_bytes",
          getattr(info, "vmem_capacity_bytes", "<absent>"))
    except Exception as e:
        p("get_tpu_info", f"FAILED {e!r}")
