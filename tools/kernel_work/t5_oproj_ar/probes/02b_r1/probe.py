"""t5-oproj-ar probe 02b timing: incumbent vs R1 arms, b1-b4, 16 ranks (2x2x2), one job.

Harness = instruments/bench.py v1.1 (same inputs, same L=8 cold-weight stack, same filler-separated chain
x = rmsnorm(h @ Wd); h = arm(x), same R_SET 32/256/1024, same trace parse), with several candidate arms:
  inc    production fp8_skinny + small_all_reduce (common.incumbent, md5s printed)        2 custom calls
  mm     fp8_skinny alone                                                                  1
  L1     R1, barrier signalled at entry and waited after the last tile, full RS after it   1
  L2     L1 + RS waves after tiles 0/1 (barrier waited after tile 0)                       1
  Rbar   L1 with the barrier signalled+waited at entry, before the weight stream (timing)  1
  Rex    L1 without RS/sum/AG (barrier balanced; output = local partial; timing only)     1
  fill   filler + XLA tile of x (the subtrahend)                                           0
Primary clock: device-trace iteration span (rank 0), net of fill. Cross-check: wall LSQ slope.
usage: python probe.py    (env INST = instruments dir, VAR = variants dir)
"""
import collections
import glob
import gzip
import json
import os
import re
import shutil
import statistics as st
import sys
import time

sys.path.insert(0, os.environ.get("INST", "/root/inst"))
sys.path.insert(0, os.environ.get("VAR", "/root/var"))
import common as C  # noqa: E402

C.init_distributed()
import hashlib  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

import fused_oproj_ar as F  # noqa: E402

mesh, N = C.make_mesh()
AX = C.AXIS
SH = NamedSharding(mesh, P(AX))
L = 8
R_SET = (32, 256, 1024)
REPS = int(os.environ.get("REPS", 7))
ROUNDS = int(os.environ.get("ROUNDS", 3))


def say(d):
    if C.is_main():
        print(json.dumps(d), flush=True)


say({"n": N, "kind": jax.devices()[0].device_kind, "jax": jax.__version__, "prov": C.provenance(),
     "cand_md5": hashlib.md5(open(F.__file__, "rb").read()).hexdigest(), "R_SET": R_SET})

VARIANTS = {"L1": {}, "L2": dict(waves=True), "Rbar": dict(barrier="exposed"), "Rex": dict(exchange=False)}
ARMS = ["fill", "inc", "mm", "L1", "L2", "Rbar", "Rex"]
EXPECTED_CC = {"fill": 0, "inc": 2, "mm": 1, "L1": 1, "L2": 1, "Rbar": 1, "Rex": 1}

# M6 driver: remote DMAs per call and the owner waves, from the code that builds the kernel
for cname, m, nv in C.CELLS:
    say({"driver": cname, "bytes_per_push": m * (C.N_OUT // N) * 2, "remote_dmas_per_call": 2 * (N - 1),
         "L2_waves": F.owner_waves(N, C.N_OUT, 2048, C.N_OUT // 2048), "W_tiles": C.N_OUT // 2048})


def put(a):
    return jax.make_array_from_callback(a.shape, SH, lambda i: a[i])


def tile6(x):
    return jnp.concatenate([x] * (C.N_OUT // C.K_IN), axis=1)


def arm_fn(name):
    def call(x, w, ws, idx, rank, nv):
        if name == "inc":
            return C.incumbent(x, w, idx, ws, rank, nv, num_cores=N, axis=AX)
        if name == "mm":
            return C.incumbent_matmul(x, w, idx, ws)
        if name == "fill":
            return tile6(x)
        return F.fused_oproj_all_reduce(x, w, idx, ws, rank, nv, num_cores=N, axis=AX, **VARIANTS[name])

    def body(h, wd, w, ws, rank, nv, r):
        def it(i, hh):
            y = jnp.dot(hh, wd, preferred_element_type=jnp.float32)
            x = (y * jax.lax.rsqrt(jnp.mean(y * y, axis=1, keepdims=True) + 1e-6)).astype(C.BF16)
            idx = jnp.full((1,), i % L, jnp.int32)
            return call(x, w, ws, idx, rank, nv).astype(hh.dtype)

        return jax.lax.fori_loop(0, r[0], it, h)

    spec = P(AX)
    return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(spec,) * 7, out_specs=spec, check_vma=False))


rng = np.random.default_rng(5)
units = {}
_, wq, ws = C.rank_inputs(N, 4, 77, n_layers=L)
W_DEV = put(wq.reshape(N * L, C.K_IN, C.N_OUT))
WS_DEV = put(ws.reshape(-1))
for cname, m, nv in C.CELLS:
    h0 = np.asarray(jnp.asarray(rng.standard_normal((N * m, C.N_OUT), dtype=np.float32), C.BF16))
    wd = np.asarray(jnp.asarray(rng.standard_normal((N * C.N_OUT, C.K_IN), dtype=np.float32)
                                / np.sqrt(C.N_OUT), C.BF16))
    ops = [put(h0), put(wd), W_DEV, WS_DEV, put(np.arange(N, dtype=np.int32)),
           put(np.full((N,), nv, np.int32))]
    for a in ARMS:
        t = time.time()
        try:
            fn = arm_fn(a)
            rr = {r: put(np.full((N,), r, np.int32)) for r in R_SET}
            comp = fn.lower(*ops, rr[R_SET[0]]).compile()
            cc = comp.as_text().count('custom_call_target="tpu_custom_call"')
            jax.block_until_ready(comp(*ops, rr[R_SET[0]]))
            units[(a, cname)] = (comp, ops, rr)
            say({"compiled": a, "cell": cname, "s": round(time.time() - t, 1), "custom_calls": cc,
                 "expected": EXPECTED_CC[a]})
        except Exception as e:
            say({"compile_error": a, "cell": cname, "err": repr(e)[:1500]})

walls = collections.defaultdict(list)
for rnd in range(ROUNDS):
    for key, (fn, ops, rr) in units.items():
        for r in R_SET:
            ts = []
            for _ in range(REPS + 1):
                t = time.perf_counter()
                jax.block_until_ready(fn(*ops, rr[r]))
                ts.append(time.perf_counter() - t)
            walls[(*key, r)].append(st.median(ts[1:]))
    say({"round_done": rnd})

res = {}
for (a, c) in units:
    xs = np.asarray(R_SET, float)
    ys = np.asarray([st.median(walls[(a, c, r)]) for r in R_SET]) * 1e6
    A = np.vstack([xs, np.ones_like(xs)]).T
    (b, a0), *_ = np.linalg.lstsq(A, ys, rcond=None)
    pr = [float(np.linalg.lstsq(A, np.asarray([walls[(a, c, r)][k] for r in R_SET]) * 1e6,
                                rcond=None)[0][0]) for k in range(ROUNDS)]
    res[(a, c)] = float(b)
    say({"arm": a, "cell": c, "wall_slope_us": round(float(b), 3), "intercept_us": round(float(a0), 1),
         "per_round": [round(v, 3) for v in pr], "hoist_ratio": round(float(ys[-1] / ys[0]), 2)})
wall_net = {}
for cname, m, nv in C.CELLS:
    if ("fill", cname) in res:
        f = res[("fill", cname)]
        wall_net[cname] = {a: round(res[(a, c)] - f, 3) for (a, c) in res if c == cname and a != "fill"}
        say({"cell": cname, "wall_net_of_fill": wall_net[cname]})


def parse(path):
    d = json.load(gzip.open(path))
    ev = d["traceEvents"]
    procs = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
    threads = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev
               if e.get("ph") == "M" and e.get("name") == "thread_name"}
    tcs = [p for p, n in procs.items() if n == "/device:TPU:0"]
    if not tcs:
        return None
    tids = [t for (p, t), n in threads.items() if p == tcs[0] and n == "XLA Ops"]
    if not tids:
        return None
    return sorted([e for e in ev if e.get("pid") == tcs[0] and e.get("tid") == tids[0]
                   and e.get("ph") == "X"], key=lambda e: e["ts"])


spans, opsets = {}, {}
for rep in range(int(os.environ.get("TRACE_REPS", 2))):
    for (a, c), (fn, ops, rr) in units.items():
        tdir = f"/tmp/tr02b_{a}_{c}_{rep}"
        shutil.rmtree(tdir, ignore_errors=True)
        try:
            with jax.profiler.trace(tdir):
                jax.block_until_ready(fn(*ops, rr[R_SET[0]]))
        except Exception as e:
            say({"trace_error": a, "cell": c, "err": repr(e)[:300]})
            continue
        if not C.is_main():
            continue
        paths = sorted(glob.glob(f"{tdir}/**/*.trace.json.gz", recursive=True))
        evs = parse(paths[-1]) if paths else None
        if not evs:
            say({"trace": a, "cell": c, "err": "no TPU:0 XLA Ops"})
            continue
        by = collections.defaultdict(list)
        for e in evs:
            by[re.sub(r"\.\d+$", "", e["name"])].append(e["dur"])
        R = R_SET[0]
        span = (evs[-1]["ts"] + evs[-1]["dur"] - evs[0]["ts"]) / R
        spans.setdefault((a, c), []).append(span)
        rows = sorted(((n, round(len(d) / R, 2), round(st.median(d[2:] if len(d) > 4 else d), 3))
                       for n, d in by.items()), key=lambda t: -t[1] * t[2])
        opsets[(a, c)] = sorted(n for n, pc, _ in rows if pc >= 0.9)
        say({"trace": a, "cell": c, "rep": rep, "iter_span_us": round(span, 3),
             "ops": [dict(name=n, per_call=pc, med_us=md) for n, pc, md in rows[:10]]})
        if rep == 0 and os.environ.get("CDK_OUTPUT_DIR"):
            dst = os.path.join(os.environ["CDK_OUTPUT_DIR"], "traces", f"{a}_{c}.trace.json.gz")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(paths[-1], dst)
if C.is_main():
    summ = {}
    for cname, m, nv in C.CELLS:
        if ("fill", cname) not in spans:
            continue
        f = st.median(spans[("fill", cname)])
        net = {a: round(st.median(v) - f, 3) for (a, c), v in spans.items() if c == cname and a != "fill"}
        reps = {a: [round(x - f, 3) for x in v] for (a, c), v in spans.items() if c == cname and a != "fill"}
        # M4: non-kernel ops present in a region (>= 0.9 per call), diffed against inc
        kern = re.compile(r"fp8_skinny|small_all_reduce|fp8_oproj_ar")
        base = {o for o in opsets.get(("inc", cname), []) if not kern.search(o)}
        sym = {a: sorted(({o for o in opsets.get((a, cname), []) if not kern.search(o)} ^ base))
               for a in ARMS if a not in ("fill", "inc")}
        d = {a: round(net[a] - net["inc"], 3) for a in net if "inc" in net}
        summ[cname] = dict(fill_span=round(f, 3), net_span=net, span_reps=reps, delta_vs_inc=d,
                           wall_net=wall_net.get(cname), region_op_symdiff_vs_inc=sym)
    say({"SPAN_SUMMARY": summ})
    out = os.environ.get("CDK_OUTPUT_DIR")
    if out:
        json.dump(summ, open(os.path.join(out, "p02b_timing.json"), "w"))
say({"DONE": True})
