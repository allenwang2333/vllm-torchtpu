"""t5-oproj-ar instrument v1.2 addition (WHITEBOARD.v1 §5 candidate-specific faults): R1 candidates and
their injected faults, single-call cells and chained cells, judged as test_accuracy.py judges.

Single-call cells (common.CELLS randn + edge, EDGE_CELLS randn), every call preceded by
fact_kernels.vmem_poison (NaN in 24 MiB of VMEM: a read of never-written scratch reads NaN):
  correct:  L1, L2, Rbar                      must be bitwise == incumbent on valid rows, agree, oracle, pad finite
  faults:   f_straddle   L2 pushing owners 5 and 10 after their FIRST tile (stale columns 2048-2303 / 4096-4223)
            f_norswait   L1 summing before the RS bytes are counted (reads NaN-poisoned / in-flight slots)
            f_early      L1 pushing the RS before the barrier wait (cross-call race; single call usually benign)
Chained cells (CH calls in one jit, x_{k+1} = out_k[:, :K]; the incumbent chain is the reference; every
intermediate output compared bitwise): run once WITHOUT poison between calls (races can manifest) and once
WITH poison (stale reads visible) for L1, L2, Rbar, f_early, f_norswait.
usage: python acc_cand.py
"""
import json
import os
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
from jax.experimental import multihost_utils  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

import fact_kernels as FK  # noqa: E402
import fused_oproj_ar as F  # noqa: E402

mesh, N = C.make_mesh()
AX = C.AXIS
SH = NamedSharding(mesh, P(AX))
CH = int(os.environ.get("CHAIN", 8))


def say(d):
    if C.is_main():
        print(json.dumps(d), flush=True)


say({"n": N, "prov": C.provenance(), "cand_md5": hashlib.md5(open(F.__file__, "rb").read()).hexdigest()})
VAR = {"L1": {}, "L2": dict(waves=True), "Rbar": dict(barrier="exposed"),
       "f_straddle": dict(waves=True, fault="straddle"), "f_norswait": dict(fault="no_rs_wait"),
       "f_early": dict(fault="early_push")}
CORRECT = ["L1", "L2", "Rbar"]
FAULTS = ["f_straddle", "f_norswait", "f_early"]


def one(name, x, w, ws, rank, nv, idx):
    if name == "inc":
        return C.incumbent(x, w, idx, ws, rank, nv, num_cores=N, axis=AX)
    return F.fused_oproj_all_reduce(x, w, idx, ws, rank, nv, num_cores=N, axis=AX, **VAR[name])


_fns = {}


def get(name, chain, poison):
    key = (name, chain, poison)
    if key in _fns:
        return _fns[key]

    def body(x, w, ws, rank, nv, idx):
        outs = []
        for _ in range(chain):
            if poison:
                x = FK.vmem_poison(x)
            o = one(name, x, w, ws, rank, nv, idx)
            outs.append(o)
            x = o[:, : C.K_IN]
        return jnp.stack(outs)

    spec = P(AX)
    _fns[key] = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(spec,) * 6, out_specs=P(None, AX),
                                      check_vma=False))
    return _fns[key]


def put(a):
    return jax.make_array_from_callback(a.shape, SH, lambda i: a[i])


def gather(y):
    return np.asarray(multihost_utils.process_allgather(y, tiled=True))


def judge(out, ref, S, tol, m, nv):
    v = slice(0, nv)
    ob, rb = C.to_bits(out[:, v]), C.to_bits(ref[:, v])
    bitwise = bool(np.array_equal(ob, rb))
    agree = bool(all(np.array_equal(ob[0], ob[r]) for r in range(N)))
    of = out[:, v].astype(np.float64)
    finite = bool(np.isfinite(of).all())
    oracle = bool(finite and (np.abs(of - S[None, v]) <= tol[None, v]).all()) if S is not None else None
    pad = bool(np.isfinite(out[:, nv:].astype(np.float64)).all()) if nv < m else True
    ok = bitwise and agree and pad and (oracle is not False)
    return dict(PASS=ok, bitwise=bitwise, agree=agree, oracle=oracle, pad=pad,
                bits_diff=int(np.count_nonzero(ob != rb)))


results = []
t0 = time.time()
cells = [(c, "randn") for c in C.CELLS] + [(c, "edge") for c in C.CELLS] + [(c, "randn") for c in C.EDGE_CELLS]
for (cname, m, nv), kind in cells:
    seed = 1000 + m * 31 + nv + (7 if kind == "edge" else 0)
    xb, wq, ws = C.rank_inputs(N, m, seed, kind=kind)
    S, tol = C.oracle(xb, wq, ws)
    ops = (put(xb.reshape(N * m, C.K_IN)), put(wq.reshape(N, C.K_IN, C.N_OUT)), put(ws.reshape(N * C.N_OUT)),
           put(np.arange(N, dtype=np.int32)), put(np.full((N,), nv, np.int32)), put(np.zeros((N,), np.int32)))
    cls = "production" if cname.startswith("b") else "edge"
    ref = gather(get("inc", 1, True)(*ops)).reshape(N, m, C.N_OUT)
    for a in CORRECT + FAULTS:
        try:
            out = gather(get(a, 1, True)(*ops)).reshape(N, m, C.N_OUT)
        except Exception as e:
            say({"cell": cname, "kind": kind, "arm": a, "error": repr(e)[:1500]})
            continue
        r = dict(test="single", cell=cname, cls=cls, kind=kind, arm=a, **judge(out, ref, S, tol, m, nv))
        results.append(r)
        say(r)
    if kind == "randn" and cls == "production":
        for poison in (False, True):
            refc = gather(get("inc", CH, poison)(*ops))
            for a in CORRECT + ["f_early", "f_norswait"]:
                try:
                    outc = gather(get(a, CH, poison)(*ops))
                except Exception as e:
                    say({"cell": cname, "chain": CH, "poison": poison, "arm": a, "error": repr(e)[:1500]})
                    continue
                outc = outc.reshape(CH, N, m, C.N_OUT)
                rc = refc.reshape(CH, N, m, C.N_OUT)
                per = [judge(outc[k], rc[k], None, None, m, nv) for k in range(CH)]
                r = dict(test="chain", poison=poison, cell=cname, cls=cls, kind=kind, arm=a,
                         PASS=all(p["PASS"] for p in per), calls_failed=sum(not p["PASS"] for p in per),
                         bits_diff=sum(p["bits_diff"] for p in per))
                results.append(r)
                say(r)

summary = {}
for a in CORRECT + FAULTS:
    for test in ("single", "chain"):
        rr = [r for r in results if r["arm"] == a and r["test"] == test]
        if not rr:
            continue
        for cls in ("production", "edge"):
            q = [r for r in rr if r["cls"] == cls]
            if q:
                key = f"{a}/{test}/{cls}"
                summary[key] = (dict(passed=sum(r["PASS"] for r in q), of=len(q)) if a in CORRECT
                                else dict(caught=sum(not r["PASS"] for r in q), of=len(q)))
cand_ok = all(r["PASS"] for r in results if r["arm"] in CORRECT)
say({"SUMMARY": summary, "candidates_ok": cand_ok, "seconds": round(time.time() - t0, 1)})
if C.is_main() and os.environ.get("CDK_OUTPUT_DIR"):
    json.dump(dict(results=results, summary=summary), open(os.path.join(os.environ["CDK_OUTPUT_DIR"], "acc_cand.json"), "w"))
say({"DONE": True, "ok": cand_ok})
