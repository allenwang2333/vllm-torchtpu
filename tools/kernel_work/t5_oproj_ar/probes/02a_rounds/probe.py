"""t5-oproj-ar probe 02a: decompose the 16-rank data round (fact arms, no candidate).

Every arm is one pallas_call (grid (), rank as an SMEM operand, collective_id 5) of the form
    barrier (signal 16 incl. self, wait 16; EXPOSED, identical in every arm)
    o_ref = x tiled 6x  (the [M, 6144] push source, as R1's o_ref windows)
    <arm body>
and is timed in the instruments' filler-separated chain (x = rmsnorm(h @ Wd); h = arm(x)),
device-trace iteration span on rank 0 (primary clock, net of `fill`), wall slope over R 32/256/1024
as the cross-check. One data round = k remote pushes of an [M, 384] bf16 chunk (the two-shot chunk)
to k peers, the k inbound chunks counted by one recv semaphore (aggregated wait) unless the arm
says otherwise, the k sends drained, the k slots f32-summed into o_ref[:, 0:384] (live sink).

Peers: rank r's i-th peer is r XOR masks[i]; it writes into the peer's slot i, so every rank sends
k and receives k (symmetric). Mask sets are chosen at run time from jax.devices() attributes
(process_index, coords, core_on_chip) and asserted uniform over all 16 ranks:
  c1     1 peer, same chip (the other TensorCore)
  h1     1 peer, same host, other chip
  x1     1 peer, other host
  h7     7 peers, every same-host peer
  x8     8 peers, every other-host peer
  chip7  7 peers, every other chip, same core index (the hierarchical exchange's cross-chip round)
  a15    15 peers (all), = one two-shot round with a direct source
Arms:
  bar        barrier only (the subtrahend)
  bar2       a second barrier round on the barrier semaphore (a signal-only round, no DMA)
  d_<set>    one data round, direct source, aggregated wait
  a15_pw     a15 with one recv semaphore per slot and 15 waits (+15 send waits) -- moe_tp_decode form
  a15_st     a15 with the 15 chunks first staged by vector copy into a scratch (incumbent `mine`)
  self15     15 LOCAL VMEM->VMEM DMAs (make_async_copy), aggregated wait: DMA issue+completion, no ICI
  inc1r      the incumbent two-shot round 1 verbatim in form (stage 16, 15 pushes under pl.when j!=me,
             own-slot copy, aggregated wait, f32 sum into `gathered`), sends drained
  inc2r      inc1r + round 2 (15 pushes of the summed chunk, wait, 16 copies out) = the exchange
             of Job #1's stream_ar arm
usage: python probe.py   (16 ranks, 2x2x2, both pods)
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
import common as C  # noqa: E402

C.init_distributed()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

MESH = pl.DeviceIdType.MESH
CID = 5
mesh, N = C.make_mesh()
AX = C.AXIS
SH = NamedSharding(mesh, P(AX))
R_SET = (32, 256, 1024)
REPS = int(os.environ.get("REPS", 7))
ROUNDS = int(os.environ.get("ROUNDS", 3))
OWN = C.N_OUT // N  # 384
CELLS = [("b1", 4), ("b4", 16)]


def say(d):
    if C.is_main():
        print(json.dumps(d), flush=True)


# ------------------------------------------------------------------ peer classes from devices
devs = list(mesh.devices.reshape(-1))
info = [(d.process_index, tuple(getattr(d, "coords", (0, 0, 0))), getattr(d, "core_on_chip", 0))
        for d in devs]
say({"devices": [dict(rank=r, proc=p, coords=c, core=k) for r, (p, c, k) in enumerate(info)]})


def cls(r, q):
    (p1, c1, k1), (p2, c2, k2) = info[r], info[q]
    if c1 == c2 and p1 == p2:
        return "chip"
    return "host" if p1 == p2 else "xhost"


def uniform(m):
    s = {cls(r, r ^ m) for r in range(N)}
    return s.pop() if len(s) == 1 else None


def same_core(m):
    return all(info[r][2] == info[r ^ m][2] for r in range(N))


MASK_CLS = {m: uniform(m) for m in range(1, N)}
SETS = {}
_c = [m for m, c in MASK_CLS.items() if c == "chip"]
_h = [m for m, c in MASK_CLS.items() if c == "host"]
_x = [m for m, c in MASK_CLS.items() if c == "xhost"]
if _c:
    SETS["c1"] = _c[:1]
if _h:
    SETS["h1"] = _h[:1]
if _x:
    SETS["x1"] = _x[:1]
if len(_c) + len(_h) == 7:
    SETS["h7"] = sorted(_c + _h)
if len(_x) == 8:
    SETS["x8"] = sorted(_x)
_chip7 = [m for m in range(1, N) if MASK_CLS[m] in ("host", "xhost") and same_core(m)]
if len(_chip7) == 7:
    SETS["chip7"] = _chip7
if all(MASK_CLS[m] for m in range(1, N)):
    SETS["a15"] = list(range(1, N))
say({"mask_classes": MASK_CLS, "sets": SETS})


# ------------------------------------------------------------------ kernels
def _peer(q):
    return {AX: q}


def _signal_all():
    sem = pltpu.get_barrier_semaphore()
    for j in range(N):
        pl.semaphore_signal(sem, 1, device_id=_peer(jnp.int32(j)), device_id_type=MESH)


def _barrier():
    _signal_all()
    pl.semaphore_wait(pltpu.get_barrier_semaphore(), N)


def _wait_bytes(view, sem):
    pltpu.make_async_copy(view, view, sem).wait()


def _tile6(x_ref, o_ref):
    k = x_ref.shape[1]
    for j in range(o_ref.shape[1] // k):
        o_ref[:, pl.ds(j * k, k)] = x_ref[...]


def _chunk(o_ref, i):
    return o_ref.at[:, pl.ds(i * OWN, OWN)]


def _sink(o_ref, slots, k):
    acc = o_ref[:, pl.ds(0, OWN)].astype(jnp.float32)
    for i in range(k):
        acc = acc + slots[i].astype(jnp.float32)
    o_ref[:, pl.ds(0, OWN)] = acc.astype(o_ref.dtype)


_CALLS = {}


def build(arm, m):
    key = (arm, m)
    if key in _CALLS:
        return _CALLS[key]
    kind = arm.split("_", 1)[0] if arm.startswith("d_") else arm
    masks = SETS.get(arm[2:]) if arm.startswith("d_") else SETS.get("a15")
    k = len(masks) if masks else 0
    scratch = [pltpu.VMEM((N, m, OWN), C.BF16)] * 3 + [
        pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA((N,)), pltpu.SemaphoreType.DMA((N,))]

    def kernel(rank_ref, x_ref, o_ref, recv, mine, gathered, s1, r1, s2, r2, sa, ra):
        me = rank_ref[0]
        _barrier()
        _tile6(x_ref, o_ref)
        if arm == "bar":
            return
        if arm == "bar2":
            _barrier()
            return
        if kind == "d" or arm in ("a15_st",):
            src = mine if arm == "a15_st" else None
            if src is not None:
                for i in range(k):
                    mine[i] = o_ref[:, pl.ds((i + 1) * OWN, OWN)]
            for i, msk in enumerate(masks):
                s = src.at[i] if src is not None else _chunk(o_ref, i + 1)
                pltpu.make_async_remote_copy(
                    src_ref=s, dst_ref=recv.at[i], send_sem=s1, recv_sem=r1,
                    device_id=_peer(jnp.bitwise_xor(me, msk)), device_id_type=MESH).start()
            _wait_bytes(recv.at[pl.ds(0, k)], r1)
            _sink(o_ref, recv, k)
            _wait_bytes(recv.at[pl.ds(0, k)], s1)
            return
        if arm == "a15_pw":
            for i, msk in enumerate(masks):
                pltpu.make_async_remote_copy(
                    src_ref=_chunk(o_ref, i + 1), dst_ref=recv.at[i], send_sem=sa.at[i],
                    recv_sem=ra.at[i], device_id=_peer(jnp.bitwise_xor(me, msk)),
                    device_id_type=MESH).start()
            for i in range(k):
                _wait_bytes(recv.at[i], ra.at[i])
            _sink(o_ref, recv, k)
            for i in range(k):
                _wait_bytes(recv.at[i], sa.at[i])
            return
        if arm == "self15":
            for i in range(k):
                pltpu.make_async_copy(_chunk(o_ref, i + 1), recv.at[i], r1).start()
            _wait_bytes(recv.at[pl.ds(0, k)], r1)
            _sink(o_ref, recv, k)
            return
        if arm in ("inc1r", "inc2r"):  # small_all_reduce._twoshot rounds, same form
            for j in range(N):
                mine[j] = o_ref[:, pl.ds(j * OWN, OWN)]
            for j in range(N):

                @pl.when(jnp.int32(j) != me)
                def _(j=j):
                    pltpu.make_async_remote_copy(
                        src_ref=mine.at[j], dst_ref=recv.at[me], send_sem=s1, recv_sem=r1,
                        device_id=_peer(jnp.int32(j)), device_id_type=MESH).start()

            recv[pl.ds(me, 1)] = mine[pl.ds(me, 1)]
            _wait_bytes(recv.at[pl.ds(0, N - 1)], r1)
            acc = recv[0].astype(jnp.float32)
            for j in range(1, N):
                acc = acc + recv[j].astype(jnp.float32)
            gathered[pl.ds(me, 1)] = acc.astype(gathered.dtype)[None]
            if arm == "inc2r":
                for j in range(N):

                    @pl.when(jnp.int32(j) != me)
                    def _(j=j):
                        pltpu.make_async_remote_copy(
                            src_ref=gathered.at[me], dst_ref=gathered.at[me], send_sem=s2,
                            recv_sem=r2, device_id=_peer(jnp.int32(j)),
                            device_id_type=MESH).start()

                _wait_bytes(gathered.at[pl.ds(0, N - 1)], r2)
                for j in range(N):
                    o_ref[:, pl.ds(j * OWN, OWN)] = gathered[j]
                _wait_bytes(gathered.at[pl.ds(0, N - 1)], s2)
            else:
                o_ref[:, pl.ds(0, OWN)] = gathered[me]
            _wait_bytes(recv.at[pl.ds(0, N - 1)], s1)
            return
        raise ValueError(arm)

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, C.N_OUT), C.BF16),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=CID),
        name=f"p02a_{arm}_m{m}",
    )
    _CALLS[key] = call
    return call


def driver(arm):
    """DMAs per call (remote, local) asserted from the same tables the kernels read (M6)."""
    if arm in ("bar", "bar2", "fill"):
        return (0, 0)
    if arm.startswith("d_"):
        return (len(SETS[arm[2:]]), 0)
    return {"a15_pw": (15, 0), "a15_st": (15, 0), "self15": (0, 15), "inc1r": (15, 1),
            "inc2r": (30, 1)}[arm]


ARMS = ["fill", "bar", "bar2"] + [f"d_{s}" for s in ("c1", "h1", "x1", "h7", "x8", "chip7", "a15")
                                  if s in SETS]
if "a15" in SETS:
    ARMS += ["a15_pw", "a15_st", "self15"]
ARMS += ["inc1r", "inc2r"]
say({"arms": ARMS, "driver_remote_local_dmas": {a: driver(a) for a in ARMS}})


def put(a):
    return jax.make_array_from_callback(a.shape, SH, lambda i: a[i])


def tile6(x):
    return jnp.concatenate([x] * (C.N_OUT // C.K_IN), axis=1)


def arm_fn(arm, m):
    def call(x, rank):
        if arm == "fill":
            return tile6(x)
        return build(arm, m)(rank.astype(jnp.int32), x)

    def body(h, wd, rank, r):
        def it(i, hh):
            y = jnp.dot(hh, wd, preferred_element_type=jnp.float32)
            x = (y * jax.lax.rsqrt(jnp.mean(y * y, axis=1, keepdims=True) + 1e-6)).astype(C.BF16)
            return call(x, rank).astype(hh.dtype)

        return jax.lax.fori_loop(0, r[0], it, h)

    spec = P(AX)
    return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(spec,) * 4, out_specs=spec,
                                 check_vma=False))


rng = np.random.default_rng(5)
units = {}
for cname, m in CELLS:
    h0 = np.asarray(jnp.asarray(rng.standard_normal((N * m, C.N_OUT), dtype=np.float32), C.BF16))
    wd = np.asarray(jnp.asarray(rng.standard_normal((N * C.N_OUT, C.K_IN), dtype=np.float32)
                                / np.sqrt(C.N_OUT), C.BF16))
    ops = [put(h0), put(wd), put(np.arange(N, dtype=np.int32))]
    for a in ARMS:
        t = time.time()
        try:
            fn = arm_fn(a, m)
            rr = {r: put(np.full((N,), r, np.int32)) for r in R_SET}
            comp = fn.lower(*ops, rr[R_SET[0]]).compile()
            cc = comp.as_text().count('custom_call_target="tpu_custom_call"')
            jax.block_until_ready(comp(*ops, rr[R_SET[0]]))
            units[(a, cname)] = (comp, ops, rr)
            say({"compiled": a, "cell": cname, "s": round(time.time() - t, 1), "custom_calls": cc,
                 "expected": 0 if a == "fill" else 1})
        except Exception as e:
            say({"compile_error": a, "cell": cname, "err": repr(e)[:800]})

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
for cname, m in CELLS:
    if ("fill", cname) in res:
        f = res[("fill", cname)]
        say({"cell": cname, "wall_net_of_fill": {a: round(res[(a, c)] - f, 3) for (a, c) in res
                                                 if c == cname and a != "fill"}})


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


spans = {}
for (a, c), (fn, ops, rr) in units.items():
    tdir = f"/tmp/tr02a_{a}_{c}"
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
    spans[(a, c)] = span
    ops_rows = sorted(((n, round(len(d) / R, 2), round(st.median(d[2:] if len(d) > 4 else d), 3))
                       for n, d in by.items()), key=lambda t: -t[1] * t[2])[:8]
    say({"trace": a, "cell": c, "iter_span_us": round(span, 3),
         "ops": [dict(name=n, per_call=pc, med_us=md) for n, pc, md in ops_rows]})
if C.is_main():
    summ = {}
    for cname, m in CELLS:
        f = spans.get(("fill", cname))
        if f is None:
            continue
        net = {a: round(spans[(a, c)] - f, 3) for (a, c) in spans if c == cname and a != "fill"}
        b = net.get("bar")
        summ[cname] = dict(fill_span=round(f, 3), net_span=net,
                           minus_bar={a: round(v - b, 3) for a, v in net.items()} if b is not None else None)
    say({"SPAN_SUMMARY": summ})
    out = os.environ.get("CDK_OUTPUT_DIR")
    if out:
        json.dump(dict(spans={f"{a}|{c}": v for (a, c), v in spans.items()},
                       wall={f"{a}|{c}": v for (a, c), v in res.items()}, sets=SETS,
                       mask_classes=MASK_CLS, summary=summ),
                  open(os.path.join(out, "p02a.json"), "w"))
say({"DONE": True})
