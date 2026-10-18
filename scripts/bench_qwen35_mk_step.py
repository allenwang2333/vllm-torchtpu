"""Standalone TPU microbenchmark of the Qwen3.5 decode/verify megakernel step (pure JAX).

Runs the production fused-pool-I/O verify kernel (`pool_io.make_flat_pooled_step`) on one
v7x-8 host (8 TensorCores, TP=EP=8) with synthetic weights / pools, at the serving geometry of
the 1P1D decode leg (EP mode, e_slots=68, DCP=4, T=4, page 128, max_pages_per_seq=515).

Weights are random fp8 (one expert-weight buffer per layer kind, shared by all 60 layers: HBM
traffic is the same as distinct buffers, TPU has no HBM cache). Routing is data-dependent
(router logits from random hidden rows); `--corr` makes the T rows of a sequence share their
hidden state (identical routing per sequence), the closest synthetic analog of MTP verify.

Usage (inside the server image, repo mapped at /workspace/vllm-torchtpu):
  python3 scripts/bench_qwen35_mk_step.py --b 16,32,48,64,80 --ctx 80000 --iters 20
  python3 scripts/bench_qwen35_mk_step.py --b 80 --skip experts      # ablation
  python3 scripts/bench_qwen35_mk_step.py --b 96 --tiles nbd=5        # tile override
  python3 scripts/bench_qwen35_mk_step.py --b 80 --xprof /tmp/xp      # jax.profiler trace

Prints one JSON line per configuration: {"B":..,"ms_mean":..,"ms_p50":..,...}.
"""

import argparse
import dataclasses
import json
import os
import statistics
import sys
import time
import types

REPO = os.environ.get("REPO", os.path.join(os.path.dirname(__file__), "..", "src"))
# Import only the kernel package (no torch / vllm imports from vllm_torchtpu/__init__).
for _name, _sub in [("vllm_torchtpu", ""), ("vllm_torchtpu.kernels", "kernels")]:
    _m = types.ModuleType(_name)
    _m.__path__ = [os.path.join(REPO, "vllm_torchtpu", _sub)]
    sys.modules[_name] = _m

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.qwen35_megakernel import config as C  # noqa: E402
from vllm_torchtpu.kernels.qwen35_megakernel import pool_io  # noqa: E402

AP = argparse.ArgumentParser()
AP.add_argument("--b", default="16,32,48,64,80", help="verify bucket rows (T=4 per seq)")
AP.add_argument("--T", type=int, default=4)
AP.add_argument("--ctx", type=int, default=80000, help="context tokens per sequence")
AP.add_argument("--iters", type=int, default=20)
AP.add_argument("--warm", type=int, default=3)
AP.add_argument("--skip", default="", help="comma list of ablations: experts,ar,attn,gdn,dense,kv")
AP.add_argument("--tiles", default="", help="comma list k=v Tiles overrides (int/bool)")
AP.add_argument("--cfg", default="", help="comma list k=v module-level config overrides")
AP.add_argument("--corr", type=int, default=1, help="1: T rows of a seq share hidden state")
AP.add_argument("--xprof", default="", help="jax.profiler trace dir (profiles 3 steps)")
AP.add_argument("--scopes", type=int, default=0)
AP.add_argument("--num_pages", type=int, default=8192)
AP.add_argument("--e_slots", type=int, default=68)
AP.add_argument("--dcp", type=int, default=4)
AP.add_argument("--tag", default="")
AP.add_argument("--nbd_search", default="", help="colon list of nbd to try (first that compiles)")
AP.add_argument("--router_scale", type=float, default=0.02,
                help="router weight std; 0 -> all rows pick experts 0..TOPK-1 (all on rank 0)")
AP.add_argument("--mode", default="ep", help="ep (EP experts, e_slots) or tp (TP-sliced experts)")
A = AP.parse_args()

for kv in filter(None, A.cfg.split(",")):
    k, v = kv.split("=")
    old = getattr(C, k)
    setattr(C, k, type(old)(int(v)) if not isinstance(old, tuple) else tuple(
        int(x) for x in v.split(":") if x))
    # context.py imported the names at module import; patch there too
    from vllm_torchtpu.kernels.qwen35_megakernel import context as _X
    if hasattr(_X, k):
        setattr(_X, k, getattr(C, k))

R = 8
ORDER = [0, 1, 4, 5, 6, 7, 2, 3]  # torch_tpu get_pallas_mesh order on tpu7x 2x2x1
devs = {d.id: d for d in jax.devices()}
mesh = jax.sharding.Mesh(np.asarray([devs[i] for i in ORDER]), ("tp",))
SH = NamedSharding(mesh, P("tp"))
print("devices:", [(d.id, getattr(d, "coords", None), getattr(d, "core_on_chip", None))
                   for d in mesh.devices.flat], flush=True)

MAX_PAGES = 515
NP = A.num_pages
base = dataclasses.replace(
    C.FULL_EP, R=R, moe_tp=(A.mode == "tp"), stack_e=False,
    e_slots=(A.e_slots if A.mode == "ep" else 0), paged_kv=True,
    fused_pool_io=True, dcp_size=A.dcp, num_pages=NP, max_pages_per_seq=MAX_PAGES,
    gdn_stride_pages=5)


def tile_over(t):
    kw = {}
    for kv in filter(None, A.tiles.split(",")):
        k, v = kv.split("=")
        old = getattr(t, k)
        kw[k] = (v not in ("0", "false", "False")) if isinstance(old, bool) else int(v)
    return dataclasses.replace(t, **kw)


def gen(shape, dtype, key, scale=1.0, kind="normal"):
    """Sharded random global array [R * shape[0], ...] built on device."""
    gshape = (R * shape[0],) + tuple(shape[1:])

    @jax.jit
    def f(k):
        if kind == "zeros":
            return jnp.zeros(gshape, dtype)
        x = jax.random.normal(k, gshape, jnp.float32) * scale
        return x.astype(dtype)

    return jax.jit(f, out_shardings=SH)(key)


def build_weights(c, tiles):
    ws = C.weight_shapes(c, tiles)
    key = jax.random.PRNGKey(0)
    lay, S = C.smalls_layout(c)
    out = []
    for i, k in enumerate(C.W_NAMES):
        key, sk = jax.random.split(key)
        if k in ("e13", "e2"):
            shp, dt = ws[k][0]
            one = gen(shp, dt, sk, 1.0)
            out.extend([one] * c.L)
            continue
        shp, dt = ws[k]
        if k == "smalls":
            row = np.zeros((S,), np.float32)
            def put(name, val):
                o, n = lay[name]
                row[o:o + n] = val
            put("in_s", 1.0 / 64)
            put("out_s", 1.0 / 32)
            put("sh13_s", 1.0 / 64)
            put("sh2_s", 1.0 / 11.3)
            put("conv", 0.25)
            put("gnorm", 1.0)
            arr = np.broadcast_to(row, (R * c.L, 1, S)).copy()
            out.append(jax.device_put(arr, SH))
        elif k == "es":
            row = np.full((2 * c.IR + c.H,), 1.0 / 64, np.float32)
            row[2 * c.IR:] = 1.0 / 32
            arr = np.broadcast_to(row, (R * shp[0],) + tuple(shp[1:])).copy()
            out.append(jax.device_put(arr, SH))
        elif k == "final_norm":
            out.append(gen(shp, dt, sk, kind="zeros"))
        elif k == "router":
            out.append(gen(shp, dt, sk, A.router_scale) if A.router_scale > 0
                       else gen(shp, dt, sk, kind="zeros"))
        else:
            out.append(gen(shp, dt, sk, 1.0))
    return out


def build_inputs(c, seed=1):
    B, T, NS, M = c.B, c.T, c.NS, c.M
    rng = np.random.default_rng(seed)
    ctxs = np.clip(rng.normal(A.ctx, A.ctx * 0.3, NS), 2048, MAX_PAGES * 128 * A.dcp - 16)
    ctxs = ctxs.astype(np.int64)
    pos = np.zeros((B,), np.int32)
    for s in range(NS):
        for t in range(T):
            pos[s * T + t] = ctxs[s] + t
    pos_g = np.tile(pos, R)
    hid = rng.normal(0, 1, (M, c.H)).astype(np.float32) * 0.5
    if A.corr:
        for s in range(NS):
            hid[s * T:(s + 1) * T] = hid[s * T] + 0.05 * rng.normal(0, 1, (T, c.H))
    hid[B:] = 0
    hid_g = np.tile(hid, (R, 1))
    # GDN slots: pages [5, 5 + 5 * 3 * NS * T); KV pages above.
    G_PER_ATT = 3
    ck = np.zeros((G_PER_ATT, NS, T), np.int32)
    slot = 1
    for g in range(G_PER_ATT):
        for s in range(NS):
            for t in range(T):
                ck[g, s, t] = slot
                slot += 1
    kv0 = (slot + 1) * 5
    pt = np.zeros((NS, MAX_PAGES), np.int32)
    span = NP - kv0
    for s in range(NS):
        n_loc = -(-int(ctxs[s] + T) // (128 * A.dcp))
        pt[s, :n_loc] = kv0 + (s * 211 + np.arange(n_loc)) % span
    ro = np.zeros((G_PER_ATT, NS), np.int32)
    put = lambda a: jax.device_put(a, SH)  # noqa: E731
    return (put(pos_g), put(jnp.asarray(hid_g, jnp.bfloat16)), put(np.tile(pt, (R, 1))),
            put(np.tile(ck, (R, 1, 1))), put(np.tile(ro, (R, 1))))


pools = None


def run(B):
    global pools
    c = dataclasses.replace(base, B=B, T=A.T)
    tiles = tile_over(C.default_tiles(c))
    skip = frozenset(filter(None, A.skip.split(",")))
    step, in_specs, out_specs, donate = pool_io.make_flat_pooled_step(
        mesh, c, tiles, fused_pool_io=True, out_rows=c.M, skip=skip, scopes=A.scopes)
    f = jax.jit(step, donate_argnums=donate)
    t0 = time.time()
    weights = build_weights(c, tiles)
    if pools is None:
        pools = [gen((NP, 2, 64, 4, 128), jnp.float8_e4m3fn, None, kind="zeros")
                 for _ in range(c.NATT)]
    pos, hid, pt, ck, ro = build_inputs(c)
    jax.block_until_ready(weights)
    t1 = time.time()
    print(f"[B={B}] compiling (nbd={tiles.nbd})...", flush=True)
    compiled = f.lower(pos, hid, *weights, *pools, pt, ck, ro).compile()
    print(f"[B={B}] compiled in {time.time() - t1:.1f}s; first run...", flush=True)
    outs = compiled(pos, hid, *weights, *pools, pt, ck, ro)
    jax.block_until_ready(outs)
    print(f"[B={B}] first run done", flush=True)
    pools = list(outs[1:])
    f = compiled
    outs = f(pos, hid, *weights, *pools, pt, ck, ro)
    jax.block_until_ready(outs)
    t2 = time.time()
    pools = list(outs[1:])
    h0 = np.asarray(outs[0][:B].astype(jnp.float32))
    finite = bool(np.isfinite(h0).all())
    import hashlib
    h0_md5 = hashlib.md5(np.ascontiguousarray(h0).tobytes()).hexdigest()[:12]
    for _ in range(A.warm):
        outs = f(pos, hid, *weights, *pools, pt, ck, ro)
        pools = list(outs[1:])
    jax.block_until_ready(outs)
    times = []
    for _ in range(A.iters):
        ts = time.perf_counter()
        outs = f(pos, hid, *weights, *pools, pt, ck, ro)
        pools = list(outs[1:])
        jax.block_until_ready(outs[0])
        times.append((time.perf_counter() - ts) * 1e3)
    # pipelined throughput (no host sync per step)
    ts = time.perf_counter()
    for _ in range(A.iters):
        outs = f(pos, hid, *weights, *pools, pt, ck, ro)
        pools = list(outs[1:])
    jax.block_until_ready(outs)
    piped = (time.perf_counter() - ts) * 1e3 / A.iters
    if A.xprof:
        jax.profiler.start_trace(A.xprof)
        for _ in range(3):
            outs = f(pos, hid, *weights, *pools, pt, ck, ro)
            pools = list(outs[1:])
        jax.block_until_ready(outs)
        jax.profiler.stop_trace()
    times.sort()
    res = {
        "tag": A.tag, "B": B, "T": A.T, "ctx": A.ctx, "skip": sorted(skip), "tiles": A.tiles,
        "cfg": A.cfg, "nbd": tiles.nbd, "nbe": tiles.nbe, "hier": tiles.hier,
        "ms_mean": round(statistics.mean(times), 3), "ms_p50": round(times[len(times) // 2], 3),
        "ms_min": round(times[0], 3), "ms_piped": round(piped, 3), "finite": finite, "md5": h0_md5,
        "compile_s": round(t2 - t1, 1), "setup_s": round(t1 - t0, 1),
    }
    print("RESULT " + json.dumps(res), flush=True)
    del weights


def run_search(b):
    base_tiles = A.tiles
    for nbd in [int(x) for x in A.nbd_search.split(":")]:
        A.tiles = ",".join(filter(None, [base_tiles, f"nbd={nbd}"]))
        try:
            run(b)
            return
        except Exception as e:  # noqa
            msg = repr(e)
            print(f"[B={b}] nbd={nbd} failed: {msg[:200]}", flush=True)
            if "vmem" not in msg.lower() and "Vmem" not in msg:
                raise
        finally:
            A.tiles = base_tiles
    raise RuntimeError("no nbd fits")


for b in [int(x) for x in A.b.split(",") if x]:
    try:
        run_search(b) if A.nbd_search else run(b)
    except Exception as e:  # keep going to the next bucket
        import traceback
        traceback.print_exc()
        print("RESULT " + json.dumps({"tag": A.tag, "B": b, "error": repr(e)[:800]}), flush=True)
