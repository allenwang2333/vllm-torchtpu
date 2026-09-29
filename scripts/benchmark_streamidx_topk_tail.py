"""Probe 18: segmented SparseCore stage 2 (topk_tail="seg") at layer scope.

  python scripts/benchmark_streamidx_topk_tail.py time    interleaved inc/seg arms, bit-exact diffs
  python scripts/benchmark_streamidx_topk_tail.py trace   xprof trace of L_inc / L_seg (needs jax>=0.11)
  python scripts/benchmark_streamidx_topk_tail.py hlo     StableHLO text of the incumbent layer (fidelity diff)
  python scripts/benchmark_streamidx_topk_tail.py counts  per-slice stage-1 live counts (synthetic + P18_DUMPS)

P18_DUMP=<npz> (time/trace) replaces the synthetic indexer inputs with a server dump
(probe_topk IDXIN: q bytes, weights, gathered K pages, meta); MLA inputs stay synthetic.
P18_BF16G sets score_bf16_group (default 0; serving uses 4).

Same layer as probe 16 (benchmark_streamidx_early_exit_overlap.py): indexer + q-glue +
sparse-MLA attention, 1024 query tokens, one sequence of P18_SEQ tokens, chunk_tokens=128,
sctopk_l1=True. Arms differ only in topk_tail ("" incumbent, "seg") and enable_early_exit.
"""
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "time"
OUT = os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "p18", MODE)
os.makedirs(OUT, exist_ok=True)
if MODE == "trace" or os.environ.get("P18_TRACE"):
    os.environ["LIBTPU_INIT_ARGS"] = (os.environ.get("LIBTPU_INIT_ARGS", "")
                                      + " --xla_enable_custom_call_region_trace=true").strip()
if os.environ.get("P18_FLAGS"):
    os.environ["GLM5_LIBTPU_EXTRA_ARGS"] = (os.environ.get("GLM5_LIBTPU_EXTRA_ARGS", "") + " " + os.environ["P18_FLAGS"]).strip()
import vllm_torchtpu.env_override  # noqa: E402,F401  (same libtpu flags as serving)

import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk  # noqa: E402
from vllm_torchtpu.kernels.mla.sparse import kernel as mla  # noqa: E402


def log(*a):
    print(*a, flush=True)


H, DN, DR, DL, TOPK = 64, 192, 64, 512, 2048
IH, ID = 32, 128
PAGE = 1024
SM = 1.0 / (256 ** 0.5)
RES = {}


def problem(n_tok, seq_len, seed=0, idx_pages=384, mla_tokens=262144):
    rng = np.random.default_rng(seed)
    # indexer cache: per token 128 fp8 bytes + 1 e8m0 scale byte (127 = 1.0), padded to 256
    kb = rng.integers(0, 256, size=(idx_pages, PAGE, 128), dtype=np.uint8)
    kb &= np.uint8(0xBE)  # finite fp8 (no 0x7f/0xff), |x| small
    cache = np.zeros((idx_pages, PAGE, 256), np.uint8)
    cache[..., :128] = kb
    cache[..., 128] = 127
    cache = cache.reshape(idx_pages, PAGE // 4, 4, 256)
    ppseq = 257
    pages_needed = -(-seq_len // PAGE)
    pi = np.zeros((ppseq,), np.int32)
    pi[:pages_needed] = rng.permutation(idx_pages)[:pages_needed]
    iq = rng.normal(size=(n_tok, IH, ID)).astype(np.float32)
    w = rng.normal(size=(n_tok, IH)).astype(np.float32)
    nope = rng.integers(0, 2**32, size=(mla_tokens, 128), dtype=np.uint64).astype(np.uint32)
    rope = rng.integers(0, 2**32, size=(mla_tokens // 4, 128), dtype=np.uint64).astype(np.uint32)
    nope &= np.uint32(0xBEBEBEBE)
    rope &= np.uint32(0xBEBEBEBE)
    qb = rng.normal(size=(n_tok, H, DN + DR)).astype(np.float32)
    wuk = (rng.normal(size=(H, DN, DL)) / np.sqrt(DN)).astype(np.float32)
    pos = np.arange(seq_len - n_tok, seq_len)
    inv = 1.0 / (10000 ** (np.arange(0, DR, 2) / DR))
    ang = pos[:, None] * inv[None]
    return dict(
        iq=jnp.asarray(iq, jnp.float8_e4m3fn), w=jnp.asarray(w),
        cache=jnp.asarray(cache), seq_lens=jnp.array([seq_len], jnp.int32),
        pi=jnp.asarray(pi), cu=jnp.array([0, n_tok], jnp.int32),
        dist=jnp.array([0, 0, 1], jnp.int32),
        qb=jnp.asarray(qb, jnp.bfloat16), wuk=jnp.asarray(wuk, jnp.bfloat16),
        cos=jnp.asarray(np.cos(ang), jnp.float32), sin=jnp.asarray(np.sin(ang), jnp.float32),
        nope=jnp.asarray(nope.reshape(-1, 256, 128)), rope=jnp.asarray(rope.reshape(-1, 64, 128)))


IDX_KEYS = ("iq", "w", "cache", "seq_lens", "pi", "cu", "dist")
LAYER_KEYS = IDX_KEYS + ("qb", "wuk", "cos", "sin", "nope", "rope")


BF16G = int(os.environ.get("P18_BF16G", "0"))


def problem_from_dump(path, seed=0, mla_tokens=262144):
    """Server dump -> the same dict as problem(); indexer inputs real, MLA synthetic."""
    z = np.load(path)
    meta = z["meta"]
    seq_len, n_tok = int(meta[0]), int(z["num_tokens"])
    p = problem(n_tok, seq_len, seed=seed, idx_pages=-(-seq_len // PAGE), mla_tokens=mla_tokens)
    k = z["k"]
    npg = k.shape[0]
    pi = np.zeros((257,), np.int32)
    pi[:npg] = np.arange(npg)
    wd = str(z["wdtype"])
    p.update(
        iq=jnp.asarray(z["q"]).view(jnp.float8_e4m3fn),
        w=jnp.asarray(z["w"], jnp.bfloat16 if wd == "bfloat16" else jnp.float32),
        cache=jnp.asarray(k), seq_lens=jnp.array([seq_len], jnp.int32), pi=jnp.asarray(pi),
        cu=jnp.array([0, n_tok], jnp.int32), dist=jnp.asarray(meta[3:6].astype(np.int32)))
    p["server_topk"] = z["topk"]
    log("DUMP", os.path.basename(path), "seq", seq_len, "n", n_tok, "npg", npg, "wdtype", wd,
        "meta", meta.tolist(), "q", z["q"].shape, "w", z["w"].shape)
    return p


def _topk(iq, w, cache, seq_lens, pi, cu, dist, ee, tail=""):
    kw = {} if tail is None else dict(topk_tail=tail)
    return streamindex_topk(
        iq, w, cache, seq_lens, pi, cu, dist, k=TOPK, compression_ratio=1,
        num_kv_pages_per_block=(2, 2, 2), num_queries_per_block=(1, 128, 128),
        enable_early_exit=ee, chunk_tokens=128, sctopk_l1=True, score_bf16_group=BF16G, **kw)


def _qglue(qb, wuk, cos, sin):
    q_nope, q_pe = qb[..., :DN], qb[..., DN:]
    ql = jnp.einsum("thd,hdc->thc", q_nope, wuk, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    x = q_pe.astype(jnp.float32)
    x1, x2 = x[..., : DR // 2], x[..., DR // 2:]
    c, s = cos[:, None, :], sin[:, None, :]
    pe = jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1).astype(jnp.bfloat16)
    return jnp.concatenate([ql, pe], -1)


def idx_fn(ee, tail=""):
    @jax.jit
    def f(iq, w, cache, seq_lens, pi, cu, dist):
        return _topk(iq, w, cache, seq_lens, pi, cu, dist, ee, tail)
    return f


def layer_fn(ee, tail=""):
    @jax.jit
    def f(iq, w, cache, seq_lens, pi, cu, dist, qb, wuk, cos, sin, nope, rope):
        topk = _topk(iq, w, cache, seq_lens, pi, cu, dist, ee, tail)
        q = _qglue(qb, wuk, cos, sin)
        out = mla.sparse_ragged_paged_attention(
            q, nope, rope, topk, None, cu, dist, sm_scale=SM,
            gather_and_attention_chunk_size=64, attention_kernel_batch_size=16,
            cache_layout="sparsecore")
        return out, topk
    return f


def timeit(fn, args, iters=30, warm=3):
    for _ in range(warm):
        jax.block_until_ready(fn(*args))
    s = []
    for _ in range(iters):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        s.append((time.perf_counter() - t) * 1e3)
    return statistics.median(s), min(s)


def ndiff(a, b):
    a = np.asarray(jax.device_get(a))
    b = np.asarray(jax.device_get(b))
    if a.dtype == jnp.bfloat16:
        a, b = a.view(np.uint16), b.view(np.uint16)
    return int((a != b).sum()), int(a.size)


def cmp(out, ref):
    if isinstance(out, tuple):
        return {f"diff{i}": ndiff(o, r) for i, (o, r) in enumerate(zip(out, ref))}
    return {"diff0": ndiff(out, ref)}


def arm(name, fn, args, ref=None, iters=30, timed=True):
    try:
        out = jax.block_until_ready(fn(*args))
        r = {}
        if timed:
            med, mn = timeit(fn, args, iters=iters)
            r.update(ms_med=round(med, 4), ms_min=round(mn, 4))
        if ref is not None:
            r.update(cmp(out, ref))
        RES[name] = r
        log(f"ARM {name:34s} {json.dumps(r)}")
        return out
    except Exception as e:  # noqa: BLE001
        RES[name] = dict(error=repr(e)[:400])
        log(f"ARM {name:34s} ERROR {e!r}"[:2000])
        traceback.print_exc()
        return None


def args_of(p, keys):
    return tuple(p[k] for k in keys)


SEQ = int(os.environ.get("P18_SEQ", "98304"))
TAILS = os.environ.get("P18_TAILS", "seg").split(",")


def run_time():
    import hashlib
    log("jax", jax.__version__, jax.devices()[0], "SEQ", SEQ, "TAILS", TAILS, "BF16G", BF16G,
        "DUMP", os.environ.get("P18_DUMP"))
    p = problem_from_dump(os.environ["P18_DUMP"]) if os.environ.get("P18_DUMP") else problem(1024, SEQ)
    LA, IA = args_of(p, LAYER_KEYS), args_of(p, IDX_KEYS)
    fns = {"inc": layer_fn(True, "")}
    for t in TAILS:
        fns[t] = layer_fn(True, t)
    ref = arm("L_inc_0", fns["inc"], LA, iters=40)
    for rep in (1, 2, 3):
        for name, f in fns.items():
            arm(f"L_{name}_{rep}", f, LA, ref, iters=40)
    # flat (ee=False) pair, the probe-16 knob
    fref = arm("Lf_inc_1", layer_fn(False, ""), LA, ref, iters=40)
    for t in TAILS:
        arm(f"Lf_{t}_1", layer_fn(False, t), LA, ref, iters=40)
    arm("Lf_inc_2", layer_fn(False, ""), LA, ref, iters=40)
    # indexer alone
    iref = arm("I_inc_1", idx_fn(True, ""), IA, iters=40)
    for t in TAILS:
        arm(f"I_{t}_1", idx_fn(True, t), IA, iref, iters=40)
    arm("I_inc_2", idx_fn(True, ""), IA, iref, iters=40)
    for t in TAILS:
        arm(f"I_{t}_2", idx_fn(True, t), IA, iref, iters=40)
    # comparator positive control: a perturbed query token must change both outputs
    pc = list(LA); pc[0] = p["iq"].at[5].set(-p["iq"][5])
    for t in TAILS:
        arm(f"PC_{t}_perturbed_iq_tok5", fns[t], tuple(pc), ref, timed=False)
    if ref is not None:
        for i, a in enumerate(ref):
            h = hashlib.sha256(np.asarray(jax.device_get(a)).tobytes()).hexdigest()[:16]
            RES[f"sha_inc_out{i}"] = h
            log(f"SHA inc {SEQ} out{i} {h}")
    tag = os.path.basename(os.environ.get("P18_DUMP", "")).replace(".npz", "") or str(SEQ)
    json.dump(RES, open(os.path.join(OUT, f"results_{tag}_g{BF16G}.json"), "w"), indent=1)
    log("RESULTS", json.dumps(RES))
    bad = [k for k, v in RES.items() if isinstance(v, dict) and "error" in v]
    log("errors:", bad)
    sys.exit(1 if bad else 0)


def run_counts():
    """Per-slice stage-1 live counts, recorded from inside the segc stage 2 call."""
    import glob
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import sctopk_l1 as l1
    rec = []
    orig = l1._sc_topk_stage2_seg

    def wrapped(cand_scores, counts, k, cand_lengths, p, k_p, **kw):
        b = cand_scores.shape[0]

        def cb(c, ln):
            rec.append((np.asarray(c)[: b * p * 16 : 16].reshape(b, p).astype(np.int64),
                        np.asarray(ln)[:b].copy(), p, k_p))
        jax.debug.callback(cb, counts, cand_lengths)
        return orig(cand_scores, counts, k, cand_lengths, p, k_p, **kw)

    l1._sc_topk_stage2_seg = wrapped
    cases = [(f"synthetic_{s}", None, s) for s in (32768, 98304, 262144)]
    for d in sorted(glob.glob(os.path.join(os.environ.get("P18_DUMPS", "/nonexistent"), "*.npz"))):
        cases.append((os.path.basename(d).replace(".npz", ""), d, None))
    out = {}
    for name, path, s in cases:
        try:
            p = problem_from_dump(path) if path else problem(1024, s)
        except Exception as e:  # noqa: BLE001
            log("COUNTS load error", name, repr(e)); continue
        IA = args_of(p, IDX_KEYS)
        for g in (0, 4):
            global BF16G
            BF16G = g
            rec.clear()
            f = jax.jit(lambda *a: _topk(*a, True, "segc"))
            topk = np.asarray(jax.block_until_ready(f(*IA)))
            jax.effects_barrier()
            if not rec:
                log("COUNTS none", name, g); continue
            cnt = np.concatenate([np.minimum(r[0][r[1] > 0], r[3]) for r in rec])
            p_, k_p = rec[0][2], rec[0][3]
            seq = int(np.asarray(p["seq_lens"])[0])
            words = p_ * k_p
            r = dict(rows=int(cnt.shape[0]), p=p_, k_p=k_p, seq=seq, calls=len(rec),
                     live_frac=float(cnt.sum() / (cnt.shape[0] * words)),
                     live_slices=float((cnt > 0).mean()),
                     cnt_mean_live_slice=float(cnt[cnt > 0].mean()) if (cnt > 0).any() else 0.0,
                     cnt_p50_live_slice=float(np.median(cnt[cnt > 0])) if (cnt > 0).any() else 0.0,
                     full_slices=float((cnt >= k_p).mean()),
                     row_sum_mean=float(cnt.sum(1).mean()), row_sum_max=int(cnt.sum(1).max()))
            for G in (16, 128, 256, 512, 2048):
                r[f"dma_frac_g{G}"] = float((-(-cnt // G) * G).sum() / (cnt.shape[0] * words))
            if "server_topk" in p and g == 4 and p["server_topk"].size:
                st = p["server_topk"]
                n = min(st.shape[0], topk.shape[0])
                same = sum(np.array_equal(np.sort(st[i]), np.sort(topk[i])) for i in range(n))
                r["server_topk_rows_equal"] = f"{same}/{n}"
                r["server_topk_slot_equal"] = int((st[:n] == topk[:n]).all(1).sum())
            out[f"{name}_g{g}"] = r
            log("COUNTS", name, "g", g, json.dumps(r))
    json.dump(out, open(os.path.join(OUT, "counts.json"), "w"), indent=1)
    log("COUNTS_ALL", json.dumps(out))


def run_trace():
    p = problem_from_dump(os.environ["P18_DUMP"]) if os.environ.get("P18_DUMP") else problem(1024, SEQ)
    LA = args_of(p, LAYER_KEYS)
    fns = {"L_inc": layer_fn(True, "")}
    for t in TAILS:
        fns[f"L_{t}"] = layer_fn(True, t)
    for f in fns.values():
        for _ in range(3):
            jax.block_until_ready(f(*LA))
    for name, f in fns.items():
        d = os.path.join(OUT, f"{name}_{SEQ}")
        jax.profiler.start_trace(d)
        for _ in range(3):
            jax.block_until_ready(f(*LA))
        jax.profiler.stop_trace()
        log("traced", name, d)


def run_hlo():
    """StableHLO of the incumbent layer, topk_tail left at its default (or absent)."""
    import inspect
    p = problem(1024, SEQ)
    LA = args_of(p, LAYER_KEYS)
    has = "topk_tail" in inspect.signature(streamindex_topk).parameters
    f = layer_fn(True, "" if has else None)
    tag = os.environ.get("P18_TAG", "tree")
    txt = f.lower(*LA).as_text()
    open(os.path.join(OUT, f"layer_{tag}_{SEQ}.stablehlo.txt"), "w").write(txt)
    log("hlo", tag, "has_topk_tail", has, "bytes", len(txt))


if __name__ == "__main__":
    {"time": run_time, "trace": run_trace, "hlo": run_hlo, "counts": run_counts}[MODE]()
