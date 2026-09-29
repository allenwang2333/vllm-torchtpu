"""Probe 16: does flattening streamindex_topk's early-exit lax.cond let XLA overlap
independent q-path TC work (W_UK bmm, q rope, concat) with the SparseCore top-k?

  python scripts/benchmark_streamidx_early_exit_overlap.py time     timing + bit-exact checks (1 chip)
  python scripts/benchmark_streamidx_early_exit_overlap.py trace    xprof trace of layer arms (needs jax>=0.11)

One GLM-5.2 layer's prefill indexer + q-glue + sparse-MLA attention, rank0 shapes:
1024 query tokens, one sequence of 262144 tokens (256k ctx incl. the chunk), indexer
q f8[1024,32,128], idx cache u8[384,256,4,256] HEAD_ALONG_SUBLANE (page 1024),
chunk_tokens=128, sctopk_l1=True, k=2048, MLA H=64 D=576 fp8 SC-layout KV cache.
Arms differ only in enable_early_exit (True = incumbent lax.cond, False = flat).
"""
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "time"
OUT = os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "p16", MODE)
os.makedirs(OUT, exist_ok=True)
if MODE == "trace" or os.environ.get("P16_TRACE"):
    os.environ["LIBTPU_INIT_ARGS"] = (os.environ.get("LIBTPU_INIT_ARGS", "")
                                      + " --xla_enable_custom_call_region_trace=true").strip()
if os.environ.get("P16_FLAGS"):
    os.environ["GLM5_LIBTPU_EXTRA_ARGS"] = (os.environ.get("GLM5_LIBTPU_EXTRA_ARGS", "") + " " + os.environ["P16_FLAGS"]).strip()
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


def _topk(iq, w, cache, seq_lens, pi, cu, dist, ee):
    return streamindex_topk(
        iq, w, cache, seq_lens, pi, cu, dist, k=TOPK, compression_ratio=1,
        num_kv_pages_per_block=(2, 2, 2), num_queries_per_block=(1, 128, 128),
        enable_early_exit=ee, chunk_tokens=128, sctopk_l1=True, score_bf16_group=0)


def _qglue(qb, wuk, cos, sin):
    q_nope, q_pe = qb[..., :DN], qb[..., DN:]
    ql = jnp.einsum("thd,hdc->thc", q_nope, wuk, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    x = q_pe.astype(jnp.float32)
    x1, x2 = x[..., : DR // 2], x[..., DR // 2:]
    c, s = cos[:, None, :], sin[:, None, :]
    pe = jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1).astype(jnp.bfloat16)
    return jnp.concatenate([ql, pe], -1)


def idx_fn(ee):
    @jax.jit
    def f(iq, w, cache, seq_lens, pi, cu, dist):
        return _topk(iq, w, cache, seq_lens, pi, cu, dist, ee)
    return f


def layer_fn(ee, barrier=False):
    """barrier=True: flat indexer but q-glue forced to start after top-k (positive control
    for the ordering: should reproduce the cond arm's serialisation)."""
    @jax.jit
    def f(iq, w, cache, seq_lens, pi, cu, dist, qb, wuk, cos, sin, nope, rope):
        topk = _topk(iq, w, cache, seq_lens, pi, cu, dist, ee)
        if barrier:
            topk, qb = jax.lax.optimization_barrier((topk, qb))
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


SEQ = int(os.environ.get("P16_SEQ", "262144"))


def run_time():
    log("jax", jax.__version__, jax.devices()[0], "SEQ", SEQ, "LIBTPU_INIT_ARGS=", os.environ.get("LIBTPU_INIT_ARGS"))
    p = problem(1024, SEQ)
    LA, IA = args_of(p, LAYER_KEYS), args_of(p, IDX_KEYS)
    # interleaved A/B/A/B for noise
    refL = arm("L_cond_1", layer_fn(True), LA)
    arm("L_flat_1", layer_fn(False), LA, refL)
    arm("L_cond_2", layer_fn(True), LA, refL)
    arm("L_flat_2", layer_fn(False), LA, refL)
    arm("L_flat_barrier", layer_fn(False, barrier=True), LA, refL)
    refI = arm("I_cond_1", idx_fn(True), IA)
    arm("I_flat_1", idx_fn(False), IA, refI)
    arm("I_cond_2", idx_fn(True), IA, refI)
    arm("I_flat_2", idx_fn(False), IA, refI)
    # q-glue + attention alone with fixed top-k (the part to be hidden)
    topk_fixed = refI

    @jax.jit
    def att_only(qb, wuk, cos, sin, nope, rope, topk, cu, dist):
        q = _qglue(qb, wuk, cos, sin)
        return mla.sparse_ragged_paged_attention(
            q, nope, rope, topk, None, cu, dist, sm_scale=SM,
            gather_and_attention_chunk_size=64, attention_kernel_batch_size=16,
            cache_layout="sparsecore")
    arm("A_qglue_attn_only", att_only, (p["qb"], p["wuk"], p["cos"], p["sin"], p["nope"], p["rope"], topk_fixed, p["cu"], p["dist"]))

    @jax.jit
    def qglue_only(qb, wuk, cos, sin):
        return _qglue(qb, wuk, cos, sin)
    arm("Q_qglue_only", qglue_only, (p["qb"], p["wuk"], p["cos"], p["sin"]))

    # positive controls: perturbed inputs must change outputs
    iq2 = p["iq"].at[5].set(-p["iq"][5])
    pc = list(LA); pc[0] = iq2
    arm("PC_flat_perturbed_iq_tok5", layer_fn(False), tuple(pc), refL, timed=False)
    pc = list(LA); pc[7] = p["qb"].at[700, 3, 0].add(1.0)
    arm("PC_flat_perturbed_qb_tok700", layer_fn(False), tuple(pc), refL, timed=False)

    if SEQ != 262144:
        json.dump(RES, open(os.path.join(OUT, f"results_{SEQ}.json"), "w"), indent=1)
        log("RESULTS", json.dumps(RES))
        bad = [k for k, v in RES.items() if "error" in v]
        sys.exit(1 if bad else 0)
    # small shape (seq > k): 256 tokens, 16384 ctx; bit-exact
    ps = problem(256, 16384, seed=2, idx_pages=32, mla_tokens=65536)
    rs = arm("S_cond", layer_fn(True), args_of(ps, LAYER_KEYS), iters=10)
    arm("S_flat", layer_fn(False), args_of(ps, LAYER_KEYS), rs, iters=10)
    # seq <= k: cond takes the fast path; flat is the common path -> compare as sets
    pk = problem(128, 1536, seed=3, idx_pages=8, mla_tokens=65536)
    a = jax.device_get(idx_fn(True)(*args_of(pk, IDX_KEYS)))
    b = jax.device_get(idx_fn(False)(*args_of(pk, IDX_KEYS)))
    same_order = bool((a == b).all())
    same_set = bool(all((np.sort(x) == np.sort(y)).all() for x, y in zip(a, b)))
    RES["K_seq_le_k"] = dict(same_order=same_order, same_set=same_set)
    log("ARM K_seq_le_k", RES["K_seq_le_k"])
    json.dump(RES, open(os.path.join(OUT, f"results_{SEQ}.json"), "w"), indent=1)
    log("RESULTS", json.dumps(RES))
    bad = [k for k, v in RES.items() if "error" in v]
    log("errors:", bad)
    sys.exit(1 if bad else 0)


def run_ab():
    """cond vs flat, interleaved x2, at SEQ; prints sha256 of outputs so arms run in
    different processes (different LIBTPU flags) can be compared bit-for-bit."""
    import hashlib
    tag = os.environ.get("P16_TAG", "base")
    log("jax", jax.__version__, "SEQ", SEQ, "TAG", tag, "LIBTPU_INIT_ARGS=", os.environ.get("LIBTPU_INIT_ARGS"))
    p = problem(1024, SEQ)
    LA = args_of(p, LAYER_KEYS)
    ref = arm(f"{tag}_{SEQ}_L_cond_1", layer_fn(True), LA, iters=40)
    arm(f"{tag}_{SEQ}_L_flat_1", layer_fn(False), LA, ref, iters=40)
    arm(f"{tag}_{SEQ}_L_cond_2", layer_fn(True), LA, ref, iters=40)
    arm(f"{tag}_{SEQ}_L_flat_2", layer_fn(False), LA, ref, iters=40)
    for i, a in enumerate(ref):
        h = hashlib.sha256(np.asarray(jax.device_get(a)).tobytes()).hexdigest()[:16]
        RES[f"{tag}_{SEQ}_sha{i}"] = h
        log(f"SHA {tag} {SEQ} out{i} {h}")
    if os.environ.get("P16_TRACE"):
        for name, f in (("L_cond", layer_fn(True)), ("L_flat", layer_fn(False))):
            d = os.path.join(OUT, f"{tag}_{name}_{SEQ}")
            jax.profiler.start_trace(d)
            for _ in range(3):
                jax.block_until_ready(f(*LA))
            jax.profiler.stop_trace()
    json.dump(RES, open(os.path.join(OUT, f"ab_{tag}_{SEQ}.json"), "w"), indent=1)
    bad = [k for k, v in RES.items() if isinstance(v, dict) and "error" in v]
    sys.exit(1 if bad else 0)


def run_trace():
    p = problem(1024, SEQ)
    LA = args_of(p, LAYER_KEYS)
    fns = {"L_cond": layer_fn(True), "L_flat": layer_fn(False), "L_flat_barrier": layer_fn(False, True)}
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
    # HLO text of each for schedule inspection
    for name, f in fns.items():
        open(os.path.join(OUT, f"{name}_{SEQ}.hlo.txt"), "w").write(f.lower(*LA).compile().as_text())


if __name__ == "__main__":
    {"time": run_time, "trace": run_trace, "ab": run_ab}[MODE]()
