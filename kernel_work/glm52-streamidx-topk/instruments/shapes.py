"""Reachable GLM-5.2 shapes for streamindex_topk, plus an input builder.

Every row here is a shape the two GLM-5.2 deployments can actually reach. The
derivation (recipe flag -> runner -> kernel) is in ../report.md; the guard in
`check_reachable` re-asserts the legality ladder so a future edit cannot quietly
benchmark a shape production never sees.

  (A) 8k1k   vllm-torchtpu-glm-5.2-ix-8k1k.yaml
             max_model_len 9216, no --block-size -> effective 1024
             pages_per_seq 9,    n = 10240
             n is align_to(pages_per_seq, bkv_p)*page_size, NOT
             pages_per_seq*page_size -- streamindex_topk.py:602-605 rounds 9
             pages up to 10 because bkv_p=2, so the score row is 11.1% wider
             than max_model_len. The pad is (A)-only; at (B) align_to(2048,2)
             is 2048 and there is none.
  (B) AgentX glm-5.2-agentx-multihost-nondisagg.yaml
             max_model_len 1048576, --block-size 256 raised to 512 by
             TpuPlatform.update_block_size_for_backend
             pages_per_seq 2048, n = 1048576
"""
import jax
import jax.numpy as jnp
import numpy as np

H = 32          # index_n_heads
DH = 128        # index_head_dim
K = 2048        # index_topk
S = 64          # num_reqs_max_model_len, fixed (not the {8,16,32,64} ladder)
CR = 1          # compression_ratio
BKV_P = (2, 2, 2)
BQ_SZ = (1, 128, 128)
DECODE_REQ_BATCH_SIZE = 4
VMEM_LIMIT = 100 * 1024 * 1024
TOKEN_LADDER = (16, 32, 64, 128, 256, 512, 1024)

# The two constants the call site folds into `weights` before the kernel sees
# them (mla_attention_op.py:278-280, deepseek_v2.py:698).
SOFTMAX_SCALE = DH ** -0.5      # 0.08838834764831845
NHEAD_SCALE = H ** -0.5         # 0.17677669529663687

A = dict(psz=1024, pages_per_seq=9)
B = dict(psz=512, pages_per_seq=2048)

ROWS = {
    # -- (A) 8k1k -------------------------------------------------------------
    "A-D64-8k":    dict(**A, T=64, R=64, D=64, seqlen=8192),
    # 9216 is the ceiling, not 14746: --random-range-ratio 0.8 makes the
    # sampler's ISL range [1638, 14746], but max_model_len=9216 caps what the
    # server admits. Note the kernel's own cap is looser -- n = 10240 -- so
    # max_model_len is what binds here, and the top 1024 score columns are
    # padding no request can ever reach.
    "A-D64-9k":    dict(**A, T=64, R=64, D=64, seqlen=9216),
    "A-D63-8k":    dict(**A, T=64, R=63, D=63, seqlen=8192),
    "A-D32-8k":    dict(**A, T=32, R=32, D=32, seqlen=8192),
    "A-D16-8k":    dict(**A, T=16, R=16, D=16, seqlen=8192),
    "A-P1024-8k":  dict(**A, T=1024, R=1, D=0, seqlen=8192, qlen=1024),
    "A-P512-8k":   dict(**A, T=512, R=1, D=0, seqlen=8192, qlen=512),
    # -- (B) AgentX -----------------------------------------------------------
    "B-D4-256k":   dict(**B, T=16, R=4, D=4, seqlen=262144),
    "B-D4-512k":   dict(**B, T=16, R=4, D=4, seqlen=524288),
    "B-D4-1M":     dict(**B, T=16, R=4, D=4, seqlen=1048576),
    "B-D16-256k":  dict(**B, T=16, R=16, D=16, seqlen=262144),
    "B-D1-1M":     dict(**B, T=16, R=1, D=1, seqlen=1048576),
    "B-P1024-256k": dict(**B, T=1024, R=1, D=0, seqlen=262144, qlen=1024),
    "B-P1024-1M":  dict(**B, T=1024, R=1, D=0, seqlen=1048576, qlen=1024),
    # -- ragged control: same total KV as B-D16-256k, spread unevenly ---------
    # isolates the seq_batch_size=4 group-max waste (streamindex_topk.py:276,
    # and the TODO at :751-754).
    "B-RAG16":     dict(**B, T=16, R=16, D=16, seqlen=1048576,
                        seqlens_override="ragged"),
}


def cdiv(a, b):
    return -(-a // b)


def check_reachable(name, r):
    """Re-assert the legality ladder. Raises on a shape production cannot reach."""
    psz, pps, T = r["psz"], r["pages_per_seq"], r["T"]
    qlen = r.get("qlen", 1)
    assert T in TOKEN_LADDER, f"{name}: T={T} not on _get_token_paddings ladder"
    assert psz * BKV_P[0] % 128 == 0, f"{name}: page_size*bkv_p not a 128 multiple"
    assert (pps * psz) % 16 == 0, f"{name}: n not a multiple of LANES"
    assert K % 16 == 0 and 0 < K <= 4096, f"{name}: k off ladder"
    assert r["R"] <= S and r["D"] <= r["R"], f"{name}: R/D out of range"
    assert r["D"] * 1 + (r["R"] - r["D"]) * qlen <= T, f"{name}: tokens exceed T"
    assert r["seqlen"] <= pps * psz, f"{name}: seqlen exceeds max_model_len"
    # The gate at streamindex_topk.py:860 is a BATCH max, so any row whose
    # longest sequence is <= k takes _fast_path and measures nothing.
    assert r["seqlen"] // CR > K, f"{name}: would take _fast_path, measures nothing"


def quantize_fp8_ue8m0(x, block_size):
    """Block fp8 quantization with UE8M0 scales. Copied verbatim in behaviour
    from tests/kernels/test_dsv4_streamindex_topk.py:39-49 so the packing this
    instrument produces is the packing the repo's own test asserts."""
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    *lead, dim = x.shape
    blocked = x.reshape(*lead, dim // block_size, block_size)
    amax = jnp.clip(jnp.max(jnp.abs(blocked), axis=-1, keepdims=True), 1e-4, None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    q = (blocked * (1.0 / scale)).astype(jnp.float8_e4m3fn).reshape(x.shape)
    scale = jnp.squeeze(scale, -1).astype(jnp.float8_e8m0fnu)
    return q, scale


def _seq_lens_for(r):
    """Per-sequence KV lengths, honouring the ragged control."""
    R, seqlen = r["R"], r["seqlen"]
    if r.get("seqlens_override") == "ragged":
        # One long sequence per group of DECODE_REQ_BATCH_SIZE, the rest short.
        # num_bkv comes from the group max (streamindex_topk.py:276), so the
        # three short ones in each group pay the long one's block count.
        out = np.full(R, 4096, dtype=np.int64)
        out[::DECODE_REQ_BATCH_SIZE] = seqlen
        return out
    return np.full(R, seqlen, dtype=np.int64)


def build(name, seed=0):
    """Build one row's kernel arguments on device. Returns (args, kwargs, meta)."""
    r = ROWS[name]
    check_reachable(name, r)
    psz, pps = r["psz"], r["pages_per_seq"]
    T, R, D = r["T"], r["R"], r["D"]
    qlen = r.get("qlen", 1)
    lens = _seq_lens_for(r)

    pages_per = np.array([cdiv(int(L), psz) for L in lens], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(pages_per)])
    num_blocks = int(starts[-1])

    key = jax.random.key(seed)
    k_cache, k_q = jax.random.split(key)

    # --- cache_kv: (num_blocks, psz//4, 4, 256) uint8 ------------------------
    # Slot s of a page lives at [., s//4, s%4], so generating in this layout
    # needs no scatter. bf16 source keeps the transient under control at 1M.
    @jax.jit
    def _pack(kk):
        keys = jax.random.normal(kk, (num_blocks, psz // 4, 4, DH),
                                 dtype=jnp.bfloat16).astype(jnp.float32)
        q8, sc = quantize_fp8_ue8m0(keys, DH)
        vb = jax.lax.bitcast_convert_type(q8, jnp.uint8)
        sb = jax.lax.bitcast_convert_type(sc, jnp.uint8)  # squeezed scale -> trailing dim 1
        pad = jnp.zeros((num_blocks, psz // 4, 4, 256 - DH - 1), jnp.uint8)
        return jnp.concatenate([vb, sb, pad], axis=-1)

    cache_kv = _pack(k_cache)

    # --- page_indices: (S * pages_per_seq,) ----------------------------------
    pi = np.zeros((S, pps), dtype=np.int32)
    for i in range(R):
        pi[i, :pages_per[i]] = np.arange(starts[i], starts[i + 1])
    page_indices = jnp.asarray(pi.reshape(-1))

    # --- seq_lens, cu_q_lens, distribution -----------------------------------
    sl = np.zeros(S, dtype=np.int32)
    sl[:R] = lens
    seq_lens = jnp.asarray(sl)

    qlens = np.zeros(S, dtype=np.int32)
    qlens[:R] = qlen
    cu = np.zeros(S + 1, dtype=np.int32)
    cu[1:] = np.cumsum(qlens)
    cu_q_lens = jnp.asarray(cu)
    assert cu[R] <= T, f"{name}: {cu[R]} tokens do not fit in T={T}"

    distribution = jnp.asarray([D, D, R], dtype=jnp.int32)

    # --- q, weights ----------------------------------------------------------
    kq, kw = jax.random.split(k_q)
    q = jax.random.normal(kq, (T, H, DH), dtype=jnp.float32).astype(
        jnp.float8_e4m3fn)
    weights = (jax.random.uniform(kw, (T, H), jnp.float32, -1.5, 1.5)
               * SOFTMAX_SCALE * NHEAD_SCALE)

    args = (q, weights, cache_kv, seq_lens, page_indices, cu_q_lens, distribution)
    kwargs = dict(k=K, compression_ratio=CR, num_kv_pages_per_block=BKV_P,
                  num_queries_per_block=BQ_SZ, vmem_limit_bytes=VMEM_LIMIT,
                  decode_req_batch_size=DECODE_REQ_BATCH_SIZE,
                  enable_early_exit=True)
    meta = dict(name=name, n=pps * psz, num_blocks=num_blocks,
                cache_bytes=num_blocks * (psz // 4) * 4 * 256,
                score_bytes=T * pps * psz * 4, lens=lens, **r)
    return args, kwargs, meta
