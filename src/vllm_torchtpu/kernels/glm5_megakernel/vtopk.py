"""Vectorized top-k extraction (v11): selected keys -> ascending token ids with no scalar loop.

The threshold search leaves T (the K-th largest key) and the selection mask
    sel = (key > T) | (key == T & pos < P)
where P keeps only the first K - #(key > T) ties in position order (a 19-pass position search that
runs only when ties exceed the need). Output slot k (ascending) then lives in the row r whose
selected-count prefix brackets it, CE[r] <= k < CI[r], at the (k - CE[r])-th selected lane of r.
Everything is counting: per-row counts and their prefix sums, and one one-hot matmul per k chunk
that reads each slot's row and within-row prefix back. Every value is a small integer: exact in bf16
(<= 256) with f32 accumulation, hi/lo split where larger.
"""
import jax.numpy as jnp
from jax import lax

F32, I32, BF16 = jnp.float32, jnp.int32, jnp.bfloat16
NT = (((1,), (1,)), ((), ()))
NN = (((1,), (0,)), ((), ()))
INT_MIN = -(2 ** 31)
INT_MAX = 2 ** 31 - 1


def shl(x, n):
    return lax.shift_left(x, jnp.asarray(n, x.dtype))


def threshold(keys_ref, K):
    """2-bit radix threshold search (v8): the K-th largest key, as a biased i32."""
    def pass_(i, p):
        sh = jnp.int32(30) - 2 * i
        kk_ = keys_ref[...]
        dig = jnp.int32(0)
        for k in (1, 2, 3):
            cs = (p | shl(jnp.int32(k), sh)) ^ jnp.int32(INT_MIN)
            dig = dig + (jnp.sum((kk_ >= cs).astype(I32)) >= K).astype(I32)
        return p | shl(dig, sh)
    p = lax.fori_loop(0, 16, pass_, jnp.int32(0))
    return p ^ jnp.int32(INT_MIN)


def extract(keys_ref, T, K, idx_v, kc=256):
    """keys_ref (ROWS, 128) i32, T from threshold() -> idx_v (K // 128, 128) i32 ascending token ids,
    -1 past the selected count. Returns the selected count."""
    ROWS = keys_ref.shape[0]
    assert ROWS % 128 == 0 and K % 128 == 0 and K <= 2 ** 15 and ROWS <= 128 * 128
    NBK = ROWS // 128
    kc = min(kc, K)
    assert K % kc == 0
    kk = keys_ref[...]
    gt = kk > T
    eq = kk == T
    n_gt = jnp.sum(gt.astype(I32))
    n_eq = jnp.sum(eq.astype(I32))
    need_eq = jnp.where(T == jnp.int32(INT_MIN), 0, K - n_gt)
    pos = (lax.broadcasted_iota(I32, (ROWS, 128), 0) * 128
           + lax.broadcasted_iota(I32, (ROWS, 128), 1))
    excess = n_eq > need_eq
    PB = (ROWS * 128).bit_length()

    def pbit(i, p):
        cand = p | shl(jnp.int32(1), jnp.int32(PB - 1) - i)
        c = jnp.sum((eq & (pos < cand)).astype(I32))
        return jnp.where(c <= need_eq, cand, p)
    P = lax.fori_loop(0, jnp.where(excess, PB, 0), pbit, jnp.int32(0))
    P = jnp.where(excess, P, jnp.int32(INT_MAX))
    sel = gt | (eq & (pos < P))
    n_sel = jnp.sum(sel.astype(I32))
    selb = sel.astype(F32).astype(BF16)                                   # (ROWS, 128)

    i128 = lax.broadcasted_iota(I32, (128, 128), 0)
    j128 = lax.broadcasted_iota(I32, (128, 128), 1)
    # per-row counts along lanes, blocked: C[b, j] = count of row 128 b + j
    cnt = lax.dot_general(jnp.ones((8, 128), BF16), selb, NT, preferred_element_type=F32)
    C = jnp.concatenate([cnt[0:1, b * 128:(b + 1) * 128] for b in range(NBK)]
                        + [jnp.zeros((128 - NBK, 128), F32)], axis=0)       # (128, 128)
    W = lax.dot_general(C.astype(BF16), (i128 <= j128).astype(BF16), NN,
                        preferred_element_type=F32)                          # within-block incl
    tot = W[:, 127:128]
    th = jnp.floor(tot * (1.0 / 128))
    tl = tot - 128 * th
    Ls = (j128 < i128).astype(BF16)                                          # Ls[b, b'] = b' < b
    off = (128 * lax.dot_general(Ls, jnp.broadcast_to(th, (128, 128)).astype(BF16), NN,
                                 preferred_element_type=F32)
           + lax.dot_general(Ls, jnp.broadcast_to(tl, (128, 128)).astype(BF16), NN,
                             preferred_element_type=F32))
    CI = W + off                                                             # row-count prefix, incl
    CE = CI - C
    CEt, CIt = CE.T, CI.T                                                    # [j, b] = row 128 b + j
    ce_col = jnp.concatenate([CEt[:, b:b + 1] for b in range(NBK)], axis=0)  # (ROWS, 1)
    ci_col = jnp.concatenate([CIt[:, b:b + 1] for b in range(NBK)], axis=0)
    ce_row = jnp.concatenate([CE[b:b + 1, :] for b in range(NBK)], axis=1)   # (1, ROWS)

    # VT: within-row inclusive prefix (lane l, row r) and each row's id / CE, hi-lo split
    wrt = lax.dot_general((j128 <= i128).astype(BF16), selb, NT, preferred_element_type=F32)
    r = lax.broadcasted_iota(I32, (1, ROWS), 1).astype(F32)
    rh = jnp.floor(r * (1.0 / 128))
    ceh = jnp.floor(ce_row * (1.0 / 128))
    tail = jnp.concatenate([rh, r - 128 * rh, ceh, ce_row - 128 * ceh,
                            jnp.zeros((12, ROWS), F32)], axis=0)
    vt = jnp.concatenate([wrt.astype(BF16), tail.astype(BF16)], axis=0)      # (144, ROWS)

    for c0 in range(0, K, kc):
        kr = (c0 + lax.broadcasted_iota(I32, (1, kc), 1)).astype(F32)
        oht = ((ce_col <= kr) & (kr < ci_col)).astype(F32).astype(BF16)      # (ROWS, kc)
        R = lax.dot_general(vt, oht, NN, preferred_element_type=F32)          # (144, kc)
        rr = 128 * R[128:129] + R[129:130]
        ce = 128 * R[130:131] + R[131:132]
        lane = jnp.sum((R[0:128] <= kr - ce).astype(F32), axis=0, keepdims=True)
        tok = jnp.where(kr < n_sel.astype(F32), rr * 128 + lane, -1.0).astype(I32)
        for q in range(kc // 128):
            idx_v[c0 // 128 + q:c0 // 128 + q + 1, :] = tok[:, q * 128:(q + 1) * 128]
    return n_sel
