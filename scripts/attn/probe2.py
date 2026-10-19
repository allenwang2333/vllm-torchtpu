"""Probe 2: per-block decode attention compute cost with KV resident in VMEM.

One "lane block" = 64 query rows (4 tokens x 16 heads) of one sequence against
BKV tokens of one fp8 KV head (head_dim 256, SEQ_ALONG_LANE: K/V are
[256, BKV]). The kernel loops n_steps x L lane blocks with an online-softmax
carry and no DMA, so its time is the compute floor of each formulation.

Prints PROBE {json} per variant: us per lane block, and max abs / rel error of
the attention output against a float64 reference.
"""

import json
import os
import sys
import types

REPO = os.environ.get("REPO", "/workspace/vllm-torchtpu/src")
for _name, _sub in [("vllm_torchtpu", ""), ("vllm_torchtpu.kernels", "kernels")]:
    _m = types.ModuleType(_name)
    _m.__path__ = [os.path.join(REPO, "vllm_torchtpu", _sub)]
    sys.modules[_name] = _m

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devtime  # noqa: E402

F32, BF16, FP8 = jnp.float32, jnp.bfloat16, jnp.float8_e4m3fn
NT = (((1,), (1,)), ((), ()))
TN = (((0,), (0,)), ((), ()))
SPLIT = 16.0
PS = 256.0


def emit(**kw):
    print("PROBE " + json.dumps(kw), flush=True)


L = int(os.environ.get("P2_L", "8"))
BKV = int(os.environ.get("P2_BKV", "4096"))
NSTEPS = int(os.environ.get("P2_STEPS", "20"))
R = 64  # query rows per lane block
INTERP = os.environ.get("P2_INTERPRET", "0") == "1"
D = 256


def block_math(variant, q, q2, q2t, k, v, m_prev, l_prev, acc_prev, iq):
    """Returns m_new [R,128], l_new [R,128], acc_new (R,D) or (D,R) for 'T' variants."""
    if variant in ("qk1pv2", "qk1pv8"):
        s = lax.dot(q, k, preferred_element_type=F32)
        m_cur = jnp.max(s, axis=-1, keepdims=True)
        m_new = jnp.maximum(m_prev, m_cur)
        alpha = jnp.exp(m_prev - m_new)
        # p * PS straight from the exp: exp(s - m + ln PS).
        pf = jnp.exp(s - m_new[:, :1] + float(np.log(PS)))
        l_new = alpha * l_prev + jnp.sum(pf, axis=-1, keepdims=True) * (1.0 / PS)
        ph = pf.astype(FP8)
        if variant == "qk1pv8":
            pv = lax.dot_general(ph, v, NT, preferred_element_type=F32) * (1.0 / PS)
        else:
            plo = ((pf - ph.astype(F32)) * SPLIT).astype(FP8)
            pv2 = lax.dot_general(jnp.concatenate([ph, plo], axis=0), v, NT, preferred_element_type=F32)
            pv = (pv2[:R] + pv2[R:] * (1.0 / SPLIT)) * (1.0 / PS)
        return m_new, l_new, alpha[:, :1] * acc_prev + pv
    if variant in ("cur", "bf16kv", "f32soft", "qkonly", "pvonly", "nat8", "nat8x2",
                   "qk2pv1", "qk2only", "pv2only", "cur_f32p"):
        if variant in ("nat8",):
            s = lax.dot(q2[:R], k, preferred_element_type=F32) * iq[:, :1]
        elif variant in ("nat8x2", "qk2pv1", "qk2only"):
            s2 = lax.dot(q2, k, preferred_element_type=F32)
            s = (s2[:R] + s2[R:] * (1.0 / SPLIT)) * iq[:, :1]
        elif variant == "pvonly" or variant == "pv2only":
            s = None
        else:
            s = lax.dot(q, k, preferred_element_type=F32)
        if s is None:
            p = jnp.full((R, k.shape[1]), 0.5, BF16 if variant == "pvonly" else F32)
            m_new, l_new, alpha = m_prev, l_prev, jnp.ones_like(m_prev)
        else:
            if variant not in ("f32soft", "cur_f32p", "nat8x2", "qk2pv1", "qk2only", "nat8"):
                s = s.astype(BF16)
            m_cur = jnp.max(s, axis=-1, keepdims=True).astype(F32)
            m_new = jnp.maximum(m_prev, m_cur)
            alpha = jnp.exp(m_prev - m_new)
            p = jnp.exp(s - m_new[:, :1].astype(s.dtype))
            l_new = alpha * l_prev + jnp.sum(p.astype(F32), axis=-1, keepdims=True)
        if variant in ("qkonly", "qk2only"):
            return m_new, l_new, acc_prev + p[:, :D].astype(F32)
        if variant in ("nat8",):
            p8 = (p * PS).astype(FP8)
            pv = lax.dot_general(p8, v, NT, preferred_element_type=F32) * (1.0 / PS)
        elif variant in ("nat8x2", "pv2only"):
            pf = p.astype(F32) * PS
            ph = pf.astype(FP8)
            plo = ((pf - ph.astype(F32)) * SPLIT).astype(FP8)
            p2 = jnp.concatenate([ph, plo], axis=0)
            pv2 = lax.dot_general(p2, v, NT, preferred_element_type=F32)
            pv = (pv2[:R] + pv2[R:] * (1.0 / SPLIT)) * (1.0 / PS)
        else:
            pv = lax.dot_general(p.astype(BF16), v, NT, preferred_element_type=F32)
        a = alpha[:, :1]
        return m_new, l_new, a * acc_prev + pv
    if variant in ("T_cur", "T_nat8x2"):
        # Transposed: s^T [BKV, R] = k^T q^T, softmax along sublanes, o^T = v p^T.
        if variant == "T_cur":
            st = lax.dot_general(k, q, (((0,), (1,)), ((), ())), preferred_element_type=F32)
        else:
            st2 = lax.dot_general(k, q2t, TN, preferred_element_type=F32)  # [BKV, 2R]
            st = (st2[:, :R] + st2[:, R:] * (1.0 / SPLIT)) * iq[:1]
        m_cur = jnp.max(st, axis=0, keepdims=True)  # [1, R]
        m_new = jnp.maximum(m_prev, m_cur)  # m_prev [8, R] (rows replicated)
        alpha = jnp.exp(m_prev - m_new)
        pt = jnp.exp(st - m_new[:1])
        l_new = alpha * l_prev + jnp.sum(pt, axis=0, keepdims=True)
        if variant == "T_cur":
            ot = lax.dot(v, pt.astype(BF16), preferred_element_type=F32)  # [D, R]
        else:
            pf = pt * PS
            ph = pf.astype(FP8)
            plo = ((pf - ph.astype(F32)) * SPLIT).astype(FP8)
            p2 = jnp.concatenate([ph, plo], axis=1)  # [BKV, 2R]
            ot2 = lax.dot(v, p2, preferred_element_type=F32)  # [D, 2R]
            ot = (ot2[:, :R] + ot2[:, R:] * (1.0 / SPLIT)) * (1.0 / PS)
        return m_new, l_new, alpha[:1] * acc_prev + ot
    if variant == "convonly":
        kb = k.astype(BF16)
        vb = v.astype(BF16)
        return m_prev, l_prev, acc_prev + (kb[:R, :D] + vb[:R, :D]).astype(F32)
    raise ValueError(variant)


def make_call(variant):
    transposed = variant.startswith("T_")
    kv_dtype = BF16 if variant == "bf16kv" else FP8
    if transposed:
        ml_shape, acc_shape = (8, R), (D, R)
    else:
        ml_shape, acc_shape = (R, 128), (R, D)

    def kernel(q_ref, q2_ref, q2t_ref, iq_ref, iqt_ref, kv_ref, o_ref, lse_ref, m_sc, l_sc, acc_sc):
        m_sc[...] = jnp.full(m_sc.shape, -jnp.inf, F32)
        l_sc[...] = jnp.zeros(l_sc.shape, F32)
        acc_sc[...] = jnp.zeros(acc_sc.shape, F32)

        @pl.loop(0, NSTEPS)
        def _(s):
            for b in range(L):
                m_new, l_new, acc_new = block_math(
                    variant, q_ref[b], q2_ref[b], q2t_ref[b], kv_ref[b, 0], kv_ref[b, 1],
                    m_sc[b], l_sc[b], acc_sc[b], iqt_ref[b] if transposed else iq_ref[b])
                m_sc[b] = m_new
                l_sc[b] = l_new
                acc_sc[b] = acc_new

        for b in range(L):
            if transposed:
                o_ref[b] = (acc_sc[b] / l_sc[b][:1]).T
                lse_ref[b] = jnp.broadcast_to((m_sc[b][:1] + jnp.log(l_sc[b][:1])).T, (R, 128))
            else:
                o_ref[b] = acc_sc[b] / l_sc[b][:, :1]
                lse_ref[b] = m_sc[b] + jnp.log(l_sc[b])

    vm = pltpu.VMEM

    def f(q, q2, q2t, iq, iqt, kv):
        return pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((L, R, D), F32), jax.ShapeDtypeStruct((L, R, 128), F32)),
            in_specs=[pl.BlockSpec(memory_space=vm)] * 6,
            out_specs=(pl.BlockSpec(memory_space=vm), pl.BlockSpec(memory_space=vm)),
            scratch_shapes=[vm((L,) + ml_shape, F32), vm((L,) + ml_shape, F32), vm((L,) + acc_shape, F32)],
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=62 << 20),
            name=f"p2_{variant}",
            **({"interpret": pltpu.InterpretParams()} if INTERP else {}),
        )(q, q2, q2t, iq, iqt, kv)

    return jax.jit(f), kv_dtype


def make_data(seed=0, peaky=False):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((L, R, D)).astype(np.float32) * 2.0
    k = rng.standard_normal((L, D, BKV)).astype(np.float32) * 0.5
    v = rng.standard_normal((L, D, BKV)).astype(np.float32) * 0.5
    if peaky:
        # A few strongly attended tokens per lane, as attention sinks do.
        hot = rng.integers(0, BKV, size=(L, 8))
        for b in range(L):
            k[b][:, hot[b]] += np.sign(q[b].mean(0))[:, None] * 1.5
    kq = np.asarray(jnp.asarray(k).astype(FP8).astype(F32))
    vq = np.asarray(jnp.asarray(v).astype(FP8).astype(F32))
    qs = np.asarray(jnp.asarray(q * (1.0 / 16)).astype(BF16).astype(F32))
    # Split q (already scaled by 1/16) into fp8 hi + lo/16 with one power-of-two
    # scale per row, computed once per sequence (not per KV block).
    amax = np.abs(qs).max(-1, keepdims=True)
    qscale = 2.0 ** np.floor(np.log2(448.0 / np.maximum(amax, 1e-30)))
    qsc = qs * qscale
    qh = np.asarray(jnp.asarray(qsc).astype(FP8).astype(F32))
    ql = np.asarray(jnp.asarray((qsc - qh) * SPLIT).astype(FP8).astype(F32))
    return qs, kq, vq, qh, ql, qscale


def reference(qs, kq, vq):
    s = np.einsum("lrd,ldt->lrt", qs.astype(np.float64), kq.astype(np.float64))
    m = s.max(-1, keepdims=True)
    p = np.exp(s - m)
    l = p.sum(-1, keepdims=True)
    o = np.einsum("lrt,ldt->lrd", p, vq.astype(np.float64)) / l
    return o, (m + np.log(l))[..., 0]


VARIANTS = os.environ.get(
    "P2_VARIANTS",
    "cur,bf16kv,f32soft,cur_f32p,qkonly,pvonly,convonly,nat8,nat8x2,qk2pv1,qk2only,pv2only,T_cur,T_nat8x2",
).split(",")

for peaky in (False, True):
    qs, kq, vq, qh, ql, qscale = make_data(peaky=peaky)
    o_ref, lse_ref = reference(qs, kq, vq)
    q_dev = jnp.asarray(qs).astype(BF16)
    q2 = jnp.asarray(np.concatenate([qh, ql], axis=1)).astype(FP8)
    q2t = jnp.swapaxes(q2, 1, 2)
    iq = jnp.asarray(np.broadcast_to(1.0 / qscale, (L, R, 128)).astype(np.float32))
    iqt = jnp.asarray(np.broadcast_to(np.swapaxes(1.0 / qscale, 1, 2), (L, 8, R)).astype(np.float32))
    for variant in VARIANTS:
        if peaky and variant in ("qkonly", "pvonly", "convonly", "qk2only", "pv2only"):
            continue
        try:
            f, kv_dtype = make_call(variant)
            kv = jnp.stack([jnp.asarray(kq), jnp.asarray(vq)], axis=1).astype(kv_dtype)
            if INTERP:
                out = jax.block_until_ready(f(q_dev, q2, q2t, iq, iqt, kv))
                us, cnt = 1.0, 1
            else:
                t, out, _ = devtime.device_times(f, (q_dev, q2, q2t, iq, iqt, kv), iters=5)
                us, cnt = devtime.summarize(t, r"^p2_")
            o, lse = (np.asarray(x, np.float64) for x in out)
            rec = dict(kind="compute", variant=variant, peaky=peaky, L=L, BKV=BKV, steps=NSTEPS,
                       us_call=round(us, 1), us_per_block=round(us / (NSTEPS * L), 3),
                       GBps_equiv=round(L * NSTEPS * BKV * 2 * D / us / 1e3, 1))
            o_r, lse_r = o_ref, lse_ref
            if variant not in ("qkonly", "pvonly", "convonly", "qk2only", "pv2only"):
                err = np.abs(o - o_r)
                rec.update(max_abs=float(err.max()), mean_abs=float(err.mean()),
                           max_rel=float((err / (np.abs(o_r) + 1e-3)).max()),
                           ref_absmean=float(np.abs(o_r).mean()))
            emit(**rec)
        except Exception as e:  # noqa: BLE001
            emit(kind="compute", variant=variant, peaky=peaky, error=f"{type(e).__name__}: {str(e)[:800]}")
print("PROBE_DONE", flush=True)
