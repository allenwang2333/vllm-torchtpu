"""Interpret-mode test: the serving form (two sequences, KV and GDN states in a region pool)."""
import os, sys, types
INTERP = os.environ.get("MK_INTERPRET", "1") == "1" and "--tpu" not in sys.argv  # --tpu: on a TPU
sys.path.insert(0, os.path.dirname(__file__))
if INTERP:
    import cpu_tpu7x  # noqa: F401
os.environ["VLLM_KV_CACHE_LAYOUT"] = "HND"
os.environ["TPU_GDN_CONV_QK_PAIR_LAYOUT"] = "1"
import jax, jax.numpy as jnp, numpy as np
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src")
pkg = types.ModuleType("vllm_torchtpu"); pkg.__path__ = [SRC + "/vllm_torchtpu"]; sys.modules["vllm_torchtpu"] = pkg
k = types.ModuleType("vllm_torchtpu.kernels"); k.__path__ = [SRC + "/vllm_torchtpu/kernels"]; sys.modules["vllm_torchtpu.kernels"] = k
from vllm_torchtpu.kernels.qwen35_prefill_mk import reference as R, megakernel as MK, harness as Hn, pool_plan

TYPES = tuple(os.environ.get("TYPES", "gdn,attn").split(","))
c = R.StageConfig(layer_types=TYPES, H=256, NH=4, NKV=2, HD=128, ROT=32, KH=2, VH=4,
                  E=16, TOPK=4, I=128, IS=128)
layout = MK.StageLayout(layer_types=c.layer_types, layer_region=(0,) * len(TYPES),
                        layer_group=(0,) * len(TYPES))
HAS_GDN = "gdn" in TYPES
lens, ctxs, BT_TOK, PPS, MAXS = [150, 200], [256, 0], 384, 8, 4
tiles = MK.MKTiles(tm_norm=64, tm_prep=64, tm_gate=128, tm=64, tn=128, chunk=64, bq=64, bkv=128,
                   att_sub=2, rt=16, tm_route=64, tm_combine=64, inflight=8, vmem_mib=64)
case = Hn.make_case(c, layout, lens, ctxs, max_seqs=MAXS, pps=PPS, block_tokens=BT_TOK)
T, split = case.t, BT_TOK // Hn.PAGE
plan, stride = pool_plan.build_gdn_plan(case.pools[0].shape, case.pools[0].dtype, conv_dim=c.QKV,
                                        n_v=c.VH, d_k=c.KD, d_v=c.VD, kernel_size=c.CONV,
                                        pool_block_tokens=BT_TOK)
_, ws, _, h, r = R.random_stage(c, T, 0)
rng = np.random.default_rng(3)
# Per-sequence states: GDN (zeros for a sequence without context) and attention context K/V.
gstates, astates = [], []
for s, (n, cx) in enumerate(zip(lens, ctxs)):
    conv = rng.standard_normal((c.CONV - 1, c.QKV)) * 0.5 if cx else np.zeros((c.CONV - 1, c.QKV))
    rec = rng.standard_normal((c.VH, c.KD, c.VD)) * 0.05 if cx else np.zeros((c.VH, c.KD, c.VD))
    gstates.append({"conv": jnp.asarray(conv, R.BF16), "rec": jnp.asarray(rec, R.BF16).astype(R.F32)})
    kv = lambda: rng.standard_normal((cx + n, c.NKV, c.HD)).astype(Hn.FP8_NP)
    kk, vv = kv(), kv()
    astates.append({"k": jnp.asarray(kk), "v": jnp.asarray(vv)})
    Hn.write_kv(case.pools[0], case.attn_blocks[s], kk[:cx], vv[:cx], 0, split, BT_TOK)
rel = lambda a, b: float(jnp.linalg.norm(a.astype(R.F32) - b.astype(R.F32)) / max(float(jnp.linalg.norm(b.astype(R.F32))), 1e-12))
pool = jnp.asarray(case.pools[0])
if HAS_GDN:
    pool = Hn.store_states(case.pools[0], plan, stride, case.gdn_slots[0],
                           np.stack([g["conv"].astype(R.F32) for g in gstates]),
                           np.stack([g["rec"] for g in gstates]), interpret=INTERP)
    cv, rc = Hn.load_states(pool, plan, stride, case.gdn_slots[0], (c.CONV - 1, c.QKV), (c.VH, c.KD, c.VD), interpret=INTERP)
    print("state pack/unpack round trip: conv", rel(cv[0], gstates[0]["conv"]), "rec", rel(rc[0], gstates[0]["rec"]))

states = [gstates if lt == "gdn" else astates for lt in TYPES]
hr, rr, sr = Hn.reference_step(c, layout, ws, h, r, case, states, flash_bkv=tiles.bkv)
ai = TYPES.index("attn") if "attn" in TYPES else None
gi = TYPES.index("gdn") if HAS_GDN else None
kw = [MK.kernel_weight_layout(c, w, lt) for w, lt in zip(ws, c.layer_types)]
step = MK.build_stage(c, layout, T, max_seqs=MAXS, pps=PPS, pool_shape=pool.shape, plan=plan,
                      stride=stride, tiles=tiles, interpret=INTERP)
h2, r2, pools2 = jax.jit(step)(case.meta, h, r, kw, [pool], MK.rope_table(c, 1024))
pool2 = np.asarray(pools2[0])
for s, (st, n, cx) in enumerate(zip(case.starts, lens, ctxs)):
    print(f"seq {s} (len {n}, ctx {cx}): hidden rel {rel(h2[st:st + n], hr[st:st + n]):.2e} residual rel {rel(r2[st:st + n], rr[st:st + n]):.2e}")
    if ai is not None:
        kk, vv = Hn.read_kv(pool2, case.attn_blocks[s], cx, n, split, BT_TOK, c.NKV, c.HD)
        print(f"   new K rel {rel(jnp.asarray(kk), sr[ai][s]['k'][cx:cx + n]):.2e} V rel {rel(jnp.asarray(vv), sr[ai][s]['v'][cx:cx + n]):.2e}")
if HAS_GDN:
    cv2, rc2 = Hn.load_states(pools2[0], plan, stride, case.gdn_slots[0], (c.CONV - 1, c.QKV), (c.VH, c.KD, c.VD), interpret=INTERP)
    for s in range(2):
        print(f"   gdn seq {s}: conv rel {rel(cv2[s], sr[gi][s]['conv']):.2e} rec rel {rel(rc2[s], sr[gi][s]['rec']):.2e}")
