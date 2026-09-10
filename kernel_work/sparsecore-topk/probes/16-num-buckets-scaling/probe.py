"""Probe 16 -- does the L0 fixed sweep scale with NUM_BUCKETS, and what is its
non-scaling intercept?

A re-run of `probes/09-sweep-split`'s four-way split at two NEW values of
`NUM_BUCKETS`, plus NB=256 as an in-pool control on the harness.

WHY. `probes/13-level-count-refit` §6 prices candidate v5 (sweep the radix) with
`fixed(NB) = 19.192 * NB/256` -- a pure proportionality with ZERO intercept,
anchored at exactly one data point (`probes/07` 19.192, corroborated by
`probes/09` 18.866). Writing the general form
`fixed(NB) = c + (19.192 - c) * NB/256`, probe 13 showed NB=64 stops winning
once `c > 11.12 us`. `c` is completely unbounded by any existing measurement --
probes/03 swept PASS COUNT, not bucket count. Three points determine both the
slope and the intercept.

WHAT IS MEASURED. Exactly probe 09's quantity: with `REFINEMENT_LEVELS = ()`
and `emit`/`fill` off (`E0`), each of the four L0 sweep primitives is dropped
independently by forcing its `parallel_loop` bound to a DYNAMIC zero, and
`E0 - E0-<prim>` is that primitive's cost. `E0 - E0-swp` is the exact joint.
The entry relayout, the score DMA, the `cvt` pass and the L0 histogram are
identical in every arm at a given NB, so they cancel exactly in the difference.

ARMS. One pool, interleaved, `us_median`.

    per NB in {64, 256, 512}:
      E0@nb       emit off, fill off, REFINEMENT_LEVELS=()   the base
      E0-zero@nb  E0 + zero_priv   bound -> dynamic 0
      E0-fold@nb  E0 + fold_priv   bound -> dynamic 0
      E0-scan@nb  E0 + scan_glob   bound -> dynamic 0
      E0-find@nb  E0 + find_bucket bound -> dynamic 0
      E0-swp@nb   E0 + all four
    E0-2d@256     E0@256 through probe 09's f32[b,n] jit entry (relayout control)
    #null_a #null_b #null_c                                  the floor. THREE.

ENTRY SHAPE. `probes/11` showed the obvious `f32[b,n]` jit signature makes XLA
materialise a 40 MiB untiling `copy` plus a retiling fusion that production
never pays. Production's operand is `f32[b, n/128, 128]{2,1,0:T(8,128)}`, whose
minor dimension is exactly one lane-tile, so the flatten is a free bitcast.
Every timed arm here takes that 3-D entry and reshapes inside the jit. `E0-2d`
and the two HLO dumps are what check it landed.

SCORES. The production DSA indexer distribution, `probes/13/level_sim.py`
`make_kv` + `make_scores` copied verbatim (gated host-side by --level-sim),
NOT `rng.standard_normal`. rho = 1 (`row_lengths = n`), matching probe 09, so
the NB=256 control is comparable; the fixed sweep is rho-independent by
construction (its trip counts are compile-time constants).

CORRECTNESS. Separate, untimed, and it runs FIRST and gates the timing stage.
Changing NUM_BUCKETS is not a one-constant edit: `L0_SHIFT` and the refinement
shift/mask schedule must move with it or `bucket` leaves `[0, NUM_BUCKETS)` and
`disable_bounds_checks=True` turns that into a wild VMEM write. Each NB gets a
fully consistent variant (`level_schedule`, verbatim from probe 13) checked
against a numpy top-k oracle on selected-score multisets, plus a negative
control (NB=256 with REFINEMENT_LEVELS=()) that MUST fail.

NOT a correctness experiment in the timing arms: every one of them computes the
wrong answer by construction, and PATCH EVIDENCE is what proves each patch took.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
import os
import random
import sys
import time
import types

import ml_dtypes
import numpy as np

SEED = 20260904          # probes 05/07/08/09 stream
SEED13 = 20260910        # probes/13/level_sim.py SEED
REPS_LO = 20
REPS_HI = 60

NBS = (64, 256, 512)
PRIMS = ("zero", "fold", "scan", "find")

# probes/13 §6, production distribution, prefill, b=512, 2 reps: crit levels.
LEVELS_CRIT = {64: 2.375, 128: 1.938, 256: 1.781, 512: 1.469}
WALK_R1 = 38.004         # us, one refinement L-walk at rho=1 (probes/07)
RHO_PRE = 0.40005        # production prefill rho (hardware_map.md:379)
P07_FIXED_256 = 19.192   # us/level, probes/07 (MARGINAL slope, level-deletion)
P07_EXECS = 3.031        # sweep executions at rho=1, probes/07 -> 58.18 us total
P09_SPLIT_256 = {"zero": 9.238, "fold": 4.900, "scan": 3.498, "find": 1.200}
P09_JOINT_256 = 18.866
P13_BREAKEVEN_C = 11.12  # us; above this, NB=64 stops winning (probes/13 §6)
# The pinned subject: the exact source probes/07/08/09 measured.
SUBJECT_MD5 = "44353b347c96e0eedbd742c710d4540a"


# ==========================================================================
# level_schedule -- VERBATIM from probes/13/level_sim.py. Gated by --level-sim.
# ==========================================================================
def level_schedule(num_buckets):
    """(l0_shift, [(lvl_shift, lvl_mask), ...]) for a given radix.

    NUM_BUCKETS=256 reproduces the shipped constants exactly:
    L0_SHIFT=24, REFINEMENT_LEVELS=((16,0xFF),(8,0xFF),(0,0xFF)).
    Smaller/larger radices keep the same rule -- consume `bits` per level,
    clamp the last level at shift 0 -- which is `modelled`, not in source.
    """
    bits = int(math.log2(num_buckets))
    assert 2**bits == num_buckets
    shifts = []
    s = 32 - bits
    while s > 0:
        s = max(0, s - bits)
        shifts.append(s)
    l0 = 32 - bits
    prev = l0
    levels = []
    for s in shifts:
        levels.append((s, (1 << (prev - s)) - 1))
        prev = s
    return l0, tuple(levels)


# ==========================================================================
# Score generators -- VERBATIM from probes/13/level_sim.py. Gated by
# --level-sim, which asserts the source text of all four is byte-identical.
# ==========================================================================
FP8 = ml_dtypes.float8_e4m3fn
FP8_MAX = float(ml_dtypes.finfo(FP8).max)  # 448.0
E8M0_MIN_EXP, E8M0_MAX_EXP = -127, 127


def _quant_fp8_rowwise(x, ue8m0):
    """`quantization.py:217-300` on the last axis, block = whole axis."""
    absmax = np.abs(x).max(axis=-1, keepdims=True).astype(np.float32)
    scale = absmax * np.float32(1.0 / FP8_MAX)
    if ue8m0:
        with np.errstate(divide="ignore"):
            e = np.ceil(np.log2(scale))
        e = np.where(np.exp2(e) < scale, e + 1, e)
        e = np.clip(e, E8M0_MIN_EXP, E8M0_MAX_EXP)
        scale = np.exp2(e).astype(np.float32)
    inv = np.where(scale > 0, 1.0 / np.maximum(scale, 1e-38), 0.0).astype(np.float32)
    q = np.clip(x * inv, -FP8_MAX, FP8_MAX).astype(FP8)
    return q, scale


def make_kv(n, head_dim, alpha, rng):
    """Compressed-KV rows after `k_norm` (a LayerNorm), fp8+ue8m0 quantized.

    `alpha` is the cosine weight of a direction shared by every KV row -- the
    residual-stream common mode.  Not determinable from source; swept.
    RoPE is skipped: it is an orthogonal per-pair rotation and leaves an
    isotropic Gaussian exactly invariant.
    """
    c = rng.standard_normal(head_dim).astype(np.float32)
    c /= np.linalg.norm(c) / math.sqrt(head_dim)
    kv = rng.standard_normal((n, head_dim)).astype(np.float32)
    kv = alpha * c[None, :] + math.sqrt(max(0.0, 1 - alpha * alpha)) * kv
    kv -= kv.mean(axis=-1, keepdims=True)                       # LayerNorm
    kv /= kv.std(axis=-1, keepdims=True)
    kq, kscale = _quant_fp8_rowwise(kv, ue8m0=True)
    # `scale_val` is read back as bf16 (streamindex_topk.py:267-268); a power
    # of two is exact in bf16.
    return kq, kscale.astype(np.float32).reshape(-1)


def make_scores(kq, kscale, b, n_head, head_dim, rng, signed_w=True,
                head_scale_sigma=0.0, chunk=32):
    """One score row per query token, exactly `streamindex_topk.py:326-338`.

        s = einsum(q_fp8, k_fp8, f32)      :326-331
        s = relu(s)                        :333
        s = s * weights_f32                :334   weights bf16, SIGNED
        s = sum over heads                 :335
        s = s * kscale                     :337
    """
    n = kq.shape[0]
    out = np.empty((b, n), np.float32)
    kqf = kq.astype(np.float32)
    softmax_scale = head_dim**-0.5
    for lo in range(0, b, chunk):
        hi = min(b, lo + chunk)
        m = hi - lo
        q = rng.standard_normal((m, n_head, head_dim)).astype(np.float32)
        if head_scale_sigma:
            q *= np.exp(head_scale_sigma * rng.standard_normal(
                (m, n_head, 1)).astype(np.float32))
        qq, qscale = _quant_fp8_rowwise(q, ue8m0=False)       # axis=-1, fp32
        w_raw = rng.standard_normal((m, n_head)).astype(np.float32)
        if not signed_w:
            w_raw = np.abs(w_raw)
        w = (w_raw * softmax_scale * (n_head**-0.5) * qscale[..., 0])
        w = w.astype(ml_dtypes.bfloat16).astype(np.float32)   # bf16 round-trip
        s = qq.astype(np.float32).reshape(m * n_head, head_dim) @ kqf.T
        np.maximum(s, 0.0, out=s)
        s = s.reshape(m, n_head, n)
        out[lo:hi] = np.einsum("bh,bhn->bn", w, s, dtype=np.float32)
    out *= kscale[None, :]
    return out


HEAD_DIM, N_HEAD = 128, 64   # probes/13/level_sim.py:293


def production_scores(b, n, rng):
    """probes/13/level_sim.py STEP 4b, verbatim: rho=1, alpha=0."""
    kq, ks = make_kv(n, HEAD_DIM, 0.0, rng)
    return make_scores(kq, ks, b, N_HEAD, HEAD_DIM, rng)


# ==========================================================================
# Timing harness -- the FROZEN instrument, or a byte-identical inline copy.
# ==========================================================================
_LAST_ORDER: list[list[str]] = []


def _time(fn, reps: int) -> float:
    t0 = time.perf_counter()
    out = None
    for _ in range(reps):
        out = fn()
    import jax
    jax.block_until_ready(out)
    return time.perf_counter() - t0


def _measure_inline(variants, rounds=9, shuffle_seed=20260902):
    import jax
    rng = random.Random(shuffle_seed)
    lo_best, hi_best, slopes, order_log = {}, {}, {}, []
    for name in variants:
        jax.block_until_ready(variants[name]())
    for _ in range(rounds):
        names = list(variants)
        rng.shuffle(names)
        order_log.append(names)
        lo_r = {}
        for name in names:
            lo_r[name] = _time(variants[name], REPS_LO)
            lo_best[name] = min(lo_best.get(name, math.inf), lo_r[name])
        for name in names:
            hi = _time(variants[name], REPS_HI)
            hi_best[name] = min(hi_best.get(name, math.inf), hi)
            slopes.setdefault(name, []).append(hi - lo_r[name])
    out = {}
    span = REPS_HI - REPS_LO
    for name in variants:
        s = sorted(slopes[name])
        n = len(s)
        med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
        per_device = s[0] / span
        naive = hi_best[name] / REPS_HI
        host = max(naive - per_device, 0.0)
        out[name] = {
            "us": per_device * 1e6,
            "us_median": med / span * 1e6,
            "us_mean": (sum(s) / n) / span * 1e6,
            "us_unpaired": (hi_best[name] - lo_best[name]) / span * 1e6,
            "hostbound_pct": (host / naive * 100.0) if naive > 0 else 0.0,
            "suspect": bool(min(s) <= 0.0 or per_device <= 0.0),
        }
        if out[name]["suspect"]:
            print(f"  !! SUSPECT {name}: min slope <= 0 -- unwarmed recompile. "
                  f"`us` invalid; read us_median "
                  f"({out[name]['us_median']:.2f}).", file=sys.stderr)
    global _LAST_ORDER
    _LAST_ORDER = order_log
    return out


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _load_harness(extra_dirs):
    for d in extra_dirs:
        cand = os.path.join(d, "glm_dsa_block_bench.py")
        if not os.path.isfile(cand):
            continue
        try:
            if d not in sys.path:
                sys.path.insert(0, d)
            mod = importlib.import_module("glm_dsa_block_bench")
            if (mod.REPS_LO, mod.REPS_HI) != (REPS_LO, REPS_HI):
                print(f"  harness at {cand} has REPS "
                      f"{mod.REPS_LO}/{mod.REPS_HI} != {REPS_LO}/{REPS_HI} "
                      f"-- refusing it, using inline", flush=True)
                continue
            print(f"  timing harness: FROZEN INSTRUMENT {cand} "
                  f"(md5 {_md5(open(cand, 'rb').read())})", flush=True)
            return mod.measure, (lambda: mod.LAST_ORDER), "frozen"
        except Exception as exc:  # pragma: no cover
            print(f"  harness import from {d} failed: {exc!r}", flush=True)
    print("  timing harness: INLINE COPY (frozen instrument not importable)",
          flush=True)
    return _measure_inline, (lambda: _LAST_ORDER), "inline"


# ==========================================================================
# Arm construction. Anchors are probe 09's, byte for byte.
# ==========================================================================
SRC_NB = "NUM_BUCKETS = 256"
SRC_L0SHIFT = "L0_SHIFT = 24"
SRC_LEVELS = "REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))"
NO_LEVELS = "REFINEMENT_LEVELS = ()"

ANCHOR_WALK = "        walk_chunks = _cdiv_dyn(i_hi)"
INJECT_ZEROS = (
    "        walk_chunks = _cdiv_dyn(i_hi)\n"
    "        z_dyn = jnp.where(w < 0, walk_chunks, 0)"
)

SRC_EMIT = "            jnp.where(active, walk_chunks, 0),"
SRC_ZERO_PRIV = "            zero_priv(NUM_BUCKETS)"
SRC_FOLD = "            fold_priv(NUM_BUCKETS // LANES)"
SRC_SCAN = "        total = scan_glob(NUM_BUCKETS // LANES)"
SRC_FIND = "        b_star = find_bucket(NUM_BUCKETS // LANES, total - k)"
SRC_FILL = "        fill_bound = jnp.where(out_gate & (total < k), k // LANES, 0)"
NOFILL = ("        fill_bound = jnp.where(out_gate & (total < k) & (w < 0), "
          "k // LANES, 0)")

ANCHORS = {
    "SRC_NB": SRC_NB, "SRC_L0SHIFT": SRC_L0SHIFT, "SRC_LEVELS": SRC_LEVELS,
    "ANCHOR_WALK": ANCHOR_WALK, "SRC_EMIT": SRC_EMIT,
    "SRC_ZERO_PRIV": SRC_ZERO_PRIV, "SRC_FOLD": SRC_FOLD, "SRC_SCAN": SRC_SCAN,
    "SRC_FIND": SRC_FIND, "SRC_FILL": SRC_FILL,
}


def _sub(src: str, old: str, new: str, what: str, expect: int = 1) -> str:
    n = src.count(old)
    assert n == expect, f"patch {what}: expected {expect} occurrence(s), found {n}"
    return src.replace(old, new)


def _radix_src(base_src: str, nb: int, full_levels: bool) -> str:
    """Retune the radix CONSISTENTLY: NUM_BUCKETS, L0_SHIFT, and the schedule.

    `bucket = (key >> L0_SHIFT) + NUM_BUCKETS//2` (sparsecore_topk.py:189-190)
    only lands in [0, NUM_BUCKETS) when L0_SHIFT = 32 - log2(NUM_BUCKETS). With
    `disable_bounds_checks=True` (:481) a mis-set shift is a wild VMEM write,
    not an error, so this is not optional even for a timing-only arm.
    """
    l0, levels = level_schedule(nb)
    src = _sub(base_src, SRC_NB, f"NUM_BUCKETS = {nb}", f"nb{nb}/NUM_BUCKETS")
    src = _sub(src, SRC_L0SHIFT, f"L0_SHIFT = {l0}", f"nb{nb}/L0_SHIFT")
    new_levels = ("REFINEMENT_LEVELS = "
                  + (repr(levels) if full_levels else "()"))
    src = _sub(src, SRC_LEVELS, new_levels, f"nb{nb}/levels")
    return src


# probes/09/arms/*.py md5s. This probe's NB=256 arms must reproduce them BIT
# FOR BIT -- that is what makes "the same probe, two new radices" a checked
# fact rather than a claim, and it is why the NB=256 column is a control.
P09_ARM_MD5 = {
    "E0":      "96e55e33f371ac63b4fc59933df923b2",   # arms/arm-E0.py
    "E0-zero": "7fde5d77d1887fe8d2baa6c33660cca2",   # arms/arm-E0nozero.py
    "E0-fold": "25255759f32a1517c2018bba6c8591f3",   # arms/arm-E0nofold.py
    "E0-scan": "8eee96cac66bc546e27a0969bf74c505",   # arms/arm-E0noscan.py
    "E0-find": "970657a0e7e8b469baceaa1a53c890cf",   # arms/arm-E0nofind.py
    "E0-swp":  "d18b8c2cb8a8a5426bcd02dd09c87a32",   # arms/arm-E0noswp.py
}

DROPS = {
    "E0":      ("emit", "fill"),
    "E0-zero": ("emit", "fill", "zero"),
    "E0-fold": ("emit", "fill", "fold"),
    "E0-scan": ("emit", "fill", "scan"),
    "E0-find": ("emit", "fill", "find"),
    "E0-swp":  ("emit", "fill", "zero", "fold", "scan", "find"),
}
ARM_KINDS = list(DROPS)


def _patch_src(base_src: str, nb: int, kind: str) -> str:
    src = _radix_src(base_src, nb, full_levels=False)
    drops = DROPS[kind]
    zero = "z_dyn"
    src = _sub(src, ANCHOR_WALK, INJECT_ZEROS, f"{kind}/inject")
    if "emit" in drops:
        src = _sub(src, SRC_EMIT, SRC_EMIT.replace("walk_chunks", zero),
                   f"{kind}/emit")
    if "zero" in drops:
        src = _sub(src, SRC_ZERO_PRIV, f"            zero_priv({zero})",
                   f"{kind}/zero_priv")
    if "fold" in drops:
        src = _sub(src, SRC_FOLD, f"            fold_priv({zero})",
                   f"{kind}/fold")
    if "scan" in drops:
        src = _sub(src, SRC_SCAN, f"        total = scan_glob({zero})",
                   f"{kind}/scan")
    if "find" in drops:
        src = _sub(src, SRC_FIND,
                   f"        b_star = find_bucket({zero}, total - k)",
                   f"{kind}/find")
    if "fill" in drops:
        src = _sub(src, SRC_FILL, NOFILL, f"{kind}/fill")
    return src


_MODCOUNT = itertools.count()


def build_mod(src: str, tag: str) -> types.ModuleType:
    mod = types.ModuleType(f"sctopk16_{tag}_{next(_MODCOUNT)}")
    mod.__file__ = f"<probe16:{tag}>"
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    mod._probe_src = src
    return mod


# ==========================================================================
# Fits
# ==========================================================================
def _linfit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    m = sxy / sxx if sxx else float("nan")
    c = my - m * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (c + m * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return c, m, r2


def _t_model(nb, fixed_nb):
    lc = LEVELS_CRIT[nb]
    return (1.0 + lc) * fixed_nb + lc * WALK_R1 * RHO_PRE


# ==========================================================================
# main
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--b", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=9216)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--harness-dir", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--level-sim", default=None,
                    help="dry-run only: probes/13 level_sim.py, for the "
                         "verbatim-generator gate")
    args = ap.parse_args()

    if args.dry_run:
        return _dry_run(args)

    os.makedirs(args.outdir, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import vllm_torchtpu  # noqa: F401  (registers the TPU platform)
    from jax.experimental.pallas import tpu as pltpu

    info = pltpu.get_tpu_info()
    print(f"host={os.uname().nodename} devices={jax.device_count()} "
          f"kind={jax.devices()[0].device_kind}", flush=True)
    print(f"sparse_core={info.sparse_core}  jax={jax.__version__}", flush=True)

    # The SUBJECT is pinned by md5, not by whatever the image happens to ship.
    # Image tag fa63d6df was re-pushed between probes/11 and this job and its
    # copy is now 1a89d451..., so the installed module is NOT the source
    # probes/07-09 measured. Load the pinned file explicitly and say which.
    import vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk as installed
    inst_md5 = _md5(open(installed.__file__, "rb").read())
    base_path = args.subject or installed.__file__
    raw = open(base_path, "rb").read()
    base_src = raw.decode()
    print(f"subject: {base_path}  md5={_md5(raw)}", flush=True)
    print(f"installed: {installed.__file__}  md5={inst_md5}  "
          f"{'SAME' if inst_md5 == _md5(raw) else 'DIFFERENT -- using the '
             'pinned --subject, not the image copy'}", flush=True)
    if _md5(raw) != SUBJECT_MD5:
        print(f"SUBJECT_GATE_FAIL: {_md5(raw)} != {SUBJECT_MD5}", flush=True)
        return 2
    print(f"SUBJECT_GATE_OK  {SUBJECT_MD5}", flush=True)
    inc = build_mod(base_src, "incumbent")

    b, k = args.b, args.k
    n = ((args.ctx + 1024) + 15) // 16 * 16
    assert n % 128 == 0, n
    p = inc._pick_partition(b, n, k)
    print(f"shape b={b} n={n} k={k} P={p} waves={(b + 31) // 32} "
          f"rho=1  entry=f32[{b},{n // 128},128]", flush=True)
    assert p == 1, "single-stage path only"

    # ---------- scores: the production distribution, rho = 1 ---------------
    t0 = time.time()
    sc_np = production_scores(b, n, np.random.default_rng(SEED13))
    print(f"scores: production distribution (probes/13 STEP 4b, alpha=0), "
          f"{time.time() - t0:.1f}s  shape={sc_np.shape}  "
          f"zeros={float((sc_np == 0).mean()) * 100:.2f}%  "
          f"md5={_md5(sc_np.tobytes())[:16]}", flush=True)
    sc3 = jnp.asarray(sc_np.reshape(b, n // 128, 128))
    sc2 = jnp.asarray(sc_np)
    lengths = jnp.full((b,), n, jnp.int32)

    results = {
        "b": b, "n": n, "k": k, "P": p, "rounds": args.rounds,
        "subject_md5": _md5(raw), "seed_scores": SEED13, "rho": 1.0,
        "entry": f"f32[{b},{n // 128},128]",
    }

    # =====================================================================
    # STAGE 1 -- CORRECTNESS. Runs first and gates the timing stage.
    # =====================================================================
    print("\n" + "=" * 78, flush=True)
    print("STAGE 1  CORRECTNESS -- does a retuned radix lower, and is it exact?",
          flush=True)
    print("=" * 78, flush=True)

    t0 = time.time()
    part = np.argpartition(-sc_np, k - 1, axis=1)[:, :k]
    ref_scores = np.sort(np.take_along_axis(sc_np, part, axis=1), axis=1)
    print(f"  numpy oracle built in {time.time() - t0:.1f}s", flush=True)

    corr = {}
    cases = [(nb, True, f"full@{nb}") for nb in NBS]
    cases.append((256, False, "NEGCTL@256(no-levels)"))
    for nb, full, label in cases:
        l0, levels = level_schedule(nb)
        rec = {"nb": nb, "full_levels": full, "l0_shift": l0,
               "levels": [list(x) for x in levels],
               "max_levels": len(levels)}
        try:
            src = _radix_src(base_src, nb, full_levels=full)
            rec["src_md5"] = _md5(src.encode())
            mod = build_mod(src, f"corr{nb}{'F' if full else 'N'}")
            g = jax.jit(lambda s, l, _m=mod: _m.sparsecore_topk(
                s.reshape(b, n), k=k, row_lengths=l))
            got = np.asarray(jax.block_until_ready(g(sc3, lengths)))
            neg = int((got < 0).sum())
            gs = np.sort(np.take_along_axis(sc_np, np.maximum(got, 0), axis=1),
                         axis=1)
            bad = int((~np.all(gs == ref_scores, axis=1)).sum())
            rec.update(lowered=True, pad_slots=neg, rows_wrong=bad,
                       exact=bool(bad == 0 and neg == 0))
            del mod, g
        except Exception as exc:
            rec.update(lowered=False, error=repr(exc)[:400], exact=False)
        corr[label] = rec
        st = ("EXACT" if rec.get("exact")
              else ("WRONG" if rec.get("lowered") else "DID-NOT-LOWER"))
        print(f"  {label:24s} L0_SHIFT={rec['l0_shift']:2d} "
              f"levels={rec['max_levels']} -> {st:14s} "
              f"rows_wrong={rec.get('rows_wrong', '-')}/{b} "
              f"pad={rec.get('pad_slots', '-')}"
              + (f"  {rec.get('error', '')}" if not rec.get("lowered", True)
                 else ""), flush=True)
    results["correctness"] = corr

    # Fail closed. The SHIPPING config must be exact (else the oracle is
    # broken), and the negative control must NOT be (else the oracle is
    # insensitive and an EXACT above establishes nothing).
    gate_ok = True
    if not corr["full@256"].get("exact"):
        print("  !! CORRECTNESS_GATE_FAIL: the SHIPPING radix (NB=256) does "
              "not match the oracle. The oracle, not the kernel, is wrong.",
              file=sys.stderr, flush=True)
        gate_ok = False
    if corr["NEGCTL@256(no-levels)"].get("exact"):
        print("  !! CORRECTNESS_GATE_FAIL: the negative control (no refinement "
              "levels) MATCHED the oracle. The check is insensitive and every "
              "EXACT above is uninterpretable.", file=sys.stderr, flush=True)
        gate_ok = False
    if not gate_ok:
        results["gate"] = "CORRECTNESS_GATE_FAIL"
        with open(os.path.join(args.outdir, "nb_scaling.json"), "w") as fh:
            json.dump(results, fh, indent=2, sort_keys=True, default=str)
        print("ABORTING before the timing stage.", file=sys.stderr, flush=True)
        return 2
    print("  CORRECTNESS_GATE_OK  (shipping radix exact AND negative control "
          "failed -- the oracle can see a broken selection)", flush=True)
    results["gate"] = "CORRECTNESS_GATE_OK"

    # =====================================================================
    # STAGE 2 -- build the timing arms
    # =====================================================================
    print("\n" + "=" * 78, flush=True)
    print("STAGE 2  ARMS", flush=True)
    print("=" * 78, flush=True)
    mods, srcs = {}, {}
    for nb in NBS:
        for kind in ARM_KINDS:
            name = f"{kind}@{nb}"
            src = _patch_src(base_src, nb, kind)
            srcs[name] = src
            mods[name] = build_mod(src, name.replace("-", "no").replace("@", "at"))
            safe = name.replace("-", "no").replace("@", "at")
            with open(os.path.join(args.outdir, f"arm-{safe}.py"), "w") as fh:
                fh.write(src)
            note = ""
            if nb == 256:
                ok = _md5(src.encode()) == P09_ARM_MD5[kind]
                note = ("  == probes/09 arm, bit for bit" if ok
                        else "  !! DIFFERS from probes/09 arm")
            print(f"  arm {name:14s} drops={','.join(DROPS[kind]):40s} "
                  f"src_md5={_md5(src.encode())}{note}", flush=True)
    p09ok = all(_md5(srcs[f"{kk}@256"].encode()) == v
                for kk, v in P09_ARM_MD5.items())
    print(f"  P09_SRC_GATE: {'OK' if p09ok else 'FAIL'} -- all six NB=256 arms "
          f"reproduce probes/09/arms/*.py bit for bit", flush=True)
    results["p09_src_gate"] = bool(p09ok)
    # The 2-D-entry control reuses E0@256's module -- SAME program, different
    # jit signature. That is the whole point of the arm.
    print(f"  arm {'E0-2d@256':14s} (E0@256 module, f32[b,n] jit entry)",
          flush=True)

    measure, last_order, harness_kind = _load_harness(
        list(args.harness_dir) + [
            os.path.dirname(os.path.abspath(__file__)),
            "/workspace/microbench", "/app/microbench", os.getcwd(),
        ])
    results["harness"] = harness_kind

    thunks, jits = {}, {}
    for name, mod in mods.items():
        jits[name] = jax.jit(lambda s, l, _m=mod: _m.sparsecore_topk(
            s.reshape(b, n), k=k, row_lengths=l))
        thunks[name] = (lambda _g=jits[name]: _g(sc3, lengths))
    jits["E0-2d@256"] = jax.jit(
        lambda s, l, _m=mods["E0@256"]: _m.sparsecore_topk(
            s, k=k, row_lengths=l))
    thunks["E0-2d@256"] = lambda _g=jits["E0-2d@256"]: _g(sc2, lengths)
    _nullf = jax.jit(lambda a: a + 1.0)
    _nullx = jnp.zeros((8, 128), jnp.float32)
    for nm in ("#null_a", "#null_b", "#null_c"):
        thunks[nm] = lambda: _nullf(_nullx)

    # =====================================================================
    # STAGE 3 -- HLO gate on the entry shape
    # =====================================================================
    print("\n" + "=" * 78, flush=True)
    print("STAGE 3  HLO -- is the 3-D entry free of the probes/11 relayout?",
          flush=True)
    print("=" * 78, flush=True)
    hlo = {}
    for tag, gg, arg in (("3d", jits["E0@256"], sc3),
                         ("2d", jits["E0-2d@256"], sc2)):
        txt = gg.lower(arg, lengths).compile().as_text()
        pth = os.path.join(args.outdir, f"hlo-E0at256-{tag}.txt")
        with open(pth, "w") as fh:
            fh.write(txt)
        df = txt.count("sparse-core-data-format-call")
        cps = [ln.strip() for ln in txt.splitlines()
               if " copy(" in ln or " copy-start(" in ln]
        big = [ln for ln in cps if "f32[" in ln or "s32[" in ln]
        hlo[tag] = {"path": pth, "chars": len(txt),
                    "sparse_core_data_format_call": df,
                    "copy_ops": len(cps), "copy_lines": big[:8]}
        print(f"  entry={tag:2s}  chars={len(txt):8d}  "
              f"sparse-core-data-format-call={df}  copy ops={len(cps)}",
              flush=True)
        for ln in big[:6]:
            print(f"      {ln[:190]}", flush=True)
    results["hlo"] = hlo
    clean = (hlo["3d"]["sparse_core_data_format_call"] == 0)
    print(f"  ENTRY_GATE: 3-D entry is "
          f"{'CLEAN (no df-call)' if clean else 'DIRTY (df-call present)'}; "
          f"2-D entry has {hlo['2d']['sparse_core_data_format_call']} df-call "
          f"line(s). Note the fixed(NB) split is a WITHIN-NB difference, so "
          f"either way the entry cancels.", flush=True)

    # =====================================================================
    # STAGE 4 -- patch evidence
    # =====================================================================
    print("\n" + "=" * 78, flush=True)
    print("STAGE 4  PATCH EVIDENCE (every arm must differ from its own E0)",
          flush=True)
    print("=" * 78, flush=True)
    ev = {}
    for nb in NBS:
        ref = np.asarray(thunks[f"E0@{nb}"]())
        for kind in ARM_KINDS:
            name = f"{kind}@{nb}"
            got = np.asarray(thunks[name]())
            same = int((np.sort(got, axis=1) == np.sort(ref, axis=1))
                       .all(axis=1).sum())
            ev[name] = {"rows_matching_E0": same, "rows": int(b)}
            flag = ""
            if kind != "E0" and same == b:
                flag = "  !! reproduces E0 exactly -- patch NOT effective"
            print(f"  {name:14s} {same:5d}/{b} rows match E0@{nb}{flag}",
                  flush=True)
    results["patch_evidence"] = ev
    ineffective = [nm for nm, v in ev.items()
                   if not nm.startswith("E0@")
                   and v["rows_matching_E0"] == b]
    if ineffective:
        print(f"  !! PATCH_EVIDENCE_FAIL: {ineffective}", file=sys.stderr,
              flush=True)

    # =====================================================================
    # STAGE 5 -- timing
    # =====================================================================
    print("\n" + "=" * 78, flush=True)
    print(f"STAGE 5  TIMING -- {len(thunks)} arms, ONE pool, "
          f"{args.rounds} interleaved rounds, us_median", flush=True)
    print("=" * 78, flush=True)
    out = measure(thunks, rounds=args.rounds)
    for name in thunks:
        r = out[name]
        print(f"  {name:14s} us_median={r['us_median']:10.3f}  "
              f"us={r['us']:10.3f}  us_mean={r['us_mean']:10.3f}  "
              f"hostbound={r['hostbound_pct']:5.1f}%  "
              f"suspect={r.get('suspect')}", flush=True)

    m = {kk: out[kk]["us_median"] for kk in out}
    nulls = [m["#null_a"], m["#null_b"], m["#null_c"]]
    null_mean = sum(nulls) / 3.0
    null_spread = max(abs(a - c) for a, c in itertools.combinations(nulls, 2))
    print(f"\n  nulls {nulls[0]:.3f} / {nulls[1]:.3f} / {nulls[2]:.3f}  "
          f"-> mean {null_mean:.3f}, RESOLUTION FLOOR (max unordered pairwise "
          f"spread) {null_spread:.3f} us", flush=True)

    # ---------- the split, per NB ------------------------------------------
    print("\n  --- the four-way sweep split, per NUM_BUCKETS ---", flush=True)
    split = {}
    for nb in NBS:
        e0 = m[f"E0@{nb}"]
        comps = {p_: e0 - m[f"E0-{p_}@{nb}"] for p_ in PRIMS}
        joint = e0 - m[f"E0-swp@{nb}"]
        ssum = sum(comps.values())
        split[nb] = {"E0": e0, "components": comps, "sum": ssum,
                     "joint": joint,
                     "additivity_ratio": (ssum / joint) if joint else float("nan")}
        print(f"  NB={nb:4d}  E0={e0:9.3f}   "
              + "  ".join(f"{p_}={comps[p_]:7.3f}" for p_ in PRIMS)
              + f"   sum={ssum:8.3f}  joint={joint:8.3f}  "
              f"addratio={split[nb]['additivity_ratio']:.3f}", flush=True)
        for p_ in PRIMS:
            if abs(comps[p_]) < null_spread:
                print(f"      ~ NB={nb} {p_} = {comps[p_]:.3f} us is UNDER the "
                      f"{null_spread:.3f} us floor -- unresolved.", flush=True)
    results["split"] = {str(kk): v for kk, v in split.items()}

    # ---------- cross-probe control ----------------------------------------
    print("\n  --- cross-probe control: NB=256 against probes/09 "
          "(j-4e0df30a) ---", flush=True)
    ctl = {}
    for p_ in PRIMS:
        got = split[256]["components"][p_]
        exp = P09_SPLIT_256[p_]
        ctl[p_] = {"probe16": got, "probe09": exp,
                   "delta_pct": 100.0 * (got - exp) / exp}
        print(f"  {p_:5s} probe16 {got:8.3f}  probes/09 {exp:8.3f}  "
              f"delta {ctl[p_]['delta_pct']:+7.2f}%", flush=True)
    dj = 100.0 * (split[256]["joint"] - P09_JOINT_256) / P09_JOINT_256
    ctl["joint"] = {"probe16": split[256]["joint"], "probe09": P09_JOINT_256,
                    "delta_pct": dj}
    print(f"  joint probe16 {split[256]['joint']:8.3f}  probes/09 "
          f"{P09_JOINT_256:8.3f}  delta {dj:+7.2f}%   "
          f"(probes/07 slope {P07_FIXED_256:.3f})", flush=True)
    results["cross_probe_nb256"] = ctl

    # ---------- ADJUDICATION: probes/07 vs probes/09 -------------------------
    # Two MEASURED numbers in this exercise price the same per-level fixed
    # sweep and disagree by 3.1x. probes/07 fitted a MARGINAL slope by deleting
    # whole levels (19.192 us/level, so 19.192*P07_EXECS = 58.2 us at rho=1);
    # probes/09 deleted the four sweep primitives in place (18.866 us). This
    # arm is built exactly like probes/09's (P09_SRC_GATE proves the sources
    # are byte-identical), so it re-measures 09's construction, not 07's.
    print("\n  --- ADJUDICATION: probes/07 (19.192 us/level, marginal, "
          "level-deletion) vs probes/09 (18.866 us, primitive-deletion) ---",
          flush=True)
    got256 = split[256]["joint"]
    d09 = 100.0 * (got256 - P09_JOINT_256) / P09_JOINT_256
    p07_total = P07_FIXED_256 * P07_EXECS
    d07l = 100.0 * (got256 - P07_FIXED_256) / P07_FIXED_256
    d07t = 100.0 * (got256 - p07_total) / p07_total
    print(f"  MEASURED NB=256 one-level fixed sweep = {got256:.3f} us "
          f"(E0 - E0-swp, {args.rounds} rounds, floor {null_spread:.3f})",
          flush=True)
    print(f"    vs probes/09 18.866            {d09:+7.2f}%", flush=True)
    print(f"    vs probes/07 19.192 per level  {d07l:+7.2f}%", flush=True)
    print(f"    vs probes/07 {p07_total:.2f} total "
          f"({P07_EXECS} execs x 19.192)  {d07t:+7.2f}%", flush=True)
    near09 = abs(got256 - P09_JOINT_256) <= max(null_spread, 0.10 * P09_JOINT_256)
    near07 = abs(got256 - p07_total) <= max(null_spread, 0.10 * p07_total)
    adj = ("probes/09" if near09 and not near07
           else "probes/07" if near07 and not near09
           else "NEITHER")
    print(f"  ADJUDICATION: the direct measurement lands on {adj}.", flush=True)
    if adj == "probes/09":
        print("  => 07's 19.192 us/LEVEL is NOT the sweep alone: deleting a "
              "level removes the sweep AND whatever else is per-level. The "
              "sweep is ~18.9 us total, ~6.2 us/level at 3.031 execs.",
              flush=True)
    results["adjudication"] = {
        "measured_nb256_us": got256, "probe09_us": P09_JOINT_256,
        "probe07_per_level_us": P07_FIXED_256, "probe07_execs": P07_EXECS,
        "probe07_implied_total_us": p07_total, "delta_vs_09_pct": d09,
        "delta_vs_07_total_pct": d07t, "lands_on": adj,
        "floor_us": null_spread}

    # ---------- is ONE scalar fixed(NB) even the right model? ----------------
    print("\n  --- do the four primitives scale together? (if not, a single "
          "fixed(NB) scalar is the wrong model) ---", flush=True)
    together = {}
    for p_ in PRIMS:
        base = split[256]["components"][p_]
        rats = [(split[nb]["components"][p_] / base) if base else float("nan")
                for nb in NBS]
        ideal = [nb / 256.0 for nb in NBS]
        together[p_] = {"ratio_to_nb256": rats, "proportional_would_be": ideal,
                        "resolved": abs(base) > null_spread}
        print(f"  {p_:5s} x(NB=64,256,512)/x(256) = ["
              + ", ".join(f"{r:.3f}" for r in rats)
              + "]  vs pure-proportional [0.250, 1.000, 2.000]"
              + ("" if abs(base) > null_spread else "   <-- UNRESOLVED at 256"),
              flush=True)
    results["scaling_together"] = together

    # ---------- entry-shape control ----------------------------------------
    relayout = m["E0-2d@256"] - m["E0@256"]
    print(f"\n  --- entry-shape control ---\n  E0-2d@256 {m['E0-2d@256']:.3f} "
          f"- E0@256 {m['E0@256']:.3f} = {relayout:+.3f} us of XLA relayout "
          f"the 3-D (production-shaped) entry does not pay. probes/11 measured "
          f"-60.416 on its own arms.", flush=True)
    results["entry_relayout_us"] = relayout

    # ---------- THE FIT ----------------------------------------------------
    print("\n  --- THE FIT: fixed(NB) = c + m*NB, over three measured points "
          "---", flush=True)
    xs = list(NBS)
    ys = [split[nb]["joint"] for nb in NBS]
    c, slope, r2 = _linfit(xs, ys)
    print(f"  points: " + "  ".join(f"NB={x}:{y:.3f}" for x, y in zip(xs, ys)),
          flush=True)
    print(f"  fixed(NB) = {c:.3f} + {slope:.6f} * NB    (r2 = {r2:.5f})",
          flush=True)
    print(f"  intercept c = {c:.3f} us = {100.0 * c / ys[1]:.1f}% of the "
          f"measured NB=256 sweep ({ys[1]:.3f})", flush=True)
    print(f"  probes/13 break-even: NB=64 stops winning at c > "
          f"{P13_BREAKEVEN_C:.2f} us. Measured c is "
          f"{'ABOVE' if c > P13_BREAKEVEN_C else 'BELOW'} it.", flush=True)
    results["fit_fixed"] = {"c_us": c, "m_us_per_bucket": slope, "r2": r2,
                            "points": dict(zip(map(str, xs), ys))}

    # per-primitive fits against their OWN trip counts
    print("\n  --- per-primitive fits (trip counts from source: "
          "zero_priv bound = NB, the other three = NB//LANES) ---", flush=True)
    perp = {}
    for p_ in PRIMS:
        trips = [nb if p_ == "zero" else nb // 16 for nb in NBS]
        vals = [split[nb]["components"][p_] for nb in NBS]
        pc, pm, pr2 = _linfit(trips, vals)
        share = 100.0 * (pm * trips[-1]) / vals[-1] if vals[-1] else float("nan")
        perp[p_] = {"intercept_us": pc, "slope_us_per_trip": pm, "r2": pr2,
                    "trips": trips, "us": vals,
                    "pct_scaling_at_nb512": share}
        print(f"  {p_:5s} trips={trips}  us=["
              + ", ".join(f"{v:.3f}" for v in vals)
              + f"]  -> {pc:+7.3f} + {pm:.5f}/trip  r2={pr2:.4f}  "
              f"intercept is {100.0 * pc / vals[1] if vals[1] else float('nan'):.0f}% "
              f"of its NB=256 cost", flush=True)
    results["fit_primitives"] = perp

    # ---------- v5 verdict --------------------------------------------------
    print("\n  --- v5: T(NB) = (1+levels_crit)*fixed(NB) + levels_crit*walk*rho "
          f"  [walk={WALK_R1} us MEASURED probes/07, rho={RHO_PRE}, "
          "levels_crit MEASURED probes/13 sim] ---", flush=True)
    t256 = _t_model(256, split[256]["joint"])
    v5 = {}
    for nb in NBS:
        t = _t_model(nb, split[nb]["joint"])
        v5[str(nb)] = {"fixed_us": split[nb]["joint"],
                       "levels_crit": LEVELS_CRIT[nb], "T_us": t,
                       "delta_us": t - t256,
                       "delta_pct": 100.0 * (t - t256) / t256}
        print(f"  NB={nb:4d}  fixed={split[nb]['joint']:8.3f}  "
              f"levels_crit={LEVELS_CRIT[nb]:.3f}  T={t:8.3f}  "
              f"delta vs NB=256 {t - t256:+8.3f} us = "
              f"{100.0 * (t - t256) / t256:+6.2f}%", flush=True)
    # probe 13's own model, for the record: fixed = 19.192*NB/256, c = 0
    print("  (probes/13 MODELLED the same table with fixed(NB)=19.192*NB/256: "
          "NB=64 -14.82%, NB=512 +19.28%)", flush=True)
    results["v5"] = v5
    verdict = ("SURVIVES" if v5["64"]["delta_us"] < -null_spread
               else ("REFUTED" if v5["64"]["delta_us"] > null_spread
                     else "UNRESOLVED"))
    print(f"\n  V5_VERDICT: NB=64 {verdict} "
          f"({v5['64']['delta_pct']:+.2f}% modelled on measured fixed(NB))",
          flush=True)
    results["v5_verdict"] = verdict

    results["raw"] = out
    results["nulls"] = {"values": nulls, "mean": null_mean,
                        "spread_us": null_spread}
    results["order"] = last_order()
    path = os.path.join(args.outdir, "nb_scaling.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=str)
    print(f"\nwrote {path}", flush=True)
    return 0


# ==========================================================================
# host-side gate
# ==========================================================================
def _dry_run(args) -> int:
    path = args.subject
    raw = open(path, "rb").read()
    src = raw.decode()
    print(f"subject: {path}  md5={_md5(raw)}")

    bad = 0
    for nm, a in ANCHORS.items():
        cnt = src.count(a)
        if cnt != 1:
            bad += 1
        print(f"  anchor {nm:14s} count={cnt} {'OK' if cnt == 1 else 'WRONG'}")
    if bad:
        print(f"ANCHOR_GATE_FAIL: {bad} anchor(s) not unique")
        return 1
    print("ANCHOR_GATE_OK")

    # the verbatim-generator gate
    if args.level_sim:
        ls = open(args.level_sim).read()
        miss = []
        for fn in (level_schedule, _quant_fp8_rowwise, make_kv, make_scores):
            import inspect
            body = inspect.getsource(fn)
            if body not in ls:
                miss.append(fn.__name__)
        if miss:
            print(f"VERBATIM_GATE_FAIL: {miss} differ from {args.level_sim}")
            return 1
        print(f"VERBATIM_GATE_OK  level_schedule/_quant_fp8_rowwise/make_kv/"
              f"make_scores are byte-identical to {args.level_sim}")
    else:
        print("VERBATIM_GATE_SKIPPED (--level-sim not given)")

    # radix schedules
    for nb in NBS:
        l0, lv = level_schedule(nb)
        cover = (32 - l0) + sum(
            (lv[i - 1][0] if i else l0) - lv[i][0] for i in range(len(lv)))
        print(f"  radix NB={nb:4d}  L0_SHIFT={l0:2d}  levels={len(lv)}  "
              f"{lv}  bits_covered={cover}")
        assert cover == 32, (nb, cover)
        assert lv[-1][0] == 0

    import difflib
    for nb in NBS:
        base = _patch_src(src, nb, "E0")
        for kind in ARM_KINDS:
            s = _patch_src(src, nb, kind)
            compile(s, f"<probe16:{kind}@{nb}>", "exec")
            d = list(difflib.unified_diff(base.splitlines(), s.splitlines(),
                                          n=0, lineterm=""))
            ch = sum(1 for ln in d
                     if (ln.startswith("+") or ln.startswith("-"))
                     and not ln.startswith(("+++", "---")))
            print(f"  arm {kind:8s}@{nb:<4d} compiles OK  "
                  f"changed_lines_vs_E0@{nb}={ch:2d}  src_md5={_md5(s.encode())}")
        for full in (True, False):
            s = _radix_src(src, nb, full_levels=full)
            compile(s, f"<probe16:corr{nb}{full}>", "exec")
    # the E0 arms across NB must differ from one another
    e0s = {nb: _md5(_patch_src(src, nb, "E0").encode()) for nb in NBS}
    assert len(set(e0s.values())) == len(NBS), e0s
    print(f"  E0 arms distinct across NB: {e0s}")
    for kk, v in P09_ARM_MD5.items():
        got = _md5(_patch_src(src, 256, kk).encode())
        print(f"  P09_SRC_GATE {kk:8s} {got} "
              f"{'OK' if got == v else 'FAIL (expected ' + v + ')'}")
        if got != v:
            return 1
    print("P09_SRC_GATE_OK")
    print("COMPILE_GATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
