"""Probe 23 -- THE RANKING JOB. Seven candidate arms plus the incumbent,
one pool, two brackets, both production rho values, b in {64, 1024}.

WHAT THIS IS. Step 3 of the write-tpu-kernels exercise. Six variant agents
wrote candidates; none of them ran a timing job. This is the only job whose
numbers decide the ranking, so every property that could make it lie is
pinned here rather than assumed.

THE ARMS (coordinator's frozen table; md5 is the gate, scope name is the proof)

    base      44353b347c96e0eedbd742c710d4540a   bare  sc_topk_direct_b*_n*_k*
    v11       67a16569d281d845aae2ad2ca0040583   _v11retile
    v12       408ef72e9afc935c928f2803764878cc   _f32in
    v4ragged  69c8834e130e23999e0284929230da3a   _v4ragged
    v4c2048   88a67892e8c063b5d9fea96de72e99c4   _v4c2048
    v2        3c041d952227dc006fb939797325aedd   _v2r1
    v2big     79f05d554b98e4681a569d1157f8e6a6   _v2bigscratch
    v10a2     f83332273f1cdc05995965c0405e8665   _sc2
              + 21f2d777419b3cf43ba940f004bd765a (sparsecore_topk_sample.py)

`v3` is dropped: it failed the frozen 68 (46 failed / 22 passed, job
j-188b0f9e) with a diagnosed mechanism, and was already modelled to lose.

TWO BRACKETS, AND HOW THE INSIDE-KERNEL ONE IS BUILT WITHOUT PARAPHRASING
ANY VARIANT'S ENTRY CODE.

  whole-call    jax.jit around `mod.sparsecore_topk(s.reshape(b, n), k, rl)`
                at the PRODUCTION 3-D entry f32[b, n/128, 128]. This is
                probes/18's P3.*/D3.* bracket and the one probes/20 settled
                is sufficient for v12 -- v12 deletes a priced op in ENTRY,
                OUTSIDE the kernel, so an inside-kernel bracket reads it as a
                flat null. That null is an instrument artifact, not a result.

  inside-kernel the `pl.kernel(...)` invocation ONLY. Built by monkeypatching
                `jax.experimental.pallas.kernel` and running the variant's own
                `_sc_topk_direct` EAGERLY (not under jit), so the spy receives
                the kernel object and its operands already CONCRETE. Nothing
                about any variant's bitcast / pad / reshape is re-implemented
                here -- the variant computes its own operands and we time what
                it handed the kernel. `v12` hands the kernel f32 where the
                incumbent hands it i32; that difference survives the capture
                untouched.

  Prize quoted against (DESIGN.md): v2/v2big/v4*/v10 -> inside-kernel;
  v11 -> whole-call (output edge); v12 -> whole-call (entry op).

RHO. Measured at the production distribution AND the production arrangement,
never reweighted from rho=1: probes/18 showed rho is not a sufficient
statistic (8.26% from row arrangement alone at an identical multiset).

  b=1024  prefillX  stratified Uniform{1..8192},    permuted, rho=0.40005 EXACT
  b=64    decodeX   stratified Uniform{8193..9215}, permuted, rho=0.850   EXACT

Both come from `p18_probe.build_lengths`, which transcribes
probes/13/level_sim.py verbatim. THE b=64 CELL IS REAL PRODUCTION DECODE,
rho=0.850 -- it is NOT the frozen bench's `_row_lengths`, which cycles one
edge set regardless of b and lands at rho=0.391, a small-batch PREFILL
occupancy run wearing a decode label.

  rho=1 is a NAMED EXCEPTION CELL, not a ladder rung: v2 (explicitly
  requested -- it separates two hypotheses 3x apart), its comparators
  base/nulls/v2big, and v4ragged, which is a null at rho=1 BY CONSTRUCTION
  and therefore a free positive control on the instrument.

POOL. One pool, all arms, >= 9 interleaved rounds, reshuffled every round,
paired REPS 20/60 slope, read `us_median` never `us`. Every thunk warmed
before it is timed (a cold thunk's paired slope is minus its compile time).
Four identical-source incumbent modules (`base` + three nulls, separately
staged, identical operands) sit in every cell: their spread IS the
resolution floor, and it also gives the incumbent the >= 2 slots the
coordinator asked for at every cell.

ARM ACTIVATION. Two independent checks, neither of which is the flag or the
env that was set:
  1. the `name=` kwarg intercepted at `pl.kernel` construction -- read
     directly, no HLO parsing;
  2. a regex over the lowered AND compiled HLO text of the whole-call jit.
Both must contain the arm's suffix as a SUBSTRING (v2/v2big PREPEND their
tag, v11/v12/v4* APPEND it -- an `endswith` test would spuriously fail v2
under emit_keys), and must contain NO OTHER arm's suffix. `base` must show a
bare name and no suffix at all. Any failure aborts before timing.

CORRECTNESS is NOT established here. It was established by the frozen 68-id
suite plus a cross-module bit/multiset gate in jobs j-188b0f9e, j-24463f36
and j-f114bd8a (all six variants: 68 passed, 30/30 exact, 30/30
bit-identical). The check in STAGE 2 below is a re-implemented smoke test on
sampled rows whose only job is to catch "the module I timed is not the module
that passed" -- it is not the gate.

NOT MEASURED HERE: end-to-end effect, anything multi-chip, and any
composition of two candidates. Composition is step 4's job; the arms overlap
and their percentages MUST NOT be added.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import types

import numpy as np

# --- the frozen arm table ---------------------------------------------------
# (tag, subdir, md5, suffix or None for the bare incumbent, extra files,
#  env set at import, kwargs for the eager `_sc_topk_direct` capture)
ARMS = [
    ("base", "base", "44353b347c96e0eedbd742c710d4540a", None, {}, {}, {}),
    ("v11", "v11", "67a16569d281d845aae2ad2ca0040583", "_v11retile", {}, {}, {}),
    ("v12", "v12", "408ef72e9afc935c928f2803764878cc", "_f32in", {}, {}, {}),
    ("v4ragged", "v4ragged", "69c8834e130e23999e0284929230da3a", "_v4ragged",
     {}, {}, {}),
    ("v4c2048", "v4c2048", "88a67892e8c063b5d9fea96de72e99c4", "_v4c2048",
     {}, {}, {}),
    ("v2", "v2", "3c041d952227dc006fb939797325aedd", "_v2r1", {}, {}, {}),
    ("v2big", "v2big", "79f05d554b98e4681a569d1157f8e6a6", "_v2bigscratch",
     {}, {}, {}),
    ("v10a2", "v10a2", "f83332273f1cdc05995965c0405e8665", "_sc2",
     {"sparsecore_topk_sample.py": "21f2d777419b3cf43ba940f004bd765a"},
     {"SCTOPK_SAMPLE_AND_COUNT": "1", "SCTOPK_COUNT_ARITY": "2"},
     {"sample_and_count": True, "count_arity": 2}),
]

ALL_SUFFIXES = [a[3] for a in ARMS if a[3]]

# Arms that additionally get the NAMED rho=1 cell at b=1024.
RHO1_ARMS = {"base", "v2", "v2big", "v4ragged", "#null_a", "#null_b",
             "#null_c"}

P18_MD5 = "ceafc734d10632672d3db48f2cea23f6"
HARNESS_MD5 = "08375d7068365e1a776c58fef09cd6d0"

# probes/18 MEASURED denominators -- the only ones that may be used.
# 155.82 us is MODELLED and is forbidden; every probes/09 number is unsafe.
DENOM_PREFILL = 275.185      # us, P3.prefillX
DENOM_PREFILL_GATED = 291.075  # us, gate-corrected per invocation
DENOM_DECODE = 49.171        # us, D3.decodeX

NAME_RE = re.compile(r"sc_topk_direct_b\d+_n\d+_k\d+(?:_[A-Za-z0-9]+)*")
BARE_RE = re.compile(r"sc_topk_direct_b\d+_n\d+_k\d+(?![_A-Za-z0-9])")


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _spread(vals):
    return max(vals) - min(vals)


def _load_p18(path: str):
    spec = importlib.util.spec_from_file_location("p18_probe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p18_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


# ==========================================================================
# The inside-kernel capture. Generic across every variant: nothing about any
# variant's entry transform is re-implemented, the variant computes it.
# ==========================================================================
class KernelSpy:
    """Intercepts `pl.kernel(...)` and records the object AND its operands.

    Used around an EAGER call to the variant's own `_sc_topk_direct`, so the
    operands the spy sees are concrete `jax.Array`s, not tracers.
    """

    def __init__(self, pl):
        self.pl = pl
        self.real = pl.kernel
        self.calls = []

    def __enter__(self):
        spy = self

        def fake_kernel(*a, **kw):
            kern = spy.real(*a, **kw)
            name = kw.get("name")

            def call(*ops):
                out = kern(*ops)
                spy.calls.append({"kern": kern, "name": name,
                                  "ops": list(ops)})
                return out

            return call

        self.pl.kernel = fake_kernel
        return self

    def __exit__(self, *exc):
        self.pl.kernel = self.real
        return False


def capture(pl, mod, scores2d, lengths, k, direct_kwargs):
    """One eager `_sc_topk_direct`; returns [{kern, name, ops}, ...]."""
    import jax
    with KernelSpy(pl) as spy:
        out = mod._sc_topk_direct(scores2d, k, lengths, **direct_kwargs)
        jax.block_until_ready(out)
    if not spy.calls:
        raise RuntimeError("CAPTURE_FAIL: pl.kernel was never called -- the "
                           "spy did not work, so an empty inside-kernel "
                           "bracket would be a silent lie")
    return spy.calls


# ==========================================================================
# Smoke check only. The AUTHORITATIVE correctness gate is the frozen 68-id
# suite, already green for every arm here (j-188b0f9e, j-24463f36,
# j-f114bd8a). This exists to catch "the module I timed is not the module
# that passed", nothing more.
# ==========================================================================
def smoke(scores_np, lengths_np, got, k, rows):
    """Per sampled row: is the SELECTED SCORE MULTISET the exact top-k?

    Multiset, not indices: v3-shaped and v10-shaped variants may return a
    different tie-broken subset and still be exact. An index byte-compare
    would fail the incumbent against itself under a different tie order.
    """
    n = scores_np.shape[1]
    bad = oor = nlive = 0
    for r in rows:
        L = int(min(lengths_np[r], n))
        g = got[r]
        live = g[g >= 0]
        want = min(k, max(L, 0))
        if live.size != want:
            nlive += 1
        if live.size and (live.min() < 0 or live.max() >= max(L, 1)):
            oor += 1
            continue
        if L <= 0:
            continue
        row = scores_np[r, :L]
        if L <= k:
            ok = np.array_equal(np.sort(live), np.arange(L))
        else:
            part = np.argpartition(-row, k - 1)[:k]
            ok = np.array_equal(np.sort(row[part]), np.sort(row[live]))
        if not ok:
            bad += 1
    return bad, oor, nlive


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stage", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=9216)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--b-prefill", type=int, default=1024)
    ap.add_argument("--b-decode", type=int, default=64)
    ap.add_argument("--smoke-rows", type=int, default=48)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stage = args.stage
    t_start = time.time()

    def log(msg):
        print(f"[{time.time() - t_start:8.1f}s] {msg}", flush=True)

    # ---------------- STAGE 0: md5 gate, in-job -----------------------------
    # A docker image tag is mutable; the gate is asserted here, in the job,
    # not in the submitter.
    log("=" * 74)
    log("STAGE 0  SUBJECT MD5 GATE (in-job)")
    log("=" * 74)
    gate = {}
    ok = True
    for path, want in (
            (os.path.join(stage, "p18_probe.py"), P18_MD5),
            (os.path.join(stage, "glm_dsa_block_bench.py"), HARNESS_MD5)):
        got = _md5(open(path, "rb").read())
        gate[os.path.basename(path)] = got
        good = got == want
        ok &= good
        log(f"  {os.path.basename(path):26s} {got}  "
            f"{'OK' if good else 'FAIL want ' + want}")
    for tag, sub, md5, suf, extra, env, dkw in ARMS:
        p = os.path.join(stage, "subjects", sub, "sparsecore_topk.py")
        got = _md5(open(p, "rb").read())
        gate[f"{tag}/sparsecore_topk.py"] = got
        good = got == md5
        ok &= good
        log(f"  {tag + '/sparsecore_topk.py':26s} {got}  "
            f"{'OK' if good else 'FAIL want ' + md5}")
        for fn, m in extra.items():
            pe = os.path.join(stage, "subjects", sub, fn)
            g2 = _md5(open(pe, "rb").read())
            gate[f"{tag}/{fn}"] = g2
            good2 = g2 == m
            ok &= good2
            log(f"  {tag + '/' + fn:26s} {g2}  "
                f"{'OK' if good2 else 'FAIL want ' + m}")
    if not ok:
        log("MD5_GATE_FAIL")
        return 2
    log("MD5_GATE_OK")

    p18 = _load_p18(os.path.join(stage, "p18_probe.py"))
    n = ((args.ctx + 1024) + 15) // 16 * 16
    assert n == 10240, (
        f"n={n}: --ctx 9216 is the shipping shape. The bench default "
        f"--ctx 132096 gives n=133120, a shape no checked-in config produces.")
    k = args.k
    bp, bd = args.b_prefill, args.b_decode

    if args.dry_run:
        lp = p18.build_lengths(bp, n, k)
        ld = p18.build_lengths(bd, n, k)
        for nm, tbl in (("prefillX", lp), ("decodeX", ld)):
            a = tbl[nm][0]
            print(f"  {nm}: rho={np.minimum(a, n).mean() / n:.6f} "
                  f"support [{a.min()},{a.max()}] b={a.shape[0]}")
        print(f"  arms={len(ARMS)} suffixes={ALL_SUFFIXES}")
        print("DRY_RUN_OK")
        return 0

    os.makedirs(args.outdir, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import vllm_torchtpu  # noqa: F401  (registers the TPU platform)
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    info = pltpu.get_tpu_info()
    log(f"host={os.uname().nodename} devices={jax.device_count()} "
        f"kind={jax.devices()[0].device_kind} sc={info.sparse_core} "
        f"jax={jax.__version__}")

    results = {
        "probe": "23-ranking", "n": n, "k": k, "ctx": args.ctx,
        "b_prefill": bp, "b_decode": bd, "rounds": args.rounds,
        "md5": gate,
        "denominators": {"prefill_us": DENOM_PREFILL,
                         "prefill_gated_us": DENOM_PREFILL_GATED,
                         "decode_us": DENOM_DECODE,
                         "source": "probes/18 MEASURED; 155.82 is MODELLED "
                                   "and forbidden"},
    }

    # ---------------- STAGE 1: build the modules ----------------------------
    log("")
    log("=" * 74)
    log("STAGE 1  MODULES -- one process, separately staged, env set per arm")
    log("=" * 74)
    mods = {}
    meta = {}
    base_src = open(os.path.join(stage, "subjects", "base",
                                 "sparsecore_topk.py"), "rb").read().decode()
    for tag, sub, md5, suf, extra, env, dkw in ARMS:
        sd = os.path.join(stage, "subjects", sub)
        src = open(os.path.join(sd, "sparsecore_topk.py"), "rb").read().decode()
        saved = {kk: os.environ.get(kk) for kk in env}
        if sd not in sys.path:
            sys.path.insert(0, sd)
        try:
            os.environ.update(env)
            m = p18.build_mod(src, tag)
        finally:
            for kk, v in saved.items():
                if v is None:
                    os.environ.pop(kk, None)
                else:
                    os.environ[kk] = v
        mods[tag] = m
        meta[tag] = {"md5": md5, "suffix": suf, "env": dict(env),
                     "direct_kwargs": dkw}
        note = ""
        if extra:
            scs = getattr(m, "_scs", None)
            f = getattr(scs, "__file__", "?") if scs else "MISSING"
            note = f"  _scs={f}"
            if scs is None or not str(f).startswith(sd):
                log(f"  !! {tag}: sparsecore_topk_sample did NOT resolve to "
                    f"the mapped copy ({f}) -- the import fell back and the "
                    f"arm would silently be something else")
                return 2
        log(f"  {tag:10s} md5={md5[:12]} suffix={str(suf):14s} env={env}{note}")
    # three nulls: IDENTICAL incumbent source, separately staged modules
    for t in "abc":
        mods[f"#null_{t}"] = p18.build_mod(base_src, f"null_{t}")
        meta[f"#null_{t}"] = {"md5": ARMS[0][2], "suffix": None, "env": {},
                              "direct_kwargs": {}}
    log(f"  + 3 nulls (identical incumbent source, separate module objects) "
        f"-> the incumbent holds 4 slots in every cell")
    TAGS = [a[0] for a in ARMS] + [f"#null_{t}" for t in "abc"]

    for t in TAGS:
        assert mods[t]._pick_partition(bp, n, k) == 1, t
        assert mods[t]._pick_partition(bd, n, k) == 1, t
    log(f"  P=1 confirmed at b={bp} and b={bd} for all {len(TAGS)} modules")

    # ---------------- operands ---------------------------------------------
    log("")
    log("STAGE 1b  OPERANDS (production scores, production length draws)")
    t0 = time.time()
    sc_pre = p18.production_scores(bp, n, np.random.default_rng(p18.SEED13))
    sc_dec = p18.production_scores(bd, n, np.random.default_rng(p18.SEED13 + 1))
    log(f"  scores built in {time.time() - t0:.1f}s  "
        f"pre.md5={_md5(sc_pre.tobytes())[:16]} "
        f"dec.md5={_md5(sc_dec.tobytes())[:16]}")

    lp = p18.build_lengths(bp, n, k)
    ld = p18.build_lengths(bd, n, k)
    # CELL = (b, rho-name). Only the production distributions, plus the named
    # rho=1 exception at b=1024.
    CELLS = {
        "P.prefillX": (bp, sc_pre, lp["prefillX"][0], lp["prefillX"][1]),
        "D.decodeX": (bd, sc_dec, ld["decodeX"][0], ld["decodeX"][1]),
        "P.rho1": (bp, sc_pre, lp["rho1"][0], lp["rho1"][1]),
    }
    cell_meta = {}
    for cn, (b, sc, la, prov) in CELLS.items():
        eff = np.minimum(la.astype(np.int64), n)
        cell_meta[cn] = {"b": b, "rho": float(eff.mean() / n),
                         "min": int(eff.min()), "max": int(eff.max()),
                         "frac_le_k": float((eff <= k).mean()),
                         "provenance": prov,
                         "rho_source": "p18_probe.build_lengths (probes/13 "
                                       "level_sim verbatim) -- NOT the frozen "
                                       "bench _row_lengths",
                         "lengths_md5": _md5(la.tobytes())[:16]}
        log(f"  cell {cn:12s} b={b:5d} rho={cell_meta[cn]['rho']:.5f} "
            f"eff in [{eff.min()},{eff.max()}] P(eff<=k)="
            f"{cell_meta[cn]['frac_le_k'] * 100:5.1f}%  {prov}")
    results["cells"] = cell_meta

    dev = {}
    for cn, (b, sc, la, prov) in CELLS.items():
        dev[cn] = {
            "s3d": jnp.asarray(sc.reshape(b, n // 128, 128)),
            "s2d": jnp.asarray(sc),
            "len": jnp.asarray(la),
            "np_s": sc, "np_l": la, "b": b,
        }

    # ---------------- STAGE 2: arm activation + smoke -----------------------
    log("")
    log("=" * 74)
    log("STAGE 2  ARM ACTIVATION (scope name) + CORRECTNESS SMOKE")
    log("=" * 74)
    whole = {}   # (tag, b) -> jitted f(s3d, len)
    act = {}
    act_ok = True
    for tag in TAGS:
        m = mods[tag]
        for b in (bp, bd):
            f = jax.jit(lambda s, l, _m=m, _b=b: _m.sparsecore_topk(
                s.reshape(_b, n), k=k, row_lengths=l))
            whole[(tag, b)] = f
    for tag in TAGS:
        suf = meta[tag]["suffix"]
        seen = set()
        for b, cn in ((bp, "P.prefillX"), (bd, "D.decodeX")):
            lowered = whole[(tag, b)].lower(dev[cn]["s3d"], dev[cn]["len"])
            for text in (lowered.as_text(), lowered.compile().as_text()):
                seen |= set(NAME_RE.findall(text))
        bare = any(BARE_RE.fullmatch(s) for s in seen)
        others = sorted({o for o in ALL_SUFFIXES
                         if o != suf and any(o in s for s in seen)})
        if suf is None:
            good = bool(seen) and bare and not others
            why = ("bare name present, no variant suffix" if good
                   else f"bare={bare} foreign={others} seen={sorted(seen)}")
        else:
            # Substring, not endswith: v2/v2big PREPEND their tag while
            # v11/v12/v4* append it, so under emit_keys v2's name would end
            # `_keys` and an endswith test would spuriously fail it.
            # `bare` is only WARNED on here -- HLO text can carry a truncated
            # copy of a name. The exact, unambiguous bare-name test is the
            # pl.kernel `name=` kwarg read in STAGE 3, and that one is fatal.
            hit = bool(seen) and any(suf in s for s in seen)
            good = hit and not others
            why = ("suffix present, no foreign suffix"
                   + ("  [WARN: a bare incumbent-shaped name also appears in "
                      "the HLO text; STAGE 3 adjudicates]" if bare else "")
                   if good else
                   f"hit={hit} bare={bare} foreign={others} "
                   f"seen={sorted(seen)}")
        act_ok &= good
        act[tag] = {"names": sorted(seen), "expect": suf, "ok": bool(good),
                    "why": why, "bare_present": bare, "foreign": others}
        log(f"  {tag:10s} expect={str(suf):14s} "
            f"{'ACTIVE' if good else '!! ARM_CHECK_FAIL'}  {why}")
        if not seen:
            log(f"  !! {tag}: no sc_topk_direct scope name found AT ALL -- "
                f"the check itself did not work")
    results["arm_check_hlo"] = act
    if not act_ok:
        log("ARM_CHECK_FAIL -- refusing to time an arm whose scope name does "
            "not prove which body lowered")
        with open(os.path.join(args.outdir, "ranking.json"), "w") as fh:
            json.dump(results, fh, indent=2, sort_keys=True, default=str)
        return 2
    log("ARM_CHECK_OK (HLO scope names)")

    rows_p = np.random.default_rng(4242).choice(bp, size=min(args.smoke_rows,
                                                             bp), replace=False)
    rows_d = np.arange(bd)
    smk = {}
    smk_ok = True
    base_out = {}
    for cn in CELLS:
        b = dev[cn]["b"]
        rows = rows_p if b == bp else rows_d
        for tag in TAGS:
            if cn == "P.rho1" and tag not in RHO1_ARMS:
                continue
            got = np.asarray(jax.block_until_ready(
                whole[(tag, b)](dev[cn]["s3d"], dev[cn]["len"])))
            if tag == "base":
                base_out[cn] = got
            bad, oor, nl = smoke(dev[cn]["np_s"], dev[cn]["np_l"], got, k, rows)
            bit = bool(np.array_equal(got, base_out[cn]))
            smk[f"{cn}/{tag}"] = {"rows_wrong": int(bad),
                                  "out_of_range": int(oor),
                                  "live_count_wrong": int(nl),
                                  "bit_equal_to_base": bit,
                                  "rows_sampled": int(len(rows))}
            fail = bool(bad or oor or nl)
            smk_ok &= not fail
            log(f"  {cn:12s} {tag:10s} rows_wrong={bad:3d} oor={oor:3d} "
                f"live_bad={nl:3d} bit==base={bit}  "
                f"{'EXACT' if not fail else '!! SMOKE_FAIL'}")
    results["smoke"] = smk
    if not smk_ok:
        log("SMOKE_FAIL -- a timed module does not agree with the numpy "
            "oracle; the frozen-68 pass does not transfer to this build")
        with open(os.path.join(args.outdir, "ranking.json"), "w") as fh:
            json.dump(results, fh, indent=2, sort_keys=True, default=str)
        return 2
    log("SMOKE_OK")

    # ---------------- STAGE 3: capture the inside-kernel bracket ------------
    log("")
    log("=" * 74)
    log("STAGE 3  INSIDE-KERNEL CAPTURE (eager _sc_topk_direct + pl.kernel spy)")
    log("=" * 74)
    # A capture failure is NOT fatal. The whole-call bracket is already
    # compiled, is the designated bracket for v11 and v12, and ranks every
    # arm; throwing it away because the second bracket could not be built
    # would spend the cluster time and report nothing. Any arm whose
    # inside-kernel bracket is missing or cannot prove its scope name is
    # DROPPED from the inside-kernel pool and listed by name below.
    inside = {}     # (tag, b) -> {"f": jitted, "ops": {cell: [ops...]}}
    cap_meta = {}
    cap_ok = True
    cap_dropped = []
    for tag in TAGS:
        m = mods[tag]
        dkw = meta[tag]["direct_kwargs"]
        suf = meta[tag]["suffix"]
        for b in (bp, bd):
            cells = [cn for cn in CELLS if dev[cn]["b"] == b
                     and (cn != "P.rho1" or tag in RHO1_ARMS)]
            per_cell = {}
            kerns = names = None
            for cn in cells:
                try:
                    calls = capture(pl, m, dev[cn]["s2d"], dev[cn]["len"], k,
                                    dkw)
                except Exception as exc:
                    log(f"  !! {tag} b={b} {cn}: CAPTURE_FAIL {exc!r}")
                    cap_ok = False
                    per_cell = None
                    break
                if kerns is None:
                    kerns = [c["kern"] for c in calls]
                    names = [c["name"] for c in calls]
                elif len(calls) != len(kerns):
                    log(f"  !! {tag} b={b} {cn}: kernel count changed with "
                        f"rho ({len(calls)} vs {len(kerns)})")
                    cap_ok = False
                per_cell[cn] = [c["ops"] for c in calls]
            if per_cell is None or kerns is None:
                cap_ok = False
                cap_dropped.append(f"{tag}/b{b}")
                continue
            # EXACT names, read from the `name=` kwarg at pl.kernel
            # construction. No HLO text, no truncation, no ambiguity: a bare
            # name here means the incumbent body ran. Fatal.
            nm_ok = (all(suf in nn for nn in names) if suf
                     else all(BARE_RE.fullmatch(nn) for nn in names))
            bare_here = any(BARE_RE.fullmatch(nn) for nn in names)
            foreign = sorted({o for o in ALL_SUFFIXES if o != suf
                              and any(o in nn for nn in names)})
            good = bool(names) and nm_ok and not foreign and (
                bare_here if suf is None else not bare_here)
            cap_ok &= good
            cap_meta[f"{tag}/b{b}"] = {
                "kernel_names": names, "n_kernels": len(kerns),
                "name_ok": bool(good), "foreign": foreign,
                "bare_name_present": bool(bare_here),
                "operand_shapes": [[str(o.shape) + str(o.dtype)
                                    for o in ops] for ops in
                                   per_cell[cells[0]]]}
            # The two rho operand sets must differ ONLY in the lengths
            # operand: same scores in, same kernel, only row_lengths moves.
            # Where they are equal, REUSE the first cell's buffer so the
            # 42 MB score operand is not resident twice per arm.
            if len(cells) > 1:
                samelog = []
                for ci in cells[1:]:
                    for ki in range(len(kerns)):
                        for oi in range(len(per_cell[cells[0]][ki])):
                            x = per_cell[cells[0]][ki][oi]
                            y = per_cell[ci][ki][oi]
                            eq = (x.shape == y.shape and x.dtype == y.dtype
                                  and bool(np.array_equal(np.asarray(x),
                                                          np.asarray(y))))
                            samelog.append([ci, ki, oi, eq])
                            if eq:
                                per_cell[ci][ki][oi] = x
                cap_meta[f"{tag}/b{b}"]["rho_operand_equality"] = samelog

            def prog(*flat, _ks=kerns, _shape=[len(o) for o in
                                               per_cell[cells[0]]]):
                outs, i = [], 0
                for kk, cnt in zip(_ks, _shape):
                    outs.append(kk(*flat[i:i + cnt]))
                    i += cnt
                return outs

            if good:
                inside[(tag, b)] = {"f": jax.jit(prog), "ops": per_cell}
            else:
                cap_dropped.append(f"{tag}/b{b}")
            log(f"  {tag:10s} b={b:5d} kernels={len(kerns)} names={names} "
                f"{'OK' if good else '!! CAPTURE_ARM_FAIL bare=' + str(bare_here) + ' foreign=' + str(foreign) + ' -- DROPPED from the inside-kernel pool'}")
    results["inside_kernel_capture"] = cap_meta
    results["inside_kernel_dropped"] = cap_dropped
    if not cap_ok:
        log("")
        log(f"!! CAPTURE_PARTIAL -- {len(cap_dropped)} inside-kernel brackets "
            f"DROPPED: {cap_dropped}")
        log("   Their whole-call rows below are unaffected and are the "
            "primary bracket for v11 and v12 in any case. Any inside-kernel "
            "row absent from a cell is absent because of this, not because "
            "it was slow.")
    else:
        log("CAPTURE_OK -- every inside-kernel bracket carries its arm's "
            "scope name, read from the pl.kernel name= kwarg directly")

    # ---------------- STAGE 4: the pool -------------------------------------
    log("")
    log("=" * 74)
    log("STAGE 4  POOL")
    log("=" * 74)
    measure, last_order, hkind, dfloor = p18._load_harness([stage])
    results["harness"] = hkind

    thunks, arm_meta = {}, {}
    for cn in CELLS:
        b = dev[cn]["b"]
        for tag in TAGS:
            if cn == "P.rho1" and tag not in RHO1_ARMS:
                continue
            nm = f"{cn}|whole|{tag}"
            thunks[nm] = (lambda _f=whole[(tag, b)], _s=dev[cn]["s3d"],
                          _l=dev[cn]["len"]: _f(_s, _l))
            arm_meta[nm] = {"cell": cn, "bracket": "whole", "tag": tag,
                            "b": b, "rho": cell_meta[cn]["rho"],
                            "suffix": meta[tag]["suffix"]}
            ent = inside.get((tag, b))
            if ent is None or cn not in ent["ops"]:
                continue
            flat = [o for ops in ent["ops"][cn] for o in ops]
            nm2 = f"{cn}|inside|{tag}"
            thunks[nm2] = (lambda _f=ent["f"], _o=tuple(flat): _f(*_o))
            arm_meta[nm2] = {"cell": cn, "bracket": "inside", "tag": tag,
                             "b": b, "rho": cell_meta[cn]["rho"],
                             "suffix": meta[tag]["suffix"]}
    log(f"  {len(thunks)} arms in ONE pool, {args.rounds} interleaved rounds, "
        f"reshuffled per round, paired REPS "
        f"{p18.REPS_LO}/{p18.REPS_HI}, read us_median")
    results["arms"] = arm_meta
    fl = dfloor()
    log(f"  dispatch floor: {fl:.2f} us/call  (it is IN every absolute below; "
        f"it cancels in an in-cell delta)")
    results["dispatch_floor_us"] = fl

    out = measure(thunks, rounds=args.rounds)
    results["raw"] = out
    results["order"] = last_order()
    m = {kk: out[kk]["us_median"] for kk in out}

    # ---------------- STAGE 5: floors ---------------------------------------
    log("")
    log("=" * 74)
    log("STAGE 5  RESOLUTION FLOOR (four identical incumbent modules per cell)")
    log("=" * 74)
    floors = {}
    for cn in CELLS:
        for br in ("whole", "inside"):
            keys = [f"{cn}|{br}|{t}" for t in
                    ("base", "#null_a", "#null_b", "#null_c")]
            keys = [x for x in keys if x in m]
            if f"{cn}|{br}|base" not in m or len(keys) < 3:
                continue
            vals = [m[x] for x in keys]
            f = _spread(vals)
            ref = m[f"{cn}|{br}|base"]
            floors[f"{cn}|{br}"] = {"vals": vals, "spread_us": f,
                                    "spread_pct": 100.0 * f / ref,
                                    "ref_us": ref}
            log(f"  {cn:12s} {br:6s} "
                + " / ".join(f"{v:8.3f}" for v in vals)
                + f"   -> FLOOR {f:7.3f} us = {100.0 * f / ref:.3f}%")
    results["floors"] = floors

    # ---------------- STAGE 6: the ranking ----------------------------------
    log("")
    log("=" * 74)
    log("STAGE 6  RANKING -- absolutes first, deltas against the in-cell base")
    log("=" * 74)
    log("  rho SOURCE for every cell: p18_probe.build_lengths, i.e. probes/13")
    log("  level_sim transcribed verbatim. b=64 is REAL PRODUCTION DECODE at")
    log("  rho=0.850, NOT the frozen bench's rho=0.391 small-batch prefill")
    log("  occupancy. Do not read the b=64 cell as the bench's 0.391 cell.")
    rank = {}
    for cn in CELLS:
        for br in ("whole", "inside"):
            key = f"{cn}|{br}"
            if key not in floors:
                continue
            fspread = floors[key]["spread_us"]
            ref = floors[key]["ref_us"]
            log("")
            log(f"  --- {cn}  bracket={br}  b={cell_meta[cn]['b']}  "
                f"rho={cell_meta[cn]['rho']:.5f}  base={ref:.3f} us  "
                f"floor={fspread:.3f} us ---")
            log(f"  {'arm':12s} {'us_median':>10s} {'delta_us':>10s} "
                f"{'delta%':>9s} {'hostbnd':>8s}  verdict")
            for tag in TAGS:
                nm = f"{cn}|{br}|{tag}"
                if nm not in m:
                    continue
                d = m[nm] - ref
                v = ("not resolvable" if abs(d) <= fspread
                     else ("FASTER" if d < 0 else "SLOWER"))
                if tag == "base":
                    v = "reference"
                elif tag.startswith("#null"):
                    v = "null (must be inside the floor)" if abs(d) <= \
                        fspread else "!! NULL OUTSIDE THE FLOOR"
                rank[nm] = {"us_median": m[nm], "delta_us": d,
                            "delta_pct": 100.0 * d / ref, "verdict": v,
                            "floor_us": fspread,
                            "hostbound_pct": out[nm]["hostbound_pct"],
                            "us": out[nm]["us"], "us_mean": out[nm]["us_mean"],
                            "suspect": out[nm].get("suspect")}
                log(f"  {tag:12s} {m[nm]:10.3f} {d:10.3f} "
                    f"{100.0 * d / ref:+8.2f}% "
                    f"{out[nm]['hostbound_pct']:7.1f}%  {v}")
    results["ranking"] = rank

    # ---------------- STAGE 7: the pre-registered readings ------------------
    log("")
    log("=" * 74)
    log("STAGE 7  PRE-REGISTERED READINGS (set before the numbers existed)")
    log("=" * 74)
    pre = {}

    def get(cn, br, tag):
        return m.get(f"{cn}|{br}|{tag}")

    # (a) tc=2048 vs ragged is NOT adjudicable; a tie ships ragged.
    for cn in ("P.prefillX", "D.decodeX"):
        for br in ("whole", "inside"):
            a, bq = get(cn, br, "v4c2048"), get(cn, br, "v4ragged")
            if a is None or bq is None or f"{cn}|{br}" not in floors:
                continue
            f = floors[f"{cn}|{br}"]["spread_us"]
            d = a - bq
            verdict = ("TIE -> SHIPS RAGGED (pre-registered)"
                       if abs(d) <= f else
                       ("c2048 faster by " if d < 0 else "ragged faster by ")
                       + f"{abs(d):.3f} us")
            pre[f"v4c2048_vs_ragged/{cn}/{br}"] = {
                "c2048_us": a, "ragged_us": bq, "delta_us": d,
                "floor_us": f, "verdict": verdict}
            log(f"  v4c2048 - v4ragged  {cn:12s} {br:6s} "
                f"{d:+8.3f} us (floor {f:.3f})  {verdict}")

    # (b) v2 reports ABSOLUTE us, not a percentage -- its two reweightings are
    #     mutually inconsistent, so a percentage would pick one of them.
    log("")
    log("  v2 -- ABSOLUTE us only (pre-registered: its two reweightings are")
    log("  mutually inconsistent, so no percentage is quoted for it here).")
    for cn in CELLS:
        for br in ("whole", "inside"):
            for tag in ("v2", "v2big"):
                nm = f"{cn}|{br}|{tag}"
                if nm not in m or f"{cn}|{br}" not in floors:
                    continue
                bs = get(cn, br, "base")
                pre[f"{tag}_absolute/{cn}/{br}"] = {
                    "us": m[nm], "base_us": bs, "delta_us": m[nm] - bs,
                    "floor_us": floors[f"{cn}|{br}"]["spread_us"]}
                log(f"    {tag:6s} {cn:12s} {br:6s} {m[nm]:9.3f} us  "
                    f"base {bs:9.3f} us  delta {m[nm] - bs:+8.3f} us")

    # (c) v4ragged at rho=1 is a null BY CONSTRUCTION -- a positive control on
    #     the instrument, not a result.
    for br in ("whole", "inside"):
        nm = f"P.rho1|{br}|v4ragged"
        if nm not in m or f"P.rho1|{br}" not in floors:
            continue
        bs = get("P.rho1", br, "base")
        f = floors[f"P.rho1|{br}"]["spread_us"]
        d = m[nm] - bs
        good = abs(d) <= f
        pre[f"v4ragged_rho1_control/{br}"] = {
            "delta_us": d, "floor_us": f, "inside_floor": bool(good)}
        log("")
        log(f"  CONTROL v4ragged at rho=1, {br}: delta {d:+.3f} us, floor "
            f"{f:.3f} us -> {'INSIDE the floor as predicted' if good else 'OUTSIDE THE FLOOR -- the rho=1 null prediction FAILED'}")

    log("")
    log("  DO NOT ADD UP the adopted candidates' percentages. The arms")
    log("  overlap and composition is step 4's job.")
    results["prereg"] = pre

    with open(os.path.join(args.outdir, "ranking.json"), "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=str)
    log(f"wrote {os.path.join(args.outdir, 'ranking.json')}")
    log("RANKING_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
