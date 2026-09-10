"""Does the pipelined copy land the same bytes? Run BEFORE the timed block.

`instruments/README.md` states it plainly: nothing in the timing harness will
catch a copy that lands the wrong bytes. `--bw` times an unchecked accumulator,
`--decomp`'s `dma_only` writes a sum nobody reads, and `--passes` compares
stubs. The frozen pytest file *does* check correctness -- but it checks
whichever subject file the job happened to map in, and v4 ships TWO rungs that
the timing job ranks against each other and against the incumbent.

So: in the same job, on the same pod, immediately before the timed rounds, run
every arm the timing block will time and compare it to the incumbent kernel
element for element. Each arm is a separate file with its own md5, printed
here, so "which arm was checked" is a fact and not an inference about the
environment. Two failure modes this catches that a pass/fail on one arm does
not:

  * a short final chunk that drops or duplicates a lane group
    (`live = clip(i_hi - c*tc, 0, tc)` is the only thing bounding it);
  * a buffer-reuse bug that is correct at one grid step and wrong at many --
    at rho = 1 and tc = 2048 there are five steps per row.

Both raggedness regimes are exercised, because they are different code paths
through the same kernel: the ragged one varies the grid length per row and
lands a short final chunk, and rho = 1 makes every grid full-length and every
buffer reused the maximum number of times.

  uv_venv/bin/python checksum.py [--ctx 9216]

Exit status is 0 only if every arm matched. `CHECKSUM: PASS` / `CHECKSUM: FAIL`
is printed on its own line -- a `#` comment cannot carry an `exit 1` through
cdk's YAML fold, so the caller greps for that string and not for the exit code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import pathlib
import sys

import vllm_torchtpu  # noqa: F401  (env_override must precede jax)
import jax
import jax.numpy as jnp
import numpy as np

HERE = pathlib.Path(__file__).parent
INSTRUMENTS = HERE.parents[2] / "instruments"
sys.path.insert(0, str(INSTRUMENTS))
from sctopk_2stage_bench import _row_lengths, describe, topk_n  # noqa: E402

INCUMBENT = (HERE.parents[3] / "src" / "vllm_torchtpu" / "kernels" /
             "deepseek_v4" / "sparsecore_topk.py")
BASE = HERE / "sparsecore_topk_base.py"
RAGGED = HERE / "sparsecore_topk_ragged.py"
C2048 = HERE / "sparsecore_topk_c2048.py"
SEED = 20260910
LANES = 16
K = 2048

# Every arm the timing block will time. Keep this list and the timing block's
# arm list identical -- an arm that is timed but not checked is the whole
# reason this file exists.
ARMS = [
    # The ragged rung goes FIRST and matters MOST. It is a null at rho = 1 by
    # construction, so the timing block cannot tell a correct ragged copy from
    # a broken one there -- and a wrong `align(i_hi, LANES)` lands SHORT reads,
    # which drop real scores silently rather than crashing. This checksum, on
    # the ragged distribution, is the only thing standing in front of that.
    ("v4-ragged", RAGGED),
    ("v4-c2048", C2048),
]


def load(tag, path):
    spec = importlib.util.spec_from_file_location(f"sctopk_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def selected_scores(idx, scores, rl):
    """The multiset of selected scores, sorted. Index order is unspecified.

    The kernel's contract is a set, not a sequence, so comparing raw index
    arrays would fail on a reordering that is not a bug. Comparing the sorted
    selected *scores* is the same bar the frozen test uses, and it is exact:
    the copy either landed the row or it did not.
    """
    out = np.full(idx.shape, -np.inf, np.float32)
    for i in range(idx.shape[0]):
        live = idx[i] >= 0
        out[i, live] = scores[i, idx[i][live]]
        out[i] = np.sort(out[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=9216,
                    help="9216 is the production P=1 shape and is NOT the "
                         "bench default")
    args = ap.parse_args()
    n = topk_n(args.ctx)
    n_p = describe(1024, n, K)["n_p"]
    rng = np.random.default_rng(SEED)

    # The subject files are the evidence. Print their md5s so a timing job can
    # gate on the same strings, and check the mapped-in base against the tree's
    # incumbent -- a stale copy would make every arm pass against the wrong
    # reference. A docker image tag is mutable; an md5 is not.
    digests = {}
    for tag, path in [("base", BASE)] + ARMS:
        digests[tag] = hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()
        print(f"  subject md5  {tag:<12} {digests[tag]}  {path.name}")
    failures = []
    if INCUMBENT.exists():
        live = hashlib.md5(INCUMBENT.read_bytes()).hexdigest()
        print(f"  tree incumbent            {live}  (src/.../sparsecore_topk.py)")
        if live != digests["base"]:
            failures.append(
                f"mapped base {digests['base']} != tree incumbent {live}; the "
                "control is not the incumbent")
    if len(set(digests.values())) != len(digests):
        failures.append("two arms are the same file -- the A/B is vacuous")

    base = load("base", BASE)

    # A test that cannot fail is not a test. The ragged arm's only functional
    # change is `align(i_hi, LANES)`, so if every row length in the ragged
    # distribution happened to be a multiple of LANES the rounding would never
    # be exercised and every arm would pass vacuously. Assert the distribution
    # actually straddles a lane group, and that it is not saturated either.
    probe = _row_lengths("random", 1024, n, describe(1024, n, K)["n_p"])
    unaligned = int((probe % LANES != 0).sum())
    partial = int(((probe > 0) & (probe < n)).sum())
    print(f"  distribution check: {unaligned}/1024 rows are not a multiple of "
          f"{LANES} words, {partial}/1024 are a partial row")
    if unaligned == 0:
        failures.append("ragged distribution never exercises the LANES "
                        "rounding -- this run proves nothing about v4-ragged")
    if partial == 0:
        failures.append("ragged distribution has no partial rows -- the "
                        "ragged arm is a no-op here")

    for b in (1024, 64):
        scores_np = rng.standard_normal((b, n), dtype=np.float32)
        scores = jnp.asarray(scores_np)
        for rho_tag, rl_np in (("ragged", _row_lengths("random", b, n, n_p)),
                               ("rho1", np.full(b, n, np.int32))):
            rl = jnp.asarray(rl_np)
            ref_idx = np.asarray(jax.block_until_ready(
                base._sc_topk_direct(scores, K, rl)))
            ref = selected_scores(ref_idx, scores_np, rl_np)
            for tag, path in ARMS:
                mod = load(f"{tag}-{b}-{rho_tag}", path)
                got_idx = np.asarray(jax.block_until_ready(
                    mod._sc_topk_direct(scores, K, rl)))
                got = selected_scores(got_idx, scores_np, rl_np)
                n_live_ref = int((ref_idx >= 0).sum())
                n_live_got = int((got_idx >= 0).sum())
                bad = float(np.abs(got - ref).max())
                ok = (bad == 0.0) and (n_live_ref == n_live_got)
                print(f"  b={b:<5} {rho_tag:<7} {tag:<18} "
                      f"live={n_live_got}/{n_live_ref}  "
                      f"max|score diff|={bad!r}  "
                      f"{'ok' if ok else 'MISMATCH'}", flush=True)
                if not ok:
                    failures.append(f"b={b} {rho_tag} {tag}")

    # Negative control: the comparator must be able to fail. Corrupt one index
    # in one row of the reference and re-run the same comparison. "N arms
    # matched" is not evidence the comparison ran.
    scores_np = rng.standard_normal((64, n), dtype=np.float32)
    idx = np.asarray(jax.block_until_ready(
        base._sc_topk_direct(jnp.asarray(scores_np), K,
                             jnp.full(64, n, jnp.int32))))
    rl_np = np.full(64, n, np.int32)
    ref = selected_scores(idx, scores_np, rl_np)
    bad_idx = idx.copy()
    row = int(np.flatnonzero((bad_idx >= 0).sum(1) > 1)[0])
    worst = int(np.argmin(scores_np[row]))
    bad_idx[row, 0] = worst
    ctl = float(np.abs(selected_scores(bad_idx, scores_np, rl_np) - ref).max())
    print(f"  negative control: one index swapped for the row minimum -> "
          f"max|score diff|={ctl!r} "
          f"({'comparator can fail' if ctl > 0 else 'COMPARATOR IS BROKEN'})")
    if ctl <= 0:
        failures.append("negative control did not fire")

    print(f"CHECKSUM: {'FAIL ' + '; '.join(failures) if failures else 'PASS'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
