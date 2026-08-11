"""Full-stack ours-vs-reference activation diff, with loud failures.

Replaces diff_all.py, which had two silent-failure modes that made a broken
comparison look like a clean one:

  1. `find_prefill()` returns None when no step matches (our tree's step000 is
     a warmup pass with constant dummy rows, so it never matched). diff_all.py
     then built paths literally containing "stepNone", found no files, printed
     "-" for every cell, and reported "FIRST DIVERGENCE: None" -- which reads
     as "everything matches". Here, a missing prefill step is a hard exit.

  2. It compared cosine only. The open DSv4 residual is a MAGNITUDE error
     (cos ~0.986 but norm ratio ~1.32), which cosine barely registers. Every
     cell here carries cos, relL2, and the ours/ref norm ratio.

Prefill detection: the step whose row count is largest AND whose row norms
actually vary. Our tree emits a warmup pass (constant rows) that the reference
does not, so step indices are NOT comparable across trees and must be detected
per-tree rather than hardcoded.
"""
import glob
import os
import re
import sys

import numpy as np

_SCRATCH = os.environ.get(
    "DSV4_SCRATCH", os.path.expanduser("~/workspace/scratch"))

O = os.path.join(_SCRATCH, "hidden_states", "vllm_torchtpu")
R = os.path.join(_SCRATCH, "hidden_states", "tpu_inference")
N = int(os.getenv("DSV4_CMP_ROWS", "256"))
ORDER = ["01_hc_pre_attn", "02_post_attn", "03_hc_post_attn",
         "04_hc_pre_ffn", "05_post_ffn", "06_hc_post_ffn"]


def describe_steps(d):
    """Every step of the layer-0 probe tensor, with the warmup tell."""
    out = []
    for f in sorted(glob.glob(f"{d}/sub_01_hc_pre_attn_step*.npy")):
        a = np.load(f)
        step = re.search(r"step(\d+)", f).group(1)
        rows = a.shape[0]
        k = min(rows, N)
        uniq = len(np.unique(np.round(np.linalg.norm(a[:k], axis=1), 3)))
        out.append((step, rows, uniq))
    return out


def find_prefill(d):
    """Largest step with genuinely varying row norms; None if none qualify."""
    best, best_rows = None, -1
    for step, rows, uniq in describe_steps(d):
        # A warmup pass is constant across rows, so uniq collapses to ~1.
        # Require several distinct norms AND a real multi-row batch.
        if rows >= 16 and uniq > 2 and rows > best_rows:
            best, best_rows = step, rows
    return best


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else float("nan")


def rl2(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def ratio(a, b):
    na = np.linalg.norm(a.ravel().astype(np.float64))
    nb = np.linalg.norm(b.ravel().astype(np.float64))
    return float(na / nb) if nb else float("nan")


for tag, d in (("ours", O), ("ref", R)):
    if not os.path.isdir(d):
        sys.exit(f"FATAL: dump dir missing: {d}")
    steps = describe_steps(d)
    if not steps:
        sys.exit(f"FATAL: no sub_01_hc_pre_attn_step*.npy in {d} -- "
                 f"the {tag} run did not dump. Check its log.")
    print(f"{tag} steps (step, rows, distinct_row_norms):")
    for s, rows, uniq in steps:
        note = "  <- constant rows (warmup?)" if uniq <= 2 else ""
        print(f"    step{s}  rows={rows:5d}  uniq={uniq:5d}{note}")

so, sr = find_prefill(O), find_prefill(R)
print(f"\ndetected prefill step: ours={so} ref={sr}")
if so is None or sr is None:
    sys.exit("FATAL: could not detect a prefill step on at least one side. "
             "Refusing to emit a comparison -- this is the silent-failure "
             "mode that made diff_all.py report a false clean result.")

# Shape agreement at the probe point is a precondition for everything below.
pa = np.load(f"{O}/sub_01_hc_pre_attn_step{so}.npy")
pb = np.load(f"{R}/sub_01_hc_pre_attn_step{sr}.npy")
if pa.shape != pb.shape:
    print(f"\nWARNING: probe shapes differ ours={pa.shape} ref={pb.shape}. "
          f"Comparing the first {N} rows only; if the two runs used different "
          f"prompts these numbers are meaningless.")

print(f"\ncomparing first {N} rows.  each cell: cos / relL2 / norm-ratio\n")
hdr = f"{'layer':>5} " + " ".join(f"{n[3:][:16]:>22}" for n in ORDER)
print(hdr)
print("-" * len(hdr))

first = None
worst_ratio = (0.0, None)
for L in range(64):
    suf = "" if L == 0 else f"_L{L}"
    row, any_found = [], False
    for n in ORDER:
        po = f"{O}/sub_{n}{suf}_step{so}.npy"
        pr = f"{R}/sub_{n}{suf}_step{sr}.npy"
        if not (os.path.exists(po) and os.path.exists(pr)):
            row.append(f"{'-':>22}")
            continue
        any_found = True
        a, b = np.load(po)[:N], np.load(pr)[:N]
        if a.shape != b.shape:
            row.append(f"{'SHAPE':>22}")
            continue
        c, l, r = cos(a, b), rl2(a, b), ratio(a, b)
        row.append(f"{c:>7.4f}/{l:>6.3f}/{r:>6.3f}")
        if (c <= 0.99 or abs(r - 1.0) > 0.05) and first is None:
            first = f"L{L}/{n}  cos={c:.5f} relL2={l:.4f} ratio={r:.4f}"
        if abs(r - 1.0) > abs(worst_ratio[0] - 1.0):
            worst_ratio = (r, f"L{L}/{n}")
    if any_found:
        print(f"{L:>5} " + " ".join(row))

print(f"\nFIRST DIVERGENCE (cos<=0.99 or |ratio-1|>0.05): {first or 'NONE'}")
print(f"WORST MAGNITUDE RATIO: {worst_ratio[0]:.4f} at {worst_ratio[1]}")
