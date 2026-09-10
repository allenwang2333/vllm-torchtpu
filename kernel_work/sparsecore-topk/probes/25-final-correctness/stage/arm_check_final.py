"""Prove the COMPOSED arm is the body that lowered -- by scope name, paired.

    python arm_check_final.py --variant <path> --variant-md5 <md5> \
                              --incumbent <path> --incumbent-md5 <md5>

Why this is not `arm_check.py` verbatim (which is reused unchanged alongside
it, so this file is an addition and not a replacement):

  * `arm_check.py` matches with `nm.endswith(suffix)` over the set of names and
    accepts if ANY matches. Two of the shapes in the frozen suite lower TWO
    kernels (stage 1 and the stage-2 `_keys` pass); an `any` there is green if
    only one of them is the new body. This requires EVERY name to carry the
    tag, and matches by SUBSTRING, because arms in this exercise both prepend
    (`v2`) and append (`v4`, `v12`) their tag and an `endswith` has already
    spuriously failed once.
  * It checks all 7 shapes the frozen suite runs, not one.
  * It runs the INCUMBENT through the identical path as a negative control.
    "the variant's names all say `_final`" is only evidence if the incumbent's
    names, read the same way, do not. Without it a bug that stamps every name
    would read as a pass.

Lowering only -- no TPU execution, no timing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"sc_topk_direct_b\d+_n\d+_k\d+(?:_[A-Za-z0-9]+)*")
TAG = "_final"
CHUNK_RE = re.compile(r"_c\d+_final$")

N_SHORT, N_128K = 10240, 133120
SHAPES = [
    (64, N_SHORT, 2048),
    (16, N_SHORT, 2048),
    (16, N_128K, 2048),
    (64, N_128K, 2048),
    (16, N_128K, 512),
    (1024, N_128K, 2048),
    (1024, N_SHORT, 2048),
]


def load(path: Path, tag: str):
    spec = importlib.util.spec_from_file_location(f"armchk_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def names_for(mod, jax, jnp, np, b, n, k):
    scores = jnp.asarray(np.zeros((b, n), np.float32))
    rl = jnp.asarray(np.full(b, n, np.int32))
    lowered = jax.jit(lambda s, l: mod.sparsecore_topk(s, k, l)).lower(
        scores, rl)
    out = set()
    for text in (lowered.as_text(), lowered.compile().as_text()):
        out |= set(NAME_RE.findall(text))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--variant-md5", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--incumbent-md5", required=True)
    a = ap.parse_args()

    for path, want, who in ((a.variant, a.variant_md5, "variant"),
                            (a.incumbent, a.incumbent_md5, "incumbent")):
        got = hashlib.md5(Path(path).read_bytes()).hexdigest()
        print(f"ARM_CHECK {who:10s} {path} md5={got}")
        if got != want:
            print(f"ARM_CHECK_MD5_FAIL {who} want={want}")
            return 2

    import jax
    import jax.numpy as jnp
    import numpy as np

    var = load(Path(a.variant), "var")
    inc = load(Path(a.incumbent), "inc")
    print(f"  variant   module __file__ = {var.__file__}")
    print(f"  incumbent module __file__ = {inc.__file__}")

    bad = []
    for b, n, k in SHAPES:
        vn = names_for(var, jax, jnp, np, b, n, k)
        inn = names_for(inc, jax, jnp, np, b, n, k)
        p = var._pick_partition(b, n, k)
        print(f"  [b{b}-n{n}-k{k}] P={p}")
        print(f"      variant  : {vn}")
        print(f"      incumbent: {inn}")
        if not vn:
            bad.append(f"b{b}-n{n}: variant lowered NO sc_topk_direct name -- "
                       "the check did not work; do not read this as a null")
            continue
        miss = [x for x in vn if TAG not in x]
        if miss:
            bad.append(f"b{b}-n{n}: variant names without {TAG!r}: {miss}")
        nochunk = [x for x in vn if not CHUNK_RE.search(x)]
        if nochunk:
            bad.append(f"b{b}-n{n}: variant names without a _c<chunk>_final "
                       f"tail: {nochunk}")
        stamped = [x for x in inn if TAG in x]
        if stamped:
            bad.append(f"b{b}-n{n}: NEGATIVE CONTROL BROKEN -- incumbent "
                       f"names carry {TAG!r}: {stamped}")
        if not inn:
            bad.append(f"b{b}-n{n}: incumbent lowered no name at all")

    if bad:
        print("ARM_CHECK_FINAL_FAIL")
        for x in bad:
            print(f"    {x}")
        return 1
    print(f"ARM_CHECK_FINAL_OK every variant scope name over "
          f"{len(SHAPES)} shapes matches _c<chunk>_final, and the incumbent "
          f"carries {TAG!r} on none of them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
