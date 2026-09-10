"""Prove the `v10` sample-and-count arm actually FIRED, before believing its 68.

`v10`'s arm is selected by an environment default read once at import
(`SCTOPK_SAMPLE_AND_COUNT`, `SCTOPK_COUNT_ARITY`). An env var that is *set*
proves nothing: if `sparsecore_topk_sample` failed to import, or the default
plumbing broke, or only one of the two files was mapped, the module happily
runs the incumbent branch and all 68 go green against code nobody changed.

The kernel names itself after its flags -- `_sc_topk_direct` builds
`sc_topk_direct_b{b}_n{n}_k{k}` and appends `_sc{arity}` on the sample path
only (`sparsecore_topk.py:578,596`). So the scope name in the lowered program
is a positive activation check that does not depend on anyone's reading of the
flag plumbing.

    python arm_check.py --subject <path> --arity 2 [--expect-sample 1]

Exits non-zero unless the expected name appears. Prints every
`sc_topk_direct_*` name it found either way, so a failure says *what* fired.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import pathlib
import re
import sys

NAME_RE = re.compile(r"sc_topk_direct_b\d+_n\d+_k\d+(?:_[A-Za-z0-9]+)*")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--subject-md5", default="")
    ap.add_argument("--arity", type=int, required=True)
    ap.add_argument("--expect-sample", type=int, default=1)
    ap.add_argument("--expect-suffix", default="",
                    help="scope-name suffix that proves this arm; overrides "
                         "--arity/--expect-sample. NONE = require no suffix.")
    ap.add_argument("--b", type=int, default=64)
    ap.add_argument("--n", type=int, default=10240)
    ap.add_argument("--k", type=int, default=2048)
    a = ap.parse_args()

    p = pathlib.Path(a.subject)
    got = hashlib.md5(p.read_bytes()).hexdigest()
    print(f"ARM_CHECK subject={p} md5={got} arity={a.arity}")
    if a.subject_md5 and got != a.subject_md5:
        print(f"ARM_CHECK_MD5_FAIL want={a.subject_md5}")
        return 2

    # Load the file under its own name so this cannot be confused with whatever
    # the image installed.
    spec = importlib.util.spec_from_file_location("arm_check_subject", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    for attr in ("_SAMPLE_AND_COUNT_DEFAULT", "_COUNT_ARITY_DEFAULT"):
        print(f"  {attr} = {getattr(mod, attr, '<absent>')}")
    print(f"  sample module = {getattr(getattr(mod, '_scs', None), '__file__', '<absent>')}")

    import jax
    import jax.numpy as jnp
    import numpy as np

    scores = jnp.asarray(np.zeros((a.b, a.n), np.float32))
    rl = jnp.asarray(np.full(a.b, a.n, np.int32))
    lowered = jax.jit(lambda s, l: mod.sparsecore_topk(s, a.k, l)).lower(scores, rl)
    names = set()
    for text in (lowered.as_text(), lowered.compile().as_text()):
        names |= set(NAME_RE.findall(text))
    print(f"  scope names found: {sorted(names) or '<NONE>'}")

    if not names:
        print("ARM_CHECK_FAIL no sc_topk_direct_* name in the lowered program -- "
              "the check itself did not work, do not read this as a null")
        return 3
    if a.expect_suffix:
        if a.expect_suffix == "NONE":
            ok = all(re.fullmatch(r"sc_topk_direct_b\d+_n\d+_k\d+", nm) for nm in names)
            print(f"ARM_CHECK_{'OK' if ok else 'FAIL'} expected a bare "
                  "sc_topk_direct_b_n_k with no arm suffix")
        else:
            hit = [nm for nm in names if nm.endswith(a.expect_suffix)]
            ok = bool(hit)
            print(f"ARM_CHECK_{'OK' if ok else 'FAIL'} expected a name ending "
                  f"{a.expect_suffix}; matched {sorted(hit)}")
        return 0 if ok else 1
    want = f"_sc{a.arity}"
    hit = [nm for nm in names if nm.endswith(want)]
    if a.expect_sample:
        ok = bool(hit)
        print(f"ARM_CHECK_{'OK' if ok else 'FAIL'} expected a name ending {want}; "
              f"matched {sorted(hit)}")
    else:
        ok = not any(re.search(r"_sc\d+$", nm) for nm in names)
        print(f"ARM_CHECK_{'OK' if ok else 'FAIL'} expected NO _sc<arity> suffix "
              "(control arm)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
