"""Fault arms for calibrating the gmm_v2 397b scorer (and the served fidelity check).

    python faults.py <golden checkout> <empty dir>

writes one kernel directory per fault under <empty dir>, each a copy of golden's
`src/vllm_torchtpu/kernels/megablox` with a wrapper appended to `gmm_v2.py` that
rebinds `gmm_v2`, the name the scorer and `fused_moe_gmm.py` call:

  golden_aa     golden, byte for byte, under another arm name (the A/A arm the
                noise floor is measured from).
  wrong_kernel  drops the last 512 columns of k for the rows of the last active
                group: a few rows wrong, which only a worst-row metric sees.
  slow_kernel   forces tile_n=128, below the tiler's floor, with the same
                arithmetic: slower and correct. It also prints one
                GMM_V2_MARKER line per traced shape, with the call chain that
                reached it, so a served run of this tree shows in its log that
                this file executes and through which entry point.

The faults live in the arms, not in bench.py, so a scoring job has no switch
that changes what it measures. A served arm uses the slow kernel directory over
golden's checkout (`--overlay`).

    python faults.py <golden checkout> <dir> --overlay slow_kernel <tree out>

copies the checkout to <tree out> with that fault's kernel directory in place.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

KERNEL_DIR = "src/vllm_torchtpu/kernels/megablox"
MARK = "# gmm_v2 397b scorer calibration fault: "

FAULTS = {
    "golden_aa": None,
    "wrong_kernel": '''
import jax.numpy as _fault_jnp

_fault_gmm_v2 = gmm_v2


def gmm_v2(lhs, rhs, group_sizes, *args, **kwargs):
    """gmm_v2 without the last 512 k columns of the last active group's rows."""
    sizes = group_sizes.astype(_fault_jnp.int32)
    last = _fault_jnp.max(_fault_jnp.where(sizes > 0, _fault_jnp.arange(sizes.shape[0]), 0))
    end = _fault_jnp.cumsum(sizes)[last]
    rows = _fault_jnp.arange(lhs.shape[0])[:, None]
    cols = _fault_jnp.arange(lhs.shape[1])[None, :]
    drop = (rows >= end - sizes[last]) & (rows < end) & (cols >= lhs.shape[1] - 512)
    lhs = _fault_jnp.where(drop, _fault_jnp.zeros((), lhs.dtype), lhs)
    return _fault_gmm_v2(lhs, rhs, group_sizes, *args, **kwargs)
''',
    "slow_kernel": '''
import dataclasses as _fault_dataclasses
import sys as _fault_sys
import traceback as _fault_traceback

_fault_gmm_v2 = gmm_v2
_fault_marked = set()


def _fault_tiling(*args, **kwargs):
    return _fault_dataclasses.replace(calculate_tiling(*args, **kwargs), tile_n=128)


def gmm_v2(lhs, rhs, *args, **kwargs):
    """gmm_v2 at tile_n=128, below the tiler's floor, with the same arithmetic."""
    key = (tuple(lhs.shape), tuple(rhs.shape), kwargs.get("fuse_act"))
    if key not in _fault_marked:
        _fault_marked.add(key)
        chain = " -> ".join(f"{Path_name(f.filename)}:{f.name}" for f in _fault_traceback.extract_stack()[:-1]
                            if "vllm_torchtpu" in f.filename)
        print(f"GMM_V2_MARKER file={__file__} lhs={key[0]} rhs={key[1]} fuse_act={key[2]} "
              f"chain={chain}", file=_fault_sys.stderr, flush=True)
    kwargs.setdefault("tile_info", _fault_tiling)
    return _fault_gmm_v2(lhs, rhs, *args, **kwargs)


def Path_name(path):
    return path.split("vllm_torchtpu/", 1)[-1]
''',
}


def make_fault_arms(golden: Path, out: Path) -> dict[str, Path]:
    src = golden / KERNEL_DIR
    if not (src / "gmm_v2.py").is_file():
        raise SystemExit(f"{golden} has no {KERNEL_DIR}/gmm_v2.py")
    made = {}
    for fault, patch in FAULTS.items():
        dest = out / fault
        if dest.exists():
            raise SystemExit(f"{dest} exists; name an empty directory")
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__"))
        if patch:
            kernel = dest / "gmm_v2.py"
            kernel.write_text(kernel.read_text() + "\n\n" + MARK + fault + "\n" + patch)
        made[fault] = dest
    return made


def overlay(golden: Path, kernel_dir: Path, tree: Path) -> None:
    shutil.copytree(golden, tree, ignore=shutil.ignore_patterns("__pycache__", ".git"))
    shutil.rmtree(tree / KERNEL_DIR)
    shutil.copytree(kernel_dir, tree / KERNEL_DIR)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("golden", type=Path, help="golden's checkout (the root holding src/)")
    p.add_argument("out", type=Path, help="an empty directory for the fault arms")
    p.add_argument("--overlay", metavar="FAULT", help="also write a served tree with this fault")
    p.add_argument("tree", type=Path, nargs="?", help="where --overlay writes the served tree")
    a = p.parse_args(argv)
    made = make_fault_arms(a.golden.resolve(), a.out.resolve())
    for fault, path in made.items():
        print(f"{fault}={path}")
    if a.overlay:
        if not a.tree:
            raise SystemExit("--overlay needs a tree path")
        overlay(a.golden.resolve(), made[a.overlay], a.tree.resolve())
        print(f"served tree with {a.overlay}: {a.tree.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
