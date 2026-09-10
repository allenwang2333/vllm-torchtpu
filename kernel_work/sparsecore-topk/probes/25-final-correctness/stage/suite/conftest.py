"""Swap the subject module into `sys.modules` before the frozen suite imports it.

The frozen `test_dsv4_sparsecore_topk.py` does, at module import:

    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
        LANES, _align_to, _pick_partition, sparsecore_topk)

so the only way to point it at a variant is to bind that dotted name before
collection. `conftest.py` is imported before test modules, which is the hook.

⛔ The obvious alternative -- `cp variant.py` over the installed file -- is
**wrong here**: four shards share one container filesystem, so shard 1's copy
would silently retarget shards 0, 2 and 3. This route is per-process and
therefore shard-safe, which is the whole reason it exists.

Selected by `SCTOPK_SUBJECT` (a path) and gated by `SCTOPK_SUBJECT_MD5`. Both
are required: an unset `SCTOPK_SUBJECT` is a hard error, not a silent fallback
to the installed copy, because "the suite ran against the image" is exactly the
pass this exercise must never report.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import pathlib
import sys

TARGET = "vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk"


def _install() -> None:
    path = os.environ.get("SCTOPK_SUBJECT")
    want = os.environ.get("SCTOPK_SUBJECT_MD5")
    if not path or not want:
        raise RuntimeError(
            "SCTOPK_SUBJECT and SCTOPK_SUBJECT_MD5 must both be set; refusing "
            "to run the suite against whatever the image happens to ship")
    p = pathlib.Path(path)
    got = hashlib.md5(p.read_bytes()).hexdigest()
    print(f"\nSUBJECT_PATH {p}")
    print(f"SUBJECT_MD5  {got}")
    if got != want:
        raise RuntimeError(f"SUBJECT_GATE_FAIL got={got} want={want}")

    # Import the parent package first, so the variant loads against a fully
    # initialised `vllm_torchtpu` exactly as the installed module would.
    parent_name = TARGET.rsplit(".", 1)[0]
    parent = importlib.import_module(parent_name)
    installed = getattr(parent, "sparsecore_topk", None)
    inst_file = getattr(installed, "__file__", None)
    if inst_file:
        inst_md5 = hashlib.md5(pathlib.Path(inst_file).read_bytes()).hexdigest()
        same = "SAME" if inst_md5 == got else "DIFFERENT -- using SCTOPK_SUBJECT"
        print(f"INSTALLED    {inst_file}  md5={inst_md5}  {same}")

    spec = importlib.util.spec_from_file_location(TARGET, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[TARGET] = mod
    spec.loader.exec_module(mod)
    setattr(parent, "sparsecore_topk", mod)

    # Prove the binding took, by identity and not by hope.
    again = importlib.import_module(TARGET)
    assert again is mod, "sys.modules swap did not take"
    assert pathlib.Path(again.__file__) == p, again.__file__
    print(f"SUBJECT_GATE_OK  {got}  bound as {TARGET}")


_install()
