"""pytest plugin: run the repo's *unmodified* kernel test file against Option B.

Loaded with `-p option_b_patch` and configured by three environment variables:

    OPTION_B=0|1          0 = inert (control run: the incumbent, unpatched)
    OPTION_B_G=N          num_groups to request
    OPTION_B_FORCE_P1=0|1 stage B via `_sc_topk_direct` (1) or `sparsecore_topk` (0)

The test file under test is `tests/kernels/test_dsv4_streamindex_topk.py`, taken
from the *checkout*, not from the image.  The image ships a stale 16-test copy
of it; a green `16 passed` is a stale-file failure, not a pass.  `shard.sh`
prints the file's md5 and this plugin prints the collection count, so both are
checkable from the job log.

The plugin rebinds the name `streamindex_topk` in the test module's namespace.
It deliberately does *not* patch
`vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk` itself,
because `option_b_topk` calls the incumbent for its unchunkable fall-back and
patching the module attribute would make that recurse.

What it binds, and why it is not a plain wrapper
------------------------------------------------

`test_streamindex_topk_shape` calls `jax.eval_shape(streamindex_topk, ...)`.
`jax.eval_shape` forwards to the callee's own tracing path -- honouring its
`static_argnames` -- only when `type(fun) is PjitFunction`; for anything else it
wraps the callable in a fresh `jit`, which turns *every* leaf into a tracer,
`k` included, and the run dies with "Non-hashable static arguments are not
supported".  So the bound object is `option_b_topk.make_dropin(...)`, a real
`jax.jit` object with the incumbent's signature.

The cost of that is that the bound object cannot plan chunks: `cu_q_lens` and
`distribution` are traced by the time its body runs, and a chunk's token bounds
are data.  It therefore runs Option B's *single-group* path -- this file's
stage-A launches, reshape/pad, `_effective_row_lengths` and stage B, at G=1.

Honesty about what a green run proves: it proves the Option B code path returns
what the repo's NumPy ground truth expects, at 20 parametrised shapes, for both
settings of `force_p1`.  It does **not** prove chunking is correct, and it
cannot: every case in this file has 1-3 sequences, so no plan with G>=2 could be
honoured even if planning were reachable from here.  The chunked-path evidence
is `probe_correctness.py`, which uses batch sizes that actually split.  The
terminal summary says which of the two this run was.
"""

from __future__ import annotations

import os
import sys

import pytest

_G = int(os.environ.get("OPTION_B_G", "1"))
_ON = os.environ.get("OPTION_B", "0") not in ("0", "", "false", "False")
_FORCE_P1 = os.environ.get("OPTION_B_FORCE_P1", "1") not in ("0", "", "false")


def pytest_report_header(config):
    return (f"option_b_patch: OPTION_B={'on' if _ON else 'off'} "
            f"num_groups={_G} force_p1={_FORCE_P1}")


def pytest_collection_modifyitems(session, config, items):
    print(f"\n[option_b_patch] collected {len(items)} tests")


def pytest_collection_finish(session):
    if not _ON:
        return
    import option_b_topk as ob

    dropin = ob.make_dropin(num_groups=_G, force_p1=_FORCE_P1, self_path=True)

    patched = 0
    # The test modules are imported at collection time; rebind in each one that
    # has the symbol.
    for name, mod in list(sys.modules.items()):
        if mod is None or "test_dsv4_streamindex_topk" not in name:
            continue
        if hasattr(mod, "streamindex_topk"):
            mod.streamindex_topk = dropin
            patched += 1
    print(f"[option_b_patch] rebound streamindex_topk -> {dropin} in "
          f"{patched} test module(s); num_groups={_G} force_p1={_FORCE_P1}")
    if patched == 0:
        raise pytest.UsageError(
            "option_b_patch: found no test module exporting "
            "`streamindex_topk` -- the patch would have been a no-op and the "
            "run would have silently exercised the incumbent")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    w = terminalreporter.write_line
    if not _ON:
        w("[option_b_patch] inert (control run: incumbent, unpatched)")
        return
    import option_b_topk as ob

    log = ob.TRACE_LOG
    paths: dict[str, int] = {}
    bad_guards = []
    for rec in log:
        paths[rec.get("path", "?")] = paths.get(rec.get("path", "?"), 0) + 1
        bad_guards += [g for g in rec.get("waves_guard", []) if not g["ok"]]
    w(f"[option_b_patch] num_groups={_G} force_p1={_FORCE_P1} "
      f"traces={len(log)}")
    for p, c in sorted(paths.items()):
        w(f"[option_b_patch]   path x{c}: {p}")
    if bad_guards:
        w(f"[option_b_patch]   waves_guard NOT satisfied: {bad_guards[:4]}")
    n_self = sum(c for p, c in paths.items() if p.startswith("self"))
    n_chunked = sum(c for p, c in paths.items() if p.startswith("chunked"))
    if n_chunked == 0:
        w("[option_b_patch] NOTE: no trace was chunked (this file's batches are "
          "1-3 sequences, and the jitted entry cannot plan). This run proves "
          "the Option B code path is TRANSPARENT at G=1, not that chunking is "
          "correct -- see probe_correctness.py for that.")
    if n_self == 0:
        w("[option_b_patch] WARNING: not one trace took an Option B path; the "
          "run exercised only the incumbent and proves nothing about Option B")
