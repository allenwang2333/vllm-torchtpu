"""pytest plugin: run the repo's own tests against Option E, unmodified.

`tests/kernels/test_dsv4_streamindex_topk.py` is run **byte-identical** to what
is committed.  Nothing in `src/` is edited either.  This plugin, loaded with
`-p option_e_patch`, rebinds the two names through which the partition rule is
reached:

  * `sparsecore_topk.sparsecore_topk`   -- the public entry, and
  * `streamindex_topk.sparsecore_topk`  -- the `from ... import` alias
    `streamindex_topk.py:25` bound at import time, which is the one the kernel
    under test actually calls.

Both are rebound because rebinding only the first is a no-op for this test
file, and a no-op patch is indistinguishable from a pass.  The plugin prints
what it swapped and asserts the alias moved, so a silent no-op is loud.

Usage:
    PYTHONPATH=<this dir>:$PYTHONPATH \
      pytest -p option_e_patch tests/kernels/test_dsv4_streamindex_topk.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ENABLED = os.environ.get("OPTION_E", "1") != "0"


def pytest_configure(config):
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as simod

    before = simod.sparsecore_topk
    print(f"\n[option-e] OPTION_E={'on' if _ENABLED else 'off (control)'}")
    print(f"[option-e] sparsecore_topk module: {scmod.__file__}")
    print(f"[option-e] streamindex_topk module: {simod.__file__}")
    if not _ENABLED:
        print("[option-e] control run: incumbent left in place")
        return

    from partition_v2 import pick_partition, sparsecore_topk_v2

    scmod.sparsecore_topk = sparsecore_topk_v2
    simod.sparsecore_topk = sparsecore_topk_v2
    assert simod.sparsecore_topk is not before, "patch did not take"
    print(f"[option-e] patched streamindex_topk.sparsecore_topk: "
          f"{before} -> {simod.sparsecore_topk}")
    # A couple of decisions printed so the log shows the rule is live.
    for b, n, k in ((8, 10240, 2048), (16, 10240, 2048), (64, 10240, 2048)):
        print(f"[option-e] pick_partition(b={b}, n={n}, k={k}) = "
              f"{pick_partition(b, n, k)} "
              f"(incumbent {scmod._pick_partition(b, n, k)})")
