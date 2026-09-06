"""Run the SHARED, UNMODIFIED correctness gate against O2.

`instruments/correctness.py` is read-only and binds `streamindex_topk` into its
own namespace at import. This driver rebinds that one name to O2's entry point
and then calls the instrument's own `main()`. The reference, both metrics and
both tolerances (`TOL_DEFICIT=0.015`, `TOL_CHURN=0.03`, validated by
`j-ae538ec1-d606-48f4-8c11`) are the instrument's and are untouched -- the only
thing swapped is the unit under test, which is the point of a variant.

Exits with the instrument's exit code.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.environ.get(
    "INST_DIR", os.path.join(_HERE, "..", "instruments")))

import api  # noqa: E402
import correctness  # noqa: E402  the shared gate, unmodified
import pallas_mosaic_tpu as policy  # noqa: E402
from shapes import VMEM_LIMIT  # noqa: E402


def _dump_vmem():
    """What each regime's heuristic picked, and what it costs in VMEM.

    Printed here rather than in the bench because it is a property of the
    config, not of a run, and because a correctness job is the cheapest place
    to record it.
    """
    common = dict(vmem_limit_bytes=VMEM_LIMIT, kv_packing=4, lkv_dim=256,
                  num_q_heads=32, head_dim=128, q_itemsize=1,
                  decode_req_batch_size=4)
    budget = policy.vmem_budget_bytes(VMEM_LIMIT)
    print(f"=== VMEM (budget {budget / 2**20:.1f} MiB from "
          f"pltpu.get_tpu_info().vmem_capacity_bytes vs vmem_limit_bytes "
          f"{VMEM_LIMIT / 2**20:.0f} MiB) ===", flush=True)
    for page_size, label in ((1024, "(A) 8k1k"), (512, "(B) AgentX")):
        cfg = policy._get_heuristics_config(page_size=page_size, verbose=True,
                                            **common)
        print(f"  {label}: page_size={page_size} bkv_p={cfg.bkv_p} "
              f"bq_sz={cfg.bq_sz}", flush=True)
        for launch, parts in policy.vmem_report(
                cfg, page_size=page_size, kv_packing=4, lkv_dim=256,
                num_q_heads=32, head_dim=128, q_itemsize=1,
                decode_req_batch_size=4).items():
            temps = parts["s"] + parts["s_summed"]
            print(f"      {launch:18} total {parts['total'] / 2**20:8.2f} MiB "
                  f"= scratch {(parts['total'] - temps) / 2**20:7.2f} "
                  f"+ einsum temps {temps / 2**20:8.2f}", flush=True)
        # The tile the design specifies, whether or not the heuristic took it.
        spec = policy._spec_config(page_size)
        if spec != cfg:
            for launch, parts in policy.vmem_report(
                    spec, page_size=page_size, kv_packing=4, lkv_dim=256,
                    num_q_heads=32, head_dim=128, q_itemsize=1,
                    decode_req_batch_size=4).items():
                temps = parts["s"] + parts["s_summed"]
                print(f"      SPEC {launch:13} total "
                      f"{parts['total'] / 2**20:8.2f} MiB "
                      f"= scratch {(parts['total'] - temps) / 2**20:7.2f} "
                      f"+ einsum temps {temps / 2**20:8.2f}", flush=True)


if __name__ == "__main__":
    _dump_vmem()
    print("\n=== gate: instruments/correctness.py, unmodified ===", flush=True)
    correctness.streamindex_topk = api.streamindex_topk
    correctness.main()
