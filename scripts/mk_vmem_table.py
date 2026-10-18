"""Print the megakernel's VMEM scratch bytes per fused verify bucket (default tiles, v7x-8 1P1D
decode geometry: TP=EP=8, e_slots=68, DCP=4, T=4). CPU only: builds the kernel spec, no compile.
The scoped-VMEM limit is tiles.vmem_mib (62 MiB); the compiler adds its own temporaries on top
of the scratch, so a bucket whose scratch is close to the limit can still fail to compile."""
import dataclasses
import json
import os
import sys
import types

REPO = os.environ.get("REPO", os.path.join(os.path.dirname(__file__), "..", "src"))
for _name, _sub in [("vllm_torchtpu", ""), ("vllm_torchtpu.kernels", "kernels")]:
    _m = types.ModuleType(_name)
    _m.__path__ = [os.path.join(REPO, "vllm_torchtpu", _sub)]
    sys.modules[_name] = _m

from vllm_torchtpu.kernels.qwen35_megakernel import config as C  # noqa: E402
from vllm_torchtpu.kernels.qwen35_megakernel import megakernel as K  # noqa: E402

BS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else
      "16,32,48,56,64,72,80,84,88,96,104,112,120,128").split(",")]
for b in BS:
    c = dataclasses.replace(
        C.FULL_EP, R=8, B=b, T=4, stack_e=False, e_slots=68, paged_kv=True, fused_pool_io=True,
        dcp_size=4, num_pages=8192, max_pages_per_seq=515, gdn_stride_pages=5,
        act_mode="from_ckpt", ckpt_2d=False, roff_1d=False, ckpt_k=4)
    t = C.default_tiles(c)
    try:
        _, info = K.build(c, t)
        mib = info["vmem_scratch_bytes"] / 2**20
        print("VMEM " + json.dumps({"B": b, "nbd": t.nbd, "nbe": t.nbe, "hier": t.hier,
                                    "rsag": t.rsag, "scratch_mib": round(mib, 2),
                                    "limit_mib": t.vmem_mib}), flush=True)
    except Exception as e:  # noqa
        print("VMEM " + json.dumps({"B": b, "error": repr(e)[:300]}), flush=True)
