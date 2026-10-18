"""List the megakernel's VMEM scratch buffers (largest first) for given fused verify buckets."""
import dataclasses
import math
import os
import sys
import types

REPO = os.environ.get("REPO", os.path.join(os.path.dirname(__file__), "..", "src"))
for _name, _sub in [("vllm_torchtpu", ""), ("vllm_torchtpu.kernels", "kernels")]:
    _m = types.ModuleType(_name)
    _m.__path__ = [os.path.join(REPO, "vllm_torchtpu", _sub)]
    sys.modules[_name] = _m

import jax.numpy as jnp  # noqa: E402
from vllm_torchtpu.kernels.qwen35_megakernel import config as C  # noqa: E402
from vllm_torchtpu.kernels.qwen35_megakernel import megakernel as K  # noqa: E402
from vllm_torchtpu.kernels.qwen35_megakernel.context import MKBuildContext  # noqa: E402

for b in [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "64,104,112").split(",")]:
    c = dataclasses.replace(
        C.FULL_EP, R=8, B=b, T=4, stack_e=False, e_slots=68, paged_kv=True, fused_pool_io=True,
        dcp_size=4, num_pages=8192, max_pages_per_seq=515, gdn_stride_pages=5,
        act_mode="from_ckpt", ckpt_2d=False, roff_1d=False, ckpt_k=4)
    t = C.default_tiles(c)
    ctx = MKBuildContext(c, t, gq_pack=K.GQ_PACK, topk_lanes=K.TOPK_LANES,
                         dmm_unroll=K.DMM_UNROLL, conv_stage_min_b=K.CONV_STAGE_MIN_B)
    rows = []
    for k in ctx.s_names:
        v = ctx.scratch[k]
        if hasattr(v, "shape") and "VMEM" in str(getattr(v, "memory_space", "")).upper():
            rows.append((math.prod(v.shape) * jnp.dtype(v.dtype).itemsize, k, tuple(v.shape),
                         jnp.dtype(v.dtype).name))
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    print(f"SCRATCH B={b} nbd={t.nbd} hier={t.hier} total={tot / 2**20:.2f} MiB", flush=True)
    for sz, k, shp, dt in rows[:24]:
        print(f"SCRATCH   {sz / 2**20:7.3f} MiB  {k:16s} {shp} {dt}", flush=True)
