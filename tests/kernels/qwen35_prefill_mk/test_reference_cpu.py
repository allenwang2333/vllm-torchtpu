import os, sys, types
os.environ["JAX_PLATFORMS"] = "cpu"
import importlib.util
import jax, jax.numpy as jnp, numpy as np
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src")
# Load the two modules without importing the vllm_torchtpu package (needs torch).
pkg = types.ModuleType("vllm_torchtpu"); pkg.__path__ = [SRC + "/vllm_torchtpu"]; sys.modules["vllm_torchtpu"] = pkg
k = types.ModuleType("vllm_torchtpu.kernels"); k.__path__ = [SRC + "/vllm_torchtpu/kernels"]; sys.modules["vllm_torchtpu.kernels"] = k
from vllm_torchtpu.kernels.qwen35_prefill_mk import reference as R
c = R.StageConfig(layer_types=("gdn", "attn"), H=256, NH=4, NKV=2, HD=128, ROT=32, KH=2, VH=4,
                  KD=128, VD=128, E=16, TOPK=4, I=64, IS=64)
t, ctx = 128, 64
types_, w, st, h, r = R.random_stage(c, t, ctx)
h2, r2, st2 = jax.jit(lambda w, h, r, st: R.stage_step(c, w, h, r, st, ctx))(w, h, r, st)
print("out", h2.shape, h2.dtype, "finite", bool(jnp.isfinite(h2.astype(jnp.float32)).all()),
      "rms", float(jnp.sqrt(jnp.mean(h2.astype(jnp.float32) ** 2))), "res rms", float(jnp.sqrt(jnp.mean(r2.astype(jnp.float32) ** 2))))
print("kv written", bool(jnp.any(st2[1]["k"][ctx:].astype(jnp.float32) != st[1]["k"][ctx:].astype(jnp.float32))))
