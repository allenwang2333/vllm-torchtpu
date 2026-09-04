"""Two-sided gate for TPU_K3_FP8_LATENT_PROJ, on CPU, no TPU.

One-sided would be worthless here: "the flag produces an fp8 method" passes
just as happily on a harness that returns fp8 unconditionally. Both arms are
asserted, from one process, on the SAME layer construction -- and the arm that
matters most is the OFF one, because a gate that is secretly always-on is the
failure that reaches the pod as a silently-changed baseline.
"""
import os
import sys

import torch

os.environ.setdefault("VLLM_TARGET_DEVICE", "tpu")
import vllm_torchtpu  # noqa: F401  (registers the platform + quant configs)
from vllm.model_executor.layers.linear import (  # noqa: E402
    ReplicatedLinear, UnquantizedLinearMethod)

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.layers.vllm import latent_proj_intra_chip as L  # noqa: E402
from vllm_torchtpu.layers.vllm.quantization.fp8 import \
    VllmFp8LinearMethodTPU  # noqa: E402

# A real ReplicatedLinear reads the TP rank/size in __init__ but never
# communicates, and a real gloo group cannot be built here (the TPU platform
# hands GroupCoordinator a "tpu" device string torch does not know on CPU).
# Stub the two readers only -- the quant method, the weight loader and
# process_weights_after_loading are all the production code.
import types  # noqa: E402

import vllm.distributed.parallel_state as _ps  # noqa: E402
_ps._TP = types.SimpleNamespace(rank_in_group=0, world_size=1,
                                device_group=None, cpu_group=None)

# Production builds these at model_config.dtype = bfloat16; a bare harness
# would default to fp32 and quietly measure a 4x saving instead of a 2x one.
torch.set_default_dtype(torch.bfloat16)

HID, LAT = 7168, 3584
checks = []


def check(name, got, want):
    checks.append((got == want, name, got, want))


def arm(on: bool):
    envs.TPU_K3_FP8_LATENT_PROJ = on          # envs is read per call, not at import
    down = L.make_latent_projection(HID, LAT, bias=False, quant_config=None,
                                    prefix="m.layers.1.block_sparse_moe."
                                           "routed_expert_down_proj")
    up = L.make_latent_projection(LAT, HID, bias=False, quant_config=None,
                                  prefix="m.layers.1.block_sparse_moe."
                                         "routed_expert_up_proj")
    out = {}
    for nm, layer in (("down", down), ("up", up)):
        m = layer.quant_method
        out[nm + ".cls"] = type(m).__name__
        out[nm + ".replicated"] = isinstance(layer, ReplicatedLinear)
        out[nm + ".wdtype_before"] = str(layer.weight.dtype)
        torch.nn.init.normal_(layer.weight, std=0.02)
        m.process_weights_after_loading(layer)
        out[nm + ".wdtype_after"] = str(layer.weight.dtype)
        out[nm + ".wshape_after"] = tuple(layer.weight.shape)
        out[nm + ".bytes"] = layer.weight.numel() * layer.weight.element_size()
        out[nm + ".has_scale"] = hasattr(layer, "weight_scale")
        sc = getattr(layer, "weight_scale", None)
        # The scale's RANK is what _quantized_matmul_jax dispatches on:
        # 1-D -> plain XLA dot (VPU rewrite at M=1, a regression);
        # rank-4 -> gmm_v2 (MXU at M=1). This is the load-bearing assertion.
        out[nm + ".scale_rank"] = None if sc is None else sc.ndim
        out[nm + ".scale_shape"] = None if sc is None else tuple(sc.shape)
    return out


off = arm(False)
on = arm(True)
BF16 = HID * LAT * 2

print(f"{'':>22}{'GATE OFF':>34}{'GATE ON':>34}")
for k in sorted(off):
    print(f"{k:>22}{str(off[k]):>34}{str(on[k]):>34}")

# --- the OFF arm: today's shipped behaviour, unchanged -----------------------
check("OFF down is UnquantizedLinearMethod", off["down.cls"], UnquantizedLinearMethod.__name__)
check("OFF up   is UnquantizedLinearMethod", off["up.cls"], UnquantizedLinearMethod.__name__)
check("OFF down weight stays bf16", off["down.wdtype_after"], "torch.bfloat16")
check("OFF down weight stays 51.38 MB", off["down.bytes"], BF16)
check("OFF up   weight stays 51.38 MB", off["up.bytes"], BF16)
check("OFF carries no weight_scale", off["down.has_scale"], False)
# --- the ON arm: fp8, half the bytes, scale present -------------------------
check("ON  down is VllmFp8LinearMethodTPU", on["down.cls"], VllmFp8LinearMethodTPU.__name__)
check("ON  up   is VllmFp8LinearMethodTPU", on["up.cls"], VllmFp8LinearMethodTPU.__name__)
check("ON  down weight is float8_e4m3fn", on["down.wdtype_after"], "torch.float8_e4m3fn")
check("ON  up   weight is float8_e4m3fn", on["up.wdtype_after"], "torch.float8_e4m3fn")
check("ON  down weight is EXACTLY half", on["down.bytes"], BF16 // 2)
check("ON  up   weight is EXACTLY half", on["up.bytes"], BF16 // 2)
check("ON  carries a weight_scale", on["down.has_scale"], True)
# The whole difference between -10% and +9%: rank 4 reaches gmm_v2, rank 1 does not.
check("ON  down scale is RANK 4 (-> gmm_v2)", on["down.scale_rank"], 4)
check("ON  up   scale is RANK 4 (-> gmm_v2)", on["up.scale_rank"], 4)
check("ON  down scale is [1, K/128, 1, N]", on["down.scale_shape"], (1, HID // 128, 1, LAT))
check("ON  up   scale is [1, K/128, 1, N]", on["up.scale_shape"], (1, LAT // 128, 1, HID))
# --- both arms still build the same LAYER, so the gate changes dtype only ----
check("both arms are ReplicatedLinear", (off["down.replicated"], on["down.replicated"]), (True, True))
# The fp8 loader transposes to canonical (k, n) at load (fp8.py:1053-1065) so the
# contraction does not transpose per step; bf16 keeps vLLM's [n_out, n_in].
check("OFF down keeps [n_out, n_in]", off["down.wshape_after"], (LAT, HID))
check("ON  down is transposed to (k, n)", on["down.wshape_after"], (HID, LAT))

print()
ok = True
for good, name, got, want in checks:
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {name:<42} got={got} want={want}")
sv = HID * LAT * 2 - on["down.bytes"]
print(f"\n  bytes/step saved = 2 proj x 92 layers x {sv/1e6:.2f} MB = "
      f"{2*92*sv/1e9:.2f} GB of {2*92*BF16/1e9:.2f} GB")
print(f"\nGATE {'PASS' if ok else 'FAIL'} -- TPU_K3_FP8_LATENT_PROJ is two-sided: "
      "OFF reproduces today's bf16 layer exactly, ON halves the weight to fp8")
sys.exit(0 if ok else 1)
