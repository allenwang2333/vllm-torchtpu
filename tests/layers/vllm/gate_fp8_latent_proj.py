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


def _relerr(ref, layer, method):
    """Round-trip error of the stored weight, for the accuracy discussion.

    Reported, not asserted -- a threshold here would be a number invented in a
    unit test standing in for an eval. It exists so the PR can quote a measured
    figure instead of "e2m1 has 1 mantissa bit".
    """
    w, sc = layer.weight, getattr(layer, "weight_scale", None)
    if sc is None:
        return 0.0
    r = ref.t().to(torch.float32)                       # [n_in, n_out]
    if w.dtype == torch.float8_e4m3fn:
        deq = w.to(torch.float32) * sc.reshape(-1, 1, sc.shape[-1]).repeat_interleave(
            r.shape[0] // sc.shape[1], dim=1).reshape(r.shape)
    else:                                               # native fp4 carrier
        lut = torch.tensor(_FP4, dtype=torch.float32)
        b = w.view(torch.uint8)
        idx = torch.stack([b & 0xF, b >> 4], dim=-1).reshape(r.shape)
        blk = sc.shape[1]
        deq = lut[idx.long()] * sc.reshape(blk, 1, sc.shape[-1]).expand(
            blk, r.shape[0] // blk, sc.shape[-1]).reshape(r.shape)
    return (deq - r).norm().item() / r.norm().item()


_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0,
        -3.0, -4.0, -6.0)


def check(name, got, want):
    checks.append((got == want, name, got, want))


def arm(on, mode="fp8"):
    # envs is read per call, not at import, so both gates flip cleanly here.
    envs.TPU_K3_FP8_LATENT_PROJ = on and mode == "fp8"
    envs.TPU_K3_MXFP4_LATENT_PROJ = on and mode == "mxfp4"
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
        torch.manual_seed(0)                      # same draw in every arm
        torch.nn.init.normal_(layer.weight, std=0.02)
        ref = layer.weight.detach().clone()
        m.process_weights_after_loading(layer)
        out[nm + ".bitident"] = (layer.weight.dtype == ref.dtype
                                 and layer.weight.shape == ref.shape
                                 and torch.equal(layer.weight, ref))
        out[nm + ".relerr"] = _relerr(ref, layer, m)
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
on = arm(True, "fp8")
m4 = arm(True, "mxfp4")
BF16 = HID * LAT * 2

print(f"{'':>22}{'GATE OFF':>30}{'FP8':>30}{'MXFP4':>30}")
for k in sorted(off):
    print(f"{k:>22}{str(off[k]):>30}{str(on[k]):>30}{str(m4[k]):>30}")

# --- the OFF arm: today's shipped behaviour, unchanged -----------------------
check("OFF down is UnquantizedLinearMethod", off["down.cls"], UnquantizedLinearMethod.__name__)
check("OFF up   is UnquantizedLinearMethod", off["up.cls"], UnquantizedLinearMethod.__name__)
check("OFF down weight stays bf16", off["down.wdtype_after"], "torch.bfloat16")
check("OFF down weight stays 51.38 MB", off["down.bytes"], BF16)
check("OFF up   weight stays 51.38 MB", off["up.bytes"], BF16)
check("OFF carries no weight_scale", off["down.has_scale"], False)
# The leg that protects everyone who does not opt in, and the one most likely
# to rot silently: OFF must leave the loaded tensor bit-for-bit untouched.
check("OFF down weight is BIT-IDENTICAL to base", off["down.bitident"], True)
check("OFF up   weight is BIT-IDENTICAL to base", off["up.bitident"], True)
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
# --- the MXFP4 arm: native fp4 carrier, rank-4 scale at block 256 ----------
# Block 256 is the whole result: at 32 or 128 gmm_v2 takes the dequantize-
# BEFORE branch and the arm is measurably worse than fp8 (13 118 / 12 428
# slots against 11 830 break-even). So the block is asserted, not assumed.
check("M4  down is VllmMxfp4LinearMethodTPU", m4["down.cls"], "VllmMxfp4LinearMethodTPU")
check("M4  down weight is float4_e2m1fn_x2", m4["down.wdtype_after"], "torch.float4_e2m1fn_x2")
check("M4  up   weight is float4_e2m1fn_x2", m4["up.wdtype_after"], "torch.float4_e2m1fn_x2")
check("M4  down scale is RANK 4 (-> gmm_v2)", m4["down.scale_rank"], 4)
check("M4  down scale is [1, K/256, 1, N]", m4["down.scale_shape"], (1, HID // 256, 1, LAT))
check("M4  up   scale is [1, K/256, 1, N]", m4["up.scale_shape"], (1, LAT // 256, 1, HID))
# gmm_v2's rhs is native float4_e2m1fn[n_in, n_out]; the torch carrier halves
# the LAST axis (sharded_jax_op.py:85), so the stored shape is [n_in, n_out/2].
check("M4  down carrier is [n_in, n_out/2]", m4["down.wshape_after"], (HID, LAT // 2))
check("M4  up   carrier is [n_in, n_out/2]", m4["up.wshape_after"], (LAT, HID // 2))
check("M4  down weight is EXACTLY a quarter", m4["down.bytes"], BF16 // 4)
check("M4  up   weight is EXACTLY a quarter", m4["up.bytes"], BF16 // 4)
check("OFF down keeps [n_out, n_in]", off["down.wshape_after"], (LAT, HID))
check("ON  down is transposed to (k, n)", on["down.wshape_after"], (HID, LAT))

print()
ok = True
for good, name, got, want in checks:
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {name:<42} got={got} want={want}")
sv = HID * LAT * 2 - on["down.bytes"]
print(f"\n  relative round-trip error (Frobenius): "
      f"fp8 down {on['down.relerr']:.4f}  up {on['up.relerr']:.4f}   |   "
      f"mxfp4 down {m4['down.relerr']:.4f}  up {m4['up.relerr']:.4f}   "
      f"(reported, not asserted -- an eval decides, not a unit test)")
print(f"\n  bytes/step saved = 2 proj x 92 layers x {sv/1e6:.2f} MB = "
      f"{2*92*sv/1e9:.2f} GB of {2*92*BF16/1e9:.2f} GB")
print(f"\nGATE {'PASS' if ok else 'FAIL'} -- TPU_K3_FP8_LATENT_PROJ is two-sided: "
      "OFF reproduces today's bf16 layer exactly, ON halves the weight to fp8")
sys.exit(0 if ok else 1)
