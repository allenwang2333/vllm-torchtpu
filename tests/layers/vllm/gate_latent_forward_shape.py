"""Consumer-side gate: a FORWARD PASS with a symbolic batch, under fake tensors.

Build 375 arm C died in `torch.compile` tracing with

    mul(FakeTensor(size=(s72, 1792)), FakeTensor(size=(3584,)))
    RuntimeError: The size of tensor a (1792) must match ... (3584)

1792 = 3584/2, the fp4 packing factor. The 32 load-time checks in
gate_fp8_latent_proj.py all passed, because every one of them constrains the
WEIGHT and its SCALE at load. The defect is entirely on the consumer side, so
no producer-side assertion could see it: the output width is taken from the
packed carrier's last dim instead of the logical one.

This gate runs the layer's own forward under FakeTensorMode with a DYNAMIC
batch, which is the exact mechanism that failed -- fake-tensor propagation, not
eager execution -- and asserts the output width for all three arms.
"""
import os
import sys

import torch

os.environ.setdefault("VLLM_TARGET_DEVICE", "tpu")
import vllm_torchtpu  # noqa: F401

import types  # noqa: E402

import vllm.distributed.parallel_state as _ps  # noqa: E402
_ps._TP = types.SimpleNamespace(rank_in_group=0, world_size=1,
                                device_group=None, cpu_group=None)
torch.set_default_dtype(torch.bfloat16)

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.layers.vllm import latent_proj_intra_chip as L  # noqa: E402

HID, LAT = 7168, 3584
checks = []


def arm(mode):
    envs.TPU_K3_FP8_LATENT_PROJ = mode == "fp8"
    envs.TPU_K3_MXFP4_LATENT_PROJ = mode == "mxfp4"
    out = {}
    for nm, (n_in, n_out) in (("down", (HID, LAT)), ("up", (LAT, HID))):
        layer = L.make_latent_projection(n_in, n_out, bias=False,
                                         quant_config=None,
                                         prefix=f"m.1.block_sparse_moe.{nm}")
        torch.manual_seed(0)
        torch.nn.init.normal_(layer.weight, std=0.02)
        layer.quant_method.process_weights_after_loading(layer)
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        mode_ = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
        with mode_:
            # A dynamic leading dim, as the compiled model sees it (`s72`).
            t = torch.empty(8, n_in, dtype=torch.bfloat16)
            torch._dynamo.mark_dynamic(t, 0)
            try:
                y, _ = layer(t)
                out[nm] = ("ok", tuple(y.shape)[-1])
            except Exception as e:  # noqa: BLE001
                out[nm] = ("raised", f"{type(e).__name__}: {str(e)[:90]}")
        # The elementwise consumer that actually failed on hardware: an
        # RMSNorm-style weight of the LOGICAL output width.
        if out[nm][0] == "ok":
            with mode_:
                t = torch.empty(8, n_in, dtype=torch.bfloat16)
                torch._dynamo.mark_dynamic(t, 0)
                y, _ = layer(t)
                try:
                    (y * torch.empty(n_out, dtype=torch.bfloat16)).shape
                except Exception as e:  # noqa: BLE001
                    out[nm] = ("consumer-raised",
                               f"{type(e).__name__}: {str(e)[:90]}")
    return out


res = {m: arm(m) for m in ("off", "fp8", "mxfp4")}
print(f"{'arm':<8}{'proj':<7}{'result':<18}{'logical n_out'}")
for m, d in res.items():
    for nm, (st, val) in d.items():
        want = LAT if nm == "down" else HID
        print(f"{m:<8}{nm:<7}{st:<18}{val}   want {want}")
        checks.append((st == "ok" and val == want,
                       f"{m:<6} {nm:<5} forward width == {want}", val))

print()
ok = True
for good, name, got in checks:
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {name}  got={got}")
print(f"\nFORWARD GATE {'PASS' if ok else 'FAIL'}"
      " -- every arm's forward must produce the LOGICAL output width under a"
      " symbolic batch")
sys.exit(0 if ok else 1)
