"""Round-trip weight error vs block size, for the accuracy/speed tradeoff.

Speed says block 256 (gmm_v2's dequantize-AFTER branch needs >= mxu_column_size
= 256). Accuracy pulls the other way, and nobody had quantified how hard.
Pure torch, no TPU. Reported, never asserted -- an eval decides.
"""
import torch

HID, LAT, FP4_MAX = 7168, 3584, 6.0
FP4 = torch.tensor([0.0, .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5,
                    -2., -3., -4., -6.])


def rt_fp4(w, block):
    n_in, n_out = w.shape
    b = w.reshape(n_in // block, block, n_out)
    s = b.abs().amax(1, keepdim=True) / FP4_MAX
    inv = torch.where(s == 0, torch.zeros_like(s), 1 / s)
    q = (b * inv).clamp(-FP4_MAX, FP4_MAX)
    i = (q.unsqueeze(-1) - FP4).abs().argmin(-1)
    return (FP4[i] * s - b).norm().item() / w.norm().item()


def rt_fp8(w, block):
    n_in, n_out = w.shape
    b = w.reshape(n_in // block, block, n_out)
    s = b.abs().amax(1, keepdim=True) / torch.finfo(torch.float8_e4m3fn).max
    inv = torch.where(s == 0, torch.zeros_like(s), 1 / s)
    q = (b * inv).to(torch.float8_e4m3fn).to(torch.float32)
    return (q * s - b).norm().item() / w.norm().item()


torch.manual_seed(0)
w = torch.randn(HID, LAT) * 0.02
print(f"weight [{HID},{LAT}] ~ N(0, 0.02)   relative Frobenius round-trip error")
print(f"{'block':>7}{'mxfp4':>10}{'fp8':>10}{'mxfp4 MB':>11}{'gmm_v2 branch':>16}")
for blk in (32, 64, 128, 256, 512, 1024):
    sc = (HID // blk) * LAT * 4 / 1e6
    br = "BEFORE (slow)" if blk < 256 else "after (fast)"
    print(f"{blk:>7}{rt_fp4(w, blk):>10.4f}{rt_fp8(w, blk):>10.4f}"
          f"{12.845 + sc:>11.2f}{br:>16}")
print("\nbf16 baseline weight = 51.38 MB, round-trip error 0 by construction.")
print("fp8 @128 (the arm on the pod) and mxfp4 @256 (K5) are the two rows to compare.")
