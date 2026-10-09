"""Torch normalization helpers with FP32 accumulation."""

import torch


def rms_norm(x, weight, eps):
    x_f32 = x if x.dtype == torch.float32 else x.float()
    w_f32 = weight if weight.dtype == torch.float32 else weight.float()
    out = x_f32 * torch.rsqrt(x_f32.square().mean(-1, keepdim=True) + eps) * w_f32
    return out if x.dtype == torch.float32 else out.to(x.dtype)
