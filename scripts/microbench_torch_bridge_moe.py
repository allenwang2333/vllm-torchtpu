"""Torch-bridge fused_moe microbench: goes through pallas.jax_op +
torch.custom_op + torch_tpu compile path (as production does), not just
jax.jit. If this shows a libtpu regression while the JAX-direct
scripts/microbench_fused_moe.py does not, the regression lives in the
bridge / dispatch layer.
"""

import argparse
import time

import torch
import torch_tpu  # noqa: F401 -- registers "tpu" device

from vllm_torchtpu.layers.vllm.fused_moe import fused_moe_gmm


def _make_inputs(num_tokens, hidden=4096, intermediate=1024,
                 num_local=64, topk=10, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    hs = (torch.randn(num_tokens, hidden, generator=g) / 10).to(torch.bfloat16)
    w1 = (torch.randn(num_local, hidden, intermediate * 2, generator=g)
          / 10).to(torch.bfloat16)
    w2 = (torch.randn(num_local, intermediate, hidden, generator=g)
          / 10).to(torch.bfloat16)
    exps = num_local * 8
    ids = torch.randint(0, exps, (num_tokens, topk), generator=g,
                        dtype=torch.int32)
    wts = torch.rand(num_tokens, topk, generator=g).to(torch.bfloat16)
    return (hs.to("tpu"), w1.to("tpu"), w2.to("tpu"),
            ids.to("tpu"), wts.to("tpu"))


def _time_run(fn, args, iters=50, warmup=10):
    for _ in range(warmup):
        out = fn(*args)
    _ = out.cpu()  # sync
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    _ = out.cpu()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    experts_start = 64 * 3  # arbitrary shard
    for name, num_tokens in [("prefill_1024", 1024), ("decode_64", 64)]:
        hs, w1, w2, ids, wts = _make_inputs(num_tokens)
        def run(hs, w1, w2, ids, wts):
            return fused_moe_gmm(
                hs, w1, w2, None, None, None, None, wts, ids,
                experts_start=experts_start,
                topk=10, activation="silu")
        try:
            us = _time_run(run, (hs, w1, w2, ids, wts),
                           iters=args.iters, warmup=args.warmup)
            print(f"{name:<16} {us:>7.2f} µs")
        except Exception as exc:
            print(f"{name:<16} FAILED: {type(exc).__name__}: {str(exc)[:120]}")


if __name__ == "__main__":
    main()
