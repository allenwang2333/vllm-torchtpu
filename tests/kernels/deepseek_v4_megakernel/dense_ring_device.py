# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""On-device check and timing of the dense ring kernel vs the XLA incumbent.

    python -m tests.kernels.deepseek_v4_megakernel.dense_ring_device --tp 16

Runs, at DSv4-Pro size with random weights: the stock-op chain (stock Pallas
mHC and wo_a kernels + XLA fp8 dots, one jit; the incumbent), the kernel-order
jnp chain, and the ring kernel. Prints RESULT lines with relative errors and
median step times against the HBM bytes floor.
"""

import argparse
import json
import time

import jax
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_config as dc
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_reference as dr
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_ring
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_weights as dw


def _rel(a, b) -> float:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _time(fn, args, iters: int, warmup: int = 3) -> dict:
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1e6)
    ts = np.asarray(ts)
    return {
        "median_us": float(np.median(ts)),
        "p10_us": float(np.percentile(ts, 10)),
        "p90_us": float(np.percentile(ts, 90)),
        "n": iters,
    }


def _compare(cfg, out, ref) -> dict:
    (x, res, post, comb, pf, pidx), ((rx, rres, rp, rc), probes) = out, ref
    r = {"x": _rel(x, rx), "res": _rel(res, rres), "post": _rel(post, rp),
         "comb": _rel(comb, rc)}  # fmt: skip
    lay = cfg.probe_layout
    for n in ("qkv", "xproj", "gate"):
        o = lay[n][0]
        r[n] = max(
            _rel(pf[li, :, o : o + p[n].shape[1]], p[n]) for li, p in enumerate(probes)
        )
    r["idx_q"] = max(
        (_rel(pidx[li], p["idx_q"]) for li, p in enumerate(probes)
         if p["idx_q"] is not None),
        default=0.0,
    )  # fmt: skip
    r["layer0_qkv"] = _rel(pf[0, :, : cfg.qkv_cols], probes[0]["qkv"])
    return r


def attribution(cfg, pk, inp, tag, iters):
    """Compute-bound time (no weight DMA) with parts of the chain left out,
    and the ring alone (weight DMA on, every part left out)."""
    all_parts = frozenset(["sinkhorn", "mhc_pass", "quant", "epilogue", "norm",
                           "dot", "hc_dot", "stub", "swiglu"])  # fmt: skip
    for keep in ([], ["dot"], ["mhc_pass"], ["dot", "mhc_pass"], ["sinkhorn"]):
        f = dense_ring.Faults(skip=all_parts - frozenset(keep))
        ring_only = jax.jit(dense_ring.build(cfg, faults=f))
        t = _time(ring_only, (pk["tiles"], pk["smalls"], inp), iters)
        print("RESULT ring_with " + json.dumps({**tag, "keep": keep, **t}), flush=True)
    parts = ["sinkhorn", "mhc_pass", "quant", "epilogue", "norm"]
    extra = ["dot", "hc_dot", "stub", "swiglu", "wo_a"]
    for skip in [[], *([p] for p in parts + extra), parts + extra]:
        f = dense_ring.Faults(no_dma=True, skip=frozenset(skip))
        nodma = jax.jit(dense_ring.build(cfg, faults=f))
        t = _time(nodma, (pk["tiles"], pk["smalls"], inp), iters)
        print("RESULT compute_only " + json.dumps({**tag, "skip": skip, **t}),
              flush=True)  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=16)
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--layers", type=int, default=0, help="0 = all 61")
    p.add_argument("--banks", type=int, default=10)
    p.add_argument("--bank-mb", type=float, default=5.0)
    p.add_argument("--timing-only", action="store_true")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--skip-xla", action="store_true")
    p.add_argument("--attribution-only", action="store_true")
    a = p.parse_args()
    kw = dict(tp=a.tp, rows=a.rows, n_banks=a.banks, bank_bytes=int(a.bank_mb * 2**20))
    if a.layers:
        kw["ratios"] = dc.DSV4_PRO_RATIOS[: a.layers]
    cfg = dc.DenseChainConfig(**kw)
    print(dc.describe(cfg), flush=True)
    tag = {"tp": a.tp, "rows": a.rows, "layers": cfg.n_layers, "banks": a.banks,
           "bank_mb": a.bank_mb}  # fmt: skip
    floor = cfg.floor_us()

    w = jax.block_until_ready(jax.jit(dw.make_weights, static_argnums=(0,))(cfg))
    inp = dw.make_inputs(cfg)
    if a.attribution_only or a.timing_only:
        pk = jax.jit(dw.pack_weights, static_argnums=(1,))(w, cfg)
        del w
        if a.timing_only:
            run = jax.jit(dense_ring.build(cfg, probes=False))
            t = _time(run, (pk["tiles"], pk["smalls"], inp), a.iters)
            print("RESULT kernel " + json.dumps({**tag, "probes": False, **t,
                  "floor_us": floor, "x_floor": t["median_us"] / floor}),
                  flush=True)  # fmt: skip
            return
        attribution(cfg, pk, inp, tag, a.iters)
        return
    chain = jax.jit(dr.dense_chain, static_argnums=(2, 3))
    ref_stock = jax.block_until_ready(chain(w, inp, cfg, "stock_kernels"))
    ref_korder = jax.block_until_ready(chain(w, inp, cfg, "kernel"))
    sx, sres = ref_stock[0][0], ref_stock[0][1]
    kx, kres = ref_korder[0][0], ref_korder[0][1]
    band = {"x": _rel(kx, sx), "res": _rel(kres, sres),
            "post": _rel(ref_korder[0][2], ref_stock[0][2]),
            "comb": _rel(ref_korder[0][3], ref_stock[0][3])}  # fmt: skip
    print("RESULT band " + json.dumps({**tag, **band}), flush=True)
    qkv_s = [p["qkv"] for p in ref_stock[1]]
    qkv_k = [p["qkv"] for p in ref_korder[1]]
    band_layers = [_rel(k, s) for k, s in zip(qkv_k, qkv_s)]
    if not a.skip_xla:
        t = _time(lambda w, i: chain(w, i, cfg, "stock_kernels"), (w, inp), a.iters)
        print("RESULT xla " + json.dumps({**tag, **t, "floor_us": floor,
              "x_floor": t["median_us"] / floor}), flush=True)  # fmt: skip

    pk = jax.block_until_ready(jax.jit(dw.pack_weights, static_argnums=(1,))(w, cfg))
    del w
    for probes in (True, False):
        run = jax.jit(dense_ring.build(cfg, probes=probes))
        out = jax.block_until_ready(run(pk["tiles"], pk["smalls"], inp))
        if probes:
            err_s = _compare(cfg, out, ref_stock)
            err_k = _compare(cfg, out, ref_korder)
            print("RESULT err_vs_stock " + json.dumps({**tag, **err_s}), flush=True)
            print("RESULT err_vs_korder " + json.dumps({**tag, **err_k}), flush=True)
            pf = np.asarray(out[4])
            per = {
                "korder_vs_stock": band_layers,
                "kernel_vs_stock": [
                    _rel(pf[li, :, : cfg.qkv_cols], s) for li, s in enumerate(qkv_s)
                ],
                "kernel_vs_korder": [
                    _rel(pf[li, :, : cfg.qkv_cols], k) for li, k in enumerate(qkv_k)
                ],
            }
            print("RESULT qkv_per_layer " + json.dumps({**tag, **per}), flush=True)
        t = _time(run, (pk["tiles"], pk["smalls"], inp), a.iters)
        print("RESULT kernel " + json.dumps({**tag, "probes": probes, **t,
              "floor_us": floor, "x_floor": t["median_us"] / floor}),
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
