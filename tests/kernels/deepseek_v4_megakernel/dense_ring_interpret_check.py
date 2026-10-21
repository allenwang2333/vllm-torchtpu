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
"""CPU-interpret check of the dense ring kernel against the kernel-order
reference; run as a script so XLA_FLAGS apply before JAX initializes.

    XLA_FLAGS=--xla_allow_excess_precision=false JAX_PLATFORMS=cpu \\
        python -m tests.kernels.deepseek_v4_megakernel.dense_ring_interpret_check \\
        --tp 1 [--fault swap|early] [--banks N]

Prints one JSON line: max |kernel - reference| per output, relative errors
of the kernel and of the stock-order reference against the kernel-order
reference, and whether the interpreter's race detector fired.
"""

import argparse
import json
import os

import jax
import numpy as np
from jax._src.pallas.mosaic.interpret import interpret_pallas_call
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_config as dc
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_reference as dr
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_ring
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_weights as dw


def _maxdiff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))))


def _rel(a, b) -> float:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--banks", type=int, default=4)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--fault", choices=("none", "swap", "early"), default="none")
    a = p.parse_args()
    assert "--xla_allow_excess_precision=false" in os.environ.get("XLA_FLAGS", "")

    cfg = dc.small_config(tp=a.tp, rows=a.rows, n_banks=a.banks, n_heads=a.heads)
    w = dw.make_weights(cfg)
    inp = dw.make_inputs(cfg)
    chain = jax.jit(dr.dense_chain, static_argnums=(2, 3))
    (rx, rres, rp, rc), probes = chain(w, inp, cfg, "kernel")
    (sx, sres, _, _), _ = chain(w, inp, cfg, "stock")

    faults = dense_ring.Faults()
    if a.fault == "swap":  # two tiles of one stream trade places
        off = cfg.stream_offsets[dc.HCA]["wq_b"]
        assert cfg.streams["wq_b"].n_tiles >= 3
        faults = dense_ring.Faults(swap_tiles=(off + 1, off + 2))
    elif a.fault == "early":
        faults = dense_ring.Faults(early_refill=True)
    params = pltpu.InterpretParams(dma_execution_mode="eager", detect_races=True)
    pk = dw.pack_weights(w, cfg)
    run = dense_ring.build(cfg, interpret=params, faults=faults)
    x, res, post, comb, pf, pidx = jax.block_until_ready(
        run(pk["tiles"], pk["smalls"], inp)
    )

    out = {
        "x": _maxdiff(x, rx),
        "res": _maxdiff(res, rres),
        "post": _maxdiff(post, rp),
        "comb": _maxdiff(comb, rc),
        "x_rel": _rel(x, rx),
        "res_rel": _rel(res, rres),
        # the same formulas summed in the stock order: the reassociation band
        "band_x_rel": _rel(sx, rx),
        "band_res_rel": _rel(sres, rres),
    }
    layout = cfg.probe_layout
    for name in ("qkv", "xproj", "gate"):
        o = layout[name][0]
        out["probe_" + name] = max(
            _maxdiff(pf[li, :, o : o + pr[name].shape[1]], pr[name])
            for li, pr in enumerate(probes)
        )
    out["layer0_probes"] = max(
        _maxdiff(pf[0, :, layout[n][0] : layout[n][0] + probes[0][n].shape[1]],
                 probes[0][n])
        for n in ("qkv", "xproj")
    )  # fmt: skip
    out["probe_idx_q"] = max(
        (_maxdiff(pidx[li], pr["idx_q"]) for li, pr in enumerate(probes)
         if pr["idx_q"] is not None),
        default=0.0,
    )  # fmt: skip
    races = interpret_pallas_call.races
    out["races"] = bool(races is not None and races.races_found)
    out["n_tiles"] = cfg.n_tiles_total
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
