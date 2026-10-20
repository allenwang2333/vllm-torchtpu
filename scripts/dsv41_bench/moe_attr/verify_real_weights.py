#!/usr/bin/env python3
"""CPU check: real_weights.py's conversion equals the serving loader's, byte for byte.

For a few experts of one layer, runs the loader's own steps on CPU -- the
MXFP4 dequant (`dequantize_mxfp4_packed` and `_dequantize_mxfp4_packed_on_device`),
`VllmDeepseekV4Mxfp4MoEMethod._dequantize_and_pad` on a stand-in layer, then
`quantize_to_native_fp4_kmajor(pack=False)` -- and compares the fp4 weights and
f32 scales with `real_weights.expert_runtime`. Run with `dsv41-docker --cpu`.
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import real_weights as rw  # noqa: E402
import torch  # noqa: E402

from vllm_torchtpu.layers.adapter.quantization import mxfp4  # noqa: E402
from vllm_torchtpu.layers.core.quantization import dequantize_mxfp4_packed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--experts", default="0,200,383")
    args = ap.parse_args()
    experts = [int(e) for e in args.experts.split(",")]
    shard = Path(args.ckpt_dir) / rw.shard_for_layer(Path(args.index), args.layer)
    raw = rw.read_layer_raw(shard, args.layer, max(experts) + 1)
    sub = {k: v[experts] for k, v in raw.items()}
    inter = sub["w1"].shape[1]

    t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sub.items()}
    layer = SimpleNamespace(
        w13_weight=torch.cat([t["w1"], t["w3"]], dim=1),
        w13_weight_scale=torch.cat([t["s1"], t["s3"]], dim=1),
        w2_weight=t["w2"],
        w2_weight_scale=t["s2"],
    )
    # Both dequant paths the loader can take must agree with each other.
    for name in ("w2",):
        cpu = dequantize_mxfp4_packed(
            t[name], t["s2"], axis=2, out_dtype=torch.bfloat16
        )
        dev = mxfp4._dequantize_mxfp4_packed_on_device(
            t[name], t["s2"], axis=2, out_dtype=torch.bfloat16
        )
        assert torch.equal(cpu.view(torch.int16), dev.view(torch.int16))
    method = SimpleNamespace(intermediate_size=inter)
    w13p, w2p, _, _, orig = mxfp4.VllmDeepseekV4Mxfp4MoEMethod._dequantize_and_pad(
        method, layer
    )
    assert orig == inter
    block = mxfp4.REQUANTIZED_BLOCK_SIZE
    ok = True
    for j, e in enumerate(experts):
        ref13, refs13 = rw.quantize_to_native_fp4_kmajor(
            jnp.asarray(w13p[j].float().numpy()).astype(jnp.bfloat16), block, pack=False
        )
        ref2, refs2 = rw.quantize_to_native_fp4_kmajor(
            jnp.asarray(w2p[j].float().numpy()).astype(jnp.bfloat16),
            min(block, inter),
            pack=False,
        )
        got = rw.expert_runtime(
            *(jnp.asarray(sub[k][j]) for k in ("w1", "w3", "w2", "s1", "s3", "s2")),
            inter_padded=w2p.shape[1],
        )
        checks = {
            "w13": np.array_equal(
                np.asarray(got[0]).view(np.uint8), np.asarray(ref13).view(np.uint8)
            ),
            "w2": np.array_equal(
                np.asarray(got[1]).view(np.uint8), np.asarray(ref2).view(np.uint8)
            ),
            "s13": np.array_equal(np.asarray(got[2]), np.asarray(refs13)[:, 0, :]),
            "s2": np.array_equal(np.asarray(got[3]), np.asarray(refs2)[:, 0, :]),
        }
        stats = dict(
            w13_shape=tuple(got[0].shape),
            w2_shape=tuple(got[1].shape),
            s13_range=(
                float(np.asarray(got[2]).min()),
                float(np.asarray(got[2]).max()),
            ),
            w13_zero_frac=float((np.asarray(got[0]).astype(np.float32) == 0).mean()),
        )
        print(f"layer {args.layer} expert {e}: {checks} {stats}")
        ok &= all(checks.values())
    print("VERIFY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
