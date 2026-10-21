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
"""Cache copies in the compiled step, aliased vs in place (deviceless tpu7x).

    python -m tests.kernels.deepseek_v4_megakernel.step_inplace_hlo --layers 3

Compiles the per-rank whole step with the real attention phase (DSv4-Pro
widths, synthetic served caches) for a 2x2x2 topology twice -- caches donated
and aliased to outputs, and `cache_inplace` -- and counts the HLO copy
instructions whose shape is a cache array's. Prints one RESULT line per mode.
"""

import argparse
import os
import re
import sys

os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import topologies  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import (
    dense_config as dc,  # noqa: E402
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_kernel as SK  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_phase_synth as S  # noqa: E402

_DT = {"int32": "s32", "uint8": "u8", "float32": "f32", "bfloat16": "bf16"}


def compile_step(inplace: bool, layers: int, T: int, B: int, max_ctx: int, topo):
    devs = topo.devices
    R = len(devs)
    cfg = dc.DenseChainConfig(tp=R, rows=T, ratios=dc.DSV4_PRO_RATIOS[:layers],
                              n_banks=4, bank_bytes=5 << 20)  # fmt: skip
    L = cfg.n_layers
    scfg = SK.StepKernelConfig(dense=cfg, q=7, B=B, attn_impl="real", max_ctx=max_ctx,
                               cache_inplace=inplace)  # fmt: skip
    mesh = jax.sharding.Mesh(devs, ("d",))
    sh = NamedSharding(mesh, P("d"))
    step = SK.build(scfg, axis="d")
    names = cfg.stream_names
    kinds = S.DSV4_KINDS[:L]
    with jax.default_device(jax.devices("cpu")[0]):
        small = S.synth_rank(jax.random.key(0), kinds, [1000] * B, 4096)
    at = jax.eval_shape(lambda: {
        k: v for k, v in S.synth_rank(jax.random.key(0), kinds, [max_ctx - 100] * B,
                                      max_ctx, served_dtypes=True).items()
        if k in ("arrays", "bt_flat", "weights")
    })  # fmt: skip
    meta = dict(cache_index=small["cache_index"], cache_layers=small["cache_layers"],
                bt_off={k: np.zeros(L, np.int32) for k in small["bt_off"]},
                bt_w={k: np.ones(L, np.int32) for k in small["bt_w"]})  # fmt: skip
    n_arr = len(at["arrays"])

    def per_rank(ids, pos, seq, embed, cs, smalls, *rest):
        tl, ex = rest[: len(names)], rest[len(names) :]
        attn = dict(arrays=list(ex[:n_arr]), bt_flat=ex[n_arr][0],
                    weights={k: v[0] for k, v in zip(at["weights"], ex[n_arr + 1 :])},
                    **meta)  # fmt: skip
        outs = step(ids[0], pos[0], seq[0], embed[0], cs[0], smalls,
                    dict(zip(names, tl)), None, attn)  # fmt: skip
        return (outs[0][None], outs[1][None], *outs[5])

    def sds(shape, dtype):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=sh)

    M = scfg.M
    args = [sds((R, M), jnp.int32), sds((R, M), jnp.int32), sds((R, B), jnp.int32),
            sds((R, 512, cfg.hidden), jnp.bfloat16),
            sds((R, 4 * max_ctx, 64), jnp.float32),
            sds((R * L, cfg.smalls_rows, 128), jnp.float32)]  # fmt: skip
    for n, s in cfg.streams.items():
        nl = max(1, len(cfg.stream_layers(n)))
        args.append(
            sds((R * nl, s.n_tiles, s.tile_rows, s.tile_cols), jnp.dtype(s.dtype))
        )
    first_cache = len(args)
    for x in at["arrays"]:
        args.append(sds((R * x.shape[0],) + x.shape[1:], x.dtype))
    args.append(sds((R,) + at["bt_flat"].shape, at["bt_flat"].dtype))
    for x in at["weights"].values():
        args.append(sds((R,) + x.shape, x.dtype))
    n_out = 2 + (0 if inplace else n_arr)
    donate = () if inplace else tuple(range(first_cache, first_cache + n_arr))
    f = jax.jit(jax.shard_map(per_rank, mesh=mesh, in_specs=(P("d"),) * len(args),
                              out_specs=(P("d"),) * n_out, check_vma=False),
                donate_argnums=donate)  # fmt: skip
    hlo = f.lower(*args).compile().as_text()
    shapes = {
        f"{_DT[str(x.dtype)]}[{','.join(map(str, x.shape))}]" for x in at["arrays"]
    }
    copies = [ln for ln in hlo.splitlines()
              if re.search(r"= \S+ copy(-start)?\(", ln)
              and any(sh_ in ln.split("=")[1] for sh_ in shapes)]  # fmt: skip
    cache_bytes = sum(int(np.prod(x.shape)) * jnp.dtype(x.dtype).itemsize
                      for x in at["arrays"])  # fmt: skip
    return len(copies), n_arr, cache_bytes, copies[:4]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--max-ctx", type=int, default=8192)
    p.add_argument("--topology", default="tpu7x:2x2x2")
    a = p.parse_args()
    topo = topologies.get_topology_desc(platform="tpu", topology_name=a.topology)
    for inplace in (False, True):
        n, n_arr, nbytes, sample = compile_step(
            inplace, a.layers, a.T, a.B, a.max_ctx, topo
        )
        mode = "inplace" if inplace else "aliased"
        print(f"RESULT {mode} cache_copies={n} cache_arrays={n_arr} "
              f"cache_bytes_per_rank={nbytes}", flush=True)  # fmt: skip
        for ln in sample:
            print("  e.g.", ln.strip()[:160])


if __name__ == "__main__":
    main()
