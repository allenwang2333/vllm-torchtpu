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
"""Deviceless tpu7x compile of the whole-step kernel at DSv4-Pro size.

    python -m tests.kernels.deepseek_v4_megakernel.step_aot --T 64 --B 9

Lowers the per-rank step under shard_map on a 2x2x2 (16-core) topology and
reports VMEM / code size, or the Mosaic error.
"""

import argparse
import os

os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import topologies  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_config as dc  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_kernel as SK  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--q", type=int, default=7)
    p.add_argument("--B", type=int, default=9)
    p.add_argument("--banks", type=int, default=6)
    p.add_argument("--bank-mb", type=float, default=5.0)
    p.add_argument("--layers", type=int, default=0)
    p.add_argument("--topology", default="tpu7x:2x2x2")
    p.add_argument("--moe", choices=("stub", "real"), default="stub")
    p.add_argument("--moe-dedicated", action="store_true")
    p.add_argument("--attn", choices=("stub", "real"), default="stub")
    p.add_argument("--max-ctx", type=int, default=131072)
    p.add_argument("--skip", default="", help="StepKernelConfig.skip, '+' separated")
    a = p.parse_args()
    topo = topologies.get_topology_desc(platform="tpu", topology_name=a.topology)
    devs = topo.devices
    R = len(devs)
    kw = dict(tp=R, rows=a.T, n_banks=a.banks, bank_bytes=int(a.bank_mb * 2**20))
    if a.layers:
        kw["ratios"] = dc.DSV4_PRO_RATIOS[: a.layers]
    cfg = dc.DenseChainConfig(**kw)
    L = cfg.n_layers
    aux = tuple(x for x in (L - 3, L - 2, L - 1) if x >= 0)
    scfg = SK.StepKernelConfig(
        dense=cfg, q=a.q, B=a.B, aux_layers=aux, moe_impl=a.moe,
        moe_leased=not a.moe_dedicated, attn_impl=a.attn, max_ctx=a.max_ctx,
        skip=tuple(x for x in a.skip.split("+") if x),
    )  # fmt: skip
    mesh = jax.sharding.Mesh(devs, ("d",))
    sh = NamedSharding(mesh, P("d"))
    M = scfg.M
    step = SK.build(scfg, axis="d")

    names = cfg.stream_names

    at = None
    if a.attn == "real":
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import attn_phase_synth as S

        seqs = [a.max_ctx - 100] * a.B
        kinds = S.DSV4_KINDS[:L]

        def arrays_only():
            d = S.synth_rank(jax.random.key(0), kinds, seqs, a.max_ctx, served_dtypes=True)
            return dict(arrays=d["arrays"], bt_flat=d["bt_flat"], weights=d["weights"])

        with jax.default_device(jax.devices("cpu")[0]):
            small = S.synth_rank(jax.random.key(0), kinds, [1000] * a.B, 4096)
        at = jax.eval_shape(arrays_only)
        at["cache_index"], at["cache_layers"] = small["cache_index"], small["cache_layers"]
        at["bt_off"], at["bt_w"] = small["bt_off"], small["bt_w"]
    n_moe = 0

    def per_rank(ids, pos, seq, embed, cs, smalls, *rest):
        tl = rest[: len(names)]
        mo = rest[len(names) : len(names) + n_moe]
        ex = rest[len(names) + n_moe :]
        moe_w = None
        if a.moe == "real":
            moe_w = dict(tid2eid=[mo[0][0], mo[1][0], mo[2][0]], bias=mo[3][0],
                         w13=list(mo[4::4]), w13_scale=list(mo[5::4]),
                         w2=list(mo[6::4]), w2_scale=list(mo[7::4]))  # fmt: skip
        attn = None
        if at is not None:
            n_arr = len(at["arrays"])
            attn = dict(
                arrays=list(ex[:n_arr]), cache_index=at_meta["cache_index"],
                cache_layers=at_meta["cache_layers"], bt_flat=ex[n_arr][0],
                bt_off=at_meta["bt_off"], bt_w=at_meta["bt_w"],
                weights={k: v[0] for k, v in zip(at["weights"], ex[n_arr + 1 :])},
            )  # fmt: skip
        outs = step(ids[0], pos[0], seq[0], embed[0], cs[0], smalls,
                    dict(zip(names, tl)), moe_w, attn,
                    eblk=jax.lax.axis_index("d"))  # fmt: skip
        return outs[0][None], outs[1][None]

    def sds(shape, dtype):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=sh)

    tiles = []
    for n, s in cfg.streams.items():
        nl = max(1, len(cfg.stream_layers(n)))
        tiles.append(
            sds((R * nl, s.n_tiles, s.tile_rows, s.tile_cols), jnp.dtype(s.dtype))
        )
    args = [
        sds((R, M), jnp.int32), sds((R, M), jnp.int32), sds((R, a.B), jnp.int32),
        sds((R, 129280, cfg.hidden), jnp.bfloat16), sds((R, 524288, 128), jnp.float32),
        sds((R * L, cfg.smalls_rows, 128), jnp.float32), *tiles,
    ]  # fmt: skip
    if a.moe == "real":
        E, H, inter = cfg.n_experts, cfg.hidden, cfg.moe_inter
        args += [sds((R, 129280, 6), jnp.int32)] * 3 + [sds((R, L, E), jnp.float32)]
        for _ in range(L):
            args += [
                sds((E, H // 8, 2 * inter), jnp.uint32),
                sds((E, 16, 2 * inter), jnp.float32),
                sds((E, inter // 8, H), jnp.uint32),
                sds((E, 8, H), jnp.float32),
            ]
    n_moe = len(args) - 6 - len(names)
    if at is not None:
        import numpy as np

        at_meta = dict(cache_index=at["cache_index"], cache_layers=at["cache_layers"],
                       bt_off={k: np.zeros(L, np.int32) for k in at["bt_off"]},
                       bt_w={k: np.ones(L, np.int32) for k in at["bt_w"]})  # fmt: skip
        for x in at["arrays"]:
            args.append(sds((R * x.shape[0],) + x.shape[1:], x.dtype))
        args.append(sds((R,) + at["bt_flat"].shape, at["bt_flat"].dtype))
        for x in at["weights"].values():
            args.append(sds((R,) + x.shape, x.dtype))
    f = jax.jit(jax.shard_map(per_rank, mesh=mesh, in_specs=(P("d"),) * len(args),
                              out_specs=(P("d"), P("d")), check_vma=False))  # fmt: skip
    c = f.lower(*args).compile()
    ma = c.memory_analysis()
    print(
        f"RESULT step compiled R={R} T={a.T} B={a.B} banks={a.banks} code_bytes="
        f"{ma.generated_code_size_in_bytes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
