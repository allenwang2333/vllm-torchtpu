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
"""Deviceless tpu7x compile of the DSpark draft step at DSv4-Pro size.

    python -m tests.kernels.deepseek_v4_megakernel.draft_aot --T 16 --B 4

Three SWA draft layers (real swa attention and MoE), K = 6 draft rows and 7
context rows per request; reports VMEM / code size, or the Mosaic error.
"""

import argparse
import os

os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import topologies  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_config as dc  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_kernel as SK  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--ctx-q", type=int, default=7)
    p.add_argument("--banks", type=int, default=4)
    p.add_argument("--max-ctx", type=int, default=1 << 20)
    p.add_argument("--topology", default="tpu7x:2x2x2")
    a = p.parse_args()
    devs = topologies.get_topology_desc(
        platform="tpu", topology_name=a.topology
    ).devices
    R, L = len(devs), 3
    cfg = dc.DenseChainConfig(tp=R, rows=a.T, n_banks=a.banks, ratios=(1,) * L)
    scfg = SK.StepKernelConfig(dense=cfg, q=a.K, B=a.B, moe_impl="real",
                               attn_impl="real", max_ctx=a.max_ctx, ctx_q=a.ctx_q,
                               num_hash_layers=0, cache_inplace=True)  # fmt: skip
    mesh = jax.sharding.Mesh(devs, ("d",))
    sh = NamedSharding(mesh, P("d"))
    step = SK.build(scfg, axis="d")
    names = cfg.stream_names
    E, H, inter, M, T = cfg.n_experts, cfg.hidden, cfg.moe_inter, scfg.M, a.T
    pages = 4096  # SWA pool pages per layer (shape only)
    wp = a.max_ctx // 128
    roles = SK.META_ROLES
    bt_len = 3 * a.B * (wp + 3) + 1024

    def sds(shape, dtype):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=sh)

    tiles = [sds((R * L, s.n_tiles, s.tile_rows, s.tile_cols), jnp.dtype(s.dtype))
             for s in cfg.streams.values()]  # fmt: skip
    moe = [sds((R, L, E), jnp.float32)]
    for _ in range(L):
        moe += [sds((E, H // 8, 2 * inter), jnp.uint32),
                sds((E, 16, 2 * inter), jnp.float32),
                sds((E, inter // 8, H), jnp.uint32), sds((E, 8, H), jnp.float32)]  # fmt: skip
    arrs = [sds((R * pages, 256, 128), jnp.int32) for _ in range(L)]
    wnames = SK.SP.AttnWeights._fields
    wts = [sds((R, L, 128) if k == "attn_sink" else (R, 8, 128), jnp.float32)
           for k in wnames]  # fmt: skip
    base = [sds((R, M), jnp.int32), sds((R, M), jnp.int32), sds((R, a.B), jnp.int32),
            sds((R, 129280, H), jnp.bfloat16), sds((R, 524288, 128), jnp.float32),
            sds((R * L, cfg.smalls_rows, 128), jnp.float32),
            sds((R, T, H), jnp.bfloat16), sds((R, T), jnp.int32)]  # fmt: skip
    args = [*base, *tiles, *moe, *arrs, sds((R, bt_len), jnp.int32), *wts]
    nt, nm = len(tiles), len(moe)

    def per_rank(ids, pos, seq, embed, cs, smalls, mx, cpos, *rest):
        tl, mo = rest[:nt], rest[nt : nt + nm]
        at = rest[nt + nm :]
        moe_w = dict(tid2eid=[], bias=mo[0][0], w13=list(mo[1::4]),
                     w13_scale=list(mo[2::4]), w2=list(mo[3::4]),
                     w2_scale=list(mo[4::4]))  # fmt: skip
        attn = dict(
            arrays=list(at[:L]),
            cache_index={r: (list(range(L)) if r == "swa" else []) for r in SK.ROLES},
            cache_layers={r: (list(range(L)) if r == "swa" else []) for r in SK.ROLES},
            bt_flat=at[L][0],
            bt_off={r: np.zeros(L, np.int32) for r in roles},
            bt_w={r: np.ones(L, np.int32) for r in roles},
            weights={k: v[0] for k, v in zip(wnames, at[L + 1 :])},
        )  # fmt: skip
        outs = step(ids[0], pos[0], seq[0], embed[0], cs[0], smalls,
                    dict(zip(names, tl)), moe_w, attn, eblk=jax.lax.axis_index("d"),
                    ctx=dict(main_x=mx[0], positions=cpos[0]))  # fmt: skip
        return outs[0][None]

    f = jax.jit(jax.shard_map(per_rank, mesh=mesh, in_specs=(P("d"),) * len(args),
                              out_specs=P("d"), check_vma=False))  # fmt: skip
    ma = f.lower(*args).compile().memory_analysis()
    print(f"RESULT draft step compiled R={R} T={T} B={a.B} K={a.K} max_ctx="
          f"{a.max_ctx} code_bytes={ma.generated_code_size_in_bytes}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
