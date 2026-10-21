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
"""Device timing of the DSpark draft step (R=16, two hosts, random weights).

python -m tests.kernels.deepseek_v4_megakernel.draft_device --T 16 --owners 1
"""

import argparse
import json
import os
import statistics
import time

import jax

jax.distributed.initialize(os.environ["COORD"], int(os.environ["NPROC"]),
                           int(os.environ["PID"]))  # fmt: skip

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_config as dc  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_kernel as SK  # noqa: E402


def result(**kw):
    if jax.process_index() == 0:
        print("RESULT " + json.dumps(kw), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--owners", type=int, default=1)
    p.add_argument("--iters", type=int, default=11)
    p.add_argument("--attn", default="real")
    a = p.parse_args()
    devs = sorted(jax.devices(), key=lambda d: d.id)
    R, L, Q_CTX = len(devs), 3, 7
    mesh = Mesh(np.array(devs), ("d",))
    EP = NamedSharding(mesh, P("d"))
    cfg = dc.DenseChainConfig(tp=R, rows=a.T, n_banks=4, ratios=(1,) * L)
    scfg = SK.StepKernelConfig(dense=cfg, q=a.K, B=a.B, moe_impl="real",
                               attn_impl=a.attn, max_ctx=131072, ctx_q=Q_CTX,
                               num_hash_layers=0, cache_inplace=a.attn == "real")  # fmt: skip
    keys = iter(jax.random.split(jax.random.key(0), 1000))

    def gen(shape, dtype, kind):
        def f(k):
            if kind == "fp8":
                return jnp.clip(jax.random.normal(k, shape) * 64, -448, 448).astype(
                    dtype
                )
            if kind == "bits":
                return jax.random.bits(k, shape, jnp.uint32)
            if kind == "scale":
                return jax.random.uniform(k, shape, jnp.float32, 0.5, 1.5) * 1e-2
            if kind == "zeros":
                return jnp.zeros(shape, dtype)
            return (jax.random.normal(k, shape) * 0.02).astype(dtype)

        return jax.jit(f, out_shardings=EP)(next(keys))

    names = cfg.stream_names
    tiles = {n: gen((R * L, s.n_tiles, s.tile_rows, s.tile_cols), jnp.dtype(s.dtype),
                    "fp8" if s.dtype == dc.FP8 else "bf16")
             for n, s in cfg.streams.items()}  # fmt: skip
    smalls = gen((R * L, cfg.smalls_rows, 128), jnp.float32, "scale")
    H, E, inter, T, M = cfg.hidden, cfg.n_experts, cfg.moe_inter, a.T, scfg.M
    embed = gen((R * 129280, H), jnp.bfloat16, "bf16")
    ang = jnp.arange(1 << 17)[:, None] * 10000.0 ** (-jnp.arange(32) / 32.0)[None]
    cs = SK.SP.pack_cos_sin(jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], 1))
    cs = jax.device_put(jnp.tile(cs, (R, 1)), EP)
    moe = [gen((R * L, E), jnp.float32, "scale")]
    for _ in range(L):
        moe += [gen((E, H // 8, 2 * inter), jnp.uint32, "bits"),
                gen((E, 16, 2 * inter), jnp.float32, "scale"),
                gen((E, inter // 8, H), jnp.uint32, "bits"),
                gen((E, 8, H), jnp.float32, "scale")]  # fmt: skip
    pages = 64
    arrs = [gen((R * pages, 256, 128), jnp.int32, "zeros") for _ in range(L)]
    bt_len = 3 * a.B * (131072 // 128 + 3) + 1024
    bt = gen((R * bt_len,), jnp.int32, "zeros")
    wnames = SK.SP.AttnWeights._fields
    wts = [gen((R * L, 128) if k == "attn_sink" else (R * 8, 128), jnp.float32, "scale")
           for k in wnames]  # fmt: skip
    ids = np.zeros((R, M), np.int32)
    pos = np.zeros((R, M), np.int32)
    seq = np.zeros((R, a.B), np.int32)
    cpos = np.full((R, T), -1, np.int32)
    for r in range(a.owners):
        b0 = 120000 + 37 * r
        ids[r, : a.K] = np.arange(a.K) + 5
        pos[r, : a.K] = np.arange(b0, b0 + a.K)
        seq[r, 0] = b0 + a.K
        cpos[:, r * 8 : r * 8 + Q_CTX] = np.arange(b0 - Q_CTX, b0)
    put = lambda x: jax.make_array_from_callback(x.shape, EP, lambda i: x[i])  # noqa: E731
    mx = gen((R * T, H), jnp.bfloat16, "bf16")
    step = SK.build(scfg, axis="d")
    nt, nm, roles = len(names), len(moe), SK.META_ROLES

    def per_rank(ids, pos, seq, embed, cs, smalls, mx, cpos, bt, *rest):
        tl, mo, at = rest[:nt], rest[nt : nt + nm], rest[nt + nm :]
        moe_w = dict(tid2eid=[], bias=mo[0], w13=list(mo[1::4]), w13_scale=list(mo[2::4]),
                     w2=list(mo[3::4]), w2_scale=list(mo[4::4]))  # fmt: skip
        attn = None
        if a.attn == "real":
            attn = dict(
                arrays=list(at[:L]),
                cache_index={r: (list(range(L)) if r == "swa" else []) for r in SK.ROLES},
                cache_layers={r: (list(range(L)) if r == "swa" else []) for r in SK.ROLES},
                bt_flat=bt, bt_off={r: np.zeros(L, np.int32) for r in roles},
                bt_w={r: np.full(L, 131072 // 128 if r == "swa" else 3, np.int32)
                      for r in roles},
                weights=dict(zip(wnames, at[L:])),
            )  # fmt: skip
        outs = step(ids[0], pos[0], seq[0], embed, cs, smalls, dict(zip(names, tl)),
                    moe_w, attn, eblk=jax.lax.axis_index("d"),
                    ctx=dict(main_x=mx, positions=cpos[0]))  # fmt: skip
        return outs[0][None]

    args = [put(ids), put(pos), put(seq), embed, cs, smalls, mx, put(cpos), bt,
            *[tiles[n] for n in names], *moe, *arrs, *wts]  # fmt: skip
    f = jax.jit(jax.shard_map(per_rank, mesh=mesh, in_specs=(P("d"),) * len(args),
                              out_specs=P("d"), check_vma=False))  # fmt: skip
    t0 = time.time()
    jax.block_until_ready(f(*args))
    result(phase="compile+first", seconds=time.time() - t0)
    for _ in range(2):
        jax.block_until_ready(f(*args))
    ts = []
    for _ in range(a.iters):
        t1 = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append((time.perf_counter() - t1) * 1e3)
    jax.block_until_ready(f(*args))
    t1 = time.perf_counter()
    for _ in range(a.iters):
        o = f(*args)
    jax.block_until_ready(o)
    pipe = (time.perf_counter() - t1) * 1e3 / a.iters
    out0 = np.asarray(o.addressable_shards[0].data, np.float32)
    result(check="draft_step_time", T=T, B=a.B, owners=a.owners, attn=a.attn,
           median_ms=statistics.median(ts), pipe_ms=pipe,
           finite=bool(np.isfinite(out0).all()))  # fmt: skip


if __name__ == "__main__":
    main()
