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
"""TPU v7 microbenchmark for Qwen3.5-397B TP-Megakernel FUSE_ES=0 vs FUSE_ES=1."""

from __future__ import annotations

import dataclasses
import functools
import gc
import json
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from vllm_torchtpu.kernels.qwen35_megakernel import megakernel as K
from vllm_torchtpu.kernels.qwen35_megakernel import pool_io


def _alloc_sharded_weights(mesh: Mesh, c: K.MKConfig, tiles: K.Tiles) -> list[jax.Array]:
    sharding = NamedSharding(mesh, P("tp"))
    shapes = K.weight_shapes(c, tiles)
    R = c.R

    @functools.partial(jax.jit, static_argnums=(1,), out_shardings=sharding)
    def _rand_fp8(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        return jax.random.randint(key, shape, 0, 64, dtype=jnp.uint8).view(K.FP8)

    @functools.partial(jax.jit, static_argnums=(1, 2), out_shardings=sharding)
    def _rand_norm(key: jax.Array, shape: tuple[int, ...], dt: Any) -> jax.Array:
        return (jax.random.normal(key, shape, dtype=jnp.float32) * 0.02).astype(dt)

    @functools.partial(jax.jit, static_argnums=(1, 2), out_shardings=sharding)
    def _rand_es(key: jax.Array, shape: tuple[int, ...], dt: Any) -> jax.Array:
        return (jnp.abs(jax.random.normal(key, shape, dtype=jnp.float32)) * 0.01 + 0.01).astype(dt)

    w_leaves: list[jax.Array] = []
    for idx, k in enumerate(K.W_NAMES):
        if not c.stack_e and k in ("e13", "e2"):
            specs = shapes[k]
            for l_idx, spec in enumerate(specs):
                sh, dt = spec.shape, spec.dtype
                leaf_sh = (R * sh[0], *sh[1:])
                key = jax.random.PRNGKey(1000 * idx + l_idx)
                if dt == K.FP8:
                    arr = _rand_fp8(key, leaf_sh)
                else:
                    arr = _rand_norm(key, leaf_sh, dt)
                w_leaves.append(arr)
        else:
            sh, dt = shapes[k]
            global_sh = (R * sh[0], *sh[1:])
            key = jax.random.PRNGKey(1000 * idx)
            if dt == K.FP8:
                arr = _rand_fp8(key, global_sh)
            elif k == "es":
                arr = _rand_es(key, global_sh, dt)
            else:
                arr = _rand_norm(key, global_sh, dt)
            w_leaves.append(arr)
    return w_leaves


def _alloc_sharded_pools(mesh: Mesh, c: K.MKConfig) -> list[jax.Array]:
    sharding = NamedSharding(mesh, P("tp"))
    R = c.R
    local_sh = (c.num_pages, 2, 64, 4, c.page_size)

    @functools.partial(jax.jit, in_shardings=(sharding,), out_shardings=sharding)
    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P("tp"),),
        out_specs=P("tp"),
        check_vma=False,
    )
    def _zero_pool(seed: jax.Array) -> jax.Array:
        return (seed + jnp.zeros(local_sh, dtype=jnp.uint8)).view(K.FP8)

    pools: list[jax.Array] = []
    for i in range(c.NATT):
        seed_np = np.zeros((R, 1, 1, 1, 1), dtype=np.uint8)
        seed_dev = jax.device_put(seed_np, sharding)
        pools.append(_zero_pool(seed_dev))
    return pools


def _bench_mode(
    mesh: Mesh,
    fuse_es: bool,
    buckets: list[tuple[int, int]],
    num_warmup: int = 5,
    num_iters: int = 30,
) -> dict[tuple[int, int], dict[str, float]]:
    R = 8
    sharding = NamedSharding(mesh, P("tp"))
    c_template = dataclasses.replace(
        K.FULL,
        R=R,
        B=16,
        T=4,
        moe_tp=True,
        fuse_es=fuse_es,
        stack_e=False,
        paged_kv=True,
        fused_pool_io=True,
        num_pages=4096,
        page_size=128,
        max_pages_per_seq=64,
        ctx=8192,
        ckpt_k=4,
    )
    tiles_template = K.default_tiles(c_template)
    print(f"\n[bench] Allocating weights for fuse_es={fuse_es} (ebank={tiles_template.ebank // 1024} KiB)...")
    w_leaves = _alloc_sharded_weights(mesh, c_template, tiles_template)
    raw_pools = _alloc_sharded_pools(mesh, c_template)
    jax.block_until_ready(w_leaves[-1])
    jax.block_until_ready(raw_pools[-1])

    results: dict[tuple[int, int], dict[str, float]] = {}
    for T, B in buckets:
        c = dataclasses.replace(
            c_template,
            B=B,
            T=T,
            ckpt_k=T,
            fused_pool_io=True,
        )
        tiles = K.default_tiles(c)
        step_fn, in_specs, out_specs, donate_argnums = pool_io.make_flat_pooled_step(
            mesh, c, tiles, fused_pool_io=True, out_rows=B
        )
        in_shardings = tuple(NamedSharding(mesh, s) for s in in_specs)
        out_shardings = tuple(NamedSharding(mesh, s) for s in out_specs)
        jit_step = jax.jit(
            step_fn,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            donate_argnums=donate_argnums,
        )

        ns = B // T
        pos_np = np.zeros((R * B,), dtype=np.int32)
        for r in range(R):
            for s in range(ns):
                for t in range(T):
                    pos_np[r * B + s * T + t] = 512 + t
        pos = jax.device_put(pos_np, sharding)
        hid_np = np.full((R * c.M, c.H), 0.02, dtype=jnp.dtype(K.BF16))
        hid = jax.device_put(hid_np, sharding)
        pt_np = np.zeros((R * ns, c.max_pages_per_seq), dtype=np.int32)
        for r in range(R):
            for s in range(ns):
                for p in range(8):
                    pt_np[r * ns + s, p] = 1 + s * 8 + p
        pt = jax.device_put(pt_np, sharding)

        ckpt_np = np.zeros((R * 3, ns, T), dtype=np.int32)
        slot = 600
        for r in range(R):
            for sub in range(3):
                for s in range(ns):
                    for t in range(T):
                        ckpt_np[r * 3 + sub, s, t] = slot
                        slot = 600 + ((slot - 599) % 200)
        ckpt = jax.device_put(ckpt_np, sharding)
        roff = jax.device_put(np.zeros((R * 3, ns), dtype=np.int32), sharding)

        for _ in range(num_warmup):
            outs = jit_step(pos, hid, *w_leaves, *raw_pools, pt, ckpt, roff)
            outs[0].block_until_ready()
            raw_pools = list(outs[1:])

        times_us: list[float] = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            outs = jit_step(pos, hid, *w_leaves, *raw_pools, pt, ckpt, roff)
            outs[0].block_until_ready()
            t1 = time.perf_counter()
            raw_pools = list(outs[1:])
            times_us.append((t1 - t0) * 1e6)

        med_us = float(np.median(times_us))
        mean_us = float(np.mean(times_us))
        min_us = float(np.min(times_us))
        print(
            f"[bench] fuse_es={int(fuse_es)} T={T} B={B:2d}: "
            f"median={med_us:7.1f} us  mean={mean_us:7.1f} us  min={min_us:7.1f} us"
        )
        results[(T, B)] = {"median_us": med_us, "mean_us": mean_us, "min_us": min_us}

    del w_leaves, raw_pools
    gc.collect()
    return results


def main() -> None:
    devs = jax.devices("tpu")
    print(f"Found {len(devs)} TPU devices: {devs}")
    assert len(devs) >= 8, f"Expected >= 8 TPU devices on 2x2x1 tpu7x, got {len(devs)}"
    mesh = Mesh(np.array(devs[:8]), ("tp",))

    buckets = [
        (1, 4),
        (1, 8),
        (1, 16),
        (4, 4),
        (4, 8),
        (4, 12),
        (4, 16),
        (4, 20),
        (4, 24),
        (4, 32),
        (4, 40),
        (4, 48),
        (4, 56),
        (4, 64),
    ]

    res_base = _bench_mode(mesh, fuse_es=False, buckets=buckets)
    res_fuse = _bench_mode(mesh, fuse_es=True, buckets=buckets)

    print("\n=== Qwen3.5-397B TP-MK FUSE_ES Microbenchmark Summary (L=60, R=8) ===")
    print(
        "| T | B | Seq | FUSE_ES=0 med (us) | FUSE_ES=1 med (us) | Delta (us) | Speedup (%) |"
    )
    print("|---|---|-----|--------------------|--------------------|------------|-------------|")
    summary_rows = []
    for T, B in buckets:
        b0 = res_base[(T, B)]["median_us"]
        b1 = res_fuse[(T, B)]["median_us"]
        delta_us = b1 - b0
        speedup_pct = (b0 - b1) / b0 * 100.0
        print(
            f"| {T} | {B:2d} | {B // T:3d} | {b0:18.1f} | {b1:18.1f} | {delta_us:+10.1f} | {speedup_pct:+11.2f}% |"
        )
        summary_rows.append(
            {
                "T": T,
                "B": B,
                "seqs": B // T,
                "fuse_es_0_median_us": b0,
                "fuse_es_1_median_us": b1,
                "delta_us": delta_us,
                "speedup_pct": speedup_pct,
                "fuse_es_0_min_us": res_base[(T, B)]["min_us"],
                "fuse_es_1_min_us": res_fuse[(T, B)]["min_us"],
            }
        )

    out_dir = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mk_fuse_es_bench.json")
    with open(out_path, "w") as f:
        json.dump({"rows": summary_rows}, f, indent=2)
    print(f"\nSaved JSON summary to {out_path}")


if __name__ == "__main__":
    main()
