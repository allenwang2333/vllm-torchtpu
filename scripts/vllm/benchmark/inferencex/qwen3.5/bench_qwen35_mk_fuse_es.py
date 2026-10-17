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
"""TPU v7 microbenchmark: Qwen3.5-397B TP-Megakernel expert-scale (es) layout ablations.

Modes (MKConfig fields), per listed expert and rank:
  sep           fuse_es=0                          3 DMAs, 1553 KiB (baseline)
  pad           es_pad=1 (bottom geometry)          3 DMAs, 1689 KiB: e13/e2 carry 32 unused rows
  bottom_reg    fuse_es=1, bottom, es_read=reg      2 DMAs, 1672 KiB: whole scale row as one value
  bottom_chunk  fuse_es=1, bottom, es_read=chunk    2 DMAs, 1672 KiB: per-chunk bitcast loads
  right         fuse_es=1, right                    2 DMAs, 1560 KiB: e2 scales in 128 dense cols
  pad_right     es_pad=1, right geometry            3 DMAs, 1577 KiB (optional)

Every mode gets bit-identical main weights and FP32 scales (the scale rows / columns are packed
from the same `es` array), so routing and outputs must match the first mode exactly; the script
checks this per bucket. Routing "same": every token identical (T=1 hits TOPK experts per layer,
T=4 at most 4 * TOPK, independent of B), as in the earlier fuse_es run. "diverse": per-token
random hidden states (identical on all ranks); with these random weights the rows' routing still
largely collapses. "forced": a forced-routing operand makes every row pick TOPK uniformly random
experts per layer, so the hit-expert count (and expert DMA traffic) grows with B as in serving.

Env: MK_ES_MODES, MK_ES_ROUTINGS, MK_ES_BUCKETS ("TxB,..."), MK_ES_WARMUP, MK_ES_ITERS,
MK_ES_PASSES (passes over the modes; odd passes run them in reverse order).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import gc
import json
import os
import time
import traceback
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from vllm_torchtpu.kernels.qwen35_megakernel import megakernel as K
from vllm_torchtpu.kernels.qwen35_megakernel import pool_io
from vllm_torchtpu.kernels.qwen35_megakernel.es_pack import (
    expert_dma_bytes,
    fuse_e2,
    fuse_e13,
)
from vllm_torchtpu.kernels.qwen35_megakernel.moe_block import MoEBlock

MODE_KW: dict[str, dict[str, Any]] = {
    "sep": {},
    "pad": {"es_pad": True},
    "bottom_reg": {"fuse_es": True, "es_read": "reg"},
    "bottom_chunk": {"fuse_es": True, "es_read": "chunk"},
    "right": {"fuse_es": True, "es_layout": "right"},
    "pad_right": {"es_pad": True, "es_layout": "right"},
    # Diagnostics reproducing the first FUSE_ES microbenchmark (j-b0b077f8): "late" loads each
    # expert's e2 weights after the next expert's gate|up in experts_piped (that build's order);
    # "rand" fills the scale rows with random FP8 bytes as that run did (outputs then differ).
    "bottom_reg_late": {"fuse_es": True, "es_read": "reg"},
    "bottom_reg_rand": {"fuse_es": True, "es_read": "reg"},
    "sep_late": {},
}
LATE_W2 = ("bottom_reg_late", "sep_late")
RAND_SCALES = ("bottom_reg_rand",)
R = 8


def _experts_piped_late(self, l, n_hit):
    """MoEBlock.experts_piped with expert i's e2 weights loaded inside down(), after expert
    i+1's gate|up (the order of the first FUSE_ES build)."""
    ctx = self.ctx
    scr, IR, T13, TE = ctx.scr, ctx.IR, ctx.T13, ctx.TE
    etake, egive = self.rings.etake, self.rings.egive
    xb2 = scr["xb2"]

    @pl.when(n_hit > 0)
    def _():
        w13 = etake("e13", 0)
        self.es_wait(0)
        xb2[:, :IR] = self.gate_up(0, w13)
        egive(0, l, 0)

        def body(i, carry):
            u13, u2 = (i + 1) * TE, i * TE + T13
            w13 = etake("e13", u13)
            self.es_wait(i + 1)
            w2r = etake("e2", u2)
            mid = xb2[:, :IR]
            nxt = self.gate_up(i + 1, w13)
            self.down(i, mid, w2r)
            xb2[:, :IR] = nxt
            egive(u13, l, 0)
            egive(u2, l, T13)
            return carry

        lax.fori_loop(0, n_hit - 1, body, 0)
        last = n_hit - 1
        u2 = last * TE + T13
        self.down(last, xb2[:, :IR], etake("e2", u2))
        egive(u2, l, T13)


@contextlib.contextmanager
def _mode_patches(mode: str):
    orig = MoEBlock.experts_piped
    if mode in LATE_W2:
        MoEBlock.experts_piped = _experts_piped_late
    try:
        yield
    finally:
        MoEBlock.experts_piped = orig


def _env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


def _template(mode: str) -> K.MKConfig:
    c = dataclasses.replace(
        K.FULL,
        R=R,
        B=16,
        T=4,
        moe_tp=True,
        stack_e=False,
        paged_kv=True,
        fused_pool_io=True,
        num_pages=4096,
        page_size=128,
        max_pages_per_seq=64,
        ctx=8192,
        ckpt_k=4,
        **MODE_KW[mode],
    )
    c.check()
    return c


class _Gen:
    """Jitted, tp-sharded weight generators shared by every mode (one compile per shape)."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        sharding = NamedSharding(mesh, P("tp"))
        self.sharding = sharding

        @functools.partial(jax.jit, static_argnums=(1,), out_shardings=sharding)
        def rand_fp8(key, shape):
            return jax.random.randint(key, shape, 0, 64, dtype=jnp.uint8).view(K.FP8)

        @functools.partial(jax.jit, static_argnums=(1, 2), out_shardings=sharding)
        def rand_norm(key, shape, dt):
            return (jax.random.normal(key, shape, dtype=jnp.float32) * 0.02).astype(dt)

        @functools.partial(jax.jit, static_argnums=(1,), out_shardings=sharding)
        def rand_es(key, shape):
            return jnp.abs(jax.random.normal(key, shape, dtype=jnp.float32)) * 0.01 + 0.01

        self.rand_fp8, self.rand_norm, self.rand_es = rand_fp8, rand_norm, rand_es
        self._fuse: dict[tuple[K.MKConfig, str], Any] = {}
        self._rand: dict[tuple[K.MKConfig, str], Any] = {}

    def fuse(self, c: K.MKConfig, kind: str):
        """(w [R*EW, 1, K, N] FP8, es [R*L, EW, 1, 2*IR+H] f32, l i32) -> fused / padded leaf."""
        fn = self._fuse.get((c, kind))
        if fn is None:
            IR = c.IR

            def body(w, es, l):
                es_l = jax.lax.dynamic_index_in_dim(es, l, axis=0, keepdims=False)
                if kind == "e13":
                    out = fuse_e13(c, w[:, 0], es_l[:, 0, : 2 * IR])
                else:
                    out = fuse_e2(c, w[:, 0], es_l[:, 0, 2 * IR :])
                return out[:, None]

            fn = jax.jit(
                jax.shard_map(
                    body,
                    mesh=self.mesh,
                    in_specs=(P("tp"), P("tp"), P()),
                    out_specs=P("tp"),
                    check_vma=False,
                ),
                out_shardings=self.sharding,
            )
            self._fuse[(c, kind)] = fn
        return fn

    def rand_rows(self, c: K.MKConfig, kind: str):
        """Overwrite a bottom-fused leaf's 32 scale rows with random FP8 bytes 0..63 (as the
        first FUSE_ES microbenchmark generated its whole fused leaves)."""
        fn = self._rand.get((c, kind))
        if fn is None:
            k0 = c.H if kind == "e13" else c.IR

            @functools.partial(jax.jit, out_shardings=self.sharding)
            def f(leaf, key):
                u = lax.bitcast_convert_type(leaf, jnp.uint8)
                r = jax.random.randint(
                    key, (*u.shape[:2], 32, u.shape[3]), 0, 64, dtype=jnp.uint8
                )
                u = lax.dynamic_update_slice(u, r, (0, 0, k0, 0))
                return lax.bitcast_convert_type(u, K.FP8)

            fn = self._rand[(c, kind)] = f
        return fn


def _alloc_weights(gen: _Gen, c: K.MKConfig, rand_scales: bool = False) -> list[jax.Array]:
    """Mode-specific weight leaves built from mode-independent random draws: every mode sees the
    same dense weights, main expert weights and FP32 expert scales."""
    c0 = dataclasses.replace(c, fuse_es=False, es_pad=False, es_layout="bottom", es_read="reg")
    shapes0 = K.weight_shapes(c0, K.default_tiles(c0))
    shapes = K.weight_shapes(c, K.default_tiles(c))
    idx_es = list(K.W_NAMES).index("es")
    sh_es, _ = shapes0["es"]
    es_full = gen.rand_es(jax.random.PRNGKey(1000 * idx_es), (R * sh_es[0], *sh_es[1:]))
    leaves: list[jax.Array] = []
    for idx, k in enumerate(K.W_NAMES):
        if k in ("e13", "e2"):
            for l, spec in enumerate(shapes0[k]):
                base = gen.rand_fp8(
                    jax.random.PRNGKey(1000 * idx + l), (R * spec.shape[0], *spec.shape[1:])
                )
                if c.fuse_es or c.es_pad:
                    leaf = gen.fuse(c, k)(base, es_full, jnp.int32(l))
                    del base
                    if rand_scales:
                        leaf = gen.rand_rows(c, k)(leaf, jax.random.PRNGKey(77_000 + 1000 * idx + l))
                else:
                    leaf = base
                want = (R * shapes[k][l].shape[0], *shapes[k][l].shape[1:])
                assert tuple(leaf.shape) == want, (k, l, leaf.shape, want)
                leaves.append(leaf)
        elif k == "es":
            if c.fuse_es:
                sh, _ = shapes["es"]
                leaves.append(
                    jax.device_put(np.zeros((R * sh[0], *sh[1:]), np.float32), gen.sharding)
                )
            else:
                leaves.append(es_full)
        else:
            sh, dt = shapes0[k]
            gsh = (R * sh[0], *sh[1:])
            key = jax.random.PRNGKey(1000 * idx)
            leaves.append(gen.rand_fp8(key, gsh) if dt == K.FP8 else gen.rand_norm(key, gsh, dt))
    jax.block_until_ready(leaves)
    return leaves


def _alloc_pools(mesh: Mesh, c: K.MKConfig) -> list[jax.Array]:
    sharding = NamedSharding(mesh, P("tp"))
    local_sh = (c.num_pages, 2, 64, 4, c.page_size)

    @functools.partial(jax.jit, in_shardings=(sharding,), out_shardings=sharding)
    @functools.partial(
        jax.shard_map, mesh=mesh, in_specs=(P("tp"),), out_specs=P("tp"), check_vma=False
    )
    def zero_pool(seed):
        return (seed + jnp.zeros(local_sh, dtype=jnp.uint8)).view(K.FP8)

    seed = jax.device_put(np.zeros((R, 1, 1, 1, 1), dtype=np.uint8), sharding)
    pools = [zero_pool(seed) for _ in range(c.NATT)]
    jax.block_until_ready(pools)
    return pools


def _hidden(c: K.MKConfig, routing: str) -> np.ndarray:
    bf16 = jnp.dtype(K.BF16)
    if routing == "same":
        return np.full((R * c.M, c.H), 0.02, dtype=bf16)
    rng = np.random.default_rng(1009 * c.T + c.B)
    h = rng.standard_normal((c.M, c.H)).astype(np.float32)
    h[c.B :] = 0.0
    return np.tile(h.astype(bf16), (R, 1))


def _forced_choice(c: K.MKConfig) -> np.ndarray:
    """[R * L, M, E] f32 routing choice, identical on every rank: each live row of each layer
    picks TOPK uniformly random distinct experts (the hit-expert count of uncorrelated routing)."""
    rng = np.random.default_rng(7919 * c.T + c.B)
    ch = rng.random((c.L, c.M, c.E), dtype=np.float32) + 0.01
    ch[:, c.B :] = 0.0
    return np.tile(ch, (R, 1, 1))


def _expected_hits(c: K.MKConfig, choice: np.ndarray | None) -> float:
    """Mean distinct experts per layer listed by the live rows (forced routing only)."""
    if choice is None:
        return float("nan")
    top = np.argsort(-choice[: c.L, : c.B], axis=-1)[..., : c.TOPK]
    return float(np.mean([len(np.unique(top[l])) for l in range(c.L)]))


def _step_inputs(mesh: Mesh, c: K.MKConfig, routing: str):
    sharding = NamedSharding(mesh, P("tp"))
    T, B = c.T, c.B
    ns = B // T
    pos = np.zeros((R * B,), dtype=np.int32)
    for r in range(R):
        for s in range(ns):
            for t in range(T):
                pos[r * B + s * T + t] = 512 + t
    pt = np.zeros((R * ns, c.max_pages_per_seq), dtype=np.int32)
    for r in range(R):
        for s in range(ns):
            for p in range(8):
                pt[r * ns + s, p] = 1 + s * 8 + p
    ckpt = np.zeros((R * 3, ns, T), dtype=np.int32)
    slot = 600
    for r in range(R):
        for sub in range(3):
            for s in range(ns):
                for t in range(T):
                    ckpt[r * 3 + sub, s, t] = slot
                    slot = 600 + ((slot - 599) % 200)
    roff = np.zeros((R * 3, ns), dtype=np.int32)
    put = functools.partial(jax.device_put, device=sharding)
    hid = _hidden(c, "diverse" if routing == "forced" else routing)
    choice = _forced_choice(c) if routing == "forced" else None
    rb = (put(choice),) if choice is not None else ()
    return (put(pos), put(hid), put(pt), put(ckpt), put(roff), rb), _expected_hits(c, choice)


def _jit_step(mesh, c, tiles, B, forced: bool):
    step_fn, in_specs, out_specs, donate = pool_io.make_flat_pooled_step(
        mesh, c, tiles, fused_pool_io=True, out_rows=B, forced_routing=forced
    )
    return jax.jit(
        step_fn,
        in_shardings=tuple(NamedSharding(mesh, s) for s in in_specs),
        out_shardings=tuple(NamedSharding(mesh, s) for s in out_specs),
        donate_argnums=donate,
    )


def _bench_mode(mesh, gen, mode, buckets, routings, warmup, iters, ref_out, tag):
    c_t = _template(mode)
    print(
        f"\n[bench] {tag} mode={mode}: {MODE_KW[mode]} expert DMA bytes/expert="
        f"{expert_dma_bytes(c_t) / 1024:.0f} KiB ebank={K.default_tiles(c_t).ebank // 1024} KiB",
        flush=True,
    )
    t0 = time.perf_counter()
    w_leaves = _alloc_weights(gen, c_t, rand_scales=mode in RAND_SCALES)
    pools = _alloc_pools(mesh, c_t)
    print(f"[bench] mode={mode}: weights ready in {time.perf_counter() - t0:.1f} s", flush=True)
    res: dict[str, dict[str, Any]] = {}
    for T, B in buckets:
        c = dataclasses.replace(c_t, B=B, T=T, ckpt_k=T)
        tiles = K.default_tiles(c)
        steps: dict[bool, Any] = {}
        for routing in routings:
            key = f"{routing}|{T}x{B}"
            forced = routing == "forced"
            try:
                if forced not in steps:
                    steps[forced] = _jit_step(mesh, c, tiles, B, forced)
                step = steps[forced]
                (pos, hid, pt, ckpt, roff, rb), hits = _step_inputs(mesh, c, routing)
                check = None
                tc = time.perf_counter()
                for w_i in range(warmup):
                    outs = step(pos, hid, *w_leaves, *pools, pt, ckpt, roff, *rb)
                    outs[0].block_until_ready()
                    pools = list(outs[1:])
                    if w_i == 0:
                        first_s = time.perf_counter() - tc
                        out0 = np.asarray(outs[0]).astype(np.float32)
                        if key not in ref_out:
                            ref_out[key] = (mode, out0)
                            check = {"vs": mode, "equal": True, "max_abs_diff": 0.0}
                        else:
                            ref_mode, ref = ref_out[key]
                            diff = np.abs(out0 - ref)
                            check = {
                                "vs": ref_mode,
                                "equal": bool(np.array_equal(out0, ref, equal_nan=True)),
                                "max_abs_diff": float(np.nanmax(diff)) if diff.size else 0.0,
                                "nan": int(np.isnan(out0).sum()),
                            }
                times = []
                for _ in range(iters):
                    t1 = time.perf_counter()
                    outs = step(pos, hid, *w_leaves, *pools, pt, ckpt, roff, *rb)
                    outs[0].block_until_ready()
                    times.append((time.perf_counter() - t1) * 1e6)
                    pools = list(outs[1:])
                med = float(np.median(times))
                res[key] = {
                    "median_us": med,
                    "mean_us": float(np.mean(times)),
                    "min_us": float(np.min(times)),
                    "p90_us": float(np.percentile(times, 90)),
                    "hits_per_layer": hits,
                    "check": check,
                }
                print(
                    f"[bench] {tag} mode={mode:12s} {routing:7s} T={T} B={B:2d}: "
                    f"median={med:8.1f} us min={np.min(times):8.1f} us first={first_s:5.1f} s "
                    f"hits/layer={hits:6.1f} equal={check['equal']} "
                    f"maxdiff={check['max_abs_diff']:.3g}",
                    flush=True,
                )
            except Exception as e:  # keep the remaining buckets / modes going
                traceback.print_exc()
                res[key] = {"error": f"{type(e).__name__}: {e}"[:2000]}
                print(f"[bench] mode={mode} {routing} T={T} B={B}: FAILED {e!r:.300}", flush=True)
        del steps
        gc.collect()
    del w_leaves, pools
    gc.collect()
    return res


def main() -> None:
    devs = jax.devices("tpu")
    print(f"Found {len(devs)} TPU devices: {devs}", flush=True)
    assert len(devs) >= R, f"Expected >= {R} TPU devices on 2x2x1 tpu7x, got {len(devs)}"
    mesh = Mesh(np.array(devs[:R]), ("tp",))
    modes = _env_list("MK_ES_MODES", "sep,pad,bottom_reg,bottom_chunk,right")
    routings = _env_list("MK_ES_ROUTINGS", "same,diverse,forced")
    buckets = [
        tuple(int(v) for v in b.split("x"))
        for b in _env_list("MK_ES_BUCKETS", "1x4,1x8,1x16,4x4,4x8,4x16,4x32,4x64")
    ]
    warmup = int(os.environ.get("MK_ES_WARMUP", "5"))
    iters = int(os.environ.get("MK_ES_ITERS", "40"))
    passes = int(os.environ.get("MK_ES_PASSES", "2"))
    print(
        f"[bench] modes={modes} routings={routings} buckets={buckets} iters={iters} "
        f"passes={passes}",
        flush=True,
    )

    ref_out: dict[str, tuple[str, np.ndarray]] = {}
    gen = _Gen(mesh)
    results: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}
    for p in range(passes):
        order = modes if p % 2 == 0 else modes[::-1]  # reversed second pass exposes drift
        for m in order:
            with _mode_patches(m):
                results[m].append(
                    _bench_mode(
                        mesh, gen, m, buckets, routings, warmup, iters, ref_out, f"pass{p}"
                    )
                )

    def med(m: str, key: str) -> float | None:
        vals = [r[key]["median_us"] for r in results[m] if "median_us" in r.get(key, {})]
        return float(np.mean(vals)) if vals else None

    def equal(m: str, key: str) -> bool:
        return all((r.get(key, {}).get("check") or {}).get("equal", True) for r in results[m])

    base = modes[0]
    print(
        f"\n=== Qwen3.5-397B TP-MK expert-scale layout ablation (L=60, R=8) ===\n"
        f"median us, mean of {passes} pass(es); (delta vs {base}); per-expert DMA bytes: "
        + ", ".join(f"{m}={expert_dma_bytes(_template(m)) / 1024:.0f}KiB" for m in modes)
    )
    for routing in routings:
        print(f"\n--- routing={routing} ---")
        print("| T | B | hits/layer | " + " | ".join(modes) + " |")
        print("|---|---|---|" + "|".join(["---"] * len(modes)) + "|")
        for T, B in buckets:
            key = f"{routing}|{T}x{B}"
            b0 = med(base, key)
            hits = next(
                (r[key].get("hits_per_layer") for r in results[base] if key in r), float("nan")
            )
            cells = []
            for m in modes:
                v = med(m, key)
                if v is None:
                    cells.append("ERR")
                    continue
                eq = "" if equal(m, key) else " MISMATCH"
                if m == base or b0 is None:
                    cells.append(f"{v:.1f}{eq}")
                else:
                    cells.append(f"{v:.1f} ({(v - b0) / b0 * 100:+.2f}%){eq}")
            print(f"| {T} | {B} | {hits:.1f} | " + " | ".join(cells) + " |")
        if passes > 1:
            spread = []
            for m in modes:
                for T, B in buckets:
                    key = f"{routing}|{T}x{B}"
                    vals = [r[key]["median_us"] for r in results[m] if "median_us" in r.get(key, {})]
                    if len(vals) > 1:
                        spread.append(abs(vals[1] - vals[0]) / np.mean(vals) * 100)
            if spread:
                print(
                    f"pass-to-pass |delta| of medians: mean {np.mean(spread):.2f}%, "
                    f"max {np.max(spread):.2f}%"
                )

    out_dir = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mk_es_ablation.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "modes": {m: MODE_KW[m] for m in modes},
                "expert_dma_bytes": {m: expert_dma_bytes(_template(m)) for m in modes},
                "routings": routings,
                "buckets": buckets,
                "passes": passes,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved JSON summary to {out_path}")


if __name__ == "__main__":
    main()
