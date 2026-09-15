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
"""Single-chip microbenchmark for RPA v3 decode at Qwen3-Coder-30B-A3B-FP8 shapes.

Measures one `ragged_paged_attention` decode call at the *per-TP-rank* shapes the
production server actually runs, so a kernel change can be judged in minutes on a
4-chip pod instead of hours on an end-to-end serving benchmark.

Per-rank shapes for Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 at TP=8:
  32 q heads / 8   -> 4 q heads
  4 kv heads       -> replicated to 1 kv head per rank
  head_dim         -> 128
  kv cache dtype   -> fp8_e4m3 (--kv-cache-dtype=fp8), page_size 256 (--block-size=256)

Why this shape matters: `get_kv_cache_shape` pads the merged-KV-head axis with
`align_to(actual_num_kv_heads * 2, kv_packing)`. For fp8 `kv_packing == 4`, so at
`num_kv_heads == 1` the axis is padded 2 -> 4 and **half of every KV page read is
zeros**. `--report-bytes` prints the resulting real-vs-padded traffic.

Two correctness-of-measurement details this script is careful about, both of which
have silently invalidated past measurements of this kernel:

1. `ragged_paged_attention` is jitted with `donate_argnames=(queries, keys, values,
   kv_cache)`. If the caller does not thread the returned `kv_cache` back in, XLA
   inserts a full copy of the cache on every call. At these shapes that copy is
   hundreds of MB and inflates the reading by an order of magnitude. This script
   threads the donated cache and pre-allocates a ring of q/k/v buffers so no call
   ever reuses a donated input.
2. Timing brackets `jax.block_until_ready`, discards warmup, and reports the median
   plus the inter-quartile range so a single scheduling hiccup cannot pass for a
   regression.

Two timings are reported, and the difference between them matters:

  `lat_ms`    one call, blocked on immediately. Includes the full host dispatch
              round-trip, which at these shapes is a fixed ~0.3 ms and dwarfs the
              kernel itself. Useful only as a latency figure.
  `dev_ms`    `--chain` calls issued back-to-back with a single block at the end,
              divided by the chain length. The threaded `kv_cache` dependency keeps
              the device executing them in order while the host runs ahead, so
              dispatch overlaps execution and this tracks device time.

**Judge kernel changes on `dev_ms`.** A `lat_ms`-only comparison buries a real
kernel win under constant per-call overhead: measured at these shapes, `lat_ms`
barely moves across a 4.5x change in KV bytes moved, so it is nearly blind to
exactly the thing this benchmark exists to detect.

The arms, and why there are three
---------------------------------
`kernels/experimental/batched_rpa` already implements a layout that removes this
padding: `KVLayout.SEQ_ALONG_LANE` fills the 32-bit packing word with head_dim
slices instead of KV-head slots, so the merged-KV axis is never padded (256
B/token here instead of 512). It is not a drop-in swap -- it rejects any
page_size but 128, while the production config runs 256. So a two-arm
"v3@256 vs SEQ_ALONG_LANE@128" comparison confounds the layout with the page
size and cannot say which one moved the number. Measure three:

  --kernel v3      --page-size 256               production baseline
  --kernel v3      --page-size 128               isolates the page-size change
  --kernel batched --kv-layout sal --page-size 128   layout, on top of that

A fourth, `--kernel batched --kv-layout has --page-size 128`, separates "the
batched kernel is different" from "the layout is better"; it is worth running
before attributing any win to the layout.

Usage:
  python scripts/bench_rpa_decode_fp8.py                       # default sweep
  python scripts/bench_rpa_decode_fp8.py --kv-lens 8192 --num-seqs 64
  python scripts/bench_rpa_decode_fp8.py --json out.json --report-bytes
  python scripts/bench_rpa_decode_fp8.py --kernel batched --kv-layout sal \
      --page-size 128 --report-bytes
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from statistics import median

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as brpa_configs
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as brpa
from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    get_kv_cache_shape, ragged_paged_attention)
from vllm_torchtpu.kernels.ragged_paged_attention.v3.util import (
    cdiv, get_dtype_packing)

# Arms. `v3` is the production kernel (PallasAttentionBackend); `batched` is the
# experimental batched RPA reachable via --attention-backend CUSTOM.
#
# SEQ_ALONG_LANE fills the 32-bit fp8 packing word with head_dim slices instead of
# KV-head slots, so the merged-KV axis is never padded -- which is the whole point
# of measuring it here. It accepts page_size 128 only (configs.py:359-363), so a
# fair read needs v3 measured at 128 too, to separate the layout effect from the
# page-size effect.
_LAYOUTS = {
    "has": brpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
    "sal": brpa_configs.KVLayout.SEQ_ALONG_LANE,
}

# Per-rank production shape at TP=8. See module docstring.
DEFAULT_NUM_Q_HEADS = 4
DEFAULT_NUM_KV_HEADS = 1
DEFAULT_HEAD_DIM = 128
DEFAULT_PAGE_SIZE = 256

_DTYPES = {
    "fp8": jnp.float8_e4m3fn,
    "fp8_e4m3": jnp.float8_e4m3fn,
    "bf16": jnp.bfloat16,
}


@dataclass
class Spec:
    """One decode measurement point."""
    num_seqs: int
    kv_len: int
    num_q_heads: int = DEFAULT_NUM_Q_HEADS
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS
    head_dim: int = DEFAULT_HEAD_DIM
    page_size: int = DEFAULT_PAGE_SIZE
    kv_dtype: str = "fp8"
    q_dtype: str = "bf16"
    kernel: str = "v3"      # v3 | batched
    kv_layout: str = "has"  # has | sal   (batched only)

    def __post_init__(self):
        if self.kernel == "v3" and self.kv_layout != "has":
            raise SystemExit("--kv-layout sal requires --kernel batched")
        if self.layout_enum == brpa_configs.KVLayout.SEQ_ALONG_LANE \
                and self.page_size != 128:
            # configs.py:359-363 rejects this, but it would surface as an opaque
            # trace from inside jit; refuse here where the message can say why.
            raise SystemExit(
                f"SEQ_ALONG_LANE requires page_size=128, got {self.page_size}. "
                "Compare it against --kernel v3 --page-size 128, not 256.")

    @property
    def layout_enum(self) -> brpa_configs.KVLayout:
        return _LAYOUTS[self.kv_layout]

    @property
    def arm(self) -> str:
        return (f"{self.kernel}@{self.page_size}"
                if self.kernel == "v3"
                else f"{self.kernel}/{self.kv_layout}@{self.page_size}")

    @property
    def pages_per_seq(self) -> int:
        return cdiv(self.kv_len, self.page_size)

    @property
    def total_num_pages(self) -> int:
        # Exactly what the workload needs; the allocator would round up, but a
        # larger pool only changes which pages are touched, not how many.
        return self.num_seqs * self.pages_per_seq

    def kv_cache_shape(self) -> tuple[int, ...]:
        if self.kernel == "v3":
            return get_kv_cache_shape(self.total_num_pages, self.page_size,
                                      self.num_kv_heads, self.head_dim,
                                      _DTYPES[self.kv_dtype])
        return brpa.get_kv_cache_shape(self.total_num_pages, self.page_size,
                                       self.num_kv_heads, self.head_dim,
                                       _DTYPES[self.kv_dtype],
                                       kv_layout=self.layout_enum)

    def traffic(self) -> dict[str, float]:
        """Real vs materialized KV bytes touched by one decode step.

        `moved` is derived from the *actual* cache shape rather than from a model
        of the padding rule, so it stays correct across layouts whose padding
        works differently -- which is the comparison this script exists to make.
        """
        itemsize = jnp.dtype(_DTYPES[self.kv_dtype]).itemsize
        shape = self.kv_cache_shape()
        per_page = math.prod(shape[1:]) * itemsize
        per_tok_moved = per_page / self.page_size
        # What the attention math actually needs: K and V, at the lane-aligned
        # head_dim the MXU would read anyway.
        per_tok_real = self.num_kv_heads * 2 * cdiv(self.head_dim, 128) * 128 * itemsize
        toks = self.num_seqs * self.kv_len
        packing = get_dtype_packing(_DTYPES[self.kv_dtype])
        return {
            "real_bytes": float(toks * per_tok_real),
            "moved_bytes": float(toks * per_tok_moved),
            "waste_frac": 1.0 - (per_tok_real / per_tok_moved),
            "bytes_per_token": per_tok_moved,
            "kv_packing": packing,
        }


def _build(spec: Spec, seed: int = 0):
    """Build a pure-decode workload: every sequence contributes exactly one query."""
    rng = np.random.default_rng(seed)
    n = spec.num_seqs
    kv_dtype = _DTYPES[spec.kv_dtype]
    q_dtype = _DTYPES[spec.q_dtype]

    # Decode: q_len == 1 for every sequence, so distribution is (n, n, n).
    cu_q_lens = np.arange(n + 1, dtype=np.int32)
    distribution = np.array([n, n, n], dtype=np.int32)
    kv_lens = np.full(n, spec.kv_len, dtype=np.int32)

    # Give every sequence a distinct, shuffled page run so the DMA pattern is
    # scattered the way a real paged cache is, not sequential.
    pool = np.arange(spec.total_num_pages, dtype=np.int32)
    rng.shuffle(pool)
    page_indices = pool[:n * spec.pages_per_seq].astype(np.int32)

    kv_cache = jnp.asarray(
        rng.standard_normal(spec.kv_cache_shape(), dtype=np.float32) * 0.1,
        dtype=kv_dtype)

    def new_qkv(i: int):
        r = np.random.default_rng(seed + 1000 + i)
        q = jnp.asarray(r.standard_normal((n, spec.num_q_heads, spec.head_dim),
                                          dtype=np.float32),
                        dtype=q_dtype)
        k = jnp.asarray(
            r.standard_normal((n, spec.num_kv_heads, spec.head_dim),
                              dtype=np.float32) * 0.1,
            dtype=kv_dtype)
        v = jnp.asarray(
            r.standard_normal((n, spec.num_kv_heads, spec.head_dim),
                              dtype=np.float32) * 0.1,
            dtype=kv_dtype)
        return q, k, v

    static = dict(
        kv_lens=jnp.asarray(kv_lens),
        page_indices=jnp.asarray(page_indices),
        cu_q_lens=jnp.asarray(cu_q_lens),
        distribution=jnp.asarray(distribution),
    )
    return kv_cache, new_qkv, static


def _stats(samples_ms: list[float]) -> dict[str, float]:
    s = sorted(samples_ms)
    med = median(s)
    q1 = s[len(s) // 4]
    q3 = s[(3 * len(s)) // 4]
    return {
        "median_ms": med,
        "p25_ms": q1,
        "p75_ms": q3,
        "min_ms": s[0],
        "iqr_pct": (q3 - q1) / med * 100.0 if med else 0.0,
    }


def run_spec(spec: Spec, iters: int, warmup: int, chain: int,
             seed: int = 0) -> dict:
    kv_cache, new_qkv, static = _build(spec, seed)
    sm_scale = 1.0 / math.sqrt(spec.head_dim)

    # Ring of q/k/v buffers: every call gets buffers it is allowed to donate,
    # and we never hand a donated (deleted) buffer back to the kernel.
    n_buf = warmup + iters + iters * chain
    ring = [new_qkv(i) for i in range(n_buf)]

    # Both entry points take the same eight positional args and return
    # (out, new_kv_cache); only the layout kwarg differs.
    if spec.kernel == "v3":
        fn, extra = ragged_paged_attention, {}
    else:
        fn, extra = brpa.ragged_paged_attention, {"kv_layout": spec.layout_enum}

    def call(idx, cache):
        q, k, v = ring[idx]
        out, cache = fn(
            q,
            k,
            v,
            cache,
            static["kv_lens"],
            static["page_indices"],
            static["cu_q_lens"],
            static["distribution"],
            sm_scale=sm_scale,
            # fp8 KV cache: q stays bf16, K/V carry unit scales here because the
            # microbenchmark measures kernel time, not quantization accuracy.
            k_scale=1.0,
            v_scale=1.0,
            **extra,
        )
        return out, cache

    cur = 0
    for _ in range(warmup):
        out, kv_cache = call(cur, kv_cache)
        cur += 1
    jax.block_until_ready((out, kv_cache))

    # (a) Per-call latency: blocked on immediately, so it carries host dispatch.
    lat_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out, kv_cache = call(cur, kv_cache)
        cur += 1
        jax.block_until_ready((out, kv_cache))
        lat_ms.append((time.perf_counter() - t0) * 1e3)

    # (b) Chained: issue `chain` calls, block once. The threaded kv_cache forces
    # in-order device execution while the host runs ahead, so per-call time here
    # reflects device occupancy rather than dispatch latency.
    dev_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for _ in range(chain):
            out, kv_cache = call(cur, kv_cache)
            cur += 1
        jax.block_until_ready((out, kv_cache))
        dev_ms.append((time.perf_counter() - t0) * 1e3 / chain)

    lat = _stats(lat_ms)
    dev = _stats(dev_ms)
    med_dev = dev["median_ms"]
    traffic = spec.traffic()
    return {
        **asdict(spec),
        "kv_cache_shape": list(spec.kv_cache_shape()),
        "pages_per_seq": spec.pages_per_seq,
        "total_num_pages": spec.total_num_pages,
        "lat": lat,
        "dev": dev,
        # Headline number for judging kernel changes.
        "dev_ms": med_dev,
        "lat_ms": lat["median_ms"],
        "dispatch_overhead_ms": lat["median_ms"] - med_dev,
        "iters": iters,
        "chain": chain,
        **traffic,
        # Effective bandwidth over device time, on bytes actually moved.
        "moved_gib_s": traffic["moved_bytes"] / (med_dev * 1e-3) / 2**30,
        "real_gib_s": traffic["real_bytes"] / (med_dev * 1e-3) / 2**30,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-seqs", type=int, nargs="+", default=[64])
    p.add_argument("--kv-lens", type=int, nargs="+",
                   default=[2048, 4096, 8192, 9216])
    p.add_argument("--num-q-heads", type=int, default=DEFAULT_NUM_Q_HEADS)
    p.add_argument("--num-kv-heads", type=int, default=DEFAULT_NUM_KV_HEADS)
    p.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    p.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    p.add_argument("--kernel", choices=["v3", "batched"], default="v3",
                   help="v3 = production PallasAttentionBackend; "
                        "batched = kernels/experimental/batched_rpa.")
    p.add_argument("--kv-layout", choices=sorted(_LAYOUTS), default="has",
                   help="batched only. sal = SEQ_ALONG_LANE (unpadded KV axis).")
    p.add_argument("--kv-dtype", choices=sorted(_DTYPES), default="fp8")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--chain", type=int, default=20,
                   help="Calls issued per block for the device-time measurement.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--report-bytes", action="store_true",
                   help="Print real-vs-padded KV traffic per point.")
    p.add_argument("--label", type=str, default="",
                   help="Free-form tag echoed into the JSON (e.g. a commit sha).")
    args = p.parse_args()

    print(f"jax {jax.__version__} backend={jax.default_backend()} "
          f"devices={jax.device_count()}")
    if jax.default_backend() != "tpu":
        raise SystemExit("Refusing to report: not running on a TPU backend.")
    print(f"device: {jax.devices()[0].device_kind}")

    rows = []
    for ns in args.num_seqs:
        for kv in args.kv_lens:
            spec = Spec(num_seqs=ns, kv_len=kv, num_q_heads=args.num_q_heads,
                        num_kv_heads=args.num_kv_heads, head_dim=args.head_dim,
                        page_size=args.page_size, kv_dtype=args.kv_dtype,
                        kernel=args.kernel, kv_layout=args.kv_layout)
            r = run_spec(spec, args.iters, args.warmup, args.chain, args.seed)
            r["label"] = args.label
            r["arm"] = spec.arm
            rows.append(r)
            print(f"[{spec.arm:>22s} seqs={ns:4d} kv_len={kv:6d}] "
                  f"dev={r['dev_ms']:8.3f} ms (iqr={r['dev']['iqr_pct']:5.2f}%)  "
                  f"lat={r['lat_ms']:8.3f} ms  "
                  f"disp={r['dispatch_overhead_ms']:6.3f} ms  "
                  f"moved={r['moved_gib_s']:7.1f} GiB/s  "
                  f"cache={tuple(r['kv_cache_shape'])}")
            if args.report_bytes:
                print(f"    real={r['real_bytes']/2**20:9.1f} MiB  "
                      f"moved={r['moved_bytes']/2**20:9.1f} MiB  "
                      f"waste={r['waste_frac']*100:.1f}%  "
                      f"{r['bytes_per_token']:.0f} B/token")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
