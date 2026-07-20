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
"""PCP prefill TTFT benchmark through the REAL attention wrapper.

Unlike micro-benchmarks that call lower-level kernel internals directly,
this drives `wrapper.sharded_pcp_ragged_paged_attention` on the production
sharded mesh: a 2D mesh (pcp x model), q/k/v sharded across PCP/TP axes, and
the KV cache sharded along the PCP block dimension.

TTFT(N) = sum over chunks i of the attention latency of chunk i (queries
[i*CH,(i+1)*CH) attending KV [0,(i+1)*CH)). Latency is measured at every CH step
with a max-size cache and dynamic `kv_lens`, so there is one compile per config.

Usage:
  python tests/kernels/ragged_paged_attention_pcp_wrapper_benchmark.py [MAXCTX] [CH]
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.wrapper import (
    PCP_AXIS_NAME, sharded_pcp_ragged_paged_attention)
from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    ragged_paged_attention)
from vllm_torchtpu.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)
from vllm_torchtpu.layers.common import sharding as sharding_mod
from vllm_torchtpu.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName,
                                                  ShardingAxisNameBase)

# Force the N-D axis names (which carry `pcp`) regardless of NEW_MODEL_DESIGN.
sharding_mod.ShardingAxisName._cls = ShardingAxisNameBase

NQ, NKV, HD = 32, 2, 128
PAGE = 256  # per-rank block_size
Q_BLOCK_SIZE = 512  # Tiling block size for Pallas vector registers
DTYPE = jnp.bfloat16
SM = HD**-0.5
KVP = get_dtype_packing(DTYPE)
NKV2 = align_to(2 * NKV, KVP)
MAX_SEQ = 4
WARMUP, ITERS = 1, 3
_rng = np.random.default_rng(0)
_rand = lambda s: jnp.array(_rng.random(size=s, dtype=np.float32)).astype(DTYPE)


def _bench(fn, *a):
    jax.block_until_ready(fn(*a))
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*a))
    t0 = time.perf_counter()
    for _ in range(ITERS):
        jax.block_until_ready(fn(*a))
    return (time.perf_counter() - t0) / ITERS * 1e3


def _mesh(pcp, tp):
    mesh_axis_names = (PCP_AXIS_NAME, ShardingAxisName.ATTN_HEAD)
    n = pcp * tp
    return Mesh(np.array(jax.devices()[:n]).reshape((pcp, tp)), mesh_axis_names)


def make_pcp_wrapper(pcp, tp, chunk, max_ctx):
    """Drive sharded_pcp_ragged_paged_attention on the real mesh."""
    if jax.device_count() < pcp * tp:
        return lambda ctx: float("nan")
    mesh = _mesh(pcp, tp)

    nq, nkv = NQ // tp, max(1, NKV // tp)
    nkv2 = align_to(2 * nkv, KVP)
    packed_kv_groups = nkv2 // KVP
    npages = max(cdiv(max_ctx, PAGE), 1)
    npages_global = align_to(npages, pcp)

    q = _rand((chunk, nq, HD))
    k = _rand((chunk, nkv, HD))
    v = _rand((chunk, nkv, HD))
    cache = jnp.zeros((npages_global, PAGE, packed_kv_groups, KVP, HD), DTYPE)

    put = lambda x, s: jax.device_put(x, NamedSharding(mesh, s))
    q = put(q, P(PCP_AXIS_NAME, None, None))
    k = put(k, P(PCP_AXIS_NAME, None, None))
    v = put(v, P(PCP_AXIS_NAME, None, None))
    cache = put(cache, P(PCP_AXIS_NAME, None, None, None, None))

    pi = jnp.arange(npages_global, dtype=jnp.int32)
    dist = jnp.array([0, 0, 1], jnp.int32)

    @jax.jit
    def fn(q, k, v, kvl, cu_q):
        out, _ = sharded_pcp_ragged_paged_attention(
            mesh=mesh,
            q=q,
            k=k,
            v=v,
            kv_cache=cache,
            kv_lens=kvl,
            page_indices=pi,
            cu_q_lens=cu_q,
            distribution=dist,
            attention_sink=None,
            sm_scale=SM,
            update_kv_cache=True,
            cp_kv_cache_interleave_size=PAGE,
            q_block_size=Q_BLOCK_SIZE,
        )
        return out

    def measure(ctx):
        kvl = jnp.zeros((MAX_SEQ,), jnp.int32).at[0].set(ctx)
        cu_q = jnp.zeros((MAX_SEQ + 1,), jnp.int32).at[1].set(chunk).at[2:].set(chunk)
        return _bench(fn, q, k, v, kvl, cu_q)

    return measure


def make_tp(tp, chunk, max_ctx):
    """TP baseline: heads sharded /tp, full context. No cross-head collective,
    so per-device latency IS the config's latency -> measure on one device."""
    nq, nkv = NQ // tp, max(1, NKV // tp)
    nkv2 = align_to(2 * nkv, KVP)
    packed_kv_groups = nkv2 // KVP
    npages = max(cdiv(max_ctx, PAGE), 1)
    q = _rand((chunk, nq, HD))
    k = _rand((chunk, nkv, HD))
    v = _rand((chunk, nkv, HD))
    pi = jnp.arange(npages, dtype=jnp.int32)

    @jax.jit
    def fn(q, k, v, ctx):
        cache = jnp.zeros((npages, PAGE, packed_kv_groups, KVP, HD), DTYPE)
        out, _ = ragged_paged_attention(
            q,
            k,
            v,
            cache,
            ctx.reshape(1),
            pi,
            jnp.array([0, chunk], jnp.int32),
            jnp.array([0, 0, 1], jnp.int32),
            sm_scale=SM,
            use_causal_mask=True,
            skip_kv_update=False,
        )
        return out

    return lambda ctx: _bench(fn, q, k, v, jnp.array(ctx, jnp.int32))


def human(n):
    return f"{n // 1024}k" if n < 1024 * 1024 else f"{n / 1024 / 1024:g}M"


if __name__ == "__main__":
    max_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 2 * 1024 * 1024
    CH = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    steps = list(range(CH, max_ctx + 1, CH))
    report = [
        n for n in [16, 32, 64, 128, 256, 512, 1024, 2048]
        if CH <= n * 1024 <= max_ctx
    ]
    print(f"# TTFT (ms) via wrapper.sharded_pcp_ragged_paged_attention, "
          f"CH={human(CH)}, NQ={NQ} NKV={NKV} HD={HD} {DTYPE.__name__} "
          f"PAGE={PAGE} Q_BLOCK={Q_BLOCK_SIZE}")
    m = {
        "tp4": make_tp(4, CH, max_ctx),
        "tp8": make_tp(8, CH, max_ctx),
        "pcp2_tp4": make_pcp_wrapper(2, 4, CH, max_ctx),
        "pcp8_tp1": make_pcp_wrapper(8, 1, CH, max_ctx),
    }
    lat = {}
    for name, f in m.items():
        lat[name] = {c: f(c) for c in steps}
        print(f"  measured {name}", flush=True)
    print("\n# Cumulative TTFT Attention Latency (ms)")
    print("{:>8} | {:>10} {:>10} {:>10} {:>10} | {:>8} {:>8} {:>8} {:>8}".format(
        "context", "tp4(ms)", "tp8(ms)", "pcp2_tp4", "pcp8_tp1", "tp4/pcp2",
        "tp4/pcp8", "tp4/tp8", "tp8/pcp8"))
    for n in report:
        N = n * 1024
        row = {k: sum(lat[k][c] for c in steps if c <= N) for k in m}
        print("{:>8} | {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} | "
              "{:>7.2f}x {:>7.2f}x {:>7.2f}x {:>7.2f}x".format(
                  human(N), row["tp4"], row["tp8"], row["pcp2_tp4"],
                  row["pcp8_tp1"], row["tp4"] / row["pcp2_tp4"],
                  row["tp4"] / row["pcp8_tp1"], row["tp4"] / row["tp8"],
                  row["tp8"] / row["pcp8_tp1"]),
              flush=True)

    print("\n# Last Block Latency (ms)")
    print("{:>8} | {:>10} {:>10} {:>10} {:>10} | {:>8} {:>8} {:>8} {:>8}".format(
        "context", "tp4(ms)", "tp8(ms)", "pcp2_tp4", "pcp8_tp1", "tp4/pcp2",
        "tp4/pcp8", "tp4/tp8", "tp8/pcp8"))
    for n in report:
        N = n * 1024
        row = {k: lat[k][N] for k in m}
        print("{:>8} | {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} | "
              "{:>7.2f}x {:>7.2f}x {:>7.2f}x {:>7.2f}x".format(
                  human(N), row["tp4"], row["tp8"], row["pcp2_tp4"],
                  row["pcp8_tp1"], row["tp4"] / row["pcp2_tp4"],
                  row["tp4"] / row["pcp8_tp1"], row["tp4"] / row["tp8"],
                  row["tp8"] / row["pcp8_tp1"]),
              flush=True)

