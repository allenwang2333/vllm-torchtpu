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
"""Benchmark donated, sharded sparse-MLA cache writing plus gather/attention.

Run under the host TPU lease, e.g.:
  python scripts/benchmark_sparse_mla_cache_update.py --json-output /tmp/decode.json
  python scripts/benchmark_sparse_mla_cache_update.py --skip-writer
  python scripts/benchmark_sparse_mla_cache_update.py --prefill --query-tokens 1024

The default replays 40 active 8K decode sequences in a 64-token bucket with
24 padding rows, top-k 2,048, logical page translation, and 327,680 cache tokens.
Updated caches are returned, donated, and retained across all steps. Timing
includes synchronization and excludes compilation. Use --profile-dir to check
that the native writer introduces no whole-cache data-formatting offloads.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout, KVCacheType, SparseMLAKVCacheSpec,
    update_sparse_mla_kv_cache)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
from vllm_torchtpu.kernels.mla.sparse import native_sc_cache_layout

NOPE_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 256
SEQ_LEN = 8192
NUM_HEADS = 64
TOPK = 2048


def make_specs(layout, config):
    num_pages = config.active_sequences * (SEQ_LEN // PAGE_SIZE)
    return tuple(
        SparseMLAKVCacheSpec.create(kind, layout, num_pages, PAGE_SIZE, dim, 4)
        for kind, dim in ((KVCacheType.NOPE, NOPE_DIM), (KVCacheType.ROPE,
                                                         ROPE_DIM)))


def finite_bytes(shape):
    pattern = (np.arange(shape[-1], dtype=np.uint8) %
               0x78).reshape((1, ) * (len(shape) - 1) + (shape[-1], ))
    return np.broadcast_to(pattern, shape).copy()


def make_problem(config):
    active = config.active_sequences
    tokens = config.query_tokens
    max_seqs = max(64, active)
    num_pages = active * (SEQ_LEN // PAGE_SIZE)
    rng = np.random.default_rng(20260905)
    tc_nope = finite_bytes((num_pages, PAGE_SIZE, 4, 128))
    tc_rope = finite_bytes((num_pages, PAGE_SIZE // 4, 4, 128))
    sc_nope = native_sc_cache_layout.pack_nope(tc_nope)
    sc_rope = native_sc_cache_layout.pack_rope_banded(tc_rope)
    block_tables = np.zeros((max_seqs, (SEQ_LEN // PAGE_SIZE)), np.int32)
    pages = rng.permutation(num_pages).reshape(active, (SEQ_LEN // PAGE_SIZE))
    block_tables[:active] = pages
    seq_lens = np.ones(max_seqs, np.int32)
    seq_lens[:active] = SEQ_LEN
    q_lens = np.zeros(max_seqs, np.int32)
    q_lens[:active] = 1
    if config.prefill:
        q_lens[:active] = tokens // active
        q_lens[:tokens % active] += 1
    active_tokens = int(q_lens.sum())
    cu = np.concatenate(([0], np.cumsum(q_lens))).astype(np.int32)

    topk = np.full((tokens, TOPK), -1, np.int32)
    logical = np.arange(SEQ_LEN, dtype=np.int64)
    for row in range(active_tokens):
        stride = int(rng.integers(1, SEQ_LEN)) | 1
        while math.gcd(stride, SEQ_LEN) != 1:
            stride += 2
        offset = int(rng.integers(0, SEQ_LEN))
        topk[row] = ((offset + stride * logical[:TOPK]) % SEQ_LEN).astype(
            np.int32)

    common = (
        jnp.asarray(
            rng.standard_normal((tokens, NUM_HEADS, NOPE_DIM),
                                dtype=np.float32), jnp.bfloat16),
        jnp.asarray(
            rng.standard_normal((tokens, NUM_HEADS, ROPE_DIM),
                                dtype=np.float32), jnp.bfloat16),
        jnp.asarray(rng.standard_normal((tokens, NOPE_DIM), dtype=np.float32),
                    jnp.float8_e4m3fn),
        jnp.asarray(rng.standard_normal((tokens, ROPE_DIM), dtype=np.float32),
                    jnp.float8_e4m3fn),
        jnp.asarray(topk),
        jnp.asarray(seq_lens),
        jnp.asarray(block_tables.reshape(-1)),
        jnp.asarray(cu),
        jnp.asarray([0 if config.prefill else active, active, active],
                    jnp.int32),
    )
    caches = {
        KVCacheLayout.TENSORCORE: (jnp.asarray(tc_nope), jnp.asarray(tc_rope)),
        KVCacheLayout.SPARSECORE: (jnp.asarray(sc_nope), jnp.asarray(sc_rope)),
    }
    jax.block_until_ready(common)
    for cache_pair in caches.values():
        jax.block_until_ready(cache_pair)
    return common, caches, active_tokens


def make_runner(layout: KVCacheLayout, mesh: Mesh, config):
    nope_spec, rope_spec = make_specs(layout, config)

    def core(nope, rope, q_nope, q_rope, kv, kpe, topk, seq_lens, block_tables,
             cu, distribution):
        if not config.skip_writer:
            nope, rope = update_sparse_mla_kv_cache(nope,
                                                    rope,
                                                    kv,
                                                    kpe,
                                                    seq_lens,
                                                    block_tables,
                                                    cu,
                                                    nope_spec=nope_spec,
                                                    rope_spec=rope_spec)
        q = jnp.concatenate((q_nope, q_rope), axis=-1)
        out = sparse_mla.sparse_ragged_paged_attention(
            q,
            nope,
            rope,
            topk,
            block_tables,
            cu,
            distribution,
            sm_scale=0.125,
            cache_layout=layout.value)
        return nope, rope, out[..., :NOPE_DIM]

    mapped = jax.shard_map(
        core,
        mesh=mesh,
        in_specs=(P(None), ) * 11,
        out_specs=(P(None), P(None), P(None, None, None)),
        check_vma=False,
    )
    return jax.jit(mapped, donate_argnums=(0, 1))


def stats(values):
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "p90_ms": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def invoke(runner, caches, common):
    start = time.perf_counter_ns()
    result = runner(*caches, *common)
    jax.block_until_ready(result)
    return result[:2], result[2], (time.perf_counter_ns() - start) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-tokens", type=int, default=64)
    parser.add_argument("--active-sequences", type=int, default=40)
    parser.add_argument(
        "--prefill",
        action="store_true",
        help="Distribute all query tokens over active sequences.")
    parser.add_argument("--skip-writer", action="store_true")
    parser.add_argument("--profile-dir")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--json-output", type=Path)
    config = parser.parse_args()
    if not 0 < config.active_sequences <= config.query_tokens:
        parser.error("active-sequences must be in [1, query-tokens]")
    if config.query_tokens % 64:
        parser.error("query-tokens must be a multiple of 64")
    if config.query_tokens > config.active_sequences * SEQ_LEN:
        parser.error("query tokens exceed the resident sequence capacity")
    if config.iterations <= 0 or config.warmups < 0:
        parser.error("iterations must be positive and warmups nonnegative")

    devices = jax.local_devices()[:1]
    mesh = Mesh(np.array(devices), ("x", ))
    common, caches, active_tokens = make_problem(config)
    runners = {
        layout: make_runner(layout, mesh, config)
        for layout in KVCacheLayout
    }

    compile_ms = {}
    outputs = {}
    for layout in KVCacheLayout:
        start = time.perf_counter_ns()
        caches[layout], outputs[layout], _ = invoke(runners[layout],
                                                    caches[layout], common)
        compile_ms[layout.value] = (time.perf_counter_ns() - start) / 1e6

    active_slice = slice(0, active_tokens)
    tc = np.asarray(outputs[KVCacheLayout.TENSORCORE][active_slice],
                    np.float32)
    sc = np.asarray(outputs[KVCacheLayout.SPARSECORE][active_slice],
                    np.float32)
    max_abs = float(np.max(np.abs(tc - sc), initial=0.0))
    if not np.allclose(tc, sc, rtol=2e-2, atol=2e-2):
        raise AssertionError(f"layout output mismatch: max_abs={max_abs}")

    for pair in range(config.warmups):
        order = tuple(KVCacheLayout)
        if pair % 2:
            order = order[::-1]
        for layout in order:
            caches[layout], outputs[layout], _ = invoke(
                runners[layout], caches[layout], common)

    profile_dir = config.profile_dir
    if profile_dir:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(profile_dir, create_perfetto_trace=True):
            for step in range(3):
                for layout in KVCacheLayout:
                    with jax.profiler.StepTraceAnnotation(
                            f"full-{layout.value}", step_num=step):
                        caches[layout], outputs[layout], _ = invoke(
                            runners[layout], caches[layout], common)

    samples = {layout.value: [] for layout in KVCacheLayout}
    ratios = []
    for pair in range(config.iterations):
        order = tuple(KVCacheLayout)
        if pair % 2:
            order = order[::-1]
        pair_ms = {}
        for layout in order:
            caches[layout], outputs[layout], elapsed = invoke(
                runners[layout], caches[layout], common)
            samples[layout.value].append(elapsed)
            pair_ms[layout] = elapsed
        ratios.append(pair_ms[KVCacheLayout.SPARSECORE] /
                      pair_ms[KVCacheLayout.TENSORCORE])

    result = {
        "jax": jax.__version__,
        "devices": [str(device) for device in devices],
        "shape": {
            "tokens": config.query_tokens,
            "active_sequences": config.active_sequences,
            "active_tokens": active_tokens,
            "prefill": config.prefill,
            "skip_writer": config.skip_writer,
            "sequence_length": SEQ_LEN,
            "topk": TOPK,
            "page_size": PAGE_SIZE,
            "num_pages": config.active_sequences * (SEQ_LEN // PAGE_SIZE),
        },
        "compile_and_first_ms": compile_ms,
        "max_abs_output_diff": max_abs,
        "timings": {
            name: stats(values)
            for name, values in samples.items()
        },
        "median_paired_sc_over_tc": statistics.median(ratios),
    }
    # Validate the outputs again after retaining donated caches for every step.
    np.testing.assert_array_equal(
        np.asarray(outputs[KVCacheLayout.TENSORCORE][active_slice]),
        np.asarray(outputs[KVCacheLayout.SPARSECORE][active_slice]))
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if config.json_output:
        config.json_output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
