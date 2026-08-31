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
"""Standalone microbenchmark for the v3 ragged-paged-attention decode kernel.

Targets the op that xprof named the single largest consumer of decode device
time for Qwen3-Coder-30B-A3B-FP8 on TPU7x at TP=8:

    RPAd-p_16-bq_1_1-bkv_9216_9216   12.433 ms/step, 66.5% of decode device
                                     time, 48 invocations/step, 259.02 us each

Defaults reproduce that shape exactly: 64 sequences of 8192 cached tokens each,
one query token per sequence, 4 q heads / 2 kv heads per rank, head_dim 128, an
fp8 paged KV cache with page_size 16 and 9216 pages-per-seq worth of slots.

Two things this harness does differently from a naive timing loop, both learned
the hard way from an earlier gmm_v2 harness whose numbers came in 1.6-6.8x above
the same op's xprof cost:

1. `--chain N` issues N dependent kernel invocations inside ONE jit, so a single
   host dispatch covers N kernels. A single decode RPA call is ~259 us of device
   work against ~50-90 us of host dispatch; timing one call at a time measures
   the dispatch. Chaining through the donated kv_cache also mirrors how the 48
   layers of a real step run back to back.
2. Every reported figure is cross-checked against the xprof scope name, which
   this script reconstructs with the kernel's own heuristic, so a row can be
   matched to a trace op by string equality rather than by eyeball.

Block sizes are the tuning lever under test. `v3/tuned_block_sizes.py` has no
importer -- it is dead code -- and `d_block_sizes` is only plumbed through
`attention_interface` on the speculative-decoding path, so for ordinary serving
the decode block sizes come solely from `get_default_block_sizes`, which pins
bq_sz=1 and bkv_sz=min(min_bkv_sz_to_peak, max_kv). This script sweeps around
that choice without needing the serving stack.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import dataclass
from statistics import median

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    RpaCase, get_default_block_sizes, get_kv_cache_shape,
    ragged_paged_attention)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 at TP=8. From the model config:
# 32 q heads, 4 kv heads, 48 layers, head_dim 128. At TP=8 that is 4 q heads
# per rank; 4 kv heads across 8 ranks means vLLM replicates to 1 kv head per
# rank, which the xprof confirms with a literal pad op
#   %pad_bitcast_fusion.19 = f8e4m3fn[64,1,4,128] fusion(f8e4m3fn[64,2,128] ...)
# i.e. 2 live values (K,V for one head) padded into 4 merged slots.
#
# NUM_KV_HEADS matters more than it looks. get_kv_cache_shape packs
# align_to(num_kv_heads*2, kv_packing) slots, and at fp8 kv_packing == 4:
#   1 kv head -> align_to(2,4) == 4 slots, 2 live  -> 50% of every KV byte is padding
#   2 kv heads -> align_to(4,4) == 4 slots, 4 live -> no padding at all
# BOTH produce an identical 5D cache shape and an identical xprof op name, so a
# harness set to 2 measures the padding-free case while production pads. It was
# 2 here previously; the config and the pad op above both say 1.
NUM_LAYERS = 48
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
HEAD_DIM = 128
NUM_SEQS = 64
KV_LEN = 8192
MAX_MODEL_LEN = 9216
PAGE_SIZE = 16
# The serving run's cache was 235067 pages of 16 tokens. Hold total *bytes*
# fixed when page_size varies so a page-size sweep is not confounded by a
# change in how far apart the pages are scattered in HBM.
CACHE_TOKENS = 235067 * 16


@dataclass(frozen=True)
class Inputs:
    q: jax.Array
    k: jax.Array
    v: jax.Array
    cache: jax.Array
    kv_lens: jax.Array
    page_indices: jax.Array
    cu_q_lens: jax.Array
    distribution: jax.Array
    page_size: int
    pages_per_seq: int


def _make_inputs(args, rng: np.random.Generator) -> Inputs:
    """Build one decode step: `num_seqs` sequences, one query token each."""
    page_size = args.page_size
    pages_per_seq = -(-args.max_model_len // page_size)
    total_pages = max(args.num_seqs * pages_per_seq,
                      CACHE_TOKENS // page_size)

    q_dtype = jnp.dtype(args.q_dtype)
    kv_dtype = jnp.dtype(args.kv_dtype)

    num_tokens = args.num_seqs  # exactly one query token per sequence
    q = jnp.asarray(
        rng.standard_normal((num_tokens, NUM_Q_HEADS, HEAD_DIM)) * 0.5,
        q_dtype)
    # The kernel requires the incoming k/v to carry the cache's dtype -- it
    # writes them straight into the paged cache before attending.
    k = jnp.asarray(
        rng.standard_normal((num_tokens, args.num_kv_heads, HEAD_DIM)) * 0.5,
        kv_dtype)
    v = jnp.asarray(
        rng.standard_normal((num_tokens, args.num_kv_heads, HEAD_DIM)) * 0.5,
        kv_dtype)

    cache_shape = get_kv_cache_shape(total_pages, page_size, args.num_kv_heads,
                                     HEAD_DIM, kv_dtype)
    # Allocate on device without a host round trip: a 1.9 GB fp8 buffer costs
    # ~30 s to build in numpy and the kernel is bandwidth-bound, so the values
    # only need to be finite and non-degenerate, not random per element.
    cache = jax.device_put(
        jnp.full(cache_shape, 0.25, dtype=jnp.float32).astype(kv_dtype))

    kv_lens = jnp.full((args.num_seqs, ), args.kv_len, jnp.int32)

    # Page allocation. A real server hands out pages from a shared pool, so a
    # sequence's pages are scattered across the whole cache rather than being
    # one contiguous run; that scatter is exactly what makes the gather
    # expensive, so it is the default.
    if args.contiguous_pages:
        pages = np.arange(args.num_seqs * pages_per_seq, dtype=np.int32)
    else:
        pages = rng.choice(total_pages,
                           size=args.num_seqs * pages_per_seq,
                           replace=False).astype(np.int32)
    page_indices = jnp.asarray(pages)

    cu_q_lens = jnp.asarray(np.arange(args.num_seqs + 1, dtype=np.int32))
    # DECODE covers seqs [0, distribution[0]); PREFILL and MIXED are empty.
    distribution = jnp.asarray([args.num_seqs] * 3, jnp.int32)

    return Inputs(q, k, v, cache, kv_lens, page_indices, cu_q_lens,
                  distribution, page_size, pages_per_seq)


def _block_sizes(args, inp: Inputs) -> tuple[int, int, int, int]:
    """The block sizes the serving path would pick, or the requested override."""
    heur = get_default_block_sizes(
        jnp.dtype(args.q_dtype),
        jnp.dtype(args.kv_dtype),
        NUM_Q_HEADS,
        args.num_kv_heads,
        HEAD_DIM,
        inp.page_size,
        args.num_seqs,  # max_num_tokens
        args.num_seqs,  # max_num_seqs
        inp.pages_per_seq,
        case=RpaCase.DECODE,
    )
    return (heur["bq_sz"], heur["bkv_sz"], heur["bq_csz"], heur["bkv_csz"])


def _scope(page_size: int, blocks: tuple[int, int, int, int]) -> str:
    """Reconstruct the xprof scope name so rows match trace ops by string."""
    bq_sz, bkv_sz, bq_csz, bkv_csz = blocks
    return (f"RPAd-p_{page_size}-bq_{bq_sz}_{bq_csz}"
            f"-bkv_{bkv_sz}_{bkv_csz}")


def _build(inp: Inputs, blocks: tuple[int, int, int, int], args):
    """One jit'd function running `--chain` dependent kernel invocations.

    The kv_cache is threaded through the chain, which both creates the data
    dependency that prevents the compiler from collapsing the repeats and
    reproduces the layer-to-layer serialization of a real decode step.
    """
    kw = dict(
        sm_scale=HEAD_DIM**-0.5,
        d_block_sizes=blocks,
        use_causal_mask=True,
    )
    if jnp.dtype(args.kv_dtype) != jnp.bfloat16:
        # fp8 cache: the kernel dequantizes with these on the fly.
        kw.update(k_scale=1.0, v_scale=1.0)

    chain = args.chain

    @jax.jit
    def run(q, k, v, cache, kv_lens, page_indices, cu_q_lens, distribution):
        out = None
        for _ in range(chain):
            out, cache = ragged_paged_attention(q, k, v, cache, kv_lens,
                                                page_indices, cu_q_lens,
                                                distribution, **kw)
        return out, cache

    def call():
        return run(inp.q, inp.k, inp.v, inp.cache, inp.kv_lens,
                   inp.page_indices, inp.cu_q_lens, inp.distribution)

    return call


def _time(fn, iterations: int) -> tuple[float, float]:
    """Return (compile_and_warmup_s, median_ms_per_chain)."""
    start = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - start

    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        out = fn()
        jax.block_until_ready(out)
        samples.append(time.perf_counter_ns() - t0)
    del out
    return compile_s, median(samples) / 1e6


def _emit(row: dict) -> None:
    print("RESULT " + json.dumps(row, sort_keys=True), flush=True)


def _candidates(args, base: tuple[int, int, int, int],
                max_kv: int) -> list[tuple[int, int, int, int]]:
    """Block-size candidates, always led by the heuristic's own choice.

    Only bkv is swept: bq_sz is structurally 1 in decode (one query token per
    sequence), so there is nothing to vary on the q axis.
    """
    bq_sz, bkv_sz, bq_csz, _ = base
    out = [base]
    if args.sweep == "none":
        return out
    if args.sweep == "split":
        # Hold the fetch block at whatever the heuristic picked -- that is the
        # DMA tile, and shrinking it costs page-table walks -- and slice the
        # compute inside it. `_validate_block_sizes` requires bkv_csz to both
        # divide bkv_sz and be a multiple of page_size, so enumerate the sizes
        # that actually satisfy that rather than a fixed list: at bkv_sz 9216
        # and page 256 the legal set is 256..9216 stepping through the divisors
        # of 36, which excludes round-looking values like 2048.
        out.extend((bq_sz, bkv_sz, bq_csz, csz)
                   for csz in range(args.page_size, bkv_sz + 1, args.page_size)
                   if bkv_sz % csz == 0)
        return _dedup(out)
    for bkv in (256, 512, 1024, 2048, 4096, 8192, max_kv):
        if bkv > max_kv or bkv % args.page_size:
            continue
        # Fetch block == compute block, the shape the heuristic uses.
        out.append((bq_sz, bkv, bq_csz, bkv))
    return _dedup(out)


def _dedup(out):
    seen, unique = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-kv-heads", type=int, default=NUM_KV_HEADS,
                   help="KV heads per rank. 1 reproduces the padded "
                        "production layout; 2 fills all 4 fp8 slots.")
    p.add_argument("--num-seqs", type=int, default=NUM_SEQS)
    p.add_argument("--kv-len", type=int, default=KV_LEN)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    p.add_argument("--page-size", type=int, default=PAGE_SIZE)
    p.add_argument("--q-dtype", default="bfloat16")
    p.add_argument("--kv-dtype", default="float8_e4m3fn")
    p.add_argument("--chain",
                   type=int,
                   default=NUM_LAYERS,
                   help="dependent kernel calls per dispatch (default: 48, "
                   "one full model pass)")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--sweep", default="none", choices=("none", "bkv", "split"))
    p.add_argument("--bkv", type=int, default=0, help="override bkv_sz")
    p.add_argument("--bkv-csz", type=int, default=0, help="override bkv_csz")
    p.add_argument("--contiguous-pages", action="store_true")
    p.add_argument("--verify",
                   action="store_true",
                   help="check each candidate's output against the "
                   "heuristic's; block sizes must not change results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="BASELINE")
    p.add_argument("--device-index", type=int, default=0)
    args = p.parse_args()

    devices = jax.local_devices()
    if not devices or devices[0].platform != "tpu":
        p.error("run this script on a TPU node")
    if args.device_index >= len(devices):
        p.error(f"--device-index must be < {len(devices)}")

    info = pltpu.get_tpu_info()
    print(json.dumps(
        {
            "device_kind": devices[0].device_kind,
            "num_devices": len(devices),
            "vmem_capacity_bytes": int(info.vmem_capacity_bytes),
            "num_lanes": int(info.num_lanes),
        },
        sort_keys=True),
          flush=True)

    rng = np.random.default_rng(args.seed)
    device = devices[args.device_index]
    with jax.default_device(device):
        inp = _make_inputs(args, rng)
        base = _block_sizes(args, inp)
        if args.bkv or args.bkv_csz:
            bkv = args.bkv or base[1]
            base = (base[0], bkv, base[2], args.bkv_csz or bkv)
        max_kv = inp.pages_per_seq * inp.page_size
        cands = _candidates(args, base, max_kv)
        logger.info(
            "page_size=%d pages_per_seq=%d max_kv=%d heuristic=%s "
            "candidates=%d chain=%d", inp.page_size, inp.pages_per_seq, max_kv,
            base, len(cands), args.chain)

        # Bytes of KV the kernel streams per invocation. Report both: the
        # merged-slot layout pads num_kv_heads*2 up to kv_packing, so PHYSICAL
        # is what the DMA actually moves and PAYLOAD is what is live. They
        # differ by 2x at the production shape (1 kv head, fp8).
        itemsize = jnp.dtype(args.kv_dtype).itemsize
        _packing = 4 // itemsize if itemsize <= 4 else 1
        _live = args.num_kv_heads * 2
        _slots = -(-_live // _packing) * _packing if _packing else _live
        payload_bytes = (args.num_seqs * args.kv_len * _live * HEAD_DIM *
                         itemsize)
        kv_bytes = (args.num_seqs * args.kv_len * _slots * HEAD_DIM * itemsize)
        logger.info(
            "kv layout: num_kv_heads=%d live_slots=%d padded_slots=%d "
            "payload=%.2f GB physical=%.2f GB (%.0f%% padding)",
            args.num_kv_heads, _live, _slots, payload_bytes / 1e9,
            kv_bytes / 1e9, 100.0 * (1.0 - _live / _slots))

        reference = None
        for blocks in cands:
            scope = _scope(inp.page_size, blocks)
            try:
                call = _build(inp, blocks, args)
                compile_s, chain_ms = _time(call, args.iterations)
                if args.verify:
                    out = np.asarray(call()[0].astype(jnp.float32))
                    # `max_abs_diff` only ever compares within one process.
                    # Comparing two *kernel* variants means two processes, so
                    # also emit a digest of the output: --seed fixes the inputs,
                    # so equal digests across processes mean equal results.
                    digest = hashlib.blake2b(np.ascontiguousarray(out).tobytes(),
                                             digest_size=8).hexdigest()
                    if reference is None:
                        reference = out
                        max_diff = 0.0
                    else:
                        max_diff = float(np.abs(out - reference).max())
                else:
                    digest = None
                    max_diff = None
            except Exception as exc:  # noqa: BLE001 - candidate triage
                text = str(exc).lower()
                kind = ("VMEM/OOM" if "vmem" in text or "out of memory" in text
                        else "error")
                logger.warning("skipped %s after %s: %s", scope, kind,
                               str(exc).partition("\n")[0])
                continue
            finally:
                gc.collect()

            per_call_us = chain_ms * 1e3 / args.chain
            row = {
                "label": args.label,
                "scope": scope,
                "page_size": inp.page_size,
                "bq_sz": blocks[0],
                "bkv_sz": blocks[1],
                "bq_csz": blocks[2],
                "bkv_csz": blocks[3],
                "num_seqs": args.num_seqs,
                "kv_len": args.kv_len,
                "chain": args.chain,
                "compile_s": round(compile_s, 3),
                "chain_ms": round(chain_ms, 4),
                "per_call_us": round(per_call_us, 2),
                "per_step_ms": round(per_call_us * NUM_LAYERS / 1e3, 3),
                "kv_gbps": round(kv_bytes / (per_call_us / 1e6) / 1e9, 1),
                "contiguous_pages": bool(args.contiguous_pages),
            }
            if max_diff is not None:
                row["max_abs_diff"] = round(max_diff, 6)
                row["out_digest"] = digest
            _emit(row)


if __name__ == "__main__":
    main()
