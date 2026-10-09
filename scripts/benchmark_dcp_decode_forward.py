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
"""Before/after benchmark of DCP attention on all-decode steps.

before  two passes, as forward_with_dcp runs by default: CACHE_ONLY over the
        all-gathered Q, then NEW_TOKENS_ONLY over own Q (it writes the new
        tokens), then merge_attn_results
after   single pass (TPU_DCP_RPA_DECODE_SINGLE_PASS): write_new_kv, then one CACHE_ONLY
        pass with skip_kv_update over the all-gathered Q

Both run the same collectives with the same shapes (all-gather Q, then the
AG-RS LSE merge of the CACHE_ONLY output), so the per-rank work above is the
whole difference and one TPU measures it. The kernels are the production
cp_attention ops. Requests carry --q-len new tokens: 1 for decode, K + 1 for
speculative verify windows.

Correctness emulates every DCP rank on one TPU, does the cross-rank LSE
merge on the host, and checks that both paths give the same output and the
same caches. Cost is per layer: the slope (t(L2) - t(L1)) / (L2 - L1) of one
compiled graph running L layers, each with its own cache.

Usage:
  python scripts/benchmark_dcp_decode_forward.py
  # Q=64, KV=4, tp=8 -> dcp<=2; MQA Q=64, KV=1, tp=8 -> dcp<=8
  python scripts/benchmark_dcp_decode_forward.py --dcp-size=8 --q-heads=64 \
      --layout=SAL --page-size=256 --q-len=4
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal import sync

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

_GROUP = 16
_LAYOUTS = {
    "HAS": configs.KVLayout.HEAD_ALONG_SUBLANE,
    "SAL": configs.KVLayout.SEQ_ALONG_LANE,
}


def _load_cp_attention():
    """Import cp_attention without `vllm_torchtpu.layers.__init__`, which pulls
    the whole attention adapter and with it vLLM APIs this benchmark does not
    need (and a mismatched vLLM may not have)."""
    import importlib.util
    import pathlib

    import vllm_torchtpu

    path = (
        pathlib.Path(vllm_torchtpu.__file__).parent / "layers/adapter/cp_attention.py"
    )
    spec = importlib.util.spec_from_file_location("_bench_cp_attention", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # Dynamo resolves globals through it
    spec.loader.exec_module(module)
    return module


_cp_attention = _load_cp_attention()


def _kernels(args, rank):
    config = dict(
        sliding_window=None,
        sm_scale=args.head_dim**-0.5,
        logits_soft_cap=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        cp_group_size=args.dcp_size,
        cp_rank=rank,
        kv_layout=_LAYOUTS[args.layout],
        decode_query_size=args.q_len,
    )
    cache_kernel, new_kernel = _cp_attention.build_dcp_kernels(**config)
    return (
        cache_kernel,
        new_kernel,
        _cp_attention.build_dcp_decode_single_pass_kernel(**config),
    )


def before_layer(kernels, own, kv_cache, q_all, key, value, *md):
    """Per-rank part of the two-pass path; returns the pre-merge pieces."""
    cache_kernel, new_kernel, _ = kernels
    ctx_out, ctx_lse = cache_kernel(kv_cache, q_all, key, value, *md, None, None, None)
    q_own = q_all[:, own].contiguous()
    new_out, new_lse = new_kernel(kv_cache, q_own, key, value, *md, None, None, None)
    return ctx_out, ctx_lse, new_out, new_lse


def after_layer(kernels, kv_cache, q_all, key, value, *md):
    return kernels[2](kv_cache, q_all, key, value, *md, None, None, None)


def _step(args, rank, mode):
    kernels = _kernels(args, rank)
    own = slice(rank * args.own_q_heads, (rank + 1) * args.own_q_heads)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def step(caches, q_all, key, value, seq_lens, block_tables, qsl, dist):
        md = (seq_lens, block_tables, qsl, dist)
        outs = []
        for cache in caches:
            if mode == "before":
                ctx_out, ctx_lse, new_out, new_lse = before_layer(
                    kernels, own, cache, q_all, key, value, *md
                )
                # Stand-in for the AG-RS result (same in both paths): this
                # rank's heads of its own CACHE_ONLY partial.
                out, _ = _cp_attention.merge_attn_results(
                    ctx_out[:, own], ctx_lse[:, own].float(), new_out, new_lse
                )
                outs.append(out.to(new_out.dtype))
            else:
                out, _ = after_layer(kernels, cache, q_all, key, value, *md)
                outs.append(out)
        return outs

    return step


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _cache_shape(num_pages, args):
    return wrapper.get_kv_cache_shape(
        num_pages,
        args.page_size,
        args.local_kv_heads,
        args.head_dim,
        jnp.bfloat16,
        kv_layout=_LAYOUTS[args.layout],
    )


def _make_batch(num_decode, kv_len, args, rng, *, random_lens):
    num_seqs = -(-num_decode // _GROUP) * _GROUP
    pps = -(-kv_len // (args.page_size * args.dcp_size))
    num_pages = num_decode * pps + 1
    lens = (
        rng.integers(args.q_len, kv_len + 1, num_decode)
        if random_lens
        else np.full(num_decode, kv_len)
    )
    seq_lens = np.zeros(num_seqs, np.int32)
    seq_lens[:num_decode] = lens
    tables = np.zeros((num_seqs, pps), np.int32)
    tables[:num_decode] = rng.permutation(np.arange(1, num_pages)).reshape(
        num_decode, pps
    )
    qsl = (np.minimum(np.arange(num_seqs + 1), num_decode) * args.q_len).astype(
        np.int32
    )
    dist = np.array([num_decode] * 3, np.int32)
    num_tokens = num_seqs * args.q_len

    def normal(*shape):
        data = rng.normal(0, 1, shape).astype(np.float32)
        return torch.from_numpy(data).to(torch.bfloat16)

    tensors = [
        normal(num_tokens, args.q_heads, args.head_dim),
        normal(num_tokens, args.local_kv_heads, args.head_dim),
        normal(num_tokens, args.local_kv_heads, args.head_dim),
    ] + [torch.from_numpy(x) for x in (seq_lens, tables.reshape(-1), qsl, dist)]
    return tensors, num_pages


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def _lse_merge(outs, lses):
    """Cross-rank merge, as cp_lse_ag_out_rs does before its reduce-scatter."""
    lses = torch.stack([lse.float() for lse in lses])
    merged_lse = torch.logsumexp(lses, dim=0)
    weight = torch.exp(lses - merged_lse).unsqueeze(-1)
    weighted = torch.stack([o.float() for o in outs]) * weight
    weighted = torch.where(weight > 0, weighted, torch.zeros_like(weighted))
    return weighted.sum(0), merged_lse


def _live_slots(inputs, rank, num_pages, args):
    """(page, offset) slots of `rank` that hold a token below its sequence's
    length, plus every slot of pages no sequence owns. Slots at or past a
    sequence's length are dead: the paths may leave different bytes there
    (the two-pass SEQ_ALONG_LANE write-back zeroes the lanes after the new
    token), and every pass masks them by length."""
    seq_lens, tables = inputs[3].numpy(), inputs[4].numpy()
    pps = tables.shape[0] // seq_lens.shape[0]
    live = np.zeros((num_pages, args.page_size), bool)
    owned = np.zeros(num_pages, bool)
    for s, length in enumerate(seq_lens):
        for vpage in range(rank, pps * args.dcp_size, args.dcp_size):
            if length == 0:
                break
            page = tables[s * pps + vpage // args.dcp_size]
            owned[page] = True
            start = vpage * args.page_size
            live[page, : max(0, min(length - start, args.page_size))] = True
    live[~owned] = True
    return live


def _by_slot(cache, args):
    """Cache as (pages, page_size, ...) whatever the layout."""
    if args.layout == "SAL":
        return cache.movedim(-1, 1)
    return cache


def check_correctness(args, device):
    rng = np.random.default_rng(0)
    kv_len = 3 * args.page_size * args.dcp_size + 5
    inputs, num_pages = _make_batch(37, kv_len, args, rng, random_lens=True)
    # Window ends on both sides of every page boundary, on every rank.
    edge = [
        p * args.page_size + d
        for p in range(1, 2 * args.dcp_size + 1)
        for d in (0, args.q_len // 2, args.q_len)
    ]
    n = min(len(edge), 37)
    inputs[3][:n] = torch.tensor(np.minimum(edge[:n], kv_len), dtype=torch.int32)
    num_tokens = 37 * args.q_len
    dev_inputs = [t.to(device) for t in inputs]
    # One cache per rank. Random contents: every slot not yet written is
    # masked by length in both paths, so garbage there must not matter.
    shape = _cache_shape(num_pages, args)
    init = [
        torch.from_numpy(rng.normal(0, 1, shape).astype(np.float32)).to(torch.bfloat16)
        for _ in range(args.dcp_size)
    ]

    results = {}
    for mode in ("before", "after"):
        pieces, caches = [], []
        for rank in range(args.dcp_size):
            cache = init[rank].to(device)
            kernels = _kernels(args, rank)
            if mode == "before":
                own = slice(rank * args.own_q_heads, (rank + 1) * args.own_q_heads)
                out = before_layer(kernels, own, cache, *dev_inputs)
            else:
                out = after_layer(kernels, cache, *dev_inputs)
            sync.synchronize([cache, *out], wait=True)
            pieces.append([t.cpu() for t in out])
            caches.append(cache.cpu())
        ctx, ctx_lse = _lse_merge([p[0] for p in pieces], [p[1] for p in pieces])
        if mode == "before":
            final = []
            for rank, p in enumerate(pieces):
                own = slice(rank * args.own_q_heads, (rank + 1) * args.own_q_heads)
                final.append(
                    _cp_attention.merge_attn_results(
                        ctx[:, own], ctx_lse[:, own], p[2].float(), p[3].float()
                    )[0]
                )
            final = torch.cat(final, dim=1)
        else:
            final = ctx
        results[mode] = (final[:num_tokens], caches)

    max_diff = float((results["before"][0] - results["after"][0]).abs().max())
    live_equal, dead_diffs = True, 0
    for rank, (b, a) in enumerate(zip(results["before"][1], results["after"][1])):
        live = torch.from_numpy(_live_slots(inputs, rank, num_pages, args))
        differs = (_by_slot(b, args) != _by_slot(a, args)).flatten(2).any(-1)
        live_equal &= not bool(differs[live].any())
        dead_diffs += int(differs[~live].sum())
    print(
        f"[correctness] dcp={args.dcp_size} layout={args.layout} q_len={args.q_len}"
        f" max|before-after|={max_diff:.4g} live_cache_equal={live_equal}"
        f" dead_slots_differing={dead_diffs}",
        flush=True,
    )
    if max_diff > 3e-2 or not live_equal:
        raise AssertionError("before and after disagree")


def _median_us(step, caches, inputs, iters):
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        outs = step(caches, *inputs)
        sync.synchronize([*caches, *outs], wait=True)
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def bench(args, device):
    rng = np.random.default_rng(1)
    l1, l2 = 1, args.layers
    print(
        f"\n[per-layer cost, rank={args.rank}]  slope (t(L={l2}) - t(L={l1}))"
        f" / {l2 - l1}",
        flush=True,
    )
    print(
        f"  {'kv_len':>7s} {'batch':>5s} {'before_us':>10s} {'after_us':>10s}"
        f" {'speedup':>8s}"
    )
    for kv_len in args.kv_lens:
        for num_decode in args.batch_sizes:
            inputs, num_pages = _make_batch(
                num_decode, kv_len, args, rng, random_lens=False
            )
            inputs = [t.to(device) for t in inputs]
            shape = _cache_shape(num_pages, args)
            caches = [
                torch.zeros(shape, dtype=torch.bfloat16, device=device)
                for _ in range(l2)
            ]
            sync.synchronize(caches, wait=True)
            per_layer = {}
            for mode in ("before", "after"):
                times = {}
                for num_layers in (l1, l2):
                    step = _step(args, args.rank, mode)
                    layer_caches = caches[:num_layers]
                    for _ in range(args.warmup):
                        outs = step(layer_caches, *inputs)
                    sync.synchronize([*layer_caches, *outs], wait=True)
                    times[num_layers] = _median_us(
                        step, layer_caches, inputs, args.iters
                    )
                per_layer[mode] = (times[l2] - times[l1]) / (l2 - l1)
            b, a = per_layer["before"], per_layer["after"]
            print(
                f"  {kv_len:>7d} {num_decode:>5d} {b:>10.1f} {a:>10.1f} {b / a:>7.2f}x",
                flush=True,
            )
            del caches


def _int_list(s):
    return [int(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dcp-size", type=int, default=2)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--layout", choices=sorted(_LAYOUTS), default="HAS")
    parser.add_argument("--q-len", type=int, default=1, help="new tokens per request")
    parser.add_argument(
        "--q-heads",
        type=int,
        default=16,
        help="Q heads after the all-gather: total_q_heads // (tp // dcp)",
    )
    parser.add_argument(
        "--local-kv-heads",
        type=int,
        default=1,
        help="vLLM caps GQA dcp at tp // total_kv_heads, so this is 1",
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=128, help="local page size")
    parser.add_argument("--kv-lens", type=_int_list, default=[8192, 32768, 131072])
    parser.add_argument("--batch-sizes", type=_int_list, default=[16, 64, 256])
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()
    assert args.q_heads % args.dcp_size == 0
    # The NEW_TOKENS_ONLY pass gets this rank's own Q heads with the same
    # local K/V; its GQA grouping only matches the global one when the rank
    # holds a single KV head, which vLLM's config check guarantees.
    assert args.local_kv_heads == 1, "vLLM GQA DCP has one KV head per rank"
    args.own_q_heads = args.q_heads // args.dcp_size

    import torch_tpu  # noqa: F401

    torch._dynamo.config.recompile_limit = 1024
    torch._dynamo.config.accumulated_recompile_limit = 4096
    device = torch.device("tpu")
    torch.empty((1,), device=device).cpu()
    print(
        f"dcp={args.dcp_size} layout={args.layout} q_len={args.q_len}"
        f" q_heads(all)={args.q_heads} own_q={args.own_q_heads}"
        f" head_dim={args.head_dim} page_size={args.page_size}",
        flush=True,
    )
    if not args.skip_correctness:
        check_correctness(args, device)
    bench(args, device)


if __name__ == "__main__":
    main()
