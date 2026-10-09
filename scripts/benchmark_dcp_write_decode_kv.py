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
"""Microbenchmark for the KV write of the DCP decode single pass.

The DCP decode single pass stores each request's new tokens into the owning
rank's KV shard before one CACHE_ONLY pass. This compares ways of doing that
write for one DCP rank on one TPU (ranks are independent; no collective):

  torch_trash   pure torch index_put_; non-owner rows land on a trash page
  torch_rmw     pure torch gather + where + index_put_; no trash page
  write_new_kv  batched_rpa.write_new_kv (Pallas DMA), as served

The torch variants exist to show why the write is a kernel: XLA's scatter
copies the whole pool on every call, so their cost grows with the pool.
They cover HEAD_ALONG_SUBLANE with one token per request only.

Correctness is checked against NumPy for every rank. Cost is per layer: the
slope (t(L2) - t(L1)) / (L2 - L1) of one compiled graph writing L separate
caches, so dispatch and sync overhead cancel. Each case runs at two pool
sizes; a cost that grows with the pool means a pool-sized copy.

Usage:
  python scripts/benchmark_dcp_write_decode_kv.py
  python scripts/benchmark_dcp_write_decode_kv.py --layout=SAL --q-len=4 \
      --page-size=256 --batch-sizes=16,64,256
"""

from __future__ import annotations

import argparse
import functools
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from torch_tpu._internal import pallas as torch_pallas
from torch_tpu._internal import sync

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper
from vllm_torchtpu.kernels.experimental.batched_rpa.write_new_kv import write_new_kv

_TRASH_PAGE = 0  # vLLM's null block; never handed to a request.
_GROUP = 16  # Batches pad to this, as the runner pads requests.
_LAYOUTS = {
    "HAS": configs.KVLayout.HEAD_ALONG_SUBLANE,
    "SAL": configs.KVLayout.SEQ_ALONG_LANE,
}


# ---------------------------------------------------------------------------
# Pure torch (HAS, one token per request)
# ---------------------------------------------------------------------------


def _merge_kv_torch(key, value, kv_cache):
    """(T, H, D) K and V -> the HEAD_ALONG_SUBLANE row (T, H2/kp, kp, Da)."""
    num_tokens, num_heads, head_dim = key.shape
    _, _, heads_per_pack, packing, aligned_dim = kv_cache.shape
    kv = torch.cat([key, value], dim=-1).reshape(num_tokens, 2 * num_heads, head_dim)
    kv = F.pad(
        kv, (0, aligned_dim - head_dim, 0, heads_per_pack * packing - 2 * num_heads)
    )
    return kv.reshape(num_tokens, heads_per_pack, packing, aligned_dim)


def _decode_slots(num_tokens, seq_lens, block_tables, qsl, dist, *, args, rank):
    num_seqs = seq_lens.shape[0]
    pps = block_tables.shape[0] // num_seqs
    seq = torch.arange(num_seqs, dtype=torch.int32, device=seq_lens.device)
    pos = torch.clamp(seq_lens - 1, min=0)
    vpage = pos // args.page_size
    local_page = torch.clamp(vpage // args.dcp_size, max=pps - 1)
    page = block_tables[seq * pps + local_page]
    owner = (seq < dist[0]) & (vpage % args.dcp_size == rank)
    token = torch.clamp(qsl[:-1], max=num_tokens - 1)
    return page, pos % args.page_size, token, owner


def torch_trash_write(kv_cache, key, value, *md, args, rank):
    merged = _merge_kv_torch(key, value, kv_cache)
    page, offset, token, owner = _decode_slots(key.shape[0], *md, args=args, rank=rank)
    page = torch.where(owner, page, torch.full_like(page, _TRASH_PAGE))
    kv_cache.index_put_((page, offset), merged[token])


def torch_rmw_write(kv_cache, key, value, *md, args, rank):
    merged = _merge_kv_torch(key, value, kv_cache)
    page, offset, token, owner = _decode_slots(key.shape[0], *md, args=args, rank=rank)
    old = kv_cache[page, offset]
    kv_cache.index_put_(
        (page, offset), torch.where(owner[:, None, None, None], merged[token], old)
    )


# ---------------------------------------------------------------------------
# write_new_kv
# ---------------------------------------------------------------------------


@functools.cache
def _write_new_kv_op(dcp_size: int, rank: int, layout: str, q_len: int):
    def fn(
        kv_cache: jax.Array,
        key: jax.Array,
        value: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
    ) -> jax.Array:
        return write_new_kv(
            kv_cache,
            key,
            value,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            jnp.array([rank], jnp.int32),
            cp_group_size=dcp_size,
            kv_layout=_LAYOUTS[layout],
            max_q_len=q_len,
        )

    op = torch_pallas.jax_op(
        f"bench::write_new_kv_{layout}_q{q_len}_d{dcp_size}_r{rank}",
        fn,
        donate_argnums=(0,),
        mesh=None,
        input_partition_specs=None,
    )
    op.register_fake(lambda kv_cache, *args: torch.empty_like(kv_cache))
    return op


def _write_fn(variant, args, rank):
    if variant == "torch_trash":
        return functools.partial(torch_trash_write, args=args, rank=rank)
    if variant == "torch_rmw":
        return functools.partial(torch_rmw_write, args=args, rank=rank)
    # Build the op here: jax_op cannot be created while Dynamo traces.
    op = _write_new_kv_op(args.dcp_size, rank, args.layout, args.q_len)
    return lambda kv_cache, *a: kv_cache.copy_(op(kv_cache, *a))


def _variants(args):
    if args.layout == "HAS" and args.q_len == 1:
        return ("torch_trash", "torch_rmw", "write_new_kv")
    return ("write_new_kv",)


def _compiled_step(write):
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def step(caches, key, value, seq_lens, block_tables, qsl, distribution):
        for cache in caches:
            write(cache, key, value, seq_lens, block_tables, qsl, distribution)

    return step


# ---------------------------------------------------------------------------
# Inputs and reference
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


def _pages_needed(num_decode, kv_len, args):
    return num_decode * -(-kv_len // (args.page_size * args.dcp_size)) + 1


def _make_batch(num_decode, kv_len, args, *, num_pages, rng, random_lens):
    """Requests padded to a multiple of 16 with q_len new tokens each.

    Real requests get distinct random pages >= 1; padding rows have no tokens
    and point at the trash page, as the runner's padding does.
    """
    num_seqs = -(-num_decode // _GROUP) * _GROUP
    pps = -(-kv_len // (args.page_size * args.dcp_size))
    if random_lens:
        lens = rng.integers(args.q_len, kv_len + 1, num_decode)
    else:
        lens = np.full(num_decode, kv_len)
    seq_lens = np.zeros(num_seqs, np.int32)
    seq_lens[:num_decode] = lens
    tables = np.full((num_seqs, pps), _TRASH_PAGE, np.int32)
    tables[:num_decode] = rng.permutation(np.arange(1, num_pages))[
        : num_decode * pps
    ].reshape(num_decode, pps)
    qsl = (np.minimum(np.arange(num_seqs + 1), num_decode) * args.q_len).astype(
        np.int32
    )
    num_tokens = num_seqs * args.q_len
    shape = (num_tokens, args.local_kv_heads, args.head_dim)
    return dict(
        key=rng.normal(0, 1, shape).astype(np.float32),
        value=rng.normal(0, 1, shape).astype(np.float32),
        seq_lens=seq_lens,
        block_tables=tables.reshape(-1),
        query_start_loc=qsl,
        request_distribution=np.array([num_decode] * 3, np.int32),
    )


def _to_device(batch, device):
    out = [
        torch.from_numpy(batch[n]).to(torch.bfloat16).to(device)
        for n in ("key", "value")
    ]
    names = ("seq_lens", "block_tables", "query_start_loc", "request_distribution")
    return out + [torch.from_numpy(batch[n]).to(device) for n in names]


def _write_slot(cache, layout, page, offset, key, value):
    """One KV head, bf16, without the kernel's packing helpers."""
    if layout == "SAL":
        cache[page, 0, :, :, offset] = key.reshape(-1, 2)
        cache[page, 1, :, :, offset] = value.reshape(-1, 2)
    else:
        cache[page, offset] = 0
        cache[page, offset, 0, 0, :] = key
        cache[page, offset, 0, 1, :] = value


def _reference(cache, batch, *, rank, args):
    key = torch.from_numpy(batch["key"]).to(torch.bfloat16).float().numpy()
    value = torch.from_numpy(batch["value"]).to(torch.bfloat16).float().numpy()
    num_seqs = batch["seq_lens"].shape[0]
    pps = batch["block_tables"].shape[0] // num_seqs
    for s in range(int(batch["request_distribution"][0])):
        for j in range(args.q_len):
            pos = int(batch["seq_lens"][s]) - args.q_len + j
            vpage = pos // args.page_size
            if vpage % args.dcp_size != rank:
                continue
            tok = batch["query_start_loc"][s] + j
            page = batch["block_tables"][s * pps + vpage // args.dcp_size]
            _write_slot(
                cache,
                args.layout,
                page,
                pos % args.page_size,
                key[tok, 0],
                value[tok, 0],
            )
    return cache


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def check_correctness(args, device):
    rng = np.random.default_rng(0)
    kv_len = 3 * args.page_size * args.dcp_size + 5
    num_decode = 37
    num_pages = _pages_needed(num_decode, kv_len, args) + 8
    batch = _make_batch(
        num_decode, kv_len, args, num_pages=num_pages, rng=rng, random_lens=True
    )
    # Window ends on both sides of every page boundary, on every rank.
    edge = [
        p * args.page_size + d
        for p in range(1, 2 * args.dcp_size + 1)
        for d in (0, args.q_len // 2, args.q_len)
    ]
    n = min(len(edge), num_decode)
    batch["seq_lens"][:n] = np.minimum(edge[:n], kv_len)
    init = rng.normal(0, 1, _cache_shape(num_pages, args)).astype(np.float32)
    init = torch.from_numpy(init).to(torch.bfloat16)
    print("[correctness]", flush=True)
    for rank in range(args.dcp_size):
        expected = _reference(init.float().numpy(), batch, rank=rank, args=args)
        for variant in _variants(args):
            cache = init.to(device)
            _compiled_step(_write_fn(variant, args, rank))(
                [cache], *_to_device(batch, device)
            )
            sync.synchronize([cache], wait=True)
            actual = cache.cpu().float().numpy()
            if variant == "torch_trash":
                actual[_TRASH_PAGE] = expected[_TRASH_PAGE]
            ok = np.array_equal(actual, expected)
            if not ok or rank == 0:
                print(f"  rank={rank} {variant:<12s} {'OK' if ok else 'MISMATCH'}")
            if not ok:
                raise AssertionError(f"rank={rank} {variant} mismatch")
    print("  all ranks match", flush=True)


def _median_us(step, caches, inputs, iters):
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        step(caches, *inputs)
        sync.synchronize(caches, wait=True)
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def bench(args, device):
    rng = np.random.default_rng(1)
    l1, l2 = 1, args.layers
    variants = _variants(args)
    print(
        f"\n[per-layer cost, rank={args.rank}]  slope (t(L={l2}) - t(L={l1}))"
        f" / {l2 - l1}",
        flush=True,
    )
    header = f"  {'batch':>5s} {'pool_pages':>10s} {'pool_MiB':>9s}"
    print(header + "".join(f" {v:>12s}" for v in variants) + "   (us/layer)")
    for num_decode in args.batch_sizes:
        needed = _pages_needed(num_decode, args.kv_len, args)
        for scale in args.pool_scales:
            num_pages = needed * scale
            batch = _make_batch(
                num_decode,
                args.kv_len,
                args,
                num_pages=num_pages,
                rng=rng,
                random_lens=False,
            )
            inputs = _to_device(batch, device)
            shape = _cache_shape(num_pages, args)
            caches = [
                torch.zeros(shape, dtype=torch.bfloat16, device=device)
                for _ in range(l2)
            ]
            sync.synchronize(caches, wait=True)
            cells = []
            for variant in variants:
                write = _write_fn(variant, args, args.rank)
                times = {}
                for num_layers in (l1, l2):
                    step = _compiled_step(write)
                    layer_caches = caches[:num_layers]
                    for _ in range(args.warmup):
                        step(layer_caches, *inputs)
                    sync.synchronize(layer_caches, wait=True)
                    times[num_layers] = _median_us(
                        step, layer_caches, inputs, args.iters
                    )
                cells.append((times[l2] - times[l1]) / (l2 - l1))
            pool_mib = np.prod(shape) * 2 / 2**20
            print(
                f"  {num_decode:>5d} {num_pages:>10d} {pool_mib:>9.0f}"
                + "".join(f" {c:>12.1f}" for c in cells),
                flush=True,
            )
            del caches
    print(
        "\nA per-layer cost that grows with pool_pages means a pool-sized copy;"
        " compare the pool rows of each batch."
    )


def _int_list(s):
    return [int(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dcp-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--layout", choices=sorted(_LAYOUTS), default="HAS")
    parser.add_argument("--q-len", type=int, default=1, help="new tokens per request")
    parser.add_argument("--page-size", type=int, default=128, help="local page size")
    parser.add_argument("--local-kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--batch-sizes", type=_int_list, default=[16, 64, 256])
    parser.add_argument(
        "--pool-scales",
        type=_int_list,
        default=[1, 8],
        help="pool size as a multiple of the pages the batch needs",
    )
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()
    assert args.local_kv_heads == 1, "the reference packs one KV head"

    import torch_tpu  # noqa: F401

    # Every variant x layer count x shape is its own graph of one `step`
    # code object; lift Dynamo's per-code-object limits so none falls over.
    torch._dynamo.config.recompile_limit = 1024
    torch._dynamo.config.accumulated_recompile_limit = 4096
    device = torch.device("tpu")
    torch.empty((1,), device=device).cpu()
    print(
        f"dcp={args.dcp_size} layout={args.layout} q_len={args.q_len} "
        f"page_size={args.page_size} head_dim={args.head_dim} kv_len={args.kv_len}",
        flush=True,
    )
    if not args.skip_correctness:
        check_correctness(args, device)
    bench(args, device)


if __name__ == "__main__":
    main()
