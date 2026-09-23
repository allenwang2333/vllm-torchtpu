#!/usr/bin/env python3
"""RPA decode-kernel microbenchmark shaped like Qwen3.5-397B-A17B-FP8.

Drives batched_rpa_longctx.wrapper.ragged_paged_attention directly -- no model,
no executor, no scheduler -- so a kernel change shows up undiluted. Defaults are
one tensor-parallel rank of the AgentX single-host recipe: 32 q heads and 2 kv
heads over tp=8 give 4 q heads and 1 (replicated) kv head, head_dim 256,
page_size 128, fp8 kv cache.

Timing is a two-depth slope, not a wall clock around one call. Host dispatch
costs several times more than the kernel at these shapes, so `depth` calls are
chained inside a single jit graph (threading kv_cache through blocks CSE and
forces serialisation), two depths are timed, and the slope between them cancels
the constant per-dispatch overhead. Dividing a single chain by its depth instead
smears that overhead into every call and shrinks any ratio you compute.

Each row carries the pallas kernel signature it actually compiled, and --baseline
refuses to diff two rows whose signatures disagree.

    # sweep the speculative-decode buckets against the long-context ladder
    qwen35_rpa_bench.py

    # record a baseline, then diff a later run against it
    qwen35_rpa_bench.py --json /tmp/before.json
    qwen35_rpa_bench.py --baseline /tmp/before.json

    # one point, deeper chains
    qwen35_rpa_bench.py --query-sizes 8 --context-lens 131072 --depth-hi 32
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import configs, wrapper

# One tp=8 rank of the Qwen3.5-397B-A17B-FP8 recipe.
MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
HEAD_DIM = 256
PAGE_SIZE = 128
NUM_SEQS = 32
Q_DTYPE = jnp.bfloat16
KV_DTYPE = jnp.float8_e4m3fn
LAYOUT = configs.KVLayout.SEQ_ALONG_LANE

_SIGNATURE = re.compile(r"RPA[dmp]-[\w.-]+")


def _int_list(value: str) -> list[int]:
    return [int(v) for v in value.replace(" ", "").split(",") if v]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--query-sizes",
        type=_int_list,
        default=[1, 4, 8, 16],
        help="decode_query_size per step (1 = no spec decode, k+1 for MTP k)",
    )
    parser.add_argument(
        "--context-lens", type=_int_list, default=[8192, 32768, 65536, 98304]
    )
    parser.add_argument("--num-seqs", type=int, default=NUM_SEQS)
    parser.add_argument("--num-q-heads", type=int, default=NUM_Q_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=NUM_KV_HEADS)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    parser.add_argument("--depth-lo", type=int, default=4)
    parser.add_argument("--depth-hi", type=int, default=16)
    parser.add_argument("--iters", type=int, default=15, help="timed runs per depth")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--kv-dtype",
        default=KV_DTYPE.__name__,
        help="kv cache dtype, e.g. float8_e4m3fn or bfloat16",
    )
    parser.add_argument(
        "--stitch",
        default=configs.StitchStrategy.WIDE.name,
        choices=[s.name for s in configs.StitchStrategy],
        help="KV-stitch implementation; WIDE_BARRIER_ONLY prices the barrier alone",
    )
    parser.add_argument(
        "--placement",
        default=configs.StitchPlacement.AFTER_Q.name,
        choices=[p.name for p in configs.StitchPlacement],
        help="where the stitch sits in the step body",
    )
    parser.add_argument("--json", help="write the raw rows here")
    parser.add_argument(
        "--jsonl",
        help="append one self-describing row per line, for accumulating sweeps",
    )
    parser.add_argument("--baseline", help="a --json file to diff this run against")
    return parser.parse_args()


def _randn(rng: np.random.Generator, shape, dtype) -> np.ndarray:
    """Fill in float32 chunks.

    rng.standard_normal(shape) would materialise the whole KV cache as float64
    first -- tens of GB at long context -- which OOM-kills the pod.
    """
    out = np.empty(shape, dtype)
    flat = out.reshape(-1)
    for start in range(0, flat.size, 1 << 24):
        chunk = flat[start : start + (1 << 24)]
        chunk[:] = (rng.standard_normal(chunk.size, np.float32) * 0.5).astype(dtype)
    return out


def _build_inputs(args: argparse.Namespace, query_size: int, context_len: int) -> dict:
    """Synthesize a full-batch decode step at one (query_size, context) point.

    Every sequence carries `context_len` cached tokens plus `query_size` new
    ones, which is the steady state the decode path is tuned for. Cache contents
    are random and never read back, so correctness is not asserted here.
    """
    rng = np.random.default_rng(args.seed)
    kv_dtype = getattr(jnp, args.kv_dtype)
    num_seqs = args.num_seqs
    kv_len = context_len + query_size
    total_tokens = num_seqs * query_size
    pages_per_seq = -(-kv_len // args.page_size) + 1
    num_pages = num_seqs * pages_per_seq + 2

    cache_shape = wrapper.get_kv_cache_shape(
        num_pages,
        args.page_size,
        args.num_kv_heads,
        args.head_dim,
        kv_dtype,
        kv_layout=LAYOUT,
    )
    page_table = rng.permutation(num_pages)[: num_seqs * pages_per_seq]

    return {
        "queries": jnp.asarray(
            _randn(rng, (total_tokens, args.num_q_heads, args.head_dim), Q_DTYPE)
        ),
        "keys": jnp.asarray(
            _randn(rng, (total_tokens, args.num_kv_heads, args.head_dim), kv_dtype)
        ),
        "values": jnp.asarray(
            _randn(rng, (total_tokens, args.num_kv_heads, args.head_dim), kv_dtype)
        ),
        "kv_cache": jnp.asarray(_randn(rng, cache_shape, kv_dtype)),
        "kv_lens": jnp.full(num_seqs, kv_len, dtype=jnp.int32),
        "page_indices": jnp.asarray(page_table.astype(np.int32)),
        "cu_q_lens": jnp.arange(num_seqs + 1, dtype=jnp.int32) * query_size,
        # (num_decode_seqs, num_decode_seqs, num_seqs): everything is decode.
        "distribution": jnp.full(3, num_seqs, dtype=jnp.int32),
    }


def _make_call(args: argparse.Namespace, query_size: int):
    def call(**kw):
        return wrapper.ragged_paged_attention(
            kw["queries"],
            kw["keys"],
            kw["values"],
            kw["kv_cache"],
            kw["kv_lens"],
            kw["page_indices"],
            kw["cu_q_lens"],
            kw["distribution"],
            sm_scale=args.head_dim**-0.5,
            kv_layout=LAYOUT,
            decode_query_size=query_size,
            stitch_strategy=configs.StitchStrategy[args.stitch],
            stitch_placement=configs.StitchPlacement[args.placement],
        )

    return call


def _make_chain(call, depth: int):
    """Chain `depth` calls in one graph, each fed the previous cache."""

    def chained(**kw):
        kw = dict(kw)
        out = None
        for _ in range(depth):
            out, kw["kv_cache"] = call(**kw)
        return out, kw["kv_cache"]

    return chained


def _time(call, inputs: dict, iters: int, warmup: int) -> list[float]:
    fn = jax.jit(call)
    for _ in range(warmup):
        jax.block_until_ready(fn(**inputs))
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        jax.block_until_ready(fn(**inputs))
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def _measure(args: argparse.Namespace, query_size: int, context_len: int) -> dict:
    inputs = _build_inputs(args, query_size, context_len)
    call = _make_call(args, query_size)

    found = _SIGNATURE.findall(jax.jit(call).lower(**inputs).as_text())
    span = args.depth_hi - args.depth_lo
    lo = _time(_make_chain(call, args.depth_lo), inputs, args.iters, args.warmup)
    hi = _time(_make_chain(call, args.depth_hi), inputs, args.iters, args.warmup)

    t_lo, t_hi = statistics.median(lo), statistics.median(hi)
    per_call = (t_hi - t_lo) / span
    # Read after the timed runs so the peak covers compilation scratch too.
    stats = jax.local_devices()[0].memory_stats() or {}
    return {
        "q": query_size,
        "kv": context_len,
        "cache_gib": inputs["kv_cache"].nbytes / 2**30,
        "input_gib": sum(v.nbytes for v in inputs.values()) / 2**30,
        "hbm_peak_gib": stats.get("peak_bytes_in_use", 0) / 2**30,
        "hbm_limit_gib": stats.get("bytes_limit", 0) / 2**30,
        "signature": sorted(set(found))[0] if found else "<unknown>",
        "per_call_ms": per_call,
        # Config travels with every row so a jsonl accumulated across runs is
        # self-describing.
        "stitch": args.stitch,
        "placement": args.placement,
        "num_seqs": args.num_seqs,
        "page_size": args.page_size,
        "head_dim": args.head_dim,
        "num_q_heads": args.num_q_heads,
        "num_kv_heads": args.num_kv_heads,
        "kv_dtype": args.kv_dtype,
        # Same slope taken on the per-depth minima: a second, noise-floor
        # estimate. Far from per_call_ms means the run was disturbed.
        "per_call_min_ms": (min(hi) - min(lo)) / span,
        "overhead_ms": t_lo - per_call * args.depth_lo,
        "t_lo_ms": t_lo,
        "t_hi_ms": t_hi,
        "depths": [args.depth_lo, args.depth_hi],
        "n": len(hi),
    }


def _print_plain(rows: list[dict]) -> None:
    print(
        f"{'q':>4} {'kv':>8} {'ms/call':>9} {'(min)':>9} {'cacheGiB':>9} "
        f"{'peakGiB':>8}  signature"
    )
    for row in rows:
        print(
            f"{row['q']:>4} {row['kv']:>8} {row['per_call_ms']:>9.4f} "
            f"{row['per_call_min_ms']:>9.4f} {row['cache_gib']:>9.2f} "
            f"{row['hbm_peak_gib']:>8.2f}  {row['signature']}"
        )


def _strategy_tag(signature: str) -> tuple[str, str]:
    """Split a signature into its shape part and its stitch-strategy tag."""
    known = {s.symbol for s in configs.StitchStrategy}
    known |= {p.symbol for p in configs.StitchPlacement}
    tags = []
    while True:
        head, _, tail = signature.rpartition("-")
        if tail not in known:
            break
        tags.insert(0, tail)
        signature = head
    return signature, "+".join(tags) or "default"


def _print_diff(rows: list[dict], baseline: list[dict]) -> None:
    """Diff against a baseline.

    Signatures must match on everything but the stitch-strategy tag, which is
    the one axis this is meant to sweep.
    """
    before = {(r["q"], r["kv"]): r for r in baseline}
    print(
        f"{'q':>4} {'kv':>8} {'before':>9} {'after':>9} {'ratio':>7} "
        f"{'saved':>8}  signature"
    )
    for row in rows:
        prev = before.get((row["q"], row["kv"]))
        if prev is None:
            print(
                f"{row['q']:>4} {row['kv']:>8} {'---':>9} "
                f"{row['per_call_ms']:>9.4f}  (no baseline row)"
            )
            continue
        prev_shape, prev_tag = _strategy_tag(prev["signature"])
        shape, tag = _strategy_tag(row["signature"])
        if prev_shape != shape:
            print(
                f"{row['q']:>4} {row['kv']:>8}  SIGNATURE MISMATCH  "
                f"{prev['signature']} -> {row['signature']}"
            )
            continue
        saved = prev["per_call_ms"] - row["per_call_ms"]
        print(
            f"{row['q']:>4} {row['kv']:>8} {prev['per_call_ms']:>9.4f} "
            f"{row['per_call_ms']:>9.4f} "
            f"{prev['per_call_ms'] / row['per_call_ms']:>6.2f}x {saved:>8.3f}  "
            f"{shape} {prev_tag}->{tag}"
        )


def main() -> None:
    args = _parse_args()
    print(
        f"model={MODEL} q_heads={args.num_q_heads} kv_heads={args.num_kv_heads} "
        f"head_dim={args.head_dim} page_size={args.page_size} "
        f"kv_dtype={args.kv_dtype} num_seqs={args.num_seqs} "
        f"depths={args.depth_lo}/{args.depth_hi} iters={args.iters} "
        f"stitch={args.stitch}/{args.placement}"
    )
    print(f"wrapper={wrapper.__file__}")

    rows = []
    for query_size in args.query_sizes:
        for context_len in args.context_lens:
            try:
                rows.append(_measure(args, query_size, context_len))
            except Exception as exc:  # noqa: BLE001
                print(
                    f"{query_size:>4} {context_len:>8}  FAILED: "
                    f"{type(exc).__name__}: {exc}"[:200]
                )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)

    if args.jsonl:
        with open(args.jsonl, "a") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    if args.baseline:
        with open(args.baseline) as handle:
            _print_diff(rows, json.load(handle))
    else:
        _print_plain(rows)


if __name__ == "__main__":
    main()
