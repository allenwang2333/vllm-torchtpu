# SPDX-License-Identifier: Apache-2.0
"""A/B numerical parity for ``VLLM_RPA_SKIP_EMPTY_PASS``.

`batched_rpa/wrapper.py` used to launch both RPA passes -- DECODE over
``[0, distribution[0])`` and MIXED over ``[distribution[1], distribution[2])``
-- unconditionally on every step. It now wraps each in a ``jax.lax.cond`` on
its own range being non-empty, so a decode-only step no longer launches MIXED
and a prefill-only step no longer launches DECODE.

Skipping a pass must be *invisible*: for every batch composition, the attention
output and the resulting KV cache must be identical to what the unconditional
path produced. This script asserts exactly that, bit for bit.

Why a script and not a pytest case
----------------------------------
``VLLM_RPA_SKIP_EMPTY_PASS`` is read at JAX trace time and baked into the
compiled kernel, so one process cannot hold both arms. Each arm is therefore a
separate process that dumps its results, and a third invocation compares them.

    VLLM_RPA_SKIP_EMPTY_PASS=0 python scripts/check_batched_rpa_skip_empty_parity.py --emit /tmp/arm0.npz
    VLLM_RPA_SKIP_EMPTY_PASS=1 python scripts/check_batched_rpa_skip_empty_parity.py --emit /tmp/arm1.npz
    python scripts/check_batched_rpa_skip_empty_parity.py --compare /tmp/arm0.npz /tmp/arm1.npz

What the cases are for
----------------------
``distribution`` is ``[decode_end, prefill_end, total]``; decode-only sequences
lead the batch. Each case makes a different pair of predicates fire:

    case              distribution   DECODE pass   MIXED pass
    decode_only       [8, 8, 8]      runs          ELIDED
    prefill_only_1    [0, 0, 1]      ELIDED        runs
    prefill_only_n    [0, 0, 3]      ELIDED        runs
    mixed             [4, 4, 8]      runs          runs
    all_empty         [0, 0, 0]      ELIDED        ELIDED

`decode_only` fails if the MIXED predicate is wrong; `prefill_only_*` fail if
the DECODE predicate is wrong. `mixed` is the case the production workload
actually runs and the one no existing test covers: both conds take the true
branch and the KV cache is threaded through *both* of them, which is where the
commit's own stated risk lives -- the true branch's custom call aliases the
cache in place, and two chained conds over a multi-hundred-megabyte buffer are
where XLA is most likely to either insert a copy or feed the second pass a
stale buffer. `all_empty` pins the degenerate batch.

Model geometry is inherited from ``microbench_batched_rpa_kv_layout.py`` rather
than restated, so the two cannot drift; that module's header records the HF
revision the counts were verified against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Verified production geometry: NUM_Q_HEADS_PER_RANK, NUM_KV_HEADS_PER_RANK,
# HEAD_DIM, MAX_NUM_SEQS, MAX_MODEL_LEN, DECODE_KV_LEN. See that module's
# header for the HF revision they were checked against.
import microbench_batched_rpa_kv_layout as mb  # noqa: E402

# The nightly config (scripts/vllm/benchmarking/configs/qwen3-coder-30b-fp8-
# tp8-ep.sh) sets no VLLM_KV_CACHE_LAYOUT, so vLLM resolves the default "NHD",
# which KV_LAYOUT_BY_VLLM_LAYOUT maps to HEAD_ALONG_SUBLANE at page_size 256 --
# the same layout and page size as the decode microbench's quoted
# (2304, 256, 1, 4, 128) fp8 cache.
DEFAULT_LAYOUT = "head_along_sublane"

# Chunked-prefill chunk used for the prefill and mixed cases. Smaller than the
# 8192-token single-chunk case the microbench times, because a realistic
# chunked-prefill step at this concurrency carries several partial prefills
# alongside the decodes rather than one maximal chunk.
PREFILL_CHUNK = 2048
PREFILL_CONTEXT = 2048


def _raw(arr) -> bytes:
    """Contiguous raw bytes of an array, dtype-agnostic (fp8/bf16 included)."""
    return np.ascontiguousarray(arr).tobytes()


def _sha(arr) -> str:
    return hashlib.sha256(_raw(arr)).hexdigest()


def _build_case(jnp, jax, name: str, *, page_size: int, pages_per_seq: int,
                max_num_seqs: int, decode_kv_len: int, kv_dtype, q_dtype,
                cache_shape, seed: int):
    """Build one step's kernel arguments for case ``name``.

    Mirrors the runner's padding conventions the same way
    ``microbench_batched_rpa_kv_layout._build_phase_inputs`` does: metadata is
    padded to ``max_num_seqs``, ``page_indices`` is the flattened block table,
    and ``kv_lens`` counts the new tokens as well as the cached ones.
    """
    if name == "decode_only":
        n_dec, n_pre = 8, 0
    elif name == "prefill_only_1":
        n_dec, n_pre = 0, 1
    elif name == "prefill_only_n":
        n_dec, n_pre = 0, 3
    elif name == "mixed":
        n_dec, n_pre = 4, 4
    elif name == "all_empty":
        n_dec, n_pre = 0, 0
    else:
        raise ValueError(f"unknown case {name!r}")

    if name == "all_empty":
        # One sequence is still described so the query tensor is non-empty;
        # distribution [0, 0, 0] is what excludes it from every range.
        q_lens = [1]
        kv_lens_list = [page_size]
        distribution = [0, 0, 0]
    else:
        # Decode-only sequences must lead the batch: `distribution` slices the
        # batch by position, it does not permute it.
        q_lens = [1] * n_dec + [PREFILL_CHUNK] * n_pre
        kv_lens_list = ([decode_kv_len] * n_dec +
                        [PREFILL_CONTEXT + PREFILL_CHUNK] * n_pre)
        distribution = [n_dec, n_dec, n_dec + n_pre]

    num_seqs = len(q_lens)
    total_q_tokens = sum(q_lens)
    for kv_len in kv_lens_list:
        if kv_len > pages_per_seq * page_size:
            raise SystemExit(
                f"{name}: kv_len={kv_len} exceeds "
                f"pages_per_seq*page_size={pages_per_seq * page_size}")

    kv_lens = np.zeros((max_num_seqs, ), dtype=np.int32)
    kv_lens[:num_seqs] = kv_lens_list

    cu_q_lens = np.full((max_num_seqs + 1, ), total_q_tokens, dtype=np.int32)
    cu_q_lens[0] = 0
    cu_q_lens[1:num_seqs + 1] = np.cumsum(q_lens, dtype=np.int32)

    page_indices = np.arange(max_num_seqs * pages_per_seq, dtype=np.int32)

    # Fixed key -> the two arms see bit-identical inputs. This is the whole
    # basis of the comparison, so it is derived from the case name rather than
    # from iteration order -- and via sha256, not `hash()`, which is salted
    # per process and would silently desync the two arms.
    name_salt = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],
                               "big") % 10_000
    key = jax.random.key(seed + name_salt)
    k_q, k_kv, k_cache = jax.random.split(key, 3)
    query = (jax.random.normal(k_q,
                               (total_q_tokens, mb.NUM_Q_HEADS_PER_RANK,
                                mb.HEAD_DIM),
                               dtype=jnp.float32) * 0.5).astype(q_dtype)
    new_kv = (jax.random.normal(k_kv,
                                (2, total_q_tokens, mb.NUM_KV_HEADS_PER_RANK,
                                 mb.HEAD_DIM),
                                dtype=jnp.float32) * 0.5).astype(kv_dtype)
    # A random cache, not a zero one: a zero cache would make a dropped read
    # indistinguishable from a correct one over the untouched region.
    kv_cache = (jax.random.normal(k_cache, cache_shape, dtype=jnp.float32) *
                0.5).astype(kv_dtype)

    args = dict(
        kv_lens=jnp.asarray(kv_lens),
        page_indices=jnp.asarray(page_indices),
        cu_q_lens=jnp.asarray(cu_q_lens),
        distribution=jnp.asarray(np.asarray(distribution, dtype=np.int32)),
    )
    meta = dict(num_seqs=num_seqs,
                n_decode=n_dec,
                n_prefill=n_pre,
                distribution=distribution,
                total_q_tokens=total_q_tokens,
                live_pages=num_seqs * pages_per_seq)
    return query, new_kv[0], new_kv[1], kv_cache, args, meta


def _emit(args) -> int:
    arm = os.environ.get("VLLM_RPA_SKIP_EMPTY_PASS", "1")
    # Export before importing the wrapper: the layout resolver is lru_cache-d.
    mb._export_vllm_layout_env(args.layout)
    import jax
    import jax.numpy as jnp
    wrapper, configs = mb._import_kernel()

    kv_layout = getattr(configs.KVLayout, args.layout.upper())
    page_size = mb._resolve_page_size(args.layout, args.page_size)
    pages_per_seq = math.ceil(args.max_model_len / page_size)
    total_pages = args.max_num_seqs * pages_per_seq
    q_dtype, kv_dtype = jnp.bfloat16, jnp.float8_e4m3fn

    cache_shape = wrapper.get_kv_cache_shape(total_pages, page_size,
                                             mb.NUM_KV_HEADS_PER_RANK,
                                             mb.HEAD_DIM, kv_dtype,
                                             kv_layout=kv_layout)
    bytes_per_token = mb._bytes_per_token(cache_shape, page_size,
                                          jnp.dtype(kv_dtype).itemsize)
    expected = mb.EXPECTED_BYTES_PER_TOKEN[args.layout]
    if bytes_per_token != expected:
        raise SystemExit(
            f"FATAL: resolved {bytes_per_token} B/token from "
            f"kv_cache_shape={cache_shape}, expected {expected}. The layout "
            "did not take effect; refusing to emit.")

    step = mb._make_step(jax,
                         wrapper,
                         kv_layout=kv_layout,
                         sm_scale=mb.HEAD_DIM**-0.5,
                         q_scale=None,
                         k_scale=None,
                         v_scale=None)

    print(f"arm VLLM_RPA_SKIP_EMPTY_PASS={arm}  layout={args.layout}  "
          f"page_size={page_size}  cache_shape={tuple(cache_shape)} "
          f"{np.dtype(kv_dtype).name}  bytes/token={bytes_per_token:.0f}",
          flush=True)

    out_blob: dict[str, np.ndarray] = {}
    manifest: dict[str, dict] = {}
    for name in args.cases:
        q, k, v, cache, kargs, meta = _build_case(jnp,
                                                  jax,
                                                  name,
                                                  page_size=page_size,
                                                  pages_per_seq=pages_per_seq,
                                                  max_num_seqs=args.max_num_seqs,
                                                  decode_kv_len=args.decode_kv_len,
                                                  kv_dtype=kv_dtype,
                                                  q_dtype=q_dtype,
                                                  cache_shape=cache_shape,
                                                  seed=args.seed)
        in_cache_sha = _sha(jax.device_get(cache))
        out, new_cache = step(q, k, v, cache, kargs["kv_lens"],
                              kargs["page_indices"], kargs["cu_q_lens"],
                              kargs["distribution"])
        out_h = jax.device_get(out)
        cache_h = jax.device_get(new_cache)

        # The live pages are a contiguous prefix because page_indices is
        # arange(); everything outside is covered by the whole-cache digest.
        live = min(meta["live_pages"], cache_h.shape[0])
        out_blob[f"{name}__out"] = np.asarray(out_h).astype(np.float32)
        out_blob[f"{name}__cache_live"] = np.frombuffer(_raw(cache_h[:live]),
                                                        dtype=np.uint8)
        manifest[name] = dict(
            distribution=meta["distribution"],
            n_decode=meta["n_decode"],
            n_prefill=meta["n_prefill"],
            total_q_tokens=meta["total_q_tokens"],
            live_pages=live,
            in_cache_sha=in_cache_sha,
            out_sha=_sha(out_h),
            out_shape=list(np.asarray(out_h).shape),
            out_dtype=str(np.asarray(out_h).dtype),
            cache_sha=_sha(cache_h),
            out_finite=bool(np.isfinite(np.asarray(out_h,
                                                   dtype=np.float32)).all()),
            out_std=float(np.asarray(out_h, dtype=np.float32).std()),
        )
        m = manifest[name]
        print(f"  [{name}] dist={m['distribution']} "
              f"q_tokens={m['total_q_tokens']} out_sha={m['out_sha'][:16]} "
              f"cache_sha={m['cache_sha'][:16]} std={m['out_std']:.6f} "
              f"finite={m['out_finite']}",
              flush=True)

    meta_blob = dict(arm=arm,
                     layout=args.layout,
                     page_size=page_size,
                     cache_shape=list(map(int, cache_shape)),
                     max_num_seqs=args.max_num_seqs,
                     max_model_len=args.max_model_len,
                     decode_kv_len=args.decode_kv_len,
                     prefill_chunk=PREFILL_CHUNK,
                     prefill_context=PREFILL_CONTEXT,
                     seed=args.seed,
                     cases=list(args.cases),
                     manifest=manifest)
    np.savez(args.emit, _meta=np.frombuffer(
        json.dumps(meta_blob).encode(), dtype=np.uint8), **out_blob)
    print(f"wrote {args.emit}", flush=True)
    return 0


def _load(path: str):
    z = np.load(path)
    meta = json.loads(bytes(z["_meta"]).decode())
    return z, meta


def _compare(path_a: str, path_b: str, *, expect_mismatch: bool) -> int:
    za, ma = _load(path_a)
    zb, mb_ = _load(path_b)

    print(f"A = {path_a}  VLLM_RPA_SKIP_EMPTY_PASS={ma['arm']}")
    print(f"B = {path_b}  VLLM_RPA_SKIP_EMPTY_PASS={mb_['arm']}")

    problems: list[str] = []
    for key in ("layout", "page_size", "cache_shape", "max_num_seqs",
                "max_model_len", "decode_kv_len", "prefill_chunk",
                "prefill_context", "seed", "cases"):
        if ma[key] != mb_[key]:
            problems.append(
                f"CONFIG MISMATCH {key}: A={ma[key]!r} B={mb_[key]!r}")
    if ma["arm"] == mb_["arm"]:
        problems.append(
            f"BOTH ARMS RAN WITH VLLM_RPA_SKIP_EMPTY_PASS={ma['arm']} -- this "
            "comparison is vacuous")
    if problems:
        for p in problems:
            print("  " + p)
        print("RESULT: INVALID")
        return 2

    n_diff = 0
    for name in ma["cases"]:
        a, b = ma["manifest"][name], mb_["manifest"][name]
        if a["in_cache_sha"] != b["in_cache_sha"]:
            print(f"  [{name}] INVALID: the two arms were fed different input "
                  "caches; the comparison means nothing")
            return 2
        # A check whose inputs cannot distinguish anything is not a check.
        if not (a["out_finite"] and b["out_finite"]):
            print(f"  [{name}] INVALID: non-finite attention output")
            return 2
        if a["out_std"] == 0.0 or b["out_std"] == 0.0:
            print(f"  [{name}] INVALID: attention output is constant, so "
                  "bit-identity between the arms proves nothing")
            return 2

        out_same = a["out_sha"] == b["out_sha"]
        cache_same = a["cache_sha"] == b["cache_sha"]
        oa = za[f"{name}__out"]
        ob = zb[f"{name}__out"]
        max_abs = float(np.abs(oa - ob).max()) if oa.shape == ob.shape else (
            float("nan"))
        live_same = np.array_equal(za[f"{name}__cache_live"],
                                   zb[f"{name}__cache_live"])
        status = "OK " if (out_same and cache_same and live_same) else "DIFF"
        if status == "DIFF":
            n_diff += 1
        print(f"  [{status}] {name:<16} dist={a['distribution']} "
              f"out_bitwise={out_same} cache_bitwise={cache_same} "
              f"live_pages_bitwise={live_same} max_abs_diff={max_abs:.6g}")

    if expect_mismatch:
        if n_diff == 0:
            print("RESULT: FAIL -- the deliberately broken predicate produced "
                  "identical results, so this check cannot detect a wrong "
                  "predicate and proves nothing.")
            return 1
        print(f"RESULT: PASS (negative control) -- {n_diff} case(s) differ, "
              "so the check discriminates.")
        return 0

    if n_diff:
        print(f"RESULT: FAIL -- {n_diff} case(s) differ between "
              "VLLM_RPA_SKIP_EMPTY_PASS=0 and =1. Skipping a pass is NOT "
              "numerically invisible.")
        return 1
    print("RESULT: PASS -- all cases bit-identical across the two arms.")
    return 0


ALL_CASES = ("decode_only", "prefill_only_1", "prefill_only_n", "mixed",
             "all_empty")

# `all_empty` is excluded by default: a step with no sequence in any range
# cannot occur in production (the runner never schedules an empty batch), and
# with both passes elided the "output" is an untouched query buffer whose
# agreement between arms is a statement about the memset guard rather than
# about attention. Available via `--cases all_empty` as a diagnostic.
DEFAULT_CASES = ("decode_only", "prefill_only_1", "prefill_only_n", "mixed")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emit", metavar="PATH")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    p.add_argument("--expect-mismatch",
                   action="store_true",
                   help=("negative control: require the two files to DIFFER. "
                         "Used to prove the check can fail."))
    p.add_argument("--layout", default=DEFAULT_LAYOUT,
                   choices=("head_along_sublane", "seq_along_lane"))
    p.add_argument("--page-size", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=mb.MAX_NUM_SEQS)
    p.add_argument("--max-model-len", type=int, default=mb.MAX_MODEL_LEN)
    p.add_argument("--decode-kv-len", type=int, default=mb.DECODE_KV_LEN)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES),
                   choices=list(ALL_CASES))
    args = p.parse_args()

    if bool(args.emit) == bool(args.compare):
        p.error("pass exactly one of --emit or --compare")
    if args.emit:
        return _emit(args)
    return _compare(*args.compare, expect_mismatch=args.expect_mismatch)


if __name__ == "__main__":
    raise SystemExit(main())
