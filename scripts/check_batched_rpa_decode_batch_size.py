# SPDX-License-Identifier: Apache-2.0
"""Correctness gate for ``VLLM_RPA_DECODE_BATCH_SIZE`` on the RPAd kernel.

``batched_rpa/wrapper.calculate_block_sizes`` hardcoded ``batch_size = 8`` for
the decode ``BlockSizes``; it is now ``int(os.getenv("VLLM_RPA_DECODE_BATCH_
SIZE", "8"))``. Lowering it does **not** merely shrink the unrolled live set:
``find_best_block_sizes`` spends the freed VMEM on a much larger ``bkv_sz``, so
the blocking of the flash-attention online softmax changes substantially. At
the production decode shape (64 seqs, kv_len 8704, page 256, max_model_len
9216):

    arm 8 -> RPAd-p256-b8-q1-k3840   bkv 3840   kv vmem (8, 3840, 1, 4, 128)
    arm 4 -> RPAd-p256-b4-q1-k7936   bkv 7936   kv vmem (4, 7936, 1, 4, 128)
    arm 2 -> RPAd-p256-b2-q1-k15872  bkv 15872  kv vmem (2, 15872, 1, 4, 128)

Arm 2's ``bkv_sz`` **exceeds ``max_model_len``**, so the KV grid is at a floor
of one block per sequence. That is precisely where a kernel that reads past the
valid KV length, or that drops sequences past the batch block, would look fast
and be wrong. A timing sweep measured 8 -> 2.71 ms, 4 -> 2.52 ms, 2 -> 1.98 ms;
this script exists to decide whether those are optimizations or missing work.

Why a script and not a pytest case
----------------------------------
``VLLM_RPA_DECODE_BATCH_SIZE`` is read at JAX trace time and baked into the
compiled kernel, so one process cannot hold two arms. Each arm is a separate
process that dumps its results; a final invocation compares them.

    VLLM_RPA_DECODE_BATCH_SIZE=8 python scripts/check_batched_rpa_decode_batch_size.py --emit /tmp/b8.npz
    VLLM_RPA_DECODE_BATCH_SIZE=4 python scripts/check_batched_rpa_decode_batch_size.py --emit /tmp/b4.npz
    VLLM_RPA_DECODE_BATCH_SIZE=2 python scripts/check_batched_rpa_decode_batch_size.py --emit /tmp/b2.npz
    python scripts/check_batched_rpa_decode_batch_size.py --compare /tmp/b8.npz /tmp/b4.npz /tmp/b2.npz

Why not a bitwise comparison
----------------------------
``check_batched_rpa_skip_empty_parity.py`` compares its arms bit for bit, and
is right to: eliding an empty pass must change nothing. Here the arms
legitimately reassociate the online-softmax accumulation over different
``bkv_sz``, so bit-identity is *not* the expected result and asserting it would
produce a false failure. The comparison is therefore a tolerance -- and a
tolerance is only as good as the evidence that it is tighter than the bugs it
must catch. Hence the four cases below, three of which exist to make the check
fail on purpose.

The cases
---------
All four are the production decode shape. ``pages_per_seq`` is 36 and
``kv_len`` 8704 is exactly 34 pages, so pages 34 and 35 of every sequence's own
allocation lie strictly beyond its valid KV.

    case            kv_lens          cache                       expectation
    base            8704 (true)      random                      the reference
    oob_poison      8704 (true)      pages 34:36 poisoned        == base, bitwise
    inband_poison   8704 (true)      pages 32:34 poisoned        != base, all 64 seqs
    overread_ctl    8960 (a lie)     random                      != base, all 64 seqs

``oob_poison`` is the direct test of the arm-2 hazard: if the kernel's mask
admits anything at or past ``kv_len``, the poison enters the softmax and the
output moves. It is an *absolute* check -- it does not trust arm 8.

``inband_poison`` is what makes ``oob_poison`` mean something. It poisons
tokens 8192..8703, which are inside every sequence's valid range and must be
attended, and requires **all 64** sequence outputs to move. If they do not, the
poison is invisible to the kernel and ``oob_poison``'s pass proves nothing.
Requiring all 64 is simultaneously an absolute anti-drop check: a batch-block
loop that skipped a block would leave those sequences' outputs unmoved.

``overread_ctl`` is the hardware negative control for the cross-arm tolerance.
It runs the *unmodified* kernel over one extra page of ordinary cache by lying
about ``kv_lens`` -- exactly the failure mode feared at arm 2 -- and its
distance from ``base`` is the size of the bug the tolerance has to catch. The
report prints that distance next to the arm-to-arm distance, so the margin is
measured rather than asserted.

Model geometry is inherited from ``microbench_batched_rpa_kv_layout.py`` rather
than restated, so the two cannot drift.
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
# HEAD_DIM, MAX_NUM_SEQS, MAX_MODEL_LEN, DECODE_KV_LEN, CONCURRENCY.
import microbench_batched_rpa_kv_layout as mb  # noqa: E402

# Same resolution as the parity script: the nightly config sets no
# VLLM_KV_CACHE_LAYOUT, so vLLM's default "NHD" maps to HEAD_ALONG_SUBLANE at
# page_size 256 -- the (2304, 256, 1, 4, 128) fp8 cache the microbench quotes.
DEFAULT_LAYOUT = "head_along_sublane"

# Poison magnitude, in fp8_e4m3fn. ~12x the 0.5-sigma scale of the ordinary
# cache, so a poisoned token that is attended dominates its softmax and the
# output cannot fail to move; far below fp8 max (448), so nothing overflows.
POISON_VALUE = 6.0

# How far past the true kv_len `overread_ctl` lies, in tokens. One page: the
# smallest over-read the blocking could plausibly produce, so the margin it
# establishes is a lower bound on the check's sensitivity.
OVERREAD_TOKENS = 256

ALL_CASES = ("base", "oob_poison", "inband_poison", "overread_ctl")

# Per-sequence relative-difference ceiling for the cross-arm comparison.
# The output is bf16 (~4e-3 relative granularity) and the arms legitimately
# reassociate an 8704-term online softmax, so a few ULPs are expected. The
# report prints the observed arm-to-arm distance against the observed
# `overread_ctl` distance; trust the printed margin, not this constant.
REL_TOL = 2e-2

_HASH_CHUNK_PAGES = 288


def _sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _chunked_sha(jax, arr) -> str:
    """sha256 of a device array's raw bytes, pulled a few pages at a time.

    The KV cache is ~302 MB; hashing it in one `device_get` is avoidable host
    pressure and buys nothing.
    """
    h = hashlib.sha256()
    n = int(arr.shape[0])
    for lo in range(0, n, _HASH_CHUNK_PAGES):
        hi = min(lo + _HASH_CHUNK_PAGES, n)
        h.update(np.ascontiguousarray(jax.device_get(arr[lo:hi])).tobytes())
    return h.hexdigest()


def _random_cache(jnp, jax, key, cache_shape, kv_dtype, chunks: int = 8):
    """A cache of *distinct* random pages.

    Deliberately not ``mb._make_cache``: that tiles one random page across the
    whole cache, which is right for a bandwidth benchmark and wrong here --
    with every page identical, a kernel reading the wrong page would be
    undetectable. Built in fp32 chunks so the transient stays ~150 MB rather
    than 1.2 GB.
    """
    n = int(cache_shape[0])
    keys = jax.random.split(key, chunks)
    edges = [round(i * n / chunks) for i in range(chunks + 1)]
    parts = []
    for i in range(chunks):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        part = jax.random.normal(keys[i], (hi - lo, ) + tuple(cache_shape[1:]),
                                 dtype=jnp.float32) * 0.5
        parts.append(part.astype(kv_dtype))
        del part
    return jnp.concatenate(parts, axis=0)


def _poison_pages(jnp, cache, pages_per_seq: int, lo_page: int, hi_page: int,
                  kv_dtype):
    """Overwrite page range ``[lo_page, hi_page)`` of every sequence."""
    n = int(cache.shape[0])
    if n % pages_per_seq:
        raise SystemExit(
            f"cache pages {n} is not a multiple of pages_per_seq "
            f"{pages_per_seq}; the poison would straddle sequences")
    nseq = n // pages_per_seq
    view = cache.reshape((nseq, pages_per_seq) + tuple(cache.shape[1:]))
    view = view.at[:, lo_page:hi_page].set(
        jnp.asarray(POISON_VALUE, dtype=kv_dtype))
    return view.reshape(cache.shape)


def _case_spec(name: str, *, page_size: int, pages_per_seq: int,
               decode_kv_len: int):
    """(kv_len fed to the kernel, poison page range or None) for ``name``."""
    valid_pages = decode_kv_len // page_size
    if decode_kv_len % page_size:
        raise SystemExit(
            f"decode_kv_len={decode_kv_len} is not a whole number of "
            f"page_size={page_size} pages; the poison boundary would be "
            "inside a page and the check would be ambiguous")
    if valid_pages >= pages_per_seq:
        raise SystemExit(
            f"kv_len {decode_kv_len} fills all {pages_per_seq} pages of the "
            "allocation, so there is no strictly-out-of-range region to "
            "poison. Raise --max-model-len or lower --decode-kv-len.")
    if name == "base":
        return decode_kv_len, None
    if name == "oob_poison":
        return decode_kv_len, (valid_pages, pages_per_seq)
    if name == "inband_poison":
        return decode_kv_len, (valid_pages - 2, valid_pages)
    if name == "overread_ctl":
        lied = decode_kv_len + OVERREAD_TOKENS
        if lied > pages_per_seq * page_size:
            raise SystemExit("overread_ctl would exceed the page allocation")
        return lied, None
    raise ValueError(f"unknown case {name!r}")


def _build_inputs(jnp, jax, *, kv_len: int, page_size: int,
                  pages_per_seq: int, max_num_seqs: int, num_live: int,
                  kv_dtype, q_dtype, cache_shape, poison, seed: int):
    """One decode step's kernel arguments, at the production decode shape."""
    q_lens = [1] * num_live
    kv_lens_list = [kv_len] * num_live
    distribution = [num_live, num_live, num_live]
    total_q_tokens = sum(q_lens)

    kv_lens = np.zeros((max_num_seqs, ), dtype=np.int32)
    kv_lens[:num_live] = kv_lens_list
    cu_q_lens = np.full((max_num_seqs + 1, ), total_q_tokens, dtype=np.int32)
    cu_q_lens[0] = 0
    cu_q_lens[1:num_live + 1] = np.cumsum(q_lens, dtype=np.int32)
    page_indices = np.arange(max_num_seqs * pages_per_seq, dtype=np.int32)

    # Every arm must see bit-identical q/k/v and an identical *base* cache, so
    # the key is fixed rather than derived from iteration order. sha256 rather
    # than hash(), which is salted per process and would silently desync arms.
    key = jax.random.key(seed)
    k_q, k_kv, k_cache = jax.random.split(key, 3)
    query = (jax.random.normal(
        k_q, (total_q_tokens, mb.NUM_Q_HEADS_PER_RANK, mb.HEAD_DIM),
        dtype=jnp.float32) * 0.5).astype(q_dtype)
    new_kv = (jax.random.normal(
        k_kv, (2, total_q_tokens, mb.NUM_KV_HEADS_PER_RANK, mb.HEAD_DIM),
        dtype=jnp.float32) * 0.5).astype(kv_dtype)

    cache = _random_cache(jnp, jax, k_cache, cache_shape, kv_dtype)
    if poison is not None:
        cache = _poison_pages(jnp, cache, pages_per_seq, poison[0], poison[1],
                              kv_dtype)

    args = dict(
        kv_lens=jnp.asarray(kv_lens),
        page_indices=jnp.asarray(page_indices),
        cu_q_lens=jnp.asarray(cu_q_lens),
        distribution=jnp.asarray(np.asarray(distribution, dtype=np.int32)),
    )
    return query, new_kv[0], new_kv[1], cache, args


def _emit(args) -> int:
    arm = os.environ.get("VLLM_RPA_DECODE_BATCH_SIZE", "<unset>")
    mb._export_vllm_layout_env(args.layout)
    import hashlib as _hl

    import jax
    import jax.numpy as jnp
    wrapper, configs = mb._import_kernel()

    kv_layout = getattr(configs.KVLayout, args.layout.upper())
    page_size = mb._resolve_page_size(args.layout, args.page_size)
    pages_per_seq = math.ceil(args.max_model_len / page_size)
    total_pages = args.max_num_seqs * pages_per_seq
    q_dtype, kv_dtype = jnp.bfloat16, jnp.float8_e4m3fn
    num_live = min(mb.CONCURRENCY, args.max_num_seqs)

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

    # Resolve, and record, the blocking this arm will actually compile. The
    # whole A/B is void if the env var did not reach `calculate_block_sizes`,
    # and "the arms differ because they differed" is not evidence.
    from jax.experimental.pallas import tpu as pltpu
    from vllm_torchtpu.kernels.experimental.batched_rpa import kernel as _krn
    vmem_limit = pltpu.get_tpu_info().vmem_capacity_bytes
    model_cfgs = configs.ModelConfigs(num_q_heads=mb.NUM_Q_HEADS_PER_RANK,
                                      num_kv_heads=mb.NUM_KV_HEADS_PER_RANK,
                                      head_dim=mb.HEAD_DIM,
                                      mask_value=0.0)
    serve_cfgs = configs.ServingConfigs(
        num_seqs=args.max_num_seqs,
        page_size=page_size,
        total_q_tokens=0,
        num_page_indices=total_pages,
        dtype_q=jnp.dtype(q_dtype),
        dtype_kv=jnp.dtype(kv_dtype),
        dtype_out=jnp.dtype(jnp.bfloat16),
        kv_layout=kv_layout,
    )
    dec_blocks, _ = wrapper.calculate_block_sizes(model_cfgs, serve_cfgs,
                                                  vmem_limit)
    rpa_cfgs = configs.RpaConfigs(block=dec_blocks,
                                  model=model_cfgs,
                                  serve=serve_cfgs,
                                  mode=configs.RpaCase.DECODE,
                                  vmem_limit_bytes=vmem_limit)
    kernel_name = _krn.get_kernel_name(rpa_cfgs)
    resolved_batch_size = int(dec_blocks.batch_size)

    step = mb._make_step(jax,
                         wrapper,
                         kv_layout=kv_layout,
                         sm_scale=mb.HEAD_DIM**-0.5,
                         q_scale=None,
                         k_scale=None,
                         v_scale=None)

    print(f"arm VLLM_RPA_DECODE_BATCH_SIZE={arm}  "
          f"resolved batch_size={resolved_batch_size}  "
          f"kernel={kernel_name}  bkv={int(dec_blocks.bkv_sz)}",
          flush=True)
    print(f"  layout={args.layout} page_size={page_size} "
          f"pages_per_seq={pages_per_seq} cache_shape={tuple(cache_shape)} "
          f"{np.dtype(kv_dtype).name} bytes/token={bytes_per_token:.0f} "
          f"live_seqs={num_live} kv_len={args.decode_kv_len}",
          flush=True)
    if arm != "<unset>" and str(resolved_batch_size) != arm:
        print(f"  !! requested {arm} but resolved {resolved_batch_size} -- "
              "the value was clamped or ignored", flush=True)

    valid_pages = args.decode_kv_len // page_size
    out_blob: dict[str, np.ndarray] = {}
    manifest: dict[str, dict] = {}
    build_hash = None

    for name in args.cases:
        kv_len, poison = _case_spec(name,
                                    page_size=page_size,
                                    pages_per_seq=pages_per_seq,
                                    decode_kv_len=args.decode_kv_len)
        q, k, v, cache, kargs = _build_inputs(jnp,
                                              jax,
                                              kv_len=kv_len,
                                              page_size=page_size,
                                              pages_per_seq=pages_per_seq,
                                              max_num_seqs=args.max_num_seqs,
                                              num_live=num_live,
                                              kv_dtype=kv_dtype,
                                              q_dtype=q_dtype,
                                              cache_shape=cache_shape,
                                              poison=poison,
                                              seed=args.seed)
        in_cache_sha = _chunked_sha(jax, cache)
        q_sha, k_sha, v_sha = (_sha(jax.device_get(q)),
                               _sha(jax.device_get(k)),
                               _sha(jax.device_get(v)))

        if build_hash is None:
            try:
                text = step.lower(q, k, v, cache, **kargs).compile().as_text()
                build_hash = _hl.sha256(text.encode()).hexdigest()[:16]
            except Exception as exc:  # noqa: BLE001 - identity only
                build_hash = f"<unavailable: {exc!r}>"

        out, new_cache = step(q, k, v, cache, **kargs)
        out_h = np.asarray(jax.device_get(out))
        # Only the in-range prefix of the cache is comparable across cases:
        # the poison itself survives the step in the out-of-range pages.
        n_slots = int(new_cache.shape[0]) // pages_per_seq
        valid_view = new_cache.reshape(
            (n_slots, pages_per_seq) + tuple(new_cache.shape[1:]))
        valid_sha = _chunked_sha(
            jax, valid_view[:, :valid_pages].reshape(
                (-1, ) + tuple(new_cache.shape[1:])))
        del new_cache, valid_view

        out32 = out_h.astype(np.float32)
        out_blob[f"{name}__out"] = out32
        manifest[name] = dict(
            kv_len_fed=int(kv_len),
            poison_pages=list(poison) if poison else None,
            in_cache_sha=in_cache_sha,
            q_sha=q_sha,
            k_sha=k_sha,
            v_sha=v_sha,
            out_sha=_sha(out_h),
            out_shape=list(out_h.shape),
            out_dtype=str(out_h.dtype),
            cache_valid_sha=valid_sha,
            out_finite=bool(np.isfinite(out32).all()),
            out_std=float(out32.std()),
            per_seq_absmax=[float(x) for x in np.abs(out32).reshape(
                out32.shape[0], -1).max(axis=1)],
        )
        m = manifest[name]
        print(f"  [{name}] kv_len_fed={m['kv_len_fed']} "
              f"poison={m['poison_pages']} out_sha={m['out_sha'][:16]} "
              f"cache_valid_sha={m['cache_valid_sha'][:16]} "
              f"std={m['out_std']:.6f} finite={m['out_finite']}",
              flush=True)

    meta_blob = dict(arm=arm,
                     resolved_batch_size=resolved_batch_size,
                     kernel_name=kernel_name,
                     bkv_sz=int(dec_blocks.bkv_sz),
                     bq_sz=int(dec_blocks.bq_sz),
                     n_buffer=int(dec_blocks.n_buffer),
                     build_hash=build_hash,
                     layout=args.layout,
                     page_size=page_size,
                     cache_shape=list(map(int, cache_shape)),
                     max_num_seqs=args.max_num_seqs,
                     max_model_len=args.max_model_len,
                     decode_kv_len=args.decode_kv_len,
                     num_live=num_live,
                     poison_value=POISON_VALUE,
                     overread_tokens=OVERREAD_TOKENS,
                     seed=args.seed,
                     cases=list(args.cases),
                     manifest=manifest)
    print("EMIT_JSON: " + json.dumps(
        dict(arm=arm, resolved_batch_size=resolved_batch_size,
             kernel_name=kernel_name, bkv_sz=int(dec_blocks.bkv_sz),
             build_hash=build_hash),
        sort_keys=True), flush=True)
    np.savez(args.emit,
             _meta=np.frombuffer(json.dumps(meta_blob).encode(),
                                 dtype=np.uint8),
             **out_blob)
    print(f"wrote {args.emit}", flush=True)
    return 0


def _load(path: str):
    z = np.load(path)
    meta = json.loads(bytes(z["_meta"]).decode())
    return z, meta


def _per_seq_rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-sequence max|a-b| normalised by that sequence's own scale."""
    flat_a = a.reshape(a.shape[0], -1)
    flat_b = b.reshape(b.shape[0], -1)
    denom = np.maximum(np.abs(flat_a).max(axis=1), 1e-6)
    return np.abs(flat_a - flat_b).max(axis=1) / denom


def _inject(out: np.ndarray, fault: str) -> np.ndarray:
    """Synthesise the bug the tolerance must catch, to prove it catches it."""
    out = out.copy()
    kind, _, param = fault.partition(":")
    if kind == "drop_last_seqs":
        n = int(param or 2)
        out[-n:] = 0.0
        return out
    if kind == "drop_first_seqs":
        n = int(param or 2)
        out[:n] = 0.0
        return out
    if kind == "scale_one_seq":
        # A single mis-rescaled sequence: the m/l/acc bug shape.
        out[0] *= float(param or 1.05)
        return out
    raise SystemExit(f"unknown --inject-fault {fault!r}")


def _compare(paths: list[str], *, base_arm: str, rel_tol: float,
             inject_fault: str | None, inject_into: str) -> int:
    loaded = [(p, ) + _load(p) for p in paths]
    metas = {m["arm"]: (p, z, m) for p, z, m in loaded}
    problems: list[str] = []

    print("=" * 78)
    print("ARMS")
    print("=" * 78)
    for p, _z, m in loaded:
        print(f"  {m['arm']:<4} {p}")
        print(f"       resolved_batch_size={m['resolved_batch_size']} "
              f"kernel={m['kernel_name']} bkv={m['bkv_sz']} "
              f"build_hash={m['build_hash']}")

    if len(metas) != len(loaded):
        problems.append("two files report the same arm -- comparison vacuous")

    # --- guard: the arms must be different programs ------------------------
    print()
    print("=" * 78)
    print("GUARD 1: the arms are distinct compiled programs")
    print("=" * 78)
    hashes = {m["build_hash"] for _p, _z, m in loaded}
    names = {m["kernel_name"] for _p, _z, m in loaded}
    bss = {m["resolved_batch_size"] for _p, _z, m in loaded}
    print(f"  distinct build_hash  = {len(hashes)} of {len(loaded)}: "
          f"{sorted(hashes)}")
    print(f"  distinct kernel_name = {len(names)} of {len(loaded)}: "
          f"{sorted(names)}")
    print(f"  resolved batch sizes = {sorted(bss)}")
    for _p, _z, m in loaded:
        if m["arm"] != "<unset>" and str(m["resolved_batch_size"]) != m["arm"]:
            problems.append(
                f"arm {m['arm']} resolved to batch_size "
                f"{m['resolved_batch_size']} -- the env var was not honoured")
    if len(hashes) != len(loaded) or len(names) != len(loaded):
        problems.append(
            "the arms do not map 1:1 onto distinct compiled programs; "
            "VLLM_RPA_DECODE_BATCH_SIZE did not reach the kernel and this "
            "check is a NULL, not a pass")
    else:
        print("  GUARD 1 PASSED: one distinct program per arm.")

    # --- guard: identical inputs across arms --------------------------------
    print()
    print("=" * 78)
    print("GUARD 2: every arm saw bit-identical inputs")
    print("=" * 78)
    ref_meta = metas.get(base_arm)
    if ref_meta is None:
        print(f"  no file for reference arm {base_arm!r}")
        print("RESULT: INVALID")
        return 2
    _rp, rz, rm = ref_meta
    for key in ("layout", "page_size", "cache_shape", "max_num_seqs",
                "max_model_len", "decode_kv_len", "num_live", "seed", "cases",
                "poison_value", "overread_tokens"):
        vals = {json.dumps(m[key]) for _p, _z, m in loaded}
        if len(vals) != 1:
            problems.append(f"CONFIG MISMATCH {key}: {sorted(vals)}")
    for case in rm["cases"]:
        for sig in ("in_cache_sha", "q_sha", "k_sha", "v_sha"):
            vals = {m["manifest"][case][sig] for _p, _z, m in loaded}
            if len(vals) != 1:
                problems.append(
                    f"INPUT MISMATCH {case}.{sig}: the arms were fed "
                    f"different inputs; the comparison means nothing")
    if not any(p.startswith(("CONFIG", "INPUT")) for p in problems):
        print("  GUARD 2 PASSED: identical config and identical "
              "q/k/v/cache digests in every arm, for every case.")

    # --- guard: the outputs are capable of distinguishing anything ----------
    print()
    print("=" * 78)
    print("GUARD 3: the outputs are finite and non-constant")
    print("=" * 78)
    for _p, _z, m in loaded:
        for case in m["cases"]:
            c = m["manifest"][case]
            if not c["out_finite"]:
                problems.append(f"arm {m['arm']} {case}: non-finite output")
            if c["out_std"] == 0.0:
                problems.append(
                    f"arm {m['arm']} {case}: output is constant, so agreement "
                    "between arms would prove nothing")
    if not any("non-finite" in p or "constant" in p for p in problems):
        print("  GUARD 3 PASSED.")

    if problems:
        print()
        for p in problems:
            print("  !! " + p)
        print("RESULT: INVALID")
        return 2

    # --- the in-arm absolute checks ----------------------------------------
    print()
    print("=" * 78)
    print("CHECK A (absolute, per arm, does not trust any other arm)")
    print("=" * 78)
    print("  A1 oob_poison    == base bitwise      -> no read at/past kv_len")
    print("  A2 inband_poison != base, ALL seqs    -> every seq's KV is read")
    print("  A3 overread_ctl  != base, ALL seqs    -> a 1-page over-read is")
    print("                                           visible at this shape")
    per_arm_ok: dict[str, bool] = {}
    overread_rel: dict[str, float] = {}
    for _p, z, m in loaded:
        arm = m["arm"]
        ok = True
        base = z["base__out"]
        mb_base = m["manifest"]["base"]
        print(f"\n  arm {arm} (batch_size={m['resolved_batch_size']}, "
              f"bkv={m['bkv_sz']}):")

        if "oob_poison" in m["cases"]:
            same_out = (m["manifest"]["oob_poison"]["out_sha"] ==
                        mb_base["out_sha"])
            same_cache = (m["manifest"]["oob_poison"]["cache_valid_sha"] ==
                          mb_base["cache_valid_sha"])
            rel = _per_seq_rel(base, z["oob_poison__out"])
            verdict = "PASS" if (same_out and same_cache) else "FAIL"
            ok &= (same_out and same_cache)
            print(f"    A1 oob_poison : out_bitwise={same_out} "
                  f"cache_valid_bitwise={same_cache} "
                  f"max_rel={rel.max():.3e}  -> {verdict}")
            if not same_out:
                bad = np.nonzero(rel > 0)[0]
                print(f"       {len(bad)} sequence(s) moved when KV strictly "
                      f"BEYOND kv_len was poisoned: seqs {bad[:12].tolist()}"
                      f"{' ...' if len(bad) > 12 else ''}")
                print("       THE KERNEL READS PAST THE VALID KV LENGTH.")

        if "inband_poison" in m["cases"]:
            rel = _per_seq_rel(base, z["inband_poison__out"])
            moved = int((rel > 0).sum())
            allmoved = moved == rel.shape[0]
            ok &= allmoved
            print(f"    A2 inband_poison: {moved}/{rel.shape[0]} seqs moved  "
                  f"min_rel={rel.min():.3e} median_rel={np.median(rel):.3e}  "
                  f"-> {'PASS' if allmoved else 'FAIL'}")
            if not allmoved:
                still = np.nonzero(rel == 0)[0]
                print(f"       seqs that did NOT move: {still.tolist()}")
                print("       Either those sequences are not attended at all "
                      "(a dropped batch block), or the poison is invisible -- "
                      "in which case A1's pass proves nothing.")

        if "overread_ctl" in m["cases"]:
            rel = _per_seq_rel(base, z["overread_ctl__out"])
            moved = int((rel > 0).sum())
            allmoved = moved == rel.shape[0]
            ok &= allmoved
            overread_rel[arm] = float(np.median(rel))
            print(f"    A3 overread_ctl : {moved}/{rel.shape[0]} seqs moved  "
                  f"min_rel={rel.min():.3e} median_rel={np.median(rel):.3e} "
                  f"max_rel={rel.max():.3e}  "
                  f"-> {'PASS' if allmoved else 'FAIL'}")
        per_arm_ok[arm] = ok

    # --- the cross-arm comparison -------------------------------------------
    print()
    print("=" * 78)
    print(f"CHECK B (cross-arm): every arm vs the shipping default {base_arm}")
    print("=" * 78)
    if inject_fault:
        print(f"  !! NEGATIVE CONTROL ACTIVE: injecting {inject_fault!r} into "
              f"arm {inject_into}'s '{'base'}' output before comparing.")
        print("     Check B MUST report FAIL below. If it does not, the "
              "tolerance is too loose to catch this bug.")
    cross_ok = True
    for _p, z, m in loaded:
        arm = m["arm"]
        if arm == base_arm:
            continue
        print(f"\n  arm {arm} vs arm {base_arm}:")
        for case in m["cases"]:
            a = rz[f"{case}__out"]
            b = z[f"{case}__out"]
            if inject_fault and arm == inject_into and case == "base":
                b = _inject(b, inject_fault)
            if a.shape != b.shape:
                print(f"    [{case}] SHAPE MISMATCH {a.shape} vs {b.shape}")
                cross_ok = False
                continue
            rel = _per_seq_rel(a, b)
            bad = np.nonzero(rel > rel_tol)[0]
            bitwise = (rel.max() == 0.0)
            passed = bad.size == 0
            cross_ok &= passed
            print(f"    [{'PASS' if passed else 'FAIL'}] {case:<14} "
                  f"max_rel={rel.max():.3e} median_rel={np.median(rel):.3e} "
                  f"bitwise={bitwise} seqs_over_tol={bad.size}/{rel.shape[0]}")
            if bad.size:
                worst = bad[np.argsort(-rel[bad])][:8]
                print("       worst seqs: " + ", ".join(
                    f"#{int(i)}={rel[int(i)]:.3e}" for i in worst))

    # --- the margin: is the tolerance tighter than the bug? -----------------
    print()
    print("=" * 78)
    print("MARGIN: the tolerance measured against a real bug, not asserted")
    print("=" * 78)
    arm_rels = {}
    for _p, z, m in loaded:
        if m["arm"] == base_arm:
            continue
        arm_rels[m["arm"]] = float(
            _per_seq_rel(rz["base__out"], z["base__out"]).max())
    for arm, r in sorted(arm_rels.items()):
        ctl = overread_rel.get(arm) or overread_rel.get(base_arm)
        print(f"  arm {arm}: legitimate reassociation vs arm {base_arm} = "
              f"{r:.3e}")
        if not ctl:
            print("          (no A3 control -- margin not established)")
            continue
        ratio = (ctl / r) if r > 0 else float("inf")
        bracketed = r < rel_tol < ctl
        print(f"          a real 1-page over-read (A3, same kernel) = "
              f"{ctl:.3e}")
        print(f"          separation = {ratio:.1f}x   tolerance "
              f"{rel_tol:.1e} "
              f"{'sits between the two' if bracketed else 'is OUTSIDE the two -- RECALIBRATE'}")

    # --- verdict ------------------------------------------------------------
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    for arm in sorted(per_arm_ok):
        print(f"  arm {arm}: absolute checks (A1/A2/A3) "
              f"{'PASS' if per_arm_ok[arm] else 'FAIL'}")
    print(f"  cross-arm vs {base_arm} (B): {'PASS' if cross_ok else 'FAIL'}")

    if inject_fault:
        if cross_ok:
            print("RESULT: FAIL (negative control) -- the injected fault "
                  f"{inject_fault!r} passed the tolerance. This check cannot "
                  "catch that bug and must not be quoted as a gate.")
            return 1
        print("RESULT: PASS (negative control) -- the injected fault "
              f"{inject_fault!r} was caught. The check discriminates.")
        return 0

    if all(per_arm_ok.values()) and cross_ok:
        print("RESULT: PASS -- every arm masks past kv_len, attends every "
              "sequence, and agrees with the shipping default within a "
              "tolerance demonstrably tighter than a one-page over-read.")
        return 0
    print("RESULT: FAIL")
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emit", metavar="PATH")
    p.add_argument("--compare", nargs="+", metavar="NPZ")
    p.add_argument("--base-arm", default="8",
                   help="the arm every other arm is compared against")
    p.add_argument("--rel-tol", type=float, default=REL_TOL)
    p.add_argument("--inject-fault", default=None,
                   help=("negative control: corrupt one arm's output before "
                         "comparing and require the check to FAIL. e.g. "
                         "drop_last_seqs:2, drop_first_seqs:2, "
                         "scale_one_seq:1.05"))
    p.add_argument("--inject-into", default="2",
                   help="which arm --inject-fault corrupts")
    p.add_argument("--layout", default=DEFAULT_LAYOUT,
                   choices=("head_along_sublane", "seq_along_lane"))
    p.add_argument("--page-size", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=mb.MAX_NUM_SEQS)
    p.add_argument("--max-model-len", type=int, default=mb.MAX_MODEL_LEN)
    p.add_argument("--decode-kv-len", type=int, default=mb.DECODE_KV_LEN)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cases", nargs="+", default=list(ALL_CASES),
                   choices=list(ALL_CASES))
    args = p.parse_args()

    if bool(args.emit) == bool(args.compare):
        p.error("pass exactly one of --emit or --compare")
    if args.emit:
        return _emit(args)
    return _compare(list(args.compare),
                    base_arm=args.base_arm,
                    rel_tol=args.rel_tol,
                    inject_fault=args.inject_fault,
                    inject_into=args.inject_into)


if __name__ == "__main__":
    raise SystemExit(main())
