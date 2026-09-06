"""The two public entry points of the composed candidate. Signatures only.

  `sparsecore_topk`   selection alone, call-compatible with the shipping
                      `vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk`.
  `streamindex_topk`  scoring + selection, call-compatible with the shipping
                      `vllm_torchtpu.kernels.deepseek_v4.streamindex_topk`.

Call-compatible is a hard requirement, not a courtesy: the shared
`instruments/correctness.py` and `instruments/bench.py` are READ-ONLY and call
the incumbent's signature. A wrapper that needed them edited would have moved
the gate rather than passed it.

Every adopted change is carried by one field of `pallas_mosaic_tpu.Config`,
which is a frozen dataclass and therefore hashable, so `cfg` is a
`static_argname`. That is what makes the leave-one-out arms cheap: each
distinct `Config` gets its own jit cache entry, all of them trace once, and
they can then be timed adjacently in one process. The alternative --
monkey-patching a module between arms -- hands arm 2 the executable arm 1
compiled, which on this op is a real hazard rather than a hypothetical one.

`Config()` is the shipping candidate. Turning a field off reproduces the
incumbent's behaviour for that one mechanism and nothing else.

Deviations from `code-style.md`, declared per its S10 and listed in
`DEVIATIONS.md`:

  0. S4 asks for jaxtyping on the public wrapper. `jaxtyping` is not installed
     in the benchmark image, and `src/vllm_torchtpu/kernels/kimi_k3/
     chunk_kda.py:1425` records that this repo removed it for that reason.
     Shapes are trailing comments; the check itself is `base.check_shapes`.
  1. S10 says `api.py` holds no config, and `streamindex_topk` names two
     config arguments: `num_kv_pages_per_block` and `num_queries_per_block`.
     They stay because the read-only instruments pass them, and under
     `wide_kv_tile` they are OVERRIDDEN rather than honoured -- the whole
     content of that option is that a caller who does not know the page-size
     regime cannot set them correctly. Overriding an argument silently is
     worse than not taking one, so it is announced once per process.
"""

import functools

import jax
import jax.numpy as jnp

import base
import pallas_mosaic_tpu as policy
import pallas_mosaic_tpu_kernel as backend

DEFAULT_VMEM_LIMIT_BYTES = backend.DEFAULT_VMEM_LIMIT_BYTES

_announced = set()


def _announce(msg):
    if msg not in _announced:
        _announced.add(msg)
        print(msg, flush=True)


@functools.partial(jax.jit,
                   static_argnames=("k", "write_empty_rows", "cfg"))
def sparsecore_topk(
    scores: jax.Array,  # [B, N] f32
    k: int,
    row_lengths: jax.Array | None = None,  # [B] i32
    *,
    write_empty_rows: bool = True,
    cfg: policy.Config = policy.Config(),
) -> jax.Array:  # [B, k] i32, -1 suffix-padded
    """Exact top-k column indices per row, unsorted.

    Ties at the cut may break differently from the incumbent's, which the
    incumbent's own docstring already permits and the shared instrument's
    churn metric covers. The index SET is expected to match, and
    `run_correctness.py` checks it as a symmetric difference against the
    incumbent at every shape rather than only against a threshold.
    """
    row_lengths = base.bind(scores, k, row_lengths)
    if not policy.supported_on():
        raise NotImplementedError(
            "this selector needs a SparseCore; the chip reports none")
    return backend.sparsecore_topk_impl(scores, k, row_lengths,
                                        write_empty_rows=write_empty_rows,
                                        cfg=cfg)


@functools.partial(
    jax.jit,
    static_argnames=("k", "compression_ratio", "num_kv_pages_per_block",
                     "num_queries_per_block", "vmem_limit_bytes",
                     "decode_req_batch_size", "enable_early_exit", "cfg"),
)
def streamindex_topk(
    q,                # [T, H, DH] fp8      per-token index queries
    indexer_weights,  # [T, H] f32          softmax and per-head scales folded
    cache_kv,         # [nb, psz//kvp, kvp, lkv] u8   paged fp8 KV + scales
    seq_lens,         # [S] i32             UNCOMPRESSED length per sequence
    page_indices,     # [S * pages_per_seq] i32
    cu_q_lens,        # [S + 1] i32
    distribution,     # [3] i32   (num_decode, num_decode_and_prefill, num_seqs)
    *,
    k: int,
    compression_ratio: int,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    enable_early_exit: bool = False,
    cfg: policy.Config = policy.Config(),
) -> jax.Array:  # [T, k] i32, -1-padded
    """Top-K compressed KV positions per token row.

    Under `cfg.wide_kv_tile` the caller's two tile arguments are replaced by
    `pallas_mosaic_tpu.tile_config`, which is keyed on `page_size` read off
    `cache_kv`. That keying IS the gating the option was measured with: at
    page_size 1024 -- the 8k1k deployment -- `tile_config` returns the
    incumbent's own tile, so the option is inert there by construction rather
    than by a benchmark flag. At page_size 512 it widens the decode and mixed
    tiles.
    """
    base.check_shapes(q, indexer_weights, cache_kv, seq_lens, page_indices,
                      cu_q_lens, distribution, k)
    if not policy.supported_on():
        raise NotImplementedError(
            "this kernel needs a SparseCore; the chip reports none")

    tiles = dict(num_kv_pages_per_block=num_kv_pages_per_block,
                 num_queries_per_block=num_queries_per_block)
    if cfg.wide_kv_tile:
        _, page_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape
        page_size = page_size_per_kv_packing * kv_packing
        # The padded head geometry `prepare_q_inputs` will produce, computed
        # rather than materialised: the tile ledger needs the padded numbers
        # and padding the array here would duplicate work the backend does.
        _, actual_h, actual_dh = q.shape
        q_packing = backend.get_dtype_packing(q.dtype)
        chosen = policy.tile_config(
            page_size=page_size,
            vmem_limit_bytes=vmem_limit_bytes,
            kv_packing=kv_packing,
            lkv_dim=lkv_dim,
            num_q_heads=backend.align_to(actual_h, q_packing),
            head_dim=backend.align_to(actual_dh, 128),
            q_itemsize=jnp.dtype(q.dtype).itemsize,
            decode_req_batch_size=decode_req_batch_size)
        override = policy.as_kwargs(chosen)
        if any(v is not None for v in tiles.values()) and override != tiles:
            _announce(
                f"  [wide_kv_tile] page_size={page_size}: overriding caller "
                f"tiles num_kv_pages_per_block={num_kv_pages_per_block} "
                f"num_queries_per_block={num_queries_per_block} with "
                f"bkv_p={chosen.bkv_p} bq_sz={chosen.bq_sz}")
        tiles = override

    return backend.streamindex_topk_impl(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k=k,
        compression_ratio=compression_ratio,
        vmem_limit_bytes=vmem_limit_bytes,
        decode_req_batch_size=decode_req_batch_size,
        enable_early_exit=enable_early_exit,
        cfg=cfg,
        **tiles,
    )
