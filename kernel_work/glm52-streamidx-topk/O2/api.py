"""O2's entry point: `streamindex_topk`, call-compatible with the incumbent.

Three deliberate deviations from
`.claude/skills/write-tpu-kernels/references/code-style.md`, declared here as
that file requires:

  0. Section 4 asks for jaxtyping annotations on the public wrapper. jaxtyping
     is NOT installed in the benchmark image (`import jaxtyping` fails in
     `uv_venv`), and `src/vllm_torchtpu/kernels/kimi_k3/chunk_kda.py:1425`
     records that this repo removed it for that reason. Adding a `pip install`
     to every TPU job to satisfy an annotation is a failure mode bought for
     nothing, so the shapes are carried as trailing comments instead.

Then the two that section 10 forbids -- `api.py` holds no config, and this
file names two:

  1. `num_kv_pages_per_block` and `num_queries_per_block` stay in the signature
     and are IGNORED. They are the incumbent's public tuning knobs and the
     shared instruments pass them positionally-by-keyword; dropping them would
     make `instruments/correctness.py` and `instruments/bench.py` uncallable
     against this variant, and the instruments are read-only. O2's whole claim
     is that these two cannot be set correctly by a caller who does not know
     the regime, so the heuristic overrides them. Ignoring an argument silently
     is worse than not taking one, so it is announced once per process.

  2. `vmem_limit_bytes` also stays, and is used -- it is a Mosaic scoped
     allocation ceiling rather than a tile knob, and `pallas_mosaic_tpu`
     intersects it with the platform's real capacity.
"""
import functools

import jax
import jax.numpy as jnp

import base
import pallas_mosaic_tpu as policy
import pallas_mosaic_tpu_kernel as backend

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024

_announced = set()


def _announce(msg):
    if msg not in _announced:
        _announced.add(msg)
        print(msg, flush=True)


@functools.partial(
    jax.jit,
    static_argnames=("k", "compression_ratio", "num_kv_pages_per_block",
                     "num_queries_per_block", "vmem_limit_bytes",
                     "decode_req_batch_size", "enable_early_exit",
                     "force_config"),
)
def streamindex_topk(
    q,                  # [T, H, D] fp8
    indexer_weights,    # [T, H] f32
    cache_kv,           # [nb, PSZ/kvp, kvp, 256] u8
    seq_lens,           # [S] i32          uncompressed lengths
    page_indices,       # [S*PPS] i32      flattened page table
    cu_q_lens,          # [S+1] i32
    distribution,       # [3] i32          (decode_end, prefill_end, num_seqs)
    *,
    k: int,
    compression_ratio: int,
    num_kv_pages_per_block=None,   # ignored -- see the module docstring
    num_queries_per_block=None,    # ignored -- see the module docstring
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    enable_early_exit: bool = False,
    force_config=None,
):                      # -> [T, k] i32, -1-padded
    """StreamIndex Top-K retrieval, with O2's regime-conditional KV tile.

    `force_config` is the bench's escape hatch: it bypasses the heuristic so a
    tile the heuristic rejects can still be handed to the compiler and the
    rejection turned from arithmetic into a measurement.
    """
    max_num_seqs = seq_lens.shape[0]
    _, page_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing

    q = base.prepare_q_inputs(q)
    prepared_indexer_weights = base.prepare_index_weights(
        indexer_weights, q.dtype)
    _, num_q_heads, head_dim = q.shape

    if force_config is None:
        cfg = policy._get_heuristics_config(
            page_size=page_size, vmem_limit_bytes=vmem_limit_bytes,
            kv_packing=kv_packing, lkv_dim=lkv_dim, num_q_heads=num_q_heads,
            head_dim=head_dim, q_itemsize=jnp.dtype(q.dtype).itemsize,
            decode_req_batch_size=decode_req_batch_size, verbose=True)
    else:
        cfg = force_config

    if num_kv_pages_per_block is not None or num_queries_per_block is not None:
        _announce(
            f"  [O2] ignoring caller tiles "
            f"num_kv_pages_per_block={num_kv_pages_per_block} "
            f"num_queries_per_block={num_queries_per_block}; page_size="
            f"{page_size} selects bkv_p={cfg.bkv_p} bq_sz={cfg.bq_sz}")

    def _common_path(_):
        scores = backend.scores(
            q, prepared_indexer_weights, cache_kv, seq_lens, page_indices,
            cu_q_lens, distribution, cfg=cfg,
            compression_ratio=compression_ratio,
            decode_req_batch_size=decode_req_batch_size,
            vmem_limit_bytes=vmem_limit_bytes)
        if scores.shape[1] < k:
            scores = jnp.pad(scores, ((0, 0), (0, k - scores.shape[1])),
                             constant_values=-jnp.inf)
        eff_row_lengths = base.effective_row_lengths(
            seq_lens=seq_lens, cu_q_lens=cu_q_lens, distribution=distribution,
            num_tokens=q.shape[0], num_positions=scores.shape[1],
            compression_ratio=compression_ratio)
        # Selection is a separate operation and O2 does not touch it: the
        # shipping selector is used as-is, so any delta measured here is the
        # scoring half's.
        topk_idxs = sparsecore_topk(scores, k, row_lengths=eff_row_lengths,
                                    write_empty_rows=True)
        return topk_idxs[:q.shape[0], :k]

    if not enable_early_exit:
        return _common_path(None)
    return jax.lax.cond(
        jnp.max(seq_lens) // compression_ratio <= k,
        lambda _: base.fast_path(q, seq_lens, cu_q_lens, compression_ratio, k),
        _common_path, None)
