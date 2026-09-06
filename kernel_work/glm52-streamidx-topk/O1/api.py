"""The public entry point for the O1 streamindex top-k: signature, and nothing else.
"""

# Dimensions.
#   T          padded token count; rows of `q` and of the result
#   R, S       live sequences, and the padded sequence slot count
#   H, DH      index heads and head dim
#   K          selected count per row
#
# No config here and no `pallas_call` (code-style.md §10): the switches this
# function forwards belong to `pallas_mosaic_tpu.Config` and are only named here
# because they must be `static_argnames` -- see the note on jit caching below.

import functools

import jax

import pallas_mosaic_tpu
import pallas_mosaic_tpu_kernel

# code-style.md §4 asks for jaxtyping on the public wrapper. DELIBERATE
# DEVIATION, declared here as §10 requires: `jaxtyping` is not installed in the
# vllm-torchtpu environment this runs in (checked in `uv_venv`), and adding a
# hard import for it would make the variant fail to load on the very image the
# microbenchmark uses. The shape contract is carried by the trailing `# [shape]`
# comments and by `_check_shapes`, which raises `ValueError` at the boundary --
# the place §4 says the caller's mistake should be caught.


def _check_shapes(q, indexer_weights, cache_kv, seq_lens, page_indices,
                  cu_q_lens, distribution, k):
    """The boundary check jaxtyping would have done, spelled out."""
    if q.ndim != 3:
        raise ValueError(f"q must be [T, H, DH], got {q.shape}")
    T, H, _ = q.shape
    if indexer_weights.shape != (T, H):
        raise ValueError(
            f"indexer_weights must be [T, H] = {(T, H)}, got "
            f"{indexer_weights.shape}; it is indexed by the same token rows "
            f"as q")
    if cache_kv.ndim != 4:
        raise ValueError(
            f"cache_kv must be [nb, psz // kv_packing, kv_packing, 256], got "
            f"{cache_kv.shape}")
    S = seq_lens.shape[0]
    if cu_q_lens.shape != (S + 1, ):
        raise ValueError(
            f"cu_q_lens must be [S + 1] = {(S + 1,)} to match seq_lens' "
            f"S = {S}, got {cu_q_lens.shape}")
    if page_indices.ndim != 1 or page_indices.shape[0] % S != 0:
        raise ValueError(
            f"page_indices must be a flat [S * pages_per_seq]; {page_indices.shape} "
            f"is not a multiple of S = {S}")
    if distribution.shape != (3, ):
        raise ValueError(
            f"distribution must be i32[3] = (num_decode, num_decode_and_prefill, "
            f"num_seqs), got {distribution.shape}")
    if k <= 0:
        raise ValueError(f"{k=} must be positive")


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "compression_ratio",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "decode_req_batch_size",
        "enable_early_exit",
        # O1's three switches. They are static, and they are named separately
        # rather than bundled into one `config` argument, for a measurement
        # reason worth stating where someone might tidy it away: a `Config`
        # instance is not hashable-as-a-static-arg in the way jit wants, and
        # -- more importantly -- three independent static args give three
        # independent jit cache entries, so the six benchmark arms can be
        # traced once each and then timed adjacently in one process. The
        # alternative, monkey-patching a module between arms, hands arm 2 the
        # executable arm 1 compiled; that is a real hazard on this op, not a
        # hypothetical one.
        "fix_growth",
        "skip_empty_launches",
        "elide_score_prefill",
    ),
)
def streamindex_topk(
    q,               # [T, H, DH] f8_e4m3
    indexer_weights,  # [T, H] f32   softmax and per-head scales pre-folded
    cache_kv,        # [nb, psz // 4, 4, 256] u8   paged fp8 KV + UE8M0 scales
    seq_lens,        # [S] i32      uncompressed KV length per sequence
    page_indices,    # [S * pages_per_seq] i32
    cu_q_lens,       # [S + 1] i32
    distribution,    # [3] i32      (num_decode, num_decode_and_prefill, num_seqs)
    *,
    k: int,
    compression_ratio: int,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int = pallas_mosaic_tpu_kernel.DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    enable_early_exit: bool = False,
    fix_growth: bool | None = None,
    skip_empty_launches: bool | None = None,
    elide_score_prefill: bool | None = None,
) -> jax.Array:
    """Top-K KV positions per token row. [T, k] i32, -1-padded.

    Signature-compatible with the shipping
    `vllm_torchtpu.kernels.deepseek_v4.streamindex_topk`, so the shared
    correctness instrument can be pointed at this without being edited. The
    three trailing switches are additions and each defaults to the heuristic in
    `pallas_mosaic_tpu._get_heuristics_config`, i.e. all three on.

    Pass one of them `False` only to reproduce the incumbent's behaviour for a
    paired measurement. `fix_growth=False, skip_empty_launches=False,
    elide_score_prefill=False` is the copy-fidelity control: it should be
    indistinguishable from the shipping kernel, and if it is not, the copy is
    wrong and every other arm in the job is meaningless.
    """
    _check_shapes(q, indexer_weights, cache_kv, seq_lens, page_indices,
                  cu_q_lens, distribution, k)
    if not pallas_mosaic_tpu.supported_on():
        raise NotImplementedError(
            "the O1 streamindex top-k needs a SparseCore; this chip reports "
            "none")

    cfg = pallas_mosaic_tpu._get_heuristics_config()
    return pallas_mosaic_tpu_kernel.streamindex_topk_impl(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k=k,
        compression_ratio=compression_ratio,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        decode_req_batch_size=decode_req_batch_size,
        enable_early_exit=enable_early_exit,
        fix_growth=cfg.fix_growth if fix_growth is None else fix_growth,
        skip_empty_launches=(cfg.skip_empty_launches
                             if skip_empty_launches is None else
                             skip_empty_launches),
        elide_score_prefill=(cfg.elide_score_prefill
                             if elide_score_prefill is None else
                             elide_score_prefill),
    )
