"""StreamIndex Top-K Configuration."""

import enum
import itertools
import threading


class MlaCase(enum.Enum):
    """Represents the different cases for MLA.

  - DECODE: Sequences are in decode-only mode (q_len = 1).
  - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
  - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
  """

    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self):
        return {
            MlaCase.DECODE: "d",
            MlaCase.PREFILL: "p",
            MlaCase.MIXED: "m",
        }[self]


class KVLayout(enum.Enum):
    """Memory layout of the packed FP8 index KV cache.

  - HEAD_ALONG_SUBLANE: `[pages, page_size // packing, packing, width]`.
    Tokens live on the sublane dimension and the `head_dim` features live on
    the lane dimension. Because per-token quantization appends a UE8M0 scale
    byte after the `head_dim` FP8 bytes, `width` must be padded up to a
    multiple of the 128-lane register width. For `head_dim=128` that means 129
    useful bytes are stored in 256, i.e. **half of the HBM traffic is padding**
    and the kernel can never exceed ~0.5 of the HBM roofline.

  - SEQ_ALONG_LANE: `[pages, head_dim // packing + 1, packing, page_size]`.
    The sequence is packed along the lane dimension and `head_dim` lives on the
    sublane dimension, so the per-token scale only needs to grow the *sublane*
    count. One extra packed sublane group (`packing` rows, of which the first
    holds the UE8M0 scale) is enough, giving `head_dim + packing` bytes per
    token (132 instead of 256 for `head_dim=128`).

    This also makes the QK product the MXU-native `[n, d] x [d, m]` form and
    yields the per-token scales directly as a `[1, bkv_sz]` row vector, so the
    lane broadcast of the dequantization scale becomes free.

  - TOKEN_MAJOR_3D: `[pages, page_size, width]`. Byte-for-byte the same cache
    as HEAD_ALONG_SUBLANE -- the `page_size // packing` and `packing` axes are
    merged into one token axis -- but stored as a rank-3 array, which XLA
    tiles `T(32,128)(4,1)` instead of `T(4,128)(4,1)`. The flatten to
    `[pages * page_size, width]` that XLA's scatter canonicalises to then
    preserves the tile and becomes a bitcast, so the write no longer pays a
    `relayout -> scatter -> relayout` round trip over the whole cache.

    The kernel's read path is unchanged apart from addressing: one page is
    `page_size` contiguous rows instead of `page_size // packing` contiguous
    `(packing, width)` groups, and the VMEM block the pipeline stages is the
    same bytes with the packing axis already folded in.
  """

    HEAD_ALONG_SUBLANE = 0
    SEQ_ALONG_LANE = 1
    TOKEN_MAJOR_3D = 2

    @classmethod
    def parse(cls, value: "str | KVLayout | None") -> "KVLayout":
        """Parses a `KVLayout` from a case-insensitive string (or passes through)."""
        if value is None:
            return cls.HEAD_ALONG_SUBLANE
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        aliases = {
            "head_along_sublane": cls.HEAD_ALONG_SUBLANE,
            "seq_along_lane": cls.SEQ_ALONG_LANE,
            # Tolerate the common typo seen in flags/configs.
            "seq_alone_lane": cls.SEQ_ALONG_LANE,
            "token_major_3d": cls.TOKEN_MAJOR_3D,
        }
        if normalized not in aliases:
            raise ValueError(
                f"Unknown kv_layout {value!r}; expected one of {sorted(aliases)}."
            )
        return aliases[normalized]


def indexer_kv_layout() -> KVLayout:
    """The DSA indexer K cache layout `TPU_DSA_INDEXER_KV_LAYOUT` selects.

  One reader for the env var, because three places have to agree on it or the
  cache is allocated in one layout and read in another: the allocator
  (`runner/kv_cache_manager.py`), the scatter that inserts this step's rows
  (`layers/adapter/custom_ops/mla_attention_op.py`) and the kernel itself.
  """
    from vllm_torchtpu import envs  # Local: envs imports torch.
    return KVLayout.parse(envs.TPU_DSA_INDEXER_KV_LAYOUT)


# Default buffer_count for (DECODE, PREFILL, MIXED) cases.
# Determined from microbenchmarks for 8k/1k and 256k/1k.
DEFAULT_BUFFER_COUNT: tuple[int, int, int] = (4, 3, 3)

# Source of `_scheduling_group_id`s for the chunked TensorCore/SparseCore
# pipeline. One counter for the process, because the ids are global to a
# compiled program and a model traces this kernel once per layer: two layers
# sharing an id would be ONE group to XLA, and the legalizer would either
# annotate everything on the path between them or fail the compile.
_scheduling_group_ids = itertools.count(1)
_scheduling_group_ids_lock = threading.Lock()


def reserve_scheduling_group_ids(count: int) -> int:
    """Reserves `count` scheduling group ids nobody else in this process gets.

  Args:
    count: Number of ids wanted; a pipeline over `n` chunks uses `n - 1` of them
      (chunk 0's TensorCore compute and the last chunk's SparseCore compute are
      not annotated).

  Returns:
    The first id; the caller owns `[first, first + count)`.
  """
    if count < 1:
        raise ValueError(f"count ({count}) must be positive.")
    with _scheduling_group_ids_lock:
        first = next(_scheduling_group_ids)
        for _ in range(count - 1):
            next(_scheduling_group_ids)
    return first
