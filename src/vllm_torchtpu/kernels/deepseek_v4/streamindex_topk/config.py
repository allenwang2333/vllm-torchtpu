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


# Second-minor extent of the `PACKING8` indexer K cache. Not a free tuning
# knob: 8 is the sublane count of the `T(8,128)(4,1)` tiling the SparseCore
# scatter fusion requires, and the whole point of the layout is that the
# buffer's *natural* tiling already equals it. Changing it reintroduces the
# retiling copies.
PACKING8_KV_PACKING = 8


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

  - TOKEN_MAJOR_3D: **not shipping; kept only as the arm that documents why
    PACKING8 exists.** `[pages, page_size, width]`. Byte-for-byte the same
    cache as HEAD_ALONG_SUBLANE -- the `page_size // packing` and `packing`
    axes are merged into one token axis -- but stored as a rank-3 array, which
    XLA tiles `T(32,128)(4,1)` instead of `T(4,128)(4,1)`.

    The intent was that the flatten to `[pages * page_size, width]` becomes a
    bitcast. It does, but that was never the binding cost: the SparseCore
    `scatter_fusion` wants the cache tiled `T(8,128)(4,1)`, so XLA wraps the
    scatter in `copy T(32,128)->T(8,128)` / `copy back` over the whole buffer.
    Measured (`j-5d800b55`, whole op, write + top-k): prefill 1024 tok 0.7504
    vs the rank-4 incumbent's 1.2427 ms, but decode x8 0.2397 vs 0.0996 --
    net **+0.365 s/request** on the bench mix, a regression. Do not ship.

  - PACKING8: `[pages, page_size // 8, 8, width]`. The *same bytes again* --
    token `t` is row-block `t // 8`, row `t % 8` -- but rank-4 with
    second-minor extent 8, whose natural tiling **is** the `T(8,128)(4,1)` the
    scatter wants. So the retiling copies are not made cheaper, they stop
    being emitted: `cost_analysis()['bytes accessed']` at decode is 0.04 MB,
    like the incumbent, against 889.26 MB for TOKEN_MAJOR_3D.

    It needs no layout API and no new kernel branch: rank-4 takes the existing
    HEAD_ALONG_SUBLANE read path, which derives `kv_packing` from the shape.
    Measured net vs the incumbent: -13.28 ms/core-step prefill, -0.023 decode,
    **-3.424 s/request** over 256 prefill + 1024 decode steps at 21 layers.
  """

    HEAD_ALONG_SUBLANE = 0
    SEQ_ALONG_LANE = 1
    TOKEN_MAJOR_3D = 2
    PACKING8 = 3

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
            "packing8": cls.PACKING8,
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
