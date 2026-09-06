"""O2 policy: which KV/query tile each of the three scoring launches gets.

O2 is a partitioning/pipelining/layout decision (areas 2, 3, 4) and nothing
else -- the kernel body in `pallas_mosaic_tpu_kernel.py` is the incumbent's,
unchanged. What moves is `bkv_p`, and it moves ONLY at regime (B).

Why the regime split is the option rather than a detail
-------------------------------------------------------
The score extent is `N_pad = align_to(pages_per_seq, bkv_p) * page_size`, taken
as a max over the whole tile tuple (`tiling.padded_n`).

  (B) page_size 512, pages_per_seq 2048.  align_to(2048, 16) = 2048.  N_pad is
      unchanged at 1048576 for every bkv_p on the ladder, so widening the KV
      tile is free of extent cost and removes 7/8 of the in-body iterations.
      `measured (j-69ee326d)`: 0.500 us per iteration, and at B-D1-1M the
      1024 iterations are 35.8% of the op.

  (A) page_size 1024, pages_per_seq 9.  align_to(9, 16) = 16 pages against 9
      live: 78% more KV read, and N_pad 10240 -> 16384, a 60% wider extent
      into a SparseCore whose cost is driven by that extent.
      design/measurement_and_design.md 9.2 prices it at +1.7% (slower) --
      `modelled`, and it is why bkv_p stays at 2 here.

There is no config that is right for both. `_get_heuristics_config` keys on
`page_size`, which is what section 17 of the design sanctions.
"""
import dataclasses

import pydantic
from jax.experimental.pallas import tpu as pltpu

import tiling

# Slot order of the three scoring launches (streamindex_topk.py:757-809).
DECODE, PREFILL, MIXED = 0, 1, 2

# The PREFILL slot is never launched -- `_common_path` uses slots [0], [0], [2]
# -- but it still feeds the `max` in `tiling.padded_n`, so raising it widens
# the score matrix for a case that never runs. Pinned at the incumbent's 2.
# (Lowering it further is carried item 10 of the design and belongs to O1.)
PREFILL_SLOT_BKV_P = 2
PREFILL_SLOT_BQ_SZ = 128

# Candidate tiles, best first. "Best" = fewest in-body iterations, which is
# `cdiv(q_len, bq_sz) * cdiv(kv_len, bkv_p*page_size)`, so it is the PRODUCT
# bq_sz*bkv_p that ranks; ties are broken toward the larger bq_sz because that
# also removes whole re-reads of the KV stream.
# 64 is on the ladder to be REFUSED, not chosen: at (B) it declares 64.53 MiB
# of decode scratch against 64.00 MiB, so the ladder terminates itself one rung
# above the answer rather than at an arbitrary cap. 16 was that arbitrary cap
# in the first two jobs, because P-ITER's sweep stopped there; 32 halves the
# in-body iteration count again for 32.53 MiB and no extent cost, since
# align_to(2048, 32) is still 2048.
DECODE_BKV_P_LADDER = (64, 32, 16, 8, 4, 2)
MIXED_TILE_LADDER = ((1024, 4), (512, 2), (256, 2), (128, 4), (128, 2))


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
    """Per-case tile extents. Index with DECODE / PREFILL / MIXED."""

    bkv_p: tuple[pydantic.PositiveInt, pydantic.PositiveInt,
                 pydantic.PositiveInt]
    bq_sz: tuple[pydantic.PositiveInt, pydantic.PositiveInt,
                 pydantic.PositiveInt]

    def __post_init__(self):
        # Cross-field invariant: the decode launches are entered with
        # static_q_len=1, and `run_scores_kernel` takes min(bq_sz, 1) there, so
        # a decode bq_sz above 1 is silently inert. Say so rather than let a
        # sweep believe it swept something.
        if self.bq_sz[DECODE] != 1:
            raise ValueError(
                f"bq_sz[DECODE]={self.bq_sz[DECODE]} is inert: the decode "
                "launches pass static_q_len=1 and clamp bq_sz to it.")


def _tile_vmem(cfg, slot, *, sbs, page_size, kv_packing, lkv_dim, num_q_heads,
               head_dim, q_itemsize):
    return tiling.vmem_bytes(
        sbs=sbs,
        bq_sz=cfg.bq_sz[slot],
        bkv_p=cfg.bkv_p[slot],
        page_size=page_size,
        kv_packing=kv_packing,
        lkv_dim=lkv_dim,
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        q_itemsize=q_itemsize,
        w_itemsize=4,      # prepared_indexer_weights is cast to f32
        score_itemsize=4,  # out_dtype is f32 (streamindex_topk.py:741)
    )


def vmem_report(cfg, *, page_size, kv_packing, lkv_dim, num_q_heads, head_dim,
                q_itemsize, decode_req_batch_size):
    """Per-launch VMEM, in bytes. Every launch that can actually run."""
    common = dict(page_size=page_size, kv_packing=kv_packing,
                  lkv_dim=lkv_dim, num_q_heads=num_q_heads,
                  head_dim=head_dim, q_itemsize=q_itemsize)
    return {
        "decode": _tile_vmem(cfg, DECODE, sbs=decode_req_batch_size, **common),
        "decode-remainder": _tile_vmem(cfg, DECODE, sbs=1, **common),
        "mixed": _tile_vmem(cfg, MIXED, sbs=1, **common),
    }


def vmem_budget_bytes(vmem_limit_bytes):
    """The binding VMEM cap, read from the platform rather than hardcoded.

    code-style.md 13: hardware constants come from `pltpu.get_tpu_info()`.
    `get_tpu_info()` describes one TensorCore, which is the unit this kernel
    runs on, so `vmem_capacity_bytes` is the right number here. The caller's
    `vmem_limit_bytes` is a Mosaic scoped-allocation ceiling and can be set
    ABOVE physical (the instruments pass 100 MB), so the smaller of the two is
    what actually binds.
    """
    return min(vmem_limit_bytes, pltpu.get_tpu_info().vmem_capacity_bytes)


def supported_on(cfg, *, vmem_limit_bytes, **kw):
    """Raise `NotImplementedError` if any launch of `cfg` cannot fit VMEM.

    `NotImplementedError`, not `ValueError`: the caller's arguments are fine,
    this tile just cannot be served on this chip (code-style.md 9).

    What is weighed against the budget is the DECLARED scratch only, not
    `tiling.vmem_bytes`'s total. That is a correction the hardware forced, and
    it is worth stating because the arithmetic pointed the other way:

      `measured (j-5136b90a-9f9e-4144-b066)`: on a chip whose
      `vmem_capacity_bytes` is 64.0 MiB, tiles this function had priced at
      289.25 MiB (bq_sz 1024, bkv_p 4) and 74.62 MiB (bq_sz 512, bkv_p 2) both
      COMPILED and produced correct output. Only the declared-scratch part of
      those figures -- 25.25 and 8.62 MiB -- is under 64. So Mosaic does not
      hold `s` whole and live alongside everything else; it tiles the einsum
      internally, and `tiling.einsum_temp_bytes` is an upper bound on a
      quantity that never all exists at once.

    Screening on the term the measurement says is binding costs nothing when
    it is right and rejects buildable, faster tiles when it is wrong -- and it
    was wrong, so the temps are now reported rather than charged.
    """
    budget = vmem_budget_bytes(vmem_limit_bytes)
    for name, parts in vmem_report(cfg, **kw).items():
        temps = parts["s"] + parts["s_summed"]
        declared = parts["total"] - temps
        if declared > budget:
            raise NotImplementedError(
                f"{name} launch declares {declared / 2**20:.2f} MiB of "
                f"scratch against a {budget / 2**20:.2f} MiB budget "
                f"(einsum temps a further {temps / 2**20:.2f} MiB, not "
                f"charged -- see supported_on); "
                f"bq_sz={cfg.bq_sz} bkv_p={cfg.bkv_p}")
    return cfg


def _spec_config(page_size):
    """The tile design/measurement_and_design.md 9.1/14 specifies for O2.

    (B) is `bkv_p` 16 at decode and the (1024, 4) MIXED tile; (A) is the
    incumbent, deliberately.
    """
    if page_size == 1024:
        return Config(bkv_p=(2, PREFILL_SLOT_BKV_P, 2),
                      bq_sz=(1, PREFILL_SLOT_BQ_SZ, 128))
    if page_size == 512:
        return Config(bkv_p=(16, PREFILL_SLOT_BKV_P, 4),
                      bq_sz=(1, PREFILL_SLOT_BQ_SZ, 1024))
    raise NotImplementedError(
        f"O2's regime split is keyed on page_size and only 1024 (the 8k1k "
        f"deployment) and 512 (AgentX) are decided; got {page_size}.")


def _get_heuristics_config(*, page_size, vmem_limit_bytes, kv_packing,
                           lkv_dim, num_q_heads, head_dim, q_itemsize,
                           decode_req_batch_size, verbose=False):
    """The largest legal tile for this regime, walking down two ladders.

    Regime (A) has one entry on each ladder and this returns the incumbent.
    Regime (B) takes the widest decode `bkv_p` and the widest MIXED tile that
    `supported_on` accepts; the fallback is loud, because a tile silently
    demoted is a measurement of something other than what was designed.
    """
    if page_size == 1024:
        return _spec_config(page_size)

    check = dict(vmem_limit_bytes=vmem_limit_bytes, page_size=page_size,
                 kv_packing=kv_packing, lkv_dim=lkv_dim,
                 num_q_heads=num_q_heads, head_dim=head_dim,
                 q_itemsize=q_itemsize,
                 decode_req_batch_size=decode_req_batch_size)
    spec = _spec_config(page_size)

    chosen = None
    for bkv_p in DECODE_BKV_P_LADDER:
        for bq_mixed, bkv_mixed in MIXED_TILE_LADDER:
            if not (tiling.is_legal_tile(page_size, bkv_p)
                    and tiling.is_legal_tile(page_size, bkv_mixed)):
                continue
            cand = Config(bkv_p=(bkv_p, PREFILL_SLOT_BKV_P, bkv_mixed),
                          bq_sz=(1, PREFILL_SLOT_BQ_SZ, bq_mixed))
            try:
                chosen = supported_on(cand, **check)
            except NotImplementedError as e:
                if verbose:
                    print(f"  [O2 heuristic] rejected "
                          f"bkv_p={cand.bkv_p} bq_sz={cand.bq_sz}: {e}",
                          flush=True)
                continue
            break
        if chosen is not None:
            break
    if chosen is None:
        raise NotImplementedError(
            "no tile on either ladder fits VMEM at page_size="
            f"{page_size}; even the incumbent tile is over budget.")
    if verbose and chosen != spec:
        # Two different things reach here and they are opposite results, so
        # say which: the ladder is ordered best-first, so a chosen tile whose
        # bkv_p exceeds the spec's is an IMPROVEMENT on the design, while one
        # below it is a demotion the design did not anticipate.
        direction = ("above" if chosen.bkv_p[DECODE] > spec.bkv_p[DECODE]
                     else "below")
        print(f"  [O2 heuristic] chosen tile bkv_p={chosen.bkv_p} "
              f"bq_sz={chosen.bq_sz} is {direction} the design's spec tile "
              f"bkv_p={spec.bkv_p} bq_sz={spec.bq_sz}", flush=True)
    return chosen


def _get_autotuning_configs(page_size):
    """The sweep the file plan asks for: decode `bkv_p` over {2,4,8,16}.

    Legality, not budget, filters here -- `supported_on` is what prices a
    candidate, and an autotuner is entitled to be told which ones it rejects.
    """
    out = []
    for bkv_p in (2, 4, 8, 16, 32, 64):
        if not tiling.is_legal_tile(page_size, bkv_p):
            continue
        out.append(Config(bkv_p=(bkv_p, PREFILL_SLOT_BKV_P, 2),
                          bq_sz=(1, PREFILL_SLOT_BQ_SZ, 128)))
    return tuple(out)


def as_kwargs(cfg):
    """The two static kwargs the kernel launcher takes."""
    return dict(num_kv_pages_per_block=tuple(cfg.bkv_p),
                num_queries_per_block=tuple(cfg.bq_sz))


def replace(cfg, **kw):
    return dataclasses.replace(cfg, **kw)
