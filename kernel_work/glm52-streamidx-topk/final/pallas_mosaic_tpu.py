"""Policy for the composed streamindex top-k: what the chip is, and what to do.

No `pl.pallas_call` here (`code-style.md` S10). This file reads the chip once
(S13), turns that into the integer arguments `tiling.py` needs, and decides
which of the composed changes are on and at what extents.

`Config` carries one field per adopted change, and each is independently
togglable. That is not a flag matrix left in for tidiness: three of the five
have effects at or below this block's 3-9% run-order drift, so the only way to
attribute the composed number is to time leave-one-out arms adjacently in one
process. `Config()` -- every field at its default -- IS the shipping candidate.

The gating each change carries is applied HERE, in the candidate, not in a note
beside it:

  `wide_kv_tile`         keys on `page_size`. At 1024 (the 8k1k deployment)
                         `_spec_tile` returns the incumbent tile deliberately,
                         so the flag cannot fire at (A) even when set.
  `fix_growth`           only reachable through the incumbent's growth loop,
                         which the capacity floor `p_min` already skips at (B).
  `pack_waves`           inert whenever the ladder has one rung, which is
                         exactly `p_max == 1` -- i.e. all of (A) once
                         `fix_growth` has collapsed P.
  `emit_keys`            structurally absent at `p == 1`: there is no stage 2
                         and therefore no seam to delete.
  `scan_count_hist`      unconditional; it is a different spelling of the same
                         histogram.
"""

import dataclasses
import functools

from jax.experimental.pallas import tpu as pltpu

import tiling

# Slot order of the three scoring launches (streamindex_topk.py:757-809).
DECODE, PREFILL, MIXED = 0, 1, 2

# The PREFILL slot is never launched -- the common path uses slots [0], [0],
# [2] -- but it still feeds the `max` in `tiling.padded_n`, so raising it
# widens the score matrix for a case that never runs. Pinned at the incumbent's
# value.
PREFILL_SLOT_BKV_P = 2
PREFILL_SLOT_BQ_SZ = 128

# Candidate tiles, best first. "Best" = fewest in-body iterations, which is
# `cdiv(q_len, bq_sz) * cdiv(kv_len, bkv_p*page_size)`, so it is the PRODUCT
# that ranks; ties break toward the larger bq_sz, which also removes whole
# re-reads of the KV stream. 64 is on the ladder to be REFUSED, not chosen: at
# (B) it declares 64.53 MiB of decode scratch against 64.00 MiB, so the ladder
# terminates itself one rung above the answer rather than at an arbitrary cap.
DECODE_BKV_P_LADDER = (64, 32, 16, 8, 4, 2)
MIXED_TILE_LADDER = ((1024, 4), (512, 2), (256, 2), (128, 4), (128, 2))


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
    """One field per adopted change. Every default on: this is what ships.

    A stdlib dataclass rather than `pydantic.dataclasses.dataclass`, which
    `code-style.md` S11 asks for: pydantic is NOT installed in the benchmark
    image, and importing it would make the candidate fail to load on the image
    the measurement runs on. Declared in `DEVIATIONS.md`. The cross-field
    checks S11 wants from a validator are in `__post_init__` instead.
    """

    # O8. Stage 1 compress-stores the winning monotone KEY beside its index and
    # DMAs both out, so stage 2 consumes keys directly and the `[B, N]`
    # `take_along_axis` at the seam does not exist.
    emit_keys: bool = True
    # O3's wave packing. `p` is chosen at RUNTIME from the batch's longest live
    # row, bucketed upward over a precompiled ladder, instead of being pinned
    # at the static capacity ceiling. NOT O3's flat-index gather, which is a
    # separate mechanism that O8 subsumes -- see DEVIATIONS.md.
    pack_waves: bool = True
    # O7. `plsc.scan_count` pre-aggregates duplicate buckets in-register, so one
    # scatter into a single 256-word histogram replaces 16 lane-private
    # histograms of ones, their zeroing and their fold.
    scan_count_hist: bool = True
    # O1a. `pick_partition` refuses to raise P unless doing so lowers the wave
    # count. It never does, so P collapses to the capacity minimum.
    fix_growth: bool = True
    # O1b. Each of the three scoring launches runs under a `lax.cond` on its own
    # sequence range being non-empty.
    skip_empty_launches: bool = True
    # O1c. The `-inf` pre-fill of the score buffer becomes `lax.empty`. Legal
    # only while the selector masks every lane on `j < i_hi`; the kernel file
    # asserts that gate before it relies on this.
    elide_score_prefill: bool = True
    # O2. The KV/query tile is chosen from `page_size` rather than taken from
    # the caller. Regime-conditional by construction: at page_size 1024 the
    # answer IS the incumbent tile.
    wide_kv_tile: bool = True

    def __post_init__(self):
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, bool):
                raise ValueError(f"{f.name} must be a bool, got {v!r}")
        # A cross-field dependency that is real but not expressible as a
        # rejected combination, so it is stated rather than enforced: eliding
        # the score pre-fill leaves every column past `row_lengths`
        # uninitialised, and the only thing that makes that legal is the
        # selector's own `j < i_hi` live-extent mask. The check that the mask
        # is still there is `pallas_mosaic_tpu_kernel._assert_valid_gate_intact`,
        # called at the one site that depends on it -- it inspects source, which
        # a dataclass validator cannot.


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class TileConfig:
    """Per-case KV/query tile extents. Index with DECODE / PREFILL / MIXED."""

    bkv_p: tuple[int, int, int]
    bq_sz: tuple[int, int, int]

    def __post_init__(self):
        # The decode launches are entered with static_q_len=1 and
        # `run_scores_kernel` takes min(bq_sz, 1) there, so a decode bq_sz above
        # 1 is silently inert. Say so rather than let a sweep believe it swept.
        if self.bq_sz[DECODE] != 1:
            raise ValueError(
                f"bq_sz[DECODE]={self.bq_sz[DECODE]} is inert: the decode "
                "launches pass static_q_len=1 and clamp bq_sz to it.")
        for v in self.bkv_p + self.bq_sz:
            if v <= 0:
                raise ValueError(f"tile extents must be positive: {self}")


# ---------------------------------------------------------------------------
# The chip.
# ---------------------------------------------------------------------------


def _sparse_core_info():
    """The chip's SparseCore geometry, or `NotImplementedError` if it has none."""
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError(
            "streamindex_topk's selector runs on SparseCore; this chip reports "
            "none. Fall back to lax.top_k.")
    return sc


def supported_on() -> bool:
    """Whether this backend can serve the op at all on the current chip."""
    try:
        _sparse_core_info()
    except NotImplementedError:
        return False
    return True


@functools.lru_cache(maxsize=None)
def hardware() -> dict:
    """Every chip number the geometry needs, read once.

    All of these were literals in the incumbent -- `LANES = 16`, an inline
    `32`, `MAX_SLICE_WORDS = 64 * 1024` -- and all are tpu7x facts rather than
    algorithm facts. Cached because the partition is recomputed on every trace
    and `get_tpu_info()` is not free.
    """
    sc = _sparse_core_info()
    return dict(
        lanes=sc.num_lanes,
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        subcores_per_wave=sc.num_cores * sc.num_subcores,
        sc_vmem_capacity_bytes=sc.vmem_capacity_bytes,
    )


def vmem_budget_bytes(vmem_limit_bytes: int) -> int:
    """The binding TensorCore VMEM cap.

    The caller's `vmem_limit_bytes` is a Mosaic scoped-allocation ceiling and
    can be set ABOVE physical (the instruments pass 100 MB), so the smaller of
    the two is what actually binds.
    """
    return min(vmem_limit_bytes, pltpu.get_tpu_info().vmem_capacity_bytes)


# ---------------------------------------------------------------------------
# Selector geometry: the ceiling, the slice width, and the ladder.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SelectGeometry:
    """What the selector runs at for one `[B, N] -> [B, k]` call."""

    p_max: int
    tn: int
    lanes: int
    subcores_per_wave: int
    num_cores: int
    num_subcores: int

    def __post_init__(self):
        if self.p_max & (self.p_max - 1):
            raise ValueError(f"p_max must be a power of two, got {self.p_max}")
        if self.tn % self.lanes:
            raise ValueError(
                f"tn={self.tn} must be a multiple of lanes={self.lanes}")

    @property
    def ladder(self) -> tuple[int, ...]:
        return tiling.partition_ladder(self.p_max)


@functools.lru_cache(maxsize=None)
def select_geometry(b: int, n: int, k: int, *, fix_growth: bool
                    ) -> SelectGeometry:
    """The partition ceiling and the slice width every ladder rung shares.

    `tn` is deliberately held at `N // p_max` rather than widened when the
    runtime `p` falls below the ceiling. A wave's cost is charged per word of
    its LONGEST slice, so a wider slice at a lower p would buy nothing; what
    the wave packing sells is a lower wave COUNT.
    """
    hw = hardware()
    p_max = tiling.pick_partition(
        b, n, k, lanes=hw["lanes"],
        subcores_per_wave=hw["subcores_per_wave"],
        sc_vmem_capacity_bytes=hw["sc_vmem_capacity_bytes"],
        fix_growth=fix_growth)
    return SelectGeometry(p_max=p_max, tn=n // p_max, lanes=hw["lanes"],
                          subcores_per_wave=hw["subcores_per_wave"],
                          num_cores=hw["num_cores"],
                          num_subcores=hw["num_subcores"])


def select_supported_on(b: int, n: int, k: int, *, fix_growth: bool) -> None:
    """Raise if the selector cannot serve this shape on this chip.

    `NotImplementedError` for a shape the backend cannot hold, `ValueError` for
    a caller mistake -- a fallback chain catches only the former
    (`code-style.md` S9).
    """
    hw = hardware()
    lanes = hw["lanes"]
    if n % lanes:
        raise ValueError(f"{n=} must be a multiple of {lanes}")
    if k % lanes or not 0 < k <= 4096:
        raise ValueError(
            f"{k=} must be a positive multiple of {lanes}, at most 4096")
    geo = select_geometry(b, n, k, fix_growth=fix_growth)
    budget = tiling.max_slice_words(hw["sc_vmem_capacity_bytes"])
    if geo.tn > budget:
        raise NotImplementedError(
            f"slice width {geo.tn} exceeds the {budget}-word subcore budget; "
            f"N={n} needs windowing this kernel does not do")


# ---------------------------------------------------------------------------
# Scoring geometry: which KV/query tile each launch gets.
# ---------------------------------------------------------------------------


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
        score_itemsize=4,  # out_dtype is f32
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


def tile_supported_on(cfg, *, vmem_limit_bytes, **kw):
    """Raise `NotImplementedError` if any launch of `cfg` cannot fit VMEM.

    What is weighed against the budget is the DECLARED scratch only, not
    `tiling.vmem_bytes`'s total. That is a correction the hardware forced:
    `measured (j-5136b90a-9f9e-4144-b066)` compiled and ran tiles this priced
    at 289.25 and 74.62 MiB on a 64.0 MiB chip, and only their declared parts
    (25.25 and 8.62 MiB) are under budget. Screening on the term the
    measurement says is binding costs nothing when it is right and rejects
    buildable, faster tiles when it is wrong -- and it was wrong.
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
                f"charged -- see tile_supported_on); "
                f"bq_sz={cfg.bq_sz} bkv_p={cfg.bkv_p}")
    return cfg


def _spec_tile(page_size: int) -> TileConfig:
    """The designed tile per regime. (A) is the incumbent, deliberately.

      (B) page_size 512, pages_per_seq 2048.  align_to(2048, 16) = 2048, so
          N_pad is unchanged at 1048576 for every bkv_p on the ladder: widening
          the KV tile is free of extent cost and removes 7/8 of the in-body
          iterations, `measured (j-69ee326d)` at 0.500 us each.
      (A) page_size 1024, pages_per_seq 9.  align_to(9, 16) = 16 pages against
          9 live: 78% more KV read, and N_pad 10240 -> 16384 into a SparseCore
          whose cost is driven by that extent. `modelled` at +1.7% (slower),
          which is why bkv_p stays at 2 here.
    """
    if page_size == 1024:
        return TileConfig(bkv_p=(2, PREFILL_SLOT_BKV_P, 2),
                          bq_sz=(1, PREFILL_SLOT_BQ_SZ, 128))
    if page_size == 512:
        return TileConfig(bkv_p=(16, PREFILL_SLOT_BKV_P, 4),
                          bq_sz=(1, PREFILL_SLOT_BQ_SZ, 1024))
    raise NotImplementedError(
        f"the regime split is keyed on page_size and only 1024 (the 8k1k "
        f"deployment) and 512 (AgentX) are decided; got {page_size}.")


def tile_config(*, page_size, vmem_limit_bytes, kv_packing, lkv_dim,
                num_q_heads, head_dim, q_itemsize, decode_req_batch_size,
                verbose=False) -> TileConfig:
    """The largest legal tile for this regime, walking down two ladders.

    Regime (A) has one entry on each ladder and this returns the incumbent.
    Regime (B) takes the widest decode `bkv_p` and the widest MIXED tile that
    `tile_supported_on` accepts; the fallback is loud, because a tile silently
    demoted is a measurement of something other than what was designed.
    """
    if page_size == 1024:
        return _spec_tile(page_size)

    check = dict(vmem_limit_bytes=vmem_limit_bytes, page_size=page_size,
                 kv_packing=kv_packing, lkv_dim=lkv_dim,
                 num_q_heads=num_q_heads, head_dim=head_dim,
                 q_itemsize=q_itemsize,
                 decode_req_batch_size=decode_req_batch_size)
    spec = _spec_tile(page_size)

    chosen = None
    for bkv_p in DECODE_BKV_P_LADDER:
        for bq_mixed, bkv_mixed in MIXED_TILE_LADDER:
            if not (tiling.is_legal_tile(page_size, bkv_p)
                    and tiling.is_legal_tile(page_size, bkv_mixed)):
                continue
            cand = TileConfig(bkv_p=(bkv_p, PREFILL_SLOT_BKV_P, bkv_mixed),
                              bq_sz=(1, PREFILL_SLOT_BQ_SZ, bq_mixed))
            try:
                chosen = tile_supported_on(cand, **check)
            except NotImplementedError as e:
                if verbose:
                    print(f"  [tile] rejected bkv_p={cand.bkv_p} "
                          f"bq_sz={cand.bq_sz}: {e}", flush=True)
                continue
            break
        if chosen is not None:
            break
    if chosen is None:
        raise NotImplementedError(
            f"no tile on either ladder fits VMEM at page_size={page_size}; "
            "even the incumbent tile is over budget.")
    if verbose and chosen != spec:
        direction = ("above" if chosen.bkv_p[DECODE] > spec.bkv_p[DECODE]
                     else "below")
        print(f"  [tile] chosen bkv_p={chosen.bkv_p} bq_sz={chosen.bq_sz} is "
              f"{direction} the design spec bkv_p={spec.bkv_p} "
              f"bq_sz={spec.bq_sz}", flush=True)
    return chosen


def as_kwargs(cfg: TileConfig) -> dict:
    """The two static kwargs the scoring launcher takes."""
    return dict(num_kv_pages_per_block=tuple(cfg.bkv_p),
                num_queries_per_block=tuple(cfg.bq_sz))
