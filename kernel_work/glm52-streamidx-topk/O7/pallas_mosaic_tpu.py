"""O7 config and Pallas call. Everything that knows about tuning lives here.

The four changes are independent booleans, all defaulting to the incumbent's
behaviour, so `Config()` is meant to be bit-for-bit the incumbent algorithm and
every arm differs from it by exactly one flag.

Declared deviation from `code-style.md` §11: `__post_init__` raises
`NotImplementedError` rather than `ValueError`. Every rejected combination here
is one this variant chose not to build -- a 4-bucket final refinement level, a
1024-bucket level 0 on the lane-private back end -- not a caller passing a value
that is wrong on its face, and `NotImplementedError` is what says which of the
two it is. The cross-field checks are in `__post_init__` as the rule requires.
"""
import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from compute import refinement_levels
from pallas_mosaic_tpu_kernel import l0_buckets, topk_kernel

# Largest per-subcore resident slice, in 32-bit words. Not derived from the
# platform: it is the incumbent's partitioning contract (`_pick_partition`
# raises P until the slice fits), and changing it would change which shapes take
# the two-stage path -- a different variant's business.
MAX_SLICE_WORDS = 64 * 1024

# Vector-subcore VMEM the kernel may actually claim. `vmem_capacity_bytes` is
# the physical 512 KiB; the reference records 480 KiB as the measured usable
# ceiling, and 8 KiB on top of that is margin for whatever the compiler spills.
VMEM_RESERVED_BYTES = 32 * 1024
VMEM_MARGIN_BYTES = 8 * 1024


def _cdiv(a, b):
    return (a + b - 1) // b


def _align_to(x, a):
    return _cdiv(x, a) * a


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
    """One point in O7's tuning space.

    All-defaults is the incumbent: separate conversion pass, lane-private
    histograms zeroed per level, full-slice masked re-scan at every refinement
    level, 8-bit digits.
    """

    fold_and_zero: bool = False
    scan_count_hist: bool = False
    carry_forward: bool = False
    fused_convert: bool = False
    digit_width: int = 8
    l0_shift: int = 24

    def __post_init__(self):
        if self.digit_width not in (4, 8):
            raise NotImplementedError(
                f"digit_width must be 4 or 8, got {self.digit_width}")
        if self.fold_and_zero and self.scan_count_hist:
            raise NotImplementedError(
                "scan_count_hist removes the lane-private histogram entirely, "
                "so there is nothing for fold_and_zero to fold; setting both "
                "would silently measure scan_count_hist alone")
        if not 16 <= self.l0_shift <= 24:
            raise NotImplementedError(f"l0_shift out of range: {self.l0_shift}")
        # Every level's histogram is walked a vector at a time, so a level
        # narrower than one vector would give it zero chunks and silently skip
        # it. `refinement_levels` only ever makes the LAST level ragged.
        narrowest = min(m + 1 for _, m in refinement_levels(self.l0_shift,
                                                            self.digit_width))
        if narrowest < 16:
            raise NotImplementedError(
                f"l0_shift={self.l0_shift} with digit_width={self.digit_width} "
                f"leaves a final level of {narrowest} buckets, under one "
                f"{16}-lane vector")
        if self.l0_shift != 24 and not self.scan_count_hist:
            # 1024 level-0 buckets replicated 16 ways is 64 KiB of VMEM and a
            # 1024-iteration fold per row. Widening level 0 is only affordable
            # on the scan_count back end, and pairing it with the lane-private
            # one would measure the VMEM pressure rather than the idea.
            raise NotImplementedError(
                "l0_shift != 24 requires scan_count_hist")

    @property
    def tag(self):
        on = [n for n in ("fold_and_zero", "scan_count_hist", "carry_forward",
                          "fused_convert") if getattr(self, n)]
        return "+".join(on + [f"d{self.digit_width}", f"s{self.l0_shift}"])


def _get_autotuning_configs():
    """The sweep. One flag at a time, then the composition, then digit width.

    Ordered so a truncated sweep still yields the attribution: arms 1-4 each
    differ from arm 0 by one flag, and only arm 5 onwards combines them.
    """
    base = Config()
    return (
        base,
        Config(fold_and_zero=True),
        Config(scan_count_hist=True),
        Config(carry_forward=True),
        Config(fused_convert=True),
        Config(scan_count_hist=True, carry_forward=True, fused_convert=True),
        Config(scan_count_hist=True, carry_forward=True, fused_convert=True,
               digit_width=4),
        # Widened level 0. Measured on this workload, the incumbent's 256-bucket
        # level 0 leaves most of the slice in the tie bucket, which is what
        # starves `carry_forward`; these are the arms that test that reading.
        Config(scan_count_hist=True, l0_shift=22),
        Config(scan_count_hist=True, carry_forward=True, l0_shift=22),
        Config(scan_count_hist=True, carry_forward=True, fused_convert=True,
               l0_shift=22),
        Config(scan_count_hist=True, carry_forward=True, fused_convert=True,
               l0_shift=20),
    )


def _hist_buckets(cfg):
    """Widest histogram any level needs: level 0, or the widest refinement."""
    return max(l0_buckets(cfg.l0_shift),
               max(m + 1 for _, m in refinement_levels(cfg.l0_shift,
                                                       cfg.digit_width)))


def _scratch_words(cfg, slice_len, k, b_pad, lanes):
    """Words of VMEM each subcore needs, excluding the candidate buffer."""
    nb = _hist_buckets(cfg)
    priv = (lanes * nb + lanes) if not cfg.scan_count_hist else lanes
    return slice_len + priv + (nb + lanes) + (k + lanes) + b_pad


def _candidate_cap(cfg, sc, slice_len, k, b_pad, lanes):
    """How many level-0 ties can be carried forward.

    Sized from what is left of the subcore's VMEM after the resident slice and
    the histograms, never from a guess. A row whose tie bucket is wider than
    this falls back to the incumbent's masked re-scan, which costs the incumbent
    and is correct; it does not cost more.
    """
    if not cfg.carry_forward:
        return 0
    budget = sc.vmem_capacity_bytes - VMEM_RESERVED_BYTES - VMEM_MARGIN_BYTES
    free = budget // 4 - _scratch_words(cfg, slice_len, k, b_pad, lanes) - lanes
    cap = min(slice_len, max(0, free))
    return (cap // lanes) * lanes


def sc_topk_direct(
    scores,  # [B, N] f32
    k,
    row_lengths,  # [B] i32
    *,
    cfg,
    write_empty=True,
):
    """Single-stage lock-free SparseCore top-k on all 32 subcores."""
    b, n = scores.shape
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    lanes = sc.num_lanes

    words = jax.lax.bitcast_convert_type(scores, jnp.int32)
    b_pad = _align_to(b, lanes)
    lengths = jnp.pad(row_lengths.astype(jnp.int32), (0, b_pad - b))

    slice_len = _align_to(n, lanes)
    num_waves = _cdiv(b, sc.num_cores * sc.num_subcores)
    cand_cap = _candidate_cap(cfg, sc, slice_len, k, b_pad, lanes)

    nb = _hist_buckets(cfg)
    priv_words = (lanes * nb + lanes) if not cfg.scan_count_hist else lanes

    out = pl.kernel(
        functools.partial(
            topk_kernel,
            cfg=cfg,
            b=b,
            n=n,
            k=k,
            slice_len=slice_len,
            num_waves=num_waves,
            write_empty=write_empty,
            lanes=lanes,
            num_subcores=sc.num_subcores,
            num_sc_cores=sc.num_cores,
            cand_cap=cand_cap,
        ),
        out_type=jax.ShapeDtypeStruct((b * k, ), jnp.int32),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                             needs_layout_passes=False),
        scratch_types=(
            pltpu.VMEM((slice_len, ), jnp.int32),
            pltpu.VMEM((priv_words, ), jnp.int32),
            pltpu.VMEM((nb + lanes, ), jnp.int32),
            pltpu.VMEM((k + lanes, ), jnp.int32),
            pltpu.VMEM((cand_cap + lanes, ), jnp.int32),
            pltpu.VMEM((b_pad, ), jnp.int32),
            pltpu.SemaphoreType.DMA((4, )),
        ),
        mesh=plsc.VectorSubcoreMesh(
            num_cores=sc.num_cores,
            num_subcores=sc.num_subcores,
            core_axis_name="core",
            subcore_axis_name="subcore",
        ),
        name=f"o7_topk_b{b}_n{n}_k{k}_{cfg.tag}",
    )(words.reshape(-1), lengths)
    return out.reshape(b, k)


def pick_partition(b, n, k, lanes=16):
    """Power-of-two partition factor P (1..32). Copied from the incumbent.

    This is deliberately unchanged -- how P is chosen is a different variant's
    axis, and moving it here would confound O7's numbers with that one's.
    """
    p_min = _cdiv(n, MAX_SLICE_WORDS)
    p = 1
    while p < p_min:
        p *= 2
    max_p = 32 // b if b < 32 else 1
    while p * 2 <= max_p:
        next_np = n // (p * 2)
        if next_np >= k and next_np >= lanes and (n % (p * 2)) == 0:
            p *= 2
        else:
            break
    return p
