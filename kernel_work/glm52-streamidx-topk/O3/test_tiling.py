"""Laptop test of O3's ladder arithmetic: no jax, no device, no TPU.

  python3 test_tiling.py

The bucketing direction is the one property here that is a correctness rule
rather than a performance one, so it gets an exhaustive sweep rather than a
table of examples.
"""
import sys

import tiling

# tpu7x, read back from pltpu.get_tpu_info().sparse_core and pinned here so the
# arithmetic is testable off-device. The kernel itself never uses these.
LANES = 16
SLICES_PER_WAVE = 32          # num_cores 2 x num_subcores 16
SC_VMEM_BYTES = 512 * 1024
MAX_SLICE_WORDS = 65536

K = 2048
N_A = 10240                   # regime (A): page_size 1024, 9 pages
N_B = 1048576                 # regime (B): page_size 512, 2048 pages


def check(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        return 1
    return 0


def main():
    bad = 0

    bad += check(tiling.slice_words_budget(SC_VMEM_BYTES) == MAX_SLICE_WORDS,
                 "slice_words_budget must reproduce the incumbent's 65536 "
                 "on a 512 KiB subcore")

    # -- the ceiling reproduces the incumbent's _pick_partition exactly -------
    for B, N, want in ((64, N_A, 1), (32, N_A, 1), (16, N_A, 2), (1, N_A, 4),
                       (16, N_B, 16), (1024, N_B, 16),
                       # b=1 is off the reachable ladder but pins the growth
                       # branch: p_min=16, then growth to 32 because 32768
                       # still clears k. O3 inherits that ceiling unchanged.
                       (1, N_B, 32)):
        got = tiling.static_partition(B, N, K, max_slice_words=MAX_SLICE_WORDS,
                                      lanes=LANES,
                                      slices_per_wave=SLICES_PER_WAVE)
        bad += check(got == want, f"static_partition({B},{N}) = {got}, want {want}")

    bad += check(tiling.partition_ladder(16) == (1, 2, 4, 8, 16), "ladder at 16")
    bad += check(tiling.partition_ladder(1) == (1, ), "ladder at 1")

    # -- BUCKET UPWARD, exhaustively -----------------------------------------
    # The rung chosen must cover the batch's longest live row. Swept over every
    # boundary and its neighbours, because off-by-one in the safe direction
    # costs a wave and off-by-one in the unsafe direction silently truncates
    # the candidate set.
    tn = N_B // 16
    ladder = tiling.partition_ladder(16)
    probes = [0, 1, K, K + 1]
    for p in ladder:
        probes += [p * tn - 1, p * tn, p * tn + 1]
    for live in probes:
        i = tiling.ladder_index(live, tn, ladder)
        p = ladder[i]
        bad += check(p * tn >= min(live, ladder[-1] * tn),
                     f"live={live} chose p={p}: coverage {p * tn} does not "
                     f"reach it -- DOWNWARD bucketing, the silent-failure case")
        if i > 0:
            bad += check(ladder[i - 1] * tn < live,
                         f"live={live} chose p={p} when p={ladder[i - 1]} "
                         f"would have covered it -- not minimal")

    # -- the five (B) shapes step 3 measures ---------------------------------
    # live_max, expected p, expected waves at the incumbent's p_max=16 vs O3.
    for name, B, live, want_p in (("B-D4-256k", 16, 262144, 4),
                                  ("B-D4-1M", 16, 1048576, 16),
                                  ("B-D1-1M", 16, 1048576, 16),
                                  ("B-RAG16", 16, 1048576, 16),
                                  ("B-P1024-256k", 1024, 262144, 4)):
        p = ladder[tiling.ladder_index(live, tn, ladder)]
        w0 = tiling.num_waves(B * 16, SLICES_PER_WAVE)
        w1 = tiling.num_waves(B * p, SLICES_PER_WAVE)
        x0 = tiling.x_crit(B, 16, tn, SLICES_PER_WAVE)
        x1 = tiling.x_crit(B, p, tn, SLICES_PER_WAVE)
        bad += check(p == want_p, f"{name}: p={p}, want {want_p}")
        print(f"  {name:14} live={live:>8}  p 16->{p:>2}  waves {w0:>3}->{w1:>3}"
              f"  X_crit {x0:>9}->{x1:>9}  modelled walk delta "
              f"{-0.895e-3 * (x0 - x1):>9.1f} us")

    # -- regime (A): p_max is 1 at b=64, so there is no switch and no change --
    cfg_a = tiling.static_partition(64, N_A, K, max_slice_words=MAX_SLICE_WORDS,
                                    lanes=LANES, slices_per_wave=SLICES_PER_WAVE)
    bad += check(tiling.partition_ladder(cfg_a) == (1, ),
                 "A-D64-8k must have a one-rung ladder: O3 is worth exactly "
                 "zero there and the code must show it, not measure it")

    print(f"\ntest_tiling: {'PASS' if not bad else f'FAIL ({bad})'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
