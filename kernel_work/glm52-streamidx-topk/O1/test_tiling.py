"""Device-free tests for the O1 integer geometry: partition, waves, launches.

  python3 test_tiling.py

No `jax`, no TPU, no `pallas_call`. That is the entire reason `tiling.py` is a
separate file (code-style.md §10): the two claims O1 rests on that are pure
arithmetic can be checked here, in milliseconds, instead of being inferred from
a timing delta on a busy pod.
"""
import sys

import tiling

# tpu7x's SparseCore, as `pltpu.get_tpu_info().sparse_core` reports it. Written
# out here rather than read, because this file must run without a device; the
# SHIPPING path reads them (`pallas_mosaic_tpu._hardware_partition_args`) and
# this table is only a fixture. If a future chip disagrees, the fixture is what
# should be extended, not the rule under test.
TPU7X = dict(lanes=16, subcores_per_wave=32, sc_vmem_capacity_bytes=512 * 1024)

FAILS = []


def check(cond, msg):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {msg}", flush=True)
    if not cond:
        FAILS.append(msg)


def test_max_slice_words():
    print("max_slice_words derives the incumbent's hardcoded 64Ki")
    check(tiling.max_slice_words(512 * 1024) == 64 * 1024,
          "512 KiB subcore -> 65536 words, matching sparsecore_topk.py's "
          "MAX_SLICE_WORDS")


def test_partition_never_grows_past_the_capacity_minimum():
    print("with the guard, P is exactly the capacity minimum, everywhere")
    for b in (1, 4, 16, 32, 64, 128, 1024):
        for n in (10240, 65536, 262144, 1048576):
            p = tiling.pick_partition(b, n, 2048, **TPU7X)
            p_min = tiling.cdiv(n, tiling.max_slice_words(
                TPU7X["sc_vmem_capacity_bytes"]))
            want = 1
            while want < p_min:
                want *= 2
            check(p == want, f"b={b:4d} n={n:7d}: P={p}, capacity minimum {want}")


def test_growth_never_lowers_the_wave_count():
    print("the incumbent's growth loop only ever raises P at equal wave count")
    for b in (1, 2, 4, 8, 16):
        for n in (10240, 1048576):
            inc = tiling.pick_partition(b, n, 2048, fix_growth=False, **TPU7X)
            fix = tiling.pick_partition(b, n, 2048, fix_growth=True, **TPU7X)
            spw = TPU7X["subcores_per_wave"]
            check(tiling.num_waves(b, inc, spw) == tiling.num_waves(b, fix, spw),
                  f"b={b:3d} n={n:7d}: P {inc} vs {fix}, waves "
                  f"{tiling.num_waves(b, inc, spw)} either way -- the extra P "
                  f"buys no parallelism")


def test_the_one_shape_the_guard_moves():
    print("A-D16-8k is where it bites, and it is the only benchmarked shape")
    moved = []
    for name, b, n in (("A-D16-8k", 16, 10240), ("A-D64-8k", 64, 10240),
                       ("B-D4-256k", 16, 1048576), ("B-D1-1M", 16, 1048576),
                       ("B-P1024-256k", 1024, 1048576)):
        inc = tiling.pick_partition(b, n, 2048, fix_growth=False, **TPU7X)
        fix = tiling.pick_partition(b, n, 2048, fix_growth=True, **TPU7X)
        if inc != fix:
            moved.append(name)
        print(f"        {name:14} B={b:4d} N={n:7d}  P {inc} -> {fix}")
    check(moved == ["A-D16-8k"],
          f"P changes on exactly ['A-D16-8k'], got {moved} -- so a `pick` arm "
          f"reading non-zero anywhere else is measuring noise, not the guard")


def test_padded_n_matches_the_shared_instrument():
    print("padded_n agrees with the shared shapes instrument")
    for psz, pps, want in ((1024, 9, 10240), (512, 2048, 1048576)):
        got = tiling.padded_n(pps, psz, (2, 2, 2), 128)
        check(got == want, f"psz={psz} pps={pps}: {got} == {want}")


def test_launch_emptiness():
    print("2 of the 3 scoring launches are empty at every benchmarked shape")
    for name, D, R in (("A-D16-8k", 16, 16), ("A-D64-8k", 64, 64),
                       ("B-D4-256k", 4, 4), ("B-D1-1M", 1, 1),
                       ("B-P1024-256k", 0, 1)):
        n_live = tiling.num_nonempty_launches(D, D, R, 4)
        print(f"        {name:14} D={D:3d} R={R:3d}  ranges "
              f"{tiling.launch_ranges(D, D, R, 4)}  non-empty {n_live}/3")
        check(n_live == 1, f"{name}: 1 of 3 launches does work, 2 are pure "
                           f"dispatch")
    # The remainder launch is not always dead: R not a multiple of 4 wakes it.
    check(tiling.num_nonempty_launches(63, 63, 63, 4) == 2,
          "D=63 wakes the ragged-remainder launch as well -- the skip is a "
          "runtime predicate, not a static one, and this is why")


def main():
    for t in (test_max_slice_words,
              test_partition_never_grows_past_the_capacity_minimum,
              test_growth_never_lowers_the_wave_count,
              test_the_one_shape_the_guard_moves,
              test_padded_n_matches_the_shared_instrument,
              test_launch_emptiness):
        t()
        print("", flush=True)
    print(f"{len(FAILS)} failures")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
