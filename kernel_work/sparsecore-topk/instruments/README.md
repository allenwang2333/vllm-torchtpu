# Instruments — frozen at step 1, READ-ONLY

Every number in `report.md`, and every variant number in `results.md`, comes from one of these three
files at **exactly these hashes**. They are chmod `a-w` on purpose. A variant that needs a different
instrument gets a new arm added to a copy under `variants/`, never an edit here — otherwise the
baseline and the challenger stop sharing a yardstick and the comparison silently stops meaning
anything.

| file | md5 | live path it was frozen from |
| --- | --- | --- |
| `test_dsv4_sparsecore_topk.py` | `b3241859e471fc87fa6e7a50e4ffec4b` | `vllm-torchtpu/tests/kernels/` |
| `sctopk_2stage_bench.py` | `bdd978e6a76a9a73043b0a72c6a7a1d7` | `microbench/` |
| `glm_dsa_block_bench.py` | `08375d7068365e1a776c58fef09cd6d0` | `microbench/` (imported by the above for its timing harness) |

Subject under test: `sparsecore_topk.py` md5 `44353b347c96e0eedbd742c710d4540a` = `b11b6eb6`.

**The copies here are the record; the live paths are what actually runs.** `test_…py` must sit in
`tests/kernels/` to import the installed package, and the microbenches import each other by
filename. Before trusting a step-3 measurement, `md5sum` the live paths against this table — that
check is what makes "read-only" a fact rather than an intention.

## Correctness

```bash
cd <checkout> && python -m pytest tests/kernels/test_dsv4_sparsecore_topk.py -v --timeout=3600
```

Needs a TPU. 68 tests, ~60 s on one chip, or ~22 s fanned 4 ways over a pod's chips with disjoint
`-k` selections. **Baseline: 68/68 PASSED**, job `j-62275e52-1498-4e01-918e`.

`-k` fan-out is safe here (the shapes are independent), but the four shards' selections must union
to exactly the 68 collected ids — check with `--collect-only -q`, not by eye. A typo in a `-k`
expression silently runs fewer tests and still prints green.

## Performance

```bash
python sctopk_2stage_bench.py --decomp --ctx 9216 --batches 16 64 1024 --rounds 9
python sctopk_2stage_bench.py --passes --ctx 9216 --batches 1024   --rounds 9
python sctopk_2stage_bench.py --bw     --ctx 9216                  --rounds 9
```

**`--ctx 9216` is not the default and must be passed every time.** The file defaults to
`--ctx 132096`, which is `n=133120` — a shape no checked-in config produces (report §4). Every
number in `report.md` is at `--ctx 9216` for that reason, and a step-3 run that omits the flag is
not comparable to the baseline table in §7.

Read **`us_median`**, not `us`. Nine rounds, arms interleaved and reshuffled per round; the paired
`REPS_LO=20`/`REPS_HI=60` slope cancels host dispatch.

## Reading the harness's own output — three prints to ignore

These are not bugs to fix (the file is frozen), they are traps to route around:

1. **`CANDIDATE B gate: … % DMA -> BUILD B`** — divides `dma_only / stage1` with the **21.3 µs launch
   intercept inside the numerator**. Prints 66% where the honest hideable figure is 3.1%. Use probe
   04's slope instead. This one would have sent the whole exercise after a fabricated prize.
2. **`CANDIDATE A/A2 prize = gatherA/gatherB`** — at `P=1` those arms time ops that do not execute.
3. **`parts/full = 2.001` … "the pieces do NOT compose"** — also a `P=1` artifact; `sum(parts)`
   includes the three two-stage arms. At `P=1` compare `full` against `stage1` alone.

## What these instruments cannot measure

- **Anything multi-chip.** One chip only. Correct for this block — `scores` is replicated and fully
  local — but it means no change affecting collectives is detectable here. Nothing proposed is.
- **End-to-end effect.** Kernel time, not model time. Converting a kernel delta to an e2e delta needs
  the block's share of the step, which is report §4 unknown 4 and is **not** established.
- **Correctness of any timing arm.** `--bw`, `--decomp` and `--passes` verify nothing;
  `dma_only`'s output is an accumulator nobody checks. **If step 3 adds a chunked or double-buffered
  arm, nothing in this file will catch a copy that lands the wrong bytes** — a checksum against the
  blocking arm has to be added by hand, in the variant's own copy.
- **Decode deltas below ~3.4%**, the null spread at `b=64` (report §7). The caller's mechanism has a
  −2.02% decode ceiling (⛔ SUPERSEDED by `probes/08` (`j-300d4690`): **−1.97%**), i.e. **inside that floor** — decode must be reported as `not resolvable`
  rather than as `no effect` unless step 3 measures it differently.
