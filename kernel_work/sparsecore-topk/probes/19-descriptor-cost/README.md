# probe 19 — what does one extra SparseCore DMA descriptor cost?

Task 2b. Ran `variants/v04-pipelined-dma/descriptor_cost.py` **unmodified** (md5
`f08c9048b0c38b0e2109e3636177132b`) on a pod. The script is `v4`'s own instrument; I own only the
dispatch, the gates and the arithmetic below.

## Answer

**`g_raw = +517.5 ns` per descriptor per subcore. `g_pipe = +248.4 ns`.**

Both thresholds the verdict was staked on are missed, but not by the same margin and not in the
direction the model assumed.

**1. The 36 ns bar is missed by 6.9×.** `v4`'s pipelining increment (4.04 µs modelled, −2.13%)
needed `g < 36 ns`. Measured through `emit_pipeline` it is 248.4 ns. At the shipping chunk of 1024
words the increment is not +4.04 µs, it is **+27.82 µs** (3.5 extra descriptors/row/subcore ×
32 waves × 248.4 ns) against a stated 13.469 µs score-DMA prize — **net −14.35 µs, a 2.1× loss.**

**2. `emit_pipeline` does not add bookkeeping — it removes 269.1 ns/descriptor.** The model treated
pipelining as pure overhead layered on a raw blocking DMA, so it expected `g_pipe > g_raw`. Measured,
`g_pipe/g_raw = 0.480`: the pipeline overlaps enough of each descriptor's latency to halve its
marginal cost. **The right `g` for `v4`'s pipelined arm is 248.4 ns, not 517.5 ns**, and any model
that adds a positive bookkeeping term to a raw-DMA baseline double-counts.

**3. "`g > ~225 ns` kills `v4` at any chunk size" is not what the arithmetic says.** The descriptor
count halves with every doubling of the chunk, so the break-even `g` doubles with it while the
score-DMA prize (13.469 µs) does not move:

| chunk | extra desc/row/subcore | pipe cost | prize | net | break-even `g` | measured 248.4 ns |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 3.500 | 27.822 µs | 13.469 | **−14.353** | 120.3 ns | 2.07× over — **loses** |
| **2048** | 1.750 | 13.911 µs | 13.469 | **−0.442** | 240.5 ns | 1.03× over — **a wash** |
| 4096 | 0.875 | 6.955 µs | 13.469 | **+6.514** | 481.0 ns | 0.52× — wins on this arithmetic |

The 240.5 ns break-even at chunk 2048 reproduces the coordinator's "~225 ns break-even for 89% of
the prize" (0.89 × 13.469 / (1.75 × 32) = 214 ns), which confirms the two models agree. **At the
newly-primary `tc = 2048` arm the measured `g` sits 3.3% past break-even — inside the model's own
error, not a kill.** ⚠ The chunk-4096 row is arithmetic, not a prediction: the ragged prize is
itself a function of chunk granularity (you can only skip whole chunks), and **this probe does not
measure that function.** Do not read row 3 as "go bigger and `v4` wins."

**4. The ragged rung is untouched by all of this.** The coordinator's `ragged-blocking-only` arm
carries 8.08 µs of the 12.12 µs modelled prize and is insensitive to `g` by construction. Nothing
here bears on it. What this probe removes is the *pipelining* rung at chunk 1024, and it downgrades
the pipelining rung at chunk 2048 from "prize" to "coin flip".

**5. At one chunk the two constructions coincide, as they must.** `raw/c1` 38.188 µs vs `pipe/c1`
38.040 µs — 0.148 µs apart, inside the 0.153 µs null spread. A pipeline over a single step is the
blocking copy; the harness reproduces that identity, which is the cheapest available check that the
two arm families are not measuring different work.

## Arms and floor

One pool, 12 interleaved reshuffled rounds, `us_median`, every thunk warmed before round 0
(inherited by import from the frozen `glm_dsa_block_bench.measure`). Bytes per wave held **constant**
at 1.25 MiB across every arm; only the number of copies moving them changes. `slice_len = 10240`
words, `waves = 16`, chunk counts `[1, 2, 4, 5, 8, 10, 20]` (divisors that keep every arm's chunk a
multiple of 16 lanes, so no arm pays a ragged tail the others do not).

| arm | c1 | c2 | c4 | c5 | c8 | c10 | c20 | slope | max resid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `raw` | 38.188 | 48.533 | 65.866 | 74.385 | 98.995 | 115.358 | 196.759 | +8.2798 µs/count | 2.104 |
| `pipe` | 38.040 | 36.934 | 45.657 | 53.630 | 63.312 | 71.945 | 111.004 | +3.9745 µs/count | 2.700 |

**Resolution floor: three `raw/c20#null{0,1,2}` drift nulls, spread 0.153 µs = 0.079%**
(193.809 .. 193.962 µs). Dispatch floor 31.19 µs/call; both fitted intercepts (raw 32.013, pipe
31.685) land on it, so the fits are anchored where they should be.

⚠ **The linear fit is imperfect and the slopes are asymptotic, not local.** Max residuals are
2.1–2.7 µs, 14–18× the null spread. Local slopes for `raw` fall monotonically 10.345 → 8.140 µs per
count and flatten (asymptote ≈ 8.14 → **509 ns**, vs the fitted 517.5); `pipe` local slopes are
noisier (−1.11, 4.36, 7.97, 3.23, 4.32, 3.91) with `pipe/c5` the one outlier (`us_median` 53.630 vs
`us` 47.323, 15.5% hostbound — the largest `us`/`us_median` gap in the table) and an asymptote
≈ 3.91 → **244 ns**, vs the fitted 248.4. Both fitted values are within 2% of their asymptotes, so
the headline `g`s stand, but do not quote them to better than ±5%.

## Reconciling the dispatch note

The brief said `descriptor_cost.py`'s "arms are selected by env read **at module import**, so one
process per arm." **That is not this file** — it has no `os.environ` read at all, takes `--ctx`,
`--waves`, `--rounds`, `--json` from argparse, and builds every `raw/*`, `pipe/*` and `#null` arm as
thunks in a **single** `measure()` pool. The coordinator's later message confirms the caution belongs
to `v4`'s *kernel* file (`variants/v04-pipelined-dma/sparsecore_topk.py`), whose chunk value and
`nopipe`/`ragged-only` selectors do read the env at import. One process was correct here, and it is
also the stronger design: a single pool is what makes the raw and pipe slopes subtractable.

## PROBING.md five fields

**probe** — fit `time(C) = a + g_eff · C` at constant bytes per wave, for a raw blocking DMA and for
the same copies through `pltpu.emit_pipeline`; `g_eff = slope / waves`. Measure a slope, do not
divide a total.

**config** — image `us-central1-.../vllm-torchtpu-hning_google_com:fa63d6df`, `--ctx 9216`
(`n = 10240`), `--waves 16`, `--rounds 12`. `--map-dir` of a staged copy of `descriptor_cost.py` and
the two frozen instruments into `/root/p19/{variants/v04-pipelined-dma,instruments}` so the script's
`pathlib.Path(__file__).parents[2] / "instruments"` resolves. In-job md5 gate on all three mapped
files, asserted inside `RUN_CMD` (`MAP_GATE_OK`); the installed image copy of `sparsecore_topk.py`
read `1a89d451…`, but nothing timed here imports it for anything but `describe`/`topk_n` shape
helpers. Verbatim `RUN_CMD` read-back md5 `32429d98138806e9a599536bd895d858`, byte-identical to the
locally `bash -n`-checked folded line; no single quote, no `#`.

**job** — `j-53755323-aeb3-43d7-b06c` (Succeeded, `FINAL_RC=0`, 80 s), tagged `agent-jobs,measurer`.
Log `j-53755323-aeb3-43d7-b06c.txt`, machine-readable `descriptor_cost.json`. No `cdk` call was
denied. The `RUN_CMD` carried a 3-attempt retry with 60 s backoff for the
`/tmp/libtpu_lockfile ABORTED` race; it did not fire.

**answer** — `g_raw = 517.5 ns`, `g_pipe = 248.4 ns`, bookkeeping `−269.1 ns` (the pipeline helps).

**decided** — `v4`'s **pipelining rung at chunk 1024 is REFUTED** (−14.35 µs net, 2.1× the prize).
At `tc = 2048` it is **not resolvable from zero** (−0.44 µs net, 3.3% past break-even) and should be
carried into the ranking job as a measured arm rather than adjudicated from this model. `v4`'s
**`ragged-blocking-only` rung is unaffected** and remains the two-thirds of the prize worth ranking.
