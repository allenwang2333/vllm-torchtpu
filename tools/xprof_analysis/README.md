# xprof_analysis — on-device step time, and where it goes

Reads TPU profiler output (`*.xplane.pb`) and answers four questions about a
serving run:

| Question | Command |
| --- | --- |
| What is in this capture? | `analyze.py phases` |
| How long is one engine step? | `analyze.py steptime` |
| Where does that step go? | `analyze.py breakdown` |
| Which ops moved between two trees? | `analyze.py diff` |
| Is candidate X actually faster than baseline? | `analyze.py compare` |

**Why on-device step time.** Client-side throughput
(`benchmark_serving.py --output-throughput`) carries scheduling, batching and
network noise; on this stack its capture-to-capture CV runs 0.2–0.8%. The
`XLA Modules` line of an xplane is the chip's own view of the same work and its
CV is about 0.5%, with the *composition* of a step held fixed by the phase
split. It is the tighter instrument for a kernel-level A/B, and it tells you
*which op* moved, which throughput never can.

Self-contained: one directory, no build step, no chip, and no dependency on the
rest of vllm-torchtpu.

---

## Quickstart

```bash
cd tools/xprof_analysis

# 1. What did I capture, and which ranks reported?
./analyze.py phases /path/to/prof/baseline_a

# 2. How long is a step? (pin the ranks that `phases` printed)
./analyze.py steptime /path/to/prof/baseline_a --ranks 0,2,4,6

# 3. Where does it go?
./analyze.py breakdown /path/to/prof/baseline_a --ranks 0,2,4,6

# 4. Candidate vs baseline, three captures each, with error bars.
./analyze.py compare --ranks 0,2,4,6 \
    --arm baseline=prof/baseline_a,prof/baseline_b,prof/baseline_c \
    --arm fusion=prof/fusion_a,prof/fusion_b,prof/fusion_c
```

First run over a capture parses the protobufs (~9 s per capture) and writes a
JSON cache into `<capture>/.xprofkit/`. Every run after that is ~0.3 s.

### Prerequisites

Python 3.10+, plus `jax` (for `jax.profiler.ProfileData`) and `pyyaml`. Both are
already in the serving venv, so if the system python cannot import jax:

```bash
~/vllm_env/bin/python analyze.py phases /path/to/prof/baseline_a
```

Nothing here installs anything or touches the serving environment.

### What a capture directory looks like

```
prof/baseline_a/
├── prefill_heavy/
│   └── plugins/profile/2026_08_20_19_04_11/
│       ├── rank0_t1v-n-xxxx.xplane.pb
│       ├── rank2_t1v-n-xxxx.xplane.pb
│       └── ...
└── decode_only/
    └── plugins/profile/...
```

One subdirectory per inference phase; the tool discovers them. A flat directory
of `.xplane.pb` files also works and is reported as phase `.`.

---

## Read this before trusting a number

Three things about these traces are counter-intuitive, and each one has produced
a wrong answer here before. The tool handles all three; you need to know about
the first.

**1. Ranks are not interchangeable — pin them.** Only about half the ranks emit
a device plane, and the ones that do disagree: the four ranks of `baseline_a`
range from 294,430.7 to 317,520.6 µs of mean step time, which `steptime` reports
as **7.55% of the mean**. A comparison that reads a different set on each side
is measuring the rank set.

The mismatch is not exhibited by anything in this repository — all nine captures
here hold exactly ranks 0,2,4,6 for `prefill_heavy`, and `diff baseline_a
baseline_b` prints +0.46% pinned and unpinned alike. Nor could it be, on two
captures: `diff` already pins to the intersection of the pair, and `compare` to
the intersection across every capture in every arm. Pass `--ranks` anyway, and
pass the *same* value to every command. The auto-pin protects one comparison; it
does not make two numbers you read out of two separate runs comparable, and that
is how the ranks get you. `phases` prints the set to use.

**2. A step is not a module event.** TorchTPU splits one engine step into the
decoder graph plus ~6 small side graphs (sampling, token scatter, `as_strided`).
Counting every module event divides the step time by about 7. The tool picks the
module with the most *total* time — which also survives recompile renames.

**3. One phase directory can hold two captures.** A restarted or re-armed
profiler leaves an older, partial dump beside the good one. Averaging across
both silently mixes two different runs; the tell is an occupancy far from 1.00.
The tool keeps the dump with the most ranks and reports what it dropped.

Two more things worth knowing when reading the output:

- **`Async XLA Ops` is not part of the step.** It totals ~2.3× the step time
  because it overlaps it. `breakdown` reports it on its own line and never sums
  it into a "% of step" column — a large value means collectives are being
  hidden successfully.
- **Step-to-step spread inside one capture is mostly real, not noise.** Per-rank
  CV runs 6–34%, driven by batch heterogeneity and ramp partials. The spread
  that matters for a go/no-go is *capture-to-capture on identical bytes*, and
  `steptime`/`compare` report that separately.

---

## Commands

### `phases` — orientation

```
$ ./analyze.py phases prof/baseline_a prof/fusion_a

| capture                  |              phase |            ranks |   n |  steps |  mean step us |
|--------------------------|--------------------|------------------|-----|--------|---------------|
| baseline_a               |      prefill_heavy |          0,2,4,6 |   4 |     36 |     305,893.4 |
| fusion_a                 |      prefill_heavy |          0,2,4,6 |   4 |     36 |     309,914.1 |

phases in every capture: prefill_heavy
  prefill_heavy: pin with --ranks 0,2,4,6
```

Run this first. The last line is the `--ranks` argument for everything else.

### `steptime` — the distribution

```
$ ./analyze.py steptime prof/baseline_a --ranks 0,2,4,6

## baseline_a
### prefill_heavy

| rank  | steps |     mean us |   median us |      sd us |   cv % |  sem % |      min us |      max us |
|-------|-------|-------------|-------------|------------|--------|--------|-------------|-------------|
| 0     |     9 |   294,430.7 |   323,537.8 |   90,329.2 |  30.68 |  10.23 |    53,659.5 |   327,981.3 |
| 2     |     9 |   317,520.6 |   323,542.0 |   21,081.1 |   6.64 |   2.21 |   261,780.8 |   327,981.2 |
| 4     |     9 |   315,686.8 |   323,532.6 |   26,344.3 |   8.35 |   2.78 |   245,840.3 |   327,976.8 |
| 6     |     9 |   295,935.2 |   323,532.5 |   85,891.7 |  29.02 |   9.67 |    66,999.6 |   327,973.8 |

rank-to-rank: 294,430.7 - 317,520.6 us (7.55% of the mean) over 4 ranks
```

Pass more than one capture and it adds a delta table and a
capture-to-capture spread table (your noise floor).

| Option | Effect |
| --- | --- |
| `--label NAME` | name each capture, in order, instead of using the directory name |
| `--estimator mean\|median\|p90\|all` | which location estimator to compare on. Default `mean`. |
| `--components` | also show summed op time and async time beside the step |
| `--emit` | print `key: value` lines for a harness to harvest |

`--components` prints `summed op time: ... 1.00x the step`. If that ratio is far
from 1.00, correction 3 above has bitten you.

### `breakdown` — where a step goes

Two tables by default: where the time went, and which part of the model spent
it. Everything else is behind a flag.

```
$ ./analyze.py breakdown prof/baseline_a --ranks 0,2,4,6

step 305,893.4 us · ops 306,221.0 us · occupancy 100.1% · 36 steps over rank(s) 0,2,4,6 · async 715,189.6 us (2.34x, overlapped)

| category             |     us/step |   % step | op families                                          |
|----------------------|-------------|----------|------------------------------------------------------|
| linear_attention     |    25,166.4 |    8.23% | fused_conv1d_gdn_per_seq +1                          |
| attention            |    47,262.0 |   15.45% | sc_ragged_gather_reduce_v2 +8                        |
| moe                  |    31,464.8 |   10.29% | gmm_v2-g_64-m_163840-k_4096-act_silu-n_2048-tm_ +1   |
| sort                 |    50,538.7 |   16.52% | sort                                                 |
| collective           |    77,356.8 |   25.29% | reduce-scatter +16                                   |
| matmul               |    14,274.3 |    4.67% | multiply_convert_fusion, fusion +2                   |
| layout               |    25,985.3 |    8.49% | copy, reshape, slice_convert_fusion +31              |
| elementwise          |    17,435.7 |    5.70% | multiply_reduce_fusion, abs_reduce_fusion +21        |
| ...
| TOTAL                |   306,221.0 |  100.11% |                                                      |

### by block — declared by the taxonomy, summed from the trace

| block                |     us/step |   % step | categories                         |
|----------------------|-------------|----------|------------------------------------|
| attn                 |    72,428.4 |   23.68% | attention, linear_attention        |
| moe                  |   159,360.2 |   52.10% | collective, sort, moe              |
| (no block)           |    74,432.3 |   24.33% | layout, elementwise, matmul +4     |
| TOTAL                |   306,221.0 |  100.11% |                                    |

  **Declared, not derived.** ...  (the full note the tool prints is below)

unclassified 11.9 us/step: custom-call.20 (1.24), custom-call.12 (1.24), ...
```

The second table is the same total again, rolled up into the part of the model
each category was **declared** to belong to — see "Block subtotals" below, and
"Declaring blocks" for how to write the declarations. A taxonomy whose rules
carry no `block:` prints no second table.

`async` in the header is work that overlaps the step rather than adding to it,
so it is never summed into a "% step" column. A large multiple is good news: it
means the collectives are being hidden rather than waited on.

The **op families** column names what is actually inside each category, largest
first, with `+N` for the rest. That distinguishes a category that is one
dominant kernel (`sort`) from one that is a long tail of small ops — a
different problem needing a different fix.

`--units` adds the same total split by execution unit:

```
### by execution unit — same total, split a different way

| unit                 |     us/step |   % step | op families                                          |
|----------------------|-------------|----------|------------------------------------------------------|
| tensorcore           |   194,325.5 |   63.53% | sort, fused_conv1d_gdn_per_seq +88                   |
| sparsecore           |   111,895.4 |   36.58% | reduce-scatter, sc_ragged_gather_reduce_v2 +5        |
```

SparseCore is a device, not a category: `sc_ragged_gather_reduce_v2` is both MoE
work and SparseCore work, so folding the two axes into one loses whichever you
were not thinking about. On this stack the ops whose *name* starts `sc_` are
0.8% of the step while the ops that actually run on SparseCore are 36.6%.
SparseCore has no VPU and its own bandwidth, so anything landing there must be
judged against the SparseCore roofline rather than TensorCore peak.

Occupancy near 100% is the sanity check: the op line is a partition of module
time, so it should come out at 1.00×. The unclassified list is your to-do list
for the taxonomy — the tool warns when it exceeds 0.5% of op time.

| Option | Effect |
| --- | --- |
| `--taxonomy PATH` | classification rules (default `taxonomies/default.yaml`) |
| `--names N` | how many families to name per category (default 3, as many as fit) |
| `--families N` | also print a separate top-N op family table with call counts |
| `--unclassified N` | how many unclassified ops to name (default 8) |
| `--sources N` | also print the top N source lines by device time (default off) |
| `--units` | also split the same total by execution unit (TensorCore / SparseCore) |
| `-v`, `--verbose` | follow every table with the full list of ops behind each row |

#### Verbose

Every table names its three biggest op families and counts the rest — enough to
tell one dominant kernel from a long tail, not enough to answer *which* op.
`-v` follows each table with the whole list, nothing folded:

```
$ ./analyze.py breakdown prof/baseline_a --taxonomy taxonomies/qwen35_397b.yaml -v

#### every op family, by category

| category             | op family                                          |     us/step |   % step |
|----------------------|----------------------------------------------------|-------------|----------|
| attn_core            | RPAm-p256-b1-q256-k256                             |     6,806.7 |    2.23% |
|                      | RPAd-p256-b8-q1-k1792                              |       677.2 |    0.22% |
|                      | rpa_metadata_schedule                              |        11.6 |    0.00% |
| linear_attn_core     | fused_conv1d_gdn_per_seq                           |    23,706.2 |    7.75% |
|                      | fused_conv1d_gdn_batched                           |     1,460.2 |    0.48% |
| ...
```

One name can be reused across categories — `multiply_convert_fusion` is the QKV
projection *and* the linear-attention input projection *and* the output
projection — so the block expansion tags each op with the category it was
counted under:

```
#### every op family, by block

| block                | op family                                          |     us/step |   % step |
|----------------------|----------------------------------------------------|-------------|----------|
| attn                 | RPAm-p256-b1-q256-k256 [attn_core]                 |     6,806.7 |    2.23% |
|                      | multiply_convert_fusion [attn_qkv_proj]            |     2,644.4 |    0.86% |
|                      | multiply_convert_fusion [out_proj]                 |     1,461.7 |    0.48% |
| ...
```

Time only, no instance count. The count used to print in these tables and got
read as structure — *this many of them, so one per layer of that kind* — which
is an inference this tool no longer makes anywhere (see "Declaring blocks").
`--families` still prints it, where the column beside it is us/call and the
reading is unambiguous, and `--json` carries it per cell — that argument is
about a page a human skims, and it does not apply to a machine-readable field.

`-v` applies to whichever tables are on, so it composes with `--units` and
`--sources N`.

#### Source lines

`--sources N` adds a table of whose code asked for the work, which is a
different question from what the compiler called it:

```
$ ./analyze.py breakdown prof/baseline_a --ranks 0,2,4,6 --sources 6 \
      --taxonomy taxonomies/qwen35_397b.yaml

### by source line — top 6, whose code asked for it

| file:line                                              |     us/step |   % step | categories                         |
|--------------------------------------------------------|-------------|----------|------------------------------------|
| …/src/vllm_torchtpu/layers/vllm/fused_moe.py:239       |   115,858.0 |   37.88% | moe_gather_reduce, moe_sort +4     |
| vllm/vllm/model_executor/parameter.py:127              |    34,130.6 |   11.16% | linear_attn_core +3                |
| …/vllm_torchtpu/kernels/quantized_matmul/util.py:94    |    21,589.8 |    7.06% | linear_attn_in_proj, quant +2      |
| …/src/vllm_torchtpu/layers/vllm/moe_routing.py:273     |    17,049.5 |    5.57% | moe_sort, moe_route                |
| …/src/vllm_torchtpu/layers/vllm/attention.py:666       |     9,753.4 |    3.19% | attn_core, attn_glue +1            |
| …/src/vllm_torchtpu/compilation/tpu_compiler.py:396    |     9,178.7 |    3.00% | collective                         |
| (+34 more lines)                                       |    16,090.4 |    5.26% |                                    |
| (compiler-generated) collective                        |    68,173.3 |   22.29% | reduce-scatter, copy-done +9       |
| (compiler-generated) layout                            |    11,847.4 |    3.87% | copy, slice-start, slice-done +5   |
| (compiler-generated) elementwise                       |     1,497.7 |    0.49% | convert, add_bitcast_fusion +4     |
| (compiler-generated) sparsecore_glue                   |     1,026.6 |    0.34% | sparse-core-data-format-call +2    |
| (compiler-generated) everything else                   |        25.5 |    0.01% |                                    |
```

Use it to *find* rules, then write them as `src:` conditions in the taxonomy —
see "Attributing ops by source line" below for the coverage limits and for why
`pallas.py:638` is half the step.

#### Block subtotals

The second default table, with the worked example
`taxonomies/qwen35_397b.yaml`, which declares three blocks:

```
$ ./analyze.py breakdown prof/baseline_a --ranks 0,2,4,6 \
      --taxonomy taxonomies/qwen35_397b.yaml

### by block — declared by the taxonomy, summed from the trace

| block                |     us/step |   % step | categories                         |
|----------------------|-------------|----------|------------------------------------|
| attn                 |    13,443.0 |    4.39% | attn_core, attn_qkv_proj +3        |
| linear_attn          |    42,641.1 |   13.94% | linear_attn_core +2                |
| moe                  |   212,325.6 |   69.41% | collective, moe_sort +5            |
| (no block)           |    37,811.3 |   12.36% | quant, layout +6                   |
| TOTAL                |   306,221.0 |  100.11% |                                    |

  **Declared, not derived.** A row is the sum of the categories whose taxonomy rule carries that `block:`, and
  nothing in the trace says which part of the model an op belongs to — there are no layer boundaries in a step,
  the compiler names glue for what it does rather than for whose it is, and a collective's source line is the
  compiler's own stack. So each row is exactly as right as the rules behind it (the `categories` column names
  the largest), and a change of sharding can move work between blocks with nothing here noticing. Before
  quoting a number from this table, read those rules in `qwen35_397b` — README, 'Declaring blocks'.
  **(no block)** is the 12.4% in categories no rule assigned to one; the rows do not nest and they sum to the step.
```

Read the note under the table before the numbers above it. **Every microsecond
in this table is measured; the split between the rows is not measured at all.**
It is a line of YAML per rule, and the tool prints it back rather than deriving
it, because the trace does not contain it:

- there are no layer boundaries in a step — the profile is one flat list of ops;
- compiler glue is named for what it does (`reshape`, `copy`), never for whose
  it is, which is why `src:` rules and not names do most of the work here;
- a collective's recorded stack is the *compilation* stack, so two all-reduces
  from different parts of the model are byte-for-byte indistinguishable. See
  "Limitations".

What the table is good for is the same thing the category table is good for, one
level up: on this stack MoE is 69% of the step, the two attention kinds are 18%
between them, and 12% is in categories nobody has declared yet. What it is *not*
is per-layer cost. Divide by your own layer count if that is the question; the
tool will not, because the divisor would have to be written into the taxonomy
and it would be wrong for every other model, sharding and phase.

**(no block)** is not a residue to be cleaned up. `norm`, `quant`,
`matmul_other` and the layout catchalls each hold work from more than one block,
and the only handle that could split them — the source line — points at a shared
utility file. A number in the wrong block is worse than a number in no block.

### `diff` — which ops moved

```
$ ./analyze.py diff prof/baseline_a prof/fusion_a --ranks 0,2,4,6

step time  305,893.4 -> 309,914.1 us  (+1.31%)   [36 / 36 steps]

### by category
| category                                       |     base us |     cand us |       delta |
|------------------------------------------------|-------------|-------------|-------------|
| collective                                     |   77,356.80 |   80,397.19 |   +3,040.40 |
| attention                                      |   47,262.00 |   48,012.04 |     +750.04 |
| moe                                            |   31,464.79 |   31,418.97 |      -45.83 |
| ...

### by op family (suffix stripped — read this one)
| op family                                      |         category |     base us |     cand us |       delta |
|------------------------------------------------|------------------|-------------|-------------|-------------|
| distributed_all_gather_into_tensor             |       collective |    7,136.65 |    9,204.01 |   +2,067.37 |
| RPAm-p256-b1-q256-k256                         |        attention |    6,806.74 |    7,383.55 |     +576.81 |
```

Everything is **microseconds per engine step**, not per capture — two captures
never contain the same number of steps, so raw totals are not comparable.

Read the family table, not the exact-name table: exact names carry
compiler-assigned suffixes (`copy.18`) that are not stable across builds, so one
unchanged op can show up as a delete plus an add. The exact-name table is for
grepping a dumped module once you know where to look.

One family can span several categories — the compiler hands one name to three
different projections — so its `category` cell names where most of the family's
time went and counts the rest as `+N`. The category table above it is the exact
split. `breakdown --families` uses the same convention.

The three tables above tell a story the step-time number cannot: this
candidate's `moe` change — the thing it actually edited — is flat at −46 µs, and
the entire +1.31% is collective wait.

**The shipped captures are not symmetric.** Of the nine under `prof/`, only
`baseline_a` and `sort_algo_c` carry the op-stats sidecar — nine
`*.op_stats_v2.pb` each; the other seven, `fusion_a` included, carry none. The
example above is therefore itself the mismatched case: run it against
`taxonomies/qwen35_397b.yaml` and the report warns that base carries source
lines and cand does not, then names the eight `src:` rules that fired on
`baseline_a` only. Those eight are not dead rules and the taxonomy needs no
edit — the instrumentation differs, not the code — which is why one-sided rules
are counted and worded separately from rules that fired on neither side.

**Calibrate it.** Diff two captures of *unchanged* code. Whatever that reports
is the floor below which a row means nothing.

### `compare` — arms, with error bars

```
$ ./analyze.py compare --ranks 0,2,4,6 \
      --arm baseline=prof/baseline_a,prof/baseline_b,prof/baseline_c \
      --arm fusion=prof/fusion_a,prof/fusion_b,prof/fusion_c

## prefill_heavy
  estimator: mean step time · ranks pinned to 0,2,4,6 · reference arm: baseline

| arm              |   n |      step us |     sd us |      cv |   vs ref |       t |                 95% CI |
|------------------|-----|--------------|-----------|---------|----------|---------|------------------------|
| baseline         |   3 |    307,339.5 |   1,461.8 |   0.48% |      ref |         |                        |
| fusion           |   3 |    308,563.9 |   1,204.0 |   0.39% |   +0.40% |   +1.12 |       [-0.73%, +1.53%] |

## verdict
  prefill_heavy: baseline is the reference (307,339.5 us, n=3, cv 0.48%)
  prefill_heavy: fusion is NOT DISTINGUISHABLE from the reference (CI [-0.73%, +1.53%], n=3)
```

The first `--arm` is the reference; every other arm is Welch-tested against it.
Ranks are pinned to the intersection across *all* captures in *all* arms, so no
arm gets credit for having reported an easier set of ranks.

The confidence interval is Welch's t with a Satterthwaite df, from a lookup
table. Welch's df is fractional almost always, and the gap between two tabled
rows is closed **downwards in df**, which is upwards in the critical value: the
interval is slightly wide on purpose, because this feeds a promotion gate.
Erring the other way at df 2.32 gives an interval 20% too narrow.

Three captures per arm is the practical minimum for a CI that means anything.
At n=1 there is no within-arm spread to estimate, so no interval exists: the
tool prints the point estimate, `n=1: not comparable` in place of the CI, and a
verdict of `NOT COMPARABLE`. It does not say "not distinguishable" — nothing
was measured to distinguish.

---

## Extending

### Adding a taxonomy rule

`taxonomies/default.yaml` is generic — nothing in it names a model or a kernel.
`taxonomies/qwen35_397b.yaml` is the worked example of a model-specific one.

```bash
cp taxonomies/default.yaml taxonomies/mystack.yaml
$EDITOR taxonomies/mystack.yaml
./analyze.py breakdown prof/baseline_a --taxonomy taxonomies/mystack.yaml
```

`categories` is an ordered list of `{name, pattern, hlo, src, block}` and **the
first match wins**. Put new rules *above* the generic `layout` / `elementwise` /
`fusion` catchalls at the bottom, or they will never fire. Two rules may share a
`name`; they fold into one row. Then re-run `breakdown` and check the
unclassified line shrank.

| Field | Matched against | Answers |
| --- | --- | --- |
| `pattern` | the op name — `multiply_convert_fusion.44` | what the compiler called it |
| `hlo` | the HLO signature | what the instruction *is* |
| `src` | `file.py:123` from the op-stats sidecar | whose code asked for it |
| `block` | nothing — it is not a condition | which part of the model it belongs to |

The three conditions are case-insensitive regexes, and a rule may use any
combination; with several, all must match. `block` is the odd one out: it
matches nothing, it only labels, and it is what the "by block" table sums. See
"Declaring blocks".

Anything whose only symptom would be a quietly wrong table is rejected at load
time, with a `ValueError` naming the file and the entry index:

* **shape** — a file that is not a mapping, an unknown section (`categorys:`),
  `categories` that is not a list, a rule that is not a mapping (`- name: a`
  with the rest at the wrong indent parses as `None`), an unknown field
  (`patern:`);
* **conditions** — a rule with no condition at all, an invalid regex, and a
  condition YAML read as *structure* rather than as text: unquoted
  `pattern: [copy]` is a list, and the character class is exactly what this
  field is for;
* **`name`** — absent, not a string (`name: 60`), blank, padded with
  whitespace (`' layout '` and `layout` would be two rows with one heading), or
  one of the three labels the renderer spends rows of its own on:
  `unclassified`, `(no block)`, `TOTAL`;
* **`block`** — every check `name` gets, for the same reasons: it is the other
  row heading and the other join key. Plus two rules that share a `name` and
  disagree about it — including one that declares a block and one that does not.

The reserved labels are one list checked in both fields, so a fourth cannot be
guarded in one and forgotten in the other. Case is *not* folded — renaming a
`MoE` somebody capitalised on purpose would be worse than the split — but a file
declaring both `moe` and `MoE`, or both `norm` and `Norm`, gets a line under the
tables saying so. Padding is not folded either: the loader is the last place
that should quietly rewrite what you wrote, so `moe ` is rejected rather than
stripped.

#### Why `hlo` exists

The op *event name in the profile is the full HLO instruction text*, not just
the op name — output shape, operand shapes and dtypes, `kind=`,
`custom_call_target=`, `async_execution_thread=`. `hlo:` matches against a
one-line summary of the fields that survive a recompile:

```
kind=kOutput out=bf16[2048,17408] in=f8e4m3fn[2048,4096],f8e4m3fn[17408,4096]
```

You need this because op names lie on a fused stack. **There is no `dot` op in
this module at all.** XLA fuses the fp8 dequant onto every matmul and names the
result after two of its *interior elementwise* ops, so the attention QKV
projection arrives called `multiply_convert_fusion` and a name-only taxonomy
files ~7% of the step under `elementwise`. `kind=kOutput` is the TPU marker for
a fusion rooted at a dot — but it is not sufficient alone, because a fusion
rooted at a *reduction* is kOutput too. The discriminator that works is a 2-D
output with two 2-D operands:

```yaml
- {name: matmul, hlo: 'kind=kOutput.* out=\w+\[\d+,\d+\] in=\S*\[\d+,\d+\],\S*\[\d+,\d+\]'}
```

Adding that rule moved `matmul` from 0.08% to 4.67% and `elementwise` from
10.10% to 5.70%. Shapes make rules sharper still — `out=\w+\[\d+,17408\]` is
the QKV projection specifically — at the cost of being tied to one model *and*
one sharding.

A stale shape rule does **not** show up as a large `unclassified` bucket. The
op falls through to the next rule that matches — the QKV projection lands back
in `elementwise` — so the report stays fully classified and is merely wrong.

Three things catch it, in the order you will meet them.

* **The dead-rule line**, printed under the tables on every run and needing no
  flag: `2 rule(s) of taxonomy 'qwen35_397b' claimed no op here: categories[18]
  'quant', categories[20] 'embedding'`. It counts **rules**, not categories, and
  the difference is the point — most categories worth writing are several rules,
  so a category-level check is blind to every death but the last. Change
  `norm`'s `src:` from `layernorm\.py` to `layer_norm\.py`, one character, and
  the category falls from 1,567.8 to 9.9 µs/step while the `norm` row keeps
  printing; the dead-rule line names the rule that died. The index is the
  rule's position in `categories:`, so it points at the line. `--json` carries
  the same list under `dead_rules`. `diff` prints it too — the shadowing that
  moves a `breakdown` row moves a delta, which is the number a go/no-go is made
  on — and there **dead means dead on both captures**. A rule that fired on one
  side and not the other is a fact about the change under review, not a defect
  in the file, so it is counted and worded separately (`one_sided_rules`).
* A rule can also be dead because the capture does not contain that work, which
  is harmless and common — both rules named above are that case, and both are
  kept on purpose. **The report cannot tell the two apart**, so it names them
  and says so.
* **`--families`** for the other half of the symptom: a large family sitting in
  a catchall, when the rule that should have claimed it is alive but too narrow.

The same applies to `src` rules against a capture with no op-stats sidecar;
`breakdown` and `diff` warn when a capture has no source lines at all, since
then *every* `src` rule is dead.

Signatures and source lines are cached, and the cache key covers the taxonomy's
inputs to them — the `PLUMBING` list, the signature field *patterns* (the regex
text, not just the field names), the group and operand limits, whether the
sidecar reader is importable and which `xprof` version it came from — so
editing a taxonomy needs nothing bumped, and tightening one of those regexes
re-derives every warm capture automatically.
Changing the deriving *code* (`hlo_signature`, a renderer body, `caller_of`, the
sidecar join) needs `CACHE_VERSION` bumped in `trace.py`.

### Declaring blocks

A **block** is a part of the model — `attn`, `moe`, whatever you name — and the
"by block" table is the sum of the categories declared to be in it. The
declaration is one field on a rule:

```yaml
- {name: moe_gmm,   block: moe,  pattern: '^gmm|grouped[-_]?matmul|ragged[-_]?dot'}
- {name: attn_core, block: attn, pattern: '^RPA[md]?[-_]|^rpa_'}
- {name: moe_glue,  block: moe,  src: 'layers/vllm/fused_moe\.py'}
```

That is the whole mechanism. A category with no `block:` is reported under
`(no block)`; a taxonomy where no rule has one prints no block table at all.
Blocks are free-form labels — the set of them is whatever the file declares, in
file order — and rules sharing a `name` must agree on the block, because the
table groups by category and one category is one row.

**The rule pointing at the kernel is not the interesting one.** `attn_core` was
already legible from the op name. What the block table buys you is the *glue*:
the reshapes, pads and slices a kernel wrapper emits around its call, which the
compiler names `reshape` and a name-only taxonomy piles into `layout`. On this
stack that is about an eighth of a kernel's true cost. The recipe is a `src:`
rule with a `block:` on it — the caller frame names the wrapper, the block says
whose wrapper it is:

```yaml
- {name: moe_glue,         block: moe,         src: 'layers/vllm/fused_moe\.py'}
- {name: linear_attn_glue, block: linear_attn, src: 'model_executor/parameter\.py'}
- {name: attn_glue,        block: attn,        src: 'layers/vllm/attention\.py'}
```

Those three rules moved 6.75% of the step out of `layout` and into the blocks
that caused it.

#### Declared, not inferred — and why

The tool used to infer the block from *arity*: an op whose instance count per
step matched one of the model's declared layer counts, and no multiple of the
others, runs once per layer of that type. It read as a counting argument rather
than a guess, and it is gone. Three reasons, in increasing order of how badly
they bite:

1. It needed the layer counts of one specific model written into the taxonomy —
   a constant that is wrong for the next model, wrong after a depth change, and
   silently wrong rather than loudly. The file stopped describing a serving
   stack and started describing a checkpoint.
2. It cannot tell two layer types of equal depth apart *at all*. Equal counts
   produce an identical fit, and the tie-break is a guess wearing arithmetic.
3. Any count that divides two ways needs a rule to break it, and every such rule
   moves real microseconds between blocks with no symptom in the output. This
   was measured, not feared: on `baseline_a`, tightening the old tolerance from
   ±1 instance to exact moved 16,508.6 µs between blocks — **5.40% of the
   305,893.4 µs step**, the denominator every `% step` column in this README
   uses. Nearly all of it left `moe` (16,257.5 µs); the remainder came out of
   `linear_attn` (247.0) and `attn` (4.2). A one-instance slack, invisible on
   the page, was worth a twentieth of the report.

A declaration has none of those properties, and it has one the inference never
had: it is visible. It is a line of YAML someone wrote, in a file under review,
and the report says so under the table.

#### What a declaration cannot fix: collectives

Both shipped taxonomies declare `collective` into `moe`. That is a claim about
the *sharding*, not a measurement. It holds when the experts are sharded and
attention is replicated or data-parallel — the common serving configuration, and
the one behind every number in this README (attn dp8 · ep8).

The one rule books a whole category, so here is the whole category. On
`baseline_a` it is 77,356.8 µs over 17 op families, and the evidence for the
declaration covers most but not all of it:

| what                                       |    us/step | % step | why it is `moe` |
|--------------------------------------------|------------|--------|-----------------|
| `reduce-scatter` ×120, `all-gather` ×120    |   69,603.3 | 22.75% | the ep8 pair around the grouped matmul, counted per step and matched to it |
| `distributed_all_gather_into_tensor` ×61    |    7,136.6 |  2.33% | inferred, not matched: it is the only other collective, and attention is dp |
| `copy-start`/`copy-done` (2,323 done)       |      575.0 |  0.19% | the transfers the two above schedule around; not separately attributed |
| 12 `prepare_*` and offload async families   |       41.8 |  0.01% | per-collective bookkeeping, ≤24.5 µs each |

So 2.53% of the step (7,753.4 µs) is booked to `moe` because nothing else in
this configuration issues a collective — a sound argument for the sharding
stated, and one no rule in the file can check. `baseline_a` does contain 0
all-reduce instructions and 0.0 µs of them, confirmed on all four ranks; the
`all-reduce` half of the pattern is there for other deployments. Under
tensor-parallel attention the whole 25.29% is wrong, and wrong with no symptom
whatsoever.

Nothing in the trace can correct it, and this is worth knowing before you try:
two all-reduce instructions from different parts of the model have the same op
name, the same shapes, the same `replica_groups`, and **byte-for-byte identical
~70-frame source stacks**. Not similar — identical, frame for frame, from
`torch/_library/autograd.py:41` all the way down.

*You cannot reproduce that from the captures in this repository, and it is not a
claim about them.* Every capture here is attn dp8 · ep8 and contains 0
all-reduce instructions; the pair above was read off a tensor-parallel
configuration not shipped with the tool. What the captures here do show is the
consequence — every collective in the module carrying the compiler's own stack,
which is why `reduce-scatter` has no caller and `all-gather` reports
`tpu_compiler.py:396`.

The reason is that a collective is functionalised into the graph at trace time,
so what the profiler records is the *compilation* stack; the model frame that
wanted it was never on the stack at execution. A Pallas kernel keeps a real
frame (`fused_moe.py:239`) only because it dispatches at runtime through
`__torch_function__`.

So no `pattern`, `hlo` or `src` rule can split them. If you shard otherwise,
delete the `block` from that rule — `(no block)` is the honest answer — or split
the rule on something you know out of band. See "Limitations".

### Attributing ops by source line

The op name is what the compiler called the instruction. `src` is whose code
asked for it — `file.py:123`, matched by regex like the other two conditions:

```yaml
- {name: norm,  src: 'layernorm\.py'}
- {name: quant, src: 'quantized_matmul/util\.py'}
```

**Which frame `src` reports.** Not the innermost one. For anything dispatched
through Pallas the innermost frame is the call wrapper,
`torch_tpu/_internal/pallas/pallas.py:638` — identically, for 51.8% of the step.
`src` is the innermost frame that is *not* dispatch plumbing
(`opstats.PLUMBING`: torch, dynamo, the Pallas wrapper, the torch_tpu
export/compile driver). That one 51.8% row resolves into:

| caller frame | % step | what is under it |
|---|---|---|
| `fused_moe.py:239` | 37.8% | gather_reduce, sort, gmm — and their glue |
| `parameter.py:127` | 11.2% | the linear-attention kernels and their glue |
| `attention.py:666` | 3.2% | RPA and its glue |
| `tpu_compiler.py:396` | 3.0% | all-gather |

What is under a frame is how you check it: the glue booked to
`parameter.py:127` sits beside `fused_conv1d_gdn_*` and nothing else, which is
why that frame reads as the linear-attention path.

On this stack, eight `src` rules moved 14.44% of the step out of `elementwise` /
`fusion` / `layout` and into rows that name the work — the three deltas below,
added up:

| category | name rules only | with `src` rules |
|---|---|---|
| `layout` | 8.49% | 3.89% |
| `elementwise` | 5.70% | 0.49% |
| `fusion` | 4.63% | 0.00% |
| `quant` | 0.00% | **6.11%** |
| `moe_glue` | — | **3.63%** |
| `linear_attn_glue` | — | **2.66%** |
| `norm` | 0.00% | **0.51%** |
| `moe_route` | — | **0.68%** |
| `attn_glue` | — | **0.46%** |
| `attn_rope` | — | **0.14%** |

`norm` reading 0.00% was not a model with no norms; it was every fused norm
arriving as an anonymous `fusion`. The `*_glue` rows are the reshapes and slices
a kernel wrapper emits around its call — previously indistinguishable from any
other `reshape`, now booked against the kernel that caused them.

**Order matters more than usual.** A `src` rule is broad — it claims a whole
file — so it must sit *below* the rules that name specific work in that file.
`moe_routing.py` contains the topk, and a `src` rule placed above `moe_sort`
would swallow the single largest row in the report. See the placement comments
in `taxonomies/qwen35_397b.yaml`.

Two limits to know before leaning on it:

- **Coverage is ~74% of op time, and a collective's coverage is worse than
  none.** The other 26% is compiler-generated: 22.3% of the step is SPMD
  `reduce-scatter` and 3.9% is the `copy` traffic around it. The table prints
  them split by category — one anonymous "26% unexplained" row reads as a hole
  in the data when it is not one. Worse than none, because a collective
  sometimes *does* carry a source line and it is not the one you want: on this
  capture `reduce-scatter` reports no caller while `all-gather` reports
  `tpu_compiler.py:396`. Both stacks are the *compilation* stack — a collective
  is functionalised into the graph at trace time — so they name the compiler,
  not the model, and every collective in a module carries the same one. Do not
  write a `src` rule that expects to separate two of them.
- **A trampoline can still swallow the owner.** `parameter.py:127` is vLLM's
  `BasevLLMParameter.__torch_function__`; it is the first non-plumbing frame for
  11.2% of the step, and the frame anyone actually wanted is gone. It reads as
  the linear-attention path because of what sits under it, not because the frame
  says so — which is why that rule carries an explicit `block:`. Widening
  `PLUMBING` does not help — everything below it is plumbing too.

Where this comes from: not the xplane, which really does carry no source
information. The profiler writes `rank<N>_<host>.op_stats_v2.pb` beside each
xplane, and every op in it has an `OpMetrics.source_info`. The join to the
xplane is by op name and is exact. Parsing costs ~0.1 s, and a capture without
a sidecar loses the source column and nothing else. See `xprofkit/opstats.py`.

### Adding a command

`analyze.py` is a thin argparse shell. To add one:

1. Write a module in `xprofkit/` exposing `analyse(capture, ...) -> dict` and
   `render(result) -> list[str]`. Take a `Capture`, return plain dicts.
2. Add a `cmd_yours(args)` in `analyze.py` that calls `_load()`, then `_emit()`.
3. Register it in `build_parser()`, calling `common(p)` so it inherits
   `--ranks` / `--phase` / `--json` / `--no-cache`.

Keeping `analyse` and `render` separate is what makes `--json` free — and the
rule is that *everything on the page* comes out of `analyse`. The block split
broke it once: `analyse_blocks` was called from inside `render`, so the headline
row (`moe`, 69.41% of the step) existed only as fixed-width text and a
regression check on block attribution had to scrape it. `breakdown --json` now
carries `blocks`, `unassigned`, `unassigned_us`, `dead_rules`, and an `n` per
cell — the last is in the payload but deliberately not in any rendered table,
where a column of instance counts reads as structure.

### Using it as a library

```python
from xprofkit import load_capture, Taxonomy, breakdown, compare

cap = load_capture("prof/baseline_a", ranks={"0", "2", "4", "6"})
print(cap.phases["prefill_heavy"].mean_over_ranks("mean_us"))

tax = Taxonomy.load()
result = breakdown.analyse(cap, tax)
print(result["prefill_heavy"]["categories"]["moe"])

# Block subtotals are a fold of that result plus the taxonomy, not a second
# pass over the trace, so they cost nothing and can be recomputed per taxonomy.
print(breakdown.analyse_blocks(result["prefill_heavy"], tax)["blocks"])

verdict = compare.compare_arms({"base": [cap], "cand": [other]}, "mean")
```

Everything is dataclasses and dicts; nothing holds a file handle or a chip.
`xprofkit/trace.py` is the only module that knows about protobufs.

---

## Files

| Path | What it is |
| --- | --- |
| `analyze.py` | the entrypoint — argparse and output only |
| `xprofkit/trace.py` | the only protobuf reader: xplane → `Capture`, plus the JSON cache |
| `xprofkit/taxonomy.py` | op → category and execution unit; the declared blocks |
| `xprofkit/opstats.py` | source lines per op, from the profiler's op-stats sidecar |
| `xprofkit/steptime.py` | per-capture distributions and capture-to-capture deltas |
| `xprofkit/compare.py` | arm-level Welch statistics and the verdict |
| `xprofkit/breakdown.py` | one capture, split by category, execution unit, family, source line and block |
| `xprofkit/diff.py` | two captures, split by category, family and exact op |
| `xprofkit/table.py` | text tables, `describe`, `welch`, the t table |
| `taxonomies/default.yaml` | generic classification rules; copy and extend |
| `taxonomies/qwen35_397b.yaml` | worked model-specific example: shape rules, `src` rules, three blocks |
| `tests/test_xprofkit.py` | hermetic unit tests — no jax, no traces, no chip |

## Cache

Each capture grows a `.xprofkit/` directory of per-rank JSON, keyed on three
things:

* the size and mtime of the xplane;
* the same for its op-stats sidecar — a capture first parsed before the sidecar
  was written cached an empty source column, and keying on the xplane alone
  would serve that forever;
* a **derivation fingerprint**: `op_hlo` and `op_src` are not copied out of the
  trace, they are derived, and the inputs to that derivation are meant to be
  edited. The fingerprint hashes the `PLUMBING` and `_TRIM` lists, the majority
  threshold, the signature field list, and whether the `xprof` schema was
  importable at all. Extend `PLUMBING` for a new stack and every warm capture
  re-derives; without this it kept serving the old callers silently.

`CACHE_VERSION` covers the deriving code itself, which no fingerprint can see.
Deleting the cache is always safe;
`--no-cache` skips it. Writing is best effort, so a read-only capture directory
works, just slowly.

Measured on the captures in this README, four ranks each (`--ranks 0,2,4,6`), so
"a capture" below is four xplanes and four sidecars. A single `breakdown` costs
~9 s cold and ~0.3 s warm. A two-arm `compare` over six captures costs ~43 s
cold — the parse dominates and scales with rank-files, not with the command —
and ~0.3 s warm, because warm it reads 24 small JSON files and does arithmetic.

## Tests

```bash
python3 tests/test_xprofkit.py        # or: python3 -m pytest tests
```

No jax, no traces, no chip. They cover the parts that decide what a number
*means* — classification (names, HLO signatures and execution unit), taxonomy
validation, block declaration and the disclosure the block table prints, dump
selection, rank pinning, cache invalidation, every command's exit status, the t
table and Welch — including regression tests for the rank-churn bug in
correction 1 and for the family table classifying from the op name alone, which
made it contradict the category table beside it.

## Limitations

Each of these is a way this tool's output can be read as saying more than it
does. They are here rather than in a comment because the person who needs them
is the one quoting a number, not the one editing the code.

- **Block assignment is declared, not evidenced.** The tool prints the split a
  human wrote in the taxonomy. It is right for the sharding it was written
  against and has no way to notice a different one. See "Declaring blocks".
- **Two collectives cannot be told apart.** Same name, same shapes, same
  `replica_groups`, byte-identical compilation stacks. The `replica_groups`
  field is in the HLO signature and is useful for other things; it is *not* a
  way to split collectives by which part of the model issued them, and no
  taxonomy rule can be. The instruction pair this rests on was observed on a
  tensor-parallel configuration that is **not among the captures here** — every
  capture in this README is attn dp8 · ep8 and holds 0 all-reduce. Take it as
  reported, not as reproducible from this repository.
- **Adjacency is the one signal nobody has tried.** The two indistinguishable
  all-reduces differ in exactly one respect: their instruction IDs (`.34` and
  `.36`), i.e. their position in the module. Position implies neighbours, and a
  collective's neighbours *are* attributable. Nothing here uses that — it would
  be a new kind of inference, and this round deliberately removed one. Recorded
  as unexplored, not as recommended.
- **Where a source line comes from is a choice, and it is not tunable.**
  `PLUMBING` and `_TRIM` are module-level constants in `opstats.py`, not
  taxonomy fields. Making them per-taxonomy was considered and deferred: the
  frame-selection policy is a property of the *stack* (torch, dynamo, Pallas),
  not of one model's categories, and the cache already keys on
  `opstats.source_fingerprint()` — the fingerprint of that policy — so a
  per-taxonomy policy would need the cache re-keyed on the taxonomy, which is
  the wrong key for a derivation the taxonomy does not participate in.
- **Ranks can disagree about a derived field.** The signature and source line
  are per-op, and two ranks occasionally disagree; the tool keeps the op's time,
  drops the disputed field, and says so under the table. On `baseline_a` that is
  6 ops: 12.4 µs of `hlo` and 0.0 µs of `src`, i.e. 0.004% of the step. A rule
  needing the dropped field cannot fire for those ops; `-v` names them.
