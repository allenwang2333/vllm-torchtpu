# Option E — cost-aware partition selection for `sparsecore_topk`

**Status: correct, measurably faster in isolation, and almost entirely out of reach in
production.** Adopt it as a small correctness-neutral cleanup if you like; do not expect an
end-to-end number from it.

Everything below is a hardware measurement on tpu7x unless it is labelled `modelled`.
Nothing in `src/` was modified. The incumbent is
`src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py` at commit `c3a18c95`
(`_pick_partition`, `sparsecore_topk`), reached unedited from this folder.

| | |
| --- | --- |
| Jobs | `j-f19bcd1c-4569-4bce-8bcd` (correctness), `j-b1a8fb7e-5fc8-48f2-8387` (timing) |
| Both | `JobStatus: Succeeded`, 4/4 fan-out shards `rc=0` |
| Image | `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b` (prebuilt; not rebuilt) |
| Recipe | `simple-run-command`, `TPU_TOPOLOGY=2x2x1`, 4 chips, `tpu-fanout.sh` |
| `~/cloud-devkit` | `c4e1d4e`, one untracked unrelated recipe |
| Raw data | `data/option-e-*.json{,.gz}` — every table here recomputes from these |

`RUN_CMD` as read back from `cdk job recipe`:

```
j-f19bcd1c: uv pip install --system -q pytest pytest-mock pytest-timeout parameterized &&
  cd /root/cloud-devkit/vllm-torchtpu && bash /root/fanout/tpu-fanout.sh
  "bash /root/kw/variants/option-e-partition/shard.sh correctness"
  "bash /root/kw/variants/option-e-partition/shard.sh pytest-optione"
  "bash /root/kw/variants/option-e-partition/shard.sh pytest-control"
  "bash /root/kw/variants/option-e-partition/shard.sh ties"

j-b1a8fb7e: cd /root/cloud-devkit/vllm-torchtpu && bash /root/fanout/tpu-fanout.sh
  "bash /root/kw/variants/option-e-partition/shard.sh timing_prod"
  "bash /root/kw/variants/option-e-partition/shard.sh timing_k"
  "bash /root/kw/variants/option-e-partition/shard.sh timing_n"
  "bash /root/kw/variants/option-e-partition/shard.sh glue"
```

---

## 1. What `_pick_partition` is actually for

Read off `sparsecore_topk.py:360-377`, the function does **two unrelated jobs in one loop, and
only the first is a constraint.**

```python
def _pick_partition(b: int, n: int, k: int) -> int:
    p_min = _cdiv(n, MAX_SLICE_WORDS)          # (1) capacity  -- mandatory
    p = 1
    while p < p_min:
        p *= 2
    max_p = 32 // b if b < 32 else 1            # (2) occupancy -- discretionary
    while p * 2 <= max_p:
        next_np = n // (p * 2)
        if next_np >= k and next_np >= LANES and (n % (p * 2)) == 0:
            p *= 2
        else:
            break
    return p
```

**(1) Capacity, `p_min` — mandatory.** `_sc_topk_direct` allocates a resident VMEM scratch of
`slice_len = _align_to(n, LANES)` int32 words per subcore (`:324`, `:347`) and DMAs the whole
slice into it (`:160`). SparseCore tile memory is 512 KiB
(`vmem_capacity_bytes=524288`, read back from the device in every probe run) and the module
reserves at most `MAX_SLICE_WORDS = 64*1024` words — 256 KiB — for that slice. So
`p >= ceil(n / 65536)` rounded up to a power of two, or the kernel cannot be built. Not
negotiable.

**(2) Occupancy, `max_p = 32 // b` — discretionary, and this is the half that costs.** The mesh
is 2 cores × 16 subcores = 32 subcores, and one *wave* processes 32 rows
(`:148, num_waves = _cdiv(b, 32)`). When `b < 32` most subcores idle in the single wave, so
splitting each row into `p` column slices manufactures `b*p <= 32` rows and fills them. The
docstring's "maximize subcore utilization" is exactly this clause. It has **no correctness
content**: the loop body is entered only once `p_min` is already satisfied, and every `p` it can
return produces a legal kernel.

**Exactness is a property of the wrapper, not of the choice of `p`.** Stage 1 returns each
slice's local top-`k_p` with `k_p = min(k, n_p)`; the global top-`k` can draw at most
`min(k, n_p)` entries from any one slice, so the `p * k_p` candidates handed to stage 2 are a
*superset* of the answer, for every legal `p`. Lowering `p` — the only thing Option E ever does
— cannot break that. `p = 1` is the degenerate case where the candidate set *is* the input and
the two-stage machinery disappears entirely.

The occupancy clause therefore buys parallelism the shape did not need and pays for it three
times: a second `_sc_topk_direct` launch, a stage-2 scan over `p*k_p` candidates, and two
`jnp.take_along_axis` gathers of `b × p*k_p` on the critical path. At `b=8, n=10240, k=2048` the
incumbent's `p=4` makes stage 2 rescan `4*2048 = 8192` candidates — as much work as the
`eff_len = 8704` row it was meant to be splitting up.

---

## 2. Is `p=1` numerically valid at b=8? — **Yes.** (the gate)

Job **`j-f19bcd1c-4569-4bce-8bcd`**, shard `correctness`, `rc=0`. Verbatim:

```
[probe] correctness: b=8 k=2048 inc_p=4 newE_p=1 chain=[1, 2, 4] -> OK
[probe] correctness: b=8 k=512  inc_p=4 newE_p=1 chain=[1, 2, 4] -> OK
[probe] correctness: b=8 k=64   inc_p=4 newE_p=1 chain=[1, 2, 4] -> OK
[probe] SUMMARY: cases=426 index_set_mismatches=0 score_bag_mismatches=0
```

Scope of that summary line: `b ∈ {1,2,3,4,5,8,9,12,16,17,32,64}` × `k ∈ {2048,512,64}` ×
6 `row_lengths` patterns (`full`, `prod_decode`, `ragged`, `short_lt_k`, `some_zero`,
`prefill_ramp`) × 2 seeds = **426 cases**, and within each case **every** feasible `p` in the
chain is run and compared, not just the one the rule picks — **1100 kernel arms**. 318 of the
426 cases are ones where Option E changes `p`. Every arm was compared against an exact numpy
top-`k` oracle on **index sets**, and separately on score multisets, emitted count, and
duplicate indices.

Recomputed from `data/option-e-correctness.json.gz`:

| | |
| --- | --- |
| Option E vs oracle, index-set mismatches | **0 / 426** |
| Option E vs **incumbent**, index-set mismatches | **0 / 426** |
| Incumbent vs oracle, index-set mismatches | 0 / 426 |
| Arms with a duplicate index | 0 / 1100 |
| Arms with a wrong emitted count | 0 / 1100 |

### The tie case, run deliberately

Scores were re-drawn quantised to 7 distinct levels — manufacturing massive ties, the one place
an exact selector is licensed to differ — and the whole matrix re-run (shard `ties`, `rc=0`):

```
[probe] SUMMARY: ties cases=426 index_set_mismatches=43 score_bag_mismatches=0
```

43 mismatches looks alarming and is not. Recomputed from `data/option-e-ties.json.gz`:

| | |
| --- | --- |
| Option E vs **incumbent**, index-set mismatches | **0 / 426** |
| Option E vs oracle | 43 / 426 |
| **Incumbent** vs oracle | **43 / 426** (the same 43) |
| Cases where only one of the two mismatches | **0** |
| Score-multiset mismatches, either side | **0 / 426** |

The 43 are a tie-breaking disagreement with numpy's `argsort` order that **the incumbent already
has**, at rows where `chain=[1]` and Option E changes nothing at all (`b=17,32,64`). Option E
reproduces the incumbent's choice bit-for-bit. The returned *scores* are identical multisets in
all 426 tied cases, so the selection is still an exact top-`k`; only which of several equal-valued
indices is named differs, and it differs identically.

**Verdict on the gate: `p=1` at `b=8` is exact. Option E is output-identical to the incumbent
across all 852 cases (426 tie-free + 426 tied).**

### The repo's own tests, unmodified

`tests/kernels/test_dsv4_streamindex_topk.py` was run **byte-identical** to what is committed,
under a pytest plugin (`option_e_patch.py`) that rebinds `sparsecore_topk` — both the module
attribute and the `from ... import` alias bound at `streamindex_topk.py:25`, which is the one the
kernel under test actually calls. The plugin asserts the rebind took and prints it:

```
[option-e] patched streamindex_topk.sparsecore_topk: <PjitFunction of <function sparsecore_topk ...>> -> <function sparsecore_topk_v2 ...>
[option-e] pick_partition(b=8,  n=10240, k=2048) = 1 (incumbent 4)
[option-e] pick_partition(b=16, n=10240, k=2048) = 1 (incumbent 2)
[option-e] pick_partition(b=64, n=10240, k=2048) = 1 (incumbent 1)
======================= 20 passed, 14 warnings in 44.27s =======================   <- Option E
======================= 20 passed, 14 warnings in 46.51s =======================   <- control (OPTION_E=0)
```

A control shard with the patch disabled was run on a neighbouring chip so a no-op patch could not
pass as a result. **Note:** the brief quoted `13 passed in 21.41s` for the incumbent. At
`c3a18c95` this file collects **20** items; both my shards report 20. The 13 is stale, not a
discrepancy in the change.

---

## 3. The measured crossover, and the rule

### Timing method

Chain-slope, as established in `probes/stage-b-ksweep.md`. `R ∈ {1,2,3,4}` kernel calls are
chained inside **one** `jax.jit` with a real data dependency (`rl + jnp.minimum(out[0,0], 0)` —
`x*0` gets constant-folded away), the whole thing is timed best-of-9 after 3 warmups, and the
**slope** of wall-clock against `R` is the per-call device time. The intercept is the host term
and is discarded. This is the only sound method here: the launch enqueues asynchronously and
blocks once, so single-call wall clock is `max(device, host)` — **a dispatch floor is a clamp,
not an addend, and subtracting it is invalid.** All 60+ chain fits in this run have
`r² ≥ 0.9997`; most are `1.0000`.

**Contention control.** The four shards ran concurrently on four chips of one pod, and other
agents were using the cluster. Every timing shard re-measured two anchors from the unpinned
k-sweep:

| anchor | reference | shard 0 | shard 1 | shard 2 | shard 3 |
| --- | --- | --- | --- | --- | --- |
| b=8, p=4 | 50.09 µs | 51.01 (+1.8%) | 50.99 (+1.8%) | 51.16 (+2.1%) | 50.86 (+1.5%) |
| b=64, p=1 | 29.00 µs | 29.07 (+0.3%) | 29.07 (+0.2%) | 28.92 (−0.3%) | 28.99 (−0.0%) |

Within 2%. The fan-out is comparable in absolute µs to the unpinned baseline.

### There is no crossover at the production width

At `n = 10240`, `p = 1` is the measured optimum at **every** `b` and **every** `k` swept. Device
µs, job `j-b1a8fb7e`:

| b, k=2048 | p=1 | p=2 | p=4 | p=8 | p=16 | p=32 | incumbent | Option E |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **15.21** | 37.76 | 42.32 | 43.95 | 43.04 | 43.15 | p=4 (42.32) | p=1 |
| 2 | **15.70** | 37.94 | 46.63 | 43.95 | 43.05 | — | p=4 (46.63) | p=1 |
| 4 | **16.90** | 40.36 | 49.69 | 47.41 | — | — | p=4 (49.69) | p=1 |
| 8 | **17.04** | 41.78 | 50.08 | — | — | — | p=4 (50.08) | p=1 |
| 16 | **17.33** | 45.64 | — | — | — | — | p=2 (45.64) | p=1 |
| 32 | **17.89** | — | — | — | — | — | p=1 | p=1 |
| 64 | **29.01** | — | — | — | — | — | p=1 | p=1 |

Two-stage never pays here, at any `k`: ~12–25 µs of XLA glue plus a stage-2 rescan of `p*k_p`
candidates swamps the stage-1 saving. `b=8, n=10240` across `k` (p=1 / p=2 / p=4, µs):

| k | 64 | 128 | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- | --- | --- |
| p=1 | **16.82** | **17.44** | **16.80** | **16.95** | **17.07** | **16.98** |
| p=2 | 33.59 | 37.96 | 35.40 | 35.75 | 38.37 | 42.21 |
| p=4 | 31.53 | 37.22 | 34.02 | 35.64 | 39.70 | 50.28 |

The `p=1` column is flat in `k` — at `n=10240` the scan is length-bounded, not `k`-bounded, so
the whole cost of `p>1` is overhead.

The glue was measured directly by decomposing whole-call time into its two stages
(shard `glue`): **11.64 – 25.34 µs, mean ≈ 17.7 µs**, e.g.

```
[probe] glue: b=8  p=4  k=2048: whole=50.09 s1=12.60 s2=16.56 -> glue=20.92us (gather elems 65536)
[probe] glue: b=8  p=4  k=512:  whole=35.31 s1=12.42 s2=11.25 -> glue=11.64us (gather elems 16384)
[probe] glue: b=2  p=16 k=512:  whole=52.76 s1=10.61 s2=16.81 -> glue=25.34us (gather elems 16384)
```

### Two-stage *does* pay at larger `n` — and there the incumbent over-partitions

| shape | incumbent p | its time | measured argmin | its time | speedup |
| --- | --- | --- | --- | --- | --- |
| b=1, n=65536, k=2048 | 32 | 89.34 µs | **4** | **53.16 µs** | 1.68× |
| b=1, n=65536, k=128 | 32 | 30.82 µs | **16** | **30.27 µs** | 1.02× |
| b=1, n=262144, k=2048 | 32 | 116.35 µs | **8** | **78.08 µs** | 1.49× |
| b=16, n=262144, k=2048 | 4 | 167.84 µs | **4** | 167.84 µs | 1.00× |

So the incumbent's failure mode is not "partitions when it shouldn't" — it is "always walks the
chain to the end". Sometimes the end is right; usually it is not.

### The rule

Keep the incumbent's feasible set **exactly**, and add one predicate: raise `p` above `p_min`
only when the *modelled* cost actually falls. Full source in `partition_v2.py`.

```python
def pick_partition(b, n, k):
    chain = feasible_partitions(b, n, k)         # the incumbent's own chain, p_min first
    return min(chain, key=lambda p: (modelled_us(b, n, k, p), p))
```

`modelled_us` is the measured stage-B cost model from `probes/stage-b-ksweep.md` §3
(`8.152 + waves*(0.702 + 0.05424e-3*n + 1.01975e-3*eff_len)`, r²=0.9982 over 35 points) applied
once per stage, plus the glue term above. **`modelled_us` is modelled, never a reported timing** —
it exists only to rank candidate `p` against each other at trace time, and it is deliberately
charged at the *low* end of the measured glue so it under-charges rather than rejects the
two-stage path.

**Invariant, and the reason correctness is cheap here:
`p_min <= pick_partition(...) <= _pick_partition(...)`.** The chain is the incumbent's; the rule
only ever stops walking it earlier. No shape gains a partition it did not already have, and no
shape loses the capacity partition it needs.

`eff_len` is a runtime value and the partition is a trace-time decision, so the model is
evaluated at `eff_len = n` — the assumption *most* favourable to partitioning. Where it declines
to partition it would decline at every shorter row too.

**Rule accuracy: on all 21 configurations where the complete feasible chain was swept, Option E
selected the measured argmin. 21/21.** (Three further points appear in `data/` at a single `p`
only, from the glue arm; they are not sweeps and are excluded.)

Known model caveat: it over-predicts `p=1` at `n=65536` (79.24 modelled vs 63.61 measured, +25%)
because `eff_len=65536` is outside the range the model was fitted on (≤10240). It still ranked
correctly, but the margin there is smaller than it looks.

---

## 4. How much of the served workload this touches — **one token bucket, honestly**

The batch dimension `b` is not the number of sequences. It is the **padded token bucket**, and
the buckets are generated by `_get_exponential_token_paddings` with a hard
`min_token_size=16` (`src/vllm_torchtpu/platforms/tpu_platform.py:1236-1276, 905-920`):

```python
compilation_config.compile_sizes = _get_token_paddings(
    min_token_size=16,
    max_token_size=scheduler_config.max_num_batched_tokens,
    extra_bucket_sizes=envs.TPU_TOKEN_BUCKET_EXTRA,
)
```

Neither the GLM-5.2 recipe nor any config in the repo sets `TPU_TOKEN_BUCKET_EXTRA` or
`compile_sizes`. With the nightly GLM-5.2 config
(`--max-num-batched-tokens=1024 --max-num-seqs=64 --max-model-len=9216 --data-parallel-size=16`,
`recipes/experimental/multihost/torchtpu/vllm-torchtpu-glm-5.2-v7x-16/vllm-torchtpu-glm-5.2-benchmark.yml`)
the reachable buckets are exactly **{16, 32, 64, 128, 256, 512, 1024}**.

**`b = 8` is not reachable.** The `prod_decode_b8_ctx8704` row in
`instruments/bench_streamidx_topk.py` — and therefore the headline 2.91× in
`probes/stage-b-ksweep.md` §5 — is a synthetic shape. The 2.91× is real as a measurement and
unreachable as a workload.

Where the rule actually differs from the incumbent, over every reachable bucket:

| n \ b | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10240 | **2→1** | – | – | – | – | – | – |
| 16384 | **2→1** | – | – | – | – | – | – |
| 32768 | **2→1** | – | – | – | – | – | – |
| 65536 | – | – | – | – | – | – | – |
| 131072 | – | – | – | – | – | – | – |
| 262144 | – | – | – | – | – | – | – |

(`k=2048`; `–` = Option E returns the incumbent's `p`.) At `b ≥ 32` the incumbent already
returns `p=1` because `max_p = 32//b <= 1`, so there is nothing to improve. At `n >= 65536` the
capacity floor `p_min` has risen to meet `max_p` and the chain collapses to one element — for
example at `n=262144, b=16`, `p_min=4` but `max_p=2`, so `chain=[4]` and Option E is a strict
no-op. **The long-context deployments are exactly the ones Option E cannot help.**

So: **one bucket out of seven, `b=16`.** Measured effect there, `n=10240`:

| k | 64 | 128 | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- | --- | --- |
| incumbent (p=2) | 38.27 | 34.92 | 36.72 | 37.84 | 40.69 | 45.76 |
| Option E (p=1) | 17.02 | 17.37 | 18.23 | 17.07 | 17.19 | 17.41 |
| speedup | 2.25× | 2.01× | 2.01× | 2.22× | 2.37× | **2.63×** |

(All six columns from the `timing_k` shard, so they are mutually comparable; the `timing_prod`
shard independently gives 45.64 / 17.33 at `k=2048`, a 0.3% shard-to-shard spread.)

At the production `k=2048` that is **28.3 µs saved per call**, per layer, per step.

**And bucket 16 is rarer than one bucket in seven suggests.** Under `--data-parallel-size=16`
the bucket is chosen from an all-reduce **MAX** across all 16 engines
(`src/vllm_torchtpu/runner/tpu_runner.py:3378-3392`):

```python
torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX,
                             group=get_dp_group().cpu_group)
bucket = _get_padded_token_len(self.num_tokens_paddings, int(t[0].item()))
```

So the step runs at bucket 16 only if **every** engine has ≤16 scheduled tokens simultaneously.
The nightly benchmark runs `--max-concurrency 1024` over 16 engines = 64 requests per engine
= `max_num_seqs`, i.e. steady-state decode sits at **bucket 64**, where Option E does nothing.
Bucket 16 is the ramp-up and ramp-down tail, and genuinely low-load serving.

Two things I did **not** measure and am not going to assert: how many `streamidx_topk` calls
there are per step (one per MLA layer, so the 28.3 µs carries a multiplier I did not count), and
what fraction of wall-clock steps in a real trace land on bucket 16.

---

## 5. Correctness evidence

Job **`j-f19bcd1c-4569-4bce-8bcd`**, `JobStatus: Succeeded`, all four shards `rc=0`.

| shard | what ran | verbatim result |
| --- | --- | --- |
| `correctness` | 426 cases × full feasible `p` chain = 1100 arms vs exact numpy oracle, tie-free scores | `[probe] SUMMARY: cases=426 index_set_mismatches=0 score_bag_mismatches=0` |
| `ties` | the same matrix on scores quantised to 7 levels | `[probe] SUMMARY: ties cases=426 index_set_mismatches=43 score_bag_mismatches=0` — all 43 are shared with the incumbent; Option-E-vs-incumbent mismatches = **0** |
| `pytest-optione` | `tests/kernels/test_dsv4_streamindex_topk.py`, byte-identical, with Option E patched in | `20 passed, 14 warnings in 44.27s` |
| `pytest-control` | same file, `OPTION_E=0`, incumbent in place | `20 passed, 14 warnings in 46.51s` |

Exact index-set equality against the incumbent, at every `b` whose behaviour changes. This meets
the bar the brief set.

---

## 6. Verdict

**Correct: yes, established on hardware, not argued.** **Faster: yes, 2.0–2.6× at `b=16`,
1.5–1.7× at long `n` with small `b`, measured by chain-slope with anchors within 2%.**
**Worth adopting end to end: probably not on its own.**

The rule is strictly safer than the incumbent (it can only lower `p` within the incumbent's own
feasible chain), it is output-identical on 852 cases, and it picked the measured optimum 21/21.
As a patch it is ~40 lines and deletes a heuristic that is wrong more often than it is right.

**The strongest argument against adopting it:** the win is almost entirely outside the served
workload. `b=8` — the shape that produced the 2.91× headline — is not a reachable token bucket;
`min_token_size=16` is hardcoded. Of the seven reachable buckets, Option E changes exactly one,
and under DP=16 that one requires all sixteen engines to be near-idle at once, which is not the
steady state the nightly benchmark measures. Meanwhile the long-context configurations, where
two-stage genuinely pays and the incumbent genuinely over-partitions, are precisely where
`p_min` has collapsed the chain to a single element and Option E is a no-op. A change that is
2.63× faster on a bucket the server rarely visits will not show up in an end-to-end number, and
spending a 2.5–4 hour benchmark to discover that would be the expensive way to learn it.

A secondary objection worth stating: `pick_partition` now depends on four fitted constants and a
glue term. They were measured, they are cited, and the rule was right 21/21 — but they are a
tpu7x-and-this-kernel fit. A future SparseCore, or a change to `_sc_topk_direct`, silently
invalidates the ranking. The invariant limits the blast radius (worst case it declines a
partition that would have helped; it can never pick an infeasible one), but a constant-free
rule — e.g. "take `p_min`, and go above it only when `p*k_p < n`" — would be more durable and
gets most of the same answer at `n=10240`.

**Recommendation.** Land it as a correctness-neutral cleanup with the numbers above attached,
or fold it into a larger stage-B change; do not fund an end-to-end benchmark for it alone. The
more valuable output of this exercise is §4: the token buckets are `{16,32,...}` and `b<16` is
unreachable, which retires any future optimisation aimed at small `b` before it is built.

---

## 7. Reproducing

```bash
# on the pod, with this folder mapped to /root/kw/variants/option-e-partition
bash shard.sh correctness      # the gate
bash shard.sh ties
bash shard.sh pytest-optione   # repo tests, unmodified, Option E patched in
bash shard.sh pytest-control   # same, incumbent
bash shard.sh timing_prod      # b x p sweep at n=10240
bash shard.sh timing_k         # k sweep
bash shard.sh timing_n         # n sweep, 65536 and 262144
bash shard.sh glue             # two-stage overhead decomposition
```

| file | what it is |
| --- | --- |
| `partition_v2.py` | the rule, plus a copy of the two-stage wrapper taking `p` as a parameter. Imports the unmodified `_sc_topk_direct` from `src/`. |
| `option_e_patch.py` | pytest plugin; rebinds both `sparsecore_topk` names, asserts the rebind took, honours `OPTION_E=0` for the control. |
| `probe_option_e.py` | the six arms. |
| `shard.sh` | fan-out entry point; keeps nested quoting out of `RUN_CMD`. |
| `data/` | raw JSON from both jobs. Every table above recomputes from these. |
