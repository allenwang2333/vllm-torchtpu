# Option B through the real caller — the runner-side change, built and measured

Option B (split the indexer's work into `G` token chunks so chunk *i*'s SparseCore top-k overlaps
chunk *i+1*'s TensorCore score kernel) had only ever been measured by a harness that *presented*
one 1024-token sequence as `G` pseudo-sequences. `variants/option-b-pipeline/RESULTS.md` §7 named
two things that had to be solved before that presentation could be produced for real. This is the
build and the measurement of the real thing: the production caller
(`mla_attention_op.py::_streamidx_topk_jax`) driving the merged kernel module, chunked.

**Headline.** It works, it is index-identical at every `G`, and at the production prefill shape
it takes **−10.59 %** off the whole caller's TensorCore time at `G=4` relative to the same caller
running the merged module unchunked (`1915.0 → 1712.2 µs`), or **−13.36 %** relative to today's
incumbent (`1976.4 µs`). Excluding the KV-cache scatter — which is 52 % of the caller and which
Option B cannot touch — the same change is **−22.08 %** / **−26.93 %**. `G=8` buys a further
0.3 pp and is not worth its compile time; **`G=4` is the knee**.

**And §7's second integration problem does not exist.** The chunk plan does *not* have to be
handed in as static metadata. Only `G` has to be static, and `G` is a pure function of two
*shapes* — `q.shape[0]` and `seq_lens.shape[0]` — both of which are already static under
`jax.jit`. Everything else is masked arithmetic on the traced `seq_lens` / `cu_q_lens` /
`distribution` the caller already has. §4 gives the formula and the evidence.

**Read the `vs incumbent` figures with §9.1 in hand.** The `bG1` baseline is the merged module, and
two of the three changes in it (D1 and the `bkv_p` ladder) were withdrawn by a concurrent effort
while this was being measured. That does not touch the Option B deltas — both arms share the same
module, so it cancels — but it does mean the compound `vs inc` column describes a configuration
nobody currently proposes to ship.

**Nothing here is an end-to-end benchmark.** Every number is device self time from `hlo_stats` on
a 4-chip `tpu7x` pod, for one call of the indexer op, at hand-built shapes. No model was run.

---

## 1. Jobs, and the md5 read-back every number depends on

The image ships a *different* `streamindex_topk.py` and **no** `sparsecore_topk.py`. A job without
`--map-dir` of the checkout's `src/` imports the image's stale module, exits 0, and benchmarks the
wrong code. Every arm reads back both md5s and refuses to produce numbers on a mismatch
(`probe_runner.py` exits 3, `bench_runner_trace.py` returns 3).

| job | arms | `JobStatus` | md5 read-back |
| --- | --- | --- | --- |
| `j-46ad1b70-e536-4e25-902a` | `plan`, `correct` | `Succeeded` | `MD5_READBACK_OK=True` |
| `j-8185ac37-8409-4177-a1fa` | `hostcost`, `trace` (no `donate_argnums`) | `Succeeded` | `MD5_READBACK_OK=True` |
| `j-b86c8fce-fd3d-4b63-9a24` | `hostcost`, `trace` (**donated `cache_kv`**, as production) | `Succeeded` | `MD5_READBACK_OK=True` |

Read back on-pod in all three:

```
MODULE .../kernels/deepseek_v4/streamindex_topk.py md5=1e221f10cbf3573aa0d2ae78c3463532 expect=1e221f10cbf3573aa0d2ae78c3463532 OK
MODULE .../kernels/deepseek_v4/sparsecore_topk.py  md5=8bc9e25b8701a53a97704564b9fbc658 expect=8bc9e25b8701a53a97704564b9fbc658 OK
VARIANT merged        .../variants/merged/streamindex_topk_merged.py    md5=5bc0578427412d56209ef8b5d764fcae
VARIANT b             .../variants/option-b-runner/streamindex_topk_b.py md5=7cc997ac1aaf6f6911264289afc2bf83
VARIANT runner_op     .../variants/option-b-runner/runner_op.py         md5=462fb9a9051c0a14dd58eaa88b6962f6
VARIANT runner_common .../variants/option-b-runner/runner_common.py     md5=e097c09d1f7afa74980b1dea99382a1f
```

`runner_common.py` was edited between job 2 and job 3 (to add `donate_argnums`); its md5 in job 3's
log is the post-edit one. Both jobs are reported, and §7 shows they agree to ≤0.23 % on every
prefill arm.

`RUN_CMD` for job 3, as read back with `cdk job recipe j-b86c8fce-fd3d-4b63-9a24`:

```
uv venv --system-site-packages /tmp/j11 && uv pip install --python /tmp/j11/bin/python -q
jax==0.11.0 jaxlib==0.11.0 && /tmp/j11/bin/python -c "import jax, jaxlib; print(\"PROFVENV\",
jax.__version__, jaxlib.__version__)" && cd /root/cloud-devkit/vllm-torchtpu && bash
/root/kw/variants/option-b-runner/runner_shard.sh hostcost; rc1=$?; bash
/root/kw/variants/option-b-runner/runner_shard.sh trace; rc2=$?; echo "HOST_RC=$rc1
TRACE_RC=$rc2"; exit $((rc1+rc2))
```

with `--map-dir <checkout>/src:tpu-job:/root/cloud-devkit/vllm-torchtpu/src` and
`--map-dir <checkout>/kernel_work/streamidx-topk:tpu-job:/root/kw`. The recipe read-back confirms
both `cdk-map-dir` unpack steps. `PROFVENV 0.11.0 0.11.0` and `HOST_RC=0 TRACE_RC=0` are in the log.

---

## 2. Method

- **Unit under test is the caller, not the kernel.** `runner_op.py` reproduces
  `_streamidx_topk_jax` — KV-cache scatter, fp8 bitcast, `streamindex_topk` — inside one
  `jax.jit`. Earlier probes measured the kernel alone; a runner change must be judged on what the
  runner runs, and the scatter turns out to be the largest single item in it (§6).
- **Device time from `hlo_stats` self time**, bucketed by `probes/bdt_hlo_stats.py` (imported, not
  reimplemented). Partition check `sum(buckets) − TensorCore total == 0` and
  `rows_kept == rows_total` pass on all 17 arms in both timing jobs. The SparseCore plane is
  reported separately and **never added** to the TensorCore total.
- **Per-call wall clock is `max(device, host)`** and is reported only as a sanity check, never as
  a result. Nothing is subtracted from it. The wall clock printed by `bench_runner_trace.py` is
  from a shared pod and is labelled `CONTENDED_NOT_A_RESULT`.
- **`cache_kv` is donated** in job 3 (`jax.jit(fn, donate_argnums=(2,))`), matching the real op's
  `pallas.jax_op(..., donate_argnums=(2,))`. The timing loops chain each call's returned cache
  into the next, which is what production does. Job 2 did not donate; comparing the two is how the
  cost of *not* donating was isolated (§6).
- Every arm is a separate `jax.profiler.trace` logdir. `hlo_stats` keys on HLO op name, so two
  arms in one trace would be summed.

### Arms

| arm | what it is |
| --- | --- |
| `inc` | production caller, production kernel: `enable_early_exit=True`, `(2,2,2)`, `(1,128,128)` — `mla_attention_op.py:107-122` verbatim |
| `bG1` | **the baseline every B number is divided by**: the merged module (bkv_p ladder + D1 + Option C) through the same caller, unchunked |
| `bG2` / `bG4` / `bG8` | the same, chunked |

Dividing by `bG1` isolates Option B from the three changes it is taken *with*; `inc` is shown too
so the compound number is visible.

### Configs

| config | slots | real seqs | tokens | why |
| --- | --- | --- | --- | --- |
| `pre_q1024_ctx9216_s1` | 1 | 1 prefill, q=1024, ctx=9216 | 1024 | the geometry every earlier probe used, so the numbers are comparable |
| `pre_q1024_ctx9216_s64` | 64 | the same one sequence | 1024 | **as vLLM actually presents it**: a padded 64-slot bucket |
| `mix_ragged_s64` | 64 | 4 ragged prefill (q 300/100/500/124) | 1024 | chunk boundaries fall *inside* sequences — the pseudo-sequence harness could not express this |
| `mixdec_s64` | 64 | 8 decode + 2×508 prefill | 1024 | decode and prefill in one call, so the static decode guard is tested rather than asserted |
| `dec_b64_ctx8704_s64` | 64 | 64 decode | 64 | production decode: the policy must refuse to chunk |

`mix_ragged_s64` is correctness-only. It is a shape vLLM *can* produce but not the one the nightly
runs, and a perf number from it would invite being quoted as production.

---

## 3. What was built, and what was deliberately not touched

**`src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py` was not modified.** The change
authorized by the brief was made as a copy, `variants/option-b-runner/runner_op.py`, so that `inc`
and `bG*` are measurable in the *same process* against the *same* inputs. The risk of a copy is
that it silently drifts from the thing it claims to represent, so `runner_op.check_fidelity()`
re-parses the real file with `ast`, slices out the scatter prologue, and asserts it is
**character-identical**. Job `j-46ad1b70` prints:

```
FIDELITY real_caller=/root/cloud-devkit/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py
FIDELITY prologue_identical=True prologue_lines=23
```

and then the full unified diff of the function, which is exactly the intended delta:

```
  q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
+ # ---- Option B: the entire caller-side change. -------------------
+ # Both operands are SHAPES, so both are static under `jax.jit`; `g`
+ # is therefore a Python int by the time `streamindex_topk` sees it.
+ g = chunk_policy(q_bytes.shape[0], seq_lens.shape[0], prefill_g) if chunked else 1
+ # -----------------------------------------------------------------
  topk_indices = streamindex_topk(
      ...
-     enable_early_exit=STREAMIDX_ENABLE_EARLY_EXIT,
+     head_major_prefill=head_major_prefill,
+     num_chunks=g,
  )
```

**Five lines in the caller.** That is the whole runner-side change.

The chunk loop itself lives in `variants/option-b-runner/streamindex_topk_b.py`, a copy of
`variants/merged/streamindex_topk_merged.py` (md5 `5bc0578427412d56209ef8b5d764fcae`) plus one new
static argument, `num_chunks`. It has to live there rather than in the caller because the thing
being looped over — `run_scores_kernel` and the score buffer it writes — is a closure the caller
cannot reach. Performance is identical either way: both inline into one XLA module.

`streamindex_topk_b.py` adds three things and changes nothing else:

- `plan_chunks(num_tokens, max_num_seqs, num_chunks) -> dict` — the **static** half of the plan.
  Pure Python on integers, no tracing. Returns the chunk ranges, which chunks need the decode
  launches, the stage-A launch count, and `ok`/`reason`. `streamindex_topk` raises `ValueError` on
  `ok=False`; the planner itself never raises, because a rejection is a reportable result.
- `chunk_metadata(seq_lens, cu_q_lens, distribution, a, b)` — the **traced** half (§4).
- a `_chunk(a, chunk_len, with_decode)` restructuring of `_common_path`, plus a driver that at
  `num_chunks == 1` calls the merged path verbatim and otherwise concatenates the per-chunk
  results.

`chunk_metadata` is module-level rather than nested inside `streamindex_topk` specifically so the
`plan` probe exercises the real function instead of a transcription of it.

---

## 4. The §7 blocker that turned out not to be one

`option-b-pipeline/RESULTS.md` §7 listed two integration problems.

### 4.1 Chunk boundaries must fall on sequence boundaries — solved, and more generally than the harness needed

Stage A's grid is over *sequences* and its masking is driven by `cu_q_lens`, so a chunk has to look
like a batch of sequences. The harness faked this by presenting one 1024-token sequence as `G`
pseudo-sequences whose `seq_lens` advance by `q_len/G`. The general construction falls out of one
requirement: **every token's absolute causal position must be unchanged** when the call is narrowed
to the token window `[a, b)`. Position is `seq_lens[j] − q_len_j + (t − cu_q_lens[j])`, so:

```
cu_q_lens'[j] = clip(cu_q_lens[j], a, b) − a
seq_lens'[j]  = max(seq_lens[j] − max(0, cu_q_lens[j+1] − b), 0)
mixed_start   = clip(#{valid j : end_j ≤ a},        distribution[1], distribution[2])
mixed_end     = clip(max(#{valid j : end_j ≤ b−1} + 1, mixed_start), mixed_start, distribution[2])
```

`#{...}` is a masked sum over `arange(max_num_seqs) < distribution[2]`, not a `searchsorted`: the
padded tail of `cu_q_lens` repeats the total and is not trustworthy as a sorted array, which is
also why `_effective_row_lengths` masks it explicitly.

Sanity-checked against the plan `variants/bxc/` validated by hand, from job `j-46ad1b70`:

```
BXC-CROSSCHECK G=4 seq_lens expected=[8448, 8704, 8960, 9216] got=[8448, 8704, 8960, 9216]
               cu_q_lens=[[0, 256], [0, 256], [0, 256], [0, 256]] match=True
```

The general formula reproduces the hand-built pseudo-sequence plan exactly — and it also handles
what the pseudo-sequence trick could not. On `mix_ragged_s64` at `G=8`, chunk `[256, 384)` cuts
through the middle of two different sequences:

```
chunk [256, 384] seq_lens=[9216, 4080, 6652, 1408] cu_q_lens=[0, 44, 128, 128, 128] mixed=[0,2)
```

### 4.2 "`cu_q_lens` and `distribution` are traced, so the plan must be static metadata" — **this is wrong**

Nothing above needs a host-computed plan. It is elementwise arithmetic and masked sums on arrays
that are already traced. The *only* value that must be static is `G`, because it fixes the number
of loop iterations, and `G` is chosen by:

```python
def chunk_policy(num_tokens, max_num_seqs, prefill_g=4):
    g = prefill_g
    while g > 1:
        if num_tokens % g == 0 and num_tokens // g >= max_num_seqs:
            return g
        g -= 1
    return 1
```

Both arguments are **shapes** (`q_bytes.shape[0]`, `seq_lens.shape[0]`). Under `jax.jit` a shape is
static by construction, so `g` is a Python `int` before tracing begins. **No plan crosses the host
boundary, and no new value has to be threaded through vLLM's calling convention.**

Two consequences worth naming:

- **`chunk_policy` doubles as the phase gate.** Phase is traced (`distribution`) and cannot be
  branched on, but `num_tokens` is static and separates the phases at this deployment: prefill is
  a 1024-token bucket, decode is a 64-token one. Verified on hardware:
  `DECODE-GATE chunk_policy(num_tokens=64, max_num_seqs=64) = 1 (must be 1)`. This matters —
  `option-c-head-major` measured **+59.62 %** (worse) when its prefill-only layout was applied at
  decode.
- **`chunk_size ≥ max_num_seqs` is a static guard that removes the launch-count penalty.** Decode
  sequences occupy global tokens `[0, distribution[0])` at one token each, and
  `distribution[0] ≤ max_num_seqs`. If `num_tokens/G ≥ max_num_seqs` — both static — every decode
  token is provably in chunk 0, so chunks `i>0` omit the two decode launches *at trace time*. That
  turns `3G` stage-A launches into `2 + G`. When the guard fails, `plan_chunks` returns `ok=False`
  and `streamindex_topk` raises rather than quietly doing the wrong thing:

  ```
  PLAN dec_b64_ctx8704_s64 G=4 ok=False :: chunk_size=16 < max_num_seqs=64: a decode token
       could fall outside chunk 0, so the decode launches cannot be statically omitted
  ```

Counted on the device (`hlo_stats` op names, job `j-b86c8fce`), the guard holds:

| arm | stage-A mixed launches | stage-A decode launches | stage-B (SparseCore top-k) launches |
| --- | ---: | ---: | ---: |
| `bG1` | 1 | 2 | 1 |
| `bG2` | 2 | 2 | 2 |
| `bG4` | 4 | 2 | 4 |
| `bG8` | 8 | 2 | 8 |

Decode stays at 2, not `2G`. (`plan_chunks`'s `num_launches` counts stage-A launches only; each
chunk also issues one SparseCore top-k, so the device sees `2 + 2G` kernel invocations.)

---

## 5. Correctness — job `j-46ad1b70-e536-4e25-902a`, 17/17, before any timing

Reference is the **production caller calling the production kernel** (`inc`), not the merged module
and not a harness. Comparison is per-row top-k **index set** equality, order-insensitive, ignoring
`-1`; the returned `cache_kv` is additionally checked bit-identical. Stage-A *scores* are not
bitwise identical under Option C and are not expected to be; what must match is the selection.

```
CORRECT pre_q1024_ctx9216_s1  G=1 bad_rows=0/1024 cache_identical=True ref_neg1=0    got_neg1=0    OK=True
CORRECT pre_q1024_ctx9216_s1  G=2 bad_rows=0/1024 cache_identical=True ref_neg1=0    got_neg1=0    OK=True
CORRECT pre_q1024_ctx9216_s1  G=4 bad_rows=0/1024 cache_identical=True ref_neg1=0    got_neg1=0    OK=True
CORRECT pre_q1024_ctx9216_s1  G=8 bad_rows=0/1024 cache_identical=True ref_neg1=0    got_neg1=0    OK=True
CORRECT pre_q1024_ctx9216_s64 G=1..8  bad_rows=0/1024  cache_identical=True          OK=True  (4 arms)
CORRECT mix_ragged_s64        G=1..8  bad_rows=0/1024  cache_identical=True  ref_neg1=7626 got_neg1=7626  OK=True  (4 arms)
CORRECT mixdec_s64            G=1..8  bad_rows=0/1024  cache_identical=True          OK=True  (4 arms)
CORRECT dec_b64_ctx8704_s64   G=1     bad_rows=0/64    cache_identical=True          OK=True
SKIP    dec_b64_ctx8704_s64   G=2,4,8: plan rejects it
CORRECT_ARM_OK=True
```

The `mix_ragged_s64` rows are the load-bearing ones: 7626 `-1` fills reproduced exactly means the
per-row *effective length* — which is derived from the rewritten `seq_lens`/`cu_q_lens` — is right
even where a chunk boundary lands mid-sequence.

This is a **microbenchmark-level correctness check on real hardware**, not an accuracy evaluation.
It establishes that chunking does not change which tokens the indexer selects. It does not
establish end-to-end model quality, and no end-to-end run was made.

---

## 6. Device time — job `j-b86c8fce-fd3d-4b63-9a24`

### A. Full caller, as the runner runs it (donated `cache_kv`)

| config | arm | full caller µs | Δ vs `bG1` | Δ vs `inc` | stage A µs | stage B on TC µs | KV-scatter block µs | other µs | SparseCore plane µs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pre_q1024_ctx9216_s1` | `inc` | **1768.5** | +3.50% | +0.00% | 249.3 | 434.0 | 999.9 | 85.3 | 432.0 |
| `pre_q1024_ctx9216_s1` | `bG1` | **1708.6** | +0.00% | -3.39% | 197.6 | 434.1 | 998.7 | 78.2 | 431.6 |
| `pre_q1024_ctx9216_s1` | `bG2` | **1601.3** | -6.28% | -9.46% | 201.5 | 307.3 | 999.2 | 93.2 | 440.9 |
| `pre_q1024_ctx9216_s1` | `bG4` | **1550.6** | **-9.25%** | **-12.32%** | 209.0 | 238.6 | 999.3 | 103.8 | 454.0 |
| `pre_q1024_ctx9216_s1` | `bG8` | **1546.0** | -9.52% | -12.58% | 223.0 | 193.8 | 998.5 | 130.7 | 477.7 |
| `pre_q1024_ctx9216_s64` **(prod)** | `inc` | **1976.4** | +3.20% | +0.00% | 247.6 | 434.6 | 999.2 | 295.0 | 432.4 |
| `pre_q1024_ctx9216_s64` **(prod)** | `bG1` | **1915.0** | +0.00% | -3.10% | 197.8 | 434.5 | 998.6 | 284.1 | 432.2 |
| `pre_q1024_ctx9216_s64` **(prod)** | `bG2` | **1809.7** | -5.50% | -8.44% | 202.2 | 268.1 | 999.2 | 340.2 | 442.6 |
| `pre_q1024_ctx9216_s64` **(prod)** | `bG4` | **1712.2** | **-10.59%** | **-13.36%** | 209.9 | 235.6 | 998.2 | 268.6 | 455.2 |
| `pre_q1024_ctx9216_s64` **(prod)** | `bG8` | **1705.7** | -10.93% | -13.69% | 224.4 | 175.6 | 998.8 | 306.9 | 477.7 |
| `mixdec_s64` | `inc` | **1874.8** | +2.80% | +0.00% | 214.5 | 367.7 | 998.2 | 294.5 | 364.2 |
| `mixdec_s64` | `bG1` | **1823.7** | +0.00% | -2.73% | 174.9 | 367.9 | 996.8 | 284.1 | 364.2 |
| `mixdec_s64` | `bG2` | **1752.6** | -3.90% | -6.52% | 202.7 | 213.2 | 996.2 | 340.4 | 372.2 |
| `mixdec_s64` | `bG4` | **1656.5** | **-9.17%** | **-11.65%** | 210.5 | 181.7 | 995.6 | 268.6 | 382.1 |
| `mixdec_s64` | `bG8` | **1652.5** | -9.39% | -11.86% | 224.4 | 124.4 | 996.8 | 306.9 | 405.0 |
| `dec_b64_ctx8704_s64` | `inc` | **181.9** | +2.60% | +0.00% | 111.9 | 32.0 | 8.7 | 29.3 | 31.1 |
| `dec_b64_ctx8704_s64` | `bG1` | **177.3** | +0.00% | -2.53% | 111.6 | 32.1 | 8.8 | 24.8 | 31.0 |

Decode has only two arms because `chunk_policy` refuses to chunk it — which is the correct
behaviour, not a gap.

### B. The same table with the KV-cache scatter block removed

| config | arm | caller − KV-scatter µs | Δ vs `bG1` | Δ vs `inc` | stage A+B µs | Δ A+B vs `bG1` | Δ A+B vs `inc` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pre_q1024_ctx9216_s1` | `inc` | **768.6** | +8.26% | +0.00% | 683.3 | +8.16% | +0.00% |
| `pre_q1024_ctx9216_s1` | `bG1` | **710.0** | +0.00% | -7.63% | 631.8 | +0.00% | -7.54% |
| `pre_q1024_ctx9216_s1` | `bG2` | **602.1** | -15.20% | -21.67% | 508.9 | -19.46% | -25.53% |
| `pre_q1024_ctx9216_s1` | `bG4` | **551.3** | **-22.35%** | **-28.27%** | 447.5 | -29.16% | -34.51% |
| `pre_q1024_ctx9216_s1` | `bG8` | **547.5** | -22.89% | -28.77% | 416.8 | -34.03% | -39.01% |
| `pre_q1024_ctx9216_s64` **(prod)** | `inc` | **977.2** | +6.63% | +0.00% | 682.3 | +7.90% | +0.00% |
| `pre_q1024_ctx9216_s64` **(prod)** | `bG1` | **916.4** | +0.00% | -6.22% | 632.3 | +0.00% | -7.32% |
| `pre_q1024_ctx9216_s64` **(prod)** | `bG2` | **810.5** | -11.56% | -17.06% | 470.3 | -25.63% | -31.07% |
| `pre_q1024_ctx9216_s64` **(prod)** | `bG4` | **714.1** | **-22.08%** | **-26.93%** | 445.5 | -29.55% | -34.70% |
| `pre_q1024_ctx9216_s64` **(prod)** | `bG8` | **706.9** | -22.86% | -27.66% | 400.1 | -36.73% | -41.36% |
| `mixdec_s64` | `inc` | **876.7** | +6.01% | +0.00% | 582.2 | +7.25% | +0.00% |
| `mixdec_s64` | `bG1` | **826.9** | +0.00% | -5.67% | 542.8 | +0.00% | -6.76% |
| `mixdec_s64` | `bG2` | **756.3** | -8.54% | -13.73% | 416.0 | -23.37% | -28.55% |
| `mixdec_s64` | `bG4` | **660.9** | **-20.08%** | **-24.62%** | 392.2 | -27.74% | -32.63% |
| `mixdec_s64` | `bG8` | **655.7** | -20.71% | -25.21% | 348.8 | -35.75% | -40.10% |
| `dec_b64_ctx8704_s64` | `inc` | **173.2** | +2.81% | +0.00% | 143.9 | +0.15% | +0.00% |
| `dec_b64_ctx8704_s64` | `bG1` | **168.5** | +0.00% | -2.73% | 143.7 | +0.00% | -0.15% |

Table B is not a subtraction of a host floor or of anything estimated. The KV-scatter block is
three *named device ops* whose measured self time is **998.2–999.9 µs in every prefill arm** (a
1.7 µs spread across ten arms), so removing it changes no comparison — it only changes the
denominator. Both denominators are given because they answer different questions: A is what the
runner costs, B is what Option B is competing against.

### Where the win comes from, and what it costs on the device

At `pre_q1024_ctx9216_s64`, `bG1 → bG4`:

| | `bG1` | `bG4` | Δ |
| --- | ---: | ---: | ---: |
| stage B on TensorCore (TC blocked waiting for SparseCore) | 434.5 | 235.6 | **−198.9 µs** |
| stage A (TensorCore score kernel) | 197.8 | 209.9 | +12.1 µs |
| SparseCore plane (concurrent, never added) | 432.2 | 455.2 | +23.0 µs |
| everything else (concatenate, per-chunk bitcasts, `-inf` inits) | 284.1 | 268.6 | −15.5 µs |

The mechanism is exactly the one Option B predicted: the TensorCore stops idling through the
SparseCore top-k. Stage B's TC time falls 45.8 % while the SparseCore's *own* work rises 5.3 % —
more, smaller waves cost a little more in total but overlap much better.

Two costs Option B genuinely adds, both small and both visible above:

- **Stage A gets slower with `G`** (+6.1 % at `G=4`, +13.4 % at `G=8`). The per-layer
  precomputation runs once per chunk; taking B together with D1 lets XLA's CSE collapse most of
  it, but not all. This is the sub-additivity the harness predicted, now measured on the real path.
- **The SparseCore plane grows monotonically** (432 → 443 → 455 → 478). At `G=8` the extra
  SparseCore work has caught up with the extra overlap, which is why `G=8` is only 0.3 pp better
  than `G=4` and costs 0.4 s more to compile. **`G=4` is the knee.**

### Comparison with the harness numbers this was supposed to make real

| | harness (kernel only) | here (caller, `−` KV-scatter, `s1`) |
| --- | ---: | ---: |
| baseline (incumbent layout, unchunked) | 728.30 µs | 768.6 µs |
| B×C at `G=4` | 537.60 µs (**−26.18 %**) | 551.3 µs (**−28.27 %**) |

The harness reproduces to within ~2 pp once the same denominator is used. What the harness did not
charge for is the KV-cache scatter, and that is not a rounding error (§6.1).

### 6.1 The finding nobody was looking for: the KV-cache scatter is 52 % of the caller at prefill

`hlo_stats`, `pre_q1024_ctx9216_s1__bG1`, three ops, 998.7 µs of 1708.6:

```
%reshape.315 = u8[589824,256]{1,0:T(8,128)(4,1)} reshape(u8[576,256,4,256]{3,2,1,0:T(4,128)(4,1)} %cache_kv.1)
    461.95 µs   HBM-bound, 608.8 GiB/s
%scatter_fusion = u8[589824,256] fusion(%reshape.315, s32[1024] %broadcast_select_fusion, u8[1024,256] %pad.0.clone)
    134.95 µs   HBM-bound, 3.6 GiB/s effective
%reshape.317.cache_kv_hbm_ref = u8[576,256,4,256]{3,2,1,0:T(4,128)(4,1)} reshape(u8[589824,256]{1,0:T(8,128)(4,1)} %scatter_fusion)
    401.76 µs   HBM-bound, 700.1 GiB/s
    Source Info: runner_op.py:188   (= mla_attention_op.py:104, cache_kv.at[...].set(k_packed))
```

XLA's scatter emitter wants the cache as a 2-D `T(8,128)` array; the cache arrives as 4-D
`T(4,128)`. So the **entire 151 MiB cache is relaid out, scattered into, and relaid out back —
twice per call, per layer** — to write 1024 rows of 129 bytes.

Three things make this worth acting on rather than filing:

1. **It is not a harness artifact.** Job 2 ran without `donate_argnums`; job 3 ran with it, exactly
   as `pallas.jax_op(..., donate_argnums=(2,))` does in production. Donation removed the decode
   arm's whole-cache copy (`copy.21`, 170.1 µs — decode fell 346.8 → 177.3 µs) and left these three
   ops **unchanged to 0.2 %**. The relayout is a layout mismatch, and donation cannot fix a layout
   mismatch.
2. **It scales with the cache, not with the work.** 462 µs is 151 MiB at 609 GiB/s. Production's
   per-layer indexer cache is larger than the 576 pages used here, and the cost is proportional.
3. **It does not happen at decode.** At `n_tok = 64` XLA picks an in-place emitter and the whole
   block is 8.8 µs. So this is a prefill-only, shape-triggered pathology, which is the kind that
   survives a long time unnoticed.

**Caveat, stated plainly:** this was measured with a `cache_kv` allocated by JAX at the shape the
op declares. Production's tensor comes from vLLM's cache engine via torch-tpu, and if it arrives in
a different layout the relayout may not occur. **This needs one look at a production xprof before
it is believed**, and it is not claimed as a production number here. But if it does reproduce, it
is a larger prize than Option B.

### 6.2 Padding to 64 slots costs 207 µs

`pre_q1024_ctx9216_s1` and `pre_q1024_ctx9216_s64` are the *same single sequence*; the only
difference is `max_num_seqs`. `bG1` goes 1708.6 → 1915.0 µs (+206.4), entirely in the "other"
bucket (78.2 → 284.1). That is the per-slot gather/searchsorted/page-index work done for 63 empty
slots. Noted, not pursued.

---

## 7. What the runner change costs — host time, launches, compiles, retracing

This is the one place where host time is the point rather than an artifact, so it was measured
separately and deliberately (`probe_runner.py --arm hostcost`, job `j-b86c8fce`).

| config | arm | wall µs = `max(device, host)` | dispatch-only µs (host floor) | compile s | StableHLO lines | `tpu_custom_call` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pre_q1024_ctx9216_s64` | `inc` | 2755.4 | 31.2 | 3.3 | 500 | 4 |
| `pre_q1024_ctx9216_s64` | `bG1` | 2712.9 | 30.4 | 2.7 | 410 | 4 |
| `pre_q1024_ctx9216_s64` | `bG2` | 2609.0 | 29.3 | 3.1 | 641 | 5 |
| `pre_q1024_ctx9216_s64` | `bG4` | 2531.7 | 34.3 | 3.0 | 921 | 7 |
| `pre_q1024_ctx9216_s64` | `bG8` | 2525.3 | 29.6 | 3.4 | 1481 | 11 |
| `pre_q1024_ctx9216_s1` | `inc` | 2055.2 | 35.2 | 2.6 | 499 | 4 |
| `pre_q1024_ctx9216_s1` | `bG1` | 1970.2 | 28.9 | 2.2 | 409 | 4 |
| `pre_q1024_ctx9216_s1` | `bG2` | 1882.4 | 29.4 | 2.4 | 639 | 5 |
| `pre_q1024_ctx9216_s1` | `bG4` | 1833.5 | 32.0 | 2.6 | 917 | 7 |
| `pre_q1024_ctx9216_s1` | `bG8` | 1848.1 | 32.3 | 3.0 | 1473 | 11 |
| `dec_b64_ctx8704_s64` | `inc` | 469.6 | 33.0 | 2.6 | 500 | 4 |
| `dec_b64_ctx8704_s64` | `bG1` | 496.2 | 31.4 | 2.3 | 410 | 4 |

Wall clock is `max(device, host)`; nothing is subtracted from it, and it is here only as a
consistency check — it tracks the device numbers in §6A with a roughly constant ~800 µs offset,
so the host is not the binding constraint at any `G`.

**Host-side cost of the runner change: none measurable.**

- **Dispatch is flat**: 28.9–35.2 µs with no trend in `G` (`bG1` 30.4, `bG8` 29.6 at the production
  shape). Job 2, an independent run, gives 20.3–25.5 µs, also flat. The absolute level differs
  between the two jobs because job 3's loop chains the donated cache; the *invariance in `G*` is
  the result, and it holds in both.
- **The reason it is flat**: all `G` chunks are inside one `jax.jit`, so there is exactly **one**
  host dispatch per call regardless of `G`. The `2 + G` stage-A and `G` stage-B launches are
  device-side, issued by the runtime from one executable. There is no `2 + G`× host cost.
- **Plan construction costs nothing at steady state.** `chunk_policy` is Python on two shapes and
  runs at *trace* time. After the first call it never executes again.

**Compile and code size**: `+0.3 s` at `G=4` and `+0.7 s` at `G=8` over `bG1`; StableHLO 410 → 921
→ 1481 lines; `tpu_custom_call` 4 → 7 → 11. Real but small, and paid once per shape bucket.

**Retracing: no additional retracing.** `G` is a function of `(q.shape[0], seq_lens.shape[0])`.
vLLM already compiles one executable per token bucket and per `max_num_seqs`, so `G` is constant
within every executable that already exists. The change adds **zero** new compilation axes. This is
the direct consequence of §4.2, and it is the reason the runner change is five lines.

**Not measured, and it should be said**: torch/Dynamo-side cost. `runner_op.py` reproduces the JAX
function, not the `pallas.jax_op` / `register_fake` / `op()` wrapper around it, because that
plumbing needs a vLLM model runner and cannot be stood up on a 4-chip pod. Since the change adds
one Python `int` to a kwargs dict at build time and nothing at call time, the wrapper cost should
be unchanged — but that is an argument, not a measurement.

---

## 8. Repeatability

Two independent jobs, same code path (the only difference is `donate_argnums`), full caller
TensorCore µs:

| config | arm | `j-8185ac37` | `j-b86c8fce` | Δ |
| --- | --- | ---: | ---: | ---: |
| `pre_q1024_ctx9216_s1` | `inc` | 1768.3 | 1768.5 | +0.01% |
| `pre_q1024_ctx9216_s1` | `bG1` | 1709.3 | 1708.6 | -0.04% |
| `pre_q1024_ctx9216_s1` | `bG4` | 1549.7 | 1550.6 | +0.06% |
| `pre_q1024_ctx9216_s1` | `bG8` | 1548.0 | 1546.0 | -0.13% |
| `pre_q1024_ctx9216_s64` | `bG1` | 1915.4 | 1915.0 | -0.02% |
| `pre_q1024_ctx9216_s64` | `bG4` | 1713.7 | 1712.2 | -0.08% |
| `mixdec_s64` | `bG4` | 1660.3 | 1656.5 | -0.23% |
| `dec_b64_ctx8704_s64` | `inc` | 354.1 | 181.9 | **-48.64%** |
| `dec_b64_ctx8704_s64` | `bG1` | 346.8 | 177.3 | **-48.88%** |

Every prefill arm reproduces to ≤0.23 %. The two decode rows are the *intended* difference: they
are the whole-cache copy that donation removes.

---

## 9. What did not work, and what is still open

- **`variants/option-b-pipeline/RESULTS.md` §7's second integration problem is not real.** It says
  the chunk plan has to be handed in as static metadata because `cu_q_lens` and `distribution` are
  traced. It does not. Only `G` is static, and `G` comes from shapes. Anyone reading §7 should read
  §4.2 here instead.
- **`G > 4` is not worth it.** `G=8` was never run before. It works, it is correct, and it is
  0.3 pp better than `G=4` at the production shape while costing 0.4 s more per compile and 560
  more lines of StableHLO. The SparseCore plane grows 432 → 478 µs across the sweep, which is what
  eats the extra overlap.
- **Option B cannot be applied at decode**, and the guard now enforces that statically rather than
  by convention. `chunk_size < max_num_seqs` is rejected with a reason string.
- **The `+59.62 %` Option C decode regression stays fenced off** by the same static
  `num_tokens`-based gate. That fence is only as good as the assumption that decode buckets are
  smaller than `max_num_seqs`; if a deployment ever compiles a decode bucket of ≥ `max_num_seqs`
  tokens, `chunk_policy` would chunk it. Worth a comment in the caller if this ships.
- **The KV-cache scatter relayout (§6.1) is unverified against production.** It is the largest
  single item in the prefill caller here and Option B does nothing about it. One production xprof
  would settle whether it is real, and it is a cheaper question than anything else on this list.
- **No end-to-end model benchmark was run**, per the brief. What is established is device time for
  one op at hand-built shapes and index-identical selection. Whether −10.6 % of this op moves
  GLM-5.2 end to end depends on the op's share of the step, which is not measured here.

### 9.1 The baseline this was built on has since been partly withdrawn — read the `vs inc` column with care

The brief said to take Option B *with* D1 rather than instead of it, so `bG1` is the merged module:
the `bkv_p` ladder + D1 + Option C. While this was being measured, a concurrent effort
(`j-d17207d1-05fb-4e10-93b3`, commit `46c62c14`) withdrew two of those three:

- **D1** deletes an early-exit branch that *does* fire. At runtime `ctx` 1024/2048 with `k=2048`
  the incumbent's fast path costs 7.7 µs and D1 costs **7.4–7.6×** that, against 4.67 % bought at
  the production shape.
- **`bkv_p` sized to `max_model_len`** makes the block the quantum of overfetch, so a short request
  scans the whole 9216-position bucket: **+98 %** on the scoring path at `ctx` 1024. Composed, the
  merged op is 14.7× slower than the incumbent at `ctx` 1024.

Option C, prefill-gated, survives unqualified.

What this does and does not do to the numbers above:

- **The Option B columns are unaffected.** Every `Δ vs bG1` divides two arms that share the same
  merged module, so D1's and `bkv_p`'s behaviour cancels exactly. `−10.59 %` at `G=4` is Option B.
- **The `Δ vs inc` columns are not a shipping claim.** They compound B with two changes that are
  now withdrawn, so `−13.36 %` describes a configuration nobody currently proposes to ship. Read
  those columns as "B on top of the merged module as it stood on 2026-09-03", not as the effect of
  landing B.
- **Every config here is long-context** (`ctx` 8704–9216), which is precisely the regime where D1
  and `bkv_p` look good. This exercise could not have caught what `j-d17207d1` caught, and did not.
- **Nothing here was re-measured against a D1-less or `bkv_p`-less base.** If the shipping candidate
  becomes C-only, Option B needs one more sweep on that base before its `vs inc` number is quoted.
  The `Δ vs bG1` mechanism — the TensorCore stops idling through the SparseCore top-k — has no
  dependence on either withdrawn change, so the expectation is that it carries over largely intact;
  that is an argument, not a measurement.

---

## Reproducing

```bash
# on-pod, one unpinned process on the 4-chip pod
bash /root/kw/variants/option-b-runner/runner_shard.sh plan
bash /root/kw/variants/option-b-runner/runner_shard.sh correct
bash /root/kw/variants/option-b-runner/runner_shard.sh hostcost
bash /root/kw/variants/option-b-runner/runner_shard.sh trace     # needs the /tmp/j11 jax 0.11 venv

# off-pod
cdk job sync-outputs <job-id>
python variants/option-b-runner/read_runner_hlo_stats.py <synced-dir> --out r.json --md r.md
```

`trace` and `hostcost` are timings: run them on the whole pod, never fanned out.
