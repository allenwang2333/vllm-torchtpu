# Low-latency all-reduce kernels in the K3 forward pass

Branch `gnovack/low-latency-allreduce-k3`. Whether the custom Pallas
collectives in `kernels/collectives/low_latency_all_reduce.py`, which beat
XLA's ring by 1.1-2.5x in JAX microbenchmarks (`docs/low_latency_allreduce.md`),
deliver on K3's critical path at TP=32.

## The answer

**They do, but only two-shot, and only where XLA was not already fusing.**

* **two-shot works in production.** On the post-attention reduction it costs
  **13.32 us/occ against XLA's 19.35 (-31%)**, and that 13.32 is essentially
  its isolated bench number of 12.67 -- a 5% production tax. The
  microbenchmark transfers.
* **one-shot is out at any arrangement.** Its egress is `M(N-1)`, so it only
  wins below ~16 KiB at 32 devices. Every arrangement that avoids the fusion
  penalty requires a *larger* payload, which is exactly where it loses.
* **The first attempt regressed for a reason unrelated to kernel speed.**
  Routing the 14 KiB reduction to a kernel broke XLA's collective combiner,
  and the lost fusion cost more than the faster kernel gained.

The best measured arm is still slower than the control, because it runs with
the combiner disabled to isolate the kernel. The arrangement projected to beat
the control -- a single two-shot over the concatenated pair -- is in
[next steps](#next-steps), not yet built.

## Every number in one place

Isolated harness (`scripts/debug/profile_ll_allreduce_shapes.py`), device self
time us/occ, bfloat16, through `torch.compile` and the real dispatcher.

**32 devices**

| shape | payload | XLA | one-shot | two-shot |
|---|--:|--:|--:|--:|
| `2x3584` | 14 KiB | 17.31 | 13.59 | **12.36** |
| `2x7168` | 28 KiB | 17.97 | 22.39 | **12.67** |
| `2x10752` | 42 KiB | 18.93 | 33.20 | **13.20** |
| `16x1024` | 32 KiB | 17.96 | 25.33 | -- |
| `16x3584` | 112 KiB | 21.86 | 89.70 | -- |
| `16x7168` | 224 KiB | 23.26 | 178.67 | -- |

**8 devices**

| shape | payload | XLA | one-shot | two-shot |
|---|--:|--:|--:|--:|
| `2x3584` | 14 KiB | 10.12 | **4.84** | 7.76 |
| `2x7168` | 28 KiB | 10.36 | **5.93** | -- |
| `16x1024` | 32 KiB | 10.37 | **6.11** | -- |
| `2x10752` | 42 KiB | 10.61 | -- | 8.21 |
| `16x3584` | 112 KiB | 14.83 | 12.53 | -- |

one-shot's advantage is real but narrow, and collapses with device count and
payload. two-shot is nearly flat in payload (`2M(N-1)/N` egress) and is the
only variant that survives past ~16 KiB at 32 devices.

Production, K3 TP=32, rank 0, decode-only, us/occ:

| collective | control | unfused XLA | one-shot@3584 | two-shot@7168 |
|---|--:|--:|--:|--:|
| fused `[2,3584]+[2,7168]`, 92 ops | **33.97** | -- | -- | -- |
| `[2,3584]`, 92 ops | -- | 32.66 | **25.19** (kernel) | 31.36 |
| `[2,7168]`, 95/187 ops | 19.10 | 19.35 | 20.15 | **13.32** (kernel) |

End to end, TPOT ms, via `arm_check.sh`:

| arm | c1 | c2 | c4 | c8 | device |
|---|--:|--:|--:|--:|--:|
| control (XLA, fused) | **16.80** | 17.95 | 23.82 | 26.99 | 259.92 ms |
| one-shot @ `[2,3584]` | 17.77 | 18.86 | 24.00 | 26.96 | 275.61 ms |
| unfused XLA (combiner off) | 18.11 | 19.21 | 24.46 | 27.61 | 280.30 ms |
| unfused + two-shot @ `[2,7168]` | 17.63 | 18.76 | 24.55 | 27.56 | 274.11 ms |

GSM8K 100/100 and a byte-identical canary on every arm. Device deltas track
TPOT deltas within 0.2 points throughout, so nothing here is off-device.

## What the 12.65 us is made of, and how close to the floor it is

`scripts/debug/two_shot_ablation.py`, jobs 118/119, `[2,7168]` bf16 at 32
devices. A ladder of kernels identical to the production one in signature,
scratch, aliasing, compiler params, shard_map and torch wrapper -- only the
body differs -- so adjacent rungs subtract. Device self time, us/occ, 300
occurrences each:

| rung | us/occ | adds |
|---|--:|---|
| `noop` | 1.42 | launch + VMEM in/out |
| `barrier1` | 4.35 | one all-to-all barrier |
| `barrier2` | 7.27 | the second barrier |
| `p1` | 8.00 | reduce-scatter |
| **`full`** | **12.65** | all-gather |
| `full_1bar` | 11.08 | (full, single barrier) |
| `full_bf` | 16.77 | (full, barrier not overlapped) |
| `xla` | 17.99 | reference |

`full` reproduces the 12.67 measured by the other harness and `xla` reproduces
17.97, both to within 0.03 us, which is what makes the subtractions worth
anything. `barrier1` reproduced at 4.35 across two independent jobs.

| component | us | share |
|---|--:|--:|
| launch + VMEM in/out | 1.42 | 11% |
| barrier #1 | 2.93 | 23% |
| barrier #2 | 2.92 | 23% |
| phase 1 (reduce-scatter), net | 0.73 | 6% |
| phase 2 (all-gather), net | 4.65 | 37% |

**A 32-device all-to-all barrier costs 2.93 us**, derived twice by independent
subtraction and agreeing to 0.01 us. The two barriers are 46% of the kernel and
move no data at all. The bandwidth floor for the 54 KiB of egress is ~0.5 us,
confirming that essentially none of this is transfer.

So the floor is `launch + one barrier + measured transit` = **9.73 us**, against
11.08 for the best configuration actually measured and 12.65 today. The kernel
is within 30% of a floor it cannot go below, and the only removable piece is the
second barrier -- worth 2.92 us standalone but **1.57 us in place**, because the
overlap already hides the rest.

Two further readings:

* **The overlap is load-bearing, not a micro-optimization.** `full_bf` puts the
  barrier back ahead of the sends and costs 16.77, within noise of XLA's 17.99.
  The 4.12 us the overlap buys is the entire margin that makes this worth
  shipping.
* **Phase 2 is the expensive half of the data path** -- 4.65 us against phase
  1's 0.73 net, for the same bytes. The all-gather is 31 small DMAs fanning out
  from one source plus the copy-out; phase 1's arrivals spread across peers.

Read the table additively with care. The barrier rungs put the barrier first
(nothing to overlap) while `p1`/`full` put it after the sends, so `p1 -
barrier2 = 0.73` is the net of adding phase 1 *and* hiding it under the barrier,
not phase 1's raw cost. The telescoping sum equals `full` by construction and is
not an independent check; `full` matching 12.67 is.

Caveat on `noop`: it ran 180 occurrences where every other rung ran 300, so XLA
eliminated 120 of its calls -- an identity kernel with no `collective_id` and no
side effects has DCE freedom the other rungs do not. Its 1.42 us is therefore
approximate. It is probably close, since barrier #1 and barrier #2 both lean on
it and land 0.01 us apart, which they would not if it were far off.

**Consequence for where effort goes.** Perfect kernel tuning is worth 1.5-2.5 us
per call, or 0.14-0.24 ms per step across the 95 post-attention reductions. The
kernel *already* beats XLA by 5.34 us per call, or 0.51 ms per step. Tuning buys
less than half of what shipping the current version buys, so the post-attention
integration should land as it stands and the kernel internals should be left
alone.

## Why the first attempt regressed

XLA's CRS combiner merges the latent-MoE `[2,3584]` reduction and the
`[2,7168]` beside it into **one** collective costing 33.97 us. Unfused they
cost 32.66 + 19.35 = **52.01 us**. The combiner is worth 18 us per layer.

Inserting a kernel on either half forfeits that. The kernel was 7.5 us cheaper
per call and the split cost 18, so the arm lost by the difference. Proven by
running the control's flags with the combiner disabled and no kernel at all:
that arm is **slower** (18.11 ms TPOT) than the kernel arm (17.77 ms). The
whole regression was the lost fusion, and the kernel was a net -4.69 ms on top
of it.

## The `[2,3584]` penalty, which no kernel can fix

At the same site, with identical `replica_groups`, ordering and `channel_id`:

| collective | payload | isolated | production | tax |
|---|--:|--:|--:|--:|
| `[2,3584]` | 14 KiB | 17.31 | 32.66 | **+88%** |
| `[2,7168]` | 28 KiB | 17.97 | 19.35 | +8% |

The *smaller* payload costs 69% more. It is uniform across ranks (CV 1.5%, so
not expert imbalance) and implementation-independent -- XLA pays +88%,
one-shot +85%. So `[2,3584]` carries a fixed ~13-15 us **wait**, not a transfer
cost: it is the first consumer of the latent-MoE result.

This is worth more than any collective swap here: ~1.4 ms per decode step,
which the control pays too. Note that the `[2,7168]` ops at that same site cost
~13.3 us under the kernel where XLA paid ~21, so the penalty attaches to the
`[2,3584]` reduction specifically and **not** to the site.

## Getting the kernels to run at all

Most of the original work, none of it in the kernels:

| what | why it fails | resolution |
|---|---|---|
| A mesh spanning peers | vLLM gives each worker a **single-device** JAX mesh; the kernels need `shard_map` over all 32 | `jax.devices()` exposes the whole slice inside a worker, so a mesh is built over the group's TPU global device ids. **Rank order is not device-id order**, so the map is gathered |
| Skipping `shard_map` | `Cannot use communication in pallas_call without shard_map` | not available; the mesh form is mandatory |
| Operand partition specs | `P("tp")` makes the module entry the *global* shape and torch_tpu refuses the rank's local tensor | declare operands **replicated**, so global == local |
| `has_side_effects=True` | XLA's partitioner rejects a side-effecting custom call inside a shard_map's manual region: `Side-effect HLO must have sharding` | parameterised; the integration passes `False` |
| Shape read in traced code | reading dims pins vLLM's dynamic token dimension: `ConstraintViolationError` | dispatch behind an opaque `torch.library.custom_op` |
| Lazy op registration | `infer_schema` is on dynamo's skip list | register in `warmup()`, before tracing |
| `mutates_args` declared | AOTAutograd adds buffers to graph outputs: `expected 3 outputs, got 1` | leave undeclared; safe because the buffers are private, unaliased and never CSE-able |
| Workspaces captured | `identified an argument that was not provided: DeviceBufferRef` | pass them as op *arguments* |

Turning off `has_side_effects` on a collective is safe here only because the
three properties it protects hold by other means: the output is consumed, each
call site has its own `collective_id`, and the aliased workspace chain orders
successive calls. It would not be safe for a stateless collective.

two-shot needs three workspace buffers where one-shot needs two, so
`get_entry_op` fixes the op's arity from the plan chosen at warmup -- a branch
at call time is a branch dynamo would have to trace.

## Correctness

`scripts/debug/test_low_latency_all_reduce.py` drives the dispatcher exactly as
`TpuDeviceCommunicator.all_reduce` does, running each shape as a **six-round
chain** whose input depends on the previous output, so a dropped, stale or
duplicated round shows up as a value mismatch rather than hiding inside an
idempotent mean -- the workspace parity is what is most exposed.

| dtype | result |
|---|---|
| float32 | **exact**, `max|diff| = 0` on every shape, both kernels |
| bfloat16 | ~3 ulps, summation order, confirmed by the float32 run |

Tolerance must scale with device count: bf16 reordering drifts to 0.024 at 32
devices, which a fixed 0.02 threshold reports as corruption.

two-shot's `RuntimeUnexpectedCoreHalt` did **not** reproduce at 8 or 32 devices
in this session, with correctness exact throughout. One clean run is not proof
it is gone, and it has not been root-caused.

## Methodology learnings

The technical result took four arms; the wrong conclusions in between are the
reusable part.

**1. A/B an op only against the same op.** For four rounds the comparison was
`kernel@[2,3584]` against `XLA@[2,7168]` -- different collectives at different
graph positions. It produced a confident, well-evidenced and entirely wrong
mechanism ("the kernel is 7x more sensitive to production", with an egress
dose-response to explain it). More analysis of the same data would never have
caught it; building the arm that made the comparison like-for-like did, and
reversed the conclusion in one run.

**2. A microbenchmark can be perfectly accurate and still mispredict.** Slope
and device self time agreed within 0.3 us everywhere, so the methodology was
sound. It measured a *standalone* collective; production fuses collectives. The
gap was in what was benchmarked, not how.

**3. Inserting an opaque custom call forfeits XLA's collective combiner.** A
generic hazard for any kernel replacing one member of a fusable group. Check
whether the op being replaced is fused before costing the replacement.

**4. Shape-based routing cannot express site-based intent.** `(2,7168)` is both
the post-attention reduction and the MoE-output half of the fused pair.
Dispatching on `(shape, dtype)` hits both, which is why "just do the
post-attention one" needed the combiner disabled instead.

**5. Distinguish across-rank from across-call-site variance.** A rank CV of 2%
with a call-site spread of 12.67-34.98 us ruled out arrival skew and pointed at
position. Same trace, two different reductions, opposite conclusions.

**6. Cheap faithful reproduction before expensive runs.** The earlier six K3
integration failures all lived in vLLM's dynamo/AOT/piecewise stack, which the
plain-`torch.compile` harness did not exercise. The Kimi-Linear TP8 vLLM smoke
test (~10 min) caught nothing this time only because it was run first.

**7. Check payload arithmetic.** Calling the concatenated pair "21.5 KiB"
(elements mistaken for bytes; it is 42 KiB) flattered one-shot twofold and
briefly made a dead approach look indicated.

**8. Loop-invariant work is silently hoisted.** The first spacing experiment
put `gap_a @ gap_a` between chained collectives; XLA CSE'd it out of the loop
and the slope did not move, which reads exactly like "spacing does not matter".
Injected work must depend on the loop variable.

**9. Shared mutable cluster state is a correctness hazard, not just a speed
one.** `VLLM_XLA_CACHE_PATH` served one arm's executable to another
(`execution supplied 2135 arguments but compiled program expected 2133`), and
the Tier-2 cache's default `/dev/shm` namespace is owned by whichever user
creates it first. Both are now namespaced per arm and per job.

**10. Estimates are cheap to falsify, so falsify them.** The 42 KiB one-shot
projection (~33 us) matched measurement (33.20) exactly; two earlier estimates
were wrong enough to invert a recommendation.

## Next steps

Ranked by value per unit of work.

**1. Donate `rs`/`ag` instead of copying them back.** two-shot writes back two
workspace buffers per call where one-shot writes one: +6096 data-formatting ops
(+3.47 ms) and +2992 reduce ops (+1.43 ms), so ~5 ms of the ~20 ms collective
saving is handed straight back. The aliasing is already set up for donation.
Cheap, and it inflates every future measurement until fixed.

**2. One two-shot call over the concatenated pair.** The only arrangement that
both keeps the fusion's amortisation and uses the faster collective. Against
the control:

* control: 92 x 33.97 + 95 x 19.10 = 4940 us/step
* fused two-shot + post-attention two-shot: 92 x ~26 + 95 x 13.32 = 3657 us/step
* about **-20 ms device, order -5% TPOT**, less the writeback leak from (1)

Needs the dispatcher to intercept *both* reductions and concatenate, rather
than replacing one of them.

**3. Find what `[2,3584]` waits on.** ~13-15 us per layer, ~1.4 ms per decode
step, roughly 8% of TPOT -- paid by the control, and untouchable by any
all-reduce implementation. Independent of all the kernel work and plausibly the
largest single win available. Start from what feeds that reduction: in the
treatment trace its operand arrives via an async `copy-done`.

**4. Validate the corrected gate.** two-shot's floor is now 16 KiB at 16/32
devices, justified by the table above, but it is unmeasured below 14 KiB where
one-shot still holds the small end. One harness run closes that.

**5. Root-cause `RuntimeUnexpectedCoreHalt`.** It did not reproduce, so it no
longer blocks; still unexplained, and it takes the slice down when it fires.

Closed by measurement, not worth pursuing:

* **one-shot in any arrangement at TP=32** -- the split penalty forbids keeping
  the pair apart, and its crossover forbids putting it together.
* **`compile_sizes=[1]`** -- addresses neither the fusion nor the site wait.
* **Padding-awareness** -- shrinks a transfer cost at a site whose cost is a
  fixed wait; the kernel's floor was never the problem.
* **Tuning the two-shot kernel itself** -- the ablation puts it within 30% of a
  9.73 us floor that is 45% barrier. The single largest lever, merging the two
  barriers, is worth 1.57 us in place and requires solving the re-entrancy
  hazard the double barrier exists for. Every remaining optimisation is smaller
  than that. Spend the effort on call *count* instead, per (2).

## Reproducing

```bash
# K3 arms (control = TPU_LOW_LATENCY_ALLREDUCE=0)
TPU_MOE_HIERARCHICAL_EP=1 TPU_LATENT_PROJ_INTRA_CHIP_TP=1 \
TPU_LOW_LATENCY_ALLREDUCE=1 TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT=1 \
TPU_LOW_LATENCY_ALLREDUCE_SHAPE=2x7168 \
VLLM_XLA_CACHE_PATH=/mnt/scratch/users/$USER/vllm-xla-cache/<arm> \
PROFILE_DIR=$PWD/results/<arm>/traces KIMI_K3_LOG_ROOT=$PWD/results/<arm>-log \
  sbatch --job-name=<arm> scripts/k3/slurm_launch_k3_tp32_tp.sbatch

# ENDPOINT is the BARE host -- arm_check appends /v1 itself. Passing the
# banner's URL (which includes /v1) 404s every request while /health stays 200.
ENDPOINT=http://<ip>:8000 ARM=<arm> OUT=results/<arm> scripts/k3/arm_check.sh
python scripts/k3/diff_arms.py results/<control> results/<arm>
python scripts/k3/rank_spread.py results/<arm>      # skew vs position
```

To break XLA's fusion without a kernel, add to `LIBTPU_INIT_ARGS`:
`--xla_jf_crs_combiner_threshold_count=1 --xla_jf_crs_combiner_threshold_in_bytes=1`.
Verify it did only that: collective occurrences should rise by exactly the
number of unfused ops, with all-gather counts unchanged.

Isolated harness, device self time comparable to a production trace:

```bash
sbatch scripts/debug/slurm_profile_ll_allreduce_tp8.sbatch --kernel-type two_shot
LL_SHAPES="2x3584,2x7168,2x10752" LL_EXTRA_ARGS="--kernel-type two_shot" \
  sbatch scripts/debug/slurm_profile_ll_allreduce_32.sbatch
python scripts/allreduce_profile_stats.py results/ll-ar-shapes-32-<job>/traces
```

Decompose the kernel's cost into launch / barriers / phase 1 / phase 2:

```bash
sbatch --partition=ici-2x2x4-b scripts/debug/two_shot_ablation.sbatch
# one rung, or a different shape:
TS_MODES="noop,barrier1" TS_SHAPE=2x7168 \
  sbatch --partition=ici-2x2x4-b scripts/debug/two_shot_ablation.sbatch
```

Integration smoke test before any K3 run (~10 min, same dynamo/AOT/piecewise
path):

```bash
ARMS=hier COMPILE_SIZES="2,16,1024" TPU_LOW_LATENCY_ALLREDUCE=1 \
TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT=1 TPU_LOW_LATENCY_ALLREDUCE_SHAPE=16x2304 \
DOCS=20 sbatch scripts/k3/slurm_kimi_linear_tp8_hier_ep.sbatch
```

Two environment notes. `jax.profiler.trace` segfaults against this libtpu
(`Unexpected PLUGIN_Profiler_Api size`), so the JAX bench cannot profile
itself -- the torch harness uses `torch.profiler` + `TpuProfilerConfig`, which
is what production uses. And the launchers namespace the Tier-2 cache per user
and job, without which a foreign `/dev/shm/torch_tpu_cache/default` kills the
run.

The banner reports what the dispatcher *did* -- how many shapes took each
kernel, and which one was wired -- read out of the server's own log. The
dispatcher silently falls back to XLA, so "the flag was set" is not evidence
that a single all-reduce changed. `warmup` defaults to the smallest
kernel-taking shape, which is rarely the one you want.
