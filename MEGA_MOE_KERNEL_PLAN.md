# Kimi-K3 Mega MoE kernel — plan

Status: **proposal**, 2026-08-26. Nothing here is built beyond stages S0a/S0b.

Update, 2026-08-26 (evening): a **single-kernel prototype of S1+S2+S4 exists**
(`src/vllm_torchtpu/kernels/kimi_k3/mega_moe.py`, harness stage `mega` in
`scripts/k3/moe_bench.py`). Gate + topk + latent down_proj + shared MLP behind
one boundary, bit-exact vs the reference, all three weight rings primed at grid
step 0 so the streams run under the topk. Measured on TPU7x, slope method,
tokens=8: **fused 32.6 us vs 35.4-35.8 us unfused (−2.8 us/layer)**, with
priming disabled 73.7 us — the overlap machinery is worth 2x, but the unfused
chain also overlaps well in this small program, so the net in-harness edge is
modest. The win is stable across repeats (32.60/32.61). Block granularity
matters: fat blocks (7 grid steps) beat the 28-step schedule by ~4 us. Not yet
run e2e; S3 (expert GMM) untouched. Whether this beats the model's MSA-bimodal
baseline is exactly the S0a-style open gate.

Update, 2026-08-27: **a decode-specialized S3 is in** (same file): the routed
experts follow the sglang-jax fused-MoE-v2 *pattern* (tokens resident in VMEM,
per-expert weight rings, dense dots with an f32 match-matrix mask — MXU is
idle at T=8 so the wasted MACs are free), NOT gmm_v2, whose emit_pipeline owns
a whole pallas_call and whose SparseCore permute can never live inside a
TensorCore kernel. Expert weight DMAs are issued at grid step 0 right after
topk resolves, so they stream underneath the down_proj/shared-MLP phases --
the ordering XLA cannot express. The HBM permute (6.81 us/layer in the model)
does not exist in this design. Harness (bf16, dense over 8 local experts,
necessarily not mxfp4-calibrated): **fused 49.1 us vs 64.7 unfused (−24%)**;
numerics bit-exact at bn_d=256, and at the tuned config limited by the known
wide-dot latent rounding (7e-4 norm-rel, 7e-3 after situ amplification in the
expert path). ahead_e 2→1 costs +2.5 us: expert ring depth is load-bearing.
VMEM (64 MiB physical) is now the binding constraint: gate 3.5 + down 24.5 +
shared 31.5 + expert rings 3.9/slot-pair forced bn1=384 for the shared gate_up
ring and caps expert ring depth at 3. The no-priming ablation for the whole
block: 77.7 vs 64.9 unfused — XLA's chain overlap beats a naive kernel; the
priming is the entire game. Remaining before e2e: mxfp4 experts (halves expert
bytes, needs in-VMEM dequant), the ragged active-expert skip (data-dependent
DMA issue), and the S0a-style paired A/B.

Update, 2026-08-27 (later): **S5 is in too — the full MoE block is one
kernel.** The tail uses a ONE-SHOT all-reduce rather than v4's two-shot:
v4's reduce-scatter needs the latent reblocked [nd, T, cols] lane-aligned,
which our [T, L] VMEM accumulator can't produce without a transpose; at 56 KiB
the payload is skew-dominated so push-to-all-peers + local sum is equivalent
and needs no reblock. Barrier/slot idioms from v4. The up_proj ring is primed
at grid step 0 so its 24.5 MiB stream hides under the whole block. Meshed
8-dev harness (lockstep — AR skew NOT reproduced): **fused 81.7-83.0 us vs
87.8-88.8 unfused (−6..−7 us/layer)**; the mega-only win at the same configs
is −19 us, and the fused tail costs more than the unfused tail in lockstep
(XLA prefetches w_up across the whole unfused graph; our up ring is
VMEM-capped at depth 3). That in-lockstep deficit is precisely the regime
where the model's 8-68 us AR skew would flip the sign: our queue keeps
streaming up_proj blocks while peers drift. Untested claim until the paired
e2e A/B. Tuned full config: bnd=256/bn1=384/aheade=2/bnup=512/aheadup=2 --
note bn_d=256 (slim down ring) is the right GLOBAL trade once the up_proj
ring needs VMEM, reversing the experts-only finding.

Update, 2026-08-27 (later still): two follow-up measurements on the tail.
(a) `scripts/k3/ar_probe.py`: one-shot vs two-shot AR alone at the tail's
payload (8x3584 bf16, 8 dev): 9.0 vs 9.6 us — protocol choice is NOT the
tail deficit. (b) **Ring-union experiment**: the down_proj ring (dead after
its phase) doubles as extra up_proj ring depth, refilled at the phase
boundary via half-block slice views (Mosaic rejects ref.reshape across the
minor dim; the union is built from [l_out, bn_d] row/column-half slices, two
strided DMAs + two half-dots per up block). Compiles, bit-exact, and...
*neutral-to-worse in lockstep* (83.1-83.5 vs 81.9): the aheadup sweep had
already plateaued at depth ~3, i.e. the up stream is just-in-time relative
to the AR end, so earlier issue buys nothing and the strided copies cost.
Kept behind `union_up=True` (off by default); the skew regime it was
designed for needs the e2e A/B to evaluate.

Update, 2026-08-27 (mxfp4): the expert phase now takes **real mxfp4
weights** -- packed uint8 pairs along K + f32 scales (block 32), dequantized
in VMEM (`pltpu.bitcast` unpack along the second-minor axis, bf16 math,
matching gmm_v2's dequant-before-matmul path). Bit-exact vs the reference in
every config tried. Timings (full layer, meshed): **84.0 us fused vs ~318
unfused** -- the unfused XLA arm materializes dequantized weights through
HBM, which is exactly the cost the model's gmm_v2 avoids in-kernel, so this
delta is not a model-relevant win claim; what it proves is the mega kernel
handles the real weight format at parity with its own bf16 prototype (84.0 vs
81.9). The expert phase is now dequant-latency-bound, not DMA-bound
(ahead_e 1..3 all ~88-89 in the serial variant): dequant pipelining across
experts was tried and measured *worse* (99.9), recorded in the kernel
docstring. VMEM freed by the packed rings (~5 MiB) is currently unspent.

---

## 0. Where we actually are

One premise worth correcting before planning on top of it: what is fused today
is **not** "gate + shared experts". The router gate is a separate
`ReplicatedLinear` producing `[T, 224]` logits and is untouched. The two kernels
that exist are:

| kernel | contents | status |
|---|---|---|
| `all_reduce_norm_matmul_v4` | all-reduce → RMS norm → latent `up_proj` | works; −6.21 us/layer at the op level, but e2e regressed +0.79 ms because MSA re-sorted around it |
| `kimi_k3/shared_mlp` | shared `gate_up` → `situ` → shared `down_proj` | bit-exact; 14.21 vs 21.13 us/layer standalone. **e2e unconfirmed** — 25.75 and 25.19 measured for the same config |

So the honest state is *two kernels that win locally and have not yet been shown
to win end to end*. That is the single most important input to this plan.

## 1. The prize

Per-layer decode cost, rank-0 control profile, one median step (92 layers,
275.6 us/layer total):

| MoE-block op | us/layer | % of step |
|---|---|---|
| router gate + topk (sort) | 9.68 | 3.5 |
| token gather / permute | 6.81 | 2.5 |
| latent `down_proj` [1792,7168] | 11.76 | 4.3 |
| expert GMM k=3584 (mxfp4) | 30.14 | 10.9 |
| expert GMM k=3072 (mxfp4) | 14.16 | 5.1 |
| shared `gate_up` [1536,7168] | 12.64 | 4.6 |
| shared `down_proj` [7168,768] | 8.49 | 3.1 |
| LL all-reduce | 35.32 | 12.8 |
| latent `up_proj` [3584,3584] | 12.08 | 4.4 |
| **total** | **141.07** | **51.2** |

**Half the decode step is this block.** That is what makes it worth a plan; it is
also why a failure here is expensive.

## 2. The thesis, stated so it can be falsified

Decode at batch 1 is a weight-streaming problem: ~8 flops/byte, MXU idle, time =
bytes ÷ achieved bandwidth. The floor for a layer is therefore
`weight bytes / achievable HBM bandwidth`. A mega kernel approaches that floor by:

1. **paying one custom-call boundary instead of nine.** Measured Pallas fixed
   cost decomposes as 1.26 us boundary + ~1.5 us pipeline fill + ~1.7 us dot
   setup. The boundary is structural (XLA cannot hoist into or out of an opaque
   custom call). The *fill* is hideable when the kernel has independent work —
   v4 hid it behind the collective, and `shared_mlp` hides phase 2's fill behind
   phase 1's dots. Nine separate kernels would pay ~11 us/layer of boundary alone.
2. **keeping intermediates in VMEM.** Today the latent, the permuted tokens, the
   expert output and the shared-expert intermediate all round-trip through HBM.
3. **opting the weights out of MSA.** XLA's memory-space assignment has a ~32 MiB
   window; a 21.0 and a 24.5 MiB weight cannot both fit, so it makes brittle
   per-layer choices that we have watched cost 3x on individual ops. A kernel
   that streams its own weights stops competing for that window.
4. **one continuous DMA stream** rather than nine ramp-up/ramp-down pipelines.

**Falsifiable form:** if the mega kernel does not beat the sum of its parts by
more than the boundary savings (~11 us/layer), thesis (2)–(4) are wrong and the
whole exercise reduces to boundary amortisation, which is not worth the
complexity.

## 3. Why NOT one monolith on the first attempt

Three reasons drawn from what already went wrong:

* **Blast radius scales with what you delete from XLA's schedule.** v4 won
  −6.2 us/layer inside the kernel and lost +17.4 us/layer to MSA re-sorting
  around it — a 3:1 loss. A kernel that removes nine ops instead of three
  perturbs MSA far more, and we cannot currently predict which way.
* **Attribution gets harder exactly when we need it most.** This session spent
  most of its time doing forensics on a *three-op* interaction and killed six
  hypotheses. A nine-op kernel that comes back 5% slow is close to undebuggable.
* **The routed GMM is already a good Pallas kernel** (`megablox/gmm_v2.py`) with
  mxfp4 dequant and ragged dispatch. Absorbing it means reimplementing that,
  not merely calling it.

So: **staged merge, each stage independently shippable and independently
revertible**, with the e2e A/B run at every stage rather than at the end.

## 4. Stages

Dependencies are the constraint. The live decode dataflow (`use_mega_moe` is
False on TPU, so the `FusedMoE`/`LatentMoERunner` path is the one that runs):

```
hidden [T,7168]  (post-attention norm)
├── router gate → logits [T,224] → grouped topk → topk_ids/weights
├── latent down_proj → [T,3584]   (intra-chip TP: 1792/core)
│      └── permute/gather → expert GMM k=3584 → situ → GMM k=3072 → unpermute
├── shared MLP: gate_up → situ → down_proj → [T,7168] partial
└── combine → all-reduce → norm → latent up_proj → [T,7168]
```

The router and the latent `down_proj` are deliberately run in parallel today
(`maybe_execute_in_parallel`), which a fused kernel must preserve or beat.

### S0a — `shared_mlp` (DONE, unconfirmed e2e)
Exit criteria: paired e2e A/B shows a real win outside noise, and a trace
confirms the op is in the graph and MSA did not regress elsewhere. **This gate
is currently open — job 653.** Do not start S1 until it closes.

### S0b — `v4` AR+norm+up_proj (BUILT, currently a regression)
Exit criteria: either the MSA regression is fixed (`xla_tpu_msa_sort_order_overrides`,
see §6) or v4 is shelved. Carrying a known −0.8 ms regression into a bigger
kernel guarantees an uninterpretable result.

### S1 — Fuse the router gate + grouped topk into one kernel
Smallest independent win: 9.68 us/layer of `sort` plus the gate GEMM. Pure
compute, no weight streaming, no mxfp4, no collective. Good first exercise in
the ragged/argsort machinery inside Pallas.
* Input: `hidden [T,7168]`, gate weight `[224,7168]`.
* Output: `topk_ids [T,16]`, `topk_weights [T,16]`.
* Must reproduce `noaux_tc` grouped topk with sigmoid scoring, `e_score_correction_bias`,
  `num_expert_group=1`, `topk_group=1`, `renormalize=True`, `routed_scaling_factor=1.0`.
* Risk: low. Reversible.

### S2 — Fuse latent `down_proj` into S1's kernel
They are already run in parallel on separate streams; inside one kernel the
gate's GEMM can overlap the down_proj's weight stream (the v4 trick again —
the gate is compute, the down_proj is bandwidth).
* Adds 11.76 us/layer of streaming to hide the gate behind.
* Output: latent `[T,3584]` + topk, both in VMEM if S3 follows.

### S3 — Absorb the expert GMM
The hard stage. `gmm_v2` already does mxfp4 dequant + ragged dispatch; the work
is to make it a *callable region* rather than its own `pallas_call` so S2's
outputs stay in VMEM.
* Preferred approach: refactor `gmm_v2`'s kernel body into an importable inner
  function parameterised by refs, then call it from the mega kernel's grid.
  Do **not** fork it — a divergent copy of the expert GMM is a maintenance trap.
* Blocker to check first: `fused_moe_kernel_0` bitcasts a 4-bit ref inside
  `emit_pipeline` and fails the second-minor-dimension check at a 1-token
  bucket. Confirm the bucket ≥ 4 workaround still holds inside a fused grid.
* Expected: 44.30 us/layer of GMM, plus removing the permute round-trip (6.81).

### S4 — Merge the shared MLP (S0a) in
The shared MLP is independent of the routed path and reads the same `hidden`.
Inside one kernel its 31.5 MiB of weight streaming can overlap the expert GMM's
compute, which is the single largest overlap opportunity in the block.
* This is where the thesis gets its real test: if S4 is not clearly better than
  S3 + separate `shared_mlp`, thesis (4) is wrong.

### S5 — Absorb the collective + norm + up_proj (v4)
Last, because it is the highest-risk: a collective inside a mega kernel means
the whole block becomes a synchronisation point, and the LL all-reduce already
shows 8–68 us per-occurrence spread from peer skew. Fusing more work *behind*
that skew may help (more to overlap) or hurt (skew propagates further).
* Only attempt if S0b closed cleanly.

## 5. Hard problems, called out up front

1. **mxfp4 inside a fused grid.** Dequant needs the scale tensors resident;
   the ragged gather uses SparseCore. Both work today in isolation.
2. **Dynamic routing.** `topk_ids` determine which expert weights are read. A
   hand-driven DMA ring must issue those reads *after* topk resolves, so the
   expert-weight stream cannot be primed at `pid==0` the way `shared_mlp` primes
   phase 2. This directly limits how much fill can be hidden — quantify it early.
3. **Compile specialisation.** The kernel specialises on token count; vLLM's
   `mark_dynamic` is an assertion, so `TPU_SOFT_DYNAMIC_TOKEN_DIM` (which swaps
   in `maybe_mark_dynamic`) is required, **set on both arms** since it changes
   compilation.
4. **VMEM budget.** Rough per-layer resident set if everything is fused: latent
   ring + expert weight ring + shared MLP rings + intermediates. A tpu7x core has
   64 MiB with ~32 MiB scoped. This will not all fit; the plan must schedule
   rings, not just allocate them.
5. **Prefill.** Every stage must fall back for large T — at prefill these ops are
   compute-bound and XLA's fusions are fine. Gate on token count.

## 6. The MSA question, which may dominate all of this

Measured: the per-layer shared-MLP cost today is *bimodal* — 32 layers at
~10.0 us (MSA promoted both weights) and 60 layers at ~27.1 us (streaming). The
Pallas kernel is a flat 14.21. So it **improves 60 layers and regresses 32**.

If MSA simply promoted both weights in all 92 layers we would get ~10.0 us/layer
— better than the kernel. **Fixing the allocator is worth ~1.6x the kernel and
needs no kernel at all.** The lever is `xla_tpu_msa_sort_order_overrides`
(`hlo_position_matcher{instruction_name_regex, size_gte, size_lte}` +
`override_options{assign_first|assign_last|assign_value}`), which overrides MSA's
greedy priority instead of arguing with its cost model.

**Do this experiment before S3.** If MSA can be steered, the value of the mega
kernel drops substantially and the plan should be re-scoped to the ops MSA cannot
help (the collective, the permute round-trips) rather than the weight streams.

## 7. Test and benchmark methodology

Hard-won this session; deviating from it has produced wrong answers repeatedly.

**Numerics.** Gate every stage against a `jnp` reference at bf16. `shared_mlp` is
bit-exact at bn=256/512; treat anything worse than ~1 ulp as a bug until
explained.

**Microbenchmark — always use the slope method.** Timing a jitted chain of DEPTH
ops and dividing by DEPTH charges each op a share of host dispatch (~100 us). At
~25 us/op that squashed four distinct cases into a common 537–886 GiB/s and
inverted a conclusion. Measure at two depths and take
`(T_hi − T_lo)/(DEPTH_hi − DEPTH_lo)`. Also give every chain step its **own**
weight buffer, or XLA hoists the load and reports a physically impossible rate,
and use enough distinct weights to exceed VMEM or MSA promotes them and reports
a fake "fast" number.

**End-to-end — paired, interleaved arms.** Run `ctl, arm, ctl, arm`, never one of
each. Two runs of an identical config have differed by 0.56 ms in this session,
~10x the spread previously assumed, and that is still unexplained.

**Compile-hash trap.** Any new env flag must be declared in
`envs.environment_variables` or the candidate silently loads the baseline's
compiled graph. Two days of A/B numbers were void for exactly this reason.
Adding a key invalidates *every* arm's cache, so re-measure the control.

**Trace analysis — step-exact, never whole-trace totals.** Use the `XLA Modules`
line (`pid=<device>, tid=2`), one event per module execution; its duration is the
step time and tracks bench TPOT. Drop event 0 (warm-up), take the step nearest
the median, window the `XLA Ops` line (`tid=3`) to it. One step vs one step needs
no normalisation. Join on `hlo_stats`' `HLO op name` for op text/category, and
match across arms by (category, result shape ← sorted operand shapes) since HLO
numbering differs. Read `S(1)` on the *operand* for VMEM vs HBM residency.
`scripts/k3/diff_arms.py` compares raw totals across unequal step counts and
double-counts its projection section — do not use it as written.

**Emitter experiments — use the standalone loop.** A small chain reproduces the
model's `window_config` choice exactly, so emitter/flag experiments cost ~1 min
instead of a 17-min model start (`scripts/k3/window_probe.py`).
`xla_tpu_scoped_vmem_limit_kib` is a `LIBTPU_INIT_ARGS` flag, not `XLA_FLAGS`.

**Cluster etiquette.** One `ici-2x2x4` slice and one `dev-2x2x1` node at a time;
queue extra work with `--dependency=afterany:<id> --nodelist=<same node>`.

## 8. Kill criteria

Stop and re-scope if any of these hold:

* S0a's paired A/B does not show a win outside noise → the local-win-to-e2e
  transfer is broken and no amount of further fusing fixes it.
* MSA sort-order overrides recover most of the bimodal loss → re-scope to the
  collective and the permute round-trips only.
* S3 cannot keep the expert GMM's ragged dispatch inside the grid without
  forking `gmm_v2` → stop at S2 + S4 and leave the GMM as its own call.
* Any stage regresses e2e by more than it wins locally, twice in a row → the
  schedule-perturbation cost exceeds the fusion benefit at this graph size.

## 9. Open questions

* Why do 19 of 24 HBM-streamed `gate_up` instances run at 0.60 TB/s while 5 run
  at 1.88 with identical shape, memory space, window config and concurrent
  traffic? Six hypotheses refuted; leading candidate is instruction-memory
  pressure in a 92-layer unrolled program. Unresolved, and it may cap what any
  kernel can achieve in situ.
* Why did two runs of an identical config differ by 0.56 ms?
* Does `scoped_vmem_limit_kib=16384` (−0.21 ms, one flag, no code) replicate?
