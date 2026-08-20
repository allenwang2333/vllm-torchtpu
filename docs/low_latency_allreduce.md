# Low-latency all-reduce on TPU7x (8 devices, 4 chips)

Machine: `tpu-rack1-w-014`, 8 × TPU7x (Ironwood), 2×2×1 chips × 2 cores, single host.
JAX 0.10.2 / libtpu 0.0.44.1 / torch_tpu 0.1.1. All numbers bfloat16, `cols=1024`.

Methodology: two-point slope. Time a chain of 20 then 60 data-dependent
all-reduces in one compiled graph, take `(T60 - T20) / 40`. Every fixed cost
(dispatch, H2D/D2H, host sync) cancels exactly, which matters because those are
~10-400 µs here and would otherwise swamp a few-µs collective. Chaining also
matches how a TP all-reduce appears in decode: on the critical path, one per
layer, nothing to overlap with.

Reproduce:

```bash
python scripts/allreduce_latency_bench.py --devices 2,4,8 --sc-offload-bytes 67108864
```

## 1. XLA's default SparseCore offload threshold is a 3.5x latency cliff

`--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes` defaults to **64 KiB**
on this build. At or above that size XLA offloads the all-reduce to SparseCore,
which costs a flat ~30 µs regardless of payload:

| payload | forced TensorCore | forced SparseCore | default |
|--------:|------------------:|------------------:|--------:|
| 16 KiB  | 10.5 µs | 31.9 µs | 10.2 µs |
| 32 KiB  | 10.6 µs | 31.9 µs | 10.2 µs |
| **64 KiB** | **11.0 µs** | **36.3 µs** | **36.4 µs** ← cliff |
| 128 KiB | 15.2 µs | 37.7 µs | 37.8 µs |
| 1 MiB   | 24.9 µs | 41.1 µs | 41.2 µs |

`env_override.py` already raises this threshold to per-tensor-core VMEM
(64 MiB on `tpu7x`), and I confirmed the prefix match fires here — GCE metadata
reports `tpu7x-8`, which hits the `"tpu7x"` entry in
`_TPU_VMEM_BYTES_PER_TENSOR_CORE`. **So vLLM already avoids this cliff.** It is
worth knowing that the override is load-bearing, not cosmetic: without it, any
all-reduce ≥64 KiB (e.g. 32 tokens × 4096 hidden bf16 = 256 KiB) pays 3.5x.

All comparisons below therefore use TensorCore-forced XLA as the baseline,
since that is what vLLM actually runs.

## 2. Custom Pallas kernels vs XLA, 8 devices

`one_shot` = single all-to-all broadcast round, 1 hop, `M·(N-1)` bytes out/device.
`two_shot` = direct (non-ring) reduce-scatter + all-gather, 2 hops, `2M(N-1)/N` out/device.
XLA's default is a *ring* RS+AG: bandwidth-optimal but `2(N-1)` = 14 sequential hops.

One consistent sweep, one shape per process, bfloat16, `cols=1024`. `–` = not
applicable (`two_shot*` scatter along rows, so they need `rows % 8 == 0`) or
past the point where that kernel is worth measuring.

| payload | XLA | one_shot_ws | two_shot | **two_shot_ws** | best vs XLA |
|--------:|----:|------------:|---------:|----------------:|------------:|
| 2 KiB   | 9.09 | **4.27** | – | – | 2.13x |
| 4 KiB   | 9.29 | **4.18** | – | – | 2.22x |
| 8 KiB   | 9.08 | **3.62** | – | – | 2.51x |
| 16 KiB  | 10.34 | **4.85** | 9.40 | 7.39 | 2.13x |
| 32 KiB  | 10.47 | **5.86** | 9.60 | 7.50 | 1.79x |
| 64 KiB  | 10.64 | 8.72 | 9.88 | **8.14** | 1.31x |
| 128 KiB | 15.26 | 13.63 | 10.61 | **8.85** | 1.72x |
| 256 KiB | 15.84 | 24.48 | 13.32 | **11.28** | 1.40x |
| 512 KiB | 20.15 | 45.69 | 18.53 | **16.54** | 1.22x |
| 1 MiB   | **25.00** | 91.39 | 28.36 | 26.85 | 0.93x |

(Plain `one_shot` is omitted; `one_shot_ws` supersedes it everywhere.)

**The workspace treatment is worth ~2 µs to two-shot as well** — 9.40 → 7.39,
10.61 → 8.85, 13.32 → 11.28, 18.53 → 16.54. Exactly the barrier-overlap saving
predicted from two-shot being synchronization-bound rather than
bandwidth-bound, and it holds across the whole range.

Crossover is ~64 KiB: `one_shot_ws` below, `two_shot_ws` from there to ~512 KiB,
XLA's ring above ~1 MiB.

At 2 and 4 devices the win is smaller (XLA's ring is only 2 and 6 hops), but
one-shot still leads at small sizes — 1.4-2.1x at 2 devices, 1.2-1.6x at 4.

## 2b. Scaling to 16 chips / 32 devices

Full `ici-2x2x4` slice: 4 hosts × 4 chips × 2 cores. One consistent run
(job 32, ~4 min wall). `✗` = takes down the slice at 32 devices; `–` = not
applicable (`two_shot` needs `rows % 32 == 0`, which at `cols=1024` puts its
smallest payload at 64 KiB).

| payload | XLA | one_shot_ws | two_shot | **two_shot_ws** | best vs XLA |
|--------:|----:|------------:|---------:|----------------:|------------:|
| 2 KiB   | 15.88 | **8.15** | – | – | 1.95x |
| 4 KiB   | 16.16 | **8.01** | – | – | 2.02x |
| 8 KiB   | 16.36 | **10.08** | – | – | 1.62x |
| 16 KiB  | 17.29 | **15.14** | – | – | 1.14x |
| 32 KiB  | **18.00** | 25.86 | – | – | 0.70x |
| 64 KiB  | 19.76 | 49.65 | 17.59 | **14.20** | 1.39x |
| 128 KiB | 22.26 | 104.30 | 17.09 | **14.25** | 1.56x |
| 256 KiB | 24.18 | 208.32 | 22.64 | **18.68** | 1.29x |
| 512 KiB | 27.92 | 425.89 | 33.54 | 28.54 | 0.98x |
| 1 MiB   | 37.84 | 857.36 | 57.60 | 51.15 | 0.74x |

Both `_ws` kernels work at 32 devices once the exit-time semaphore check is
disabled (§3a). `one_shot_ws` holds the same ~2x peak as at 8 devices, with an
earlier crossover (~8 KiB vs ~64 KiB) for the usual bandwidth reason.
`two_shot_ws` is worth ~3 µs over plain two-shot here and lifts the 64-256 KiB
range from 1.05-1.21x to **1.29-1.56x**.

**The remaining gap is 32 KiB at 32 devices**, where XLA still wins: one-shot is
already bandwidth-bound at 31 peers, and neither two-shot variant can run there.
That second part is an artifact, not physics — both scatter along rows, so they
inherit `rows % N == 0`, which at `cols=1024` and 32 devices means no payload
between 16 KiB and 64 KiB is expressible. Scattering the flattened array instead
would close it, and is much cheaper than another kernel.

Two more things to read off this:

* **one-shot's peak win is stable at ~1.4x**, same as 8 devices — its floor
  (11.1 µs) and XLA's floor (16.3 µs) both grow with device count, but at
  roughly the same rate.
* **Its useful range collapses from ≤32 KiB to ≤8 KiB.** One-shot's egress is
  `M·(N-1)`, so 4x the devices costs 4x the bandwidth at the same payload,
  and the slope above the crossover is brutal — 477 µs at 512 KiB.
* **two_shot is sync-bound here too**: 17.57 → 17.91 µs from 64 to 128 KiB,
  double the payload for +2%. Same conclusion as at 8 devices — its floor is
  hops plus barrier, so the workspace treatment is where its headroom is.

XLA scales well here: 4x the devices costs only 1.8x latency (9.1 → 16.1 µs at
2 KiB), so its lowering is clearly not a naive 2(N-1)-hop ring at this scale.

`two_shot`'s egress is `2M(N-1)/N ≈ 2M`, essentially independent of N, so it
holds a ~17.5 µs floor where one-shot has already exploded. But XLA is only
20-24 µs there, so the win is 1.05-1.21x. Two-shot's value at 32 devices today
is that it *bounds the downside* above the one-shot crossover, not that it is
fast -- which is exactly why giving it the workspace treatment is the
interesting move.

Caveat: sub-meshes (8 or 16 of the 32) were **not** measured in the multi-host
job. Every process must participate in each jitted collective, so a mesh over a
device subset risks a hang; the 8-device numbers above come from a single-host
run instead.

## 3. Remaining headroom

Both barriers in the kernel are pure overhead. Dropping the second
(re-entrancy) barrier — unsafe as shipped, diagnostic only via `oneshot_sb`:

| payload | XLA | one_shot | one_shot, single barrier |
|--------:|----:|---------:|-------------------------:|
| 2 KiB  | 9.05 | 6.45 | **4.95** (1.83x) |
| 8 KiB  | 8.68 | 6.27 | **4.88** (1.78x) |
| 32 KiB | 10.52 | 7.58 | **6.73** (1.56x) |

So ~1.4 µs of the 6.4 µs floor is the second barrier. `scripts/allreduce_breakdown.py`
strips the kernel down further — 8 devices, bfloat16, µs:

| payload | copy (launch) | bar1 | bar2 | dma1bar | full |
|--------:|--------------:|-----:|-----:|--------:|-----:|
| 2 KiB   | – | 1.54 | 3.01 | 4.90 | 6.28 |
| 8 KiB   | 1.28 | 1.77 | 3.37 | 4.48 | 6.20 |
| 32 KiB  | – | 1.91 | 2.87 | 6.67 | 8.21 |
| 128 KiB | 0.20 | 1.68 | 3.03 | 14.69 | 16.29 |

At 2 KiB the 6.28 µs splits roughly as **1.5 µs barrier #1 (incl. launch) +
1.5 µs barrier #2 + 3.4 µs DMA+reduce**. So **about half the floor is barrier
overhead, not communication.**

Both barriers are removable in principle:

* **Barrier #2 (~1.5 µs)** — double-buffered scratch keyed on invocation
  parity, so a peer's receive is itself permission to send again.
* **Barrier #1 (~1.5 µs)** — the bigger structural win. Today it is a full
  round trip *before any data moves*. Replacing it with per-peer capacity
  semaphores lets the DMAs start immediately and overlaps the synchronization
  with the transfer instead of serializing them.

That puts a realistic floor near **3.4-3.6 µs, or ~2.6x over XLA's 9.1 µs**, up
from 1.4x today. Both need invocation-parity state carried across calls, so this
is real work, not a flag flip.

### Barrier #1 removed: `one_shot_all_reduce_ws` (~2.2x over XLA)

Measured on 8 devices, bfloat16, one shape per process:

| payload | XLA | `one_shot` | `one_shot_ws` | speedup |
|--------:|----:|-----------:|--------------:|--------:|
| 2 KiB   | 9.92 | 6.83 | **3.90** | 2.54x |
| 4 KiB   | 9.22 | 5.97 | **4.21** | 2.19x |
| 8 KiB   | 9.14 | 6.34 | **4.20** | 2.18x |
| 16 KiB  | 10.12 | 6.80 | **4.81** | 2.10x |
| 32 KiB  | 10.43 | 8.26 | **6.59** | 1.58x |
| 64 KiB  | 11.07 | 10.91 | **9.02** | 1.23x |

3.90 µs at 2 KiB lands on the 3.4-3.6 µs floor the decomposition predicted.

**How.** Barrier #1's real job is not buffer reuse — it guarantees the
destination is *owned by this kernel on the receiver* when the data lands.
Kernel scratch is live only while the kernel runs, so a write that arrives
early scribbles on VMEM XLA has given to another op. The fix is to make the
destination a caller-owned workspace threaded through every call: it stays live
between kernels (XLA cannot reuse a value that is still needed), so an early
write is harmless and the barrier can move *after* the sends, where it overlaps
the transfer. Three things must be double-buffered together:

* **data slots** — round `k` writes `ws[k%2]`, last read by round `k-2`;
  passing round `k-1`'s barrier proves peers finished round `k-2`'s reduce;
* **DMA semaphores** — otherwise a round-`k` send satisfies a peer's pending
  round `k-1` byte-count wait early;
* **the barrier must stay a *double* barrier** — double-buffering the data does
  not license a single one, since a fast peer can signal two rounds' arrivals.

Round parity is carried at runtime in an aliased `step` counter, so a call site
that repeats across decode steps works without trace-time constants.

**Validation.** `scripts/allreduce_ws_stress.py`: 700 executions across 7 shapes
× 5 chain lengths (1-64 rounds), values checked against `psum` every time, zero
failures. The chain uses a per-round twist rather than a plain mean-of-means, so
a dropped, doubled or stale round shows up as a value mismatch. In float32 a
200-round chain matches to 1.5e-7; the ~0.02 bfloat16 deltas are accumulation
drift, not corruption (confirmed by the float32 run).

**Known limit.** Validated to 64 KiB per shard; **128 KiB takes down the slice**
(`SLICE_FAILURE_SW_INJECT_ERROR`), diagnosis pending. Not currently blocking:
one-shot is bandwidth-bound well before that and `two_shot` already wins from
~64 KiB, so the limit sits outside the range where this kernel is the right
choice. It does mean the dispatch gate is mandatory, not optional.

### 3a. Root cause: Mosaic's exit-time semaphore check

The slice failures had a single root cause, found in the TPU driver log the
`SLICE_FAILURE_SW_INJECT_ERROR` message points at (`/tmp/tpu_logs_<user>/`,
which is where to look first next time -- the libtpu message itself is generic):

```
Node 1 halted unexpectedly at tag:pc TensorCoreSequencer:1:0x584:
Semaphore (scratch argument 3) has a nonzero value upon exit from a Mosaic
kernel. Make sure every DMA is awaited, and every semaphore signal is paired
with a wait.
```

Scratch argument 3 is `recv_sem`. **It is not a data race.** With the barrier
*after* the sends, a peer that has passed round `k`'s barrier issues its round
`k+1` sends *before* its own round `k+1` barrier. Those DMAs land in our
`ws[(k+1)%2]` and bump `recv_sem[(k+1)%2]` while we are still finishing round
`k`. The data is fine -- that is the other slot, which is exactly what double
buffering is for -- but Mosaic asserts every scratch semaphore is zero on
kernel exit, and that one legitimately is not. Double-buffering the semaphores
did not fix it; it only moved which index is nonzero.

This accounts for every observation:

* **Nondeterminism at 32 devices** -- it depends on whether a peer's next-round
  DMA beats our exit, so more devices and more cross-host dispatch skew make it
  more likely.
* **Byte sensitivity at 8 devices** (784 KiB fine, 896 KiB fatal) -- larger
  transfers widen the window.
* **`barrier_first` always clean** -- no peer can send round `k+1` until
  everyone has entered it, so nothing arrives early.

**Fix:** `disable_semaphore_checks=True` in `CompilerParams`, now the default
for this kernel. The early counts are not lost -- they carry over and are
consumed by the next round's wait, verified exact against `psum` in float32:
160 executions at 8 devices (including the previous reliable crasher at 64-round
chains) and 280 at 32 devices, all to ~1e-6. The device cap and in-flight byte
cap are gone; both were symptoms of this one bug.

`barrier_first` remains as an escape hatch and as the control that isolated the
bug. Note it must stay possible to *reproduce* the failure --
`--enable-sem-checks` does that.

**Process note.** Two separate crashes came from the same argparse trap:
`action="store_true"` defaults to `False`, so a flag meant to say "leave the
default alone" instead actively forced the unsafe setting. Both flags now
default to `None`.

### The failed attempt that got here (kept for the record)

First attempt (`one_shot_all_reduce_db`, since deleted) double-buffered the recv
slots in *kernel scratch* and moved the barrier after the sends. Measured on
8 devices, 2 KiB:

| variant | latency | vs XLA |
|---|---:|---:|
| `one_shot` (shipped) | 6.3-6.6 µs | 1.4x |
| barrier after sends, double barrier | **4.25 µs** | 1.91x |
| barrier after sends, single barrier | **3.53 µs** | 2.33x |

Neither variant was correct — both eventually killed the slice or hung. Three
real bugs surfaced, all of which the workspace version above had to fix too:

1. Double-buffering the *data* doesn't license a single barrier — a fast peer
   can signal two rounds' arrivals, so "I collected N-1 signals" stops meaning
   "everyone arrived".
2. Round 0 issued a leading *and* trailing barrier while later rounds issued
   one, so barrier-semaphore counts drifted and later rounds ate stale signals.
3. The *DMA semaphores* are shared across rounds too: a round-`k` send can land
   while a peer drains round `k-1`'s byte-count waits and satisfy them early.

It still failed after all three, because kernel scratch is the wrong home for
the destination — hence the workspace design above, which is the same
symmetric-memory idea as the paper's `ncclLLBuffer`.

One methodology note worth carrying forward: a mid-bisect hypothesis that the
crash was "multi-shape" was **wrong**. Python block-buffers stdout when
redirected to a file, so results printed before an abort were lost and the crash
looked like it happened one config earlier than it did. The real trigger was
payload size. Run these with `python -u`.

Caveat on the attribution: the `copy` variant's slope fell below noise at 2 and
32 KiB (shown `-`), so launch is somewhere in 0.2-1.3 µs and the split between
"launch" and "barrier #1" is soft. The `bar1`/`bar2`/`dma1bar`/`full` columns
are directly measured and solid; only their *decomposition* carries that slack.

Note also there is no barrier-free variant here, deliberately: removing every
barrier lets a peer's DMA land after the receiving kernel exits, which corrupts
semaphore state and kills the slice (`SLICE_FAILURE_SW_INJECT_ERROR`). One
barrier is the hard floor for a stateless kernel.

## 4. torchtpu's all-reduce carries large per-collective overhead

Measured through the stack vLLM actually uses (`tpu_dist` functional collectives
under `torch.compile(backend="tpu")`, 8 processes), same slope methodology:

| payload | torchtpu | JAX/XLA same collective |
|--------:|---------:|------------------------:|
| 2 KiB   | 139.6 µs | 9.1 µs |
| 8 KiB   | 139.1 µs | 9.2 µs |
| 32 KiB  | 140.4 µs | 10.8 µs |
| 128 KiB | 139.7 µs | 15.2 µs |

Controls run:

* Compute-only chain (`--op noop`): T20 = T60 = 0.38 ms, ~0 µs/op. So the
  harness's fixed cost genuinely cancels and the 139 µs is per-all-reduce.
* Absolute times are linear: T20 = 2.97 ms, T60 = 8.54 ms.
* With a matmul interleaved before each collective (`--interleave-matmul`),
  still ~130 µs/op.

**The cost is flat from 8 KiB to 128 KiB.** No collective is flat in payload
size, so this is not wire time — it is per-collective overhead ~15x larger than
the collective itself.

Likely mechanism, *not yet confirmed*: `torch_tpu/_internal/compile/fx_passes/
force_collectives_output.py` promotes every collective and `wait_tensor` to a
submodule output, so a graph of N collectives becomes N separately dispatched
executables (`_SplitCompiledExecutable`). A host round-trip per collective is
consistent with a flat, size-independent ~139 µs.

Caveat worth resolving before acting: at 139 µs × ~60 layers a decode step would
be ~8 ms of pure all-reduce, which does not match observed vLLM decode times —
so either production graphs avoid the per-collective split, or the host pipelines
the dispatches in a way this microbenchmark's blocking chain prevents. **The next
step is a profile of a real decode step** (e.g. the traces in
`k3_pruned_profiles/`) to see the actual device-side collective duration and the
gaps between them.

## 5. Barrier-free steady state (`steady_barrier=False`)

§3 said "there is no barrier-free variant here, deliberately ... one barrier is
the hard floor for a **stateless** kernel." That qualifier was load-bearing and
has since stopped applying: the `_ws` kernels are not stateless, so a late DMA
lands in a live caller-owned slot rather than in memory the kernel no longer
owns. `steady_barrier=False` removes the barrier from every round except round 0.

**Why it is safe.** This is the TPU analogue of the barrier-free (Lamport)
collectives used on NVLink, but only at the level of the idea -- the mechanism
does not port. There the data word doubles as the ready flag so the consumer
polls it, skipping a flag round-trip. On TPU there is nothing to poll: transfers
are DMA, not core-issued stores to peer memory, and the recv semaphore is
already delivered by the same hardware transaction as the data. Arrival
detection never cost us anything. What the barrier buys is the *buffer-reuse*
guarantee -- exactly the job Lamport's rotating buffers do -- and the data
dependency supplies it:

* Phase 1 is all-to-all, so completing round `k-1` required receiving from every
  peer. Being in round `k` proves every peer entered round `k-1`, which (`ws` and
  `step` are aliased, so rounds are strictly ordered per device) proves every
  peer completed round `k-2` -- including its reads of slot `(k-2)%2 == k%2`,
  the slot we are about to write.
* Skew is bounded at one round symmetrically: peer `j` cannot enter round `k+1`
  until our round-`k` sends land, so while we are in round `k`, `j` is at most in
  `k+1` and touches the *other* slot. Two slots suffice; no third is needed.

Round 0 keeps its barrier and must -- nothing has synchronized yet, the workspace
may not be allocated on every peer, and the `collective_id` may not be
registered.

**Correctness.** `scripts/allreduce_ws_stress.py --no-steady-barrier`, float32
oracle, non-idempotent twist chain, rounds 1-200, 20 repeats:

| devices | hosts | arms | result |
|--------:|------:|-----:|--------|
| 8 | 1 | 4 | 360 exec each, 0 failing |
| 32 | 4 | 4 | 360 exec each, 0 failing (job 254) |

Errors sit at the float32 floor and are *bit-identical* between control and
barrier-free within each kernel, which is the signature to expect: removing the
barrier changes synchronization, not arithmetic. The two kernels do differ from
each other, confirming the switch is live rather than the flag inert.

**What it is worth: ~1.5 us, flat.** Measured at the shapes production actually
runs. `_build_two_shot` reshapes the flattened payload to `(N, numel/N)`, so K3's
`(1,7168)` post-attention reduce is `32x224` and its `(1,10752)` latent-MoE pair
is `32x336`; reach them with `LL_SHAPES`. Two independent runs (jobs 256/257),
us:

| shape | KiB | site | xla | `twoshot_ws` | `twoshot_ws_nb` |
|-------|----:|------|----:|-------------:|----------------:|
| 32x224 | 14 | post-attention | 17.48 / 17.11 | 12.73 / 12.25 | **10.66 / 10.92** |
| 32x336 | 21 | latent-MoE pair | 17.99 / 17.50 | 12.79 / 12.45 | **11.76 / 10.96** |
| 32x448 | 28 | -- | 18.06 / 18.03 | 12.62 / 13.14 | **11.63 / 11.07** |

Read the *levels*, not the per-run deltas: `twoshot_ws` is 12.6 +/- 0.4 and
`twoshot_ws_nb` is 11.0 +/- 0.5, so the saving is ~1.5 us (12%) and flat across
14-28 KiB. Run-to-run variation is comparable to the effect -- a 2.07 outlier
appeared at 14 KiB in job 256 and at 28 KiB in job 257 -- so single-run deltas
mean nothing here. Against XLA the post-attention kernel goes 1.39x -> 1.59x.

Flatness is the mechanism: the barrier costs 62 semaphore signals plus two waits,
a fixed message count independent of payload.

**Do not read 2.93 us from §3a's ablation as the available win.** That figure
comes from *adding* a barrier to a bare `noop` rung -- a serial position -- while
the shipping `_ws` kernels already overlap theirs. The same ablation's `full_bf`
rung (16.77 against `full`'s 12.65) shows the overlap already recovers 4.12 of
the 5.85 us serial double-barrier cost, leaving ~1.7 us exposed. And not even all
of that returns, because the barrier and the recv wait block on the same event --
peer arrival. Removing the barrier removes its bookkeeping, not the waiting.

**Not enabled by default, and why the correctness result is not sufficient.**
The failure mode is a silent stale read, and soundness requires every rank to
call the collective the same number of times in the same order -- speculative
decode, early exit, a `_dummy_run` firing on a subset of ranks, or a
non-unanimous bucket switch all break it. Worse, the stress test cannot see the
case that matters: every rank runs identical work in lockstep, so there is almost
no arrival skew, and skew is precisely what the barrier absorbs. Under the MoE
call site's skew -- the suspected cause of the pair kernel's 12.98 -> 25.16 us
inflation in production -- the exposed portion could be much larger, in either
direction. Testing that needs a rank-dependent delay, and it has to be a
side-effecting Pallas spin kernel: multiply-by-zero and `s*1.0` loops get folded
away, static per-process variation is impossible under one SPMD program, and
anything that perturbs values breaks the oracle.

### 5a. End-to-end: 15.22 -> 14.88 ms, -2.2% (jobs 286/287)

A/B on four pinned nodes, back to back, one bit different in the config, separate
XLA cache trees, the same measurement script driven from the launcher's
POST_READY_CMD hook so the arms cannot drift in what they ran. Coherent canary on
every check, including under barrier-free -- the first end-to-end correctness
signal for a collective with no synchronization primitive in its steady state.

| arm | reps | mean | spread |
|-----|------|-----:|-------:|
| control (barrier on) | 15.20 / 15.18 / 15.28 | 15.22 | 0.10 |
| barrier-free | 14.88 / 14.88 / 14.88 | **14.88** | **0.00** |

**-0.34 ms, -2.23%**, about 7x the control's standard error.

Per site, from the traces (`scripts/debug/llar_chip_skew.py`, 16 chips x 92/95
layers, us):

| site | control | barrier-free | delta | predicted |
|------|--------:|-------------:|------:|----------:|
| post-attention | 12.46 | **10.74** | **-1.72** | -1.70 |
| MoE pair | 24.64 | **23.07** | **-1.57** | ~0 |

`95 x 1.72 + 92 x 1.57 = 307 us` against a measured 340 us. The accounting
closes.

**The MoE prediction was wrong, and the decomposition says why.** Splitting each
site into work and skew:

| | work (ctl -> free) | skew (ctl -> free) |
|---|---|---|
| post-attention | 12.49 -> 10.79 | 1.01 -> 0.40 |
| MoE pair | 12.98 -> 11.36 | 11.66 -> 11.71 |

The MoE site's skew is unchanged, as §6 argues it must be. But its *work* fell by
1.57 us, essentially the microbenchmark's 1.62. **The two terms are additive, not
overlapping.** The mistaken reasoning was that because the barrier waits on the
same event as the recv wait, its cost would be swallowed by the skew; that
conflates the barrier's *wait* (genuinely redundant) with its *message traffic* --
62 semaphore signals plus two waits, which is work on the critical path whether
or not the rank then waits for data.

So the simple statement is the correct one: **barrier-free removes ~1.6 us of work
from every collective at every site, independent of skew.**

Two corroborating observations. Uniformity improved everywhere -- per-chip spread
0.41 -> 0.19 at post-attention and 3.11 -> 1.74 at the MoE site, per-layer skew
1.01 -> 0.40 and 14.61 -> 13.66 -- which is also why the treatment's TPOT reps
came out identical to two decimals while the control's spread 0.10. And the
post-attention treatment lands at 10.74 against the microbenchmark's 10.79, so
the microbenchmark predicts production for this kernel in *both* barrier modes.

Caveat: one control and one treatment, so day-scale drift is not excluded, though
back-to-back scheduling on pinned nodes makes it unlikely.

**Still not enabled by default**, for the reasons above: the failure mode is a
silent stale read, and soundness needs every rank calling the collective the same
number of times in the same order. The skew-injection test matters more now, not
less, because the change is worth 2.2%.

**Two configuration traps this run exposed**, both of which cost a slice:

* `TPU_LOW_LATENCY_ALLREDUCE_SHAPE` is what puts K3's post-attention reduce on the
  kernel. Without it the dispatcher takes the *smallest* payload that takes a
  kernel -- the 7 KiB `(1,3584)` latent reduce -- leaving the 14 KiB `(1,7168)`
  on XLA at 17.1 us against two_shot's 12.5. That is ~433 us/step over 95 layers,
  and it fully explains a 15.73-vs-15.31 discrepancy first misattributed to node
  variation. Four samples at 15.68-15.81 with the wrong wiring against three at
  15.18-15.28 with the right one.
* The value is `ROWSxCOLS`, not a bare width. `7168` parses to a 1-tuple, matches
  nothing, and falls back -- warning in the server log only. Use `1x7168`.

The launcher now tags the XLA cache per flag and refuses to measure when the
dispatcher rejected the override.

At ~1.5 us across K3's 187 collectives/step this is ~281 us predicted; measured
end to end it is 340 us, **2.2% of TPOT**.

## 6. The slice is not flat, and the kernels do not know it

`scripts/debug/probe_pairwise_latency.py` measures one bidirectional exchange
between every pair of devices at 2 KiB. On `ici-2x2x4` TPU7x (16 chips x 2 cores),
496 isolated pairs, jobs 248/250/251:

| component | us | how |
|-----------|---:|-----|
| kernel invocation -- launch, plumbing, local HBM->VMEM stage, no network | **2.59** | `idle` mode |
| + on-package DMA and its semaphore | +0.94 | same-chip 3.53 - idle |
| + leaving the chip (first ICI hop) | +1.15 | manhattan-1 0/1 4.68 - same-chip |
| + each additional hop | +0.455 | slope over manhattan 1..5 |
| + each core-0 endpoint | +0.45 | class 0/1, 0/0 vs 1/1 |

Full range 3.44 us (on-chip) to 6.94 us (5 hops, both endpoints core 0), a 2.02x
spread. `fanout` reproduced across three jobs with per-tier medians within
0.03 us.

Three things fall out:

* **The length-4 dimension is a line, not a ring.** `(0,0,0)->(0,0,3)` lands in
  the manhattan-3 tier, so there is no torus wraparound and the diameter is 5.
* **Invocation costs 5.7x a hop** (2.59 vs 0.455). Any algorithm that spends a
  *kernel* to save hops must win back 5.7 hops and the diameter is only 5. This
  is the first direct evidence for the flat one-shot design, which until now
  rested on hop-count arithmetic; it also reframes why XLA loses at decode sizes
  -- not long hops, but having phases at all.
* **The core penalty is additive per endpoint, not a max.** The exchange is
  bidirectional, so if addressing core 0 were simply slower the measurement would
  be the max of the two directions and 0/0 would match 0/1. It does not:

  | class | manh 1 | 2 | 3 | 4 | 5 | vs 1/1 |
  |-------|-------:|--:|--:|--:|--:|-------:|
  | 1/1 | 4.23 | 4.68 | 5.18 | 5.60 | 6.02 | -- |
  | 0/1 | 4.68 | 5.15 | 5.59 | 6.05 | 6.49 | +0.45 |
  | 0/0 | 5.12 | 5.58 | 6.04 | 6.49 | 6.94 | +0.89 |

  Additive means the two directions partly serialize, so the absolute figures
  include per-endpoint serialization and overstate one-way latency; tier and
  class *differences* are differential and unaffected. A unidirectional variant
  would separate a source-core effect from a destination-core one.

### 6a. Hierarchy does not help, and the reason is algebraic

The cost model above says a two-level collective should be built as *one* kernel
with extra DMA phases (~0.9 us each) rather than two kernels (2.59 us each), and
that the egress cut should pay for it. `scripts/debug/probe_hierarchical_dma.py`
tests that by issuing each scheme's exact DMA pattern in one kernel, against flat
controls on the same harness. Every arm computes a real all-reduce and is checked
against `psum`, so a pattern cannot pass by quietly moving less data.

32 devices, us:

| KiB | flat_oneshot | **flat_twoshot** | hier_oneshot | hier_oneshot_noc | hier_twoshot |
|----:|-------------:|-----------------:|-------------:|-----------------:|-------------:|
| 14 | 31.70 | **15.73** | 16.37 | 15.51 | 17.27 |
| 21 | 43.90 | **15.77** | 19.11 | 18.23 | 17.56 |
| 28 | 57.10 | **16.00** | 21.96 | 21.02 | 17.55 |
| 56 | 100.23 | **17.21** | 31.05 | 30.21 | 18.08 |

**Flat two-shot wins everywhere.** `hier_oneshot` beats `flat_oneshot` by 15-69
us, which confirms the egress model works exactly as predicted -- but that was
never the relevant comparison, because the gate already sends these payloads to
two-shot.

Against two-shot, hierarchy is **egress-neutral by identity**, not by a narrow
margin. With `C` chips and `N = 2C` devices:

* flat two-shot: `2M(N-1)/N` = `M(2C-1)/C`
* hier two-shot: `M/2` (phase A) + `2(M/2)(C-1)/C` (phase B) + `M/2` (phase C)
  = `M(1 + (C-1)/C)` = `M(2C-1)/C`

The intra-chip round trip costs precisely what the smaller inter-chip collective
saves, for *any* chip count. At 32 devices both are 1.9375M. So hierarchical
two-shot can only lose by the cost of its extra phases, and the measured penalty
(1.54 / 1.79 / 1.55 / 0.87 us) is exactly that.

Dropping phase C does not rescue it. `hier_oneshot_noc` reaches 15.51 against
flat two-shot's 15.73 -- a tie -- and it is not a usable collective (each core
holds only its own half). Its two-shot equivalent would land near 16.4.

The underlying reason is that two-shot's egress is already within 3% of the `2M`
floor for any two-phase algorithm, so there is nothing left for hierarchy to cut.
Hierarchy pays off only against one-shot's `M(N-1)`, and one-shot is not the
kernel we run at these sizes.

Two things worth keeping from the exercise:

* **Phase C costs 0.77-0.94 us**, matching the pairwise probe's 0.94 us
  on-package DMA. The cost model transfers, which is what makes it usable for
  ruling designs out on paper before implementing them.
* **An all-1/1-class routing does not compose with a split scheme.** If core `c`
  owns half `c`, then core 0 of chip `i` must send half 0 to core 0 of chip `j`,
  so the slow 0/0 legs (+0.89 us) are forced. Buying the far-peer term requires
  idling core 0, which halves the egress cut.

**Prerequisite nobody has noticed.** `allreduce_latency_bench.py` builds its mesh
with `mesh_utils.create_device_mesh`, which returns a topology-optimized order
(`[0,1,2,3,6,7,4,5,...]`), while production's `_get_mesh` uses plain TP-rank
order. This is benign for the current kernels -- both are all-to-all, so the
destination set is identical and only DMA issue order changes -- but it stops
being benign for a hierarchical kernel, where "my on-chip partner" and "one core
per chip" must be real physical relationships. `_get_mesh` would need a 2-D mesh
built from `chip_topology`. A mis-ordered mesh fails *silently*: it still
computes the right answer, just slowly.

**Also worth knowing:** `_GATE` keys on device count, which works only because
count proxies slice geometry on this hardware. The cost model is dominated by
invocation plus diameter, and diameter is a property of slice *shape* -- 32
devices as `4x4x2` would have a different diameter than `2x2x4` and hit the same
gate entry.

## Recommendation

1. **Confirm §4 first.** If torchtpu really adds ~130 µs per collective in
   production graphs, that dwarfs everything else here — a 2x kernel win on a
   9 µs op is irrelevant next to a 139 µs dispatch. If it is a microbenchmark
   artifact, discard it and go to #2.
2. **Ship the `_ws` kernels with a size/device dispatch gate.** Measured best
   choice, bfloat16, `cols=1024`:

   | devices | payload | use | vs XLA |
   |--------:|--------:|-----|-------:|
   | 8  | ≤ 32 KiB | `one_shot_all_reduce_ws` | 1.8-2.5x |
   | 8  | 64 KiB - 512 KiB | `two_shot_all_reduce_ws` | 1.2-1.7x |
   | 8  | > 512 KiB | XLA | — |
   | 32 | ≤ 16 KiB | `one_shot_all_reduce_ws` | 1.1-2.0x |
   | 32 | 64 - 256 KiB | `two_shot_all_reduce_ws` | 1.3-1.6x |
   | 32 | 32 KiB, > 256 KiB | XLA | — |

   The gate is mandatory, not an optimization: off-crossover the one-shot family
   is catastrophically worse (857 µs at 1 MiB / 32 devices).
3. **Cost of adoption.** Each call site needs a workspace and round counter
   threaded through it (`make_workspace` / `make_two_shot_workspace`). On an
   8-device TP decode with ~60 all-reduces/step, `one_shot_ws` saves ~5 µs each,
   ≈ **300 µs/step**.
4. **Cheap follow-up: drop the `rows % N == 0` constraint** by scattering the
   flattened array. At 32 devices it is the only reason nothing beats XLA at
   32 KiB, and it is much less work than another kernel.
5. **Keep the `env_override.py` SparseCore threshold.** Worth 3.5x at ≥64 KiB,
   and the `tpu7x` prefix match is what makes it fire.

## Files

* `src/vllm_torchtpu/kernels/collectives/low_latency_all_reduce.py` — the kernels.
* `scripts/allreduce_latency_bench.py` — JAX benchmark (XLA vs one-shot vs `_ws` vs two-shot).
* `scripts/allreduce_ws_stress.py` — correctness stress for the workspace
  kernels (`--kernel oneshot|twoshot`), float32 oracle. Run this to a clean exit
  before trusting any `_ws` change; it caught every bug in both.
* `scripts/allreduce_breakdown.py` — latency attribution (launch / barrier / DMA).
* `scripts/debug/probe_pairwise_latency.py` — pairwise ICI latency matrix
  (`idle` / `fanout` / `allpairs` / `matrix`). Run `--sweep-n` first on any new
  slice shape: the fixed cost of one blocked dispatch is ~1 ms, so a short chain
  measures noise, and the harness will happily report a figure below the launch
  floor if you let it.
* `scripts/debug/slurm_stress_barrier_free_32.sbatch` — the 2x2 correctness
  matrix (one-shot/two-shot x barriered/barrier-free) at 32 devices.
* `scripts/slurm_allreduce_32.sbatch` — 32-device (16-chip) multi-host run.
* `scripts/torchtpu_allreduce_bench.py` — torchrun benchmark of torchtpu's own all-reduce.
