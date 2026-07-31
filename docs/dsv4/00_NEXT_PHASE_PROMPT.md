# START HERE — DeepSeek-V4 TPU: the accuracy gap (next phase)

Seed document for the next investigation session. Read this **first and in
full**. It is deliberately short; the rest of `docs/dsv4/` is reference.

**Applies to whichever model/agent picks this up.** Before reporting anything,
satisfy §2.1 — five non-negotiable checks. They exist because each one
corresponds to a real, expensive mistake already made here.

---

## 1. Where things stand

The catastrophic failure is **fixed**. DSV4-Flash used to emit one correct token
and then collapse to token id `0` forever. Root cause was a dtype bug: the
compressor handed `float8_e4m3fn`-typed cache tensors to kernels that treat them
as raw byte buffers, so `.at[...].set(uint8_record)` performed a **numeric
conversion instead of a bitcast** and corrupted every record it ever wrote.
Fix: `.view(torch.uint8)` before the op. Full write-up:
[`compressor_fp8_dtype_bug.md`](compressor_fp8_dtype_bug.md).

Current output for `"Compare XLA and LLVM in two bullet points."`:

> `Here are two bullet points comparing the two given entities:`
> `**Entity 1: "The quick brown fox jumps over the fox""`
> `Entity 21`

Fluent and grammatical, but **not faithful to the prompt** — it invents
"entities" and recites a pangram instead of comparing XLA and LLVM. That
accuracy gap is the remaining work.

Kernel suite: **27/27 pass**. The 3 `test_mxfp4_*` failures are pre-existing
(`FusedMoEConfig.__init__() missing 'intermediate_size'`, a vLLM-API/test-harness
mismatch), A/B-confirmed via `git stash`.

---

## 2. ⚠️ All prior per-layer divergence numbers are VOID

Every hidden-state / per-layer divergence measurement in
`history_bisection_progress_report.md` was taken while compressed KV was corrupt
across **all** CSA layers. Those numbers describe a machine that no longer
exists.

**Do not reason from them. Re-measure before forming any hypothesis.** The real
divergence may be much smaller, or somewhere else entirely.

---

## 2.1 Non-negotiables — evidence you must produce

Five checks. Each maps to a specific expensive failure from the previous
session. **The user will ask for these; produce them unprompted.** They are
model-agnostic — they protect against confident wrong answers from any agent,
including the one that wrote this document (whose hypothesis hit rate here was
**0 for 7**).

### 1. Show the control before claiming a negative

Never report "layer L is not the culprit" — or any other negative — without a
positive control proving your instrument works.

For the substitution sweep specifically: run
`DSV4_SUBSTITUTE_LAYER=<L> DSV4_SUBSTITUTE_MODE=noise` and show that generation
**visibly degrades**. If injecting noise changes nothing, the hook is not taking
effect and every "no effect" result in the sweep is worthless.

*Why:* a state-cache probe last session returned a clean, convincing `nz=0` for
every window position. It was reading unrelated memory. With no control, that
false negative was indistinguishable from a real finding and nearly became a
published conclusion.

### 2. Report evidence, not conclusions

Every claim ships with the raw numbers that support it — byte counts, the sweep
table, before/after magnitudes — in the same output. "Slot 3 is corrupt" is not
a finding; `nz[nope/rope/scale/pad]=0/5/7/0` beside a healthy `448/128/7/0` and
a never-written `0/0/0/0` is.

*Why:* every genuine finding last session came with a control row in the same
line. Every retracted one came from a summarised conclusion.

### 3. Do not cite pre-fix numbers

Any per-layer divergence figure from `history_*.md` is void (§2). If your
reasoning references one, you skipped §2 and your conclusion is built on a
machine that no longer exists. Re-measure.

*Why:* those numbers were all taken while compressed KV was corrupt across every
CSA layer.

### 4. Confirm the environment before trusting any run

State explicitly, per run batch, that you have:

- cleared **both** compile caches (`default` *and* `tpu_tier2_cache`)
- checked for stray `/dev/vfio` holders
- used an **absolute** `cd` in the command

*Why:* a half-cleared cache silently ran stale code and produced a
bit-identical result that was misread as "the fix is a no-op." Stray workers and
a drifting cwd each killed runs outright — the cwd one three separate times,
including once immediately after it was documented.

### 5. A passing control does not mean the experiment is valid

The control (rule 1) proves your instrument *fires*. It does **not** prove your
instrument measures what you think. Separately check that your intervention
changes **only** the thing you intend.

*Why:* the substitution hook was once written to return
`res = (_new, None, None, None)`. The layer output is a 4-tuple threaded through
every subsequent layer (`hidden_states, residual, post_mix, res_mix`), and
`None` is the *initial* state — so substituting at layer L silently reset the
Hyper-Connections streams for layers L+1..42. The sweep would then measure
"replace layer L's output **and** wipe the HC state", two confounded effects.

The noise control passed the whole time: generation degraded exactly as
expected. A passing control was fully compatible with an uninterpretable
experiment.

**Concrete check for any intervention:** enumerate everything the intervention
touches, and confirm each is either (a) the intended target or (b) restored to
what it would have been. For tuple/stream-carrying returns, preserve every
element you are not deliberately replacing.

---

---

## 3. First tasks, in order

**0. Run the one-run ablation FIRST** (`DSV4_FORCE_SWA_ONLY=1`). See §3.0 — it
can halve the search space in a single run, before any 43-run sweep.

1. **Re-baseline.** Re-run the per-layer hidden-state comparison against
   `tpu-inference` on an identical prompt+checkpoint. Identify the first layer
   that diverges materially. This is a *measurement*, not a hypothesis — do it
   before theorising. Per §4.0, prefer a **scripted sweep** that emits one
   summary table over an interactive read-and-iterate loop; and consider the
   layer-substitution sweep, which localises the culprit by brute force.
2. **Audit for the same dtype bug elsewhere.** Grep every site that hands a raw
   `kv_cache` (or any `kv_cache_dtype`-typed tensor) to a kernel that expects
   packed bytes. The attention path is safe because it routes through
   `_as_kernel_cache_view()`; the compressor was not. This bug class is
   invisible until something reads the data back, and one instance was already
   found this way.
3. **Only then** bisect within the first diverging layer (sub-block
   checkpoints), against `tpu-inference` as ground truth.

---

## 3.0 Static narrowing: the symptom already localises the bug

Do not treat the accuracy gap as a black box. **The failure mode is itself
strong evidence**, and reading it costs nothing.

Current output:

> `Here are two bullet points comparing the two given entities:`
> `**Entity 1: "The quick brown fox jumps over the lazy dog"`

Two independent signals:

- **Fluent, grammatical, well-structured** => the LM head, FFN, MoE, embeddings
  and the overall forward pass are working. This is not "numbers drifted
  everywhere."
- **Generic filler + a pangram** ("the two given entities", "the quick brown
  fox") => the model is generating from priors, *not from the prompt*. It is not
  seeing its context.

That isolates **context retrieval**, which in DSV4 is the sparse-attention path:
CSA (`compress_ratio=4`) + the top-k indexer, and HCA (`compress_ratio=128`).
Dense local SWA is a separate, simpler path.

### RESULT: the ablation has been run — sparse path is EXONERATED

```
baseline  : 'Here are two bullet points comparing the two given entities:
             **Entity 1: "The quick brown fox jumps over the lazy dog"  Entity 2:'
swa_only=1: 'Here are two bullet points comparing the key aspects of the two main
             characters in the novel "The Kite Runner. The novel is about the kite runner.'
```

Disabling CSA/HCA/indexer entirely changed *which* topic is confabulated but not
the failure mode: still fluent, still unfaithful, no crash, no degradation. So
**the sparse-attention path is not the cause**, and the §3.0 hypothesis below
(kept for the record) was wrong.

### Ablation validity: confirmed

`sliding_window = 128` (config.json) and the test prompt is ~15 tokens, so SWA
alone covers the **entire** prompt. The ablation is therefore a fair test: with
the sparse path disabled every token is still attendable via dense local
attention, so losing content cannot be blamed on "no long-range mechanism".
The exoneration holds.

### The sharpened symptom — read this carefully

Both outputs correctly absorb **"compare two things in two bullet points"** and
produce exactly that structure. What is lost is **"XLA" and "LLVM"**
specifically, replaced by a confabulated topic.

So the model is *not* blind to its context — the **instruction survives while
specific content tokens do not**. Since this persists with sparse attention
fully disabled, and SWA alone covers a 15-token prompt, the defect is upstream
of which-tokens-are-retrievable. Candidates worth checking in roughly this
order:

1. **Tokenization / input_ids** — confirm the prompt tokenises identically to
   the reference. `[DEBUG_INPUT_IDS]` already prints these. Cheap, and a
   mismatch here would explain everything.
2. **Embedding lookup** for those specific token ids.
3. **Position encoding / RoPE** — content-position binding rather than content
   itself.
4. **The attention computation** (not the sparse *selection*, which is now
   exonerated).

**Tokenization is already ruled out** (checked, no run needed):
`'Compare'|' X','LA'|' and'|' LL','VM'|' in',' two',' bullet',' points','.'`
= `43651 | 2137 11318 | 305 | 33792 24695 | 295 1234 37561 4365 16`, clean
roundtrip. So candidate 1 above is closed.

Cross-referencing those ids against the generated ones is informative: the model
**copies** `' two'(1234) ' bullet'(37561) ' points'(4365)` verbatim from the
prompt, but never emits `' X'(2137) 'LA'(11318) ' LL'(33792) 'VM'(24695)`. It
retains the *late, generic* prompt tokens and drops the *early, specific* ones.

Two readings fit that equally well, and they are **cleanly discriminable in one
run**:

- **Positional** — early prompt positions are mishandled (RoPE / position
  binding).
- **Salience** — residual numerical error degrades the signal, and rare/specific
  content is the first casualty while high-frequency instruction patterns
  survive.

**The discriminator:** move the content to the end of the prompt, e.g.
`"In two bullet points, compare XLA and LLVM."` If XLA/LLVM now survive, it is
positional. If they are still lost, it is salience — i.e. diffuse numerical
error, which argues for the per-layer sweep rather than a single localised bug.
Run this before the 43-run sweep; it is one run and it changes which
investigation you start.

Note the ablation did its job either way by eliminating a subsystem.

### The one-run discriminator (superseded — already run, see above)

```bash
DSV4_FORCE_SWA_ONLY=1   # forces swa_only on every layer; sparse path disabled
```

- **Output becomes prompt-faithful** -> the bug is in CSA/HCA/indexer. You have
  eliminated most of the model in a single run; go straight to that subsystem.
- **Output stays unfaithful** -> the sparse path is exonerated, and the 43-layer
  sweep (§3.1) is now warranted rather than speculative.

Either way it is one run, and it makes the sweep either unnecessary or
well-targeted.

### Live candidates in that path

Both are undocumented deviations from `tpu-inference`:

1. **The top-k `max_valid_kv` clamp** in `forward_mqa` — vllm-torchtpu masks
   `topk_indices >= (positions+1)//compress_ratio`; the reference passes the
   indices raw. Plausibly a workaround for the *now-fixed* corrupt compressed
   KV, i.e. possibly obsolete and now harmful.
2. **`two_caches_same_buffer` hardcoded** to `not swa_only`, where the reference
   derives it (`main_cache_index == swa_cache_index`).

## 3.1 Concrete recipe: the layer-substitution sweep

This is the recommended **first experiment**, written out so it can be built in
one turn rather than designed from scratch. It localises the culprit layer by
brute force, which is the right trade when runs are cheap and turns are not
(§4.0).

### Idea

For each layer `L` in `0..42`: run generation normally, but **overwrite layer
`L`'s output with the reference's** (from `tpu-inference`) for every token, then
record the generated text. The first `L` whose substitution repairs the output
is where the divergence originates. One script, ~43 runs, one summary table, no
reasoning required.

### Prerequisites — verify BEFORE launching 43 runs

1. **Regenerate or validate the reference dumps.** Existing ones live at
   `scratch/hidden_states/tpu_inference/`. The reference implementation is
   unchanged, so they *should* still be valid — but confirm the prompt,
   checkpoint snapshot, `max_tokens`, and batch shape match the run you are
   about to do. A 43-run sweep built on mismatched dumps is 43 wasted runs and,
   worse, a confident wrong answer. Reference runner:
   `scratch/run_dsv4_reference.py` (separate venv — see `running_and_testing.md`).
2. **Mind the Hyper-Connections axis.** `config.json` has `"hc_mult": 4`, so
   dumped hidden states are `(num_tokens, 4, hidden_size)` and the size-4 axis
   is **4 stream copies, not 4 tokens**. A previous session lost significant
   time to a comparison script that mistook it for a token axis and manufactured
   a fake "embedding corruption" finding. Substitute the full
   `(tokens, 4, hidden)` tensor.
3. **Confirm head-sharding alignment.** `vllm-torchtpu` shards attention heads
   across TP ranks; `tpu-inference` does not. `scratch/compare_sublayer.py`
   already contains the slicing fix — reuse it rather than re-deriving.

### Status: the hook is BUILT and the control PASSES

Already implemented in `models/vllm/deepseek_v4_patch.py`
(`patched_layer_forward`), gated on `DSV4_SUBSTITUTE_LAYER` /
`DSV4_SUBSTITUTE_MODE` / `DSV4_SUBSTITUTE_DIR`, inert when unset. You do not
need to build it — you need to drive it.

**Use `DSV4_SUBSTITUTE_MODE=shuffle` as the control**, not `noise`. Shuffle
permutes the real tensor along the token axis: identical value distribution,
wrong content, so generation degrades while every value stays in-distribution.
`noise` replaces with Gaussian values and — now that the stream tuple is
correctly preserved — propagates far out-of-distribution data through the full
Hyper-Connections machinery, which has been observed to trip a TPU `UserFatal`
(`RuntimeUnexpectedCoreHalt`) in a downstream Pallas kernel. A crash does prove
causal effect, but you cannot compare generated text against it.

Control result (`DSV4_SUBSTITUTE_LAYER=0 DSV4_SUBSTITUTE_MODE=shuffle`) —
this is the one to reproduce:

```
SUBSTITUTE fired: 264
[SUBSTITUTE] layer=0 step=0 mode=shuffle shape=(64, 4, 4096) absmax 20.75 -> 20.75
crash lines: 0
Generated: ',###package#package##packagepackage##package#package...'
```

Values preserved exactly (it is a permutation), no TPU fatal, and generation is
comprehensively destroyed versus the fluent baseline. Causal effect proven, and
you still get comparable text out.

Earlier control result, kept for the record
(`DSV4_SUBSTITUTE_MODE=noise`, before the stream-preservation fix):

```
SUBSTITUTE fired: 136
[SUBSTITUTE] layer=0 step=0 mode=noise shape=(64, 4, 4096) absmax 20.75 -> 15.5
Generated: ', ,系Ş,en,  \n,,,,,, and'      (normal: 'Here are two bullet points comparing...')
```

All three links verified: the hook **fires** on real forwards, it **actually
replaces** the tensor, and it **causally changes** generation. Negatives from
this sweep are therefore trustworthy — but re-run this control if you touch the
hook (§2.1 rule 1).

`shape=(64, 4, 4096)` also confirms the `hc_mult=4` layout empirically: 64
tokens x 4 stream copies x 4096 hidden. Reference dumps share this layout, so
direct substitution should be shape-compatible — the hook will say so loudly if
it is not.

### Implementation notes

- Hook: the per-layer loop in `models/vllm/deepseek_v4_patch.py`, alongside the
  existing `DSV4_DUMP_HIDDEN_STATES` machinery (which already dumps
  `layer{NN}_step{NNN}.npy`, so the write path and naming are established —
  invert it to *read*).
- Gate on a new env var, e.g. `DSV4_SUBSTITUTE_LAYER=<int>` plus
  `DSV4_SUBSTITUTE_DIR=<path>`; inert when unset.
- Driver script: loop `L` over `0..42`, launch a run per `L`, capture generated
  text, emit **one table** `layer -> generated_text_prefix -> repaired?`.
- Run it unattended and read the table once. Do not poll (§4.0).

### Interpreting the result

| outcome | meaning |
|---|---|
| substitution at layer `L` repairs output, `L-1` does not | divergence originates in layer `L`; bisect its sub-blocks next |
| **no** single layer repairs it | the error is distributed/accumulating, not one bad layer — switch to per-layer `cos_sim` profiling to find where it starts growing |
| **every** layer from some point repairs it | consistent with a single origin upstream; take the earliest |
| substitution changes nothing anywhere | the substitution hook itself is not taking effect — verify with a deliberate corruption (substitute noise and confirm output degrades) before trusting any negative result |

That last row is the control, and it is not optional — see §4.4.

### Cheaper variants worth batching alongside

Since runs are cheap, launch these in the same sweep rather than sequentially:

- **Ablations**: disable the compressor entirely; disable SWA; force
  `swa_only` on all layers. Each isolates a subsystem.
- **Prompt sensitivity**: 2–3 different prompts, to distinguish a systematic
  numerical bug from prompt-specific behaviour.
- **`max_tokens` sweep**: confirms whether quality degrades with depth.

---

## 4. How to work — rules earned the hard way

The previous session had a **0/7 hit rate on hypotheses and ~100% on
measurements**. Weight your effort accordingly.

### 4.0 THE BINDING CONSTRAINT: Claude usage, not TPU time

**TPU experiments are cheap and plentiful here. Claude usage is the scarce
resource.** Optimise for *fewer Claude turns*, not fewer runs. This inverts the
usual instinct and should shape every decision below.

- **One mega-instrumented run, not eight iterative ones.** The previous
  session's biggest waste was the *shape* of the loop: add a probe -> run ->
  read -> add another probe -> run. Each cycle costs several turns. Add every
  plausible probe at once and produce a single rich log.
- **Prefer brute-force experimental search over clever reasoning.** Example for
  task 1: rather than reasoning about which layer diverges, run a
  **layer-substitution sweep** — for each of the 43 layers, force that layer's
  output to the reference value and see which run repairs generation. ~43 cheap
  runs identify the culprit directly, with near-zero reasoning. Same for
  ablations: disable the compressor, disable SWA, swap a kernel for the
  reference's. Let the hardware search the space.
- **Script the analysis; don't read logs by hand.** A harness that prints
  `first divergent layer = X, cos_sim profile = [...]` costs one turn to write
  and zero thereafter. Eyeballing logs costs a turn every time.
- **Never poll.** Launch, then check once at the end. Monitor setup and repeated
  log checks are pure cost with no analytical value.
- **Batch questions.** If three things are worth knowing, instrument all three
  before running, even if only one seems likely.

### 4.1 A contradiction in measured data outranks every open hypothesis

If two measurements are mutually inconsistent under your model, **stop and
resolve that first** — it localises a wrong *assumption*, which is worth more
than any new idea.

The previous session detected its key contradiction (`nope` all-zero bytes *with*
scale byte 109) and twice waved it off as "don't over-derive from logged
numbers." That contradiction *was* the answer: a 30-second CPU sweep proved the
combination unreachable, demolishing the "one atomic record write" assumption
that had misdirected everything. Chasing it on first sight would have saved most
of the session.

### 4.2 Can a CPU script answer this?

These kernels are **pure JAX** and run fine on CPU. Still worth using, but note
the reason under 4.0's constraint: CPU scripts are valuable because they iterate
in seconds and so **cost fewer Claude turns**, not because TPU runs are
expensive. Where a question needs the real model, just run it — and instrument
it heavily while you're there. Reusable harness in [`scratch/cpu_repro/`](../../scratch/cpu_repro):

| script | answers |
|---|---|
| `a1_quantize_reachability.py` | which `(q_nonzero, scale_byte)` pairs `quantize_fp8_ue8m0` can produce |
| `a2_record_roundtrip.py` | record assembly + fp8 byte-lane fidelity |
| `a4_state_roundtrip.py` | `pack_state_cache`/`unpack_state_cache` losslessness |
| `a5_store_cpu_vs_tpu.py` | full store path — run under `JAX_PLATFORMS=cpu` **and** `=tpu` to separate math from lowering |

Run as `JAX_PLATFORMS=cpu PYTHONPATH=src python3 scratch/cpu_repro/<f>.py`.
Imitate their structure: each states **up front what result would falsify
what**.

### 4.3 Derive bounds before running anything

Invariants refute hypotheses for free. Two that each killed a leading suspect
last session at zero cost:

- `mla.py` clamps `L = max(L, 1.0)` before `out = acc / L`, so `|out| ≤ |acc|` —
  the attention-sinks epilogue can only ever *shrink* the output.
- RMSNorm divides by `sqrt(mean(x²))`, so `|normed| ≤ √512 · max|w| ≈ 22.6`
  whenever variance is finite and nonzero — `1e29` is unreachable, full stop.

### 4.4 Every probe needs a control in the same output line

The KV probe was trustworthy because healthy slots 0–2 printed alongside the bad
slot 3. The state probe produced a convincing, entirely **false** `nz=0` because
it had no control. A probe without a control is not evidence.

### 4.5 Never hand-derive a paged-cache address

Import the kernel's own helper (`_state_chunk_dims`) and mirror
`unpack_state_cache`'s indexing. `slot_mapping` indexes the *state view*
(`[num_pages, state_block_size, state_dim]`), **not** the raw byte buffer
(`[num_pages, rows, packing, width]`). Getting this wrong silently read the
wrong page and produced a full false conclusion.

### 4.6 Prefer raw bytes and invariants over decoded values

Decoded views misled repeatedly: a `~3/4` nonzero-byte ratio read as "one of
four blocks is dead" was actually a **bf16-in-f32 storage artifact**; the
`1.5845e29` value looked like corruption but is exactly `2^97`, i.e. structure.

### 4.7 When components all pass, interrogate the interface

The bug lived at a **call site**, not in any component. `kv_score`, the gather,
`compress_norm_rope_store` (byte-identical on CPU *and* TPU), and the
pack/unpack round-trip all passed in isolation — because each test allocated its
own `uint8` array. When every piece passes but integration fails, stop testing
pieces and interrogate what crosses the boundary: **dtype, aliasing, identity,
lifetime**.

### 4.8 Validate end-to-end, not just tests

A lint "cleanup" last session passed syntax checks *and* the 27/27 kernel suite
while leaving the model unable to start (`NameError` on every worker, engine core
dead). Run the actual model after any change, including cleanups.

Corollary: avoid gratuitous changes. That break came from fixing a cosmetic
`E741` in unrelated prior-session code — risk taken for zero benefit.

---

## 5. Operational gotchas (each cost real time)

- **Clear BOTH compile caches.** `/dev/shm/torch_tpu_cache/` holds `default`
  *and* `tpu_tier2_cache` (~4.7 GB). Older docs mention only the first.
  A bit-identical result after a real edit means **stale code ran**, not a no-op.

  ```bash
  rm -rf /dev/shm/torch_tpu_cache/default/* /dev/shm/torch_tpu_cache/tpu_tier2_cache/*
  ```

- **Never launch with `nohup … & disown` from a tool-driven shell** — workers die
  mid-load and leave stray `/dev/vfio` holders. Run in the foreground of a
  backgrounded task. Check before every run:

```bash
  for i in 0 1 2 3 4 5 6 7; do sudo fuser /dev/vfio/$i; done
  ```

- **`scratch/` is ~255 GB and is now gitignored.** Never `git add .` here; the
CPU harness is force-added.
- **Use an absolute `cd`** in run commands — a drifting shell cwd silently killed
  one run at startup.

### Repro command

```bash
cd /home/patemotter_google_com/workspace/vllm-torchtpu
source /home/patemotter_google_com/uv_venv/bin/activate
PYTHONPATH=/home/patemotter_google_com/workspace/vllm-torchtpu/src \
DSV4_MODEL_PATH=/home/patemotter_google_com/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1 \
DSV4_PROMPT="Compare XLA and LLVM in two bullet points." \
VLLM_ENFORCE_EAGER=1 GPU_MEMORY_UTILIZATION=0.40 DSV4_MAX_NUM_SEQS=8 \
DSV4_NUM_GPU_BLOCKS=512 DSV4_MAX_TOKENS=32 \
python examples/run_dsv4_inference.py
```

Debug flags (all inert by default): `DSV4_DUMP_HIDDEN_STATES=1` (the tool for
task 1), `DSV4_DEBUG_MAIN_KV=1` (read-side KV dequant with a control row),
`DSV4_DEBUG_COMPRESSOR=1`.

Kernel suite (fast gate):

```bash
PYTHONPATH=src python3 -m pytest -q --use-tpu \
  tests/kernels/test_dsv4_mla.py tests/kernels/test_dsv4_mla_swa.py \
  tests/kernels/test_dsv4_compressor.py tests/kernels/test_dsv4_moe_routing.py \
  tests/kernels/test_dsv4_streamindex_topk.py
```

---

## 6. Do not re-chase these

Refuted with evidence. Details in
[`compressor_fp8_dtype_bug.md`](compressor_fp8_dtype_bug.md) §5.4.

| hypothesis | why it's dead |
|---|---|
| attention-sinks epilogue `exp()` | `L ≥ 1` clamp ⇒ `\|out\| ≤ \|acc\|`; it can only shrink the output |
| `pl.debug_print` needs unblocking | never needed; still non-functional, still uninvestigated |
| `tp_size` double-division | real but **inert** — per-worker mesh is single-chip, `tp_size == 1` |
| second writer / SWA overlay corrupting rows | value already present in the compressor op's own return |
| `interleaved_rope` allocate-then-scatter | reverted to reference; output bit-identical → true no-op |
| `unpack_state_cache` `ndim<4` fallback | never fires; buffer is 5-D, squeezed to 4-D |
| state not persisting across calls | refuted; positions 12–14 read back populated |
| "one of four head-blocks is zero" | the ratio is a bf16-in-f32 artifact |
| HCA `compress_ratio` clamp (`min(comp_ratio, block_size)`) | real deviation from the reference, but **inert here**: a live run declares 1/4/**128** unclamped, so `block_size` >= 128. Would bite only at `block_size < 128`. |

---

## 7. Known deviations from the reference, deliberately left in place

Worth revisiting now that records are real. Standing rule: **where there's no
documented reason, default to `tpu-inference`'s form.**

- `mla.py`'s `L = jnp.maximum(L, 1.0)` — not in the reference (benign, but an addition).
- `unpack_state_cache`'s `ndim < 4` silent-zeros fallback — no reference equivalent.
- `two_caches_same_buffer` hardcoded to `not swa_only` — the reference *derives* it
  (`main_cache_index == swa_cache_index`).
- E8M0 scale bytes are built by hand (libtpu lacks `f8E8M0FNU`); now saturating
  rather than wrapping mod 256.

---

## 8. Map of `docs/dsv4/`

| file | what it is |
|---|---|
| `00_NEXT_PHASE_PROMPT.md` | **this file** — start here |
| `compressor_fp8_dtype_bug.md` | the solved bug; authoritative post-mortem |
| `architecture_and_rationale.md` | DSV4 design: the *why* |
| `running_and_testing.md` | how to run, env vars, docker |
| `tpu_inference_comparison_design.md` | reference-comparison harness design |
| `pr_splitting_plan.md` | how this work is intended to land |
| `history_*.md` | **superseded session logs.** Useful for measurements and for what was refuted; their *framing* and all per-layer numbers are void. Do not seed a session with these. |
