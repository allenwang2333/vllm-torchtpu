# START HERE — DeepSeek-V4 TPU: the accuracy gap (next phase)

Seed document for the next investigation session. Read this **first and in
full**. It is deliberately short; the rest of `docs/dsv4/` is reference.

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

## 3. First tasks, in order

1. **Re-baseline.** Re-run the per-layer hidden-state comparison against
   `tpu-inference` on an identical prompt+checkpoint. Identify the first layer
   that diverges materially. This is a *measurement*, not a hypothesis — do it
   before theorising.
2. **Audit for the same dtype bug elsewhere.** Grep every site that hands a raw
   `kv_cache` (or any `kv_cache_dtype`-typed tensor) to a kernel that expects
   packed bytes. The attention path is safe because it routes through
   `_as_kernel_cache_view()`; the compressor was not. This bug class is
   invisible until something reads the data back, and one instance was already
   found this way.
3. **Only then** bisect within the first diverging layer (sub-block
   checkpoints), against `tpu-inference` as ground truth.

---

## 4. How to work — rules earned the hard way

The previous session had a **0/7 hit rate on hypotheses and ~100% on
measurements**. Weight your effort accordingly.

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

These kernels are **pure JAX** and run fine on CPU. Nearly every ~8-minute
full-model run last session only refuted a hypothesis a 30-second script could
have refuted. Reusable harness in [`scratch/cpu_repro/`](../../scratch/cpu_repro):

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
