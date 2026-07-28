# How compilation caching works in vllm-torchtpu

Everything below was read out of source: `vllm-torchtpu` (this repo), `../vllm`,
and `google-pytorch/torch_tpu` + `google-pytorch/torchtpu-vllm` via the GitHub API.
No TPU was needed — this is all static reading.

Written for someone new to TPUs, new to inference serving, and new to PyTorch
internals. Skim Part 0 first; every later section leans on it.

---

## Part 0 — Vocabulary (read me first)

**The big picture in one paragraph.** A PyTorch model is ordinary Python code.
Python is far too slow to run a large language model directly, so before serving
we *compile* the model: translate the Python into a single fused program of TPU
machine code. That translation takes minutes. Caching exists so you pay those
minutes once ("cold start") and every later server launch reuses the saved
result ("warm start"). Everything in this document is about *what exactly gets
saved, where, and by whom*.

| Term | Plain-language meaning |
|---|---|
| **Inference / serving** | Running a trained model to answer requests (vs. training it). vLLM is a server that does this efficiently for LLMs. |
| **`torch.compile`** | PyTorch's "make my model fast" button. Under the hood it runs several stages, each named below. |
| **TorchDynamo ("Dynamo")** | Stage 1 of `torch.compile`. Watches your Python function execute and records the tensor operations it performs, producing a graph instead of running line-by-line Python. |
| **FX graph** | The recording Dynamo produces: a flowchart of math operations ("matmul → add → softmax → …") as a Python data structure. Still hardware-independent. |
| **Guards** | Conditions Dynamo attaches to a recording, e.g. "this is only valid if the input is a bf16 tensor of width 4096." Checked on every call; if a guard fails, Dynamo re-records. |
| **`aot_autograd`** | A PyTorch layer that takes an FX graph and normalizes it (splits out gradients — irrelevant for inference — and breaks composite ops into primitive ones) before handing it to a backend compiler. |
| **Backend** | The thing that turns the normalized graph into machine code. On GPUs it's "Inductor"; here it's `torch_tpu`'s `TpuBackend`. |
| **XLA** | Google's compiler for TPUs (and other accelerators). Takes a portable math description (HLO/StableHLO) and produces TPU machine code. The slow, expensive step. |
| **HLO / StableHLO / MLIR** | Intermediate languages between "PyTorch graph" and "TPU machine code" — think of them as the compiler's internal paperwork formats. |
| **PjRt executable** | The final product: actual TPU machine code, loaded onto the chip and ready to run. Can be serialized to bytes and saved. |
| **HBM / VMEM / MXU** | TPU hardware terms: HBM is the big off-chip memory, VMEM is small fast on-chip memory, MXU is the matrix-multiply engine. Compilation is largely about choreographing data movement among these. |
| **KV cache** | Not a compilation cache! The per-request memory where an LLM stores attention state for tokens it has already processed. Its *size* is baked into the compiled program, which is why it shows up in cache keys here. |
| **Shape bucket / padding** | TPU programs are compiled for one fixed input size. Real batches vary, so vLLM pads every batch up to the nearest of ~20 preset sizes ("buckets") and compiles one program per bucket. |
| **Piecewise compilation** | vLLM splits the model graph at the attention operation and compiles the pieces separately (attention has custom handling). So the unit of compilation is (subgraph, bucket), not the whole model. |
| **Pickle** | Python's built-in "save this object to a file" serializer. `pickle.dump(x, f)` writes it; `pickle.load(f)` recreates it. A class can customize what gets written via a `__reduce__` method — that detail matters a lot in Part 4. |
| **Rank / DP** | When a model runs across many TPU chips, each participating process is a "rank." DP = data parallelism (several replicas each serving different requests). Each rank compiles its own programs, so caches are often per-rank. |
| **Cache key / hash** | A fingerprint summarizing everything that could change the compiled output (model config, library versions, kernel source, …). Same fingerprint → safe to reuse the saved result; any change → different fingerprint → recompile. |
| **Cold / warm start** | Cold: first launch, nothing cached, pay full compile time. Warm: relaunch with caches populated, skip most of it. |

---

## Part 1 — The pipeline, end to end

Think of one model forward pass going through five translations. Each translation
is expensive, and each layer caches its own output.

```
  Python nn.Module forward()
        │  ① TorchDynamo  (bytecode tracing)
        ▼
  FX graph  (a Python-level dataflow graph of ATen ops)
        │  ② vLLM splits it into "piecewise" subgraphs, one compile per shape bucket
        ▼
  FX subgraph + concrete shapes
        │  ③ aot_autograd → decomposition → torch_tpu's fx_to_mlir
        ▼
  StableHLO / MLIR module
        │  ④ XLA:TPU compiler (inside libtpu) — this is the slow one, tens of seconds
        ▼
  PjRt "loaded executable"  = actual TPU machine code + memory layouts
        │  ⑤ SerializeExecutable() → an opaque blob of bytes
        ▼
  the thing every disk cache in this system is ultimately storing
```

### What "lowering HLO to TPU instructions" actually means

You asked for this specifically. In plain terms:

- **HLO / StableHLO** is a *portable, hardware-independent* description of the
  math: "matmul this [8192,4096] bf16 tensor with that one, add bias, softmax."
  It knows shapes and dtypes but nothing about the chip.

- **The XLA compiler stack** takes that and does the work a normal compiler does,
  plus a lot that's specific to how a TPU works (a fixed grid of multiply-add
  units fed by carefully choreographed memory transfers — nothing like a CPU):
  - *layout assignment* — TPUs want data tiled into (8, 128) blocks; XLA decides
    the physical tiling of every tensor and inserts transposes where it must.
  - *fusion* — merges elementwise chains into one kernel so intermediates stay in
    on-chip memory (VMEM) instead of round-tripping through HBM.
  - *scheduling / double-buffering* — overlaps HBM↔VMEM DMA copies with MXU
    (matrix unit) work, which is where most of the perf comes from.
  - *memory assignment* — statically assigns every buffer an offset. TPU has no
    malloc at runtime; the whole program's memory plan is baked in at compile time.
  - *collectives lowering* — turns `all-reduce` into actual ICI link operations
    with a fixed topology.
  - *code emission* — writes out the final machine instructions for the TPU's
    scalar/vector units and the MXU.

- **The result** is a `PjRtLoadedExecutable`. `SerializeExecutable()` turns it into
  a byte blob you can write to disk and reload later without recompiling.

This is why the whole caching apparatus exists: step ④ dominates cold-start time,
and steps ①–③ together are maybe a few seconds while ④ can be minutes for a large
model × 20 shape buckets.

**Two important consequences:**

1. The blob is only valid for the exact chip generation, libtpu build, and topology
   it was compiled for. That's why the tier-3 cache directory is namespaced by a
   *binary fingerprint* (see below).
2. Because memory assignment is baked in, KV-cache block count is a **compile-time
   literal**. That's exactly why `compute_tpu_compilation_hash()` in
   `src/vllm_torchtpu/compilation/tpu_compiler.py:170` hashes `num_gpu_blocks`.

---

## Part 2 — The three-tier native cache (torch_tpu)

`torch_tpu` has its own cache, completely independent of vLLM. Source:
`torch_tpu/common/compilation_cache.cc`, `tier2_compilation_cache.cc`,
`tier3_compilation_cache.cc`.

The "tiers" are the same idea at three distances: the same compiled blob kept in
this process's memory (fastest, dies with the process), in a shared spot on this
machine (survives the process, dies at reboot), and on real storage (survives
everything, can even be a network bucket shared across machines).

| Tier | Where | Lifetime | Purpose |
|---|---|---|---|
| **Tier-1** | Process memory (`CompilationCache` singleton) | one process | Same graph compiled twice in one run → free |
| **Tier-2** | `/dev/shm/torch_tpu_cache/<name>/<fingerprint>/<key>.bin` | node reboot | The 8 ranks on one host compile the *same* graph; only one pays |
| **Tier-3** | `$TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT/<fingerprint>/<key>.bin` | persistent (can be GCS) | Survives process restart; this is your `xla_cache/torch_tpu_tier3/*.bin` |

Lookup order on a compile request (`GetFromTier2OrCompile` → `GetFromTier3OrCompile`):

```
tier-1 (memory)  →  tier-2 (/dev/shm)  →  tier-3 (disk)  →  actually compile

write-back: a tier-3 hit is written back to tier-2; a real compile (tier-1)
is written back to BOTH tier-2 and tier-3.
```

Nice detail: on a tier-3 read, torch_tpu *also* kicks off a local compile in a
background thread pool as a hedge (`UsesLocalBackupTaskForTier3Read`, default on),
and whichever finishes first wins. That matters when tier-3 is remote/GCS-backed.

**How tier-3 files are named** (`GetTier3CacheEntryPath`):

```
<root>/<TorchTpuBinaryFingerprint:16hex>/<shapeless_key>_<dimensions_key>_<compile_options_key>.bin
       └── which libtpu/torch_tpu build          └── CompilationCacheKey::CompactFormat()
```

- `shapeless_key` — the graph structure (ops, connectivity, dtypes), shapes excluded
- `dimensions_key` — the concrete shapes for this bucket
- `compile_options_key` — XLA flags, topology, `LIBTPU_INIT_ARGS`, etc.

**The file contents are literally `SerializeExecutable()` output** —
`AtomicWriteToCacheFile()` in `tier2_compilation_cache.cc:387` calls
`executable->GetExecutable()->SerializeExecutable()` and writes those bytes.
Remember this; it's the crux of the duplication question.

**How it gets turned on here** — `src/vllm_torchtpu/env_override.py:188`:

```python
if VLLM_XLA_CACHE_PATH:
    TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT = f"{VLLM_XLA_CACHE_PATH}/torch_tpu_tier3"
    TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE    = "tpu_tier2_cache"
```

Tier-2 is force-enabled because torch_tpu refuses tier-3 without tier-2
(`UsesTier3CompilationCache() = UsesTier2CompilationCache() && root non-empty`).

---

## Part 3 — The vLLM-level caches

vLLM has two *separate* disk caches, at two different altitudes.

### 3a. Piecewise / per-shape cache — `torch_compile_cache/<hash10>/rank_i_j/<prefix>/`

Written by `vllm/compilation/backends.py` (`CompilerManager`) + our
`TpuCompilerAdaptor`. Contains:

- `vllm_compile_cache.py` — a plain-text dict: `(compile_range, graph_index, "tpu") → handle`
- `artifact_compile_range_<start>_<end>_subgraph_<i>` — **one pickle per shape bucket
  per subgraph**, written by `pickle.dump(inner_exe, f)` at
  `src/vllm_torchtpu/compilation/tpu_compiler.py:340`
- `computation_graph.py`, `transformed_code.py`, `cache_key_factors.json` — diagnostics

**Careful — this directory is actually split across two roots.** The
`VLLM_XLA_CACHE_PATH` relocation happens *inside* `TpuCompilerAdaptor.initialize_cache()`
(`tpu_compiler.py:256`), so it only moves the adaptor's own writes — the
`artifact_compile_range_*` pickles. `CompilerManager` keeps the **original**
`local_cache_dir` under `VLLM_CACHE_ROOT` for its own files
(`cache_file_path` is set at `backends.py:183` *before* the compiler's
`initialize_cache` is called at `backends.py:212`). So with `VLLM_XLA_CACHE_PATH`
set you get the same relative layout in two places:

```
vllm_cache/torch_compile_cache/<h>/rank_i_j/<p>/vllm_compile_cache.py   ← the index
vllm_cache/torch_compile_cache/<h>/rank_i_j/<p>/cache_key_factors.json
vllm_cache/torch_compile_cache/<h>/rank_i_j/<p>/computation_graph.py
xla_cache/torch_compile_cache/<h>/rank_i_j/<p>/artifact_compile_range_*  ← the payloads
```

The index in `vllm_cache` points into `xla_cache` via the absolute paths stored
in each handle.

The key naming comes from `backends.py:306` — vLLM only auto-names artifacts for
non-Inductor backends, which is us:

```python
maybe_key = f"artifact_compile_range_{start}_{end}_subgraph_{graph_index}"
```

**What's inside that pickle?** `TorchTpuCompiledExecutable.__reduce__` in
`torch_tpu/_internal/compile/torch_tpu_compiled_executable.py:340`:

```python
def __reduce__(self):
    serialized = tpu_torch_compile.serialize_executable(self._executable)
    return (_unpickle_compiled_executable, (serialized, self._reconstruct_fx_outputs_fn, ...))
```

and `PySerializeExecutable` (`tpu_torch_compile.cc:970`) calls
`executable->GetExecutable()->SerializeExecutable()`.

> **That is the same function call tier-3 uses.** The bytes inside
> `artifact_compile_range_*` and the bytes inside `torch_tpu_tier3/*.bin` are the
> same blob, differing only by pickle framing.

### 3b. Dynamo AOT cache — `torch_compile_cache/torch_aot_compile/<hash>/rank_i_j/model`

Written by `vllm/compilation/decorators.py` (path constructed at `:543`).
**This is on by default**, not opt-in: `use_aot_compile()` in `vllm/envs.py:340`
defaults `VLLM_USE_AOT_COMPILE` to `"1"` whenever torch ≥ 2.10 and
`VLLM_DISABLE_COMPILE_CACHE` is unset — and this repo pins `torch==2.11.0`
(`pyproject.toml:28`). The only other off-switch is `backend == "eager"`
(`decorators.py:653`); ours is `"tpu"`, so the AOT path runs.

It caches a *different* layer: the Dynamo trace itself — bytecode transform,
guards, and the FX graph — so warm start skips tracing, not just compiling.

The payload (`VllmSerializableFunction.serialize_compile_artifacts`,
`vllm/compilation/caching.py:252`) is a pickle of: the serialized FX graph module,
meta-device example inputs, prefix/config, and — *only if
`VLLM_USE_MEGA_AOT_ARTIFACT=1`* — the collected backend artifacts.

Crucially, this path uses `envs.VLLM_CACHE_ROOT` **directly**. It never consults
`VLLM_XLA_CACHE_PATH`. That is exactly why you see it under `vllm_cache/` while
3a lives under `xla_cache/`.

`vllm_torchtpu/__init__.py:11` patches `aot_compile_hash_factors` so the TPU
compiler hash participates in this cache key too — otherwise a kernel edit would
be invisible to the AOT layer.

---

## Part 4 — So where are the actual duplicates?

### Confirmed duplicate: tier-3 `.bin` ⟷ `artifact_compile_range_*`

**Same bytes, two files, always both written on a cold start.** For a model with
N piecewise subgraphs × M shape buckets you get N×M copies in each place. On a
large model that is easily gigabytes doubled.

### Why it was introduced — the answer from PR #38

[torchtpu-vllm#38](https://github.com/google-pytorch/torchtpu-vllm/pull/38)
("Integrate aot compile path", by @angelayi) created `TpuCompilerAdaptor`. Its
"Gaps" section says it directly:

> "Since we cannot serialize the `aot_autograd` callable due to other closures, we
> can only serialize the `_TorchTpuCompiledExecutable` artifact. Therefore we must
> call `aot_autograd` to construct the python wrapper again to apply the additional
> `copy_`."

Unpacked, in plain language:

vLLM's `CompilerInterface` contract is *"compile() returns a callable and a handle;
load(handle) must give me back an equivalent callable."* The callable that
`TpuBackend` produces is an `aot_autograd` wrapper — a Python function that
secretly holds references to live, in-memory objects (tracing machinery,
bookkeeping for which inputs get mutated, the epilogue that copies mutated
values back). Pickle cannot save functions that hold live references like that.
So the wrapper as a whole can't be written to disk.

The only picklable piece is the inner executable. So the adaptor:
1. reaches into `_tpu_backend._compiled_executables[-1]` and pickles that
   (`tpu_compiler.py:336`), and
2. on `load()`, re-runs `aot_autograd` with a stub compiler that just hands back
   the unpickled executable (`tpu_compiler.py:384`), rebuilding the Python wrapper
   around a pre-compiled core.

At the time, tier-3 wasn't wired into the vLLM path at all — `VLLM_XLA_CACHE_PATH`
and the tier-3 routing came later (commit `ee5f214a` / `6ef917cf`, "support
VLLM_XLA_CACHE_PATH for unified compilation caching", torchtpu-vllm#279). So the
duplication wasn't a decision anyone made; it's two independently-correct caches
that were built at different times and now both run.

### Not actually a duplicate (but easy to mistake for one): `torch_aot_compile/`

I checked whether the AOT `model` file contains a third copy of the TPU binaries.
The path exists but is currently mostly closed:

- `VllmBackend.collect_standalone_compile_artifacts()` (`backends.py:872`) returns
  `None, None, None` immediately unless `VLLM_USE_MEGA_AOT_ARTIFACT=1`.
- Even with it on, `PiecewiseBackend.to_bytes()` (`piecewise_backend.py:240`) only
  serializes entries where `hasattr(entry.runnable, "serialize")`.
- `TpuBackend.__call__` *does* attach `.serialize` now (default
  `enable_serialization=True`, `_backend.py:264`/`347`, producing a
  `BundledAOTAutogradResult` that contains the compiled forward). **But** our
  adaptor wraps it: when `_ensure_tuple_output` had to wrap the graph,
  `compile()` returns the plain `unwrap_fn` closure (`tpu_compiler.py:319`), which
  drops the `.serialize` attribute.

So today: `torch_aot_compile/` holds the Dynamo layer (bytecode + FX + guards),
not the TPU machine code. It is a *different* cache, not a third copy.

**But this is one torch bump away from changing.** `use_mega_aot_artifact()`
(`vllm/envs.py:352`) defaults to `"1"` when torch ≥ 2.12.0.dev *and* AOT compile
is on — and AOT compile is already on by default (see 3b). This repo pins
`torch==2.11.0`, which is the only thing keeping mega-AOT off. When the pin
moves to 2.12+, every subgraph whose runnable kept its `.serialize` attribute
(i.e. the ones `_ensure_tuple_output` did *not* wrap) starts contributing a
genuine third copy of the executable bytes inside `torch_aot_compile/model`.
Worth adding an explicit guard or a conscious decision before that bump.

---

## Part 5 — "Store only a link" — what your colleague meant, and what it costs

The idea is sound. Instead of `artifact_compile_range_*` holding ~200 MB of
executable bytes, it holds ~50 bytes: the tier-3 `CompilationCacheKey`. On load,
you ask torch_tpu for that key and it comes back from tier-1/2/3.

Here's the honest assessment of each option.

### Option A — make `__reduce__` emit a key instead of bytes (the clean fix, needs torch_tpu)

```python
def __reduce__(self):
    if tier3_enabled and self._cache_key is not None:
        return (_unpickle_by_cache_key, (self._cache_key, self._reconstruct_fx_outputs_fn, ...))
    return (_unpickle_compiled_executable, (serialize_executable(self._executable), ...))
```

**Blocker:** `TorchTpuCompiledExecutable.__init__`
(`torch_tpu_compiled_executable.py:124`) takes `(executable,
reconstruct_fx_outputs_fn, updates_default_generator_state)` — **the
`CompilationCacheKey` is never stored on the Python object.** It's computed and
consumed entirely inside C++ (`CompileModuleWithCache`, `tpu_torch_compile.cc:662`).

So this needs a torch_tpu change: return the key from `CompileModuleWithCache`,
stash it on the artifact, and add a `load_from_cache_key(key)` binding that goes
through `GetFromTier2OrCompile`. Not large, but it's their repo.

**Correctness caveats to design for:**
- The key is namespaced by `TorchTpuBinaryFingerprint` — a link is only valid for
  the same libtpu/torch_tpu build. A stale link must fail *soft* (fall back to
  recompile), not crash.
- The pickle and the tier-3 blob would then have different lifetimes. Anything
  that GCs `xla_cache/torch_tpu_tier3/` independently of `torch_compile_cache/`
  would silently turn every cache hit into a full recompile. Needs to be one
  eviction domain, or the fallback path must be genuinely fast.

### Option B — don't write the pickle at all when tier-3 is on (smallest change, ours alone)

In `TpuCompilerAdaptor.compile()`, skip the `pickle.dump` when tier-3 is enabled,
and store a handle that means "no blob — recompile me." `load()` then re-runs
`_tpu_backend(graph, example_inputs)`, which re-does ① FX→MLIR lowering (seconds)
but hits tier-3 for ④ the XLA compile (the expensive part).

- **Pro:** entirely in this repo, no torch_tpu dependency, no new failure mode.
- **Con:** you pay MLIR lowering per subgraph per bucket on every warm start.
  For a 20-bucket model that's real but bounded — and it's the same work a
  tier-3-only setup already does today.
- This is the pragmatic version of "store the link": the link *is* the cache key
  that torch_tpu recomputes deterministically from the graph.

### Option C — hardlink / reflink the two files

Tempting, and wrong. The pickle isn't the raw blob — it has pickle framing plus
`_reconstruct_fx_outputs_fn`. Byte ranges don't line up, so you can't hardlink
them. You'd first have to change the on-disk format, at which point you're doing
Option A anyway.

### Recommendation

Ship **Option B** behind a flag now (it's a ~20-line change in `tpu_compiler.py`
and reclaims half the disk immediately), and file **Option A** against torch_tpu
as the real fix, since it eliminates the duplicate *and* keeps the "skip MLIR
lowering" win.

---

## Part 6 — Other things worth knowing (found while reading)

1. **The handle stores an absolute path.**
   `handle = (key, save_path, was_wrapped)` (`tpu_compiler.py:341`), and it's
   written verbatim into `vllm_compile_cache.py`. Move or rename the cache
   directory — or mount it at a different path in a container — and every load
   misses. A path relative to `cache_dir` would make the cache relocatable.
   Worth fixing regardless of which dedup option you pick.

2. **Two roots by accident — twice over.**
   `VLLM_XLA_CACHE_PATH` relocates only the artifact pickles (the relocation
   lives inside `TpuCompilerAdaptor.initialize_cache`), so even *within* the
   piecewise cache the index (`vllm_compile_cache.py`) sits in `vllm_cache/`
   while the payloads it points to sit in `xla_cache/` (see 3a). And the AOT
   cache (3b) never consults `VLLM_XLA_CACHE_PATH` at all — `decorators.py`
   reads `VLLM_CACHE_ROOT` directly. If you want one root, either set
   `VLLM_CACHE_ROOT` itself to the shared location, or extend the relocation to
   cover both the `CompilerManager` files and the AOT path.

3. **Tier-2 lives in `/dev/shm` and counts against your RAM.**
   `/dev/shm/torch_tpu_cache/...` is tmpfs. On a multi-host job with a big model
   the tier-2 copy is a third full copy of every executable, in memory. It's
   force-enabled whenever tier-3 is on. `run_qwen35_raiden_pcp8_dp8_live_pair.sh:445`
   already sets `TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=disabled` for that
   reason (`"disabled"` is a real sentinel, `tier2_compilation_cache.cc:122`;
   that script also unsets the tier-3 root, so it is knowingly opting out of
   both). Corollary worth remembering: because tier-3 requires tier-2, setting
   tier-2 to `disabled` **silently disables tier-3 too**, even if the tier-3
   root is still set.

4. **Cache-key discipline in this repo is unusually careful, and deliberately so.**
   `compute_tpu_compilation_hash` hashes the *bytes of Pallas kernel source files*
   (`_RUNTIME_CACHE_KEY_PATHS`), because those kernels run behind custom-op
   boundaries where Dynamo's traced-file validation can't see them. Without that,
   editing a kernel would silently reuse a stale binary. Conversely
   `_TPU_COMPILE_ENV_IGNORED` and `HASH_IGNORED_ADDITIONAL_CONFIG_KEYS` exclude
   orchestration-only knobs so a new profiler trace dir doesn't nuke the cache.
   Any dedup work has to preserve both directions.

---

## Quick reference — who writes what

| Path | Written by | Contains | Honors `VLLM_XLA_CACHE_PATH`? |
|---|---|---|---|
| `xla_cache/torch_tpu_tier3/<fp>/<key>.bin` | torch_tpu C++ | `SerializeExecutable()` bytes | yes (via `env_override.py`) |
| `/dev/shm/torch_tpu_cache/<name>/<fp>/<key>.bin` | torch_tpu C++ (tier-2) | same bytes | n/a (always tmpfs) |
| `xla_cache/torch_compile_cache/<h>/rank_i_j/<p>/artifact_compile_range_*` | `TpuCompilerAdaptor.compile` | pickle of the **same bytes** | yes |
| `vllm_cache/torch_compile_cache/<h>/rank_i_j/<p>/vllm_compile_cache.py` | `CompilerManager.save_to_file` | index: range → handle (abs path into `xla_cache/`) | **no** (only the pickles move) |
| `vllm_cache/torch_compile_cache/torch_aot_compile/<h>/rank_i_j/model` | `decorators.py` | Dynamo bytecode + FX graph + guards | **no** |
| `vllm_cache/torch_compile_cache/torch_aot_compile/<h>/inductor_cache/` | torch inductor | ~empty on TPU (no Inductor) | **no** |
