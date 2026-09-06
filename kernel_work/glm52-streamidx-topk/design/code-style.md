# `code-style.md` file plan, instantiated for this block

## 1. The file decomposition, instantiated

`code-style.md` §10 prescribes exactly four files — "One operation, four files, each with a job you can state in a sentence." That is the number the plan has. Each variant directory `kernel_work/glm52-streamidx-topk/<variant>/` gets these four, and the composed `final/` gets the same four.

| File | "Contains" (reference's words) | "Must not contain" (reference's words) | Instantiated for this block |
| --- | --- | --- | --- |
| `api.py` | "the free function users call; jaxtyping annotations; the backend fallback chain" | "tuning knobs — **no `config` at all**" | The free function the indexer call site invokes (what `mla_attention_op.py` reaches today). jaxtyping on `q`, `k_cache`, `weights`, `kv_scale`, `seq_lens`, `block_tables` → `Integer[Array, "T k"]`, using the glossary's letters. The fallback chain that raises per §9 when a shape is unsupported. No `bq_sz`, `bkv_p`, `seq_batch_size`, radix width or any other knob appears here. |
| `base.py` | "the `Op` subclass, `bind()` validation, the plain-XLA reference `_fwd`, the module-level `vjp`" | "any backend-specific code" | `bind()` validation of the legality ladder (token bucket, `page_size*bkv_p % 128`, `n % 16`, `k` bounds). The plain-XLA `_fwd`: dequantize, score, mask, `jax.lax.top_k`. Per §12 this `_fwd` **is** the test reference — "A kernel is tested against the same operation written in ordinary JAX, in the same repo" — so the reference implementation currently living in `instruments/correctness.py` belongs here. No Pallas, no Mosaic, no SparseCore. |
| `pallas_mosaic_tpu.py` | "the backend `Op` subclass, its `Config`, `_get_heuristics_config`, `_get_autotuning_configs`, `supported_on`" | "**`pl.pallas_call`**" | The `Config` (§11: frozen/kw_only/slots pydantic) carrying tile and grid knobs; `_get_heuristics_config` choosing them from `T`, `n`, `k`, `page_size` and `pltpu.get_tpu_info()`; `_get_autotuning_configs`; `supported_on`. This is where the decode-batched / decode-remainder / mixed dispatch decision is made. Exactly the spelling `_get_heuristics_config` — "Not `_default_config`, not `_heuristic`." |
| `pallas_mosaic_tpu_kernel.py` | "the `pallas_call`, the kernel body, the index maps" | "policy decisions about *which* config to use" | Every `pl.pallas_call` in the variant — scoring launches **and** any SparseCore selection launch — plus the kernel bodies and the `index_map`s over the KV-block and token grids. It receives a `Config`; it never picks one. |

Three notes, all from §10 and the checklist, that bear on this block specifically:

- **"No `pallas_call` outside the last."** The incumbent's compute-unit split (`streamindex_topk.py` / `sparsecore_topk.py`) is not a boundary the reference sanctions at the top level. A variant that keeps both a TensorCore scoring launch and a SparseCore select launch puts **both** in `pallas_mosaic_tpu_kernel.py`. The alternative the reference permits is treating the selector as its own operation — but then it gets its own four files, not a bare second module. "One operation, four files."
- **"Split at around 250 lines."** The reference gives this as the trigger for the `api`/`base`/backend/kernel split; the incumbent is far past it, so the split is not optional here.
- **The opt-out is a real option, but it has a form.** §10: "A kernel doing dynamic-offset DMA, in-kernel metadata construction and selective triple buffering may have nothing the four-file shape can hold. Opting out is a legitimate outcome, not a mess to be cleaned up later — **say so in the module docstring and leave.**" What is ruled out is the middle: "an `Op` subclass whose `_fwd` is a thin shim over a thousand lines that ignore the abstraction is worse than either choice made cleanly." A variant taking this route must state it in the docstring; it may not half-take it.

## 2. Further decomposition: yes, it applies

§10's second axis is "**what each module is allowed to touch** — refs, DMA, arrays, or plain ints. Make 'does this function take a `Ref`?' a file boundary rather than a per-function surprise." It applies to this block: the operation is well past four files' worth, it carries substantial host-side metadata algebra, and the payoff the reference names — "A function that takes only arrays can be called from XLA, from a unit test, and from inside a kernel unchanged" — lands directly on the parts of this op that are hardest to test through a `pallas_call`.

Modules, and which side of the ref boundary each falls on:

**Ref-free side (arrays or ints only — unit-testable against `base.py`'s `_fwd`, no device):**

- `compute.py` — "arrays only — **no refs**". The scoring math as pure array functions: the fp8×fp8 dot with f32 accumulation, ReLU, the `weights[t,h]` head-weighted sum, the e8m0 dequant by `kv_scale`, and the validity/causal masks. Also the selection algebra a variant introduces — histogram, per-level threshold, bucket-count refinement — written array-in/array-out. Every function annotated `jax.Array` in and out, per the ragged-dot instance.
- `tiling.py` — "ints only — block extents, grid extents". `n = pages_per_seq * page_size`, `num_bkv = cdiv(max_i kv_lens[i], bkv_sz)` and the group-max arithmetic behind Finding 3, the token-bucket ladder, the `bq_sz`/`bkv_p` grid extents, and the sequence-length sort permutation if a variant adds one. "testable with no device at all" — this is where the Finding 2/Finding 3 structural terms become checkable by `pytest` on a laptop.
- `config.py` — the config dataclass (§11).

**Ref side:**

- `load_store.py` — "refs in, arrays out"; "the only file where `ref[...]` appears". Unpacking the 256-byte packed cache row into fp8 values plus the e8m0 scale byte, the masked store of the `int32[T,k]` index output with its `-1` padding.
- `dma.py` — "`make_async_copy`, semaphores, memory spaces"; "the only file that can deadlock". The `_fetch_bkv` KV-block copies, any strided 128-of-256 read a variant tries, and any score-tile staging that keeps the matrix out of HBM.
- `wrapper.py` — "assembly: builds the `pallas_call` out of the above".

The splash-attention instance in §10 is the closest structural precedent and maps cleanly: "pure host logic → host-built arrays → device refs." Here that is `tiling.py` (block-table and `num_bkv` metadata, no `jax`) → the host-built index/length arrays → the kernel file consuming them as refs.

For a small variant that does not need six modules, §10 gives the function-granularity form of the same rule (ragged dot): keep the ref-free helpers above the kernel bodies in one file, "the array functions are annotated and above; the ref functions are bare and below." Either granularity satisfies the checklist item; a per-function surprise does not.

## 3. The checklist, verbatim

Step 3 returns this filled in per variant, read out of `references/code-style.md` rather than from this copy.

```
- [ ] Dimension glossary at the top of the file; uppercase logical, `t`-prefixed tile, `n`-prefixed
      grid extent. Padded extents get their own letter.
- [ ] Every ref parameter has a trailing `# [shape] dtype` comment in the glossary's letters, saying
      whether it is the block or the whole array.
- [ ] Extents derived from the refs, not re-read from the config. No bare `x.shape[2]` in
      arithmetic.
- [ ] jaxtyping on the public wrapper using the same letters; bare refs inside — never a
      `jax.Array` annotation on a ref.
- [ ] Kernel signature carries `# Inputs. / # Outputs. / # Scratch.`
- [ ] Kernel body in order: indices → shapes → semaphores → init guard → loads → compute → masked
      store → epilogue.
- [ ] Single characters only for loop indices, annotated math tensors, uppercase extents and `d<x>`
      gradients. Grid indices spelled out.
- [ ] Memory space in the ref name wherever two spaces share a signature. Never `_ptr`.
- [ ] Comments name the compromise, not the operation. One-line module docstring.
- [ ] Unsupported features raise `NotImplementedError`, not `ValueError`.
- [ ] Four files: `api.py` / `base.py` / `pallas_mosaic_tpu.py` / `..._kernel.py`. No `pallas_call`
      outside the last, no `config` in the first.
- [ ] Where it splits further, the boundary is what the module may touch: arrays / refs / DMA /
      ints.
- [ ] Config is a frozen, kw_only, slots pydantic dataclass; cross-field checks in `__post_init__`.
- [ ] Tested against the XLA `_fwd` in `base.py`, tolerance visible at the call site, runtime skips.
- [ ] Hardware constants from `pltpu.get_tpu_info()`, never hardcoded.
```

## Conflicts with the brief

One, and it is minor. The brief describes the incumbent as "two files and two compute units" and says the plan should not assume that split survives — which agrees with `code-style.md`, but it is worth stating what the reference replaces it with: the sanctioned top-level boundary is `api` / `base` / backend / kernel, **not** TensorCore-vs-SparseCore. A variant that keeps a standalone `sparsecore_topk.py` holding its own `pl.pallas_call` violates the checklist's "No `pallas_call` outside the last" unless the selector is promoted to its own operation with its own four files. `code-style.md` wins.

Nothing else conflicts. The `kernel_work/glm52-streamidx-topk/<variant>/` layout and the composed `final/` are orthogonal to the reference, which governs file names within a directory, not the directory tree.
