# Deviations from `references/code-style.md`, and from the image's assumptions

Declared here because S10 of that file requires a deviation to be named where a
reader will find it, not argued for in a commit message. Each entry says what
the rule is, what this directory does instead, and why the substitute is not
just convenience.

## Forced by the benchmark image

These are not judgement calls. Each was checked in the image the microbenchmark
runs in (`vllm-torchtpu-annyan_google_com:13617694`) and each would turn a
measurement into an import error.

1. **`jaxtyping` is not installed.** S4 asks for jaxtyping annotations on the
   public wrapper. `import jaxtyping` fails in the image's `uv_venv`, and
   `src/vllm_torchtpu/kernels/kimi_k3/chunk_kda.py:1425` records that this repo
   removed it for that reason. Shapes are carried as trailing comments in the
   letters of each file's dimension glossary, and the check S4 actually wants —
   a `ValueError` at the boundary — is `base.bind` and `base.check_shapes`.

2. **`pydantic` is not installed.** Every variant's `pallas_mosaic_tpu.py`
   imports it for its config dataclass. `final/pallas_mosaic_tpu.py` uses stdlib
   `dataclasses` with `frozen=True, kw_only=True, slots=True` instead. This is
   not only an availability fix: a frozen dataclass is hashable, which is what
   lets `cfg` be a `static_argname` and gives every leave-one-out arm its own
   jit cache entry. The alternative — rebinding a module attribute between arms
   — hands arm 2 the executable arm 1 compiled, and on this op that is a real
   hazard rather than a hypothetical one.

3. **`pytest` is absent.** There is no test file here in the pytest sense. The
   correctness gate is `instruments/correctness.py`, invoked through
   `run_correctness.py`, which exits non-zero on failure — so it composes with
   `&&` in a shard command, which is how the benchmark is gated.

4. **`jnp.top_k` is gone from this image's JAX.** `base.fwd`, the plain-XLA
   reference S12 asks for, uses `jax.lax.top_k`.

## Judgement calls

5. **Two operations share four files** rather than taking four each. The
   TensorCore scoring half is present only because O1 changes it, and splitting
   it out would put a ~700-line verbatim copy in a file of its own for no gain.
   The cost is that `pallas_mosaic_tpu_kernel.py` is ~1500 lines against the
   ~250 S10 suggests.

6. **The copied TensorCore body keeps the literal `128` lane width** instead of
   `pltpu.get_tpu_info().num_lanes`, against S13. Rewriting ~370 lines of index
   arithmetic is unrelated to any adopted change and is pure regression risk in
   a half whose entire contract is that its output does not move. The
   SparseCore body, which *is* changed, has all four of its hardware constants
   — lane count, subcore count, core count, tile-memory capacity — read from
   `pltpu.get_tpu_info()` via `pallas_mosaic_tpu.hardware()` and threaded down
   as parameters.

7. **`api.py` names two config arguments**, which S10 forbids:
   `num_kv_pages_per_block` and `num_queries_per_block`. They stay in the
   signature because the shared, read-only instruments pass them, and a wrapper
   that could not be called by the instruments would have moved the gate rather
   than passed it. Under `wide_kv_tile` they are *overridden*, which is the
   whole content of that option: a caller who does not know the page-size
   regime cannot set them correctly. Overriding an argument silently is worse
   than not taking one, so the override is announced once per process.

8. **The gate is shared, not local.** S12 wants the test beside the kernel.
   `instruments/correctness.py` is one directory up and is imported, never
   copied and never edited: a candidate that changes its own test has moved the
   gate rather than passed it. **No tolerance value appears anywhere in this
   directory** — both thresholds live in the shared gate, which re-derives and
   prints their separation windows on every `--negative-control` run.

## Deliberately *not* carried from step 3

9. **O3's flat-index gather is not ported.** O3 contained two mechanisms and
   only one is adopted. The wave packing is here; the flat-index form of the
   stage-2 gather (`jnp.take(words_flat, row_base + idx)`) is not. Its
   full-level win had no identified mechanism (`unmodelled`), it was
   *unresolved* at selector level at every shape, and its gather measured
   2.7–28.5% slower than `take_along_axis` in isolation. Under `emit_keys` the
   gather it replaces does not exist at all, so stacking the two is not a
   composition — it is a change to a line that has been deleted.

10. **O9 (score-slice DMA overlap) and O5 (never built) are absent.** O9 did not
    resolve at any shape; O5 has no measurement behind it. Neither has a flag
    here, because a flag implies a measured arm.
