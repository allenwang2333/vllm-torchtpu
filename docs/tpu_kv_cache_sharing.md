# Why TPU can't share one KV cache the way CUDA does

Written 2026-08-11. Answers a question that keeps coming up: why did the DSv4
bring-up hit "we need two physical KV arrays where vLLM expects one", and why
hadn't that shown up before.

**Short answer:** it had, and other models here hit it too. It isn't a DSv4
bug — it's a structural mismatch between how vLLM's block manager accounts for
memory and what a `jax.Array` can be. What's unusual about DSv4 is that the two
cheap ways out don't work for it, so it needs a third.

| | who | how it's handled | cost |
|---|---|---|---|
| **1** | most models | layers sharing a `KVCacheTensor` genuinely share one array | none — this is what vLLM intends |
| **2** | Qwen3.5 and other attention+mamba hybrids | one array per layer, block accounting compensated | HBM; caps context above ~2048 tokens |
| **3** | DSv4 only | sharing rebuilt by hand from the specs | complexity, and invariants that aren't all checked |

DSv4 needs (3) because (1) is impossible — the layers that must share have
incompatible page geometry — and (2) is unaffordable at 43 layers x 3 caches.

---

## The assumption vLLM makes

vLLM's KV cache manager allocates one `KVCacheTensor` and records the set of
layers that share it (`shared_by`). Those layers are expected to sit in **one
physical allocation**, each indexing disjoint byte ranges of it. Device kernels
reinterpret the same bytes as whatever that layer needs.

Upstream states it directly in the sparse-MLA path DSv4 uses
(`vllm/v1/attention/backends/mla/sparse_swa.py`):

> Block size is constrained by tensor sharing between SWA and C4A KV blocks.
> **Since both block types share the same physical tensor**, they must use the
> same page size.

On CUDA this is nearly free. You allocate a slab and take byte-offset views:

```
    one cudaMalloc
    +---------------------------+---------------------------+
    | layer A's blocks          | layer B's blocks          |
    +---------------------------+---------------------------+
      view A (offset 0)           view B (offset N)
```

Two logically distinct caches, one allocation, different offsets, possibly
different dtypes over the same bytes. The block manager hands out page indices
on the assumption that this is what exists.

## Why that doesn't work on TPU

`jax.Array` is strongly typed and has no byte-view. There is no way to express
"this array is bytes 4096..8192 of that array, reinterpreted as fp8". You get
whole, typed arrays.

So the moment vLLM says *"these N layers share one tensor"*, the TPU backend
has to choose:

1. allocate **one array per layer** and fix up the block accounting, or
2. rebuild the sharing explicitly by **pointing several layer entries at the
   same array**

Both appear in this codebase. Which one is right depends on whether the layers
sharing the tensor have compatible geometry.

## The three strategies in our runner

`tpu_runner.py` handles `shared_by` groups three different ways depending on
whether the layers involved can actually share an array. Which one a model gets
is decided by geometry, not by the model.

### Strategy 1 — share one array (most models)

Default path, `_alloc_or_share`. When a `KVCacheTensor` is `shared_by` several
layers that agree on shape and dtype, allocate **one** array and hand the same
object to every layer, keyed on `(shape, dtype)`.

This works because sharing at *whole-array* granularity needs no byte-views. It
is what vLLM intends, minus the offsets. Most models never need anything else.

Getting this wrong is expensive and quiet: we previously gave each layer its own
array against a budget vLLM had sized for one, measured 12.66 MiB/block
allocated against the 6.09 MiB/block vLLM assumed (~2.08x), which overcommitted
HBM and killed the server at 9216 context (fixed 2026-08-08).

The keying matters. A mismatch in `(shape, dtype)` allocates rather than
silently aliasing incompatible layouts — better a wasted array than two layers
writing over each other.

### Strategy 2 — duplicate per layer, fix the accounting (Mamba hybrids)

Qwen3.5 and other attention+mamba models. Attention KV and mamba recurrent
state cannot share typed bytes, so every layer gets its own array.

That alone breaks the block arithmetic: vLLM's pool would hold
`num_shared_layers`x more block IDs than each per-layer array has slots. The
scheduler would hand out IDs past the end of a layer's leading dimension, JAX's
indexed writes would clip them silently, and multiple requests' mamba states
would collapse onto the same slot — corrupted state, gibberish output, no
crash.

The compensation is to report each layer's `page_size_padded` as the *whole*
`shared_by` footprint (`attn_page + N x mamba_unpadded`, using TPU-actual
per-block bytes so fp8 packing is accounted for), so vLLM derives a smaller
`num_blocks` that matches what we really allocate. HBM usage is unchanged; only
the block-ID accounting lines up.

The cost is real and the code says so:

> duplication is a Mamba-only workaround. Fixing this is what makes a
> >2048-token context testable here.

So this class of model **has** hit the limitation. It pays memory instead of
aliasing, and the price is a capped context.

`tpu-inference` made the same choice for Mamba, with a TODO naming it:

> we should not be replicating the kv cache for each layer and instead should
> follow the native GPU/Torch approach where every group of layers
> (`shared_by`) shares the same underlying raw tensor

and

> MambaSpec does not support shared layers for now, defaulting to single KV
> cache per layer...

### Strategy 3 — rebuild the sharing by hand (DSv4 only)

`_plan_ds_v4_swa_overlay` / `_alias_ds_v4_auxiliary_caches`. Ignore vLLM's byte
offsets, work out from the specs which layers *can* host which, and point one
layer's cache entry at another's array:

```python
kv_caches[layer_name] = kv_caches[host_layer_name]
```

## Why DSv4 needs a third strategy

The limitation is not DSv4-specific. What is specific is that **neither escape
hatch works for it**:

- Strategy 1 can't apply: the layers that must share have *incompatible*
  geometry — a 128-row SWA page against 256-row CSA pages and 8-row HCA pages.
  There is no single `(shape, dtype)` they agree on.
- Strategy 2 is unaffordable: 43 layers x 3 caches each. The duplication that
  Mamba absorbs would not fit.

Two conditions have to hold together for a model to land here, and DSv4 is the
first to satisfy both:

**Multiple live caches in one layer, with different geometry.** Hybrid models
like Gemma alternate — one layer sliding-window, the next full attention — so
each layer still owns exactly one cache and Strategy 1 handles it. DSv4 needs,
in the *same* layer: a sliding-window cache (last 128 tokens, full resolution),
a compressed main-branch cache (4:1 for CSA, 128:1 for HCA), and a compressor
state cache.

**Upstream assumes they are one allocation.** That assumption lives only in the
sparse-MLA path, which only DSv4 uses.

So the honest summary is not "nobody noticed this before". It is "everyone who
hit it could afford the cheap way out, and DSv4 can't".

## What the overlay actually does

Not "two copies of everything". Hosts are the `MLAAttentionSpec` arrays; a
sliding-window cache may overlay one whose page is at least as wide and holds
at least as many entries as the SWA logical page.

On DSv4-Flash that gives roughly a 21-of-43 split: a 128-row SWA page fits
inside a 256-row CSA page, and cannot fit an 8-row HCA page. The layers that
don't fit get their own arrays.

Assignment is positional within a cache group, and each host backs at most one
sliding-window cache. That constraint is load-bearing: layers in a group share
a block table, so two of them on one array would be handed the same block ID
and overwrite each other.

`DSV4_DISABLE_SWA_OVERLAY=1` turns the overlay off entirely — every SWA cache
gets its own array. It costs KV capacity, and it's the controlled way to ask
whether the overlay is responsible for a correctness symptom.

## Known-shaky parts

Worth stating plainly, since the mechanism has been flagged in review (PR #322)
and parts of it rest on assumption rather than a checked property.

- The planner pairs layers **purely by shape compatibility** and never consults
  `KVCacheTensor.shared_by`. The code asserts cross-group block-ID disjointness
  as the justification for why that's safe, but nothing in this repo verifies
  it. Marked UNVERIFIED in the source since 2026-08-07.
- It is a **runner-level tensor alias**, which reviewers objected to on PR #322
  as the wrong layer to solve this at. The objection is reasonable; the
  counter-argument is that the reference does the same thing, and there is no
  obvious place above the runner that knows both the specs and the allocation.
- Overlay is **not** the cause of the long-range correctness failure. Running
  with `DSV4_DISABLE_SWA_OVERLAY=1` produced byte-identical output, so it is
  eliminated as a suspect for that bug — but that also means it has never been
  stress-tested by a workload that would expose an aliasing mistake.

## The general fix, if anyone wants it

All three cases are symptoms of one gap: **the TPU backend has no way to
express a byte-level view into a shared allocation**, so every model that
triggers vLLM's `shared_by` grouping needs a bespoke workaround.

Things that don't help, worth ruling out because they get suggested:

- **Buffer donation.** Different mechanism. Donation lets a jitted function
  reuse an input's memory for its output, avoiding a copy. It doesn't let you
  address a byte range of an array as a differently-typed array. `tpu-inference`
  uses `donate_argnums` for token buffers and passes `donate=False` when
  creating KV caches — and still needs Strategy 2 for Mamba and Strategy 3 for
  DSv4.
- **Copying `tpu-inference`'s implementation.** Their allocator builds
  `jnp.empty` arrays; ours builds `torch` tensors on the TPU device. The
  allocation code doesn't port. What does port is the *plan* — which layer
  hosts which, and why — and we already took that; our
  `_plan_ds_v4_swa_overlay` mirrors their `_initialize_ds_v4_kv_cache` rules.
  There is also nothing better to copy: they have the same TODO and the same
  per-layer duplication for Mamba.
- **One big allocation plus index arithmetic.** Fine when every layer in the
  group shares a dtype and page geometry — which is exactly what's violated in
  the cases that need help.

A real fix is one of:

- a TPU-side buffer abstraction supporting typed views into a shared
  allocation. This is a `jax.Array` limitation, so not ours to make, and it's a
  long ask on someone else's roadmap.
- **a backend hook in vLLM** letting a platform declare "I cannot share
  tensors, account for these layers separately", turning the block-ID
  compensation currently hand-rolled for Qwen3.5 into a supported path. This is
  the one worth pursuing: small, well-scoped, clearly a platform-capability
  question rather than a TPU hack, and both TPU stacks would use it
  immediately. Note neither stack can land it alone, which is probably why
  neither has.

Cheapest thing available right now, independent of either: **make the
invariants checkable.** `tpu-inference` asserts that the compressor's two layer
names resolve to the same array index. We have a `data_ptr()` guard for that
case, but our SWA overlay's cross-group block-ID disjointness is still an
assumption in a comment. Turning it into an assert converts a
silent-corruption class into a loud failure.

Until something lands, expect each new model with a `shared_by` group to need
its own handling, and expect the failure mode to be silent — clipped block IDs
and collapsed state rather than a crash.
