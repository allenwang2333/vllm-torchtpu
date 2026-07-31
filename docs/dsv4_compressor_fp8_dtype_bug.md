# DeepSeek-V4 on TPU: the compressor fp8/uint8 dtype bug

**Status: root cause found, fixed, and verified (2026-07-31).**

Single authoritative write-up of the bug that made DeepSeek-V4-Flash emit one
correct token and then collapse. Supersedes the framing in
`docs/bisection_progress_report.md` §7 and §8 (kept for their measurements and
for the record of what was refuted).

---

## 1. Symptom

Prompt: `"Compare XLA and LLVM in two bullet points."`

```
Generated token IDs: [9544, 0, 0, 0, 0, 0, 0, 0, ...]
Generated text: 'Here'
```

The first token (`'Here'`, logprob 0.0) matched the JAX reference exactly; every
token after it was id `0`. Not a gradual degradation into gibberish — a hard
collapse.

Bisected to **layer 2's decode-step attention output blowing up to ~1e29**,
while its input and (apparently) the KV cache it read were both clean.

---

## 2. Root cause

vLLM allocates the KV and compressor-state caches as `kv_cache_dtype`, i.e.
**`torch.float8_e4m3fn`**. Every compressor kernel treats them as raw **byte
buffers**:

- `compress_norm_rope_store` packs `[fp8 nope | bf16 rope | e8m0 scale]` into a
  `uint8` record and scatters it with `flat.at[dest].set(record)`.
- `pack_state_cache` writes f32 state through `_to_byte_lane(...)`, also `uint8`.

**A JAX `.at[...].set(uint8_values)` into an fp8-typed array performs a numeric
conversion, not a bitcast.** Byte `108` becomes the fp8 *number* `108.0`, which
has an entirely different bit pattern. Demonstrated directly:

```python
jnp.zeros((4,), jnp.float8_e4m3fn).at[:].set(jnp.asarray([108,119,112,109], jnp.uint8))
#   -> reads back as bytes [110, 111, 110, 110]      BYTES DESTROYED
jnp.zeros((4,), jnp.uint8).at[:].set(...)
#   -> reads back as bytes [108, 119, 112, 109]      correct
```

`layers/vllm/custom_ops/deepseek_v4/deepseek_v4_compressor.py`'s `forward`
passed `self.k_cache.kv_cache` and `self.state_cache.kv_cache` **raw** into the
op. Every record the compressor ever wrote was corrupted on write.

The attention path never hit this because it routes through
`_as_kernel_cache_view()` (`deepseek_v4_attention.py`), which applies exactly
the missing `.view(torch.uint8)`. **The compressor was the only path that handed
raw `kv_cache_dtype`-typed tensors to a byte-level kernel.**

### 2.1 Causal chain (every link independently measured)

| # | step | measured |
|---|---|---|
| 1 | compressor writes packed byte records, numerically cast into fp8 storage | bit patterns destroyed |
| 2 | state cache decodes back as garbage | `win_absmax = 7.87e+31` |
| 3 | RMSNorm `variance = mean(x²)` ≈ 6e63 | **overflows float32 → inf** |
| 4 | `rsqrt(inf) = 0` → `normed = x·0 = 0` | `compressed` nope=0, rope=0 |
| 5 | `quantize_fp8_ue8m0(zeros)` → empty record | `q` 0/448, scale byte 105 |
| 6 | empty record scattered over the healthy row 259 | `nz` 448/128/7 → 0/5/7 |
| 7 | layer 2 decode attention reads it | ~1e29 blowup → garbled text |

---

## 3. The fix

```python
# deepseek_v4_compressor.py :: VllmDeepseekCompressor.forward
orig_cache = self.k_cache.kv_cache
state_cache_buf = self.state_cache.kv_cache
if orig_cache is not None and orig_cache.dtype != torch.uint8:
    orig_cache = orig_cache.view(torch.uint8)
if state_cache_buf is not None and state_cache_buf.dtype != torch.uint8:
    state_cache_buf = state_cache_buf.view(torch.uint8)
```

`.view()` shares storage, so the later `copy_` write-backs still target the
memory the caches are bound to. This mirrors `_as_kernel_cache_view()`.

---

## 4. Verification

| metric | before | after |
|---|---|---|
| gather window absmax (pos 15) | `7.87e+31` | `3.736` |
| `compressed` nope / rope | `0 / 0` | `2.396 / 3.351` |
| quantised nonzero | `0/448` | `448/448` |
| scale bytes | `[105,…]` | `[120,120,120,120,120,120,120]` |
| slot 3 (row 259) | `nope 0, rope 1.585e+29, nz 0/5/7` | `nope 320, rope 3.344, nz 448/128/7` |
| generation | `[9544, 0, 0, 0, …]` | fluent English, 32 tokens |

Post-fix output:

> `Here are two bullet points comparing the two given entities:\n\n**Entity 1: "The quick brown fox jumps over the fox""\nEntity 21`

**Kernel suite: 27/27 pass** (`test_dsv4_mla`, `test_dsv4_mla_swa`,
`test_dsv4_compressor`, `test_dsv4_moe_routing`, `test_dsv4_streamindex_topk`).

The 3 `test_mxfp4_*` failures are **pre-existing** —
`TypeError: FusedMoEConfig.__init__() missing 1 required positional argument:
'intermediate_size'`, a vLLM-API/test-harness mismatch — confirmed identical
with the fix stashed (`git stash` A/B).

---

## 5. Why this took so long to find

### 5.1 The damage was continuously masked

Layer 2's **SWA cache is the same tensor** as its compressed-KV cache
(`_get_active_sw_cache()` falls through to `self.kv_cache` because
`model.layers.2.attn.swa_cache` has no KV-cache group), and `forward_mqa` ends
with `orig_sw_cache.copy_(new_sw_cache)` over the whole buffer. The SWA kernel
*does* get a correct uint8 view, so it kept overwriting the compressor's damage
with valid data.

Compressed slot 3 (row 259) was simply the one row read in the window between
corruption and masking — which made a **global** bug look like a single
anomalous row, and sent the investigation hunting for what was special about
slot 3. Nothing was.

### 5.2 Every isolated test passed

| component | verdict | how tested |
|---|---|---|
| `kv_score` (op input) | healthy | absmax 3.03, 0 NaN/Inf, 2048/2048 nonzero |
| `gather_state_windows` | correct | 8/8 valid window rows |
| `compress_norm_rope` | correct | nope 4.54, rope 3.32 |
| `compress_norm_rope_store` | correct | **byte-identical on CPU and TPU** |
| `pack`/`unpack` round-trip | lossless | exact, incl. via `save_partial_states` |

Each test allocated **its own `uint8` array**, so none could see the defect. The
bug lived only in the dtype of the buffer handed in at the call site.

> **Transferable rule: when every component passes in isolation but the
> integration fails, suspect the dtype/identity/aliasing of the buffers supplied
> at the call site, not the math.**

### 5.3 Decoded-value probes repeatedly misled; byte probes with a control cracked it

- `rope_absmax = 1.5845632502852868e+29` is **exactly 2^97** (bf16 `0x7000`).
  Random corruption does not land on an exact power of two — that was the first
  real hint the bytes were structured, not "garbage data".
- A state row reading `~6120/8192` nonzero bytes was interpreted as "one of four
  head-blocks is dead (¾ ratio)". It was actually
  `(2·1024 + 2·2048)/8192` — a **bf16-in-f32 storage artifact**, since kv blocks
  hold bf16-precision values whose low two mantissa bytes are zero.
- The probe that finally worked dumped **raw byte-region populations plus a
  never-written control row**, making "written but empty" distinguishable from
  "never written" at a glance.

### 5.4 Hypotheses raised and refuted (do not re-chase)

| hypothesis | how it died |
|---|---|
| attention-sinks epilogue `exp()` | refuted *by construction*: `mla.py` clamps `L = max(L, 1.0)` before `out = acc/L`, so `\|out\| ≤ \|acc\|` — the epilogue can only shrink the output |
| `pl.debug_print` needed unblocking | never needed; still non-functional, still uninvestigated |
| `tp_size` double-division | real latent bug but **inert** — the per-worker mesh is single-chip, so `tp_size == 1` |
| second writer / SWA-overlay corrupting the row | refuted by adding `rope_absmax` to the compressor's own probe: value already present in the op's return |
| `interleaved_rope` allocate-then-scatter | reverted to the reference form; output **bit-identical** with both caches cleared → a true no-op |
| `unpack_state_cache` `ndim<4` silent-zeros fallback | never fires; buffer is 5-D, squeezed to 4-D |
| state failing to persist across calls | refuted: positions 12–14 read back at `nz≈6120` |
| "one of four head-blocks is zero" | refuted: the ratio is a bf16-in-f32 artifact |

---

## 6. Operational gotchas discovered along the way

- **Clear BOTH compile caches.** `/dev/shm/torch_tpu_cache/` holds `default`
  (~1.1 GB) **and** `tpu_tier2_cache` (~4.7 GB). Older docs mention only the
  first; clearing just that one silently runs stale kernels. A bit-identical
  result after a real code edit is the tell.
  ```bash
  rm -rf /dev/shm/torch_tpu_cache/default/* /dev/shm/torch_tpu_cache/tpu_tier2_cache/*
  ```
- **Never launch the repro with `nohup … & disown` from a tool-driven shell** —
  workers are killed mid-load and leave stray `/dev/vfio` holders. Run in the
  foreground of a backgrounded task. Check before every run:
  ```bash
  for i in 0 1 2 3 4 5 6 7; do sudo fuser /dev/vfio/$i; done
  ```
- **Never hand-derive a paged-cache address in a probe.** Import the kernel's
  own `_state_chunk_dims` and mirror `unpack_state_cache`'s indexing.
  `slot_mapping` indexes the *state view*
  (`[num_pages, state_block_size, state_dim]`), **not** the raw byte buffer
  (`[num_pages, rows, packing, width]`). A hand-derived address silently read
  the wrong page and produced a convincing but entirely false `nz=0` result.
- **`scratch/` is ~255 GB and is not gitignored.** Never `git add .` here.

---

## 7. Reusable CPU harness (`scratch/cpu_repro/`)

All of these run in **seconds with no TPU** (`JAX_PLATFORMS=cpu PYTHONPATH=src`):

| script | what it decides |
|---|---|
| `a1_quantize_reachability.py` | which `(q_nonzero, scale_byte)` pairs `quantize_fp8_ue8m0` can produce — proved the observed row was not a coherent record |
| `a2_record_roundtrip.py` | record assembly + fp8 byte-lane fidelity |
| `a4_state_roundtrip.py` | `pack_state_cache`/`unpack_state_cache` losslessness at real geometry |
| `a5_store_cpu_vs_tpu.py` | full `compress_norm_rope_store` — run under both `JAX_PLATFORMS=cpu` and `=tpu` to separate math from lowering |

Most ~8-minute full-model runs in this investigation only served to refute a
hypothesis one of these could have refuted in 30 seconds. **Front-load CPU
experiments.**

---

## 8. Still open

1. **Output is fluent but semantically wrong** — it answers about "entities" and
   "the quick brown fox" rather than XLA vs LLVM. The catastrophic failure is
   fixed; an accuracy gap remains.
2. **All prior per-layer divergence numbers are void.** Every measurement in
   `bisection_progress_report.md` was taken while compressed KV was corrupt
   across all CSA layers. Re-bisect from scratch against `tpu-inference`.
3. **Audit for the same dtype bug elsewhere.** Any other call site handing a raw
   `kv_cache` to a kernel that expects packed bytes has this defect, and it is
   invisible until something reads the data back.
4. Reference deviations deliberately left in place, worth revisiting now that
   records are real: `mla.py`'s `L = jnp.maximum(L, 1.0)`;
   `unpack_state_cache`'s `ndim < 4` silent-zeros fallback;
   `two_caches_same_buffer` hardcoded rather than derived.
