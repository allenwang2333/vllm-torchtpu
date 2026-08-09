# DSv4 MMLU-Pro bring-up log (2026-08-08)

Bringing DeepSeek-V4-Flash up on the online serving path to get an MMLU-Pro
baseline against tpu-inference, after the zero-MoE-weight fix (`665006c8`).

Scripts: `scripts/dsv4/` (`dsv4_env.sh`, `serve_dsv4.sh`, `run_mmlu_pro.sh`).
Reference-command mapping and the verified/unverified split: `scripts/dsv4/README.md`.

## Status

- **Online serving path works.** The server reached healthy at
  `max_model_len=1024`, `max_num_seqs=128`, chunked prefill, non-eager, TP=8 --
  the first end-to-end server run since the MoE fix.
- **ROOT CAUSE of the 9216 failure: we allocated one KV array per layer where
  vLLM budgeted one per `KVCacheTensor`** (~2.08x over). Fixed; see Finding 0.
- Finding 1's KV budget over-estimate is real but was *not* what blocked 9216.
  Raising `gpu_memory_utilization` to clear it made things worse, because the
  pool grows with utilization and so does the 2.08x overshoot.

## Finding 0 (THE root cause): per-layer duplication of shared KV tensors

`initialize_kv_cache` (`runner/tpu_runner.py`) allocated a separate array for
every layer in `kv_cache_tensor.shared_by`. vLLM derives `num_blocks` from
`kv_cache_tensor.size`, which budgets **one** array for that whole set, so each
extra layer spends the same budget again: 12.66 MiB/block allocated against
6.09 MiB/block assumed.

vLLM always sizes the pool to consume available memory, so the overshoot scales
with the pool. That is why:

- lowering `gpu_memory_utilization` never helped (smaller cap, same 2.08x);
- lowering `max_num_seqs` never helped (it *raised* the pool -- arm 3 got a
  bigger pool, 737,786 tokens, and failed the same way);
- `max_model_len=1024` survived (2x of a small pool still fits total HBM) while
  9216 died in `.to(self.device)` with `Attempting to allocate 910.62M`.

**Fix:** one array per `KVCacheTensor`, shared layers aliased to it, keyed on
(shape, dtype) so mismatched geometry allocates instead of silently aliasing.
Mamba keeps per-layer duplication (per-layer states by construction, and its
`num_blocks` comes from the summed group page size). This is what the reference
does -- `tpu_inference/runner/kv_cache_manager.py:914`, `if j == 0 or
duplicate_shared_layers`, where duplication is a Mamba-only workaround.

**This was already documented in a comment at the exact failing line**, dated
2026-08-07, with the ratio, the symptom, the reference's approach, and the note
that fixing it "is what makes a >2048-token context testable here." It was
rediscovered the slow way. When a traceback names a line in our own code, read
that line first.

## Finding 1: vLLM's DSv4 KV budget check over-estimates by ~62x

`vllm serve ... --max-model-len=9216` refused to start:

```
ValueError: To serve at least one request with the model's max seq len (9216),
(42.86 GiB KV cache is needed, which is larger than the available KV cache
memory (37.64 GiB).
```

The demand is wrong. Summing what vLLM bills each spec for one max-length
request (`spec.max_memory_usage_bytes(vllm_config)`, the same quantity the
check is built from) gives:

| max_model_len | billed total | check demanded | ratio |
|---|---|---|---|
| 1024 | 0.45 GiB | ~27.9 GiB | 62.1x |
| 9216 | 0.69 GiB | 42.86 GiB | 62.1x |

The multiplier is constant; only the context length moved it across the
available-memory line, which is why 1024 starts and 9216 does not.

**Cause** -- `_max_memory_usage_bytes_from_groups` (`vllm/v1/core/kv_cache_utils.py`)
has a branch whose comment reads *"Special case (only DeepseekV4 for now): all
groups are UniformTypeKVCacheSpecs"*. It bills every group for the **global
maximum** layer-tuple count times the **sum of all page sizes in a tuple**, then
sums over groups:

```python
num_layer_tuples = max(g.kv_cache_spec.get_num_layer_tuples() for g in kv_cache_groups)
layer_tuple_bytes = sum(full_mla_spec.get_page_sizes())
g_bytes = num_layer_tuples * g.max_memory_usage_pages(vllm_config) * layer_tuple_bytes
```

With DSv4's 172 specs across many groups (4 per CSA layer, 3 per HCA layer),
that product is ~62x the real requirement.

**This is a startup guard only.** The engine allocates from available memory
regardless -- the 1024 run allocated 68.5 GiB of blocks against a 0.45 GiB
per-request need. Raising `gpu_memory_utilization` to clear the guard is
therefore safe, not a fudge.

**Open question, worth answering before reporting any perf number:** the
reference runs this at `gpu_memory_utilization=0.4` and does not trip the
check, so tpu-inference must produce a different grouping (fewer groups, or
not all `UniformTypeKVCacheSpecs`). Whether our grouping shape is wrong -- or
merely different -- is not yet established. Do not assume our specs are correct
just because the real billing is small.

## Finding 2: per-spec KV geometry, for reference

At `max_model_len=1024`, `max_in_flight_tokens=2048`, `cache block_size=1024`:

| spec | count | block_size | storage | head | window | page | billed/layer |
|---|---|---|---|---|---|---|---|
| `attn` (uncompressed, ratio 0 layers) | 2 | 1024 | 1024 | 640 | - | 655360 | 655360 |
| `attn` (CSA, ratio 4) | 21 | 1024 | 256 | 640 | - | 163840 | 163840 |
| `attn` (HCA, ratio 128) | 20 | 1024 | 8 | 640 | - | 5120 | 5120 |
| `attn.swa_cache` | 43 | 128 | 128 | 640 | 128 | 163840 | 1474560 |
| `attn.compressor.state_cache` (CSA) | 21 | 16 | 16 | 8192 | 8 | 163840 | 10649600 |
| `attn.compressor.state_cache` (HCA) | 20 | **1** | 1 | 4096 | 128 | 5120 | 5248000 |
| `attn.indexer.compressor.state_cache` | 21 | 16 | 16 | 2048 | 8 | 65536 | 4259840 |
| `attn.indexer.k_cache` | 21 | 1024 | 256 | 256 | - | 65536 | 65536 |

The compressor state caches dominate, for a non-obvious reason:
`SlidingWindowSpec` bills `cdiv(min(window-1+in_flight, max_model_len),
block_size) + 1` blocks, so a `block_size=1` spec (HCA) is billed roughly
**per token**, not per page.

## Note: `dsv4_kill_server` killed its own caller (self-inflicted, same session)

Not a discovery -- a bug introduced by these scripts earlier the same session
and fixed an hour later. Recorded only because the failure mode is worth
recognising, not as a finding.

The cleanup used `pkill -9 -f "vllm serve"`. Any caller whose command line
contains that string -- a wrapper that echoes the command, a shell invoked with
it inline -- matches its own pattern, so the cleanup killed the process running
it. Observed twice: a probe died seconds after launch with a truncated log and
no error. The same trap bites interactively (`pkill -f "VLLM::"` typed into a
shell whose own command line contains `VLLM::`); use `[V]LLM::` there.

Fixed in `scripts/dsv4/dsv4_env.sh`: the kill walks the match list and skips
this process and its ancestors.

The one genuine finding in this document is Finding 1 (upstream vLLM's ~62x
KV budget over-estimate). Finding 2 is reference geometry. The tooling change
below is an extension of a dump that already existed, not a new capability.

## Tooling added

`DSV4_DUMP_KV_PLAN=1` now also dumps, from `TPUModelRunner.get_kv_cache_spec`
(`runner/tpu_runner.py`), each spec's `page_size_bytes`, `page_size_padded`,
`sliding_window`, exemption status, and **the bytes vLLM bills it for one
max-length request**, plus a total.

It runs there, not only in `initialize_kv_cache`, because vLLM's budget check
sits between the two -- so on exactly the configs that fail to start, the old
dump never executed and the specs stayed invisible. That gap cost this session
two runs of blind estimation.
