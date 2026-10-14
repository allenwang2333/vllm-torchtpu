# DeepSeek-V4 decode integration snapshot

This directory accompanies the adopted optimization set validated on 2026-10-13.
It preserves the process-local huge-page staging hook separately from model code.
The integration includes mHC, Indexer/TopK, Q/KV preprocessing, the single-token
Wq_b MXU path, CSA/HCA composition and cache fixes, and resident output projection.
Existing guarded fallback paths and the disabled CSA FP8 compute option remain.
The later experimental HCA Wq_b output-layout candidate is excluded.

## Source provenance

The frozen service tree has 72 changed or added files relative to
`2fac4ce4f355a96447f568679d09b0457465a2a5`. Unmodified files match that base.
`frozen_source_sha256.json` records those 72 original file hashes before migration
onto current main. The integration preserves newer upstream changes through a
three-way application; hashes are provenance, not a promise that every migrated
file is byte-identical. Public-path documentation and formatting may also differ.

## Main integration and local checks

The migration is based on main commit
`ea114d4a7ce76ac28b53f51749a9d1dc8266540c`. Eight source files overlapped with
upstream changes. The merge retains upstream CSA tile selection, Indexer chunk
configuration and prefill tiles, and weight-loading buffer fixes. The Indexer
adapter leaves request-group selection to the decode dispatcher to avoid passing
`decode_req_batch_size` twice.

Local checks passed for all 72 changed Python files: Ruff 0.14.0 formatting and
lint, syntax parsing, and Apache license headers. The repository's environment
read and B023 baseline checks passed, as did seven CPU benchmark-script tests.
Five additional host-only dispatch cases exercised dense decode, generic and
chunked fallbacks using extracted functions with a simulated conditional; they
do not validate JAX tracing or TPU kernels. Full pre-commit/type checking, TPU
tests and full-service performance were not rerun for this main integration.

## Huge-page staging hook

On Linux, set an existing writable audit directory and prepend the hook directory
to `PYTHONPATH` before launching the server:

```bash
mkdir -p /path/to/staging-audit
export DSV4_STAGE_AUDIT_DIR=/path/to/staging-audit
export PYTHONPATH="$PWD/scripts/deepseek_v4_decode/hugepage_hook:${PYTHONPATH:-}"
```

The hook uses 2 MiB-aligned anonymous memory and `MADV_HUGEPAGE` for the existing
CPU block-table staging buffer. It checks `/proc/self/smaps` and fails if the
required huge-page allocation is unavailable. It changes allocation only; the
normal staging function still populates and uploads the block tables. This is
an opt-in reproduction hook, not a new automatic model default. It requires the
runner's `stage_block_table_uploads` interface and Linux transparent huge pages.
The full-service run recorded 8 MiB of anonymous huge pages per rank.

## Historical validation

The original frozen tree completed 158 targeted TPU pytest cases, 1,752 exact
metadata comparisons, 272 exact QKV/compressor/attention combinations, and four
64-step full-cache comparisons. Earlier component validation is separate and
must not be added as a count of unique test cases.

The full service used 43 layers, TP1/DP8/EP, FP4 MoE block size 512, FP8 E4M3 KV,
max model length 66,560, max batched tokens 4,096, max sequences 256, memory
utilization 0.90, async scheduling, chunked prefill, and disabled prefix caching.
MTP/speculative decoding was disabled. Requests contained exactly 65,536 input
and 1,024 output tokens. C1 and C8 each ran three rounds without a profiler.

TPOT is `(last_token_time - first_token_time) / (output_tokens - 1)` per request,
excluding TTFT. Historical means were C1 8.787100719 ms and C8 9.290572064 ms.
These are historical frozen-service results, not measurements of the new main
integration. There was no qualifying 10-second full-concurrency window.

Numerical limitations:

- In the original 160 Wq_b weight/input cases, 116 projections were bitwise exact;
  the others were within one BF16 ULP in that sample set. This is not a global
  one-ULP guarantee. Later expanded testing found an inherited near-zero outlier
  exceeding one BF16 ULP while satisfying the tested FP32 forward-error bound.
- Historical formal outputs differed from the prior service in 16 of 27
  requests. Kernel comparisons and successful serving do not establish model
  quality. No complete model-quality evaluation is claimed.
- Current-main integration requires its own TPU validation before historical
  numerical or performance conclusions can be extended to it.

Raw traces, model weights, prompt payloads, machine-specific paths and runtime
logs are retained with the original experiments and are not part of this source
branch. No TPU service is started by importing these documentation files.
