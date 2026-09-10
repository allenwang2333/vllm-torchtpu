# Qwen3.5 routing: captured performance and native code

This branch contains historical evidence for the opt-in 128-route stable
ordering/inverse implementation. It is separate from the implementation branch
so the full native dumps do not enter the implementation PR or main history.
The artifacts describe the frozen captured revisions, before the feature flag
was added and the source was ported to current upstream. They do not benchmark
the flag-gated upstream port.

## Serving performance

Qwen/Qwen3.5-35B-A3B, eight TPUv6e chips, TP8, concurrency one, 512 input tokens
and 256 output tokens, CUSTOM batched-RPA attention, prefix caching disabled.

Ten fresh confirmation pairs: **1512.167191 ms to 1469.135896 ms**, a **2.845670%
reduction** in mean HTTP end-to-end request latency. The 95% bootstrap interval
for the mean paired candidate/baseline ratio is [0.957763172, 0.987362437].
That interval does not establish a reduction greater than 2% throughout it.

Each role used a fresh server and compile cache, ten warmups and 30 timed
requests. The three pilot pairs are reported separately. Exact output-token
replays were performed separately from timing. The final audit records source,
checkpoint and same-role native-program identity checks. Distributed device
critical-path timing and formal Phase-5 model acceptance remain unestablished.

- [Per-request timings](serving-request-latencies.csv): all 780 timed requests.
- [Summary and runtime pins](summary.json).
- [Final closeout audit](serving-closeout-audit.json).
- [Independent statistics audit](serving-statistics-audit.json).
- [Exact captured patch](captured-candidate.patch).

## Before and after LLO

| Scope | Before | After |
| --- | --- | --- |
| Full changed model decode program | [Compressed LLO](model-decode-baseline.llo.gz) | [Compressed LLO](model-decode-candidate.llo.gz) |
| Resolved model routing routine | [Sort body](model-routing-baseline-body.llo) | [Joint body](model-routing-candidate-body.llo) |
| Complete local preparation | [LLO](preparation-baseline.llo) | [LLO](preparation-candidate.llo) |

The model baseline invokes the same 342-record sort body twice per layer.
The candidate invokes one 129-record joint body. Across 40 MoE layers, 80 sort
calls become 40 joint calls. The other 18 active programs retain identical LLO.
Both joint outputs have live consumers.

| Instruction inventory per routing operation | Two baseline visits | One joint visit |
| --- | ---: | ---: |
| Numbered instruction records | 684 | 129 |
| `vlt.8x128.s32` | 224 | 32 |
| `vsel.8x128` | 672 | 21 |
| `vperm.lane.8x128` | 112 | 0 |
| `vadd.8x128.s32` | 0 | 36 |

The baseline column accounts for two visits to a shared routine; it does not
represent two static code copies. These are instruction inventories, not
cycle measurements. [Native review](model-native-review.json) records exact
callee resolution and half-open numbered-record ranges. The full model dumps
are approximately 202 MB each before gzip compression.

The local preparation comparison uses the intermediate first-sort plus
inverse-helper baseline, rather than the original two-sort model baseline.
Its ten-pair result is **9.648066 us to 7.204613 us**, a **25.325832% reduction**.
It covers all six outputs at hidden[16,2048], routes[16,8], 256 experts, no EP,
no SparseCore and synthetic seed 1729. See the
[local confirmation](preparation-confirmation.json) and
[lowering review](preparation-lowering-review.json).

## Verification and scope

With NumPy installed, run:

```bash
python verify.py
```

The verifier checks all hashes in [manifest.json](manifest.json), including
decompressed full LLO hashes, recounts native instructions in both extracted
bodies, and recomputes pilot/confirmation statistics from every timing sample.
The manifest binds the measured artifacts and reports; this README and verifier
are versioned by Git. Captured artifacts are excluded from formatters to retain
their original bytes.

Original XPlane traces, model weights, output tensors, token replay payloads,
and serialized executables are not included. Their recorded hashes and audit
results are retained. Verification of this bundle does not independently
reconstruct the source-to-profile identity joins or establish a new speedup.
