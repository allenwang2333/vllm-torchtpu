# DeepSeek-V4 native model: remaining work

The native model branch is done and working: eager and compiled gates both
pass, with the in-window needle control exact.

What is left is one correctness bug, three cleanups found while building it,
and two measurements that were promised but never taken. The whole list costs
two TPU runs.

## 1. Find out which quantization path DSv4 actually takes

Add temporary logging to `_process_fp8_linear_weights`
(`layers/vllm/quantization/fp8.py:219`) recording, per layer prefix, whether
`block_quant` is set, which of `weight_scale` or `weight_scale_inv` exists, and
the final scale shape.

This cannot be settled by reading. The requantizer produces a per-channel
`weight_scale`, while the linear method's block branch requires
`weight_scale_inv`; both cannot hold on a run that works, and the answer decides
whether task 2 is a deletion or a narrowing. The logging rides along with run 1
rather than costing a run of its own.

## 2. Stop requantizing block scales to per-channel

`models/vllm/deepseek_v4_requant.py` rewrites eight attention and shared-expert
projections from 128x128 block scales to one scale per output row, at a measured
2.71% relative Frobenius error on `layers.0.attn.wo_a`. Neither upstream vLLM
nor tpu-inference does this, and it was raised in review on #462.

The fix depends on task 1 and is one of:

- delete the module, if nothing needs a 1-D scale
- narrow it to `wo_a`, the one place that indexes a scale directly
  (`deepseek_v4_attention.py:752`), forced 1-D by its grouped-BMM einsum
- switch DSv4 to the blockwise linear runtime, which already exists behind
  `ENABLE_QUANTIZED_MATMUL_KERNEL` and `REQUANTIZE_BLOCK_SIZE` (`fp8.py:166`)
  and is off by default (`envs.py:21`)

This also closes the TODO at `deepseek_v4_requant.py:16`.

## 3. Drop the recursion limit if it is no longer needed

`model.py:751` raises the interpreter recursion limit to 50000, inherited from
the compile-enablement PR where it was measured against the patch-based
architecture and its deeper monkey-patched frames. Remove it and let run 1
answer; if it was still load-bearing, restore it with the measured depth in the
comment rather than a round number. Only a compiled run tests this, because
eager never traces.

## 4. Stop the legacy cache filling RAM

`examples/run_dsv4_inference.py:6` defaults `VLLM_FP4_CACHE_DIR` to a
`/dev/shm` path, which `quant_weight_cache.py:48` honours as a legacy alias, so
any run that does not set `VLLM_QUANT_WEIGHT_CACHE_DIR` builds its cache in RAM.
The last one reached 162 GB. Point it at disk here and in the two gate scripts
that set the same variable.

## 5. Remove dead references

Three vestigial `ROCMAiter` references remain in `deepseek_v4_attention.py`,
including a registry assignment that the native model makes pointless.

## Runs

Run 1 carries tasks 1, 3, 4 and 5 together under the compiled gate. It passes
on `tracebacks=0 sweeps=13 raises=0`, an exact needle control, and generations
identical to the previous compiled run, since none of those changes should move
numerics. It doubles as the warm-cache test still owed: the quantized-weight
cache has so far only ever been written, never read, so confirm hits and compare
startup against the cold-cache baseline.

Run 2 carries task 2 under the same gate. Generations are expected to move here,
because the weights genuinely change; the needle control must stay exact and the
continuations coherent.

Before each run, `ruff check --select F821,F841,F401,F811` over the changed
files and the two unit-test files under `tests/layers` and `tests/models/vllm`.

## Open question

`deepseek_v4_stubs.py` defines `patch_cuda_stubs()`, which has had no caller
since the patch module was removed. Either the native model needs no CUDA stubs
and the module can go, or a path exists that no run has exercised.

## Deferred

DP attention needs its own planning pass. It is a sharding change through the
runner and KV cache manager rather than a model edit, and tpu-inference requires
it for MLA models. It also unblocks the first genuine ours-vs-reference
comparison; every "identical" result recorded so far compares us against
ourselves.

An MMLU run with `--log_samples` would settle whether the ~24% strict-match
score reflects parsing or the model. It is one serving run, independent of the
work above.

Splitting the package to match `kimi_k3`'s layout, with `layers.py` and `moe.py`
carved out of `model.py`, is cosmetic and would need a re-gate.
