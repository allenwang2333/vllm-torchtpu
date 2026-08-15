# DeepSeek-V4 native model

DeepSeek-V4 now runs from a real model module registered in vLLM's
`ModelRegistry` instead of a context manager that monkey-patched upstream AMD
classes at load time. This follows the pattern `kimi_k3` already uses in this
repo, and mirrors the reference implementation in
`tpu_inference/models/vllm/experimental/deepseek_v4.py`.

## What changed

`src/vllm_torchtpu/models/vllm/deepseek_v4/model.py` holds the model, decoder
layers, MoE, MLP and head. `register_models()` points
`DeepseekV4ForCausalLM` at it, and `deepseek_v4_patch.py` is gone along with the
runner's `_maybe_patch_for_deepseek_v4` wrapper around `load_model`.

`VllmDeepseekV4MLAAttention` now subclasses the generic
`DeepseekV4Attention` ABC rather than `DeepseekV4ROCMAiterMLAAttention`, so the
AMD module is no longer a base-class dependency.

Each decoder layer closes its own residual stream. `forward` takes
`(x, positions, input_ids)` and returns one tensor, instead of threading
`post_mix`/`res_mix`/`residual` through to the next layer and into the head.
This matches the path the reference actually calls,
`_forward_unfused_post_pre`. The reference keeps a 4-tuple return so a fused
post+pre op can be swapped in; we return a bare tensor, so restoring
`MHCFusedPostPreOp` later would mean restoring the wider signature and its
`support_torch_compile` dims.

Expert weights are recorded under their mapped name only when the loader
reports success, rather than unconditionally after the mapping loop.

## Attention requantization is required

`attention_requantizer_generator` rebases DSv4 attention weights from
128x128 block scales to per-channel FP8 scales. Its only call site used to be
inside the patch module's load path, so removing that module drops it — and
`_o_proj` reads `wo_a.weight_scale` as a 1-D per-channel vector. The native
`load_weights` therefore calls it directly.

tpu-inference has no equivalent, which gxd3 raised on #462 as worth revisiting.
The deviation is carried with a TODO. Settling it means comparing our scale
handling against the reference and deciding which side moves.

## What the patch module did that the model does not

Three behaviors were dropped deliberately, each checked:

- The `update_param_tp_status()` sweep over `named_modules()` — vLLM performs
  the same sweep in `model_executor/model_loader/utils.py:119`.
- The `FusedMoEParallelConfig.make` override — our `FusedMoE(...)` construction
  is byte-identical to the reference's, which uses no such override.
- A post-load `dist.barrier()`, `sync.synchronize(wait=True)` and
  `gc.collect()` — the reference has none of these.

One is worth tracking: `patch_cuda_stubs()` was called from the patch module and
is now called nowhere, leaving `deepseek_v4_stubs.py` defined but uninstalled.
No eager run has needed it. The module is kept rather than deleted.

## Verification

Hardware gate on tpu7x-8: TP=8, EP=8, `max_model_len=1024`, `max_num_seqs=16`,
`gpu_memory_utilization=0.48`, eager. 26 prompt sweeps, zero tracebacks, zero
raises, and the 81-token in-window needle control returns its code exactly.

`ruff check --select F821,F841,F401,F811` is clean over the model package, the
runner and the tests.

Everything above ran with `VLLM_ENFORCE_EAGER=1`. The `support_torch_compile`
`dynamic_arg_dims` changes and the `PiecewiseBackend` bucket-dispatch fix in
`src/vllm_torchtpu/__init__.py` are therefore unexercised, as is the removal of
the `torch._dynamo` metrics-logging bypass the patch module carried.

## Open items

`model.py` raises the interpreter recursion limit to 50000 before tracing,
carried over from #464 where the comment reads "tracing this model exceeds the
default 1000 frames". That was measured against the patch-based architecture,
whose monkey-patched call stacks were deeper than the native model's. Whether
the native model still needs it is unverified. To settle it: drop the
`sys.setrecursionlimit` call, run the compile gate, and see whether tracing
raises `RecursionError`. If it does, record the depth it actually needs instead
of keeping a round number.

`deepseek_v4_stubs.py` defines `patch_cuda_stubs()` but nothing calls it now
that the patch module is gone. Either the native model needs no CUDA stubs, in
which case the module can go, or a path exists that has not been exercised.

## Prior MMLU measurement

An earlier run of this code, before the in-layer mixing change, scored 0.2390
+/- 0.0036 exact match on `mmlu_llama` across all 57 subjects with
`--kv-cache-dtype fp8`, completing without crashes. On four-option questions
that is close to chance. The likely explanation is strict-match parsing against
free-form chat output rather than model degradation, but that is an inference:
the run used no `--log_samples`, so no generation was inspected. Re-running with
`--log_samples`, or with a flexible-extract filter, would settle it.
