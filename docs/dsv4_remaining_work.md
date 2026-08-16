# DeepSeek-V4 native model: remaining work

The native model branch is done and working: eager and compiled gates both
pass, with the in-window needle control exact.

What is left is one correctness bug, four cleanups, and two measurements that
were promised but never taken. The whole list costs two TPU runs.

Each task below is written to be handed to one agent on its own. An agent needs
this header plus its own section; it does not need the other sections.

Status: tasks 1, 3, 4, 5 and 6 are applied in the working tree and ruff is
clean; they are waiting on run 1. Task 2 is still blocked on run 1's log.

## Common to every task

The TPU is exclusive, not off limits. The machine has one set of eight chips, so
one run at a time — check nothing is already running before starting a gate, and
never start `pytest` while a server is up, since it takes the chips out from
under it. A gate is 25-35 minutes; use `setsid`, a silence watchdog and
`chain_progress.log` for anything long.

That exclusivity is why the tasks are batched into two runs rather than gated one
at a time. Parallel agents must not each start a gate: make every edit, verify
statically, then one chain covers the lot.

Static check before handing back, over the files you changed:

```
ruff check --select F821,F841,F401,F811 <changed files> \
    tests/layers/test_quant_weight_cache.py tests/models/vllm/test_dsv4_model.py
```

`py_compile` is not a substitute — it has passed over a deleted call before.

The gate is `scratchpad/gate_dsv4_compile.sh <tree> <tag>`. It passes on
`tracebacks=0 sweeps=13 raises=0` in `scratchpad/chain_progress.log`, an exact
needle control, and coherent continuations. The baseline to diff generations
against is `scratchpad/gate_cache_compile.sub`, the most recent passing compiled
gate (`chain_progress.log`, 21:40:10, `sweeps=13 compile_lines=72`).

Run 1 carries tasks 1, 3, 4, 5 and 6 together. None of them should move
numerics, so run 1 passes only if generations are identical to the baseline. It
doubles as the warm-cache measurement still owed: the quantized-weight cache has
so far only ever been written, never read, so confirm cache hits in the log and
compare startup time against the cold-cache baseline.

Run 2 carries task 2 alone. Generations are expected to move there, because the
weights genuinely change. The needle control must stay exact.

Two files have more than one task on them. Tasks 1 and 2 both touch
`layers/vllm/quantization/fp8.py`; tasks 2 and 5 both touch
`layers/vllm/custom_ops/deepseek_v4/deepseek_v4_attention.py`. Task 2 runs after
run 1 anyway, so sequence it last on both.

Paths are relative to `src/vllm_torchtpu/` unless they start with `examples/`,
`tests/`, `scratchpad/` or `scratch/`.

## 1. Find out which quantization path DSv4 actually takes

Temporary logging only. No behavior change. Rides along with run 1, so it costs
no run of its own, and comes back out once the answer is recorded.

The requantizer emits `f"{base_key}.weight_scale"` for every projection it
touches and consumes `.weight_scale_inv` if that is what arrived
(`models/vllm/deepseek_v4_requant.py:79-93`). The block branch of
`_dequantize_fp8_linear` opens with `assert weight_scale_inv is not None`
(`fp8.py:103`), reached only when `block_quant` is set. The gates pass without
that assert firing, so one of these holds, and reading cannot say which
(inferred):

- `block_quant` is false for DSv4, and these layers take the per-channel branch
  at `fp8.py:118`
- `is_checkpoint_fp8_serialized` is false, so `process_weights_after_loading`
  takes its `else` branch at `fp8.py:986` and never calls
  `_process_fp8_linear_weights` at all
- the requantized projections are not vLLM linear layers and never reach this
  linear method

Add two log lines, both `logger.info` and not `info_once`, since one line per
layer is the point:

- at the top of `process_weights_after_loading` (`fp8.py:974`), before the
  `is_checkpoint_fp8_serialized` test: the layer identity, that flag, and
  `self.block_quant`
- at the top of `_process_fp8_linear_weights` (`fp8.py:219`): the layer
  identity, `block_quant`, whether each of `weight_scale` and
  `weight_scale_inv` exists on the layer, and the shape of whichever does

The first line is what makes absence meaningful: a projection that logs nothing
at all is the third outcome above. That holds even though run 1 is the first
warm-cache run — `fp8.py` has no reference to `quant_weight_cache`, so a cache
hit cannot suppress the probe. Only the MoE path is cached.

Identity comes from `getattr(layer, "prefix", None)`, which vLLM's `LinearBase`
sets at `linear.py:269`, with `type(layer).__name__` and the weight shape as the
fallback.

Done when: the two lines are in, ruff is clean, and the section records what to
grep for in the run-1 log. Report the outcome for the requantized projections by
name, not in aggregate. Task 2 takes the logging back out.

## 2. Stop requantizing block scales to per-channel

Blocked on task 1. Do not start this before run 1's log has been read.

`models/vllm/deepseek_v4_requant.py:29-47` dequantizes 128x128 block scales and
requantizes to one scale per output row, at a measured 2.71% relative Frobenius
error on `layers.0.attn.wo_a` (unverified — a measurement from an earlier
session, not reproducible statically). Neither upstream vLLM nor tpu-inference
does this, and it was raised in review on #462. It applies to nine name patterns
(`deepseek_v4_requant.py:54-63`): `wq_a`, `wq_b`, `wo_a`, `wo_b`, `wkv`,
`indexer.wq_b` under `.attn.`, and `w1`, `w2`, `w3` under `.shared_experts.`.
Whether that is nine actual projections per layer is unverified; only a run can
say.

One constraint holds in every branch below. `deepseek_v4_attention.py:752` does
`self.wo_a.weight_scale.view(n_local_groups, o_lora_rank)`, a 2-D view of a 1-D
scale, applied at :758 after the grouped-BMM einsum at :756. `wo_a` needs a
per-channel scale, or that call site changes too.

Pick the branch that run 1's log supports:

- if nothing needs a 1-D scale except `wo_a`, narrow the filter to `wo_a` and
  delete the rest
- if the layers bypass the linear method entirely, work out what does consume
  these scales before deleting anything
- if the blockwise runtime is the right answer, it already exists behind
  `ENABLE_QUANTIZED_MATMUL_KERNEL` and `REQUANTIZE_BLOCK_SIZE`
  (`fp8.py:166-172`), both off unless set in the environment
  (`envs.py:200-205`, `env_bool` defaulting to `False` at :98)

Whichever branch, remove the call in `models/vllm/deepseek_v4/model.py:791` if
the module goes, and close the TODO at `deepseek_v4_requant.py:16`.

Also take out task 1's temporary logging in the same edit. This task is the only
one sequenced after run 1 on `fp8.py`, so it owns the removal.

Done when: the edit is in, task 1's log lines are gone, ruff is clean, and the
section states which branch was taken and on what evidence from the run-1 log.

## 3. Drop the recursion limit if it is no longer needed

`models/vllm/deepseek_v4/model.py:739` calls `sys.setrecursionlimit(50000)` in
`DeepseekV4ForCausalLM.__init__`. It came from the compile-enablement PR, where
it was measured against the patch-based architecture and its deeper
monkey-patched frames. The native model does not have those frames.

Delete the line and let run 1 answer. Only a compiled run tests this, because
eager never traces deeply enough to hit the limit. If it turns out to still be
load-bearing, restore it with the measured depth in the comment rather than a
round number.

Done when: the line is gone, `import sys` is either still used elsewhere in the
file or removed with it (ruff F401 catches this), and ruff is clean.

## 4. Stop the legacy cache filling RAM

`examples/run_dsv4_inference.py:6` does
`os.environ.setdefault("VLLM_FP4_CACHE_DIR", "/dev/shm/dsv4-flash-real-fp4")`.
`quant_weight_cache.py:56-62` resolves the directory by returning the first
non-empty of `VLLM_QUANT_WEIGHT_CACHE_DIR` then `VLLM_FP4_CACHE_DIR`, so the
primary variable wins when both are set. Any run that does not set the primary
one builds its cache in RAM. The last such run reached 162 GB.

`scratchpad/gate_dsv4_compile.sh:32` already sets the primary variable to
`/home/patemotter_google_com/workspace/dsv4_quant_cache`, so the compiled gate
is unaffected. Two scripts and the example are not:

- `examples/run_dsv4_inference.py:6`
- `scratchpad/gate_dsv4.sh:31`
- `scratchpad/run_dsv4_mmlu.sh:19`

Change all three to `setdefault`/`export` `VLLM_QUANT_WEIGHT_CACHE_DIR` at the
disk path above. Do not perpetuate the legacy alias.

Out of scope: seven scripts under `scratch/` set the same `/dev/shm` path
(`run_e2e.py:22`, `eval_mmlu_pro_direct.py:14`, `run_mmlu_limit100.sh`,
`run_mmlu_server_eval.sh`, `run_official_mmlu.sh`, `run_mmlu_eval.sh`,
`run_dsv4_docker.sh:19`). `scratch/run_e2e.py` is a known-bad harness we no
longer use. Leave that directory alone; it is Pate's call.

Done when: the three files point at disk, and no other file in `examples/` or
`scratchpad/` still sets `VLLM_FP4_CACHE_DIR`.

## 5. Delete the dead ROCm patch function

`patch_deepseek_v4_mla_cls` (`deepseek_v4_attention.py:801-812`) has no caller.
A whole-repo `grep -rn "patch_deepseek_v4_mla_cls" --include=*.py` matches only
the definition, and grep would also catch a string reference for a dynamic
lookup.

Delete the function. That removes the `DeepseekV4ROCMAiterMLAAttention`
assignment at :803, the log string at :811, and the dead `DeepseekV4SWACache`
rebinding and `get_min_page_size` fallback at :804-809 — all of which live only
inside it, so none of it changes behavior. Then remove any import that only that
function used (`ruff --select F401`), and check whether
`vllm.models.deepseek_v4.amd.model` is still imported anywhere in the file.

Separately, `deepseek_v4_attention.py:253` has
`"DeepseekV4ROCMAiterMLAAttention"` as one entry in a live class-name tuple that
decides which submodules get `get_kv_cache_spec` overridden (:249-254). It is a
string comparison against `module.__class__.__name__` over `self.modules()`, so
whether the entry ever matches depends on what upstream's `DeepseekV4Attention`
builds as children — not something this repo can settle by reading. Left in
place: it costs one tuple entry, and removing it would edit a validated numerics
path on an unproven argument.

Done: the function is deleted; ruff clean; `:253` left as above.

## 6. Delete the CUDA stubs module

`models/vllm/deepseek_v4_stubs.py` defines `patch_cuda_stubs` (:74) and the
three dummy classes it installs (:23, :38, :56). No production code imports it.
The only importer in the repo is `tests/models/vllm/test_dsv4_model.py:21`,
which calls it at :33 in a test named for it; the dummy classes are referenced
only inside the module itself
(`grep -rn "deepseek_v4_stubs\|DummyDeviceCapability\|DummyCudaStream\|DummyCudaEvent"`).

Delete three things:

- the module `models/vllm/deepseek_v4_stubs.py`
- the import at `tests/models/vllm/test_dsv4_model.py:21`
- the test `test_deepseek_v4_cuda_stubs_patching` (`test_dsv4_model.py:31-33`)

Leave `test_dsv4_model.py` itself. Its other test,
`test_deepseek_v4_model_registration` (:24), is unrelated, and the header's ruff
command names the file.

The one path that touches CUDA events supplies its own:
`deepseek_v4_attention.py:229-260` saves and restores `torch.Event` and a CUDA
event around its patched construction.

Done when: the module and that one test are gone, ruff is clean, and nothing
under `src/` or `tests/` still names the module.

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
