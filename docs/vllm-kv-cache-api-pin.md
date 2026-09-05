# The KV-cache spec API this repo may use

The TPU image installs vLLM from `pyproject.toml`, which pins the upstream tag
`v0.27.0`. A local run usually puts a vLLM *source checkout* on `PYTHONPATH`
instead, and that checkout is normally much newer. The two have incompatibly
different KV-cache spec APIs, so **a cache spec that constructs on a developer
box can raise inside the pod, and nothing on the CPU test rung will notice**.

This page records the two shapes and the rule that follows from them.
`tests/models/vllm/test_vllm_kv_cache_api_pin.py` is the machine-checked half:
change the pin and it fails, pointing here.

## The rule

Write every KV-cache spec against `v0.27.0`. Concretely: use `compress_ratio=`
and `storage_block_size`; never `tokens_per_state`, `num_states`,
`CircularBufferSpec` or `SlidingWindowSpec.extra_retained_tokens`.

The test enforces the negative half by parsing `src/` and refusing those three
names as identifiers anywhere. It tokenizes rather than grepping, so a comment
explaining why a name is avoided does not trip it -- a tripwire whose cheapest
repair is deleting the documentation is not worth having.

## The two API shapes

| API | `v0.27.0` (the deployment) | a recent vLLM `main` |
| --- | --- | --- |
| `MLAAttentionSpec.compress_ratio: int = 1` | present (`vllm/v1/kv_cache_interface.py:394`) | removed |
| `MLAAttentionSpec.storage_block_size` (property, `block_size // compress_ratio`) | present (`:404-405`) | removed |
| `SlidingWindowMLASpec.compress_ratio` / `.storage_block_size` | present (`:637`, `:644-645`) | removed |
| `MLAAttentionSpec.alignment`, `.model_version`, `.cache_dtype_str` | present (`:391-395`) | present |
| `AttentionSpec.tokens_per_state` / `.num_states` | **absent** | present |
| `CircularBufferSpec` | **absent** | present |
| `SlidingWindowSpec.extra_retained_tokens` | **absent** | present |

Against a checkout that has made that move, `MLAAttentionSpec(compress_ratio=4)`
raises `TypeError: unexpected keyword argument`. That is not hypothetical: it is
why `tests/runner/test_tpu_runner.py` (which constructs one at `:2498` and
`:2558`) does not import on a developer box whose vLLM is newer than the pin,
and why `tests/models/vllm/test_kimi_k3.py` reports failures there.

The repo's own call sites, all written against the pin:
`layers/vllm/custom_ops/deepseek_v4/deepseek_v4_indexer.py:113`,
`models/vllm/deepseek_v4/attention.py:244`, and `runner/tpu_runner.py`, which
reads `spec.storage_block_size` and branches on `spec.compress_ratio`.

## How the pin was confirmed

Reading `pyproject.toml` establishes what *should* be installed. Two further
checks establish what *is*, because a `dev` image stage can overlay a vLLM
source checkout:

* **The engine banner.** A pod log line
  `Initializing a V1 LLM engine (v0.27.0)` is a bare version string. A build
  from any commit after the tag renders as `0.27.1.devN+g<sha>` under
  setuptools-scm, so a bare `0.27.0` is the tag itself.
* **A line-number fingerprint**, which pins the tag rather than merely the
  version. The same log carries `[model.py:645] Resolved architecture` and
  `[model.py:1883] Using max model len`. At `v0.27.0`,
  `vllm/config/model.py:645` is exactly
  `logger.info("Resolved architecture: %s", arch)` and `:1883` is
  `logger.info("Using max model len %s", max_model_len)`. On a newer `main`
  those two lines are unrelated code.

The pod's `PYTHONPATH` carries this repo's `src/` and nothing else, so a vLLM
checkout on the developer box shadows nothing there. That asymmetry is why a
change this repo needs *in* vLLM has to be applied at runtime from inside this
package: editing a vLLM checkout cannot reach the pod, because the pod never
reads one.

## Consequences worth knowing

`CircularBufferSpec` is the natural way to express a fixed-depth per-request
ring, and it is not available. `SlidingWindowSpec.extra_retained_tokens` -- the
way to keep a few tokens past the window's free edge -- is not available either;
at `v0.27.0` the free edge is unconditional:

```python
# vllm/v1/core/single_type_kv_cache_manager.py:1083 @ v0.27.0
return max(0, num_computed_tokens - self.sliding_window + 1)
```

That is what leaves a homemade fixed-depth ring as the expressible option for
the GLM-5.3-Flash kpool indexer's pending-pool state.

## If you bump the vLLM pin

`tests/models/vllm/test_vllm_kv_cache_api_pin.py` will fail. In order:

1. Re-derive the table above from the new ref.
2. Re-check the repo call sites listed under "The two API shapes".
3. Re-check the three names the test forbids in `src/` --
   `CircularBufferSpec`, `extra_retained_tokens`, `tokens_per_state` -- and
   unforbid whichever the new pin actually provides.
