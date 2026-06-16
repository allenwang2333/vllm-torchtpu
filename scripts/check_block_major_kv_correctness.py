"""Single-chip correctness gate for the block-major KV cache layout.

Runs Qwen3-0.6B with greedy decoding on a fixed prompt set, dumps the
generated text + token IDs to a JSON file. Run twice (flag off, flag on)
and diff the JSONs:

  VLLM_TPU_BLOCK_MAJOR_KV=0 python3 scripts/check_block_major_kv_correctness.py \
      --out /tmp/kv_layer_major.json
  VLLM_TPU_BLOCK_MAJOR_KV=1 python3 scripts/check_block_major_kv_correctness.py \
      --out /tmp/kv_block_major.json
  python3 scripts/check_block_major_kv_correctness.py --diff \
      /tmp/kv_layer_major.json /tmp/kv_block_major.json

Identical token IDs => strided per-layer views survive the torch_tpu
Pallas-op bridge with donation. Divergence => the central hypothesis is
wrong and we need to rework the approach.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# scripts/ contains a sibling `vllm` subdir (benchmarking/integration
# helpers) that would shadow the real vllm package when this file is
# launched as `python3 scripts/check_block_major_kv_correctness.py`,
# because Python prepends the script's parent dir to sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path and sys.path[0] == _HERE:
    sys.path.pop(0)

# Some envs have torchvision built against an incompatible torch ABI
# and crash on `torchvision::nms` registration. Transformers'
# image_processing_base would pull it in. Probe; suppress only if the
# probe fails — harmless in healthy envs.
try:  # noqa: E402
    import torchvision as _tv  # noqa: F401, E402
    _ = _tv.transforms.InterpolationMode  # forces _meta_registrations
    del _tv
except Exception:
    try:
        import transformers.utils.import_utils as _iu  # noqa: E402
        _iu._torchvision_available = False
        del _iu
    except Exception:
        pass

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "The colors of the rainbow are",
    "The future of AI is",
    "In what year did the Titanic sink?",
    "What is the largest planet in our solar system?",
    "Who painted the ceiling of the Sistine Chapel?",
    "What is the chemical symbol for gold?",
]

# A long fixed prefix that repeats verbatim across the prompts below.
# Used for prefix-cache reuse stress: the SECOND call should warm-hit on
# everything except the per-prompt suffix.
_SHARED_PREFIX = (
    "You are a careful, concise assistant. You answer in one short "
    "sentence. You always begin your reply with the word 'Answer'. "
    "Do not include any qualifiers or hedging. Do not ask for "
    "clarification. Do not repeat the question. Use lowercase only. "
    "Here is the question: ")

SHARED_PREFIX_PROMPTS = [
    _SHARED_PREFIX + "What color is the sky on a clear day?",
    _SHARED_PREFIX + "Who wrote Pride and Prejudice?",
    _SHARED_PREFIX + "What is two plus two?",
    _SHARED_PREFIX + "What is the capital of Japan?",
]


def run_and_dump(out_path: str,
                 model: str,
                 max_model_len: int,
                 max_tokens: int,
                 enable_prefix_cache: bool = True,
                 use_shared_prefix_prompts: bool = False,
                 tensor_parallel_size: int = 1,
                 kv_cache_dtype: str = "auto",
                 kv_offload_size_gib: float | None = None) -> None:
    # vllm imports must be lazy (see tests/conftest.py rationale).
    # vllm_torchtpu must be imported first to apply env_override and
    # vLLM patches before vllm's own modules are loaded.
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    import vllm_torchtpu  # noqa: F401, PLC0415

    os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")
    # Path 2 (bundle-direct kernel) eliminates the per-layer-fresh-alloc
    # behavior of the per-layer-view approach, so we can go back to a
    # reasonable HBM utilization.
    llm_kwargs = dict(model=model,
                      max_model_len=max_model_len,
                      max_num_batched_tokens=max_model_len,
                      gpu_memory_utilization=0.5,
                      tensor_parallel_size=tensor_parallel_size,
                      kv_cache_dtype=kv_cache_dtype,
                      enable_prefix_caching=enable_prefix_cache)
    if kv_offload_size_gib is not None:
        # OffloadingConnector with the TPU↔CPU spec, in kv_both role
        # (single process sources and sinks its own KV).
        llm_kwargs["kv_transfer_config"] = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "TPUCPUOffloadingSpec",
                "spec_module_path": "vllm_torchtpu.offload.cpu_tpu",
            },
        }
        llm_kwargs["kv_offloading_size"] = kv_offload_size_gib
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    prompts = (SHARED_PREFIX_PROMPTS if use_shared_prefix_prompts else PROMPTS)
    if use_shared_prefix_prompts:
        # Send the batch twice so the second call warm-hits prefix cache.
        # We diff tokens of the SECOND batch — they must match the
        # layer-major run's second batch.
        _ = llm.generate(prompts, sp)
    outputs = llm.generate(prompts, sp)

    dump = []
    for o in outputs:
        gen = o.outputs[0]
        dump.append(
            dict(prompt=o.prompt,
                 text=gen.text,
                 token_ids=list(gen.token_ids),
                 finish_reason=gen.finish_reason))
    with open(out_path, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"Wrote {out_path} ({len(dump)} prompts)")


def diff_dumps(path_a: str, path_b: str) -> int:
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)
    if len(a) != len(b):
        print(f"FAIL: prompt count differs: {len(a)} vs {len(b)}")
        return 2
    diverged = 0
    for i, (ra, rb) in enumerate(zip(a, b)):
        ta, tb = ra["token_ids"], rb["token_ids"]
        if ta == tb:
            continue
        diverged += 1
        prefix_match = 0
        for x, y in zip(ta, tb):
            if x != y:
                break
            prefix_match += 1
        print(f"DIVERGE prompt {i}: {ra['prompt']!r}")
        print(
            f"  matching prefix tokens: {prefix_match}/{min(len(ta), len(tb))}"
        )
        print(f"  A first 16: {ta[:16]}")
        print(f"  B first 16: {tb[:16]}")
        print(f"  A text: {ra['text']!r}")
        print(f"  B text: {rb['text']!r}")
    if diverged == 0:
        print(f"PASS: all {len(a)} prompts produce identical token IDs.")
        return 0
    print(f"FAIL: {diverged}/{len(a)} prompts diverged.")
    return 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", help="Where to write the run dump")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-model-len", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--no-prefix-cache",
                   dest="enable_prefix_cache",
                   action="store_false",
                   help="Disable prefix caching (default: on).")
    p.add_argument("--shared-prefix",
                   dest="shared_prefix",
                   action="store_true",
                   help="Use prompts that share a 100-token prefix; send "
                   "twice so the second batch tests prefix-cache reuse.")
    p.add_argument("--tp-size",
                   type=int,
                   default=1,
                   help="Tensor parallel size (TP=1 default).")
    p.add_argument("--kv-cache-dtype",
                   default="auto",
                   choices=["auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"],
                   help="kv_cache_dtype passed to LLM(...).")
    p.add_argument("--kv-offload-size",
                   type=float,
                   default=None,
                   help="If set, enable OffloadingConnector with this "
                   "many GiB of host-resident KV cache.")
    p.add_argument("--diff",
                   nargs=2,
                   metavar=("A", "B"),
                   help="Diff two run dumps and exit")
    args = p.parse_args()

    if args.diff:
        return diff_dumps(*args.diff)
    if not args.out:
        p.error("--out is required for a run (or pass --diff A B)")
    run_and_dump(args.out,
                 args.model,
                 args.max_model_len,
                 args.max_tokens,
                 enable_prefix_cache=args.enable_prefix_cache,
                 use_shared_prefix_prompts=args.shared_prefix,
                 tensor_parallel_size=args.tp_size,
                 kv_cache_dtype=args.kv_cache_dtype,
                 kv_offload_size_gib=args.kv_offload_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
