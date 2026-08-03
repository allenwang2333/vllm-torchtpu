#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steppable offline inference for the Kimi KDA layer.

Runs a tiny random-weight Kimi Linear model in one process with torch.compile
off, so a debugger can step through real model execution. Three settings make
that possible, and all three are load-bearing:

- ``enforce_eager=True`` sets ``CompilationMode.NONE``, so the model runs as
  plain Python instead of being traced once by
  ``TorchCompileWithNoGuardsWrapper``. Under the wrapper there is no Python
  frame left to step through.
- ``tensor_parallel_size=1`` keeps the platform on ``UniProcExecutor``. Any
  higher and the model runs in ``TpuMultiprocExecutor`` child processes, where
  a breakpoint in the parent never fires.
- ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` keeps the engine core in-process too.

The weights are random (``load_format="dummy"``) and the model is two KDA
layers, so the output is meaningless -- this is for stepping through *control
flow and shapes*, not for checking generation quality. Use
``scripts/kernels/bench_chunk_kda.py`` for numbers and the layer tests for
correctness.

Usage::

    # Break inside KimiDeltaAttention._core_attention on its 1st call.
    python scripts/debug/offline_eager_kimi.py

    # Break on the prefill call for layer 1 instead.
    python scripts/debug/offline_eager_kimi.py --break-on-call 2

    # No breakpoint, just run and print the shapes each call sees.
    python scripts/debug/offline_eager_kimi.py --break-on-call 0

Once stopped, the useful frame locals are ``mixed_qkv``, ``raw_gate``,
``beta``, ``recurrent_cache`` and ``metadata``; ``self`` is the layer. ``u``
walks out to ``forward``, ``s`` steps into ``sconv_op`` / the recurrence.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

# Must precede `import vllm`: the engine core forks otherwise.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("TPU_STDERR_LOG_LEVEL", "3")

HEAD_DIM = 128  # what the chunked kernel tiles; keep it at the real value
NUM_HEADS = 2  # local KDA heads, since tensor_parallel_size is 1


def write_tiny_config(directory: Path, num_kda_layers: int = 2) -> Path:
    """A minimal Kimi Linear config: ``num_kda_layers`` KDA layers, then MLA.

    ``kda_layers`` is 1-based -- ``is_kda_layer`` checks ``layer_idx + 1`` --
    and any layer not listed there becomes ``MultiHeadLatentAttention``.

    The trailing full-attention layer is not optional. A KDA-only model leaves
    ``_find_non_ssm_backend`` with nothing to return, and
    ``update_block_size_for_backend`` then fails on ``None``. Its MLA head dims
    are the checkpoint's real ones, because the TPU MLA kernels are picky about
    them; only the head *count* is shrunk.

    Leaving ``num_experts`` unset keeps every MLP dense, which keeps the fused
    MoE kernels out of the way.
    """
    num_layers = num_kda_layers + 1
    config = {
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "dtype": "bfloat16",
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 72,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        # Small on purpose: with skip_tokenizer_init nothing needs a real vocab,
        # and this keeps the embedding from dominating the dummy weights.
        "vocab_size": 1024,
        "tie_word_embeddings": False,
        "first_k_dense_replace": 0,
        # MLA, for the one full-attention layer.
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": True,
        "q_lora_rank": None,
        "linear_attn_config": {
            "kda_layers": list(range(1, num_kda_layers + 1)),
            "full_attn_layers": [num_layers],
            "head_dim": HEAD_DIM,
            "num_heads": NUM_HEADS,
            "short_conv_kernel_size": 4,
        },
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "rope_theta": 10000.0,
    }
    path = directory / "config.json"
    path.write_text(json.dumps(config, indent=2))
    return path


def instrument(break_on_call: int) -> None:
    """Wrap ``_core_attention`` so the debugger lands in a real forward pass.

    Patching here rather than telling you a line number keeps the script
    working when the file moves, and lands you in the frame that matters with
    the activations already in scope.

    Called *after* the engine is built, on purpose. Importing the model module
    before vLLM has resolved its platform plugin leaves the TPU runtime
    half-initialised and the first device transfer fails with "pjrt client not
    initialized". Patching the class after the fact still takes effect, because
    every layer looks the method up on the class at call time.
    """
    from vllm_torchtpu.models.vllm.kimi_k3.attention import KimiDeltaAttention

    original = KimiDeltaAttention._core_attention
    calls = {"n": 0}

    def traced(self, mixed_qkv, raw_gate, beta, output_gate, sconv_cache,
               recurrent_cache, metadata):
        calls["n"] += 1
        num_tokens = mixed_qkv.shape[0]
        query_lens = (metadata.query_start_loc[1:] -
                      metadata.query_start_loc[:-1])
        scheduled = int((query_lens > 0).sum())
        print(
            f"[call {calls['n']:>3}] {self.prefix}  "
            f"tokens={num_tokens} scheduled_reqs={scheduled} "
            f"recurrence={'naive' if self.use_naive_kda else 'chunked'}",
            flush=True)
        if calls["n"] == break_on_call:
            print(
                "  -- entering the debugger; `s` steps into the ops, "
                "`u` goes out to forward",
                flush=True)
            breakpoint()
        return original(self, mixed_qkv, raw_gate, beta, output_gate,
                        sconv_cache, recurrent_cache, metadata)

    KimiDeltaAttention._core_attention = traced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--break-on-call",
        type=int,
        default=1,
        help="1-based _core_attention call to break on; 0 to only trace")
    parser.add_argument("--num-kda-layers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--naive",
                        action="store_true",
                        help="run the reference scan instead of the kernel")
    args = parser.parse_args()

    if args.naive:
        os.environ["VLLM_TPU_USE_NAIVE_KDA"] = "1"

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    with tempfile.TemporaryDirectory(prefix="kimi-tiny-") as tmp:
        model_dir = Path(tmp)
        write_tiny_config(model_dir, args.num_kda_layers)

        llm = LLM(
            model=str(model_dir),
            load_format="dummy",  # random weights; nothing to download
            skip_tokenizer_init=True,  # feed token ids, need no tokenizer
            tensor_parallel_size=1,  # keeps UniProcExecutor
            enforce_eager=True,  # disables torch.compile
            max_model_len=256,
            max_num_seqs=4,
            trust_remote_code=True,
        )
        instrument(args.break_on_call)

        # Two requests of different lengths, so the batch is ragged and the
        # chunked kernel sees more than one segment.
        prompts = [
            TokensPrompt(prompt_token_ids=list(range(3, 3 + args.prompt_len))),
            TokensPrompt(
                prompt_token_ids=list(range(5, 5 + args.prompt_len // 2))),
        ]
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
            # The progress bar rewrites the line pdb is prompting on.
            use_tqdm=False,
        )

    for i, out in enumerate(outputs):
        print(f"request {i}: prompt={len(out.prompt_token_ids)} tokens -> "
              f"generated {list(out.outputs[0].token_ids)}")
    print("EAGER_RUN_OK (weights are random; token ids are meaningless)")


if __name__ == "__main__":
    main()
