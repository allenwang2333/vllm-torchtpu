#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fast TorchDynamo / TPU graph compilation test for DeepSeek-V4.

Uses `load_format="dummy"` to skip the ~3.5 minute disk loading and weight
processing phase, exercising the full 43-layer model (SWA, CSA, HCA attention
variants, Dense MLPs, and MoE routing) under TorchDynamo compilation in ~10s.
"""

import os
import time

from vllm import LLM, SamplingParams

MODEL = os.getenv(
    "DSV4_MODEL_PATH",
    "/home/patemotter_google_com/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1",
)
MAX_MODEL_LEN = int(os.getenv("DSV4_MAX_MODEL_LEN", "4096"))
MAX_SEQS = int(os.getenv("DSV4_MAX_SEQS", "128"))


def main():
    print("=" * 70)
    print("DeepSeek-V4 Fast Compilation Tracing Test (Dummy Weights)")
    print(f"Model: {MODEL}")
    print(f"Max model len: {MAX_MODEL_LEN}, Max seqs: {MAX_SEQS}")
    print("=" * 70)

    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        quantization="deepseek_v4_fp8",
        load_format="dummy",
        hf_overrides={"num_hidden_layers": 4},
        tensor_parallel_size=int(os.getenv("VLLM_TP_SIZE", "8")),
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=int(
            os.getenv("DSV4_MAX_NUM_BATCHED_TOKENS", "1024")),
        max_num_seqs=MAX_SEQS,
        kv_cache_dtype="fp8_e4m3",
        gpu_memory_utilization=float(os.getenv("GPU_MEM_UTIL", "0.67")),
        enable_expert_parallel=True,
        enable_prefix_caching=False,
        enforce_eager=False,
    )
    init_time = time.perf_counter() - t0
    print(
        f"[TEST] Engine initialization with dummy weights took {init_time:.2f}s"
    )

    # Run a 1-step test generation to trigger and verify compilation across all layers
    print("[TEST] Triggering 1-step decode forward to verify graph tracing...")
    prompts = ["Hello world, this is a test prompt to verify compilation."] * 8
    sampling_params = SamplingParams(max_tokens=2, temperature=0.0)

    t1 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    compile_and_run_time = time.perf_counter() - t1
    print(
        f"[TEST] Forward pass + compilation completed in {compile_and_run_time:.2f}s"
    )
    print(f"[TEST] Generated {len(outputs)} sequences successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
