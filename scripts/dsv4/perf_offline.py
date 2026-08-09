"""Offline DSv4 throughput sweep: one model load, several batch sizes.

Answers one question: is the ~1 tok/s seen while serving a batching artifact
(a graph compiled for `max_num_seqs` running a single real sequence) or a real
per-step cost? If tokens/s scales with batch size, it is the former.

    PYTHONPATH=src python scripts/dsv4/perf_offline.py

Env knobs: DSV4_PERF_BATCHES (default "1,8,32,128"), DSV4_PERF_PROMPT_TOKENS
(512), DSV4_PERF_GEN_TOKENS (64), DSV4_MAX_MODEL_LEN (4096), VLLM_TP_SIZE (8).
"""
import os
import time

os.environ.setdefault("NEW_MODEL_DESIGN", "1")
os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("MOE_REQUANTIZE_WEIGHT_DTYPE", "fp4")
os.environ.setdefault("MOE_REQUANTIZE_BLOCK_SIZE", "512")
os.environ.setdefault("VLLM_PLUGINS", "torchtpu,torchtpu_general")
os.environ.setdefault("VLLM_FP4_CACHE_DIR", "/dev/shm/dsv4-fp4-fp4x512")
os.environ.setdefault("HF_HOME",
                      "/home/patemotter_google_com/workspace/hf_cache")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.getenv(
    "DSV4_MODEL_PATH", "/home/patemotter_google_com/workspace/hf_cache/hub/"
    "models--deepseek-ai--DeepSeek-V4-Flash/snapshots/"
    "60d8d70770c6776ff598c94bb586a859a38244f1")
BATCHES = [
    int(b) for b in os.getenv("DSV4_PERF_BATCHES", "1,8,32,128").split(",")
]
PROMPT_TOKENS = int(os.getenv("DSV4_PERF_PROMPT_TOKENS", "512"))
GEN_TOKENS = int(os.getenv("DSV4_PERF_GEN_TOKENS", "64"))
MAX_MODEL_LEN = int(os.getenv("DSV4_MAX_MODEL_LEN", "4096"))
MAX_SEQS = max(BATCHES)


def main():
    llm = LLM(
        model=MODEL,
        quantization="deepseek_v4_fp8",
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
        enforce_eager=os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
    )

    tokenizer = llm.get_tokenizer()
    # A prompt of a known token length, so tokens/s is computed from real
    # counts rather than an assumed prompt size.
    filler = "The quick brown fox jumps over the lazy dog. " * 200
    ids = tokenizer(filler)["input_ids"][:PROMPT_TOKENS]
    prompt = tokenizer.decode(ids)
    actual_prompt_tokens = len(tokenizer(prompt)["input_ids"])

    # ignore_eos so every sequence emits exactly GEN_TOKENS: throughput is then
    # a fixed amount of work, not a function of when the model chose to stop.
    params = SamplingParams(temperature=0.0,
                            max_tokens=GEN_TOKENS,
                            ignore_eos=True)

    print(
        f"[PERF] model_len={MAX_MODEL_LEN} prompt_tokens={actual_prompt_tokens} "
        f"gen_tokens={GEN_TOKENS} max_num_seqs={MAX_SEQS}",
        flush=True)

    print("[PERF] warmup (excluded from results)", flush=True)
    llm.generate([prompt], params)

    rows = []
    for batch in BATCHES:
        prompts = [prompt] * batch
        t0 = time.perf_counter()
        outs = llm.generate(prompts, params)
        elapsed = time.perf_counter() - t0
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        pro = sum(len(o.prompt_token_ids) for o in outs)
        rows.append(
            (batch, elapsed, gen, gen / elapsed, (pro + gen) / elapsed))
        print(
            f"[PERF] batch={batch:4d} wall={elapsed:7.2f}s "
            f"gen_tokens={gen:6d} decode={gen/elapsed:8.2f} tok/s "
            f"total={(pro+gen)/elapsed:9.2f} tok/s",
            flush=True)

    print("\n[PERF] === summary ===", flush=True)
    print(
        f"[PERF] {'batch':>6} {'wall_s':>9} {'decode_tok_s':>13} "
        f"{'total_tok_s':>12} {'speedup_vs_b1':>14}",
        flush=True)
    base = rows[0][3] if rows else 1.0
    for batch, elapsed, _gen, dec, tot in rows:
        print(
            f"[PERF] {batch:6d} {elapsed:9.2f} {dec:13.2f} {tot:12.2f} "
            f"{dec/base:13.2f}x",
            flush=True)
    print(
        "[PERF] If decode tok/s scales with batch, the serving slowness is "
        "empty-batch padding, not per-step kernel cost.",
        flush=True)


if __name__ == "__main__":
    main()
