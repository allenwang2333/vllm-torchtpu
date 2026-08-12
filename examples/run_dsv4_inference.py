import os

os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["MODEL_IMPL_TYPE"] = "vllm"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ.setdefault("VLLM_FP4_CACHE_DIR", "/dev/shm/dsv4-flash-real-fp4")

# NOTE: this file used to hot-patch torch_tpu._internal.pallas from
# src/vllm_torchtpu/patches/pallas_patched.py at startup. That file was
# removed from the tree (upstream torch_tpu covers it), and the loader's
# os.path.exists guard made the whole block a silent no-op -- while its
# presence misled the 20260812 decode-corruption investigation into a
# false "hot-patched bridge" premise (docs/dsv4_decode_race_postmortem.md).
# Stock torch_tpu pallas is what actually runs; keep it that way.

from vllm import LLM, SamplingParams


def check_tpu_device_availability(timeout_seconds=60):
    """Wait until TPU devices /dev/accel* are free from locks before launching LLM."""
    import glob
    import shutil
    import subprocess
    import time

    # Clean up stale shared memory and IPC handles first
    for pattern in ("/tmp/ipc*", "/tmp/psm*", "/tmp/vllm*"):
        for p in glob.glob(pattern):
            try:
                if shutil.os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    shutil.os.remove(p)
            except Exception:
                pass

    start_time = time.time()
    waited = False
    while time.time() - start_time < timeout_seconds:
        accel_files = glob.glob("/dev/accel*") + glob.glob("/dev/vfio/*")
        if not accel_files:
            break
        try:
            res = subprocess.run(["lsof"] + accel_files,
                                 capture_output=True,
                                 text=True)
            if res.returncode != 0 or not res.stdout.strip():
                break
            print(
                "[PREFLIGHT_WAIT] TPU chips busy (locked by background processes). Waiting for teardown...",
                flush=True)
            waited = True
        except Exception:
            break
        time.sleep(3)

    if waited:
        print(
            "[PREFLIGHT_SUCCESS] TPU chips released. Waiting 5s buffer for driver stabilization...",
            flush=True)
        time.sleep(5)
    print(
        "[PREFLIGHT_CHECK] All TPU chips free. Proceeding with LLM initialization.",
        flush=True)


def main():
    check_tpu_device_availability()
    # Instantiate the LLM. It will automatically use our TPU platform and patches.
    # We use TP=8 to utilize all 8 TPU chips on the v7x-8 node.
    tp_size = int(os.getenv("VLLM_TP_SIZE", "8"))
    ep_size = int(os.getenv("VLLM_EP_SIZE", "8"))
    max_model_len = int(os.getenv("DSV4_MAX_MODEL_LEN", "64"))
    # The production benchmark runs DSv4 with 512 batched tokens against a
    # 9216-token context (`tests/e2e/benchmarking/mmlu.sh`), so every prompt
    # over 512 tokens is prefilled in chunks. This harness defaulted the two
    # to the same value, which prefills in one shot -- a path the reference
    # never takes, and vice versa.
    max_num_batched_tokens = int(
        os.getenv("DSV4_MAX_NUM_BATCHED_TOKENS", str(max_model_len)))
    compile_sizes = sorted({64, max_num_batched_tokens, max_model_len})
    llm = LLM(
        model=os.getenv("DSV4_MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash"),
        quantization="deepseek_v4_fp8",
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=int(os.getenv("DSV4_MAX_NUM_SEQS", "256")),
        kv_cache_dtype="fp8_e4m3",
        gpu_memory_utilization=float(
            os.getenv("GPU_MEMORY_UTILIZATION",
                      os.getenv("GPU_MEM_UTIL", "0.40"))),
        enable_expert_parallel=ep_size > 1,
        # The production benchmark runs DSv4 with prefix caching OFF; this
        # harness silently defaulted it ON, which changes block allocation and
        # therefore interacts with the SWA/compressed-KV cache overlay.
        enable_prefix_caching=os.getenv("DSV4_ENABLE_PREFIX_CACHING",
                                        "1") == "1",
        enforce_eager=os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
        # DSV4_ASYNC_SCHEDULING=0 disables vLLM's async scheduling
        # (auto-enabled when the executor supports it). Intervention #2 for
        # the decode-corruption race: async scheduling preps step s+1's
        # buffers while step s's deferred side-effect branches (compressor
        # kv_score) are still unmaterialized -- see
        # docs/dsv4_decode_race_postmortem.md.
        async_scheduling=(False if os.getenv("DSV4_ASYNC_SCHEDULING") == "0"
                          else None),
        # Auto-prefetch only engages for network filesystems (NFS/Lustre);
        # our checkpoint sits on local EXT4 so it never kicks in even though
        # it comfortably fits in RAM. Force it on request.
        safetensors_load_strategy=os.getenv("DSV4_SAFETENSORS_LOAD_STRATEGY"),
        additional_config={
            "compilation_config": {
                "compile_sizes": compile_sizes,
            },
            "cache_config": {
                "num_gpu_blocks_override":
                int(os.getenv("DSV4_NUM_GPU_BLOCKS", "512")),
            },
        },
    )

    # `DSV4_PROMPT_FILES` (comma-separated paths) runs several prompts against
    # ONE model load, each in its own `generate` call so every prompt keeps
    # single-sequence conditions. Loading and requantizing DSv4 costs ~12 min,
    # so a context-length sweep this way is one load instead of N.
    prompt_files = [
        p for p in os.getenv("DSV4_PROMPT_FILES", "").split(",") if p
    ]
    if prompt_files:
        prompt_contents = []
        for path in prompt_files:
            with open(path) as f:
                prompt_contents.append((os.path.basename(path), f.read()))
    else:
        prompt_contents = [("DSV4_PROMPT",
                            os.getenv(
                                "DSV4_PROMPT",
                                "Explain the difference between XLA and LLVM "
                                "in one sentence."))]

    tokenizer = llm.get_tokenizer()
    prompt_texts = [(label,
                     tokenizer.apply_chat_template([{
                         "role": "user",
                         "content": content
                     }],
                                                   tokenize=False,
                                                   add_generation_prompt=True))
                    for label, content in prompt_contents]

    _temp = float(os.getenv("DSV4_TEMPERATURE", "0.0"))
    _max_tokens = int(os.getenv("DSV4_MAX_TOKENS", "50"))
    sampling_params = SamplingParams(temperature=_temp,
                                     max_tokens=_max_tokens,
                                     ignore_eos=False,
                                     logprobs=5)

    # Teacher-forcing mode: score a FIXED token sequence step by step instead
    # of free-running generation. `DSV4_TF_FILES` is a comma-separated list of
    # JSON files, each {"label", "prompt_token_ids", "forced_token_ids"}. For
    # every step t we prefill `prompt + forced[:t]` (token IDs, never re-
    # tokenized text) and read the top-5 logprobs of the next token -- i.e.
    # the model's distribution at the exact context the forced sequence had at
    # step t, but with caches built by ONE prefill rather than t decode steps.
    # This runner's own runner path does not support vLLM `prompt_logprobs`
    # (tpu_runner.py hardcodes prompt_logprobs_dict=None), hence the
    # growing-prefix loop. Requires prefix caching OFF or every call after the
    # first would reuse decode-era blocks and measure nothing.
    #
    # DSV4_TF_AFTER_PROMPTS=1 runs the normal DSV4_PROMPT_FILES generation
    # loop FIRST and then the TF scoring, in the same process -- one model
    # load produces both the decode-built and the prefill-built data for a
    # differential (e.g. with DSV4_DUMP_PAGES=1).
    tf_files = [p for p in os.getenv("DSV4_TF_FILES", "").split(",") if p]
    if tf_files and os.getenv("DSV4_TF_AFTER_PROMPTS") == "1":
        print("=== Prompt generation pass (before TF scoring) ===")
        for label, prompt_text in prompt_texts:
            print(f"[SWEEP] ===== prompt: {label} =====")
            outputs = llm.generate([prompt_text], sampling_params)
            for output in outputs:
                print(f"[SWEEP] {label}: prompt_tokens="
                      f"{len(output.prompt_token_ids)} "
                      f"generated={output.outputs[0].text!r}")
            _report(outputs)
    if tf_files:
        import json

        from vllm.inputs import TokensPrompt
        assert os.getenv("DSV4_ENABLE_PREFIX_CACHING", "1") == "0", \
            "teacher forcing needs DSV4_ENABLE_PREFIX_CACHING=0"
        tf_params = SamplingParams(temperature=0.0,
                                   max_tokens=1,
                                   ignore_eos=True,
                                   logprobs=5)
        for path in tf_files:
            with open(path) as f:
                spec = json.load(f)
            label = spec["label"]
            prompt_ids = spec["prompt_token_ids"]
            forced = spec["forced_token_ids"]
            print(f"[TF] ===== {label}: prompt={len(prompt_ids)} "
                  f"forced={len(forced)} =====")
            for t in range(len(forced)):
                ids = prompt_ids + forced[:t]
                outs = llm.generate(
                    [TokensPrompt(prompt_token_ids=ids)],
                    tf_params,
                    use_tqdm=False,
                )
                out = outs[0].outputs[0]
                lp = out.logprobs[0] if out.logprobs else {}
                print(
                    f"[TF_LOGPROBS] file={label} step={t} "
                    f"forced_next={forced[t]} picked={out.token_ids[0]} {lp}",
                    flush=True)
        return

    print("=== Testing formatted prompt with llm.generate ===")
    for label, prompt_text in prompt_texts:
        print(f"[SWEEP] ===== prompt: {label} =====")
        outputs = llm.generate([prompt_text], sampling_params)
        for output in outputs:
            print(f"[SWEEP] {label}: prompt_tokens="
                  f"{len(output.prompt_token_ids)} "
                  f"generated={output.outputs[0].text!r}")
        _report(outputs)


def _report(outputs):
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Prompt token IDs: {output.prompt_token_ids}")
        print(f"Generated token IDs: {output.outputs[0].token_ids}")
        print(f"Generated text: {output.outputs[0].text!r}")
        if output.outputs[0].logprobs is not None and len(
                output.outputs[0].logprobs) > 0:
            print(
                f"Generated logprobs (step 1): {output.outputs[0].logprobs[0]}"
            )
            for i, lp in enumerate(output.outputs[0].logprobs):
                print(f"[STEP_LOGPROBS] step={i} {lp}")
        print("-" * 80)


if __name__ == "__main__":
    main()
