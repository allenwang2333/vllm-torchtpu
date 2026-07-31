import os

os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["MODEL_IMPL_TYPE"] = "vllm"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ.setdefault("VLLM_FP4_CACHE_DIR", "/dev/shm/dsv4-flash-real-fp4")

import importlib.util
import sys

# Dynamic hot-patch to import our local patched pallas module before any vllm imports occur.
# This registers output_partition_specs support on GKE without modifying the container site-packages.
try:
    patch_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src",
        "vllm_torchtpu", "patches", "pallas_patched.py")
    if os.path.exists(patch_path):
        # 1. Import original module and submodule
        import torch_tpu._internal.pallas as original_pallas
        import torch_tpu._internal.pallas.pallas as original_pallas_submodule

        # 2. Load our patched module as a temporary standalone module
        spec = importlib.util.spec_from_file_location("patched_pallas_temp",
                                                      patch_path)
        patched_module = importlib.util.module_from_spec(spec)
        sys.modules["patched_pallas_temp"] = patched_module
        spec.loader.exec_module(patched_module)

        # 3. Patch both package and submodule namespaces
        for target in [original_pallas, original_pallas_submodule]:
            target.jax_op = patched_module.jax_op
            target.custom_jax_kernel = patched_module.custom_jax_kernel
            target.JaxCallable = patched_module.JaxCallable
            target.torch_placeholder = patched_module.torch_placeholder
            target._convert_shape = patched_module._convert_shape

        print(
            f"=== [HOT PATCH] Successfully hot-patched pallas package and submodule from {patch_path} ==="
        )
except Exception as e:
    print(f"=== [HOT PATCH ERROR] Failed to hot-patch pallas.py: {e} ===")

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
    compile_sizes = sorted({64, max_model_len})
    llm = LLM(
        model=os.getenv("DSV4_MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash"),
        quantization="deepseek_v4_fp8",
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=int(os.getenv("DSV4_MAX_NUM_SEQS", "256")),
        kv_cache_dtype="fp8_e4m3",
        gpu_memory_utilization=float(
            os.getenv("GPU_MEMORY_UTILIZATION",
                      os.getenv("GPU_MEM_UTIL", "0.40"))),
        enable_expert_parallel=ep_size > 1,
        enforce_eager=os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
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

    prompt_content = os.getenv(
        "DSV4_PROMPT",
        "Explain the difference between XLA and LLVM in one sentence.")
    conversation = [{"role": "user", "content": prompt_content}]

    tokenizer = llm.get_tokenizer()
    prompt_text = tokenizer.apply_chat_template(conversation,
                                                tokenize=False,
                                                add_generation_prompt=True)

    _temp = float(os.getenv("DSV4_TEMPERATURE", "0.0"))
    _max_tokens = int(os.getenv("DSV4_MAX_TOKENS", "50"))
    sampling_params = SamplingParams(temperature=_temp,
                                     max_tokens=_max_tokens,
                                     ignore_eos=False,
                                     logprobs=5)

    print("=== Testing formatted prompt with llm.generate ===")
    outputs = llm.generate([prompt_text], sampling_params)
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
