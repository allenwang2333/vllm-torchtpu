import os
os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["MODEL_IMPL_TYPE"] = "vllm"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ.setdefault("VLLM_FP4_CACHE_DIR", "/dev/shm/dsv4-flash-real-fp4")

import sys
import importlib.util

try:
    patch_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "vllm_torchtpu", "patches", "pallas_patched.py")
    if os.path.exists(patch_path):
        import torch_tpu._internal.pallas as original_pallas
        import torch_tpu._internal.pallas.pallas as original_pallas_submodule
        spec = importlib.util.spec_from_file_location("patched_pallas_temp", patch_path)
        patched_module = importlib.util.module_from_spec(spec)
        sys.modules["patched_pallas_temp"] = patched_module
        spec.loader.exec_module(patched_module)
        for target in [original_pallas, original_pallas_submodule]:
            target.jax_op = patched_module.jax_op
            target.custom_jax_kernel = patched_module.custom_jax_kernel
            target.JaxCallable = patched_module.JaxCallable
            target.torch_placeholder = patched_module.torch_placeholder
            target._convert_shape = patched_module._convert_shape
        print(f"=== [HOT PATCH] Successfully hot-patched pallas package and submodule from {patch_path} ===")
except Exception as e:
    print(f"=== [HOT PATCH ERROR] Failed to hot-patch pallas.py: {e} ===")

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def check_tpu_device_availability(timeout_seconds=60):
    import time
    import glob
    import subprocess
    import shutil

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
            res = subprocess.run(["lsof"] + accel_files, capture_output=True, text=True)
            if res.returncode != 0 or not res.stdout.strip():
                break
            print(f"[PREFLIGHT_WAIT] TPU chips busy. Waiting for teardown...", flush=True)
            waited = True
        except Exception:
            break
        time.sleep(3)

    if waited:
        print("[PREFLIGHT_SUCCESS] TPU chips released. Waiting 5s buffer.", flush=True)
        time.sleep(5)
    print("[PREFLIGHT_CHECK] All TPU chips free. Proceeding with LLM initialization.", flush=True)


def main():
    check_tpu_device_availability()
    tp_size = int(os.getenv("VLLM_TP_SIZE", "8"))
    ep_size = int(os.getenv("VLLM_EP_SIZE", "8"))
    llm = LLM(
        model=os.getenv("DSV4_MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash"),
        quantization="deepseek_v4_fp8",
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=64,
        max_num_batched_tokens=64,
        kv_cache_dtype="fp8_e4m3",
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", os.getenv("GPU_MEM_UTIL", "0.40"))),
        enable_expert_parallel=ep_size > 1,
        enforce_eager=os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
        additional_config={
            "compilation_config": {
                "compile_sizes": [64],
            },
            "cache_config": {
                "num_gpu_blocks_override": 512,
            },
        },
    )

    prompt_content = os.getenv("DSV4_PROMPT", "Compare XLA and LLVM in two bullet points.")
    conversation = [{"role": "user", "content": prompt_content}]
    tokenizer = llm.get_tokenizer()
    prompt_text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    print(f"[SETUP] prompt_token_ids ({len(prompt_token_ids)}): {prompt_token_ids}", flush=True)

    sampling_2step = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=False, logprobs=5)

    print("=== REQUEST A: normal continuous generation, 2 decode steps ===", flush=True)
    outputs_a = llm.generate([TokensPrompt(prompt_token_ids=prompt_token_ids)], sampling_2step)
    out_a = outputs_a[0]
    gen_ids_a = list(out_a.outputs[0].token_ids)
    print(f"[REQUEST_A] generated token ids: {gen_ids_a}", flush=True)
    for i, lp in enumerate(out_a.outputs[0].logprobs):
        print(f"[REQUEST_A][STEP_LOGPROBS] step={i} {lp}", flush=True)

    first_token = gen_ids_a[0]
    extended_prompt_ids = list(prompt_token_ids) + [first_token]
    print(f"[SETUP] extended_prompt_ids ({len(extended_prompt_ids)}): {extended_prompt_ids}", flush=True)

    sampling_1step = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=False, logprobs=5)

    print("=== REQUEST B: fresh single-shot prefill of prompt+first_token, predict what would be decode-step-2's token ===", flush=True)
    outputs_b = llm.generate([TokensPrompt(prompt_token_ids=extended_prompt_ids)], sampling_1step)
    out_b = outputs_b[0]
    gen_ids_b = list(out_b.outputs[0].token_ids)
    print(f"[REQUEST_B] generated token ids: {gen_ids_b}", flush=True)
    for i, lp in enumerate(out_b.outputs[0].logprobs):
        print(f"[REQUEST_B][STEP_LOGPROBS] step={i} {lp}", flush=True)

    print("=== COMPARISON ===", flush=True)
    print(f"Request A step=1 (decode, real 2nd generated token) logprobs: {out_a.outputs[0].logprobs[1]}", flush=True)
    print(f"Request B step=0 (fresh prefill of same 16-token context) logprobs: {out_b.outputs[0].logprobs[0]}", flush=True)
    a_top = list(out_a.outputs[0].logprobs[1].keys())[0]
    b_top = list(out_b.outputs[0].logprobs[0].keys())[0]
    print(f"MATCH={a_top == b_top} (A_top_token_id={a_top}, B_top_token_id={b_top})", flush=True)


if __name__ == "__main__":
    main()
