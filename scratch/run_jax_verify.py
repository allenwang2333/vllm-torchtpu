# Copyright 2026 Google LLC
# Wrapper to run JAX reference model with backend patch

import os
import sys

os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.32"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["VLLM_ENFORCE_EAGER"] = "1"
os.environ["VLLM_BYPASS_TPU_INIT"] = "1"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

# Compatibility alias for renamed DeepseekV4Attention class in newer vLLM versions
try:
    import vllm.models.deepseek_v4.attention as att_mod
    if not hasattr(att_mod, "DeepseekV4Attention") and hasattr(att_mod, "DeepseekV4MLAAttention"):
        att_mod.DeepseekV4Attention = att_mod.DeepseekV4MLAAttention
except ImportError:
    pass



# Add torchtpu-vllm/src to path so we can import env_override
sys.path.insert(0, "/mnt/workspace/vllm-torchtpu/src")

# Manually patch DeepSeek V4 MLA class to JAX wrapper to run under CPU/JAX platform
from tpu_inference.layers.vllm.custom_ops.deepseek_v4_attention import patch_deepseek_v4_mla_cls
patch_deepseek_v4_mla_cls()

# Mock CUDA device capability, Stream, and Event allocation since they are unavailable on CPU/TPU
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

current_platform.get_device_capability = lambda *args, **kwargs: DeviceCapability(9, 0)

class DummyCudaEvent:
    def __init__(self, *args, **kwargs):
        pass
    def record(self, *args, **kwargs):
        pass
    def wait(self, *args, **kwargs):
        pass
    def query(self):
        return True
    def synchronize(self):
        pass

import torch
torch.cuda.Stream = lambda *args, **kwargs: None
torch.cuda.Event = DummyCudaEvent

# Now add our repo to path and import offline_inference
sys.path.insert(0, "/mnt/workspace/vllm-torchtpu")

# Monkey-patch JAX quantization mapping for deepseek_v4_fp8
import tpu_inference.layers.common.quant_methods as qm
qm.FP8 = "deepseek_v4_fp8"

from examples.offline_inference import create_parser
from vllm import LLM, SamplingParams

# Monkey patch JAX MoE apply_jax to log inputs
from tpu_inference.layers.jax.quantization.fp8 import Fp8FusedMoEMethod
_orig_apply_jax = Fp8FusedMoEMethod.apply_jax

def patched_apply_jax(self, layer, x, *, router_logits):
    prefix = getattr(layer, "prefix", "")
    if "layers.0." in prefix or "layers.3." in prefix:
        import jax
        def print_stats(prefix, x, r):
            print(f"[DEBUG_JAX] {prefix} input x: shape={list(x.shape)} dtype={str(x.dtype)} | mean={float(x.mean()):.6f} std={float(x.std()):.6f} min={float(x.min()):.6f} max={float(x.max()):.6f}", flush=True)
            print(f"[DEBUG_JAX] {prefix} input router_logits: shape={list(r.shape)} dtype={str(r.dtype)} | mean={float(r.mean()):.6f} std={float(r.std()):.6f} min={float(r.min()):.6f} max={float(r.max()):.6f}", flush=True)
        jax.debug.callback(print_stats, prefix, x, router_logits)
    return _orig_apply_jax(self, layer, x, router_logits=router_logits)

Fp8FusedMoEMethod.apply_jax = patched_apply_jax

def custom_main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    use_chat_template = args.pop("use_chat_template")
    chat_template_kwargs = args.pop('chat_template_kwargs')

    llm = LLM(**args)

    sampling_params = llm.get_default_sampling_params()
    sampling_params.logprobs = 5
    sampling_params.max_tokens = 5
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    prompts = [
        "What is the speed of light in a vacuum?",
    ]

    print("Using custom main to dump logits and logprobs", flush=True)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0]
        print(f"Prompt: {prompt!r}")
        print(f"Prompt token IDs: {output.prompt_token_ids}")
        print(f"Generated text: {generated.text!r}")
        print(f"Generated token IDs: {generated.token_ids}")
        if generated.logprobs is not None and len(generated.logprobs) > 0:
            print(f"Generated logprobs (step 1): {generated.logprobs[0]}")
        print("-" * 50)

if __name__ == "__main__":
    os.environ.setdefault('SKIP_JAX_PRECOMPILE', '1')
    parser = create_parser()
    args = vars(parser.parse_args())
    custom_main(args)
