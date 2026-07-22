# Copyright 2026 Google LLC
# Wrapper to run JAX reference model with backend patch

import os
import sys

os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.32"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["VLLM_ENFORCE_EAGER"] = "1"
os.environ["VLLM_BYPASS_TPU_INIT"] = "1"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
torch._dynamo.config.disable = True

# Patch DeviceConfig to fallback to cpu if tpu device is not registered in PyTorch
try:
    from vllm.config.device import DeviceConfig
    _orig_device_config_post_init = DeviceConfig.__post_init__
    def patched_device_config_post_init(self):
        try:
            _orig_device_config_post_init(self)
        except RuntimeError as e:
            if "device string: tpu" in str(e):
                print("[JAX_VERIFY_PATCH] Caught device tpu RuntimeError, falling back to cpu device object", flush=True)
                self.device = torch.device("cpu")
            else:
                raise
    DeviceConfig.__post_init__ = patched_device_config_post_init
except ImportError:
    pass

# Patch custom_op.CustomOp.register_oot to ignore duplicate registrations
try:
    from vllm.model_executor.custom_op import CustomOp, op_registry_oot
    _orig_register_oot = CustomOp.register_oot
    @classmethod
    def patched_register_oot(cls, reg_name):
        print(f"[JAX_VERIFY_PATCH] register_oot called for: {reg_name} (current registry: {list(op_registry_oot.keys())})", flush=True)
        if reg_name in op_registry_oot:
            print(f"[JAX_VERIFY_PATCH] Ignoring duplicate custom op registration for: {reg_name}", flush=True)
            return lambda subclass: subclass
        return _orig_register_oot.__func__(cls, reg_name)
    CustomOp.register_oot = patched_register_oot
except Exception as e:
    print(f"[JAX_VERIFY_PATCH] Failed to patch register_oot: {e}", flush=True)

# Compatibility alias for renamed DeepseekV4Attention class in newer vLLM versions
try:
    import vllm.models.deepseek_v4.attention as att_mod
    if not hasattr(att_mod, "DeepseekV4Attention") and hasattr(att_mod, "DeepseekV4MLAAttention"):
        att_mod.DeepseekV4Attention = att_mod.DeepseekV4MLAAttention
except ImportError:
    pass

# sys.path.insert(0, "/mnt/workspace/vllm-torchtpu/src")

# Manually patch DeepSeek V4 MLA class to JAX wrapper to run under CPU/JAX platform
from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4_attention import patch_deepseek_v4_mla_cls
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

# Manually define weight name loader patch for the AMD model class
from vllm.models.deepseek_v4.amd.model import DeepseekV4ForCausalLM as _AmdDeepseekV4ForCausalLM
_amd_orig_load_weights = _AmdDeepseekV4ForCausalLM.load_weights

def patched_amd_load_weights(self, weights):
    try:
        if hasattr(self, "model") and hasattr(self.model, "layers") and len(self.model.layers) > 0:
            l = self.model.layers[0]
            if hasattr(l, "ffn") and hasattr(l.ffn, "shared_experts") and hasattr(l.ffn.shared_experts, "down_proj"):
                proj = l.ffn.shared_experts.down_proj
                print(f"[DEBUG_SHARED_EXPERTS_PARAMS] down_proj state_dict keys: {list(proj.state_dict().keys())}", flush=True)
                # Print class details of down_proj's linear method
                if hasattr(proj, "quant_method"):
                    print(f"[DEBUG_SHARED_EXPERTS_PARAMS] down_proj quant_method: {proj.quant_method.__class__.__name__}", flush=True)
        if hasattr(self, "model") and hasattr(self.model, "hc_head"):
            print(f"[DEBUG_HC_HEAD_PARAMS] hc_head state_dict keys: {list(self.model.hc_head.state_dict().keys())}", flush=True)
    except Exception as e:
        print(f"[DEBUG_SHARED_EXPERTS_PARAMS] Failed to print: {e}", flush=True)

    self.model.has_tilelang = False
    for idx, l in enumerate(self.model.layers):
        l.layer_idx = idx
        l.has_tilelang = False
        if hasattr(l, "ffn"):
            l.ffn.layer_idx = idx
            if hasattr(l.ffn, "experts"):
                l.ffn.experts.layer_idx = idx

    orig_mapper = self.hf_to_vllm_mapper
    class WrappedMapper:
        def apply(self, w):
            for name, loaded_weight in orig_mapper.apply(w):
                if ".experts." in name and name.endswith(".weight"):
                    if loaded_weight.ndim == 2:
                        loaded_weight = loaded_weight.t()

                if "hc_head.hc_" in name:
                    name = name.replace("hc_head.hc_", "hc_head_")
                    
                # Map MHC layer parameters from HF (nested) to vLLM (flat)
                if ".attn_hc.base" in name:
                    name = name.replace(".attn_hc.base", ".hc_attn_base")
                elif ".attn_hc.fn" in name:
                    name = name.replace(".attn_hc.fn", ".hc_attn_fn")
                elif ".attn_hc.weight_scale_inv" in name:
                    name = name.replace(".attn_hc.weight_scale_inv", ".hc_attn_scale")
                elif ".ffn_hc.base" in name:
                    name = name.replace(".ffn_hc.base", ".hc_ffn_base")
                elif ".ffn_hc.fn" in name:
                    name = name.replace(".ffn_hc.fn", ".hc_ffn_fn")
                elif ".ffn_hc.weight_scale_inv" in name:
                    name = name.replace(".ffn_hc.weight_scale_inv", ".hc_ffn_scale")
                    
                # Map shared experts from HF (gate_proj/up_proj/down_proj) to vLLM (w1/w3/w2)
                if ".shared_experts.gate_proj" in name:
                    name = name.replace(".shared_experts.gate_proj", ".shared_experts.w1")
                elif ".shared_experts.up_proj" in name:
                    name = name.replace(".shared_experts.up_proj", ".shared_experts.w3")
                    
                # Map layernorms
                if ".input_layernorm" in name:
                    name = name.replace(".input_layernorm", ".attn_norm")
                elif ".post_attention_layernorm" in name:
                    name = name.replace(".post_attention_layernorm", ".ffn_norm")
                
                # Map mlp to ffn
                if ".mlp." in name:
                    name = name.replace(".mlp.", ".ffn.")
                
                # Map attention parameters
                # 1. Indexer Compressor mappings (HF -> vLLM)
                if ".self_attn.compressor.indexer.gate_proj" in name:
                    name = name.replace(".self_attn.compressor.indexer.gate_proj", ".attn.indexer.compressor.wgate")
                elif ".self_attn.compressor.indexer.kv_proj" in name:
                    name = name.replace(".self_attn.compressor.indexer.kv_proj", ".attn.indexer.compressor.wkv")
                elif ".self_attn.compressor.indexer.kv_norm" in name:
                    name = name.replace(".self_attn.compressor.indexer.kv_norm", ".attn.indexer.compressor.norm")
                elif ".self_attn.compressor.indexer.position_bias" in name:
                    name = name.replace(".self_attn.compressor.indexer.position_bias", ".attn.indexer.compressor.ape")
                    
                # 2. Indexer main mappings (HF -> vLLM)
                elif ".self_attn.compressor.indexer.q_b_proj" in name:
                    name = name.replace(".self_attn.compressor.indexer.q_b_proj", ".attn.indexer.wq_b")
                elif ".self_attn.compressor.indexer.scorer.weights_proj" in name:
                    name = name.replace(".self_attn.compressor.indexer.scorer.weights_proj", ".attn.indexer.weights_proj")
                    
                # 3. Attention Compressor mappings (HF -> vLLM)
                elif ".self_attn.compressor.gate_proj" in name:
                    name = name.replace(".self_attn.compressor.gate_proj", ".attn.mla_attn.compressor.wgate")
                elif ".self_attn.compressor.kv_proj" in name:
                    name = name.replace(".self_attn.compressor.kv_proj", ".attn.mla_attn.compressor.wkv")
                elif ".self_attn.compressor.kv_norm" in name:
                    name = name.replace(".self_attn.compressor.kv_norm", ".attn.mla_attn.compressor.norm")
                elif ".self_attn.compressor.position_bias" in name:
                    name = name.replace(".self_attn.compressor.position_bias", ".attn.mla_attn.compressor.ape")
                elif ".self_attn.q_a_proj" in name:
                    name = name.replace(".self_attn.q_a_proj", ".attn.wq_a")
                elif ".self_attn.q_a_norm" in name:
                    name = name.replace(".self_attn.q_a_norm", ".attn.q_norm")
                elif ".self_attn.q_b_proj" in name:
                    name = name.replace(".self_attn.q_b_proj", ".attn.wq_b")
                elif ".self_attn.kv_proj" in name:
                    name = name.replace(".self_attn.kv_proj", ".attn.wkv")
                elif ".self_attn.kv_norm" in name:
                    name = name.replace(".self_attn.kv_norm", ".attn.kv_norm")
                elif ".self_attn.o_a_proj" in name:
                    name = name.replace(".self_attn.o_a_proj", ".attn.wo_a")
                elif ".self_attn.o_b_proj" in name:
                    name = name.replace(".self_attn.o_b_proj", ".attn.wo_b")
                elif ".self_attn.sinks" in name:
                    name = name.replace(".self_attn.sinks", ".attn.attn_sink")

                if (".experts." not in name or ".shared_experts." in name) and name.endswith("weight_scale"):
                    name = name.replace("weight_scale", "weight_scale_inv")

                yield name, loaded_weight

    self.hf_to_vllm_mapper = WrappedMapper()
    res = _amd_orig_load_weights(self, weights)

    # Wrap get_kv_cache_spec to return dummy specs instead of None to bypass JAX KV cache manager assertions
    from vllm.v1.attention.backends.mla.sparse_swa import SlidingWindowMLASpec
    import torch

    for name, module in self.model.named_modules():
        if hasattr(module, "get_kv_cache_spec"):
            orig_spec_fn = module.get_kv_cache_spec
            def make_wrapped_fn(orig_fn, mod_name):
                def wrapped_spec_fn(*args, **kwargs):
                    spec = orig_fn(*args, **kwargs)
                    if spec is None:
                        print(f"[JAX_VERIFY_PATCH] Returning dummy KVCacheSpec for unused module: {mod_name}", flush=True)
                        return SlidingWindowMLASpec(
                            block_size=1,
                            num_kv_heads=1,
                            head_size=128,
                            dtype=torch.uint8,
                            sliding_window=1,
                            cache_dtype_str="fp8",
                            model_version="deepseek_v4",
                        )
                    return spec
                return wrapped_spec_fn
            module.get_kv_cache_spec = make_wrapped_fn(orig_spec_fn, name)

    import torchax
    torchax.enable_globally()
    print("[RUN_JAX_VERIFY_PATCH] Enabled torchax globally after loading weights", flush=True)

    return res

_AmdDeepseekV4ForCausalLM.load_weights = patched_amd_load_weights

# Also patch inner DeepseekV4Model.load_weights
from vllm.models.deepseek_v4.amd.model import DeepseekV4Model
_amd_orig_inner_load_weights = DeepseekV4Model.load_weights

def patched_amd_inner_load_weights(self, weights):
    def wrap_weights(w_iter):
        for name, loaded_weight in w_iter:
            if ".attn." in name or ".self_attn." in name:
                continue
            if "hc_head.hc_" in name:
                name = name.replace("hc_head.hc_", "hc_head_")
            if (".experts." not in name or ".shared_experts." in name) and name.endswith("weight_scale"):
                name = name.replace("weight_scale", "weight_scale_inv")
            yield name, loaded_weight

    return _amd_orig_inner_load_weights(self, wrap_weights(weights))

DeepseekV4Model.load_weights = patched_amd_inner_load_weights

# Patch forward method to execute within torchax.default_env()
_orig_forward = _AmdDeepseekV4ForCausalLM.forward

def patched_forward(self, *args, **kwargs):
    import torchax
    with torchax.default_env():
        return _orig_forward(self, *args, **kwargs)

_AmdDeepseekV4ForCausalLM.forward = patched_forward

# Now add our repo to path and import offline_inference
sys.path.insert(0, "/root/cloud-devkit/vllm-torchtpu")

# Monkey-patch JAX quantization mapping for deepseek_v4_fp8
import vllm_torchtpu.layers.common.quant_methods as qm
qm.FP8 = "deepseek_v4_fp8"

from examples.offline_inference import create_parser
from vllm import LLM, SamplingParams

# Monkey patch JAX MoE apply_monolithic to log inputs
from vllm_torchtpu.layers.vllm.quantization.mxfp4 import VllmMxfp4MoEMethod
_orig_apply_monolithic = VllmMxfp4MoEMethod.apply_monolithic

def patched_apply_monolithic(self, layer, x, router_logits, input_ids=None):
    prefix = getattr(layer, "prefix", "")
    if "layers.0." in prefix or "layers.3." in prefix:
        import jax
        def print_stats(prefix, x_arr, r_arr):
            print(f"[DEBUG_JAX] {prefix} input x: shape={list(x_arr.shape)} dtype={str(x_arr.dtype)} | mean={float(x_arr.mean()):.6f} std={float(x_arr.std()):.6f} min={float(x_arr.min()):.6f} max={float(x_arr.max()):.6f}", flush=True)
            print(f"[DEBUG_JAX] {prefix} input router_logits: shape={list(r_arr.shape)} dtype={str(r_arr.dtype)} | mean={float(r_arr.mean()):.6f} std={float(r_arr.std()):.6f} min={float(r_arr.min()):.6f} max={float(r_arr.max()):.6f}", flush=True)
        x_elem = getattr(x, "_elem", x)
        r_elem = getattr(router_logits, "_elem", router_logits)
        jax.debug.callback(print_stats, prefix, x_elem, r_elem)

    res = _orig_apply_monolithic(self, layer, x, router_logits, input_ids)

    if "layers.0." in prefix or "layers.3." in prefix:
        import jax
        def print_output_stats(prefix, res_arr):
            print(f"[DEBUG_JAX] {prefix} output: shape={list(res_arr.shape)} dtype={str(res_arr.dtype)} | mean={float(res_arr.mean()):.6f} std={float(res_arr.std()):.6f} min={float(res_arr.min()):.6f} max={float(res_arr.max()):.6f}", flush=True)
        res_elem = getattr(res, "_elem", res)
        jax.debug.callback(print_output_stats, prefix, res_elem)

    return res

VllmMxfp4MoEMethod.apply_monolithic = patched_apply_monolithic

def custom_main(args: dict):
    # Patch DeepseekV4Attention to VllmDeepseekV4MLAAttention
    import vllm.models.deepseek_v4.amd.model as ds_v4_amd_model
    from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4_attention import VllmDeepseekV4MLAAttention
    ds_v4_amd_model.DeepseekV4Attention = VllmDeepseekV4MLAAttention
    print("[RUN_JAX_VERIFY_PATCH] Patched ds_v4_amd_model.DeepseekV4Attention -> VllmDeepseekV4MLAAttention", flush=True)

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
