"""Process-wide defaults required before torch_tpu autoloads."""

import os

# torch_tpu reads this during import. If unset, it marks c10d functional
# collectives as Dynamo graph breaks, which prevents vLLM TP graphs from
# compiling. This file is imported by Python before vLLM imports torch.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
                      "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

# Compatibility alias for renamed DeepseekV4Attention class in newer vLLM versions
try:
    import vllm.models.deepseek_v4.attention as att_mod
    if not hasattr(att_mod, "DeepseekV4Attention") and hasattr(att_mod, "DeepseekV4MLAAttention"):
        att_mod.DeepseekV4Attention = att_mod.DeepseekV4MLAAttention
except ImportError:
    pass

try:
    import vllm.models.deepseek_v4.amd.model as ds_v4_amd_model
    if not hasattr(ds_v4_amd_model, "DeepseekV4Attention"):
        if hasattr(ds_v4_amd_model, "DeepseekV4ROCMAiterMLAAttention"):
            ds_v4_amd_model.DeepseekV4Attention = ds_v4_amd_model.DeepseekV4ROCMAiterMLAAttention
        elif hasattr(ds_v4_amd_model, "DeepseekV4MLAAttention"):
            ds_v4_amd_model.DeepseekV4Attention = ds_v4_amd_model.DeepseekV4MLAAttention
except ImportError:
    pass
