# Imported for their registration side effects; the out-of-tree
# RowParallelLinear in `linear` only affects layers built after the import, so
# this module must be imported before any model is constructed.
import inspect
import os
import vllm.model_executor.layers.quantization.modelopt as _modelopt
if not hasattr(_modelopt, "ModelOptNvFp4LinearMethod") and hasattr(_modelopt, "ModelOptLinearMethod"):
    _modelopt.ModelOptNvFp4LinearMethod = _modelopt.ModelOptLinearMethod

import vllm.v1.kv_cache_interface as _kv_iface
if hasattr(_kv_iface, "KVCacheTensor") and not hasattr(_kv_iface.KVCacheTensor, "shared_by"):
    _kv_iface.KVCacheTensor.shared_by = property(lambda self: self.layers)

for _cls_name in ("AttentionSpec", "FullAttentionSpec", "SlidingWindowSpec", "MLAAttentionSpec"):
    _spec_cls = getattr(_kv_iface, _cls_name, None)
    if _spec_cls is not None and "indexes_kv_by_block_stride" not in inspect.signature(_spec_cls.__init__).parameters:
        def _make_compat_init(_orig_init):
            def _compat_init(self, *args, **kwargs):
                stride = kwargs.pop("indexes_kv_by_block_stride", True)
                _orig_init(self, *args, **kwargs)
                object.__setattr__(self, "indexes_kv_by_block_stride", stride)
            return _compat_init
        _spec_cls.__init__ = _make_compat_init(_spec_cls.__init__)

import vllm.v1.attention.backends.utils as _attn_utils

if not hasattr(_attn_utils, "get_kv_cache_layout"):
    def get_kv_cache_layout() -> str:
        return os.environ.get("VLLM_KV_CACHE_LAYOUT", "NHD")

    get_kv_cache_layout.cache_clear = lambda: None
    _attn_utils.get_kv_cache_layout = get_kv_cache_layout

from vllm_torchtpu.layers.vllm import attention as attention
from vllm_torchtpu.layers.vllm import linear as linear
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.tokenizers import register_tokenizers


def register_layers() -> None:
    register_models()
    register_tokenizers()
