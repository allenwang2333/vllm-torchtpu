# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Out-of-tree TPU custom ops and layer adapters for upstream Qwen4Exp."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from vllm.logger import init_logger

logger = init_logger(__name__)

_CONFIG_MOD = None


def _load_upstream_config_module():
    global _CONFIG_MOD
    if _CONFIG_MOD is not None:
        return _CONFIG_MOD

    from transformers.models.qwen3_next.configuration_qwen3_next import (
        Qwen3NextConfig,
    )

    if "vllm.transformers_utils.configs.qwen3_next" not in sys.modules:
        q3n_mod = types.ModuleType("vllm.transformers_utils.configs.qwen3_next")
        q3n_mod.Qwen3NextConfig = Qwen3NextConfig
        sys.modules["vllm.transformers_utils.configs.qwen3_next"] = q3n_mod

    vllm_spec = importlib.util.find_spec("vllm")
    vllm_dir = os.path.dirname(vllm_spec.origin)
    models_dir = os.path.join(vllm_dir, "models")
    qwen4_dir = os.path.join(models_dir, "qwen4_exp")
    cfg_path = os.path.join(qwen4_dir, "config.py")
    if "vllm.models" not in sys.modules:
        models_mod = types.ModuleType("vllm.models")
        models_mod.__path__ = [models_dir]
        sys.modules["vllm.models"] = models_mod
    if "vllm.models.qwen4_exp" not in sys.modules:
        qwen4_mod = types.ModuleType("vllm.models.qwen4_exp")
        qwen4_mod.__path__ = [qwen4_dir]
        sys.modules["vllm.models.qwen4_exp"] = qwen4_mod
    if "vllm.models.qwen4_exp.config" in sys.modules:
        mod = sys.modules["vllm.models.qwen4_exp.config"]
    else:
        spec = importlib.util.spec_from_file_location(
            "vllm.models.qwen4_exp.config", cfg_path
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["vllm.models.qwen4_exp.config"] = mod
        spec.loader.exec_module(mod)
    _CONFIG_MOD = mod
    return mod


def register_qwen4_exp_config() -> None:
    """Register Qwen4Exp configs with HuggingFace AutoConfig without importing vllm.model_executor."""
    cfg_mod = _load_upstream_config_module()
    CONFIG_MAPPING.register("qwen4_exp", cfg_mod.Qwen4ExpConfig, exist_ok=True)
    CONFIG_MAPPING.register(
        "qwen4_exp_text", cfg_mod.Qwen4ExpTextConfig, exist_ok=True
    )


def _inject_upstream_shims() -> None:
    """Inject missing symbols into upstream vllm before importing qwen4_exp.nvidia."""
    from vllm.distributed import get_tp_group
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig as _BaseQuantConfig,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    import vllm.model_executor.layers.fused_moe.utils as _moe_utils

    if not hasattr(_moe_utils, "is_model_fused_shared_expert_compatible"):
        _moe_utils.is_model_fused_shared_expert_compatible = (
            lambda *a, **kw: False
        )

    import vllm.model_executor.layers.mamba.mamba_utils as _mamba_utils

    if not hasattr(_mamba_utils, "MambaStateCopyFuncsByType"):
        _mamba_utils.MambaStateCopyFuncsByType = dict

    import vllm.distributed as _dist
    import vllm.distributed.parallel_state as _parallel_state

    def _safe_get_etp_group():
        etp = getattr(_parallel_state, "_ETP", None)
        return etp if etp is not None else get_tp_group()

    _dist.get_etp_group = _safe_get_etp_group
    _parallel_state.get_etp_group = _safe_get_etp_group

    import vllm.model_executor.layers.quantization.modelopt as _modelopt

    if not hasattr(_modelopt, "ModelOptMixedPrecisionConfig"):
        _modelopt.ModelOptMixedPrecisionConfig = type(
            "ModelOptMixedPrecisionConfig", (), {}
        )

    if "vllm.model_executor.layers.quantization.unquantized" not in sys.modules:
        _unq_mod = types.ModuleType(
            "vllm.model_executor.layers.quantization.unquantized"
        )
        _unq_mod.VllmUnquantizedConfig = _BaseQuantConfig
        sys.modules[
            "vllm.model_executor.layers.quantization.unquantized"
        ] = _unq_mod

    import vllm.v1.attention.backends.short_conv_attn as _short_conv_attn

    if not hasattr(_short_conv_attn, "PleShortConvAttentionBackend"):
        _short_conv_attn.PleShortConvAttentionBackend = type(
            "PleShortConvAttentionBackend", (), {}
        )
    if not hasattr(_short_conv_attn, "PleShortConvAttentionMetadata"):
        _short_conv_attn.PleShortConvAttentionMetadata = type(
            "PleShortConvAttentionMetadata", (), {}
        )

    if not getattr(VocabParallelEmbedding, "_qwen4_patched", False):
        _orig_vpe_init = VocabParallelEmbedding.__init__

        def _patched_vpe_init(self, *args, **kwargs):
            kwargs.pop("shard_by_vocab", None)
            qc = kwargs.get("quant_config")
            if isinstance(qc, type) and issubclass(qc, _BaseQuantConfig):
                kwargs["quant_config"] = None
            _orig_vpe_init(self, *args, **kwargs)

        VocabParallelEmbedding.__init__ = _patched_vpe_init
        VocabParallelEmbedding._qwen4_patched = True


def _build_tpu_hc_ops_module() -> types.ModuleType:
    """Create PyTorch-XLA implementation of vllm.models.qwen4_exp.nvidia.ops.hc."""
    mod = types.ModuleType("vllm.models.qwen4_exp.nvidia.ops.hc")

    def grouped_gemma_rmsnorm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        num_groups: int,
    ) -> torch.Tensor:
        dim = x.shape[-1]
        group_dim = dim // num_groups
        grouped = x.float().unflatten(-1, (num_groups, group_dim))
        variance = grouped.square().mean(-1, keepdim=True)
        expected = grouped * torch.rsqrt(variance + eps)
        expected = expected.flatten(-2) * (1.0 + weight.float())
        return expected.to(x.dtype)

    def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
        return F.silu(x.float() / hc_count).to(x.dtype)

    def hc_gate_mix(
        x: torch.Tensor,
        gate: torch.Tensor,
        hc_count: int,
    ) -> torch.Tensor:
        dim = gate.shape[-1]
        hc_dim = dim // hc_count
        expected = (
            torch.sigmoid(gate.float().unflatten(-1, (hc_count, hc_dim)))
            * x.float().unflatten(-1, (hc_count, hc_dim))
        ).mean(-2)
        return expected.to(x.dtype)

    def hc_combine(
        residual: torch.Tensor,
        block_output: torch.Tensor,
        injection_logits: torch.Tensor | None,
        hc_count: int,
    ) -> torch.Tensor:
        dim = residual.shape[-1]
        hc_dim = dim // hc_count
        expected = residual.float().unflatten(-1, (hc_count, hc_dim))
        if injection_logits is not None:
            injection_weight = 2.0 * torch.sigmoid(
                injection_logits.float() / hc_count
            )
            expected = expected + block_output.float().unsqueeze(
                -2
            ) * injection_weight.unsqueeze(-1)
        else:
            expected = expected + block_output.float().unsqueeze(-2)
        return expected.flatten(-2).to(residual.dtype)

    def hc_combine_norm(
        residual: torch.Tensor,
        block_output: torch.Tensor,
        injection_logits: torch.Tensor | None,
        weight: torch.Tensor,
        eps: float,
        hc_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = hc_combine(
            residual, block_output, injection_logits, hc_count
        )
        normed = grouped_gemma_rmsnorm(combined, weight, eps, hc_count)
        return combined, normed

    mod.grouped_gemma_rmsnorm = grouped_gemma_rmsnorm
    mod.hc_silu = hc_silu
    mod.hc_gate_mix = hc_gate_mix
    mod.hc_combine = hc_combine
    mod.hc_combine_norm = hc_combine_norm
    return mod


def _build_tpu_qsa_attention_cls():
    from vllm.config import CacheConfig, VllmConfig
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization import QuantizationConfig
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

    cfg_mod = _load_upstream_config_module()
    Qwen4ExpTextConfig = cfg_mod.Qwen4ExpTextConfig

    class Qwen4ExpQSAAttention(Qwen3NextAttention):
        """TPU QSA Attention adapter inheriting from Qwen3NextAttention."""

        def __init__(
            self,
            vllm_config: VllmConfig,
            config: Qwen4ExpTextConfig,
            layer_id: int = 0,
            layer_idx: int | None = None,
            prefix: str = "",
            cache_config: CacheConfig | None = None,
            quant_config: QuantizationConfig | None = None,
            reduce_results: bool = True,
        ) -> None:
            if cache_config is None and vllm_config is not None:
                cache_config = getattr(vllm_config, "cache_config", None)
            super().__init__(
                config=config,
                model_config=vllm_config.model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                reduce_results=reduce_results,
                prefix=prefix,
            )
            self.layer_idx = layer_idx if layer_idx is not None else layer_id

            indexer_n_heads = getattr(config, "indexer_n_heads", 4)
            indexer_kv_heads = getattr(config, "indexer_kv_heads", 1)
            indexer_head_dim = getattr(config, "indexer_head_dim", 128)

            class _TPUQSAIndexer(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.index_qk_proj = ReplicatedLinear(
                        config.hidden_size,
                        (indexer_n_heads + indexer_kv_heads) * indexer_head_dim,
                        bias=False,
                        quant_config=quant_config,
                        prefix=f"{prefix}.indexer.index_qk_proj",
                    )
                    self.q_layernorm = GemmaRMSNorm(
                        indexer_head_dim, eps=config.rms_norm_eps
                    )
                    self.k_layernorm = GemmaRMSNorm(
                        indexer_head_dim, eps=config.rms_norm_eps
                    )

            self.indexer = _TPUQSAIndexer()

    return Qwen4ExpQSAAttention


def _build_tpu_ple_ops_module() -> types.ModuleType:
    """Create a module implementing vllm.models.qwen4_exp.nvidia.ops.ple on PyTorch/XLA."""
    ple_ops = types.ModuleType("vllm.models.qwen4_exp.nvidia.ops.ple")

    def ple_gate(
        key: torch.Tensor,
        value: torch.Tensor,
        hidden: torch.Tensor,
        norm_key_w: torch.Tensor,
        norm_query_w: torch.Tensor,
        norm_conv_w: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = hidden.dtype
        T = hidden.shape[0]
        H = value.shape[-1]
        HC = hidden.shape[-1] // H

        k = key.float().view(T, HC, H)
        q = hidden.float().view(T, HC, H)
        nk = norm_key_w.float().view(1, HC, H)
        nq = norm_query_w.float().view(1, HC, H)

        k_n = (
            k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + nk)
        ).to(orig_dtype)
        q_n = (
            q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + nq)
        ).to(orig_dtype)

        products = (k_n.float() * q_n.float()).to(orig_dtype)
        dot = products.float().sum(dim=-1, keepdim=True).to(orig_dtype).float()
        d = (dot / (H**0.5)).to(orig_dtype).float()
        sign = torch.where(d < 0, -1.0, torch.where(d > 0, 1.0, 0.0))
        magnitude = torch.sqrt(torch.clamp(torch.abs(d), min=1e-6)).to(
            orig_dtype
        )
        g = torch.sigmoid(sign * magnitude).to(orig_dtype).float()

        v = value.float().view(T, 1, H)
        gated = (g * v).to(orig_dtype)
        gf = gated.float()
        ncw = norm_conv_w.float().view(1, HC, H)
        normed = (
            gf
            * torch.rsqrt(gf.pow(2).mean(dim=-1, keepdim=True) + eps)
            * (1.0 + ncw)
        ).to(orig_dtype)

        return gated.reshape(T, HC * H), normed.reshape(T, HC * H)

    def ple_conv(*args, **kwargs):
        return None

    def ple_ngram_ids(*args, **kwargs):
        raise RuntimeError("ple_ngram_ids CUDA kernel is unused on TPU")

    ple_ops.ple_gate = ple_gate
    ple_ops.ple_conv = ple_conv
    ple_ops.ple_ngram_ids = ple_ngram_ids
    return ple_ops


def _to_int_list(val) -> list[int] | None:
    if val is None:
        return None
    import numpy as np

    if isinstance(val, torch.Tensor):
        return val.detach().cpu().reshape(-1).tolist()
    try:
        return np.asarray(val).reshape(-1).astype(np.int64).tolist()
    except Exception:
        return None


def _extract_tpu_batch_meta(
    first_meta, num_tokens: int
) -> list[tuple[int, int, int, int]]:
    """Return list of (start, end, seq_len, slot_id) for active requests in batch."""
    if first_meta is None:
        return [(0, num_tokens, num_tokens, 0)]
    qsl_raw = getattr(first_meta, "query_start_loc_cpu", None)
    if qsl_raw is None:
        qsl_raw = getattr(first_meta, "query_start_loc", None)
    qsl_list = _to_int_list(qsl_raw)

    sl_raw = getattr(first_meta, "seq_lens_cpu", None)
    if sl_raw is None:
        sl_raw = getattr(first_meta, "seq_lens", None)
    sl_list = _to_int_list(sl_raw)

    slot_raw = getattr(first_meta, "mamba_state_indices", None)
    slot_list = _to_int_list(slot_raw)

    active_reqs: list[tuple[int, int, int, int]] = []
    if qsl_list is not None and len(qsl_list) >= 2:
        num_cand = len(qsl_list) - 1
        for i in range(num_cand):
            start = int(qsl_list[i])
            end = int(qsl_list[i + 1])
            length = end - start
            if length <= 0:
                continue
            seq_len = (
                int(sl_list[i])
                if (sl_list is not None and i < len(sl_list))
                else length
            )
            if seq_len <= 0:
                continue
            slot_id = (
                int(slot_list[i])
                if (slot_list is not None and i < len(slot_list))
                else i
            )
            active_reqs.append((start, end, seq_len, slot_id))
    elif sl_list is not None and len(sl_list) > 0:
        offset = 0
        for i, s_len in enumerate(sl_list):
            seq_len = int(s_len)
            if seq_len <= 0:
                continue
            slot_id = (
                int(slot_list[i])
                if (slot_list is not None and i < len(slot_list))
                else i
            )
            active_reqs.append((offset, offset + 1, seq_len, slot_id))
            offset += 1

    if not active_reqs:
        return [(0, num_tokens, num_tokens, 0)]
    return active_reqs


def _tpu_short_conv(
    self,
    conv_input: torch.Tensor,
    gated_output: torch.Tensor,
    query_start_loc: torch.Tensor | None = None,
) -> None:
    if self.conv_state_len == 0:
        return
    from vllm.forward_context import get_forward_context

    first_meta = None
    try:
        attn_meta = get_forward_context().attn_metadata
        if isinstance(attn_meta, dict) and attn_meta:
            first_meta = next(iter(attn_meta.values()))
        else:
            first_meta = attn_meta
    except Exception:
        pass

    active_reqs = _extract_tpu_batch_meta(first_meta, conv_input.shape[0])
    conv_weight = self.conv1d.weight.to(conv_input.dtype)
    for start, end, seq_len, slot_id in active_reqs:
        length = end - start
        if length <= 0:
            continue
        slot_idx = int(slot_id) % self._conv_history_cache.shape[0]
        is_new_prefill = seq_len <= length
        if is_new_prefill:
            self._conv_history_cache[slot_idx].zero_()
        initial_history = self._conv_history_cache[slot_idx].to(
            conv_input.dtype
        )
        req_tokens = conv_input[start:end].transpose(0, 1)  # [C, L]
        history = torch.cat([initial_history, req_tokens], dim=-1)
        conv_out = F.conv1d(
            history.unsqueeze(0),
            conv_weight,
            groups=self.hc_hidden_size,
            dilation=self.short_conv_dilation,
        ).squeeze(0)
        conv_out = F.silu(conv_out).transpose(0, 1)  # [L, C]
        gated_output[start:end].add_(conv_out)
        self._conv_history_cache[slot_idx].copy_(
            history[:, -self.conv_state_len :].to(
                self._conv_history_cache.dtype
            )
        )


def _tpu_ple_forward(
    self,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
) -> torch.Tensor:
    input_ids = input_ids.reshape(-1)
    if input_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            "PLE expects input_ids and hidden_states to have the same "
            f"token length, got {input_ids.shape[0]} and "
            f"{hidden_states.shape[0]}"
        )
    embeddings = self.ple_embedding(
        hidden_states,
        input_ids,
        query_start_loc,
        ngram_context,
    )
    embeddings = self._dequantize_embeddings(
        embeddings, hidden_states.dtype
    )
    kv, _ = self.kv_proj(embeddings)
    key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
    gated_output, conv_input = self.ple_gate_fn(
        key,
        value,
        hidden_states,
        self.norm_key.weight,
        self.norm_query.weight,
        self.norm_conv.weight,
        self.norm_key.eps,
    )
    self._short_conv(conv_input, gated_output, query_start_loc)
    return gated_output


_QWEN4_OPS_REGISTERED = False


def register_qwen4_exp_ops() -> None:
    """Register Qwen4Exp TPU ops and wire upstream model into ModelRegistry."""
    global _QWEN4_OPS_REGISTERED
    register_qwen4_exp_config()
    if _QWEN4_OPS_REGISTERED:
        return

    _inject_upstream_shims()

    # 1. Register TPU HC and PLE custom ops so upstream hyperconnection and ple_layer work directly
    hc_ops = _build_tpu_hc_ops_module()
    ple_ops = _build_tpu_ple_ops_module()
    sys.modules["vllm.models.qwen4_exp.nvidia.ops.hc"] = hc_ops
    sys.modules["vllm.models.qwen4_exp.nvidia.ops.ple"] = ple_ops

    Qwen4ExpQSAAttention = _build_tpu_qsa_attention_cls()

    llg_mod = types.ModuleType("vllm.models.qwen4_exp.nvidia.low_latency_gemm")
    llg_mod.enable_qwen4_exp_low_latency_gemm = lambda *a, **kw: None
    sys.modules["vllm.models.qwen4_exp.nvidia.low_latency_gemm"] = llg_mod

    qsa_mod = types.ModuleType("vllm.models.qwen4_exp.nvidia.qsa")
    qsa_mod.Qwen4ExpQSAAttention = Qwen4ExpQSAAttention
    sys.modules["vllm.models.qwen4_exp.nvidia.qsa"] = qsa_mod

    import vllm.models.qwen4_exp.nvidia.hyperconnection as _upstream_hc

    _upstream_hc.grouped_gemma_rmsnorm = hc_ops.grouped_gemma_rmsnorm
    _upstream_hc.hc_silu = hc_ops.hc_silu
    _upstream_hc.hc_gate_mix = hc_ops.hc_gate_mix
    _upstream_hc.hc_combine = hc_ops.hc_combine
    _upstream_hc.hc_combine_norm = hc_ops.hc_combine_norm

    import vllm.models.qwen4_exp.nvidia.ngram_embedding as _upstream_ngram

    _orig_ple_from_quant = (
        _upstream_ngram.Qwen4ExpPLEEmbeddingMethod.from_quant_config
    )

    def _tpu_ple_from_quant(quant_config, prefix, embedding_dtype=None):
        if (
            quant_config is not None
            and type(quant_config).__name__ == "VllmUnquantizedConfig"
        ):
            quant_config = None
        return _orig_ple_from_quant(quant_config, prefix, embedding_dtype)

    _upstream_ngram.Qwen4ExpPLEEmbeddingMethod.from_quant_config = staticmethod(
        _tpu_ple_from_quant
    )

    import vllm.models.qwen4_exp.nvidia.ple_layer as _upstream_ple

    _upstream_ple.ple_gate = ple_ops.ple_gate
    _upstream_ple.ple_conv = ple_ops.ple_conv
    _orig_ple_init = _upstream_ple.Qwen4ExpPLELayer.__init__

    def _tpu_ple_init(self, *args, **kwargs):
        _orig_ple_init(self, *args, **kwargs)
        try:
            from vllm.config import get_current_vllm_config

            ctx = (
                get_current_vllm_config().compilation_config.static_forward_context
            )
            ctx.pop(self.prefix, None)
        except Exception:
            pass
        vllm_config = kwargs.get("vllm_config") or (
            args[1] if len(args) > 1 else None
        )
        max_num_seqs = 256
        dtype = torch.bfloat16
        if vllm_config is not None:
            max_num_seqs = max(
                getattr(vllm_config.scheduler_config, "max_num_seqs", 256), 256
            )
            dtype = getattr(vllm_config.model_config, "dtype", torch.bfloat16)
        self.register_buffer(
            "_conv_history_cache",
            torch.zeros(
                max_num_seqs,
                self.hc_hidden_size,
                self.conv_state_len,
                dtype=dtype,
            ),
            persistent=False,
        )

    _upstream_ple.Qwen4ExpPLELayer.__bases__ = (nn.Module,)
    for _attr in (
        "mamba_type",
        "get_attn_backend",
        "get_state_dtype",
        "get_state_shape",
        "is_kv_cache_tp_replicated",
    ):
        if hasattr(_upstream_ple.Qwen4ExpPLELayer, _attr):
            try:
                delattr(_upstream_ple.Qwen4ExpPLELayer, _attr)
            except Exception:
                pass
    _upstream_ple.Qwen4ExpPLELayer.ple_gate_fn = staticmethod(ple_ops.ple_gate)
    _upstream_ple.Qwen4ExpPLELayer.__init__ = _tpu_ple_init
    _upstream_ple.Qwen4ExpPLELayer._short_conv = _tpu_short_conv
    _upstream_ple.Qwen4ExpPLELayer.forward = _tpu_ple_forward
    _upstream_ple.Qwen4ExpPLELayer.get_kv_cache_spec = (
        lambda self, vllm_config: None
    )

    import vllm.models.qwen4_exp.nvidia.model as _upstream_model
    from vllm.forward_context import get_forward_context

    _upstream_model.Qwen4ExpQSAAttention = Qwen4ExpQSAAttention
    _upstream_model.enable_qwen4_exp_low_latency_gemm = lambda *a, **kw: None
    _orig_model_init = _upstream_model.Qwen4ExpModel.__init__
    _orig_embed = _upstream_model.Qwen4ExpModel.embed_input_ids
    _orig_forward = _upstream_model.Qwen4ExpModel.forward

    def _tpu_model_init(self, *args, **kwargs):
        _orig_model_init(self, *args, **kwargs)
        self.do_not_compile = True
        try:
            from vllm.compilation.counter import compilation_counter

            if compilation_counter.num_models_seen > 0:
                compilation_counter.num_models_seen -= 1
        except Exception:
            pass
        vllm_config = kwargs.get("vllm_config") or (
            args[0] if len(args) > 0 else None
        )
        max_num_seqs = 256
        if vllm_config is not None:
            max_num_seqs = max(
                getattr(vllm_config.scheduler_config, "max_num_seqs", 256), 256
            )
        eos_val = getattr(self.config, "eos_token_id", 0)
        if isinstance(eos_val, (list, tuple)):
            eos_val = eos_val[0] if len(eos_val) > 0 else 0
        eos_id = int(eos_val) if eos_val is not None else 0
        ngram_ctx_len = max(int(getattr(self.config, "ngram_size", 3)) - 1, 1)
        self.register_buffer(
            "_ngram_context_cache",
            torch.full(
                (max_num_seqs, ngram_ctx_len),
                eos_id,
                dtype=torch.int32,
            ),
            persistent=False,
        )

    def _tpu_embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._last_embedded_input_ids = input_ids
        return _orig_embed(self, input_ids)

    def _prepare_tpu_ple_inputs(
        self,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if input_ids is None or not getattr(self.config, "ple_layer_ids", None):
            return query_start_loc, ngram_context
        num_tokens = input_ids.numel()
        eos_val = getattr(self.config, "eos_token_id", 0)
        if isinstance(eos_val, (list, tuple)):
            eos_val = eos_val[0] if len(eos_val) > 0 else 0
        eos_id = int(eos_val) if eos_val is not None else 0
        ngram_ctx_len = max(int(getattr(self.config, "ngram_size", 3)) - 1, 1)

        if query_start_loc is not None and ngram_context is not None:
            return query_start_loc, ngram_context

        first_meta = None
        try:
            attn_meta = get_forward_context().attn_metadata
            if isinstance(attn_meta, dict) and attn_meta:
                first_meta = next(iter(attn_meta.values()))
            else:
                first_meta = attn_meta
        except Exception:
            pass

        active_reqs = _extract_tpu_batch_meta(first_meta, num_tokens)
        starts_list: list[int] = [0]
        ngram_rows: list[torch.Tensor] = []
        flat_ids = input_ids.reshape(-1)

        for start, end, seq_len, slot_id in active_reqs:
            length = end - start
            if length <= 0:
                continue
            slot_idx = int(slot_id) % self._ngram_context_cache.shape[0]
            is_new_prefill = seq_len <= length
            if is_new_prefill:
                self._ngram_context_cache[slot_idx].fill_(eos_id)
            ctx_row = self._ngram_context_cache[slot_idx].to(input_ids.device)
            ngram_rows.append(ctx_row)
            starts_list.append(end)
            history = torch.cat(
                [
                    ctx_row.to(flat_ids.dtype),
                    flat_ids[start:end],
                ]
            )
            self._ngram_context_cache[slot_idx].copy_(
                history[-ngram_ctx_len:].to(torch.int32)
            )

        if starts_list[-1] < num_tokens:
            starts_list.append(num_tokens)
            ngram_rows.append(
                torch.full(
                    (ngram_ctx_len,),
                    eos_id,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )

        query_start_loc = torch.tensor(
            starts_list, dtype=torch.int32, device=input_ids.device
        )
        ngram_context = torch.stack(ngram_rows, dim=0)
        return query_start_loc, ngram_context

    def _tpu_forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        deepstack_input_embeds=None,
    ):
        T = (
            inputs_embeds.shape[0]
            if inputs_embeds is not None
            else (
                input_ids.shape[0]
                if input_ids is not None
                else positions.shape[-1]
            )
        )
        if input_ids is None:
            cached_ids = getattr(self, "_last_embedded_input_ids", None)
            if cached_ids is not None and cached_ids.numel() == T:
                input_ids = cached_ids.reshape(-1)
                self._last_embedded_input_ids = None
        if input_ids is None or input_ids.numel() != T:
            input_ids = torch.zeros(
                (T,),
                device=positions.device,
                dtype=torch.int64,
            )
        query_start_loc, ngram_context = _prepare_tpu_ple_inputs(
            self, input_ids, query_start_loc, ngram_context
        )
        return _orig_forward(
            self,
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
            deepstack_input_embeds=deepstack_input_embeds,
        )

    from vllm.compilation.decorators import IGNORE_COMPILE_KEY

    setattr(_upstream_model.Qwen4ExpModel, IGNORE_COMPILE_KEY, True)
    _upstream_model.Qwen4ExpModel.__bases__ = (nn.Module,)
    _upstream_model.Qwen4ExpModel.__call__ = nn.Module.__call__
    _upstream_model.Qwen4ExpModel.__init__ = _tpu_model_init
    _upstream_model.Qwen4ExpModel.embed_input_ids = _tpu_embed_input_ids
    _upstream_model.Qwen4ExpModel.forward = _tpu_forward

    from vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op import (
        VllmGatedDeltaNetAttention,
    )

    _upstream_model.QwenGatedDeltaNetAttention = VllmGatedDeltaNetAttention

    import vllm.models.qwen4_exp.nvidia.ngram_embedding as _upstream_ngram_emb

    _orig_from_quant_config = (
        _upstream_ngram_emb.Qwen4ExpPLEEmbeddingMethod.from_quant_config
    )

    def _tpu_from_quant_config(
        quant_config,
        prefix: str,
        embedding_dtype: str | None = None,
    ):
        try:
            return _orig_from_quant_config(
                quant_config, prefix, embedding_dtype
            )
        except NotImplementedError:
            return _upstream_ngram_emb.Qwen4ExpPLEUnquantizedEmbeddingMethod()

    _upstream_ngram_emb.Qwen4ExpPLEEmbeddingMethod.from_quant_config = (
        staticmethod(_tpu_from_quant_config)
    )

    from vllm.distributed import get_tp_group
    import vllm.model_executor.layers.vocab_parallel_embedding as vpe
    import vllm.models.qwen4_exp.common.ple as _upstream_common_ple

    _upstream_common_ple.PLEVocabParallelEmbedding.__init__ = (
        vpe.VocabParallelEmbedding.__init__
    )

    def _tpu_ple_embedding_init(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
    ) -> None:
        del num_ngram_heads, max_total_tokens
        vpe.VocabParallelEmbedding.__init__(
            self,
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            quant_method=embedding_method,
            parallel_group=get_tp_group(),
        )
        if not hasattr(self, "org_vocab_start_index"):
            self.register_buffer(
                "org_vocab_start_index",
                torch.tensor(
                    self.shard_indices.org_vocab_start_index, dtype=torch.int64
                ),
                persistent=False,
            )
            self.register_buffer(
                "org_vocab_end_index",
                torch.tensor(
                    self.shard_indices.org_vocab_end_index, dtype=torch.int64
                ),
                persistent=False,
            )
            self.register_buffer(
                "num_org_vocab_padding",
                torch.tensor(
                    self.shard_indices.num_org_vocab_padding, dtype=torch.int64
                ),
                persistent=False,
            )
            self.register_buffer(
                "added_vocab_start_index",
                torch.tensor(
                    self.shard_indices.added_vocab_start_index,
                    dtype=torch.int64,
                ),
                persistent=False,
            )
            self.register_buffer(
                "added_vocab_end_index",
                torch.tensor(
                    self.shard_indices.added_vocab_end_index, dtype=torch.int64
                ),
                persistent=False,
            )
        self.embedding_method = embedding_method
        self.data_parallel_rank = data_parallel_rank
        self.etp_data_parallel_size = 1

    _upstream_ngram_emb.Qwen4ExpPLEEmbedding.__init__ = _tpu_ple_embedding_init

    from vllm.v1.kv_cache_interface import MambaSpec

    def _tpu_get_mamba_specs_from_config(cls, vllm_config):
        return (
            MambaSpec(
                shapes=_upstream_model.Qwen4ExpForCausalLM.get_gdn_mamba_state_shape_from_config(
                    vllm_config
                ),
                dtypes=_upstream_model.Qwen4ExpForCausalLM.get_gdn_mamba_state_dtype_from_config(
                    vllm_config
                ),
                block_size=-1,
            ),
        )

    _upstream_model.Qwen4ExpForCausalLM.get_mamba_specs_from_config = (
        classmethod(_tpu_get_mamba_specs_from_config)
    )
    _upstream_model.Qwen4ExpForConditionalGeneration.get_mamba_specs_from_config = (
        classmethod(_tpu_get_mamba_specs_from_config)
    )
    _upstream_model.Qwen4ExpForCausalLM.hf_to_vllm_mapper.orig_to_new_prefix.update(
        {"model.visual.": None, "visual.": None}
    )
    for _cls in (
        _upstream_model.Qwen4ExpForCausalLM,
        _upstream_model.Qwen4ExpForConditionalGeneration,
    ):
        for _attr in ("get_mamba_state_copy_funcs", "get_model_state_cls"):
            if hasattr(_cls, _attr):
                try:
                    delattr(_cls, _attr)
                except Exception:
                    pass

    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "Qwen4ExpForCausalLM",
        "vllm.models.qwen4_exp.nvidia.model:Qwen4ExpForCausalLM",
    )
    ModelRegistry.register_model(
        "Qwen4ExpForConditionalGeneration",
        "vllm.models.qwen4_exp.nvidia.model:Qwen4ExpForConditionalGeneration",
    )
    _QWEN4_OPS_REGISTERED = True
    logger.info("Applied TPU patch: Qwen4Exp custom ops and ModelRegistry adapter.")

