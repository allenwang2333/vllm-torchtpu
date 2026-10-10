# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU implementation of the DeepSeek-V4.1 DSpark speculative decoding draft model."""

from __future__ import annotations

import re
import typing
from collections.abc import Callable, Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.offloader import NoopOffloader, set_offloader

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.tp_collectives import (
    tp_all_gather,
    tp_shard_slice,
    tp_token_shard_ok,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_flash_mhc_op import (  # noqa: E501
    get_mhc_final_collapse_norm_v41_op,
)
from vllm_torchtpu.models.vllm.deepseek_v4.model import _HostStagedParams
from vllm_torchtpu.models.vllm.deepseek_v41.model import DeepseekV41DecoderLayer

logger = init_logger(__name__)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


class DSparkDeepseekV41Model(nn.Module):
    """DeepSeek-V4.1 DSpark 3-layer parallel draft transformer on TPU."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        set_offloader(NoopOffloader())
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.rms_norm_eps = float(config.rms_norm_eps)
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.num_dspark_layers = (
            getattr(config, "n_mtp_layers", None)
            or getattr(config, "num_nextn_predict_layers", None)
            or 3
        )

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
        self.register_buffer(
            "tp_rank_idx",
            torch.tensor([tp_rank], dtype=torch.int64),
            persistent=False,
        )
        init_pre_mix = torch.zeros((1, config.hc_mult), dtype=torch.float32)
        init_pre_mix[0, 0] = 1.0
        self.register_buffer("init_pre_mix", init_pre_mix, persistent=False)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )
        from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

        _ROPE_DICT.clear()
        # Draft layers use the smaller DSpark expert set.
        moe_overrides = {
            "n_routed_experts": config.dspark_n_routed_experts,
            "num_experts_per_tok": config.dspark_num_experts_per_tok,
        }
        self.layers = nn.ModuleList(
            [
                DeepseekV41DecoderLayer(
                    vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                    topk_indices_buffer=self.topk_indices_buffer,
                    engram_layout=None,
                    moe_overrides=moe_overrides,
                )
                for i in range(self.num_dspark_layers)
            ]
        )
        for layer in self.layers:
            layer.attn.non_causal_multi_token_decode = True

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self._fused_ctx_wkv_weight: torch.Tensor | None = None
        self._fused_ctx_wkv_scale: torch.Tensor | None = None

        from vllm_torchtpu.models.vllm.deepseek_v41.attention import (
            _get_dspark_swa_store_op,
        )
        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            get_vllm_model_wrapper_context,
        )

        if get_vllm_model_wrapper_context().mesh is not None:
            for _ps in (16, 32, 64, 128):
                _ = _get_dspark_swa_store_op(_ps)

    def process_weights_after_loading(self) -> None:
        from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
        from vllm_torchtpu.utils import synchronize_tensors

        slices_w: list[torch.Tensor] = []
        slices_s: list[torch.Tensor] = []
        use_quant = False
        for layer in self.layers:
            attn = layer.attn
            attn.process_weights_after_loading(torch.bfloat16)
            if hasattr(attn, "wkv_weight") and hasattr(attn, "wkv_weight_scale"):
                use_quant = True
                slices_w.append(attn.wkv_weight)
                slices_s.append(attn.wkv_weight_scale)
            else:
                f_wqa = getattr(attn, "fused_wqa_wkv", None)
                w = getattr(f_wqa, "weight", None)
                if w is None or w.dtype not in (
                    torch.bfloat16,
                    torch.float16,
                    torch.float32,
                ):
                    self._fused_ctx_wkv_weight = None
                    self._fused_ctx_wkv_scale = None
                    return
                if getattr(f_wqa, KEEP_VLLM_LAYOUT_ATTR, False):
                    if w.shape[1] != attn.q_lora_rank + attn.head_dim:
                        self._fused_ctx_wkv_weight = None
                        self._fused_ctx_wkv_scale = None
                        return
                    slices_w.append(w[:, attn.q_lora_rank :].transpose(0, 1))
                else:
                    if w.shape[0] != attn.q_lora_rank + attn.head_dim:
                        self._fused_ctx_wkv_weight = None
                        self._fused_ctx_wkv_scale = None
                        return
                    slices_w.append(w[attn.q_lora_rank :])
        if (
            use_quant
            and len(slices_w) == len(self.layers)
            and len(slices_s) == len(self.layers)
        ):
            self._fused_ctx_wkv_weight = torch.cat(slices_w, dim=1).contiguous()
            self._fused_ctx_wkv_scale = torch.cat(slices_s, dim=-1).contiguous()
            if self._fused_ctx_wkv_weight.device.type == "tpu":
                synchronize_tensors(
                    [self._fused_ctx_wkv_weight, self._fused_ctx_wkv_scale],
                    wait=True,
                )
        elif slices_w and not use_quant and len(slices_w) == len(self.layers):
            self._fused_ctx_wkv_weight = torch.cat(slices_w, dim=0).contiguous()
            self._fused_ctx_wkv_scale = None
            if self._fused_ctx_wkv_weight.device.type == "tpu":
                synchronize_tensors([self._fused_ctx_wkv_weight], wait=True)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def get_draft_attn_causal(self) -> list[bool]:
        return [False] * len(self.layers)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        proj = self.main_proj(aux_hidden_states)
        if isinstance(proj, tuple):
            proj = proj[0]
        return self.main_norm(proj)

    def precompute_and_store_context_kv(
        self,
        target_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: tuple,
    ) -> torch.Tensor:
        main_x = self.combine_hidden_states(target_hidden_states)
        if positions.ndim > 1:
            positions = positions.flatten()
        fused_wkv = self._fused_ctx_wkv_weight
        fused_scale = self._fused_ctx_wkv_scale
        if fused_wkv is not None:
            if fused_scale is not None:
                from vllm_torchtpu.layers.adapter.linear_common import (
                    quantized_matmul,
                )

                orig_m = main_x.shape[0]
                m_pad = (orig_m + 7) & ~7
                x_qmm = (
                    torch.nn.functional.pad(main_x, (0, 0, 0, m_pad - orig_m))
                    if m_pad > orig_m
                    else main_x
                )
                kv_all = quantized_matmul(x_qmm, fused_wkv, fused_scale)[:orig_m].view(
                    orig_m, len(self.layers), -1
                )
            else:
                kv_all = torch.nn.functional.linear(main_x, fused_wkv).view(
                    main_x.shape[0], len(self.layers), -1
                )
            for layer_idx, layer in enumerate(self.layers):
                attn = layer.attn
                kv = attn.kv_norm(kv_all[:, layer_idx])
                attn.store_context_kv(kv, positions, metadata[layer_idx])
            return main_x
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.attn
            kv = attn.project_wkv(main_x)
            kv = attn.kv_norm(kv)
            attn.store_context_kv(kv, positions, metadata[layer_idx])
        return main_x

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1 and envs.VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS > 0:
            _ = torch.ops.aten.alias.default(self.tp_rank_idx)
        positions = torch.ops.aten.alias.default(positions)
        num_tokens = positions.shape[0]
        token_sharded_mhc = tp_token_shard_ok(num_tokens, tp_size)

        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        if token_sharded_mhc:
            tp_rank = get_tensor_model_parallel_rank()
            inputs_embeds = tp_shard_slice(
                inputs_embeds, tp_size, tp_rank, self.tp_rank_idx
            )
        pre_mix = inputs_embeds[:, :1].to(torch.float32) * 0.0 + self.init_pre_mix
        hidden_states = inputs_embeds.repeat(1, self.hc_mult).view(
            -1, self.hc_mult, inputs_embeds.shape[-1]
        )

        previous_x: torch.Tensor | None = None
        previous_post_mix: torch.Tensor | None = None
        previous_res_mix: torch.Tensor | None = None

        for layer in self.layers:
            (
                previous_x,
                hidden_states,
                pre_mix,
                previous_post_mix,
                previous_res_mix,
                _,
            ) = layer(
                positions,
                hidden_states,
                pre_mix,
                previous_x=previous_x,
                previous_post_mix=previous_post_mix,
                previous_res_mix=previous_res_mix,
                engram_hashes=None,
                token_sharded=token_sharded_mhc,
                tp_rank_idx=self.tp_rank_idx,
            )

        out = get_mhc_final_collapse_norm_v41_op(self.rms_norm_eps)(
            previous_x,
            hidden_states,
            previous_post_mix,
            previous_res_mix,
            pre_mix,
            self.norm.weight,
        )
        if token_sharded_mhc:
            out = tp_all_gather(out, dim=0)
        return out


class DSparkDeepseekV41ForCausalLM(nn.Module):
    """vLLM model-registry entry for DeepSeek-V4.1 DSpark (`DSparkV41DraftModel`)."""

    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DSparkDeepseekV41Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head: nn.Module | None = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_attn_causal(self) -> list[bool]:
        return self.model.get_draft_attn_causal()

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.attn.swa_cache_layer.prefix for layer in self.model.layers]

    def tpu_precompute_and_store_context_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: tuple,
    ) -> torch.Tensor:
        return self.model.precompute_and_store_context_kv(
            hidden_states, positions, metadata
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def _remap_dspark_name(self, name: str) -> str | None:
        if name.endswith(".bias_vl"):
            return None
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m is None:
            return None
        stage = int(m.group(1))
        rest = m.group(2)
        if rest.startswith("confidence_head."):
            return None
        if rest.startswith("markov_head.embed."):
            return "model.markov_head.markov_w1." + rest.removeprefix(
                "markov_head.embed."
            )
        if rest.startswith("markov_head.head."):
            return "model.markov_head.markov_w2." + rest.removeprefix(
                "markov_head.head."
            )
        if rest.startswith(("main_proj.", "main_norm.", "norm.", "markov_head.")):
            return f"model.{rest}"
        return f"model.layers.{stage}.{rest}"

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.dspark_n_routed_experts,
        )
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        staged = _HostStagedParams()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        try:
            for orig_name, loaded_weight in weights:
                mapped = self._remap_dspark_name(orig_name)
                if mapped is None:
                    continue
                name = mapped

                if name.endswith(".scale"):
                    suffix = (
                        expert_scale_suffix
                        if _EXPERT_SCALE_RE.search(name)
                        else ".weight_scale_inv"
                    )
                    name = name.removesuffix(".scale") + suffix
                if ".shared_experts.w2" in name:
                    name = name.replace(
                        ".shared_experts.w2", ".shared_experts.down_proj"
                    )

                if ".experts." in name:
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for (
                        param_name,
                        weight_name,
                        expert_id,
                        expert_shard_id,
                    ) in expert_mapping:
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if name_mapped not in params_dict:
                            continue
                        param = staged.stage(name_mapped, params_dict[name_mapped])
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            loaded_params.add(name_mapped)
                            break
                    continue

                is_layer_param = name.startswith("model.layers.")
                for (
                    param_name,
                    weight_name,
                    stacked_shard_id,
                ) in stacked_params_mapping:
                    if not is_layer_param or weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        break
                    param = staged.stage(name, params_dict[name])
                    param.weight_loader(param, loaded_weight, stacked_shard_id)
                    loaded_params.add(name)
                    break
                else:
                    if "attn_sink" in name:
                        if name not in params_dict:
                            continue
                        narrow = loaded_weight[head_start:head_end]
                        param = staged.stage(name, params_dict[name])
                        param.data[: narrow.shape[0]].copy_(narrow)
                        loaded_params.add(name)
                        continue
                    if name.endswith(".ffn.gate.bias"):
                        name = name.replace(
                            ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                        )
                    if name not in params_dict:
                        continue
                    param = staged.stage(name, params_dict[name])
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
        finally:
            staged.flush_all()

        for layer in self.model.layers:
            layer._ensure_mhc_buffers(force=True)

        logger.info(
            "DSparkDeepseekV41ForCausalLM loaded %d parameters",
            len(loaded_params),
        )
        return loaded_params

    def sample_draft_tokens(
        self,
        pred_hidden: torch.Tensor,
        prev_init: torch.Tensor,
        num_pred: int,
    ) -> torch.Tensor:
        padded_num_reqs = prev_init.shape[0]
        lm_head = self.lm_head
        markov_w2 = getattr(self.model.markov_head, "markov_w2", None)
        tp_size = int(getattr(lm_head, "tp_size", 1))
        shard_indices = getattr(lm_head, "shard_indices", None)
        if (
            lm_head is None
            or markov_w2 is None
            or shard_indices is None
            or tp_size <= 1
        ):
            bias = self.markov_bias(self.markov_embed(prev_init))
            logits = self.compute_draft_logits(pred_hidden)
            logits_3d = logits.view(padded_num_reqs, num_pred, logits.shape[-1])
            draft_tokens = []
            prev = prev_init
            for idx in range(num_pred):
                if idx > 0:
                    bias = self.markov_bias(self.markov_embed(prev))
                step = self.map_draft_to_target(
                    (logits_3d[:, idx] + bias).argmax(dim=-1)
                )
                draft_tokens.append(step)
                prev = step
            return torch.stack(draft_tokens, dim=1)

        shard_cols = int(lm_head.weight.shape[0])
        meta = getattr(self, "_tp_markov_meta", None)
        if meta is None or meta[0].device != pred_hidden.device:
            start_idx = int(shard_indices.org_vocab_start_index)
            end_idx = int(shard_indices.org_vocab_end_index)
            valid_vocab = end_idx - start_idx
            w2_raw = markov_w2.weight[start_idx:end_idx]
            w2_pad = (
                torch.nn.functional.pad(w2_raw, (0, 0, 0, shard_cols - valid_vocab))
                if valid_vocab < shard_cols
                else w2_raw.contiguous()
            )
            pad_bias = torch.zeros(
                (1, shard_cols), dtype=w2_raw.dtype, device=pred_hidden.device
            )
            if valid_vocab < shard_cols:
                pad_bias[:, valid_vocab:] = float("-inf")
            start_t = torch.tensor(
                [start_idx], dtype=torch.int32, device=pred_hidden.device
            )
            meta = (w2_pad, pad_bias, start_t)
            object.__setattr__(self, "_tp_markov_meta", meta)
        w2_pad, pad_bias, start_t = meta

        prev = prev_init
        bias_local = (
            torch.nn.functional.linear(self.markov_embed(prev), w2_pad) + pad_bias
        )
        logits_local = self.logits_processor._get_logits(
            pred_hidden, lm_head, None, skip_gather=True
        )
        assert logits_local is not None
        logits_local_3d = logits_local.view(
            padded_num_reqs, num_pred, shard_cols
        )

        draft_tokens = []
        for idx in range(num_pred):
            if idx > 0:
                bias_local = (
                    torch.nn.functional.linear(self.markov_embed(prev), w2_pad)
                    + pad_bias
                )
            local_val, local_idx = (
                logits_local_3d[:, idx] + bias_local
            ).max(dim=-1, keepdim=True)
            global_idx = local_idx.to(torch.int32) + start_t
            all_vals = tp_all_gather(local_val, dim=-1)
            all_idxs = tp_all_gather(global_idx, dim=-1)
            best_shard = all_vals.argmax(dim=-1, keepdim=True)
            step = all_idxs.gather(-1, best_shard).squeeze(-1)
            draft_tokens.append(step)
            prev = step
        return torch.stack(draft_tokens, dim=1)

    def process_weights_after_loading(self) -> None:
        for layer in self.model.layers:
            layer._ensure_mhc_buffers()
        self.model.process_weights_after_loading()
