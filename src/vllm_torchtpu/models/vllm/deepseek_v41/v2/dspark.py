# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for DeepSeek-V4.1 on TPU."""

import re
from collections.abc import Callable, Iterable
from typing import cast

import torch
import torch.nn as nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v41.nvidia import model as upstream

from vllm_torchtpu.layers.adapter.linear_common import (
    WEIGHT_FLIPPED_ATTR,
    quantized_matmul,
)
from vllm_torchtpu.layers.adapter.quantization.deepseek_v41 import DeepseekV41MoE
from vllm_torchtpu.logger import init_logger

from ..common.mhc import (
    initialize_delayed_mhc,
    initialize_mhc_post,
    pack_mix_weight,
    pack_mix_weights,
)
from . import attention as v2_attn
from .attention import (
    DeepseekV41Attention,
    _block_table,
    _cache_slots,
    _insert,
    _token_requests,
    stage_attention_cache_scope,
)
from .attention_kernels import initialize_indexer, initialize_paged_attention
from .cache import initialize_cache_ops
from .engram import Engram, NgramHashState
from .model import DeepseekV41DecoderLayer
from .ops import compact_rope_caches, device_hooks

logger = init_logger(__name__)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def get_dspark_swa_index_width(window_size: int, dspark_block_size: int) -> int:
    """Padded 64-aligned SWA key width covering window_size + dspark_block_size."""
    return -(-(window_size + dspark_block_size) // 64) * 64


def _remap_dspark_name(name: str, *, enable_confidence_head: bool = True) -> str | None:
    """Map a checkpoint ``mtp.{i}.*`` name to this model's parameter path.

    Returns None for non-mtp weights (owned by the target model).
    """
    m = re.match(r"mtp\.(\d+)\.(.*)", name)
    if m is None:
        return None
    stage = int(m.group(1))
    rest = m.group(2)
    if rest.startswith("confidence_head.") and not enable_confidence_head:
        return None
    if rest.startswith("markov_head.embed."):
        return "model.markov_head.markov_w1." + rest.removeprefix("markov_head.embed.")
    if rest.startswith("markov_head.head."):
        return "model.markov_head.markov_w2." + rest.removeprefix("markov_head.head.")
    head_prefixes = (
        "norm.",
        "markov_head.",
        "confidence_head.",
    )
    if rest.startswith(("main_proj.", "main_norm.")) or rest.startswith(head_prefixes):
        return f"model.{rest}"
    if rest.endswith("ffn.gate.bias_vl"):
        rest = rest.removesuffix("ffn.gate.bias_vl") + "ffn.gate.bias"
    return f"model.layers.{stage}.{rest}"


def _embed_draft_blocks(
    embed: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    block: int,
) -> torch.Tensor:
    """Embed a draft batch laid out as ``[base, noise x (block - 1)]`` per request.

    The draft forward's rows are ``[r * block, (r + 1) * block)`` for request
    ``r`` with one shared noise (MASK) id after each base id, and the tail past
    the last full block is noise too (``DFlashProposer._layout_draft_block``).
    Only the per-request base ids and the noise id go through ``embed`` (a
    vocab-parallel lookup plus a TP all-reduce), and the rows are broadcast
    back, so the gather and the all-reduce shrink by ~``block``x. Each row is
    the same single-rank sum as the full lookup, so the result is bitwise
    equal. Rows whose id breaks the layout are NaN instead of silently wrong.

    Args:
        embed: Token-id -> ``[n, hidden]`` embedding lookup.
        input_ids: ``[tokens]`` draft input ids.
        block: Query rows per request.

    Returns:
        ``[tokens, hidden]`` embeddings.
    """
    tokens = input_ids.shape[0]
    num_blocks = tokens // block
    if block <= 1 or num_blocks == 0:
        return embed(input_ids)
    tail = tokens - num_blocks * block
    ids_2d = input_ids[: num_blocks * block].view(num_blocks, block)
    base = ids_2d[:, 0]
    noise = input_ids[1:2]
    rows = embed(torch.cat([base, noise]))
    hidden = rows.shape[-1]
    noise_row = rows[num_blocks:]
    # Pure 2-D [num_blocks * block, hidden] selection: a [num_blocks, block,
    # hidden] build pads the middle `block` dim to 8 sublanes on TPU.
    total, owner = num_blocks * block, getattr(embed, "__self__", embed)
    key, cache = (
        (total, block, input_ids.device),
        getattr(owner, "_draft_block_idx_cache", None),
    )
    _, is_base_col, block_idx = (
        cache
        if cache is not None and cache[0] == key
        else setattr(
            owner,
            "_draft_block_idx_cache",
            (
                cache := (
                    key,
                    (
                        (
                            idx := torch.arange(
                                total, dtype=torch.int32, device=input_ids.device
                            )
                        )
                        % block
                        == 0
                    )[:, None],
                    idx // block,
                )
            ),
        )
        or cache
    )
    base_rows = rows[:num_blocks].index_select(0, block_idx)
    blocks = torch.where(
        is_base_col, base_rows, noise_row.expand(total, hidden)
    ).masked_fill(~is_base_col & (ids_2d.reshape(-1) != noise)[:, None], float("nan"))
    if tail:
        tail_rows = noise_row.expand(tail, hidden).masked_fill(
            (input_ids[num_blocks * block :] != noise).unsqueeze(-1), float("nan")
        )
        blocks = torch.cat([blocks, tail_rows])
    return blocks


def _insert_context_kv(
    attn: DeepseekV41Attention,
    kv: torch.Tensor,
    positions: torch.Tensor,
    metadata,
) -> None:
    """Rotate and write normalized context KV rows into one draft layer's SWA cache."""
    swa_cache = attn.swa_cache_layer.kv_cache
    if metadata is None or not swa_cache.numel():
        return
    stage_cache = v2_attn._active_stage_attn_cache
    if stage_cache is None or not stage_cache.get("context_kv_pre_rotated", False):
        kv = attn._rotate(kv, positions)
    write = None
    if isinstance(metadata, torch.Tensor):
        swa_slots = metadata
        swa_valid = swa_slots >= 0
    else:
        rows_per_page = swa_cache.shape[1]
        fast_offset = v2_attn._has_fast_cache_offset(swa_cache)
        slot_key = (
            "context_kv_slots",
            metadata.seq_lens.shape,
            rows_per_page,
            attn.window_size,
            v2_attn._fast_cache_offset_key(swa_cache) if fast_offset else None,
        )
        block_tables = getattr(metadata, "block_tables", None)
        cached_slots = stage_cache.get(slot_key) if stage_cache is not None else None
        if cached_slots is not None and cached_slots[0] is block_tables:
            _, swa_slots, swa_valid, _, _, write = cached_slots
        else:
            table = None
            if cached_slots is not None:
                _, _, _, in_window, addr, _ = cached_slots
            elif metadata.seq_lens.numel() > 0:
                pos_i32 = (
                    positions if positions.dtype == torch.int32 else positions.int()
                )
                requests = _token_requests(metadata, pos_i32).clamp(
                    max=metadata.seq_lens.shape[0] - 1
                )
                offsets = (
                    pos_i32
                    - (
                        metadata.seq_lens
                        if metadata.seq_lens.dtype == torch.int32
                        else metadata.seq_lens.int()
                    ).index_select(
                        0,
                        req_i32 := (
                            requests
                            if requests.dtype == torch.int32
                            else requests.int()
                        ),
                    )
                    + attn.window_size
                )
                in_window = (offsets >= 0) & (offsets < attn.window_size)
                if block_tables is not None:
                    table = _block_table(metadata)
                    width = table.shape[1]
                    pos_col = (pos_i32 // rows_per_page).clamp(max=width - 1)
                    # Flat int32 row-major index into the block table: one
                    # 1-D index_select per layer instead of a 2-D int64
                    # (requests, pos_col) gather.
                    block_idx = req_i32 * width + pos_col
                    pos_rem = pos_i32 % rows_per_page
                    rows = torch.arange(
                        positions.shape[0],
                        dtype=torch.int32,
                        device=positions.device,
                    )
                    valid_token = (
                        rows
                        < (
                            metadata.query_start_loc
                            if metadata.query_start_loc.dtype == torch.int32
                            else metadata.query_start_loc.int()
                        )[-1]
                    )
                    addr = (
                        block_idx,
                        width,
                        pos_rem,
                        valid_token,
                        valid_token & in_window,
                    )
                else:
                    addr = None
            else:
                in_window = None
                addr = None
            if addr is not None:
                if table is None:
                    table = _block_table(metadata)
                block_idx, width, pos_rem, valid_token, valid_in_window = addr
                # A reused address must index a table of the same width.
                assert table.shape[1] == width, (table.shape, width)
                # int32 block / slot math (block tables and positions are
                # int32; slots fit int32 like every PagedCacheRows record):
                # TPU emulates int64 multiplies and adds.
                blocks = (
                    (table if table.dtype == torch.int32 else table.int())
                    .reshape(-1)
                    .index_select(0, block_idx)
                )
                swa_slots = (blocks * rows_per_page + pos_rem).masked_fill(
                    ~valid_token, -1
                )
                swa_valid = (blocks >= 0) & valid_in_window
                # Offset-free write rows shared by every draft layer; each
                # layer adds its row_offset instead of re-deriving the block
                # and remainder from swa_slots in PagedCacheRows.write.
                if fast_offset:
                    write = v2_attn._fast_swa_row_slots(
                        swa_cache, blocks, pos_rem, ~swa_valid | (swa_slots < 0)
                    )
            else:
                swa_slots = _cache_slots(swa_cache, metadata, positions)
                swa_valid = (
                    (swa_slots >= 0) & in_window
                    if in_window is not None
                    else swa_slots >= 0
                )
            if stage_cache is not None:
                stage_cache[slot_key] = (
                    block_tables,
                    swa_slots,
                    swa_valid,
                    in_window,
                    addr,
                    write,
                )
    if write is not None:
        _insert(
            swa_cache,
            swa_slots,
            kv,
            swa_valid,
            records=v2_attn._offset_records(swa_cache, *write),
        )
    else:
        _insert(swa_cache, swa_slots, kv, swa_valid)


class DSparkDeepseekV41Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        initialize_cache_ops()
        initialize_delayed_mhc()
        initialize_mhc_post()
        initialize_paged_attention()
        initialize_indexer()

        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.num_dspark_layers = (
            getattr(config, "n_mtp_layers", None)
            or getattr(config, "num_nextn_predict_layers", None)
            or 3
        )

        # Aliased to the target's embedding by the proposer's sharing utility
        # after load (has_own_embed_tokens=False).
        self.embed_tokens: nn.Module | None = None
        self._fused_context_wkv_weight: torch.Tensor | None = None
        self._fused_context_wkv_scale: torch.Tensor | None = None
        self._fused_context_kv_norm_weight: torch.Tensor | None = None

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

        current_vllm_config = get_current_vllm_config()
        with (
            device_hooks(
                upstream,
                _select_dsv4_attn_cls=lambda _: DeepseekV41Attention,
                Engram=Engram,
                NgramHashState=NgramHashState,
                DeepseekV4DecoderLayer=DeepseekV41DecoderLayer,
                DeepseekV4MoE=DeepseekV41MoE,
                _use_sequence_parallel=lambda _: False,
            ),
            device_hooks(torch.cuda, Stream=lambda: None),
        ):
            self.layers = nn.ModuleList(
                [
                    DeepseekV41DecoderLayer(
                        current_vllm_config,
                        prefix=maybe_prefix(
                            prefix, f"layers.{self.num_hidden_layers + i}"
                        ),
                        topk_indices_buffer=self.topk_indices_buffer,
                    )
                    for i in range(self.num_dspark_layers)
                ]
            )

        dspark_block_size = (
            getattr(config, "dspark_block_size", None)
            or getattr(config, "block_size", None)
            or vllm_config.speculative_config.num_speculative_tokens
        )
        num_spec = vllm_config.speculative_config.num_speculative_tokens
        max_block_rows = num_spec + 1
        # Query tokens per request in a draft forward, laid out as rows
        # [r * block, (r + 1) * block) (DSparkProposer._query_block_size).
        query_block_size = (
            num_spec if getattr(config, "sample_from_anchor", True) else num_spec + 1
        )
        self.query_block_size = int(query_block_size)
        for layer in cast(Iterable[DeepseekV41DecoderLayer], self.layers):
            if max_block_rows > layer.attn.window_size:
                raise ValueError(
                    f"DSpark draft block of up to {max_block_rows} rows "
                    f"exceeds the DSV4.1 sliding window ({layer.attn.window_size})."
                )
            layer.attn.non_causal_multi_token_decode = True
            layer.attn.dspark_block_size = int(dspark_block_size)
            layer.attn.query_block_size = int(query_block_size)

        compact_rope_caches(
            (layer.attn for layer in self.layers),
            vllm_config.model_config.max_model_len,
        )

        # V4.1 collapses hyper-connections with the last layer's FFN pre-mix
        # instead of a learned hc_head.
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
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", True):
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + config.dspark_markov_rank,
                prefix=maybe_prefix(prefix, "confidence_head"),
            )

    def _finalize_context_kv_weights(
        self, layers: list[DeepseekV41DecoderLayer]
    ) -> None:
        if self._fused_context_wkv_weight is not None:
            return
        attn0 = layers[0].attn
        w0_mod = getattr(attn0, "fused_wqa_wkv", None)
        if not getattr(w0_mod, WEIGHT_FLIPPED_ATTR, False):
            return
        q_rank = attn0.q_lora_rank
        head_dim = attn0.head_dim
        wkv_weights = [
            layer.attn.fused_wqa_wkv.weight[:, q_rank : q_rank + head_dim]
            for layer in layers
        ]
        self._fused_context_wkv_weight = torch.cat(wkv_weights, dim=-1).contiguous()
        if getattr(w0_mod, "weight_scale", None) is not None:
            wkv_scales = [
                layer.attn.fused_wqa_wkv.weight_scale[..., q_rank : q_rank + head_dim]
                for layer in layers
            ]
            self._fused_context_wkv_scale = torch.cat(wkv_scales, dim=-1).contiguous()
        self._fused_context_kv_norm_weight = torch.stack(
            [layer.attn.kv_norm.weight.detach().float() for layer in layers], dim=0
        ).contiguous()

    def finalize_mhc_broadcast_weights(self) -> None:
        layers = list(self.layers)
        if not layers:
            return
        first_layer = layers[0]
        if first_layer.hc_attn_fn.dtype == torch.float32:
            if first_layer.hc_attn_fn.shape[-1] == self.hc_mult * self.hidden_size:
                broadcast_fn = (
                    first_layer.hc_attn_fn.detach()
                    .view(-1, self.hc_mult, self.hidden_size)
                    .sum(dim=1)
                )
                if first_layer.hc_attn_fn.device.type == "tpu":
                    broadcast_fn = pack_mix_weight(broadcast_fn)
                first_layer.hc_attn_fn_broadcast = broadcast_fn
            pack_mix_weights(layers)
        self._finalize_context_kv_weights(layers)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """main_x = main_norm(main_proj(concat of target aux hidden states))."""
        return self.main_norm(self.main_proj(aux_hidden_states))

    def get_draft_attn_causal(self) -> list[bool]:
        return [False] * len(self.layers)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [
            layer.attn.swa_cache_layer.prefix
            for layer in cast(Iterable[DeepseekV41DecoderLayer], self.layers)
        ]

    def precompute_and_store_context_kv(
        self,
        target_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: tuple | list | None = None,
    ) -> torch.Tensor:
        """Insert the sliding-window context KV for every draft layer."""
        if target_hidden_states.shape[-1] != self.config.hidden_size:
            main_x = self.combine_hidden_states(target_hidden_states)
        else:
            main_x = target_hidden_states
        layers = list(cast(Iterable[DeepseekV41DecoderLayer], self.layers))
        with stage_attention_cache_scope():
            if self._fused_context_wkv_weight is not None:
                head_dim = layers[0].attn.head_dim
                if self._fused_context_wkv_scale is not None:
                    all_kv_raw = quantized_matmul(
                        main_x,
                        self._fused_context_wkv_weight,
                        self._fused_context_wkv_scale,
                    )
                else:
                    all_kv_raw = torch.matmul(main_x, self._fused_context_wkv_weight)
                all_kv_3d = all_kv_raw.view(-1, len(layers), head_dim)
                x_f32 = all_kv_3d.to(torch.float32)
                w_norm = self._fused_context_kv_norm_weight
                assert w_norm is not None
                kv_normed = (
                    x_f32
                    * torch.rsqrt(
                        x_f32.pow(2).mean(dim=-1, keepdim=True)
                        + layers[0].attn.kv_norm.variance_epsilon
                    )
                    * w_norm
                ).to(all_kv_3d.dtype)
                kv_rotated = layers[0].attn._rotate(kv_normed, positions)
                assert v2_attn._active_stage_attn_cache is not None
                v2_attn._active_stage_attn_cache["context_kv_pre_rotated"] = True
                # Per-layer 128-lane-aligned column slices of the flat
                # [T, layers * head_dim] rows: transposing the 3-D
                # [T, layers, head_dim] view (layers padded to 8 sublanes on
                # TPU) costs XLA a relayout copy of the whole tensor.
                kv_flat = kv_rotated.reshape(kv_rotated.shape[0], -1)
                for i, layer in enumerate(layers):
                    layer_md = None if metadata is None else metadata[i]
                    kv_i = kv_flat[:, i * head_dim : (i + 1) * head_dim].contiguous()
                    _insert_context_kv(layer.attn, kv_i, positions, layer_md)
            else:
                for i, layer in enumerate(layers):
                    attn = layer.attn
                    qr_kv = attn._fused_wqa_wkv_gemm(main_x)
                    _, _, kv = attn._split_qkv_and_norm(qr_kv)
                    layer_md = None if metadata is None else metadata[i]
                    _insert_context_kv(attn, kv, positions, layer_md)
        return main_x

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = _embed_draft_blocks(
                self.embed_input_ids, input_ids, self.query_block_size
            )
        if getattr(self.layers[0], "hc_attn_fn_broadcast", None) is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        residual = post_mix = res_mix = pre_mix = None
        with stage_attention_cache_scope():
            for layer in self.layers:
                hidden_states, residual, post_mix, res_mix, pre_mix, _ = layer(
                    hidden_states,
                    positions,
                    input_ids,
                    pre_mix,
                    post_mix,
                    res_mix,
                    residual,
                )
        assert (
            residual is not None
            and post_mix is not None
            and res_mix is not None
            and pre_mix is not None
        )
        return _post_and_collapse(hidden_states, residual, post_mix, res_mix, pre_mix)


def _post_and_collapse(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    pre_mix: torch.Tensor,
) -> torch.Tensor:
    """Fuse the final draft layer's mHC post and stream collapse.

    Slicing the flat ``[T, M * D]`` residual avoids the Pallas ``mhc_post``
    64-token pad/trim + ``[T, M * D]`` HBM round-trip and XLA's ``[T, M, D]``
    ``T(4, 128)`` relayout copy in ``collapse``, while matching ``_post_kernel``
    and ``_collapse_flat`` accumulation order.
    """
    tokens, mult, dim = residual.shape
    flat = residual.reshape(tokens, mult * dim)
    x_f32 = x.reshape(tokens, dim).to(residual.dtype).float()
    post_f32 = post_layer_mix.reshape(tokens, mult).float()
    comb_f32 = comb_res_mix.reshape(tokens, mult * mult).float()
    pre_f32 = pre_mix.reshape(tokens, mult).float()
    streams = [flat[:, i * dim : (i + 1) * dim].float() for i in range(mult)]
    terms = []
    for j in range(mult):
        acc = post_f32[:, j : j + 1] * x_f32
        for i in range(mult):
            acc = acc + comb_f32[:, (i * mult + j) : (i * mult + j + 1)] * streams[i]
        terms.append(pre_f32[:, j : j + 1] * acc.to(residual.dtype).float())
    value = (
        (terms[0] + terms[2]) + (terms[1] + terms[3])
        if mult == 4
        else torch.stack(terms, dim=0).sum(dim=0)
    )
    return value.to(residual.dtype)


class DSparkDeepseekV41ForCausalLM(nn.Module):
    """Registry entry for ``architectures=["DSparkV41DraftModel"]`` on TPU."""

    has_own_embed_tokens = False
    has_own_lm_head = False
    owns_context_kv = True
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        # TPU FP8 linear adapters register weight_scale_inv.
        self.linear_scale_name = "weight_scale_inv"
        self.pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
        )
        self.model = DSparkDeepseekV41Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Aliased to the target's lm_head by the proposer's sharing utility.
        self.lm_head: nn.Module | None = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    # --- Hooks consumed by DFlashProposer / DSparkProposer -----------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_attn_causal(self) -> list[bool]:
        return self.model.get_draft_attn_causal()

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def precompute_and_store_context_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: tuple | list | None = None,
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
        """Base logits U_k = lm_head(norm(head_hidden))."""
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)

    def compute_draft_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's vocab shard of U_k, without the TP all-gather.

        Columns index ``lm_head``'s padded local vocab partition; the
        proposer's TP-sharded Markov sampler scores each shard locally and
        reduces only the per-request (max, token id) pair across ranks.
        """
        assert self.lm_head is not None
        return self.logits_processor(
            self.lm_head, self.model.norm(hidden_states), skip_gather=True
        )

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    # --- Weight loading -----------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``mtp.{0,1,2}.*`` draft weights from the target checkpoint."""
        first_layer = self.model.layers[0]
        use_native_mega_moe = getattr(
            first_layer.ffn,
            "use_native_mega_moe",
            getattr(first_layer.ffn, "use_mega_moe", False),
        )
        n_draft_experts = (
            getattr(self.config, "dspark_n_routed_experts", 0)
            or self.config.n_routed_experts
        )
        if use_native_mega_moe:
            expert_mapping = upstream.make_deepseek_v4_expert_params_mapping(
                n_draft_experts
            )
        else:
            expert_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=n_draft_experts,
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
        loaded_confidence_head = False

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        for name, loaded_weight in weights:
            mapped = self._remap_dspark_name(name)
            if mapped is None:
                continue
            name = mapped
            if "confidence_head." in name:
                loaded_confidence_head = True

            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else f".{self.linear_scale_name}"
                )
                name = name.removesuffix(".scale") + suffix
            if ".shared_experts.w2" in name:
                name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
            if self.pad_shared_expert and ".shared_experts." in name:
                loaded_weight = upstream.DeepseekV4Model._pad_shared_expert_weight(
                    self.quant_config, name, loaded_weight
                )

            if ".experts." in name:
                if (
                    "weight_scale" in name
                    and loaded_weight.dtype == torch.float8_e8m0fnu
                ):
                    loaded_weight = loaded_weight.view(torch.uint8)
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    success = param.weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                continue

            is_layer_param = name.startswith("model.layers.")
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if not is_layer_param or weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(name)
                break
            else:
                if "attn_sink" in name:
                    narrow = loaded_weight[head_start:head_end]
                    params_dict[name].data[: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if self.model.confidence_head is not None and not loaded_confidence_head:
            self.model.confidence_head = None
        self.process_weights_after_loading()
        logger.info_once(
            "DSpark V4.1 draft model loaded: %d params", len(loaded_params)
        )
        return loaded_params

    def process_weights_after_loading(self) -> None:
        for layer in self.model.layers:
            layer.ffn.finalize_mega_moe_weights()
        self.model.finalize_mhc_broadcast_weights()

    def _remap_dspark_name(self, name: str) -> str | None:
        return _remap_dspark_name(
            name, enable_confidence_head=self.model.confidence_head is not None
        )
