# SPDX-License-Identifier: Apache-2.0
"""DSpark speculative decoding for TPU.

DSpark is semi-autoregressive parallel drafting: one DFlash-style parallel
block forward (anchor + noise tokens, non-causal within the block, context
KV precomputed from the target's aux hidden states), followed by a
lightweight sequential Markov head that biases each position's draft logits
on the previously sampled token. Upstream implements DSparkSpeculator as a
small DFlashSpeculator subclass; this proposer mirrors that on top of the
TPU DFlashProposer.

Dense-checkpoint layout (``sample_from_anchor`` absent or true, as in
upstream DSparkSpeculator): the query block is K tokens — the anchor is the
FIRST prediction position and every slot predicts the NEXT token — versus
DFlash's 1 + K (anchor is bonus only). Speculators-format checkpoints set
``sample_from_anchor=False`` and keep the DFlash layout.

PoC scope: greedy draft sampling only (argmax + d2t remap); probabilistic
Gumbel-coupled sampling is not implemented.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.utils import resolve_draft_logits_processor

logger = init_logger(__name__)

# Token ids ride through the per-step all-gather as exact float32 integers.
_MAX_EXACT_FLOAT32_INT = 1 << 24


class DSparkProposer(DFlashProposer):
    def __init__(self, runner, vllm_config: VllmConfig):
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        hf_config = speculative_config.draft_model_config.hf_config
        hf_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else {}

        super().__init__(runner, vllm_config)

        if not self.dflash_config.get("target_layer_ids"):
            raise ValueError(
                "DSpark requires target layer ids in the draft config (one of "
                "eagle_aux_hidden_state_layer_ids, dspark_target_layer_ids, "
                "target_layer_ids, or dflash_config.target_layer_ids)."
            )

        # Same key/default as upstream DSparkSpeculator: dense checkpoints
        # omit it (-> True); the speculators converter writes it explicitly.
        self.sample_from_anchor = bool(hf_dict.get("sample_from_anchor", True))
        block = hf_dict.get("dspark_block_size") or hf_dict.get("block_size")
        K = self.speculative_config.num_speculative_tokens
        if block is not None and int(block) != K:
            raise ValueError(
                f"num_speculative_tokens={K} does not match the DSpark "
                f"checkpoint's block size {block}; the block/Markov-head "
                "machinery is trained for exactly that block length."
            )
        # vLLM 0.29 drafting options this proposer does not implement yet.
        spec = self.speculative_config
        if (
            spec.draft_sample_method != "greedy"
            or spec.enable_adaptive_verification
            or spec.dspark_draft_topk is not None
        ):
            raise ValueError(
                "DSpark on TPU does not support draft_sample_method="
                f"{spec.draft_sample_method!r}, enable_adaptive_verification="
                f"{spec.enable_adaptive_verification}, dspark_draft_topk="
                f"{spec.dspark_draft_topk}; only greedy drafting of a fixed "
                "block is implemented."
            )
        logger.info("DSpark: sample_from_anchor=%s, K=%d.", self.sample_from_anchor, K)

        # Full-vocab Markov sampling until load_model() has seen the lm_head
        # layout (see _init_sharded_markov_sampling).
        self._sharded_markov_tp = 1
        self._markov_w1_local: torch.Tensor | None = None
        self._markov_w2_local: torch.Tensor | None = None
        self._vocab_shard_offset: torch.Tensor | None = None
        self._vocab_shard_mask: torch.Tensor | None = None

    # A reduced-vocab DSpark checkpoint carries its own lm_head
    # ([draft_vocab, hidden] plus a d2t remap) and possibly its own
    # embed_tokens; force-sharing the target's would clobber them. Honor the
    # has_own_embed_tokens / has_own_lm_head flags the loader sets (upstream
    # load_dspark_model does the same via _should_share).
    _force_share_target_embeddings = False

    def _query_block_size(self, K: int) -> int:
        # Dense layout: K query tokens (anchor + K-1 noise), every slot
        # predicts the next token. Speculators-format checkpoints keep the
        # DFlash 1 + K layout.
        return K if self.sample_from_anchor else K + 1

    def load_model(self, target_model) -> None:
        super().load_model(target_model)
        for attr in (
            "compute_draft_logits",
            "markov_embed",
            "markov_bias",
            "map_draft_to_target",
            "has_own_lm_head",
            "has_own_embed_tokens",
        ):
            if not hasattr(self.draft_model, attr):
                raise RuntimeError(
                    f"DSpark draft model lacks {attr}(); expected a "
                    "Qwen3DSpark/Gemma4DSpark-family model."
                )
        self._init_sharded_markov_sampling()

    # --- TP-sharded greedy Markov sampling ----------------------------------
    #
    # With a TP-sharded lm_head the stock path all-gathers the full
    # [reqs * K, vocab] draft logits (DeviceCommunicatorBase.all_gather then
    # also transposes the [tp, N, V/tp] gather into [N, V]) and every rank
    # redundantly runs the K-step Markov bias + argmax over the whole vocab.
    # Here each rank keeps only its own vocab shard of the base logits, adds
    # the matching rows of the replicated Markov w2, takes a local max, and
    # the ranks exchange a [reqs, 2] (max value, global token id) pair per
    # step. Rank-specific state (vocab offset, w2 shard, padding mask) lives
    # in device tensors so the compiled graph is identical across TP ranks.

    def _init_sharded_markov_sampling(self) -> None:
        """Pre-slice the Markov head for this rank's lm_head vocab shard."""
        self._sharded_markov_tp = 1
        self._markov_w1_local = None
        self._markov_w2_local = None
        self._vocab_shard_offset = None
        self._vocab_shard_mask = None

        draft = self.draft_model
        assert draft is not None
        reason = self._sharded_markov_unsupported_reason(draft)
        if reason is not None:
            logger.info("DSpark: full-vocab Markov sampling (%s).", reason)
            return

        lm_head = draft.lm_head
        shard_indices = lm_head.shard_indices
        markov_head = draft.model.markov_head
        markov_w2 = markov_head.markov_w2
        logits_processor = resolve_draft_logits_processor(draft)
        head_dtype = getattr(logits_processor, "head_dtype", None)

        width = lm_head.num_embeddings_per_partition
        start = shard_indices.padded_org_vocab_start_index
        rows = markov_w2.weight.detach()[start : start + width]
        if rows.shape[0] < width:
            rows = F.pad(rows, (0, 0, 0, width - rows.shape[0]))
        if head_dtype is not None:
            rows = rows.to(head_dtype)
        self._markov_w2_local = rows.contiguous().clone()
        markov_w1 = getattr(markov_head, "markov_w1", None)
        w1_weight = getattr(markov_w1, "weight", None)
        if w1_weight is not None:
            self._markov_w1_local = (
                w1_weight.detach()
                .to(dtype=rows.dtype, device=rows.device)
                .contiguous()
                .clone()
            )
        self._vocab_shard_offset = torch.tensor(
            start, dtype=torch.float32, device=rows.device
        )
        # Whether any shard carries vocab padding is a global property, so
        # the mask add is present in every rank's graph or in none.
        if lm_head.num_embeddings_padded != lm_head.org_vocab_size:
            mask = torch.zeros(width, dtype=rows.dtype, device=rows.device)
            mask[shard_indices.num_org_elements :] = float("-inf")
            self._vocab_shard_mask = mask
        # The lm_head's TP size is the TP group's world size by construction.
        self._sharded_markov_tp = int(lm_head.tp_size)
        logger.info(
            "DSpark: TP-sharded Markov sampling over %d x %d vocab columns.",
            self._sharded_markov_tp,
            width,
        )

    def _sharded_markov_unsupported_reason(self, draft) -> str | None:
        if self._draft_replicated:
            return "draft lm_head is replicated (draft_tp=1)"
        if not hasattr(draft, "compute_draft_logits_local"):
            return f"{type(draft).__name__} lacks compute_draft_logits_local()"
        lm_head = getattr(draft, "lm_head", None)
        shard_indices = getattr(lm_head, "shard_indices", None)
        tp_size = getattr(lm_head, "tp_size", 1)
        if shard_indices is None or not isinstance(tp_size, int) or tp_size <= 1:
            return "draft lm_head is not a TP-sharded ParallelLMHead"
        if shard_indices.num_added_elements_padded != 0:
            return "lm_head carries added vocab"
        if getattr(lm_head, "num_embeddings_padded", 0) >= _MAX_EXACT_FLOAT32_INT:
            return "vocab exceeds the exact float32 integer range"
        markov_head = getattr(getattr(draft, "model", None), "markov_head", None)
        markov_w2 = getattr(markov_head, "markov_w2", None)
        weight = getattr(markov_w2, "weight", None)
        if (
            weight is None
            or weight.dim() != 2
            or not weight.is_floating_point()
            or getattr(markov_w2, "tp_size", 1) != 1
            or not isinstance(
                getattr(markov_w2, "quant_method", None),
                (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
            )
        ):
            return "markov_w2 is not a replicated unquantized head"
        if resolve_draft_logits_processor(draft) is None:
            return "draft exposes no LogitsProcessor"
        return None

    def _markov_bias_local(self, prev: torch.Tensor) -> torch.Tensor:
        """Markov bias restricted to this rank's vocab shard ([B] -> [B, W])."""
        w2_local = self._markov_w2_local
        assert w2_local is not None
        w1_local = self._markov_w1_local
        if w1_local is not None:
            return F.linear(F.embedding(prev, w1_local), w2_local)
        assert self.draft_model is not None
        embed = self.draft_model.markov_embed(prev)
        if embed.dtype != w2_local.dtype:
            embed = embed.to(w2_local.dtype)
        return F.linear(embed, w2_local)

    @staticmethod
    def _pack_local_candidate(
        scores: torch.Tensor, vocab_offset: torch.Tensor
    ) -> torch.Tensor:
        """Per-row (max value, global token id) of a vocab shard as [B, 2] f32."""
        local_max, local_idx = scores.max(dim=-1)
        return torch.stack(
            [
                local_max.to(torch.float32),
                local_idx.to(torch.float32) + vocab_offset.to(torch.float32),
            ],
            dim=-1,
        )

    @staticmethod
    def _select_gathered_candidate(
        gathered: torch.Tensor, tp_size: int, num_reqs: int
    ) -> torch.Tensor:
        """Global argmax from the [tp * B, 2] all-gathered shard candidates.

        Shards are ordered by vocab offset and ``argmax`` keeps the first
        maximum, so ties resolve to the lowest token id exactly like a
        full-vocab ``argmax``.
        """
        gathered = gathered.view(tp_size, num_reqs, 2)
        best_rank = gathered[..., 0].argmax(dim=0, keepdim=True)
        token_ids = gathered[..., 1].gather(0, best_rank).squeeze(0)
        return token_ids.to(torch.int64)

    def _sample_markov_sharded(
        self,
        valid_hidden: torch.Tensor,
        ids_3d: torch.Tensor,
        block_size: int,
        sample_start: int,
    ) -> list[torch.Tensor]:
        assert self.draft_model is not None
        vocab_offset = self._vocab_shard_offset
        assert vocab_offset is not None
        padded_num_reqs = ids_3d.shape[0]
        tp_size = self._sharded_markov_tp
        slot_major_hidden = (
            valid_hidden.view(padded_num_reqs, block_size, valid_hidden.shape[-1])
            .transpose(0, 1)
            .contiguous()
            .view(valid_hidden.shape)
        )
        logits = self.draft_model.compute_draft_logits_local(slot_major_hidden)
        if self._vocab_shard_mask is not None:
            logits = logits + self._vocab_shard_mask
        logits_slot_major = logits.view(block_size, padded_num_reqs, logits.shape[-1])

        prev = ids_3d[:, 0]
        map_to_target = self.draft_model.map_draft_to_target
        draft_tokens = []
        for i in range(sample_start, block_size):
            scores = logits_slot_major[i] + self._markov_bias_local(prev)
            packed = self._pack_local_candidate(scores, vocab_offset)
            gathered = tensor_model_parallel_all_gather(packed, dim=0)
            step = map_to_target(
                self._select_gathered_candidate(gathered, tp_size, padded_num_reqs)
            )
            draft_tokens.append(step)
            prev = step
        return draft_tokens

    def _sample_markov_replicated(
        self,
        slot_major_hidden: torch.Tensor,
        ids_3d: torch.Tensor,
        block_size: int,
        sample_start: int,
    ) -> list[torch.Tensor]:
        assert self.draft_model is not None
        padded_num_reqs = ids_3d.shape[0]
        num_valid_tokens = padded_num_reqs * block_size
        # Draft-vocab logits for every slot; the Markov bias is added in
        # draft space and ids are remapped to target vocab after argmax.
        logits = self.draft_model.compute_draft_logits(slot_major_hidden)
        logits_slot_major = logits[:num_valid_tokens].view(
            block_size, padded_num_reqs, logits.shape[-1]
        )

        prev = ids_3d[:, 0]
        markov_embed = self.draft_model.markov_embed
        markov_bias = self.draft_model.markov_bias
        map_to_target = self.draft_model.map_draft_to_target
        draft_tokens = []
        for i in range(sample_start, block_size):
            bias = markov_bias(markov_embed(prev))
            step = map_to_target((logits_slot_major[i] + bias).argmax(dim=-1))
            draft_tokens.append(step)
            prev = step
        return draft_tokens

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _dflash_forward_and_sample(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backbone forward + sequential greedy Markov sampling, in-graph.

        Dense layout: every query slot predicts the token at its position + 1,
        so all block_size slots are sampled (DFlash samples only the mask
        slots). Bonus-anchor layout: the anchor is a pure bonus token, so only
        the K mask slots (1..block_size-1) are sampled. Either way the Markov
        chain seeds from the anchor input id and runs a static K-iteration
        loop: embedding gather + rank-r bias + argmax, either over this rank's
        vocab shard with a [B, 2] all-gather per step (sharded lm_head) or
        over the full gathered vocab (replicated lm_head).
        """
        assert self.draft_model is not None
        hidden = self.draft_model.model(
            input_ids=input_ids,
            positions=positions,
        )

        padded_num_reqs = hidden.shape[0] // block_size
        num_valid_tokens = padded_num_reqs * block_size
        ids_3d = input_ids[:num_valid_tokens].view(padded_num_reqs, block_size)
        sample_start = 0 if self.sample_from_anchor else 1
        if self._sharded_markov_tp > 1:
            valid_hidden = hidden[:num_valid_tokens]
            draft_tokens = self._sample_markov_sharded(
                valid_hidden, ids_3d, block_size, sample_start
            )
        else:
            # Transpose [padded_num_reqs, block_size, H] ->
            # [block_size, padded_num_reqs, H] before compute_draft_logits so
            # each Markov step i indexes a contiguous [padded_num_reqs, vocab]
            # slice instead of a strided [:, i] view across the full draft-vocab
            # logits tensor, and pad back to hidden.shape[0] so the TP all-gather
            # inside compute_draft_logits stays 8-row aligned.
            slot_major_hidden = (
                hidden[:num_valid_tokens]
                .view(padded_num_reqs, block_size, hidden.shape[-1])
                .transpose(0, 1)
                .contiguous()
                .view(num_valid_tokens, hidden.shape[-1])
            )
            pad_tokens = hidden.shape[0] - num_valid_tokens
            if pad_tokens > 0:
                slot_major_hidden = F.pad(slot_major_hidden, (0, 0, 0, pad_tokens))
            draft_tokens = self._sample_markov_replicated(
                slot_major_hidden, ids_3d, block_size, sample_start
            )
        draft_tokens_chunk = torch.stack(draft_tokens, dim=1)

        return draft_tokens_chunk, hidden
