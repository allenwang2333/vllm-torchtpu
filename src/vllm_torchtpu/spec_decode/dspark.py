# SPDX-License-Identifier: Apache-2.0
"""DSpark speculative decoding for TPU.

DSpark is semi-autoregressive parallel drafting: one DFlash-style parallel
block forward (anchor + noise tokens, non-causal within the block, context
KV precomputed from the target's aux hidden states), followed by a
lightweight sequential Markov head that biases each position's draft logits
on the previously sampled token. Upstream implements DSparkSpeculator as a
small DFlashSpeculator subclass; this proposer mirrors that on top of the
TPU DFlashProposer.

Dense-checkpoint layout (``dspark_bonus_anchor`` unset): the query block is
K tokens — the anchor is the FIRST prediction position and every slot
predicts the NEXT token — versus DFlash's 1 + K (anchor is bonus only).

Draft sampling: greedy (argmax + d2t remap) or probabilistic
(``draft_sample_method="probabilistic"``) -- Gumbel-coupled with the target's
rejection sampler via deterministic (seed, position)-keyed noise.
"""
from __future__ import annotations

import numpy as np
import torch
from vllm.config import VllmConfig

from vllm_torchtpu.layers.vllm.sample.keyed_gumbel import keyed_gumbel_noise
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.utils import normalize_draft_config

logger = init_logger(__name__)

# Alias for backwards compatibility
normalize_dspark_draft_config = normalize_draft_config


class DSparkProposer(DFlashProposer):

    def __init__(self, runner, vllm_config: VllmConfig):
        hf_config = (
            vllm_config.speculative_config.draft_model_config.hf_config)
        hf_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else {}

        super().__init__(runner, vllm_config)

        if not self.dflash_config.get("target_layer_ids"):
            raise ValueError(
                "DSpark requires target layer ids in the draft config (one of "
                "eagle_aux_hidden_state_layer_ids, dspark_target_layer_ids, "
                "target_layer_ids, or dflash_config.target_layer_ids).")

        self.sample_from_anchor = not hf_dict.get("dspark_bonus_anchor", False)
        block = hf_dict.get("dspark_block_size") or hf_dict.get("block_size")
        K = self.speculative_config.num_speculative_tokens
        if block is not None and K < int(block):
            raise ValueError(
                f"num_speculative_tokens={K} is smaller than the DSpark "
                f"checkpoint's block size {block}; the block/Markov-head "
                "machinery yields garbled output below the trained block.")

        # Probabilistic (Gumbel-coupled) drafting caches the per-step proposal
        # logits so the verifier can recompute q(x) bitwise; greedy needs none
        # of that state. Buffers are allocated in load_model (vocab/dtype).
        self.draft_sample_probabilistic = (getattr(
            self.speculative_config, "draft_sample_method",
            "greedy") == "probabilistic")
        if (self.draft_sample_probabilistic
                and vllm_config.scheduler_config.async_scheduling):
            # The coupled metadata (key positions, cache-row gather) is built
            # from this step's host state; the async path substitutes draft
            # ids on device after the host build. Not wired up yet.
            raise NotImplementedError(
                "draft_sample_method='probabilistic' is not supported with "
                "async scheduling on TPU yet; use sync scheduling.")
        self.draft_logits_cache: torch.Tensor | None = None
        self._d2t_scatter_index: torch.Tensor | None = None
        # req_id -> cache row occupied at the most recent draft step; the
        # runner uses it to gather cache rows for next-step verification.
        self.draft_cache_rows: dict[str, int] = {}

        logger.info(
            "DSpark: sample_from_anchor=%s, K=%d, draft_sampling=%s.",
            self.sample_from_anchor, K,
            "probabilistic" if self.draft_sample_probabilistic else "greedy")

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
        for attr in ("compute_draft_logits", "markov_embed", "markov_bias",
                     "map_draft_to_target"):
            if not hasattr(self.draft_model, attr):
                raise RuntimeError(
                    f"DSpark draft model lacks {attr}(); expected a "
                    "Qwen3DSpark/Gemma4DSpark-family model.")

        if self.draft_sample_probabilistic:
            device = next(self.draft_model.parameters()).device
            dtype = next(self.draft_model.parameters()).dtype
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            max_num_reqs = self.runner.max_num_reqs
            K = self.speculative_config.num_speculative_tokens
            # Pre-temperature proposal logits in TARGET vocab space, one row
            # per (request slot, draft step). ~[reqs, K, V] in model dtype;
            # e.g. 256 x 7 x 152k bf16 = ~545 MB, sized by the engine's
            # max_num_seqs. Row max_num_reqs is a trash row that padding
            # slots scatter into.
            self.draft_logits_cache = torch.zeros(max_num_reqs + 1,
                                                  K,
                                                  vocab_size,
                                                  dtype=dtype,
                                                  device=device)
            d2t = getattr(self.draft_model, "draft_id_to_target_id", None)
            if d2t is not None:
                # Reduced draft vocab: scatter draft columns into their
                # target-vocab rows before caching/sampling (upstream
                # DSparkSpeculator._d2t_scatter_index).
                self._d2t_scatter_index = (
                    torch.arange(d2t.shape[0], device=device) + d2t)

    def _chunk_sample_args(self, chunk, padded_num_reqs: int) -> tuple:
        """Per-chunk sampling state for the compiled draft+sample graph.

        Greedy drafting needs none. Probabilistic drafting needs, per padded
        chunk row: the request's Gumbel seed, its temperature, and the
        persistent cache row its proposal logits land in (trash row for
        padding slots). Also snapshots req_id -> cache row for the runner's
        next-step verification gather.
        """
        if not self.draft_sample_probabilistic:
            return ()
        runner = self.runner
        state_buf = self.draft_logits_cache
        device = state_buf.device
        trash_row = state_buf.shape[0] - 1

        seeds_np = np.zeros(padded_num_reqs, dtype=np.int64)
        temps_np = np.zeros(padded_num_reqs, dtype=np.float32)
        rows_np = np.full(padded_num_reqs, trash_row, dtype=np.int64)
        for i in range(chunk.num_reqs):
            batch_row = chunk.start_index + i
            req_id = runner.input_batch.req_ids[batch_row]
            seeds_np[i] = runner.get_spec_seed(req_id)
            temps_np[i] = runner.input_batch.temperature_cpu[batch_row]
            rows_np[i] = batch_row
            self.draft_cache_rows[req_id] = batch_row

        seeds = torch.from_numpy(seeds_np).to(device)
        temps = torch.from_numpy(temps_np).to(device)
        cache_rows = torch.from_numpy(rows_np).to(device)
        return (seeds, temps, cache_rows)

    def _dummy_sample_args(self, padded_num_reqs: int) -> tuple:
        if not self.draft_sample_probabilistic:
            return ()
        state_buf = self.draft_logits_cache
        device = state_buf.device
        trash_row = state_buf.shape[0] - 1
        return (
            torch.zeros(padded_num_reqs, dtype=torch.int64, device=device),
            torch.ones(padded_num_reqs, dtype=torch.float32, device=device),
            torch.full((padded_num_reqs, ),
                       trash_row,
                       dtype=torch.int64,
                       device=device),
        )

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _dflash_forward_and_sample(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
        seeds: torch.Tensor | None = None,
        temperatures: torch.Tensor | None = None,
        cache_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backbone forward + sequential Markov sampling, in-graph.

        Dense layout: every query slot predicts the token at its position + 1,
        so all block_size slots are sampled (DFlash samples only the mask
        slots). Bonus-anchor layout: the anchor is a pure bonus token, so only
        the K mask slots (1..block_size-1) are sampled. Either way the Markov
        chain seeds from the anchor input id and runs a static K-iteration
        loop: embedding gather + rank-r bias + sample.

        Greedy mode: argmax over draft-vocab logits + d2t remap. Probabilistic
        mode: proposal logits are scattered into target vocab, cached
        pre-temperature for the verifier (Gumbel-coupled rejection), and
        sampled via keyed-Gumbel argmax with the predicted position - 1 as the
        noise key -- the verifier re-derives the same key from its input-token
        positions.
        """
        hidden = self.draft_model.model(
            input_ids=input_ids,
            positions=positions,
        )

        padded_num_reqs = hidden.shape[0] // block_size
        valid_hidden = hidden[:padded_num_reqs * block_size]

        logits = self.draft_model.compute_draft_logits(valid_hidden)
        logits_3d = logits.view(padded_num_reqs, block_size, logits.shape[-1])

        ids_3d = input_ids[:padded_num_reqs * block_size].view(
            padded_num_reqs, block_size)
        pos_3d = positions[:padded_num_reqs * block_size].view(
            padded_num_reqs, block_size)
        prev = ids_3d[:, 0]
        draft_tokens = []
        sample_start = 0 if self.sample_from_anchor else 1

        if self.draft_sample_probabilistic:
            vocab_size = self.draft_logits_cache.shape[-1]
            is_greedy = temperatures == 0.0
            safe_temp = torch.where(is_greedy, torch.ones_like(temperatures),
                                    temperatures)

        for i in range(sample_start, block_size):
            markov_embed = self.draft_model.markov_embed(prev)
            bias = self.draft_model.markov_bias(markov_embed)
            proposal = logits_3d[:, i] + bias

            if not self.draft_sample_probabilistic:
                step = self.draft_model.map_draft_to_target(
                    proposal.argmax(dim=-1))
            else:
                if self._d2t_scatter_index is not None:
                    full = torch.full(
                        (padded_num_reqs, vocab_size),
                        float("-inf"),
                        dtype=proposal.dtype,
                        device=proposal.device,
                    ).index_copy_(1, self._d2t_scatter_index, proposal)
                else:
                    full = proposal
                # Cache pre-temperature; the verifier divides by the same
                # temperature on load, reproducing the sampled value bitwise.
                step_col = i - sample_start
                self.draft_logits_cache[:, step_col].index_copy_(
                    0, cache_rows, full)
                # The token predicted here lands at query position + 1
                # (dense) or at the slot position (bonus-anchor); either way
                # the verifier sees its predecessor at key position, so key
                # on that predecessor.
                pos_key = (pos_3d[:, i]
                           if self.sample_from_anchor else pos_3d[:, i] - 1)
                gumbel = keyed_gumbel_noise(seeds, pos_key, vocab_size)
                noisy = (full.to(torch.float32) /
                         safe_temp.unsqueeze(-1)) + torch.where(
                             is_greedy.unsqueeze(-1), torch.zeros_like(gumbel),
                             gumbel)
                step = noisy.argmax(dim=-1).to(torch.int32)

            draft_tokens.append(step)
            prev = step
        draft_tokens_chunk = torch.stack(draft_tokens, dim=1)

        return draft_tokens_chunk, hidden
