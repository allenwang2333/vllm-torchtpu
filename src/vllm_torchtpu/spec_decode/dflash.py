# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import copy
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from vllm.compilation.backends import set_model_tag
from vllm.config import (VllmConfig, get_layers_from_vllm_config,
                         set_current_vllm_config)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model_loader

from vllm_torchtpu.layers.common.attention_metadata import (
    AttentionMetadata, AttentionMetadataBuilderContext)
from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackendImpl
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID
from vllm_torchtpu.spec_decode.utils import (DraftChunkInputs,
                                             _force_draft_tp1,
                                             maybe_share_embeddings,
                                             maybe_share_lm_head)
from vllm_torchtpu.utils import synchronize_tensors

if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


class DFlashProposer:
    """DFlash draft proposer for TPU.

    Implements a parallel block-level speculative decoding framework.
    """

    def __init__(
        self,
        runner: "TPUModelRunner",
        vllm_config: VllmConfig,
    ):
        self.runner = runner
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None

        # Read DFlash specific configurations
        hf_config = self.speculative_config.draft_model_config.hf_config
        hf_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else {}
        self.dflash_config = hf_dict.get("dflash_config", {})
        self.mask_token_id = self.dflash_config.get("mask_token_id")
        assert self.mask_token_id is not None, (
            "DFlash requires `mask_token_id` to be defined in the draft model's config.json "
            "under the `dflash_config` dictionary.")
        # Without tagged target layers no aux hidden states flow to the
        # draft; the failure would otherwise surface much later as a width
        # mismatch inside the compiled KV update.
        self._target_layer_ids: list[int] = list(
            self.dflash_config.get("target_layer_ids") or [])
        if not self._target_layer_ids:
            raise ValueError(
                "DFlash requires a non-empty `target_layer_ids` list in the "
                "draft model's config.json under `dflash_config`.")

        draft_tp = self.speculative_config.draft_tensor_parallel_size
        target_tp = self.vllm_config.parallel_config.tensor_parallel_size
        # Default to (draft_tp == target tp) when unset. vLLM's
        # _verify_and_get_draft_tp already resolves an unset value to target tp
        # for dflash, but set it explicitly here so self-documented.
        if draft_tp is None:
            draft_tp = target_tp
            self.speculative_config.draft_tensor_parallel_size = draft_tp
        # Only two draft parallelisms are supported: fully REPLICATED (tp=1) or
        # SHARDED across the whole TP group (draft_tp == target_tp).
        if draft_tp not in (1, target_tp):
            raise ValueError(
                f"dflash draft_tensor_parallel_size={draft_tp} is unsupported "
                f"on TPU: it must be 1 (replicated draft) or {target_tp} "
                f"(== target tensor_parallel_size, sharded draft).")
        self._draft_replicated = (draft_tp == 1)
        # Independent from replication: target TP1 / draft TP1 can alias.
        self._draft_tp_matches_target = (draft_tp == target_tp)
        logger.info(
            "DFlash draft parallelism: %s (draft_tp=%s).",
            "REPLICATED (tp=1)" if self._draft_replicated else "SHARDED",
            draft_tp)

        self._max_model_len = vllm_config.model_config.max_model_len
        self.draft_model = None
        self._draft_attn_layer_names: set[str] | None = None
        self.draft_chunks: list[DraftChunkInputs] | None = None
        self._static_attn_tensors_cache: dict[int, tuple[torch.Tensor,
                                                         torch.Tensor]] = {}
        # Keyed by (num_tokens, num_reqs, num_blocks); one entry per replay
        # shape, see `_dummy_kv_update_inputs`.
        self._dummy_kv_input_cache: dict[tuple,
                                         tuple[list[torch.Tensor],
                                               "AttentionMetadata"]] = {}
        # Keyed by num_tokens; constant inputs of one dummy draft forward,
        # see `_dummy_forward_inputs`.
        self._dummy_forward_input_cache: dict[int, tuple] = {}
        # Keyed by the chunk's padded request count; the all-zero
        # `draft_lengths` a chunk without spec metadata routes on.
        self._zero_draft_lengths_cache: dict[int, torch.Tensor] = {}

    @property
    def block_size(self) -> int:
        """Tokens per request in a draft forward: 1 base + K mask slots."""
        return self.speculative_config.num_speculative_tokens + 1

    def _get_static_attn_tensors(
            self, padded_num_reqs: int,
            block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch or create persistent static TPU tensors for draft attention metadata."""
        if padded_num_reqs not in self._static_attn_tensors_cache:
            qsl = torch.arange(padded_num_reqs + 1,
                               dtype=torch.int32,
                               device=self.runner.device) * block_size
            rd = torch.tensor([0, 0, padded_num_reqs],
                              dtype=torch.int32,
                              device=self.runner.device)
            self._static_attn_tensors_cache[padded_num_reqs] = (qsl, rd)
        return self._static_attn_tensors_cache[padded_num_reqs]

    def load_model(self, target_model) -> None:
        """Load the draft model, share embeddings, and tag target layers."""
        # Tag the target model to extract aux hidden states
        target_layers = [idx + 1 for idx in self._target_layer_ids]
        if hasattr(target_model, "set_aux_hidden_state_layers"):
            target_model.set_aux_hidden_state_layers(tuple(target_layers))
        elif hasattr(target_model, "model") and hasattr(
                target_model.model, "_set_aux_hidden_state_layers"):
            target_model.model._set_aux_hidden_state_layers(
                tuple(target_layers))
        else:
            raise RuntimeError(
                "Target model does not support _set_aux_hidden_state_layers")
        logger.info(
            "Tagged target model for DFlash auxiliary hidden states extraction."
        )

        # Snapshot the target's attention layer names before the draft model
        # adds its own; diff after load gives us the draft layer names.
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config,
                                        AttentionLayerBase).keys())

        self._load_draft_model()

        all_attn_layers = get_layers_from_vllm_config(self.vllm_config,
                                                      AttentionLayerBase)
        self._draft_attn_layer_names = (set(all_attn_layers.keys()) -
                                        target_attn_layer_names)

        # Share Embeddings and LM Head
        maybe_share_embeddings(self.draft_model,
                               target_model,
                               self._draft_tp_matches_target,
                               force_share=True)
        maybe_share_lm_head(self.draft_model,
                            target_model,
                            self._draft_replicated,
                            self._draft_tp_matches_target,
                            force_share=True)

        if hasattr(self.draft_model, "get_draft_attn_causal"):
            layer_causal_list = self.draft_model.get_draft_attn_causal()
        elif hasattr(self.draft_model.model, "get_draft_attn_causal"):
            layer_causal_list = self.draft_model.model.get_draft_attn_causal()
        else:
            raise RuntimeError(
                "Draft model does not support get_draft_attn_causal")

        with set_vllm_model_wrapper_context(mesh=self.runner.mesh):
            for i, layer in enumerate(self.draft_model.model.layers):
                attn_impl = layer.self_attn.attn.impl
                if isinstance(attn_impl, PallasAttentionBackendImpl):
                    attn_impl.use_causal_mask = layer_causal_list[i]
                    attn_impl.initialize_kernel(layer.self_attn.attn)

    def _load_draft_model(self) -> None:
        logger.info("Loading DFlash draft model...")
        model_loader = get_model_loader(self.vllm_config.load_config)
        draft_tp1_ctx = (_force_draft_tp1() if self._draft_replicated else
                         contextlib.nullcontext())
        draft_vllm_config = copy.copy(self.vllm_config)
        draft_model_config = copy.copy(
            self.speculative_config.draft_model_config)
        draft_model_config.runner_type = "draft"
        with set_model_tag("dflash_head"), set_vllm_model_wrapper_context(
                mesh=self.runner.mesh), set_current_vllm_config(
                    draft_vllm_config), draft_tp1_ctx:
            self.draft_model = model_loader.load_model(
                vllm_config=draft_vllm_config,
                model_config=draft_model_config,
            )

    def _get_padded_len(self, target_len: int) -> int:
        for p in self.runner.num_tokens_paddings:
            if p >= target_len:
                return p
        return target_len

    def _build_draft_attn_metadata(
        self,
        chunk: DraftChunkInputs,
        num_tokens_padded: int,
        seq_lens: torch.Tensor,
    ) -> dict:
        """
        Construct attention metadata for DFlash draft forward.
        Scales query boundaries and request token distribution to block_size.
        """
        runner = self.runner
        chunk_ctx = chunk.attn_ctx
        block_size = self.block_size
        # `num_tokens_padded` is a bucket of at least one full block.
        padded_num_reqs = num_tokens_padded // block_size

        query_start_loc, request_distribution = self._get_static_attn_tensors(
            padded_num_reqs, block_size)

        assert self._draft_attn_layer_names is not None, (
            "load_model() must run before draft attention metadata is built")

        saved_ctx = runner._attn_metadata_builder_ctx
        # num_reqs must be the REAL request count: the staging walk copies
        # exactly ctx.num_reqs block-table rows, and rows beyond the live
        # batch hold departed requests' freed block ids. With the real
        # count, padding rows stage as zeros -> the null block.
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=chunk.num_reqs,
            start_index=chunk.start_index,
            use_max_model_len=chunk_ctx.use_max_model_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            position_ids_override=chunk_ctx.position_ids_override,
        )
        # No mamba fields: DFlash drafters are attention-only, so the build
        # below visits no mamba group and nothing would read them. A drafter
        # with recurrent layers needs more than a slot tensor here anyway —
        # `unified_mamba_state_indices`, the read-offset scatter, and the
        # state seed copies are all target-path only.
        try:
            # Only the draft layers' KV cache groups. The full build walks
            # every group in the deployment and we would discard all but
            # these — see `build_attention_metadata_for_layers`.
            return runner.build_attention_metadata_for_layers(
                self._draft_attn_layer_names, padded_num_reqs)
        finally:
            runner._attn_metadata_builder_ctx = saved_ctx

    def _prepare_dflash_inputs(
        self,
        chunk: DraftChunkInputs,
        next_tokens_device: torch.Tensor | None = None,
        device_seed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepares input_ids and positions tensors for DFlash draft execution.

        Each request block is formatted as:
          input_ids: [base_token, MASK_TOKEN, ..., MASK_TOKEN]  (len = block_size)
          positions: [base_pos, base_pos+1, ..., base_pos+K]    (len = block_size)

        The base token is the last verified token from the target model.
        """
        # 1. Calculate static padded shapes
        num_reqs = chunk.num_reqs
        block_size = self.block_size
        total_draft_tokens = num_reqs * block_size

        # Round up to the nearest static padded length to avoid XLA recompilation
        if self.runner._dp_lockstep_enabled():
            # EP-DP lockstep: every rank must emit identical collective
            # shapes, and num_reqs is rank-local. Pad to the step's DP-wide
            # max per-chunk request count, coordinated by the same
            # all-reduce that sizes the target bucket
            # (`_dp_coordinated_step`); run_dp_dummy_draft replays the same
            # shape on idle ranks.
            padded_len = self._dp_draft_bucket()
            # The coordinated max is an exact per-chunk bound
            # (_count_input_chunks); a shortfall here would silently
            # truncate draft inputs via negative F.pad, so fail loudly.
            assert padded_len >= total_draft_tokens, (
                f"lockstep draft bucket {padded_len} < local chunk's "
                f"{total_draft_tokens} draft tokens")
        else:
            padded_len = self._get_padded_len(total_draft_tokens)

        query_start_loc_device = chunk.attn_ctx.query_start_loc
        position_ids_device = chunk.position_ids
        if position_ids_device.dim() > 1:
            position_ids_device = position_ids_device[0]

        # Route to the appropriate pre-compiled XLA Graph
        if next_tokens_device is not None:
            # No spec metadata means nothing was verified this step, so every
            # request anchors on the prompt-chunk path.
            draft_lengths = chunk.draft_lengths
            if draft_lengths is None:
                draft_lengths = self._zero_draft_lengths(num_reqs)
            input_ids, positions, seq_lens = self._tpu_build_dflash_inputs_next_tokens(
                next_tokens_device,
                query_start_loc_device,
                position_ids_device,
                draft_lengths,
                num_reqs,
                block_size,
                padded_len,
                self.mask_token_id,
                INVALID_TOKEN_ID,
            )
        elif device_seed is not None:
            input_ids, positions, seq_lens = self._tpu_build_dflash_inputs_seed(
                device_seed,
                query_start_loc_device,
                position_ids_device,
                num_reqs,
                block_size,
                padded_len,
                self.mask_token_id,
            )
        else:
            raise RuntimeError(
                "DFlash propose called without next_tokens_device or device_seed. "
                "This drafter is designed strictly for the fused device path.")

        return input_ids, positions, seq_lens

    def _zero_draft_lengths(self, num_reqs: int) -> torch.Tensor:
        """All-zero `draft_lengths` for a chunk without spec metadata.

        Padded exactly like the real tensor (`get_spec_decode_metadata`
        pads to the chunk's padded request count) so both steps share one
        compiled input-builder program, and cached so a non-spec step
        allocates nothing.
        """
        from vllm_torchtpu.runner.tpu_runner import \
            _get_padded_num_reqs_with_upper_limit
        padded_num_reqs = _get_padded_num_reqs_with_upper_limit(
            num_reqs, self.runner.max_num_reqs)
        zeros = self._zero_draft_lengths_cache.get(padded_num_reqs)
        if zeros is None:
            zeros = torch.zeros(padded_num_reqs,
                                dtype=torch.int32,
                                device=self.runner.device)
            self._zero_draft_lengths_cache[padded_num_reqs] = zeros
        return zeros

    def _build_draft_layer_metadata(
        self,
        attn_metadata: dict[str, "AttentionMetadata"],
    ) -> tuple["AttentionMetadata", ...]:
        """Per-layer metadata for the draft's KV write, in layer order.

        `attn_metadata` is the *target's* full per-layer dict, so a draft layer
        that fails to resolve must raise rather than fall back to an arbitrary
        entry: the first entry is whichever layer hashed first — plausibly a
        mamba group, whose block table indexes a different block size. Writing
        the draft's KV through it would corrupt the cache silently instead of
        failing.
        """
        draft_md_list = []
        for layer in self.draft_model.model.layers:
            layer_name = getattr(layer.self_attn.attn, "layer_name", None)
            md = attn_metadata.get(layer_name) if layer_name else None
            if md is None:
                raise RuntimeError(
                    f"No attention metadata for draft layer {layer_name!r}; "
                    "the draft layers must own a KV cache group of their own.")
            draft_md_list.append(md)
        return tuple(draft_md_list)

    def _update_draft_kv_cache_from_target(self, chunk) -> None:
        """
        Projects the target model's hidden states into the draft model's KV cache.

        This allows the draft model to maintain context without needing to run a full
        forward pass on the tokens that were just verified by the target model.
        It applies RoPE to the target's hidden states and scatters the tokens directly
        into the draft KV cache.
        """
        target_hidden = chunk.aux_hidden_states
        if not target_hidden:
            raise RuntimeError(
                "DFlash chunk has no aux hidden states; the target model must "
                "be tagged via set_aux_hidden_state_layers before propose().")
        num_tokens = target_hidden[0].shape[0]

        positions = chunk.position_ids
        if positions.dim() > 1:
            positions = positions[0]
        positions = positions[:num_tokens]
        draft_md_tuple = self._build_draft_layer_metadata(chunk.attn_metadata)

        with set_vllm_model_wrapper_context(mesh=self.runner.mesh):
            self._tpu_precompute_and_update_kv_cache(target_hidden, positions,
                                                     draft_md_tuple)

    @torch.no_grad()
    def propose(
        self,
        sampled_token_ids: list[list[int]],
        discard_sampled_tokens_req_indices: list[int],
        num_rejected_tokens_np: np.ndarray | None,
        scheduler_output,
        return_device: bool = False,
        next_tokens_per_chunk: list[torch.Tensor] | None = None,
        device_seed: torch.Tensor | None = None,
    ) -> list[list[int]] | torch.Tensor:
        """Main execution step to propose draft tokens.

        `sampled_token_ids`, `discard_sampled_tokens_req_indices`,
        `num_rejected_tokens_np` and `scheduler_output` are unused here: this
        drafter is strictly device-fused and seeds from `next_tokens_per_chunk`
        or `device_seed`. They stay in the signature because
        `SpeculativeDecodingManager` calls `propose` polymorphically across
        this and `Eagle3Proposer`, which does read them.
        """
        runner = self.runner
        num_reqs = runner.input_batch.num_reqs
        K = self.speculative_config.num_speculative_tokens
        if num_reqs == 0:
            if return_device:
                return torch.empty((0, K),
                                   dtype=torch.int32,
                                   device=runner.device)
            return []

        chunks = self.draft_chunks
        if not chunks:
            raise RuntimeError(
                "DFlash propose called but no draft chunks captured.")

        block_size = self.block_size

        # EP-DP lockstep: a rank whose step produced fewer chunks than the
        # DP-group max must pad each draft phase to the group's chunk count —
        # collectives pair across ranks by emission order, and propose emits
        # ALL KV updates before ALL draft forwards, so the padding must
        # follow the same phase structure (never interleave per chunk).
        extra_chunks = 0
        if runner._dp_lockstep_enabled():
            extra_chunks = max(0, runner._dp_step_num_chunks - len(chunks))

        # 1. Hoist all KV cache updates across chunks to the top of propose
        for chunk in chunks:
            self._update_draft_kv_cache_from_target(chunk)
        for _ in range(extra_chunks):
            self._dp_dummy_kv_update()

        # 2. Process each chunk
        draft_tokens_per_chunk = []
        for i, chunk in enumerate(chunks):
            next_tokens_device = next_tokens_per_chunk[
                i] if next_tokens_per_chunk else None
            # device_seed spans the whole batch; each chunk seeds only its own
            # requests.
            chunk_seed = (device_seed[chunk.start_index:chunk.start_index +
                                      chunk.num_reqs]
                          if device_seed is not None else None)
            input_ids, positions, seq_lens = self._prepare_dflash_inputs(
                chunk,
                next_tokens_device=next_tokens_device,
                device_seed=chunk_seed)
            padded_len = input_ids.shape[0]
            draft_attn_metadata = self._build_draft_attn_metadata(
                chunk, padded_len, seq_lens)

            with (
                    # Precomputed so set_forward_context does not run its
                    # own cross-DP CPU all_reduce per draft forward; must
                    # pair exactly with the idle replay's calls.
                    set_forward_context(
                        draft_attn_metadata,
                        self.vllm_config,
                        num_tokens=padded_len,
                        num_tokens_across_dp=runner._dp_num_tokens_across_dp(
                            padded_len)),
                    set_vllm_model_wrapper_context(mesh=self.runner.mesh),
            ):
                draft_tokens_chunk, _ = self._dflash_forward_and_sample(
                    input_ids, positions, block_size)

            draft_tokens_per_chunk.append(draft_tokens_chunk)
        for _ in range(extra_chunks):
            self._dp_dummy_forward()

        # 3. Drop each chunk's padding rows; [total_num_reqs, K].
        if return_device:
            draft_tokens = [
                tokens[:chunk.num_reqs]
                for tokens, chunk in zip(draft_tokens_per_chunk, chunks)
            ]
            if len(draft_tokens) == 1:
                return draft_tokens[0]
            return torch.cat(draft_tokens, dim=0)

        # Sync path: transfer each chunk's ENTIRE padded tensor and slice on
        # the host, so the D2H keeps a static shape per bucket.
        draft_tokens_list = []
        for tokens, chunk in zip(draft_tokens_per_chunk, chunks):
            draft_tokens_list.extend(tokens.cpu().tolist()[:chunk.num_reqs])
        return draft_tokens_list

    @torch.no_grad()
    def precompile(self) -> None:
        """Warm up the DFlash draft model across all padding buckets.
        Called by tpu_runner.py during _dummy_run.
        """
        if self.draft_model is None:
            return

        logger.info("Precompiling DFlash draft model buckets...")

        runner = self.runner
        block_size = self.block_size
        # A most-model-len chunk holds up to num_reqs_most_model_len
        # requests (> num_reqs_max_model_len), and under EP-DP lockstep the
        # coordinated bucket is sized from the DP-wide max chunk request
        # count — so warm draft forwards up to the larger bound, or the
        # first such step compiles XLA mid-serving.
        max_chunk_reqs = max(runner.num_reqs_max_model_len,
                             runner.num_reqs_most_model_len or 0)
        max_draft_tokens = max_chunk_reqs * block_size

        with runner._precompile_timed("drafter first pass"):
            for num_tokens in runner.num_tokens_paddings:
                # The KV update runs on every chunk (prefill and decode)
                # through the TARGET chunk's metadata, so warm it at each
                # chunk geometry the target uses.
                self._dummy_precompute_and_update_kv_cache(
                    num_tokens=num_tokens,
                    num_reqs=runner.num_reqs_max_model_len,
                    num_blocks=runner.max_num_blocks_per_req,
                    sync=True,
                )
                if runner.most_model_len is not None:
                    self._dummy_precompute_and_update_kv_cache(
                        num_tokens=num_tokens,
                        num_reqs=runner.num_reqs_most_model_len,
                        num_blocks=runner.num_blocks_per_most_len_req,
                        sync=True,
                    )

                # A draft forward is always at least one full block and
                # never more than the largest chunk's request count worth.
                if num_tokens < block_size or num_tokens > max_draft_tokens:
                    continue

                self._dummy_draft_forward(num_tokens=num_tokens,
                                          use_max_model_len=True,
                                          sync=True)
                if runner.most_model_len is not None:
                    self._dummy_draft_forward(num_tokens=num_tokens,
                                              use_max_model_len=False,
                                              sync=True)

    def _kv_update_hidden_spec(self) -> tuple[list[int], torch.dtype]:
        """`(per-aux-tensor widths, dtype)` of the KV update's hidden input.

        The runner hands `_update_draft_kv_cache_from_target` one tensor per
        tagged target layer, which the compiled program concatenates; the
        dummies must have the same list length and widths or they trace a
        different program and the real one compiles live on the first step.
        """
        # `load_model` tags exactly `target_layer_ids` target layers, and
        # each yields one [num_tokens, target_hidden] tensor.
        target_hidden = self.vllm_config.model_config.get_hidden_size()
        return ([target_hidden] * len(self._target_layer_ids),
                self.draft_model.model.embed_tokens.weight.dtype)

    def _dummy_kv_update_inputs(
        self,
        num_tokens: int,
        num_reqs: int,
        num_blocks: int,
    ) -> tuple[list[torch.Tensor], "AttentionMetadata"]:
        """Reusable `(aux hidden list, metadata)` for one KV-update shape.

        Under lockstep this runs once per padding chunk per step, so build
        the constant dummies once per bucket instead of reallocating them
        and re-issuing scalar H2D writes every idle step. `(num_reqs,
        num_blocks)` is the block-table geometry of the target chunk whose
        metadata the real call reuses.
        """
        key = (num_tokens, num_reqs, num_blocks)
        entry = self._dummy_kv_input_cache.get(key)
        if entry is not None:
            return entry

        runner = self.runner
        aux_widths, dtype = self._kv_update_hidden_spec()
        dummy_hidden = [
            torch.zeros((num_tokens, width), dtype=dtype, device=runner.device)
            for width in aux_widths
        ]
        dummy_positions = torch.zeros(num_tokens,
                                      dtype=torch.int32,
                                      device=runner.device)

        block_tables = torch.zeros((num_reqs * num_blocks, ),
                                   dtype=torch.int32,
                                   device=runner.device)
        seq_lens = torch.zeros((num_reqs, ),
                               dtype=torch.int32,
                               device=runner.device)
        seq_lens[0] = num_tokens

        query_start_loc = torch.zeros((num_reqs + 1, ),
                                      dtype=torch.int32,
                                      device=runner.device)
        query_start_loc[1:] = num_tokens

        request_distribution = torch.tensor([0, 0, num_reqs],
                                            dtype=torch.int32,
                                            device=runner.device)

        dummy_attn_metadata = AttentionMetadata(
            input_positions=dummy_positions,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
        )
        entry = (dummy_hidden, dummy_attn_metadata)
        self._dummy_kv_input_cache[key] = entry
        return entry

    def _dummy_precompute_and_update_kv_cache(
        self,
        num_tokens: int,
        num_reqs: int,
        num_blocks: int,
        sync: bool,
    ) -> None:
        """One KV-update pass on constant dummies.

        The dummy's result is never consumed, so the deferred program is
        always materialized and enqueued here (in emission order, which the
        cross-rank collective pairing relies on). `sync` additionally waits
        on the host — what `precompile` wants, and what the per-step
        lockstep replays must NOT do: they run inside `propose` on the
        decode path, where a host wait per padding chunk stalls every step.
        """
        runner = self.runner
        dummy_hidden, dummy_attn_metadata = self._dummy_kv_update_inputs(
            num_tokens, num_reqs, num_blocks)
        dummy_positions = dummy_attn_metadata.input_positions

        with set_vllm_model_wrapper_context(mesh=runner.mesh):
            out = self._tpu_precompute_and_update_kv_cache(
                dummy_hidden, dummy_positions,
                tuple([dummy_attn_metadata] *
                      len(self.draft_model.model.layers)))
            synchronize_tensors(out, wait=sync)

    def _dummy_forward_inputs(
        self,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, int]:
        """Constant inputs of one dummy draft forward at `num_tokens`.

        Under lockstep `_dp_dummy_forward` runs once per padding chunk per
        step, so build the dummies once per bucket instead of allocating and
        uploading five tensors every idle step.
        """
        entry = self._dummy_forward_input_cache.get(num_tokens)
        if entry is not None:
            return entry

        device = self.runner.device
        block_size = self.block_size
        num_reqs = num_tokens // block_size
        assert num_reqs >= 1, (
            f"dummy draft forward at {num_tokens} tokens < block {block_size}")
        num_tokens_per_req = num_tokens // num_reqs

        input_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        positions = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        query_start_loc = torch.arange(num_reqs + 1,
                                       dtype=torch.int32,
                                       device=device) * num_tokens_per_req
        seq_lens = torch.full((num_reqs, ),
                              num_tokens_per_req,
                              dtype=torch.int32,
                              device=device)
        request_distribution = torch.tensor([0, 0, num_reqs],
                                            dtype=torch.int32,
                                            device=device)
        entry = (input_ids, positions, query_start_loc, seq_lens,
                 request_distribution, num_reqs)
        self._dummy_forward_input_cache[num_tokens] = entry
        return entry

    def _dummy_draft_forward(
        self,
        num_tokens: int,
        use_max_model_len: bool,
        sync: bool,
    ) -> None:
        """One draft forward on constant dummies; see
        `_dummy_precompute_and_update_kv_cache` for `sync`."""
        runner = self.runner
        (input_ids, positions, query_start_loc, seq_lens, request_distribution,
         actual_num_reqs) = self._dummy_forward_inputs(num_tokens)

        # Must match `_build_draft_attn_metadata` exactly: torch-tpu fuses
        # programs by dispatch sequence, so precompiling a different
        # sequence leaves the real one to compile live.
        assert self._draft_attn_layer_names is not None, (
            "load_model() must run before precompile()")
        saved_ctx = runner._attn_metadata_builder_ctx
        # A fresh context per call: the staging walk caches its uploads on
        # the context, and the real path stages once per chunk.
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=actual_num_reqs,
            start_index=0,
            use_max_model_len=use_max_model_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            position_ids_override=positions,
        )
        try:
            per_layer_attn_metadata = (
                runner.build_attention_metadata_for_layers(
                    self._draft_attn_layer_names, actual_num_reqs))
        finally:
            runner._attn_metadata_builder_ctx = saved_ctx

        with (
                # Same contract as propose(): precomputed DP metadata, no
                # hidden per-forward cross-DP all_reduce.
                set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=runner._dp_num_tokens_across_dp(
                        num_tokens)),
                set_vllm_model_wrapper_context(mesh=self.runner.mesh),
        ):
            draft_tokens_chunk, _ = self._dflash_forward_and_sample(
                input_ids, positions, self.block_size)
            synchronize_tensors(draft_tokens_chunk, wait=sync)

    def _dp_lockstep_sharded(self) -> bool:
        """True when the draft emits cross-DP collectives under EP-DP
        lockstep. A DFlash draft always runs TP-sharded (draft_tp must equal
        the target's TP), so this is simply whether lockstep is on.
        """
        return self.runner._dp_lockstep_enabled()

    def _dp_draft_bucket(self) -> int:
        """Draft-forward token bucket every lockstep rank runs this step,
        from the DP-wide max per-chunk request count."""
        coordinated_reqs = max(1, self.runner._dp_step_max_reqs)
        return self._get_padded_len(coordinated_reqs * self.block_size)

    @torch.no_grad()
    def _dp_dummy_kv_update(self) -> None:
        """One padding KV-update pass at the DP-coordinated target bucket —
        the same program a busy rank's `_update_draft_kv_cache_from_target`
        dispatches."""
        runner = self.runner
        bucket = runner._dp_target_bucket
        assert bucket is not None, "_dp_target_bucket unset in lockstep step"
        self._dummy_precompute_and_update_kv_cache(
            num_tokens=bucket,
            num_reqs=runner.num_reqs_max_model_len,
            num_blocks=runner.max_num_blocks_per_req,
            sync=False,
        )

    @torch.no_grad()
    def _dp_dummy_forward(self) -> None:
        """One padding draft forward at the step's coordinated draft bucket —
        the shape `_prepare_dflash_inputs` pins every rank to under
        lockstep, derived from the same `_dp_coordinated_step` all-reduce."""
        self._dummy_draft_forward(num_tokens=self._dp_draft_bucket(),
                                  use_max_model_len=True,
                                  sync=False)

    @torch.no_grad()
    def run_dp_dummy_draft(self, num_chunks: int) -> None:
        """Replay a busy rank's DRAFT collective trace on an idle/padding
        rank, for `num_chunks` padding chunks (EP-DP lockstep).

        Collectives pair across ranks by emission order, and a busy
        propose() emits ALL context KV-updates before ALL draft forwards
        (the hoist at the top of propose), so the replay keeps the same
        phase structure: KV-updates x num_chunks, then forwards x
        num_chunks — never interleaved per chunk.
        """
        if num_chunks <= 0 or self.draft_model is None:
            return
        if not self._dp_lockstep_sharded():
            return
        for _ in range(num_chunks):
            self._dp_dummy_kv_update()
        for _ in range(num_chunks):
            self._dp_dummy_forward()

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _tpu_precompute_and_update_kv_cache(
        self,
        hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor]
        | torch.Tensor,
        positions: torch.Tensor,
        draft_md_tuple: tuple["AttentionMetadata", ...],
    ) -> torch.Tensor:
        """
        Projects target model hidden states directly into the draft model's KV space.

        Instead of computing KV projections and RoPE iteratively per layer (which would
        create L tiny, sequential XLA graphs), we project to a single massive flat tensor
        for all layers at once using `_fused_kv_weight`. We then reshape and compute
        RoPE across all layers simultaneously in a single vectorized operation.
        """
        self_model = self.draft_model.model
        if isinstance(hidden_states, (list, tuple)):
            target_hidden = hidden_states[0] if len(
                hidden_states) == 1 else torch.cat(hidden_states, dim=-1)
        else:
            target_hidden = hidden_states

        # Collapse the concatenated aux hidden states to the draft's width.
        # `combine_hidden_states` is the modern spelling; bare `fc` is what
        # drafters that predate it expose. The concatenated width is one
        # target hidden size per tagged layer -- see `_kv_update_hidden_spec`.
        if hasattr(self.draft_model, "combine_hidden_states"):
            target_hidden = self.draft_model.combine_hidden_states(
                target_hidden)
        elif (fc_layer := getattr(self_model, "fc", None)) is not None:
            target_hidden = fc_layer(target_hidden)
            if isinstance(target_hidden, tuple):
                target_hidden = target_hidden[0]

        if hasattr(self_model, "hidden_norm"):
            target_hidden = self_model.hidden_norm(target_hidden)

        # Project target_hidden to K and V dims for all draft layers
        # all_kv_flat shape: [num_ctx, L * 2 * nkv * hd]
        all_kv_flat = F.linear(target_hidden, self_model._fused_kv_weight,
                               self_model._fused_kv_bias)

        num_ctx = all_kv_flat.shape[0]
        L = len(self_model.layers)
        hd = self_model.layers[0].self_attn.head_dim
        nkv = self_model.layers[0].self_attn.num_kv_heads

        # Single contiguous copy that separates K/V and transposes to layer-major layout
        all_kv = all_kv_flat.view(num_ctx, L, 2, nkv,
                                  hd).permute(2, 1, 0, 3, 4).contiguous()
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd]
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd]

        if hasattr(self_model.layers[0].self_attn, "k_norm"):
            kw = [layer.self_attn.k_norm.weight for layer in self_model.layers]
            kw_stack = torch.stack(kw, dim=0).view(L, 1, 1, hd)
            eps = self_model.layers[0].self_attn.k_norm.variance_epsilon
            variance = all_k.pow(2).mean(-1, keepdim=True)
            all_k_normed = all_k * torch.rsqrt(variance + eps) * kw_stack
        else:
            all_k_normed = all_k

        # Apply rotary_emb once across all layers by flattening L and nkv
        all_k_flat2 = all_k_normed.view(L * num_ctx, nkv, hd)
        positions_repeated = positions.repeat(L)
        nq = self_model.layers[0].self_attn.num_heads
        dummy_q = torch.zeros((L * num_ctx, nq, hd),
                              device=target_hidden.device,
                              dtype=target_hidden.dtype)
        dummy_q_roped_flat, roped_k_flat = self_model.layers[
            0].self_attn.rotary_emb(positions_repeated, dummy_q, all_k_flat2)
        roped_k_all = roped_k_flat.view(L, num_ctx, nkv, hd)
        roped_q_all = dummy_q_roped_flat.view(L, num_ctx, nq, hd)

        # 2. update kv cache in the compiled graph so XLA can fuse the mutations
        for i, layer in enumerate(self.draft_model.model.layers):
            attn_obj = layer.self_attn.attn
            kv_cache = attn_obj.kv_cache
            attn_impl = attn_obj.impl

            # Use normal forward pass to run RPA v3 and update KV cache
            attn_impl.forward(
                layer=attn_obj,
                query=roped_q_all[i],
                key=roped_k_all[i],
                value=all_v[i],
                kv_cache=kv_cache,
                attn_metadata=draft_md_tuple[i],
            )

        return hidden_states

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _dflash_forward_and_sample(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compiled XLA graph for DFlash backbone forward pass and greedy sampling.

        Performs:
          1. Backbone model forward pass on input_ids and positions.
          2. Slices out the K mask tokens (skipping slot 0 base token).
          3. Computes draft logits and performs greedy argmax sampling in-graph.
        """
        # 1. Forward pass through draft model backbone
        hidden = self.draft_model.model(
            input_ids=input_ids,
            positions=positions,
        )

        # 2. Slice off padding tokens added by static TPU bucketing
        padded_num_reqs = hidden.shape[0] // block_size
        valid_hidden = hidden[:padded_num_reqs * block_size]

        # 3. Reshape and slice off slot 0 (base token), keeping only the K mask token slots
        hidden_reshaped = valid_hidden.view(padded_num_reqs, block_size,
                                            hidden.shape[-1])
        draft_hidden = hidden_reshaped[:, 1:, :].reshape(-1, hidden.shape[-1])

        # 4. Compute draft logits and run greedy argmax sampling inside the XLA graph
        logits = self.draft_model.compute_logits(draft_hidden)
        logits_3d = logits.view(padded_num_reqs, block_size - 1,
                                logits.shape[-1])
        draft_tokens_chunk = logits_3d.argmax(dim=-1)

        return draft_tokens_chunk, hidden

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _tpu_build_dflash_inputs_next_tokens(
        self,
        next_tokens_device: torch.Tensor,
        query_start_loc: torch.Tensor,
        position_ids: torch.Tensor,
        draft_lengths: torch.Tensor,
        num_reqs: int,
        block_size: int,
        padded_len: int,
        mask_token_id: int,
        invalid_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compiled XLA graph that builds the drafter's inputs directly from the target model's output.

        Args:
            next_tokens_device: Tensor of shape [num_reqs, K+1] containing the target
                model's accepted tokens for this step. Padded with `invalid_token_id`.
            query_start_loc: 1D Tensor containing starting query locations per request.
            position_ids: 1D Tensor containing position IDs.
            draft_lengths: Per-request drafts the target verified this step
                (chunk-local, padded). Zero marks a request whose query was a
                prompt chunk, which anchors differently -- see `base_pos`.
            num_reqs: Number of active requests.
            block_size: K + 1 (1 base token + K draft mask slots).
            padded_len: Total length of the flattened tensors (padded_num_reqs * block_size).
            mask_token_id: The vocabulary ID for the MASK token.
            invalid_token_id: The ID denoting unused/padding tokens in next_tokens_device.

        Returns:
            input_ids: 1D Tensor of shape [padded_len] formatted as:
                [Base, MASK, MASK, ..., Base, MASK, MASK, ...]
            positions: 1D Tensor of shape [padded_len] containing the absolute positional encodings:
                [Pos, Pos+1, Pos+2, ..., Pos', Pos'+1, Pos'+2, ...]
            seq_lens: 1D Tensor of shape [padded_num_reqs] containing the new sequence lengths.
        """
        padded_num_reqs = padded_len // block_size

        # The rejection sampler emits the TARGET's padded row count, which
        # can exceed the draft's; slice before padding so the pad is never
        # negative (a negative F.pad truncates silently).
        next_tokens_rows = next_tokens_device[:padded_num_reqs]
        next_tokens_device_padded = F.pad(
            next_tokens_rows,
            (0, 0, 0, padded_num_reqs - next_tokens_rows.shape[0]),
            value=invalid_token_id).to(torch.int32)

        # Extract start and end positions on-device
        req_indices = torch.arange(padded_num_reqs,
                                   dtype=torch.int32,
                                   device=query_start_loc.device)
        safe_req_indices = torch.clamp(req_indices, max=max(num_reqs - 1, 0))
        start_indices = query_start_loc[safe_req_indices]
        start_positions = position_ids[start_indices].unsqueeze(1)
        valid_mask = (req_indices < num_reqs).unsqueeze(1).to(torch.int32)
        start_positions_padded = (start_positions * valid_mask).to(torch.int32)
        end_indices = torch.clamp(query_start_loc[safe_req_indices + 1] - 1,
                                  min=0)
        last_positions = position_ids[end_indices].unsqueeze(1)

        # Count accepted tokens per request and extract the base token
        num_valid = (next_tokens_device_padded
                     != invalid_token_id).sum(dim=1, dtype=torch.int32)
        last_valid_col = torch.clamp(num_valid, min=1) - 1
        base_tokens = next_tokens_device_padded.gather(
            1, last_valid_col.unsqueeze(1)).squeeze(1)
        # A padding row has no valid token, so the gather returns
        # invalid_token_id (-1) and scattering that into input_ids would index
        # the embedding table negatively. Live rows always hold the bonus
        # token, so this only rewrites padding.
        base_tokens = torch.where(base_tokens == invalid_token_id,
                                  torch.full_like(base_tokens, mask_token_id),
                                  base_tokens)
        num_accepted = num_valid.unsqueeze(1).to(torch.int32)

        # Format input_ids with base tokens and MASK placeholders
        input_ids = torch.full((padded_len, ),
                               mask_token_id,
                               dtype=torch.int32,
                               device=next_tokens_device_padded.device)
        scatter_indices = torch.arange(0,
                                       padded_num_reqs * block_size,
                                       block_size,
                                       dtype=torch.int32,
                                       device=next_tokens_device_padded.device)
        input_ids.scatter_(0, scatter_indices, base_tokens)

        # `start + num_accepted` is only correct for a (K+1)-token verify
        # window; a request whose query this step was a PROMPT chunk (it
        # finished prefill while others verified) must anchor one past its
        # last token, or RoPE runs at the wrong offset and mask K/V lands on
        # its context KV. The chunk's draft count is the exact discriminator:
        # a prompt chunk verified no drafts, whatever its query length.
        num_draft = draft_lengths[safe_req_indices].unsqueeze(1)
        is_verify_window = (num_draft > 0).to(torch.int32)
        verify_base = start_positions_padded + num_accepted
        prompt_base = ((last_positions + 1) * valid_mask).to(torch.int32)
        base_pos = (is_verify_window * verify_base +
                    (1 - is_verify_window) * prompt_base)
        offsets = torch.arange(
            0,
            block_size,
            dtype=torch.int32,
            device=next_tokens_device_padded.device).unsqueeze(0)
        positions_unpadded = (base_pos + offsets).flatten()
        positions = F.pad(positions_unpadded,
                          (0, padded_len - positions_unpadded.shape[0]),
                          value=0)
        # Clamp to max_model_len to prevent out-of-bounds RoPE and kernel indexing.
        positions = torch.clamp(positions, max=self._max_model_len - 1)
        seq_lens = torch.clamp(base_pos.squeeze(1) + block_size,
                               max=self._max_model_len)

        return input_ids, positions, seq_lens

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _tpu_build_dflash_inputs_seed(
        self,
        device_seed: torch.Tensor,
        query_start_loc: torch.Tensor,
        position_ids: torch.Tensor,
        num_reqs: int,
        block_size: int,
        padded_len: int,
        mask_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compiled XLA graph that builds the drafter's inputs from a 1D tensor of seed tokens.

        Args:
            device_seed: 1D Tensor of shape [num_reqs] containing base tokens.
            query_start_loc: 1D Tensor containing starting query locations per request.
            position_ids: 1D Tensor containing position IDs.
            num_reqs: Number of active requests.
            block_size: K + 1 (1 base token + K draft mask slots).
            padded_len: Total length of the flattened tensors (padded_num_reqs * block_size).
            mask_token_id: The vocabulary ID for the MASK token.

        Returns:
            input_ids: 1D Tensor of shape [padded_len] formatted as:
                [SeedToken, MASK, MASK, ..., SeedToken2, MASK, MASK, ...]
            positions: 1D Tensor of shape [padded_len] containing the absolute positional encodings:
                [Pos, Pos+1, Pos+2, ..., Pos', Pos'+1, Pos'+2, ...]
            seq_lens: 1D Tensor of shape [padded_num_reqs] containing the new sequence lengths.
        """
        padded_num_reqs = padded_len // block_size

        seed_rows = device_seed[:padded_num_reqs]
        base_tokens = F.pad(seed_rows, (0, padded_num_reqs - seed_rows.shape[0]),
                            value=mask_token_id).to(torch.int32)

        # Anchor one past the LAST prompt token of the chunk (not the chunk
        # start): a wrong anchor runs RoPE at the wrong offset and writes
        # mask K/V over the context KV.
        req_indices = torch.arange(padded_num_reqs,
                                   dtype=torch.int32,
                                   device=query_start_loc.device)
        safe_req_indices = torch.clamp(req_indices, max=max(num_reqs - 1, 0))
        end_indices = torch.clamp(query_start_loc[safe_req_indices + 1] - 1,
                                  min=0)
        last_positions = position_ids[end_indices].unsqueeze(1) + 1
        valid_mask = (req_indices < num_reqs).unsqueeze(1).to(torch.int32)
        base_pos = (last_positions * valid_mask).to(torch.int32)

        # Format input_ids with base tokens and MASK placeholders
        input_ids = torch.full((padded_len, ),
                               mask_token_id,
                               dtype=torch.int32,
                               device=base_tokens.device)
        scatter_indices = torch.arange(0,
                                       padded_num_reqs * block_size,
                                       block_size,
                                       dtype=torch.int32,
                                       device=base_tokens.device)
        input_ids.scatter_(0, scatter_indices, base_tokens)

        offsets = torch.arange(
            0,
            block_size,
            dtype=torch.int32,
            device=base_tokens.device).unsqueeze(0)
        positions_unpadded = (base_pos + offsets).flatten()
        positions = F.pad(positions_unpadded,
                          (0, padded_len - positions_unpadded.shape[0]),
                          value=0)
        # Clamp to max_model_len to prevent out-of-bounds RoPE and kernel indexing.
        positions = torch.clamp(positions, max=self._max_model_len - 1)
        seq_lens = torch.clamp(base_pos.squeeze(1) + block_size,
                               max=self._max_model_len)

        return input_ids, positions, seq_lens
