from typing import Any, Dict, List

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors, tp_all_gather_materialized

logger = init_logger(__name__)


class MMEncoderManager(EncoderCudaGraphManager):
    """Per-budget PyTorch XLA-cache manager for the vision encoder forward.

    Vision encoders (e.g. ViTs in Qwen2.5-VL / Qwen3-VL) produce variable
    token counts based on dynamic image resolutions and aspect ratios.
    On TPU, dynamic tensor shapes trigger expensive XLA JIT recompilations
    on every distinct image size. While GPU vLLM uses CUDA Graphs with fixed
    replay buffers to avoid launch overhead, TPU requires static XLA
    compilation. ``MMEncoderManager`` solves this by mapping dynamic
    multimodal inputs into a discrete set of static token budgets
    (e.g., 256, 512, 1024, 2048), zero-padding inputs to match pre-compiled templates.

    How it is used:
        1. Initialization: Created by ``maybe_create_mm_encoder_manager`` during
           runner setup. It pre-allocates dummy CPU input templates for each token
           budget and compiles the model's ``encoder_cudagraph_forward`` via
           ``torch.compile(backend="tpu", fullgraph=True, dynamic=False)``.
        2. Warmup: During ``capture_model()``, ``precompile_vision_encoder()``
           runs a dummy forward pass for every budget in ``self.token_budgets``
           to prime the XLA binary cache before serving traffic begins.
        3. Runtime: During request execution, incoming multimodal items are
           greedily packed into the smallest fitting token budget. Dynamic inputs
           are zero-padded to the static budget template via ``_pad_to_template``,
           executed through the compiled XLA graph via ``_run_budget_graph``,
           and the resulting embeddings are cached for the decoder.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        vllm_model: torch.nn.Module,
    ) -> None:
        """Initialize the MMEncoderManager with budget templates and XLA cache.

        Args:
            vllm_config: The global vLLM configuration.
            device: Target torch device (TPU) where operations run.
            vllm_model: The underlying multimodal model module.
        """
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            dtype=vllm_config.model_config.dtype,
            model=vllm_model,
        )

        # Create dummy input templates for every budget size and path directly
        # from self.path_token_budgets prepared by EncoderCudaGraphManager.
        self.budget_templates: Dict[str, Dict[int, Dict[
            str, torch.Tensor]]] = {
                path: {
                    budget:
                    self.model.prepare_encoder_cudagraph_capture_inputs(
                        budget,
                        self.max_batch_size,
                        self.max_frames_per_batch,
                        self.device,
                        self.dtype,
                        path=path).values
                    for budget in budgets if budget > 0
                }
                for path, budgets in self.path_token_budgets.items()
            }

        # We compile the forward pass function using the TPU backend.
        # This acts as our XLA cache. We compile it once, and it will generate
        # separate XLA graphs for each unique budget size passed to it.
        self._compiled_budget_forward = torch.compile(
            self.model.encoder_cudagraph_forward,
            backend="tpu",
            fullgraph=True,
            dynamic=False)

        logger.info(
            f"[mm_encoder_manager] Initialized XLA path_budgets={self.path_token_budgets} "
            f"max_batch_size={self.max_batch_size}")

    def _pad_to_template(
        self,
        replay_values: Dict[str, torch.Tensor],
        budget: int,
        path: str = "default",
    ) -> Dict[str, torch.Tensor]:
        """Zero-pads dynamic multimodal inputs up to the static budget template."""
        template = self.budget_templates[path][budget]
        padded: Dict[str, torch.Tensor] = {}

        for key, tmpl in template.items():
            src = replay_values.get(key)
            if src is None:
                padded[key] = tmpl
                continue

            if not hasattr(src, "shape") or src.ndim == 0:
                # Only scalar values, copy directly to template
                tmpl.copy_(src.to(device=tmpl.device))
                padded[key] = tmpl
                continue

            if src.shape == tmpl.shape:
                # Already perfectly bucket-sized
                padded[key] = src.to(device=tmpl.device)
                continue

            padding_logic = self.config.padding_logics.get(
                key, self._copy_padded_buffer)
            padding_logic(tmpl, src)
            padded[key] = tmpl

        return padded

    @torch.no_grad()
    def _capture_budget_graph(self,
                              token_budget: int,
                              path: str = "default") -> None:
        """Primes the XLA cache for a specific budget bucket."""
        template = self.budget_templates[path][token_budget]

        # Pass a copy of template so models calling values.pop() don't mutate the stored template.
        try:
            out = self._compiled_budget_forward(dict(template), path=path)
            synchronize_tensors(out, wait=True)
            self._get_graph_set(path)[token_budget] = template
        except Exception as e:
            logger.warning(
                "[mm_encoder_manager] Failed to precompile vision encoder "
                "for budget=%d, path='%s': %s", token_budget, path, e)

    @torch.no_grad()
    def _run_budget_graph(
        self,
        mm_kwargs: Dict[str, Any],
        token_budget: int,
        path: str = "default",
    ) -> torch.Tensor | None:
        """Pads actual inputs and runs the compiled XLA graph."""
        num_items = len(self._get_item_specs(mm_kwargs))
        if (path not in self.budget_templates
                or token_budget not in self.budget_templates[path]):
            self.graph_misses += num_items
            return None

        # Prepare dynamic inputs
        replay = self.model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs,
            self.max_batch_size,
            self.max_frames_per_batch,
            path=path)

        # Pad dynamic inputs to strictly match the static budget shape
        padded = self._pad_to_template(replay.values, token_budget, path=path)

        # Run encoder with eager fallback on compilation/execution failure
        try:
            out_tensor = self._compiled_budget_forward(dict(padded), path=path)
            self.graph_hits += num_items
            return out_tensor
        except Exception as e:
            logger.warning(
                "[mm_encoder_manager] Compiled vision encoder forward failed "
                "for budget=%d, path='%s': %s. Falling back to "
                "encoder_eager_forward.", token_budget, path, e)
            self.graph_misses += num_items
            with torch.inference_mode():
                return self.model.encoder_eager_forward(mm_kwargs, path=path)

    def precompile_vision_encoder(self) -> None:
        """Prime the XLA compilation cache for all configured token budgets."""
        for path, templates in self.budget_templates.items():
            for budget in templates:
                self._capture_budget_graph(budget, path=path)

    # -- encoder data parallelism (--mm-encoder-tp-mode data) --

    def _mm_embed_bucket(self, num_tokens: int) -> int:
        """Padded row count the model keeps a ``num_tokens`` item at."""
        bucket_fn = getattr(self.model, "mm_embed_bucket", None)
        if bucket_fn is None or not envs.TPU_MM_BUCKET_SCATTER:
            return num_tokens
        return bucket_fn(num_tokens)

    def _dp_gather(
        self,
        local_outputs: List[torch.Tensor],
        per_item_out_tokens: List[int],
        image_rank_assignment: List[int],
        images_per_rank: List[int],
        max_output_tokens_per_rank: int,
    ) -> List[torch.Tensor]:
        """Collect every rank's encoder outputs and restore the batch order.

        Replaces the base implementation for two TPU-specific reasons.

        The collective: the base version all-gathers a freshly built lazy
        tensor, which drags each rank's local encoder graph into the
        collective's program. Ranks that ran different image shapes then
        disagree on what to execute and the device halts. Everything here goes
        through ``tp_all_gather_materialized`` instead.

        The layout: ``postprocess_encoder_output`` pads each item up to a
        bucket, so a rank's outputs are longer than ``per_item_out_tokens``
        says and the base version's slice offsets land mid-item. Offsets are
        recomputed from the padded lengths, which every rank can derive from
        the global ``per_item_out_tokens`` without communicating. The
        extraction is index-based rather than sliced so the programs stay
        keyed on the buckets rather than on this batch's offsets.
        """
        # The base version's send length assumes unpadded items; recomputed
        # below from the padded lengths instead.
        del max_output_tokens_per_rank

        hidden_size = self.config.out_hidden_size
        padded_lens = [self._mm_embed_bucket(t) for t in per_item_out_tokens]

        # Original item indices handled by each rank, in that rank's order --
        # the same order _dp_shard fed to _execute_local.
        items_by_rank: List[List[int]] = []
        cursor = 0
        for count in images_per_rank:
            items_by_rank.append(image_rank_assignment[cursor:cursor + count])
            cursor += count

        # One send length for every rank. Bucketing it keeps the collective's
        # shape off the critical path of "new image size -> new program".
        send_len = self._mm_embed_bucket(
            max((sum(padded_lens[i] for i in items)
                 for items in items_by_rank),
                default=0))

        local_items = items_by_rank[self._dp_rank()]
        send = torch.zeros((send_len, hidden_size),
                           dtype=self.dtype,
                           device=self.device)
        offset = 0
        for item_idx, out in zip(local_items, local_outputs):
            rows = padded_lens[item_idx]
            # Every rank derives this layout from the global token counts, so
            # a local output of the wrong height would silently shift all the
            # items behind it -- and the receiving ranks could not tell.
            if out.shape[0] != rows:
                raise RuntimeError(
                    f"encoder output for item {item_idx} has {out.shape[0]} "
                    f"rows, expected {rows}; the model's postprocess_encoder_"
                    f"output must pad to mm_embed_bucket for DP gather to "
                    f"agree across ranks")
            # Host-built indices: the copy is keyed on the bucket, and the
            # varying offset arrives as data instead of as a slice bound.
            dst = torch.arange(offset, offset + rows, dtype=torch.long)
            send.index_copy_(0, dst.to(self.device), out.to(self.dtype))
            offset += rows

        gathered = tp_all_gather_materialized(send, dim=0)

        last_row = gathered.shape[0] - 1
        result: List[torch.Tensor | None] = [None] * len(per_item_out_tokens)
        for rank, items in enumerate(items_by_rank):
            offset = rank * send_len
            for item_idx in items:
                rows = padded_lens[item_idx]
                src = torch.arange(offset, offset + rows, dtype=torch.long)
                src.clamp_(max=last_row)
                result[item_idx] = gathered.index_select(
                    0, src.to(self.device))
                offset += rows

        return [t for t in result if t is not None]

    @staticmethod
    def _dp_rank() -> int:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()


def maybe_create_mm_encoder_manager(
    vllm_config: VllmConfig,
    device: torch.device,
    vllm_model: torch.nn.Module,
) -> MMEncoderManager | None:
    """Create an MMEncoderManager instance if enabled and supported.

    Args:
        vllm_config: The global vLLM configuration object.
        device: Target torch device (TPU) for model execution.
        vllm_model: The underlying multimodal model module.

    Returns:
        An initialized ``MMEncoderManager`` if encoder compilation is enabled
        (``cudagraph_mm_encoder=True``) and supported by the model (implements
        ``SupportsEncoderCudaGraph``). Returns ``None`` otherwise, in which case
        the runner falls back to the eager multimodal path
        (``model.embed_multimodal(**mm_kwargs)``) without budget padding.
    """
    if not vllm_config.compilation_config.cudagraph_mm_encoder:
        return None
    if not supports_encoder_cudagraph(vllm_model):
        return None
    return MMEncoderManager(vllm_config, device, vllm_model)
