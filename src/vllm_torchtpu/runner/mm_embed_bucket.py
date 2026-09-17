# SPDX-License-Identifier: Apache-2.0
"""One bucket ladder for per-item vision embeddings, for any model.

vLLM's default ``postprocess_encoder_output`` cuts the encoder's packed output
into exact-length per-item slices (``scatter_output_slices``). That packed
output is the one static thing in the whole vision path -- it is the encoder
cudagraph's budget buffer -- and slicing it to the image's own token count
throws that away: every later op keyed on those rows recompiles once per image
resolution, and no amount of padding afterwards puts it back, because the pad
itself reads a tensor whose height is the token count.

Gathering instead of slicing keeps it. ``index_select`` with a host-built index
of bucket length reads from a static source into a static destination, and the
image size arrives as the *values* of the index rather than as a shape. Rows
past the item's real length repeat its last row; every consumer bounds its read
by the true token count, which it knows host-side from ``mm_position.length``.

Kimi K3 has carried its own copy of this as ``mm_embed_bucket``. Nothing in it
is model-specific, so it is applied here to every model that inherits vLLM's
default, and models with their own ``postprocess_encoder_output`` (Step3-VL,
DeepSeek-OCR, Kimi) are left alone.
"""

from typing import Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Below this, buckets cost more in wasted rows than they save in programs; an
# item of 3 tokens and one of 60 share a program either way.
MM_EMBED_BUCKET_MIN = 64


def mm_embed_bucket(num_tokens: int) -> int:
    """Padded row count an item of ``num_tokens`` is kept at."""
    bucket = MM_EMBED_BUCKET_MIN
    while bucket < num_tokens:
        bucket *= 2
    return bucket


def gather_output_buckets(
    output: torch.Tensor,
    indices: list[int],
    per_item_out_tokens: list[int],
    dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
) -> None:
    """``scatter_output_slices``, but every item comes back bucket-height.

    The gather is keyed on ``(packed rows, bucket)``, both of which come from
    a finite set, rather than on this item's token count.
    """
    last_row = output.shape[0] - 1
    offset = 0
    for idx in indices:
        num_tokens = per_item_out_tokens[idx]
        padded = mm_embed_bucket(num_tokens)
        src_idx = torch.arange(offset, offset + padded, dtype=torch.long)
        # A bucket can overhang the packed output's last item; those rows are
        # padding either way, so clamping them is enough to stay in bounds.
        src_idx.clamp_(max=last_row)
        dest[idx] = output.index_select(0, src_idx.to(output.device))
        offset += num_tokens


def _patch_encoder_postprocess_to_buckets() -> None:
    """Install the bucketed gather as vLLM's default postprocess."""
    if not envs.TPU_MM_BUCKET_SCATTER:
        return

    from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph

    if getattr(SupportsEncoderCudaGraph.postprocess_encoder_output,
               "_tpu_bucketed", False):
        return

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # `clone` is moot: index_select already returns fresh storage, which
        # is what the caller wants the flag for.
        del clone, batch_mm_kwargs
        gather_output_buckets(outputs["default"], indices,
                              per_item_out_tokens, dest)

    postprocess_encoder_output._tpu_bucketed = True
    SupportsEncoderCudaGraph.postprocess_encoder_output = (
        postprocess_encoder_output)
    logger.info("Applied TPU patch: bucketed encoder output postprocess "
                "(floor %d).", MM_EMBED_BUCKET_MIN)
