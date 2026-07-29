# SPDX-License-Identifier: Apache-2.0
"""Skeleton Kimi-Linear model for TPU — step 1: dummy logits only.

**This computes nothing.** It exists to prove one thing end to end: that a model
definition living in ``vllm_torchtpu`` is the class vLLM instantiates and drives,
through construction, weight loading, forward and sampling. Every output is
zeros, so generated tokens are meaningless.

Real layers get added incrementally on top of this. See ``KIMI_K3_TPU_GAPS.md``
for the gap inventory from the full-port spike, and
``artifacts/qwen35-gdn-debug/kimi_k3_full_port/`` for that spike's code.

Interface implemented (``vllm/model_executor/models/interfaces_base.py``):

* ``__init__(vllm_config, prefix)``
* ``embed_input_ids(input_ids)``
* ``forward(input_ids, positions, ...)`` -> hidden states
* ``compute_logits(hidden_states)`` -> logits
* ``load_weights(weights)`` -> names actually loaded

``is_attention_free = True`` (the ``IsAttentionFree`` marker at
``interfaces.py:763``) declares that this model has no attention layers, so vLLM
does not ask it for a KV cache spec. That is correct while there are no layers at
all; it must be removed once real attention or KDA layers are added.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class KimiLinearForCausalLM(nn.Module):
    """Dummy-logits skeleton. Produces zeros of the correct shape."""

    # No attention/KDA layers yet -> no KV cache. Remove when layers land.
    is_attention_free = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        # Kimi-K3 nests the text config; a standalone Kimi-Linear config does not.
        text_config = getattr(config, "text_config", config)

        self.config = text_config
        self.vocab_size: int = text_config.vocab_size
        self.hidden_size: int = text_config.hidden_size
        self.dtype: torch.dtype = vllm_config.model_config.dtype

        # One real parameter, so the module has a device/dtype, `.to()` works,
        # and the dummy weight loader has something to initialise.
        self._placeholder = nn.Parameter(torch.zeros(1, dtype=self.dtype),
                                        requires_grad=False)

        logger.warning_once(
            "Kimi TPU: using the SKELETON model (%s) — all outputs are zeros, "
            "generated text is meaningless. vocab_size=%d hidden_size=%d",
            type(self).__name__, self.vocab_size, self.hidden_size)

    # -- VllmModel ---------------------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (input_ids.shape[0], self.hidden_size),
            dtype=self.dtype,
            device=self._placeholder.device,
        )

    # `inputs_embeds`/`intermediate_tensors` are accepted because the runner
    # passes them positionally or by keyword depending on the path taken.
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            num_tokens = inputs_embeds.shape[0]
        else:
            num_tokens = input_ids.shape[0]
        return torch.zeros(
            (num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self._placeholder.device,
        )

    # -- VllmModelForTextGeneration ----------------------------------------
    def compute_logits(self, hidden_states: torch.Tensor,
                       *args) -> torch.Tensor:
        # *args absorbs a sampling_metadata argument on vLLM versions that still
        # pass one; the current protocol is (hidden_states,) only.
        return torch.zeros(
            (hidden_states.shape[0], self.vocab_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )

    # -- weight loading ----------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Consume and discard checkpoint weights.

        Returns an empty set: nothing in the checkpoint maps onto this skeleton.
        The iterator is still drained so the loader does not report unconsumed
        tensors, and so a real checkpoint can be pointed at this model without
        erroring.
        """
        count = 0
        for _name, _tensor in weights:
            count += 1
        if count:
            logger.warning_once(
                "Kimi TPU skeleton: discarded %d checkpoint tensors (this model "
                "has no real weights yet).", count)
        return set()
