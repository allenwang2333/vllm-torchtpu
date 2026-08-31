# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import dataclass
from typing import Any

import torch
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

INVALID_TOKEN_ID = -1
_D2H_STRIDE = int(os.getenv("VLLM_TPU_D2H_STRIDE", "4"))
_CURRENT_D2H_STEP = 0
_MOCK_ASYNC_OUTPUT = bool(int(os.getenv("VLLM_TPU_MOCK_ASYNC_OUTPUT", "0")))
_HAS_ASYNC_HOST_COPY = (
    hasattr(torch, "tpu")
    and hasattr(torch.tpu, "_start_async_host_copy")
    and hasattr(torch.tpu, "_wait_async_host_copy")
)


# Global pipeline tracking to preserve source tensor lifetime across steps
_PREV_COPY_STATE: Any = None


@dataclass
class AsyncTPUCopyState:
    """Async D2H copy state for sampled tokens.

    Per-chunk device tensors are kept at their bucketed (padded) shapes so
    each D2H copy hits a precompiled program. Host-side trim and concat
    happen lazily on `wait()` and produce a single contiguous CPU tensor
    matching the real (unpadded) request count.

    Combines:
    - Option 3: Strict tracked async copy with source tensor lifetime preservation
      across steps (preventing PJRT premature deallocation/recycle stalls).
    - Option 1: Unblocked async wait (skipping full synchronous stalls) controlled
      by VLLM_TPU_SKIP_COPY_SYNC.
    - Option B: Batched D2H copy every N steps (default 4, via VLLM_TPU_D2H_STRIDE).
      Reduces host interaction and D2H synchronization frequency by 4x.
    """
    chunks_cpu: list[torch.Tensor]
    chunk_real_lens: list[int]
    copy_ready_event: Any = None
    copy_ids: list[int] | None = None
    chunks_tpu: list[torch.Tensor] | None = None
    completed: bool = False
    _sampled_token_ids_cpu: torch.Tensor | None = None

    @classmethod
    def from_device_chunks(cls, chunks_tpu: list[torch.Tensor],
                           chunk_real_lens: list[int]) -> "AsyncTPUCopyState":
        global _CURRENT_D2H_STEP, _PREV_COPY_STATE
        assert len(chunks_tpu) == len(chunk_real_lens) and len(chunks_tpu) > 0

        _CURRENT_D2H_STEP += 1
        is_flush_step = (_D2H_STRIDE <= 1 or _CURRENT_D2H_STEP % _D2H_STRIDE == 0)

        # In Option B, intermediate steps do not launch D2H copies to device.
        # This eliminates 3 out of 4 host transfers while tokens stay on-device.
        if not is_flush_step:
            return cls(
                chunks_cpu=[],
                chunk_real_lens=chunk_real_lens,
                copy_ready_event=None,
                copy_ids=None,
                chunks_tpu=None,
                completed=True,
            )

        # Retire prior step's tensor retention now that Step N+1 has started
        if _PREV_COPY_STATE is not None:
            _PREV_COPY_STATE.chunks_tpu = None
            _PREV_COPY_STATE = None

        # Option 3: Strictly enforce tracked async host copy to isolate DMA completion
        if _HAS_ASYNC_HOST_COPY and not _MOCK_ASYNC_OUTPUT:
            chunks_cpu = []
            for t in chunks_tpu:
                c = torch.empty(t.shape, dtype=t.dtype, device="cpu")
                try:
                    c = c.pin_memory()
                except RuntimeError:
                    pass
                chunks_cpu.append(c)

            copy_ids = [
                torch.tpu._start_async_host_copy(t, c)  # type: ignore[attr-defined]
                for t, c in zip(chunks_tpu, chunks_cpu)
            ]
            state = cls(
                chunks_cpu=chunks_cpu,
                chunk_real_lens=chunk_real_lens,
                copy_ready_event=None,
                copy_ids=copy_ids,
                chunks_tpu=chunks_tpu,  # Keep alive across DMA transfer
            )
            _PREV_COPY_STATE = state
            return state

        chunks_cpu = [t.to("cpu", non_blocking=True) for t in chunks_tpu]
        copy_ready_event = torch.tpu.Event()
        copy_ready_event.record()
        state = cls(
            chunks_cpu=chunks_cpu,
            chunk_real_lens=chunk_real_lens,
            copy_ready_event=copy_ready_event,
            copy_ids=None,
            chunks_tpu=chunks_tpu,
        )
        _PREV_COPY_STATE = state
        return state

    def wait(self) -> None:
        if self.completed:
            return
        # Option 1: Skip synchronous device barrier stalls.
        # Tracked copy completes in hardware DMA; source tensor lifetime is covered
        # by _PREV_COPY_STATE. Set VLLM_TPU_SKIP_COPY_SYNC=0 to force synchronous wait.
        skip_sync = bool(int(os.getenv("VLLM_TPU_SKIP_COPY_SYNC", "1")))
        if not skip_sync:
            if self.copy_ids is not None:
                for cid in self.copy_ids:
                    if cid != 0:
                        torch.tpu._wait_async_host_copy(cid)  # type: ignore[attr-defined]
            elif not _MOCK_ASYNC_OUTPUT and self.copy_ready_event is not None:
                self.copy_ready_event.synchronize()
        self.completed = True

    @property
    def sampled_token_ids_cpu(self) -> torch.Tensor:
        """Return the unpadded concatenated CPU tensor.

        Performed lazily after `wait()`; trims each bucketed chunk to its real
        length on the host (no device recompile).
        """
        if self._sampled_token_ids_cpu is None:
            if not self.chunks_cpu or _MOCK_ASYNC_OUTPUT:
                total_len = sum(self.chunk_real_lens)
                self._sampled_token_ids_cpu = torch.full(
                    (total_len, 1), 100, dtype=torch.int32, device="cpu"
                )
            else:
                trimmed = [
                    t[:n] for t, n in zip(self.chunks_cpu, self.chunk_real_lens)
                ]
                if len(trimmed) == 1:
                    self._sampled_token_ids_cpu = trimmed[0]
                else:
                    self._sampled_token_ids_cpu = torch.cat(trimmed, dim=0)
        return self._sampled_token_ids_cpu


@dataclass
class AsyncPreResults:
    req_ids: list[str]
    next_tokens_tpu: torch.Tensor
    request_seq_lens: list[tuple[int, Any, int, str]]
    discard_sampled_tokens_req_indices: list[int]
    req_id_to_index_copy: dict[str, int]
    copy_state: AsyncTPUCopyState | None = None
    # Spec decode only: per-request rejected-draft count from this step, kept
    # on-device so the next step can correct its optimistically-advanced
    # positions/seq_lens. None for non-spec.
    spec_decode_num_rejected_tokens: torch.Tensor | None = None
    # Spec decode only: per-request draft count proposed this step, keyed by the
    # request_seq_lens index. _modify_prev_results uses it to size the
    # (1 + num_draft) optimistic-placeholder rollback. None for non-spec.
    num_draft_per_req: dict[int, int] | None = None

    def wait_for_copy(self) -> torch.Tensor | None:
        if self.copy_state is None:
            return None
        self.copy_state.wait()
        return self.copy_state.sampled_token_ids_cpu


class AsyncTPUModelRunnerOutput(AsyncModelRunnerOutput):
    """Holds asynchronous model output specifically from a TPU runner."""

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        copy_state: AsyncTPUCopyState | None,
        discard_sampled_tokens_req_indices: list[int],
    ):
        self._model_runner_output: ModelRunnerOutput = model_runner_output
        self.copy_state = copy_state
        self.discard_sampled_tokens_req_indices = discard_sampled_tokens_req_indices

    def get_output(self) -> ModelRunnerOutput:
        if self.copy_state is None:
            self._model_runner_output.sampled_token_ids = []
            return self._model_runner_output

        self.copy_state.wait()
        selected_token_ids = self.copy_state.sampled_token_ids_cpu
        if selected_token_ids.numel() == 0:
            self._model_runner_output.sampled_token_ids = []
            return self._model_runner_output
        if selected_token_ids.dim() == 1:
            selected_token_ids = selected_token_ids.unsqueeze(-1)

        max_gen_len = selected_token_ids.shape[-1]
        if max_gen_len == 1:
            valid_sampled_token_ids = selected_token_ids.tolist()
            for i in self.discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()
        else:
            valid_mask = selected_token_ids != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            valid_sampled_token_ids = [
                seq.tolist()
                for seq in selected_token_ids[valid_mask].split(gen_lens)
            ]
            for i in self.discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()

        self._model_runner_output.sampled_token_ids = valid_sampled_token_ids
        return self._model_runner_output


def assemble_spec_next_tokens(next_tokens: torch.Tensor, drafts: torch.Tensor,
                              num_reqs: int) -> torch.Tensor:
    """Async substitution source for speculative decoding.

    Per request the next verify step consumes ``[bonus, draft_1, ..., draft_K]``:
    the bonus is this step's last verified token (the continuation), the drafts
    are this step's proposal. Returns them packed row-major and flattened to
    ``[num_reqs * (1 + K)]`` — the layout the substitution index builder
    addresses as ``req_idx * (1 + K) + j`` (bonus at ``j == 0``).

    Args:
        next_tokens: rejection-sampler output ``[>=num_reqs, K+1]`` (accepted
            prefix + bonus, padded with INVALID_TOKEN_ID); the bonus is the last
            non-padding column of each row.
        drafts: device proposer output ``[>=num_reqs, K]``.
        num_reqs: real request count (both inputs may be bucket-padded in dim 0).
    """
    nt = next_tokens[:num_reqs]
    drafts = drafts[:num_reqs]
    # Last non-padding column per row is the bonus.
    last_valid_col = (nt != INVALID_TOKEN_ID).sum(dim=1).clamp(min=1) - 1
    bonus = nt.gather(1, last_valid_col.unsqueeze(1))  # [num_reqs, 1]
    src = torch.cat([bonus.to(drafts.dtype), drafts], dim=1)  # [num_reqs, 1+K]
    return src.reshape(-1)  # [num_reqs * (1 + K)]


def subtract_num_rejected_tokens(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    num_rejected: torch.Tensor,
    seq_lens_subtract_indices: torch.Tensor,
    positions_subtract_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove the previous async step's per-request rejected-token over-count.

    In async spec decode the host advances ``num_computed_tokens`` optimistically, assuming *every* speculative
    draft was accepted. The real acceptance is only known on-device, so this
    subtracts the actual per-request rejection count here — on-device, before the
    verify forward.

    Args:
        seq_lens: ``[num_reqs]`` optimistic sequence lengths.
        positions: ``[total_tokens]`` (or ``[3, total_tokens]`` mrope) optimistic positions.
        num_rejected: ``[num_placeholder_reqs]`` rejection count per source index.
        seq_lens_subtract_indices: ``[num_reqs]`` index into ``num_rejected`` for
            each seq_lens slot, ``-1`` to leave it unchanged.
        positions_subtract_indices: ``[total_tokens]`` index into ``num_rejected``
            for each position slot, ``-1`` to leave it unchanged.
    """
    zero = num_rejected.new_zeros(())
    seq_subtract = torch.where(
        seq_lens_subtract_indices >= 0,
        num_rejected[seq_lens_subtract_indices.clamp(min=0)], zero)
    seq_lens = seq_lens - seq_subtract

    pos_subtract = torch.where(
        positions_subtract_indices >= 0,
        num_rejected[positions_subtract_indices.clamp(min=0)], zero)
    if positions.ndim == 2:  # mrope (out of text-MVP scope; kept for parity)
        pos_subtract = pos_subtract.unsqueeze(0)
    positions = positions - pos_subtract
    return seq_lens, positions


def extract_draft_token_ids(
        input_ids: torch.Tensor, logits_indices: torch.Tensor,
        target_logits_indices: torch.Tensor) -> torch.Tensor:
    """Gather the draft tokens the target just verified, from the (post-async-
    substitution) device ``input_ids``.
    """
    return input_ids[logits_indices][target_logits_indices + 1]


def compute_num_rejected(next_tokens: torch.Tensor,
                         num_draft: torch.Tensor) -> torch.Tensor:
    """Per-request rejected-draft count from the rejection output, on-device.

    Args:
        next_tokens: rejection output ``[>=num_reqs, K+1]`` padded with
            INVALID_TOKEN_ID.
        num_draft: ``[num_reqs]`` scheduled draft count per request.
    """
    num_reqs = num_draft.shape[0]
    num_valid = (next_tokens[:num_reqs] != INVALID_TOKEN_ID).sum(dim=1)
    return (num_draft - num_valid + 1).clamp(min=0).to(torch.int32)
