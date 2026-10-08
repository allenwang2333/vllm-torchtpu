"""Host copies of a step's Engram hash inputs, for Host-DRAM Engram tables.

With ``TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD=1`` the Engram rows are gathered on
the host, so the n-gram hashes must reach the host every step. Hashing on the
device and copying back (``gathered_hashes.to("cpu")``) blocks on the previous
step and stops async scheduling from overlapping host work with the device.
The runner already holds every hash input on the host before uploading it, so
V4.1 publishes those host copies for the model instead.

Like ``async_fence.py``, this wraps the runner's lookback upload rather than
editing the shared runner: ``forward_model`` calls
``_prepare_lookback_token_ids`` inside the step's forward context, after
``input_ids_cpu``, ``positions_cpu`` and ``query_start_loc_cpu`` are final.
"""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import repeat
from typing import Any

import numpy as np
import torch
from vllm.distributed import get_engram_dp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.models.deepseek_v41.common.engram import DEAD_ID
from vllm.models.deepseek_v41.nvidia.engram import engram_gathered_num_tokens

from vllm_torchtpu import envs

HOST_ENGRAM_KEY = "deepseek_v41_host_engram"


@dataclass(frozen=True)
class HostEngramInputs:
    """Host views matching the device tensors of one forward step.

    ``input_ids``/``positions`` cover the padded token count from the start
    of the runner's host buffers (full-batch layout), ``query_start_loc`` the
    whole padded request table and ``lookback_token_ids`` the whole lookback
    buffer, so callers slice them to the device tensors' static shapes.
    ``all_prompt`` is True when every scheduled token of every request is a
    prompt token. ``host_ready`` indicates whether decode/mixed token and
    lookback inputs have been resolved on the host for ``hash_on_host``.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    lookback_token_ids: torch.Tensor
    all_prompt: bool
    host_ready: bool | None = None

    @property
    def can_hash_on_host(self) -> bool:
        return self.all_prompt if self.host_ready is None else self.host_ready


def _async_source_tokens_cpu(
    pre_async: Any, *, prefer_next_tokens_tpu: bool = False
) -> torch.Tensor | None:
    """Resolve the previous step's sampled tokens on CPU for async substitution."""
    if prefer_next_tokens_tpu:
        next_tpu_spec: Any = getattr(pre_async, "next_tokens_tpu", None)
        if isinstance(next_tpu_spec, torch.Tensor):
            return next_tpu_spec.to("cpu").reshape(-1)
    wait_for_copy = getattr(pre_async, "wait_for_copy", None)
    waited: Any = wait_for_copy() if callable(wait_for_copy) else None
    copy_state = getattr(pre_async, "copy_state", None)
    chunks_cpu = getattr(copy_state, "chunks_cpu", None)
    if chunks_cpu:
        flat_chunks = [c.reshape(-1) for c in chunks_cpu]
        return (
            flat_chunks[0] if len(flat_chunks) == 1 else torch.cat(flat_chunks, dim=0)
        )
    if isinstance(waited, torch.Tensor):
        return waited.reshape(-1)
    next_tpu: Any = getattr(pre_async, "next_tokens_tpu", None)
    if isinstance(next_tpu, torch.Tensor):
        return next_tpu.to("cpu").reshape(-1)
    return None


@dataclass(frozen=True)
class _AsyncSpecState:
    """Previous-step speculative results, indexed by the previous step's
    request position (``pre_idx``), as flat NumPy arrays.

    ``tokens[pre_idx, :lengths[pre_idx]]`` are the accepted tokens of that
    request in order (``row[row != -1]`` of the sampled copy, empty for
    discarded requests), ``rejected`` the draft count rolled back for it
    (``compute_num_rejected``) and ``num_draft`` its scheduled drafts.
    """

    stride: int
    tokens: np.ndarray
    lengths: np.ndarray
    rejected: np.ndarray
    num_draft: np.ndarray


def _resolve_async_spec_state(runner: Any, pre_async: Any) -> _AsyncSpecState | None:
    """Resolve async speculative stride, rejection counts, and committed tokens."""
    spec_config = getattr(runner, "speculative_config", None)
    num_spec = getattr(spec_config, "num_speculative_tokens", None)
    req_id_to_index = getattr(pre_async, "req_id_to_index_copy", None)
    if not isinstance(num_spec, int) or num_spec <= 0 or not req_id_to_index:
        return None
    wait_for_copy = getattr(pre_async, "wait_for_copy", None)
    sampled_cpu: Any = wait_for_copy() if callable(wait_for_copy) else None
    if not isinstance(sampled_cpu, torch.Tensor):
        return None
    sampled = sampled_cpu.numpy()
    if sampled.ndim == 1:
        sampled = sampled[:, None]
    num_rows = int(sampled.shape[0])
    pre_positions = np.fromiter(
        req_id_to_index.values(), dtype=np.int64, count=len(req_id_to_index)
    )
    if int(pre_positions.min()) < 0 or int(pre_positions.max()) >= num_rows:
        return None
    spec_num_rejected = getattr(pre_async, "spec_decode_num_rejected_tokens", None)
    stride = 1 + num_spec if spec_num_rejected is not None else 1
    valid = sampled != -1
    lengths = valid.sum(axis=1, dtype=np.int64)
    # Compact each row's valid tokens to its front, preserving their order.
    if valid.shape[1] <= 1 or bool(np.all(valid[:, :-1] >= valid[:, 1:])):
        tokens = sampled
    else:
        order = np.argsort(~valid, axis=1, kind="stable")
        tokens = np.take_along_axis(sampled, order, axis=1)
    num_draft = np.zeros(num_rows, dtype=np.int64)
    num_draft_per_req = getattr(pre_async, "num_draft_per_req", None) or {}
    if num_draft_per_req:
        draft_pos = np.fromiter(
            num_draft_per_req.keys(), dtype=np.int64, count=len(num_draft_per_req)
        )
        draft_cnt = np.fromiter(
            num_draft_per_req.values(), dtype=np.int64, count=len(num_draft_per_req)
        )
        in_range = (draft_pos >= 0) & (draft_pos < num_rows)
        num_draft[draft_pos[in_range]] = draft_cnt[in_range]
    if spec_num_rejected is not None:
        rejected = np.where(num_draft > 0, np.maximum(num_draft - lengths + 1, 0), 0)
    else:
        rejected = np.zeros(num_rows, dtype=np.int64)
    discard = getattr(pre_async, "discard_sampled_tokens_req_indices", None) or ()
    if discard:
        discard_idx = np.fromiter((int(i) for i in discard), dtype=np.int64)
        discard_idx = discard_idx[(discard_idx >= 0) & (discard_idx < num_rows)]
        if discard_idx.size:
            lengths = lengths.copy()
            lengths[discard_idx] = 0
    return _AsyncSpecState(stride, tokens, lengths, rejected, num_draft)


@dataclass(frozen=True)
class _SampledOverlay:
    """Tokens committed in the previous step, per local request row.

    ``tokens[i, :lengths[i]]`` were written at positions ``starts[i] ..
    starts[i] + lengths[i]`` of request ``i`` after ``token_ids_cpu`` was
    snapshotted, so lookback positions in that range read from here.
    """

    tokens: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray


def _acquire_host_scratch(
    runner: Any, attr: str, template: torch.Tensor, active_len: int
) -> torch.Tensor:
    """Return a reusable host scratch tensor synced over ``[:active_len]``."""
    scratch: Any = getattr(runner, attr, None)
    dirty_attr = f"{attr}_dirty"
    if (
        not isinstance(scratch, torch.Tensor)
        or scratch.shape != template.shape
        or scratch.dtype != template.dtype
    ):
        scratch = template.clone()
        setattr(runner, attr, scratch)
        setattr(runner, dirty_attr, active_len)
        return scratch
    sync_len = min(
        int(template.shape[0]),
        max(active_len, int(getattr(runner, dirty_attr, int(template.shape[0])))),
    )
    scratch.numpy()[:sync_len] = template.numpy()[:sync_len]
    setattr(runner, dirty_attr, active_len)
    return scratch


def _resolve_decode_lookback_cpu(
    runner,
    batch,
    num_reqs: int,
    start_index: int,
    *,
    rejected_by_req: np.ndarray | None = None,
    sampled_overlay: _SampledOverlay | None = None,
) -> torch.Tensor:
    """Fill lookback_token_ids on CPU for both prompt and generated positions."""
    src_lookback = runner.lookback_token_ids.cpu
    if num_reqs <= 0:
        return src_lookback.clone()
    lookback: Any = getattr(runner, "_host_engram_lookback_scratch", None)
    if (
        not isinstance(lookback, torch.Tensor)
        or lookback.shape != src_lookback.shape
        or lookback.dtype != src_lookback.dtype
    ):
        lookback = torch.empty_like(src_lookback)
        runner._host_engram_lookback_scratch = lookback
    lookback_np = lookback.numpy()
    if num_reqs < int(lookback.shape[0]):
        lookback_np[num_reqs:] = src_lookback.numpy()[num_reqs:]
    depth = int(lookback.shape[1])
    starts_1d = np.asarray(
        batch.num_computed_tokens_cpu[start_index : start_index + num_reqs],
        dtype=np.int64,
    )
    if rejected_by_req is not None:
        starts_1d = starts_1d - rejected_by_req
    starts = starts_1d[:, None]
    pos = starts - np.arange(1, depth + 1, dtype=np.int64)
    token_ids_cpu = np.asarray(batch.token_ids_cpu)
    max_cols = int(token_ids_cpu.shape[1])
    valid = (pos >= 0) & (pos < max_cols)
    local = np.arange(num_reqs, dtype=np.int64)[:, None]
    rows = local + start_index if start_index else local
    clipped = np.where(valid, pos, 0)
    gathered = np.where(valid, token_ids_cpu[rows, clipped], -1)
    if sampled_overlay is not None:
        rel = pos - sampled_overlay.starts[:, None]
        hit = (rel >= 0) & (rel < sampled_overlay.lengths[:, None])
        if hit.any():
            overlay = sampled_overlay.tokens[local, np.where(hit, rel, 0)]
            gathered = np.where(hit, overlay, gathered)
    lookback_np[:num_reqs] = gathered
    return lookback


def _ragged_arange(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenation of ``range(starts[i], starts[i] + lengths[i])``."""
    if lengths.size and bool(np.all(lengths == lengths[0])):
        return (
            np.asarray(starts, dtype=np.int64)[:, None]
            + np.arange(int(lengths[0]), dtype=np.int64)
        ).reshape(-1)
    offsets = np.repeat(np.cumsum(lengths) - lengths, lengths)
    return (
        np.arange(offsets.shape[0], dtype=np.int64)
        - offsets
        + np.repeat(starts, lengths)
    )


def _unresolved_host_inputs(runner, all_prompt: bool) -> HostEngramInputs:
    return HostEngramInputs(
        input_ids=runner.input_ids_cpu,
        positions=runner.positions_cpu,
        query_start_loc=runner.query_start_loc_cpu,
        lookback_token_ids=runner.lookback_token_ids.cpu,
        all_prompt=all_prompt,
        host_ready=False,
    )


def _dummy_host_inputs(runner, ctx) -> HostEngramInputs:
    pos_override = ctx.position_ids_override
    num_tokens = max(
        int(pos_override.shape[-1]),
        int(getattr(runner, "max_num_tokens", 0) or 0),
    )
    qsl = getattr(ctx, "query_start_loc", None)
    qsl_len = max(
        int(qsl.shape[0]) if qsl is not None else int(getattr(ctx, "num_reqs", 0)) + 1,
        int(getattr(runner, "max_num_reqs", 0) or 0) + 1,
    )
    lb_cpu = runner.lookback_token_ids.cpu
    lb_rows = max(int(lb_cpu.shape[0]), int(getattr(ctx, "num_reqs", 0)))
    depth = int(lb_cpu.shape[1])
    key = (num_tokens, qsl_len, lb_rows, depth)
    cached = getattr(runner, "_dummy_host_engram_inputs", None)
    if cached is None or cached[0] != key:
        inputs = HostEngramInputs(
            input_ids=torch.zeros(num_tokens, dtype=torch.int32),
            positions=torch.zeros(num_tokens, dtype=torch.int64),
            query_start_loc=torch.zeros(qsl_len, dtype=torch.int32),
            lookback_token_ids=torch.full((lb_rows, depth), -1, dtype=torch.int32),
            all_prompt=True,
            host_ready=True,
        )
        runner._dummy_host_engram_inputs = (key, inputs)
        return inputs
    return cached[1]


def capture_host_engram_inputs(runner, num_reqs: int) -> HostEngramInputs | None:
    """Snapshot the runner's host hash inputs, or None if they do not apply.

    This runs on the host between the previous step's completion and the
    next step's launch under ``OFFLOAD=1``, so everything per request is
    expressed as NumPy array operations over the chunk rather than per-request
    tensor ops.
    """
    from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind

    ctx = getattr(runner, "_attn_metadata_builder_ctx", None)
    if getattr(ctx, "position_ids_override", None) is not None:
        desc = getattr(ctx, "sequence_layout_descriptor", None)
        if getattr(desc, "kind", SequenceLayoutKind.ALL) is not SequenceLayoutKind.ALL:
            return None
        return _dummy_host_inputs(runner, ctx)
    plan = getattr(runner, "_last_sequence_layout_plan", None)
    if plan is None or plan.kind is not SequenceLayoutKind.ALL:
        return None
    start_index = int(getattr(ctx, "start_index", getattr(plan, "start_index", 0)))
    chunk_num_reqs = min(
        num_reqs,
        int(getattr(ctx, "num_reqs", getattr(plan, "num_reqs", num_reqs))),
    )
    batch = runner.input_batch
    seq_lens = runner.seq_lens_np[:chunk_num_reqs]
    prompt_lens = batch.num_prompt_tokens[start_index : start_index + chunk_num_reqs]
    all_prompt = bool(np.all(seq_lens <= prompt_lens))
    if all_prompt and start_index == 0:
        return HostEngramInputs(
            input_ids=runner.input_ids_cpu,
            positions=runner.positions_cpu,
            query_start_loc=runner.query_start_loc_cpu,
            lookback_token_ids=runner.lookback_token_ids.cpu,
            all_prompt=True,
            host_ready=True,
        )

    if (
        getattr(batch, "num_computed_tokens_cpu", None) is None
        or getattr(batch, "token_ids_cpu", None) is None
    ):
        return _unresolved_host_inputs(runner, all_prompt)

    pre_async = getattr(runner, "_pre_async_results", None)
    is_async_spec = pre_async is not None and (
        getattr(pre_async, "spec_decode_num_rejected_tokens", None) is not None
        or getattr(runner, "speculative_config", None) is not None
    )
    spec_state = _resolve_async_spec_state(runner, pre_async) if is_async_spec else None
    if is_async_spec and spec_state is None:
        return _unresolved_host_inputs(runner, all_prompt)

    stride = spec_state.stride if spec_state is not None else 1
    ids = runner.input_ids_cpu
    positions = runner.positions_cpu
    rejected_by_req: np.ndarray | None = None
    sampled_overlay: _SampledOverlay | None = None
    req_id_to_index = (
        getattr(pre_async, "req_id_to_index_copy", None)
        if pre_async is not None
        else None
    )
    req_ids = getattr(batch, "req_ids", None)
    if req_id_to_index and req_ids is not None:
        chunk_req_ids = req_ids[start_index : start_index + chunk_num_reqs]
        pre_pos = np.fromiter(
            map(req_id_to_index.get, chunk_req_ids, repeat(-1)),
            dtype=np.int64,
            count=chunk_num_reqs,
        )
        qsl = runner.query_start_loc_cpu.numpy()[: chunk_num_reqs + 1].astype(
            np.int64, copy=False
        )
        q_start = qsl[:-1]
        n_sched = qsl[1:] - q_start
        matched = (pre_pos >= 0) & (n_sched > 0)
        if spec_state is not None:
            # Rows this step did not carry over keep zero rejections and no
            # overlay; the lookback then reads token_ids_cpu as-is.
            safe_pos = np.where(matched, pre_pos, 0)
            rejected_by_req = np.where(matched, spec_state.rejected[safe_pos], 0)
            num_tokens_no_spec = getattr(batch, "num_tokens_no_spec", None)
            n_draft = spec_state.num_draft[safe_pos]
            if num_tokens_no_spec is not None:
                write_start = (
                    np.asarray(
                        num_tokens_no_spec[start_index : start_index + chunk_num_reqs],
                        dtype=np.int64,
                    )
                    - 1
                    - n_draft
                )
            else:
                write_start = (
                    np.asarray(
                        batch.num_computed_tokens_cpu[
                            start_index : start_index + chunk_num_reqs
                        ],
                        dtype=np.int64,
                    )
                    - n_draft
                )
            sampled_overlay = _SampledOverlay(
                tokens=spec_state.tokens[safe_pos],
                starts=write_start,
                lengths=np.where(matched, spec_state.lengths[safe_pos], 0),
            )
        if matched.any():
            m_len = n_sched[matched]
            dst_idx: slice | np.ndarray = (
                slice(int(qsl[0]), int(qsl[-1]))
                if bool(matched.all())
                else _ragged_arange(q_start[matched], m_len)
            )
            src_idx = _ragged_arange(pre_pos[matched] * stride, m_len)
            flat_src = _async_source_tokens_cpu(
                pre_async, prefer_next_tokens_tpu=(spec_state is not None)
            )
            if flat_src is None or int(flat_src.numel()) <= int(src_idx.max()):
                return _unresolved_host_inputs(runner, all_prompt)
            active_len = int(qsl[-1])
            ids = _acquire_host_scratch(
                runner, "_host_engram_ids_scratch", runner.input_ids_cpu, active_len
            )
            ids.numpy()[dst_idx] = flat_src.numpy()[src_idx]
            if rejected_by_req is not None:
                m_rej = rejected_by_req[matched]
                if m_rej.any():
                    positions = _acquire_host_scratch(
                        runner,
                        "_host_engram_pos_scratch",
                        runner.positions_cpu,
                        active_len,
                    )
                    positions.numpy()[dst_idx] -= np.repeat(m_rej, m_len)

    lookback = _resolve_decode_lookback_cpu(
        runner,
        batch,
        chunk_num_reqs,
        start_index,
        rejected_by_req=rejected_by_req,
        sampled_overlay=sampled_overlay,
    )
    return HostEngramInputs(
        input_ids=ids,
        positions=positions,
        query_start_loc=runner.query_start_loc_cpu,
        lookback_token_ids=lookback,
        all_prompt=all_prompt,
        host_ready=True,
    )


def gather_host_engram_hashes(
    engram_hashes: torch.Tensor,
    *,
    dp_shared_memory: bool = False,
    gather_fn: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Gather Engram hashes across DP, using ``dp_group.cpu_group`` on CPU."""
    if not dp_shared_memory:
        dp_group = get_engram_dp_group()
        if dp_group is not None and getattr(dp_group, "world_size", 1) > 1:
            dp_metadata = (
                getattr(get_forward_context(), "dp_metadata", None)
                if is_forward_context_available()
                else None
            )
            raw_slot = engram_gathered_num_tokens() if dp_metadata is not None else 0
            slot = raw_slot if raw_slot > 0 else int(engram_hashes.shape[0])
            dp_cpu_group = getattr(dp_group, "cpu_group", None)
            if (
                dp_cpu_group is not None
                and engram_hashes.device.type != "cpu"
                and envs.TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD
                and not envs.TPU_DEEPSEEK_V41_ENGRAM_HOST_DMA
                and not torch.compiler.is_compiling()
            ):
                engram_hashes = engram_hashes.to("cpu")
            cpu_group = dp_cpu_group if engram_hashes.device.type == "cpu" else None
            if cpu_group is not None:
                if engram_hashes.shape[0] > slot:
                    raise ValueError("Engram token count exceeds the DP token slot")
                if engram_hashes.shape[0] < slot:
                    pad = engram_hashes.new_full(
                        (slot - engram_hashes.shape[0], *engram_hashes.shape[1:]),
                        DEAD_ID,
                    )
                    engram_hashes = torch.cat((engram_hashes, pad))
                out = engram_hashes.new_empty(
                    (dp_group.world_size * slot, *engram_hashes.shape[1:])
                )
                torch.distributed.all_gather_into_tensor(
                    out, engram_hashes.contiguous(), group=cpu_group
                )
                return out
            if raw_slot <= 0 and gather_fn is None:
                return dp_group.all_gather(engram_hashes, dim=0)
    if gather_fn is not None:
        return gather_fn(engram_hashes, dp_shared_memory=dp_shared_memory)
    from vllm.models.deepseek_v41.nvidia.engram import gather_engram_hashes

    return gather_engram_hashes(engram_hashes, dp_shared_memory=dp_shared_memory)


def install_host_engram_inputs() -> None:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    original = TPUModelRunner._prepare_lookback_token_ids
    if getattr(original, "_deepseek_v41_host_engram", False):
        return

    def prepare(self, num_reqs: int) -> torch.Tensor:
        ids = original(self, num_reqs)
        if is_forward_context_available():
            get_forward_context().additional_kwargs[HOST_ENGRAM_KEY] = (
                capture_host_engram_inputs(self, num_reqs)
            )
        return ids

    prepare._deepseek_v41_host_engram = True  # type: ignore[attr-defined]
    TPUModelRunner._prepare_lookback_token_ids = prepare  # type: ignore[method-assign]
