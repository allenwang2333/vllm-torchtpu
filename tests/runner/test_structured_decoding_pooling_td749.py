# SPDX-License-Identifier: Apache-2.0
"""Tests for Track TD7.49: StructuredDecodingManager and TPUMeanPool optimizations."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.model_executor.layers.pooler.seqwise import heads, methods
from vllm.model_executor.layers.pooler.seqwise.heads import EmbeddingPoolerHead
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

import vllm_torchtpu.runner.tpu_pooling as tpu_pooling_mod
from vllm_torchtpu.runner.structured_decoding_manager import StructuredDecodingManager
from vllm_torchtpu.runner.tpu_pooling import (
    PromptTokenIdsTensor,
    TPUMeanPool,
    build_pooling_metadata,
    install_tpu_pooling_patches,
)

VOCAB_SIZE = 64
ALLOW_ALL = -1
BLOCK_ALL = 0


def make_manager(
    vocab_size: int = VOCAB_SIZE,
    max_num_reqs: int = 8,
    req_id_to_index: dict[str, int] | None = None,
    num_spec_tokens: int | None = None,
    num_tokens_paddings: list[int] | None = None,
    pcp_enabled: bool = False,
    mtp_enabled: bool = False,
    device: torch.device | str = "cpu",
) -> StructuredDecodingManager:
    speculative_config = (
        SimpleNamespace(num_speculative_tokens=num_spec_tokens)
        if num_spec_tokens is not None
        else None
    )
    runner = SimpleNamespace(
        vocab_size=vocab_size,
        max_num_reqs=max_num_reqs,
        device=torch.device(device),
        input_batch=SimpleNamespace(req_id_to_index=req_id_to_index or {}),
        speculative_config=speculative_config,
        num_tokens_paddings=num_tokens_paddings or [16, 32, 64],
        _pcp_enabled=pcp_enabled,
        _mtp_enabled=mtp_enabled,
    )
    return StructuredDecodingManager(runner)


def make_grammar_output(req_ids: list[str], rows: list[list[int]]):
    return SimpleNamespace(
        structured_output_request_ids=req_ids,
        grammar_bitmask=np.array(rows, dtype=np.int32)
        if rows
        else np.zeros((0, 2), dtype=np.int32),
    )


def row(value: int) -> list[int]:
    return [value, value]


@contextlib.contextmanager
def capture_trace():
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        yield prof


class TestStructuredDecodingManagerTD749:
    def test_init_and_non_spec_paths(self):
        # Non-spec manager and PCP+MTP K1 skip target buffer allocation.
        m_plain = make_manager()
        assert not hasattr(m_plain, "target_grammar_bitmask_cpu")
        assert not hasattr(m_plain, "device_all_false_require_target")

        m_pcp = make_manager(num_spec_tokens=1, pcp_enabled=True, mtp_enabled=True)
        assert not hasattr(m_pcp, "target_grammar_bitmask_cpu")

        # Spec manager allocates both CPU and device-resident target buffers.
        m_spec = make_manager(
            num_spec_tokens=2,
            max_num_reqs=4,
            num_tokens_paddings=[8, 16, 32],
            req_id_to_index={"req0": 0, "req1": 1, "req2": 2},
        )
        assert m_spec.device_all_false_require_target.shape == (16, 1)
        assert m_spec.device_all_false_require_target.dtype == torch.bool
        assert not m_spec.device_all_false_require_target.any()
        assert m_spec.device_dummy_bitmask_target.shape == (16, 2)
        assert m_spec.device_dummy_bitmask_target.dtype == torch.int32

        # Non-spec fast path (0 structured in chunk).
        logits = torch.zeros(4, VOCAB_SIZE)
        req, bm, arange = m_spec.prepare_structured_decoding_input(
            logits, make_grammar_output([], []), 0, 2
        )
        assert req.data_ptr() == m_spec.device_all_false_require.data_ptr()
        assert bm.data_ptr() == m_spec.device_dummy_bitmask.data_ptr()
        assert arange.tolist() == list(range(32))

        # Non-spec single local request (num_local == 1).
        go_single = make_grammar_output(
            ["req_missing", "req0", "req2"],
            [row(1), [ALLOW_ALL, BLOCK_ALL], row(2)],
        )
        req1, bm1, _ = m_spec.prepare_structured_decoding_input(logits, go_single, 0, 2)
        assert req1[:, 0].tolist() == [True, False, False, False]
        assert bm1[0].tolist() == [ALLOW_ALL, BLOCK_ALL]

        # Non-spec multiple local requests (num_local > 1).
        go_multi = make_grammar_output(
            ["req0", "req1"],
            [[ALLOW_ALL, BLOCK_ALL], [BLOCK_ALL, ALLOW_ALL]],
        )
        req2, bm2, _ = m_spec.prepare_structured_decoding_input(logits, go_multi, 0, 2)
        assert req2[:, 0].tolist() == [True, True, False, False]
        assert bm2[0].tolist() == [ALLOW_ALL, BLOCK_ALL]
        assert bm2[1].tolist() == [BLOCK_ALL, ALLOW_ALL]

    def test_spec_zero_h2d_fast_path_with_and_without_target_logits(self):
        manager = make_manager(
            num_spec_tokens=2,
            req_id_to_index={"req0": 0, "req1": 1, "req2": 2, "req3": 3},
        )
        # Populate CPU buffers with non-zero sentinels to verify fast path does NOT
        # touch or zero CPU buffers when no structured requests fall in the chunk.
        manager.grammar_bitmask_cpu.fill_(99)
        manager.require_structured_out_cpu.fill_(True)
        manager.target_grammar_bitmask_cpu.fill_(77)
        manager.require_structured_out_target_cpu.fill_(True)

        # req0 is in chunk [0, 2), while we query chunk [2, 4).
        grammar_output = make_grammar_output(["req0"], [row(10), row(11), row(12)])
        scheduled = {"req0": [1, 2], "req2": [3]}
        draft_lengths = np.array([1, 0], dtype=np.int32)
        target_logits = torch.zeros(4, VOCAB_SIZE)
        bonus_logits = torch.zeros(2, VOCAB_SIZE)

        (req_target, bm_target, req_bonus, bm_bonus, arange) = (
            manager.prepare_spec_structured_decoding_input(
                target_logits,
                bonus_logits,
                grammar_output,
                scheduled,
                draft_lengths,
                cur_start_idx=2,
                cur_end_idx=4,
            )
        )

        assert req_target is not None
        assert bm_target is not None
        assert (
            req_target.data_ptr() == manager.device_all_false_require_target.data_ptr()
        )
        assert bm_target.data_ptr() == manager.device_dummy_bitmask_target.data_ptr()
        assert req_target.shape == (4, 1)
        assert bm_target.shape == (4, 2)
        assert req_bonus.data_ptr() == manager.device_all_false_require.data_ptr()
        assert bm_bonus.data_ptr() == manager.device_dummy_bitmask.data_ptr()
        assert req_bonus.shape == (2, 1)
        assert bm_bonus.shape == (2, 2)
        assert arange.data_ptr() == manager.structured_decode_arange.data_ptr()
        # Sentinel values in CPU buffers remain untouched on the fast path.
        assert int(manager.grammar_bitmask_cpu[0, 0].item()) == 99
        assert int(manager.target_grammar_bitmask_cpu[0, 0].item()) == 77

        # Also test fast path when target_logits is None (non-draft chunk).
        (req_target_none, bm_target_none, req_bonus2, bm_bonus2, _) = (
            manager.prepare_spec_structured_decoding_input(
                None,
                bonus_logits,
                grammar_output,
                scheduled,
                None,
                cur_start_idx=2,
                cur_end_idx=4,
            )
        )
        assert req_target_none is None
        assert bm_target_none is None
        assert req_bonus2.data_ptr() == manager.device_all_false_require.data_ptr()
        assert bm_bonus2.data_ptr() == manager.device_dummy_bitmask.data_ptr()

    def test_spec_standard_path_and_draft_mismatch_error(self):
        manager = make_manager(
            num_spec_tokens=2,
            req_id_to_index={"req0": 0, "req1": 1},
        )
        grammar_output = make_grammar_output(
            ["req0", "req1"],
            [row(10), row(11), row(12), row(20)],
        )
        scheduled = {"req0": [1, 2], "req1": []}
        draft_lengths = np.array([2, 0], dtype=np.int32)
        target_logits = torch.zeros(4, VOCAB_SIZE)
        bonus_logits = torch.zeros(2, VOCAB_SIZE)

        (req_target, bm_target, req_bonus, bm_bonus, _) = (
            manager.prepare_spec_structured_decoding_input(
                target_logits,
                bonus_logits,
                grammar_output,
                scheduled,
                draft_lengths,
                0,
                2,
            )
        )
        assert req_target[:, 0].tolist() == [True, True, False, False]
        assert bm_target[0].tolist() == row(10)
        assert bm_target[1].tolist() == row(11)
        assert req_bonus[:, 0].tolist() == [True, True]
        assert bm_bonus[0].tolist() == row(12)
        assert bm_bonus[1].tolist() == row(20)

        # Standard path with target_logits=None and 0 drafts for structured req.
        go_no_drafts = make_grammar_output(["req1"], [row(30)])
        (req_t_none, bm_t_none, req_b, bm_b, _) = (
            manager.prepare_spec_structured_decoding_input(
                None,
                bonus_logits,
                go_no_drafts,
                {},
                None,
                0,
                2,
            )
        )
        assert req_t_none is None
        assert bm_t_none is None
        assert req_b[:, 0].tolist() == [False, True]
        assert bm_b[1].tolist() == row(30)

        # Draft lengths mismatch raises ValueError.
        bad_draft_lengths = np.array([1, 0], dtype=np.int32)
        with pytest.raises(
            ValueError,
            match="Draft lengths mismatch for request req0",
        ):
            manager.prepare_spec_structured_decoding_input(
                target_logits,
                bonus_logits,
                grammar_output,
                scheduled,
                bad_draft_lengths,
                0,
                2,
            )

    def test_structured_decode_and_mask_logits_all_branches(self, monkeypatch):
        manager = make_manager(
            vocab_size=64,
            num_spec_tokens=1,
            req_id_to_index={"req0": 0},
        )
        logits = torch.zeros(1, 64)
        bitmask = torch.tensor([[ALLOW_ALL, BLOCK_ALL]], dtype=torch.int32)
        require = torch.tensor([[True]])
        arange = torch.arange(32, dtype=torch.int32)

        # Exercise unwrapped structured_decode directly on CPU.
        unwrapped_out = StructuredDecodingManager.structured_decode.__wrapped__(
            manager, require, bitmask, logits, arange
        )
        assert torch.equal(unwrapped_out[0, :32], logits[0, :32])
        assert torch.isneginf(unwrapped_out[0, 32:]).all()

        # Padded vocab dimension (target_dim > vocab_size).
        padded_logits = torch.zeros(1, 128)
        padded_out = manager.apply_grammar_bitmask(padded_logits, bitmask, arange)
        assert padded_out.shape == (1, 128)
        assert torch.isneginf(padded_out[0, 32:]).all()

        # Smaller vocab dimension raises ValueError.
        with pytest.raises(ValueError, match="must be at least vocab_size"):
            manager.apply_grammar_bitmask(torch.zeros(1, 32), bitmask, arange)

        # mask_logits and mask_spec_logits with monkeypatched structured_decode.
        monkeypatch.setattr(manager, "structured_decode", manager._structured_decode)
        go = make_grammar_output(["req0"], [[ALLOW_ALL, BLOCK_ALL]])
        masked = manager.mask_logits(logits, go, 0, 1)
        assert torch.isneginf(masked[0, 32:]).all()

        go_spec = make_grammar_output(
            ["req0"], [[ALLOW_ALL, BLOCK_ALL], [BLOCK_ALL, ALLOW_ALL]]
        )
        t_out, b_out = manager.mask_spec_logits(
            torch.zeros(1, 64),
            torch.zeros(1, 64),
            go_spec,
            {"req0": [1]},
            np.array([1], dtype=np.int32),
            0,
            1,
        )
        assert t_out is not None
        assert torch.isneginf(t_out[0, 32:]).all()
        assert torch.isneginf(b_out[0, :32]).all()

        t_none, b_out2 = manager.mask_spec_logits(
            None,
            torch.zeros(1, 64),
            go,
            {},
            None,
            0,
            1,
        )
        assert t_none is None
        assert torch.isneginf(b_out2[0, 32:]).all()


class TestTPUPoolingTD749:
    def test_install_patches_idempotency_and_prompt_token_ids_tensor(self):
        # Reset flag once to cover both branches of install_tpu_pooling_patches().
        orig_embed = heads.EmbeddingPoolerHead.forward
        orig_get_pti = PoolingMetadata._get_prompt_token_ids
        try:
            tpu_pooling_mod._tpu_pooling_patches_installed = False
            install_tpu_pooling_patches()
            assert tpu_pooling_mod._tpu_pooling_patches_installed is True
            # Second call hits early return branch.
            install_tpu_pooling_patches()
        finally:
            tpu_pooling_mod._tpu_pooling_patches_installed = True

        raw = torch.tensor([[10, 20, 30], [40, 50, 0]], dtype=torch.int32)
        req_map = {"r1": [10, 20, 30], "r2": [40, 50]}
        pt = PromptTokenIdsTensor(raw, req_map)

        # String and non-string keys for PromptTokenIdsTensor methods.
        assert pt["r1"] == [10, 20, 30]
        assert isinstance(pt[0:1], PromptTokenIdsTensor)
        assert not isinstance(pt[0], PromptTokenIdsTensor)
        assert pt.get("r1") == [10, 20, 30]
        assert pt.get("missing", "def") == "def"
        assert pt.get(123, "non_str_def") == "non_str_def"
        assert "r1" in pt
        assert "missing" not in pt
        assert (123 in pt) is False
        assert list(pt.keys()) == ["r1", "r2"]
        assert list(pt.values()) == [[10, 20, 30], [40, 50]]
        assert dict(pt.items()) == req_map
        _ = (orig_embed, orig_get_pti)

    def test_tpu_mean_pool_non_partial_all_branches(self):
        pooler = TPUMeanPool()
        dim = 4

        # 1. Empty batch (num_seqs == 0).
        meta_empty = MagicMock()
        cursor_empty = MagicMock()
        cursor_empty.is_partial_prefill.return_value = False
        cursor_empty.prompt_lens_cpu = torch.tensor([], dtype=torch.int32)
        meta_empty.get_pooling_cursor.return_value = cursor_empty
        out_empty = pooler.forward(torch.zeros(0, dim, dtype=torch.float32), meta_empty)
        assert out_empty.shape == (0, dim)
        assert out_empty.dtype == torch.float32

        # 2. prompt_lens_cpu as torch.Tensor + FP32 hidden_states (no cast needed).
        hs_fp32 = torch.randn(5, dim, dtype=torch.float32)
        meta_tensor = MagicMock()
        cursor_tensor = MagicMock()
        cursor_tensor.is_partial_prefill.return_value = False
        cursor_tensor.prompt_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
        meta_tensor.get_pooling_cursor.return_value = cursor_tensor
        out_fp32 = pooler.forward(hs_fp32, meta_tensor)
        assert torch.allclose(out_fp32[0], hs_fp32[:2].mean(dim=0))
        assert torch.allclose(out_fp32[1], hs_fp32[2:].mean(dim=0))

        # 3. prompt_lens_cpu as non-Tensor sequence + BF16 hidden_states.
        class _HostLensList(list):
            def numel(self) -> int:
                return len(self)

            def sum(self) -> int:
                return sum(int(x) for x in self)

        hs_bf16 = torch.randn(5, dim, dtype=torch.bfloat16)
        meta_np = MagicMock()
        cursor_np = MagicMock()
        cursor_np.is_partial_prefill.return_value = False
        cursor_np.prompt_lens_cpu = _HostLensList([2, 3])
        meta_np.get_pooling_cursor.return_value = cursor_np
        out_bf16 = pooler.forward(hs_bf16, meta_np)
        assert out_bf16.dtype == torch.float32
        assert torch.allclose(
            out_bf16[0], hs_bf16[:2].to(torch.float32).mean(dim=0), atol=1e-2
        )

    def test_tpu_mean_pool_chunked_prefill_fp32_and_bf16_and_state_cleanup(self):
        pooler = TPUMeanPool()
        dim = 4

        # Request 0: PoolingStates (has .clean()), BF16 hidden_states, 2 steps.
        # Request 1: SimpleNamespace without .clean(), FP32 hidden_states, 1 step.
        state_with_clean = PoolingStates()
        state_no_clean = SimpleNamespace()

        hs_step1_bf16 = torch.randn(3, dim, dtype=torch.bfloat16)
        meta_step1 = MagicMock()
        cursor_step1 = MagicMock()
        cursor_step1.is_partial_prefill.return_value = True
        cursor_step1.num_scheduled_tokens_cpu = torch.tensor([3], dtype=torch.int32)
        cursor_step1.is_finished.return_value = torch.tensor([False])
        meta_step1.get_pooling_cursor.return_value = cursor_step1
        meta_step1.pooling_states = [state_with_clean]

        out1 = pooler.forward(hs_step1_bf16, meta_step1)
        assert out1 == [None]
        assert state_with_clean.mean_pool_count == 3

        # Step 2 for Request 0 (finished, BF16 cast + clean() called).
        hs_step2_bf16 = torch.randn(2, dim, dtype=torch.bfloat16)
        cursor_step1.num_scheduled_tokens_cpu = torch.tensor([2], dtype=torch.int32)
        cursor_step1.is_finished.return_value = torch.tensor([True])
        out2 = pooler.forward(hs_step2_bf16, meta_step1)
        assert len(out2) == 1
        assert out2[0] is not None
        assert out2[0].dtype == torch.bfloat16
        assert state_with_clean.mean_pool_sum is None
        assert state_with_clean.mean_pool_count == 0

        # Request 1 with FP32 hidden_states and state without .clean().
        hs_fp32 = torch.randn(4, dim, dtype=torch.float32)
        meta_fp32 = MagicMock()
        cursor_fp32 = MagicMock()
        cursor_fp32.is_partial_prefill.return_value = True
        cursor_fp32.num_scheduled_tokens_cpu = torch.tensor([4], dtype=torch.int32)
        cursor_fp32.is_finished.return_value = torch.tensor([True])
        meta_fp32.get_pooling_cursor.return_value = cursor_fp32
        meta_fp32.pooling_states = [state_no_clean]

        out_fp32 = pooler.forward(hs_fp32, meta_fp32)
        assert len(out_fp32) == 1
        assert out_fp32[0] is not None
        assert out_fp32[0].dtype == torch.float32
        assert torch.allclose(out_fp32[0], hs_fp32.mean(dim=0))
        assert state_no_clean.mean_pool_sum is None

    def test_safe_embed_forward_and_safe_get_prompt_token_ids_branches(
        self, monkeypatch
    ):
        install_tpu_pooling_patches()
        head = EmbeddingPoolerHead()
        p1 = PoolingParams(task="embed")
        p2 = PoolingParams(task="embed")
        meta = PoolingMetadata(
            prompt_lens=torch.tensor([3, 3], dtype=torch.int32),
            prompt_token_ids=None,
            prompt_token_ids_cpu=None,
            pooling_params=[p1, p2],
            pooling_states=[],
        )

        # 1. List with None -> exercises per-item loop and shape[0] == 1 branch.
        emb = torch.randn(4)
        out_with_none = head.forward([None, emb], meta)
        assert out_with_none[0] is None
        assert torch.allclose(out_with_none[1], emb)

        # 2. List WITHOUT None (or Tensor) -> exercises line 203 direct pass-through.
        out_no_none = head.forward(torch.randn(2, 4), meta)
        assert out_no_none.shape == (2, 4)

        # 3. List with None when orig_embed_forward returns a non-1-row result.
        orig_head_forward = heads.EmbeddingPoolerHead.forward
        try:
            heads.EmbeddingPoolerHead.forward = lambda self_obj, pd, pm: torch.zeros(
                2, 4
            )
            tpu_pooling_mod._tpu_pooling_patches_installed = False
            install_tpu_pooling_patches()
            out_multi_row = head.forward([None, emb], meta)
            assert out_multi_row[0] is None
            assert out_multi_row[1].shape == (2, 4)
        finally:
            heads.EmbeddingPoolerHead.forward = orig_head_forward
            tpu_pooling_mod._tpu_pooling_patches_installed = True

        # 4. safe_get_prompt_token_ids: None, dict, and Tensor branches.
        with pytest.raises(ValueError, match="prompt_token_ids is required"):
            meta._get_prompt_token_ids(None)

        dict_out = meta._get_prompt_token_ids({"r1": [1, 2, 3], "r2": [4, 5]})
        assert len(dict_out) == 2
        assert torch.equal(dict_out[0], torch.tensor([1, 2, 3], dtype=torch.int32))

        tensor_pti = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        tensor_out = meta._get_prompt_token_ids(tensor_pti)
        assert len(tensor_out) == 2

    def test_build_pooling_metadata_all_fallback_branches(self):
        # 1. batch.requests is None and runner_requests is None -> reqs = []
        empty_batch = SimpleNamespace(requests=None)
        meta_empty = build_pooling_metadata(empty_batch, None)
        assert len(meta_empty.prompt_lens) == 0

        # 2. batch.requests is provided directly with various request attribute shapes:
        # - r_len_attr: has prompt_token_ids_len (int), pooling_states=None (has attr)
        r_len_attr = SimpleNamespace(
            req_id="r_len",
            prompt_token_ids_len=3,
            num_prompt_tokens=None,
            prompt_token_ids=None,
            pooling_params=PoolingParams(task="embed"),
            pooling_states=None,
        )
        # - r_num_prompt: num_prompt_tokens=4, no pooling_states attr
        r_num_prompt = SimpleNamespace(
            req_id="r_num",
            prompt_token_ids_len=None,
            num_prompt_tokens=4,
            prompt_token_ids=None,
            pooling_params=PoolingParams(task="embed"),
        )
        # - r_tokens_only: only prompt_token_ids=[7, 8] (length 2 > 0)
        r_tokens_only = SimpleNamespace(
            req_id="r_tok",
            prompt_token_ids_len=None,
            num_prompt_tokens=None,
            prompt_token_ids=[7, 8],
            pooling_params=PoolingParams(task="embed"),
            pooling_states=PoolingStates(),
        )
        # - r_tokens_shorter: prompt_token_ids=[9] (length 1 <= max_prompt_len=2)
        r_tokens_shorter = SimpleNamespace(
            req_id="r_tok_short",
            prompt_token_ids_len=None,
            num_prompt_tokens=None,
            prompt_token_ids=[9],
            pooling_params=PoolingParams(task="embed"),
            pooling_states=PoolingStates(),
        )
        # - r_none_all: none of the length attributes set -> returns 0
        r_none_all = SimpleNamespace(
            req_id="r_zero",
            prompt_token_ids_len=None,
            num_prompt_tokens=None,
            prompt_token_ids=None,
            pooling_params=PoolingParams(task="embed"),
        )

        batch_direct = SimpleNamespace(
            requests=[
                r_len_attr,
                r_num_prompt,
                r_tokens_only,
                r_tokens_shorter,
                r_none_all,
            ],
            num_reqs=5,
        )
        meta_direct = build_pooling_metadata(batch_direct)
        assert meta_direct.prompt_lens.tolist() == [3, 4, 2, 1, 0]
        assert isinstance(r_len_attr.pooling_states, PoolingStates)
        assert not hasattr(r_num_prompt, "pooling_states")
        assert meta_direct.prompt_token_ids_cpu["r_len"] == []
        assert meta_direct.prompt_token_ids_cpu["r_tok"] == [7, 8]
        assert meta_direct.prompt_token_ids_cpu["r_tok_short"] == [9]

        # 3. All requests have prompt_token_ids=None but non-zero prompt_lens
        # -> hits `if max_prompt_len == 0 and len(prompt_lens) > 0:` (line 279).
        batch_no_tokens = SimpleNamespace(
            requests=[r_num_prompt],
            num_reqs=1,
        )
        meta_no_tokens = build_pooling_metadata(batch_no_tokens)
        assert meta_no_tokens.prompt_token_ids_cpu.shape == (1, 4)

        # 4. runner_requests path via batch.req_ids.
        batch_runner = SimpleNamespace(
            requests=None,
            req_ids=["r_tok"],
            num_reqs=1,
        )
        meta_runner = build_pooling_metadata(batch_runner, {"r_tok": r_tokens_only})
        assert meta_runner.prompt_lens.tolist() == [2]
        assert methods.MeanPool.forward == TPUMeanPool.forward
