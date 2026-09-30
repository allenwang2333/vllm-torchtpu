# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GaaD module: constants, invariants, FSMBridge, and GaadProposer."""

from unittest.mock import MagicMock

from vllm_torchtpu.spec_decode.gaad import (
    DRAFT_FLAG_EMPTY,
    DRAFT_FLAG_GRAMMAR,
    DRAFT_FLAG_MODEL,
    FSMBridge,
    GaadProposer,
    validate_draft_row_invariant,
)


class FakeMatcher:
    """Mock XGrammar matcher for deterministic traversal."""

    def __init__(self, jump_string="", accept_map=None):
        self._jump_string = jump_string
        self._accept_map = accept_map or {}
        self.accepted_tokens = []
        self.rollback_calls = []

    def accept_token(self, token_id: int) -> bool:
        can_accept = self._accept_map.get(token_id, True)
        if can_accept:
            self.accepted_tokens.append(token_id)
        return can_accept

    def find_jump_forward_string(self) -> str:
        return self._jump_string

    def rollback(self, count: int) -> None:
        self.rollback_calls.append(count)
        del self.accepted_tokens[-count:]


class FakeGrammar:
    """Mock StructuredOutputGrammar."""

    def __init__(self, matcher, is_terminated=False):
        self.matcher = matcher
        self._is_terminated = is_terminated

    def is_terminated(self) -> bool:
        return self._is_terminated


class FakeTokenizer:
    """Mock tokenizer for character-level or sub-word mapping."""

    def __init__(self, vocab_map=None):
        self.vocab_map = vocab_map or {
            ": ": [10, 20],
            '{"name": "': [100, 101, 102],
            "invalid": [999],
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.vocab_map.get(text, [ord(c) for c in text])


class TestGaadConstantsAndRowInvariant:
    """Tests for flag constants and validate_draft_row_invariant."""

    def test_constants_values(self):
        assert DRAFT_FLAG_EMPTY == 0
        assert DRAFT_FLAG_MODEL == 1
        assert DRAFT_FLAG_GRAMMAR == 2

    def test_valid_draft_rows(self):
        # Empty rows or pure padding
        assert validate_draft_row_invariant([])
        assert validate_draft_row_invariant([0, 0, 0, 0])

        # Pure grammar jumps
        assert validate_draft_row_invariant([2, 2, 2, 0])
        assert validate_draft_row_invariant([2, 2, 2, 2])

        # Pure model drafts
        assert validate_draft_row_invariant([1, 1, 0, 0])

        # Interleaved grammar and model drafts, ending with zeros
        assert validate_draft_row_invariant([2, 2, 1, 2, 0])
        assert validate_draft_row_invariant([1, 2, 1, 2, 2])

    def test_invalid_draft_rows_internal_holes(self):
        # Internal zeros (padding hole before valid draft) are forbidden
        assert not validate_draft_row_invariant([2, 0, 2, 0])
        assert not validate_draft_row_invariant([0, 1, 0, 0])
        assert not validate_draft_row_invariant([1, 0, 2, 0])

    def test_invalid_draft_rows_illegal_flags(self):
        # Flags outside {0, 1, 2}
        assert not validate_draft_row_invariant([3, 0, 0])
        assert not validate_draft_row_invariant([-1, 0, 0])
        assert not validate_draft_row_invariant([2, 4, 0])


class TestFSMBridge:
    """Tests for FSMBridge deterministic jump extraction and token
    boundary validation."""

    def test_extract_deterministic_jump_clean(self):
        bridge = FSMBridge()
        matcher = FakeMatcher(jump_string='{"name": "')
        grammar = FakeGrammar(matcher)
        tokenizer = FakeTokenizer()

        tokens, flags = bridge.extract_deterministic_jump(
            grammar=grammar,
            tokenizer=tokenizer,
            validated_draft=[],
            num_spec_tokens=5,
        )

        assert tokens == [100, 101, 102]
        assert flags == [DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR]
        # Matcher must be rolled back completely to preserve peek semantics
        assert len(matcher.accepted_tokens) == 0
        assert matcher.rollback_calls == [3]

    def test_extract_deterministic_jump_budget_cap(self):
        bridge = FSMBridge()
        matcher = FakeMatcher(jump_string='{"name": "')
        grammar = FakeGrammar(matcher)
        tokenizer = FakeTokenizer()

        tokens, flags = bridge.extract_deterministic_jump(
            grammar=grammar,
            tokenizer=tokenizer,
            validated_draft=[],
            num_spec_tokens=2,  # budget capped to 2
        )

        assert tokens == [100, 101]
        assert flags == [DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR]
        assert len(matcher.accepted_tokens) == 0
        assert matcher.rollback_calls == [2]

    def test_extract_deterministic_jump_boundary_rejection(self):
        bridge = FSMBridge()
        # Token 102 rejected by grammar
        matcher = FakeMatcher(
            jump_string='{"name": "', accept_map={100: True, 101: True, 102: False}
        )
        grammar = FakeGrammar(matcher)
        tokenizer = FakeTokenizer()

        tokens, flags = bridge.extract_deterministic_jump(
            grammar=grammar,
            tokenizer=tokenizer,
            validated_draft=[],
            num_spec_tokens=5,
        )

        assert tokens == [100, 101]
        assert flags == [DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR]
        assert len(matcher.accepted_tokens) == 0
        assert matcher.rollback_calls == [2]

    def test_extract_deterministic_jump_terminated_grammar(self):
        bridge = FSMBridge()
        matcher = FakeMatcher(jump_string='{"name": "')
        grammar = FakeGrammar(matcher, is_terminated=True)
        tokenizer = FakeTokenizer()

        tokens, flags = bridge.extract_deterministic_jump(
            grammar=grammar,
            tokenizer=tokenizer,
            validated_draft=[],
            num_spec_tokens=5,
        )

        assert tokens == []
        assert flags == []


class TestGaadProposer:
    """Tests for GaadProposer batch proposal interface."""

    def test_propose_batch_mixed_requests(self):
        vllm_config = MagicMock()
        vllm_config.speculative_config.num_speculative_tokens = 4
        proposer = GaadProposer(vllm_config)

        matcher1 = FakeMatcher(jump_string=": ")
        grammar1 = FakeGrammar(matcher1)

        req_structured = MagicMock()
        req_structured.is_finished.return_value = False
        req_structured.is_prefill_chunk = False
        req_structured.output_token_ids = [1]
        req_structured.max_tokens = 100
        req_structured.structured_output_request.grammar = grammar1

        req_unconstrained = MagicMock()
        req_unconstrained.is_finished.return_value = False
        req_unconstrained.is_prefill_chunk = False
        req_unconstrained.output_token_ids = [1]
        req_unconstrained.max_tokens = 100
        req_unconstrained.structured_output_request = None

        tokenizer = FakeTokenizer()

        batch_proposals = proposer.propose(
            requests=[req_structured, req_unconstrained],
            tokenizer=tokenizer,
        )

        assert len(batch_proposals) == 2
        # First request emitted grammar jump tokens [10, 20]
        assert batch_proposals[0] == [10, 20]
        # Second unconstrained request emitted empty list
        assert batch_proposals[1] == []
