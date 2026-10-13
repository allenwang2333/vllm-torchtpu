# Copyright 2026 Google LLC
"""Full 100% statement and branch coverage tests for tracing.annotation."""

from unittest.mock import MagicMock, patch

import pytest

from vllm_torchtpu.tracing.annotation import (
    TraceAnnotation,
    is_trace_annotation_enabled,
)

pytestmark = pytest.mark.cpu_test


def test_trace_annotation_disabled_and_enabled_branches() -> None:
    with patch("torch.autograd._profiler_enabled", return_value=False):
        assert is_trace_annotation_enabled() is False
        assert TraceAnnotation.is_enabled() is False
        with patch("torch.profiler.record_function") as mock_rf:
            with TraceAnnotation("PP:Handoff") as ann:
                assert ann.name == "PP:Handoff"
                assert ann.kwargs == {}
                assert ann._rf is None
            mock_rf.assert_not_called()

    with patch("torch.autograd._profiler_enabled", return_value=True):
        assert is_trace_annotation_enabled() is True
        assert TraceAnnotation.is_enabled() is True

        with patch("torch.profiler.record_function") as mock_rf:
            mock_ctx = MagicMock()
            mock_rf.return_value = mock_ctx

            with TraceAnnotation("ModelForward") as ann_no_kwargs:
                assert ann_no_kwargs._rf is mock_ctx
            mock_rf.assert_called_once_with("ModelForward")
            mock_ctx.__enter__.assert_called_once_with()
            mock_ctx.__exit__.assert_called_once_with(None, None, None)
            assert ann_no_kwargs._rf is None

        with patch("torch.profiler.record_function") as mock_rf_kw:
            mock_ctx_kw = MagicMock()
            mock_rf_kw.return_value = mock_ctx_kw

            with TraceAnnotation(
                "SD:MaskLogits",
                num_reqs=4,
                padded_num_reqs=8,
                phase="DECODE",
            ) as ann_kw:
                assert ann_kw._rf is mock_ctx_kw
            mock_rf_kw.assert_called_once_with(
                "SD:MaskLogits#num_reqs=4,padded_num_reqs=8,phase=DECODE#"
            )
            mock_ctx_kw.__enter__.assert_called_once_with()
            mock_ctx_kw.__exit__.assert_called_once_with(None, None, None)
            assert ann_kw._rf is None
