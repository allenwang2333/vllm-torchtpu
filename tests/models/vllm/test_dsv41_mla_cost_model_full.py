"""100% statement and branch coverage test suite for MLA masked-dense cost model."""

from __future__ import annotations

import pytest

from vllm_torchtpu.kernels.mla.masked_dense.cost_model import (
    GLM52_TPU7X_PREFILL_COST_MODEL,
    KV_BLOCK_SIZE,
    QUERY_BLOCK_SIZE,
    MaskedDensePrefillCostModel,
)


def test_cost_model_constants_and_iteration() -> None:
    """Verify GLM52_TPU7X_PREFILL_COST_MODEL constants, _fields, and iteration."""
    assert KV_BLOCK_SIZE == 1024
    assert QUERY_BLOCK_SIZE == 32
    assert MaskedDensePrefillCostModel._fields == (
        "analytic_fixed_ns",
        "analytic_token_ns",
        "bitmap_fixed_ns",
        "bitmap_token_ns",
        "q_kv_block_ns",
        "sparse_fixed_ns",
        "sparse_token_ns",
    )

    values = tuple(GLM52_TPU7X_PREFILL_COST_MODEL)
    assert values == (215_000, 245, 227_000, 451, 8_700, 259_000, 2_156)

    with pytest.raises((AttributeError, TypeError)):
        GLM52_TPU7X_PREFILL_COST_MODEL.analytic_fixed_ns = 0  # type: ignore[misc]


def test_cost_model_estimates() -> None:
    """Verify analytic, bitmap, and sparse cost estimates across models."""
    default_cm = GLM52_TPU7X_PREFILL_COST_MODEL
    assert (
        default_cm.estimate_analytic_cost_ns(512, 8) == 215_000 + 512 * 245 + 8 * 8_700
    )
    assert default_cm.estimate_bitmap_cost_ns(512, 8) == 227_000 + 512 * 451 + 8 * 8_700
    assert default_cm.estimate_sparse_cost_ns(512) == 259_000 + 512 * 2_156

    custom_cm = MaskedDensePrefillCostModel(
        analytic_fixed_ns=100,
        analytic_token_ns=2,
        bitmap_fixed_ns=200,
        bitmap_token_ns=3,
        q_kv_block_ns=10,
        sparse_fixed_ns=300,
        sparse_token_ns=4,
    )
    assert custom_cm.estimate_analytic_cost_ns(16, 5) == 100 + 16 * 2 + 5 * 10
    assert custom_cm.estimate_bitmap_cost_ns(16, 5) == 200 + 16 * 3 + 5 * 10
    assert custom_cm.estimate_sparse_cost_ns(16) == 300 + 16 * 4

    assert (
        MaskedDensePrefillCostModel.estimate_analytic_cost_ns(custom_cm, 16, 5)
        == 100 + 16 * 2 + 5 * 10
    )
    assert (
        MaskedDensePrefillCostModel.estimate_bitmap_cost_ns(custom_cm, 16, 5)
        == 200 + 16 * 3 + 5 * 10
    )
    assert (
        MaskedDensePrefillCostModel.estimate_sparse_cost_ns(custom_cm, 16)
        == 300 + 16 * 4
    )
