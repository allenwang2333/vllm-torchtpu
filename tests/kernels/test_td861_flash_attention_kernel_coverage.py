# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""100% statement and branch coverage tests for TD8.61 Flash Attention Pallas kernel."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest  # type: ignore[missing-import]
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.flash_attention import kernel
from vllm_torchtpu.kernels.flash_attention.kernel import (
    BlockSizes,
    SegmentIds,
    below_or_on_diag,
    calculate_vmem_usage_bytes,
    flash_attention,
    mha_reference,
    mha_reference_no_custom_vjp,
)
from vllm_torchtpu.kernels.flash_attention.tuned_params import TunableParams

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Helpers for kernel execution
# ---------------------------------------------------------------------------


class _MockRef:
    """Mutable array reference simulating Pallas tile refs in CPU tests."""

    def __init__(self, arr: jax.Array):
        self.val = arr
        self.shape = arr.shape
        self.dtype = arr.dtype

    def __getitem__(self, idx: Any) -> Any:
        return self.val[idx]

    def __setitem__(self, idx: Any, val: Any) -> None:
        self.val = self.val.at[idx].set(val)


# ---------------------------------------------------------------------------
# 1. BlockSizes & VMEM calculation tests
# ---------------------------------------------------------------------------


def test_block_sizes_slots_and_validation() -> None:
    # slots=True verified
    bs = BlockSizes(block_q=128, block_k_major=128, block_k=128, block_b=1)
    assert hasattr(bs, "__slots__")
    assert bs.block_q == 128

    # get_default singleton
    bs_def = BlockSizes.get_default(1, 1, 128, 128, 128)
    assert bs_def is kernel._DEFAULT_BLOCK_SIZES
    assert bs_def.block_q == 128

    # verify_major_minor: minor > major
    with pytest.raises(ValueError, match="should be smaller than"):
        BlockSizes(block_q=128, block_k_major=128, block_k=256, block_b=1)

    # verify_major_minor: major % minor != 0
    with pytest.raises(ValueError, match="should divide"):
        BlockSizes(block_q=128, block_k_major=128, block_k=96, block_b=1)


def test_calculate_vmem_usage_bytes(monkeypatch) -> None:
    monkeypatch.setattr(
        kernel.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            num_lanes=128, num_sublanes=8, vmem_capacity_bytes=64 * 1024 * 1024
        ),
    )
    bs = BlockSizes(block_q=128, block_k_major=128, block_k=128, block_b=1)

    # 1. ab is None, segment_ids is None
    vmem1 = calculate_vmem_usage_bytes(
        bs, jnp.bfloat16, jnp.bfloat16, head_dim=128, kv_seq_len=256
    )
    assert vmem1 > 0

    # 2. ab is not None, segment_ids is not None
    ab_dummy = SimpleNamespace(dtype=jnp.float32)
    seg_dummy = SimpleNamespace()
    vmem2 = calculate_vmem_usage_bytes(
        bs,
        jnp.bfloat16,
        jnp.bfloat16,
        head_dim=128,
        kv_seq_len=256,
        ab=ab_dummy,
        segment_ids=seg_dummy,
    )
    assert vmem2 > vmem1


# ---------------------------------------------------------------------------
# 2. flash_attention argument validation and planning
# ---------------------------------------------------------------------------


def test_flash_attention_validation_errors() -> None:
    fn = flash_attention.__wrapped__

    q_ok = jnp.zeros((2, 4, 128, 64), dtype=jnp.bfloat16)
    k_ok = jnp.zeros((2, 4, 128, 64), dtype=jnp.bfloat16)
    v_ok = jnp.zeros((2, 4, 128, 64), dtype=jnp.bfloat16)

    # Batch size mismatch
    k_bad_b = jnp.zeros((3, 4, 128, 64), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="Batch size mismatch"):
        fn(q_ok, k_bad_b, v_ok, vmem_limit_bytes=1000)

    # Head count mismatch
    k_bad_h = jnp.zeros((2, 8, 128, 64), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="Head count mismatch"):
        fn(q_ok, k_bad_h, v_ok, vmem_limit_bytes=1000)

    # Model dimension mismatch: q vs k
    k_bad_d = jnp.zeros((2, 4, 128, 128), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="Model dimension mismatch"):
        fn(q_ok, k_bad_d, v_ok, vmem_limit_bytes=1000)

    # Model dimension mismatch: q vs v
    v_bad_d = jnp.zeros((2, 4, 128, 128), dtype=jnp.bfloat16)
    with pytest.raises(NotImplementedError, match="V model dimension"):
        fn(q_ok, k_ok, v_bad_d, vmem_limit_bytes=1000)

    # KV sequence length mismatch: k vs v
    v_bad_seq = jnp.zeros((2, 4, 64, 64), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="KV sequence length mismatch"):
        fn(q_ok, k_ok, v_bad_seq, vmem_limit_bytes=1000)

    # Attention bias shape mismatch
    ab_bad = jnp.zeros((2, 4, 128, 64), dtype=jnp.float32)  # expected (2, 4, 128, 128)
    with pytest.raises(ValueError, match="Attention bias shape mismatch"):
        fn(q_ok, k_ok, v_ok, ab=ab_bad, vmem_limit_bytes=1000)

    # Segment IDs shape mismatch: q
    seg_bad_q = SegmentIds(q=jnp.zeros((1, 128)), kv=jnp.zeros((2, 128)))
    with pytest.raises(ValueError, match="Q segment ids shape mismatch"):
        fn(q_ok, k_ok, v_ok, segment_ids=seg_bad_q, vmem_limit_bytes=1000)

    # Segment IDs shape mismatch: kv
    seg_bad_kv = SegmentIds(q=jnp.zeros((2, 128)), kv=jnp.zeros((2, 64)))
    with pytest.raises(ValueError, match="KV segment ids shape mismatch"):
        fn(q_ok, k_ok, v_ok, segment_ids=seg_bad_kv, vmem_limit_bytes=1000)


def test_flash_attention_block_size_resolution(monkeypatch) -> None:
    fn = flash_attention.__wrapped__

    called_args: list[Any] = []

    def mock_flash_attention(*args):
        called_args.append(args)
        return args[0]  # return q

    monkeypatch.setattr(kernel, "_flash_attention", mock_flash_attention)
    monkeypatch.setattr(
        kernel.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            num_lanes=128, num_sublanes=8, vmem_capacity_bytes=64 * 1024 * 1024
        ),
    )

    q = jnp.zeros((1, 1, 128, 128), dtype=jnp.bfloat16)
    k = jnp.zeros((1, 1, 128, 128), dtype=jnp.bfloat16)
    v = jnp.zeros((1, 1, 128, 128), dtype=jnp.bfloat16)

    # 1. tuned_params is not None
    monkeypatch.setattr(
        kernel,
        "get_tuned_params",
        lambda key: TunableParams(block_q=64, block_k_major=128, block_k=64, block_b=1),
    )
    fn(q, k, v, vmem_limit_bytes=1000)
    resolved_bs = called_args[-1][8]
    assert resolved_bs.block_q == 64
    assert resolved_bs.block_k == 64

    # 2. tuned_params is None, vmem <= limit * 0.9 (single-step fit)
    monkeypatch.setattr(kernel, "get_tuned_params", lambda key: None)
    fn(q, k, v, vmem_limit_bytes=64 * 1024 * 1024)
    resolved_fit = called_args[-1][8]
    assert resolved_fit.block_k_major == 128
    assert resolved_fit.block_k == 128

    # 3. tuned_params is None, vmem > limit * 0.9 (fallback to default block sizes)
    # Also tests vmem_limit_bytes is None (uses get_tpu_info().vmem_capacity_bytes)
    monkeypatch.setattr(
        kernel.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(num_lanes=128, num_sublanes=8, vmem_capacity_bytes=100),
    )
    fn(q, k, v, vmem_limit_bytes=None)
    resolved_fallback = called_args[-1][8]
    assert resolved_fallback is kernel._DEFAULT_BLOCK_SIZES

    # 4. Explicit block_sizes passed in (bypasses block_sizes is None branch: 197->234)
    # Also pass valid ab and valid segment_ids to cover 181->186 and 192->197
    explicit_bs = BlockSizes(block_q=64, block_k_major=64, block_k=64, block_b=1)
    ab_valid = jnp.zeros((1, 1, 128, 128))
    seg_valid = SegmentIds(q=jnp.zeros((1, 128)), kv=jnp.zeros((1, 128)))
    res = fn(
        q,
        k,
        v,
        ab=ab_valid,
        segment_ids=seg_valid,
        block_sizes=explicit_bs,
        vmem_limit_bytes=1000,
    )
    assert res is q
    assert called_args[-1][8] is explicit_bs
    assert called_args[-1][3] is ab_valid
    assert called_args[-1][4] is seg_valid


# ---------------------------------------------------------------------------
# 3. below_or_on_diag, verify_block, bytes, cost estimate & compiler params
# ---------------------------------------------------------------------------


def test_below_or_on_diag_and_verify_block() -> None:
    # below_or_on_diag True vs False
    assert below_or_on_diag(0, 128, 0, 128) is True
    assert below_or_on_diag(0, 128, 1, 128) is False

    # _verify_block: block > dim
    with pytest.raises(ValueError, match="should be smaller or equal"):
        kernel._verify_block("block_q", "q_seq_len", 256, 128)

    # _verify_block: dim % block != 0
    with pytest.raises(ValueError, match="should be divisible"):
        kernel._verify_block("block_q", "q_seq_len", 80, 128, should_divide=True)

    # _verify_block: should_divide=False allows non-divisible
    kernel._verify_block("block_q", "q_seq_len", 80, 128, should_divide=False)


def test_bytes_and_fwd_cost_estimate() -> None:
    # _bytes with array (.size present)
    arr = jnp.zeros((2, 4, 8), dtype=jnp.float32)
    assert kernel._bytes(arr) == 64 * 4

    # _bytes without .size fallback
    class _MockShapeOnly:
        shape = (2, 4, 8)
        dtype = jnp.dtype("float32")

    assert kernel._bytes(_MockShapeOnly()) == 64 * 4

    # _fwd_cost_estimate
    q = jnp.ones((1, 1, 128, 128), dtype=jnp.bfloat16)
    cost = kernel._fwd_cost_estimate(
        q,
        q,
        q,
        None,
        None,
        causal=False,
        sm_scale=1.0,
        kernel_inputs_specs=(q, q, q, None),
        kernel_outputs_specs=(q, None),
    )
    assert cost is not None
    assert cost.flops > 0
    assert cost.bytes_accessed > 0


def test_compiler_params_cache() -> None:
    p1 = kernel._get_compiler_params(1024)
    p2 = kernel._get_compiler_params(1024)
    p3 = kernel._get_compiler_params(2048)
    p_none = kernel._get_compiler_params(None)
    assert p1 is p2
    assert p1 is not p3
    assert p_none.vmem_limit_bytes is None


# ---------------------------------------------------------------------------
# 4. Module-level index map functions
# ---------------------------------------------------------------------------


def test_module_level_index_maps() -> None:
    assert kernel._q_index_map(1, 2, 3, 4) == (1, 2, 3, 0)
    assert kernel._o_index_map(1, 2, 3, 4) == (1, 2, 3, 0)
    assert kernel._lm_index_map(1, 2, 3, 4) == (1, 2, 3, 0)
    assert kernel._kv_noncausal_index_map(1, 2, 3, 4) == (1, 2, 4, 0)
    assert kernel._ab_noncausal_index_map(1, 2, 3, 4) == (1, 2, 3, 4)
    assert kernel._q_segment_ids_index_map(1, 2, 3, 4) == (1, 3, 0)
    assert kernel._kv_segment_ids_noncausal_index_map(1, 2, 3, 4) == (1, 0, 4)


# ---------------------------------------------------------------------------
# 5. _flash_attention_impl branches and Pallas call invocation
# ---------------------------------------------------------------------------


def test_flash_attention_impl_all_branches(monkeypatch) -> None:
    captured_calls: list[dict[str, Any]] = []

    def mock_pallas_call(k, **kwargs):
        captured_calls.append(kwargs)

        def invoke(*args):
            o = jnp.zeros_like(args[0])
            l_dummy = jnp.zeros(
                (args[0].shape[0], args[0].shape[1], args[0].shape[2], 128, 1),
                dtype=jnp.float32,
            )
            m_dummy = jnp.zeros(
                (args[0].shape[0], args[0].shape[1], args[0].shape[2], 128, 1),
                dtype=jnp.float32,
            )
            return (o, l_dummy, m_dummy)

        return invoke

    monkeypatch.setattr(kernel.pl, "pallas_call", mock_pallas_call)

    q = jnp.ones((1, 2, 128, 128), dtype=jnp.bfloat16)
    k = jnp.ones((1, 2, 256, 128), dtype=jnp.bfloat16)
    v = jnp.ones((1, 2, 256, 128), dtype=jnp.bfloat16)
    ab = jnp.zeros((1, 2, 128, 256), dtype=jnp.float32)
    seg = SegmentIds(q=jnp.zeros((1, 128)), kv=jnp.zeros((1, 256)))

    # A. Causal=True, save_residuals=True, ab, segment_ids, block_k != kv_seq_len
    res_causal = kernel._flash_attention(
        q,
        k,
        v,
        ab,
        seg,
        save_residuals=True,
        causal=True,
        sm_scale=0.5,
        block_sizes=BlockSizes(128, 128, 128, 1),
        vmem_limit_bytes=1000,
        debug=False,
    )
    assert isinstance(res_causal, tuple)
    assert len(res_causal) == 3

    # Exercise causal index map closures created inside _flash_attention_impl
    grid_spec = captured_calls[-1]["grid_spec"]
    kv_map = grid_spec.in_specs[1].index_map
    ab_map = grid_spec.in_specs[3].index_map
    kv_seg_map = grid_spec.in_specs[5].index_map

    # Test kv_index_map causal branch
    assert kv_map(0, 0, 0, 0) == (0, 0, 0, 0)
    assert kv_map(0, 0, 0, 1) == (0, 0, 0, 0)

    # Test ab_index_map causal branch (below_diag vs skipped with q_seq_index rollover)
    assert ab_map(0, 0, 0, 0) == (0, 0, 0, 0)
    # q_seq_index == 0 == (q_seq_len // block_q) - 1: next_q_index wraps to 0
    assert ab_map(0, 0, 0, 1) == (0, 0, 0, 0)

    # Test kv_segment_ids_index_map causal branch
    assert kv_seg_map(0, 0, 0, 0) == (0, 0, 0)
    assert kv_seg_map(0, 0, 0, 1) == (0, 0, 0)

    # B. Causal=False, save_residuals=False, no ab/seg, block_k == kv_seq_len
    res_noncausal = kernel._flash_attention(
        q,
        k,
        v,
        None,
        None,
        save_residuals=False,
        causal=False,
        sm_scale=1.0,
        block_sizes=BlockSizes(128, 256, 256, 1),
        vmem_limit_bytes=None,
        debug=False,
    )
    assert not isinstance(res_noncausal, tuple)
    assert res_noncausal.shape == q.shape

    # Exercise non-causal segment_ids index map assignment
    kernel._flash_attention(
        q,
        k,
        v,
        None,
        seg,
        save_residuals=False,
        causal=False,
        sm_scale=1.0,
        block_sizes=BlockSizes(128, 256, 256, 1),
        vmem_limit_bytes=None,
        debug=False,
    )

    # Test ab_index_map causal rollover where q_seq_index < max
    q_long = jnp.ones((1, 1, 256, 128), dtype=jnp.bfloat16)
    k_long = jnp.ones((1, 1, 256, 128), dtype=jnp.bfloat16)
    v_long = jnp.ones((1, 1, 256, 128), dtype=jnp.bfloat16)
    ab_long = jnp.zeros((1, 1, 256, 256), dtype=jnp.float32)
    kernel._flash_attention(
        q_long,
        k_long,
        v_long,
        ab_long,
        None,
        save_residuals=False,
        causal=True,
        sm_scale=1.0,
        block_sizes=BlockSizes(128, 128, 128, 1),
        vmem_limit_bytes=None,
        debug=False,
    )
    ab_map_long = captured_calls[-1]["grid_spec"].in_specs[3].index_map
    # q_seq_index = 0, kv_seq_index = 1: should_run is False, next_q_index becomes 1
    assert ab_map_long(0, 0, 0, 1) == (0, 0, 1, 0)


# ---------------------------------------------------------------------------
# 6. Direct execution of Pallas kernel bodies and error branches
# ---------------------------------------------------------------------------


def test_flash_attention_kernel_single_batch_execution(monkeypatch) -> None:
    # Mock Pallas primitives for direct execution
    prog_ids = {2: 0, 3: 0}
    monkeypatch.setattr(pl, "program_id", lambda i: prog_ids.get(i, 0))
    monkeypatch.setattr(
        pl, "when", lambda cond: (lambda fn: fn() if bool(cond) else fn)
    )

    def mock_loop(start, stop, step=1, unroll=True):
        def dec(fn):
            for i in range(start, stop, step):
                fn(i)
            return fn

        return dec

    monkeypatch.setattr(pl, "loop", mock_loop)

    batch_idx = (0, 0)
    q_ref = _MockRef(jnp.ones((1, 1, 128, 128)))
    k_ref = _MockRef(jnp.ones((1, 1, 128, 128)))
    v_ref = _MockRef(jnp.ones((1, 1, 128, 128)))
    ab_ref = _MockRef(jnp.zeros((1, 1, 128, 128)))
    q_seg_ref = _MockRef(jnp.zeros((1, 128, 128), dtype=jnp.int32))
    kv_seg_ref = _MockRef(jnp.zeros((1, 1, 128), dtype=jnp.int32))
    o_ref = _MockRef(jnp.zeros((1, 1, 128, 128)))
    m_ref = _MockRef(jnp.zeros((1, 1, 128, 128)))
    l_ref = _MockRef(jnp.zeros((1, 1, 128, 128)))
    m_scratch = _MockRef(jnp.zeros((1, 1, 128, 128)))
    l_scratch = _MockRef(jnp.zeros((1, 1, 128, 128)))
    acc_scratch = _MockRef(jnp.zeros((1, 1, 128, 128)))

    # 1. Full execution with causal=True, ab, sm_scale != 1.0, segment_ids, store_output
    kernel._flash_attention_kernel_single_batch(
        batch_idx,
        q_ref,
        k_ref,
        v_ref,
        ab_ref,
        q_seg_ref,
        kv_seg_ref,
        o_ref,
        l_ref,
        m_ref,
        m_scratch,
        l_scratch,
        acc_scratch,
        causal=True,
        sm_scale=0.5,
        block_k=128,
        kv_seq_len=128,
        mask_value=kernel.DEFAULT_MASK_VALUE,
    )
    assert o_ref[(0, 0)].shape == (128, 128)

    # 2. Execution with causal=False, l_ref is None, m_ref is None, ab/q_seg is None
    kernel._flash_attention_kernel_single_batch(
        batch_idx,
        q_ref,
        k_ref,
        v_ref,
        None,
        None,
        None,
        o_ref,
        None,
        None,
        m_scratch,
        l_scratch,
        acc_scratch,
        causal=False,
        sm_scale=1.0,
        block_k=128,
        kv_seq_len=128,
        mask_value=kernel.DEFAULT_MASK_VALUE,
    )

    # 3. Head dim not multiple of 128 but < 128 (head_dim_repeats == 0 branch)
    q_small = _MockRef(jnp.ones((1, 1, 128, 64)))
    k_small = _MockRef(jnp.ones((1, 1, 128, 64)))
    v_small = _MockRef(jnp.ones((1, 1, 128, 64)))
    o_small = _MockRef(jnp.zeros((1, 1, 128, 64)))
    acc_small = _MockRef(jnp.zeros((1, 1, 128, 64)))
    kernel._flash_attention_kernel_single_batch(
        batch_idx,
        q_small,
        k_small,
        v_small,
        None,
        None,
        None,
        o_small,
        None,
        None,
        m_scratch,
        l_scratch,
        acc_small,
        causal=False,
        sm_scale=1.0,
        block_k=128,
        kv_seq_len=128,
        mask_value=kernel.DEFAULT_MASK_VALUE,
    )
    assert o_small[(0, 0)].shape == (128, 64)

    # 4. Error: block_k not multiple of MIN_BLOCK_SIZE
    with pytest.raises(NotImplementedError, match="should be a multiple"):
        kernel._flash_attention_kernel_single_batch(
            batch_idx,
            q_ref,
            k_ref,
            v_ref,
            None,
            None,
            None,
            o_ref,
            None,
            None,
            m_scratch,
            l_scratch,
            acc_scratch,
            causal=False,
            sm_scale=1.0,
            block_k=96,
            kv_seq_len=128,
            mask_value=kernel.DEFAULT_MASK_VALUE,
        )

    # 5. Error: head_dim > 128 and not multiple of MIN_BLOCK_SIZE
    q_bad_hd = _MockRef(jnp.ones((1, 1, 128, 192)))
    with pytest.raises(NotImplementedError, match="if larger"):
        kernel._flash_attention_kernel_single_batch(
            batch_idx,
            q_bad_hd,
            k_ref,
            v_ref,
            None,
            None,
            None,
            o_ref,
            None,
            None,
            m_scratch,
            l_scratch,
            acc_scratch,
            causal=False,
            sm_scale=1.0,
            block_k=128,
            kv_seq_len=128,
            mask_value=kernel.DEFAULT_MASK_VALUE,
        )

    # 6. Error: block_k not multiple of NUM_LANES when q_segment_ids present
    monkeypatch.setattr(kernel, "NUM_LANES", 256)
    try:
        with pytest.raises(NotImplementedError, match="must be a multiple of 256"):
            kernel._flash_attention_kernel_single_batch(
                batch_idx,
                q_ref,
                k_ref,
                v_ref,
                None,
                q_seg_ref,
                kv_seg_ref,
                o_ref,
                None,
                None,
                m_scratch,
                l_scratch,
                acc_scratch,
                causal=False,
                sm_scale=1.0,
                block_k=128,
                kv_seq_len=128,
                mask_value=kernel.DEFAULT_MASK_VALUE,
            )
    finally:
        monkeypatch.setattr(kernel, "NUM_LANES", 128)


def test_flash_attention_kernel_single_step_execution(monkeypatch) -> None:
    prog_ids = {2: 0}
    monkeypatch.setattr(pl, "program_id", lambda i: prog_ids.get(i, 0))

    batch_idx = (0, 0)
    q_ref = _MockRef(jnp.ones((1, 1, 128, 128)))
    k_ref = _MockRef(jnp.ones((1, 1, 128, 128)))
    v_ref = _MockRef(jnp.ones((1, 1, 128, 128)))
    ab_ref = _MockRef(jnp.zeros((1, 1, 128, 128)))
    q_seg_ref = _MockRef(jnp.zeros((1, 128, 128), dtype=jnp.int32))
    kv_seg_ref = _MockRef(jnp.zeros((1, 1, 128), dtype=jnp.int32))
    o_ref = _MockRef(jnp.zeros((1, 1, 128, 128)))
    m_ref = _MockRef(jnp.zeros((1, 1, 128, 128)))
    l_ref = _MockRef(jnp.zeros((1, 1, 128, 128)))

    # 1. Full execution with causal=True, ab, sm_scale != 1.0, segment_ids, m_ref, l_ref
    kernel._flash_attention_kernel_single_batch_single_step(
        batch_idx,
        q_ref,
        k_ref,
        v_ref,
        ab_ref,
        q_seg_ref,
        kv_seg_ref,
        o_ref,
        l_ref,
        m_ref,
        causal=True,
        sm_scale=0.5,
        block_k=128,
        kv_seq_len=128,
        mask_value=kernel.DEFAULT_MASK_VALUE,
    )
    assert o_ref[(0, 0)].shape == (128, 128)

    # 2. Execution with causal=False, ab=None, segment_ids=None
    kernel._flash_attention_kernel_single_batch_single_step(
        batch_idx,
        q_ref,
        k_ref,
        v_ref,
        None,
        None,
        None,
        o_ref,
        None,
        None,
        causal=False,
        sm_scale=1.0,
        block_k=128,
        kv_seq_len=128,
        mask_value=kernel.DEFAULT_MASK_VALUE,
    )

    # 3. Error: segment_ids present and block_k not multiple of NUM_LANES
    k_bad = _MockRef(jnp.ones((1, 1, 64, 128)))
    with pytest.raises(NotImplementedError, match="must be a multiple of 128"):
        kernel._flash_attention_kernel_single_batch_single_step(
            batch_idx,
            q_ref,
            k_bad,
            v_ref,
            None,
            q_seg_ref,
            kv_seg_ref,
            o_ref,
            None,
            None,
            causal=False,
            sm_scale=1.0,
            block_k=64,
            kv_seq_len=64,
            mask_value=kernel.DEFAULT_MASK_VALUE,
        )


def test_flash_attention_kernel_wrapper(monkeypatch) -> None:
    single_step_called = False
    multi_step_called = False

    def mock_single_step(*args, **kwargs):
        nonlocal single_step_called
        single_step_called = True

    def mock_multi_step(*args, **kwargs):
        nonlocal multi_step_called
        multi_step_called = True

    monkeypatch.setattr(
        kernel,
        "_flash_attention_kernel_single_batch_single_step",
        mock_single_step,
    )
    monkeypatch.setattr(kernel, "_flash_attention_kernel_single_batch", mock_multi_step)

    q_ref = _MockRef(jnp.ones((2, 1, 128, 128)))

    # Case 1: block_k == kv_seq_len uses single step
    kernel._flash_attention_kernel(q_ref, block_k=128, kv_seq_len=128)
    assert single_step_called is True

    # Case 2: block_k != kv_seq_len uses multi step
    kernel._flash_attention_kernel(q_ref, block_k=64, kv_seq_len=128)
    assert multi_step_called is True


# ---------------------------------------------------------------------------
# 7. MHA Reference functions
# ---------------------------------------------------------------------------


def test_mha_reference_functions() -> None:
    q = jnp.ones((1, 1, 16, 32), dtype=jnp.float32)
    k = jnp.ones((1, 1, 16, 32), dtype=jnp.float32)
    v = jnp.ones((1, 1, 16, 32), dtype=jnp.float32)
    ab = jnp.zeros((1, 1, 16, 16), dtype=jnp.float32)
    seg = SegmentIds(
        q=jnp.zeros((1, 16), dtype=jnp.int32),
        kv=jnp.zeros((1, 16), dtype=jnp.int32),
    )

    # 1. mha_reference_no_custom_vjp: all options active
    out, l_res, m = mha_reference_no_custom_vjp(
        q,
        k,
        v,
        ab=ab,
        segment_ids=seg,
        causal=True,
        sm_scale=0.5,
        save_residuals=True,
    )
    assert out.shape == q.shape
    assert l_res.shape == (1, 1, 16)
    assert m.shape == (1, 1, 16)

    # 2. mha_reference_no_custom_vjp: options disabled (save_residuals=False)
    out_simple = mha_reference_no_custom_vjp(
        q, k, v, ab=None, segment_ids=None, causal=False, sm_scale=1.0
    )
    assert cast(Any, out_simple).shape == q.shape

    # 3. mha_reference JIT wrapper & _mha_reference
    out_ref = mha_reference(q, k, v, ab, seg, causal=True, sm_scale=0.5)
    assert out_ref.shape == q.shape

    # 4. Direct get_tuned_params lookup (miss & hit)
    tkey = kernel.make_tuning_key(
        q,
        k,
        v,
        causal=True,
        has_segment_ids=False,
        has_attention_bias=False,
        vmem_limit_bytes=None,
        device_kind="TPU v5",
    )
    assert kernel.get_tuned_params(tkey) is None
