"""CPU contracts for the fused sliding-window SparseMLA decode path.

Targets:
- kernels/deepseek_v4/core_attention/sparse_mla.py
  (`_attention_kernel_fused_swa`, `fused_swa_attention_call`)
- models/vllm/deepseek_v41/v2/attention_kernels.py
  (`fused_swa_eligible`, `fused_swa_inputs`, the fused branch of
  `sparse_ragged_paged_attention_union_gather`, `sparse_fp8_attention_jax`
  routing, the ``fused_swa`` op and wrapper plumbing)
- envs.py (``TPU_DEEPSEEK_V41_FUSED_SWA``)

The kernel body runs on CPU against mock HBM/VMEM refs whose async copies
really move data, so its output is checked against a plain-JAX softmax over
the concatenated ``[window | top-k]`` keys, not only for coverage.
"""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.core_attention import sparse_mla
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    gather_cache_values,
    split_halves,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

HEAD_DIM = 512
NOPE_DIM = sparse_mla.NOPE_DIM
ROPE_DIM = HEAD_DIM - NOPE_DIM


# ==============================================================================
# Mock Pallas refs with real data movement
# ==============================================================================


def _as_index(idx):
    if isinstance(idx, tuple):
        return tuple(_as_index(i) for i in idx)
    if hasattr(idx, "start") and hasattr(idx, "size"):
        start = int(np.asarray(idx.start))
        size = int(np.asarray(idx.size))
        return slice(start, start + size)
    if isinstance(idx, jax.Array) and idx.ndim == 0:
        return int(idx)
    return idx


class _Ref:
    """numpy-backed ref; views are shared so DMA copies and stores persist."""

    def __init__(self, arr):
        self.arr = arr

    @property
    def shape(self):
        return self.arr.shape

    @property
    def dtype(self):
        return self.arr.dtype

    def __getitem__(self, idx):
        return jnp.asarray(self.arr[_as_index(idx)])

    def __setitem__(self, idx, val):
        self.arr[_as_index(idx)] = np.asarray(val)

    @property
    def at(self):
        return _At(self)

    def bitcast(self, dtype):
        # Pallas TPU bitcast keeps the lane axis and scales the sublane axis
        # by the element-size ratio.
        ratio = self.arr.dtype.itemsize // np.dtype(dtype).itemsize
        *lead, rows, lanes = self.arr.shape
        return _Ref(self.arr.view(dtype).reshape(*lead, rows * ratio, lanes))


class _At:
    def __init__(self, ref):
        self.ref = ref

    def __getitem__(self, idx):
        return _Ref(self.ref.arr[_as_index(idx)])


class _Copy:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

    def start(self):
        assert self.src.arr.shape == self.dst.arr.shape
        np.copyto(self.dst.arr, self.src.arr)

    def wait(self):
        pass


def _when(cond):
    def decorator(fn):
        if bool(cond):
            fn()
        return fn

    return decorator


def _bitcast(x, dtype):
    arr = x.arr if isinstance(x, _Ref) else np.asarray(x)
    return jnp.asarray(arr.view(dtype))


def _kernel_patches(program_id):
    return (
        mock.patch.object(
            sparse_mla.pltpu,
            "make_async_copy",
            side_effect=lambda src, dst, sem: _Copy(src, dst),
        ),
        mock.patch.object(sparse_mla.pltpu, "bitcast", side_effect=_bitcast),
        mock.patch.object(
            sparse_mla.pltpu, "roll", side_effect=lambda x, s, a: jnp.roll(x, s, a)
        ),
        mock.patch.object(
            sparse_mla.pl, "reciprocal", side_effect=lambda x, approx=True: 1.0 / x
        ),
        mock.patch.object(sparse_mla.pl, "when", side_effect=_when),
        mock.patch.object(sparse_mla.pl, "program_id", side_effect=program_id),
    )


# ==============================================================================
# Test data
# ==============================================================================


def _fp8_rows(rng, tokens, topk):
    """Packed NoPE words ``int32[tokens, topk, 128]`` and their BF16 values."""
    values = rng.standard_normal((tokens, topk, NOPE_DIM)).astype(np.float32)
    fp8 = values.astype(jnp.float8_e4m3fn)
    # The 64 trailing bytes hold one e8m0 scale per element modulo 64.
    scale_bytes = np.full((tokens, topk, 64), 127, np.uint8)  # 2**0
    scale_bytes[..., :32] = 128  # 2**1 for the first half of each period
    row_bytes = np.concatenate([fp8.view(np.uint8), scale_bytes], axis=-1)
    words = np.ascontiguousarray(row_bytes).view(np.int32)
    return words.reshape(tokens, topk, 128)


def _rope_words(rope):
    """``bf16[tokens, topk, 64]`` -> ``int32[tokens, topk // 4, 128]``."""
    tokens, topk, _ = rope.shape
    half = topk // 2
    flat = np.concatenate([rope[:, :half], rope[:, half:]], axis=-1)
    return np.ascontiguousarray(flat).view(np.int32).reshape(tokens, topk // 4, 128)


def _dequant_rows(words, rope):
    """The ``[tokens, topk, HEAD_DIM]`` keys `load_bkv` sees for ``words``."""
    tokens, topk, _ = words.shape
    with (
        mock.patch.object(sparse_mla.pltpu, "bitcast", side_effect=_bitcast),
        mock.patch.object(
            sparse_mla.pltpu, "roll", side_effect=lambda x, s, a: jnp.roll(x, s, a)
        ),
    ):
        nope = jnp.stack(
            [
                sparse_mla._dequant_dsv4_fp8(
                    jnp.asarray(words[t].view(np.uint8).reshape(topk, 512))
                )
                for t in range(tokens)
            ]
        )
    return jnp.concatenate([nope, jnp.asarray(rope)], axis=-1).astype(jnp.float32)


def _reference(q, keys, kv_lens, swa_kv, swa_bias, sinks, scale):
    """Plain softmax over ``[window | top-k]`` with the kernel's roundings.

    ``swa_bias`` is 2-D ``[tokens, window]``; the kernel takes it as
    ``[tokens, window // 128, 128]`` (see `_kernel_bias`).
    """
    q = q.astype(jnp.float32)
    swa = swa_kv.astype(jnp.float32)
    k_span = jnp.arange(keys.shape[1])[None]
    keys = jnp.where((k_span < kv_lens[:, None])[..., None], keys, 0)
    s_swa = jnp.einsum("thd,twd->thw", q, swa) * scale + swa_bias[:, None]
    s_kv = jnp.einsum("thd,tkd->thk", q, keys) * scale
    s_kv = jnp.where(
        (k_span < kv_lens[:, None])[:, None], s_kv, jnp.finfo(jnp.float32).min
    )
    s = jnp.concatenate([s_swa, s_kv], axis=-1)
    m = s.max(-1, keepdims=True)
    p = jnp.exp(s - m)
    denominator = p.sum(-1) + jnp.exp(sinks[None] - m[..., 0])
    p = p.astype(jnp.bfloat16).astype(jnp.float32)
    window = swa.shape[1]
    acc = jnp.einsum("thw,twd->thd", p[..., :window], swa)
    acc += jnp.einsum("thk,tkd->thd", p[..., window:], keys)
    return acc / denominator[..., None]


def _kernel_bias(swa_bias):
    """The kernel's lane-tiled bias layout, one sublane row per 128 columns."""
    tokens, window = swa_bias.shape
    return jnp.asarray(swa_bias, jnp.float32).reshape(tokens, window // 128, 128)


def _run_fused_kernel(
    q, nope_words, rope_words, swa_kv, swa_bias, kv_lens, sinks, *, scale, batch_size
):
    """Drive `_attention_kernel_fused_swa` over its whole grid on CPU."""
    tokens, heads, dim = q.shape
    topk = nope_words.shape[1]
    window = swa_kv.shape[1]
    swa_bias = _kernel_bias(swa_bias)
    valid = int(jnp.sum(kv_lens >= 0))
    grid = -(-tokens // batch_size)
    out = np.zeros(q.shape, q.dtype)
    refs = dict(
        kv_lens_ref=_Ref(np.asarray(kv_lens, np.int32)),
        start_end_seq_idx_ref=_Ref(np.array([0, tokens, valid], np.int32)),
        sem_ids_ref=_Ref(np.zeros((2,), np.int32)),
        bo_ids_ref=_Ref(np.full((2,), -1, np.int32)),
        attention_sinks_ref=_Ref(np.asarray(sinks, np.float32)),
        q_hbm_ref=_Ref(np.asarray(q)),
        cache_kv_nope_hbm_ref=_Ref(nope_words),
        cache_kv_rope_hbm_ref=_Ref(rope_words),
        swa_kv_hbm_ref=_Ref(np.asarray(swa_kv)),
        swa_bias_hbm_ref=_Ref(np.asarray(swa_bias, np.float32)),
        o_hbm_ref=_Ref(out),
        bkv_nope_x2_ref=_Ref(np.zeros((2, batch_size, topk * 4, 128), np.uint8)),
        bkv_rope_x2_ref=_Ref(np.zeros((2, batch_size, topk // 2, 128), jnp.bfloat16)),
        bq_x2_ref=_Ref(np.zeros((2, batch_size, heads, dim), q.dtype)),
        bo_x2_ref=_Ref(np.zeros((2, batch_size, heads, dim), q.dtype)),
        swa_kv_x2_ref=_Ref(np.zeros((2, batch_size, window, dim), jnp.bfloat16)),
        swa_bias_x2_ref=_Ref(np.zeros((2, batch_size, window // 128, 128), np.float32)),
        sems=_Ref(np.zeros((6, 2), np.int32)),
    )
    for step in range(grid):
        patches = _kernel_patches(lambda axis, step=step: step)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            mock.patch.object(sparse_mla.pl, "num_programs", return_value=grid),
        ):
            sparse_mla._attention_kernel_fused_swa(
                *refs.values(), sm_scale=scale, batch_size=batch_size
            )
    return jnp.asarray(out)


# ==============================================================================
# 1. Kernel body numerics
# ==============================================================================


class TestFusedSwaKernel:
    @pytest.mark.parametrize(
        ("q_dtype", "tokens"),
        [(jnp.float32, 8), (jnp.bfloat16, 8), (jnp.float32, 12)],
    )
    def test_matches_joint_softmax(self, q_dtype, tokens):
        rng = np.random.default_rng(0)
        heads, topk, window, batch_size = 8, 16, 128, 4
        scale = 0.1
        q = jnp.asarray(rng.standard_normal((tokens, heads, HEAD_DIM)), q_dtype)
        nope_words = _fp8_rows(rng, tokens, topk)
        rope = jnp.asarray(rng.standard_normal((tokens, topk, ROPE_DIM)), jnp.bfloat16)
        rope_words = _rope_words(np.asarray(rope))
        keys = _dequant_rows(nope_words, rope)
        swa_kv = rng.standard_normal((tokens, window, HEAD_DIM)).astype(np.float32)
        live = np.ones((tokens, window), bool)
        live[:, 100:] = False  # padded -1 records
        live[3] = False  # a token with no live window row at all
        swa_kv[~live] = 0  # `fused_swa_inputs` zeroes dead rows
        swa_kv = jnp.asarray(swa_kv, jnp.bfloat16)
        swa_bias = jnp.where(live, 0.0, jnp.finfo(jnp.float32).min)
        kv_lens = jnp.asarray(
            ([topk, 1, 5, topk, 9, topk, 2, topk] * 2)[:tokens], jnp.int32
        )
        sinks = jnp.asarray(rng.standard_normal(heads), jnp.float32)

        out = _run_fused_kernel(
            q,
            nope_words,
            rope_words,
            swa_kv,
            swa_bias,
            kv_lens,
            sinks,
            scale=scale,
            batch_size=batch_size,
        )
        want = _reference(q, keys, kv_lens, swa_kv, swa_bias, sinks, scale)
        assert out.dtype == q_dtype
        assert not jnp.isnan(out).any()
        tol = 1e-3 if q_dtype == jnp.float32 else 2e-2
        np.testing.assert_allclose(out.astype(jnp.float32), want, rtol=tol, atol=tol)

    def test_partial_last_batch_writes_valid_rows_only(self):
        rng = np.random.default_rng(1)
        tokens, heads, topk, window, batch_size = 6, 8, 16, 128, 3
        q = jnp.asarray(rng.standard_normal((tokens, heads, HEAD_DIM)), jnp.float32)
        nope_words = _fp8_rows(rng, tokens, topk)
        rope = jnp.asarray(rng.standard_normal((tokens, topk, ROPE_DIM)), jnp.bfloat16)
        rope_words = _rope_words(np.asarray(rope))
        keys = _dequant_rows(nope_words, rope)
        swa_kv = jnp.asarray(
            rng.standard_normal((tokens, window, HEAD_DIM)), jnp.bfloat16
        )
        swa_bias = jnp.zeros((tokens, window), jnp.float32)
        # Tokens 5 is padding: ``kv_lens`` -1 marks it beyond ``num_valid``.
        kv_lens = jnp.asarray([topk, 4, topk, topk, 7, -1], jnp.int32)
        sinks = jnp.zeros((heads,), jnp.float32)

        out = _run_fused_kernel(
            q,
            nope_words,
            rope_words,
            swa_kv,
            swa_bias,
            kv_lens,
            sinks,
            scale=0.05,
            batch_size=batch_size,
        )
        want = _reference(q, keys, kv_lens, swa_kv, swa_bias, sinks, 0.05)
        np.testing.assert_allclose(out[:5], want[:5], rtol=1e-3, atol=1e-3)
        assert jnp.all(out[5] == 0)  # never sent to HBM

    def test_rejects_mismatched_shapes(self):
        q = _Ref(np.zeros((4, 8, HEAD_DIM), np.float32))
        good = dict(
            kv_lens_ref=_Ref(np.zeros((4,), np.int32)),
            start_end_seq_idx_ref=_Ref(np.array([0, 4, 4], np.int32)),
            sem_ids_ref=_Ref(np.zeros((2,), np.int32)),
            bo_ids_ref=_Ref(np.full((2,), -1, np.int32)),
            attention_sinks_ref=_Ref(np.zeros((8,), np.float32)),
            q_hbm_ref=q,
            cache_kv_nope_hbm_ref=_Ref(np.zeros((4, 16, 128), np.int32)),
            cache_kv_rope_hbm_ref=_Ref(np.zeros((4, 4, 128), np.int32)),
            swa_kv_hbm_ref=_Ref(np.zeros((4, 128, HEAD_DIM), jnp.bfloat16)),
            swa_bias_hbm_ref=_Ref(np.zeros((4, 1, 128), np.float32)),
            o_hbm_ref=q,
        )
        scratch = [None] * 7
        bad = dict(good, swa_bias_hbm_ref=_Ref(np.zeros((4, 128), np.float32)))
        with pytest.raises(AssertionError):
            sparse_mla._attention_kernel_fused_swa(
                *bad.values(), *scratch, sm_scale=1.0, batch_size=4
            )


# ==============================================================================
# 2. Kernel builder
# ==============================================================================


class TestFusedSwaAttentionCall:
    def test_builds_pallas_call(self):
        captured = {}

        def fake_pallas_call(kernel, **kwargs):
            captured.update(kwargs)
            captured["kernel"] = kernel
            return lambda *args: args[5]  # aliased q

        tokens, heads, topk, window = 24, 8, 128, 256
        q = jnp.zeros((tokens, heads, HEAD_DIM), jnp.bfloat16)
        with mock.patch.object(
            sparse_mla.pl, "pallas_call", side_effect=fake_pallas_call
        ):
            out = sparse_mla.fused_swa_attention_call(
                q,
                jnp.zeros((tokens, topk, 128), jnp.int32),
                jnp.zeros((tokens, topk // 4, 128), jnp.int32),
                jnp.zeros((tokens, window, HEAD_DIM), jnp.bfloat16),
                jnp.zeros((tokens, window // 128, 128), jnp.float32),
                jnp.zeros((tokens,), jnp.int32),
                jnp.zeros((heads,), jnp.float32),
                start_seq_idx=0,
                end_seq_idx=jnp.int32(tokens),
                num_valid_tokens=jnp.int32(tokens),
                kernel_batch_size=8,
                sm_scale=0.5,
            )
        assert out is q
        assert captured["input_output_aliases"] == {5: 0}
        assert captured["name"] == f"SparseMLA-fswa_{window}-p_{topk}-bz_8-gcz_{tokens}"
        assert captured["out_shape"].shape == q.shape
        assert captured["kernel"].keywords == {"sm_scale": 0.5, "batch_size": 8}
        grid_spec = captured["grid_spec"]
        assert grid_spec.num_scalar_prefetch == 4
        assert len(grid_spec.in_specs) == 6
        scratch = grid_spec.scratch_shapes
        assert len(scratch) == 7
        assert scratch[4].shape == (2, 8, window, HEAD_DIM)
        assert scratch[5].shape == (2, 8, window // 128, 128)
        assert int(grid_spec.grid[0]) == 3


# ==============================================================================
# 3. Driver helpers and routing
# ==============================================================================


class TestFusedSwaHelpers:
    @pytest.mark.parametrize(
        ("tokens", "width", "max_requests", "want"),
        [
            (24, 192, 48, True),  # decode verify batch, tokens < 128
            (200, 128, 48, True),  # not a 64 multiple
            (192, 128, None, True),  # DSpark verify multiple of 6 without bound
            (128, 192, None, True),  # power-of-two multi-request verify shard (w!=128)
            (256, 192, None, True),  # B=2048 multi-request verify shard (w!=128)
            (256, 128, None, False),  # 2048-token prefill chunk keeps shared union
            (192, 128, 48, False),  # prefill tile with bound: shared union
            (128, 64, 48, False),
            (24, 96, 48, True),  # width not a 64 multiple still falls back
            (24, 640, 48, False),  # over the 512 lane budget
            (24, 0, 48, False),  # no window at all
        ],
    )
    def test_fused_swa_eligible(self, tokens, width, max_requests, want):
        assert attention_kernels.fused_swa_eligible(tokens, width, max_requests) is want

    def test_fused_swa_eligible_tracks_attend_window(self):
        """Eligible exactly where `_attend_window` would call `_attend`."""
        pool = jnp.zeros((64, 128), jnp.int32)
        state = attention_kernels._attention_state(jnp.zeros((1, 1, 64)))
        for tokens, width, max_requests in [
            (24, 192, 48),
            (128, 64, 48),
            (192, 128, None),
            (256, 192, None),
            (256, 128, None),
            (128, 512, 4),
        ]:
            q = jnp.zeros((tokens, 1, 64), jnp.bfloat16)
            records = jnp.zeros((tokens, width), jnp.int32)
            with (
                mock.patch.object(
                    attention_kernels, "_attend", return_value=state
                ) as attend,
                mock.patch.object(
                    attention_kernels,
                    "_window_union",
                    side_effect=RuntimeError("union"),
                ),
                contextlib.suppress(RuntimeError),
            ):
                attention_kernels._attend_window(
                    q,
                    pool,
                    records,
                    attention_kernels._attention_state(q),
                    1.0,
                    max_requests=max_requests,
                )
            assert attention_kernels.fused_swa_eligible(
                tokens, width, max_requests
            ) is bool(attend.called)

    def test_fused_swa_inputs(self):
        rng = np.random.default_rng(2)
        dim = 256
        rows = jnp.asarray(rng.standard_normal((40, dim)), jnp.bfloat16)
        # Poison the rows a -1 record may read (clamped or wrapped index).
        rows = rows.at[0].set(jnp.nan).at[-1].set(jnp.nan)
        # Pack BF16 pairs into int32 words: element 2i low, element 2i + 1 high.
        bits = jax.lax.bitcast_convert_type(rows, jnp.uint16).astype(jnp.uint32)
        pool = (bits[:, 0::2] | (bits[:, 1::2] << 16)).astype(jnp.int32)
        records = jnp.asarray(
            [[3, 7, -1, 11] + [-1] * 60, [1] * 64],
            jnp.int32,
        )
        kv, bias = attention_kernels.fused_swa_inputs(pool, records, dim)
        assert kv.shape == (2, 128, dim) and kv.dtype == jnp.bfloat16
        assert bias.shape == (2, 1, 128) and bias.dtype == jnp.float32
        assert jnp.array_equal(kv[0, 0], rows[3])
        assert jnp.array_equal(kv[0, 1], rows[7])
        assert jnp.array_equal(kv[0, 3], rows[11])
        assert jnp.array_equal(kv[1, :64], jnp.broadcast_to(rows[1], (64, dim)))
        padded = jnp.pad(records, ((0, 0), (0, 64)), constant_values=-1)
        want_bias = jnp.where(padded >= 0, 0.0, jnp.finfo(jnp.float32).min)
        assert jnp.array_equal(bias[:, 0], want_bias)
        # Dead records (-1 and the pad) are zero rows, whatever the pool holds.
        assert jnp.all(kv[0, 2] == 0) and jnp.all(kv[0, 4:] == 0)
        assert jnp.all(kv[1, 64:] == 0)
        assert not jnp.isnan(kv).any()
        # Natural order agrees with the lane-gather reader as well.
        natural = gather_cache_values(pool, records[0, :2], dim)
        assert jnp.array_equal(kv[0, :2], natural)
        assert not jnp.array_equal(split_halves(natural), natural)

    @pytest.mark.parametrize("tokens", [50, 96, 144, 192])
    def test_fused_swa_inputs_two_banks_and_wide_window(self, tokens):
        dim = 256
        pool = jnp.arange(32 * 128, dtype=jnp.int32).reshape(32, 128)
        banks = (pool[:16], pool[16:])
        records = jnp.full((tokens, 200), 20, jnp.int32)
        with mock.patch.object(
            attention_kernels,
            "gather_cache_values",
            wraps=attention_kernels.gather_cache_values,
        ) as spy_gather:
            kv, bias = attention_kernels.fused_swa_inputs(banks, records, dim)
        assert spy_gather.call_args.args[1].shape == (tokens * 200,)
        assert kv.shape == (tokens, 256, dim)
        assert bias.shape == (tokens, 2, 128)
        flat = bias.reshape(tokens, 256)
        assert jnp.all(flat[:, :200] == 0)
        assert jnp.all(flat[:, 200:] == jnp.finfo(jnp.float32).min)
        assert jnp.all(kv[:, 200:] == 0)
        single_kv, _ = attention_kernels.fused_swa_inputs(pool, records, dim)
        assert jnp.array_equal(kv, single_kv)


class TestFusedSwaDriver:
    def _inputs(self, tokens=24, heads=4, topk=16, window=128):
        return dict(
            q=jnp.ones((tokens, heads, 64), jnp.bfloat16),
            cache_nope=jnp.ones((4, 16, 64), jnp.uint8),
            cache_rope=jnp.ones((4, 4, 128), jnp.int32),
            topk_indices=jnp.zeros((tokens, topk), jnp.int32),
            page_indices=jnp.zeros((4,), jnp.int32),
            cu_q_lens=jnp.array([0, tokens], jnp.int32),
            distribution=jnp.array([0, 0, 1], jnp.int32),
            sinks=jnp.zeros((heads,), jnp.float32),
            swa_kv=jnp.ones((tokens, window, 64), jnp.bfloat16),
            swa_bias=jnp.zeros((tokens, window // 128, 128), jnp.float32),
        )

    @pytest.mark.parametrize("two_bank", [False, True])
    def test_union_gather_fused_branch(self, two_bank):
        i = self._inputs()
        calls = []

        def fake_fused(q, nope, rope, swa_kv, swa_bias, kv_lens, sinks, **kw):
            calls.append((q.shape, swa_kv.shape, swa_bias.shape, kw))
            return q

        nope = (
            (i["cache_nope"][:2], i["cache_nope"][2:]) if two_bank else i["cache_nope"]
        )
        rope = (
            (i["cache_rope"][:2], i["cache_rope"][2:]) if two_bank else i["cache_rope"]
        )
        with (
            mock.patch.object(
                sparse_mla, "fused_swa_attention_call", side_effect=fake_fused
            ),
            mock.patch("jax.experimental.pallas.pallas_call") as pallas_call,
        ):
            out = attention_kernels.sparse_ragged_paged_attention_union_gather(
                i["q"],
                nope,
                rope,
                i["topk_indices"],
                i["page_indices"],
                i["cu_q_lens"],
                i["distribution"],
                i["sinks"],
                None,
                None,
                None,
                sm_scale=0.25,
                union_gather=not two_bank,
                gather_and_attention_chunk_size=16,
                swa_kv=i["swa_kv"],
                swa_bias=i["swa_bias"],
            )
        assert out.shape == i["q"].shape
        assert not pallas_call.called
        assert len(calls) == 2  # 24 tokens in 16-token chunks
        q_shape, kv_shape, bias_shape, kw = calls[0]
        assert q_shape[-1] == 128  # padded head dim
        assert kv_shape == (24, 128, 128)  # window rows padded alongside q
        assert bias_shape == (24, 1, 128)
        assert kw["sm_scale"] == 0.25 and kw["kernel_batch_size"] == 16
        assert calls[1][3]["start_seq_idx"] == 16
        assert calls[1][3]["kernel_batch_size"] == 8

    def test_union_gather_requires_both_fused_inputs(self):
        i = self._inputs()
        with pytest.raises(AssertionError):
            attention_kernels.sparse_ragged_paged_attention_union_gather(
                i["q"],
                i["cache_nope"],
                i["cache_rope"],
                i["topk_indices"],
                i["page_indices"],
                i["cu_q_lens"],
                i["distribution"],
                i["sinks"],
                None,
                None,
                None,
                swa_kv=i["swa_kv"],
            )
        with pytest.raises(AssertionError):
            attention_kernels.sparse_ragged_paged_attention_union_gather(
                i["q"],
                i["cache_nope"],
                i["cache_rope"],
                i["topk_indices"],
                i["page_indices"],
                i["cu_q_lens"],
                i["distribution"],
                i["sinks"],
                None,
                None,
                None,
            )

    def _jax_args(self, tokens, width):
        return dict(
            q=jnp.ones((tokens, 4, 64), jnp.bfloat16),
            swa_pool=jnp.zeros((256, 128), jnp.int32),
            main_pool=jnp.ones((256, 128), jnp.int32),
            rope_pool=jnp.ones((64, 128), jnp.int32),
            swa_records=jnp.zeros((tokens, width), jnp.int32),
            main_records=jnp.zeros((tokens, 128), jnp.int32),
            pages=jnp.zeros((1, 1), jnp.int32),
            query_start_loc=jnp.array([0, tokens], jnp.int32),
            sink=jnp.zeros((4,), jnp.float32),
        )

    def test_sparse_fp8_attention_jax_routes_eligible_shapes(self):
        for tokens, want_chunk in ((24, 128), (144, 144), (168, 168), (192, 192)):
            a = self._jax_args(tokens=tokens, width=64)
            with (
                mock.patch.object(
                    attention_kernels,
                    "sparse_ragged_paged_attention_union_gather",
                    return_value=a["q"],
                ) as union,
                mock.patch.object(
                    attention_kernels, "_attend_window", side_effect=AssertionError
                ),
            ):
                for union_gather in (False, True):
                    out = attention_kernels.sparse_fp8_attention_jax(
                        *a.values(),
                        scale=0.125,
                        page_size=16,
                        max_requests=None if tokens == 192 else 48,
                        union_gather=union_gather,
                        fused_swa=True,
                    )
                    assert out is a["q"]
                    kw = union.call_args.kwargs
                    assert kw["union_gather"] is union_gather
                    assert kw["gather_and_attention_chunk_size"] == want_chunk
                    assert kw["swa_kv"].shape == (tokens, 128, 64)
                    assert kw["swa_bias"].shape == (tokens, 1, 128)
                    assert union.call_args.args[8:11] == (None, None, None)

    def test_sparse_fp8_attention_jax_keeps_prefill_path(self):
        for tokens, want_chunk in ((128, 128), (192, 192), (256, 128)):
            for union_gather in (False, True):
                a = self._jax_args(tokens=tokens, width=64)
                with (
                    mock.patch(
                        "vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla.sparse_ragged_paged_attention",
                        return_value=a["q"],
                    ) as plain,
                    mock.patch.object(
                        attention_kernels,
                        "sparse_ragged_paged_attention_union_gather",
                        return_value=a["q"],
                    ) as union,
                    mock.patch.object(
                        attention_kernels,
                        "fused_swa_inputs",
                        side_effect=AssertionError,
                    ),
                ):
                    out = attention_kernels.sparse_fp8_attention_jax(
                        *a.values(),
                        scale=0.125,
                        page_size=16,
                        max_requests=48,
                        union_gather=union_gather,
                        fused_swa=True,
                    )
                assert out is a["q"]
                if union_gather:
                    union.assert_called_once()
                    assert (
                        union.call_args.kwargs["gather_and_attention_chunk_size"]
                        == want_chunk
                    )
                else:
                    plain.assert_called_once()
                    assert (
                        plain.call_args.kwargs["gather_and_attention_chunk_size"]
                        == want_chunk
                    )


# ==============================================================================
# 4. Op and torch wrapper plumbing, env flag
# ==============================================================================


class TestFusedSwaPlumbing:
    def test_fp8_attention_ops_forward_fused_swa(self):
        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        captured = {}

        def mock_jax_op(name, fn, **kwargs):
            captured[name] = (fn, kwargs["input_partition_specs"])
            return mock.MagicMock()

        tp_group = mock.MagicMock()
        tp_group.size.return_value = 2
        with (
            set_vllm_model_wrapper_context(mesh=mesh),
            mock.patch(
                "vllm.distributed.get_tp_group",
                return_value=SimpleNamespace(device_group=tp_group),
            ),
            mock.patch("torch_tpu._internal.pallas.get_pallas_mesh", return_value=mesh),
            mock.patch("torch_tpu._internal.pallas.jax_op", side_effect=mock_jax_op),
        ):
            attention_kernels._attention_op = None
            attention_kernels._fp8_attention_op = None
            attention_kernels._fp8_attention_qo_op = None
            attention_kernels.initialize_paged_attention()
        attention_kernels._attention_op = None

        fn, specs = captured["pallas::dsv41v2_sparse_fp8_attention"]
        assert specs[-8:] == (None,) * 8
        fn_2b, specs_2b = captured["pallas::dsv41v2_sparse_fp8_attention_2bank"]
        assert specs_2b[-8:] == (None,) * 8 and len(specs_2b) == 22

        seen = []

        def fake_jax(q, *args, **kw):
            seen.append(kw["fused_swa"])
            return q

        tensors = lambda tokens: (  # noqa: E731
            jnp.zeros((tokens, 4, 64)),
            jnp.zeros((256, 64)),
            jnp.zeros((256, 64)),
            jnp.zeros((64, 128)),
            jnp.zeros((tokens, 64), jnp.int32),
            jnp.zeros((tokens, 128), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((2,), jnp.int32),
            jnp.zeros((4,)),
            jnp.zeros((4,)),
            jnp.zeros((1,), jnp.int32),
        )
        tail = (0.125, 16, None, False)
        with (
            mock.patch("jax.shard_map", side_effect=lambda fn, **kw: lambda *a: fn(*a)),
            mock.patch("jax.lax.all_to_all", side_effect=lambda x, **kw: x),
            mock.patch.object(
                attention_kernels, "sparse_fp8_attention_jax", side_effect=fake_jax
            ),
        ):
            fn(*tensors(256), *tail, True, False, 4096, True)  # token parallel
            fn(*tensors(64), *tail, False, False, 4096, True)
            fn(*tensors(64), *tail, False, False, 4096)  # default
            t = tensors(64)
            fn_2b(
                t[0],
                t[1],
                t[1],
                t[2],
                t[2],
                t[3],
                t[3],
                *t[4:],
                *tail,
                False,
                False,
                4096,
                True,
            )
        assert seen == [True, True, False, True]

    def test_torch_wrapper_appends_fused_swa(self, monkeypatch):
        op = mock.MagicMock(return_value=torch.zeros(4, 2, 64))
        op_2b = mock.MagicMock(return_value=torch.zeros(4, 2, 64))
        monkeypatch.setattr(attention_kernels, "_fp8_attention_op", op)
        monkeypatch.setattr(attention_kernels, "_fp8_attention_2bank_op", op_2b)
        raw = torch.zeros(16, 128, dtype=torch.int32)
        rope = torch.zeros(16, 32, dtype=torch.int32)
        swa = SimpleNamespace(raw=raw, raw_hi=None, rope=None, rope_hi=None)
        main = SimpleNamespace(
            raw=raw, raw_hi=None, rope=rope, rope_hi=None, shape=(1, 16), total_rows=16
        )
        common = (
            torch.zeros(4, 2, 64),
            swa,
            main,
            torch.zeros(4, 64, dtype=torch.int32),
            torch.zeros(4, 128, dtype=torch.int32),
            torch.zeros(2),
            0.125,
        )
        kwargs = dict(
            pages=torch.zeros((1, 1), dtype=torch.int32),
            starts=torch.tensor([0, 4], dtype=torch.int32),
        )
        attention_kernels.sparse_fp8_attention(*common, **kwargs)
        assert op.call_args.args[-1] is False
        attention_kernels.sparse_fp8_attention(*common, fused_swa=True, **kwargs)
        assert op.call_args.args[-1] is True
        assert op.call_args.args[-2] == 4096

        swa_2b = SimpleNamespace(raw=raw, raw_hi=raw, rope=None, rope_hi=None)
        main_2b = SimpleNamespace(
            raw=raw, raw_hi=raw, rope=rope, rope_hi=rope, shape=(1, 16), total_rows=32
        )
        with mock.patch.object(attention_kernels, "_has_hi_bank", return_value=True):
            attention_kernels.sparse_fp8_attention(
                common[0], swa_2b, main_2b, *common[3:], fused_swa=True, **kwargs
            )
        assert op_2b.call_args.args[-1] is True

    def test_env_flag(self, monkeypatch):
        monkeypatch.delenv("TPU_DEEPSEEK_V41_FUSED_SWA", raising=False)
        resolver = envs.environment_variables["TPU_DEEPSEEK_V41_FUSED_SWA"]
        assert resolver() is True
        monkeypatch.setenv("TPU_DEEPSEEK_V41_FUSED_SWA", "0")
        assert resolver() is False
