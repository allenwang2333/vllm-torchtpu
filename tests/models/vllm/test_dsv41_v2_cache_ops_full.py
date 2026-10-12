"""Unit and bit-exactness tests for DeepSeek-V4.1 v2 cache.py and ops.py."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


class _StubModule(ModuleType):
    __path__ = []  # type: ignore[assignment]

    def __getattr__(self, name: str):
        return MagicMock()


for _mod_name in (
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.models",
    "vllm.model_executor.models.config",
    "vllm.v1",
    "vllm.v1.kv_cache_interface",
    "vllm.v1.worker",
    "vllm.v1.worker.utils",
    "torch_tpu",
    "torch_tpu._internal",
    "torch_tpu._internal.pallas",
):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _StubModule(_mod_name)

cache = importlib.import_module("vllm_torchtpu.models.vllm.deepseek_v41.v2.cache")
ops = importlib.import_module("vllm_torchtpu.models.vllm.deepseek_v41.v2.ops")

_module_target = SimpleNamespace(alpha=10)


def _sample_global_fn(x: int, y: int = 2, *, scale: int = 1) -> int:
    return (_module_target.alpha + x + y) * scale


def test_ops_device_hooks_and_bind_device_ops():
    mod = SimpleNamespace(alpha=10, beta=20)
    with ops.device_hooks(mod, alpha=99, beta=123):
        assert mod.alpha == 99
        assert mod.beta == 123
    assert mod.alpha == 10
    assert mod.beta == 20

    with pytest.raises(RuntimeError), ops.device_hooks(mod, alpha=777):
        assert mod.alpha == 777
        raise RuntimeError("boom")
    assert mod.alpha == 10

    multiplier = 3

    def sample_closure(z: int) -> int:
        return z * multiplier

    bound = ops.bind_device_ops(
        _sample_global_fn,
        namespace_suffix="_tpu_test",
        _module_target=SimpleNamespace(alpha=5),
    )
    assert bound(1) == (5 + 1 + 2) * 1
    assert bound(2, 3, scale=2) == (5 + 2 + 3) * 2
    assert ops.bind_device_ops(sample_closure)(4) == 12


def test_ops_compact_rope_caches_and_rotary_and_fp8_rmsnorm():
    with pytest.raises(ValueError, match="max_model_len must be positive"):
        ops.compact_rope_caches([], 0)

    shared_large = torch.randn(32, 8)
    small = torch.randn(8, 8)
    a1 = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=shared_large))
    a2 = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=shared_large))
    a3 = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=small))
    ops.compact_rope_caches([a1, a2, a3], 16)
    assert a1._tpu_rope_cache.shape == (16, 8)
    assert a1._tpu_rope_cache is a2._tpu_rope_cache
    assert a3._tpu_rope_cache is small

    x_f32 = torch.randn(1, 4, 2, 16, dtype=torch.float32)
    pos_i64 = torch.arange(4, dtype=torch.int64)
    pos_f32 = pos_i64.float()

    with pytest.raises(ValueError, match="invalid rotary dimension"):
        ops.rotary(x_f32, pos_f32, {"qk_rope_head_dim": 0}, compressed=False)
    with pytest.raises(ValueError, match="invalid rotary dimension"):
        ops.rotary(x_f32, pos_f32, {"qk_rope_head_dim": 3}, compressed=False)
    with pytest.raises(ValueError, match="invalid rotary dimension"):
        ops.rotary(x_f32, pos_f32, {"qk_rope_head_dim": 32}, compressed=False)

    cfg_uncompressed = {"qk_rope_head_dim": 8, "rope_theta": 10000.0}
    out_fwd = ops.rotary(
        x_f32, pos_i64, cfg_uncompressed, compressed=False, inverse=False
    )
    out_inv = ops.rotary(
        out_fwd, pos_f32, cfg_uncompressed, compressed=False, inverse=True
    )
    torch.testing.assert_close(out_inv, x_f32, atol=1e-5, rtol=1e-5)

    cfg_full_bf16 = {
        "qk_rope_head_dim": 16,
        "compress_rope_theta": 50000.0,
        "rope_scaling": {
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 8.0,
        },
    }
    x_bf16 = x_f32.to(torch.bfloat16)
    r1 = ops.rotary(x_bf16, pos_f32, cfg_full_bf16, compressed=True)
    r2 = ops.rotary(x_bf16, pos_f32, cfg_full_bf16, compressed=True)
    assert r1.dtype == torch.bfloat16
    assert torch.equal(r1, r2)

    cfg_zero_yarn = {
        "qk_rope_head_dim": 8,
        "compress_rope_theta": 10000.0,
        "rope_parameters": {"original_max_position_embeddings": 0},
    }
    assert (
        ops.rotary(x_f32, pos_f32, cfg_zero_yarn, compressed=True).shape == x_f32.shape
    )

    with pytest.raises(ValueError, match="FP8 block size"):
        ops.round_fp8_activation(torch.randn(4, 30), block_size=32)

    act_f32 = torch.randn(4, 64, dtype=torch.float32) * 12.0
    blocks = act_f32.unflatten(-1, (-1, 32))
    scales_ref = torch.pow(
        2.0, torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
    )
    expected_fp8 = (
        (blocks / scales_ref.unsqueeze(-1))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .float()
        * scales_ref.unsqueeze(-1)
    ).flatten(-2)
    got_f32 = ops.round_fp8_activation(act_f32, block_size=32)
    assert torch.equal(got_f32, expected_fp8)

    got_bf16 = ops.round_fp8_activation(act_f32.to(torch.bfloat16), block_size=32)
    assert got_bf16.dtype == torch.bfloat16

    w_f32 = torch.ones(16, dtype=torch.float32)
    norm_f32 = ops.rms_norm(x_f32, w_f32, 1e-6)
    assert norm_f32.dtype == torch.float32
    norm_bf16 = ops.rms_norm(x_bf16, w_f32.to(torch.bfloat16), 1e-6)
    assert norm_bf16.dtype == torch.bfloat16


def test_cache_sizing_and_materializer():
    cfg_bf16 = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"])
        ),
        cache_config=SimpleNamespace(cache_dtype="bfloat16"),
    )
    cfg_fp8 = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"])
        ),
        cache_config=SimpleNamespace(cache_dtype="fp8_e4m3"),
    )
    assert not cache.uses_fp8_rope_companion(cfg_bf16)
    assert cache.uses_fp8_rope_companion(cfg_fp8)
    assert cache.reserve_rope_companion(cfg_bf16, 1000) == 1000
    assert cache.reserve_rope_companion(cfg_fp8, 1000) == 800
    huge = cache.SPARSECORE_MAX_POOL_BYTES * 3
    assert (
        cache.reserve_rope_companion(cfg_fp8, huge)
        == cache.SPARSECORE_MAX_POOL_BANKS * cache.SPARSECORE_MAX_POOL_BYTES
    )

    dummy_cfg = SimpleNamespace(
        num_blocks=2,
        kv_cache_tensors=[SimpleNamespace(size=4096)],
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(page_size_bytes=1024))
        ],
    )
    with patch.object(cache, "allocate_upstream_kv_cache", return_value={}):
        empty_res = cache.materialize_upstream_kv_cache(
            dummy_cfg, torch.device("cpu"), SimpleNamespace(is_layer_compact=False)
        )
        assert empty_res.kv_caches == {}
        assert empty_res.raw_tensors == []

    backing_u8 = torch.zeros(4096, dtype=torch.uint8)
    view_a = backing_u8[:2048].view(2, 1, 2, 512)
    view_b = backing_u8[2048:].view(torch.bfloat16).view(2, 1, 2, 256)
    with (
        patch.object(
            cache,
            "allocate_upstream_kv_cache",
            return_value={"layer.0": view_a, "layer.1": view_b},
        ),
        patch(
            "vllm_torchtpu.kv_cache_materializer.layer_to_pool_index",
            return_value={"layer.0": 0, "layer.1": 1},
        ),
    ):
        res_compact = cache.materialize_upstream_kv_cache(
            dummy_cfg,
            torch.device("cpu"),
            SimpleNamespace(is_layer_compact=True),
            word_rows=True,
        )
        assert len(res_compact.kv_caches) == 2
        assert "layer.0" in res_compact.rope_pools

        with patch.object(cache, "SPARSECORE_MAX_POOL_BYTES", 2048):
            dummy_split = SimpleNamespace(
                num_blocks=2,
                kv_cache_tensors=[SimpleNamespace(size=4096)],
                kv_cache_groups=[
                    SimpleNamespace(kv_cache_spec=SimpleNamespace(page_size_bytes=2048))
                ],
            )
            res_split = cache.materialize_upstream_kv_cache(
                dummy_split,
                torch.device("cpu"),
                SimpleNamespace(is_layer_compact=False),
                word_rows=True,
            )
            assert isinstance(res_split.native_raw_cache["layer.0"], tuple)
            assert isinstance(res_split.rope_pools["layer.0"], tuple)

    # Also cover non-FP8 materialization (word_rows=False)
    with patch.object(
        cache, "allocate_upstream_kv_cache", return_value={"layer.1": view_b}
    ):
        res_bf16_flat = cache.materialize_upstream_kv_cache(
            dummy_cfg,
            torch.device("cpu"),
            SimpleNamespace(is_layer_compact=False),
            word_rows=False,
        )
        assert res_bf16_flat.rope_pools == {}
        assert res_bf16_flat.raw_tensors[0].dtype == torch.int8


def test_paged_cache_rows_validation_cpu_and_device_paths():
    raw_lo = torch.zeros((8, cache.ROW_WORDS), dtype=torch.int32)
    raw_hi = torch.zeros((8, cache.ROW_WORDS), dtype=torch.int32)
    rope_lo = torch.zeros((2, cache.ROW_WORDS), dtype=torch.int32)
    bf16_layout = ((4, 2, 256), torch.bfloat16, 1024, 0)

    with pytest.raises(ValueError, match="two banks"):
        cache.PagedCacheRows((raw_lo, raw_hi, raw_lo), bf16_layout)
    with pytest.raises(ValueError, match="two banks"):
        cache.PagedCacheRows((raw_lo, raw_hi), bf16_layout, rope=(rope_lo,))
    with pytest.raises(ValueError, match="int32"):
        cache.PagedCacheRows(torch.zeros((8, 128), dtype=torch.float32), bf16_layout)
    with pytest.raises(ValueError, match="BF16, FP32 or FP8"):
        cache.PagedCacheRows(raw_lo, ((4, 2, 256), torch.int16, 1024, 0))
    with pytest.raises(ValueError, match="RoPE companion"):
        cache.PagedCacheRows(raw_lo, ((4, 2, 512), torch.uint8, 1024, 0), rope=None)
    with pytest.raises(ValueError, match="FP8 records"):
        cache.PagedCacheRows(raw_lo, ((4, 2, 256), torch.uint8, 512, 0), rope=rope_lo)
    with pytest.raises(ValueError, match="divide or be a multiple"):
        cache.PagedCacheRows(raw_lo, ((4, 2, 3), torch.bfloat16, 12, 0))
    with pytest.raises(ValueError, match="divide or be a multiple"):
        cache.PagedCacheRows(raw_lo, ((4, 2, 192), torch.bfloat16, 768, 0))
    with pytest.raises(ValueError, match="record-aligned"):
        cache.PagedCacheRows(raw_lo, ((4, 2, 256), torch.bfloat16, 1028, 0))

    big_raw = MagicMock()
    big_raw.dtype = torch.int32
    big_raw.ndim = 2
    big_raw.shape = (3_000_000_000, cache.ROW_WORDS)
    big_raw.nbytes = 3_000_000_000 * cache.ROW_BYTES
    with pytest.raises(ValueError, match="int32"):
        cache.PagedCacheRows(big_raw, bf16_layout)

    p_unaligned = cache.PagedCacheRows(raw_lo.clone(), bf16_layout)
    assert p_unaligned.numel() == 4 * 2 * 256
    assert p_unaligned.device.type == "cpu"
    assert p_unaligned.pages(torch.tensor([0, 1])) is None
    assert (
        p_unaligned.row_indices(torch.tensor([0, 1], dtype=torch.int64)).dtype
        == torch.int32
    )

    aligned_layout = ((2, 4, 256), torch.bfloat16, 4096, 2048)
    raw_16 = torch.zeros((16, cache.ROW_WORDS), dtype=torch.int32)
    p_aligned = cache.PagedCacheRows(raw_16, aligned_layout)
    pages_mapped = p_aligned.pages(torch.tensor([0, 1], dtype=torch.int64))
    assert pages_mapped is not None and pages_mapped.tolist() == [1, 3]
    assert p_aligned.row_indices(torch.tensor([0, 4], dtype=torch.int32)).tolist() == [
        4,
        12,
    ]

    # Also test pages() when page_scale == 1 and page_offset == 0 with int32 input
    p_unit_aligned = cache.PagedCacheRows(
        raw_lo.clone(), ((2, 4, 256), torch.bfloat16, 2048, 0)
    )
    assert p_unit_aligned.pages(torch.tensor([0, 1], dtype=torch.int32)).tolist() == [
        0,
        1,
    ]

    vals = torch.randn((4, 256), dtype=torch.bfloat16)
    slots = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    p_unaligned.write(
        slots, vals.float(), valid=torch.tensor([True, True, False, True])
    )
    p_unaligned.write(torch.tensor([0, 1], dtype=torch.int32), vals[:2], valid=None)
    fetched = p_unaligned[
        torch.tensor([0, 0], dtype=torch.int32), torch.tensor([0, 1], dtype=torch.int32)
    ]
    assert fetched.shape == (2, 256)

    p_offset = cache.PagedCacheRows(
        torch.zeros((12, cache.ROW_WORDS), dtype=torch.int32),
        ((4, 2, 256), torch.bfloat16, 1024, 1024),
    )
    p_offset.write(torch.tensor([0, 1], dtype=torch.int32), vals[:2])
    p_stride = cache.PagedCacheRows(
        torch.zeros((16, cache.ROW_WORDS), dtype=torch.int32),
        ((4, 2, 256), torch.bfloat16, 2048, 0),
    )
    p_stride.write(
        torch.tensor([0, 1], dtype=torch.int32),
        vals[:2],
        valid=torch.tensor([True, False]),
    )
    p_stride.write(torch.tensor([0, 1], dtype=torch.int32), vals[:2], valid=None)

    p_2b = cache.PagedCacheRows(
        (raw_lo.clone(), raw_hi.clone()), ((8, 2, 256), torch.bfloat16, 1024, 0)
    )
    p_2b.write(torch.tensor([0, 9], dtype=torch.int32), vals[:2])
    got_2b = p_2b[
        torch.tensor([0, 4], dtype=torch.int32), torch.tensor([0, 1], dtype=torch.int32)
    ]
    assert got_2b.shape == (2, 256)
    p_2b.copy_raw((raw_lo.clone(), raw_hi.clone()))
    p_unaligned.copy_raw(raw_lo.clone())

    fp8_1b = cache.PagedCacheRows(
        torch.zeros((8, cache.ROW_WORDS), dtype=torch.int32),
        ((4, 2, 512), torch.uint8, 1024, 0),
        rope=torch.zeros((2, cache.ROW_WORDS), dtype=torch.int32),
    )
    fp8_1b.write(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.randn((2, 512), dtype=torch.float32),
    )
    fp8_1b.write(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.randn((2, 512), dtype=torch.bfloat16),
    )

    fp8_2b = cache.PagedCacheRows(
        (
            torch.zeros((8, cache.ROW_WORDS), dtype=torch.int32),
            torch.zeros((8, cache.ROW_WORDS), dtype=torch.int32),
        ),
        ((8, 2, 512), torch.uint8, 1024, 0),
        rope=(
            torch.zeros((2, cache.ROW_WORDS), dtype=torch.int32),
            torch.zeros((2, cache.ROW_WORDS), dtype=torch.int32),
        ),
    )
    fp8_2b.write(
        torch.tensor([0, 9], dtype=torch.int32),
        torch.randn((2, 512), dtype=torch.bfloat16),
    )


def test_paged_cache_rows_non_cpu_dispatch_and_native_cache_view():
    class FakeTPU(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="xla")

    t_lo = torch.zeros((8, cache.ROW_WORDS), dtype=torch.int32).as_subclass(FakeTPU)
    t_hi = torch.zeros((8, cache.ROW_WORDS), dtype=torch.int32).as_subclass(FakeTPU)
    r_lo = torch.zeros((2, cache.ROW_WORDS), dtype=torch.int32).as_subclass(FakeTPU)
    r_hi = torch.zeros((2, cache.ROW_WORDS), dtype=torch.int32).as_subclass(FakeTPU)

    p1 = cache.PagedCacheRows(t_lo, ((4, 2, 256), torch.bfloat16, 1024, 0))
    p2 = cache.PagedCacheRows((t_lo, t_hi), ((8, 2, 256), torch.bfloat16, 1024, 0))
    pf1 = cache.PagedCacheRows(t_lo, ((4, 2, 512), torch.uint8, 1024, 0), rope=r_lo)
    pf2 = cache.PagedCacheRows(
        (t_lo, t_hi), ((8, 2, 512), torch.uint8, 1024, 0), rope=(r_lo, r_hi)
    )

    with (
        patch.object(
            cache,
            "_read_values_op",
            side_effect=lambda c, r, e, f: torch.zeros(
                (r.numel(), e), dtype=torch.bfloat16
            ),
        ),
        patch.object(
            cache,
            "_read_values_2bank_op",
            side_effect=lambda lo, hi, r, e, f: torch.zeros(
                (r.numel(), e), dtype=torch.bfloat16
            ),
        ),
        patch.object(
            cache, "_write_values_op", side_effect=lambda c, r, v: torch.zeros_like(c)
        ),
        patch.object(
            cache,
            "_write_values_2bank_op",
            side_effect=lambda lo, hi, r, v: (
                torch.zeros_like(lo),
                torch.zeros_like(hi),
            ),
        ),
        patch.object(
            cache,
            "_write_values_after_op",
            side_effect=lambda c, r, v, a: torch.zeros_like(c),
        ),
        patch.object(
            cache,
            "_encode_fp8_op",
            side_effect=lambda v: (
                torch.zeros((v.shape[0], cache.ROW_WORDS), dtype=torch.int32),
                torch.zeros((v.shape[0], cache.ROPE_WORDS), dtype=torch.int32),
            ),
        ),
    ):
        assert p1[torch.tensor([0]), torch.tensor([0])].shape == (1, 256)
        assert p2[torch.tensor([0]), torch.tensor([0])].shape == (1, 256)
        v_bf16 = torch.randn((2, 256), dtype=torch.bfloat16)
        p1.write(torch.tensor([0, 1], dtype=torch.int64), v_bf16)
        p1.write(torch.tensor([0, 1], dtype=torch.int32), v_bf16, after=v_bf16)
        p2.write(torch.tensor([0, 9], dtype=torch.int32), v_bf16)
        p2.write(torch.tensor([0, 9], dtype=torch.int32), v_bf16, after=v_bf16)
        pf1.write(
            torch.tensor([0, 1], dtype=torch.int64),
            torch.randn((2, 512), dtype=torch.bfloat16),
        )
        pf2.write(
            torch.tensor([0, 9], dtype=torch.int32),
            torch.randn((2, 512), dtype=torch.bfloat16),
        )

        cache.write_cache(
            p1,
            torch.tensor([0, 1]),
            v_bf16,
            torch.tensor([True, False]),
            records=torch.tensor([0, 1]),
        )
        cache.write_cache(p1, torch.tensor([0, 1]), v_bf16, torch.tensor([True, False]))
        cpu_2d = torch.zeros((4, 256), dtype=torch.bfloat16)
        cache.write_cache(
            cpu_2d, torch.tensor([0, 2]), v_bf16, torch.tensor([True, True])
        )
        cpu_3d = torch.zeros((2, 2, 256), dtype=torch.bfloat16)
        cache.write_cache(
            cpu_3d, torch.tensor([0, 3]), v_bf16, torch.tensor([True, True])
        )

    with patch.object(
        cache, "_write_op", side_effect=lambda c, s, v: torch.zeros_like(c)
    ):
        cache.write_cache(
            t_lo,
            torch.tensor([0, 1], dtype=torch.int64),
            torch.zeros((2, 128), dtype=torch.int32),
            torch.tensor([True, False]),
        )
        cache.write_cache(
            t_lo,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.zeros((2, 128), dtype=torch.int32),
            torch.tensor([True, False]),
        )

    class BaseLayer:
        def bind_kv_cache(self, c):
            self._base_bound = c

    class DummyAttention(cache.NativeCacheView, BaseLayer):
        pass

    layer = DummyAttention()
    view_cpu = torch.zeros((4, 2, 256), dtype=torch.bfloat16)
    layer.kv_cache = view_cpu
    assert layer.kv_cache is view_cpu
    layer.bind_kv_cache(view_cpu, raw_cache=None)
    assert layer._tpu_paged_cache_rows is None
    raw_cpu = torch.zeros((8, cache.ROW_WORDS), dtype=torch.int32)
    layer.bind_kv_cache(view_cpu, raw_cache=raw_cpu)
    assert layer._tpu_paged_cache_rows is not None
    explicit_layout = ((4, 2, 256), torch.bfloat16, 1024, 0)
    layer.bind_kv_cache(view_cpu, raw_cache=raw_cpu, layout=explicit_layout)
    assert layer._tpu_cache_layout == explicit_layout
    layer._tpu_paged_cache_rows = None
    with patch("torch.compiler.is_compiling", return_value=True):
        p_first = layer.kv_cache
        p_second = layer.kv_cache
        assert p_first is p_second

    with pytest.raises(ValueError, match="explicit allocator layout"):
        layer.bind_kv_cache(view_cpu, raw_cache=t_lo, layout=None)


def test_jax_cache_ops_bit_exact_and_pallas_kernels():
    x_bf16 = jnp.asarray(np.random.randn(4, 512).astype(np.float32)).astype(
        jnp.bfloat16
    )
    x_f32 = jnp.asarray(np.random.randn(4, 256).astype(np.float32))

    # Verify split_halves, merge_halves, split_pieces, and merge_pieces bit-exactness
    for arr in (x_bf16, x_f32):
        order = cache.piece_order(arr.shape[-1])
        s_ref = cache._permute(arr, order)
        s_opt = cache.split_halves(arr)
        assert jnp.array_equal(s_opt, s_ref)
        m_ref = cache._permute(s_opt, jnp.argsort(order))
        m_opt = cache.merge_halves(s_opt)
        assert jnp.array_equal(m_opt, m_ref)

        sp = cache.split_pieces(arr, 4)
        mp = cache.merge_pieces(sp)
        assert jnp.array_equal(mp, arr)

    # Verify encode_fp8_kv 100.0000% bit-exactness against division by scales
    latent = jnp.asarray(np.random.randn(8, 512).astype(np.float32) * 15.0).astype(
        jnp.bfloat16
    )
    words_opt, rope_opt = cache.encode_fp8_kv(latent)
    nope_3d = latent[:, : cache.NOPE_DIM].astype(jnp.float32).reshape(-1, 7, 64)
    exp = jnp.ceil(
        jnp.log2(jnp.maximum(jnp.abs(nope_3d).max(axis=1), 1e-4) / 448.0)
    ).astype(jnp.int32)
    scales_div = jax.lax.bitcast_convert_type((exp + 127) << 23, jnp.float32)
    vals_div = (
        (nope_3d / scales_div[:, None, :])
        .reshape(-1, cache.NOPE_DIM)
        .astype(jnp.float8_e4m3fn)
    )
    inv_scales = jax.lax.bitcast_convert_type((127 - exp) << 23, jnp.float32)
    vals_mul = (
        (nope_3d * inv_scales[:, None, :])
        .reshape(-1, cache.NOPE_DIM)
        .astype(jnp.float8_e4m3fn)
    )
    assert jnp.array_equal(
        jax.lax.bitcast_convert_type(vals_mul, jnp.uint8),
        jax.lax.bitcast_convert_type(vals_div, jnp.uint8),
    )
    assert words_opt.shape == (8, cache.ROW_WORDS)
    assert rope_opt.shape == (8, cache.ROPE_WORDS)

    pool_lo = jnp.arange(16 * cache.ROW_WORDS, dtype=jnp.int32).reshape(
        16, cache.ROW_WORDS
    )
    pool_hi = (
        jnp.arange(16 * cache.ROW_WORDS, dtype=jnp.int32).reshape(16, cache.ROW_WORDS)
        + 10000
    )
    assert (
        cache._scatter_rows(
            pool_lo,
            jnp.array([0, -1], dtype=jnp.int32),
            jnp.ones((2, cache.ROW_WORDS), dtype=jnp.int32),
        ).shape
        == pool_lo.shape
    )
    assert cache._cache_num_rows(pool_lo) == 16
    assert cache._cache_num_rows((pool_lo, pool_hi)) == 32

    recs = jnp.array([0, 2, -1, 17], dtype=jnp.int32)
    assert cache._gather_words(pool_lo, jnp.array([0, 1, 2, 3]), 32).shape == (4, 32)
    assert cache._gather_words(pool_lo, jnp.array([0, 1]), 256).shape == (2, 256)
    assert cache.gather_cache_pieces(pool_lo, jnp.array([0, 1]), 256).shape == (
        2,
        2,
        128,
    )
    assert cache.gather_cache_pieces((pool_lo, pool_hi), recs, 512).shape == (4, 4, 128)
    assert cache.gather_cache_values(
        pool_lo, jnp.array([0, 1]), 256, split=True
    ).shape == (2, 256)
    assert cache.gather_cache_values(
        pool_lo, jnp.array([0, 1]), 512, split=True
    ).shape == (2, 512)
    assert cache.gather_cache_values(
        pool_lo, jnp.array([0, 1]), 128, fp32=True
    ).shape == (2, 128)
    assert cache.gather_cache_values(
        pool_lo, jnp.array([0, 1]), 128, fp32=False
    ).shape == (2, 128)

    with pytest.raises(ValueError, match="whole pool word rows"):
        cache.gather_cache_page_words(pool_lo, jnp.array([0]), 1, 128)
    assert cache.gather_cache_page_words(pool_lo, jnp.array([0, -1]), 2, 128).shape == (
        2,
        128,
    )
    assert cache.gather_cache_page_words(
        (pool_lo, pool_hi), jnp.array([0, 8]), 2, 128
    ).shape == (2, 128)
    assert cache.gather_cache_page_values(pool_lo, jnp.array([0]), 2, 128).shape == (
        2,
        128,
    )

    merged_r, merged_w = cache.merge_sub_rows(
        pool_lo,
        jnp.array([0, 1, -1], dtype=jnp.int32),
        jnp.ones((3, 64), dtype=jnp.int32),
    )
    assert merged_r.shape == (3,) and merged_w.shape == (3, cache.ROW_WORDS)
    packed_bf16 = cache._pack_bf16_words(x_bf16)
    assert packed_bf16.shape == (4, 256)

    # Exercise CPU branches of scatter_cache_values
    assert (
        cache.scatter_cache_values(
            pool_lo,
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0, 128), dtype=jnp.int32),
        )
        is pool_lo
    )
    assert (
        cache.scatter_cache_values(
            pool_lo, jnp.array([0, 1]), jnp.ones((2, 64), dtype=jnp.int32)
        ).shape
        == pool_lo.shape
    )
    assert (
        cache.scatter_cache_values(
            pool_lo, jnp.array([0, 1]), jnp.ones((2, 128), dtype=jnp.float32)
        ).shape
        == pool_lo.shape
    )
    assert (
        cache.scatter_cache_values(
            pool_lo, jnp.array([0, 1]), jnp.ones((2, 256), dtype=jnp.bfloat16)
        ).shape
        == pool_lo.shape
    )
    assert (
        len(
            cache.scatter_cache_values(
                (pool_lo, pool_hi),
                jnp.array([0, 17]),
                jnp.ones((2, 64), dtype=jnp.int32),
            )
        )
        == 2
    )

    # Exercise TPU Pallas inner kernel bodies & initialize_cache_ops
    class DummyCopy:
        def start(self):
            pass

        def wait(self):
            pass

    def fake_pallas_call(kernel_fn, out_shape, **kwargs):
        def _runner(records_arg, flat_words, *banks):
            rec_jnp = jnp.asarray(records_arg)
            words_jnp = jnp.asarray(flat_words)
            if len(banks) == 1:
                out_buf = jnp.asarray(banks[0])
                kernel_fn(rec_jnp, words_jnp, banks[0], out_buf, None)
                return out_buf
            out_lo = jnp.asarray(banks[0])
            out_hi = jnp.asarray(banks[1])
            kernel_fn(rec_jnp, words_jnp, banks[0], banks[1], out_lo, out_hi, None)
            return out_lo, out_hi

        return _runner

    with (
        patch.object(jax, "default_backend", return_value="tpu"),
        patch("jax.experimental.pallas.pallas_call", side_effect=fake_pallas_call),
        patch("jax.experimental.pallas.program_id", return_value=0),
        patch(
            "jax.experimental.pallas.when", side_effect=lambda cond: (lambda fn: fn())
        ),
        patch("jax.experimental.pallas.tpu.make_async_copy", return_value=DummyCopy()),
        patch(
            "jax.lax.fori_loop",
            side_effect=lambda lo, hi, body, init: body(0, init),
        ),
    ):
        assert (
            cache.scatter_cache_values(pool_lo, jnp.array([0, 1]), x_bf16[:2]).shape
            == pool_lo.shape
        )
        assert (
            cache.scatter_cache_values(
                pool_lo, jnp.array([0, 1]), jnp.ones((2, 64), dtype=jnp.int32)
            ).shape
            == pool_lo.shape
        )
        lo_res, hi_res = cache.scatter_cache_values(
            (pool_lo, pool_hi),
            jnp.array([0, 17], dtype=jnp.int32),
            jnp.ones((2, 128), dtype=jnp.int32),
        )
        assert lo_res.shape == pool_lo.shape and hi_res.shape == pool_hi.shape

    class FakeOp:
        def __init__(self):
            self.fake_fn = None

        def register_fake(self, fn):
            self.fake_fn = fn

    fake_ops = []

    def fake_jax_op(name, fn, **kwargs):
        op = FakeOp()
        op.fn = fn
        fake_ops.append(op)
        return op

    with (
        patch.object(
            sys.modules["torch_tpu._internal"],
            "pallas",
            SimpleNamespace(jax_op=fake_jax_op),
            create=True,
        ),
        patch(
            "vllm_torchtpu.models.vllm.vllm_model_wrapper_context.get_vllm_model_wrapper_context",
            return_value=SimpleNamespace(mesh=SimpleNamespace()),
        ),
        patch.object(jax, "shard_map", side_effect=lambda fn, **kw: fn),
    ):
        cache._write_op = None
        cache.initialize_cache_ops()
        assert cache._write_op is not None
        # Second call returns early cleanly
        cache.initialize_cache_ops()

        dummy_c = torch.zeros((4, 128), dtype=torch.int32)
        dummy_r = torch.tensor([0, 1], dtype=torch.int32)
        dummy_v = torch.zeros((2, 128), dtype=torch.int32)
        for op in fake_ops:
            assert op.fake_fn is not None
        assert fake_ops[0].fake_fn(dummy_c, dummy_r, 128, False).shape == (2, 128)
        assert fake_ops[1].fake_fn(dummy_c, dummy_c, dummy_r, 128, True).shape == (
            2,
            128,
        )
        assert fake_ops[2].fake_fn(dummy_c, dummy_r, dummy_v).shape == dummy_c.shape
        assert fake_ops[3].fake_fn(dummy_c, dummy_r, dummy_v).shape == dummy_c.shape
        assert len(fake_ops[4].fake_fn(dummy_c, dummy_c, dummy_r, dummy_v)) == 2
        assert (
            fake_ops[5].fake_fn(dummy_c, dummy_r, dummy_v, dummy_v).shape
            == dummy_c.shape
        )
        assert (
            len(fake_ops[6].fake_fn(torch.zeros((2, 512), dtype=torch.bfloat16))) == 2
        )

        assert fake_ops[1].fn(
            pool_lo, pool_hi, jnp.array([0, 17]), 128, False
        ).shape == (2, 128)
        assert (
            len(
                fake_ops[4].fn(
                    pool_lo,
                    pool_hi,
                    jnp.array([0, 17]),
                    jnp.ones((2, 64), dtype=jnp.int32),
                )
            )
            == 2
        )
        assert (
            fake_ops[5]
            .fn(
                pool_lo,
                jnp.array([0]),
                jnp.ones((1, 64), dtype=jnp.int32),
                jnp.ones((1, 128), dtype=jnp.int32),
            )
            .shape
            == pool_lo.shape
        )
        cache._write_op = None
