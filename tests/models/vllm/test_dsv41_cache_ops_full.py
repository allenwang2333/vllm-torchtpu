# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Comprehensive 100% statement and branch coverage tests for deepseek_v41_cache."""

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
import torch

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

for _pkg_name, _rel in (
    ("vllm_torchtpu", "vllm_torchtpu"),
    ("vllm_torchtpu.kernels", "vllm_torchtpu/kernels"),
    ("vllm_torchtpu.kernels.deepseek_v4", "vllm_torchtpu/kernels/deepseek_v4"),
    ("vllm_torchtpu.kernels.deepseek_v41", "vllm_torchtpu/kernels/deepseek_v41"),
    ("vllm_torchtpu.layers", "vllm_torchtpu/layers"),
    ("vllm_torchtpu.layers.adapter", "vllm_torchtpu/layers/adapter"),
    (
        "vllm_torchtpu.layers.adapter.custom_ops",
        "vllm_torchtpu/layers/adapter/custom_ops",
    ),
    (
        "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41",
        "vllm_torchtpu/layers/adapter/custom_ops/deepseek_v41",
    ),
    ("vllm_torchtpu.models", "vllm_torchtpu/models"),
    ("vllm_torchtpu.models.vllm", "vllm_torchtpu/models/vllm"),
    ("vllm_torchtpu.runner", "vllm_torchtpu/runner"),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_SRC_DIR / _rel)] if _SRC_DIR.is_dir() else []  # type: ignore[attr-defined]
        sys.modules[_pkg_name] = _pkg
    if "." in _pkg_name:
        _parent, _child = _pkg_name.rsplit(".", 1)
        setattr(sys.modules[_parent], _child, sys.modules[_pkg_name])

for _mod_name, _attrs in (
    ("torch_tpu", {}),
    ("torch_tpu._internal", {}),
    ("torch_tpu._internal.pallas", {"jax_op": MagicMock()}),
    (
        "vllm_torchtpu.models.vllm.vllm_model_wrapper_context",
        {"get_vllm_model_wrapper_context": MagicMock()},
    ),
    (
        "vllm_torchtpu.runner.kv_cache_dsv4",
        {"deepseek_v41_cache_spec": MagicMock()},
    ),
    (
        "vllm_torchtpu.kernels.deepseek_v41.cache_write",
        {"write_cache_words": MagicMock()},
    ),
):
    if _mod_name not in sys.modules:
        _m = types.ModuleType(_mod_name)
        for _k, _v in _attrs.items():
            setattr(_m, _k, _v)
        sys.modules[_mod_name] = _m
    if "." in _mod_name:
        _parent, _child = _mod_name.rsplit(".", 1)
        setattr(sys.modules[_parent], _child, sys.modules[_mod_name])


cache_mod = importlib.import_module(
    "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_cache"
)
ROW_WORDS = cache_mod.ROW_WORDS
ROW_BYTES = cache_mod.ROW_BYTES
ByteLayout = cache_mod.ByteLayout
IndexPageGeometry = cache_mod.IndexPageGeometry
MaterializedNativeKVCache = cache_mod.MaterializedNativeKVCache
NativeBinding = cache_mod.NativeBinding
NativeCacheBinding = cache_mod.NativeCacheBinding
RecordLayout = cache_mod.RecordLayout
_gather_words = cache_mod._gather_words
_scatter_rows = cache_mod._scatter_rows
block_table = cache_mod.block_table
cache_byte_layout = cache_mod.cache_byte_layout
cache_record_layout = cache_mod.cache_record_layout
cache_slots = cache_mod.cache_slots
gather_cache_values = cache_mod.gather_cache_values
initialize_cache_ops = cache_mod.initialize_cache_ops
masked_row_indices = cache_mod.masked_row_indices
native_cache_view = cache_mod.native_cache_view
read_cache = cache_mod.read_cache
read_cache_values = cache_mod.read_cache_values
row_indices = cache_mod.row_indices
row_slots = cache_mod.row_slots
scatter_cache_values = cache_mod.scatter_cache_values
scatter_cache_words = cache_mod.scatter_cache_words
token_requests = cache_mod.token_requests
write_cache = cache_mod.write_cache


class _BaseBinding:
    def bind_kv_cache(self, cache):
        self.kv_cache = cache

    def get_kv_cache_spec(self, vllm_config):
        return ("base_spec", vllm_config)


class _DummyCacheLayer(NativeCacheBinding, _BaseBinding):
    pass


def test_index_page_geometry_and_layouts():
    geom = IndexPageGeometry(nope_bytes=ROW_BYTES * 2, scale_bytes=64)
    assert geom.keys_per_page == (ROW_BYTES * 2) // 128

    with pytest.raises(ValueError, match="complete native word rows"):
        IndexPageGeometry(nope_bytes=0, scale_bytes=64)
    with pytest.raises(ValueError, match="complete native word rows"):
        IndexPageGeometry(nope_bytes=128, scale_bytes=64)
    with pytest.raises(ValueError, match="companion array budget"):
        IndexPageGeometry(nope_bytes=ROW_BYTES * 2, scale_bytes=4)

    raw_cpu = torch.zeros((16, ROW_WORDS), dtype=torch.int32)
    cache_cpu = torch.zeros((2, 4, 256), dtype=torch.bfloat16)
    bl = ByteLayout((2, 4, 256), torch.bfloat16, 2048, 0)
    assert cache_byte_layout(raw_cpu, cache_cpu, bl) is bl
    assert cache_byte_layout(raw_cpu, cache_cpu, tuple(bl)) == bl
    inferred = cache_byte_layout(raw_cpu, cache_cpu, None)
    assert inferred.shape == (2, 4, 256)

    fake_tpu_raw = SimpleNamespace(device=SimpleNamespace(type="tpu"))
    with pytest.raises(ValueError, match="explicit allocator layout"):
        cache_byte_layout(fake_tpu_raw, cache_cpu, None)

    rec = cache_record_layout(raw_cpu, bl)
    assert rec.geometry == (4, 4, 0)

    with pytest.raises(ValueError, match="word rows"):
        cache_record_layout(torch.zeros((16, 64), dtype=torch.int32), bl)
    with pytest.raises(ValueError, match="BF16 or FP32"):
        cache_record_layout(raw_cpu, ByteLayout((2, 4, 256), torch.int16, 2048, 0))
    with pytest.raises(ValueError, match="divide or be a multiple of 512"):
        cache_record_layout(raw_cpu, ByteLayout((2, 4, 3), torch.bfloat16, 24, 0))
    with pytest.raises(ValueError, match="divide or be a multiple of 512"):
        cache_record_layout(raw_cpu, ByteLayout((2, 4, 192), torch.bfloat16, 1536, 0))
    with pytest.raises(ValueError, match="record-aligned"):
        cache_record_layout(
            raw_cpu, ByteLayout((2, 4, 256), torch.bfloat16, 1024 + 64, 0)
        )
    with pytest.raises(ValueError, match="fit int32"):
        huge_raw = SimpleNamespace(
            dtype=torch.int32,
            ndim=2,
            shape=(torch.iinfo(torch.int32).max + 1, ROW_WORDS),
            nbytes=(torch.iinfo(torch.int32).max + 1) * ROW_BYTES,
        )
        cache_record_layout(huge_raw, bl)


def test_native_binding_and_materialized_cache():
    raw = torch.zeros((8, ROW_WORDS), dtype=torch.int32)
    rec = RecordLayout(4, 4, 0, 256, torch.bfloat16)
    nb_direct = NativeBinding(raw, rec)
    assert nb_direct.viewed_pool is raw
    nb_viewed = NativeBinding(raw, rec, kernel_shape=(2, 4, ROW_WORDS))
    assert nb_viewed.viewed_pool.shape == (2, 4, ROW_WORDS)

    layer = _DummyCacheLayer()
    assert not layer.has_native_cache
    with pytest.raises(RuntimeError, match="requires native cache binding"):
        layer.native()

    layer.bind_native(None, None)
    assert not layer.has_native_cache
    with pytest.raises(ValueError, match="validated record geometry"):
        layer.bind_native(raw, None)

    with pytest.raises(TypeError, match="single Tensor"):
        layer.bind_kv_cache(["not_a_tensor"])

    cache = torch.zeros((2, 4, 256), dtype=torch.bfloat16)
    layer.bind_kv_cache(cache)
    assert layer.kv_cache is cache
    assert layer.bound_cache is cache

    layer.bind_kv_cache(cache, raw_cache=raw)
    assert layer.has_native_cache
    assert layer.native().pool is raw
    assert layer.kv_cache is raw

    with patch(
        "vllm_torchtpu.runner.kv_cache_dsv4.deepseek_v41_cache_spec",
        side_effect=lambda base, cfg: ("wrapped", base, cfg),
    ):
        spec = layer.get_kv_cache_spec("cfg")
        assert spec == ("wrapped", ("base_spec", "cfg"), "cfg")

    mat = MaterializedNativeKVCache(kv_caches={"k": cache}, raw_tensors=[raw])
    assert "k" in mat.kv_caches

    byte_pool = torch.arange(2 * 1024, dtype=torch.uint8)
    view = native_cache_view(byte_pool, (2, 2, 128), torch.bfloat16, 1024, 256)
    assert view.shape == (2, 2, 128)


def test_row_indices_and_read_write_cache():
    slots = torch.tensor([0, 1, 4, 5], dtype=torch.int32)
    # page_size == page_stride, row_offset == 0 and != 0
    assert torch.equal(row_indices(slots, 4, 4, 0), slots)
    assert torch.equal(row_indices(slots, 4, 4, 2), slots + 2)
    # power-of-two page_size with power-of-two and non-power-of-two page_stride
    assert torch.equal(
        row_indices(slots, 4, 8, 0),
        slots // 4 * 8 + slots % 4,
    )
    assert torch.equal(
        row_indices(slots, 4, 6, 3),
        3 + slots // 4 * 6 + slots % 4,
    )
    # non-power-of-two page_size
    assert torch.equal(
        row_indices(slots, 3, 6, 0),
        slots // 3 * 6 + slots % 3,
    )

    jslots = jnp.array([-1, 0, 3, 4], dtype=jnp.int32)
    m_idx = masked_row_indices(jslots, 4, 8, 2)
    assert list(m_idx) == [-1, 2, 5, 10]

    # read_cache without layout (int32 and int64 slots)
    cache_3d = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    out_i32 = read_cache(cache_3d, torch.tensor([0, 5], dtype=torch.int32))
    out_i64 = read_cache(cache_3d, torch.tensor([0, 5], dtype=torch.int64))
    assert torch.equal(out_i32, out_i64)

    # read_cache and write_cache on CPU with layout
    raw_pool = torch.zeros((8, ROW_WORDS), dtype=torch.int32)
    rec_layout = RecordLayout(2, 4, 0, 256, torch.bfloat16)
    vals = torch.arange(3 * 256, dtype=torch.float32).reshape(3, 256)
    write_slots = torch.tensor([0, 1, 2], dtype=torch.int64)
    valid = torch.tensor([True, False, True])
    write_cache(raw_pool, write_slots, vals, valid, layout=rec_layout)

    read_back = read_cache(
        raw_pool, torch.tensor([0, 2], dtype=torch.int64), layout=rec_layout
    )
    assert torch.equal(read_back[0], vals[0].to(torch.bfloat16))
    assert torch.equal(read_back[1], vals[2].to(torch.bfloat16))
    read_back_i32 = read_cache(
        raw_pool, torch.tensor([0, 2], dtype=torch.int32), layout=rec_layout
    )
    assert torch.equal(read_back, read_back_i32)

    # write_cache on CPU without layout (2D and 3D cache)
    cache_2d = torch.zeros((8, 16), dtype=torch.float32)
    v2 = torch.ones((2, 16), dtype=torch.float32)
    write_cache(
        cache_2d,
        torch.tensor([1, 3], dtype=torch.int32),
        v2,
        torch.tensor([True, True]),
    )
    assert torch.equal(cache_2d[1], v2[0])

    cache_3d_w = torch.zeros((2, 4, 16), dtype=torch.float32)
    write_cache(
        cache_3d_w,
        torch.tensor([1, 5], dtype=torch.int32),
        v2,
        torch.tensor([True, True]),
    )
    assert torch.equal(cache_3d_w[0, 1], v2[0])
    assert torch.equal(cache_3d_w[1, 1], v2[1])


def test_tpu_dispatched_read_and_write_cache_and_init_ops():
    class _FakeTensor:
        def __init__(self, t, device_type="tpu"):
            self._t = t
            self.device = SimpleNamespace(type=device_type)
            self.dtype = t.dtype
            self.shape = t.shape
            self.ndim = t.ndim

        def copy_(self, other):
            self.copied = other

    fake_cache = _FakeTensor(torch.zeros((8, ROW_WORDS), dtype=torch.int32))
    rec_layout = RecordLayout(2, 4, 0, 256, torch.bfloat16)

    with (
        patch.object(
            cache_mod,
            "_read_values_op",
            side_effect=lambda c, s, el, fp32, ps, pst, ro: torch.zeros(
                (s.numel(), el), dtype=torch.float32 if fp32 else torch.bfloat16
            ),
        ),
        patch.object(
            cache_mod,
            "_write_values_op",
            side_effect=lambda c, s, v, ps, pst, ro: "written_values",
        ),
        patch.object(
            cache_mod,
            "_write_op",
            side_effect=lambda c, s, v: "written_raw",
        ),
    ):
        r1 = read_cache(
            fake_cache,
            torch.tensor([0, 1], dtype=torch.int64),
            layout=rec_layout,
        )
        assert r1.shape == (2, 256)
        r2 = read_cache(
            fake_cache,
            torch.tensor([0, 1], dtype=torch.int32),
            layout=rec_layout,
        )
        assert r2.shape == (2, 256)

        vals_bf16 = torch.ones((2, 256), dtype=torch.bfloat16)
        write_cache(
            fake_cache,
            torch.tensor([0, 1], dtype=torch.int32),
            vals_bf16,
            torch.tensor([True, True]),
            layout=rec_layout,
        )
        assert fake_cache.copied == "written_values"

        write_cache(
            fake_cache,
            torch.tensor([0, 1], dtype=torch.int64),
            vals_bf16,
            torch.tensor([True, True]),
            layout=rec_layout,
        )
        assert fake_cache.copied == "written_values"

        write_cache(
            fake_cache,
            torch.tensor([0, 1], dtype=torch.int32),
            vals_bf16,
            torch.tensor([True, True]),
        )
        assert fake_cache.copied == "written_raw"

        write_cache(
            fake_cache,
            torch.tensor([0, 1], dtype=torch.int64),
            vals_bf16,
            torch.tensor([True, True]),
        )
        assert fake_cache.copied == "written_raw"

    # Test initialize_cache_ops registration & early return
    registered_ops = []

    def _fake_jax_op(name, fn, **kwargs):
        op = MagicMock(name=name)
        op.fn = fn

        def _reg_fake(fake_fn):
            op.fake_fn = fake_fn

        op.register_fake = _reg_fake
        registered_ops.append(op)
        return op

    old_ops = (
        cache_mod._read_values_op,
        cache_mod._write_op,
        cache_mod._write_values_op,
    )
    try:
        cache_mod._read_values_op = None
        cache_mod._write_op = None
        cache_mod._write_values_op = None
        with (
            patch("torch_tpu._internal.pallas.jax_op", side_effect=_fake_jax_op),
            patch(
                "vllm_torchtpu.models.vllm.vllm_model_wrapper_context.get_vllm_model_wrapper_context",
                return_value=SimpleNamespace(mesh="dummy_mesh"),
            ),
            patch("jax.shard_map", side_effect=lambda fn, **kw: fn),
        ):
            initialize_cache_ops()
            assert len(registered_ops) == 3
            # Call second time to hit early return
            initialize_cache_ops()
            assert len(registered_ops) == 3

            # Exercise fake registrations and write_values inner function
            dummy_c = torch.zeros((4, ROW_WORDS), dtype=torch.int32)
            dummy_s = torch.zeros((2,), dtype=torch.int32)
            dummy_v = torch.zeros((2, 256), dtype=torch.bfloat16)
            assert registered_ops[0].fake_fn(
                dummy_c, dummy_s, 256, False, 2, 2, 0
            ).shape == (2, 256)
            assert registered_ops[1].fake_fn(dummy_c, dummy_s, dummy_v).shape == (
                4,
                ROW_WORDS,
            )
            assert registered_ops[2].fake_fn(
                dummy_c, dummy_s, dummy_v, 2, 2, 0
            ).shape == (4, ROW_WORDS)

            j_cache = jnp.zeros((4, ROW_WORDS), dtype=jnp.int32)
            j_slots = jnp.array([0, -1], dtype=jnp.int32)
            j_vals = jnp.ones((2, 256), dtype=jnp.bfloat16)
            out_wv = registered_ops[2].fn(j_cache, j_slots, j_vals, 2, 2, 0)
            assert out_wv.shape == (4, ROW_WORDS)
    finally:
        (
            cache_mod._read_values_op,
            cache_mod._write_op,
            cache_mod._write_values_op,
        ) = old_ops


def test_jax_gather_and_scatter_kernels():
    j_cache = jnp.arange(8 * ROW_WORDS, dtype=jnp.int32).reshape(8, ROW_WORDS)
    j_slots = jnp.array([0, 2], dtype=jnp.int32)
    j_vals = jnp.ones((2, ROW_WORDS), dtype=jnp.int32)
    scattered = _scatter_rows(j_cache, j_slots, j_vals)
    assert scattered.shape == (8, ROW_WORDS)

    # _gather_words: chunks == 1, chunks > 1, and sub-row (< ROW_WORDS)
    g1 = _gather_words(j_cache, j_slots, ROW_WORDS)
    assert g1.shape == (2, ROW_WORDS)
    g2 = _gather_words(j_cache, j_slots, ROW_WORDS * 2)
    assert g2.shape == (2, ROW_WORDS * 2)
    g_sub = _gather_words(
        j_cache, jnp.array([0, 1, 2, 3], dtype=jnp.int32), ROW_WORDS // 4
    )
    assert g_sub.shape == (4, ROW_WORDS // 4)

    # read_cache_values & gather_cache_values (fp32=True and fp32=False)
    rv_fp32 = read_cache_values(j_cache, j_slots, ROW_WORDS, True, 2, 2, 0)
    assert rv_fp32.dtype == jnp.float32
    rv_bf16 = gather_cache_values(
        j_cache, jnp.array([0, 1], dtype=jnp.int32), 64, fp32=False
    )
    assert rv_bf16.dtype == jnp.bfloat16 and rv_bf16.shape == (2, 64)

    # scatter_cache_values & scatter_cache_words: empty and non-empty
    empty_vals = jnp.zeros((0, 256), dtype=jnp.bfloat16)
    assert (
        scatter_cache_values(j_cache, jnp.zeros((0,), dtype=jnp.int32), empty_vals)
        is j_cache
    )
    empty_words = jnp.zeros((0, ROW_WORDS), dtype=jnp.int32)
    assert (
        scatter_cache_words(j_cache, jnp.zeros((0,), dtype=jnp.int32), empty_words)
        is j_cache
    )

    with patch("jax.default_backend", return_value="cpu"):
        # Sub-row on CPU
        s_sub = scatter_cache_values(
            j_cache,
            jnp.array([0, 1], dtype=jnp.int32),
            jnp.ones((2, 64), dtype=jnp.bfloat16),
        )
        assert s_sub.shape == j_cache.shape
        # Full-row / multi-row on CPU (fp32)
        s_full = scatter_cache_values(
            j_cache,
            jnp.array([0, 1], dtype=jnp.int32),
            jnp.ones((2, ROW_WORDS * 2), dtype=jnp.float32),
        )
        assert s_full.shape == j_cache.shape

    # Sub-row on TPU dispatches to write_cache_words
    with (
        patch("jax.default_backend", return_value="tpu"),
        patch(
            "vllm_torchtpu.kernels.deepseek_v41.cache_write.write_cache_words",
            return_value="tpu_subrow_written",
        ),
    ):
        res_sub_tpu = scatter_cache_words(
            j_cache,
            jnp.array([0, 1], dtype=jnp.int32),
            jnp.ones((2, 32), dtype=jnp.int32),
        )
        assert res_sub_tpu == "tpu_subrow_written"

    # Full-row on TPU dispatches to pl.pallas_call and executes kernel body
    def _fake_pallas_call(kernel_fn, **kwargs):
        def _runner(records, flat_words, cache):
            sem = MagicMock()
            with (
                patch("jax.experimental.pallas.program_id", side_effect=[0, 0]),
                patch(
                    "jax.experimental.pallas.when",
                    side_effect=lambda cond: (lambda fn: fn()),
                ),
                patch(
                    "jax.experimental.pallas.ds",
                    side_effect=lambda s, sz: slice(s, s + sz),
                ),
                patch(
                    "jax.experimental.pallas.tpu.make_async_copy",
                    return_value=MagicMock(),
                ),
            ):
                kernel_fn(
                    records,
                    flat_words,
                    cache,
                    cache,
                    jnp.zeros((ROW_WORDS,), dtype=jnp.int32),
                    sem,
                )
            return cache

        return _runner

    with (
        patch("jax.default_backend", return_value="tpu"),
        patch("jax.experimental.pallas.pallas_call", side_effect=_fake_pallas_call),
    ):
        res_full_tpu = scatter_cache_words(
            j_cache,
            jnp.array([0, 1], dtype=jnp.int32),
            jnp.ones((2, ROW_WORDS), dtype=jnp.int32),
        )
        assert res_full_tpu.shape == j_cache.shape


def test_cache_slots_and_row_slots():
    empty_meta = SimpleNamespace(
        seq_lens=torch.zeros((0,), dtype=torch.int32),
        block_tables=torch.zeros((0, 4), dtype=torch.int32),
        query_start_loc=torch.tensor([0], dtype=torch.int32),
    )
    pos = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    assert torch.equal(
        cache_slots(empty_meta, pos, rows_per_page=4),
        torch.full((4,), -1, dtype=torch.int32),
    )

    meta = SimpleNamespace(
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        block_tables=torch.tensor([[2, 3], [5, 6]], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
    )
    assert block_table(meta).shape == (2, 2)
    reqs = token_requests(meta, pos)
    assert reqs.dtype == torch.int32

    # compress_ratio == 1, power-of-two rows_per_page, ring=False
    s1 = cache_slots(meta, pos, rows_per_page=4, compress_ratio=1, ring=False)
    assert s1[-1].item() == -1
    # power-of-two compress_ratio, ring=True, explicit int64 requests
    s2 = cache_slots(
        meta,
        pos.int(),
        rows_per_page=4,
        compress_ratio=2,
        ring=True,
        requests=reqs.long(),
    )
    assert s2.shape == (4,)
    # non-power-of-two compress_ratio and non-power-of-two rows_per_page
    s3 = cache_slots(
        meta, pos, rows_per_page=3, compress_ratio=3, ring=False, requests=reqs
    )
    assert s3.shape == (4,)

    # row_slots: power-of-two and non-power-of-two rows_per_page, int32 and int64 inputs
    bt_i64 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    bt_i32 = bt_i64.int()
    rq_i64 = torch.tensor([0, 1], dtype=torch.int64)
    rq_i32 = rq_i64.int()
    logical_i64 = torch.tensor([[0, 3, 5], [1, 4, 7]], dtype=torch.int64)
    logical_i32 = logical_i64.int()

    rs_pow2_64 = row_slots(bt_i64, rq_i64, logical_i64, rows_per_page=4)
    rs_pow2_32 = row_slots(bt_i32, rq_i32, logical_i32, rows_per_page=4)
    assert torch.equal(rs_pow2_64, rs_pow2_32)

    rs_npow2 = row_slots(bt_i32, rq_i32, logical_i32, rows_per_page=3)
    assert rs_npow2.shape == (2, 3)
