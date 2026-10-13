# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement + branch unit tests for DeepSeek-V4.1 CED SWA attention op."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "vllm_torchtpu"


@pytest.fixture(autouse=True)
def _mock_torch_tpu_and_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    if "torch_tpu" not in sys.modules:
        tt_mod = types.ModuleType("torch_tpu")
        tt_int = types.ModuleType("torch_tpu._internal")
        pallas_mod = types.ModuleType("torch_tpu._internal.pallas")

        class _FakeOp:
            def __init__(self, fn):
                self.fn = fn
                self.fake_fn = None

            def register_fake(self, fake_fn):
                self.fake_fn = fake_fn

            def __call__(self, *args):
                return self.fn(*args)

        pallas_mod.jax_op = lambda name, fn, **kwargs: _FakeOp(fn)
        tt_int.pallas = pallas_mod
        tt_mod._internal = tt_int
        monkeypatch.setitem(sys.modules, "torch_tpu", tt_mod)
        monkeypatch.setitem(sys.modules, "torch_tpu._internal", tt_int)
        monkeypatch.setitem(sys.modules, "torch_tpu._internal.pallas", pallas_mod)

    for pkg_name, rel_dir in (
        ("vllm_torchtpu.layers", "layers"),
        ("vllm_torchtpu.layers.adapter", "layers/adapter"),
        ("vllm_torchtpu.layers.adapter.custom_ops", "layers/adapter/custom_ops"),
        (
            "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41",
            "layers/adapter/custom_ops/deepseek_v41",
        ),
        ("vllm_torchtpu.models", "models"),
        ("vllm_torchtpu.models.vllm", "models/vllm"),
    ):
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(_SRC_ROOT / rel_dir)]
            monkeypatch.setitem(sys.modules, pkg_name, pkg)

    dsv4_pkg_name = "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4"
    dsv4_op_name = f"{dsv4_pkg_name}.deepseek_v4_attention_op"
    if dsv4_op_name not in sys.modules:
        dsv4_pkg = types.ModuleType(dsv4_pkg_name)
        dsv4_op = types.ModuleType(dsv4_op_name)
        dsv4_op._attention_csa_from_swa = lambda q, *a, **kw: q  # type: ignore[attr-defined]
        dsv4_pkg.deepseek_v4_attention_op = dsv4_op  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, dsv4_pkg_name, dsv4_pkg)
        monkeypatch.setitem(sys.modules, dsv4_op_name, dsv4_op)

    ctx_name = "vllm_torchtpu.models.vllm.vllm_model_wrapper_context"
    if ctx_name not in sys.modules:
        ctx_mod = types.ModuleType(ctx_name)
        ctx_mod.get_vllm_model_wrapper_context = lambda: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, ctx_name, ctx_mod)

    if jax.default_backend() == "cpu":
        from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
            deepseek_v41_cache as cache_mod,
        )
        from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
            deepseek_v41_ced_attention_op as ced_mod,
        )

        orig_gather = cache_mod.gather_cache_values

        def _cpu_gather_f32(
            pool: jax.Array, indices: jax.Array, width: int
        ) -> jax.Array:
            return orig_gather(pool, indices, width).astype(jnp.float32)

        monkeypatch.setattr(cache_mod, "gather_cache_values", _cpu_gather_f32)
        monkeypatch.setattr(ced_mod, "gather_cache_values", _cpu_gather_f32)


def _reference_swa_attention_state(
    q: jax.Array,
    pool: jax.Array,
    records: jax.Array,
    scale: float,
):
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
        deepseek_v41_cache as cache_mod,
    )

    tokens, heads, dim = q.shape
    tile = 64
    state = (
        jnp.full((tokens, heads), -1e30, jnp.float32),
        jnp.zeros((tokens, heads), jnp.float32),
        jnp.zeros(q.shape, jnp.float32),
    )
    width = records.shape[1]
    if not width:
        return state
    records = jnp.pad(records, ((0, 0), (0, -width % tile)), constant_values=-1)
    gathered = cache_mod.gather_cache_values(
        pool, jnp.maximum(records, 0).reshape(-1), dim
    ).reshape(tokens, records.shape[1], dim)
    q_f32 = q if q.dtype == jnp.float32 else q.astype(jnp.float32)

    for index in range(records.shape[1] // tile):
        maximum, denominator, numerator = state
        start = index * tile
        addresses = records[:, start : start + tile]
        kv = gathered[:, start : start + tile]
        valid = addresses >= 0
        kv = jnp.where(valid[..., None], kv, 0)
        scores = (
            jnp.einsum(
                "thd,tkd->thk",
                q_f32,
                kv.astype(jnp.float32),
                preferred_element_type=jnp.float32,
            )
            * scale
        )
        scores = jnp.where(valid[:, None], scores, -jnp.inf)
        new_max = jnp.maximum(maximum, scores.max(-1))
        correction = jnp.exp(maximum - new_max)
        exponentials = jnp.exp(scores - new_max[..., None])
        denominator = denominator * correction + exponentials.sum(-1)
        numerator = numerator * correction[..., None] + jnp.einsum(
            "thk,tkd->thd",
            exponentials.astype(jnp.bfloat16),
            kv,
            preferred_element_type=jnp.float32,
        )
        state = (new_max, denominator, numerator)
    return state


def test_swa_attention_state_empty_padded_aligned_and_fori_loop() -> None:
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
        deepseek_v41_ced_attention_op as ced_mod,
    )

    swa_attention_state = ced_mod.swa_attention_state

    rng = np.random.default_rng(7)
    dim = 128
    scale = float(dim**-0.5)
    pool_bf16 = jnp.asarray(
        rng.standard_normal((256, dim), dtype=np.float32), dtype=jnp.bfloat16
    )
    pool_words = jax.lax.bitcast_convert_type(
        pool_bf16.reshape(-1, 2, 128).swapaxes(1, 2), jnp.int32
    ).reshape(-1, 128)

    q_bf16 = jnp.asarray(
        rng.standard_normal((2, 4, dim), dtype=np.float32), dtype=jnp.bfloat16
    )
    q_f32 = q_bf16.astype(jnp.float32)

    # 1. Empty width == 0
    empty_recs = jnp.zeros((2, 0), dtype=jnp.int32)
    m0, d0, n0 = swa_attention_state(q_bf16, pool_words, empty_recs, scale)
    assert m0.shape == (2, 4) and d0.shape == (2, 4) and n0.shape == (2, 4, dim)
    assert np.all(np.asarray(d0) == 0.0)

    # 2. Unaligned width = 100 (pad_width != 0, steps = 2 <= 8) with float32 q
    recs_unaligned = jnp.asarray(rng.integers(-5, 200, size=(2, 100), dtype=np.int32))
    got_u = swa_attention_state(q_f32, pool_words, recs_unaligned, scale)
    ref_u = _reference_swa_attention_state(q_f32, pool_words, recs_unaligned, scale)
    for g, r in zip(got_u, ref_u):
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-5, atol=1e-5)

    # 3. Aligned width = 128 (pad_width == 0, steps = 2 <= 8) with bfloat16 q
    recs_aligned = jnp.asarray(rng.integers(0, 200, size=(2, 128), dtype=np.int32))
    got_a = swa_attention_state(q_bf16, pool_words, recs_aligned, scale)
    ref_a = _reference_swa_attention_state(q_bf16, pool_words, recs_aligned, scale)
    for g, r in zip(got_a, ref_a):
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-5, atol=1e-5)

    # 4. Large window width = 832 (steps = 13 > 12 -> executes fori_loop)
    recs_long = jnp.asarray(rng.integers(-4, 200, size=(2, 832), dtype=np.int32))
    got_l = swa_attention_state(q_bf16, pool_words, recs_long, scale)
    ref_l = _reference_swa_attention_state(q_bf16, pool_words, recs_long, scale)
    for g, r in zip(got_l, ref_l):
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-5, atol=1e-5)


def test_ced_attention_jax_and_initialize_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_torchtpu.models.vllm.vllm_model_wrapper_context as ctx_mod
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
        deepseek_v41_ced_attention_op as ced_mod,
    )

    rng = np.random.default_rng(19)
    dim = 128
    scale = float(dim**-0.5)
    pool_bf16 = jnp.asarray(
        rng.standard_normal((128, dim), dtype=np.float32), dtype=jnp.bfloat16
    )
    swa_pool = jax.lax.bitcast_convert_type(
        pool_bf16.reshape(-1, 2, 128).swapaxes(1, 2), jnp.int32
    ).reshape(-1, 128)
    q = jnp.asarray(
        rng.standard_normal((2, 2, dim), dtype=np.float32), dtype=jnp.bfloat16
    )
    main_nope = jnp.zeros((4, 128), dtype=jnp.int32)
    main_rope = jnp.zeros((4, 128), dtype=jnp.int32)
    swa_slots = jnp.asarray([[0, 1, -1, -1], [2, 3, 4, -1]], dtype=jnp.int32)
    page_indices = jnp.zeros((2, 4), dtype=jnp.int32)
    cu_q_lens = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    distribution = jnp.asarray([0, 0, 2], dtype=jnp.int32)
    sink_bf16 = jnp.zeros((2,), dtype=jnp.bfloat16)
    sink_f32 = jnp.zeros((2,), dtype=jnp.float32)

    # 1. selected.shape[1] == 0 path (SWA + sink only)
    out_zero_sel = ced_mod.ced_attention_jax(
        q,
        swa_pool,
        main_nope,
        main_rope,
        swa_slots,
        jnp.zeros((2, 0), dtype=jnp.int32),
        page_indices,
        cu_q_lens,
        distribution,
        sink_bf16,
        scale=scale,
        logical_main_page_size=64,
        page_size=64,
        page_stride=64,
        row_offset=0,
    )
    assert out_zero_sel.shape == q.shape and out_zero_sel.dtype == jnp.bfloat16

    # 2. selected.shape[1] > 0 path (delegates to _attention_csa_from_swa)
    captured = {}

    def _fake_csa_from_swa(q_in, *args, **kwargs):
        captured["kwargs"] = kwargs
        return q_in

    monkeypatch.setattr(
        ced_mod.attention_ops, "_attention_csa_from_swa", _fake_csa_from_swa
    )
    out_sel = ced_mod.ced_attention_jax(
        q,
        swa_pool,
        main_nope,
        main_rope,
        swa_slots,
        jnp.zeros((2, 16), dtype=jnp.int32),
        page_indices,
        cu_q_lens,
        distribution,
        sink_f32,
        scale=scale,
        logical_main_page_size=64,
        page_size=64,
        page_stride=64,
        row_offset=0,
    )
    assert out_sel.shape == q.shape
    assert captured["kwargs"]["attention_kernel_batch_size"] == 4

    # 3. initialize_ced_attention (first call + idempotent second call) & ced_attention
    mesh = Mesh(np.array(jax.devices()[:1]), ("model",))
    monkeypatch.setattr(
        ctx_mod,
        "get_vllm_model_wrapper_context",
        lambda: SimpleNamespace(mesh=mesh),
    )
    monkeypatch.setattr(ced_mod, "_attention_op", None)
    ced_mod.initialize_ced_attention()
    assert ced_mod._attention_op is not None
    # Idempotent early return branch
    ced_mod.initialize_ced_attention()

    # Exercise fake_fn registration
    q_torch = torch.zeros((2, 2, dim), dtype=torch.bfloat16)
    fake_out = ced_mod._attention_op.fake_fn(q_torch)
    assert fake_out.shape == q_torch.shape

    # Exercise attend() shard_map body via _attention_op
    attend_res = ced_mod._attention_op.fn(
        q,
        swa_pool,
        main_nope,
        main_rope,
        swa_slots,
        jnp.zeros((2, 16), dtype=jnp.int32),
        page_indices,
        cu_q_lens,
        distribution,
        sink_f32,
        scale,
        64,
        64,
        64,
        0,
    )
    assert attend_res.shape == q.shape

    # Exercise ced_attention with both int64 and int32 slots/selected
    recorded_args = []
    monkeypatch.setattr(
        ced_mod,
        "_attention_op",
        lambda *a: (recorded_args.append(a), a[0])[1],
    )
    meta = SimpleNamespace(
        block_tables="bt",
        query_start_loc="qsl",
        request_distribution="dist",
    )
    layout = SimpleNamespace(page_size=64, page_stride=64, row_offset=0)
    res_i64 = ced_mod.ced_attention(
        q_torch,
        "swa",
        ("nope", "rope"),
        torch.zeros((2, 4), dtype=torch.int64),
        torch.zeros((2, 8), dtype=torch.int64),
        meta,
        "sink",
        scale,
        64,
        layout=layout,
    )
    assert res_i64 is q_torch
    assert recorded_args[-1][4].dtype == torch.int32
    assert recorded_args[-1][5].dtype == torch.int32

    res_i32 = ced_mod.ced_attention(
        q_torch,
        "swa",
        ("nope", "rope"),
        torch.zeros((2, 4), dtype=torch.int32),
        torch.zeros((2, 8), dtype=torch.int32),
        meta,
        "sink",
        scale,
        64,
        layout=layout,
    )
    assert res_i32 is q_torch
