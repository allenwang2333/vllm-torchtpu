# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for TD7.34 deepseek_v41_backend, ced_attention_op, and stage_timing."""

import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import vllm_torchtpu.runner.stage_timing as stage_timing_mod
from vllm_torchtpu.kernels.deepseek_v4.csa_cache_layout import ROW_WORDS
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    deepseek_v41_ced_attention_op as ced_op,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_backend import (
    AttentionBackend,
)


def test_attention_backend_contract() -> None:
    assert AttentionBackend.get_name() == "TORCH_TPU_DEEPSEEK_V41"
    assert AttentionBackend.get_kv_cache_shape(
        16, 64, 4, 128, "extra_arg", extra_kw=True
    ) == (16, 4, 64, 128)


def test_swa_attention_state_empty_unrolled_and_fori_loop() -> None:
    tokens, heads, dim = 2, 4, 8
    q_bf16 = jnp.ones((tokens, heads, dim), dtype=jnp.bfloat16) * 0.25
    q_f32 = q_bf16.astype(jnp.float32)
    pool = jnp.zeros((16, ROW_WORDS), dtype=jnp.int32)

    # Empty width == 0 early return
    empty_records = jnp.zeros((tokens, 0), dtype=jnp.int32)
    m0, d0, n0 = ced_op.swa_attention_state(q_bf16, pool, empty_records, 0.5)
    assert m0.shape == (tokens, heads)
    assert d0.shape == (tokens, heads)
    assert n0.shape == (tokens, heads, dim)
    assert jnp.all(d0 == 0.0)
    assert jnp.all(n0 == 0.0)

    # Unrolled path: steps <= 8 (e.g., width = 4 -> padded to 64 -> steps = 1)
    records_short = jnp.array([[0, 1, -1, 2], [1, -1, -1, -1]], dtype=jnp.int32)
    m1, d1, n1 = ced_op.swa_attention_state(q_bf16, pool, records_short, 0.5)
    m1_f32, d1_f32, n1_f32 = ced_op.swa_attention_state(q_f32, pool, records_short, 0.5)
    np.testing.assert_allclose(np.asarray(m1), np.asarray(m1_f32))
    np.testing.assert_allclose(np.asarray(d1), np.asarray(d1_f32))
    np.testing.assert_allclose(np.asarray(n1), np.asarray(n1_f32))

    # fori_loop path: steps > 8 (width = 9 * 64 = 576 -> steps = 9)
    records_long = jnp.full((tokens, 9 * 64), -1, dtype=jnp.int32)
    records_long = records_long.at[:, :4].set(records_short)
    m9, d9, n9 = ced_op.swa_attention_state(q_f32, pool, records_long, 0.5)
    np.testing.assert_allclose(np.asarray(m9), np.asarray(m1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(d9), np.asarray(d1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(n9), np.asarray(n1), rtol=1e-5, atol=1e-5)


def test_ced_attention_jax_and_initialize_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
        deepseek_v4_attention_op as attention_ops,
    )
    from vllm_torchtpu.models.vllm import vllm_model_wrapper_context as ctx_mod

    tokens, heads, dim = 2, 2, 8
    q = jnp.ones((tokens, heads, dim), dtype=jnp.bfloat16)
    swa = jnp.zeros((16, ROW_WORDS), dtype=jnp.int32)
    main_nope = jnp.zeros((4, 8), dtype=jnp.int32)
    main_rope = jnp.zeros((4, 8), dtype=jnp.int32)
    swa_slots = jnp.array([[0, 1], [2, -1]], dtype=jnp.int32)
    page_indices = jnp.zeros((2, 4), dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 1, 2], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)
    sink = jnp.zeros((heads,), dtype=jnp.bfloat16)

    # Branch 1: selected.shape[1] == 0 (SWA-only return path)
    selected_empty = jnp.zeros((tokens, 0), dtype=jnp.int32)
    out_swa_only = ced_op.ced_attention_jax(
        q,
        swa,
        main_nope,
        main_rope,
        swa_slots,
        selected_empty,
        page_indices,
        cu_q_lens,
        distribution,
        sink,
        scale=0.5,
        logical_main_page_size=64,
        page_size=64,
        page_stride=1,
        row_offset=0,
    )
    assert out_swa_only.shape == (tokens, heads, dim)
    assert out_swa_only.dtype == jnp.bfloat16

    # Branch 2: selected.shape[1] > 0 (delegates to _attention_csa_from_swa)
    csa_calls: list[dict] = []

    def _fake_csa(*args, **kwargs):
        csa_calls.append({"args": args, "kwargs": kwargs})
        return jnp.full_like(args[0], 3.0)

    monkeypatch.setattr(attention_ops, "_attention_csa_from_swa", _fake_csa)
    selected_nonempty = jnp.array([[0, 1], [1, -1]], dtype=jnp.int32)
    out_csa = ced_op.ced_attention_jax(
        q,
        swa,
        main_nope,
        main_rope,
        swa_slots,
        selected_nonempty,
        page_indices,
        cu_q_lens,
        distribution,
        sink,
        scale=0.5,
        logical_main_page_size=64,
        page_size=64,
        page_stride=1,
        row_offset=0,
    )
    assert len(csa_calls) == 1
    assert out_csa.shape == (tokens, heads, dim)

    # Test initialize_ced_attention() and attend / register_fake
    monkeypatch.setattr(ced_op, "_attention_op", None)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    monkeypatch.setattr(
        ctx_mod,
        "get_vllm_model_wrapper_context",
        lambda: SimpleNamespace(mesh=mesh),
    )

    captured_jax_op: dict = {}

    class _FakeOp:
        def __init__(self, fn):
            self.fn = fn
            self.fake_fn = None
            self.calls = []

        def register_fake(self, fake_fn):
            self.fake_fn = fake_fn

        def __call__(self, *args):
            self.calls.append(args)
            return torch.zeros_like(args[0])

    def _fake_pallas_jax_op(name, fn, **kwargs):
        op = _FakeOp(fn)
        captured_jax_op["name"] = name
        captured_jax_op["op"] = op
        return op

    from torch_tpu._internal import pallas

    monkeypatch.setattr(pallas, "jax_op", _fake_pallas_jax_op)

    ced_op.initialize_ced_attention()
    op_instance = captured_jax_op["op"]
    assert ced_op._attention_op is op_instance

    # Second call hits early return (`if _attention_op is not None: return`)
    ced_op.initialize_ced_attention()
    assert ced_op._attention_op is op_instance

    # Execute captured `attend` inner function and `register_fake` callback
    attend_out = op_instance.fn(
        q,
        swa,
        main_nope,
        main_rope,
        swa_slots,
        selected_empty,
        page_indices,
        cu_q_lens,
        distribution,
        sink,
        0.5,
        64,
        64,
        1,
        0,
    )
    assert attend_out.shape == (tokens, heads, dim)

    q_t = torch.zeros((tokens, heads, dim), dtype=torch.bfloat16)
    fake_out = op_instance.fake_fn(q_t, "ignored")
    assert fake_out.shape == q_t.shape

    # Test ced_attention with int32 (fast path) and int64 (.int() conversion)
    metadata = SimpleNamespace(
        block_tables=torch.zeros((2, 4), dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
    )
    layout = SimpleNamespace(geometry=(64, 1, 0))
    slots_i32 = torch.tensor([[0, 1], [2, -1]], dtype=torch.int32)
    sel_i32 = torch.tensor([[0, 1], [1, -1]], dtype=torch.int32)
    out_t1 = ced_op.ced_attention(
        q_t,
        torch.zeros((16, ROW_WORDS), dtype=torch.int32),
        (
            torch.zeros((4, 8), dtype=torch.int32),
            torch.zeros((4, 8), dtype=torch.int32),
        ),
        slots_i32,
        sel_i32,
        metadata,
        torch.zeros((heads,), dtype=torch.bfloat16),
        0.5,
        64,
        layout=layout,
    )
    assert out_t1.shape == q_t.shape
    assert op_instance.calls[-1][4] is slots_i32
    assert op_instance.calls[-1][5] is sel_i32

    out_t2 = ced_op.ced_attention(
        q_t,
        torch.zeros((16, ROW_WORDS), dtype=torch.int32),
        (
            torch.zeros((4, 8), dtype=torch.int32),
            torch.zeros((4, 8), dtype=torch.int32),
        ),
        slots_i32.long(),
        sel_i32.long(),
        metadata,
        torch.zeros((heads,), dtype=torch.bfloat16),
        0.5,
        64,
        layout=layout,
    )
    assert out_t2.shape == q_t.shape
    assert op_instance.calls[-1][4].dtype == torch.int32
    assert op_instance.calls[-1][5].dtype == torch.int32


def test_runner_stage_timing_cpu_and_device_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTpuEvent:
        def __init__(self) -> None:
            self.recorded = False
            self.synced = False

        def record(self) -> None:
            self.recorded = True

        def synchronize(self) -> None:
            self.synced = True

    monkeypatch.setattr(stage_timing_mod, "TpuEvent", _FakeTpuEvent)

    timing = stage_timing_mod.RunnerStageTiming(rank=3)

    # 1. host_stage
    with timing.host_stage("host_prep"):
        pass

    # 2. stage (normal path)
    with timing.stage("forward"):
        pass

    # 3. stage when body raises (still closes pending timing in finally)
    with pytest.raises(ValueError, match="boom"), timing.stage("failing_stage"):
        raise ValueError("boom")

    # Wait briefly for background collector thread to drain the 2 device stages
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with timing._samples_lock:
            if len(timing._device_samples) >= 2:
                break
        time.sleep(0.005)

    # 4. stage fallback when _slots semaphore is exhausted
    monkeypatch.setattr(timing._slots, "acquire", lambda blocking=False: False)
    with timing.stage("overflow_stage"):
        pass

    cpu_samples, device_samples = timing.take_samples()
    assert [s[1] for s in cpu_samples] == [
        "host_prep",
        "forward",
        "failing_stage",
        "overflow_stage",
    ]
    assert [s[1] for s in device_samples] == ["forward", "failing_stage"]
    assert all(s[0] == "3" for s in cpu_samples)
    assert all(s[0] == "3" for s in device_samples)

    # Calling take_samples again returns empty lists
    assert timing.take_samples() == ([], [])

    # 5. Exercise _collect_device_time on the main thread for deterministic
    # coverage tracing without requiring --concurrency=thread
    start_ev = _FakeTpuEvent()
    end_ev = _FakeTpuEvent()
    start_ev.record()
    end_ev.record()
    item = stage_timing_mod._PendingTiming("direct_stage", start_ev, end_ev)  # type: ignore[arg-type]
    item.end_recorded.set()

    items = [item]

    def _get_once():
        if items:
            return items.pop(0)
        raise StopIteration("done")

    timing._slots = SimpleNamespace(release=lambda: None)  # type: ignore[assignment]
    timing._pending = SimpleNamespace(get=_get_once)  # type: ignore[assignment]
    with pytest.raises(StopIteration, match="done"):
        timing._collect_device_time()

    _, dev2 = timing.take_samples()
    assert len(dev2) == 1
    assert dev2[0][1] == "direct_stage"
