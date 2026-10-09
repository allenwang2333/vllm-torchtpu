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
"""TD7.102: all-gather matmul host path, kernel body, collectives util,
MLA scatter wrapper and masked-dense cost model on CPU."""

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.collectives import all_gather_matmul as agm
from vllm_torchtpu.kernels.collectives import (
    all_gather_matmul_tuned_block_sizes as agm_tbs,
)
from vllm_torchtpu.kernels.collectives import util as coll_util
from vllm_torchtpu.kernels.mla import kv_cache_utils, scatter, scatter_utils
from vllm_torchtpu.kernels.mla.masked_dense import cost_model


@pytest.fixture(autouse=True)
def _clear_caches():
    agm._build_all_gather_matmul.cache_clear()
    agm_tbs.get_tpu_version.cache_clear()
    yield
    agm._build_all_gather_matmul.cache_clear()
    agm_tbs.get_tpu_version.cache_clear()


# ---------------------------------------------------------------- tuned sizes


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("cpu", -1),
        ("TPU v4", 4),
        ("TPU v5 lite", 5),
        ("TPU v5e", 5),
        ("TPU v5p", 5),
        ("TPU v6 lite", 6),
        ("TPU7x", 7),
    ],
)
def test_get_tpu_version_parses_and_caches(kind, expected):
    calls = []

    def devices():
        calls.append(1)
        return [SimpleNamespace(device_kind=kind)]

    with patch.object(agm_tbs.jax, "devices", devices):
        assert agm_tbs.get_tpu_version() == expected
        assert agm_tbs.get_tpu_version() == expected
    assert len(calls) == 1


@pytest.mark.cpu_test
def test_get_tpu_version_rejects_unknown_kind():
    with (
        patch.object(
            agm_tbs.jax, "devices", lambda: [SimpleNamespace(device_kind="TPU vXY")]
        ),
        pytest.raises(AssertionError),
    ):
        agm_tbs.get_tpu_version()


@pytest.mark.cpu_test
def test_tuned_block_sizes_hit_and_miss():
    with patch.object(agm_tbs, "get_tpu_version", lambda: 6):
        assert agm_tbs.get_key(1, 2, 3, "bfloat16", 8) == (6, 1, 2, 3, "bfloat16", 8)
        assert agm_tbs.get_tuned_block_sizes(1024, 51200, 5120, "bfloat16", 8) == (
            6400,
            2560,
        )
        assert agm_tbs.get_tuned_block_sizes(8, 128, 128, "bfloat16", 8) == (
            None,
            None,
        )


# ------------------------------------------------------------ host / dispatch


@pytest.mark.cpu_test
def test_validate_inputs_errors():
    def a(*shape, dtype=jnp.bfloat16):
        return jax.ShapeDtypeStruct(shape, dtype)

    agm.validate_inputs(a(32, 256), a(256, 384), 1)
    agm.validate_inputs(a(32, 256), a(384, 256), 1, rhs_transpose=True)
    cases = [
        ((a(32, 256, 1), a(256, 384), 1), "2D"),
        ((a(32, 256), a(256, 384, dtype=jnp.float32), 1), "dtypes must match"),
        ((a(32, 256), a(128, 384), 1), "contracting dimension"),
        ((a(32, 200), a(200, 384), 1), "k \\(200\\) must be divisible by 128"),
        ((a(32, 256), a(256, 200), 1), "n \\(200\\) must be divisible by 128"),
        ((a(24, 256), a(256, 384), 1), "must be divisible by 16\\.$"),
        ((a(33, 256), a(256, 384), 1), "tp_size \\* 2"),
        ((a(80, 256), a(256, 384), 5), "tp_size \\(5\\) on axis"),
    ]
    for args, match in cases:
        with pytest.raises(ValueError, match=match):
            agm.validate_inputs(*args)


@pytest.mark.cpu_test
def test_vmem_estimate_and_kernel_name():
    got = agm.get_vmem_estimate_bytes(
        64, 512, 256, 128, 4096, 2, jnp.bfloat16, jnp.bfloat16, jnp.bfloat16
    )
    assert got == 2 * 32 * 256 * 2 + 256 * 256 * 2 + 2 * 64 * 128 * 2 + 4096
    assert (
        agm.get_kernel_name(128, 256, True)
        == "all_gather_matmul_kernel_bn_128_bk_256_rhs_transpose_True"
    )


def _fake_pallas_call(captured):
    def fake(kernel, *, out_shape, grid_spec, compiler_params, cost_estimate, name):
        captured.append(
            SimpleNamespace(
                keywords=kernel.keywords,
                out_shape=out_shape,
                grid_spec=grid_spec,
                compiler_params=compiler_params,
                name=name,
            )
        )

        def run(x, y):
            rhs_t = kernel.keywords["rhs_transpose"]
            g = jax.lax.all_gather(x, kernel.keywords["axis_name"], tiled=True)
            out = jnp.dot(g, y.T if rhs_t else y, preferred_element_type=jnp.float32)
            return out.astype(x.dtype), jnp.zeros(out_shape[1].shape, x.dtype)

        return run

    return fake


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("kwargs", "tuned", "want_bn", "want_bk", "want_acc"),
    [
        ({}, (None, None), 384, 256, (8, 128)),
        ({}, (128, 128), 128, 128, (32, 128)),
        ({"bn": 128, "bk": 256, "rhs_transpose": True}, (None, None), 128, 256, None),
        ({"rhs_transpose": True, "collective_id": 3}, (None, None), 384, 256, None),
    ],
)
def test_all_gather_matmul_matches_reference_and_caches(
    kwargs, tuned, want_bn, want_bk, want_acc
):
    captured = []
    mesh = jax.sharding.Mesh(jax.devices()[:1], ("tp",))
    x = (jnp.arange(32 * 256, dtype=jnp.float32).reshape(32, 256) % 7).astype(
        jnp.bfloat16
    )
    y = (jnp.arange(256 * 384, dtype=jnp.float32).reshape(256, 384) % 5).astype(
        jnp.bfloat16
    )
    rhs_t = kwargs.get("rhs_transpose", False)
    yy = y.T if rhs_t else y
    with (
        patch.object(agm.pl, "pallas_call", _fake_pallas_call(captured)),
        patch.object(
            agm.all_gather_matmul_tuned_block_sizes,
            "get_tuned_block_sizes",
            lambda *a: tuned,
        ),
    ):
        out1 = agm.all_gather_matmul(x, yy, mesh, "tp", **kwargs)
        out2 = agm.all_gather_matmul(x, yy, mesh, "tp", **kwargs)
    ref = jnp.dot(x, y, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(out1), np.asarray(ref))
    np.testing.assert_array_equal(np.asarray(out2), np.asarray(ref))
    # Built and traced once for both calls.
    assert len(captured) == 1
    assert agm._build_all_gather_matmul.cache_info().hits == 1
    cap = captured[0]
    assert cap.keywords["bn"] == want_bn and cap.keywords["bk"] == want_bk
    assert cap.compiler_params.collective_id == kwargs.get("collective_id", 0)
    assert cap.name == agm.get_kernel_name(want_bn, want_bk, rhs_t)
    if want_acc is not None:
        assert cap.grid_spec.scratch_shapes[-1].shape == want_acc


# ---------------------------------------------------------------- kernel body


def _np_index(idx):
    # 0-d JAX indices must become ints: as arrays they make NumPy take a
    # fancy-indexing copy, and `.at[...]` stores would be lost.
    idx = idx if isinstance(idx, tuple) else (idx,)
    return tuple(i if isinstance(i, (slice, int, type(...))) else int(i) for i in idx)


class _Ref:
    """NumPy-backed stand-in for a Pallas ref (`.at[...]`, load, store)."""

    def __init__(self, data):
        self.data = data

    shape = property(lambda self: self.data.shape)
    dtype = property(lambda self: self.data.dtype)

    @property
    def at(self):
        ref = self

        class _At:
            def __getitem__(self, idx):
                return _Ref(ref.data[_np_index(idx)])

        return _At()

    def __getitem__(self, idx):
        return jnp.asarray(self.data[_np_index(idx)])

    def __setitem__(self, idx, value):
        self.data[_np_index(idx)] = np.asarray(value)

    def __jax_array__(self):
        return jnp.asarray(self.data)


class _Copy:
    def __init__(self, log, kind, kw):
        self.log, self.kind, self.kw = log, kind, kw

    def start(self):
        self.log.append((self.kind, "start"))

    def wait(self):
        self.log.append((self.kind, "wait"))


def _fake_modules(num_programs, program_ids, log):
    def when(cond):
        log.append(("when", bool(cond)))
        return lambda fn: fn()

    def run_scoped(fn, **scratch):
        return fn(**{k: f"scoped:{k}" for k in scratch})

    fake_pl = SimpleNamespace(
        num_programs=lambda i: num_programs[i],
        program_id=lambda i: program_ids[i],
        when=when,
        ds=lambda start, size: slice(int(start), int(start) + size),
        debug_print=lambda msg, *args: log.append(("debug", msg)),
        run_scoped=run_scoped,
    )
    fake_pltpu = SimpleNamespace(
        make_async_copy=lambda **kw: _Copy(log, "local", kw),
        make_async_remote_copy=lambda **kw: _Copy(log, "remote", kw),
        DeviceIdType=SimpleNamespace(MESH="mesh"),
        get_barrier_semaphore=lambda: "barrier",
        semaphore_signal=lambda sem, **kw: log.append(("signal", sem)),
        semaphore_wait=lambda sem, n: log.append(("sem_wait", sem, n)),
        SemaphoreType=SimpleNamespace(REGULAR="regular"),
    )
    fake_lax = SimpleNamespace(
        axis_index=lambda name: 0,
        rem=jax.lax.rem,
        dot_general=jax.lax.dot_general,
    )
    return fake_pl, fake_pltpu, fake_lax


def _run_kernel(*, grid_k, rhs_transpose, debug_mode, program_ids=(1, 0, 0)):
    m_per_device, k, n_per_device, num_devices = 16, 256, 256, 2
    bn, bk = 128, k // grid_k
    log = []
    fake_pl, fake_pltpu, fake_lax = _fake_modules(
        (num_devices + 2, n_per_device // bn, grid_k), program_ids, log
    )
    f32 = np.float32
    y_shape = (n_per_device, k) if rhs_transpose else (k, n_per_device)
    refs = dict(
        x_hbm_ref=_Ref(np.ones((m_per_device, k), f32)),
        y_hbm_ref=_Ref(np.ones(y_shape, f32)),
        o_hbm_ref=_Ref(np.zeros((m_per_device * num_devices, n_per_device), f32)),
        # Kernel indexes slots by outer_step; size generously for the fake.
        x_hbm_scratch_ref=_Ref(np.zeros((num_devices + 2, m_per_device, k), f32)),
        x_local_copy_sem="x_sem",
        y_local_copy_sem="y_sem",
        o_local_copy_sem="o_sem",
        send_sems=_Ref(np.zeros((2, num_devices + 2), f32)),
        recv_sems=_Ref(np.zeros((2, num_devices + 2), f32)),
        x_vmem_scratch_ref=_Ref(np.ones((2, m_per_device, k), f32)),
        y_vmem_scratch_ref=_Ref(np.ones(y_shape, f32)),
        o_vmem_scratch_ref=_Ref(np.zeros((2, m_per_device, bn), f32)),
        acc_vmem_scratch_ref=_Ref(np.zeros((m_per_device, bn), f32)),
    )
    with (
        patch.object(agm, "pl", fake_pl),
        patch.object(agm, "pltpu", fake_pltpu),
        patch.object(agm, "lax", fake_lax),
        patch.object(coll_util, "pl", fake_pl),
        patch.object(coll_util, "pltpu", fake_pltpu),
    ):
        agm._all_gather_kernel(
            **refs,
            axis_name="tp",
            bn=bn,
            bk=bk,
            debug_mode=debug_mode,
            rhs_transpose=rhs_transpose,
        )
    return log, refs


@pytest.mark.cpu_test
@pytest.mark.parametrize("grid_k", [1, 2])
@pytest.mark.parametrize("rhs_transpose", [False, True])
@pytest.mark.parametrize("debug_mode", [False, True])
def test_all_gather_kernel_body_issues_every_phase(grid_k, rhs_transpose, debug_mode):
    log, refs = _run_kernel(
        grid_k=grid_k, rhs_transpose=rhs_transpose, debug_mode=debug_mode
    )
    kinds = {(e[0], e[1]) for e in log if e[0] in ("local", "remote")}
    assert kinds == {
        ("local", "start"),
        ("local", "wait"),
        ("remote", "start"),
        ("remote", "wait"),
    }
    assert any(e[0] == "signal" for e in log)
    assert any(e[0] == "debug" for e in log) is debug_mode
    k = 256
    # x (ones) @ y (ones) over the whole contraction: every MXU output is k
    # (grid_k == 1) or the acc was flushed to o and re-zeroed (grid_k > 1).
    o = refs["o_vmem_scratch_ref"].data
    if grid_k == 1:
        assert any(np.all(o[s] == k) for s in range(2))
    else:
        assert np.all(refs["acc_vmem_scratch_ref"].data == 0)
        assert any(np.all(o[s] == k // grid_k) for s in range(2))


# ------------------------------------------------------------ collectives util


@pytest.mark.cpu_test
@pytest.mark.parametrize("double_barrier", [False, True])
def test_local_barrier(double_barrier):
    log = []
    fake_pl, fake_pltpu, _ = _fake_modules((1,), (0,), log)
    with (
        patch.object(coll_util, "pl", fake_pl),
        patch.object(coll_util, "pltpu", fake_pltpu),
    ):
        coll_util.local_barrier(3, 5, double_barrier=double_barrier)
    expected = [
        ("signal", "barrier"),
        ("signal", "barrier"),
        ("sem_wait", "barrier", 2),
    ]
    if double_barrier:
        sem = "scoped:second_barrier"
        expected += [("signal", sem), ("signal", sem), ("sem_wait", sem, 2)]
    assert log == expected


# ---------------------------------------------------------------- MLA scatter


def _spec(layout):
    return SimpleNamespace(
        shape=(4, 8),
        jax_dtype=jnp.int32,
        page_size=16,
        token_bytes=32,
        layout=layout,
        cache_type=SimpleNamespace(value="nope"),
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "layout",
    [kv_cache_utils.KVCacheLayout.TENSORCORE, SimpleNamespace(value="packed")],
)
@pytest.mark.parametrize("with_dst_rows", [False, True])
def test_scatter_wrapper_routes_layout_and_rows(layout, with_dst_rows):
    cache = jnp.zeros((4, 8), jnp.int32)
    values = jnp.ones((3, 5), jnp.float32)
    calls = {}

    def scatter_rows(cache, src, dst_rows, *, name):
        calls["scatter"] = (src, dst_rows, name)
        return cache + 1

    with (
        patch.object(kv_cache_utils, "as_token_bytes", lambda v, b: ("tc", b)),
        patch.object(kv_cache_utils, "pack_tokens", lambda v, b: ("packed", b)),
        patch.object(
            kv_cache_utils, "get_dst_rows", lambda **kw: ("rows", kw["page_size"])
        ),
        patch.object(scatter_utils, "scatter_rows", scatter_rows),
    ):
        kwargs = (
            {"dst_rows": "given"}
            if with_dst_rows
            else {"seq_lens": 1, "block_tables": 2, "query_start_loc": 3}
        )
        out = scatter.scatter(cache, values, _spec(layout), **kwargs)
    np.testing.assert_array_equal(np.asarray(out), np.ones((4, 8), np.int32))
    src, dst_rows, name = calls["scatter"]
    is_tc = layout is kv_cache_utils.KVCacheLayout.TENSORCORE
    assert src == (("tc" if is_tc else "packed"), 32)
    assert dst_rows == ("given" if with_dst_rows else ("rows", 16))
    assert name == f"scatter_nope_{layout.value}_cache"


@pytest.mark.cpu_test
def test_scatter_wrapper_asserts():
    spec = _spec(kv_cache_utils.KVCacheLayout.TENSORCORE)
    with pytest.raises(AssertionError, match="does not match spec"):
        scatter.scatter(jnp.zeros((4, 9), jnp.int32), jnp.ones((3, 5)), spec)
    with pytest.raises(AssertionError, match="pass dst_rows"):
        scatter.scatter(jnp.zeros((4, 8), jnp.int32), jnp.ones((3, 5)), spec)


# ------------------------------------------------------------ cost model


@pytest.mark.cpu_test
def test_masked_dense_cost_model_constants():
    model = cost_model.GLM52_TPU7X_PREFILL_COST_MODEL
    assert isinstance(model, cost_model.MaskedDensePrefillCostModel)
    assert model._fields == (
        "analytic_fixed_ns",
        "analytic_token_ns",
        "bitmap_fixed_ns",
        "bitmap_token_ns",
        "q_kv_block_ns",
        "sparse_fixed_ns",
        "sparse_token_ns",
    )
    assert all(isinstance(v, int) and v > 0 for v in model)
    assert (cost_model.KV_BLOCK_SIZE, cost_model.QUERY_BLOCK_SIZE) == (1024, 32)
