# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import jax
import pytest
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal.pallas import pallas

from vllm_torchtpu.distributed import pallas_shapes


@pytest.mark.parametrize(
    "spec,expected",
    [
        (P(), (3, 8, 16)),
        (P(None, "tp"), (3, 16, 16)),
        (P(None, ("tp", "pcp")), (3, 64, 16)),
        (P("pcp", "tp"), (12, 16, 16)),
    ],
)
def test_native_placeholder_local_global_roundtrip(monkeypatch, spec, expected):
    # The abstract Pallas devices do not require a TPU runtime.
    devices = [
        pallas._PallasDevice(id=i, process_index=i, device_kind="TPU v7")
        for i in range(8)
    ]
    import numpy as np

    mesh = jax.sharding.Mesh(np.array(devices).reshape(4, 2), ("pcp", "tp"))
    monkeypatch.setattr(pallas, "get_global_shape", pallas_shapes.get_global_shape)
    monkeypatch.setattr(pallas, "get_local_shape", pallas_shapes.get_local_shape)
    value = pallas.jax_placeholder(torch.zeros(3, 8, 16), mesh, spec)
    assert value.shape == expected
    # Inspect native output allocation without creating an actual TPU tensor.
    allocation = {}

    def empty(shape, **kwargs):
        allocation["shape"] = tuple(shape)
        return allocation

    monkeypatch.setattr(torch, "empty", empty)
    pallas.torch_placeholder(
        jax.core.ShapedArray(expected, value.dtype), mesh=mesh, partition_spec=spec
    )
    assert allocation["shape"] == (3, 8, 16)


def test_compound_axis_requires_divisible_output(monkeypatch):
    pallas_shapes.install_pallas_shape_conversion()
    assert pallas.get_global_shape is pallas_shapes.get_global_shape
    assert pallas.get_local_shape is pallas_shapes.get_local_shape

    assert pallas_shapes.get_global_shape((3, 8), None, P("tp")) == (3, 8)
    assert pallas_shapes.get_local_shape([3, 8], None, None) == (3, 8)
    assert pallas_shapes.get_global_shape(
        [3, 8], SimpleNamespace(shape={"tp": 2}), P()
    ) == (
        3,
        8,
    )

    with pytest.raises(ValueError, match="not divisible"):
        pallas_shapes.get_local_shape(
            (63,), SimpleNamespace(shape={"pcp": 4, "tp": 2}), P(("tp", "pcp"))
        )


def test_sharded_jax_op_full_coverage(monkeypatch):
    import inspect

    import jax.numpy as jnp

    from vllm_torchtpu.distributed import sharded_jax_op as sjo

    pallas_shapes.install_pallas_shape_conversion()
    mesh = SimpleNamespace(shape={"tp": 2})

    allocations = []

    def fake_empty(shape, *, dtype=None, device=None):
        res = SimpleNamespace(shape=tuple(shape), dtype=dtype, device=device)
        allocations.append(res)
        return res

    monkeypatch.setattr(torch, "empty", fake_empty)

    # 1. _placeholder branches: spec is None or non-ShapedArray, unsupported dtype,
    #    normal dtype, float4 empty shape, float4 odd last dim, float4 even last dim.
    fallback_sentinel = object()
    monkeypatch.setattr(
        sjo.pallas_impl, "torch_placeholder", lambda aval, mesh=None: fallback_sentinel
    )
    aval_f32 = jax.core.ShapedArray((4, 8), jnp.dtype(jnp.float32))
    assert sjo._placeholder(aval_f32, None, mesh) is fallback_sentinel
    assert sjo._placeholder(object(), P("tp"), mesh) is fallback_sentinel

    orig_shaped_array = jax.core.ShapedArray
    aval_bad = SimpleNamespace(shape=(4, 8), dtype="unsupported_dtype")
    monkeypatch.setattr(
        sjo.jax.core, "ShapedArray", (orig_shaped_array, SimpleNamespace)
    )
    with pytest.raises(NotImplementedError, match="Unsupported dtype"):
        sjo._placeholder(aval_bad, P("tp"), mesh)
    monkeypatch.setattr(sjo.jax.core, "ShapedArray", orig_shaped_array)

    out_f32 = sjo._placeholder(aval_f32, P("tp"), mesh)
    assert out_f32.shape == (2, 8)
    assert out_f32.dtype == torch.float32

    fp4_jax_dtype = next(
        dt
        for dt, t_dt in sjo.pallas_impl.JAX_TO_TORCH_DTYPE_MAP.items()
        if t_dt == torch.float4_e2m1fn_x2
    )
    aval_fp4_scalar = jax.core.ShapedArray((), fp4_jax_dtype)
    out_fp4_scalar = sjo._placeholder(aval_fp4_scalar, P(), mesh)
    assert out_fp4_scalar.shape == ()

    aval_fp4_odd = jax.core.ShapedArray((4, 3), fp4_jax_dtype)
    with pytest.raises(ValueError, match="expected the last dimension"):
        sjo._placeholder(aval_fp4_odd, P("tp"), mesh)

    aval_fp4_even = jax.core.ShapedArray((4, 6), fp4_jax_dtype)
    out_fp4_even = sjo._placeholder(aval_fp4_even, P("tp"), mesh)
    assert out_fp4_even.shape == (2, 3)

    # 2. _sharded_output_placeholders branches: missing vs matching shardings.
    lowered_no_sharding = SimpleNamespace(
        out_avals=[aval_f32], _out_named_shardings=None
    )
    assert sjo._sharded_output_placeholders(lowered_no_sharding, mesh) == [
        fallback_sentinel
    ]

    lowered_with_sharding = SimpleNamespace(
        out_avals=[aval_f32],
        _out_named_shardings=[SimpleNamespace(spec=P("tp"))],
    )
    placeholders = sjo._sharded_output_placeholders(lowered_with_sharding, mesh)
    assert len(placeholders) == 1 and placeholders[0].shape == (2, 8)

    # 3. _build_sharded_callable_cls: needle missing fallback vs compilation.
    orig_getsource = inspect.getsource
    monkeypatch.setattr(
        inspect, "getsource", lambda obj: "  def __call__(self): pass\n"
    )
    assert sjo._build_sharded_callable_cls() is sjo.pallas_impl.JaxCallable

    synthetic_src = (
        "  def __call__(self, lowered):\n"
        f"{sjo._STOCK_PLACEHOLDER_LINE}\n"
        "      return output_shapes\n"
    )
    monkeypatch.setattr(inspect, "getsource", lambda obj: synthetic_src)
    cls_ok = sjo._build_sharded_callable_cls()
    assert issubclass(cls_ok, sjo.pallas_impl.JaxCallable)
    assert cls_ok is not sjo.pallas_impl.JaxCallable
    called_out = cls_ok.__call__(SimpleNamespace(mesh=mesh), lowered_with_sharding)
    assert len(called_out) == 1 and called_out[0].shape == (2, 8)
    monkeypatch.setattr(inspect, "getsource", orig_getsource)

    # 4. sharded_jax_op: lazy init, invalid name, registration & fake_fn.
    monkeypatch.setattr(sjo, "_SHARDED_CALLABLE_CLS", None)
    devices = [
        pallas._PallasDevice(id=i, process_index=i, device_kind="TPU v7")
        for i in range(2)
    ]
    import numpy as np

    real_mesh = jax.sharding.Mesh(np.array(devices), ("tp",))

    def sample_kernel(x: jax.Array) -> jax.Array:
        return x

    with pytest.raises(ValueError, match="must be 'namespace::op'"):
        sjo.sharded_jax_op(
            "invalid_op_name",
            sample_kernel,
            mesh=real_mesh,
            input_partition_specs=(P("tp"),),
            output_partition_specs=P("tp"),
        )
    assert sjo._SHARDED_CALLABLE_CLS is not None

    registered_fakes = []

    class _FakeCustomOp:
        def __init__(self, name, fn, mutates_args):
            self.name = name
            self.fn = fn
            self.mutates_args = mutates_args

        def register_fake(self, fake_fn):
            registered_fakes.append(fake_fn)

    monkeypatch.setattr(
        torch.library,
        "custom_op",
        lambda name, fn, mutates_args=(): _FakeCustomOp(name, fn, mutates_args),
    )
    op = sjo.sharded_jax_op(
        "test_td752::sample_kernel",
        sample_kernel,
        mesh=real_mesh,
        input_partition_specs=(P("tp"),),
        output_partition_specs=P("tp"),
    )
    assert isinstance(op, _FakeCustomOp)
    assert len(registered_fakes) == 1

    fake_lowered = SimpleNamespace(
        out_avals=[aval_f32],
        _out_named_shardings=[SimpleNamespace(spec=P("tp"))],
        out_tree=SimpleNamespace(unflatten=lambda items: items[0]),
    )
    monkeypatch.setattr(
        sjo.pallas_impl,
        "jax_placeholders",
        lambda args, mesh=None, partition_specs=None: args,
    )
    monkeypatch.setattr(op.fn, "exported", lambda *a, **kw: fake_lowered)
    fake_out = registered_fakes[0](torch.zeros(2, 8))
    assert fake_out.shape == (2, 8)
