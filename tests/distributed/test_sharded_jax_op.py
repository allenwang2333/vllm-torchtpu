# SPDX-License-Identifier: Apache-2.0
"""The bridge passes local output specifications to current TorchTPU."""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.distributed import sharded_jax_op as bridge

pytestmark = pytest.mark.cpu_test


def test_native_output_specs_do_not_need_source_rewriting(monkeypatch):
    if (
        "output_partition_specs"
        not in inspect.signature(bridge.pallas_impl.JaxCallable).parameters
    ):
        pytest.skip("TorchTPU predates native output partition specs")
    monkeypatch.setattr(
        inspect, "getsource", Mock(side_effect=AssertionError("no patch"))
    )
    assert bridge._build_sharded_callable_cls() is bridge.pallas_impl.JaxCallable


@pytest.mark.parametrize("specs", [P("rank", None, None), (P(), P("rank", None))])
def test_constructor_receives_explicit_output_leaves(monkeypatch, specs):
    captured = {}

    class NativeCallable:
        def __init__(self, *, output_partition_specs=None, **kwargs):
            captured["specs"] = output_partition_specs

    monkeypatch.setattr(bridge.pallas_impl, "JaxCallable", NativeCallable)
    monkeypatch.setattr(bridge, "_SHARDED_CALLABLE_CLS", None)
    monkeypatch.setattr(bridge, "NamedSharding", lambda mesh, spec: (mesh, spec))
    monkeypatch.setattr(bridge.jax, "jit", lambda fn, **kwargs: fn)
    monkeypatch.setattr(
        bridge.pallas_impl, "_get_kernel_invocation_key", lambda *args: "key"
    )
    monkeypatch.setattr(
        bridge.torch.library, "custom_op", lambda *args, **kwargs: Mock()
    )

    def fn(x: jax.Array) -> jax.Array:
        return x

    bridge.sharded_jax_op(
        "test::explicit_specs",
        fn,
        mesh=object(),
        input_partition_specs=(P("rank", None, None),),
        output_partition_specs=specs,
    )
    expected = (specs,) if isinstance(specs, P) else specs
    assert captured["specs"] == expected


def test_local_output_placeholder_uses_rank_slot(monkeypatch):
    mesh = SimpleNamespace(shape={"rank": 32})
    monkeypatch.setattr(
        bridge.pallas_impl, "get_local_shape", lambda shape, mesh, spec: (1, *shape[1:])
    )
    create = Mock(return_value="placeholder")
    monkeypatch.setattr(bridge.torch, "empty", create)
    assert (
        bridge._placeholder(
            jax.core.ShapedArray((32, 4, 7168), jnp.bfloat16),
            P("rank", None, None),
            mesh,
        )
        == "placeholder"
    )
    assert create.call_args.args[0] == (1, 4, 7168)
