# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Unit tests for the PCP streaming RPA kernel's vLLM adapter.

These moved out of `tests/layers/adapter/test_attention_pcp.py` when
`layers/adapter/attention.py` stopped reaching for the streaming kernel: they
only ever exercised `vllm_adapter` itself, which still ships. They are pure
mock tests -- no TPU, no real Pallas lowering.
"""

import inspect

import pytest
import torch

import vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter as pcp_adapter


def test_adapter_make_streaming_kernel_has_eight_tensor_signature_and_calls_wrapper(
        monkeypatch):
    captured = {}
    new_cache = object()
    output = object()

    def fake_pcp_wrapper(**kwargs):
        captured.update(kwargs)
        return output, new_cache

    monkeypatch.setattr(pcp_adapter, "sharded_pcp_ragged_paged_attention",
                        fake_pcp_wrapper)

    mesh = object()
    kernel = pcp_adapter.make_pcp_streaming_rpa_kernel(
        q_scale=None,
        k_scale=0.125,
        v_scale=0.25,
        mesh=mesh,
        sliding_window=None,
        sm_scale=1.0,
        soft_cap=None,
        skip_kv_update=False,
        cp_kv_cache_interleave_size=4,
    )

    params = list(inspect.signature(kernel).parameters.values())
    assert len(params) == 8
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
               for p in params)

    args = tuple(object() for _ in range(8))
    assert kernel(*args) == (new_cache, output)
    assert captured["mesh"] is mesh
    assert captured["q"] is args[1]
    assert captured["k"] is args[2]
    assert captured["v"] is args[3]
    assert captured["kv_cache"] is args[0]
    assert captured["kv_lens"] is args[4]
    assert captured["page_indices"] is args[5]
    assert captured["cu_q_lens"] is args[6]
    assert captured["distribution"] is args[7]
    assert captured["attention_sink"] is None
    assert captured["sm_scale"] == 1.0
    assert captured["attention_chunk_size"] is None
    assert captured["q_scale"] is None
    assert captured["k_scale"] == 0.125
    assert captured["v_scale"] == 0.25
    assert captured["update_kv_cache"] is True
    assert captured["cp_kv_cache_interleave_size"] == 4


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({
            "soft_cap": 30.0,
            "skip_kv_update": False
        }, "logits soft cap"),
        ({
            "soft_cap": None,
            "skip_kv_update": True
        }, "skip_kv_update"),
    ],
)
def test_adapter_make_streaming_kernel_rejects_unsupported_features(
        kwargs, expected_message):
    with pytest.raises(NotImplementedError, match=expected_message):
        pcp_adapter.make_pcp_streaming_rpa_kernel(
            q_scale=None,
            k_scale=None,
            v_scale=None,
            mesh=object(),
            sliding_window=None,
            sm_scale=1.0,
            cp_kv_cache_interleave_size=4,
            **kwargs,
        )


def test_adapter_scoped_callable_uses_lowered_out_shardings(monkeypatch):
    captured = {}
    original_call = pcp_adapter.pallas_impl.JaxCallable.__call__

    class FakeOutTree:

        def unflatten(self, values):
            return tuple(values)

    class FakeLowered:
        out_avals = ("aval0", "aval1")
        _out_named_shardings = ("sharding0", "sharding1")
        out_tree = FakeOutTree()
        mlir_module_serialized = b"mlir"

        def mlir_module(self):
            return "module {}"

    class FakePcpCallable(pcp_adapter._PcpStreamingJaxCallable):

        def __init__(self):
            self.trace_key = "trace"
            self.static_argnums = ()
            self.output_shapes = {}
            self.kernel_key_to_mlir_fingerprint = {}
            self.mesh = "mesh"
            self.input_partition_specs = ("input_spec", )
            self.name = "op"
            self.exported = lambda *args, **kwargs: FakeLowered()
            self.donate_argnums = ()
            self.input_output_aliases = {}

        def _validate_args(self, *args):
            captured["validated_args"] = args

    class FakePallasRuntime:

        def lookup_custom_kernel(self, name, key):
            captured["lookup"] = (name, key)
            return False

        def register_custom_kernel(self, name, key, *, serialized_mlir_module):
            captured["registered"] = (name, key, serialized_mlir_module)

        def call_custom_kernel(self, name, key, *, inputs, output_shapes,
                               donate_argnums):
            captured["call"] = {
                "name": name,
                "key": key,
                "inputs": inputs,
                "output_shapes": output_shapes,
                "donate_argnums": donate_argnums,
            }
            return ("result0", "result1")

    monkeypatch.setattr(pcp_adapter.pallas_impl, "tpu_torch_pallas",
                        FakePallasRuntime())
    monkeypatch.setattr(pcp_adapter.pallas_impl, "_get_kernel_invocation_key",
                        lambda *_args, **_kwargs: "kernel_key")
    monkeypatch.setattr(pcp_adapter.pallas_impl, "jax_placeholders",
                        lambda *_args, **_kwargs: ("jax_arg", ))
    monkeypatch.setattr(
        pcp_adapter, "_torch_placeholder_with_sharding",
        lambda aval, sharding, mesh: f"{aval}:{sharding}:{mesh}")

    result = FakePcpCallable()(torch.tensor(1.0))

    assert result == ("result0", "result1")
    assert captured["call"]["output_shapes"] == [
        "aval0:sharding0:mesh", "aval1:sharding1:mesh"
    ]
    assert pcp_adapter.pallas_impl.JaxCallable.__call__ is original_call


def test_adapter_pcp_jax_op_uses_scoped_callable(monkeypatch):
    captured = {}
    original_call = pcp_adapter.pallas_impl.JaxCallable.__call__

    class FakePcpCallable:

        def __init__(self, **kwargs):
            captured["callable_kwargs"] = kwargs

    class FakeCustomOp:

        def register_fake(self, fake_impl):
            captured["fake_impl"] = fake_impl

    def fake_custom_op(name, wrapped_fn, *, mutates_args):
        captured["custom_op"] = (name, wrapped_fn, mutates_args)
        return FakeCustomOp()

    monkeypatch.setattr(pcp_adapter, "_PcpStreamingJaxCallable",
                        FakePcpCallable)
    monkeypatch.setattr(pcp_adapter.pallas_impl, "_verify_signature",
                        lambda _signature: None)
    monkeypatch.setattr(pcp_adapter.pallas_impl, "_infer_static_argnums",
                        lambda _signature: ())
    monkeypatch.setattr(pcp_adapter.pallas_impl, "_get_kernel_invocation_key",
                        lambda *_args, **_kwargs: "trace_key")
    monkeypatch.setattr(pcp_adapter, "_named_shardings",
                        lambda *_args, **_kwargs: ("out0", "out1"))
    monkeypatch.setattr(pcp_adapter.jax, "jit", lambda fn, **kwargs:
                        ("jit_fn", fn, kwargs))
    monkeypatch.setattr(pcp_adapter.torch.library, "custom_op", fake_custom_op)

    def fn(tensor):
        return tensor

    result = pcp_adapter.pcp_streaming_jax_op(
        "pcp::op",
        fn,
        donate_argnums=(0, ),
        mesh=object(),
        input_partition_specs=("input_spec", ),
    )

    assert isinstance(result, FakeCustomOp)
    assert isinstance(captured["custom_op"][1], FakePcpCallable)
    assert captured["callable_kwargs"]["name"] == "pcp::op"
    assert captured["callable_kwargs"]["jit_fn"][0] == "jit_fn"
    assert captured["callable_kwargs"]["donate_argnums"] == (0, )
    assert captured["custom_op"][2] == ()
    assert pcp_adapter.pallas_impl.JaxCallable.__call__ is original_call


def test_adapter_invoke_streaming_op_trims_generic_tensor_args():
    captured = {}
    kv_cache = object()
    generic_args = (*tuple(object() for _ in range(7)), None)
    expected = object()

    def fake_op(*args):
        captured["op_args"] = args
        return expected

    actual = pcp_adapter.invoke_pcp_streaming_op(fake_op, kv_cache,
                                                 generic_args, {})

    assert actual is expected
    assert captured["op_args"] == (kv_cache, *generic_args[:7])


def test_adapter_invoke_streaming_op_rejects_unsupported_sink():
    args = tuple(object() for _ in range(8))

    with pytest.raises(NotImplementedError, match="attention sinks"):
        pcp_adapter.invoke_pcp_streaming_op(lambda *_args: None, object(),
                                            args, {})


def test_adapter_invoke_streaming_op_rejects_kwargs_and_wrong_arg_count():
    with pytest.raises(ValueError, match="does not accept keyword"):
        pcp_adapter.invoke_pcp_streaming_op(lambda *_args: None, object(), (),
                                            {"sink": object()})

    with pytest.raises(ValueError, match="expects 8 tensor args"):
        pcp_adapter.invoke_pcp_streaming_op(lambda *_args: None, object(), (),
                                            {})


