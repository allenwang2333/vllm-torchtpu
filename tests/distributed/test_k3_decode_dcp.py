# SPDX-License-Identifier: Apache-2.0
"""Admission tests independent of vLLM/TPU imports."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

_PATH = (Path(__file__).resolve().parents[2] /
         "src/vllm_torchtpu/distributed" / "k3_decode_dcp.py")
_SPEC = importlib.util.spec_from_file_location("k3_decode_groups", _PATH)
groups = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(groups)


def native_config():
    return NS(
        parallel_config=NS(tensor_parallel_size=8,
                           decode_context_parallel_size=2,
                           pipeline_parallel_size=1,
                           prefill_context_parallel_size=1,
                           world_size=8),
        model_config=NS(architectures=["KimiK3ForConditionalGeneration"]),
        speculative_config=None,
        kv_transfer_config=NS(kv_connector="DecodeBenchConnector"),
        cache_config=NS(cache_dtype="auto", block_size=16))


@pytest.mark.parametrize("dp", [1, 4])
def test_native_tp8_keeps_topology(dp):
    c = native_config()
    c.parallel_config.data_parallel_size = dp
    before = vars(c.parallel_config).copy()
    groups.validate_native_config(c)
    assert vars(c.parallel_config) == before
    assert groups.native_size(c) == 2
    c.model_config.architectures = ["KimiLinearForCausalLM"]
    groups.validate_native_config(c)
    assert groups.native_size(c) == 2


def test_native_interleave_resolved_before_attention_build():
    c = native_config()
    c.parallel_config.cp_kv_cache_interleave_size = 1
    c.parallel_config.dcp_kv_cache_interleave_size = 4
    groups.configure_native_interleave(c)
    assert c.parallel_config.cp_kv_cache_interleave_size == 16
    assert c.parallel_config.dcp_kv_cache_interleave_size == 16
    c.cache_config.block_size = 32
    groups.configure_native_interleave(c)
    assert c.parallel_config.cp_kv_cache_interleave_size == 32
    assert c.parallel_config.dcp_kv_cache_interleave_size == 32


@pytest.mark.parametrize("connector", [None, "TPURaidenConnector"])
def test_native_rejects_unimplemented_prefill_and_pd(connector):
    c = native_config()
    c.kv_transfer_config = NS(kv_connector=connector) if connector else None
    with pytest.raises(ValueError, match="live PD KV placement"):
        groups.validate_native_config(c)


def test_native_rejects_extend_but_allows_idle_steps():
    c = native_config()
    groups.validate_decode_step(c, NS(num_scheduled_tokens={}))
    groups.validate_decode_step(c, NS(num_scheduled_tokens={"a": 1}))
    with pytest.raises(ValueError, match="prefill/extend"):
        groups.validate_decode_step(c, NS(num_scheduled_tokens={"a": 2}))


@pytest.mark.parametrize("field,value", [
    ("decode_context_parallel_size", 4),
    ("tensor_parallel_size", 7),
    ("pipeline_parallel_size", 2),
    ("prefill_context_parallel_size", 2),
])
def test_native_rejects_unsupported_topology(field, value):
    c = native_config()
    setattr(c.parallel_config, field, value)
    with pytest.raises(ValueError, match="Native K3"):
        groups.validate_native_config(c)


def test_native_rejects_speculation_and_quantized_cache():
    c = native_config()
    c.speculative_config = NS()
    with pytest.raises(ValueError, match="speculative"):
        groups.validate_native_config(c)
    c.speculative_config = None
    c.cache_config.cache_dtype = "fp8"
    with pytest.raises(ValueError, match="BF16 KV"):
        groups.validate_native_config(c)


def test_non_k3_native_dcp_unchanged():
    c = native_config()
    c.model_config.architectures = ["DeepseekV3ForCausalLM"]
    c.kv_transfer_config = None
    groups.validate_native_config(c)
    assert groups.native_size(c) == 1
