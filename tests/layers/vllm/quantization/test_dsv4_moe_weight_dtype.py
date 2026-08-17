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
"""
Tests for DeepSeek-V4 expert weight dtype resolution.

The expert path requantizes to fp4 or int8. fp4 weights are paired with bf16
activations by design, so on a pre-v7 TPU they leave the quantized MXU unused;
int8 weights keep the activations quantized and stay on the integer MXU. The
resolver encodes that, and these tests pin the defaults and the override.
"""

import jax.numpy as jnp
import pytest

from vllm_torchtpu.layers.vllm.quantization import mxfp4


@pytest.fixture(autouse=True)
def _clear_resolver_cache():
    """The resolver is process-cached; each test needs a clean read."""
    mxfp4._dsv4_moe_weight_dtype.cache_clear()
    yield
    mxfp4._dsv4_moe_weight_dtype.cache_clear()


@pytest.mark.parametrize("tpu_version,expected", [
    (5, jnp.int8),
    (6, jnp.int8),
    (7, jnp.float4_e2m1fn),
    (-1, jnp.float4_e2m1fn),
])
def test_default_follows_tpu_generation(monkeypatch, tpu_version, expected):
    monkeypatch.delenv("MOE_REQUANTIZE_WEIGHT_DTYPE", raising=False)
    monkeypatch.setattr(mxfp4, "get_tpu_version", lambda: tpu_version)
    assert mxfp4._dsv4_moe_weight_dtype() == expected


@pytest.mark.parametrize("value,expected", [
    ("int8", jnp.int8),
    ("fp4", jnp.float4_e2m1fn),
    ("float4_e2m1fn", jnp.float4_e2m1fn),
    ("INT8", jnp.int8),
])
def test_env_overrides_the_default(monkeypatch, value, expected):
    monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", value)
    # Pick the generation whose default is the *other* dtype, so a passing
    # assertion can only come from the override.
    monkeypatch.setattr(mxfp4, "get_tpu_version",
                        lambda: 7 if expected == jnp.int8 else 6)
    assert mxfp4._dsv4_moe_weight_dtype() == expected


def test_unsupported_env_value_falls_back_without_raising(monkeypatch):
    """`MOE_REQUANTIZE_WEIGHT_DTYPE` is shared with the fp8 and dense paths,
    where `float8_e4m3fn` is the documented value. This path cannot produce an
    fp8 rhs, but it must not break a run that sets the variable globally for
    its siblings -- so it warns and keeps the generation default."""
    monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "float8_e4m3fn")
    monkeypatch.setattr(mxfp4, "get_tpu_version", lambda: 6)
    assert mxfp4._dsv4_moe_weight_dtype() == jnp.int8


def test_unset_env_does_not_inherit_the_envs_module_default(monkeypatch):
    """`vllm_torchtpu.envs.MOE_REQUANTIZE_WEIGHT_DTYPE` defaults to
    `float8_e4m3fn`. Reading it through `envs` rather than the environment
    would make every v7 run silently stop producing fp4."""
    monkeypatch.delenv("MOE_REQUANTIZE_WEIGHT_DTYPE", raising=False)
    monkeypatch.setattr(mxfp4, "get_tpu_version", lambda: 7)
    assert mxfp4._dsv4_moe_weight_dtype() == jnp.float4_e2m1fn
