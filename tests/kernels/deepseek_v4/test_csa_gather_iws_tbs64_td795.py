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
"""TD7.95: CSA gather / index word scores params and hd64 tuned-size lookup."""

from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather
from vllm_torchtpu.kernels.deepseek_v41 import index_word_scores
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import (
    tuned_block_sizes_hd64 as tbs64,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3.util import next_power_of_2


@pytest.mark.cpu_test
def test_module_compiler_params_match_inline_construction():
    assert (
        pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        )
        == csa_gather._COMPILER_PARAMS
    )
    assert (
        pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
        == index_word_scores._COMPILER_PARAMS
    )


def _run_csa_gather(indices, num_valid_indices):
    captured = {}
    info = SimpleNamespace(
        sparse_core=SimpleNamespace(num_lanes=8, num_cores=2, num_subcores=16)
    )

    def fake_kernel(fn, *, out_type, compiler_params, **kwargs):
        captured["compiler_params"] = compiler_params

        def run(nope, rope, idx, valid):
            captured["valid"] = np.asarray(valid)
            n = out_type[0].shape[0]
            return (
                jnp.zeros((n, 128), jnp.int32),
                jnp.zeros((n // 4, 128), jnp.int32),
            )

        return run

    with (
        patch.object(csa_gather.pltpu, "get_tpu_info", return_value=info),
        patch.object(
            csa_gather.plsc, "VectorSubcoreMesh", lambda **kw: SimpleNamespace(**kw)
        ),
        patch.object(csa_gather.pl, "kernel", fake_kernel),
    ):
        csa_gather.csa_gather.__wrapped__(
            jnp.zeros((8, 16, 128), jnp.int32),
            jnp.zeros((8, 4, 128), jnp.int32),
            indices,
            num_valid_indices,
            top_k=512,
        )
    return captured


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("kind", "expected"),
    [("none", 1024), ("int", 512), ("scalar", 512), ("vector", 512)],
)
def test_csa_gather_valid_indices_vector(kind, expected):
    # Built here, not in parametrize: a JAX array at collection time
    # initializes the backend, which the test harness rejects.
    num_valid_indices = {
        "none": lambda: None,
        "int": lambda: 512,
        "scalar": lambda: jnp.int32(512),
        "vector": lambda: jnp.array([512], jnp.int32),
    }[kind]()
    captured = _run_csa_gather(jnp.zeros((1024,), jnp.int32), num_valid_indices)
    assert captured["compiler_params"] is csa_gather._COMPILER_PARAMS
    assert captured["valid"].dtype == np.int32
    np.testing.assert_array_equal(captured["valid"], np.full((8,), expected))


@pytest.mark.cpu_test
@pytest.mark.parametrize("page_size", [16, 100, 128, 256])
@pytest.mark.parametrize("max_model_len", [128, 3000, 4096, 100_000])
def test_lookup_keys_power_of_two_without_rerounding(
    monkeypatch, page_size, max_model_len
):
    monkeypatch.setattr(tbs64, "get_device_name", lambda: "TPU7x")
    keys = tbs64.get_lookup_keys(
        page_size, jnp.bfloat16, jnp.bfloat16, 16, 2, 64, max_model_len
    )
    assert keys[1] == next_power_of_2(page_size)
    assert keys[4] == next_power_of_2(max_model_len)


@pytest.mark.cpu_test
def test_missing_tuned_entry_logs_warning_with_keys(monkeypatch):
    monkeypatch.setattr(tbs64, "get_tpu_version", lambda: 6)
    monkeypatch.setattr(tbs64, "get_device_name", lambda: "TPU unknown")
    calls = []
    monkeypatch.setattr(
        tbs64.logger, "warning", lambda msg, *args: calls.append((msg, args))
    )
    assert tbs64.get_tuned_block_sizes(
        jnp.bfloat16, jnp.bfloat16, 8, 2, 64, 128, 4096, 32
    ) == (2048 // 128, 32)
    assert len(calls) == 1
    msg, args = calls[0]
    assert "%s" in msg
    assert args[0][0] == "TPU unknown"
