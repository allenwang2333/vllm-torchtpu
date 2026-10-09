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
"""TD7.88: shared mHC CompilerParams and closed-form hc_mult inversion."""

import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import o_projection
from vllm_torchtpu.kernels.deepseek_v4.mhc import utils


@pytest.mark.cpu_test
def test_compiler_params_is_cached_and_matches_inline_construction():
    params = utils.compiler_params(utils.DEFAULT_VMEM_LIMIT_BYTES)
    assert params is utils.compiler_params(utils.DEFAULT_VMEM_LIMIT_BYTES)
    assert params == pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=utils.DEFAULT_VMEM_LIMIT_BYTES,
        disable_bounds_checks=True,
    )
    other = utils.compiler_params(utils.DEFAULT_VMEM_LIMIT_BYTES // 2)
    assert other.vmem_limit_bytes == utils.DEFAULT_VMEM_LIMIT_BYTES // 2
    assert other is not params


@pytest.mark.cpu_test
def test_o_projection_compiler_params_match_inline_construction():
    assert (
        pltpu.CompilerParams(
            vmem_limit_bytes=o_projection.DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        )
        == o_projection._COMPILER_PARAMS
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("hc_mult", [1, 2, 4, 8, 63, 64, 1000])
def test_hc_mult_from_mix_dim_inverts(hc_mult):
    assert utils.hc_mult_from_mix_dim(hc_mult * (hc_mult + 2)) == hc_mult


@pytest.mark.cpu_test
def test_hc_mult_from_mix_dim_matches_linear_search():
    # The replaced search: next(m for m in range(1, 64) if m*(m+2) == n).
    for n in range(0, 64 * 66):
        expected = next((m for m in range(1, 64) if m * (m + 2) == n), None)
        if expected is None:
            if n < 63 * 65:
                with pytest.raises(ValueError, match="hc_mult3"):
                    utils.hc_mult_from_mix_dim(n)
        else:
            assert utils.hc_mult_from_mix_dim(n) == expected
