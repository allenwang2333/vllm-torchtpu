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
"""Unit tests for the cache execution extension (`pool.config`)."""

from __future__ import annotations

import pytest

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import pool


def test_cache_exec_pool_gather_default_is_split():
    assert pool.POOL_IO_NUMERICS == {
        "fused_pool_io": "approx",
        "pool_gather": "exact",
        "pool_gather_eligible": "approx",
    }
    assert mk.ExecSpec().cache is None
    assert mk.cache_exec(mk.ExecSpec()).pool_gather is False
    assert not mk.CacheExec(paged_kv=True).GPR
    for val in pool.POOL_GATHER_VALUES:
        assert mk.CacheExec(pool_gather=val).pool_gather == val
    with pytest.raises(ValueError, match="pool_gather must be one of"):
        mk.CacheExec(pool_gather="sometimes")  # type: ignore[arg-type]
    ex = mk.with_cache(mk.ExecSpec(B=24, T=4), paged_kv=True, pool_gather=True)
    assert mk.cache_exec(ex).GPR and not mk.cache_exec(ex).FUSED
