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

"""100% full-module line and branch coverage tests for streamindex_topk/config.py."""

from __future__ import annotations

import importlib
import os

import pytest

os.environ.setdefault("PJRT_DEVICE", "CPU")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config


def test_module_reload_for_full_coverage():
    """Reload module under coverage tracer so module-level definitions are traced."""
    importlib.reload(config)
    assert config.DEFAULT_BUFFER_COUNT == (4, 3, 3)


def test_mla_case_values_and_symbols():
    """Verify MlaCase enum values and pre-bound symbol attributes."""
    assert config.MlaCase.DECODE.value == 0
    assert config.MlaCase.PREFILL.value == 1
    assert config.MlaCase.MIXED.value == 2

    assert config.MlaCase.DECODE.symbol == "d"
    assert config.MlaCase.PREFILL.symbol == "p"
    assert config.MlaCase.MIXED.symbol == "m"
    assert config._MLA_CASE_SYMBOLS == {
        config.MlaCase.DECODE: "d",
        config.MlaCase.PREFILL: "p",
        config.MlaCase.MIXED: "m",
    }


def test_kv_layout_parse_fast_and_slow_paths():
    """Verify KVLayout.parse across None, enum, fast map, fallback, and errors."""
    # None defaults to HEAD_ALONG_SUBLANE (fast path)
    assert config.KVLayout.parse(None) is config.KVLayout.HEAD_ALONG_SUBLANE

    # Enum instances pass through (fast path)
    assert (
        config.KVLayout.parse(config.KVLayout.HEAD_ALONG_SUBLANE)
        is config.KVLayout.HEAD_ALONG_SUBLANE
    )
    assert (
        config.KVLayout.parse(config.KVLayout.SEQ_ALONG_LANE)
        is config.KVLayout.SEQ_ALONG_LANE
    )

    # Exact lowercase and uppercase aliases (fast path)
    assert (
        config.KVLayout.parse("head_along_sublane")
        is config.KVLayout.HEAD_ALONG_SUBLANE
    )
    assert (
        config.KVLayout.parse("HEAD_ALONG_SUBLANE")
        is config.KVLayout.HEAD_ALONG_SUBLANE
    )
    assert config.KVLayout.parse("seq_along_lane") is config.KVLayout.SEQ_ALONG_LANE
    assert config.KVLayout.parse("SEQ_ALONG_LANE") is config.KVLayout.SEQ_ALONG_LANE
    assert config.KVLayout.parse("seq_alone_lane") is config.KVLayout.SEQ_ALONG_LANE
    assert config.KVLayout.parse("SEQ_ALONE_LANE") is config.KVLayout.SEQ_ALONG_LANE

    # Whitespace-padded and mixed-case strings hit the strip().lower() fallback path
    assert (
        config.KVLayout.parse("  Head_Along_Sublane  ")
        is config.KVLayout.HEAD_ALONG_SUBLANE
    )
    assert config.KVLayout.parse("\tSeq_Along_Lane\n") is config.KVLayout.SEQ_ALONG_LANE
    assert config.KVLayout.parse("  Seq_Alone_Lane ") is config.KVLayout.SEQ_ALONG_LANE

    # Invalid string raises ValueError
    with pytest.raises(ValueError, match="Unknown kv_layout 'invalid_layout'"):
        config.KVLayout.parse("invalid_layout")


def test_scheduling_group_ids_reserve_and_reset():
    """Verify reserve_scheduling_group_ids and reset_scheduling_group_ids."""
    config.reset_scheduling_group_ids()

    with pytest.raises(ValueError, match="must be positive"):
        config.reserve_scheduling_group_ids(0)

    with pytest.raises(ValueError, match="must be positive"):
        config.reserve_scheduling_group_ids(-3)

    assert config.reserve_scheduling_group_ids(1) == 1
    assert config.reserve_scheduling_group_ids(4) == 2
    assert config.reserve_scheduling_group_ids(2) == 6

    config.reset_scheduling_group_ids()
    assert config.reserve_scheduling_group_ids(1) == 1
