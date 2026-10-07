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
"""TPU_RPA_DECODE_ONLY: small buckets run batched RPA's DECODE pass only.

It reuses TPU_GDN_DECODE_ONLY_MAX_TOKENS and the runner's prefill floor. These
tests pin the predicate, the kernel kwargs, and that the floor also moves a
chunk holding an RPA mixed-segment request when RPA's decode segment is
narrower than the windowed one (reorder_batch_threshold 1).
"""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.adapter import attention
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("TPU_GDN_DECODE_ONLY_MAX_TOKENS", "64")
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY", "1")
    yield 64


def _floor(start_index, num_reqs, num_windowed, num_rpa_decode=None):
    return TPUModelRunner._gdn_mixed_min_tokens(
        SimpleNamespace(), start_index, num_reqs, num_windowed, num_rpa_decode
    )


def test_off_by_default(monkeypatch):
    monkeypatch.setenv("TPU_GDN_DECODE_ONLY_MAX_TOKENS", "64")
    monkeypatch.delenv("TPU_RPA_DECODE_ONLY", raising=False)
    assert not attention.rpa_decode_only_program(8)
    # The RPA decode count is ignored without the flag.
    assert _floor(0, 3, 3, 1) == 0


def test_needs_the_gdn_limit(monkeypatch):
    monkeypatch.delenv("TPU_GDN_DECODE_ONLY_MAX_TOKENS", raising=False)
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY", "1")
    assert not attention.rpa_decode_only_program(8)


def test_predicate(on):
    assert attention.rpa_decode_only_program(4)
    assert attention.rpa_decode_only_program(64)
    assert not attention.rpa_decode_only_program(128)


def test_floor_uses_the_rpa_decode_segment(on):
    # Windowed (decode + verify) covers all 3 requests, but RPA's decode
    # segment holds 1 (reorder_batch_threshold 1): the chunk must move.
    assert _floor(0, 3, 3, 1) == on + 1
    # Same segments: no floor.
    assert _floor(0, 3, 3, 3) == 0
    # A prefill request still moves the chunk.
    assert _floor(0, 3, 2, 2) == on + 1


def test_kernel_kwargs(on, monkeypatch):
    monkeypatch.setenv("TPU_RPA_SCHEDULE_FAST", "1")
    seen = {}

    def fake_impl(*args, rpa_func, **kwargs):
        seen["kw"] = dict(getattr(rpa_func, "keywords", {}))
        return None

    monkeypatch.setattr(attention, "_pallas_rpa_kernel_impl", fake_impl)
    q = SimpleNamespace(shape=(48, 4, 256))
    common = dict(
        mesh=None, sliding_window=None, skip_kv_update=False, kv_layout=None
    )
    attention._pallas_rpa_kernel_batched(
        None, q, None, None, None, None, None, None, None, None, None, None, **common
    )
    assert seen["kw"] == {"decode_only": True, "schedule_fast": True}
    q = SimpleNamespace(shape=(128, 4, 256))
    attention._pallas_rpa_kernel_batched(
        None, q, None, None, None, None, None, None, None, None, None, None, **common
    )
    assert seen["kw"] == {"schedule_fast": True}
