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
"""The dense-MLA substitution has a context bound, and it must be enforced.

Running GLM-5.3-Flash's full-attention layers densely is exact only while a
sparse pooled indexer would have selected every causal token, i.e. while
``max_model_len <= index_topk``. Past that the reference attends sparsely and
a dense build does not, and the output is wrong while still looking plausible
-- the worst failure shape there is. So it raises rather than warns.
"""

import types

import pytest

from vllm_torchtpu.models.vllm.glm5_next.model import (
    _ALLOW_UNSOUND_CONTEXT, _check_context_within_dense_mla_bound)


def _cfg(max_model_len):
    return types.SimpleNamespace(model_config=types.SimpleNamespace(
        max_model_len=max_model_len))


def _text(index_topk=2048):
    return types.SimpleNamespace(index_topk=index_topk)


@pytest.mark.parametrize("max_len", [128, 2047, 2048])
def test_within_bound_is_allowed(max_len):
    _check_context_within_dense_mla_bound(_cfg(max_len), _text())


@pytest.mark.parametrize("max_len", [2049, 4096, 1_048_576])
def test_beyond_bound_raises(max_len):
    with pytest.raises(ValueError, match="exceeds index_topk"):
        _check_context_within_dense_mla_bound(_cfg(max_len), _text())


def test_the_escape_hatch_downgrades_to_a_warning(monkeypatch):
    """Deliberately available: measuring throughput past the bound is valid.

    Judging output past it is not, which is why it is opt-in and loud.
    """
    monkeypatch.setenv(_ALLOW_UNSOUND_CONTEXT, "1")
    _check_context_within_dense_mla_bound(_cfg(1_048_576), _text())


def test_bound_tracks_the_config_not_a_hardcoded_2048():
    """The tiny fixture and the real checkpoint must both be governed."""
    with pytest.raises(ValueError, match="index_topk=256"):
        _check_context_within_dense_mla_bound(_cfg(512), _text(index_topk=256))
    _check_context_within_dense_mla_bound(_cfg(512), _text(index_topk=4096))


def test_missing_fields_do_not_raise():
    """A config without index_topk is not this check's business."""
    _check_context_within_dense_mla_bound(_cfg(4096), types.SimpleNamespace())
    _check_context_within_dense_mla_bound(
        types.SimpleNamespace(model_config=types.SimpleNamespace()), _text())
