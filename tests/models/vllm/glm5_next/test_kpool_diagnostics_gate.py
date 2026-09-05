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
"""The switch that decides whether ``kpool_write``'s counters are read.

The counters themselves are computed on device unconditionally; this is only
the decision to pay a host round trip and look. Two properties matter and
neither is obvious from the call site:

* it is **fixed for the life of the process**, so the traced forward emits the
  same ops on every step. A gate that flips part-way through a run -- an
  earlier version checked the first eight forwards of each layer -- re-keys
  the compiled/fused decode program across all eleven DSA layers for nothing,
  and spends its whole budget inside warmup, where the shapes are dummies;
* a value that is neither ``0`` nor ``1`` **raises**. This switch is set by
  hand, in a bring-up run, by somebody who already suspects a bug; a typo that
  quietly means "off" costs that whole run.
"""

import pytest

from vllm_torchtpu.models.vllm.glm5_next.indexer import (
    KPOOL_CHECK_DIAGNOSTICS_ENV, kpool_check_diagnostics_enabled)

ENV = KPOOL_CHECK_DIAGNOSTICS_ENV


def test_the_env_var_has_the_documented_name():
    """It appears in probe scripts and in the docs, so it is API."""
    assert ENV == "VLLM_TORCHTPU_GLM5_KPOOL_CHECK_DIAGNOSTICS"


def test_off_when_unset():
    assert kpool_check_diagnostics_enabled({}) is False


def test_off_when_empty():
    """An exported-but-empty variable is how a shell spells "not set"."""
    assert kpool_check_diagnostics_enabled({ENV: ""}) is False


@pytest.mark.parametrize("value,expected", [("1", True), ("0", False)])
def test_the_two_accepted_values(value, expected):
    assert kpool_check_diagnostics_enabled({ENV: value}) is expected


@pytest.mark.parametrize("value", ["true", "True", "yes", "on", "8", "01"])
def test_anything_else_raises_rather_than_meaning_off(value):
    with pytest.raises(ValueError, match=ENV):
        kpool_check_diagnostics_enabled({ENV: value})


def test_it_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert kpool_check_diagnostics_enabled() is True
    monkeypatch.delenv(ENV)
    assert kpool_check_diagnostics_enabled() is False
