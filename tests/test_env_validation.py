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
"""Tests for reporting set environment variables that nothing reads."""

import os
from unittest import mock

import pytest

from vllm_torchtpu.env_validation import UnreadEnvVar, find_unread_env_vars
from vllm_torchtpu.platforms import tpu_platform
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

TORCHTPU = {
    "TPU_PARALLEL_PRECOMPILE",
    "MOE_FUSED_EP_KERNEL_IMPL",
    "VLLM_TPU_MOST_MODEL_LEN",
    "VLLM_XLA_CHECK_RECOMPILATION",
}
VLLM = {"VLLM_USE_V1", "VLLM_TPU_USING_PATHWAYS", "VLLM_LOGGING_LEVEL"}

# Names GKE, libtpu and JAX set in TPU pods. None of them may be reported.
EXTERNAL_TPU_NAMES = [
    "TPU_ACCELERATOR_TYPE",
    "TPU_TOPOLOGY",
    "TPU_HOST_BOUNDS",
    "TPU_WORKER_ID",
    "TPU_WORKER_HOSTNAMES",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_SKIP_MDS_QUERY",
    "TPU_TOPOLOGY_WRAP",
    "TPU_RUNTIME_METRICS_PORTS",
    "TPU_PROCESS_BOUNDS",
    "TPU_STDERR_LOG_LEVEL",
    "TPU_MIN_LOG_LEVEL",
]


@pytest.mark.parametrize(
    "set_names, expected",
    [
        pytest.param(
            ["TPU_PARALELL_PRECOMPILE"],
            [UnreadEnvVar("TPU_PARALELL_PRECOMPILE", "TPU_PARALLEL_PRECOMPILE")],
            id="typo_in_shared_prefix",
        ),
        pytest.param(
            ["MOE_FUSED_EP_KERNAL_IMPL"],
            [UnreadEnvVar("MOE_FUSED_EP_KERNAL_IMPL", "MOE_FUSED_EP_KERNEL_IMPL")],
            id="typo_in_other_package_prefix",
        ),
        pytest.param(
            ["VLLM_TPU_USE_NAIVE_KDA"],
            [UnreadEnvVar("VLLM_TPU_USE_NAIVE_KDA")],
            id="removed_vllm_tpu_knob",
        ),
        pytest.param(
            ["VLLM_TPU_MOST_MODEL_LEM"],
            [UnreadEnvVar("VLLM_TPU_MOST_MODEL_LEM", "VLLM_TPU_MOST_MODEL_LEN")],
            id="vllm_tpu_typo_gets_a_suggestion",
        ),
        pytest.param(
            ["VLLM_TPU_MOST_MODEL_LEN", "VLLM_XLA_CHECK_RECOMPILATION"],
            [],
            id="this_package_vllm_names_are_known",
        ),
        pytest.param(
            ["VLLM_TPU_USING_PATHWAYS", "VLLM_LOGGING_LEVEL"],
            [],
            id="vllm_core_names_are_known",
        ),
        pytest.param(EXTERNAL_TPU_NAMES, [], id="external_tpu_names_are_left_alone"),
        pytest.param(
            ["PATH", "HOME", "PYTHONPATH", "JAX_PLATFORMS"],
            [],
            id="unrelated_prefixes_are_left_alone",
        ),
        pytest.param([], [], id="empty_environment"),
    ],
)
def test_find_unread_env_vars(set_names, expected):
    assert find_unread_env_vars(set_names, TORCHTPU, VLLM) == expected


def test_findings_are_sorted_and_deduplicated():
    found = find_unread_env_vars(["VLLM_ZZZ", "VLLM_AAA", "VLLM_AAA"], TORCHTPU, VLLM)
    assert [u.name for u in found] == ["VLLM_AAA", "VLLM_ZZZ"]


def test_str_includes_the_suggestion():
    assert str(UnreadEnvVar("A_B", "A_C")) == "A_B (did you mean A_C?)"
    assert str(UnreadEnvVar("A_B")) == "A_B"


@pytest.fixture
def _clean_environ(monkeypatch):
    """Runs the platform check against only the variables a test sets."""
    for name in list(os.environ):
        monkeypatch.delenv(name)


@pytest.fixture
def _warning():
    with mock.patch.object(tpu_platform.logger, "warning", autospec=True) as warning:
        yield warning


def test_platform_warns_once_listing_every_unread_name(
    monkeypatch, _clean_environ, _warning
):
    monkeypatch.setenv("VLLM_TPU_USE_NAIVE_KDA", "1")
    monkeypatch.setenv("TPU_PARALELL_PRECOMPILE", "1")
    monkeypatch.setenv("VLLM_TPU_MOST_MODEL_LEN", "8192")
    TpuPlatform.validate_environ(hard_fail=False)
    _warning.assert_called_once_with(
        "Environment variables that nothing reads: "
        "TPU_PARALELL_PRECOMPILE (did you mean TPU_PARALLEL_PRECOMPILE?), "
        "VLLM_TPU_USE_NAIVE_KDA"
    )


def test_platform_raises_when_hard_fail(monkeypatch, _clean_environ):
    monkeypatch.setenv("VLLM_TPU_USE_NAIVE_KDA", "1")
    with pytest.raises(ValueError, match="nothing reads: VLLM_TPU_USE_NAIVE_KDA"):
        TpuPlatform.validate_environ(hard_fail=True)


def test_platform_accepts_registered_and_device_control_names(
    monkeypatch, _clean_environ, _warning
):
    monkeypatch.setenv("VLLM_TPU_MOST_MODEL_LEN", "8192")
    monkeypatch.setenv("TPU_VISIBLE_CHIPS", "0")
    monkeypatch.setenv("VLLM_TPU_USING_PATHWAYS", "0")
    TpuPlatform.validate_environ(hard_fail=True)
    _warning.assert_not_called()
