# SPDX-License-Identifier: Apache-2.0
"""The llm-d proxy reads its engine sizing variables as PROXY_<name> when the
unprefixed name is absent (the standard agentx-397b-pd-baseline recipe passes only
PROXY_* variables to the proxy container); the unprefixed name still wins."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "examples/disagg/toy_proxy_server_llmd.py"
_spec = importlib.util.spec_from_file_location("toy_proxy_server_llmd_std", _PATH)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

NAMES = [
    "PIPELINE_PARALLEL_SIZE_PREFILL",
    "BLOCK_SIZE_PREFILL",
    "BLOCK_SIZE_DECODE",
    "KV_OFFLOAD_BYTES_PREFILL",
    "DECODE_CONTEXT_PARALLEL_SIZE_DECODE",
    "MAX_NUM_SEQS_DECODE",
]


@pytest.mark.parametrize("name", NAMES)
def test_unprefixed_wins(monkeypatch, name):
    monkeypatch.setenv(name, "8")
    monkeypatch.setenv("PROXY_" + name, "2176")
    assert _m._env_int_std(name, 1) == 8
    assert _m._env_float_std(name, 1.0) == 8.0


@pytest.mark.parametrize("name", NAMES)
def test_proxy_prefixed_fallback(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PROXY_" + name, "2176")
    assert _m._env_int_std(name, 1) == 2176


@pytest.mark.parametrize("name", NAMES)
def test_default_when_neither(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("PROXY_" + name, raising=False)
    assert _m._env_int_std(name, 7) == 7


def test_lifespan_reads_through_std_names():
    src = _PATH.read_text()
    for name in NAMES:
        assert f'_std("{name}"' in src, name
