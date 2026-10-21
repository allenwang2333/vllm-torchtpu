# SPDX-License-Identifier: Apache-2.0
"""PROXY_*_ROUTING=llmd runs examples/disagg/toy_proxy_server_llmd.py (the final
lineage's llm-d proxy); every other routing value keeps toy_proxy_server.py."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_DISAGG = Path(__file__).resolve().parents[2] / "examples" / "disagg"


def _load():
    spec = importlib.util.spec_from_file_location(
        "toy_proxy_server_dispatch", _DISAGG / "toy_proxy_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_m = _load()


@pytest.mark.parametrize(
    "prefill,decode,want",
    [
        ("llmd", "llmd", True),
        ("llmd", None, True),
        (None, "llmd", True),
        ("LLMD ", None, True),
        ("prefix_hash", "prefix_hash", False),
        ("prefix_hash", "round_robin", False),
        (None, None, False),
    ],
)
def test_wants_llmd_proxy(monkeypatch, prefill, decode, want):
    for k, v in (("PROXY_PREFILL_ROUTING", prefill), ("PROXY_DECODE_ROUTING", decode)):
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    assert _m._wants_llmd_proxy() is want


def test_llmd_proxy_path_found_beside_file():
    assert _m._llmd_proxy_path() == str(_DISAGG / "toy_proxy_server_llmd.py")


def _help(routing: str) -> str:
    env = dict(os.environ, PROXY_PREFILL_ROUTING=routing, PROXY_DECODE_ROUTING=routing)
    r = subprocess.run(
        [sys.executable, str(_DISAGG / "toy_proxy_server.py"), "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_main_dispatches_to_llmd_proxy():
    out = _help("llmd")
    assert "running the llm-d proxy" in out
    assert "toy_proxy_server_llmd.py" in out


def test_main_keeps_this_proxy_for_prefix_hash():
    out = _help("prefix_hash")
    assert "running the llm-d proxy" not in out
