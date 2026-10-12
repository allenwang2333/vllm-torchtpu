# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest


def _load_session_router():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "dp_router"
        / "session_router.py"
    )
    spec = importlib.util.spec_from_file_location("session_router", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_sr = _load_session_router()


def _router(policy="session", n=2, **kw):
    backends = [f"http://b{i}:8000" for i in range(n)]
    return _sr.Router(_sr.RouterConfig(backends=backends, policy=policy, **kw))


def _body(text: str) -> bytes:
    return json.dumps({"messages": [{"role": "user", "content": text}]}).encode()


def test_session_is_sticky_and_new_sessions_go_least_loaded() -> None:
    r = _router()
    first = r.choose({"X-Correlation-ID": "a"}, b"{}")
    r.inflight[first] += 5
    second = r.choose({"X-Correlation-ID": "b"}, b"{}")
    assert second != first
    for _ in range(3):
        assert r.choose({"X-Correlation-ID": "a"}, b"{}") == first
    assert r.sessions_per_backend == [1, 1]
    assert r.snapshot()["sessions_total"] == 2


def test_session_without_header_falls_back_to_prefix() -> None:
    r = _router(chunk_chars=16)
    body = _body("x" * 200)
    idx = r.choose({}, body)
    assert r.fallback_no_header == 1
    assert r.choose({}, body) == idx
    assert r.snapshot()["prefix_chunk_match_rate"] > 0


def test_prefix_prefers_longest_match_until_load_guard() -> None:
    r = _router(policy="prefix", chunk_chars=16, balance_abs=2)
    shared = "y" * 160
    idx = r.choose({}, _body(shared))
    assert r.choose({}, _body(shared + "more text here " * 4)) == idx
    r.inflight[idx] = 3
    assert r.choose({}, _body(shared)) != idx


def test_round_robin_cycles() -> None:
    r = _router(policy="round_robin", n=3)
    assert [r.choose({}, b"{}") for _ in range(6)] == [0, 1, 2, 0, 1, 2]


def test_upstream_headers_drop_hop_by_hop_and_add_dp_rank() -> None:
    r = _router(dp_rank_header="X-data-parallel-rank")
    out = r.upstream_headers(
        {"Connection": "keep-alive", "Host": "h", "X-Correlation-ID": "s"}, 1, "POST"
    )
    assert out == {"X-Correlation-ID": "s", "X-data-parallel-rank": "1"}
    assert "X-data-parallel-rank" not in r.upstream_headers({}, 1, "GET")
    assert "X-data-parallel-rank" not in _router().upstream_headers({}, 1, "POST")


def test_config_validation_and_cli_env_defaults(monkeypatch) -> None:
    with pytest.raises(ValueError):
        _sr.RouterConfig(backends=[])
    with pytest.raises(ValueError):
        _sr.RouterConfig(backends=["http://b"], policy="random")
    monkeypatch.setenv("ROUTER_BACKENDS", "http://a:1/ http://b:2")
    monkeypatch.setenv("ROUTER_POLICY", "prefix")
    args = _sr.parse_args([])
    assert args.backends == ["http://a:1/", "http://b:2"]
    assert args.policy == "prefix"
    assert _sr.RouterConfig(backends=args.backends).backends == [
        "http://a:1",
        "http://b:2",
    ]
