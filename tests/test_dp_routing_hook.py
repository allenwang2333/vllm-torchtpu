"""The DP routing hook the DSv4 agentx recipe installs into vLLM's DPLBAsyncMPClient."""

import asyncio
import base64
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from vllm.v1.engine.core_client import DPLBAsyncMPClient

RECIPE = (
    Path(__file__).resolve().parents[1]
    / "forge/recipes/deepseek-v4/infx-deepseek-v4-agentx-evolve.jsonnet"
)


def _hook_source() -> str:
    blob = re.search(r'echo "(ZGVmIF9pbnN0YWxs[^"]*)"', RECIPE.read_text()).group(1)
    return base64.b64decode(blob).decode()


@pytest.fixture
def install(monkeypatch):
    for name in ("get_core_engine_for_request", "process_engine_outputs"):
        monkeypatch.setattr(DPLBAsyncMPClient, name, DPLBAsyncMPClient.__dict__[name])

    def _install(context_load: bool):
        monkeypatch.setenv("ENABLE_PREFIX_ROUTING", "1")
        monkeypatch.setenv("PREFIX_ROUTING_NEW_SESSION_AFFINITY", "0")
        monkeypatch.setenv("PREFIX_ROUTING_CONTEXT_LOAD", "1" if context_load else "0")
        exec(_hook_source(), {})

    return _install


def _client(n=3):
    return SimpleNamespace(
        core_engines=[f"e{i}" for i in range(n)],
        lb_engines=[[0, 0, 0.0] for _ in range(n)],
        engine_inflight=Counter(),
        reqs_in_flight={},
        client_count=1,
    )


def _route(client, sid, prompt_tokens):
    req = SimpleNamespace(
        request_id=f"{sid}-req",
        session_id=sid,
        prompt_token_ids=[1] * prompt_tokens,
        data_parallel_rank=None,
    )
    return DPLBAsyncMPClient.get_core_engine_for_request(client, req)


def _route_four(client):
    return [
        _route(client, s, n)
        for s, n in (("a", 100_000), ("b", 50_000), ("c", 10), ("d", 10))
    ]


def test_count_routing_is_unchanged_by_default(install):
    install(context_load=False)
    # equal request counts: the tie goes to the engine routed to longest ago
    assert _route_four(_client()) == ["e0", "e1", "e2", "e0"]
    assert "context_load" not in DPLBAsyncMPClient.process_engine_outputs.__name__


def test_context_load_routes_to_the_fewest_prompt_tokens_in_flight(install):
    install(context_load=True)
    assert _route_four(_client()) == ["e0", "e1", "e2", "e2"]


def test_context_load_forgets_finished_requests(install):
    install(context_load=True)
    client = _client()
    _route_four(client)
    asyncio.run(
        DPLBAsyncMPClient.process_engine_outputs(
            client, SimpleNamespace(finished_requests={"a-req"})
        )
    )
    assert client.engine_inflight["e0"] == 0  # upstream bookkeeping still runs
    assert _route(client, "e", 10) == "e0"  # e0 now holds no prompt tokens
