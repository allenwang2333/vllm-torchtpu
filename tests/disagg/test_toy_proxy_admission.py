# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
from pathlib import Path


def _load_toy_proxy_server():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "disagg"
        / "toy_proxy_server.py"
    )
    spec = importlib.util.spec_from_file_location("toy_proxy_server_adm", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_m = _load_toy_proxy_server()
_AdmissionBudget = _m._AdmissionBudget
_PrefixAffinityRouter = _m._PrefixAffinityRouter


def _run(coro):
    return asyncio.run(coro)


def test_budget_count_cap_blocks_and_releases() -> None:
    async def body():
        b = _AdmissionBudget("d", max_count=1, max_tokens=0)
        await b.acquire(10)
        second = asyncio.create_task(b.acquire(10))
        await asyncio.sleep(0.01)
        assert not second.done()
        assert b.waiting == 1
        await b.release(10)
        await asyncio.wait_for(second, 1.0)
        assert b.count == 1 and b.waiting == 0

    _run(body())


def test_budget_token_cap_counts_tokens_not_requests() -> None:
    async def body():
        b = _AdmissionBudget("d", max_count=0, max_tokens=100)
        await b.acquire(60)
        await b.acquire(30)
        third = asyncio.create_task(b.acquire(30))
        await asyncio.sleep(0.01)
        assert not third.done()
        await b.release(60)
        await asyncio.wait_for(third, 1.0)
        assert b.tokens == 60 and b.count == 2

    _run(body())


def test_budget_admits_oversize_request_only_when_empty() -> None:
    async def body():
        b = _AdmissionBudget("d", max_count=0, max_tokens=100)
        await b.acquire(500)
        assert b.count == 1
        nxt = asyncio.create_task(b.acquire(1))
        await asyncio.sleep(0.01)
        assert not nxt.done()
        await b.release(500)
        await asyncio.wait_for(nxt, 1.0)

    _run(body())


def test_budget_adjust_frees_room_when_estimate_was_high() -> None:
    async def body():
        b = _AdmissionBudget("d", max_count=0, max_tokens=100)
        await b.acquire(90)
        nxt = asyncio.create_task(b.acquire(50))
        await asyncio.sleep(0.01)
        assert not nxt.done()
        await b.adjust(-60)
        await asyncio.wait_for(nxt, 1.0)
        assert b.tokens == 80

    _run(body())


def _turns(router, text_prefix, n_turns, turn_len=4096):
    text = text_prefix
    ranks = []
    for _ in range(n_turns):
        text += "x" * turn_len
        rank, _ = router.route(router.hashes(text))
        router.done(rank)
        ranks.append(rank)
    return ranks


def test_router_residency_spreads_cold_sessions_and_keeps_turns_sticky() -> None:
    r = _PrefixAffinityRouter(n=2, chunk=1024, max_sessions=4)
    firsts = []
    for i in range(4):
        ranks = _turns(r, f"session-{i}-" * 100, 3)
        assert len(set(ranks)) == 1
        firsts.append(ranks[0])
    assert sorted(firsts) == [0, 0, 1, 1]
    assert r.stats()["resident"] == [2, 2]
    assert r.evicted == 0


def test_router_evicted_session_is_rerouted_as_cold() -> None:
    r = _PrefixAffinityRouter(n=2, chunk=1024, max_sessions=1)
    a = "aaaa-" * 1000
    b = "bbbb-" * 1000
    c = "cccc-" * 1000
    ra = r.route(r.hashes(a))[0]
    r.done(ra)
    rb = r.route(r.hashes(b))[0]
    r.done(rb)
    assert ra != rb
    # A third conversation evicts the LRU session of the rank it lands on.
    rc = r.route(r.hashes(c))[0]
    r.done(rc)
    assert r.evicted == 1
    evicted_text = a if rc == ra else b
    rank, sticky = r.route(r.hashes(evicted_text + "x" * 4096))
    assert not sticky
    assert r.rerouted == 1
    r.done(rank)


def test_router_saturation_relaxes_slack() -> None:
    r = _PrefixAffinityRouter(n=2, chunk=1024, slack=0, saturated_inflight=2)
    a = "aaaa-" * 1000
    ra, _ = r.route(r.hashes(a))
    r.done(ra)
    other = 1 - ra
    # Owner leg busier than the other beyond the slack: not saturated, so cold.
    r.inflight = [0, 0]
    r.inflight[ra] = 1
    rank, sticky = r.route(r.hashes(a + "x" * 4096))
    assert not sticky and rank == other
    r.done(rank)
    # Both legs at or above saturated_inflight: sticky wins despite imbalance.
    r.inflight = [2, 2]
    r.inflight[rank] = 5
    rank2, sticky2 = r.route(r.hashes(a + "x" * 8192))
    assert sticky2 and rank2 == rank
    assert r.sticky_saturated == 1


def test_prompt_token_helpers() -> None:
    assert _m._estimate_prompt_tokens("a" * 400, 4.0) == 100
    assert _m._expected_output_tokens({"max_tokens": 50}, 1024) == 50
    assert _m._expected_output_tokens({"max_tokens": 5000}, 1024) == 1024
    assert _m._expected_output_tokens({}, 1024) == 1024
    assert _m._prompt_tokens_from_prefill({"usage": {"prompt_tokens": 7}}, {}) == 7
    assert _m._prompt_tokens_from_prefill({}, {"num_tokens": 9}) == 10
    assert _m._prompt_tokens_from_prefill({}, {}) is None
