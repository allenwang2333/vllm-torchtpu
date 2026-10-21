"""PR #2243 quadratic prefill cost in the llm-d style prefill router (final-p2)."""

import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[2] / "examples/disagg/toy_proxy_server_llmd.py"
    spec = importlib.util.spec_from_file_location("toy_proxy_server_quad", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load()
Router = _mod._PrefixAffinityRouter
K = 1.23e-5


def _router(coeff):
    return Router(
        n=2,
        chunk=2176 * 4,
        role="prefill",
        block_tokens=2176,
        hbm_max_blocks=1616,
        dram_max_blocks=15427,
        peak_prefill_tok_s=62000.0,
        quad_attn_coeff=coeff,
        peak_prefill_eff_tok_s=2.1 * 62000.0,
    )


def _backlog(r):
    # Pod 0: one 8K-token turn behind a 150K-token cached prefix in flight.
    # Pod 1: one 12K-token cold request in flight. Linear tokens say pod 0 is
    # lighter (8K < 12K); attention FLOPs say pod 0 is heavier.
    r.inflight[:] = [1, 1]
    r.inflight_uncached_tokens[:] = [8000, 12000]
    r.inflight_eff_tokens[:] = [
        r.effective_tokens(8000, 150000),
        r.effective_tokens(12000, 0),
    ]


def test_effective_tokens_matches_the_pr_formula():
    r = _router(K)
    assert r.effective_tokens(0, 5000) == 0.0
    assert r.effective_tokens(4096, 0) == 4096 + K * 4096 * 2048
    assert r.effective_tokens(4096, 200000) > 2.0 * r.effective_tokens(4096, 0)


def test_coeff_zero_keeps_linear_routing_and_no_eff_accounting():
    r = _router(0.0)
    _backlog(r)
    r.inflight_eff_tokens[:] = [0.0, 0.0]
    rank, sticky, _, _ = r.route_prefill([], 4096, None, req_id="a")
    assert (rank, sticky) == (0, False)
    assert r.inflight_eff_tokens == [0.0, 0.0] and r.quad_flips == 0


def test_quadratic_cost_sends_cold_work_to_the_flops_lighter_pod():
    r = _router(K)
    _backlog(r)
    rank, sticky, u, d = r.route_prefill([], 4096, None, req_id="cold")
    assert (rank, sticky) == (1, False)
    assert r.quad_flips == 1
    before = r.inflight_eff_tokens[1]
    r.done_prefill(1, u, d, req_id="cold")
    assert abs(r.inflight_eff_tokens[1] - (before - r.effective_tokens(4096, 0))) < 1e-6


def test_warm_turn_stays_sticky_and_accounting_drains_to_zero():
    r = _router(K)
    hs = r.hashes("x" * (2176 * 4 * 40))
    rank0, _, u0, d0 = r.route_prefill(hs, 87040, None, req_id="t1")
    r.done_prefill(rank0, u0, d0, req_id="t1")
    hs2 = r.hashes("x" * (2176 * 4 * 40) + "y" * (2176 * 4 * 2))
    rank1, sticky, u1, d1 = r.route_prefill(hs2, 91392, None, req_id="t2")
    assert rank1 == rank0 and sticky
    assert r.inflight_eff_tokens[rank1] > 0
    r.done_prefill(rank1, u1, d1, req_id="t2")
    assert r.inflight_eff_tokens == [0.0, 0.0]
