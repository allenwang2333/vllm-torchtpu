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
"""The microbenchmark clock, and a guard against the bug it replaced.

No TPU: the helpers are exercised against a fake callable, and the guard is a
static AST scan. The bug being guarded is `jax.block_until_ready` *inside* a
timing loop, which charged a host round trip to every sample and so read a fast
kernel as far slower than it is.
"""

import ast
import pathlib
import sys
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

TIMED_SCRIPTS = (
    "scripts/benchmark_masked_dense_vs_sparse_mla.py",
    "scripts/mla_mixed_prefill_tuner.py",
    "scripts/flash_attention_tuner.py",
)


@pytest.fixture(autouse=True)
def fake_jax(monkeypatch):
    """Stub jax so the module imports and blocks are counted, with no TPU."""
    calls = {"blocks": 0}

    def block_until_ready(x):
        calls["blocks"] += 1
        return x

    monkeypatch.setitem(sys.modules, "jax",
                        types.SimpleNamespace(
                            block_until_ready=block_until_ready))
    sys.modules.pop("kernel_timing", None)   # _mod re-execs it per test
    return calls


def _mod():
    """Fresh module per test, so each binds to that test's fake jax."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "kernel_timing", REPO / "scripts" / "kernel_timing.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kernel_timing"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_burst_blocks_once_not_per_call(fake_jax):
    """The whole point: N dispatches, one block."""
    kt = _mod()
    dispatched = []
    kt.burst_ms(lambda: dispatched.append(1), iters=32)
    assert len(dispatched) == 32
    assert fake_jax["blocks"] == 1, (
        "one block per burst; blocking per call is the bug this replaced")


def test_pipelined_reports_per_call_not_per_burst(fake_jax):
    """A burst's elapsed time is divided by the call count."""
    kt = _mod()
    slept = 0.002

    def slow():
        import time
        time.sleep(slept)

    per_call_ms = kt.pipelined_ms(slow, iters=20, warmup=1, bursts=1)
    assert slept * 1e3 * 0.5 < per_call_ms < slept * 1e3 * 2.0, (
        f"{per_call_ms} ms is not near the {slept * 1e3} ms per call slept")


def test_blocking_syncs_every_call(fake_jax):
    """The legacy clock is kept only as a diagnostic, so it must still block."""
    kt = _mod()
    kt.blocking_ms(lambda: None, iters=20, warmup=0)
    assert fake_jax["blocks"] == 20, (
        "blocking_ms must sync per call -- it exists to expose the gap")


def test_both_returns_pipelined_first(fake_jax):
    """Order matters: callers quote [0] and diagnose with [1]."""
    kt = _mod()
    slept = 0.002

    def slow():
        import time
        time.sleep(slept)

    pipe, wall = kt.both_ms(slow, iters=20, warmup=1, bursts=1)
    # No real device, so both clocks measure the same work: pin order, not gap.
    assert pipe == pytest.approx(wall, rel=0.5)
    assert pipe > 0 and wall > 0


def test_slope_cancels_a_fixed_per_burst_cost(fake_jax):
    """A constant added to every burst must not survive the subtraction."""
    kt = _mod()
    calls = {"n": 0}

    def work():
        calls["n"] += 1

    # slope must run both an n-call and a 2n-call burst to take a difference.
    kt.slope_ms(work, iters=20, warmup=1, bursts=1)
    assert calls["n"] == 1 + 20 + 40, (
        f"slope must time both n and 2n bursts, saw {calls['n']} calls")


def test_short_bursts_are_refused(fake_jax):
    """A burst below the floor is refused, not silently reported."""
    kt = _mod()
    with pytest.raises(ValueError, match="MIN_ITERS"):
        kt.pipelined_ms(lambda: None, iters=kt.MIN_ITERS - 1)
    with pytest.raises(ValueError, match="MIN_ITERS"):
        kt.paired_ms(lambda: None, lambda: None, iters=1)
    with pytest.raises(ValueError, match="MIN_ITERS"):
        kt.slope_ms(lambda: None, iters=1)


def test_paired_alternates_arm_order(fake_jax):
    """Arms interleave so drift penalizes both, not whichever ran second."""
    kt = _mod()
    order = []
    kt.paired_ms(lambda: order.append("a"), lambda: order.append("b"),
                 iters=20, warmup=1, bursts=2)
    # Skip the two warmup calls; each of the two bursts that follow runs 20
    # calls of each arm.
    measured = order[2:]
    burst1, burst2 = measured[:40], measured[40:]
    assert burst1[0] == "a" and burst2[0] == "b", (
        f"arms did not alternate between bursts: {burst1[0]}, {burst2[0]}")


def test_paired_refuses_an_odd_burst_count(fake_jax):
    """An odd count lets the median land on the majority lead position."""
    kt = _mod()
    with pytest.raises(ValueError, match="odd"):
        kt.paired_ms(lambda: None, lambda: None, iters=20, bursts=3)
    assert kt.PAIRED_BURSTS % 2 == 0


def _blocking_timing_loops(tree):
    """Loops whose body both reads a clock and blocks -- the bug's signature."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        clock = block = False
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            name = ast.unparse(inner.func)
            if "perf_counter" in name or name.endswith("time.time"):
                clock = True
            if "block_until_ready" in name:
                block = True
        if clock and block:
            found.append(node.lineno)
    return found


@pytest.mark.parametrize("rel", TIMED_SCRIPTS)
def test_no_blocking_inside_a_timing_loop(rel):
    """Regression guard: reintroducing the per-call block fails here.

    A loop that both reads `perf_counter` and calls `block_until_ready` is
    timing a synchronous round trip. Time a pipelined burst with
    `scripts/kernel_timing.py` instead.
    """
    path = REPO / rel
    tree = ast.parse(path.read_text())
    assert not _blocking_timing_loops(tree), (
        f"{rel}: blocking inside a timing loop at line(s) "
        f"{_blocking_timing_loops(tree)}")
