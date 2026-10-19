"""HostSchedDiag: off by default, one parsable line every N steps."""

import re
import threading

from vllm_torchtpu.runner import host_sched_diag as diag_mod


def test_off_reads_nothing(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("an off diag must not read /proc")

    monkeypatch.setattr(diag_mod, "_schedstat", boom)
    d = diag_mod.HostSchedDiag(0, 0)
    assert all(d.tick() is None for _ in range(5))


def test_line_every_n_steps_with_sane_values():
    d = diag_mod.HostSchedDiag(7, 3, fwd_timers=False)
    stop = threading.Event()
    worker = threading.Thread(
        target=lambda: [sum(range(1000)) for _ in iter(stop.is_set, True)]
    )
    worker.start()
    try:
        lines = []
        for _ in range(9):
            sum(range(20000))  # some CPU on the calling thread
            lines.append(d.tick())
    finally:
        stop.set()
        worker.join()
    assert [i for i, ln in enumerate(lines) if ln] == [2, 5, 8]
    line = lines[-1]
    assert line.startswith("HOSTDIAG rank=7 steps=3 ")
    vals = dict(re.findall(r"(\w[\w/]*)=([-\d.]+)\b", line))
    for k in (
        "wall_ms/step",
        "main_run_ms/step",
        "main_wait_ms/step",
        "proc_run_ms/step",
        "proc_wait_ms/step",
    ):
        assert float(vals[k]) >= 0, (k, line)
    assert float(vals["main_run_ms/step"]) > 0
    # the process total includes the serving thread
    assert float(vals["proc_run_ms/step"]) >= float(vals["main_run_ms/step"]) - 1e-3
    assert int(vals["threads"]) >= 2
    assert "top=" in line


def test_missing_cgroup_is_tolerated(monkeypatch):
    monkeypatch.setattr(diag_mod, "_CGROUP_STATS", ("/nonexistent/cpu.stat",))
    d = diag_mod.HostSchedDiag(0, 1, fwd_timers=False)
    line = d.tick()
    assert "cg_throttled=0/0" in line


def test_line_carries_fwd_timers_and_thread_placement(monkeypatch):
    calls = []

    class Fake:
        def __call__(self, x):
            calls.append(x)
            return x + 1

    # the timer wrapper keeps the callee's behaviour and accumulates per name
    wrapped = diag_mod._timed("unit_fake", Fake.__call__)
    monkeypatch.setattr(Fake, "__call__", wrapped)
    f = Fake()
    assert [f(i) for i in range(4)] == [1, 2, 3, 4] and calls == [0, 1, 2, 3]
    ns, n = diag_mod.FWD_TIMERS["unit_fake"]
    assert n == 4 and ns > 0
    d = diag_mod.HostSchedDiag(1, 2, fwd_timers=False)
    d.tick()
    line = d.tick()
    assert re.search(r" cpu=\d+ nvcsw/step=[\d.]+( sib_busy=[\d.]+)?$", line), line
    assert "fwd_ms/step(calls)=" in line


def test_install_fwd_timers_is_idempotent(monkeypatch):
    from torch._dynamo.aot_compile import AOTCompiledFunction

    # restore the classes after the test (monkeypatch undoes these setattrs)
    for attr in ("__call__", "guard_check"):
        monkeypatch.setattr(
            AOTCompiledFunction, attr, AOTCompiledFunction.__dict__[attr]
        )
    try:
        from vllm.compilation.caching import VllmSerializableFunction

        monkeypatch.setattr(
            VllmSerializableFunction,
            "__call__",
            VllmSerializableFunction.__dict__["__call__"],
        )
    except ImportError:
        pass
    first = diag_mod.install_fwd_timers()
    assert "aot" in first and "guard" in first
    assert diag_mod.install_fwd_timers() == []
    fn = AOTCompiledFunction.__dict__["__call__"]
    assert fn._h5_timed and not getattr(fn.__wrapped__, "_h5_timed", False)


def test_proc_stat_and_siblings():
    st = diag_mod._proc_stat()
    assert st and all(0 <= b <= t for b, t in st.values())
    import os

    cpu = sorted(os.sched_getaffinity(0))[0]
    assert cpu not in diag_mod._siblings(cpu)
