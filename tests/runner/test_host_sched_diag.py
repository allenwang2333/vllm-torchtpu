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
    d = diag_mod.HostSchedDiag(7, 3)
    stop = threading.Event()
    worker = threading.Thread(target=lambda: [sum(range(1000)) for _ in iter(stop.is_set, True)])
    worker.start()
    try:
        lines = []
        for _ in range(9):
            sum(range(20000))  # some CPU on the calling thread
            lines.append(d.tick())
    finally:
        stop.set()
        worker.join()
    assert [i for i, l in enumerate(lines) if l] == [2, 5, 8]
    line = lines[-1]
    assert line.startswith("HOSTDIAG rank=7 steps=3 ")
    vals = dict(re.findall(r"(\w[\w/]*)=([-\d.]+)\b", line))
    for k in ("wall_ms/step", "main_run_ms/step", "main_wait_ms/step", "proc_run_ms/step", "proc_wait_ms/step"):
        assert float(vals[k]) >= 0, (k, line)
    assert float(vals["main_run_ms/step"]) > 0
    # the process total includes the serving thread
    assert float(vals["proc_run_ms/step"]) >= float(vals["main_run_ms/step"]) - 1e-3
    assert int(vals["threads"]) >= 2
    assert "top=" in line


def test_missing_cgroup_is_tolerated(monkeypatch):
    monkeypatch.setattr(diag_mod, "_CGROUP_STATS", ("/nonexistent/cpu.stat",))
    d = diag_mod.HostSchedDiag(0, 1)
    line = d.tick()
    assert "cg_throttled=0/0" in line
