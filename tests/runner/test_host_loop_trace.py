# SPDX-License-Identifier: Apache-2.0
import json

from vllm_torchtpu.runner import host_loop_trace
from vllm_torchtpu.runner.host_loop_trace import HostLoopTrace, _timed


def _lines(directory):
    files = list(directory.iterdir())
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def test_records_one_line_per_step(tmp_path, monkeypatch):
    monkeypatch.setattr(host_loop_trace, "_FLUSH_EVERY", 2)
    trace = HostLoopTrace(str(tmp_path), "worker")
    for step in range(4):
        trace.mark("a")
        trace.mark("a")  # first occurrence wins
        trace.note("step", step)
        trace.end_step()
    trace.end_step()  # empty record is dropped
    recs = _lines(tmp_path)
    assert [r["step"] for r in recs] == [0, 1, 2, 3]
    assert all(isinstance(r["a"], int) for r in recs)


def test_timed_marks_start_and_end_even_on_error(tmp_path):
    trace = HostLoopTrace(str(tmp_path), "engine")

    def boom():
        raise ValueError

    wrapped = _timed(trace, "sched", boom)
    try:
        wrapped()
    except ValueError:
        pass
    assert trace._rec["sched_s"] <= trace._rec["sched_e"]


def test_disabled_without_directory(monkeypatch):
    monkeypatch.delenv("TPU_HOST_LOOP_TRACE_DIR", raising=False)
    assert host_loop_trace.make_trace("worker") is None
    host_loop_trace.install_engine_core_trace()  # no-op, no vLLM import needed
