# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming accounting and paired serving artifact validation."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "collectives_serving_client",
    Path(__file__).resolve().parents[2] / "scripts/collectives/serving_client.py",
)
client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client)

pytestmark = pytest.mark.cpu_test


def _event(value):
    return ["data: " + json.dumps(value), ""]


def test_stream_token_accounting_and_finish_contract():
    lines = _event({"choices": [{"token_ids": [1], "text": "a"}]})
    lines += _event(
        {"choices": [{"token_ids": [2, 3], "text": "bc", "finish_reason": "length"}]}
    )
    lines += _event({"choices": [], "usage": {"completion_tokens": 3}})
    lines += ["data: [DONE]", ""]
    times = iter((12.0, 14.0))
    row = client.consume_stream(lines, started=10.0, clock=lambda: next(times))
    assert row["token_ids"] == [1, 2, 3]
    assert row["text"] == "abc"
    assert [value["seconds"] for value in row["arrivals"]] == [2, 4]
    with pytest.raises(ValueError, match="incomplete"):
        client.consume_stream(lines[:-2], started=0, clock=lambda: 1)
    bad = lines.copy()
    bad[4] = 'data: {"usage": {"completion_tokens": 4}}'
    with pytest.raises(ValueError, match="disagree"):
        client.consume_stream(bad, started=0, clock=lambda: 1)


def test_strict_math_answer_does_not_use_intermediate_numbers():
    assert client.answer("12 apples minus 2 apples is #### 10") == "1E+1"
    assert client.answer("#### -1,234.50") == "-1234.5"
    assert client.answer("There are 12 apples, then 2 disappear.") is None


def _row(index, *, failed=False):
    if failed:
        return {"index": index, "started": 1.0, "ended": 4.0, "error": "timeout"}
    return {
        "index": index,
        "started": 1.0,
        "ended": 3.0,
        "token_ids": [1, 2],
        "ttft_ms": 1000.0,
        "tpot_ms": 1000.0,
        "e2e_ms": 2000.0,
        "single_token_chunks": True,
    }


def test_failures_remain_in_throughput_window():
    summary = client.summarize_requests([_row(0), _row(1, failed=True)])
    assert summary["requested"] == 2
    assert summary["completed"] == 1
    assert summary["failures"] == 1
    assert summary["output_tokens_per_second"] == pytest.approx(2 / 3)
    assert summary["wall_seconds"] == 3


def _campaign(tmp_path):
    paths = []
    for slot, arm in enumerate(client.ARMS):
        path = tmp_path / str(slot)
        path.mkdir()
        rows = []
        for isl, concurrency in client.CELLS:
            for index in range(10):
                rows.append(
                    dict(
                        _row(index), phase="measured", isl=isl, concurrency=concurrency
                    )
                )
        (path / "requests.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        client.save(
            path / "summary.json",
            {
                "slot": slot,
                "arm": arm,
                "model": "test",
                "inputs_sha256": "inputs",
                "sampling": client.SAMPLING,
                "performance_settings": client.PERFORMANCE_SETTINGS,
                "quality_output_tokens": 1024,
                "requests_sha256": client.digest(path / "requests.jsonl"),
                "server_metadata": {"start_id": str(slot), "compile_cache": str(path)},
                "started_unix": slot * 10,
                "ended_unix": slot * 10 + 1,
                "complete": True,
            },
        )
        paths.append(path)
    return paths


def test_abba_summary_uses_fresh_start_means_and_checks_cached_inputs(tmp_path):
    paths = _campaign(tmp_path)
    args = SimpleNamespace(runs=paths, output=tmp_path / "report.json")
    client.summarize(args)
    report = json.loads(args.output.read_text())
    assert report["performance_complete"]
    assert not report["complete"]
    assert not report["accuracy_complete"]
    assert len(report["cells"]) == 4
    assert report["cells"][0]["comparison"]["ttft_ms"]["pair_delta_percent"] == [0, 0]
    changed = json.loads((paths[2] / "summary.json").read_text())
    changed["inputs_sha256"] = "different"
    client.save(paths[2] / "summary.json", changed)
    with pytest.raises(ValueError, match="inputs_sha256"):
        client.summarize(args)


def test_abba_summary_rejects_reused_caches(tmp_path):
    paths = _campaign(tmp_path)
    changed = json.loads((paths[2] / "summary.json").read_text())
    changed["server_metadata"]["compile_cache"] = str(paths[1])
    client.save(paths[2] / "summary.json", changed)
    with pytest.raises(ValueError, match="compile_cache"):
        client.summarize(SimpleNamespace(runs=paths, output=tmp_path / "report.json"))


def test_paired_accuracy_retains_changed_answers(tmp_path):
    paths = _campaign(tmp_path)
    for slot in (0, 1):
        with (paths[slot] / "requests.jsonl").open("a") as output:
            for index in range(100):
                correct = slot == 0 or index != 99
                row = dict(
                    _row(index),
                    phase="quality",
                    gold="42",
                    prediction="42" if correct else "41",
                    correct=correct,
                )
                if not correct:
                    row["token_ids"] = [4, 1]
                output.write(json.dumps(row) + "\n")
        summary = json.loads((paths[slot] / "summary.json").read_text())
        summary["requests_sha256"] = client.digest(paths[slot] / "requests.jsonl")
        client.save(paths[slot] / "summary.json", summary)
    args = SimpleNamespace(runs=paths, output=tmp_path / "report.json")
    client.summarize(args)
    report = json.loads(args.output.read_text())
    assert report["complete"]
    assert report["accuracy_complete"]
    pair = report["quality_pairs"][0]
    assert pair["baseline_correct"] == 100
    assert pair["candidate_correct"] == 99
    assert pair["answer_disagreements"] == pair["token_disagreements"] == 1


def test_worker_snapshot_waits_for_matching_ack(tmp_path):
    import threading
    import time

    def observer():
        request = tmp_path / "snapshot-request.json"
        deadline = time.monotonic() + 2
        while not request.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        value = json.loads(request.read_text())
        client.save(
            tmp_path / f"snapshot-{value['request_id']}.json",
            {"request_id": value["request_id"], "unix": 123.0, "calls": {"8": 12}},
        )

    thread = threading.Thread(target=observer, daemon=True)
    thread.start()
    receipt = client.worker_snapshot(tmp_path, "measured/start", timeout=2)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert receipt["unix"] == 123
    assert receipt["sha256"] == client.digest(receipt["path"])
    requested = json.loads((tmp_path / "snapshot-request.json").read_text())
    assert requested["phase"] == "measured/start"
    assert requested["request_id"] == receipt["request_id"]


def test_worker_snapshot_requires_ack_and_matching_id(tmp_path, monkeypatch):
    monkeypatch.setattr(client.uuid, "uuid4", lambda: "fixed")
    with pytest.raises(TimeoutError, match="not acknowledged"):
        client.worker_snapshot(tmp_path, "measured/end", timeout=0.001)
    client.save(
        tmp_path / "snapshot-fixed.json",
        {"request_id": "wrong", "unix": 123.0, "calls": {}},
    )
    with pytest.raises(ValueError, match="ID mismatch"):
        client.worker_snapshot(tmp_path, "measured/end", timeout=1)


def test_clock_anchor_brackets_wall_clock(monkeypatch):
    clocks = iter((10.0, 10.002))
    monkeypatch.setattr(client.time, "perf_counter", lambda: next(clocks))
    monkeypatch.setattr(client.time, "time", lambda: 1000.0)
    anchor = client.clock_anchor()
    assert anchor["unix"] == 1000
    assert anchor["monotonic"] == pytest.approx(10.001)
    assert anchor["clock_uncertainty_seconds"] == pytest.approx(0.001)


def test_run_records_separate_phase_and_metrics_boundaries(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}")
    metadata = tmp_path / "server.json"
    client.save(
        metadata,
        {
            "arm": "baseline",
            "start_id": "start",
            "source_commit": "commit",
            "compile_cache": "cache",
        },
    )
    monkeypatch.setattr(
        client,
        "load_inputs",
        lambda *_: {"performance_prompts": {"1": [[1]] * 18, "8192": [[2]] * 18}},
    )
    monkeypatch.setattr(client, "request", lambda *_, **__: _row(0))
    monkeypatch.setattr(client, "metrics", lambda _: {"unix": 123.0, "text": ""})
    args = SimpleNamespace(
        inputs=inputs,
        model="test",
        output=tmp_path / "run",
        server_metadata=metadata,
        slot=0,
        quality_output_tokens=1024,
        worker_evidence_dir=None,
        base_url="unused",
        timeout=1,
        accuracy=False,
    )
    client.run(args)
    result = json.loads((args.output / "summary.json").read_text())
    assert result["complete"]
    assert len(result["phases"]) == 24
    assert result["phases_sha256"] == client.digest(args.output / "phases.jsonl")
    for start, end in zip(result["phases"][::2], result["phases"][1::2]):
        assert (start["edge"], end["edge"]) == ("start", "end")
        assert start["phase"] == end["phase"]
        assert start["unix"] <= end["unix"]
        assert start["monotonic"] <= end["monotonic"]
    for cell in result["cells"]:
        assert cell["metrics"]["before_unix"] == cell["metrics"]["after_unix"] == 123
        assert cell["metrics"]["sha256"] == client.digest(cell["metrics"]["path"])
