# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sanity probe must be able to return both answers.

A gate validated only on the input it should pass is not validated. Each case
here is a transcript a real server can produce, and the probe's verdict on it
is asserted in both directions.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_PROBE = (pathlib.Path(__file__).resolve().parents[2] / "scripts" / "vllm" /
          "benchmarking" / "moe_sanity_probe.py")


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("moe_sanity_probe", _PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _transcript(probe, texts):
    return [{
        "prompt": prompt,
        "expect": list(expect),
        "text": text
    } for (prompt, expect), text in zip(probe.PROMPTS, texts)]


_HEALTHY = [
    " Paris, the largest city in France.",
    " 4",
    " H2O, two hydrogen atoms bonded to one oxygen.",
    " Thursday, Friday, Saturday, Sunday.",
    " 5, 6, 7, 8",
    " Jupiter, a gas giant about 318 times Earth's mass.",
    " Bonjour",
    " cold",
]

# What half-uninitialized routed experts produce: fluent-looking tokens with
# no relation to the prompt. This is the transcript BK #299 would have had.
_CORRUPT = [
    " the the of and in the to a the of and in the to a",
    " lorem ipsum dolor sit amet",
    " 1948 1948 1948 1948 1948 1948 1948 1948",
    " ??? ??? ???",
    " and then the and then the and then the and then the",
    " Wednesday elephant seventeen",
    " glorp",
    " warm and also warm and also warm and also warm and also",
]


def test_a_healthy_transcript_passes(probe) -> None:
    passed, reasons, metrics = probe.score(_transcript(probe, _HEALTHY), 6,
                                           0.6)
    assert passed, reasons
    assert metrics["hits"] == 8


def test_the_corrupt_transcript_fails(probe) -> None:
    passed, reasons, metrics = probe.score(_transcript(probe, _CORRUPT), 6,
                                           0.6)
    assert not passed
    assert metrics["hits"] <= 1
    assert any("answered correctly" in reason for reason in reasons)


def test_an_empty_completion_is_a_failure_not_a_zero(probe) -> None:
    """A server that returns "" must not read as a clean, unanswered probe."""
    passed, reasons, _ = probe.score(_transcript(probe, [""] * 8), 0, 0.6)
    assert not passed
    assert any("empty" in reason for reason in reasons)


def test_a_degenerate_loop_fails_even_when_the_answers_are_right(
        probe) -> None:
    """Right answer, then a cycle: correctness alone cannot be the gate."""
    looped = [text + (" and so on and so forth" * 6) for text in _HEALTHY]
    passed, reasons, _ = probe.score(_transcript(probe, looped), 6, 0.6)
    assert not passed
    assert any("degenerate" in reason for reason in reasons)


def test_repeat_fraction_ignores_short_answers(probe) -> None:
    """Eight words cannot be shown to loop; the detector must not guess."""
    assert probe._repeat_fraction(" 4") == 0.0
    assert probe._repeat_fraction("a b c d " * 6) == pytest.approx(1.0)


def _fake_server(texts):
    """A stdlib OpenAI-completions server that replays `texts` in order."""
    import http.server
    import json as _json
    import threading

    remaining = list(texts)

    class Handler(http.server.BaseHTTPRequestHandler):

        def log_message(self, *args):  # noqa: A003 - silence the access log
            pass

        def do_GET(self):
            self._reply({"data": [{"id": "fake-k3"}]})

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self._reply({"choices": [{"text": remaining.pop(0)}]})

        def _reply(self, body):
            payload = _json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.parametrize("texts,expected_exit", [(_HEALTHY, 0), (_CORRUPT, 1)])
def test_the_probe_exits_nonzero_end_to_end(probe, monkeypatch, tmp_path,
                                            texts, expected_exit) -> None:
    """Through the real HTTP path and the real argv, not just `score`.

    `run_benchmarks.sh` voids the arm on this exit code, so the code is the
    contract -- and a gate whose transport is untested is a gate that can
    return 0 because the request never went out.
    """
    server = _fake_server(texts)
    try:
        monkeypatch.setattr("sys.argv", [
            "moe_sanity_probe.py", "--port",
            str(server.server_address[1]), "--out",
            str(tmp_path / "probe.json")
        ])
        assert probe.main() == expected_exit
    finally:
        server.shutdown()

    import json as _json
    report = _json.loads((tmp_path / "probe.json").read_text())
    assert report["passed"] == (expected_exit == 0)
    assert len(report["records"]) == len(probe.PROMPTS)
