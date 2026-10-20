# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the GSM8K accuracy client against a mock OpenAI-style server."""

import asyncio
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/dsv41_bench/server_steps"
sys.path.insert(0, str(_SCRIPTS))

import gsm8k_client as g  # noqa: E402

pytestmark = pytest.mark.cpu_test


def test_mcnemar_matches_the_published_paired_result():
    # #2061: seven regressions, five improvements -> p = 0.7744.
    assert g.mcnemar_exact(7, 5) == pytest.approx(0.7744, abs=1e-4)
    assert g.mcnemar_exact(5, 7) == g.mcnemar_exact(7, 5)
    assert g.mcnemar_exact(0, 0) == 1.0
    assert g.mcnemar_exact(0, 6) == pytest.approx(2 / 64)


@pytest.mark.parametrize(
    "text,value",
    [
        ("so 3 + 4 = 7.\n#### 7", 7.0),
        ("#### 1,234", 1234.0),
        ("#### $18", 18.0),
        ("The answer is -5.", -5.0),
        ("She pays $18.50 in total", 18.5),
        ("no digits here", None),
    ],
)
def test_parse_number(text, value):
    assert g.parse_number(text) == value


def _dataset(tmp_path, n_train=10, n_test=12):
    data = tmp_path / "data"
    data.mkdir()
    for name, n in (("train", n_train), ("test", n_test)):
        with (data / f"{name}.jsonl").open("w") as f:
            for i in range(n):
                f.write(
                    json.dumps(
                        {"question": f"{name} q{i}?", "answer": f"work\n#### {i}"}
                    )
                    + "\n"
                )
    return data


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _serve_and_run(args, *, chat_mode=True):
    """A mock server: question i answered correctly unless i % 4 == 1 (wrong),
    i % 6 == 5 (truncated) or i == 3 (HTTP 500)."""
    from aiohttp import web

    seen = []

    async def chat(request):
        body = await request.json()
        seen.append(body)
        q = body["messages"][-1]["content"]
        i = int(q.split("q")[1].split("?")[0])
        if i == 3:
            return web.Response(status=500, text="boom")
        answer = f"#### {i + 1 if i % 4 == 1 else i}"
        finish = "length" if i % 6 == 5 else "stop"
        return web.json_response(
            {
                "choices": [{"message": {"content": answer}, "finish_reason": finish}],
                "usage": {"completion_tokens": 5, "prompt_tokens": 100},
            }
        )

    async def tokenize(request):
        body = await request.json()
        assert body["chat_template_kwargs"]["enable_thinking"] is False
        return web.json_response({"tokens": [1, 2, 3]})

    async def detokenize(request):
        tail = g.CHAT_MODE_SUFFIX if chat_mode else "<\uff5cAssistant\uff5c><think>"
        return web.json_response({"prompt": "..." + tail})

    async def metrics(request):
        return web.Response(
            text='vllm:num_requests_running{engine="0",model_name="m"} 3.0\n'
            'vllm:num_requests_running{engine="1",model_name="m"} 2.0\n'
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/detokenize", detokenize)
    app.router.add_get("/metrics", metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    port = _free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    try:
        args.url = f"http://127.0.0.1:{port}"
        return await g.run_client(args), seen
    finally:
        await runner.cleanup()


def _args(tmp_path, data, out="out", **kw):
    base = dict(
        data=str(data),
        out=str(tmp_path / out),
        model="m",
        concurrency=4,
        num_shots=4,
        seed=42,
        max_tokens=2048,
        timeout=30,
        limit=0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_client_scores_counts_and_records_protocol(tmp_path):
    data = _dataset(tmp_path)
    summary, seen = asyncio.run(_serve_and_run(_args(tmp_path, data)))
    # 12 questions; wrong: 1, 5, 9; error: 3 -> 8 correct.
    assert summary["questions"] == 12
    assert summary["correct"] == 8
    assert summary["request_errors"] == 1
    assert summary["length_truncations"] == 2  # i = 5, 11
    assert summary["running_peaks"]["global"] == 5.0
    train = g.read_jsonl(data / "train.jsonl")
    shots = g.build_conversations(train, [], num_shots=4, shot_seed=42)[1]
    assert summary["protocol"]["shot_train_indices"] == shots
    assert (
        seen[0]["messages"][0]["content"]
        == f"Question: {train[shots[0]]['question']}\nAnswer:"
    )
    body = seen[0]
    assert body["temperature"] == 0.0 and body["seed"] == 42
    assert body["max_tokens"] == 2048
    assert len(body["messages"]) == 2 * 4 + 1
    assert [m["role"] for m in body["messages"][:2]] == ["user", "assistant"]
    lines = (tmp_path / "out" / "results.jsonl").read_text().splitlines()
    assert [json.loads(line)["index"] for line in lines] == list(range(12))


def test_client_refuses_a_thinking_mode_prompt(tmp_path):
    data = _dataset(tmp_path)
    with pytest.raises(RuntimeError, match="chat-mode"):
        asyncio.run(_serve_and_run(_args(tmp_path, data), chat_mode=False))


def test_compare_pairs_questions(tmp_path):
    def write(name, correct):
        d = tmp_path / name
        d.mkdir()
        with (d / "results.jsonl").open("w") as f:
            for i, c in enumerate(correct):
                f.write(
                    json.dumps({"index": i, "correct": c, "response": str(c)}) + "\n"
                )
        return d

    a = write("a", [True, True, False, True, False])
    b = write("b", [True, False, True, True, True])
    res = g.compare(a, b)
    assert (res["regressions"], res["improvements"]) == (1, 2)
    assert res["regression_indices"] == [1] and res["improvement_indices"] == [2, 4]
    assert res["mcnemar_exact_p"] == pytest.approx(1.0)


def test_launcher_gsm8k_scenario_serves_the_decode_configuration(tmp_path, monkeypatch):
    import run as launcher

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__file__=__file__))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--scenario", "gsm8k", "--input", "gsm8k-data", "--out", "results"],
    )
    monkeypatch.setattr(launcher, "source_identity", lambda *a, **kw: {})
    monkeypatch.setattr(launcher.importlib.metadata, "version", lambda _: "1")
    monkeypatch.setattr(launcher, "bootstrap_ports", lambda *a: None)
    health = iter([SimpleNamespace(ok=False), SimpleNamespace(ok=True)])
    monkeypatch.setattr(launcher.requests, "get", lambda *a, **kw: next(health))
    server = SimpleNamespace(poll=lambda: None)
    starts, clients, audits = [], [], []

    def start(command, *, env, **kwargs):
        starts.append((command, env))
        return server

    monkeypatch.setattr(launcher.subprocess, "Popen", start)
    monkeypatch.setattr(launcher.subprocess, "run", lambda c, **kw: clients.append(c))
    monkeypatch.setattr(launcher, "stop", lambda s: None)
    monkeypatch.setattr(launcher, "assert_no_compilation", lambda *a: audits.append(a))
    launcher.main()

    decode = launcher.DECODE_SCENARIO
    command, env = starts[0]
    verify_rows = decode.local_concurrency * (decode.dspark_tokens + 1)
    assert json.loads(command[command.index("--compilation-config") + 1]) == {
        "compile_sizes": [verify_rows, decode.chunk_tokens]
    }
    assert env["MAX_SEQS"] == str(decode.local_concurrency)
    assert env["MAX_MODEL_LEN"] == str(decode.prefix_tokens + decode.output_tokens)
    (client,) = clients
    assert client[1].endswith("gsm8k_client.py") and client[2] == "client"
    assert client[client.index("--data") + 1] == "gsm8k-data"
    assert client[client.index("--concurrency") + 1] == str(
        8 * decode.local_concurrency
    )
    assert not audits
    status = json.loads((tmp_path / "results/startup.json").read_text())
    assert status["formal_status"] == "passed" and status["completed"]
