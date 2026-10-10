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
"""The programs a run starts, as stubs: each records its call and fakes its work.

Run as `python stubs.py <program> <args...>` from a shell shim per program
name (runtime_helpers.stub_env writes them). Every call is recorded as one
JSON file under $STUB_RECORDS: the program, argv, env, cwd and pid.

`vllm serve` is a small OpenAI-compatible HTTP server: /health,
/v1/models, /v1/completions, /v1/chat/completions and /metrics, plus
/stub/die, which ends it as a crash would. It forks one child that stands in
for an engine core. Knobs, all env:

    STUB_SERVE_EXIT=N     exit N at once, before serving
    STUB_READY_AFTER=S    /health answers 503 for S seconds
    STUB_NEVER_READY=1    /health never answers 200
    STUB_DIE_AFTER=S      crash S seconds after start
"""

import http.server
import json
import os
import pathlib
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

RECORDS = os.environ.get("STUB_RECORDS")


def record(program: str, argv: list[str], **extra: Any) -> None:
    if not RECORDS:
        return
    data = {
        "program": program,
        "argv": argv,
        "env": dict(os.environ),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        **extra,
    }
    path = pathlib.Path(RECORDS) / f"{time.time_ns()}-{os.getpid()}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.rename(path)


def flag(args: list[str], name: str, default: str | None = None) -> str | None:
    """The value of --name, as `--name v` or `--name=v`; the last one wins."""
    value = default
    for i, arg in enumerate(args):
        if arg == name and i + 1 < len(args):
            value = args[i + 1]
        elif arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
    return value


def number(name: str, default: float = 0.0) -> float:
    return float(os.environ.get(name, default))


# vllm serve -----------------------------------------------------------------


def serve(args: list[str]) -> int:
    if "STUB_SERVE_EXIT" in os.environ:
        print("ERROR stub: engine failed to start", flush=True)
        return int(os.environ["STUB_SERVE_EXIT"])
    port = int(flag(args, "--port", "8000") or 8000)
    served = flag(args, "--served-model-name") or args[0]
    max_len = int(flag(args, "--max-model-len", "4096") or 4096)
    started = time.monotonic()
    ready_after = number("STUB_READY_AFTER")
    never_ready = os.environ.get("STUB_NEVER_READY") == "1"
    engine = os.fork()
    if engine == 0:
        # The engine core: it lives as long as the group does.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        while True:
            time.sleep(60)
    record("vllm serve", args, engine_pid=engine)

    class Handler(http.server.BaseHTTPRequestHandler):
        def reply(self, code: int, body: Any, kind: str = "application/json") -> None:
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                ready = not never_ready and time.monotonic() - started >= ready_after
                self.reply(200 if ready else 503, {})
            elif self.path == "/v1/models":
                card = {"id": served, "object": "model", "max_model_len": max_len}
                self.reply(200, {"object": "list", "data": [card]})
            elif self.path == "/metrics":
                self.reply(200, "vllm:num_requests_running 0\n", "text/plain")
            elif self.path == "/stub/die":
                self.reply(200, {})
                print("ERROR stub: engine core died", flush=True)
                os.kill(engine, signal.SIGKILL)
                os._exit(1)
            else:
                self.reply(404, {"error": self.path})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            if self.path == "/v1/completions":
                self.reply(200, {"choices": [{"text": " 5", "finish_reason": "stop"}]})
            elif self.path == "/v1/chat/completions":
                message = {"role": "assistant", "content": "5"}
                self.reply(200, {"choices": [{"message": message}]})
            else:
                self.reply(404, {"error": self.path})

        def log_message(self, fmt: str, *values: Any) -> None:
            print(f"INFO stub: {fmt % values}", flush=True)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"INFO stub: serving {served} on port {port}", flush=True)
    die_after = number("STUB_DIE_AFTER")
    if die_after:

        def crash() -> None:
            time.sleep(die_after)
            print("ERROR stub: engine core died", flush=True)
            os.kill(engine, signal.SIGKILL)
            os._exit(1)

        threading.Thread(target=crash, daemon=True).start()

    def stop(signum: int, frame: Any) -> None:
        print("INFO stub: shutting down", flush=True)
        os.kill(engine, signal.SIGTERM)
        os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    httpd.serve_forever()
    return 0


def vllm(args: list[str]) -> int:
    if args[:1] == ["serve"]:
        return serve(args[1:])
    record("vllm", args)
    print(f"stub vllm: unknown command {args}", file=sys.stderr)
    return 2


PROGRAMS: dict[str, Callable[[list[str]], int]] = {"vllm": vllm}


def main() -> int:
    program, args = sys.argv[1], sys.argv[2:]
    return PROGRAMS[program](args)


if __name__ == "__main__":
    sys.exit(main())
