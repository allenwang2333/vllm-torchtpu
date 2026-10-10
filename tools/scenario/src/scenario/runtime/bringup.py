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
"""Bringing up this host's part: its pre commands, then its server.

Pre commands run in order before the server (the remote-code warm), each
within its own timeout. A best-effort one that fails is a warning; any
other is a bring-up failure (exit 3).

The readiness wait is Readiness.timeout_s from the spawn, the number `show`
prints: the engine's own startup timeout plus a margin, so a slow start
fails with vLLM's message rather than the wait's.
"""

import pathlib
import time
from collections.abc import Callable, Mapping

from scenario.plan.types import HostPlan, Readiness
from scenario.runtime import net
from scenario.runtime.exits import BRINGUP, NOT_READY, RunError
from scenario.runtime.log import Log
from scenario.runtime.placement import LOOPBACK, Refs
from scenario.runtime.proc import Proc
from scenario.runtime.proc import run as run_command

POLL_INTERVAL_S = 1.0
# Lines of the server's output repeated in a failure report; all of it was
# streamed above.
FAILURE_TAIL = 50


def host_env(
    base: Mapping[str, str], host: HostPlan, refs: Refs
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """The host's env: inherited, then the scenario's, then the tool's.

    Returns that env and the two parts the runtime adds, for the spawn log.
    """
    scenario = refs.env(host.scenario_env)
    tool = refs.env(host.tool_env)
    return {**base, **scenario, **tool}, scenario, tool


def run_pre(
    host: HostPlan,
    *,
    refs: Refs,
    env: Mapping[str, str],
    added: Mapping[str, Mapping[str, str]],
    cwd: pathlib.Path,
    log: Log,
    check: Callable[[], None] | None = None,
) -> None:
    """Run the host's pre commands in order, each within its own timeout.

    `check` is called while a command runs and may raise to stop it.

    Raises:
        RunError: exit 3, a command that is not best effort failed.
    """
    for index, command in enumerate(host.pre):
        what = f"pre[{index}]"
        argv = refs.argv(command.argv)
        extra = refs.env(command.env)
        log.spawn(what, argv, {**added, **({"env": extra} if extra else {})})
        done = run_command(
            argv,
            env={**env, **extra},
            cwd=cwd,
            timeout_s=command.timeout_s,
            check=check,
        )
        if done.ok:
            continue
        if done.timed_out:
            problem = f"{what} ({argv[0]}) did not finish in {command.timeout_s}s"
        else:
            problem = f"{what} ({argv[0]}) exited {done.code}"
        if command.best_effort:
            warning = f"{problem}; it is best effort, so bring-up goes on"
            log.warning(warning)
            continue
        raise RunError(BRINGUP, problem)


def start_server(
    host: HostPlan,
    readiness: Readiness,
    *,
    refs: Refs,
    env: Mapping[str, str],
    added: Mapping[str, Mapping[str, str]],
    cwd: pathlib.Path,
    log_path: pathlib.Path,
    log: Log,
) -> Proc:
    """Spawn the host's server; its output goes to stdout and `log_path`.

    Raises:
        RunError: exit 3, something already answers on the API port of a
            host that serves the API.
    """
    assert host.serve is not None
    if host.head and net.port_open(LOOPBACK, readiness.port):
        raise RunError(
            BRINGUP,
            f"port {readiness.port} is already taken: another server runs on this host",
        )
    argv = refs.argv(host.serve.argv)
    serve_env = refs.env(host.serve.env)
    log.spawn("serve", argv, {**added, **({"env": serve_env} if serve_env else {})})
    return Proc(argv, env={**env, **serve_env}, cwd=cwd, log=log_path).start()


def wait_ready(
    server: Proc,
    readiness: Readiness,
    *,
    spawned_at: float,
    log: Log,
    note: Callable[[], str] = lambda: "",
) -> float:
    """Wait for the server's readiness; the seconds it took.

    `note` adds to each progress line, such as the hosts not heard from.

    Raises:
        RunError: exit 4, the server died while starting or the wait ran out.
    """
    url = f"http://{LOOPBACK}:{readiness.port}{readiness.path}"
    outcome = net.wait_http(
        url,
        deadline=spawned_at + readiness.timeout_s,
        proc=server,
        interval_s=POLL_INTERVAL_S,
        progress=lambda waited: log.say(
            f"Waiting for server... ({waited:.0f}s elapsed){note()}"
        ),
    )
    waited = time.monotonic() - spawned_at
    if outcome == "ready":
        log.say(f"Server ready on port {readiness.port} ({waited:.0f}s)")
        return waited
    if outcome == "died":
        code = server.poll()
        report_tail(server, log)
        raise RunError(
            NOT_READY,
            f"the server died while starting (exit {code}) after {waited:.0f}s",
        )
    report_tail(server, log)
    raise RunError(
        NOT_READY,
        f"the server was not ready within {readiness.timeout_s}s "
        f"(GET {readiness.path} on port {readiness.port})",
    )


def report_tail(server: Proc, log: Log, count: int = FAILURE_TAIL) -> None:
    lines = server.tail(count)
    log.say(f"========== Last {len(lines)} lines of server.log ==========")
    log.say("\n".join(lines))
    log.say("=" * 51)
