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
"""Validation rules V-01..V-17 and V-42.

Field types enforce the rules about one value. The rules here need several
fields of one validated scenario (a scenario without a sweep, or one point
of a sweep), and report every problem with its key path. Rules that need
other files or several points run during resolution or `validate`; they are
declared here too, so the reference lists every rule in one table.
"""

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from scenario.errors import Issue
from scenario.model.common import format_duration
from scenario.model.scenario import Scenario
from scenario.model.server import HARDWARE_FIELDS, RoleView
from scenario.model.tasks import LmEvalTask
from scenario.paths import KeyPath
from scenario.reserved import (
    AGENTX_CLIENT_ENV,
    CHIPLESS_DRIVER_ENV,
    DP_HOST_ARGS,
    INFERENCEX_EVAL_ENV,
    INFERENCEX_PERF_ENV,
    PERF_CLIENT_ARGS,
    RAY_ENV,
    READY_TIMEOUT_ENV,
    ROUTER_ARGS,
    SERVE_ARGS,
    tool_env_owner,
)

Stage = Literal["field", "scenario", "resolve", "validate"]
Check = Callable[[Scenario], Iterator[Issue]]


@dataclass(frozen=True)
class Rule:
    id: str
    summary: str
    stage: Stage
    check: Check | None = None


RULES: dict[str, Rule] = {}


def _declare(rule_id: str, summary: str, stage: Stage) -> None:
    RULES[rule_id] = Rule(rule_id, summary, stage)


def _rule(rule_id: str, summary: str) -> Callable[[Check], Check]:
    def register(check: Check) -> Check:
        RULES[rule_id] = Rule(rule_id, summary, "scenario", check)
        return check

    return register


def check_scenario(scenario: Scenario) -> list[Issue]:
    """Every scenario-stage problem of a validated scenario, in rule order."""
    issues = []
    for rule in RULES.values():
        if rule.check is not None:
            issues.extend(
                issue if issue.rule else replace(issue, rule=rule.id)
                for issue in rule.check(scenario)
            )
    return issues


def _int_arg(role: RoleView, key: str) -> int | None:
    value = role.args.get(key, 1)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _and(names: Sequence[str]) -> str:
    """`a, b and c`."""
    *rest, last = names
    return f"{', '.join(rest)} and {last}" if rest else last


# Rules of the field, resolve and validate stages are declared without a
# check; their code is in the field types, resolve.py and `validate`.

_declare("V-01", "schema is 1", "field")


@_rule("V-02", "server sets the hardware fields of one role, or server.roles")
def _one_placement(s: Scenario) -> Iterator[Issue]:
    server = s.server
    have = [name for name in HARDWARE_FIELDS if getattr(server, name) is not None]
    if server.roles is not None:
        if have:
            listed = ", ".join(f"server.{name}" for name in have)
            yield Issue(
                ("server", "roles"),
                f"{listed} {'is' if len(have) == 1 else 'are'} set too; with "
                "roles, each role sets its own hardware",
                hint="set them to null under server to drop inherited values",
            )
    elif not have:
        yield Issue(
            ("server",),
            f"set server.{_and(HARDWARE_FIELDS)} for one role, or server.roles "
            "for several",
        )
    elif len(have) < len(HARDWARE_FIELDS):
        missing = [name for name in HARDWARE_FIELDS if name not in have]
        yield Issue(
            ("server",),
            f"needs {_and(missing)} too; one role sets {_and(HARDWARE_FIELDS)}",
        )


@_rule("V-03", "server.roles has two or more roles")
def _several_roles(s: Scenario) -> Iterator[Issue]:
    if s.server.roles is not None and len(s.server.roles) < 2:
        yield Issue(
            ("server", "roles"),
            "has one role; set a single role's hardware on server instead",
        )


@_rule("V-04", "server.multihost is set exactly when a role has more than one host")
def _multihost(s: Scenario) -> Iterator[Issue]:
    roles = s.server.role_views()
    multi = [role for role in roles if role.hardware.hosts > 1]
    if multi and s.server.multihost is None:
        role = multi[0]
        yield Issue(
            (*role.path, "hosts"),
            f"is {role.hardware.hosts}; set server.multihost: mp or ray",
        )
    elif roles and not multi and s.server.multihost is not None:
        yield Issue(
            ("server", "multihost"),
            "is set but every role has one host; remove it",
        )


@_rule("V-42", "a role's topology has hosts x chips_per_host chips")
def _topology_chips(s: Scenario) -> Iterator[Issue]:
    for role in s.server.role_views():
        hardware = role.hardware
        chips = math.prod(int(size) for size in hardware.topology.split("x"))
        if chips != hardware.chips:
            yield Issue(
                (*role.path, "topology"),
                f"topology {hardware.topology} has {chips} chips, but hosts "
                f"{hardware.hosts} x chips_per_host {hardware.chips_per_host} = "
                f"{hardware.chips}",
            )


@_rule("V-05", "server.router only with server.roles")
def _router_needs_roles(s: Scenario) -> Iterator[Issue]:
    if s.server.router is not None and s.server.roles is None:
        yield Issue(("server", "router"), "needs server.roles")


@_rule("V-06", "router toy-proxy fronts one producer and one consumer role")
def _router_roles(s: Scenario) -> Iterator[Issue]:
    if s.server.router is None or s.server.roles is None:
        return
    kv = {name: role.kv.role for name, role in s.server.roles.items() if role.kv}
    producers = [name for name, role in kv.items() if role == "producer"]
    consumers = [name for name, role in kv.items() if role == "consumer"]
    if len(producers) != 1 or len(consumers) != 1:
        yield Issue(
            ("server", "router", "type"),
            "toy-proxy needs one producer and one consumer role; found "
            f"producers [{', '.join(producers)}] and consumers "
            f"[{', '.join(consumers)}]",
        )


# Kv.connector allows one connector, so the type enforces this today.
_declare("V-07", "every kv block uses the same connector", "field")


@_rule("V-08", "tasks.driver names a role or client, and is required with roles")
def _driver(s: Scenario) -> Iterator[Issue]:
    driver = s.tasks.driver
    if s.server.roles is not None:
        choices = [*s.server.roles, "client"]
        if driver is None:
            yield Issue(
                ("tasks",),
                "tasks.driver is required with server.roles (one of: "
                f"{', '.join(choices)})",
            )
        elif driver not in choices:
            yield Issue(
                ("tasks", "driver"),
                f"{driver} is not a role; one of: {', '.join(choices)}",
            )
    elif driver not in (None, "server"):
        yield Issue(
            ("tasks", "driver"),
            "a scenario without roles runs tasks on its first host; remove "
            "tasks.driver",
        )


STREAMING_LOADERS = ("runai_streamer", "runai_streamer_sharded")


@_rule("V-09", "an object-storage model path has a loader that reads it")
def _object_storage(s: Scenario) -> Iterator[Issue]:
    if not s.server.model.in_object_storage:
        return
    for role in s.server.role_views():
        if role.args.get("load-format") not in STREAMING_LOADERS:
            scope = f" for role {role.name}" if s.server.roles else ""
            yield Issue(
                ("server", "model", "path"),
                f"is object storage; set load-format: runai_streamer{scope} (no "
                "other loader reads object storage)",
            )


@_rule("V-10", "distributed-executor-backend agrees with server.multihost")
def _executor_backend(s: Scenario) -> Iterator[Issue]:
    multihost = s.server.multihost
    for role in s.server.role_views():
        backend = role.args.get("distributed-executor-backend")
        if (
            backend is not None
            and multihost is not None
            and role.hardware.hosts > 1
            and backend != multihost
        ):
            yield Issue(
                role.arg_path("distributed-executor-backend"),
                f"is {backend} but server.multihost is {multihost}",
            )


def _env_blocks(s: Scenario) -> Iterator[tuple[KeyPath, Mapping[str, str | None], str]]:
    yield ("server", "env"), s.server.env, "server"
    for name, role in (s.server.roles or {}).items():
        yield ("server", "roles", name, "env"), role.env, "server"
    tasks = s.tasks
    if tasks.perf is not None:
        yield ("tasks", "perf", "env"), tasks.perf.env, f"perf:{tasks.perf.client}"
    if tasks.agentx is not None:
        yield ("tasks", "agentx", "env"), tasks.agentx.env, "agentx"
    if tasks.eval is not None and tasks.eval.client == "inferencex":
        yield ("tasks", "eval", "env"), tasks.eval.env, "eval:inferencex"
    elif tasks.eval is not None and tasks.eval.lm_eval is not None:
        for index, item in enumerate(tasks.eval.lm_eval.tasks):
            if isinstance(item, LmEvalTask):
                path = ("tasks", "eval", "lm_eval", "tasks", index, "env")
                yield path, item.env, "task"
    for index, task in enumerate(tasks.custom):
        yield ("tasks", "custom", index, "env"), task.env, "task"


@_rule("V-11", "no env block sets a name the tool owns")
def _tool_env(s: Scenario) -> Iterator[Issue]:
    chipless = s.server.roles is not None and s.tasks.driver == "client"
    for path, env, kind in _env_blocks(s):
        for name, value in env.items():
            where = (*path, name)
            if value is None:
                continue
            if name == READY_TIMEOUT_ENV:
                yield Issue(
                    where,
                    f"{name} is set from server.startup_timeout; remove it",
                    hint="set server.startup_timeout instead (now "
                    f"{format_duration(s.server.startup_timeout)})",
                    rule="V-12",
                )
            elif owner := tool_env_owner(name):
                yield Issue(where, f"{name} is set by the tool ({owner}); remove it")
            elif kind == "perf:inferencex" and name in INFERENCEX_PERF_ENV:
                yield Issue(
                    where,
                    f"{name} is set by the tool for the inferencex client "
                    f"({INFERENCEX_PERF_ENV[name]}); remove it",
                )
            elif kind == "agentx" and name in AGENTX_CLIENT_ENV:
                yield Issue(
                    where,
                    f"{name} is set by the tool from {AGENTX_CLIENT_ENV[name]}; "
                    "remove it",
                )
            elif kind == "eval:inferencex" and name in INFERENCEX_EVAL_ENV:
                yield Issue(
                    where,
                    f"{name} is set by the tool from {INFERENCEX_EVAL_ENV[name]}; "
                    "remove it",
                )
            elif kind != "server" and chipless and name in CHIPLESS_DRIVER_ENV:
                yield Issue(
                    where,
                    f"{name} is set by the tool on a client driver "
                    f"({CHIPLESS_DRIVER_ENV[name]}); remove it",
                )


_declare(
    "V-12", f"no env block sets {READY_TIMEOUT_ENV} (use startup_timeout)", "scenario"
)


@_rule("V-13", "engines on the ray backend get no JAX_PLATFORMS")
def _ray_env(s: Scenario) -> Iterator[Issue]:
    if s.server.multihost != "ray":
        return
    reported = set()
    for role in s.server.role_views():
        if role.hardware.hosts == 1:
            continue
        for name, owner in RAY_ENV.items():
            path = role.env_path(name)
            if name in role.env and path not in reported:
                reported.add(path)
                yield Issue(path, f"{name} is set by {owner}; remove it")


@_rule("V-14", "free-form args do not set flags the tool renders")
def _reserved_args(s: Scenario) -> Iterator[Issue]:
    server = s.server
    blocks: list[tuple[KeyPath, Mapping[str, object], Mapping[str, str]]] = [
        (("server", "args"), server.args, SERVE_ARGS)
    ]
    for name, role in (server.roles or {}).items():
        blocks.append((("server", "roles", name, "args"), role.args, SERVE_ARGS))
    if server.router is not None:
        blocks.append((("server", "router", "args"), server.router.args, ROUTER_ARGS))
    if s.tasks.perf is not None:
        path = ("tasks", "perf", "client_args")
        blocks.append((path, s.tasks.perf.client_args, PERF_CLIENT_ARGS))
    for path, args, reserved in blocks:
        for key in args:
            if reserved is SERVE_ARGS and key in DP_HOST_ARGS:
                yield Issue(
                    (*path, key),
                    f"srt-slurm style: do not set {key}; the tool derives it from "
                    "data-parallel-size and hosts",
                )
            elif key in reserved:
                yield Issue((*path, key), f"is set by {reserved[key]}; remove it")


_declare("V-15", "arg keys are flag names without dashes", "field")


@_rule("V-17", "data-parallel-size is a positive integer, split evenly over mp hosts")
def _mp_split(s: Scenario) -> Iterator[Issue]:
    for role in s.server.role_views():
        dp = _int_arg(role, "data-parallel-size")
        if dp is None:
            yield Issue(
                role.arg_path("data-parallel-size"),
                f"must be a positive integer; got {role.args['data-parallel-size']!r}",
            )
            continue
        hosts = role.hardware.hosts
        if s.server.multihost == "mp" and hosts > 1 and dp % hosts:
            yield Issue(
                role.arg_path("data-parallel-size"),
                f"multihost mp needs data-parallel-size divisible by {hosts} "
                f"hosts; got {dp}",
            )


def rule_table() -> list[Rule]:
    """Every rule, in id order."""
    return sorted(RULES.values(), key=lambda rule: int(rule.id[2:]))
