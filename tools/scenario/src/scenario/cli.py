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
"""The `scenario` command line.

Exit codes: 0 when everything is fine (warnings allowed), 1 when a scenario
has problems, 2 for usage errors, including a Plan that cannot be selected
or decoded.
"""

import pathlib
import sys
from collections.abc import Sequence
from typing import NoReturn

import click

from scenario import PLAN_FORMAT, __version__, gen
from scenario.errors import DiagnosticError
from scenario.keycheck import check_keys
from scenario.model.scenario import Scenario
from scenario.overrides import sparse_layer
from scenario.paths import PathSyntaxError, parse_path
from scenario.plan import codec
from scenario.plan.build import PlanError, plan, select_point
from scenario.plan.types import Plan
from scenario.resolve import Resolved, Resolver, find_root
from scenario.show import render_plan, render_resolved, render_why

EXIT_PROBLEMS = 1
EXIT_USAGE = 2

FileArg = click.Path(path_type=pathlib.Path, dir_okay=False)
RootOption = click.option(
    "--root",
    type=click.Path(path_type=pathlib.Path, file_okay=False, exists=True),
    help="The directory holding scenarios/defaults.yaml; found from the file "
    "or the current directory when not given.",
)
SetOption = click.option(
    "--set",
    "sets",
    multiple=True,
    metavar="PATH=VALUE",
    help="Override one value, e.g. server.args.max-model-len=4096; repeatable. "
    "The run becomes an experiment and skips its gates.",
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(
    __version__,
    "--version",
    prog_name="scenario",
    message=f"%(prog)s %(version)s (plan format {PLAN_FORMAT})",
)
def main() -> None:
    """Validate, show, render and run vLLM server scenarios."""


def _fail(message: str, code: int = EXIT_USAGE) -> NoReturn:
    click.echo(f"error: {message}", err=True)
    sys.exit(code)


def _resolve(
    file: pathlib.Path,
    sets: Sequence[str],
    root: pathlib.Path | None,
    point: str | None = None,
) -> tuple[Resolved, ...]:
    """Resolve FILE, or print every problem and exit 1."""
    try:
        resolver = Resolver(find_root(file, root, cwd=pathlib.Path.cwd()))
        resolved = resolver.resolve(file.resolve(), tuple(sets), point)
    except DiagnosticError as error:
        click.echo("\n".join(d.render() for d in error.diagnostics), err=True)
        sys.exit(EXIT_PROBLEMS)
    for warning in resolved[0].warnings if resolved else ():
        click.echo(warning.render(), err=True)
    return resolved


def _plan(points: Sequence[Resolved], point: str | None, only: str | None) -> Plan:
    try:
        tasks = [name.strip() for name in only.split(",")] if only else None
        return plan(select_point(points, point), only=tasks)
    except PlanError as error:
        _fail(str(error))


# show -------------------------------------------------------------------


def _key_path(key: str) -> tuple[str | int, ...]:
    try:
        path = parse_path(key)
    except PathSyntaxError as error:
        _fail(str(error))
    unknown = check_keys(sparse_layer(path, 0), Scenario)
    if unknown:
        _fail(f"{key}: {unknown[0].message}", EXIT_PROBLEMS)
    return path


def _read_encoded(value: str) -> Plan:
    text = sys.stdin.read() if value == "-" else value
    try:
        return codec.decode(text)
    except codec.PlanFormatError as error:
        _fail(str(error))


@main.command()
@click.argument("file", required=False, type=FileArg)
@SetOption
@click.option("--point", help="The sweep point to show.")
@click.option("--plan", "show_plan", is_flag=True, help="Show the Plan of a point.")
@click.option("--why", metavar="KEY", help="Show every layer that set KEY.")
@click.option(
    "--encoded",
    metavar="VALUE",
    help="Show the Plan in a pod's SCENARIO_PLAN value ('-' reads stdin).",
)
@RootOption
def show(
    file: pathlib.Path | None,
    sets: tuple[str, ...],
    point: str | None,
    show_plan: bool,
    why: str | None,
    encoded: str | None,
    root: pathlib.Path | None,
) -> None:
    """Show a scenario's resolved values and where each came from.

    --plan shows what every host runs; --why KEY the layers that set KEY;
    --encoded the Plan a pod received.
    """
    if encoded is not None:
        if file is not None or sets or show_plan or why or point:
            _fail("--encoded takes no scenario file and no other option")
        click.echo(render_plan(_read_encoded(encoded)), nl=False)
        return
    if file is None:
        _fail("show needs a scenario FILE, or --encoded VALUE")
    if show_plan and why:
        _fail("pass --plan or --why, not both")
    path = _key_path(why) if why else None
    points = _resolve(file, sets, root, point)
    if show_plan:
        click.echo(render_plan(_plan(points, point, None)), nl=False)
        return
    texts = []
    for resolved in points:
        if path is None:
            texts.append(render_resolved(resolved))
        elif len(points) > 1:
            texts.append(f"point {resolved.point}:\n" + render_why(resolved, path))
        else:
            texts.append(render_why(resolved, path))
    click.echo("\n".join(texts), nl=False)


# dev --------------------------------------------------------------------


@main.group()
def dev() -> None:
    """Commands for working on the tool itself."""


@dev.command()
@click.option(
    "--check", is_flag=True, help="Write nothing; fail when a generated file is stale."
)
def regen(check: bool) -> None:
    """Regenerate the JSON schemas and REFERENCE.md from the models."""
    try:
        root = gen.tool_dir()
    except FileNotFoundError as error:
        _fail(str(error))
    if check:
        names = gen.stale(root)
        for name in names:
            click.echo(f"stale: tools/scenario/{name}; run: {gen.REGEN}", err=True)
        sys.exit(EXIT_PROBLEMS if names else 0)
    for name in gen.write(root):
        click.echo(f"wrote tools/scenario/{name}")
