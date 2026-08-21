#!/usr/bin/env python3
"""Analyse TPU serving profiles: step time, where it goes, and A/B comparison.

    analyze.py phases     CAPTURE...                  what is in a capture
    analyze.py steptime   CAPTURE...                  on-device step time
    analyze.py breakdown  CAPTURE                     where a step goes
    analyze.py diff       BASE CAND                   which ops moved
    analyze.py compare    --arm N=C,C,C --arm N=C,C   arms, with error bars

A CAPTURE is a directory of profiler output: one subdirectory per inference
phase, each holding ``rank<N>_*.xplane.pb``. A flat directory of traces also
works.

Every command accepts ``--ranks`` and every command should be given it when the
output will be compared against anything. Only about half the ranks emit a
device plane, which half is not stable between captures, and they differ by up
to 8% in step time -- so an unpinned comparison partly measures which ranks
happened to report. ``phases`` prints the rank set to pin to.

Run ``analyze.py <command> --help`` for the options of one command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from xprofkit import breakdown, compare, diff, steptime  # noqa: E402
from xprofkit.taxonomy import Taxonomy  # noqa: E402
from xprofkit.trace import (  # noqa: E402
    DEVICE_PLANE, load_capture, phase_dirs, rank_key, rank_of, shared_ranks,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ranks(value: str | None) -> set[str] | None:
    return {r.strip() for r in value.split(",") if r.strip()} if value else None


def _phases(value: str | None) -> set[str] | None:
    return {p.strip() for p in value.split(",") if p.strip()} if value else None


#: Directory a phase puts its traces in. Its presence is what tells a phase
#: directory (`prefill_heavy/plugins/profile/<timestamp>/rank0_*.xplane.pb`)
#: apart from a directory of captures, whose children are phases in turn.
TRACE_SUBDIR = "plugins"


def _captures_below(path: str) -> list[str]:
    """Subdirectories of *path* that are whole captures rather than phases.

    Pointing at the directory *holding* the captures is the easy mistake -- it
    is what tab-completion lands on -- and it does not fail. Every capture
    below it is read as one "phase", so a run's prefill and decode phases are
    averaged into a single number per capture and the report looks ordinary
    while measuring nothing. The tell is one level down: a phase directory
    keeps its traces in ``plugins/``, so a candidate phase that has *named*
    phases of its own is a capture.
    """
    below = []
    for name in phase_dirs(path):
        # `.` is a flat capture, and `plugins` means *path itself* is a phase
        # directory -- its traces are below it, not in phases of its own.
        if name in (".", TRACE_SUBDIR):
            continue
        inner = phase_dirs(os.path.join(path, name))
        if [d for d in inner if d != TRACE_SUBDIR]:
            below.append(name)
    return below


def _load(paths: list[str], args, labels: list[str] | None = None) -> list:
    if labels and len(labels) != len(paths):
        raise SystemExit(
            f"got {len(labels)} --label for {len(paths)} captures; pass one "
            "per capture, in order, or none at all")
    labels = labels or [os.path.basename(os.path.normpath(p)) for p in paths]
    out = []
    for label, path in zip(labels, paths):
        if not os.path.isdir(path):
            raise SystemExit(f"not a directory: {path}")
        # Skipped when phases were named explicitly: that is the escape hatch
        # for a layout this heuristic reads wrongly, and naming a phase means
        # the caller knows what the directory holds.
        below = ([] if getattr(args, "phase", None) else _captures_below(path))
        if below:
            listed = " ".join(os.path.join(path, name) for name in below[:3])
            raise SystemExit(
                f"{path} looks like a directory of captures, not a capture: "
                f"{', '.join(below[:6])}{' ...' if len(below) > 6 else ''}.\n"
                f"Each of those has phases of its own, and reading them as "
                f"phases would average a whole run into one row.\n"
                f"Point at one capture (or pass several): {listed}")
        capture = load_capture(
            path, _ranks(args.ranks), label=label,
            phases=_phases(getattr(args, "phase", None)),
            use_cache=not args.no_cache)
        if not capture:
            print(f"no step data from {_why_empty(path, args)}", file=sys.stderr)
        out.append(capture)
    return out


def _why_empty(path: str, args) -> str:
    """Why a capture directory produced nothing, in the user's terms.

    "No readable device plane" is true of a mistyped ``--ranks``, of a
    directory one level above the captures, and of a genuinely empty trace
    alike, and a first-time user has no way to tell which they hit. Say which
    of the three it was.
    """
    traces = list(Path(path).glob("**/*.xplane.pb"))
    if not traces:
        subdirs = ", ".join(sorted(p.name for p in Path(path).iterdir()
                                   if p.is_dir())[:4]) or "none"
        return (f"{path}: no *.xplane.pb anywhere under it. Point at the "
                f"capture directory itself, not its parent (subdirectories: "
                f"{subdirs}).")
    found = sorted({rank_of(p.name) for p in traces}, key=rank_key)
    if args.ranks and not (_ranks(args.ranks) & set(found)):
        return (f"{path}: --ranks {args.ranks} matched none of the ranks that "
                f"reported ({','.join(found)}).")
    phases = getattr(args, "phase", None)
    if phases:
        return (f"{path}: --phase {phases} matched no phase directory "
                f"({', '.join(phase_dirs(path)) or 'none'}).")
    # Only the traces that were actually read. Counting all of them under a
    # `--ranks 0,2` that matched two of eight reports "8 trace(s) found, none
    # of which holds a device plane" -- which is a claim about six files
    # nothing opened, and sends the reader looking for a capture-wide fault.
    selected = _ranks(args.ranks)
    if selected:
        traces = [p for p in traces if rank_of(p.name) in selected]
    scope = "" if not selected else f" for rank(s) {','.join(sorted(selected, key=rank_key))}"
    return (f"{path}: {len(traces)} trace(s) found{scope}, none of which holds "
            f"a '{DEVICE_PLANE}' plane. A host-only capture looks like this.")


def _taxonomy(args) -> Taxonomy:
    """Load the taxonomy, reporting a bad file as a message rather than a stack.

    The taxonomy is the part users are expected to edit, so a typo in it is an
    ordinary event and should read like one.
    """
    try:
        return Taxonomy.load(args.taxonomy)
    except FileNotFoundError:
        raise SystemExit(f"no such taxonomy: {args.taxonomy}") from None
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(f"cannot load taxonomy {args.taxonomy}: {exc}") from None


def _emit(lines: list[str], payload: dict | None, args, *,
          problems: list[str] = ()) -> int:
    """Print the report, write the JSON, and return the process's exit status.

    ``problems`` is why the report does not answer the question that was asked
    -- an arm with no usable capture, a phase nothing measured. They are
    printed *and* they make the status non-zero, because the caller this tool
    is documented for is ``analyze.py compare ... && promote``: a command that
    prints diagnostics and exits 0 promotes on a mistyped ``--ranks``.
    """
    print("\n".join(lines))
    if getattr(args, "json", None) and payload is not None:
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {args.json}")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


def _unusable(captures: list) -> list[str]:
    """The captures that were named and produced nothing."""
    return [f"{c.label}: no phase with device data" for c in captures if not c]


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_phases(args) -> int:
    """Inventory: which phases exist, which ranks reported, how many steps.

    Cheap orientation before anything else, and the way to find the rank set to
    pass to ``--ranks`` for every later command.
    """
    from xprofkit.table import table, us

    captures = _load(args.captures, args)
    rows = []
    for cap in captures:
        if not cap:
            rows.append([cap.label, "(no device plane)", "", "", "", ""])
            continue
        for phase, data in cap.phases.items():
            rows.append([
                cap.label, phase, ",".join(data.rank_ids), len(data.ranks),
                data.steps, us(data.mean_over_ranks("mean_us")),
            ])
    lines = table(["capture", "phase", "ranks", "n", "steps", "mean step us"],
                  rows, widths=[24, 18, 16, 3, 6, 13])

    # The pin line is the point of this command, so it prints for one capture
    # as well as for several -- one capture is how everybody starts.
    common_phases = (set.intersection(*(set(c.phases) for c in captures))
                     if all(c.phases for c in captures) else set())
    pins = {phase: shared_ranks([c.phases[phase] for c in captures])
            for phase in sorted(common_phases)}
    lines.append("")
    if len(captures) > 1:
        lines.append(f"phases in every capture: "
                     f"{','.join(sorted(common_phases)) or '(none)'}")
    for phase, ranks in pins.items():
        lines.append(f"  {phase}: pin with --ranks {','.join(ranks) or '(none)'}")
    if not pins:
        lines.append("  no phase is present in every capture; nothing to pin")

    payload = {
        "captures": [
            {"label": c.label, "root": c.root,
             "phases": {p: {"ranks": d.rank_ids, "steps": d.steps,
                            "mean_us": d.mean_over_ranks("mean_us"),
                            "dropped": list(d.dropped)}
                        for p, d in c.phases.items()}}
            for c in captures
        ],
        "pin_ranks": pins,
    }
    return _emit(lines, payload, args, problems=_unusable(captures))


def cmd_steptime(args) -> int:
    """Step-time distribution per capture, plus capture-to-capture deltas."""
    captures = _load(args.captures, args, args.label)
    if not any(captures):
        return 1
    # Not `not any(...)`: with two captures named and one readable, the
    # capture-to-capture comparison this command exists for did not happen, and
    # the one table that did print looks exactly like a successful run.
    problems = _unusable(captures)
    estimators = (steptime.ESTIMATORS if args.estimator == "all"
                  else (args.estimator,))

    lines: list[str] = []
    for cap in captures:
        lines += steptime.render_capture(cap, components=args.components)
    comparison = {}
    if len(captures) > 1:
        comparison = steptime.compare(captures, estimators)
        lines += steptime.render_compare(comparison, estimators)
        lines += compare.render_spread(
            compare.spread(captures, args.estimator if args.estimator != "all"
                           else "mean"),
            args.estimator if args.estimator != "all" else "mean")
    if args.emit:
        # One capture keeps the bare keys a harness already greps for; several
        # get the label in front, because otherwise the last one silently wins.
        lines.append("")
        for cap in captures:
            lines += steptime.emit_metrics(
                cap, prefix="" if len(captures) == 1 else f"{cap.label}.")

    payload = {
        "captures": [{"label": c.label, "root": c.root,
                      "phases": steptime.summarise(c)} for c in captures],
        "comparison": comparison,
    }
    return _emit(lines, payload, args, problems=problems)


def cmd_breakdown(args) -> int:
    """Where one capture's step time goes, by category and op family."""
    taxonomy = _taxonomy(args)
    cap = _load([args.capture], args)[0]
    if not cap:
        return 1
    result = breakdown.analyse(cap, taxonomy)
    lines = [f"# {cap.label}", f"taxonomy: {taxonomy.name}"]
    lines += breakdown.render(result, taxonomy, families=args.families,
                              unclassified=args.unclassified, names=args.names,
                              sources=args.sources, units=args.units,
                              verbose=args.verbose)
    return _emit(lines, result, args)


def cmd_diff(args) -> int:
    """Op-level A/B of two captures."""
    taxonomy = _taxonomy(args)
    base, cand = _load([args.base, args.cand], args)
    if not base or not cand:
        return 1
    result = diff.diff(base, cand, taxonomy)
    if not result["phases"]:
        print("no phase was captured on both sides", file=sys.stderr)
        return 1
    lines = diff.render(result, ops=args.ops, families=args.families,
                        taxonomy=taxonomy)
    return _emit(lines, result, args)


def cmd_compare(args) -> int:
    """Arms of repeated captures, with confidence intervals."""
    arms: dict[str, list] = {}
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm wants NAME=dir[,dir...], got {spec!r}")
        name, _, paths = spec.partition("=")
        dirs = [p for p in paths.split(",") if p]
        if not dirs:
            raise SystemExit(f"--arm {name} has no capture directories")
        if name in arms:
            # Silently merging them would put a candidate's captures in the
            # baseline's arm and read as a much tighter baseline.
            raise SystemExit(f"--arm {name} given twice; arm names must differ")
        arms[name] = _load(
            dirs, args, [f"{name}/{os.path.basename(os.path.normpath(p))}"
                         for p in dirs])
    if len(arms) < 2:
        raise SystemExit("compare needs at least two --arm")

    result = compare.compare_arms(arms, args.estimator)
    lines = [f"# compare ({args.estimator} on-device step time)"]
    lines += compare.render(result)
    lines += compare.verdict(result)
    return _emit(lines, result, args, problems=_compare_problems(arms, result))


def _compare_problems(arms: dict[str, list], result: dict) -> list[str]:
    """Why this comparison did not compare anything.

    ``compare`` is the command the README puts behind a promotion gate, and it
    was the one command that could not fail: the arm check counted arm *names*,
    which a mistyped ``--ranks``, a renamed phase or a capture that never
    landed all pass, and every one of them printed two diagnostics, an empty
    verdict and exit 0. ``diff``, ``breakdown`` and ``steptime`` all exit 1 on
    the equivalent input.
    """
    problems = [f"arm {name!r}: no capture with usable step data"
                for name, captures in arms.items() if not any(captures)]
    compared = [phase for phase, data in result.items()
                if "skipped" not in data
                and any(arm["vs_ref"] for arm in data["arms"].values())]
    if not compared:
        skipped = "; ".join(f"{phase}: {data['skipped']}"
                            for phase, data in result.items() if "skipped" in data)
        problems.append("no phase produced a comparison"
                        + (f" ({skipped})" if skipped else ""))
    return problems


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="analyze.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def common(parser, *, ranks=True, phase=True, taxonomy=False):
        if ranks:
            parser.add_argument(
                "--ranks", metavar="0,2,4",
                help="only these ranks. Pass it whenever the output will be "
                     "compared; `phases` prints the set to use.")
        if phase:
            parser.add_argument(
                "--phase", metavar="NAME[,NAME]",
                help="only these phases (default: all). Each phase costs a few "
                     "seconds per rank on a cold cache.")
        if taxonomy:
            parser.add_argument(
                "--taxonomy", metavar="FILE",
                help="op classification rules (default: taxonomies/default.yaml)")
        parser.add_argument("--json", metavar="FILE", help="also write the full result here")
        parser.add_argument("--no-cache", action="store_true",
                            help="re-parse the protobufs instead of using .xprofkit/")

    p = sub.add_parser("phases", help="what is in a capture",
                       description=cmd_phases.__doc__)
    p.add_argument("captures", nargs="+")
    common(p)
    p.set_defaults(func=cmd_phases)

    p = sub.add_parser("steptime", help="on-device step time",
                       description=steptime.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("captures", nargs="+")
    p.add_argument("--label", action="append",
                   help="name for each capture, in order (default: directory name)")
    p.add_argument("--estimator", choices=(*steptime.ESTIMATORS, "all"),
                   default="mean",
                   help="which location estimator to compare on (default: mean)")
    p.add_argument("--components", action="store_true",
                   help="also show summed op time and async time beside the step")
    p.add_argument("--emit", action="store_true",
                   help="print `key: value` lines for a harness to harvest")
    common(p)
    p.set_defaults(func=cmd_steptime)

    p = sub.add_parser("breakdown", help="where a step goes",
                       description=breakdown.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("capture")
    p.add_argument("--families", type=int, default=0, metavar="N",
                   help="also list the top N op families (default: 0, off)")
    p.add_argument("--unclassified", type=int, default=8, metavar="N",
                   help="how many unclassified ops to name (default: 8)")
    p.add_argument("--sources", type=int, default=0, metavar="N",
                   help="also print the top N source lines by device time")
    p.add_argument("--names", type=int, default=3, metavar="N",
                   help="how many op families to name per category, as many as "
                        "fit on the line (default: 3)")
    p.add_argument("--units", action="store_true",
                   help="also split the same total by execution unit "
                        "(TensorCore / SparseCore)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="follow every table with the full list of op families "
                        "behind each of its rows, nothing folded")
    common(p, taxonomy=True)
    p.set_defaults(func=cmd_breakdown)

    p = sub.add_parser("diff", help="which ops moved between two captures",
                       description=diff.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base")
    p.add_argument("cand")
    p.add_argument("--families", type=int, default=20, metavar="N",
                   help="how many op-family rows to show (default: 20). This "
                        "is the table to read.")
    p.add_argument("--ops", type=int, default=25, metavar="N",
                   help="how many exact-op rows to show (default: 25). Names "
                        "carry build-local suffixes; for grepping a module.")
    common(p, taxonomy=True)
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("compare", help="arms of repeated captures, with error bars",
                       description=compare.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=D1,D2,D3",
                   help="a named set of captures of one tree. Repeat for each "
                        "arm; the first is the reference.")
    p.add_argument("--estimator", choices=steptime.ESTIMATORS, default="mean")
    common(p)
    p.set_defaults(func=cmd_compare)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
