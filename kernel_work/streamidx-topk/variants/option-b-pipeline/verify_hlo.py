#!/usr/bin/env python3
"""Static verification that the stage-A -> stage-B dependency survived into the
compiled, *scheduled* HLO, and that the TC/SC overlap the Flip-3 timings
attribute to Option B is actually present in the schedule.

Runs offline on the `*.hlo.txt` files `probe_flip3.py --arm hlo` writes to
$CDK_OUTPUT_DIR/hlo/.  It needs no TPU: everything here is compile-time
structure, NOT a measurement.  Nothing this script prints is a timing.

Why this exists separately from the `--arm hlo` checks inside `probe_flip3.py`:
those checks looked for stage B among instructions whose *opcode* is
`custom-call`.  That is wrong.  In the compiled module the SparseCore top-k is
an async call into an `execution_thread="sparsecore"` computation:

    %sc_topk_direct_b32_... .call-start = (...) call-start(%bitcast.37, ...),
        async_execution_thread="sparsecore", to_apply=%called_computation
    %sc_topk_direct_b32_... .call-done  = s32[65536] call-done(%... .call-start)

so the opcode is `call-start` / `call-done`, and the only `custom-call`s bearing
the kernel's name are (a) the ROOT inside the callee, which takes *parameters*
and is therefore unreachable from anything in the entry computation, and (b) the
`PrepareAsyncCallStart` / `PrepareAsyncCallDone` pair, which has no data
operands at all.  The in-job checker consequently reported
`every_B_depends_on_an_A=False` on a module where the dependency is present and
load-bearing.  It was a bug in the checker, not a finding about the module.

The four questions this answers, per (phase, G):

  1. does every stage-B call-start reach a stage-A custom-call through the
     operand DAG?  (the dependency is real -- and it is a real one, not a
     zero-valued trick: stage B's data operand *is* stage A's output buffer,
     via layout-only bitcasts)
  2. which stage A does each stage B depend on?  (correct pairing: B_i <- A_i)
  3. how many *other* chunks' stage-A kernels are scheduled strictly between
     B_i.call-start and B_i.call-done?  This is the overlap, read off the
     schedule.  > 0 means the TensorCore has work in flight while the
     SparseCore call is outstanding.
  4. does copy r+1's stage A depend on *every* one of copy r's stage-B
     call-dones?  If not, the chain-slope fit is measuring cross-copy overlap
     as well and the per-copy slope is not the per-copy device time.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

# `%name = shape opcode(operands), attr=...`, optionally prefixed by ROOT.
_DEF = re.compile(r"^\s*(?:ROOT\s+)?(%[\w.\-]+)\s*=\s*(.*)$")
# The opcode is the identifier immediately before the operand list.  Shapes can
# contain parentheses of their own (tuple shapes), so anchor on the *first*
# `ident(` that follows a `]` / `)` / `}` / space -- shapes always end in one of
# those.
_OPCODE = re.compile(r"([a-z][\w\-]*)\(")
_OPERAND = re.compile(r"%[\w.\-]+")

STAGE_A_PREFIX = "%StreamIdxTC"
STAGE_B_KERNEL = "sc_topk_direct"


def _operand_span(rest: str, after: int) -> str:
    """Return the text of the balanced paren group starting at `after`."""
    depth, j = 1, after
    while j < len(rest) and depth > 0:
        if rest[j] == "(":
            depth += 1
        elif rest[j] == ")":
            depth -= 1
        j += 1
    return rest[after : j - 1]


def parse_entry(path: pathlib.Path) -> list[dict]:
    """Parse the ENTRY computation into schedule-ordered instructions.

    A scheduled HLO module prints each computation in schedule order, so the
    textual order *is* the schedule.  We assert the module is scheduled before
    relying on that.
    """
    text = path.read_text()
    if "is_scheduled=true" not in text:
        raise SystemExit(
            f"{path.name}: module is not scheduled; textual order is not a "
            "schedule and question 3 cannot be answered from it"
        )
    lines = text.splitlines()
    entries = [i for i, l in enumerate(lines) if l.startswith("ENTRY")]
    if not entries:
        raise SystemExit(f"{path.name}: no ENTRY computation")
    out = []
    for line in lines[entries[0] + 1 :]:
        if line.startswith("}"):
            break
        m = _DEF.match(line)
        if not m:
            continue
        name, rest = m.group(1), m.group(2)
        om = _OPCODE.search(rest)
        if not om:
            continue
        opcode = om.group(1)
        ops = _OPERAND.findall(_operand_span(rest, om.end()))
        out.append(dict(name=name, opcode=opcode, operands=ops, text=rest))
    return out


def analyse(path: pathlib.Path) -> dict:
    instrs = parse_entry(path)
    order = {ins["name"]: i for i, ins in enumerate(instrs)}
    by_name = {ins["name"]: ins for ins in instrs}

    stage_a = [
        ins["name"]
        for ins in instrs
        if ins["opcode"] == "custom-call" and ins["name"].startswith(STAGE_A_PREFIX)
    ]
    b_start = [
        ins["name"]
        for ins in instrs
        if ins["opcode"] == "call-start" and STAGE_B_KERNEL in ins["name"]
    ]
    b_done = [
        ins["name"]
        for ins in instrs
        if ins["opcode"] == "call-done" and STAGE_B_KERNEL in ins["name"]
    ]
    done_of = {}
    for d in b_done:
        src = by_name[d]["operands"]
        assert len(src) == 1, d
        done_of[src[0]] = d

    a_set = set(stage_a)

    def reaches(root: str, targets: set[str], skip: set[str] = frozenset()) -> set[str]:
        """Every member of `targets` transitively reachable from `root`.

        Traversal stops at `skip` so that "does A_{r+1} reach B_r" can be asked
        without the answer leaking through an even earlier copy.
        """
        seen, stack, hit = set(), [root], set()
        while stack:
            n = stack.pop()
            if n in seen or n not in by_name:
                continue
            seen.add(n)
            if n in targets:
                hit.add(n)
            if n in skip:
                continue
            stack.extend(by_name[n]["operands"])
        return hit

    # --- Q1/Q2: which stage A does each stage B depend on? -------------------
    pairing = {}
    # The *data* operand of the call-start (operand 0) is the one that should
    # carry stage A's scores; operand 1 is the eff-lengths vector and operand 2
    # is the prepare token.  Check both the whole instruction and operand 0, so
    # a dependency that arrived only via the token would be visible as a
    # difference between the two.
    pairing_via_data = {}
    for s in b_start:
        pairing[s] = sorted(reaches(s, a_set), key=lambda n: order[n])
        op0 = by_name[s]["operands"][0]
        pairing_via_data[s] = sorted(reaches(op0, a_set), key=lambda n: order[n])

    # --- how does the data get from A to B? ---------------------------------
    # Walk operand 0 of each call-start back to its stage A and record the
    # opcodes on the path.  If every op on that path is layout-only
    # (bitcast / bitcast-convert / reshape / copy) then stage B is reading
    # stage A's actual buffer and there is nothing for the compiler to fold.
    def path_to_a(start: str) -> tuple[str, list[str]] | tuple[None, None]:
        stack = [(start, [])]
        while stack:
            n, acc = stack.pop()
            if n in a_set:
                return n, acc
            if n not in by_name:
                continue
            ins = by_name[n]
            for o in ins["operands"]:
                stack.append((o, acc + [ins["opcode"]]))
        return None, None

    _walk = {s: path_to_a(by_name[s]["operands"][0]) for s in b_start}
    carrier_a = {s: v[0] for s, v in _walk.items()}
    carrier_paths = {s: v[1] for s, v in _walk.items()}
    LAYOUT_ONLY = {"bitcast", "bitcast-convert", "reshape", "copy", "transpose"}

    # --- Q3: stage-A kernels in flight during each stage-B async call --------
    windows = []
    for s in b_start:
        d = done_of.get(s)
        if d is None:
            windows.append(dict(start=s, done=None, a_in_window=[]))
            continue
        lo, hi = order[s], order[d]
        inside = [a for a in stage_a if lo < order[a] < hi]
        windows.append(
            dict(start=s, done=d, span_instrs=hi - lo, a_in_window=inside)
        )

    # --- Q4: does each copy's stage A depend on the previous copy's stage B? -
    # Copies are contiguous in the schedule: G stage-A's then G stage-B's, R
    # times over.  Recover G from the count.
    n_a, n_b = len(stage_a), len(b_start)
    g_r = n_a
    copies = None
    cross = None
    if n_a == n_b and n_a > 0:
        for R in (4, 3, 2, 1):
            if n_a % R == 0:
                pass
        # R is fixed by the caller (2 for these dumps); derive G from the
        # filename instead of guessing.
        m = re.search(r"_G(\d+)_R(\d+)", path.name)
        if m:
            G, R = int(m.group(1)), int(m.group(2))
            if G * R == n_a:
                copies = [
                    dict(
                        a=stage_a[r * G : (r + 1) * G],
                        b=b_start[r * G : (r + 1) * G],
                    )
                    for r in range(R)
                ]
                cross = []
                for r in range(1, R):
                    prev_done = {done_of[s] for s in copies[r - 1]["b"]}
                    first_a = copies[r]["a"][0]
                    hit = reaches(first_a, prev_done)
                    cross.append(
                        dict(
                            copy=r,
                            first_stage_a=first_a,
                            prev_stage_b_dones=len(prev_done),
                            reached=len(hit),
                            all_reached=hit == prev_done,
                        )
                    )
                g_r = G

    return dict(
        file=path.name,
        num_stage_a=n_a,
        num_stage_b_call_start=n_b,
        num_stage_b_call_done=len(b_done),
        every_b_reaches_an_a=all(bool(v) for v in pairing.values()),
        every_b_reaches_an_a_via_data_operand=all(
            bool(v) for v in pairing_via_data.values()
        ),
        # 1:1 is asked of the *direct* carrier -- the layout-only chain from the
        # call-start's data operand back to the buffer it reads.  The full
        # transitive set is deliberately not used here: copy r+1's stage B also
        # reaches copy r's stage A's through the inter-copy `eps` carrier, which
        # is by design (see `cross_copy` below) and is not a pairing violation.
        pairing_is_one_to_one=(
            all(v is not None for v in carrier_a.values())
            and len(set(carrier_a.values())) == n_b
        ),
        pairing_direct={k: v for k, v in carrier_a.items()},
        pairing_transitive={k: v for k, v in pairing_via_data.items()},
        carrier_path_opcodes={k: v for k, v in carrier_paths.items()},
        carrier_is_layout_only_everywhere=all(
            v is not None and set(v) <= LAYOUT_ONLY for v in carrier_paths.values()
        ),
        stage_a_in_flight_during_stage_b=[
            dict(start=w["start"], n_stage_a_in_window=len(w["a_in_window"]),
                 stage_a_in_window=w["a_in_window"])
            for w in windows
        ],
        max_stage_a_in_flight=max(
            (len(w["a_in_window"]) for w in windows), default=0
        ),
        cross_copy=cross,
        cross_copy_fully_serialized=(
            None if cross is None else all(c["all_reached"] for c in cross)
        ),
        G=g_r,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hlo_dir")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    d = pathlib.Path(args.hlo_dir)
    files = sorted(d.glob("*.hlo.txt"))
    if not files:
        print(f"no *.hlo.txt under {d}", file=sys.stderr)
        return 2

    results = {}
    ok = True
    for f in files:
        r = analyse(f)
        results[f.name] = r
        checks = {
            "B->A dependency survives (via data operand)":
                r["every_b_reaches_an_a_via_data_operand"],
            "one stage B per stage A, 1:1": r["pairing_is_one_to_one"],
            "carrier is layout-only (unfoldable, real buffer)":
                r["carrier_is_layout_only_everywhere"],
            "cross-copy fully serialized": r["cross_copy_fully_serialized"],
        }
        print(f"=== {f.name}  (A={r['num_stage_a']} "
              f"B.call-start={r['num_stage_b_call_start']} "
              f"B.call-done={r['num_stage_b_call_done']})")
        for label, v in checks.items():
            if v is None:
                print(f"    n/a   {label}")
                continue
            print(f"    {'PASS' if v else 'FAIL':5s} {label}")
            ok = ok and bool(v)
        n_in = [w["n_stage_a_in_window"]
                for w in r["stage_a_in_flight_during_stage_b"]]
        print(f"    stage-A kernels in flight per stage-B async window: {n_in}"
              f"  (max {r['max_stage_a_in_flight']})")
        # show one carrier path as evidence
        first = next(iter(r["carrier_path_opcodes"].items()))
        print(f"    carrier path {first[0]} -> stage A: {first[1]}")
        print()

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json_out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
