"""Does the kLoop fusion v12 deletes sit INSIDE the program probe-18's
`P3.prefillX` arm times, or upstream of it?

No pod time. AOT-compiles the EXACT jitted program probe 18 registered for its
`P3.*` arms

    jax.jit(lambda s, l: sparsecore_topk(s.reshape(b, n), k=k, row_lengths=l))
    s : f32[b, n // 128, 128]     <- the production 3-D entry
    l : i32[b]

against two sources -- the pinned probe-18 subject and v12's file -- and reads
the optimized ENTRY back.  The `P2.*` (2-D entry, `f32[b, n]`) program is
compiled too, because that is the layout v12's own dumps used and the only way
to tell a v12 deletion apart from a probe-18 entry-layout artifact.

Counting rule (the trap this probe exists to avoid): **match opcodes, not
substrings.**  `%bitcast-convert_bitcast_fusion` is an instruction whose *name*
contains "bitcast-convert" and whose *opcode* is `fusion`; `%bitcast.3` is a
free reinterpretation and `%bitcast_convert_type.3` is a materialized
TensorCore op.  Every count below is parsed from the opcode position

    [ROOT] %<name> = <shape> <opcode>(<operands>)

and a materialization test is applied separately: an instruction is *priced* if
its backend_config carries a `window_config` with a nonzero `estimated_cycles`,
which a free `bitcast` never has.

Usage:
    uv_venv/bin/python aot_entry.py            # all four cells
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
import sys

import jax
import jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info as _tpu_info

_T = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
_tpu_info.get_tpu_info = lambda: _T
_tpu_info.is_tpu_device = lambda: True
_tpu_info.get_device_kind = lambda: "TPU7x"
_tpu_info.get_num_device_cores = lambda: 1

from jax.experimental import topologies  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import SingleDeviceSharding  # noqa: E402

pltpu.get_tpu_info = lambda: _T

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUBJECT = ROOT / "probes/18-rho-denominator/subject/sparsecore_topk.py"
V12 = ROOT / "variants/v12-f32-input/sparsecore_topk.py"

SUBJECT_MD5 = "44353b347c96e0eedbd742c710d4540a"
V12_MD5 = "408ef72e9afc935c928f2803764878cc"

B, N, K = 1024, 10240, 2048          # probe 18's prefill shape at --ctx 9216
BD = 64                               # probe 18's decode shape

INSTR = re.compile(r"^\s*(?:ROOT\s+)?%([\w\-.]+)\s*=\s*(.+?)\s+"
                   r"([a-z][a-z0-9-]*)\((.*)$")


def load(path: pathlib.Path, want_md5: str, tag: str):
    got = hashlib.md5(path.read_bytes()).hexdigest()
    print(f"  source {tag:9s} {path.name}  md5={got}", end="")
    if got != want_md5:
        print(f"  MD5_GATE_FAIL want={want_md5}")
        sys.exit(1)
    print("  MD5_GATE_OK")
    spec = importlib.util.spec_from_file_location(f"arm_{tag}", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def parse_entry(txt: str):
    e = re.search(r"^ENTRY .*?^}", txt, re.S | re.M).group(0)
    rows = []
    for ln in e.splitlines()[1:-1]:
        m = INSTR.match(ln)
        if not m:
            continue
        name, shape, opcode, rest = m.groups()
        cyc = re.search(r'"estimated_cycles":"(\d+)"', ln)
        used = re.findall(r'"used_scoped_memory_configs":\[(.*?)\]', ln)
        used_b = 0
        if used and used[0]:
            used_b = max(int(x) for x in re.findall(r'"size":"(\d+)"',
                                                    used[0]))
        rows.append({
            "name": name,
            "opcode": opcode,
            "shape": shape,
            "priced": bool(cyc) and int(cyc.group(1)) > 0,
            "cycles": int(cyc.group(1)) if cyc else 0,
            "used_scoped_bytes": used_b,
            "kind": (re.search(r"kind=(\w+)", ln).group(1)
                     if "kind=" in ln else ""),
        })
    return e, rows


def cell(tag, module, entry_rank, b):
    """entry_rank 3 -> f32[b, n//128, 128] with a reshape; 2 -> f32[b, n]."""
    dev = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                       platform="tpu").devices[0]
    sh = SingleDeviceSharding(dev)
    if entry_rank == 3:
        s = jax.ShapeDtypeStruct((b, N // 128, 128), jnp.float32, sharding=sh)
        fn = lambda a, l: module.sparsecore_topk(a.reshape(b, N), K,
                                                 row_lengths=l)
    else:
        s = jax.ShapeDtypeStruct((b, N), jnp.float32, sharding=sh)
        fn = lambda a, l: module.sparsecore_topk(a, K, row_lengths=l)
    l = jax.ShapeDtypeStruct((b,), jnp.int32, sharding=sh)
    c = jax.jit(fn, out_shardings=sh).lower(s, l).compile()
    e, rows = parse_entry(c.as_text())
    ma = c.memory_analysis()
    return {
        "cell": tag,
        "entry_rank": entry_rank,
        "b": b,
        "entry_text": e,
        "rows": rows,
        "temp_bytes": getattr(ma, "temp_size_in_bytes", None),
        "arg_bytes": getattr(ma, "argument_size_in_bytes", None),
        "out_bytes": getattr(ma, "output_size_in_bytes", None),
    }


def show(r):
    print(f"\n--- {r['cell']}  ({r['entry_rank']}-D entry, b={r['b']}) ---")
    print(f"    temp={r['temp_bytes']}  arg={r['arg_bytes']}  "
          f"out={r['out_bytes']}")
    hist = {}
    for row in r["rows"]:
        hist[row["opcode"]] = hist.get(row["opcode"], 0) + 1
    print("    opcode histogram:",
          "  ".join(f"{k}={v}" for k, v in sorted(hist.items())))
    print("    priced instructions (window_config estimated_cycles > 0):")
    any_priced = False
    for row in r["rows"]:
        if row["priced"]:
            any_priced = True
            print(f"      opcode={row['opcode']:16s} kind={row['kind']:8s} "
                  f"cycles={row['cycles']:>8d} "
                  f"scoped={row['used_scoped_bytes']:>10d}  "
                  f"shape={row['shape']}")
            print(f"        name=%{row['name']}")
    if not any_priced:
        print("      (none)")
    print("    free/unpriced:", ", ".join(
        f"{row['opcode']}" for row in r["rows"] if not row["priced"]))


def main():
    print("AOT, no TPU.  jax", jax.__version__)
    inc = load(SUBJECT, SUBJECT_MD5, "incumbent")
    v12 = load(V12, V12_MD5, "v12")
    assert inc is not v12
    mods = [p for p, m in sys.modules.items()
            if getattr(m, "__file__", None)
            and pathlib.Path(m.__file__).name == "sparsecore_topk.py"]
    print("  resident sparsecore_topk modules:", mods)

    out = []
    for tag, mod in (("incumbent", inc), ("v12", v12)):
        for rank, b in ((3, B), (2, B), (3, BD)):
            r = cell(f"{tag}/{rank}d/b{b}", mod, rank, b)
            show(r)
            out.append(r)

    print("\n" + "=" * 74)
    print("ADJUDICATION")
    print("=" * 74)
    by = {r["cell"]: r for r in out}

    def priced(cellname):
        return [row for row in by[cellname]["rows"] if row["priced"]]

    for rank, b in ((3, B), (2, B), (3, BD)):
        a = f"incumbent/{rank}d/b{b}"
        c = f"v12/{rank}d/b{b}"
        pa, pc = priced(a), priced(c)
        cyc_a = sum(x["cycles"] for x in pa)
        cyc_c = sum(x["cycles"] for x in pc)
        print(f"\n  {rank}-D entry, b={b}:")
        print(f"    incumbent priced ops: {len(pa)}  total cycles {cyc_a}")
        for x in pa:
            print(f"        {x['opcode']} kind={x['kind'] or '-'} "
                  f"{x['cycles']} cyc  {x['shape']}")
        print(f"    v12       priced ops: {len(pc)}  total cycles {cyc_c}")
        for x in pc:
            print(f"        {x['opcode']} kind={x['kind'] or '-'} "
                  f"{x['cycles']} cyc  {x['shape']}")
        verdict = ("INSIDE  -- v12 deletes a priced op that the incumbent "
                   f"{rank}-D program contains"
                   if len(pa) > len(pc) else
                   "NOT VISIBLE -- no priced op is deleted at this entry rank")
        print(f"    -> {verdict}")

    (HERE / "aot_entry.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "entry_text"} for r in out],
        indent=1))
    for r in out:
        (HERE / f"entry-{r['cell'].replace('/', '-')}.txt").write_text(
            r["entry_text"] + "\n")
    print("\nwrote aot_entry.json and per-cell entry-*.txt")


if __name__ == "__main__":
    main()
