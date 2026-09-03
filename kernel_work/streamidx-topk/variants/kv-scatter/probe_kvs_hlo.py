#!/usr/bin/env python3
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
"""The pass condition: does the scatter's operand reshape become a `bitcast`?

Compiler only -- no profiler, no execution of the traced program.  Runs on the
image's pinned jax 0.10.2, so it cannot hit the `PLUGIN_Profiler_Api` SIGSEGV,
and it runs before the timing steps so that a trace that dies still leaves the
static half of the evidence behind.

Two modes.

`--mode layout` answers the prerequisite question, which is **not** the same as
the bitcast question: do `[415,256,4,256]` and `[415,256,8,128]` really hold the
same bytes in the same order?  Row-major equivalence is asserted elementwise
against the explicit index map `a[p,s,k,d] == b[p,s,2k+d//128,d%128]`, not by
comparing one `reshape` against another (which would be a tautology).  It also
prints the tiled layout XLA assigns each shape, which is what decides whether
the reshape can be a bitcast at all.

`--mode hlo` compiles each (config, arm) and writes the **optimized** HLO to
`<root>/hlo/<config>__<arm>.hlo.txt`, then prints a digest: every instruction
that could be a relayout of the whole cache (`reshape` / `bitcast` / `copy` /
`transpose` over >= 1e6 elements) and every instruction that scatters, with its
operands.  The files are written under `$CDK_OUTPUT_DIR` so `cdk job
sync-outputs` brings them back and the HLO can be quoted rather than
paraphrased.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import kvs_common as KC  # noqa: E402
import kvs_op as KO  # noqa: E402

# `%name = <dtype>[dims]{layout} opcode(operands...)`
_INSTR = re.compile(
    r"^\s*(?:ROOT\s+)?(%?[\w.\-]+)\s*=\s*"
    r"([a-z0-9]+\[[^\]]*\](?:\{[^}]*\})?(?:\s*,\s*[a-z0-9]+\[[^\]]*\]"
    r"(?:\{[^}]*\})?)*|\(\s*.*?\s*\))\s+"
    r"([a-z][\w-]*)\((.*)$")
_SHAPE = re.compile(r"([a-z0-9]+)\[([0-9,]*)\](\{[^}]*\})?")
RELAYOUT_OPS = ("reshape", "bitcast", "copy", "transpose", "bitcast-convert")
BIG = 1_000_000


def log(msg):
    print(f"[kvs-hlo] {msg}", flush=True)


def _elems(dims: str) -> int:
    if not dims:
        return 1
    n = 1
    for d in dims.split(","):
        n *= int(d)
    return n


def digest_hlo(text: str) -> dict:
    """Pull the relayout-sized instructions and the scatters out of an HLO."""
    defs: dict[str, str] = {}
    big: list[dict] = []
    scatters: list[dict] = []
    fusion_bodies: dict[str, list[str]] = {}
    cur_comp = None

    for line in text.splitlines():
        m_comp = re.match(r"^\s*(?:ENTRY\s+)?([\w.\-]+)\s*(?:\(|\{)", line)
        if m_comp and "= " not in line:
            cur_comp = m_comp.group(1)
        m = _INSTR.match(line)
        if not m:
            continue
        name, shape, opcode, rest = m.groups()
        name = name.lstrip("%")
        defs[name] = line.strip()
        fusion_bodies.setdefault(cur_comp or "?", []).append(opcode)
        sm = _SHAPE.match(shape)
        n = _elems(sm.group(2)) if sm else 0
        rec = {
            "name": name,
            "opcode": opcode,
            "shape": shape.strip(),
            "elems": n,
            "comp": cur_comp,
            "line": line.strip()[:400],
            "operands": [
                o.lstrip("%")
                for o in re.findall(r"%([\w.\-]+)", rest)
            ],
        }
        if opcode in RELAYOUT_OPS and n >= BIG:
            big.append(rec)
        if opcode == "scatter" or "scatter" in name:
            scatters.append(rec)

    # Attach each scatter's operand definitions, which is where the reshape /
    # bitcast question is actually decided.
    for s in scatters:
        s["operand_defs"] = [
            defs.get(o, f"<{o}: not an instruction in this module>")[:400]
            for o in s["operands"]
        ]
    return {
        "big_relayouts": big,
        "scatters": scatters,
        "n_big_relayouts": len(big),
        "big_relayout_bytes": sum(r["elems"] for r in big),
    }


def entry_param_layouts(text: str) -> list[str]:
    for line in text.splitlines():
        if line.startswith("ENTRY "):
            return [line.strip()[:1200]]
    return []


# ---------------------------------------------------------------------------
# --mode layout
# ---------------------------------------------------------------------------
def mode_layout(out_root: str) -> dict:
    res: dict = {"checks": {}}
    rows = KC.ROWS_PER_TOKEN
    lane = KC.LKV_DIM // rows

    # A small array with a byte pattern that makes every position identifiable
    # modulo 251 (prime, so no accidental periodicity with 128 or 256).
    p, s = 3, 5
    a_np = (np.arange(p * s * KC.KV_PACKING * KC.LKV_DIM, dtype=np.int64) %
            251).astype(np.uint8).reshape(p, s, KC.KV_PACKING, KC.LKV_DIM)
    b_np = a_np.reshape(p, s, KC.KV_PACKING * rows, lane)

    # The claim, stated as an index map rather than as reshape-vs-reshape.
    ok_map = True
    for k in range(KC.KV_PACKING):
        for d in range(KC.LKV_DIM):
            if a_np[:, :, k, d].tolist() != b_np[:, :, rows * k + d // lane,
                                                 d % lane].tolist():
                ok_map = False
    res["checks"]["index_map_a_k_d_equals_b_2k_plus_d_div_128"] = ok_map

    # The same reshape performed on device, compared with the host's.
    a_dev = jnp.asarray(a_np)
    b_dev = np.asarray(jax.jit(lambda x: x.reshape(b_np.shape))(a_dev))
    res["checks"]["device_reshape_matches_numpy"] = bool(
        np.array_equal(b_np, b_dev))

    # And the inverse, which is what the correctness readback relies on.
    a_back = np.asarray(
        jax.jit(lambda x: x.reshape(a_np.shape))(jnp.asarray(b_np)))
    res["checks"]["device_reshape_round_trips"] = bool(
        np.array_equal(a_np, a_back))

    # The tiled layout XLA assigns each production-sized shape.  This is the
    # thing that decides whether the flatten to 2-D can be a bitcast: a minor
    # dim of 256 does not fill one 128-lane tile, and the second-minor dim
    # picks the tile height.
    layouts = {}
    for tag, shape in (("base_4d", KC.BASE_CACHE_SHAPE),
                       ("v128_4d", KC.V128_CACHE_SHAPE),
                       ("base_2d", (KC.TOTAL_PAGES * KC.BLOCK_PER_KV_PACKING *
                                    KC.KV_PACKING, KC.LKV_DIM)),
                       ("v128_2d",
                        (KC.TOTAL_PAGES * KC.BLOCK_PER_KV_PACKING *
                         KC.KV_PACKING * rows, lane))):
        x = jax.ShapeDtypeStruct(shape, jnp.uint8)
        txt = jax.jit(lambda v: v + jnp.uint8(0)).lower(x).compile().as_text()
        layouts[tag] = entry_param_layouts(txt)
    res["layouts"] = layouts

    # And the reshape between the two 4-D forms, alone, at production size.
    for tag, src, dst in (("base_to_v128", KC.BASE_CACHE_SHAPE,
                           KC.V128_CACHE_SHAPE),
                          ("v128_to_base", KC.V128_CACHE_SHAPE,
                           KC.BASE_CACHE_SHAPE)):
        x = jax.ShapeDtypeStruct(src, jnp.uint8)
        txt = jax.jit(lambda v: v.reshape(dst)).lower(x).compile().as_text()
        path = os.path.join(out_root, "hlo", f"reshape__{tag}.hlo.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(txt)
        d = digest_hlo(txt)
        res.setdefault("reshape_only", {})[tag] = {
            "n_big_relayouts": d["n_big_relayouts"],
            "instructions": [r["line"] for r in d["big_relayouts"]],
            "hlo": path,
        }
        log(f"reshape-only {tag}: {d['n_big_relayouts']} big relayout op(s)")
        for r in d["big_relayouts"]:
            log(f"  {r['line']}")

    for k, v in res["checks"].items():
        log(f"LAYOUT {k} = {v}")
    for k, v in layouts.items():
        log(f"LAYOUT entry({k}) = {v}")
    return res


# ---------------------------------------------------------------------------
# --mode hlo
# ---------------------------------------------------------------------------
def mode_hlo(mods, configs, arms, out_root: str) -> dict:
    hlo_dir = os.path.join(out_root, "hlo")
    os.makedirs(hlo_dir, exist_ok=True)
    out: dict = {}
    for cname in configs:
        cfg = KC.CONFIG_BY_NAME[cname]
        for arm in arms:
            key = f"{cname}__{arm}"
            rec: dict = {"config": cname, "arm": arm}
            try:
                inputs = KC.build_inputs(cfg,
                                         layout=KC.CACHE_LAYOUT[arm],
                                         seed=0)
                args = KC.as_args(inputs)
                fn = KO.make_arm(mods, arm, donate=True)
                txt = fn.lower(*args).compile().as_text()
                path = os.path.join(hlo_dir, f"{key}.hlo.txt")
                with open(path, "w") as fh:
                    fh.write(txt)
                d = digest_hlo(txt)
                rec.update(d)
                rec["hlo"] = path
                rec["hlo_bytes"] = len(txt)
                rec["cache_shape"] = list(inputs["cache_kv"].shape)
                rec["ok"] = True
                log(f"=== {key}: cache{rec['cache_shape']} "
                    f"big_relayouts={d['n_big_relayouts']} "
                    f"scatters={len(d['scatters'])}")
                for r in d["big_relayouts"]:
                    log(f"  BIG   {r['line']}")
                for sc in d["scatters"]:
                    log(f"  SCAT  {sc['line']}")
                    for od in sc["operand_defs"]:
                        log(f"  SCAT-OPERAND  {od}")
                for v in inputs.values():
                    try:
                        v.delete()
                    except (AttributeError, RuntimeError):
                        pass
            except Exception as err:  # noqa: BLE001
                rec["ok"] = False
                rec["error"] = repr(err)[:1500]
                log(f"=== {key}: FAILED {err!r}")
                traceback.print_exc()
            out[key] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("layout", "hlo"), required=True)
    ap.add_argument("--configs", default="all")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--root", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    root = a.root or os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"),
                                  "kvs")
    os.makedirs(root, exist_ok=True)

    log(f"jax={jax.__version__} devices={jax.devices()}")
    mods = KC.load_modules()
    if not KC.report_md5s(mods, log):
        log("FATAL: kernel md5 read-back mismatch -- the pod is running the "
            "image's stale kernels.  Refusing to produce a result.")
        return 3
    KO.check_fidelity(log)

    if a.mode == "layout":
        payload = mode_layout(root)
        rc = 0 if all(payload["checks"].values()) else 1
    else:
        configs = ([c.name for c in KC.CONFIGS]
                   if a.configs == "all" else a.configs.split(","))
        arms = list(KC.ALL_ARMS) if a.arms == "all" else a.arms.split(",")
        payload = mode_hlo(mods, configs, arms, root)
        rc = 0 if all(v.get("ok") for v in payload.values()) else 1

    path = a.json or os.path.join(root, f"{a.mode}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    log(f"wrote {path}  rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
