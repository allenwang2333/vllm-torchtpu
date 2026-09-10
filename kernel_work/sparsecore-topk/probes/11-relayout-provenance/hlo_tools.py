"""A tolerant optimized-HLO reader.

An HLO line prints ITS OWN output shape and names its operands BY NAME. So a
grep for an operand's shape on an instruction line can never match. Everything
here resolves operands by walking names into a global instruction index, and
every reported shape is the shape the instruction that owns it printed.
"""

from __future__ import annotations

import re

_ELT_BYTES = {
    "pred": 1, "s8": 1, "u8": 1, "s16": 2, "u16": 2, "bf16": 2, "f16": 2,
    "s32": 4, "u32": 4, "f32": 4, "s64": 8, "u64": 8, "f64": 8,
    "c64": 8, "c128": 16, "s4": 0.5, "u4": 0.5,
    "f8e4m3fn": 1, "f8e5m2": 1, "f8e4m3fnuz": 1, "f8e5m2fnuz": 1,
    "f8e4m3b11fnuz": 1, "f8e3m4": 1, "f8e4m3": 1,
}
_SHAPE_RE = re.compile(r"\b([a-z][a-z0-9]*)\[([0-9,\s]*)\]")
_INSTR_RE = re.compile(r"^\s*(ROOT\s+)?%([A-Za-z0-9_.\-]+)\s*=\s*(.*)$")
_OPCODE_RE = re.compile(r"^(.*?)\s+([a-z][a-z0-9\-]*)\(")
_COMP_RE = re.compile(r"^\s*(ENTRY\s+)?%?([A-Za-z0-9_.\-]+)\s*\((.*?)\)\s*->\s*(.*?)\s*\{\s*$")
_NAME_RE = re.compile(r"%([A-Za-z0-9_.\-]+)")
_LAYOUT_RE = re.compile(r"\{([^{}]*)\}")

MIB = 1024.0 * 1024.0


def shape_bytes(shape_str: str) -> float:
    """Bytes of the (possibly tuple) shape an instruction printed."""
    tot = 0.0
    for dt, dims in _SHAPE_RE.findall(shape_str):
        if dt not in _ELT_BYTES:
            continue
        n = 1
        for d in dims.split(","):
            d = d.strip()
            if d:
                n *= int(d)
        tot += n * _ELT_BYTES[dt]
    return tot


def layouts_of(shape_str: str) -> list[str]:
    return _LAYOUT_RE.findall(shape_str)


def _split_top(s: str, sep: str = ",") -> list[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return [x.strip() for x in out]


class Hlo:
    """Every instruction of a module, indexed by name."""

    def __init__(self, text: str):
        self.text = text
        self.lines = text.splitlines()
        self.instrs: dict[str, dict] = {}      # name -> first definition
        self.dupes: dict[str, int] = {}
        self.order: list[dict] = []
        self.comps: dict[str, dict] = {}       # computation -> header info
        cur = "<toplevel>"
        for raw in self.lines:
            ch = _COMP_RE.match(raw)
            if ch and " = " not in raw:
                cur = ch.group(2)
                params = {}
                for p in _split_top(ch.group(3)):
                    if ":" in p:
                        pn, ps = p.split(":", 1)
                        params[pn.strip().lstrip("%")] = ps.strip()
                self.comps[cur] = {
                    "name": cur, "is_entry": bool(ch.group(1)),
                    "params": params, "result": ch.group(4), "line": raw.strip(),
                }
                continue
            ins = self._parse(raw, cur)
            if ins is None:
                continue
            self.order.append(ins)
            if ins["name"] in self.instrs:
                self.dupes[ins["name"]] = self.dupes.get(ins["name"], 1) + 1
            else:
                self.instrs[ins["name"]] = ins

    @staticmethod
    def _parse(raw: str, comp: str):
        m = _INSTR_RE.match(raw)
        if not m:
            return None
        rhs = m.group(3)
        m2 = _OPCODE_RE.match(rhs)
        if not m2:
            return None
        start = m2.end() - 1
        depth = 0
        end = len(rhs) - 1
        for j in range(start, len(rhs)):
            if rhs[j] == "(":
                depth += 1
            elif rhs[j] == ")":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        ops_s = rhs[start + 1:end]
        rest = rhs[end + 1:]
        shape = m2.group(1).strip()
        return {
            "name": m.group(2), "root": bool(m.group(1)), "comp": comp,
            "shape": shape, "opcode": m2.group(2),
            "operands": [x for x in _NAME_RE.findall(ops_s)],
            "operands_raw": _split_top(ops_s),
            "bytes": shape_bytes(shape), "layouts": layouts_of(shape),
            "calls": (_NAME_RE.findall(rest.split("calls=")[1])[:8]
                      if "calls=" in rest else []),
            "line": raw.strip(),
        }

    # ---------------------------------------------------------------- views
    def find(self, pred) -> list[dict]:
        return [i for i in self.order if pred(i)]

    def by_name(self, name: str):
        return self.instrs.get(name)

    def shape_of(self, name: str, comp: str | None = None) -> str:
        i = self.instrs.get(name)
        if i is not None:
            return i["shape"]
        if comp and comp in self.comps:
            p = self.comps[comp]["params"].get(name)
            if p:
                return p + "  (parameter of %" + comp + ")"
        for c in self.comps.values():
            p = c["params"].get(name)
            if p:
                return p + "  (parameter of %" + c["name"] + ")"
        return "<unresolved>"

    def walk(self, seed_names, max_nodes: int = 60) -> list[dict]:
        """Transitive operand closure, BFS, deduped, in visit order."""
        seen, out, q = set(), [], list(seed_names)
        while q and len(out) < max_nodes:
            nm = q.pop(0)
            if nm in seen:
                continue
            seen.add(nm)
            i = self.instrs.get(nm)
            if i is None:
                out.append({"name": nm, "opcode": "<param/extern>",
                            "shape": self.shape_of(nm), "comp": "",
                            "operands": [], "bytes": 0.0, "line": ""})
                continue
            out.append(i)
            for c in i["calls"]:
                comp = self.comps.get(c)
                if comp:
                    for j in self.order:
                        if j["comp"] == c and j["root"]:
                            q.append(j["name"])
            q.extend(i["operands"])
        return out

    # ------------------------------------------------------------- reports
    def data_format_lines(self) -> list[str]:
        return [ln.strip() for ln in self.lines
                if "sparse-core-data-format" in ln]

    def big(self, min_mib: float = 1.0, drop_bitcast: bool = True) -> list[dict]:
        out = []
        for i in self.order:
            if i["bytes"] < min_mib * MIB:
                continue
            if drop_bitcast and i["opcode"] in ("bitcast", "parameter",
                                                "get-tuple-element", "tuple"):
                continue
            out.append(i)
        return out

    def copies(self) -> list[dict]:
        res = []
        for i in self.order:
            if i["opcode"] not in ("copy", "transpose", "reshape"):
                continue
            src = i["operands"][0] if i["operands"] else None
            src_shape = self.shape_of(src, i["comp"]) if src else "<none>"
            lo, lin = layouts_of(i["shape"]), layouts_of(src_shape)
            res.append({**i, "src": src, "src_shape": src_shape,
                        "layout_out": lo[0] if lo else "",
                        "layout_in": lin[0] if lin else "",
                        "layout_changes": bool(lo and lin and lo[0] != lin[0])})
        return res


def summarize(text: str, kernel_tag: str = "sc_topk_direct") -> dict:
    """The per-arm activation record. Never a measurement.

    `materialized_MiB` counts only `copy` and `fusion` outputs at or above
    1 MiB. Those are the two opcodes that unambiguously write a whole new
    buffer. `async-start`/`async-done` are deliberately NOT counted: they wrap
    a called computation whose own instructions are already in the list, and
    counting both double-counts. A materialization of X MiB moves ~2X MiB
    (a full read plus a full write).
    """
    h = Hlo(text)
    df = h.data_format_lines()
    cps = h.copies()
    big_copy = [c for c in cps if c["opcode"] == "copy" and c["bytes"] >= MIB]
    lay_copy = [c for c in big_copy if c["layout_changes"]]
    big_fusion = [i for i in h.order
                  if i["opcode"] == "fusion" and i["bytes"] >= MIB]
    kern = [i for i in h.order
            if kernel_tag in i["name"] and i["opcode"] == "custom-call"]
    copy_mib = sum(c["bytes"] for c in big_copy) / MIB
    fus_mib = sum(i["bytes"] for i in big_fusion) / MIB
    inventory = sorted(
        [i for i in h.big(1.0)
         if i["opcode"] not in ("async-start", "async-done")],
        key=lambda x: -x["bytes"])
    return {
        "lines": len(h.lines),
        "data_format_call": bool(df),
        "data_format_line_count": len(df),
        "data_format_lines": df[:8],
        "big_copy": bool(big_copy),
        "big_copy_count": len(big_copy),
        "layout_changing_big_copy": bool(lay_copy),
        "layout_changing_big_copy_count": len(lay_copy),
        "big_fusion_count": len(big_fusion),
        "big_copy_MiB": round(copy_mib, 3),
        "big_fusion_MiB": round(fus_mib, 3),
        "materialized_MiB": round(copy_mib + fus_mib, 3),
        "kernel_calls": [i["name"] for i in kern],
        "inventory": [
            {"name": i["name"], "opcode": i["opcode"], "comp": i["comp"],
             "MiB": round(i["bytes"] / MIB, 3), "shape": i["shape"][:100]}
            for i in inventory[:16]],
        "big_copy_detail": [
            {"name": c["name"], "MiB": round(c["bytes"] / MIB, 3),
             "comp": c["comp"],
             "layout_in": c["layout_in"], "layout_out": c["layout_out"],
             "changes": c["layout_changes"], "src": c["src"],
             "src_shape": c["src_shape"][:100], "shape": c["shape"][:100]}
            for c in sorted(big_copy, key=lambda x: -x["bytes"])[:12]],
    }
