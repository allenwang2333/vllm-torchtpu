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
"""Shrink an xprof capture on the machine that produced it.

A 93-layer / 32-rank decode capture is ~10 GB of `.xplane.pb`, which is more
than a CI artifact upload will carry and more than the node's GCS scope will
let out. Almost all of that is per-op *metadata* repeated once per plane, plus
a `counters_0` line nothing downstream reads. This turns each capture into:

    census.json                 every plane/line with its event count and bytes
                                (so "we dropped nothing that mattered" is a
                                check, not a promise)
    optext.jsonl.gz             the HLO text of every op, once, content-hashed
    planes/<file>__<plane>.json.gz
                                per device plane: XLA Ops as (meta, offset,
                                duration) arrays in issue order, XLA Modules,
                                plane stats
    hlometa/<file>.metaplane.pb.gz   the /host:metadata plane, verbatim
    slim/<file>.slim.xplane.pb       a valid xplane with the fat lines and the
                                unreferenced event metadata removed, for the
                                ranks named by --slim (xprof still reads it)

Stdlib only, on purpose: it has to run inside the serving container and on the
Buildkite agent without installing anything.

Wire format (tsl/profiler/protobuf/xplane.proto):
  XSpace{planes=1 errors=2 warnings=3 hostnames=4}
  XPlane{id=1 name=2 lines=3 event_metadata=4 stat_metadata=5 stats=6}
  XLine {id=1 name=2 timestamp_ns=3 events=4 duration_ps=9 display_id=10
         display_name=11}
  XEvent{metadata_id=1 offset_ps=2 duration_ps=3 stats=4 num_occurrences=5}
  XStat {metadata_id=1 double=2 uint64=3 int64=4 str=5 bytes=6 ref=7}
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import struct
import sys
import time

# Lines worth carrying off the machine. Everything else is censused and dropped.
KEEP_LINES = ("XLA Modules", "XLA Ops", "Async XLA Ops", "XLA TraceMe",
              "Steps", "Sparse Core Ops", "Sparse Core Modules",
              "Sparse Core Step Breakdown", "Mosaic Regions")
KEEP_PLANE_PREFIXES = ("/device:TPU", "/device:CUSTOM", "/host:metadata",
                       "Task Environment", "/host:CPU")
DEVICE_PLANE = "/device:TPU"


# --------------------------------------------------------------------------
# wire-format read
# --------------------------------------------------------------------------
def _varint(b, i):
    r = s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        if not x & 0x80:
            return r, i
        s += 7


def fields(b, start=0, end=None):
    """Yield (field_no, wire_type, payload) over one protobuf message."""
    i, end = start, len(b) if end is None else end
    while i < end:
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v = struct.unpack_from("<Q", b, i)[0]
            i += 8
        elif wt == 2:
            n, i = _varint(b, i)
            v = b[i:i + n]
            i += n
        elif wt == 5:
            v = struct.unpack_from("<I", b, i)[0]
            i += 4
        else:
            raise ValueError("wire type %d at %d" % (wt, i))
        yield fn, wt, v


def parse_stat(b):
    mid = val = None
    for fn, _wt, v in fields(b):
        if fn == 1:
            mid = v
        elif fn == 2:
            val = struct.unpack("<d", struct.pack("<Q", v))[0]
        elif fn in (3, 4):
            val = v
        elif fn == 5:
            val = v.decode("utf-8", "replace")
        elif fn == 6:
            val = "<%d bytes>" % len(v)
        elif fn == 7:
            val = ["ref", v]
    return mid, val


def _map_entry(b):
    """map<int64, X> entry -> (key, value_bytes)."""
    k = v = None
    for fn, _wt, val in fields(b):
        if fn == 1:
            k = val
        elif fn == 2:
            v = val
    return k, v


def _named(b):
    """XEventMetadata / XStatMetadata -> its `name` field."""
    for fn, _wt, v in fields(b):
        if fn == 2:
            return v.decode("utf-8", "replace")
    return None


# --------------------------------------------------------------------------
# wire-format write
# --------------------------------------------------------------------------
def _wv(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _tag(fn, wt):
    return _wv((fn << 3) | wt)


def _bytes_field(fn, payload):
    return _tag(fn, 2) + _wv(len(payload)) + payload


# --------------------------------------------------------------------------
# parsing one xplane file
# --------------------------------------------------------------------------
def _line_parts(raw_line):
    name = None
    events = []
    for fn, _wt, v in fields(raw_line):
        if fn == 2:
            name = v.decode("utf-8", "replace")
        elif fn == 4:
            events.append(v)
        elif fn == 11 and name is None:
            name = v.decode("utf-8", "replace")
    return name, events


def _plane_parts(raw_plane):
    pid = name = None
    lines, stats = [], []
    emeta, smeta = {}, {}
    for fn, _wt, v in fields(raw_plane):
        if fn == 1:
            pid = v
        elif fn == 2:
            name = v.decode("utf-8", "replace")
        elif fn == 3:
            lines.append(v)
        elif fn == 4:
            k, val = _map_entry(v)
            if k is not None:
                emeta[k] = val
        elif fn == 5:
            k, val = _map_entry(v)
            if k is not None:
                smeta[k] = _named(val)
        elif fn == 6:
            stats.append(v)
    return pid, name, lines, stats, emeta, smeta


def _events_arrays(events, smeta):
    """XEvents -> parallel arrays, plus the sparse stats we actually read."""
    mid, off, dur = [], [], []
    extra = {}
    for e in events:
        m = o = d = None
        st = {}
        for fn, _wt, v in fields(e):
            if fn == 1:
                m = v
            elif fn == 2:
                o = v
            elif fn == 3:
                d = v
            elif fn == 4:
                k, val = parse_stat(v)
                st[smeta.get(k, "s%s" % k)] = val
        if o is None:
            o = st.get("device_offset_ps") or 0
        if d is None:
            d = st.get("device_duration_ps") or 0
        i = len(mid)
        mid.append(m or 0)
        off.append(o)
        dur.append(d)
        for k in ("all_reduce_unique_id", "run_id", "replica_id", "queue_id",
                  "core_type", "group_id", "step_num"):
            if k in st:
                extra.setdefault(k, {})[i] = st[k]
    order = sorted(range(len(off)), key=off.__getitem__)
    if order != list(range(len(off))):
        mid = [mid[i] for i in order]
        off = [off[i] for i in order]
        dur = [dur[i] for i in order]
        pos = {o: i for i, o in enumerate(order)}
        extra = {k: {pos[i]: v for i, v in d.items()} for k, d in extra.items()}
    return mid, off, dur, extra


def _sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


class Reducer:

    def __init__(self, out_dir):
        self.out = out_dir
        self.texts = {}          # hash -> text
        self.census = []
        for sub in ("planes", "hlometa", "slim"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    def _hash(self, text):
        h = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]
        if h not in self.texts:
            self.texts[h] = text
        return h

    def file(self, path, slim=False, keep_metaplane=False):
        raw = open(path, "rb").read()
        base = os.path.basename(path).replace(".xplane.pb", "")
        frec = {"file": path, "bytes": len(raw), "planes": []}
        slim_planes = []
        errs, warns = [], []
        for fn, _wt, v in fields(raw):
            if fn == 2:
                errs.append(v.decode("utf-8", "replace"))
            elif fn == 3:
                warns.append(v.decode("utf-8", "replace"))
        frec["errors"], frec["warnings"] = errs, warns

        for fn, _wt, raw_plane in fields(raw):
            if fn != 1:
                continue
            pid, pname, lines, stats, emeta, smeta = _plane_parts(raw_plane)
            prec = {"plane": pname, "id": pid, "bytes": len(raw_plane),
                    "n_event_meta": len(emeta), "lines": []}
            keep_line_bytes = []
            used_mids = set()
            for rl in lines:
                lname, events = _line_parts(rl)
                prec["lines"].append({"name": lname, "events": len(events),
                                      "bytes": len(rl)})
                if lname in KEEP_LINES:
                    keep_line_bytes.append(rl)
                    for e in events:
                        for f2, _w2, v2 in fields(e):
                            if f2 == 1:
                                used_mids.add(v2)
                                break
            pstats = {}
            for s in stats:
                mid, val = parse_stat(s)
                pstats[smeta.get(mid, "stat%s" % mid)] = val
            prec["stats"] = pstats
            frec["planes"].append(prec)

            if pname and pname.startswith(DEVICE_PLANE):
                self._emit_plane(base, pid, pname, lines, pstats, emeta, smeta)
            if pname == "/host:metadata" and keep_metaplane:
                with gzip.open(os.path.join(
                        self.out, "hlometa", "%s.metaplane.pb.gz" % base),
                        "wb", compresslevel=6) as fh:
                    fh.write(raw_plane)
            if slim and pname and pname.startswith(KEEP_PLANE_PREFIXES):
                slim_planes.append(
                    self._slim_plane(raw_plane, keep_line_bytes, used_mids,
                                     emeta, pname))

        if slim and slim_planes:
            body = b"".join(_bytes_field(1, p) for p in slim_planes)
            dst = os.path.join(self.out, "slim", "%s.slim.xplane.pb" % base)
            with open(dst, "wb") as fh:
                fh.write(body)
            frec["slim_bytes"] = len(body)
        self.census.append(frec)
        return frec

    def _emit_plane(self, base, pid, pname, lines, pstats, emeta, smeta):
        rec = {"file": base, "plane": pname, "plane_id": pid, "stats": pstats}
        meta = {}
        for rl in lines:
            lname, events = _line_parts(rl)
            if lname not in ("XLA Ops", "XLA Modules", "Async XLA Ops",
                             "XLA TraceMe", "Sparse Core Ops",
                             "Sparse Core Modules"):
                continue
            mid, off, dur, extra = _events_arrays(events, smeta)
            key = {"XLA Ops": "ops", "XLA Modules": "modules",
                   "Async XLA Ops": "async", "XLA TraceMe": "traceme",
                   "Sparse Core Ops": "sc_ops",
                   "Sparse Core Modules": "sc_modules"}[lname]
            rec[key] = {"mid": mid, "off": off, "dur": dur,
                        "extra": {k: {str(i): v for i, v in d.items()}
                                  for k, d in extra.items()}}
            for m in set(mid):
                if m not in meta:
                    nm = _named(emeta.get(m, b"")) if m in emeta else None
                    meta[m] = self._hash(nm) if nm is not None else None
        rec["meta"] = {str(k): v for k, v in meta.items()}
        dst = os.path.join(self.out, "planes",
                           "%s__%s.json.gz" % (base, _sanitize(pname)))
        with gzip.open(dst, "wt", compresslevel=6) as fh:
            json.dump(rec, fh, separators=(",", ":"))

    @staticmethod
    def _slim_plane(raw_plane, keep_line_bytes, used_mids, emeta, pname):
        head = bytearray()
        tail = bytearray()
        for fn, wt, v in fields(raw_plane):
            if fn == 1:
                head += _tag(1, 0) + _wv(v)
            elif fn == 2:
                head += _bytes_field(2, v)
            elif fn == 3:
                continue                      # lines: re-added from keep list
            elif fn == 4:
                k, _ = _map_entry(v)
                # /host:metadata carries the HLO protos as event metadata and
                # has no lines to reference them, so keep all of its map.
                if pname == "/host:metadata" or k in used_mids:
                    tail += _bytes_field(4, v)
            elif fn == 5:
                tail += _bytes_field(5, v)
            elif fn == 6:
                tail += _bytes_field(6, v)
            elif wt == 2:
                tail += _bytes_field(fn, v)
            elif wt == 0:
                tail += _tag(fn, 0) + _wv(v)
        mid = b"".join(_bytes_field(3, l) for l in keep_line_bytes)
        return bytes(head) + mid + bytes(tail)

    def finish(self):
        with gzip.open(os.path.join(self.out, "optext.jsonl.gz"), "wt",
                       compresslevel=6) as fh:
            for h, t in self.texts.items():
                fh.write(json.dumps({"h": h, "t": t},
                                    separators=(",", ":")) + "\n")
        with open(os.path.join(self.out, "census.json"), "w") as fh:
            json.dump({"files": self.census, "n_optext": len(self.texts)},
                      fh, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True,
                    help="directory searched recursively for *.xplane.pb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--slim", default="rank0",
                    help="comma-separated basename prefixes to also emit a "
                         "slim xplane for ('none' to disable)")
    ap.add_argument("--meta-from", default="rank0",
                    help="prefixes whose /host:metadata plane is kept verbatim")
    ap.add_argument("--delete-source", action="store_true",
                    help="unlink each xplane.pb once it has been reduced")
    args = ap.parse_args(argv)

    pbs = sorted(glob.glob(os.path.join(args.root, "**", "*.xplane.pb"),
                           recursive=True))
    if not pbs:
        print("xprof_reduce: no xplane.pb under %s" % args.root)
        return 1
    slim_pfx = () if args.slim == "none" else tuple(args.slim.split(","))
    meta_pfx = () if args.meta_from == "none" else tuple(
        args.meta_from.split(","))
    os.makedirs(args.out, exist_ok=True)
    r = Reducer(args.out)
    t0 = time.time()
    for p in pbs:
        b = os.path.basename(p)
        try:
            rec = r.file(p, slim=b.startswith(slim_pfx),
                         keep_metaplane=b.startswith(meta_pfx))
            print("xprof_reduce: %s %.1f MB -> slim %.1f MB (%.0fs)"
                  % (b, rec["bytes"] / 1e6, rec.get("slim_bytes", 0) / 1e6,
                     time.time() - t0), flush=True)
            if args.delete_source:
                os.unlink(p)
        except Exception as exc:                       # keep going: partial
            print("xprof_reduce: FAILED on %s: %r" % (p, exc), flush=True)
            r.census.append({"file": p, "error": repr(exc)})
    r.finish()
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(args.out) for f in fs)
    print("xprof_reduce: %d files -> %.1f MB in %s (%.0fs)"
          % (len(pbs), total / 1e6, args.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
