#!/usr/bin/env python3
"""Per-call unit-utilization timeline from a packed-layout-probe trace.

Extracts the MXU / Scalar ALU / Vector ALU / XLU / Vector Load|Store windows
for one fused-kernel call (event index given) from a trace.json.gz, plus the
bundle-event span, so the overlap structure of two variants can be compared.

Usage: call_timeline.py <trace.json.gz> [call-index]
"""
import gzip
import json
import sys

path = sys.argv[1]
want_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 25  # mid-run call

d = json.load(gzip.open(path))
ev = d['traceEvents'] if isinstance(d, dict) else d

TRACKS = ('MXU', 'Scalar ALU', 'Vector ALU', 'XLU', 'Vector Load',
          'Vector Store', 'Vector EUP', 'Vector Fills', 'Vector Spills')
per_track = {t: [] for t in TRACKS}
for e in ev:
    if e.get('ph') == 'X' and e.get('pid') == 3 and e.get('name') in per_track:
        per_track[e['name']].append((e['ts'], e['dur']))

for t in per_track:
    per_track[t].sort()

# Use the MXU event's call to anchor: all tracks share the call cadence.
mxu = per_track['MXU']
if not mxu:
    print('no MXU events')
    sys.exit(1)
ts, dur = mxu[want_idx]
# The call window: from the earliest same-call event on any track to the
# latest. Take each track's event whose ts is within [-30us, +30us] of the
# MXU event's ts (same call).
print(f'call #{want_idx}: MXU busy {dur / 1e3:.2f} us at ts={ts}')
lo, hi = ts - 50e3, ts + 100e3
first, last = ts + dur, ts + dur
for t in TRACKS:
    cands = [(s, u) for s, u in per_track[t] if lo <= s <= hi]
    if not cands:
        continue
    s, u = min(cands, key=lambda su: abs(su[0] - ts))
    first = min(first, s)
    last = max(last, s + u)
    print(f'  {t:14s} start {(s - ts) / 1e3:+8.2f} us   dur {u / 1e3:7.2f} us')
print(f'  total call span: {(last - first) / 1e3:.2f} us '
      f'(MXU busy {dur / 1e3:.2f} us = {dur / (last - first) * 100:.0f}%)')
