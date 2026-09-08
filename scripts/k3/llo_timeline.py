#!/usr/bin/env python3
"""Reconstruct the per-call LLO timeline of a probe kernel.

Pairs the dense LLO bundle events in a trace.json.gz (instrumentation_id +
ts) with the id->opcode map from `xprof get_llo_debug_string`, clusters the
bundle stream into kernel calls, and reports where the matmul, remote-DMA,
and semaphore bundles land within one call -- i.e. whether the sends
overlap the matmul.

Usage: llo_timeline.py <trace.json.gz> <llo-dbg.json> [call-index]
"""
import gzip
import json
import re
import sys

trace_path, dbg_path = sys.argv[1], sys.argv[2]
want = int(sys.argv[3]) if len(sys.argv) > 3 else None

# id -> opcode, from ONE module only (instruction ids collide across
# modules). Default: the fused kernel's module.
want_module = sys.argv[4] if len(sys.argv) > 4 else 'tpu_custom_call'
dbg = json.load(open(dbg_path))
id2op = {}
for m in re.split(r'Module: ', dbg['debug_string'])[1:]:
    if not m.split('\n')[0].startswith(want_module):
        continue
    for mm in re.finditer(r'\[(\d+)\]\s+(?:%\w+ = )?(OPCODE_\w+)', m):
        id2op[int(mm.group(1))] = mm.group(2)
print(f'{len(id2op)} instruction ids from module {want_module}')

d = json.load(gzip.open(trace_path))
ev = d['traceEvents'] if isinstance(d, dict) else d
bundles = []
for e in ev:
    if e.get('ph') == 'X' and str(e.get('name', '')).startswith('bundle.'):
        bid = int(e['name'].split('.')[1])
        if bid in id2op:  # only the target module's bundles
            bundles.append((e['ts'], bid, e.get('tid')))
bundles.sort()

# Cluster into calls: a new call starts after a >30us gap.
calls = []
cur = [bundles[0]]
for prev, b in zip(bundles, bundles[1:]):
    if b[0] - prev[0] > 30:
        calls.append(cur)
        cur = []
    cur.append(b)
calls.append(cur)
print(f'{len(calls)} call clusters, '
      f'median {sorted(len(c) for c in calls)[len(calls) // 2]} bundles/call')

idx = want if want is not None else len(calls) // 2
call = calls[idx]
t0 = call[0][0]

CLASSES = (
    ('MATMUL', lambda op: 'MATMUL' in op),
    ('DMA_REMOTE', lambda op: op == 'OPCODE_DMA_GENERAL'),
    ('DMA_WAIT', lambda op: op == 'OPCODE_DMA_DONE_WAIT'),
    ('SEM_SIG', lambda op: 'SEMAPHORE_SIGNAL' in op),
    ('SEM_WAIT', lambda op: 'SEMAPHORE_WAIT' in op),
    ('BARRIER', lambda op: 'BARRIER' in op),
)
print(f'call #{idx}: {len(call)} bundles over '
      f'{(call[-1][0] - t0):.2f} us')
for name, pred in CLASSES:
    ts = [ts - t0 for ts, bid, _ in call if pred(id2op.get(bid, ''))]
    if not ts:
        print(f'  {name:10s}  none')
        continue
    ts.sort()
    print(f'  {name:10s}  n={len(ts):3d}  span [{ts[0]:8.2f}, '
          f'{ts[-1]:8.2f}] us  (median {ts[len(ts) // 2]:8.2f})')

# Ordered summary: quantiles of each class as a coarse timeline.
marks = []
for name, pred in CLASSES:
    for ts, bid, _ in call:
        if pred(id2op.get(bid, '')):
            marks.append((ts - t0, name[0]))
marks.sort()
# Bucket into 20 time slices for a compact ASCII timeline.
span = marks[-1][0] - marks[0][0] if marks else 1.0
buckets = [[] for _ in range(20)]
for ts, c in marks:
    buckets[min(19, int((ts - marks[0][0]) / (span + 1e-9) * 20))].append(c)
print('timeline (M=matmul D=dma W=dma-wait S=sem-sig s=sem-wait B=barrier):')
for i, b in enumerate(buckets):
    print(f'  {marks[0][0] + span * i / 20:8.2f} us  {"".join(sorted(b))}')
