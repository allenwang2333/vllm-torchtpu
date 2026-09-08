#!/usr/bin/env python3
"""Decode the embedded Mosaic kernel from an HLO dump's custom-call body.

Usage: decode_kernel_body.py <after_optimizations.txt> [opname-filter]

Prints the kernel's op sequence (opcode + location), so the DMA/barrier
ORDER of a hanging kernel can be diffed against a working one.
"""
import base64
import re
import sys

from jax.experimental.mosaic.dialects import tpu  # noqa: F401  (registers tpu)
from jaxlib.mlir import ir

try:
    from jaxlib.mlir.dialects import register_stablehlo_dialect  # noqa: F401
except ImportError:
    # jaxlib >= 0.11 self-registers dialects on import.
    from jaxlib.mlir.dialects import stablehlo  # noqa: F401

path = sys.argv[1]
pat = sys.argv[2] if len(sys.argv) > 2 else None
raw = open(path).read()

# Every custom-call in the module carries its kernel as base64 in
# backend_config.custom_call_config.body.
bodies = []
for m in re.finditer(
        r'custom_call_config\\": \{\\"body\\": \\"([A-Za-z0-9+/=]+)\\"', raw):
    bodies.append(m.group(1))
if not bodies:
    # HLO text escapes quotes differently: "body": "..." with \" inside
    for m in re.finditer(r'body\\?": \s*\\?"([A-Za-z0-9+/=]{100,})\\?"', raw):
        bodies.append(m.group(1))
print(f"found {len(bodies)} kernel bodies")

for bi, b64 in enumerate(bodies):
    blob = base64.b64decode(b64)
    with ir.Context() as ctx:
        tpu.register_dialect(ctx)
        try:
            mod = ir.Module.parse(blob, context=ctx)
        except Exception as e:
            print(f"body {bi}: parse failed: {e}")
            continue
    asm = str(mod)
    lines = asm.splitlines()
    print(f"=== body {bi}: {len(lines)} lines")
    interesting = re.compile(
        r'(dma_start|dma_wait|semaphore|barrier|vmatmul|matmul|tpu\.|return|func\.)'
    )
    n = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if pat and pat not in s:
            continue
        if any(k in s for k in ('dma_start', 'dma_wait', 'semaphore_signal',
                                'semaphore_wait', 'matmul', 'vmatmul',
                                'get_barrier_semaphore', 'vector_store',
                                'vector_load', 'tpu.loop', 'scf.', 'launch')):
            print(f"  {i:5d}: {s[:180]}")
            n += 1
            if n > 250:
                print("  ... (truncated)")
                break
