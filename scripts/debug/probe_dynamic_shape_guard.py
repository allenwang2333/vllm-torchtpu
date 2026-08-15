#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""How do you tell, inside dynamo, that a dim is dynamic -- without pinning it?

K3 fails to compile with the low-latency all-reduce enabled:

    ConstraintViolationError: Constraints violated (L['input_ids'].size()[0])!
    You marked it as dynamic but your code specialized it to be a constant

attributed to the dispatcher's ``self._plans.get(key)`` -- hashing a shape
reads its dims, and reading a dynamic dim pins it. An ``isinstance(dim, int)``
guard did *not* prevent it, so whatever dynamo hands back there is not a
plain SymInt.

The earlier TP8 check missed this because it marked the reduced tensor itself
dynamic. vLLM marks ``input_ids`` and the reduced tensor is *derived* from it,
which is what this reproduces: dynamic ids -> embedding -> all-reduce.

Records, for each candidate predicate, what it answers on a derived dynamic
dim, and whether the graph still compiles afterwards. The one that both
answers "dynamic" and leaves compilation intact is the guard to use.

    torchrun --nproc_per_node=8 scripts/debug/probe_dynamic_shape_guard.py
"""

import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401

OBSERVED: list = []


def main() -> int:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="tpu_dist",
                            rank=rank,
                            world_size=world_size)

    device = torch.device("tpu")
    hidden = 1024
    weight = torch.randn(256, hidden,
                         dtype=torch.float32).to(torch.bfloat16).to(device)

    def probe_dims(t: torch.Tensor) -> None:
        """Record what each predicate says, without using the answer."""
        if not OBSERVED:
            d0 = t.shape[0]
            # Only isinstance: anything reflective (getattr/hasattr on the
            # symbolic node) is itself untraceable and aborts the probe.
            OBSERVED.append({
                "isinstance_int": isinstance(d0, int),
                "isinstance_symint": isinstance(d0, torch.SymInt),
            })

    # -- variant A: ask the predicates, take neither branch --
    def model_reads_shape(ids):
        x = torch.nn.functional.embedding(ids, weight)
        probe_dims(x)
        return x * 2.0

    # -- variant B: never read the shape --
    def model_no_read(ids):
        x = torch.nn.functional.embedding(ids, weight)
        return x * 2.0

    # vLLM's combination: the token dim is marked dynamic, but the TPU backend
    # cannot compile dynamic shapes, so the graph is compiled dynamic=False.
    # The violation then surfaces at guard-building time, which is exactly the
    # K3 failure and what has to be reproduced here.
    results = {}
    for label, fn in (("reads_shape", model_reads_shape), ("no_read",
                                                           model_no_read)):
        ids = torch.randint(0, 256, (512, ), dtype=torch.int32).to(device)
        torch._dynamo.mark_dynamic(ids, 0)
        try:
            compiled = torch.compile(fn,
                                     backend="tpu",
                                     fullgraph=True,
                                     dynamic=False)
            out = compiled(ids)
            out.cpu()
            results[label] = "compiled"
        except Exception as exc:  # noqa: BLE001
            results[label] = f"{type(exc).__name__}: {str(exc)[:160]}"
            if rank == 0 and label == "reads_shape":
                print(traceback.format_exc()[-500:], flush=True)
        torch._dynamo.reset()

    if rank == 0:
        print("\n=== what a derived dynamic dim looks like inside dynamo ===",
              flush=True)
        for entry in OBSERVED:
            for k, v in entry.items():
                print(f"  {k:>18}: {v}", flush=True)
        print("\n=== does the graph still compile? ===", flush=True)
        for label, outcome in results.items():
            print(f"  {label:>12}: {outcome}", flush=True)

    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
