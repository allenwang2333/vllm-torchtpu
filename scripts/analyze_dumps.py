#!/usr/bin/env python3
"""Classify py-spy DUMP snapshots of DP workers into host-time buckets.

For each dump we take the MainThread stack (the worker execute loop) and bucket
it by its deepest signature frame:

  BARRIER   - CPU-gloo DP all_reduce (the lockstep barrier)
  WAIT_DEV  - device sync: wait_for_copy / synchronize / event wait / .item on tpu
  FORWARD   - model forward / logits compute (executable dispatch or in-graph)
  SAMPLE    - sampling / async copy launch
  PREP      - host input prep / update_states / python step overhead
  IDLE      - blocked in shm_broadcast dequeue (waiting for scheduler; no step)

Prints per-worker and aggregate percentages. IDLE is reported separately and the
"active" barrier fraction (BARRIER / (total - IDLE)) is also shown, since the
prior ~50-58% claim is about time *within* running steps.

Usage: analyze_dumps.py <dumps_dir>
"""
import glob
import os
import re
import sys

# leaf->root; first hit wins. Substrings matched against "func (file:line)".
SIGS = [
    ("IDLE", ("shm_broadcast", "acquire_read", "dequeue", "worker_busy_loop",
              "zmq/sugar/poll", "death_pipe", "async_output_busy_loop")),
    ("BARRIER", ("_dp_coordinated_step", "all_reduce", "distributed_c10d",
                 "reduce.py", "->allreduce", "processgroupgloo")),
    ("WAIT_DEV", ("wait_for_copy", "_modify_prev_results", "synchronize",
                  "copy_ready_event", ".wait (", "event", "mark_step",
                  "flush_disjoint")),
    ("SAMPLE", ("sample_tokens", "sampler", "from_device_chunks",
                "sample_from_logits", "gather_logprobs")),
    ("FORWARD", ("forward_model", "compute_selected_logits", "compute_logits",
                 "forward", "_backend", "compiled", "fused_moe", "attention",
                 "decoder_layer", "model(")),
    ("PREP", ("_prepare_inputs", "_update_states", "execute_model",
              "_reorder_batch", "_get_model_inputs", "_apply_async",
              "_count_input_chunks", "maybe_setup_kv", "prepare")),
]

ORDER = ["BARRIER", "WAIT_DEV", "FORWARD", "SAMPLE", "PREP", "IDLE"]


def mainthread_frames(text):
    """Return the MainThread stack as a list of 'func (file:line)' lowercased,
    leaf-first (as printed by py-spy)."""
    lines = text.splitlines()
    frames = []
    in_main = False
    for ln in lines:
        m = re.match(r'^Thread .*"(.+?)"', ln)
        if m:
            in_main = (m.group(1) == "MainThread")
            continue
        if in_main:
            s = ln.strip()
            if s:
                frames.append(s.lower())
    return frames


def bucket_of(frames):
    for fr in frames:  # leaf -> root
        for label, subs in SIGS:
            for s in subs:
                if s in fr:
                    return label
    return "PREP"  # default: some python step work


def worker_of(path):
    m = re.match(r'(w\d+)_', os.path.basename(path))
    return m.group(1) if m else "?"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*.txt")))
    if not files:
        print("no dumps found in", sys.argv[1])
        sys.exit(2)

    per = {}   # worker -> {bucket: count}
    for fp in files:
        with open(fp) as f:
            frames = mainthread_frames(f.read())
        if not frames:
            continue
        b = bucket_of(frames)
        w = worker_of(fp)
        per.setdefault(w, {}).setdefault(b, 0)
        per[w][b] += 1

    # header
    print(f"{'worker':<10}{'N':>6}" + "".join(f"{k:>9}" for k in ORDER)
          + f"{'BAR/act':>9}")
    agg = {k: 0 for k in ORDER}
    tot = 0
    for w in sorted(per):
        c = per[w]
        n = sum(c.values())
        tot += n
        for k in ORDER:
            agg[k] += c.get(k, 0)
        active = n - c.get("IDLE", 0)
        bar_act = (100 * c.get("BARRIER", 0) / active) if active else 0
        row = "".join(f"{100*c.get(k,0)/n:8.1f}%" for k in ORDER)
        print(f"{w:<10}{n:>6}{row}{bar_act:8.1f}%")
    print("-" * (10 + 6 + 9 * len(ORDER) + 9))
    active = tot - agg["IDLE"]
    bar_act = (100 * agg["BARRIER"] / active) if active else 0
    row = "".join(f"{100*agg[k]/tot:8.1f}%" for k in ORDER)
    print(f"{'ALL':<10}{tot:>6}{row}{bar_act:8.1f}%")
    print(f"\nBARRIER = {100*agg['BARRIER']/tot:.1f}% of all samples, "
          f"{bar_act:.1f}% of ACTIVE (non-idle) samples.")
    print(f"IDLE = {100*agg['IDLE']/tot:.1f}% (inter-step waiting for scheduler)")
    print("prior claim: ~50-58% of prefill worker time in the barrier")


if __name__ == "__main__":
    main()
