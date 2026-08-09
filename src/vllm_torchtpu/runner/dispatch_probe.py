# SPDX-License-Identifier: Apache-2.0
"""Per-rank dispatch-sequence trace, for diagnosing lockstep divergence.

Under torch-tpu's ``DEFER_AND_FUSE`` the fused programs are keyed on the
live dispatch sequence, and on a TPU slice program launches are collective.
A rank that takes a data-dependent early return therefore skips a launch its
peers still make, and the step deadlocks with the ranks split across phases.

This module records, per rank, the ordered list of decision points reached
in a step so two ranks' traces can be diffed line-by-line. The first line
where they disagree is the divergence.

Deliberately host-only: every value logged is a Python int, a bool, or a
tensor *shape*. Nothing here reads a device value, so the probe never forces
materialization and never perturbs the timing that makes the hang
reproduce. That is the difference between this and the earlier
``debug: drain probes`` commits, which forced a flush per stage and so
changed what they were measuring.

Enable with ``VLLM_TPU_DISPATCH_PROBE=1``. Output goes to
``$VLLM_TPU_DISPATCH_PROBE_DIR`` (default ``/tmp/dispatch_trace``) as one
``rank<N>.tsv`` per rank. Disabled, ``probe()`` is a single module-level
bool test.
"""
import os
import threading

_ENABLED = os.environ.get("VLLM_TPU_DISPATCH_PROBE", "") == "1"
_DIR = os.environ.get("VLLM_TPU_DISPATCH_PROBE_DIR", "/tmp/dispatch_trace")

_lock = threading.Lock()
_handle = None
_seq = 0
_step = 0


def _rank() -> int:
    """Best-effort global rank, without importing torch.distributed."""
    for var in ("RANK", "VLLM_DP_RANK", "LOCAL_RANK"):
        val = os.environ.get(var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return os.getpid()


def _open():
    global _handle
    if _handle is None:
        os.makedirs(_DIR, exist_ok=True)
        path = os.path.join(_DIR, f"rank{_rank()}.tsv")
        # Line-buffered: a hang must not lose the tail of the trace in a
        # buffer that is never flushed, which is the whole point here.
        _handle = open(path, "w", buffering=1)
        _handle.write("#seq\tstep\tevent\tdetail\n")
    return _handle


def new_step() -> None:
    """Mark a step boundary so traces stay aligned when one rank lags."""
    global _step
    if not _ENABLED:
        return
    with _lock:
        _step += 1
    probe("step_begin")


def probe(event: str, **detail) -> None:
    """Record one decision point.

    Args:
        event: stable identifier for the site, e.g. ``"seed_copy.flush"``.
        detail: host-side scalars only. Never pass a device tensor; read a
            ``.shape`` or a Python-side count instead.
    """
    if not _ENABLED:
        return
    global _seq
    with _lock:
        _seq += 1
        seq = _seq
        step = _step
        fields = " ".join(f"{k}={v}" for k, v in sorted(detail.items()))
        _open().write(f"{seq}\t{step}\t{event}\t{fields}\n")
