"""Pytest plugin: per-file JAX compile time, optionally with caches cleared per file.

Load with `-p compile_probe` (module on PYTHONPATH).

COMPILE_PROBE_MODE=normal  record only.
COMPILE_PROBE_MODE=clear   also call jax.clear_caches() before the first test of
                           every file, so no file can reuse another file's
                           compiled executables.
COMPILE_PROBE_OUT          JSON output path (default
                           /perf_eval_results/compile_probe_<mode>.json).

Registering a jax.monitoring listener and calling jax.clear_caches() do not
create a PJRT client, so the plugin does not initialize the TPU in the pytest
process by itself.
"""

import collections
import json
import os
import sys
import time

_EVENTS = {
    "/jax/core/compile/jaxpr_trace_duration": "trace_s",
    "/jax/core/compile/jaxpr_to_mlir_module_duration": "lower_s",
    "/jax/core/compile/backend_compile_duration": "backend_s",
}
_MODE = os.environ.get("COMPILE_PROBE_MODE", "normal")
_OUT = os.environ.get(
    "COMPILE_PROBE_OUT", f"/perf_eval_results/compile_probe_{_MODE}.json"
)

_stats = collections.defaultdict(lambda: collections.defaultdict(float))
_order = []
_current = {"file": "<outside tests>"}
_clears = {"n": 0, "s": 0.0}


def _listener(event, duration_secs, **kwargs):
    key = _EVENTS.get(event)
    if key is None:
        return
    s = _stats[_current["file"]]
    s[key] += duration_secs
    if key == "backend_s":
        s["n_backend"] += 1


def pytest_configure(config):
    import jax

    jax.monitoring.register_event_duration_secs_listener(_listener)


def pytest_runtest_setup(item):
    f = item.nodeid.split("::")[0]
    if f != _current["file"]:
        _current["file"] = f
        _order.append(f)
        if _MODE == "clear":
            import jax

            t0 = time.perf_counter()
            jax.clear_caches()
            _clears["n"] += 1
            _clears["s"] += time.perf_counter() - t0


def pytest_runtest_logreport(report):
    f = report.nodeid.split("::")[0]
    _stats[f]["wall_s"] += report.duration
    if report.when == "call":
        _stats[f]["n_tests"] += 1
        if report.failed:
            _stats[f]["n_failed"] += 1


def pytest_sessionfinish(session, exitstatus):
    out = {
        "mode": _MODE,
        "clears": _clears,
        "order": _order,
        "files": {f: dict(v) for f, v in _stats.items()},
    }
    os.makedirs(os.path.dirname(_OUT) or ".", exist_ok=True)
    with open(_OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    tot = collections.Counter()
    for v in _stats.values():
        for k, x in v.items():
            tot[k] += x
    sys.stdout.write(
        f"\n[compile_probe] mode={_MODE} files={len(_order)} "
        f"backend_compiles={int(tot['n_backend'])} "
        f"trace={tot['trace_s']:.1f}s lower={tot['lower_s']:.1f}s "
        f"backend={tot['backend_s']:.1f}s wall={tot['wall_s']:.1f}s "
        f"clears={_clears['n']} ({_clears['s']:.2f}s) -> {_OUT}\n"
    )
