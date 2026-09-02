"""Shared timing helper for the TPU microbenchmarks in this directory.

Timing method: dispatch a jitted callable ``iters`` times asynchronously and
block once at the end. That only measures device time if device time per call
exceeds the Python dispatch cost, so every run also measures a *dispatch floor*
-- the same loop over a trivial jitted op -- and prints it. Any reported number
within ~2x of the floor is dispatch-bound, not kernel-bound, and is labelled so.
"""

import time

import jax
import jax.numpy as jnp

_FLOOR_US = None


def _time_loop(fn, args, iters):
    out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1e6


def _compiled(fn, args):
    """Pre-compiled executable: dispatches faster than a fresh jit call."""
    try:
        return jax.jit(fn).lower(*args).compile()
    except Exception:  # noqa: BLE001
        return jax.jit(fn)


def dispatch_floor(dev, iters=2000):
    """us/call for a jitted op with negligible device work."""
    global _FLOOR_US
    if _FLOOR_US is None:
        x = jax.device_put(jnp.ones((8, 128), jnp.float32), dev)
        f = _compiled(lambda a: a + 1.0, (x, ))
        _FLOOR_US = _time_loop(f, (x, ), iters)
    return _FLOOR_US


def bench(name, fn, args, dev, iters=200, reps=3, notes=""):
    """Compile, run, and report min-of-``reps`` mean us/call."""
    try:
        t_compile0 = time.perf_counter()
        f = _compiled(fn, args)
        out = f(*args)
        jax.block_until_ready(out)
        compile_s = time.perf_counter() - t_compile0
        times = [_time_loop(f, args, iters) for _ in range(reps)]
        best = min(times)
        floor = dispatch_floor(dev)
        flag = "  <-- DISPATCH-BOUND, not a kernel time" if best < 2 * floor else ""
        spread = (max(times) - best) / best * 100.0
        print(f"RESULT {name:<38s} {best:9.2f} us/call  "
              f"(spread {spread:5.1f}%, compile {compile_s:5.1f}s, "
              f"iters={iters}x{reps}){flag}"
              + (f"   # {notes}" if notes else ""),
              flush=True)
        return best
    except Exception as e:  # noqa: BLE001 - a variant failing must not stop the run
        msg = str(e).replace("\n", " | ")[:600]
        print(f"RESULT {name:<38s} FAILED: {type(e).__name__}: {msg}",
              flush=True)
        return None


def banner(title):
    print(f"\n===== {title} =====", flush=True)


def env_report():
    import jax as _jax
    print(f"jax {_jax.__version__}", flush=True)
    try:
        import jaxlib
        print(f"jaxlib {jaxlib.__version__}", flush=True)
    except Exception:
        pass
    devs = _jax.devices()
    print(f"devices ({len(devs)}): {devs}", flush=True)
    d = devs[0]
    for attr in ("device_kind", "platform", "coords", "core_on_chip",
                 "num_cores", "memory_stats"):
        try:
            v = getattr(d, attr)
            print(f"  device.{attr} = {v() if callable(v) else v}", flush=True)
        except Exception as e:
            print(f"  device.{attr} -> {type(e).__name__}", flush=True)
    return d
