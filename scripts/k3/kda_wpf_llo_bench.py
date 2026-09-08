#!/usr/bin/env python3
"""LLO-traced microbench: the fused KDA core with and without the o_proj
weight prefetch (T=1, K3 TP=32 per-rank shapes).

The question: the fused KDA kernel measured 9.0 -> 10.3 us when the
prefetch rides it (job 569 vs 377). This bench captures the dense LLO trace
per variant so the +1.3 us can be attributed (bundle timeline +
instruction mix), with wall-clock timings alongside.

Variants: stock (no prefetch), wpf (DMA at kernel entry), wpf_late (DMA
after the pipeline). Runs the production kernel from
src/vllm_torchtpu/kernels/gdn/v3/wrapper.py -- the probe venv imports it
fine with sys.path (the package __init__ is light).
"""

import os
import sys
import time

# These slices boot libtpu in multislice mode: a JAX process that does not
# join the distributed runtime hangs in backend init waiting for the other
# hosts. Join on every task.
if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    import jax
    jax.distributed.initialize()

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.layout import Format, Layout
from jax.sharding import SingleDeviceSharding

_REPO = os.environ.get(
    "KIMI_K3_REPO_DIR",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_REPO, "src"))

from vllm_torchtpu.kernels.gdn.v3 import config  # noqa: E402
from vllm_torchtpu.kernels.gdn.v3 import wrapper  # noqa: E402


def _stamp(rank, msg):
    print(f"[rank {rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _inputs():
    """T=1 decode, K3 per-rank KDA shapes: n_v=3, d_k=d_v=128."""
    n_kq = n_v = 3
    d_k = d_v = 128
    kernel_size = 4
    dim_size = 2 * n_kq * d_k + n_v * d_v

    rngs = iter(jax.random.split(jax.random.key(0), 9))
    qkv = jax.random.normal(next(rngs), (1, dim_size), jnp.bfloat16)
    b = jax.random.normal(next(rngs), (1, n_v), jnp.bfloat16)
    a = jax.random.normal(next(rngs), (1, n_v * d_k), jnp.bfloat16)
    conv_state = jax.random.normal(next(rngs), (2, kernel_size - 1, dim_size),
                                   jnp.bfloat16) * 0.1
    recurrent_state = jax.random.normal(next(rngs),
                                        (2, n_v, d_k, d_v), jnp.float32) * 0.05
    conv_weight = jax.random.normal(next(rngs), (dim_size, 1, kernel_size),
                                    jnp.bfloat16)
    a_log = jax.random.normal(next(rngs), (n_v, ), jnp.float32)
    dt_bias = jax.random.normal(next(rngs), (n_v, d_k), jnp.float32)
    w_o = jax.random.normal(next(rngs), (384, 7168), jnp.bfloat16)

    return dict(
        qkv=qkv,
        b=b,
        a=a,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        conv_weight=conv_weight,
        conv_bias=None,
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=jnp.array([0, 1], jnp.int32),
        state_indices=jnp.array([1], jnp.int32),
        distribution=jnp.array([1, 1, 1], jnp.int32),
        seq_lens=jnp.array([5], jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        attention_mode=config.AttentionMode.KDA,
        gate_lower_bound=-5.0,
    ), w_o


def main() -> int:
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ok = True
    dev = jax.local_devices()[0]
    kwargs, w_o = _inputs()

    # The prefetch DMA inherits the HBM layout of w_o; an untiled {1,0}
    # source makes the HBM->tiled-VMEM transfer strided (~0.4 TB/s,
    # measured 13.7 us for 5.5 MB in this bench). Pinning to the VMEM
    # tile order makes it linear.
    w_o_pinned = jax.device_put(
        w_o,
        Format(Layout(major_to_minor=(0, 1), tiling=((8, 128), (2, 1))),
               SingleDeviceSharding(dev)))
    variants = {
        "stock": dict(w_o_prefetch=None, wpf_late=False),
        "wpf": dict(w_o_prefetch=w_o, wpf_late=False),
        "wpf_late": dict(w_o_prefetch=w_o, wpf_late=True),
        "wpf_pinned": dict(w_o_prefetch=w_o_pinned, wpf_late=False),
    }
    only = os.environ.get("WPB_VARIANTS")
    if only:
        variants = {k: v for k, v in variants.items() if k in only.split(",")}

    prof_root = os.environ.get("WPB_PROFILE")
    ref = None
    with jax.default_device(dev):
        static_names = {
            "n_kq", "n_v", "d_k", "d_v", "kernel_size", "attention_mode",
            "gate_lower_bound", "wpf_late"
        }
        for name, extra in variants.items():
            # Re-jit the RAW function (._fun skips the wrapper's donation,
            # which would eat the state buffers after one call and break the
            # timing loop). Statics are baked at lower(); the Compiled call
            # takes only the traced args.
            fn = jax.jit(wrapper.fused_conv1d_gdn._fun,
                         static_argnames=list(static_names))
            all_args = {**kwargs, **extra}
            t0 = time.time()
            compiled = fn.lower(**all_args).compile()
            _stamp(rank,
                   f"{name}: lowered+compiled in {time.time() - t0:.1f}s")
            call_args = {
                k: v
                for k, v in all_args.items() if k not in static_names
            }

            result = compiled(**call_args)
            if extra["w_o_prefetch"] is None:
                (_, _), out = result
                wv = None
            else:
                (_, _), out, wv = result
            jax.block_until_ready(out)
            if ref is None:
                ref = np.asarray(out, np.float64)
            else:
                np.testing.assert_allclose(np.asarray(out, np.float64),
                                           ref,
                                           rtol=2e-2,
                                           atol=5e-3)
            if wv is not None:
                np.testing.assert_array_equal(np.asarray(wv), np.asarray(w_o))
            _stamp(rank, f"{name}: numerics ok")

            for _ in range(5):
                out = compiled(**call_args)
            jax.block_until_ready(out[0] if isinstance(out, tuple) else out)
            t0 = time.time()
            for _ in range(100):
                result = compiled(**call_args)
            jax.block_until_ready(out)
            _stamp(
                rank, f"{name}: {(time.time() - t0) / 100 * 1e6:.2f} "
                "us/call (host dispatch included)")

            if prof_root:
                from jax import profiler
                tdir = os.path.join(prof_root, f"{name}/rank{rank}")
                profiler.start_trace(tdir)
                for _ in range(50):
                    result = compiled(**call_args)
                jax.block_until_ready(out)
                profiler.stop_trace()
                _stamp(rank, f"{name}: trace captured -> {tdir}")

    _stamp(rank, "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
