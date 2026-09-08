#!/usr/bin/env python3
"""VMEM chaining / w-prefetch probe (single-device per rank, no collectives).

Context: the fused o_proj+AR kernel stages its weight shard w [7168, 224]
from HBM into VMEM on every call (~4.6 us, jobs 462/499/501/502), and the
dump shows why: w arrives in HBM as {0,1:T(8,128)} while the kernel's VMEM
operand is {1,0:T(8,128)(2,1)S(1)} -- the staging "copy" is a dimension-
order transpose, which is also presumably why XLA emits it as a blocking
copy while the small same-layout workspace operands got ASYNC
copy-start/copy-done pairs.

Questions under test (answers come from the HLO dumps + traces, not
assumptions):

1. chained: does a pallas kernel's VMEM out_spec result feed a second
   pallas kernel's VMEM in_spec directly, or does XLA roundtrip/retile it?
2. prelaid: if w's HBM layout matches the VMEM tiling (jax.experimental
   .layout.Layout), does the staging become a pure async DMA
   (copy-start/copy-done, like the rs/ag workspaces)?
3. prefetch / prefetch_first: with an unrelated compute kernel C between
   the stage and the consumer, does the scheduler run the stage early
   enough to overlap C (the "prefetch w during attention" pattern)?

Run under srun on a 4-host ici-2x2x4 slice; each rank tests its own local
device only (no collectives), so all variants share one job safely.
"""

import os
import sys
import time

# These slices boot libtpu in multislice mode: a JAX process that does not
# join the distributed runtime hangs in backend init waiting for the other
# hosts ("Did you run your code on all TPU hosts?"). Join on every task.
if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    import jax
    jax.distributed.initialize()

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.layout import Format, Layout
from jax.experimental.pallas import tpu as pltpu

HID, SHARD = 7168, 224  # w is [7168, 224]: 3.2 MiB bf16

# Variants that perturb x or otherwise break the plain o_proj numerics;
# scheduling/timing is their signal.

# The VMEM layout the o_proj dot's w operand gets in production (measured in
# the fused-check dump, job 505): {1,0:T(8,128)(2,1)S(1)} -- XLA layout
# strings are MINOR-to-MAJOR, so {1,0} is major_to_minor=(0, 1). (An earlier
# revision of this probe pinned (1, 0) -- the layout w already had -- and
# was a no-op.)
W_LAYOUT = Layout(major_to_minor=(0, 1), tiling=((8, 128), (2, 1)))


def _stamp(rank, msg):
    print(f"[rank {rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stage_to_vmem(w):
    """Kernel A: pass-through HBM -> VMEM-outspec staging kernel."""

    def kernel(w_ref, o_ref):
        o_ref[...] = w_ref[...]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(w.shape, w.dtype),
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        compiler_params=pltpu.CompilerParams(has_side_effects=False),
    )(w)


def oproj(x, w):
    """Kernel B: the o_proj dot consuming w from a VMEM in_spec."""

    def kernel(x_ref, w_ref, o_ref):
        acc = jax.lax.dot_general(x_ref[...],
                                  w_ref[...],
                                  dimension_numbers=(((1, ), (1, )), ((), ())),
                                  preferred_element_type=jnp.float32)
        o_ref[...] = acc.astype(o_ref.dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, HID), x.dtype),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        compiler_params=pltpu.CompilerParams(has_side_effects=False),
    )(x, w)


def standin(xb):
    """Kernel C: attention stand-in -- 48 dependent elementwise passes over
    a [7168, 224] VMEM buffer (~8-10 us of VMEM-bound work, enough to hide
    the ~4.5 us w staging behind)."""

    def kernel(x_ref, o_ref):
        v = x_ref[...]
        for _ in range(48):
            v = v * jnp.bfloat16(1.0001) + jnp.bfloat16(0.001)
        o_ref[...] = v

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(xb.shape, xb.dtype),
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        compiler_params=pltpu.CompilerParams(has_side_effects=False),
    )(xb)


def standin_xla(xb):
    """Kernel C as plain XLA (no pallas): same dependent elementwise passes,
    so the latency hider sees a fusion rather than a custom call."""
    v = xb
    for _ in range(48):
        v = v * jnp.bfloat16(1.0001) + jnp.bfloat16(0.001)
    return v


def standin_xla_dep(xb):
    """standin_xla with tanh so XLA cannot fold the 48 affine passes into
    one; output feeds the consumer (the production dependency shape)."""
    v = xb
    for _ in range(48):
        v = jnp.tanh(v) * jnp.bfloat16(1.0001)
    return v


def standin_big(xb):
    """The pallas stand-in at 192 passes (~34 us), so a hoisted w copy-start
    has a wide, unambiguous window to overlap with."""

    def kernel(x_ref, o_ref):
        v = x_ref[...]
        for _ in range(192):
            v = v * jnp.bfloat16(1.0001) + jnp.bfloat16(0.001)
        o_ref[...] = v

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(xb.shape, xb.dtype),
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        compiler_params=pltpu.CompilerParams(has_side_effects=False),
    )(xb)


_TIMING_ONLY = {"prelaid_dep", "prelaid_dep_pallas", "prelaid_dep_2layer"}


def _two_layer(x, xb, w1, w2):
    """C1 -> B1 -> C2 -> B2: two chained attention/oproj-ish layers."""
    h1 = standin_xla_dep(xb)
    o1 = oproj(x + h1[0:1, :], w1)
    h2 = standin_xla_dep(h1)
    o2 = oproj(x + h2[0:1, :], w2)
    return o1, o2


def attn_with_prefetch(xb, w):
    """Kernel A: attention stand-in that ALSO prefetches w.

    Takes w as an HBM ref (pl.ANY -- no XLA staging), starts an async
    HBM->VMEM DMA of it into a VMEM out ref at kernel ENTRY, does the
    stand-in compute, and waits the DMA just before exit. The DMA runs
    under the compute; the VMEM w buffer is the kernel's second output and
    chains to the consumer (pallas->pallas VMEM chaining is glueless, jobs
    511/512). This is the scheduler-proof prefetch: placement is explicit
    in the program, not delegated to XLA's latency hider.
    """

    def kernel(xb_ref, w_ref, h_ref, wv_ref, sem):
        pf = pltpu.make_async_copy(src_ref=w_ref, dst_ref=wv_ref, sem=sem)
        pf.start()
        v = xb_ref[...]
        for _ in range(48):
            v = v * jnp.bfloat16(1.0001) + jnp.bfloat16(0.001)
        h_ref[...] = v
        pf.wait()

    return pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct(xb.shape, xb.dtype),
                   jax.ShapeDtypeStruct(w.shape, w.dtype)),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            # HBM ref: the kernel does its own DMA, no XLA staging of w.
            # (pl.ANY lets XLA choose -- and it chooses VMEM staging, which
            # defeats the purpose; measured in job 530's dump.)
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        scratch_shapes=[pltpu.SemaphoreType.DMA],
        compiler_params=pltpu.CompilerParams(has_side_effects=False),
    )(xb, w)


def attn_with_prefetch_nowait(xb, w):
    """attn_with_prefetch WITHOUT the pf.wait(): does an async HBM->VMEM DMA
    started in-kernel drain/fence at kernel exit (consumer sees the data),
    or does the consumer race it? Numerics across 250 timed iterations are
    the signal. Needs disable_semaphore_checks (pending DMA at exit is a
    semaphore-check violation by construction)."""

    def kernel(xb_ref, w_ref, h_ref, wv_ref, sem):
        pf = pltpu.make_async_copy(src_ref=w_ref, dst_ref=wv_ref, sem=sem)
        pf.start()
        v = xb_ref[...]
        for _ in range(48):
            v = v * jnp.bfloat16(1.0001) + jnp.bfloat16(0.001)
        h_ref[...] = v
        del pf  # no wait

    return pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct(xb.shape, xb.dtype),
                   jax.ShapeDtypeStruct(w.shape, w.dtype)),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        scratch_shapes=[pltpu.SemaphoreType.DMA],
        compiler_params=pltpu.CompilerParams(has_side_effects=False,
                                             disable_semaphore_checks=True),
    )(xb, w)


def oproj_stream(x, w):
    """o_proj with w as RAW HBM blocks: grid over the 28 output tiles; each
    iteration explicitly DMAs its [256,224] w block HBM->VMEM and dots it.

    The grid-free probes (jobs 530/534) showed XLA stages every whole-tensor
    custom-call operand to VMEM regardless of the BlockSpec memory_space
    (pl.ANY and pltpu.HBM alike) -- so an in-kernel HBM->VMEM DMA of w is
    impossible in that form. The gridded/blocked form is how pallas exposes
    raw HBM refs; the dump answers whether it escapes XLA staging here.
    Numerics are checkable: equals x @ w.T.
    """

    def kernel(x_ref, w_ref, o_ref, w_vmem, sem):
        cp = pltpu.make_async_copy(src_ref=w_ref, dst_ref=w_vmem, sem=sem)
        cp.start()
        cp.wait()
        acc = jax.lax.dot_general(x_ref[...],
                                  w_vmem[...],
                                  dimension_numbers=(((1, ), (1, )), ((), ())),
                                  preferred_element_type=jnp.float32)
        o_ref[...] = acc.astype(o_ref.dtype)

    return pl.pallas_call(
        kernel,
        grid=(HID // 256, ),
        in_specs=[
            pl.BlockSpec(block_shape=(1, SHARD),
                         index_map=lambda i: (0, 0),
                         memory_space=pltpu.VMEM),
            pl.BlockSpec(block_shape=(256, SHARD),
                         index_map=lambda i: (i, 0),
                         memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec(block_shape=(1, 256),
                               index_map=lambda i: (0, i),
                               memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct((1, HID), x.dtype),
        scratch_shapes=[
            pltpu.VMEM((256, SHARD), x.dtype), pltpu.SemaphoreType.DMA
        ],
        compiler_params=pltpu.CompilerParams(has_side_effects=False),
    )(x, w)


def main() -> int:
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ok = True
    dev = jax.local_devices()[0]

    with jax.default_device(dev):
        x = jax.random.normal(jax.random.key(0), (1, SHARD), jnp.bfloat16)
        xb = jax.random.normal(jax.random.key(2), (HID, SHARD), jnp.bfloat16)
        w = jax.random.normal(jax.random.key(1), (HID, SHARD), jnp.bfloat16)
        w_pre = jax.device_put(
            w, Format(W_LAYOUT, jax.sharding.SingleDeviceSharding(dev)))
        w2 = jax.random.normal(jax.random.key(7), (HID, SHARD), jnp.bfloat16)
        w2_pre = jax.device_put(
            w2, Format(W_LAYOUT, jax.sharding.SingleDeviceSharding(dev)))
        _stamp(rank, f"w layout: {jax.typeof(w)}; w_pre: {jax.typeof(w_pre)}")

        want = np.asarray(
            jax.jit(lambda a, b: jnp.dot(a, b.T).astype(jnp.float32))(x, w))
        scale = max(float(np.max(np.abs(want))), 1e-6)

        # NOTE: w must be an ARGUMENT, not a closure capture -- captured
        # arrays become XLA constants whose layouts XLA re-assigns freely,
        # which silently discards the pinned layout (measured: the constant
        # got {1,0:T(16,128)(2,1)} regardless of the device_put). And the
        # stand-in's output must be RETURNED, or the jaxpr drops it (DCE).
        variants = {
            # The current shape: w staged inside the consumer's call.
            "baseline": (lambda x, w: oproj(x, w), (x, w)),
            # w pre-laid-out in HBM to the kernel's VMEM tiling.
            "prelaid": (lambda x, w: oproj(x, w), (x, w_pre)),
            # pallas->pallas VMEM chaining: A's VMEM output feeds B.
            "chained": (lambda x, w: oproj(x, stage_to_vmem(w)), (x, w)),
            # Stand-in compute C traced BEFORE the stage+consumer (program
            # order C, A, B); h is returned to keep C live. prefetch_first
            # traces A first instead (A, C, B). Whether XLA's scheduler
            # honors or overrides that order is part of the answer.
            "prefetch": (lambda x, xb, w: (lambda h:
                                           (oproj(x, stage_to_vmem(w)), h))
                         (standin(xb)), (x, xb, w)),
            "prefetch_first": (lambda x, xb, w: (lambda wv:
                                                 (oproj(x, wv), standin(xb)))
                               (stage_to_vmem(w)), (x, xb, w)),
            # prelaid + C traced first: does XLA's latency hiding hoist the
            # (now async?) w staging over C on its own?
            "prelaid_prefetch": (lambda x, xb, w: (lambda h: (oproj(x, w), h))
                                 (standin(xb)), (x, xb, w_pre)),
            # Same, but C is a plain XLA fusion instead of a pallas custom
            # call: is the hoisting blocker the custom-call boundary?
            "prelaid_prefetch_xla": (lambda x, xb, w: (lambda h:
                                                       (oproj(x, w), h))
                                     (standin_xla(xb)), (x, xb, w_pre)),
            # The production dependency shape: o_proj CONSUMES C's output
            # (attention -> o_proj), w staging independent. tanh keeps the
            # 48-pass chain from constant-folding into one affine op.
            # Timing/scheduling only -- numerics n/a (x is perturbed by C).
            "prelaid_dep":
            (lambda x, xb, w: oproj(x + standin_xla_dep(xb)[0:1, :], w),
             (x, xb, w_pre)),
            # Same dependency shape but C is a PALLAS custom call (like the
            # real attention kernel) -- prelaid_dep hoisted the w copy-start
            # over a 40 us XLA fusion (job 515); does it over a custom call?
            "prelaid_dep_pallas":
            (lambda x, xb, w: oproj(x + standin_big(xb)[0:1, :], w), (x, xb,
                                                                      w_pre)),
            # Two chained layers (C1 -> B1 -> C2 -> B2), the e2e structure:
            # is w2's copy-start still hoisted over C2 when layer 1's tail
            # is in the way?
            "prelaid_dep_2layer": (_two_layer, (x, xb, w_pre, w2_pre)),
            # THE PATTERN: the attention stand-in kernel itself issues the
            # async HBM->VMEM w DMA at entry and waits at exit; o_proj
            # consumes the VMEM buffer directly (no XLA staging of w at
            # all). inkernel uses the unpinned {0,1} HBM w, inkernel_pre the
            # pinned {1,0:T(8,128)(2,1)} one -- testing the in-kernel DMA's
            # layout tolerance both ways. Compare against prelaid_prefetch
            # (same work, XLA-managed staging, no hoisting).
            "inkernel": (lambda x, xb, w: (lambda hw: (oproj(x, hw[1]), hw[0]))
                         (attn_with_prefetch(xb, w)), (x, xb, w)),
            "inkernel_pre": (lambda x, xb, w: (lambda hw:
                                               (oproj(x, hw[1]), hw[0]))
                             (attn_with_prefetch(xb, w)), (x, xb, w_pre)),
            # No in-kernel wait on the prefetch DMA: tests whether kernel
            # exit drains outstanding DMAs (numerics across the timing loop
            # are the signal -- a race would produce wrong numbers).
            "inkernel_nowait":
            (lambda x, xb, w: (lambda hw: (oproj(x, hw[1]), hw[0]))
             (attn_with_prefetch_nowait(xb, w)), (x, xb, w_pre)),
            # The intended API for this: with_memory_space_constraint marks
            # the ARGUMENT's aval HBM, which pallas reads (not the
            # BlockSpec!) when it sets the custom call's
            # input_memory_space_colors -- the colors are what libtpu's
            # staging consults. If honored: no staging copies at all.
            "wmark":
            (lambda x, xb, w: (lambda hw: (oproj(x, hw[1]), hw[0]))
             (attn_with_prefetch(
                 xb, pltpu.with_memory_space_constraint(w, pltpu.HBM))),
             (x, xb, w)),
            # wmark + pinned layout: w arrives in HBM already in the
            # kernel's {1,0:T(8,128)(2,1)} layout, so no staging of any kind
            # should be needed (job 539 showed the HBM mark alone just moves
            # the retile: HBM->VMEM->HBM for an unpinned w).
            "wmark_pre":
            (lambda x, xb, w: (lambda hw: (oproj(x, hw[1]), hw[0]))
             (attn_with_prefetch(
                 xb, pltpu.with_memory_space_constraint(w, pltpu.HBM))),
             (x, xb, w_pre)),
            # Gridded o_proj reading w from raw HBM blocks (explicit
            # in-kernel DMA per block): does the operand escape XLA's VMEM
            # staging? And what does streamed-vs-staged cost at T=1?
            "stream": (lambda x, w: oproj_stream(x, w), (x, w)),
            "stream_pre": (lambda x, w: oproj_stream(x, w), (x, w_pre)),
        }

        prof_root = os.environ.get("VMEM_PROBE_PROFILE")

        only = os.environ.get("VMEM_PROBE_VARIANTS")
        if only:
            keep = set(only.split(","))
            variants = {k: v for k, v in variants.items() if k in keep}

        for name, (f, args) in variants.items():
            fn = jax.jit(f)
            t0 = time.time()
            compiled = fn.lower(*args).compile()
            _stamp(rank,
                   f"{name}: lowered+compiled in {time.time() - t0:.1f}s")
            out = compiled(*args)
            if name in _TIMING_ONLY:
                _stamp(rank, f"{name}: (timing/scheduling only, numerics "
                       "n/a)")
            else:
                out0 = out[0] if isinstance(out, tuple) else out
                rel = float(
                    np.max(
                        np.abs(
                            np.asarray(out0, jnp.float32).reshape(-1) -
                            want.reshape(-1)))) / scale
                _stamp(rank, f"{name}: rel err = {rel:.3g}")
                ok = ok and rel < 0.05
            for _ in range(3):
                out = compiled(*args)
            jax.block_until_ready(out)
            t0 = time.time()
            for _ in range(200):
                out = compiled(*args)
            jax.block_until_ready(out)
            _stamp(
                rank, f"{name}: {(time.time() - t0) / 200 * 1e6:.2f} us/call"
                " (host dispatch included)")
            if prof_root:
                from jax import profiler
                tdir = os.path.join(prof_root, f"{name}/rank{rank}")
                profiler.start_trace(tdir)
                for _ in range(50):
                    out = compiled(*args)
                jax.block_until_ready(out)
                profiler.stop_trace()
                _stamp(rank, f"{name}: trace captured -> {tdir}")

    _stamp(rank, "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
