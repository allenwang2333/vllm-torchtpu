"""Probe 1: HBM DMA roofline under RPA-like page patterns, and the current
batched-RPA DCP decode kernel's time split into DMA-only / compute-only floors.

Prints one line per measurement: PROBE {json}.
"""

import functools
import json
import os
import sys
import time
import types

REPO = os.environ.get("REPO", "/workspace/vllm-torchtpu/src")
for _name, _sub in [("vllm_torchtpu", ""), ("vllm_torchtpu.kernels", "kernels")]:
    _m = types.ModuleType(_name)
    _m.__path__ = [os.path.join(REPO, "vllm_torchtpu", _sub)]
    sys.modules[_name] = _m

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devtime  # noqa: E402


def emit(**kw):
    print("PROBE " + json.dumps(kw), flush=True)


print("jax", jax.__version__, "jaxlib", getattr(jax, "__version_info__", None), flush=True)
try:
    import jaxlib

    print("jaxlib", jaxlib.__version__, flush=True)
except Exception as e:  # noqa: BLE001
    print("jaxlib?", e)
print("devices", jax.devices(), flush=True)
info = pltpu.get_tpu_info()
emit(kind="tpu_info", **{k: str(getattr(info, k)) for k in dir(info) if not k.startswith("_")})

SECTIONS = os.environ.get("PROBE_SECTIONS", "dma,rpa").split(",")

# ----------------------------------------------------------------------------
# DMA roofline: u32 pages [P, 2, 64, 128] (64 KiB, the fp8 K+V of 128 tokens
# of one KV head with head_dim 256), gathered per step into VMEM.
# ----------------------------------------------------------------------------
PAGE_SHAPE = (2, 64, 128)
PAGE_BYTES = 2 * 64 * 128 * 4


def gather_call(n_steps, lanes, ppi, layout, run, slots):
    if layout == "lane":
        buf_shape = (slots, lanes, 2, 64, ppi * 128)
    else:
        buf_shape = (slots, lanes, ppi, 2, 64, 128)

    def kernel(pidx_ref, kv_hbm, o_ref, buf, sem):
        def issue(step, slot):
            for b in range(lanes):
                base = (step * lanes + b) * ppi
                for i in range(0, ppi, run):
                    p = pidx_ref[base + i]
                    if layout == "lane":
                        src = kv_hbm.at[p]
                        dst = buf.at[slot, b, :, :, pl.ds(i * 128, 128)]
                    else:
                        src = kv_hbm.at[pl.ds(p, run)]
                        dst = buf.at[slot, b, pl.ds(i, run)]
                    pltpu.make_async_copy(src, dst, sem.at[slot]).start()

        def wait(slot):
            pltpu.make_async_copy(buf.at[slot], buf.at[slot], sem.at[slot]).wait()

        for s0 in range(min(slots - 1, n_steps)):
            issue(s0, s0)

        @pl.loop(0, n_steps)
        def _(s):
            slot = lax.rem(s, slots)
            wait(slot)

            @pl.when(s + slots - 1 < n_steps)
            def _():
                issue(s + slots - 1, lax.rem(s + slots - 1, slots))

        o_ref[0] = pidx_ref[0]

    def f(pidx, kv):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
                out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
                scratch_shapes=[
                    pltpu.VMEM(buf_shape, jnp.uint32),
                    pltpu.SemaphoreType.DMA((slots,)),
                ],
            ),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=62 << 20),
            name=f"gather_{layout}_r{run}_l{lanes}_s{slots}",
        )(pidx, kv)

    return jax.jit(f)


def dma_section():
    total_pages_target = 320 * 32
    rng = np.random.default_rng(0)
    P = total_pages_target + 512
    kv = jax.random.bits(jax.random.PRNGKey(1), (P,) + PAGE_SHAPE, jnp.uint32)
    cases = [
        # name, lanes, ppi, layout, run, slots, order
        ("lane_r1_rand", 8, 32, "lane", 1, 3, "rand"),
        ("lane_r1_seq", 8, 32, "lane", 1, 3, "seq"),
        ("page_r1_rand", 8, 32, "page", 1, 3, "rand"),
        ("page_r2_rand", 8, 32, "page", 2, 3, "rand"),
        ("page_r4_rand", 8, 32, "page", 4, 3, "rand"),
        ("page_r5_rand", 8, 30, "page", 5, 3, "rand"),
        ("page_r8_rand", 8, 32, "page", 8, 3, "rand"),
        ("page_r32_seq", 8, 32, "page", 32, 3, "seq"),
        ("lane_r1_rand_l4s4", 4, 32, "lane", 1, 4, "rand"),
        ("lane_r1_rand_l4s2", 4, 32, "lane", 1, 2, "rand"),
        ("lane_r1_rand_l16s2", 16, 16, "lane", 1, 2, "rand"),
        ("page_r5_rand_l4s4", 4, 30, "page", 5, 4, "rand"),
    ]
    for name, lanes, ppi, layout, run, slots, order in cases:
        n_items = total_pages_target // ppi
        n_items -= n_items % lanes
        n_steps = n_items // lanes
        npg = n_items * ppi
        if order == "seq":
            pidx = np.arange(npg, dtype=np.int32)
        else:
            nruns = npg // run
            starts = rng.permutation(P // run)[:nruns] * run
            pidx = (starts[:, None] + np.arange(run)[None, :]).reshape(-1).astype(np.int32)
        try:
            f = gather_call(n_steps, lanes, ppi, layout, run, slots)
            t, _, _ = devtime.device_times(f, (jnp.asarray(pidx), kv), iters=8)
            us, cnt = devtime.summarize(t, r"^gather_")
            nbytes = npg * PAGE_BYTES
            emit(kind="dma", case=name, us=round(us, 1), n=cnt, MB=round(nbytes / 1e6, 1),
                 GBps=round(nbytes / us / 1e3, 1), dmas=npg // run, steps=n_steps)
        except Exception as e:  # noqa: BLE001
            emit(kind="dma", case=name, error=f"{type(e).__name__}: {str(e)[:600]}")


# ----------------------------------------------------------------------------
# Current kernel: batched RPA DCP CACHE_ONLY decode pass, engine geometry.
# ----------------------------------------------------------------------------
def rpa_section():
    from vllm_torchtpu.kernels.experimental.batched_rpa import configs as C
    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as W

    PAGE, D, T, DCP, QH, CTX = 128, 256, 4, 4, 16, 80000
    max_pages = 515

    def build(n, run=5):
        local = -(-CTX // DCP)
        per = -(-local // PAGE) + 1
        num_pages = n * per * 2 + 64
        shape = W.get_kv_cache_shape(num_pages, PAGE, 1, D, jnp.float8_e4m3fn,
                                     kv_layout=C.KVLayout.SEQ_ALONG_LANE)
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
        kv = (jax.random.normal(k1, shape, jnp.float32) * 0.5).astype(jnp.float8_e4m3fn)
        q = jax.random.normal(k2, (n * T, QH, D), jnp.float32).astype(jnp.bfloat16)
        knew = (jax.random.normal(k3, (n * T, 1, D), jnp.float32) * 0.5).astype(jnp.float8_e4m3fn)
        vnew = (jax.random.normal(k4, (n * T, 1, D), jnp.float32) * 0.5).astype(jnp.float8_e4m3fn)
        kv_lens = jnp.full((n,), CTX + T, jnp.int32)
        pages = np.zeros((n, max_pages), np.int32)
        R = run
        nruns = -(-per // R)
        slots = (num_pages - 1) // R
        starts = np.random.default_rng(0).permutation(slots)[: n * nruns].reshape(n, nruns) * R + 1
        allp = (starts[:, :, None] + np.arange(R)[None, None, :]).reshape(n, nruns * R)[:, :per]
        pages[:, :per] = allp
        cu = jnp.arange(n + 1, dtype=jnp.int32) * T
        dist = jnp.array([n, n, n], jnp.int32)
        return kv, q, knew, vnew, kv_lens, jnp.asarray(pages.reshape(-1)), cu, dist, n * local * 2 * D

    @functools.partial(jax.jit, donate_argnums=(0,))
    def f(kv, q, knew, vnew, kv_lens, pidx, cu, dist):
        out, new_kv, lse = W.ragged_paged_attention(
            q, knew, vnew, kv, kv_lens, pidx, cu, dist,
            sm_scale=1 / 16, k_scale=1.0, v_scale=1.0,
            cp_group_size=DCP, cp_rank=jnp.array([0], jnp.int32),
            attention_scope=C.AttentionScope.CACHE_ONLY, decode_query_size=T,
            return_lse=True, kv_layout=C.KVLayout.SEQ_ALONG_LANE)
        return new_kv, out, lse

    def carry(out, args):
        return (out[0],) + tuple(args[1:])

    for n in [int(x) for x in os.environ.get("PROBE_N", "64,128").split(",")]:
        data = build(n)
        nbytes = data[-1]
        args = data[:-1]
        for mode in os.environ.get("PROBE_RPA_MODES", "base,nocompute,nodma").split(","):
            os.environ["TPU_RPA_DBG_NOCOMPUTE"] = "1" if mode == "nocompute" else "0"
            os.environ["TPU_RPA_DBG_NODMA"] = "1" if mode == "nodma" else "0"
            jax.clear_caches()
            try:
                args = (jnp.copy(args[0]),) + tuple(args[1:])
                t, out, args = devtime.device_times(f, args, iters=8, carry=carry)
                us, cnt = devtime.summarize(t, r"^RPAd")
                sus, scnt = devtime.summarize(t, r"rpa_metadata_schedule")
                top = sorted(((k, round(sum(v) / len(v), 1), len(v)) for k, v in t.items()),
                             key=lambda r: -r[1] * r[2])[:8]
                emit(kind="rpa", n=n, mode=mode, us=round(us, 1), cnt=cnt,
                     GBps=round(nbytes / us / 1e3, 1), sched_us=sus, top=top)
            except Exception as e:  # noqa: BLE001
                emit(kind="rpa", n=n, mode=mode, error=f"{type(e).__name__}: {str(e)[:600]}")
        os.environ["TPU_RPA_DBG_NOCOMPUTE"] = "0"
        os.environ["TPU_RPA_DBG_NODMA"] = "0"


if "dma" in SECTIONS:
    dma_section()
if "rpa" in SECTIONS:
    rpa_section()
print("PROBE_DONE", flush=True)
