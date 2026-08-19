# SPDX-License-Identifier: Apache-2.0
"""The sized slab overflows into an out-of-bounds DMA, and the default avoids it.

A slab smaller than the drop-free bound is only safe if a shard whose experts
overflow it drops the excess. It does not -- it writes past the end of the
buffer, which the Pallas interpreter catches as an out-of-bounds write and a
chip reports as an unexpected core halt with leaked semaphores.

Why no existing test saw it: the sizing formula takes ``min(sized, drop-free)``
and its floor is ``(ROWBLK - 1) * num_experts + CAPACITY``, which dominates at
every shape our suite executes -- so the slab our tests run with is *always* the
drop-free one, and the drop path never runs. Serving at 16,384 node-wide tokens
was the first place it did.

The first test pins the defect so the fix has a target; the second pins that the
shipped default cannot reach it.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# Same two fixtures as the numerics gate, deliberately copied rather than
# shared: that module forces the process device count at import, and moving
# either of them into a conftest would change when that happens for every file
# in this directory. Four devices here, not eight -- this is about one shard's
# slab, not about mesh order.
_EP_DEVICES = 4


@pytest.fixture(scope="module", autouse=True)
def cpu_device_pool():
    """Force the process's device count once, before anything here traces."""
    devices = jax.devices("cpu")
    if len(devices) < _EP_DEVICES:
        # Force it the way the numerics gate does, and fail rather than skip:
        # a slab test that quietly does not run is worse than no slab test.
        # Not `import jax.extend`: that binds `jax` as a local for the whole
        # function and the line above stops seeing the module.
        from jax.extend import backend as jax_backend

        jax_backend.clear_backends()
        jax.config.update("jax_num_cpu_devices", _EP_DEVICES)
        devices = jax.devices("cpu")
    if len(devices) < _EP_DEVICES:
        pytest.fail(
            f"asked for {_EP_DEVICES} CPU devices, got {len(devices)}; "
            "do not turn this into a skip")
    return tuple(devices[:_EP_DEVICES])


@pytest.fixture
def tpu7x(monkeypatch):
    """Name the chip the kernel's VMEM accounting reads, on a host with none."""
    from jax._src.pallas.mosaic import tpu_info
    from jax.experimental.pallas import tpu as pltpu

    info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(tpu_info, "get_tpu_info", lambda: info)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    return pltpu


EP, EXPERTS, TOP_K, HIDDEN, INTER = 4, 16, 2, 256, 128
G_LOCAL = EXPERTS // EP


def _concentrated_inputs(num_tokens, seed=0):
    """Every token's top-k on shard 0's experts: the worst case for one slab."""
    rng = np.random.default_rng(seed)
    fp8 = jnp.float8_e4m3fn
    logits = np.full((num_tokens, EXPERTS), -30.0, np.float32)
    logits[:, :G_LOCAL] = rng.normal(0, 1, (num_tokens, G_LOCAL))
    return dict(
        x=jnp.asarray(
            rng.normal(0, 1, (num_tokens, HIDDEN)) / 4, jnp.bfloat16),
        gating=jnp.asarray(logits, jnp.bfloat16),
        w1=jnp.asarray(rng.normal(0, 1, (EXPERTS, HIDDEN, 2 * INTER)) *
                       0.1).astype(fp8),
        w2=jnp.asarray(rng.normal(0, 1,
                                  (EXPERTS, INTER, HIDDEN)) * 0.1).astype(fp8),
        w1s=jnp.asarray(rng.uniform(0.5, 1.5, (EXPERTS, 1, 1, 2 * INTER)),
                        jnp.float32),
        w2s=jnp.asarray(rng.uniform(0.5, 1.5, (EXPERTS, 1, 1, HIDDEN)),
                        jnp.float32),
    )


def _run(devices, tpu7x, num_tokens, skew):
    import functools

    from jax.sharding import AxisType, Mesh

    from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME_1D
    from vllm_torchtpu.layers.common.fused_moe_ep_v2 import \
        fused_ep_moe_v2_func

    op = _concentrated_inputs(num_tokens)
    mesh = Mesh(np.asarray(devices[:EP]).reshape(EP),
                axis_names=(EP_AXIS_NAME_1D, ),
                axis_types=(AxisType.Explicit, ))
    fn = functools.partial(fused_ep_moe_v2_func,
                           mesh=mesh,
                           top_k=TOP_K,
                           renormalize=True,
                           activation="silu",
                           capacity_skew=skew)
    with tpu7x.force_tpu_interpret_mode():
        out = jax.jit(fn)(op["x"], op["gating"], op["w1"], op["w2"], op["w1s"],
                          op["w2s"])
    return np.asarray(jax.device_get(out), np.float32)


def _slabs(num_tokens, skew):
    """(what this skew allocates, the drop-free bound). ``_slab_rows`` is the
    driver's own chooser, so skew 0 means the bound rather than an error."""
    from vllm_torchtpu.kernels.fused_moe.v2.host import ragged_stride_bound
    from vllm_torchtpu.layers.common.fused_moe_ep_v2 import (CAPACITY,
                                                             _slab_rows)

    return (_slab_rows(num_tokens, TOP_K, EXPERTS, EP, skew),
            ragged_stride_bound(num_tokens, TOP_K, EXPERTS, CAPACITY))


# Order matters: the overflow below aborts a DMA inside the interpreter and
# leaves its shared-memory bookkeeping inconsistent for anything that runs
# after it in the same process, so the clean case goes first.
def test_the_shipped_default_never_takes_that_path(cpu_device_pool, tpu7x):
    """The drop-free slab holds every pair by construction, concentrated or not.

    This is the guarantee the default buys, and it is why the default is 0.0
    until the drop path works: the same routing that overflows above runs here.
    """
    from vllm_torchtpu.layers.common.fused_moe_ep_v2 import CAPACITY_SKEW

    assert CAPACITY_SKEW == 0.0, (
        "the default slab is sized again -- re-validate the drop path first: "
        "the test above is what it has to survive")
    sized, drop_free = _slabs(256, CAPACITY_SKEW)
    assert sized == drop_free

    out = _run(cpu_device_pool, tpu7x, 256, CAPACITY_SKEW)
    assert out.shape == (256, HIDDEN)
    assert np.isfinite(out).all()


def test_a_slab_smaller_than_the_routing_writes_out_of_bounds(
        cpu_device_pool, tpu7x):
    """The defect, pinned. Delete this test when the drop path drops.

    ``skew=0.25`` is only a way to reach a state the shipped 3.0 reaches at
    serving size: what matters is ``sized < rows a shard is asked to hold``,
    and this is the smallest shape that gets there in an interpreter.
    """
    sized, drop_free = _slabs(256, 0.25)
    assert sized < drop_free, "this shape no longer exercises a sized slab"
    assert 256 * TOP_K > sized, "the routing no longer overflows the slab"

    with pytest.raises(Exception, match="[Oo]ut-of-bounds"):
        _run(cpu_device_pool, tpu7x, 256, 0.25)
