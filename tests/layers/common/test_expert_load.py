# SPDX-License-Identifier: Apache-2.0
"""Inertness and trap coverage for the MoE expert-load tap.

`layers/common/expert_load.py` shipped with no tests. Its call sites are
unconditional -- `VllmFp8MoEMethodTPU.process_weights_after_loading` calls
`register_expert_load_buffer`, the fp8 MoE *forward* calls `maybe_tap`, and
`TPUWorker.execute_model` calls `maybe_dump_after_step` on every step -- so
"inert unless `TPU_MOE_DUMP_EXPERT_LOAD=1`" is a property that has to hold at
runtime, not merely a claim in a docstring. The merge's whole attribution
argument depends on it.

Four architectural traps killed this ~30-line op in four separate jobs, and
three of them fire at engine-core init from a traceback that points elsewhere.
The cheap ones are checked here so they cannot come back silently:

* **postponed annotations** -- `from __future__ import annotations` anywhere in
  `expert_load` makes `_infer_static_argnums` classify every tensor argument as
  static (job `j-ba60be15-fc34-4868-bdc9`);
* **host callbacks** -- `jax.debug.callback` and friends cannot survive
  `jax.export` (job `j-b7f40703-7881-4383-9178`);
* **module import** -- `expert_load` imports the private
  `torch_tpu._internal.pallas.pallas._infer_static_argnums` at module scope, and
  `fp8.py` and `tpu_worker.py` both import `expert_load` unconditionally, so if
  that private symbol ever moves the server fails to start *with the flag off*.

The fourth trap -- building the `jax_op` inside a compiled region -- is
deliberately **not** claimed to be covered here. A `fullgraph=True` check does
not catch it unless it leaves the op unbuilt, and two standalone TPU checks
passed while the server was failing. See `test_op_is_not_built_when_disabled`,
which pins the half of it that is checkable.
"""

import inspect
import os
from unittest.mock import patch

import pytest

from vllm_torchtpu.layers.common import expert_load


@pytest.fixture(autouse=True)
def _reset_module_state():
    op, buffers, steps = expert_load._OP, dict(
        expert_load._BUFFERS), expert_load._STEPS
    expert_load._BUFFERS.clear()
    expert_load._STEPS = 0
    yield
    expert_load._OP = op
    expert_load._BUFFERS.clear()
    expert_load._BUFFERS.update(buffers)
    expert_load._STEPS = steps


def _off():
    return patch.dict(os.environ, {"TPU_MOE_DUMP_EXPERT_LOAD": "0"})


# --------------------------------------------------------------------------
# Trap coverage
# --------------------------------------------------------------------------


def test_module_imports_and_the_private_pallas_symbol_still_exists():
    """`expert_load` is imported unconditionally by fp8.py and tpu_worker.py.

    Reaching this test at all means the module-scope
    `from torch_tpu._internal.pallas.pallas import _infer_static_argnums`
    resolved. Assert it explicitly so a future torch_tpu that moves the symbol
    fails here rather than at engine-core init on all 8 ranks.
    """
    from torch_tpu._internal.pallas.pallas import _infer_static_argnums
    assert callable(_infer_static_argnums)


def test_accumulate_annotations_are_not_stringified():
    """The postponed-annotations trap, checked without building the op.

    `_get_op` raises on this too, but only when something builds the op --
    which with the flag off never happens. Check it unconditionally.
    """
    from torch_tpu._internal.pallas.pallas import _infer_static_argnums
    sig = inspect.signature(expert_load._accumulate)
    for name, param in sig.parameters.items():
        assert not isinstance(param.annotation, str), (
            f"{name} is annotated with the string {param.annotation!r}; "
            "`from __future__ import annotations` has come back to "
            "expert_load.py and will make every tensor arg static")
    assert not _infer_static_argnums(sig)


def test_no_host_callbacks_in_the_module_source():
    """`jax.debug.callback` / `io_callback` cannot survive `jax.export`."""
    src = inspect.getsource(expert_load)
    # Strip the module docstring, which discusses the trap by name.
    body = src.split('"""', 2)[-1]
    for forbidden in ("jax.debug.callback", "io_callback", "jax.debug.print",
                      "debug_callback"):
        assert forbidden not in body, (
            f"{forbidden} is architecturally impossible here: torch_tpu "
            "lowers through jax.export, which rejects host callbacks")


def test_maybe_tap_has_no_shape_dependent_branch():
    """Every trace-visible branch must be trace-invariant.

    A branch on a symbolic dimension becomes a Dynamo guard and vLLM's
    artifact cache rejects the second trace (job
    `j-f4752055-9b99-4402-9b07`). What is allowed to remain is the env flag
    and the two `is None` checks.
    """
    src = inspect.getsource(expert_load.maybe_tap)
    body = src.split('"""', 2)[-1]
    for forbidden in (".shape[", ".numel(", ".size("):
        assert forbidden not in body, (
            f"maybe_tap's body reads {forbidden} -- a shape-dependent branch "
            "here specialises the trace and the compile-artifact cache "
            "rejects it")


# --------------------------------------------------------------------------
# Q1: inertness with the flag off
# --------------------------------------------------------------------------


def test_enabled_is_false_by_default():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TPU_MOE_DUMP_EXPERT_LOAD", None)
        assert expert_load.enabled() is False
    with _off():
        assert expert_load.enabled() is False
    for falsey in ("", "false", "False"):
        with patch.dict(os.environ, {"TPU_MOE_DUMP_EXPERT_LOAD": falsey}):
            assert expert_load.enabled() is False
    with patch.dict(os.environ, {"TPU_MOE_DUMP_EXPERT_LOAD": "1"}):
        assert expert_load.enabled() is True


def test_register_is_a_noop_and_allocates_nothing_when_disabled():
    class _Layer:
        layer_name = "model.layers.3.mlp.experts"
        global_num_experts = 512

    layer = _Layer()
    with _off():
        expert_load.register_expert_load_buffer(layer)
    assert not hasattr(layer, "_expert_load_counts")
    assert expert_load._BUFFERS == {}


def test_op_is_not_built_when_disabled():
    """With the flag off nothing ever calls `_get_op`, so `_OP` stays None.

    This is the checkable half of the eager-construction trap: `maybe_tap`
    reads `_OP` directly and returns when it is None, so with the flag off it
    cannot construct anything inside a traced region. It does **not** establish
    that the enabled path builds eagerly -- only a server run does that.
    """
    expert_load._OP = None
    with _off():
        expert_load.maybe_tap(object(), None, None)
        expert_load.maybe_dump_after_step()
    assert expert_load._OP is None, (
        "the op was constructed with the tap disabled; with the flag on this "
        "would be happening inside the compiled MoE forward")


def test_maybe_tap_returns_before_touching_its_arguments():
    """The tap sits in the fp8 MoE forward and must not perturb it.

    Passing sentinels that raise on any attribute access proves the disabled
    path does no work at all -- not even a `getattr` on the layer.
    """

    class _Explode:

        def __getattr__(self, name):
            raise AssertionError(
                f"disabled maybe_tap touched .{name} on its argument")

    with _off():
        expert_load.maybe_tap(_Explode(), _Explode(), _Explode())


def test_dump_and_reset_are_noops_when_disabled(capsys):
    with _off():
        expert_load.dump(tag="serving")
        for _ in range(5):
            expert_load.maybe_dump_after_step()
    out = capsys.readouterr().out
    assert "EXPERTLOAD_JSON" not in out
    assert expert_load._STEPS == 0, (
        "the step counter advanced with the tap disabled")


def test_disabled_tap_does_not_graph_break_the_compiled_forward():
    """The inertness claim that actually costs money if it is wrong.

    `maybe_tap` is called from `VllmFp8MoEMethodTPU.apply`, inside the
    Dynamo-traced MoE forward -- the hot path. With the flag off it returns on
    the first line, but Dynamo still has to *trace* that line, and
    `expert_load.enabled()` is a raw `os.getenv` rather than the
    `vllm_torchtpu.envs` module attribute the rest of this codebase uses (env
    module attributes constant-fold; an arbitrary `os` call is not guaranteed
    to). If it graph-breaks, the merge adds an unconditional graph break to the
    MoE forward with every flag at its default, and every "the flags caused
    this delta" attribution built on top of that is wrong.

    `fullgraph=True` turns a graph break into an exception, so this test fails
    loudly rather than silently costing throughput.
    """
    import torch

    class _Layer:
        pass

    layer = _Layer()

    def f(x, ids, weights):
        expert_load.maybe_tap(layer, ids, weights)
        return x * 2

    compiled = torch.compile(f, fullgraph=True, backend="eager", dynamic=False)
    x = torch.ones(4)
    ids = torch.zeros(4, 2, dtype=torch.int32)
    weights = torch.ones(4, 2)
    with _off():
        out = compiled(x, ids, weights)
    assert torch.equal(out, torch.full((4, ), 2.0))


def test_fullgraph_really_rejects_a_graph_break():
    """Negative control for the test above.

    If `fullgraph=True` were not enforcing on this torch build, the previous
    test would pass no matter what `maybe_tap` did. Compile a region with an
    unambiguous graph break and require it to raise.
    """
    import torch

    def g(x):
        print("this print is a graph break")  # noqa: T201
        return x * 2

    compiled = torch.compile(g, fullgraph=True, backend="eager", dynamic=False)
    with pytest.raises(Exception):
        compiled(torch.ones(4))


def test_layer_without_a_buffer_is_skipped_even_when_enabled():
    """A layer that never registered must not crash the forward."""

    class _Layer:
        pass

    with patch.dict(os.environ, {"TPU_MOE_DUMP_EXPERT_LOAD": "1"}):
        expert_load.maybe_tap(_Layer(), None, None)


# --------------------------------------------------------------------------
# The histogram itself
# --------------------------------------------------------------------------


def test_accumulate_counts_valid_rows_and_masks_padding():
    """`_accumulate` is plain JAX; check its arithmetic on the host.

    Padding rows keep their expert ids under `TPU_MOE_SKIP_PADDED_TOKENS` and
    only their weights are zeroed, so counting ids alone would attribute
    phantom load. This is the assertion that would catch a dropped mask.
    """
    jnp = pytest.importorskip("jax.numpy")

    counts = jnp.zeros(4, dtype=jnp.int32)
    ids = jnp.asarray([[0, 1], [1, 2], [3, 3]], dtype=jnp.int32)
    weights = jnp.asarray(
        [[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]],  # last row is padding
        dtype=jnp.float32)

    got = expert_load._accumulate(counts, ids, weights)
    # expert 0 once, expert 1 twice, expert 2 once, expert 3 zero (masked).
    assert list(map(int, got)) == [1, 2, 1, 0]

    # Negative control: without the weight mask expert 3 would count twice.
    unmasked = expert_load._accumulate(counts, ids, jnp.ones_like(weights))
    assert list(map(int, unmasked)) == [1, 2, 1, 2]


def test_accumulate_ignores_out_of_range_expert_ids():
    jnp = pytest.importorskip("jax.numpy")
    counts = jnp.zeros(4, dtype=jnp.int32)
    ids = jnp.asarray([[-1, 9], [2, 2]], dtype=jnp.int32)
    weights = jnp.ones((2, 2), dtype=jnp.float32)
    got = expert_load._accumulate(counts, ids, weights)
    assert list(map(int, got)) == [0, 0, 2, 0]


def test_accumulate_is_additive():
    jnp = pytest.importorskip("jax.numpy")
    ids = jnp.asarray([[0, 1]], dtype=jnp.int32)
    weights = jnp.ones((1, 2), dtype=jnp.float32)
    counts = jnp.asarray([5, 7, 0, 0], dtype=jnp.int32)
    got = expert_load._accumulate(counts, ids, weights)
    assert list(map(int, got)) == [6, 8, 0, 0]
