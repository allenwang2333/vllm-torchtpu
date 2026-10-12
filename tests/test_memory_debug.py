# SPDX-License-Identifier: Apache-2.0
import gc
import weakref

import pytest

from vllm_torchtpu import memory_debug


class _Entry:
    pass


def _backend_runnable(entry):
    """Shaped like TpuBackend's return: a wrapper closing over the aot
    runnable, both carrying ``serialize = lambda: entry``."""

    def result(*args):
        return sum(args)

    def skip_execution_on_warmup_if_async(*args):
        return result(*args)

    result.serialize = lambda: entry
    skip_execution_on_warmup_if_async.serialize = lambda: entry
    return skip_execution_on_warmup_if_async


@pytest.mark.cpu_test
@pytest.mark.parametrize("enabled", [False, True])
def test_drop_serialize_refs_releases_the_bundled_entry(monkeypatch, enabled):
    monkeypatch.setattr(memory_debug, "DROP_SERIALIZE_ENTRY", enabled)
    entry = _Entry()
    alive = weakref.ref(entry)
    runnable = _backend_runnable(entry)
    del entry

    memory_debug.maybe_drop_serialize_refs(runnable)
    gc.collect()

    assert (alive() is None) == enabled
    assert runnable(2, 3) == 5


@pytest.mark.cpu_test
def test_clear_precompile_context_drops_parked_entries(monkeypatch):
    from torch._dynamo.precompile_context import PrecompileContext

    monkeypatch.setattr(memory_debug, "CLEAR_PRECOMPILE_CONTEXT", True)
    entry = _Entry()
    alive = weakref.ref(entry)
    monkeypatch.setitem(PrecompileContext._backend_artifacts_by_key, "k", entry)
    del entry

    memory_debug.maybe_clear_precompile_context("test")

    assert memory_debug._precompile_artifacts() == 0
    assert alive() is None


@pytest.mark.cpu_test
def test_holders_names_the_owner_chain(monkeypatch, caplog):
    from torch_tpu._internal.compile import torch_tpu_compiled_executable as tce

    class FakeExecutable:
        pass

    class Owner:
        def __init__(self):
            self.copy = FakeExecutable()

    monkeypatch.setattr(tce, "TorchTpuCompiledExecutable", FakeExecutable)
    monkeypatch.setattr(memory_debug, "TRACE", True)
    monkeypatch.setattr(memory_debug, "_is_rank0", True)
    keep = Owner()  # noqa: F841 (holds the fake executable)

    with caplog.at_level("INFO"):
        memory_debug.log_executable_holders("test", depth=2)

    assert "executables=1" in caplog.text
    # Python 3.12 inlines instance dicts, so the owner is the direct referrer.
    assert "x test_holders_names_the_owner_chain.<locals>.Owner" in caplog.text
