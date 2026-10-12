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
