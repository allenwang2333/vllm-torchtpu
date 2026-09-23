# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared pytest configuration for all tests."""

import os
import sys

# Keep TP collectives compiled in-graph (no graph break) for fullgraph TP.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
                      "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

# Forking after TorchTPU/JAX/PyTorch background threads have started can
# deadlock vLLM engine or worker subprocess startup. Set this before test
# modules import vLLM so every LLM test uses a clean spawned process.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Disable vLLM background usage telemetry in tests to prevent background threads
# and IPC sockets from hanging multi-process worker teardown.
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_DO_NOT_TRACK", "1")

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """Run whole-slice worker-spawning tests before in-process TPU tests.

    Once the pytest process itself initializes the TPU runtime (any test that
    runs jax/torch_tpu in-process), a test that later spawns a fresh
    ``torch.distributed.run`` worker group can no longer bootstrap the slice:
    the workers fail with "PjRtClient is not initialized" and libtpu may tear
    down the whole session with SLICE_FAILURE_SW_INJECT_ERROR. libtpu exposes
    no release API, so the only safe ordering is spawn-first. The sort is
    stable, so relative order within each group is unchanged.
    """
    items.sort(key=lambda item: 0
               if item.get_closest_marker("spawns_tpu_workers") else 1)
    _select_shard(config, items)


def pytest_addoption(parser):
    """Add --use-tpu and --shard command line options."""
    parser.addoption(
        "--use-tpu",
        action="store_true",
        default=True,
        help="Run tests on TPU device instead of CPU",
    )
    parser.addoption(
        "--shard",
        default=None,
        metavar="I/N",
        help="Run only shard I of N (1-based). Tests sharing a "
        "shard_group marker always land on the same shard.",
    )


def _shard_weight(item):
    """Relative cost of one test, used only to balance --shard.

    A module can publish measured costs as SHARD_MINUTES, keyed by test id
    (``test_x[param]``) or by function name; anything missing counts as
    SHARD_DEFAULT_MINUTES, or 1 when the module publishes neither.
    """
    table = getattr(item.module, "SHARD_MINUTES", {})
    default = getattr(item.module, "SHARD_DEFAULT_MINUTES", 1.0)
    return table.get(item.name, table.get(item.originalname, default))


def _select_shard(config, items):
    """Deselect everything outside this process's --shard.

    Groups (a shard_group marker, else the test on its own) are placed
    heaviest-first on the least-loaded shard. Every shard computes the same
    placement from the same collected items, so each test runs exactly once
    across the N shards.

    Skipped under --collect-only: Buildkite's test engine client (bktec)
    plans its run from a plain collection and may hand one step's plan to
    every job of that step. Listing the full selection there keeps every
    job's plan identical, and the split happens when the tests actually run.
    """
    spec = config.getoption("--shard")
    if not spec or config.getoption("collectonly"):
        return
    try:
        index, count = (int(part) for part in spec.split("/"))
    except ValueError:
        raise pytest.UsageError(f"--shard expects I/N, got {spec!r}") from None
    if not 1 <= index <= count:
        raise pytest.UsageError(f"--shard {spec}: need 1 <= I <= N")

    def group_of(item):
        marker = item.get_closest_marker("shard_group")
        return marker.args[0] if marker else item.nodeid

    weights = {}
    for item in items:
        group = group_of(item)
        weights[group] = weights.get(group, 0.0) + _shard_weight(item)
    loads = [0.0] * count
    placement = {}
    for group in sorted(weights, key=lambda g: (-weights[g], g)):
        shard = min(range(count), key=lambda s: (loads[s], s))
        placement[group] = shard
        loads[shard] += weights[group]

    keep = [item for item in items if placement[group_of(item)] == index - 1]
    dropped = [
        item for item in items if placement[group_of(item)] != index - 1
    ]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = keep


@pytest.fixture(autouse=True)
def _reset_warning_once():
    """`warning_once` dedups by message for the life of the process, so without
    this a test passes or fails depending on what ran before it."""
    try:
        import vllm.logger
    except ImportError:
        yield
        return

    vllm.logger._print_warning_once.cache_clear()
    yield
    vllm.logger._print_warning_once.cache_clear()


@pytest.fixture
def vllm_config_context():
    """Run backend helpers with the same resolved layout as worker startup."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    config = VllmConfig()
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


@pytest.fixture
def device(request):
    """Get the device to run tests on (CPU or TPU).

    Usage:
        pytest tests/ -v              # Run on CPU
        pytest tests/ -v --use-tpu    # Run on TPU
    """
    use_tpu = request.config.getoption("--use-tpu")
    if use_tpu:
        try:
            import torch_tpu  # noqa: F401
            return torch.device("tpu")
        except ImportError:
            pytest.skip("torch_tpu not available")
        except Exception as e:
            pytest.fail(f"TPU requested but failed to initialize: {e}")
    return torch.device("cpu")


def pytest_collection_finish(session):
    """Fail fast if collecting the suite claimed the accelerator.

    libtpu is process-exclusive. A PJRT client built in the pytest PARENT locks
    out every child, and vLLM's engine and workers run in spawned children (see
    VLLM_WORKER_MULTIPROC_METHOD above), so every engine-starting test in the
    run then dies with "pjrt client not initialized" -- far from the module
    that actually claimed the device.

    Collection is import-time, so anything evaluated at module scope counts:
    the usual culprit is a `pytest.mark.skipif(jax.device_count() < N)`, which
    builds a client to answer and does so even when the marker goes on to
    deselect every test in the file. Put that check in a fixture instead; a
    fixture runs at setup, after deselection.

    Checked here rather than left to CI because the failure is invisible to any
    run that does not collect the offending file and an engine test together.
    """
    if "jax" not in sys.modules:
        return
    from jax._src import xla_bridge
    if not xla_bridge.backends_are_initialized():
        return
    raise pytest.UsageError(
        "A JAX backend was initialized during test collection, which claims "
        "the TPU for this process and makes every spawned engine/worker child "
        "fail with 'pjrt client not initialized'. Some test module touches the "
        "accelerator at import time -- most often a module-level "
        "`jax.device_count()` in a `skipif`. Move it into a fixture. To find "
        "the module: bisect with `pytest --collect-only <subset>` and check "
        "`jax._src.xla_bridge.backends_are_initialized()` after each.")
