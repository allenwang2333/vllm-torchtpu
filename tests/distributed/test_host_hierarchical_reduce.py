# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only topology checks; device numerics live in the bridge benchmark."""

import importlib.util
import random
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / (
    "src/vllm_torchtpu/distributed/host_hierarchical_reduce.py"
)
_SPEC = importlib.util.spec_from_file_location("host_reduce", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_shuffled_ranks_preserve_every_contribution():
    devices = list(range(32))
    random.Random(219).shuffle(devices)
    hosts = [f"host-{device // 8}" for device in devices]
    local, cross = _MODULE.host_replica_groups(hosts, devices)
    values = [
        [(device + 1) * (column + 1) for column in range(16)] for device in devices
    ]
    expected = [sum(row[column] for row in values) for column in range(16)]
    shards = {}
    for group in local:
        assert len({hosts[i] for i in group}) == 1
        for shard, rank in enumerate(group):
            shards[rank] = [
                sum(values[i][column] for i in group)
                for column in range(shard * 2, shard * 2 + 2)
            ]
    reduced = {}
    for group in cross:
        assert len({hosts[i] for i in group}) == 4
        result = [sum(shards[i][column] for i in group) for column in range(2)]
        for rank in group:
            reduced[rank] = result
    for group in local:
        assert [value for rank in group for value in reduced[rank]] == expected


@pytest.mark.parametrize(
    "hosts,devices",
    [
        (["one-host"] * 32, list(range(32))),
        ([str(i // 8) for i in range(31)], list(range(31))),
        ([str(i // 8) for i in range(32)], [0] * 32),
        (["a"] * 7 + ["b"] * 9 + ["c"] * 8 + ["d"] * 8, list(range(32))),
    ],
)
def test_unsupported_or_ambiguous_topology_fails(hosts, devices):
    with pytest.raises(ValueError):
        _MODULE.host_replica_groups(hosts, devices)
