# SPDX-License-Identifier: Apache-2.0
"""The dispatch gate for the low-latency all-reduce kernels.

The gate is the part of the integration that has to be right without a TPU
present: off-crossover the one-shot family is not merely slower but
catastrophically slower (857 us at 1 MiB / 32 devices against XLA's 38), so a
gate that lets a payload through in the wrong band turns a speedup into a large
regression. Everything else -- correctness of the reduction, the workspace
parity -- needs real devices and lives in
``scripts/debug/test_low_latency_all_reduce.py``.

Thresholds trace back to the measured sweep in ``docs/low_latency_allreduce.md``.
"""

import pytest

from vllm_torchtpu.distributed.low_latency_all_reduce import _choose_kernel

KIB = 1024


def _route(devices: int, payload: int, numel: int | None = None) -> str:
    # numel only matters for two-shot's divisibility check; default to a value
    # that divides evenly so the size bands are what is under test.
    return _choose_kernel(devices, payload,
                          numel if numel is not None else devices * 1024)


@pytest.mark.parametrize("payload", [2 * KIB, 8 * KIB, 16 * KIB])
def test_small_payloads_use_one_shot_at_32_devices(payload):
    assert _route(32, payload) == "one_shot"


def test_32kib_at_32_devices_stays_on_xla():
    # One-shot is bandwidth-bound at 31 peers here (0.70x vs XLA) and
    # two-shot's measured range starts at 64 KiB. Neither wins, so neither runs.
    assert _route(32, 32 * KIB) == "xla"


@pytest.mark.parametrize("payload", [2 * KIB, 16 * KIB, 32 * KIB])
def test_one_shot_range_is_wider_at_8_devices(payload):
    # Egress is M*(N-1), so fewer peers keep one-shot cheap further up.
    assert _route(8, payload) == "one_shot"


@pytest.mark.parametrize("devices,payload", [(8, 1024 * KIB), (32, 512 * KIB),
                                             (32, 4096 * KIB)])
def test_large_payloads_fall_back_to_xla(devices, payload):
    assert _route(devices, payload) == "xla"


def test_prefill_sized_payload_never_takes_a_kernel():
    # A 512-token prefill of K3's 7168-wide hidden state in bf16 is 7 MiB --
    # the regime where XLA's ring is simply the right algorithm.
    assert _route(32, 512 * 7168 * 2, numel=512 * 7168) == "xla"


def test_unknown_device_count_falls_back_to_xla():
    # Nothing was measured at 5 devices; guessing is how the one-shot cliff
    # gets hit.
    assert _route(5, 2 * KIB) == "xla"


def test_two_shot_is_off_unless_explicitly_enabled(monkeypatch):
    monkeypatch.delenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", raising=False)
    assert _route(32, 128 * KIB) == "xla"


def test_two_shot_band_when_enabled(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "1")
    assert _route(32, 128 * KIB) == "two_shot"


def test_two_shot_needs_a_divisible_element_count(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "1")
    # two-shot scatters the flattened payload across devices; an element count
    # that does not divide cannot be expressed and must not be forced.
    assert _route(32, 128 * KIB, numel=65537) == "xla"
