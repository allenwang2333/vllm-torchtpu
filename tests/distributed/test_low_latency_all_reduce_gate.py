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


@pytest.mark.parametrize("payload", [2 * KIB, 8 * KIB, 13 * KIB])
def test_small_payloads_use_one_shot_at_32_devices(payload):
    # one_shot owns everything up to 13 KiB, below where two_shot is measured.
    assert _route(32, payload) == "one_shot"


def test_32kib_at_32_devices_stays_on_xla():
    # With two_shot off, one_shot is bandwidth-bound at 31 peers here (0.70x vs
    # XLA), so nothing runs.
    assert _route(32, 32 * KIB) == "xla"


def test_two_shot_takes_14kib_from_one_shot_at_32_devices(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "1")
    # 14 KiB is the smallest payload where both are measured, and two_shot wins
    # it: 12.36 against one_shot's 13.59. A 16 KiB one_shot boundary handed it
    # to the slower kernel, which made K3's post-attention reduce *slower* at a
    # single-token bucket (one_shot 13.59) than at a two-token one (two_shot
    # 12.67) -- a smaller batch reducing slower.
    assert _route(32, 14 * KIB, numel=7168) == "two_shot"


def test_lowering_the_one_shot_boundary_leaves_no_gap(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "1")
    # one_shot_max and two_shot_min must move together. Leaving the floor at 16
    # KiB while dropping the boundary to 13 opens a 13-16 KiB hole to XLA --
    # the same bug that sent K3's 42 KiB pair to XLA at 8 devices.
    for kib in (13, 14, 15, 16, 17):
        assert _route(32, kib * KIB, numel=kib * 512) != "xla", f"{kib} KiB"


def test_16_devices_keeps_its_own_one_shot_boundary(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "1")
    # The sweep behind the 13 KiB boundary is at 32 devices. one_shot's egress
    # is M(N-1) and two_shot's 2M(N-1)/N, so the crossover really does move with
    # the device count; extrapolating it across counts is how the 42 KiB hole at
    # 8 devices got there.
    assert _route(16, 14 * KIB, numel=7168) == "one_shot"


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


def test_concat_payload_takes_two_shot_at_8_devices(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "1")
    # K3's concatenated latent-MoE pair is 2x10752 bf16 = 42 KiB. The 8-device
    # band used to start at 64 KiB purely because the original sweep did, so
    # this fell through to XLA and the single-host pruned-K3 proxy could not
    # exercise the concat path. Measured at 8 devices two_shot is 8.21 us here
    # against XLA's 10.61, so the band starts at one_shot_max instead.
    assert _route(8, 42 * KIB, numel=21504) == "two_shot"


def test_one_shot_still_owns_the_small_end_at_8_devices(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "1")
    # Lowering two_shot's floor to one_shot_max must not steal the range where
    # one_shot is nearly twice as fast (4.84 vs 7.76 us at 14 KiB).
    assert _route(8, 14 * KIB, numel=7168) == "one_shot"


@pytest.mark.parametrize("payload,numel", [
    (224 * KIB, 16 * 7168),
    (336 * KIB, 16 * 10752),
    (448 * KIB, 16 * 14336),
])
def test_two_shot_ceiling_excludes_the_payloads_it_loses(
        monkeypatch, payload, numel):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "1")
    # two_shot is only locally flat in payload. Measured at 32 devices (job
    # 143, device self time): 0.91x XLA at 224 KiB, 0.75x at 336 KiB, 0.61x at
    # 448 KiB -- it is bandwidth-bound past ~112 KiB while the ring amortises.
    # The ceiling was 256 KiB, which was the top of the original sweep's range
    # rather than a measurement, and it let 224 KiB take a kernel 9% slower
    # than the XLA it replaced.
    assert _route(32, payload, numel=numel) == "xla"


@pytest.mark.parametrize("devices", [16, 32])
def test_two_shot_keeps_the_band_it_wins(monkeypatch, devices):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "1")
    # Bringing the ceiling down must not cut into the measured winning region:
    # 42 KiB is 1.44x XLA and 112 KiB -- K3's (16,3584) at decode bucket 16 --
    # is still 1.26x.
    assert _route(devices, 42 * KIB, numel=21504) == "two_shot"
    assert _route(devices, 112 * KIB, numel=16 * 3584) == "two_shot"
