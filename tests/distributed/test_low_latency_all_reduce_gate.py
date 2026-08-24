# SPDX-License-Identifier: Apache-2.0
"""The dispatch gate for the low-latency all-reduce kernels.

The gate is the part of the integration that has to be right without a TPU
present: off-crossover the kernel is not merely slower but much slower (0.61x
XLA at 448 KiB / 32 devices, and the one-shot arm this table used to carry read
857 us at 1 MiB against XLA's 38), so a gate that lets a payload through in the
wrong band turns a speedup into a regression. Everything else -- correctness of the reduction, the workspace
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
def test_small_payloads_take_two_shot_at_32_devices(payload):
    # These used to belong to one_shot, which owned everything below 13 KiB.
    # With that kernel gone two_shot takes them, and extending it downwards is
    # safe: two_shot is nearly flat in payload (12.36 / 12.67 / 13.20 us at
    # 14 / 28 / 42 KiB) while XLA is flat but higher (17.31 / 17.97 / 18.93),
    # so a smaller payload makes two_shot's job easier and XLA's no harder.
    assert _route(32, payload) == "two_shot"


def test_32kib_at_32_devices_now_takes_a_kernel():
    # This used to assert "xla": one_shot was bandwidth-bound at 31 peers here
    # (0.70x XLA) and two_shot's band was believed to start at 64 KiB, so the
    # cell was a deliberate hole. Both halves of that reasoning are gone -- the
    # 64 KiB floor was the bottom of a sweep rather than a result, and two_shot
    # reads 12.67 us at 28 KiB against XLA's 17.97.
    assert _route(32, 32 * KIB) == "two_shot"


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


def test_16_devices_covers_the_small_end_too(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "1")
    # The sweep behind the 13 KiB boundary is at 32 devices. one_shot's egress
    # is M(N-1) and two_shot's 2M(N-1)/N, so the crossover really does move with
    # the device count; extrapolating it across counts is how the 42 KiB hole at
    # 8 devices got there.
    assert _route(16, 14 * KIB, numel=7168) == "two_shot"


@pytest.mark.parametrize("payload", [2 * KIB, 16 * KIB, 32 * KIB])
def test_small_payloads_take_two_shot_at_8_devices(payload):
    # Egress is M*(N-1), so fewer peers keep one-shot cheap further up.
    assert _route(8, payload) == "two_shot"


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


def test_two_shot_is_on_by_default(monkeypatch):
    # Inverted when one_shot was removed: two_shot is the only kernel left, so
    # defaulting it off would make the dispatcher a no-op.
    monkeypatch.delenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", raising=False)
    assert _route(32, 128 * KIB) == "two_shot"


def test_two_shot_can_be_forced_off(monkeypatch):
    # The fastest way to tell whether a regression is this kernel's fault.
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "0")
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
    # against XLA's 10.61.
    assert _route(8, 42 * KIB, numel=21504) == "two_shot"


def test_the_small_end_at_8_devices_is_a_knowing_tradeoff(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "1")
    # This is the one cell where removing one_shot costs measured performance:
    # at 8 devices and 14 KiB one_shot read 4.84 us against two_shot's 7.76.
    # two_shot still beats XLA's 10.12 there, so the band covers it rather than
    # leaving a hole, but the 1.6x is genuinely given up. 8 devices is the
    # pruned-K3 proxy, not a serving target.
    assert _route(8, 14 * KIB, numel=7168) == "two_shot"


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


@pytest.mark.parametrize("devices", [2, 4])
def test_two_and_four_devices_fall_through_to_xla(devices, monkeypatch):
    """Those entries were one_shot-only, so they did not survive its removal."""
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "1")
    assert _route(devices, 8 * KIB) == "xla"


def test_8_device_ceiling_is_the_measured_one(monkeypatch):
    """64 KiB, not the old 512 KiB, which was the top of a sweep not a result.

    8 devices is measured only to 42 KiB. 64 KiB is one step past the last
    measured point; 512 KiB was three doublings past it.
    """
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT", "1")
    assert _route(8, 64 * KIB, numel=32768) == "two_shot"
    assert _route(8, 128 * KIB, numel=65536) == "xla"
