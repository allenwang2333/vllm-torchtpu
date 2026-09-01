# SPDX-License-Identifier: Apache-2.0
"""The dispatch gate for the low-latency all-reduce kernel.

The gate is the part of the integration that has to be right without a TPU
present: off-crossover the kernel is not merely slower but much slower (0.61x
XLA at 448 KiB / 32 devices, and a 256 KiB payload once took a core down with
RuntimeUnexpectedCoreHalt), so a gate that lets a payload through in the wrong
band turns a speedup into a liability. Everything else -- correctness of the
reduction, the workspace parity -- needs real devices and lives in
``scripts/k3/llar_out_proj_check.sbatch``.

Thresholds trace back to the measured sweep quoted in
``distributed/low_latency_all_reduce.py``.
"""

import pytest
import torch

from vllm_torchtpu.distributed.low_latency_all_reduce import (
    _DEFAULT_MAX_BATCH_SIZE, _choose_kernel, max_batch_size, wired_max_numel)

KIB = 1024

# K3's out-proj reduce: bf16, hidden width.
WIDTH = 7168
DTYPE = torch.bfloat16


def _route(devices: int, payload: int, numel: int | None = None) -> str:
    # numel only matters for two-shot's divisibility check; default to a value
    # that divides evenly so the size bands are what is under test.
    return _choose_kernel(devices, payload,
                          numel if numel is not None else devices * 1024)


@pytest.mark.parametrize("payload", [2 * KIB, 8 * KIB, 13 * KIB])
def test_small_payloads_take_two_shot_at_32_devices(payload):
    # two_shot is nearly flat in payload (12.36 / 12.67 / 13.20 us at
    # 14 / 28 / 42 KiB) while XLA is flat but higher (17.31 / 17.97 / 18.93),
    # so extending it downwards is safe: a smaller payload makes two_shot's
    # job easier and XLA's no harder.
    assert _route(32, payload) == "two_shot"


def test_32kib_at_32_devices_takes_two_shot():
    assert _route(32, 32 * KIB) == "two_shot"


def test_14kib_at_32_devices_takes_two_shot():
    # 14 KiB is one decode token of K3's hidden width -- the c=1 payload this
    # integration exists for.
    assert _route(32, 14 * KIB, numel=7168) == "two_shot"


def test_16_devices_covers_the_small_end_too():
    # The sweep behind the low end is at 32 devices. Egress is M(N-1), so the
    # crossover moves with the device count; do not extrapolate it blindly.
    assert _route(16, 14 * KIB, numel=7168) == "two_shot"


@pytest.mark.parametrize("payload", [2 * KIB, 16 * KIB, 32 * KIB])
def test_small_payloads_take_two_shot_at_8_devices(payload):
    assert _route(8, payload) == "two_shot"


@pytest.mark.parametrize(
    "devices,payload",
    [
        (32, 224 * KIB),  # the measured tie point
        (32, 448 * KIB),  # 0.61x XLA here
        (16, 256 * KIB),
        (8, 128 * KIB),
        (2, 4 * KIB),  # nothing to win at N=2: 2 hops either way
        (4, 4 * KIB),
    ])
def test_past_the_crossover_stays_on_xla(devices, payload):
    assert _route(devices, payload) == "xla"


def test_indivisible_payload_stays_on_xla():
    # two_shot scatters along rows; a payload that does not divide the device
    # count cannot be reduced by it.
    assert _route(32, 14 * KIB, numel=7168 + 1) == "xla"


# --- the wired ceiling the op is built with ---------------------------------


def test_k3_out_proj_wires_through_8_tokens_by_default(monkeypatch):
    monkeypatch.delenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE",
                       raising=False)
    # T=8 is 112 KiB, the last measured win; T=16 is the 224 KiB tie and is
    # deliberately not wired.
    assert wired_max_numel(WIDTH, DTYPE, 32) == 8 * WIDTH


def test_gate_binds_before_the_batch_knob(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE", "16")
    # Asking for 16 tokens does not wire T=16: the measured 128 KiB gate caps
    # the payload at 9 tokens of this width, and the lane-alignment rule then
    # drops T=9 (9*224 = 2016 is not a 128-multiple), leaving T=8. The knob
    # lowers the ceiling; it does not raise the gate.
    assert wired_max_numel(WIDTH, DTYPE, 32) == 8 * WIDTH


def test_batch_knob_can_only_lower_the_ceiling(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE", "4")
    assert wired_max_numel(WIDTH, DTYPE, 32) == 4 * WIDTH


def test_ungated_device_count_wires_nothing(monkeypatch):
    monkeypatch.delenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE",
                       raising=False)
    assert wired_max_numel(WIDTH, DTYPE, 4) == 0


# --- the batch knob's parsing ------------------------------------------------


def test_default_max_batch_size(monkeypatch):
    monkeypatch.delenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE",
                       raising=False)
    assert max_batch_size() == _DEFAULT_MAX_BATCH_SIZE


def test_blank_max_batch_size_uses_default(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE", "")
    assert max_batch_size() == _DEFAULT_MAX_BATCH_SIZE


def test_garbage_max_batch_size_uses_default(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE", "lots")
    assert max_batch_size() == _DEFAULT_MAX_BATCH_SIZE


def test_nonpositive_max_batch_size_uses_default(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE", "0")
    assert max_batch_size() == _DEFAULT_MAX_BATCH_SIZE


def test_max_batch_size_reads_the_env(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE", "2")
    assert max_batch_size() == 2
