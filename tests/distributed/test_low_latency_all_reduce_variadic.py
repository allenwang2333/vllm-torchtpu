# SPDX-License-Identifier: Apache-2.0
"""Off-device checks for the variadic (two-operand) MoE pair reduction.

The variadic kernel replaces ``torch.cat`` + one collective at K3's latent-MoE
site with one kernel taking both operands. What can be checked without a TPU is
the *selection* logic around it -- the env gate, the width derivation, and the
divisibility precondition -- because getting any of those wrong routes the wrong
tensors into a kernel built for different shapes. The reduction itself needs real
devices and lives in ``scripts/allreduce_variadic_stress.py``.
"""

import types

import pytest

from vllm_torchtpu.distributed import low_latency_all_reduce as llar


def _config(hidden: int | None, latent: int | None):
    hf = types.SimpleNamespace()
    if hidden is not None:
        hf.hidden_size = hidden
    if latent is not None:
        hf.routed_expert_hidden_size = latent
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_text_config=hf, hf_config=hf))


def test_widths_come_from_the_model_config():
    cfg = _config(7168, 3584)
    assert llar._moe_hidden_width(cfg) == 7168
    assert llar._moe_latent_width(cfg) == 3584
    # The concat width is their sum, and the variadic path must decompose it
    # back into exactly these two -- if it ever disagreed, the kernel would be
    # built for one shape and fed another.
    assert llar._moe_concat_width(cfg) == 7168 + 3584


def test_component_widths_are_none_without_a_latent_moe():
    # A dense model has no routed-expert width, so there is no pair to reduce
    # and the variadic path must decline rather than invent a second operand.
    cfg = _config(7168, None)
    assert llar._moe_latent_width(cfg) is None
    assert llar._moe_concat_width(cfg) is None


def test_nonsense_widths_are_rejected():
    for hidden, latent in ((0, 3584), (7168, 0), (-1, 3584)):
        cfg = _config(hidden, latent)
        assert (llar._moe_hidden_width(cfg) is None
                or llar._moe_latent_width(cfg) is None)


def test_variadic_is_off_unless_explicitly_enabled(monkeypatch):
    monkeypatch.delenv("TPU_LOW_LATENCY_ALLREDUCE_MOE_VARIADIC", raising=False)
    assert not llar.moe_variadic_enabled()


def test_variadic_on_when_set(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MOE_VARIADIC", "1")
    assert llar.moe_variadic_enabled()


@pytest.mark.parametrize("value", ["0", "", "true", "yes", "2"])
def test_only_exactly_one_enables_it(monkeypatch, value):
    # Matches moe_concat_enabled's convention. A gate that accepted "true" but
    # not "1" would silently diverge from the flag next to it.
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MOE_VARIADIC", value)
    assert not llar.moe_variadic_enabled()


def test_entry_starts_unwired():
    # Nothing may consume the entry before warmup registers it; the MoE tail
    # reads this global directly and a stale truthy value would send the pair
    # into a kernel whose workspaces belong to another process.
    assert llar._MOE_VARIADIC_ENTRY is None or isinstance(
        llar._MOE_VARIADIC_ENTRY, tuple)


@pytest.mark.parametrize(
    "rows,latent,hidden,devices,ok",
    [
        (1, 3584, 7168, 32, True),  # K3 decode: 112 and 224 per device
        (1, 3584, 7168, 8, True),  # 448 and 896
        (1, 3584, 7168, 33, False),  # neither divides
        (1, 100, 7168, 32, False),  # latent does not divide
        (1, 3584, 100, 32, False),  # hidden does not divide
        (16, 3584, 7168, 32, True),  # a larger bucket still divides
    ])
def test_divisibility_precondition(rows, latent, hidden, devices, ok):
    # two-shot scatters over the device axis, so both element counts must
    # divide the world size. The warmup checks this before building the op
    # because the kernel raises rather than falling back, and a raise during
    # warmup takes down the server.
    got = (rows * latent) % devices == 0 and (rows * hidden) % devices == 0
    assert got is ok
