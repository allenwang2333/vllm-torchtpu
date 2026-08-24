# SPDX-License-Identifier: Apache-2.0
"""The batch threshold and the entry lookup it feeds.

Which batches take the kernel is the one policy decision callers make in traced
Python, so it has to be right without a TPU present and it has to read a Python
int rather than a tensor dim -- reading a traced dim pins vLLM's dynamic token
dimension and fails compilation outright.

It also has to be *conservative in the right direction*. An entry that should
not have been selected does not produce a wrong answer -- the op checks its own
shape and falls back to XLA -- but it does produce a slow one, from inside an
opaque custom call the compiler can no longer fuse or reorder, after the caller
has already prepared a payload the kernel will not accept. On K3's latent-MoE
tail that preparation is a torch.cat plus two slices per layer: job 139 gained
3.3% at a wired bucket and lost 5.2% / 4.0% at two that were not.

The context is installed by assigning ``forward_context._forward_context``
directly rather than through ``set_forward_context``, because that helper needs
a real ``VllmConfig`` and one cannot be built off-device: ``DeviceConfig`` calls
``torch.device("tpu")``, which no CPU-only torch accepts. Assigning the global
is what ``set_forward_context`` does anyway, and the field this reads is the
same one either way.
"""

import pytest
from vllm import forward_context as fc
from vllm.forward_context import BatchDescriptor

from vllm_torchtpu.distributed.low_latency_all_reduce import (
    _DEFAULT_MAX_BATCH_SIZE, _candidate_rows, max_batch_size, select_entry)

# The value is never called here; only which one comes back is under test.
ENTRIES = {1: ("op1", ()), 2: ("op2", ()), 8: ("op8", ())}


@pytest.fixture
def context(monkeypatch):
    """Install a ForwardContext with the given padded token count, or none."""

    def install(num_tokens: int | None):
        if num_tokens is None:
            monkeypatch.setattr(fc, "_forward_context", None, raising=False)
            return None
        ctx = fc.ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
        )
        monkeypatch.setattr(fc, "_forward_context", ctx, raising=False)
        return ctx

    return install


# --- the lookup -------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", [1, 2, 8])
def test_returns_the_entry_for_this_bucket(context, num_tokens):
    context(num_tokens)
    assert select_entry(ENTRIES) is ENTRIES[num_tokens]


@pytest.mark.parametrize("num_tokens", [3, 16, 512])
def test_unwired_buckets_select_nothing(context, num_tokens):
    # Above the threshold, or simply not a compile bucket. Either way the
    # caller keeps the path it would have had without any of this.
    context(num_tokens)
    assert select_entry(ENTRIES) is None


def test_no_forward_context_selects_nothing(context):
    context(None)
    assert select_entry(ENTRIES) is None


def test_missing_descriptor_selects_nothing(monkeypatch):
    # Upstream only synthesises batch_descriptor under CUDA graphs; the TPU
    # runner sets it explicitly. If that ever regresses, this must degrade to
    # XLA rather than guess a bucket.
    ctx = fc.ForwardContext(no_compile_layers={},
                            attn_metadata={},
                            slot_mapping={},
                            batch_descriptor=None)
    monkeypatch.setattr(fc, "_forward_context", ctx, raising=False)
    assert select_entry(ENTRIES) is None


def test_empty_entries_select_nothing(context):
    context(2)
    assert select_entry({}) is None


def test_reads_an_int_not_a_tensor(context):
    # The whole point of going through the forward context: the TPU runner
    # passes num_tokens=num_tokens_padded, an ordinary int set outside anything
    # traced. If this ever became a tensor the lookup would stop folding at
    # trace time and start pinning the dynamic token dimension -- the failure
    # this module is arranged to avoid.
    ctx = context(2)
    assert type(ctx.batch_descriptor.num_tokens) is int


# --- the threshold ----------------------------------------------------------


def test_defaults_to_letting_the_payload_gate_decide(monkeypatch):
    monkeypatch.delenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE",
                       raising=False)
    assert max_batch_size() == _DEFAULT_MAX_BATCH_SIZE


@pytest.mark.parametrize("raw,expected", [("1", 1), ("4", 4), ("64", 64),
                                          (" 8 ", 8)])
def test_reads_the_configured_limit(monkeypatch, raw, expected):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE", raw)
    assert max_batch_size() == expected


@pytest.mark.parametrize("raw", ["", "   "])
def test_blank_is_unset(monkeypatch, raw):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE", raw)
    assert max_batch_size() == _DEFAULT_MAX_BATCH_SIZE


@pytest.mark.parametrize("raw", ["2x7168", "many", "1.5"])
def test_garbage_falls_back_rather_than_raising(monkeypatch, raw):
    # A bad value here must not take a 32-device job down at warmup. "2x7168"
    # in particular is the shape-shaped value someone will try after reading an
    # older revision of the docs.
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE", raw)
    assert max_batch_size() == _DEFAULT_MAX_BATCH_SIZE


@pytest.mark.parametrize("raw", ["0", "-1"])
def test_nonpositive_is_refused(monkeypatch, raw):
    # 0 would disable the kernel while leaving TPU_LOW_LATENCY_ALLREDUCE on,
    # which reads as "enabled" everywhere else and would be a silent no-op.
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE", raw)
    assert max_batch_size() == _DEFAULT_MAX_BATCH_SIZE


# --- which buckets warmup builds plans for ----------------------------------


def _config(compile_sizes, max_num_batched_tokens=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        compilation_config=SimpleNamespace(compile_sizes=compile_sizes),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens),
    )


def test_only_buckets_within_the_threshold_get_plans(monkeypatch):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE", "8")
    assert _candidate_rows(_config([1, 2, 16, 512])) == [1, 2]


def test_uncompiled_sizes_are_excluded(monkeypatch):
    # A plan for a token count the model never runs at allocates a workspace
    # nothing reads, and could never be selected anyway: the forward context
    # only ever reports a compile bucket.
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE", "512")
    assert _candidate_rows(_config([2], max_num_batched_tokens=1024)) == [2]


def test_max_batched_tokens_counts_when_it_is_small_enough(monkeypatch):
    # A decode-only server can leave compile_sizes empty and still have one
    # real shape, which is max_num_batched_tokens.
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE", "8")
    assert _candidate_rows(_config([], max_num_batched_tokens=4)) == [4]


def test_nothing_within_the_threshold_is_not_an_error(monkeypatch):
    # warmup logs and leaves every all-reduce on XLA rather than raising.
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE", "2")
    assert _candidate_rows(_config([16, 512])) == []
