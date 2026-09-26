# SPDX-License-Identifier: Apache-2.0
"""Splitting precompile buckets between the ranks of a host (rank_share).

Each "rank" here is a Session on one shared in-process store, standing in for
the default process group's store the workers share.
"""

import dataclasses
import os

import pytest
import torch
import torch.distributed as dist
from torch import fx, nn
from vllm.config.utils import Range

from vllm_torchtpu.compilation import rank_share, tpu_compiler

BUCKET = "0/16_16"
KEY = "artifact_compile_range_16_16_subgraph_0"


@pytest.fixture
def store():
    return dist.HashStore()


def test_a_bucket_is_compiled_by_the_rank_that_claims_it_first(store):
    first = rank_share.Session(store, "ns")
    second = rank_share.Session(store, "ns")

    assert first.claim(BUCKET)
    assert not second.claim(BUCKET)
    assert second.artifact(BUCKET) is None  # claimed, not yet published

    first.publish(BUCKET, "/cache/rank_0_0/artifact")
    assert second.artifact(BUCKET) == "/cache/rank_0_0/artifact"
    # The claimant compiles its own bucket; it never replays it.
    assert first.artifact(BUCKET) is None


def test_a_failed_compile_releases_the_waiting_peers(store):
    """The peers learn there is no artifact and compile the bucket themselves,
    instead of waiting on a claim nobody will publish."""
    first = rank_share.Session(store, "ns")
    second = rank_share.Session(store, "ns")
    assert first.claim(BUCKET)
    assert not second.claim(BUCKET)

    with pytest.raises(RuntimeError), rank_share.compiling(first, BUCKET):
        raise RuntimeError("compile failed")

    assert second.published(BUCKET)
    assert second.artifact(BUCKET) is None


def test_peers_replay_buckets_in_the_order_they_are_published(store):
    owner = rank_share.Session(store, "ns")
    waiter = rank_share.Session(store, "ns")
    for bucket in ("0/16_16", "0/2048_2048"):
        assert owner.claim(bucket)
        assert not waiter.claim(bucket)

    owner.publish("0/2048_2048", "/cache/2048")
    ready = rank_share.as_published(
        waiter, [("0/16_16", "small"), ("0/2048_2048", "large")]
    )
    assert next(ready) == ("0/2048_2048", "large")
    owner.publish("0/16_16", "/cache/16")
    assert next(ready) == ("0/16_16", "small")
    assert next(ready, None) is None


def test_ranks_share_only_when_their_graphs_agree():
    class AddConstant(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.register_buffer("constant", torch.tensor([value]))

        def forward(self, x):
            return x + self.constant

    same = rank_share.fingerprint(fx.symbolic_trace(AddConstant(1.0)))
    assert same == rank_share.fingerprint(fx.symbolic_trace(AddConstant(1.0)))
    # Same code, but a rank-specific constant: sharing would run the wrong one.
    assert same != rank_share.fingerprint(fx.symbolic_trace(AddConstant(2.0)))


@dataclasses.dataclass
class FakeExecutable:
    """Picklable stand-in for the runnable and its bundled entry."""

    rows: int

    def __call__(self, *args):
        return (torch.zeros(self.rows),)

    def serialize(self):
        return self


class FakeTpuBackend:
    def __init__(self):
        self.compiled = 0

    def __call__(self, graph, example_inputs):
        self.compiled += 1
        return FakeExecutable(example_inputs[0].shape[0])


class Double(nn.Module):
    def forward(self, x):
        return x * 2


def test_a_peer_replays_the_claimants_artifact(store, tmp_path, monkeypatch):
    import copy

    backend = FakeTpuBackend()
    monkeypatch.setattr(tpu_compiler, "_tpu_backend", backend)
    monkeypatch.setattr(tpu_compiler, "_deserialize_entry", lambda entry: entry)
    graph = fx.symbolic_trace(Double())
    inputs = [torch.zeros(16)]

    real_deepcopy = copy.deepcopy
    deepcopy_calls = 0

    def counting_deepcopy(obj, *args, **kwargs):
        nonlocal deepcopy_calls
        if obj is graph:
            deepcopy_calls += 1
        return real_deepcopy(obj, *args, **kwargs)

    monkeypatch.setattr(tpu_compiler.copy, "deepcopy", counting_deepcopy)

    ranks = []
    for rank in range(2):
        compiler = tpu_compiler.TpuCompilerAdaptor()
        compiler.initialize_cache(str(tmp_path / f"rank_{rank}_0" / "prefix"))
        ranks.append((compiler, rank_share.Session(store, "ns")))

    handles = []
    for compiler, session in ranks:
        if session.claim(BUCKET):
            with rank_share.compiling(session, BUCKET):
                runnable, handle = compiler.compile(
                    graph, inputs, {}, Range(start=16, end=16), KEY
                )
        else:
            for bucket, rng in rank_share.as_published(
                session, [(BUCKET, Range(start=16, end=16))]
            ):
                with rank_share.compiling(session, bucket):
                    runnable, handle = compiler.compile(graph, inputs, {}, rng, KEY)
        assert runnable(*inputs).shape == (16,)
        handles.append(handle)

    assert backend.compiled == 1
    assert deepcopy_calls == 1
    owner_path, peer_path = handles[0][1], handles[1][1]
    assert peer_path == os.path.join(ranks[1][0].cache_dir, KEY)
    assert os.path.samefile(owner_path, peer_path)
    assert ranks[1][1].replayed == 1


def test_deserialize_entry_does_not_deepcopy_entry(monkeypatch):
    """deserialize_bundled_cache_entry deep-copies its input so the returned
    runnable can be re-serialized; for a TorchTPU entry that loads a second copy
    of the PjRt executable into HBM. _deserialize_entry neutralizes that copy."""
    import copy

    from torch._functorch._aot_autograd import aot_autograd_result

    class ExpensiveEntry:
        copied = 0

        def __deepcopy__(self, _memo):
            ExpensiveEntry.copied += 1
            return ExpensiveEntry()

    seen_copy = []

    def fake_deserialize(entry):
        seen_copy.append(copy.deepcopy(entry))
        return lambda *args: args

    monkeypatch.setattr(
        aot_autograd_result, "deserialize_bundled_cache_entry", fake_deserialize
    )
    entry = ExpensiveEntry()
    tpu_compiler._deserialize_entry(entry)
    assert ExpensiveEntry.copied == 0
    assert seen_copy == [None]
