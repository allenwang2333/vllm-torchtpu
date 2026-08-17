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
"""MMEncoderManager._dp_gather must reassemble the batch it was handed.

With ``--mm-encoder-tp-mode data`` each TP rank encodes a different subset of
the images and one all-gather puts the batch back together. The override exists
because the base implementation computes its offsets from the *unpadded* token
counts, while the TPU path keeps every item padded to a bucket -- so the base
slice bounds land mid-item and each rank reads a neighbour's rows.

There is no TPU or process group here: the all-gather is simulated by running
every rank's send path in the same process and concatenating, which is exactly
what the real collective produces.
"""

import random
import types

import pytest
import torch

from vllm_torchtpu.runner.mm_encoder_manager import MMEncoderManager

HIDDEN = 8
BUCKET_MIN = 64


def bucket(num_tokens):
    size = BUCKET_MIN
    while size < num_tokens:
        size *= 2
    return size


def make_manager(rank, bucketed):
    """An MMEncoderManager with just the fields _dp_gather touches."""
    mgr = object.__new__(MMEncoderManager)
    mgr.config = types.SimpleNamespace(out_hidden_size=HIDDEN)
    mgr.dtype = torch.float32
    mgr.device = torch.device("cpu")
    mgr.model = (types.SimpleNamespace(mm_embed_bucket=staticmethod(bucket))
                 if bucketed else types.SimpleNamespace())
    mgr._dp_rank = staticmethod(lambda: rank)
    return mgr


def round_robin_assignment(num_items, tp_size):
    """Stand-in for get_load_balance_assignment's flattened rank ordering."""
    per_rank = [[i for i in range(num_items) if i % tp_size == r]
                for r in range(tp_size)]
    flat = [i for items in per_rank for i in items]
    return flat, [len(items) for items in per_rank], per_rank


def run_gather(per_item_out_tokens, tp_size, bucketed, monkeypatch):
    """Encode every item, shard it, gather it back, return per-item outputs."""
    num_items = len(per_item_out_tokens)
    flat, counts, per_rank = round_robin_assignment(num_items, tp_size)

    # Distinct values per item so a misplaced row cannot go unnoticed. Each
    # item is padded to its bucket the way postprocess_encoder_output does.
    truth = {}
    stored = {}
    for i, n in enumerate(per_item_out_tokens):
        rows = torch.arange(n, dtype=torch.float32).reshape(n,
                                                            1) + (i + 1) * 1000
        truth[i] = rows.expand(n, HIDDEN).contiguous()
        pad = (bucket(n) if bucketed else n) - n
        stored[i] = (torch.cat([truth[i], truth[i][-1:].expand(pad, HIDDEN)])
                     if pad else truth[i])

    # Every rank builds its send buffer; the concatenation of those buffers is
    # what tensor_model_parallel_all_gather(dim=0) yields on every rank.
    sends = []

    def fake_all_gather(tensor, dim=0):
        assert dim == 0
        return torch.cat(sends, dim=0)

    monkeypatch.setattr(
        "vllm_torchtpu.runner.mm_encoder_manager.tp_all_gather_materialized",
        fake_all_gather)

    results = []
    for rank in range(tp_size):
        mgr = make_manager(rank, bucketed)
        local = [stored[i] for i in per_rank[rank]]
        # First pass: capture this rank's send buffer via the fake collective.
        captured = {}

        def capture(tensor, dim=0, _c=captured):
            _c["send"] = tensor.clone()
            return torch.cat(sends + [tensor], dim=0)

        monkeypatch.setattr(
            "vllm_torchtpu.runner.mm_encoder_manager."
            "tp_all_gather_materialized", capture)
        mgr._dp_gather(
            local, list(per_item_out_tokens), flat, counts,
            max(
                sum(per_item_out_tokens[i] for i in items)
                for items in per_rank))
        sends.append(captured["send"])

    # Now that every rank's contribution exists, replay the gather for real.
    monkeypatch.setattr(
        "vllm_torchtpu.runner.mm_encoder_manager.tp_all_gather_materialized",
        fake_all_gather)
    for rank in range(tp_size):
        mgr = make_manager(rank, bucketed)
        local = [stored[i] for i in per_rank[rank]]
        results.append(
            mgr._dp_gather(
                local, list(per_item_out_tokens), flat, counts,
                max(
                    sum(per_item_out_tokens[i] for i in items)
                    for items in per_rank)))
    return results, truth


@pytest.mark.parametrize("bucketed", [True, False], ids=["bucketed", "exact"])
@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_dp_gather_restores_every_item(tp_size, bucketed, monkeypatch):
    rng = random.Random(tp_size)
    per_item = [rng.randint(1, 400) for _ in range(tp_size * 2 + 1)]

    results, truth = run_gather(per_item, tp_size, bucketed, monkeypatch)

    for rank, per_item_outputs in enumerate(results):
        assert len(per_item_outputs) == len(per_item), f"rank {rank}"
        for i, out in enumerate(per_item_outputs):
            n = per_item[i]
            assert torch.equal(out[:n], truth[i]), (
                f"rank {rank} item {i} ({n} tokens) got the wrong rows")


def test_dp_gather_keeps_bucket_padding(monkeypatch):
    """Downstream indexing clamps into the bucket, so it must still be there."""
    per_item = [100, 300, 5]
    results, _ = run_gather(per_item, 2, True, monkeypatch)
    assert [out.shape[0] for out in results[0]] == [128, 512, 64]


def test_dp_gather_send_length_is_bucketed(monkeypatch):
    """Two batches whose totals share a bucket must all-gather the same shape."""
    seen = set()
    for per_item in ([100, 100], [120, 110]):  # both total into the 256 bucket
        sends = []

        def capture(tensor, dim=0, _s=sends):
            _s.append(tensor.shape)
            return torch.cat([tensor, tensor], dim=0)

        monkeypatch.setattr(
            "vllm_torchtpu.runner.mm_encoder_manager."
            "tp_all_gather_materialized", capture)
        mgr = make_manager(0, True)
        stored = [torch.zeros(bucket(n), HIDDEN) for n in per_item]
        mgr._dp_gather(stored[:1], per_item, [0, 1], [1, 1], max(per_item))
        seen.add(sends[0])
    assert len(seen) == 1, f"send shape varied with image size: {seen}"
