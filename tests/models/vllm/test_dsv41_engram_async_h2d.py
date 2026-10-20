# SPDX-License-Identifier: Apache-2.0
"""Non-blocking host Engram upload and the cached zero `window_starts`.

With TPU_DEEPSEEK_V41_ENGRAM_ASYNC_H2D the packed H2D returns before the DMA
has read the host buffer, so the staged bytes must match the blocking upload
and no host write may land before the upload's fence is synchronized.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as v2_model
from vllm_torchtpu.models.vllm.deepseek_v41.v2.engram import Engram

pytestmark = pytest.mark.cpu_test

_COLS, _DIM, _BLOCK, _MAX_TOKENS = 2, 32, 16, 16


def _gather(ids, raw, scales):
    # Distinct bytes per token and hash so a stale or torn upload shows.
    seed = ids.to(torch.int64).sum(-1, keepdim=True)
    raw.copy_(
        (
            (seed[:, :, None] * 7 + torch.arange(raw[0].numel()).view(raw[0].shape))
            % 251
        ).to(torch.uint8)
    )
    scales.copy_(((seed[:, :, None] + 120) % 8 + 120).to(torch.uint8).expand_as(scales))


def _engram(dp_size: int = 1) -> Engram:
    engram = object.__new__(Engram)
    engram.embed_tokens = SimpleNamespace(
        cpu_offload=True,
        host_offload=True,
        dp_size=dp_size,
        part_n_hash_cols=_COLS,
        n_hash_cols=_COLS * dp_size,
        head_dim=_DIM,
        block_size=_BLOCK,
        tp_size=1,
        gather_raw=_gather,
    )
    engram._init_staging(max_tokens=_MAX_TOKENS, head_dim=_DIM)
    return engram


class _Fence:
    """Stands in for the device event: the upload is in flight until synchronize()."""

    def __init__(self, engram, log):
        self.engram, self.log = engram, log
        self.uploaded = engram._host_packed_buf.clone()
        self.synchronized = False

    def synchronize(self):
        # The DMA reads the host buffer only now: it must still hold the upload.
        assert torch.equal(self.engram._host_packed_buf, self.uploaded)
        self.synchronized = True
        self.log.append("sync")


def _install_fences(monkeypatch, engram):
    log, fences = [], []

    def record(device):
        fences.append(_Fence(engram, log))
        log.append("record")
        return fences[-1]

    monkeypatch.setattr(common, "_record_upload_fence", record)
    return log, fences


_STEPS = (16, 8, 16, 4, 4, 16)


def _hash_ids(step: int, tokens: int, dp_size: int = 1) -> torch.Tensor:
    return (
        torch.arange(tokens * dp_size * _COLS, dtype=torch.int32).view(-1, _COLS)
        + step * 31
    )


def _staged_bytes(engram) -> torch.Tensor:
    return engram._active_raw.clone()


def test_async_upload_stages_identical_bytes(monkeypatch):
    blocking, asynchronous = _engram(), _engram()
    _, fences = _install_fences(monkeypatch, asynchronous)
    for step, tokens in enumerate(_STEPS):
        monkeypatch.setattr(
            envs, "TPU_DEEPSEEK_V41_ENGRAM_ASYNC_H2D", False, raising=False
        )
        blocking.prepare_embeddings(_hash_ids(step, tokens))
        monkeypatch.setattr(
            envs, "TPU_DEEPSEEK_V41_ENGRAM_ASYNC_H2D", True, raising=False
        )
        asynchronous.prepare_embeddings(_hash_ids(step, tokens))
        assert torch.equal(_staged_bytes(asynchronous), _staged_bytes(blocking))
        # Bitwise: some byte patterns decode to NaN.
        rows = asynchronous._ready_rows(tokens).view(torch.int16)
        assert torch.equal(rows, blocking._ready_rows(tokens).view(torch.int16))
    assert len(fences) == len(_STEPS)
    assert blocking._host_upload_fence is None


def test_host_buffer_not_rewritten_while_upload_in_flight(monkeypatch):
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_ASYNC_H2D", True, raising=False)
    engram = _engram()
    log, fences = _install_fences(monkeypatch, engram)

    def gather(ids, raw, scales):
        # Every write into the host staging buffer follows the previous fence.
        assert all(f.synchronized for f in fences)
        log.append("write")
        _gather(ids, raw, scales)

    engram.embed_tokens.gather_raw = gather
    for step, tokens in enumerate(_STEPS):
        engram.prepare_embeddings(_hash_ids(step, tokens))
    assert log == ["write", "record"] + ["sync", "write", "record"] * (len(_STEPS) - 1)
    assert not fences[-1].synchronized


def test_dp_exchange_writes_wait_for_previous_upload(monkeypatch):
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_ASYNC_H2D", True, raising=False)
    dp_size = 4
    engram = _engram(dp_size)
    log, fences = _install_fences(monkeypatch, engram)
    group = SimpleNamespace(world_size=dp_size, cpu_group=object())
    monkeypatch.setattr(common.upstream, "get_engram_dp_group", lambda: group)

    def exchange(raw, scales, cpu_group, size):
        tokens = raw.shape[0] // size

        def wait():
            # The exchanged rows are copied into the host buffer after this.
            assert all(f.synchronized for f in fences)
            return raw[:tokens].repeat(1, size, 1), scales[:tokens].repeat(1, size, 1)

        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(common, "start_exchange_rows", exchange)
    for step, tokens in enumerate(_STEPS):
        engram.prepare_embeddings(_hash_ids(step, tokens, dp_size))
        assert len(fences) == step + 1
    assert log == ["record"] + ["sync", "record"] * (len(_STEPS) - 1)


def test_cpu_staging_records_no_fence():
    assert common._record_upload_fence(torch.device("cpu")) is None


def _first(requests, hits=None):
    return SimpleNamespace(
        seq_lens=torch.ones(requests, dtype=torch.int32), replay_starts_cpu=hits
    )


@pytest.mark.parametrize("hits", [None, (), (0, 0, 0), (0, 128, 0), (256,)])
def test_window_starts_cache_matches_upload(monkeypatch, hits):
    positions = torch.zeros(8, dtype=torch.int64)
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_ASYNC_H2D", False, raising=False)
    expected = v2_model._window_starts(_first(3, hits), positions)
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_ASYNC_H2D", True, raising=False)
    cached = v2_model._window_starts(_first(3, hits), positions)
    assert cached.dtype == expected.dtype and torch.equal(cached, expected)
    again = v2_model._window_starts(_first(3, hits), positions)
    assert (again is cached) == (not any(hits or ()))


def test_window_starts_cache_is_per_batch_size(monkeypatch):
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_ASYNC_H2D", True, raising=False)
    positions = torch.zeros(8, dtype=torch.int64)
    four = v2_model._window_starts(_first(4), positions)
    two = v2_model._window_starts(_first(2), positions)
    assert four.shape == (4,) and two.shape == (2,)
    assert v2_model._window_starts(None, positions).shape == (1,)
    assert v2_model._window_starts(_first(4), positions) is four
