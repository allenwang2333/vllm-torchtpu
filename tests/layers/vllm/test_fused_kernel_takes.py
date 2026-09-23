"""The fused EP kernel takes a step only when the rows it sees are this rank's
own: always under data parallelism, and under tensor parallelism only once
sequence parallelism has split the step."""
import pytest

from vllm_torchtpu.layers.vllm import fused_moe_ep_v3, sequence_parallel
from vllm_torchtpu.layers.vllm.quantization import fp8


@pytest.mark.parametrize("rows,expected", [(32, False), (48, True),
                                           (256, True)])
def test_data_parallel_rows_are_the_ranks_own(monkeypatch, rows, expected):
    monkeypatch.setattr(fused_moe_ep_v3, "_tensor_parallel_size", lambda: 1)
    assert fp8._fused_kernel_takes(rows, 33) is expected


@pytest.mark.parametrize("rows", [48, 64, 128, 256])
def test_replicated_rows_under_tensor_parallelism_stay_below(
        monkeypatch, rows):
    monkeypatch.setattr(fused_moe_ep_v3, "_tensor_parallel_size", lambda: 8)
    monkeypatch.setattr(sequence_parallel, "_ACTIVE", False)
    assert fp8._fused_kernel_takes(rows, 33) is False


def test_split_rows_under_tensor_parallelism_go_to_the_kernel(monkeypatch):
    monkeypatch.setattr(fused_moe_ep_v3, "_tensor_parallel_size", lambda: 8)
    monkeypatch.setattr(sequence_parallel, "_ACTIVE", True)
    assert fp8._fused_kernel_takes(33, 33) is True
    assert fp8._fused_kernel_takes(32, 33) is False


def test_the_width_recorded_at_loading_wins_over_the_config(monkeypatch):
    """The warm-up traces run outside vLLM's config context, where the config
    lookup reads 1; the width recorded at weight loading decides instead."""
    monkeypatch.setattr(fused_moe_ep_v3, "_tensor_parallel_size", lambda: 1)
    monkeypatch.setattr(fused_moe_ep_v3, "_TP_SIZE", 8)
    monkeypatch.setattr(sequence_parallel, "_ACTIVE", False)
    assert fused_moe_ep_v3.tensor_parallel_size() == 8
    assert fp8._fused_kernel_takes(48, 33) is False
    monkeypatch.setattr(fused_moe_ep_v3, "_TP_SIZE", None)
    assert fused_moe_ep_v3.tensor_parallel_size() == 1
