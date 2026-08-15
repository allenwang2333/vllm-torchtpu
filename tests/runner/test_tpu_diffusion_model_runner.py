import sys
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("vllm_omni")

import torch
from vllm_omni.diffusion.data import DiffusionOutput

from vllm_torchtpu.worker.tpu_diffusion_model_runner import (
    TpuDiffusionModelRunner, _to_cpu)


def test_to_cpu_basic():
    # Test CPU tensor
    cpu_tensor = torch.tensor([1, 2, 3])
    out = _to_cpu(cpu_tensor)
    assert out is cpu_tensor
    assert out.device.type == "cpu"

    # Non-tensor types should be returned as-is
    assert _to_cpu("hello") == "hello"
    assert _to_cpu(42) == 42


@patch("vllm_torchtpu.worker.tpu_diffusion_model_runner.torch_tpu",
       create=True)
def test_to_cpu_tpu_tensor(mock_torch_tpu):
    mock_tpu_tensor = MagicMock(spec=torch.Tensor)
    mock_tpu_tensor.device.type = "tpu"
    mock_cpu_tensor = torch.tensor([1, 2, 3])

    mock_tpu_tensor.detach.return_value.cpu.return_value = mock_cpu_tensor

    # We patch sys.modules temporarily so `import torch_tpu` succeeds inside _to_cpu
    with patch.dict(sys.modules, {"torch_tpu": mock_torch_tpu}):
        out = _to_cpu(mock_tpu_tensor)

        # Verify mark_step was called
        mock_torch_tpu.core.tpu_model.mark_step.assert_called_once()

        # Verify it returns the CPU tensor
        assert out is mock_cpu_tensor


def test_to_cpu_recursive():
    cpu_tensor = torch.tensor([1])

    mock_tpu_tensor = MagicMock(spec=torch.Tensor)
    mock_tpu_tensor.device.type = "tpu"
    mock_tpu_tensor.detach.return_value.cpu.return_value = cpu_tensor

    input_data = [
        mock_tpu_tensor, (mock_tpu_tensor, "string"), {
            "key": mock_tpu_tensor
        }
    ]

    out = _to_cpu(input_data)

    # Verify structure is preserved and tensors are converted
    assert isinstance(out, list)
    assert out[0] is cpu_tensor
    assert isinstance(out[1], tuple)
    assert out[1][0] is cpu_tensor
    assert out[1][1] == "string"
    assert isinstance(out[2], dict)
    assert out[2]["key"] is cpu_tensor


@patch(
    "vllm_omni.diffusion.worker.diffusion_model_runner.DiffusionModelRunner.execute_model"
)
def test_tpu_diffusion_model_runner_execute_model(mock_super_execute):
    runner = object.__new__(TpuDiffusionModelRunner)

    cpu_tensor = torch.tensor([1, 2, 3])
    mock_tpu_tensor = MagicMock(spec=torch.Tensor)
    mock_tpu_tensor.device.type = "tpu"
    mock_tpu_tensor.detach.return_value.cpu.return_value = cpu_tensor

    # Create fake output
    fake_output = DiffusionOutput(output=mock_tpu_tensor)
    mock_super_execute.return_value = fake_output

    result = runner.execute_model(execute_model_req=MagicMock())

    # Verify the tensor was converted to CPU
    assert result.output is cpu_tensor


@patch(
    "vllm_omni.diffusion.worker.diffusion_model_runner.DiffusionModelRunner.execute_model"
)
def test_tpu_diffusion_model_runner_execute_model_list(mock_super_execute):
    runner = object.__new__(TpuDiffusionModelRunner)

    cpu_tensor = torch.tensor([1, 2, 3])
    mock_tpu_tensor = MagicMock(spec=torch.Tensor)
    mock_tpu_tensor.device.type = "tpu"
    mock_tpu_tensor.detach.return_value.cpu.return_value = cpu_tensor

    fake_output1 = DiffusionOutput(output=mock_tpu_tensor)
    fake_output2 = DiffusionOutput(output=mock_tpu_tensor)

    mock_super_execute.return_value = [fake_output1, fake_output2]

    results = runner.execute_model(execute_model_req=MagicMock())

    assert len(results) == 2
    assert results[0].output is cpu_tensor
    assert results[1].output is cpu_tensor
