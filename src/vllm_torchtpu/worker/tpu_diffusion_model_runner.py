import torch

from vllm_torchtpu.utils import is_tpu_device

try:
    from vllm_omni.diffusion.data import DiffusionOutput  # type: ignore
    from vllm_omni.diffusion.worker.diffusion_model_runner import \
        DiffusionModelRunner  # type: ignore
except ImportError:
    DiffusionOutput = None
    DiffusionModelRunner = object


def _to_cpu(val):
    try:
        import torch_tpu
        if hasattr(torch_tpu, "mark_step"):
            torch_tpu.mark_step()
    except (ImportError, AttributeError):
        pass

    def _convert(v):
        if isinstance(v, torch.Tensor):
            if is_tpu_device(v):
                return v.detach().cpu()
            return v

        elif isinstance(v, list):
            return [_convert(x) for x in v]
        elif isinstance(v, tuple):
            return tuple(_convert(x) for x in v)
        elif isinstance(v, dict):
            return {k: _convert(x) for k, x in v.items()}
        return v

    return _convert(val)


class TpuDiffusionModelRunner(DiffusionModelRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute_model(self, *args, **kwargs) -> DiffusionOutput:
        outputs = super().execute_model(*args, **kwargs)
        if isinstance(outputs, list):
            for out in outputs:
                out.output = _to_cpu(out.output)
        else:
            outputs.output = _to_cpu(outputs.output)
        return outputs
