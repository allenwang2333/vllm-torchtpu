# SPDX-License-Identifier: Apache-2.0
"""Private layer hooks and late, round-local NaN logs."""

import json
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import torch
from torch_tpu._internal.stream.streams import TpuEvent
from vllm.model_executor.models.interfaces import supports_multimodal

from vllm_torchtpu.layers.adapter.custom_ops.hidden_nan_flags_op import record_layer_nan
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclass
class ModelLayerNaNFlags:
    role: str
    layer_ids: tuple[int, ...]
    flags: torch.Tensor


@dataclass
class HiddenNaNReadback:
    round_id: int
    models: tuple[ModelLayerNaNFlags, ...]
    flags_cpu: tuple[torch.Tensor, ...]
    ready_event: TpuEvent

    def consume(self) -> None:
        self.ready_event.synchronize()
        for model, flags in zip(self.models, self.flags_cpu):
            if flags.any().item():
                logger.error(
                    "Hidden NaN detected: role=%s round=%d layer_ids=%s isnan=%s",
                    model.role,
                    self.round_id,
                    list(model.layer_ids),
                    json.dumps(flags.tolist()),
                )


class HiddenNaNDetector:
    """Rank-local booleans; hidden tensors are inspected but never retained."""

    def __init__(self, models: tuple[ModelLayerNaNFlags, ...]) -> None:
        self.models = models
        self.armed_tpu = torch.zeros(1, dtype=torch.bool, device=models[0].flags.device)
        self._armed = False
        self._round_id = 0

    def after_layer(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: torch.Tensor | tuple[Any, ...],
        *,
        model_index: int,
        layer_slot: int,
    ) -> None:
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        record_layer_nan(
            self.models[model_index].flags, self.armed_tpu, hidden, layer_slot
        )

    def arm(self) -> None:
        self.armed_tpu.fill_(True)
        self._armed = True
        self._round_id = 0

    def begin_round(self) -> None:
        self._round_id += 1
        for model in self.models:
            model.flags.zero_()

    def readback(self) -> HiddenNaNReadback | None:
        # Warmup outputs may be discarded; never enqueue an unconsumed D2H then.
        if not self._armed:
            return None
        flags_cpu = tuple(
            model.flags.to("cpu", non_blocking=True) for model in self.models
        )
        event = TpuEvent()
        event.record()
        return HiddenNaNReadback(self._round_id, self.models, flags_cpu, event)


def install_hidden_nan_detector(
    models: list[tuple[str, torch.nn.Module]], device: torch.device
) -> HiddenNaNDetector:
    """Validate the model boundary once and attach standard decoder-layer hooks."""
    model_layers: list[tuple[str, list[tuple[int, torch.nn.Module]]]] = []
    for role, model in models:
        if supports_multimodal(model):
            model = cast(torch.nn.Module, model.get_language_model())
        layers = []
        containers = []
        for name, module in model.named_modules():
            if not isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                continue
            # A block's nested containers are implementation details, not layers.
            if any(
                not parent or name.startswith(parent + ".") for parent in containers
            ):
                continue
            containers.append(name)
            for index, layer in module.named_children():
                if not index.isdecimal():
                    raise ValueError(
                        "Hidden NaN detection requires integer-indexed layer "
                        f"containers: role={role} module={name}.{index}"
                    )
                layers.append((int(index), layer))
        if not layers:
            raise ValueError(
                "Hidden NaN detection requires layers registered in a ModuleList "
                f"or an integer-indexed ModuleDict: role={role}"
            )
        model_layers.append((role, sorted(layers, key=lambda item: item[0])))
    detector = HiddenNaNDetector(
        tuple(
            ModelLayerNaNFlags(
                role,
                tuple(index for index, _ in layers),
                torch.zeros(len(layers), dtype=torch.bool, device=device),
            )
            for role, layers in model_layers
        )
    )
    for model_index, ((role, layers), model) in enumerate(
        zip(model_layers, detector.models)
    ):
        for slot, (_, layer) in enumerate(layers):
            layer.register_forward_hook(
                partial(detector.after_layer, model_index=model_index, layer_slot=slot)
            )
        logger.info(
            "Hidden NaN detection installed: role=%s layers=%d flag_bytes=%d",
            role,
            len(layers),
            model.flags.numel() * model.flags.element_size(),
        )
    return detector
