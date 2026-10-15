# SPDX-License-Identifier: Apache-2.0
"""Private layer hooks and sampled, round-local first-NaN snapshots."""

import os
import socket
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch_tpu._internal.stream.streams import TpuEvent
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import supports_multimodal

from vllm_torchtpu.layers.adapter.custom_ops.hidden_nan_flags_op import record_layer_nan
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import ModelConfig


@dataclass
class ModelHiddenNaNSnapshot:
    role: str
    layer_ids: tuple[int, ...]
    state: torch.Tensor
    hidden: torch.Tensor
    input_ids: torch.Tensor
    current_input_ids: torch.Tensor
    current_input_rows: torch.Tensor


@dataclass
class HiddenNaNReadback:
    round_id: int
    req_ids: tuple[str, ...]
    models: tuple[ModelHiddenNaNSnapshot, ...]
    states_cpu: tuple[torch.Tensor, ...]
    snapshots: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    ready_event: TpuEvent
    dump_dir: Path

    def consume(self) -> None:
        self.ready_event.synchronize()
        for model, state, (hidden, tokens) in zip(
            self.models, self.states_cpu, self.snapshots
        ):
            slot, phase, hidden_rows, token_rows = state.tolist()
            if slot < 0 or not 1 <= phase <= 3:
                continue
            layer_id = model.layer_ids[slot]
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            path = self.dump_dir / (
                f"hidden-nan-{socket.gethostname()}-{os.getpid()}-"
                f"{model.role}-round{self.round_id}-layer{layer_id}.pt"
            )
            torch.save(
                {
                    "role": model.role,
                    "round": self.round_id,
                    "layer_id": layer_id,
                    "nan_event_in_window": phase,
                    "req_ids": list(self.req_ids),
                    "hidden_states": hidden.to("cpu")[:hidden_rows].clone(),
                    "input_ids": tokens.to("cpu")[:token_rows].clone(),
                },
                path,
            )
            logger.error(
                "Hidden NaN snapshot saved: role=%s round=%d layer=%d "
                "event_in_window=%d hidden_shape=%s input_tokens=%d path=%s",
                model.role,
                self.round_id,
                layer_id,
                phase,
                (hidden_rows, hidden.shape[1]),
                token_rows,
                path,
            )


class HiddenNaNDetector:
    """Feature-owned buffers capture the first bad layer across a round's forwards."""

    def __init__(
        self, models: tuple[ModelHiddenNaNSnapshot, ...], dump_dir: Path
    ) -> None:
        self.models = models
        self.dump_dir = dump_dir
        self.armed_tpu = torch.zeros(1, dtype=torch.bool, device=models[0].state.device)
        self._armed = False
        self._round_id = 0
        self._req_ids: tuple[str, ...] = ()

    def record_input_ids(self, input_ids: torch.Tensor) -> None:
        # The target may receive only inputs_embeds. Preserve its actual device
        # tokens after async placeholder substitution, before that conversion.
        self._bind_input_ids(0, input_ids)

    def _bind_input_ids(self, model_index: int, input_ids: torch.Tensor) -> None:
        model = self.models[model_index]
        model.current_input_ids[: input_ids.shape[0]].copy_(input_ids)
        model.current_input_rows.fill_(input_ids.shape[0])

    def before_model(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        model_index: int,
    ) -> None:
        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else inputs[0]
        if input_ids is not None:
            self._bind_input_ids(model_index, input_ids)

    def before_embedding(
        self, module: torch.nn.Module, inputs: tuple[Any, ...], *, model_index: int
    ) -> None:
        # MTP loop programs call the model's forward unbound, bypassing its
        # root prehook. The token embedding still receives each step's IDs.
        self._bind_input_ids(model_index, inputs[0])

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
        model = self.models[model_index]
        record_layer_nan(
            model.state,
            model.hidden,
            model.input_ids,
            self.armed_tpu,
            hidden,
            model.current_input_ids,
            model.current_input_rows,
            layer_slot,
        )

    def arm(self) -> None:
        self.armed_tpu.fill_(False)
        self._armed = True
        self._round_id = 0
        for model in self.models:
            model.state.copy_(
                torch.tensor(
                    [-1, 0, 0, 0], dtype=torch.int32, device=model.state.device
                )
            )

    def begin_round(self, req_ids: list[str]) -> None:
        self._round_id += 1
        self._req_ids = tuple(req_ids)
        self.armed_tpu.fill_(self._armed)
        for model in self.models:
            model.state[0].fill_(-1)

    def readback(self) -> HiddenNaNReadback | None:
        # Warmup outputs may be discarded; never enqueue an unconsumed D2H then.
        if not self._armed:
            return None
        states_cpu = tuple(
            model.state.to("cpu", non_blocking=True) for model in self.models
        )
        # Snapshot outputs are never donated: a clone can alias its PJRT buffer.
        # Retain this round's values before later forwards replace the buffers.
        snapshots = tuple(
            (model.hidden.clone(), model.input_ids.clone()) for model in self.models
        )
        event = TpuEvent()
        event.record()
        self.armed_tpu.fill_(False)
        return HiddenNaNReadback(
            self._round_id,
            self._req_ids,
            self.models,
            states_cpu,
            snapshots,
            event,
            self.dump_dir,
        )


def install_hidden_nan_detector(
    models: list[tuple[str, torch.nn.Module, "ModelConfig", int]],
    device: torch.device,
    *,
    dump_dir: Path | None = None,
) -> HiddenNaNDetector:
    """Validate the model boundary once and attach standard decoder-layer hooks."""
    model_layers: list[tuple[str, list[tuple[int, torch.nn.Module]]]] = []
    model_embeddings: list[list[torch.nn.Module]] = []
    for role, model, _, _ in models:
        if supports_multimodal(model):
            model = cast(torch.nn.Module, model.get_language_model())
        model_embeddings.append(
            [
                module
                for name, module in model.named_modules()
                if (
                    isinstance(module, VocabParallelEmbedding)
                    and not isinstance(module, ParallelLMHead)
                )
                # The draft loader replaces embed_tokens with nn.Embedding
                # when its TP differs. Position embeddings are not token inputs.
                or (
                    name.rsplit(".", 1)[-1] == "embed_tokens"
                    and isinstance(module, torch.nn.Embedding)
                )
            ]
        )
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
    # Target and Draft may share a token embedding. Both hooks receive its
    # inputs, including draft query shapes larger than the target's buckets.
    token_capacity = max(capacity for _, _, _, capacity in models)
    detector = HiddenNaNDetector(
        tuple(
            ModelHiddenNaNSnapshot(
                role,
                tuple(index for index, _ in layers),
                torch.tensor([-1, 0, 0, 0], dtype=torch.int32, device=device),
                torch.zeros(
                    (capacity, config.get_hidden_size()),
                    dtype=cast(torch.dtype, config.dtype),
                    device=device,
                ),
                torch.zeros(token_capacity, dtype=torch.int32, device=device),
                torch.zeros(token_capacity, dtype=torch.int32, device=device),
                torch.zeros(1, dtype=torch.int32, device=device),
            )
            for (role, layers), (_, _, config, capacity) in zip(model_layers, models)
        ),
        dump_dir
        if dump_dir is not None
        else Path(tempfile.gettempdir()) / "vllm-hidden-nan",
    )
    for model_index, ((_, model, _, _), embeddings) in enumerate(
        zip(models, model_embeddings)
    ):
        if embeddings:
            for embedding in embeddings:
                embedding.register_forward_pre_hook(
                    partial(detector.before_embedding, model_index=model_index)
                )
        else:
            model.register_forward_pre_hook(
                partial(detector.before_model, model_index=model_index),
                with_kwargs=True,
            )
    for model_index, ((role, layers), model) in enumerate(
        zip(model_layers, detector.models)
    ):
        for slot, (_, layer) in enumerate(layers):
            layer.register_forward_hook(
                partial(detector.after_layer, model_index=model_index, layer_slot=slot)
            )
        logger.info(
            "Hidden NaN detection installed: role=%s layers=%d snapshot_shape=%s "
            "sample=3/100_nan_rounds dump_dir=%s",
            role,
            len(layers),
            tuple(model.hidden.shape),
            detector.dump_dir,
        )
    return detector
