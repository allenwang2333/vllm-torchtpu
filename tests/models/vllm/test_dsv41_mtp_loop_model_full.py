# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vllm_torchtpu.spec_decode.mtp_loop_model."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "vllm_torchtpu"
if (_SRC_ROOT / "spec_decode").is_dir():
    for _pkg_name, _pkg_dir in (
        ("vllm_torchtpu", _SRC_ROOT),
        ("vllm_torchtpu.spec_decode", _SRC_ROOT / "spec_decode"),
    ):
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_pkg_dir)]
        sys.modules.setdefault(_pkg_name, _pkg)

if "vllm" not in sys.modules:
    _vllm_mod = types.ModuleType("vllm")
    _vllm_comp_mod = types.ModuleType("vllm.compilation")
    _vllm_dec_mod = types.ModuleType("vllm.compilation.decorators")
    _vllm_cfg_mod = types.ModuleType("vllm.config")
    _vllm_seq_mod = types.ModuleType("vllm.sequence")

    def _support_torch_compile(
        dynamic_arg_dims: dict[str, Any] | None = None,
    ) -> Any:
        def _decorator(cls: type) -> type:
            cls._dynamic_arg_dims = dynamic_arg_dims  # type: ignore[attr-defined]
            orig_init = cls.__init__

            def _wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
                vllm_config = kwargs.get("vllm_config")
                orig_init(self, *args, **kwargs)
                self.vllm_config = vllm_config

            cls.__init__ = _wrapped_init  # type: ignore[method-assign]
            return cls

        return _decorator

    _vllm_dec_mod.support_torch_compile = _support_torch_compile  # type: ignore[attr-defined]
    _vllm_cfg_mod.VllmConfig = Any  # type: ignore[attr-defined]
    _vllm_seq_mod.IntermediateTensors = Any  # type: ignore[attr-defined]

    sys.modules["vllm"] = _vllm_mod
    sys.modules["vllm.compilation"] = _vllm_comp_mod
    sys.modules["vllm.compilation.decorators"] = _vllm_dec_mod
    sys.modules["vllm.config"] = _vllm_cfg_mod
    sys.modules["vllm.sequence"] = _vllm_seq_mod

from vllm_torchtpu.spec_decode.mtp_loop_model import (  # noqa: E402
    MTP_DYNAMIC_ARG_DIMS,
    MtpLoopStepModel,
)


class _DraftMtpModule(nn.Module):
    """Test double for DeepSeek-V4.1 MTP draft module."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "self": self,
                "input_ids": input_ids,
                "positions": positions,
                "hidden_states": hidden_states,
                "intermediate_tensors": intermediate_tensors,
                "inputs_embeds": inputs_embeds,
                "spec_step_idx": spec_step_idx,
            }
        )
        base = inputs_embeds if inputs_embeds is not None else hidden_states
        return base + float(spec_step_idx + 1)


def test_mtp_dynamic_arg_dims_contract() -> None:
    """Verify MTP_DYNAMIC_ARG_DIMS specifies dim 0 for all 5 tensor arguments."""
    assert MTP_DYNAMIC_ARG_DIMS == {
        "input_ids": 0,
        "positions": 0,
        "hidden_states": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }


def test_mtp_loop_step_model_init_and_forward_delegation() -> None:
    """Verify MtpLoopStepModel binds mtp.forward and passes all arguments through."""
    draft = _DraftMtpModule()
    fake_cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=0),
    )
    loop_model = MtpLoopStepModel(
        vllm_config=fake_cfg,  # type: ignore[arg-type]
        prefix="mtp_index_share_loop",
        mtp=draft,
    )

    assert loop_model.prefix == "mtp_index_share_loop"
    assert loop_model.mtp is draft
    assert loop_model._mtp_forward.__self__ is draft
    assert loop_model._mtp_forward.__func__ is _DraftMtpModule.forward

    input_ids = torch.tensor([10, 20, 30], dtype=torch.int32)
    positions = torch.tensor([1, 2, 3], dtype=torch.int32)
    hidden_states = torch.full((3, 8), 2.0, dtype=torch.float32)

    out_step1 = loop_model.forward(
        input_ids=input_ids,
        positions=positions,
        hidden_states=hidden_states,
        spec_step_idx=1,
    )
    assert torch.equal(out_step1, torch.full((3, 8), 4.0, dtype=torch.float32))
    assert len(draft.calls) == 1
    assert draft.calls[0]["self"] is draft
    assert draft.calls[0]["input_ids"] is input_ids
    assert draft.calls[0]["positions"] is positions
    assert draft.calls[0]["intermediate_tensors"] is None
    assert draft.calls[0]["inputs_embeds"] is None
    assert draft.calls[0]["spec_step_idx"] == 1

    embeds = torch.full((3, 8), 5.0, dtype=torch.float32)
    interm = SimpleNamespace(tensors={"h": hidden_states})
    out_step2 = loop_model.forward(
        input_ids=None,
        positions=positions,
        hidden_states=hidden_states,
        intermediate_tensors=interm,  # type: ignore[arg-type]
        inputs_embeds=embeds,
        spec_step_idx=2,
    )
    assert torch.equal(out_step2, torch.full((3, 8), 8.0, dtype=torch.float32))
    assert len(draft.calls) == 2
    assert draft.calls[1]["intermediate_tensors"] is interm
    assert draft.calls[1]["inputs_embeds"] is embeds
    assert draft.calls[1]["spec_step_idx"] == 2

    draft2 = _DraftMtpModule()
    loop_model.mtp = draft2
    assert loop_model.mtp is draft2
    assert loop_model._mtp_forward.__self__ is draft2
    out_step3 = loop_model.forward(
        input_ids=input_ids,
        positions=positions,
        hidden_states=hidden_states,
        spec_step_idx=3,
    )
    assert torch.equal(out_step3, torch.full((3, 8), 6.0, dtype=torch.float32))
    assert len(draft2.calls) == 1
