# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model registration for the TorchTPU vLLM plugin.

``register_models`` is re-exported here because this module is the target of the
``torchtpu_models`` entry point in ``pyproject.toml``. vLLM resolves that target
at startup, so the name has to live on the top-level package -- alongside
``vllm_torchtpu.layers:register_layers`` and
``vllm_torchtpu.platforms:register_tpu_platform`` -- rather than on the nested
implementation module. Moving it without updating the entry point surfaces only
at serving time, as an ``AttributeError`` from the plugin loader.
"""

from vllm_torchtpu.models.vllm import register_models

__all__ = ["register_models"]
