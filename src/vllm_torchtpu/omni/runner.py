# SPDX-License-Identifier: Apache-2.0
"""TPU model runner with Omni connector mixin support."""

from typing import Any
import numpy as np
import torch

from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class TPUOmniModelRunner(OmniConnectorModelRunnerMixin, TPUModelRunner):
    """TPU Model Runner with Omni Connector Mixin for inter-stage transfers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_intermediate_buffer: dict[str, dict[str, Any]] = {}
        self._omni_num_scheduled_tokens_np: np.ndarray | None = None
        self._omni_last_model_output: object | None = None
        self.omni_prefix_cache = None
        self._sampled_token_ids_cpu_override = None
        self._omni_query_start_loc_model_kwarg = False
        if hasattr(self, "model_config"):
            try:
                self.init_omni_connectors(self.model_config)
            except Exception:
                pass

    def execute_model(self, scheduler_output: Any) -> Any:
        output = super().execute_model(scheduler_output)
        if hasattr(self, "attach_omni_connector_output"):
            try:
                return self.attach_omni_connector_output(output)
            except Exception:
                pass
        return output
