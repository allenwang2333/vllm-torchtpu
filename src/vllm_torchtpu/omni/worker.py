# SPDX-License-Identifier: Apache-2.0
"""TPU Worker with Omni Model Runner support."""

from vllm_torchtpu.worker.tpu_worker import TPUWorker
from vllm_torchtpu.omni.runner import TPUOmniModelRunner


class TPUOmniWorker(TPUWorker):
    """TPU Worker configured with TPUOmniModelRunner."""
    model_runner_cls = TPUOmniModelRunner
