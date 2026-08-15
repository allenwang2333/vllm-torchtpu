# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepSeek-V4 model loading compatibility helpers."""

from contextlib import contextmanager

from vllm.config import VllmConfig

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@contextmanager
def _maybe_patch_for_deepseek_v4(vllm_config: VllmConfig):
    """Compatibility context manager kept for runner invocation."""
    # Bypasses a Dynamo json serialization bug during metrics logging.
    try:
        from torch._dynamo import utils as dynamo_utils
        dynamo_utils._get_dynamo_config_for_logging = lambda: {}
    except (ImportError, AttributeError):
        pass

    yield
