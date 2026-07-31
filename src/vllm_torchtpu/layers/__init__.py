import os

from vllm_torchtpu.layers.vllm import attention as attention
from vllm_torchtpu.models.vllm import register_models


def register_layers() -> None:
    """Plugin entry point (``torchtpu_layers``).

    ``models.vllm.register_models()`` registers the in-plugin Kimi model under
    the architecture ``KimiLinearForCausalLM`` -- which is also the architecture a
    plain Kimi-Linear checkpoint declares. Registering it therefore *replaces* the
    validated Kimi-Linear path (vLLM's in-tree ``kimi_linear.py`` with only the KDA
    layer substituted via ``PluggableLayer.register_oot``, set up by
    ``models.register_models()`` from the separate ``torchtpu_models`` entry
    point) on every Kimi-Linear run.

    Until the two converge, keep it opt-in: Kimi-Linear stays on the path that
    scores 0.93 on gsm8k by default, and K3 bring-up sets
    ``TPU_ENABLE_KIMI_K3_MODEL=1``. That also preserves a known-good
    implementation to bisect against if K3 numerics go wrong.
    """
    if os.environ.get("TPU_ENABLE_KIMI_K3_MODEL") == "1":
        register_models()
