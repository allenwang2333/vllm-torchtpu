# SPDX-License-Identifier: Apache-2.0
"""API-server (frontend) patches, installed from the general plugin.

``TPU_DETOK_PRIME_TAIL_TOKENS``: vLLM primes every request's
``tokenizers.decoders.DecodeStream`` with the full prompt, and the stream's
first ``step`` decodes all of it on the API server's event loop (~35 ms at
120k prompt tokens, ~0.3 s at 1M), stalling every other request's streaming.
The text a step emits depends only on the last few prompt tokens, so priming
with a short tail streams the same text.

``--renderer-num-workers > 1`` with DeepSeek-V4: vLLM's HF renderer gives each
renderer thread its own tokenizer copy (``maybe_make_thread_pool``), but the
DeepSeek-V4 renderer shares one, and concurrent encodes on one HF fast
tokenizer fail with ``RuntimeError: Already borrowed`` (HTTP 500).
``TPU_DSV4_RENDERER_THREAD_POOL=1`` gives it the same per-thread copies.
"""

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class _TailPromptRequest:
    """Read-only view of an ``EngineCoreRequest`` with a truncated prompt."""

    __slots__ = ("_request", "prompt_token_ids")

    def __init__(self, request, tail_tokens: int):
        self._request = request
        ids = request.prompt_token_ids
        self.prompt_token_ids = ids[-tail_tokens:] if ids is not None else None

    def __getattr__(self, name):
        return getattr(self._request, name)


def patch_detok_prime_tail(tail_tokens: int) -> None:
    from vllm.v1.engine import detokenizer

    cls = detokenizer.FastIncrementalDetokenizer
    if getattr(cls, "_tpu_prime_tail_tokens", None) is not None:
        return
    orig_init = cls.__init__

    def __init__(self, tokenizer, request):
        orig_init(self, tokenizer, _TailPromptRequest(request, tail_tokens))

    cls.__init__ = __init__
    cls._tpu_prime_tail_tokens = tail_tokens
    logger.info("Detokenizer primed with the last %d prompt tokens", tail_tokens)


def patch_dsv4_renderer_thread_pool() -> None:
    import copy

    from vllm.renderers import deepseek_v4
    from vllm.tokenizers.hf import maybe_make_thread_pool

    cls = deepseek_v4.DeepseekV4Renderer
    if getattr(cls, "_tpu_thread_pool_patched", False):
        return
    orig_init = cls.__init__

    def __init__(self, config, tokenizer):
        workers = config.model_config.renderer_num_workers
        if tokenizer is not None and workers > 1:
            # A copy, so the class swap below leaves the caller's tokenizer as is.
            tokenizer = copy.copy(tokenizer)
        orig_init(self, config, tokenizer)
        if self.tokenizer is not None and workers > 1:
            maybe_make_thread_pool(self.tokenizer, workers + 1)

    cls.__init__ = __init__
    cls._tpu_thread_pool_patched = True


def install_frontend_patches() -> None:
    from vllm_torchtpu import envs

    if envs.TPU_DSV4_RENDERER_THREAD_POOL:
        patch_dsv4_renderer_thread_pool()
    if envs.TPU_DETOK_PRIME_TAIL_TOKENS > 0:
        patch_detok_prime_tail(envs.TPU_DETOK_PRIME_TAIL_TOKENS)
