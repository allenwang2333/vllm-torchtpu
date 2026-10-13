# Copyright 2025 Google LLC
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

from dataclasses import dataclass

import torch
from jax.sharding import Mesh


@dataclass(slots=True)
class VllmModelWrapperContext:
    # TODO need discussion, ullm use this context to manager kv_caches
    # while vllm binds the KV cache to the attention layer
    mesh: Mesh
    kv_caches: list[torch.Tensor] | None = None
    layer_name_to_kvcache_index: dict[str, int] | None = None
    vllm_config: object | None = None
    # Full-model block-major KV cache bundle shared across layers. When set,
    # attention forward execution routes through the bundled RPA custom op with
    # in-place buffer donation, bypassing the per-layer tensor views bound by vLLM.
    kv_cache_bundle: torch.Tensor | None = None
    _prev: object = None

    def __enter__(self) -> "VllmModelWrapperContext":
        global _vllm_model_wrapper_context
        self._prev, _vllm_model_wrapper_context = _vllm_model_wrapper_context, self
        return self

    def __exit__(
        self,
        exc_type: object = None,
        exc_val: object = None,
        exc_tb: object = None,
    ) -> None:
        global _vllm_model_wrapper_context
        _vllm_model_wrapper_context, self._prev = self._prev, None  # type: ignore[assignment]


_vllm_model_wrapper_context: VllmModelWrapperContext | None = None
_last_context: VllmModelWrapperContext | None = None
_context_cache: dict[tuple[int, int, int, int, int], VllmModelWrapperContext] = {}
_reentrant_cache: dict[tuple[int, int, int, int, int], VllmModelWrapperContext] = {}


def get_vllm_model_wrapper_context() -> VllmModelWrapperContext:
    if (ctx := _vllm_model_wrapper_context) is not None:
        return ctx
    raise AssertionError(
        "VllmModelWrapperContext is not set. "
        "Please use `set_vllm_model_wrapper_context` to set the "
        "VllmModelWrapperContext."
    )


def set_vllm_model_wrapper_context(
    *,
    kv_caches: list[torch.Tensor] | None = None,
    mesh: Mesh,
    layer_name_to_kvcache_index: dict[str, int] | None = None,
    vllm_config: object | None = None,
    kv_cache_bundle: torch.Tensor | None = None,
) -> VllmModelWrapperContext:
    global _last_context
    if (
        (ctx := _last_context) is not None
        and ctx.mesh is mesh
        and ctx.kv_caches is kv_caches
        and ctx.layer_name_to_kvcache_index is layer_name_to_kvcache_index
        and ctx.vllm_config is vllm_config
        and ctx.kv_cache_bundle is kv_cache_bundle
        and ctx is not _vllm_model_wrapper_context
        and ctx._prev is None
    ):
        return ctx
    key = (
        id(mesh),
        id(kv_caches),
        id(layer_name_to_kvcache_index),
        id(vllm_config),
        id(kv_cache_bundle),
    )
    cached = _context_cache.get(key)
    if (
        cached is not None
        and cached.mesh is mesh
        and cached.kv_caches is kv_caches
        and cached.layer_name_to_kvcache_index is layer_name_to_kvcache_index
        and cached.vllm_config is vllm_config
        and cached.kv_cache_bundle is kv_cache_bundle
    ):
        if cached is not _vllm_model_wrapper_context and cached._prev is None:
            _last_context = cached
            return cached
        rctx = _reentrant_cache.get(key)
        if (
            rctx is not None
            and rctx is not _vllm_model_wrapper_context
            and rctx._prev is None
            and rctx.mesh is mesh
            and rctx.kv_caches is kv_caches
            and rctx.layer_name_to_kvcache_index is layer_name_to_kvcache_index
            and rctx.vllm_config is vllm_config
            and rctx.kv_cache_bundle is kv_cache_bundle
        ):
            return rctx
        rctx = VllmModelWrapperContext(
            kv_caches=kv_caches,
            mesh=mesh,
            layer_name_to_kvcache_index=layer_name_to_kvcache_index,
            vllm_config=vllm_config,
            kv_cache_bundle=kv_cache_bundle,
        )
        _reentrant_cache[key] = rctx
        return rctx
    ctx = VllmModelWrapperContext(
        kv_caches=kv_caches,
        mesh=mesh,
        layer_name_to_kvcache_index=layer_name_to_kvcache_index,
        vllm_config=vllm_config,
        kv_cache_bundle=kv_cache_bundle,
    )
    _context_cache[key] = ctx
    _last_context = ctx
    return ctx
