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
"""FP32 rotary embedding for TPU as out-of-tree CustomOps.

vLLM's CUDA rotary kernel (``csrc/pos_encoding_kernels.cu``) loads the
(activation-dtype) cos/sin cache entries and the q/k values, upcasts all of
them to ``float``, rotates, and casts once on store. The PyTorch reference,
``ApplyRotaryEmb.forward_static``, only does that when
``enable_fp32_compute=True``; the default is ``False`` and
``RotaryEmbedding.forward_static`` never passes it, so on TPU the rotation
``x1 * cos - x2 * sin`` runs in BF16. For RL rollouts the resulting drift in
q/k shows up as trainer/sampler log-prob mismatch.

Two registrations restore the CUDA numerics, following the vLLM
device-extension design (``docs/design/custom_op.md``) rather than patching
vLLM internals. ``CustomOp.__new__`` substitutes an out-of-tree class by the
upstream class name, so each upstream class that needs the fix gets one entry:

* ``ApplyRotaryEmb``: constructed with ``enable_fp32_compute=True``. Covers
  every caller that goes through an instance (the M-RoPE family and its
  subclasses).
* ``RotaryEmbedding`` and every upstream subclass that inherits its
  ``forward_native`` (Llama3, YaRN, NTK, linear, Gemma4 scaling, ...; found by
  walking the class hierarchy): ``forward_static`` calls
  ``ApplyRotaryEmb.forward_static`` directly, so the instance flag does not
  reach it. The TPU subclass replays ``forward_static`` with
  ``enable_fp32_compute=True``.
* ``MRotaryEmbedding`` / ``MRotaryEmbeddingInterleaved`` (Qwen2.5-VL,
  Qwen3-VL, Qwen3.5): precision already comes from ``TpuApplyRotaryEmb``; the
  TPU subclasses exist for the in-graph path below.

``forward_native`` is overridden as well as ``forward_oot``/``forward_tpu`` so
the numerics do not depend on ``CompilationConfig.custom_ops`` enabling the op:
CUDA runs the FP32 kernel in both cases, and so should TPU.

In-graph cos/sin (``TPU_ROPE_IN_GRAPH=1``)
------------------------------------------
Upstream builds a ``[max_position_embeddings, rotary_dim]`` ``cos_sin_cache``
buffer on the host and the compiled graph gathers rows from it. That buffer is
an input of every compiled program (tens of MB at long context) and is pure
math. With the flag on, eligible modules instead keep only the ``[rotary_dim/2]``
``inv_freq`` vector and compute ``cos``/``sin`` of ``positions * inv_freq`` on
the TPU, inside the graph; the big buffer is dropped. Eligible means the cache
is a pure function of position and ``inv_freq``, i.e. the class does not
override ``_compute_cos_sin_cache`` (plain, Llama3, NTK, Gemma4 scaling, and
M-RoPE without YaRN ``scaling_factor``; not linear / dynamic-NTK / YaRN, which
keep the cache path). For M-RoPE the per-frequency T/H/W stream selection that
upstream applies to gathered cos/sin rows is applied to the positions instead
(``mrope_dim_modalities``). The in-graph values are rounded to the cache dtype
before the FP32 rotation, so the numerics stay those of the CUDA kernel
reading an activation-dtype cache.
"""

import torch
from vllm.model_executor.custom_op import CustomOp, op_registry_oot
from vllm.model_executor.layers import rotary_embedding as upstream_rope
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    MRotaryEmbeddingInterleaved,
    RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_INV_FREQ_BUFFER = "tpu_rope_inv_freq"
_MODALITY_BUFFER = "tpu_rope_modality"


@CustomOp.register_oot(name="ApplyRotaryEmb")
class TpuApplyRotaryEmb(ApplyRotaryEmb):
    """``ApplyRotaryEmb`` that always rotates in FP32, like the CUDA kernel."""

    def __init__(
        self,
        enforce_enable: bool = False,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
        **kwargs,
    ) -> None:
        # Mirror the upstream positional signature so callers passing
        # enable_fp32_compute positionally are overridden too, not rejected.
        del enable_fp32_compute  # always FP32 on TPU
        super().__init__(
            enforce_enable=enforce_enable,
            is_neox_style=is_neox_style,
            enable_fp32_compute=True,
            **kwargs,
        )

    def forward_oot(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return self.forward_native(x, cos, sin)

    forward_tpu = forward_oot


def in_graph_cos_sin(
    positions: torch.Tensor, inv_freq: torch.Tensor, cache_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """``cos``/``sin`` of ``positions * inv_freq`` computed on device.

    ``positions`` is ``[N]`` (one position per token) or ``[N, rotary_dim/2]``
    (one position per token and frequency, as M-RoPE needs). Same arithmetic
    as ``RotaryEmbeddingBase._compute_cos_sin_cache`` (FP32 product, FP32
    trig), then rounded to ``cache_dtype`` exactly like the stored cache rows
    would be, so swapping the gather for in-graph compute does not move the
    reference numerics.
    """
    pos = positions.to(torch.float32)
    if pos.ndim == 1:
        pos = pos[:, None]
    freqs = pos * inv_freq[None, :]
    return freqs.cos().to(cache_dtype), freqs.sin().to(cache_dtype)


def is_in_graph_eligible(cls: type[RotaryEmbeddingBase]) -> bool:
    """Cache is a pure function of position and ``inv_freq`` (class-level)."""
    return cls._compute_cos_sin_cache is RotaryEmbeddingBase._compute_cos_sin_cache


def _instance_in_graph_eligible(module: RotaryEmbeddingBase, base_cls: type) -> bool:
    if isinstance(module, MRotaryEmbedding):
        # MRotaryEmbedding only overrides the cache builders to add YaRN;
        # without a scaling factor they delegate to the base implementation.
        return module.scaling_factor is None
    return is_in_graph_eligible(base_cls)


def mrope_dim_modalities(module: MRotaryEmbedding) -> tuple[int, ...] | None:
    """Per-frequency modality (0=T, 1=H, 2=W) for ``[3, N]`` M-RoPE positions.

    Reproduces how upstream picks cos/sin rows from the three position
    streams: the chunked ``mrope_section`` layout, ``apply_interleaved_rope``
    for ``mrope_interleaved``, and ``MRotaryEmbeddingInterleaved.mrope_dim``.
    None when the module has no sections (1-D positions only).
    """
    half = module.rotary_dim // 2
    mrope_dim = getattr(module, "mrope_dim", None)
    if mrope_dim is not None:
        return tuple(int(m) for m in mrope_dim[:half])
    section = module.mrope_section
    if not section:
        return None
    if module.mrope_interleaved:
        out = []
        for ch in range(half):
            if ch % 3 == 1 and ch < section[1] * 3:
                out.append(1)
            elif ch % 3 == 2 and len(section) > 2 and ch < section[2] * 3:
                out.append(2)
            else:
                out.append(0)
        return tuple(out)
    out = []
    for modality, count in enumerate(section):
        out.extend([modality] * int(count))
    return tuple(out)


def _select_mrope_positions(
    positions: torch.Tensor, modality: torch.Tensor
) -> torch.Tensor:
    """``[3, N]`` T/H/W positions -> ``[N, rotary_dim/2]`` per-frequency positions."""
    t, h, w = (p[:, None] for p in positions.reshape(3, -1))
    return torch.where(modality == 1, h, torch.where(modality == 2, w, t))


def _enable_in_graph_rope(self: RotaryEmbeddingBase, base_cls: type) -> None:
    """After upstream init: switch the instance to in-graph cos/sin if asked."""
    self._tpu_rope_in_graph = False
    if not envs.TPU_ROPE_IN_GRAPH:
        return
    cache = getattr(self, "cos_sin_cache", None)
    if not isinstance(cache, torch.Tensor):
        return
    if not _instance_in_graph_eligible(self, base_cls):
        logger.warning_once(
            "TPU_ROPE_IN_GRAPH: %s builds a scaled cos_sin_cache; keeping its buffer.",
            base_cls.__name__,
        )
        return
    if isinstance(self, MRotaryEmbedding):
        modality = mrope_dim_modalities(self)
        if modality is not None:
            self.register_buffer(
                _MODALITY_BUFFER,
                torch.tensor(modality, dtype=torch.int64, device=cache.device),
                persistent=False,
            )
    inv_freq = self._compute_inv_freq(self.base).to(
        device=cache.device, dtype=torch.float32
    )
    self.register_buffer(_INV_FREQ_BUFFER, inv_freq, persistent=False)
    self._tpu_rope_cache_dtype = cache.dtype
    self._tpu_rope_in_graph = True
    # Drop the host-built table; nothing on the in-graph path reads it.
    for buf_name in ("cos_sin_cache", "cos_sin_cache_bf16"):
        self._buffers.pop(buf_name, None)
        self.__dict__.pop(buf_name, None)
    logger.info_once(
        "TPU_ROPE_IN_GRAPH: %s computes cos/sin in-graph (dropped %s cache).",
        base_cls.__name__,
        tuple(cache.shape),
    )


def _rotate_qk_fp32(
    self: RotaryEmbeddingBase,
    num_tokens: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Rotate the leading ``rotary_dim`` of every head in FP32 (CUDA numerics)."""
    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_size)
    query_rot = query[..., : self.rotary_dim]
    query_pass = query[..., self.rotary_dim :]
    query_rot = ApplyRotaryEmb.forward_static(
        query_rot, cos, sin, self.is_neox_style, enable_fp32_compute=True
    )
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    if key is not None:
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = ApplyRotaryEmb.forward_static(
            key_rot, cos, sin, self.is_neox_style, enable_fp32_compute=True
        )
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


def _rotary_forward_fp32(
    self: RotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``RotaryEmbedding.forward_static`` with ``enable_fp32_compute=True``.

    cos/sin come either from the cache matched to the query dtype or, with
    ``TPU_ROPE_IN_GRAPH``, from in-graph trig rounded to that same dtype: the
    CUDA kernel reads an activation-dtype cache and upcasts the loaded
    entries, so the rounded values are part of the reference numerics.
    """
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    if getattr(self, "_tpu_rope_in_graph", False):
        cos, sin = in_graph_cos_sin(
            positions, getattr(self, _INV_FREQ_BUFFER), self._tpu_rope_cache_dtype
        )
        cos, sin = cos.to(query.dtype), sin.to(query.dtype)
    else:
        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        cos_sin = cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)
    return _rotate_qk_fp32(self, num_tokens, cos, sin, query, key)


def _mrope_forward_fp32(
    self: MRotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """M-RoPE forward: in-graph cos/sin when enabled, else upstream's.

    Upstream's cached path already rotates in FP32 here because
    ``self.apply_rotary_emb`` is ``TpuApplyRotaryEmb``. The in-graph path
    selects, per frequency, the T/H/W position stream the upstream
    ``mrope_section`` / interleaved layouts would have gathered cos/sin for,
    so it is bit-identical to the cached result.
    """
    if not getattr(self, "_tpu_rope_in_graph", False):
        if isinstance(self, MRotaryEmbeddingInterleaved):
            return MRotaryEmbeddingInterleaved.forward(self, positions, query, key)
        return MRotaryEmbedding.forward_native(self, positions, query, key, offsets)

    assert positions.ndim in (1, 2)
    if offsets is not None:
        positions = positions + offsets
    num_tokens = positions.shape[-1]
    modality = getattr(self, _MODALITY_BUFFER, None)
    if positions.ndim == 2:
        assert modality is not None, "3-D M-RoPE positions need mrope_section"
        pos = _select_mrope_positions(positions, modality)
    else:
        pos = positions
    cos, sin = in_graph_cos_sin(
        pos, getattr(self, _INV_FREQ_BUFFER), self._tpu_rope_cache_dtype
    )
    cos, sin = cos.to(query.dtype), sin.to(query.dtype)
    return _rotate_qk_fp32(self, num_tokens, cos, sin, query, key)


def _all_subclasses(cls: type) -> set[type]:
    out: set[type] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _all_subclasses(sub)
    return out


def _uses_static_rotary_forward(cls: type[RotaryEmbedding]) -> bool:
    """True if ``cls.forward_native`` resolves to ``RotaryEmbedding``'s.

    Classes with their own forward_native (MRotaryEmbedding family, FoPE,
    DeepSeek scaling, ...) do not run the static path; MRotaryEmbedding and
    MRotaryEmbeddingInterleaved get explicit TPU classes below.
    """
    for c in cls.__mro__:
        if "forward_native" in c.__dict__:
            return c is RotaryEmbedding
    return False


def _discover_static_rope_classes() -> tuple[type[RotaryEmbedding], ...]:
    """Upstream RotaryEmbedding classes whose forward is the static path.

    Discovered from the class hierarchy (``vllm.model_executor.layers.
    rotary_embedding`` imports every variant), so new upstream scaling
    variants are picked up without editing this file.
    """
    upstream_prefix = upstream_rope.__name__ + "."
    found = {
        c
        for c in _all_subclasses(RotaryEmbedding)
        if c.__module__.startswith(upstream_prefix)
        and _uses_static_rotary_forward(c)
        and c.__name__ not in op_registry_oot
    }
    return (RotaryEmbedding, *sorted(found, key=lambda c: c.__name__))


_STATIC_ROPE_CLASSES = _discover_static_rope_classes()


def _make_tpu_rope_class(
    base: type[RotaryEmbeddingBase], forward, *, override_forward: bool = False
) -> type[RotaryEmbeddingBase]:
    def __init__(self, *args, **kwargs) -> None:
        base.__init__(self, *args, **kwargs)
        _enable_in_graph_rope(self, base)

    namespace = {
        "__module__": __name__,
        "__doc__": f"``{base.__name__}`` rotating in FP32 (CUDA numerics).",
        "__init__": __init__,
        "forward_native": forward,
        "forward_oot": forward,
        "forward_tpu": forward,
    }
    if override_forward:
        # MRotaryEmbeddingInterleaved bypasses CustomOp dispatch with its own
        # forward(); route that through the TPU implementation as well.
        namespace["forward"] = forward
    tpu_cls = type(f"Tpu{base.__name__}", (base,), namespace)
    return CustomOp.register_oot(name=base.__name__)(tpu_cls)


def _make_static_rope_class(base: type[RotaryEmbedding]) -> type[RotaryEmbedding]:
    assert _uses_static_rotary_forward(base), (
        f"{base.__name__} defines its own forward_native; it needs its own TPU "
        "override instead of the shared RotaryEmbedding.forward_static replay."
    )
    return _make_tpu_rope_class(base, _rotary_forward_fp32)


TPU_ROPE_CLASSES: dict[str, type[RotaryEmbeddingBase]] = {
    base.__name__: _make_static_rope_class(base) for base in _STATIC_ROPE_CLASSES
}
# M-RoPE (Qwen2.5-VL / Qwen3-VL / Qwen3.5): own forward_native, so the
# static-path discovery above does not cover them. Subclasses with their own
# forward_native (Bailing, Ernie4.5-VL) stay upstream and get FP32 rotation
# through TpuApplyRotaryEmb only.
TPU_ROPE_CLASSES["MRotaryEmbedding"] = _make_tpu_rope_class(
    MRotaryEmbedding, _mrope_forward_fp32
)
TPU_ROPE_CLASSES["MRotaryEmbeddingInterleaved"] = _make_tpu_rope_class(
    MRotaryEmbeddingInterleaved, _mrope_forward_fp32, override_forward=True
)
TpuRotaryEmbedding = TPU_ROPE_CLASSES["RotaryEmbedding"]
TpuMRotaryEmbedding = TPU_ROPE_CLASSES["MRotaryEmbedding"]

logger.info(
    "Registered FP32 rotary embedding for TPU: ApplyRotaryEmb, %s.",
    ", ".join(TPU_ROPE_CLASSES),
)
