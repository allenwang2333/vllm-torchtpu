"""Scoped upstream device bindings and V4.1 tensor helpers."""

import math
from contextlib import contextmanager
from threading import RLock

import torch

_lock = RLock()


@contextmanager
def device_hooks(module, **replacements):
    with _lock:
        previous = {name: getattr(module, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(module, name, value)
            yield
        finally:
            for name, value in previous.items():
                setattr(module, name, value)


def bind_device_ops(function, *, namespace_suffix="", **operations):
    """Reuse upstream bytecode with per-adapter device function bindings.

    Unlike runtime module monkeypatches, bindings are immutable during forward
    and TorchDynamo can trace them without locks or global mutation. A second
    binding of the same function needs its own ``namespace_suffix``: the
    namespace is registered by name, and Dynamo guards resolve through it.
    """
    import sys
    from types import FunctionType, ModuleType

    # Dynamo guards resolve global functions by module name. A copied globals
    # dict with the upstream name would make guards inspect the GPU operation.
    name = f"{__name__}.{function.__qualname__.replace('.', '_')}{namespace_suffix}"
    namespace = ModuleType(name)
    namespace.__dict__.update(function.__globals__ | operations | {"__name__": name})
    sys.modules[name] = namespace
    bound = FunctionType(
        function.__code__,
        namespace.__dict__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    bound.__kwdefaults__ = function.__kwdefaults__
    bound.__annotations__ = function.__annotations__
    bound.__qualname__ = function.__qualname__
    setattr(namespace, function.__name__, bound)
    return bound


def compact_rope_caches(attentions, max_model_len):
    """Own compact serving tables without changing upstream's shared YaRN state.

    Frequencies and their original context stay unchanged. Only rows reachable
    by this engine are copied; layers sharing an upstream table share one copy.
    """
    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")
    compact = {}
    for attention in attentions:
        original = attention.rotary_emb.cos_sin_cache
        key = id(original)
        if key not in compact:
            compact[key] = (
                original[:max_model_len].clone()
                if original.shape[0] > max_model_len
                else original
            )
        attention._tpu_rope_cache = compact[key]


_rotary_freq_cache: dict[tuple, torch.Tensor] = {}


def _get_rotary_frequencies(dim: int, base: float, yarn, device: torch.device):
    yarn_key = (
        (
            yarn["original_max_position_embeddings"],
            yarn["beta_fast"],
            yarn["beta_slow"],
            yarn["factor"],
        )
        if yarn and yarn["original_max_position_embeddings"] > 0
        else None
    )
    cache_key = (dim, base, device, yarn_key)
    cached = _rotary_freq_cache.get(cache_key)
    if cached is not None:
        return cached
    frequencies = 1 / base ** (
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
    )
    if yarn_key is not None:
        original, beta_fast, beta_slow, factor = yarn_key

        def correction(rotations):
            return (
                dim
                * math.log(original / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction(beta_fast)), 0)
        high = min(math.ceil(correction(beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, device=device, dtype=torch.float32) - low)
            / max(high - low, 1e-3)
        ).clamp(0, 1)
        frequencies = frequencies * (1 - ramp + ramp / factor)
    _rotary_freq_cache[cache_key] = frequencies
    return frequencies


def rotary(x, positions, config, *, compressed, inverse=False):
    dim = config["qk_rope_head_dim"]
    if dim < 2 or dim % 2 or dim > x.shape[-1]:
        raise ValueError("invalid rotary dimension")
    base = config["compress_rope_theta"] if compressed else config["rope_theta"]
    yarn = (
        (config.get("rope_scaling") or config.get("rope_parameters"))
        if compressed
        else None
    )
    frequencies = _get_rotary_frequencies(dim, base, yarn, x.device)
    pos_f32 = (
        positions
        if positions.device == x.device and positions.dtype == torch.float32
        else positions.to(device=x.device, dtype=torch.float32)
    )
    angles = pos_f32[:, None] * frequencies
    shape = (1, len(positions), *((1,) * (x.ndim - 3)), dim // 2)
    cosine, sine = angles.cos().reshape(shape), angles.sin().reshape(shape)
    if inverse:
        sine = -sine
    x_rope = x[..., -dim:]
    x_rope_f32 = x_rope if x.dtype == torch.float32 else x_rope.float()
    pairs = x_rope_f32.unflatten(-1, (-1, 2))
    real, imag = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack(
        (real * cosine - imag * sine, real * sine + imag * cosine), -1
    ).flatten(-2)
    rotated_out = rotated if x.dtype == torch.float32 else rotated.to(x.dtype)
    if dim == x.shape[-1]:
        return rotated_out
    return torch.cat((x[..., :-dim], rotated_out), -1)


def round_fp8_activation(x, block_size=32):
    if x.shape[-1] % block_size:
        raise ValueError("FP8 block size must divide contraction dimension")
    x_f32 = x if x.dtype == torch.float32 else x.float()
    blocks = x_f32.unflatten(-1, (-1, block_size))
    exp = torch.ceil(
        torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)
    ).to(torch.int32)
    inv_scales = ((127 - exp) << 23).view(torch.float32)
    scales = ((exp + 127) << 23).view(torch.float32)
    values = (blocks * inv_scales).clamp(-448, 448).to(torch.float8_e4m3fn)
    dequant = (values.float() * scales).flatten(-2)
    return dequant if x.dtype == torch.float32 else dequant.to(x.dtype)


def rms_norm(x, weight, eps):
    x_f32 = x if x.dtype == torch.float32 else x.float()
    w_f32 = weight if weight.dtype == torch.float32 else weight.float()
    out = x_f32 * torch.rsqrt(x_f32.square().mean(-1, keepdim=True) + eps) * w_f32
    return out if x.dtype == torch.float32 else out.to(x.dtype)
