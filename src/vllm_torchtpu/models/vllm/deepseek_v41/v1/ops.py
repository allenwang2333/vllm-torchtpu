"""RoPE cache tables and tensor rotary operations."""

import math

import torch


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
    cache_key = (dim, base, device, yarn_key, torch.arange)
    cached = _rotary_freq_cache.get(cache_key)
    if cached is not None:
        return cached
    frequencies = 1 / base ** (
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
    )
    if yarn_key is not None:
        original, beta_fast, beta_slow, factor = yarn_key
        inv_two_log_base = dim / (2 * math.log(base))
        two_pi = 2 * math.pi

        def correction(rotations):
            return inv_two_log_base * math.log(original / (rotations * two_pi))

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
    x_rope = x if dim == x.shape[-1] else x[..., -dim:]
    x_tail = x_rope if x.dtype == torch.float32 else x_rope.float()
    pairs = x_tail.unflatten(-1, (-1, 2))
    real, imag = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack(
        (real * cosine - imag * sine, real * sine + imag * cosine), -1
    ).flatten(-2)
    rotated_out = rotated if x.dtype == torch.float32 else rotated.to(x.dtype)
    if dim == x.shape[-1]:
        return rotated_out
    return torch.cat((x[..., :-dim], rotated_out), -1)
