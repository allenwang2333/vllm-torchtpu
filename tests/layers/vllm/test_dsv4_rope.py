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

import numpy as np
import torch
import jax
import jax.numpy as jnp
from vllm.model_executor.layers.rotary_embedding import DeepseekV4ScalingRotaryEmbedding

# --- JAX Reference implementation of RoPE ---
def jax_yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * np.log(max_position_embeddings / (num_rotations * 2 * np.pi))) / (2 * np.log(base))

def jax_yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = np.floor(jax_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = np.ceil(jax_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return int(max(low, 0)), int(min(high, dim - 1))

def jax_yarn_linear_ramp_mask(low, high, dim):
    if low == high:
        high += 0.001
    linear_func = (jnp.arange(dim, dtype=jnp.float32) - low) / (high - low)
    return jnp.clip(linear_func, 0.0, 1.0)

def jax_yarn_get_mscale(scale, mscale):
    return jnp.where(scale <= 1, 1.0, 0.1 * mscale * jnp.log(scale) + 1.0)

class JaxDeepseekScalingRotaryEmbedding:
    def __init__(self, head_size, rotary_dim, max_position_embeddings, base, scaling_factor, dtype=jnp.bfloat16):
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.dtype = dtype
        self.beta_fast = 32
        self.beta_slow = 1
        self.mscale_value = 1.0
        self.mscale_all_dim = 0.0
        
        self.mscale = jax_yarn_get_mscale(self.scaling_factor, self.mscale_value) / jax_yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        self.sin_cos_cache = self._compute_sin_cos()
        
    def _compute_inv_freq(self):
        fractions = jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim
        inv_freq_extrapolation = 1.0 / (self.base**fractions)
        inv_freq_interpolation = 1.0 / (self.scaling_factor * self.base**fractions)
        
        low, high = jax_yarn_find_correction_range(
            self.beta_fast, self.beta_slow, self.rotary_dim, self.base,
            self.max_position_embeddings)
            
        inv_freq_mask = 1 - jax_yarn_linear_ramp_mask(low, high, self.rotary_dim // 2).astype(jnp.float32)
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq
        
    def _compute_sin_cos(self):
        inv_freq_H = self._compute_inv_freq()
        t = jnp.arange(self.max_position_embeddings * self.scaling_factor, dtype=jnp.float32)
        freqs = jnp.einsum("T,k->Tk", t, inv_freq_H)
        sin, cos = jnp.sin(freqs) * self.mscale, jnp.cos(freqs) * self.mscale
        cache = jnp.concatenate((cos, sin), axis=-1)
        
        # pad cache to 128 boundary
        H = cache.shape[1]
        target_dim = ((H - 1) // 128 + 1) * 128
        padding_amount = target_dim - self.rotary_dim
        pad_width = ((0, 0), (0, padding_amount))
        cache_padded = jnp.pad(cache, pad_width, mode='constant')
        return cache_padded

    def apply_rope(self, positions, x_TNH, inverse=False):
        assert x_TNH.ndim == 3
        cos_sin_padded = self.sin_cos_cache[positions]
        cos_sin_TH = cos_sin_padded[:, :self.rotary_dim]
        cos_TH, sin_TH = jnp.split(cos_sin_TH, 2, axis=-1)
        if inverse:
            sin_TH = -sin_TH
        cos_T1H, sin_T1H = cos_TH[:, None, :], sin_TH[:, None, :]
        even_TNH, odd_TNH = x_TNH[..., ::2], x_TNH[..., 1::2]
        combined_TNH = jnp.stack([
            even_TNH * cos_T1H - odd_TNH * sin_T1H,
            odd_TNH * cos_T1H + even_TNH * sin_T1H
        ], axis=-1).reshape(x_TNH.shape)
        return combined_TNH.astype(self.dtype)


def test_rope_parity():
    from unittest.mock import MagicMock
    from vllm.config.vllm import CompilationConfig
    import vllm.config.vllm as vllm_config_mod
    from vllm.platforms import current_platform

    comp_config = CompilationConfig()
    comp_config.custom_ops = ["all"]

    mock_vllm_config = MagicMock()
    mock_vllm_config.compilation_config = comp_config
    vllm_config_mod.get_current_vllm_config = MagicMock(return_value=mock_vllm_config)

    # Force current_platform to be CPU in this test to avoid tensor device mismatches during cache creation
    current_platform.device_type = "cpu"

    # Model config
    head_size = 512
    rotary_dim = 64
    original_max_position_embeddings = 65536
    base = 10000.0
    scaling_factor = 16
    
    # Generate test inputs
    num_tokens = 8
    num_heads = 8
    
    # query: [num_tokens, num_heads, head_size]
    # We will generate a query tensor with values that are easy to track
    query_np = np.random.normal(size=(num_tokens, num_heads, head_size)).astype(np.float32)
    positions_np = np.array([0, 1, 2, 3, 10, 20, 50, 100], dtype=np.int32)
    
    # --- Run JAX Reference ---
    # In DeepSeek MLA, RoPE is applied to the last 64 dimensions of the head:
    # JAX model passes q_rope_TNH which has shape [num_tokens, num_heads, rotary_dim] to apply_rope
    q_rope_jax = query_np[..., -rotary_dim:]
    jax_rope_module = JaxDeepseekScalingRotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=original_max_position_embeddings,
        base=base,
        scaling_factor=scaling_factor,
        dtype=jnp.float32,
    )
    
    out_rope_jax = jax_rope_module.apply_rope(positions_np, q_rope_jax, inverse=False)
    # The JAX output concatenates nope and rope:
    q_nope_jax = query_np[..., :-rotary_dim]
    out_jax = np.concatenate([q_nope_jax, out_rope_jax], axis=-1)
    
    # Now run JAX inverse RoPE (to simulate _o_proj)
    # We take the rotated output, and apply inverse RoPE to the last rotary_dim channels
    out_o_proj_rope_jax = out_jax[..., -rotary_dim:]
    inv_out_rope_jax = jax_rope_module.apply_rope(positions_np, out_o_proj_rope_jax, inverse=True)
    inv_out_jax = np.concatenate([q_nope_jax, inv_out_rope_jax], axis=-1)

    # --- Run PyTorch/vLLM Reference ---
    torch_device = "cpu"
    query_torch = torch.from_numpy(query_np).to(torch_device)
    positions_torch = torch.from_numpy(positions_np).to(torch_device)
    
    # Instantiate the vLLM rotary embedding
    py_rope_module = DeepseekV4ScalingRotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=original_max_position_embeddings,
        base=base,
        is_neox_style=False, # DeepSeek is GPT-J style / interleaved
        scaling_factor=scaling_factor,
        dtype=torch.float32,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
        mscale_all_dim=0.0,
    )
    
    # In PyTorch, DeepseekV4ScalingRotaryEmbedding.forward takes (positions, query, key) and applies RoPE.
    # Note: DeepSeekV4 RoPE applies to the last rotary_dim channels in-place, or returning new tensors.
    # Let's call forward_native:
    out_q_torch, _ = py_rope_module.forward_native(
        positions=positions_torch,
        query=query_torch.clone(),
        key=None,
        inverse=False
    )
    out_py = out_q_torch.numpy()
    
    # And run PyTorch inverse RoPE
    inv_out_q_torch, _ = py_rope_module.forward_native(
        positions=positions_torch,
        query=out_q_torch.clone(),
        key=None,
        inverse=True
    )
    inv_out_py = inv_out_q_torch.numpy()
    
    # Compare caches
    # Compare mscale
    print(f"JAX mscale: {jax_rope_module.mscale}")
    print(f"PyTorch mscale: {py_rope_module.mscale}")

    # Compare caches
    print("\n--- Cache comparison ---")
    jax_cache = np.array(jax_rope_module.sin_cos_cache)[:, :rotary_dim]
    py_cache = py_rope_module.cos_sin_cache.numpy()
    cache_diff = np.abs(jax_cache - py_cache)
    print(f"Max cache diff: {np.max(cache_diff)}")
    
    # Compare forward outputs
    print("\n--- Forward output comparison ---")
    fwd_diff = np.abs(out_jax - out_py)
    print(f"Max forward output diff: {np.max(fwd_diff)}")
    
    # Compare inverse outputs
    print("\n--- Inverse output comparison ---")
    inv_diff = np.abs(inv_out_jax - inv_out_py)
    print(f"Max inverse output diff: {np.max(inv_diff)}")
    
    # Compare double-rotated output against input
    print("\n--- Double-rotated output (inv(fwd(x))) vs input comparison ---")
    recon_diff_jax = np.abs(query_np - inv_out_jax)
    recon_diff_py = np.abs(query_np - inv_out_py)
    print(f"Max JAX reconstruction diff: {np.max(recon_diff_jax)}")
    print(f"Max PyTorch reconstruction diff: {np.max(recon_diff_py)}")
    
    assert np.max(fwd_diff) < 1e-4, "Forward outputs differ!"
    assert np.max(inv_diff) < 1e-4, "Inverse outputs differ!"
    print("\nParity test PASSED!")

if __name__ == "__main__":
    test_rope_parity()
