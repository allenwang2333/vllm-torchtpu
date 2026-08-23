# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

# Architectures whose recurrent layers read and write their state through the
# attention-shaped pool. These build on QwenGatedDeltaNetAttention, which the
# TPU backend replaces with VllmGatedDeltaNetAttention; that subclass is what
# accepts a single pooled buffer instead of a separate conv/ssm pair. Other
# hybrid architectures unpack two state tensors unconditionally, so they stay
# on the per-layer KV caches until they grow a pooled path. Extend this set in
# the same change that adds one.
POOLED_GDN_ARCHITECTURES = frozenset({
    "Qwen3NextForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
})

# The pooled GDN kernel's state dtypes inside the unified pool are controlled
# here: conv rows are always bf16 and the SSM region defaults to fp32. The
# env-gated experiment stores SSM checkpoints and live recurrence/carry as
# bf16 while retaining fp32 MXU accumulation. These physical dtypes are
# independent of the layer's declared MambaSpec dtypes.
# The default ABI constants below remain fp32 for page sizing and Raiden.
# The bf16 selector is consumed only by the single-server mixed-mode kernel
# plan, which uses a compact prefix of that unchanged allocation.
#
# These are torch dtype *names*, not torch.dtype objects: this module is
# imported by the JAX kernel path and by deviceless tests, so it must stay
# importable without torch.
POOLED_GDN_CONV_STATE_DTYPE = "torch.bfloat16"
POOLED_GDN_SSM_STATE_DTYPE = "torch.float32"
POOLED_GDN_SSM_BF16_STATE_DTYPE = "torch.bfloat16"

# Derived from the dtypes above rather than declared separately, so changing
# a state dtype cannot leave a stale byte width behind.
_ITEMSIZE_BY_DTYPE = {"torch.bfloat16": 2, "torch.float32": 4}
POOLED_GDN_CONV_STATE_ITEMSIZE = _ITEMSIZE_BY_DTYPE[
    POOLED_GDN_CONV_STATE_DTYPE]
POOLED_GDN_SSM_STATE_ITEMSIZE = _ITEMSIZE_BY_DTYPE[POOLED_GDN_SSM_STATE_DTYPE]


def pooled_gdn_ssm_state_dtype(*, use_bf16: bool = False) -> str:
    """Physical pooled SSM checkpoint dtype for the selected ABI."""
    return (POOLED_GDN_SSM_BF16_STATE_DTYPE
            if use_bf16 else POOLED_GDN_SSM_STATE_DTYPE)


def pooled_gdn_ssm_state_itemsize(*, use_bf16: bool = False) -> int:
    """Bytes per physical pooled SSM checkpoint element."""
    return _ITEMSIZE_BY_DTYPE[
        pooled_gdn_ssm_state_dtype(use_bf16=use_bf16)]


def pooled_gdn_conv_state_bytes(*, kernel_size: int, conv_dim: int) -> int:
    """Live bytes of one slot's conv region: (kernel_size - 1) dense bf16
    rows of conv_dim channels. The kernel keeps no spec-decode widening in
    the pool (verify windows roll back via per-slot checkpoints instead)."""
    return (kernel_size - 1) * conv_dim * POOLED_GDN_CONV_STATE_ITEMSIZE


def pooled_gdn_ssm_state_bytes(*,
                               num_v_heads: int,
                               head_k_dim: int,
                               head_v_dim: int,
                               use_bf16: bool = False) -> int:
    """Live bytes of one slot's dense, head-major SSM checkpoint."""
    return (num_v_heads * head_k_dim * head_v_dim *
            pooled_gdn_ssm_state_itemsize(use_bf16=use_bf16))


@dataclass(frozen=True)
class PooledGDNStateLayout:
    ssm_bytes: int
    conv_bytes: int
    token_bytes: int
    ssm_tokens: int
    conv_tokens: int

    @property
    def required_tokens(self) -> int:
        return self.ssm_tokens + self.conv_tokens

    @property
    def required_bytes(self) -> int:
        return self.required_tokens * self.token_bytes


def derive_pooled_gdn_state_layout(
    *,
    ssm_bytes: int,
    conv_bytes: int,
    token_bytes: int,
) -> PooledGDNStateLayout:
    """Derive the physical token extent of pooled GDN state regions."""
    if min(ssm_bytes, conv_bytes, token_bytes) <= 0:
        raise ValueError("pooled GDN byte sizes must be positive")
    if ssm_bytes % token_bytes:
        raise ValueError(
            "pooled GDN SSM state must occupy complete FA token rows")

    ssm_tokens = ssm_bytes // token_bytes
    conv_tokens = 1
    while (conv_tokens * token_bytes < conv_bytes
           or ssm_tokens % conv_tokens != 0):
        conv_tokens *= 2
        if conv_tokens > ssm_tokens:
            raise ValueError(
                "pooled GDN conv tile cannot be placed after the SSM region")

    return PooledGDNStateLayout(
        ssm_bytes=ssm_bytes,
        conv_bytes=conv_bytes,
        token_bytes=token_bytes,
        ssm_tokens=ssm_tokens,
        conv_tokens=conv_tokens,
    )
