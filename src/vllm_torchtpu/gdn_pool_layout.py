# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

# Architectures whose recurrent layers read and write their state through the
# attention-shaped pool. The Qwen3 families build on QwenGatedDeltaNetAttention,
# which the TPU backend replaces with VllmGatedDeltaNetAttention; that subclass
# accepts a single pooled buffer instead of a separate conv/ssm pair.
# KimiLinearForCausalLM runs its KDA layers through the gather/scatter pooled
# op (build_kimi_pooled_kda_op) instead. Other hybrid architectures unpack two
# state tensors unconditionally, so they stay on the per-layer KV caches until
# they grow a pooled path. Extend this set in the same change that adds one.
POOLED_GDN_ARCHITECTURES = frozenset({
    "Qwen3NextForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    # Kimi-Linear's KDA layers run the gather/scatter pooled op
    # (build_kimi_pooled_kda_op); Kimi-K3 stays opt-in until its TP32
    # geometry is validated.
    "KimiLinearForCausalLM",
})


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
    """Derive the physical token extent of pooled GDN state regions.

    Regions span whole FA token rows; a state whose byte size does not
    divide the token row leaves the tail of its last row as padding (the
    kernel's copy-plan zero-fills it via ``StateRegion.rows_used``). The
    conv tile stays a power of two so layouts derived before padding was
    allowed keep their exact token extents.
    """
    if min(ssm_bytes, conv_bytes, token_bytes) <= 0:
        raise ValueError("pooled GDN byte sizes must be positive")

    ssm_tokens = (ssm_bytes + token_bytes - 1) // token_bytes
    conv_tokens = 1
    while conv_tokens * token_bytes < conv_bytes:
        conv_tokens *= 2

    return PooledGDNStateLayout(
        ssm_bytes=ssm_bytes,
        conv_bytes=conv_bytes,
        token_bytes=token_bytes,
        ssm_tokens=ssm_tokens,
        conv_tokens=conv_tokens,
    )
