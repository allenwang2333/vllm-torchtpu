# SPDX-License-Identifier: Apache-2.0

import os
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


def _pad_ssm_to_token_rows() -> bool:
    return bool(int(os.getenv("TPU_GDN_PAD_SSM_TO_TOKEN_ROWS") or "0"))


def derive_pooled_gdn_state_layout(
    *,
    ssm_bytes: int,
    conv_bytes: int,
    token_bytes: int,
) -> PooledGDNStateLayout:
    """Derive the physical token extent of pooled GDN state regions."""
    if min(ssm_bytes, conv_bytes, token_bytes) <= 0:
        raise ValueError("pooled GDN byte sizes must be positive")
    # The SSM region is placed as a whole number of attention token rows, and
    # for Kimi-K3 the two sizes are not commensurate: ssm_bytes is
    # n_v * d_k * d_v * 4 (a power of two times n_v = 96/TP) while an MLA token
    # row is 576 elements = 2^6 * 3^2, so the row carries a factor of 9 that
    # n_v never supplies. Reporting the numbers matters more than the rule --
    # without them the failure is unactionable, and whether any dtype or block
    # size can satisfy it is pure arithmetic once they are visible.
    if ssm_bytes % token_bytes:
        if not _pad_ssm_to_token_rows():
            raise ValueError(
                "pooled GDN SSM state must occupy complete FA token rows: "
                f"ssm_bytes={ssm_bytes} is not a multiple of "
                f"token_bytes={token_bytes} (remainder "
                f"{ssm_bytes % token_bytes}, conv_bytes={conv_bytes}). "
                "Set TPU_GDN_PAD_SSM_TO_TOKEN_ROWS=1 to round the SSM region "
                "up to the next whole row instead.")
        # Round up. The region only has to RESERVE space -- v3_state_source
        # uses ssm_tokens as the offset at which the conv tile starts, so a
        # larger reservation moves conv later and wastes under one row per
        # sequence. It does not change how much state is read or written.
        #
        # Rounding to the next row alone is not enough: the conv tile below
        # must also divide ssm_tokens, and the next row up is frequently prime
        # (683 in the first case measured), which leaves the doubling loop with
        # no legal tile. So round up to a multiple of the conv tile instead,
        # which satisfies both constraints at once and costs at most one conv
        # tile of extra reservation.
        conv_tile = 1
        while conv_tile * token_bytes < conv_bytes:
            conv_tile *= 2
        ssm_tokens = -(-ssm_bytes // token_bytes)
        ssm_tokens = -(-ssm_tokens // conv_tile) * conv_tile
    else:
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
