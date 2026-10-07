# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

# Architectures whose recurrent layers read and write their state through the
# attention-shaped pool. The Qwen3 families build on QwenGatedDeltaNetAttention,
# which the TPU backend replaces with VllmGatedDeltaNetAttention; that subclass
# accepts a single pooled buffer instead of a separate conv/ssm pair.
# KimiLinearForCausalLM runs its KDA layers through the gather/scatter pooled
# op (build_kimi_pooled_kda_op, around the fused conv1d + GDN v3 kernel)
# instead. Other hybrid architectures unpack two state tensors unconditionally,
# so they stay on the per-layer KV caches until they grow a pooled path.
# Extend this set in the same change that adds one.
QWEN_GDN_ARCHITECTURES = frozenset(
    {
        "Qwen3NextForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    }
)
POOLED_GDN_ARCHITECTURES = QWEN_GDN_ARCHITECTURES | frozenset(
    {
        # Kimi-Linear and Kimi-K3 run their KDA layers through the gather/scatter
        # pooled op (build_kimi_pooled_kda_op, around the fused conv1d + GDN v3
        # kernel). Kimi-K3's TP32 geometry is validated: prefix-cache hits are
        # byte-identical to misses and GSM8K scores 100/100 with and without the
        # pool.
        "KimiLinearForCausalLM",
        "KimiK3ForConditionalGeneration",
    }
)

# The pooled GDN kernel's state dtypes inside the unified pool are fixed for
# Conv, while SSM follows the layer's declared MambaSpec dtype. Everything
# that describes pool bytes — the kernel's state plan and the Raiden reshard
# manifest — must take its dtypes and sizes from here, so no two consumers can
# disagree about the physical layout.
#
# These are torch dtype *names*, not torch.dtype objects: this module is
# imported by the JAX kernel path and by deviceless tests, so it must stay
# importable without torch.
DEFAULT_POOLED_GDN_CONV_STATE_DTYPE = "torch.bfloat16"
DEFAULT_POOLED_GDN_SSM_STATE_DTYPE = "torch.bfloat16"
_ITEMSIZE_BY_DTYPE = {
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float8_e4m3fn": 1,
}

# When the SSM state is quantized to FP8 (float8_e4m3fn) or FP4 (float4_e2m1fn),
# a per-(v_head, v_col) FP32 scale sits at the head of the conv region block.
# Because a bfloat16 VREG on TPU v7 spans 16 rows (8,192 bytes at 256 lanes),
# the conv region starts at VREG-aligned byte offset 8,192 and its touched rows
# are also rounded up to whole bfloat16 VREGs (8,192-byte grain).
POOLED_GDN_SCALE_HEADER_BYTES = 8192
POOLED_GDN_CONV_VREG_GRAIN_BYTES = 8192


def pooled_gdn_state_dtypes(
    dtypes: tuple[object, ...],
) -> tuple[object, object]:
    """Resolve declared Conv/SSM dtypes to their pooled physical dtypes."""
    ssm_dtype = dtypes[1]
    if str(ssm_dtype) in ("torch.float32", "float32", "auto"):
        ssm_dtype = DEFAULT_POOLED_GDN_SSM_STATE_DTYPE
    return DEFAULT_POOLED_GDN_CONV_STATE_DTYPE, ssm_dtype


def pooled_gdn_state_itemsize(dtype: object) -> int:
    name = str(dtype)
    if not name.startswith("torch."):
        name = f"torch.{name}"
    try:
        return _ITEMSIZE_BY_DTYPE[name]
    except KeyError as exc:
        raise ValueError(f"unsupported pooled GDN state dtype: {dtype}") from exc


def unified_kv_layout_enabled_for_architecture(architecture: str | None) -> bool:
    """Whether `architecture` runs on the attention-shaped unified KV pool.

    Split out of `platforms.tpu_block_size_utils.unified_kv_layout_enabled`
    (which delegates here) so the GDN layer can ask the same question without
    importing `vllm_torchtpu.platforms` -- that package's __init__ pulls in
    TpuPlatform, which imports the layers back.
    """
    from vllm_torchtpu import envs as tpu_envs

    override = tpu_envs.TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL
    if override is not None:
        return override
    return architecture in POOLED_GDN_ARCHITECTURES


def pooled_gdn_conv_state_bytes(*, kernel_size: int, conv_dim: int) -> int:
    """Live bytes of one slot's conv region: (kernel_size - 1) dense bf16
    rows of conv_dim channels. The kernel keeps no spec-decode widening in
    the pool (verify windows roll back via per-slot checkpoints instead)."""
    return (
        (kernel_size - 1)
        * conv_dim
        * pooled_gdn_state_itemsize(DEFAULT_POOLED_GDN_CONV_STATE_DTYPE)
    )


def pooled_gdn_conv_and_scale_state_bytes(
    *,
    kernel_size: int,
    conv_dim: int,
    ssm_dtype: object = DEFAULT_POOLED_GDN_SSM_STATE_DTYPE,
    raw_conv_bytes: int | None = None,
) -> int:
    """Bytes required in the conv block, including any co-located FP32 scale."""
    base_bytes = (
        raw_conv_bytes
        if raw_conv_bytes is not None
        else pooled_gdn_conv_state_bytes(kernel_size=kernel_size, conv_dim=conv_dim)
    )
    # vmem_ldst loads and stores conv state at whole bfloat16 VREG granularity
    # (16 rows = 8,192 bytes at 256 lanes) regardless of ssm_dtype, so always
    # round conv state bytes up to POOLED_GDN_CONV_VREG_GRAIN_BYTES.
    aligned_conv = (
        (base_bytes + POOLED_GDN_CONV_VREG_GRAIN_BYTES - 1)
        // POOLED_GDN_CONV_VREG_GRAIN_BYTES
    ) * POOLED_GDN_CONV_VREG_GRAIN_BYTES
    name = str(ssm_dtype)
    if "float8_e4m3fn" in name or "float4_e2m1fn" in name:
        return POOLED_GDN_SCALE_HEADER_BYTES + aligned_conv
    return aligned_conv


def pooled_gdn_ssm_state_bytes(
    *,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: object = DEFAULT_POOLED_GDN_SSM_STATE_DTYPE,
) -> int:
    """Live bytes of one slot's head-major, dense SSM region in ``dtype``."""
    name = str(dtype)
    if "float4_e2m1fn" in name:
        return (num_v_heads * head_k_dim * head_v_dim) // 2
    return num_v_heads * head_k_dim * head_v_dim * pooled_gdn_state_itemsize(dtype)


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

    If ssm_bytes cannot be evenly divided into FA token rows, its last row
    is padded.
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
