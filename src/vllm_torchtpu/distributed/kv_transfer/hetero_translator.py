# SPDX-License-Identifier: Apache-2.0
"""Generic Cross-Hardware Heterogeneous Translator for Disaggregated TPU Serving.

Translates tensor sharding, memory layout, quantization datatypes, and Raiden
channel metadata across heterogeneous hardware generations (e.g. TPU v6e Trillium
prefill -> TPU v7x Ironwood decode).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Literal, Sequence

import torch

from vllm_torchtpu.distributed.kv_transfer.hetero_config import HeteroTierConfig

DTypeStr = Literal["bfloat16", "float16", "float8_e4m3fn", "float8_e5m2", "int8"]
LayoutStr = Literal["canonical", "pallas_tiled", "contiguous"]


@dataclasses.dataclass(frozen=True)
class ShardChannelMapping:
    """Exact channel mapping between producer rank and consumer rank."""

    producer_rank: int
    consumer_rank: int
    head_indices: tuple[int, ...]
    consumer_head_offset: int


@dataclasses.dataclass(frozen=True)
class HeteroTransferPlan:
    """Deterministic transfer specification across heterogeneous tiers."""

    request_id: str
    producer_tp: int
    consumer_tp: int
    num_tokens: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    bytes_per_token_per_layer: int
    total_transfer_bytes: int
    channel_mappings: list[ShardChannelMapping]
    communication_pattern: Literal["point_to_point", "gather", "scatter", "all_to_all"]

    def estimated_transfer_latency_ms(self, dcn_bandwidth_gbps: float = 100.0) -> float:
        """Estimate DCN wire transfer latency in milliseconds."""
        bytes_per_sec = (dcn_bandwidth_gbps * 1e9) / 8.0
        return (self.total_transfer_bytes / bytes_per_sec) * 1000.0


class ShardingTranslator:
    """Translates KV cache tensor parallelism, head slicing, and mesh topology."""

    def __init__(
        self,
        *,
        producer_tp: int,
        consumer_tp: int,
        num_kv_heads: int,
        producer_cores_per_chip: int = 1,
        consumer_cores_per_chip: int = 2,
    ) -> None:
        if producer_tp <= 0 or consumer_tp <= 0:
            raise ValueError(f"TP sizes must be positive, got producer={producer_tp}, consumer={consumer_tp}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        if num_kv_heads % min(producer_tp, consumer_tp) != 0:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be divisible by min(producer_tp, consumer_tp) "
                f"({min(producer_tp, consumer_tp)})"
            )

        self.producer_tp = producer_tp
        self.consumer_tp = consumer_tp
        self.num_kv_heads = num_kv_heads
        self.producer_cores_per_chip = producer_cores_per_chip
        self.consumer_cores_per_chip = consumer_cores_per_chip

        self.heads_per_producer_rank = max(1, num_kv_heads // producer_tp)
        self.heads_per_consumer_rank = max(1, num_kv_heads // consumer_tp)

    @property
    def communication_pattern(self) -> Literal["point_to_point", "gather", "scatter", "all_to_all"]:
        if self.producer_tp == self.consumer_tp:
            return "point_to_point"
        elif self.producer_tp > self.consumer_tp:
            return "gather"
        elif self.producer_tp < self.consumer_tp:
            return "scatter"
        return "all_to_all"

    def get_producer_head_slice(self, producer_rank: int) -> tuple[int, int]:
        """Return (start_head, end_head) owned by the producer rank."""
        if not (0 <= producer_rank < self.producer_tp):
            raise IndexError(f"producer_rank {producer_rank} out of bounds [0, {self.producer_tp})")
        start = (producer_rank * self.num_kv_heads) // self.producer_tp
        end = ((producer_rank + 1) * self.num_kv_heads) // self.producer_tp
        return (start, end)

    def get_consumer_head_slice(self, consumer_rank: int) -> tuple[int, int]:
        """Return (start_head, end_head) expected by the consumer rank."""
        if not (0 <= consumer_rank < self.consumer_tp):
            raise IndexError(f"consumer_rank {consumer_rank} out of bounds [0, {self.consumer_tp})")
        start = (consumer_rank * self.num_kv_heads) // self.consumer_tp
        end = ((consumer_rank + 1) * self.num_kv_heads) // self.consumer_tp
        return (start, end)

    def build_channel_mappings(self) -> list[ShardChannelMapping]:
        """Compute the full set of producer->consumer shard channel mappings."""
        mappings: list[ShardChannelMapping] = []

        for p_rank in range(self.producer_tp):
            p_start, p_end = self.get_producer_head_slice(p_rank)
            p_heads = set(range(p_start, p_end))

            for c_rank in range(self.consumer_tp):
                c_start, c_end = self.get_consumer_head_slice(c_rank)
                c_heads = set(range(c_start, c_end))

                overlap = sorted(p_heads.intersection(c_heads))
                if overlap:
                    offset = overlap[0] - c_start
                    mappings.append(
                        ShardChannelMapping(
                            producer_rank=p_rank,
                            consumer_rank=c_rank,
                            head_indices=tuple(overlap),
                            consumer_head_offset=offset,
                        )
                    )

        return mappings


class MemoryLayoutTranslator:
    """Translates physical block sizes, dimension strides, and hardware alignment."""

    def __init__(
        self,
        *,
        producer_block_size: int = 16,
        consumer_block_size: int = 16,
        head_dim: int = 128,
        producer_layout: LayoutStr = "canonical",
        consumer_layout: LayoutStr = "canonical",
        producer_alignment_bytes: int = 128,
        consumer_alignment_bytes: int = 256,
    ) -> None:
        self.producer_block_size = producer_block_size
        self.consumer_block_size = consumer_block_size
        self.head_dim = head_dim
        self.producer_layout = producer_layout
        self.consumer_layout = consumer_layout
        self.producer_alignment_bytes = producer_alignment_bytes
        self.consumer_alignment_bytes = consumer_alignment_bytes

    def calculate_num_blocks(self, num_tokens: int, block_size: int) -> int:
        """Calculate number of allocated blocks given token length."""
        return math.ceil(num_tokens / block_size)

    def get_strides(
        self,
        *,
        layout: LayoutStr,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype_bytes: int,
    ) -> tuple[int, int, int, int]:
        """Compute tensor byte strides for the specified layout.

        canonical: [num_blocks, block_size, num_heads, head_dim]
        pallas_tiled: [num_blocks, num_heads, block_size, head_dim]
        """
        if layout == "canonical":
            s3 = dtype_bytes
            s2 = head_dim * s3
            s1 = num_heads * s2
            s0 = block_size * s1
            return (s0, s1, s2, s3)
        elif layout == "pallas_tiled":
            s3 = dtype_bytes
            s2 = head_dim * s3
            s1 = block_size * s2
            s0 = num_heads * s1
            return (s0, s1, s2, s3)
        else:
            # contiguous flat buffer
            total = num_blocks * block_size * num_heads * head_dim
            return (total * dtype_bytes, 0, 0, 0)

    def align_buffer_bytes(self, raw_bytes: int, is_consumer: bool = False) -> int:
        """Align byte size to the hardware burst boundary (128B v6e, 256B v7x)."""
        align = self.consumer_alignment_bytes if is_consumer else self.producer_alignment_bytes
        return math.ceil(raw_bytes / align) * align


class DTypeTranslator:
    """Translates quantization dtypes, precision scalers, and element sizes."""

    DTYPE_SIZES: dict[str, int] = {
        "bfloat16": 2,
        "float16": 2,
        "float32": 4,
        "fp8": 1,
        "fp8_e4m3": 1,
        "float8_e4m3fn": 1,
        "float8_e5m2": 1,
        "int8": 1,
        "uint8": 1,
    }

    TORCH_DTYPES: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "int8": torch.int8,
        "uint8": torch.uint8,
    }

    def __init__(
        self,
        *,
        producer_dtype: str = "float8_e4m3fn",
        consumer_dtype: str = "float8_e4m3fn",
    ) -> None:
        self.producer_dtype = self._normalize_dtype(producer_dtype)
        self.consumer_dtype = self._normalize_dtype(consumer_dtype)

    @staticmethod
    def _normalize_dtype(dtype: str | torch.dtype) -> str:
        if isinstance(dtype, torch.dtype):
            d = str(dtype).replace("torch.", "").strip().lower()
        else:
            d = dtype.strip().lower()
            if d.startswith("torch."):
                d = d[len("torch.") :]
        if d in ("fp8", "fp8_e4m3", "float8_e4m3fn"):
            return "float8_e4m3fn"
        if d in ("fp8_e5m2", "float8_e5m2"):
            return "float8_e5m2"
        if d in ("bf16", "bfloat16"):
            return "bfloat16"
        if d in ("fp16", "half", "float16"):
            return "float16"
        if d in ("fp32", "float", "float32"):
            return "float32"
        return d

    @classmethod
    def resolve_torch_dtype(cls, dtype: str | torch.dtype) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        norm = cls._normalize_dtype(dtype)
        if norm in cls.TORCH_DTYPES:
            return cls.TORCH_DTYPES[norm]
        maybe_dtype = getattr(torch, norm, None)
        if isinstance(maybe_dtype, torch.dtype):
            return maybe_dtype
        raise ValueError(f"Unsupported KV cache dtype for translation: {dtype!r}")

    @classmethod
    def translate_pallas_kv_buffer(
        cls,
        src_buf: memoryview | bytes | bytearray,
        dst_buf: Any,
        *,
        dst_offset: int = 0,
        num_blocks: int,
        src_dtype: str | torch.dtype,
        dst_dtype: str | torch.dtype,
        src_block_shape: Sequence[int],
        dst_block_shape: Sequence[int],
    ) -> None:
        """Translate a Pallas 5D KV cache shard between dtypes/packing layouts in host memory.

        In TPU Pallas 5D KV cache layout ``(num_blocks, page_size, k2_per_pack, kv_packing, head_dim)``,
        the ``kv_packing`` axis is sublane-packed into each 32-bit word along ``head_dim``.
        After ``batch_transfer_d2h``, the physical C-contiguous memory layout in host shared memory
        is ``(num_blocks, page_size, k2_per_pack, head_dim, kv_packing)``.
        """
        s_dtype = cls.resolve_torch_dtype(src_dtype)
        d_dtype = cls.resolve_torch_dtype(dst_dtype)
        src_writable = (
            bytearray(src_buf)
            if isinstance(src_buf, bytes)
            or (isinstance(src_buf, memoryview) and src_buf.readonly)
            else src_buf
        )
        if len(src_block_shape) != 4 or len(dst_block_shape) != 4:
            s_count = num_blocks * math.prod(src_block_shape)
            d_count = num_blocks * math.prod(dst_block_shape)
            if s_count != d_count:
                raise ValueError(
                    f"Cannot translate non-4D block shapes with mismatched element counts: "
                    f"src={tuple(src_block_shape)} vs dst={tuple(dst_block_shape)}"
                )
            src_flat = torch.frombuffer(src_writable, dtype=s_dtype, count=s_count)
            dst_flat = torch.frombuffer(
                dst_buf, dtype=d_dtype, count=d_count, offset=dst_offset
            )
            dst_flat.copy_(src_flat.to(d_dtype))
            return

        s_page, s_k2, s_pack, s_hdim = (int(x) for x in src_block_shape)
        d_page, d_k2, d_pack, d_hdim = (int(x) for x in dst_block_shape)
        if s_page != d_page or s_hdim != d_hdim:
            raise ValueError(
                f"Incompatible Pallas KV block geometry for dtype translation: "
                f"src={tuple(src_block_shape)} vs dst={tuple(dst_block_shape)}"
            )

        s_count = num_blocks * s_page * s_k2 * s_pack * s_hdim
        d_count = num_blocks * d_page * d_k2 * d_pack * d_hdim
        src_t = torch.frombuffer(
            src_writable, dtype=s_dtype, count=s_count
        ).view(num_blocks, s_page, s_k2, s_hdim, s_pack)
        dst_t = torch.frombuffer(
            dst_buf, dtype=d_dtype, count=d_count, offset=dst_offset
        ).view(num_blocks, d_page, d_k2, d_hdim, d_pack)

        if s_k2 == d_k2:
            if s_pack >= d_pack:
                dst_t.copy_(src_t[..., :d_pack].to(d_dtype))
            else:
                dst_t[..., :s_pack].copy_(src_t.to(d_dtype))
                dst_t[..., s_pack:].zero_()
        else:
            unpacked = src_t.permute(0, 1, 2, 4, 3).reshape(
                num_blocks, s_page, s_k2 * s_pack, s_hdim
            )
            valid_k2 = min(s_k2 * s_pack, d_k2 * d_pack)
            converted = unpacked[:, :, :valid_k2, :].to(d_dtype)
            if d_k2 * d_pack > valid_k2:
                padded = torch.zeros(
                    (num_blocks, d_page, d_k2 * d_pack, d_hdim),
                    dtype=d_dtype,
                )
                padded[:, :, :valid_k2, :].copy_(converted)
                converted = padded
            dst_t.copy_(
                converted.reshape(num_blocks, d_page, d_k2, d_pack, d_hdim).permute(
                    0, 1, 2, 4, 3
                )
            )

    def get_element_size(self, dtype: str) -> int:
        norm = self._normalize_dtype(dtype)
        return self.DTYPE_SIZES.get(norm, 2)

    @property
    def producer_bytes(self) -> int:
        return self.get_element_size(self.producer_dtype)

    @property
    def consumer_bytes(self) -> int:
        return self.get_element_size(self.consumer_dtype)

    @property
    def is_lossless(self) -> bool:
        return self.producer_dtype == self.consumer_dtype


class TransportMetadataTranslator:
    """Translates network descriptors and Raiden channel tags."""

    @staticmethod
    def build_raiden_channel_tag(
        *,
        request_id: str,
        layer_idx: int,
        kv_component: Literal["key", "value"],
        head_idx: int,
    ) -> str:
        """Format canonical Raiden transfer channel tag."""
        return f"req_{request_id}_l{layer_idx}_{kv_component}_h{head_idx}"

    @staticmethod
    def parse_raiden_channel_tag(tag: str) -> dict[str, Any]:
        """Parse components from a Raiden channel tag."""
        parts = tag.split("_")
        # Expected: req_<uuid>_l<layer>_<comp>_h<head>
        return {
            "request_id": parts[1] if len(parts) > 1 else "",
            "tag": tag,
        }


class HeteroTranslator:
    """Top-level coordinator translating KV cache properties across hardware tiers."""

    def __init__(
        self,
        *,
        sharding: ShardingTranslator,
        layout: MemoryLayoutTranslator,
        dtype: DTypeTranslator,
        transport: TransportMetadataTranslator | None = None,
    ) -> None:
        self.sharding = sharding
        self.layout = layout
        self.dtype = dtype
        self.transport = transport or TransportMetadataTranslator()

    @classmethod
    def from_tier_configs(
        cls,
        *,
        producer_tier: HeteroTierConfig,
        consumer_tier: HeteroTierConfig,
        num_kv_heads: int,
        head_dim: int = 128,
        producer_block_size: int = 16,
        consumer_block_size: int = 16,
        producer_layout: LayoutStr = "canonical",
        consumer_layout: LayoutStr = "canonical",
    ) -> HeteroTranslator:
        """Factory creating a HeteroTranslator directly from two HeteroTierConfigs."""
        sharding = ShardingTranslator(
            producer_tp=producer_tier.tensor_parallel_size,
            consumer_tp=consumer_tier.tensor_parallel_size,
            num_kv_heads=num_kv_heads,
            producer_cores_per_chip=1 if producer_tier.hardware_generation == "v6e" else 2,
            consumer_cores_per_chip=2 if consumer_tier.hardware_generation == "v7x" else 1,
        )

        layout = MemoryLayoutTranslator(
            producer_block_size=producer_block_size,
            consumer_block_size=consumer_block_size,
            head_dim=head_dim,
            producer_layout=producer_layout,
            consumer_layout=consumer_layout,
            producer_alignment_bytes=128 if producer_tier.hardware_generation == "v6e" else 256,
            consumer_alignment_bytes=256 if consumer_tier.hardware_generation == "v7x" else 128,
        )

        dtype = DTypeTranslator(
            producer_dtype=producer_tier.kv_cache_dtype,
            consumer_dtype=consumer_tier.kv_cache_dtype,
        )

        return cls(sharding=sharding, layout=layout, dtype=dtype)

    def plan_transfer(
        self,
        *,
        request_id: str,
        num_tokens: int,
        num_layers: int,
    ) -> HeteroTransferPlan:
        """Synthesize a complete peer-to-peer transfer plan across tiers."""
        # 2 for Key and Value
        bytes_per_token_per_layer = 2 * self.sharding.num_kv_heads * self.layout.head_dim * self.dtype.producer_bytes
        total_transfer_bytes = num_tokens * num_layers * bytes_per_token_per_layer

        channel_mappings = self.sharding.build_channel_mappings()

        return HeteroTransferPlan(
            request_id=request_id,
            producer_tp=self.sharding.producer_tp,
            consumer_tp=self.sharding.consumer_tp,
            num_tokens=num_tokens,
            num_layers=num_layers,
            num_kv_heads=self.sharding.num_kv_heads,
            head_dim=self.layout.head_dim,
            bytes_per_token_per_layer=bytes_per_token_per_layer,
            total_transfer_bytes=total_transfer_bytes,
            channel_mappings=channel_mappings,
            communication_pattern=self.sharding.communication_pattern,
        )


__all__ = [
    "DTypeStr",
    "DTypeTranslator",
    "HeteroTransferPlan",
    "HeteroTranslator",
    "LayoutStr",
    "MemoryLayoutTranslator",
    "ShardChannelMapping",
    "ShardingTranslator",
    "TransportMetadataTranslator",
]
