# MXFP4 MoE runtime scale decoding

Set this before loading the model to keep checkpoint MXFP4 scales in their
one-byte E8M0 representation:

```bash
export TPU_MOE_RUNTIME_SCALE_CAST=1
```

The default is `0`, which retains the existing FP32 scale loading behavior.
Restart and reload the model when changing the setting.

With the setting enabled, the MXFP4 MoE loader keeps uint8 scales in the GMM
layout `[experts, quant_blocks, output_size]`. The broadcast dimension is
introduced only inside the VMEM tile. The local GMM pipeline copies
each scale tile from HBM into VMEM and constructs its FP32 exponent bits.
For pre-matmul weight dequantization it casts the small scale tile to the
activation dtype before broadcasting and the fused gate/up interleave.
The scale is then applied in the existing weight dequantization path.
There is no full-tensor FP32 conversion at the GMM entry point. Both gate/up
and down projections support this representation, including fused activation.

The decoder preserves the existing loader's `2**(byte - 127)` convention:
byte 0 represents `2**-127`, and byte 255 becomes positive infinity. It does
not use the native E8M0 dtype's NaN convention for byte 255.

If `MOE_REQUANTIZE_BLOCK_SIZE` is set, scales are still decoded during loading
and the requantized FP32 scales are retained. Those scales are general floating
point values and cannot be stored as E8M0 without further quantization. The
runtime setting has no effect on that path or on other quantization loaders.

For unchanged checkpoint block sizes, persistent scale storage falls from
4 bytes to 1 byte per element (75% smaller). Actual available KV capacity
also depends on padding, state pools, workspace, and runtime allocations.

## Layout compatibility

Encoded scales remain three-dimensional through the fused MoE wrapper.
For compact uint8 DMA, K tiles must span a multiple of four quantization
blocks. Smaller or unaligned tiles use the previous four-dimensional
broadcast-axis representation inside the GMM boundary for compatibility;
that fallback can retain layout-copy overhead. The tested Kimi-K3 block32
gate/up and down tiles both use the compact path. No decoded-scale or
expanded-weight buffering stage is added, and tile-size selection keeps
its existing conservative VMEM allowance.
