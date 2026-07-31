# Kimi K3 / Kimi-Linear TPU model

This package is a text-only Kimi K3 and Kimi-Linear implementation in the same
style as other vLLM models. It accepts `VllmConfig`, flattened token tensors,
engine-managed attention state, and standard vLLM checkpoint loading.

The model reuses vLLM's tensor-parallel linear, embedding, logits,
weight-loading, attention-registration, and inner-state infrastructure. Its
K3-specific execution paths use Torch custom ops backed by TPU Pallas kernels:

- MLA uses vLLM's latent-cache interface and the TorchTPU Pallas MLA backend.
- KDA short convolution and recurrent delta updates are separate Pallas custom
  ops over engine-managed state.
- Standard and Stable LatentMoE use vLLM's `FusedMoE`, whose TorchTPU
  quantization methods dispatch to the Pallas grouped-matmul kernel.
- SITU is fused into the Pallas grouped matmul. Attention residuals remain a
  short, direct PyTorch equation.

MLA KV cache and KDA recurrent state are allocated and passed by vLLM through
their registered attention layers.

`KimiLinearForCausalLM` is registered directly, so the public
`moonshotai/Kimi-Linear-48B-A3B-Instruct` checkpoint needs no architecture
override. Pipeline parallelism is intentionally unsupported. Tensor parallelism
uses vLLM's standard sharded linear, embedding, and collective layers.
