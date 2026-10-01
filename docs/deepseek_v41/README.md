# DeepSeek V4.1 on TorchTPU

The runtime entry is
`vllm_torchtpu.models.vllm.deepseek_v41.model:DeepseekV41ForCausalLM`.
Model construction, decoder order, checkpoint names and cache ownership are
inherited from pinned vLLM v0.30.0.

The retained path targets DeepSeek-V4.1-Flash text generation with MXFP4 routed
experts, FP8 indexer values, BF16 KV storage, TP/EP, PP1 and DP1. Scheduling is
synchronous, without speculation or EPLB. Vision and alternative expert/indexer
formats are outside this implementation.

## Runtime files

| File | Responsibility |
| --- | --- |
| `models/vllm/deepseek_v41/model.py` | Model assembly and compiled encoder/decoder stages |
| `ced.py` | Prompt-tail replay planning across scheduler chunks |
| `attention.py` | MLA, compressor state and output projection |
| `attention_kernels.py` | Paged attention, exact index selection and publication |
| `cache.py` | Native allocation, paged record addressing and mutation |
| `engram.py` | Upstream hashing, request lookbacks and TPU lookup |
| `mhc.py` | Delayed mHC, compact Sinkhorn and residual flow |
| `ops.py` | Scoped device bindings, RoPE and small tensor helpers |
| `layers/adapter/quantization/deepseek_v41.py` | V4 quantization dispatch and MoE model bindings |

Paths in the table are relative to `src/vllm_torchtpu/`; model filenames after
the first row use the same model directory.

V4.1 uses the same quantization methods and runtime kernels as the native
TorchTPU V4 model. Dense FP8 weights are dequantized and requantized using the
shared linear configuration (per-channel FP8 by default). MXFP4 expert weights
use the V4 requantizer (`MOE_REQUANTIZE_BLOCK_SIZE`, default 512). The shared
GMM/fused-EP paths own activation quantization, expert weighting and reduction.
Shared experts use the V4 MLP and the MoE runner's combined TP reduction;
`apply_routed_scale_to_output=True` matches the native V4 model. `wo_a` retains
runtime FP8 weights and uses V4's fused inverse-RoPE/FP8 output-projection kernel.
There is no V4.1-specific source-format arithmetic path.

KV storage and ownership remain V4.1's BF16 cache implementation; this does not
adopt V4's packed SWA/CSA cache layout or change V4.1's compressor architecture.
Cache kernels use record indices and V4's 128-word row geometry, retaining
two-dimensional row/lane addressing for large shared pools.
Like V4, the attention and compressor paths consume the shared TPU
`AttentionMetadata`; each cache derives its slot or ring addresses in the model
from its own block table and compression geometry.
Engram's checkpoint table format and V4.1 indexer rounding are unchanged.

Hierarchical reindex builds a compact key-record list before scoring. For a
fixed candidate pool, later indexers' key reads and QK work are bounded by that
pool, including tile padding. The first indexer still scores the full visible
context. Candidate sorting, duplicate removal, short-context bypass and
underfull top-k padding preserve the full-score path's selected logical rows.

`TPU_DEEPSEEK_V41_CED=1` enables decoder SWA bounded replay. The encoder processes
every scheduled token and publishes decoder global KV before selecting decoder
rows. Only the final SWA window of the complete prompt enters the decoder;
the window starts at the same position across all prefill chunks and subsequent
decode steps. The two stages compile and dispatch with independent token
buckets. Mixed batches retain request block-table ownership and restore output
rows to the runner's original layout.

Bounded replay deliberately approximates decoder SWA state; it is not
numerically equivalent to a full decoder pass for long prompts. The default
remains the full-forward reference until matched quality and performance
validation is complete. Prefix caching uses the full-forward path even when
the option is set: rebuilding the missing decoder window from cached encoder
outputs is not implemented. Bounded replay requires unsharded token rows
(PCP1, no sequence parallelism); existing PP1/DP1 and speculation restrictions
also apply.

The shared runner changes provide only registration, native-cache binding,
metadata, Engram lookbacks and compilation-cache invalidation. The vLLM version
bump also requires the small NVFP4 linear schema adaptation imported by the
shared quantization registry; it does not select NVFP4 for this model.

## Launch and validation

Use the existing benchmark flow with
`scripts/vllm/benchmarking/configs/deepseek-v41-flash-tp8-ep.sh`.
That configuration sets the V4.1 gather workaround explicitly, without changing
other models' automatic libtpu flags.

Validation scripts and raw experiment evidence are maintained separately from
the runtime change. BF16 KV storage differs from the official packed-cache
numerics. Cache validation alone does not establish full-model numerical
equivalence or serving performance.
