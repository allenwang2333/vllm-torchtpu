# The Engineering Mechanics of LLMs & DeepSeek-V4

This guide bridges the gap between high-level conceptual analogies and low-level technical execution inside modern Large Language Models, using the exact architectural specifications of **DeepSeek-V4-Flash**.

---

## 1. Unified Architectural Glossary & Specifications

Rather than simple tables, each model dimension is mapped below from concept to implementation, detailing **what it is**, **how to think of it**, **why it is chosen**, and **its real value and context**.

### Transformer Block (Layer)
*   **Technical Definition**: A modular network block consisting of an [attention](#self-attention) mechanism followed by a [Feed-Forward Network (FFN/MLP)](#feed-forward-network-ffn--mlp), surrounded by [layer normalization](#rmsnorm--layer-normalization) and [residual connections](#residual-stream-residual_connection).
*   **The Analogy**: *A single stage in a conceptual assembly-line factory.*
*   **Why?**: Stacked transformer blocks allow hierarchical feature learning. Early blocks resolve raw syntax. Middle blocks resolve semantic contexts. Final blocks resolve logic, tone, and next-word constraints.
*   **DeepSeek-V4 Value & Context**: `43` total blocks.

### Hidden Size (hidden_size)
*   **Technical Definition**: The dimension of the activation vector (the [residual stream](#residual-stream-residual_connection)) flowing between [transformer layers](#transformer-block-layer).
*   **The Analogy**: *The number of semantic descriptors/adjectives used to define a single word.*
*   **Why?**: 4096 dimensions provides enough degrees of freedom to represent complex semantics and grammar. Because modern TPU/GPU matrix cores operate on blocks of 128/256 dimensions, a hidden size of 4096 matches TPU hardware alignments perfectly, maximizing hardware instruction efficiency.
*   **DeepSeek-V4 Value & Context**: `4096`

### Feed-Forward Network (FFN / MLP)
*   **Technical Definition**: A multi-layer perceptron block containing up-projection, [activation function](#swishsilu-activation-function), and down-projection matrices to map vectors non-linearly.
*   **The Analogy**: *The factual encyclopedia reference lookup.*
*   **Why?**: While [attention](#self-attention) moves information *between* words to resolve context, the FFN updates information *within* a single word's vector. It stores the model's factual database (e.g. associating the coordinates of "Paris" with "capital of France").
*   **DeepSeek-V4 Value & Context**: Configured as a sparse [Mixture of Experts (MoE)](#routed-experts-n_routed_experts) containing 256 routed experts and 1 dense shared expert.

### Self-Attention
*   **Technical Definition**: The core transformer operation where [Query, Key, and Value (Q, K, V)](#query-key-and-value-q-k-v) projections are computed from the same hidden state, aligning tokens based on dot-product matching.
*   **The Analogy**: *Each word looking at the rest of the sentence to see who it should talk to.*
*   **Why?**: Self-attention allows dynamic, prompt-dependent routing of information. A word's coordinate is modified based on its context within the prompt, enabling the model to parse pronouns and syntax.
*   **DeepSeek-V4 Value & Context**: Configured as Multi-head Latent Attention (MLA) and sliding window attention.

### Query, Key, and Value (Q, K, V)
*   **Technical Definition**: The three vector projections created from the hidden state during self-attention:
    *   **Query ($Q$)**: What the current word is looking for.
    *   **Key ($K$)**: What properties the other words offer.
    *   **Value ($V$)**: The actual content that is extracted once a match is made.
*   **The Analogy**: *The Question (Query), the Index Card Title (Key), and the Index Card Content (Value).*
*   **Why?**: Splitting a word vector into Q, K, and V allows the model to calculate relationship weights (using $Q \cdot K^T$) separately from the actual semantic information exchange ($Score \cdot V$).
*   **DeepSeek-V4 Value & Context**: Managed via latent low-rank matrices.

### Attention Head (Head)
*   **Technical Definition**: An independent slot inside the multi-head attention layer that calculates its own $Q, K, V$ dot products.
*   **The Analogy**: *One specialized context examiner in a panel of experts.*
*   **Why?**: Each head focuses on a different linguistic relation (e.g., Head 1 matches verbs to subjects, Head 2 matches pronouns to nouns, Head 3 tracks punctuation). 64 heads allow the model to build 64 separate contextual maps in parallel.
*   **DeepSeek-V4 Value & Context**: `64` heads.

### Head Dimension (head_dim)
*   **Technical Definition**: The size of the Query, Key, and Value vectors computed by an individual [attention head](#attention-head-head).
*   **The Analogy**: *The number of details used to answer a single context examiner's question.*
*   **Why?**: A larger head dimension (e.g. 512) allows capturing highly detailed coordinate correlations. DeepSeek uses a massive head dimension because its low-rank Key-Value compression mechanism projects multiple heads dynamically from a shared latent space.
*   **DeepSeek-V4 Value & Context**: `512`

### KV Latent Rank (kv_lora_rank / compress_rates)
*   **Technical Definition**: The dimension of the [latent space](#latent-space-compression) to which the Key and Value vectors are compressed before being cached in the [KV Cache](#kv-cache).
*   **The Analogy**: *A zip-file compression format for the model's short-term memory.*
*   **Why?**: In standard attention, storing raw Key-Value caches for thousands of tokens fills up the TPU's [High Bandwidth Memory (HBM)](#high-bandwidth-memory-hbm), leading to out-of-memory errors. Compressing them 8-fold reduces cache memory usage by 90%, enabling massive context limits.
*   **DeepSeek-V4 Value & Context**: `512` (down-projected from 4096)

### Latent Space / Compression
*   **Technical Definition**: A lower-dimensional vector space that captures the essential features of a higher-dimensional representation.
*   **The Analogy**: *Reducing a high-resolution photo to a thumbnail to save space, but keeping enough detail to reconstruct the image when needed.*
*   **Why?**: Storing coordinates in a smaller latent space minimizes memory throughput bottlenecks, moving the bottleneck from memory-bandwidth bound to compute bound.
*   **DeepSeek-V4 Value & Context**: Compresses hidden dimension $4096$ to latent dimension $512$ for KV projections.

### Rotary Positional Embedding (RoPE)
*   **Technical Definition**: A method of encoding position by multiplying Query and Key coordinate pairs by rotation matrices in 2D planes.
*   **The Analogy**: *Attaching clock hands to vectors and rotating them based on word positions.*
*   **Why?**: Attention is order-blind. By rotating vectors based on position index, the dot product between queries and keys automatically measures relative distance. 
*   **DeepSeek-V4 Value & Context**: `qk_rope_head_dim = 64`, `rope_theta = 10000`, `rope_scaling = yarn`.

### YaRN RoPE Scaling
*   **Technical Definition**: Yet another RoPE extensioN, a scaling method that interpolates and extrapolates rotary angles to support context lengths far beyond training limits.
*   **The Analogy**: *Adjusting the numbers on the clock face so the hands rotate slower for long sentences.*
*   **Why?**: Without scaling, rotating angles beyond training limits causes the dot products to go out of distribution, degrading generation quality. YaRN stretches the angles smoothly, supporting context windows up to 1M tokens.
*   **DeepSeek-V4 Value & Context**: Scale factor `16` is used to stretch coordinates.

### Sliding Window Size (sliding_window)
*   **Technical Definition**: The maximum distance over which attention dot products are calculated in Sliding Window Attention layers.
*   **The Analogy**: *The word-limit of the model's immediate visual window.*
*   **Why**: Using local attention windows on early layers reduces computational complexity and stabilizers training dynamics.
*   **DeepSeek-V4 Value & Context**: `128` (tokens)

### Routed Experts (n_routed_experts)
*   **Technical Definition**: The number of independent feed-forward networks (MLPs) available in an MoE layer.
*   **The Analogy**: *The number of specialized textbooks inside the library.*
*   **Why?**: Instead of a single massive MLP, the model divides its parameter storage into 256 experts. This allows the model to expand its total parameter capacity (its database of facts) without scaling up the compute cost per token.
*   **DeepSeek-V4 Value & Context**: `256` experts, sharded across devices.

### Shared Experts (n_shared_experts)
*   **Technical Definition**: The number of dense, non-sparse MLPs that are executed for every token.
*   **The Analogy**: *The general reference dictionary that stays open at all times.*
*   **Why?**: Routed experts are highly specialized. The 1 shared expert captures general language flow and common syntax patterns, reducing redundant calculations inside the routed experts.
*   **DeepSeek-V4 Value & Context**: `1` shared expert.

### Experts Per Token (num_experts_per_tok / topk_method)
*   **Technical Definition**: The number of active experts selected by the routing gate for each token.
*   **The Analogy**: *The number of textbooks the librarian pulls to answer a single question.*
*   **Why?**: Setting Top-K to 6 ensures that while the model has 256 total experts, only 6 are executed per token, keeping the active parameter count (and computational latency) low.
*   **DeepSeek-V4 Value & Context**: `6` experts active per token.

### Tensor Parallelism (TP)
*   **Technical Definition**: A sharding strategy that splits individual weight matrices (layers) across multiple chips within a single transformer step (e.g. slicing Queries/Keys/Values across TP=8 chips).
*   **The Analogy**: *Dividing a single massive math equation so 8 calculators solve different parts of it in parallel, then adding the results together.*
*   **Why?**: When a single weight matrix is too large to fit in the [HBM](#high-bandwidth-memory-hbm) of a single chip, TP splits it. This requires high-bandwidth [Inter-Chip Interconnects (ICI)](#inter-chip-interconnect-ici) since the chips must sync their intermediate values via `all-reduce` or `all-gather` operations during the forward pass.
*   **DeepSeek-V4 Value & Context**: Typically set to TP=1 (replicated) for Flash/Flash-lite sizes, or TP=2/4 for large models.

### Expert Parallelism (EP)
*   **Technical Definition**: A sharding strategy for MoE models where different [routed experts](#routed-experts-n_routed_experts) are placed on different TPU chips.
*   **The Analogy**: *Distributing 256 textbooks across 8 scholars; if Scholar 1 gets a word needing the physics textbook (stored on Scholar 5), they dispatch the word vector to Scholar 5 for computation.*
*   **Why?**: Because expert weights are executed independently, storing them on separate chips keeps memory footprint low while allowing all 256 experts to exist in the cluster. This requires `all-to-all` communication to route tokens to their target expert hosts.
*   **DeepSeek-V4 Value & Context**: Set to EP=8 on our 8-chip TPU v7 GKE cluster.

### Data Parallelism (DP) / Attention DP (attn_dp)
*   **Technical Definition**: A sharding strategy that replicates the entire model parameters on all chips, but processes different batches of input tokens on each chip.
*   **The Analogy**: *Printing 8 copies of the same book so 8 students can grade different chapters of homework concurrently.*
*   **Why?**: It is the simplest scaling format. In DeepSeek, "Attention DP" is used to process attention blocks in parallel across the mesh while the MoE experts are sharded via EP.
*   **DeepSeek-V4 Value & Context**: Dynamic scaling axis mapped dynamically inside the sharding manager.

### Pipeline Parallelism (PP)
*   **Technical Definition**: A sharding strategy that splits the layers of the model sequentially across different chips (e.g. layers 0-20 on Chip A, layers 21-43 on Chip B).
*   **The Analogy**: *An assembly line where worker A finishes stage 1, then passes the coordinate vector down the pipeline to worker B for stage 2.*
*   **Why?**: Used when the total model layers exceed the aggregate memory capacity of a single host group. It introduces pipeline bubbles (idle time where Chip B waits for Chip A to finish), so it is avoided unless memory capacity limits are exceeded.
*   **DeepSeek-V4 Value & Context**: Set to PP=1 (no pipelining) on our 8-chip single-host configuration.

### 3D Parallelism Mesh
*   **Technical Definition**: The logical coordinate grid structure (TP, PP, DP/EP) used to assign TPU device nodes into communication mesh groups.
*   **The Analogy**: *Structuring a classroom of 8 students into a 3D grid: Row size is TP, Column size is EP, and Stack depth is PP.*
*   **Why?**: Different operators (attention vs. MoE) scale best on different communication layouts. By defining a logical mesh grid, PyTorch/JAX can automatically route sharded coordinates along specific physical dimensions (like row-mesh `all-reduce` or column-mesh `all-to-all`).
*   **DeepSeek-V4 Value & Context**: Managed dynamically based on active JAX topology.

### RMSNorm / Layer Normalization
*   **Technical Definition**: Root Mean Square Normalization, scaling the vector coordinates to have a mean square value of 1.0 before passing them to attention/FFN blocks.
*   **The Analogy**: *A volume control knob that keeps coordinates from blowing up.*
*   **Why?**: As values pass through matrix multiplications in deep layers, their coordinates can scale exponentially (causing numerical overflow). Normalization stabilizes the math. RMSNorm is mathematically cheaper than standard LayerNorm because it skips computing the vector mean, saving valuable TPU clock cycles.
*   **DeepSeek-V4 Value & Context**: `rms_norm_eps = 1e-06`.

### Residual Stream (residual_connection)
*   **Technical Definition**: Adding the input tensor of a transformer block directly back to its output tensor: $X_{next} = X + \text{Output}$.
*   **The Analogy**: *A clean high-speed highway running straight through all 43 factory stages.*
*   **Why?**: In deep networks, gradients fade or explode during backward propagation (the vanishing gradient problem). The residual stream allows original word features to flow cleanly to the end, while layers only calculate adjustments to add to this highway.
*   **DeepSeek-V4 Value & Context**: Wrapped around all major blocks.

### Tokenizer
*   **Technical Definition**: A software preprocessing program that converts text strings into lists of token IDs (integers) and vice versa.
*   **The Analogy**: *The codebook translator.*
*   **Why?**: Tokenizers ensure that words are broken down into optimal mathematical pieces. Common words get a single integer ID, while rare or misspelled words are chopped into smaller sub-word pieces (syllables/letters), ensuring the model never encounters an "unknown word" error.
*   **DeepSeek-V4 Value & Context**: Byte-level Byte-Pair Encoding (BPE) tokenizer.

### KV Cache
*   **Technical Definition**: A memory cache storing the Key and Value activations of all generated tokens in HBM.
*   **The Analogy**: *Scratchpad memory storing past thoughts to avoid re-reading them.*
*   **Why?**: Without a KV Cache, generating token #100 would require re-running the entire model on tokens #1 to #99. Caching key/value states keeps generation latency low, but consumes massive memory capacity.
*   **DeepSeek-V4 Value & Context**: Compressed dynamically using MLA (latent rank 512).

### Logits
*   **Technical Definition**: The raw, unnormalized scoring outputs produced by the final linear projection ([LM Head](#language-model-head-lm-head)).
*   **The Analogy**: *The raw votes for every word in the dictionary.*
*   **Why?**: Logits are arbitrary real numbers (e.g. `[12.5, -4.2, 0.1]`). They indicate how strongly the model votes for each word. Before sampling, they must be converted to probabilities.
*   **DeepSeek-V4 Value & Context**: Shape `[batch_size, 129280]`.

### Language Model Head (LM Head)
*   **Technical Definition**: The final linear layer (`lm_head`) that projects the output hidden state of shape `[hidden_size]` to `[vocab_size]`.
*   **The Analogy**: *The final megaphone that projects coordinates back into vocabulary slots.*
*   **Why?**: The LM head translates the finalized semantic coordinates back into raw votes ([logits](#logits)) for every word in the vocabulary list.
*   **DeepSeek-V4 Value & Context**: Output projection weight matrix.

### Softmax
*   **Technical Definition**: A normalization function that exponentiates elements and divides them by the sum of exponents: $\sigma(z)_i = \frac{e^{z_i}}{\sum e^{z_j}}$.
*   **The Analogy**: *Converting raw votes into percentages that sum to 100%.*
*   **Why?**: Exponentiating logits via Softmax ensures that all negative scores become positive, and the final list sums to exactly 1.0 (probabilities), allowing a sampler to choose the next token.
*   **DeepSeek-V4 Value & Context**: Computed dynamically at final head.

### Temperature & Sampling
*   **Technical Definition**: Parameters controlling how tokens are selected from the Softmax probability distribution.
    *   **Temperature ($T$)**: Modifies logits before Softmax: $\frac{z_i}{T}$. A low temperature ($0.2$) flattens minor votes, making outputs predictable. A high temperature ($0.9$) spreads probabilities, making outputs creative.
    *   **Top-P / Top-K**: Filters out unlikely words before sampling to prevent the model from generating random nonsense tokens.
*   **The Analogy**: *The creativity dial.*
*   **Why?**: Sampling controls the determinism and diversity of generation.

### Swish/SiLU Activation Function
*   **Technical Definition**: Sigmoid Linear Unit, defined as $f(x) = x \cdot \text{sigmoid}(x)$.
*   **The Analogy**: *A smooth gate filter.*
*   **Why?**: Gated activation functions replace older functions like ReLU. Because SiLU has a smooth gradient profile, it allows gradient signals to propagate more cleanly during backpropagation, leading to superior parameter convergence.
*   **DeepSeek-V4 Value & Context**: Activates intermediate MLPs.

### Parameters / Weights
*   **Technical Definition**: The learned floating-point values stored in matrices that perform the scaling and rotation operations on vectors.
*   **The Analogy**: *The locked knobs inside the control panel.*
*   **Why?**: Weights represent the cached learning memory of the model.
*   **DeepSeek-V4 Value & Context**: ~103B total parameters.

### High Bandwidth Memory (HBM)
*   **Technical Definition**: High-performance 3D-stacked RAM memory integrated directly onto the TPU/GPU silicon package.
*   **The Analogy**: *A super-wide multi-lane highway connecting the memory storage to the CPU/TPU core.*
*   **Why?**: LLM generation is memory-bandwidth bound (reading weights from HBM to SRAM limit generation speeds). HBM provides TB/s bandwidth speeds to feed weights to matrix calculation units.
*   **DeepSeek-V4 Value & Context**: 32GB HBM per TPU chip.

### Inter-Chip Interconnect (ICI)
*   **Technical Definition**: High-speed, low-latency direct link ports that connect TPUs within a pod mesh.
*   **The Analogy**: *A fiber-optic network connecting the TPU nodes together in a single room.*
*   **Why?**: EP routing requires constant collective dispatching (`all-to-all` token sends). ICI links bypass the host VM networking stack, enabling sub-millisecond tensor exchange.
*   **DeepSeek-V4 Value & Context**: Connected via custom TPU ring links.

### Weight Block Size (weight_block_size)
*   **Technical Definition**: The dimensions of the sub-matrices that share a single quantization scaling factor.
*   **The Analogy**: *The size of the sub-blocks that share a single multiplier.*
*   **Why?**: FP4 numbers have extremely limited range. If you scale a whole matrix with a single factor, small parameters are rounded to zero, ruining model accuracy. Scaling in tiny $128 \times 128$ sub-blocks ensures the scale factor matches the local range of parameters perfectly, preserving precision.
*   **DeepSeek-V4 Value & Context**: `[128, 128]`

---

## 2. Step-by-Step Data Flow & Mechanics

Here is the exact journey of a token through a single layer of DeepSeek-V4-Flash:

```
[Input Tokens: (batch, seq)] 
       |
       v  (Embedding Lookup)
[Hidden State X: (batch, seq, 4096)]
       |
       +----> [Multi-head Latent Attention (MLA)]
       |             |
       |             +--> Down-project keys/values to latent space (dim 512).
       |             +--> Split off positional vectors (dim 64) and rotate via RoPE.
       |             +--> Perform dot-product query-key matching.
       |             +--> Average value coordinates and project back to dim 4096.
       |             |
       |             v
       |      [Attention Output: (batch, seq, 4096)]
       |             |
       |             v  (Residual Addition)
       |      [X_attn = X + Attention Output]
       |             |
       |             v  (RMSNorm Normalization)
       +------------>+
                     |
                     v
       [Contextualized State: (batch, seq, 4096)]
                     |
                     +----> [Mixture of Experts (MoE)]
                     |             |
                     |             +--> Router projects state to 256 expert logits.
                     |             +--> Select top-6 experts per token.
                     |             +--> Dispatch tokens via All-to-All communication.
                     |             +--> Execute Grouped GEMM SwiGLU MLP kernels on TPU.
                     |             +--> Execute Shared Expert MLP.
                     |             +--> Combine outputs and All-to-All collect.
                     |             |
                     |             v
                     |      [MoE Output: (batch, seq, 4096)]
                     |             |
                     |             v  (Residual Addition)
                     |      [X_next = X_attn + MoE Output]
                     |             |
                     +------------>+
                                   |
                                   v
                      [Layer Output: (batch, seq, 4096)]
```

---

## 3. Hardware Co-Design: Low-Precision FP4 Quantization

Low-precision formats reduce the memory storage footprint of model weights.

| Format | Bits | Mantissa (Precision) | Exponent (Range) | Memory per 100B parameters |
| :--- | :--- | :--- | :--- | :--- |
| **BF16** | 16 | 7 bits | 8 bits | 200 GB |
| **FP8** | 8 | 3 or 4 bits | 4 or 5 bits | 100 GB |
| **FP4** | 4 | 1 bit | 2 bits | 50 GB |

*   **Exponent vs. Mantissa**: The exponent controls the scale of the number, while the mantissa controls the precision/details. At FP4 precision, we only have **1 bit** for the mantissa, which provides extremely limited precision.
*   **Blockwise Scaling**: To prevent mathematical overflow/underflow at FP4 precision, DeepSeek-V4 uses a blockwise scale factors matrix. A separate scale factor (usually stored at FP8 or BF16 precision) is assigned to tiny sub-blocks of the weight matrix (size $128 \times 128$). This shifts the exponent scale dynamically per block, keeping the FP4 numbers within their representable range and preserving output token accuracy.

---

## 4. Behind the Scenes: The Startup & JIT Compilation Sequence

When a model run is submitted to the GKE TPU cluster, a highly coordinated sequence of hardware and software compilation passes occurs before a single token can be generated.

### Phase 1: Platform & Device Binding
1.  **Backend Registration**: The Python container invokes PyTorch bindings (`torchax`), which register the custom `PrivateUse1` backend dynamically as `'tpu'`.
2.  **Workload Admission**: Kueue validates the cluster resource quota and schedules the JobSet onto 8 physical TPU v7 chips.
3.  **Topology Discovery**: The engine queries local metadata to discover the pod's Inter-Chip Interconnect (ICI) layout (in our case, topology mesh shape `2x2x1`).

### Phase 2: Graph Tracing (HLO IR Generation)
1.  **Dynamic Tracing**: JAX executes the model's forward math loops in Python, recording the tensor shape transitions.
2.  **HLO IR Emission**: The JAX tracer translates Python operations into an intermediate representation graph called **High-Level Optimizer (HLO) IR**.
3.  **Sharding Annotation**: JAX scans the `ShardingAxis` configurations, annotating the HLO graph edges with device location bindings (e.g. marking MoE expert layers to shard along `"expert"` axis, while marking attention layers as replicated).

### Phase 3: XLA Compilation & Memory Allocation
1.  **SPMD Partitioning**: XLA (Accelerated Linear Algebra) compiler compiles the annotated HLO graph. It splits the graph nodes across the 8 devices, inserting collective operations (like `all-to-all` for MoE routing and `all-gather` for attention outputs) at the partition boundaries.
2.  **Operator Fusion**: XLA merges adjacent mathematical nodes (e.g. fusing a dot product multiplication with a Sigmoid gating activation) into a single unified kernel. This prevents the TPU from having to write intermediate coordinates back to HBM, dramatically increasing speed.
3.  **Memory Space Assignment (MSA)**:
    *   **HBM Allocation**: Reserves storage blocks for static weight tables (totaling ~60GB).
    *   **VMEM (On-Chip SRAM) Routing**: Configures the high-speed local vector memory cache (SRAM) for active activation tensors (intermediate coordinates during forward loops).
    *   **Local Scratchpad**: Allocates local flag locks (sflags, smem) on each core to synchronize parallel threads.
4.  **Pallas Low-Level Compilation**: The custom Pallas kernels (grouped GMM MoE matrix math, MLA custom attention, streamindex top-k) are lowered to hardware-specific ISA binary code.

### Phase 5: Warmup & Execution Loop
1.  **Dry Run (Warmup)**: The TPU executes a zero-input forward pass to trigger final runtime bindings.
2.  **Generation Loop**: The model processes the prompt token IDs, executing the compiled executable binaries on the TPU cores, and returns the output tokens.
