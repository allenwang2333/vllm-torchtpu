# vLLM-Omni on Google Cloud TPU Architecture & Design

This document details the end-to-end architecture, platform plugin integration, and multi-stage request execution flow for running **vLLM-Omni** models on **Google Cloud TPUs** using **`vllm-torchtpu`**.

---

## 🏗️ 1. End-to-End System Architecture

```mermaid
flowchart TB
    %% Entry Layer
    subgraph EntryLayer["1. Entry Points & User Interface"]
        CLI["vllm_omni serve / vllm serve --omni"]
        PythonAPI["Omni / AsyncOmni Python API"]
        OpenAIAPI["OpenAI-Compatible & Duplex REST/WS API"]
    end

    %% vLLM-Omni Control Plane
    subgraph OmniCore["2. vLLM-Omni Core Engine & Orchestrator"]
        Engine["AsyncOmniEngine<br/>(Background loop, stage lifecycle, async chunking)"]
        Orchestrator["Stage Orchestrator<br/>(Cross-stage routing, request state, cancellation)"]
        StageRuntime["StageRuntime<br/>(Process placement, replica management)"]
        OmniResolver["Platform Discovery & Resolver<br/>load_omni_plugins_by_group('vllm_omni.platform_plugins')"]
    end

    %% Plugin Boundary
    subgraph PluginBoundary["3. vLLM-TorchTPU Platform Plugin (vllm_torchtpu.omni)"]
        EntryPoint["Entry Point: torchtpu<br/>register_omni_tpu_platform()"]
        OmniTpuPlatform["OmniTpuPlatform<br/>(Inherits OmniPlatform & TpuPlatform)"]
        
        subgraph PlatformServices["Platform Services"]
            DevControl["Device & Memory Manager<br/>• torch.device('tpu')<br/>• HBM memory querying<br/>• torch.tpu.synchronize()"]
            AttnBackend["Diffusion Attention Backend<br/>• TpuSDPABackend<br/>• TpuSDPAImpl (F.scaled_dot_product_attention)"]
            WorkerBindings["Worker Class Resolver<br/>• get_omni_ar_worker_cls()<br/>• get_omni_generation_worker_cls()"]
        end
    end

    %% Execution Engine
    subgraph TPUExecution["4. TPU Execution Runtime (vllm-torchtpu)"]
        TPUWorker["TPUWorker / TPUModelRunner"]
        
        subgraph KernelPipelines["Stage & Kernel Pipelines"]
            ARPipeline["AR / VLM Stages<br/>(Thinker / Talker / Text-AR)"]
            DiffusionPipeline["Diffusion / DiT Stages<br/>(Image, Video, Audio Generation)"]
            CustomKernels["TPU Kernels & Acceleration<br/>• PagedAttention / FlashAttention<br/>• Fused MoE / MegaBlox<br/>• SDPA Attention Kernels"]
        end

        subgraph DistributedEngine["Distributed & Parallelism"]
            TP_EP["Tensor & Expert Parallelism (TP / EP)"]
            PCP["Context Parallelism (PCP)"]
            Raiden["Raiden KV DMA / Disaggregated Transfer"]
        end
    end

    %% Hardware
    subgraph HardwareLayer["5. Google Cloud TPU Infrastructure"]
        TorchTPU["Torch-TPU / PyTorch XLA Runtime"]
        LibTPU["LibTPU / PJRT Driver"]
        TPUHardware["TPU Hardware (v5e / v5p / v6e / v7x Trillium Chips)"]
    end

    %% Connections
    EntryLayer --> Engine
    Engine --> Orchestrator
    Engine --> StageRuntime
    StageRuntime --> OmniResolver

    OmniResolver -->|Discovers entry point| EntryPoint
    EntryPoint --> OmniTpuPlatform
    OmniTpuPlatform --> DevControl
    OmniTpuPlatform --> AttnBackend
    OmniTpuPlatform --> WorkerBindings

    WorkerBindings --> TPUWorker
    AttnBackend -.-> DiffusionPipeline

    StageRuntime --> TPUWorker
    TPUWorker --> ARPipeline
    TPUWorker --> DiffusionPipeline
    ARPipeline --> CustomKernels
    DiffusionPipeline --> CustomKernels

    TPUWorker --> DistributedEngine
    CustomKernels --> TorchTPU
    DistributedEngine --> TorchTPU
    TorchTPU --> LibTPU
    LibTPU --> TPUHardware
```

---

## 🔄 2. Multi-Stage Request Execution Lifecycle

The sequence diagram below illustrates how an end-to-end omni request (such as **Qwen3-Omni**: *Multimodal Audio/Vision Input → Thinker → Talker → Code2Wav Audio Output*) executes across pipeline stages on TPUs:

```mermaid
sequenceDiagram
    autonumber
    actor User as Client / User Request
    participant Engine as AsyncOmniEngine
    participant Orch as Orchestrator
    participant Plat as OmniTpuPlatform
    participant S1 as Stage 1: Thinker (AR TPUWorker)
    participant S2 as Stage 2: Talker (AR TPUWorker)
    participant S3 as Stage 3: Code2Wav / Decoder
    participant TPU as Google Cloud TPU Chips

    User->>Engine: Submit Multimodal Request (Text + Audio/Image)
    Engine->>Plat: Query hardware capabilities & memory
    Plat-->>Engine: HBM budget & TPU device configuration
    Engine->>Orch: Initialize cross-stage request context

    rect rgb(240, 248, 255)
        note over Orch,S1: Stage 1 - Multimodal Understanding (Thinker)
        Orch->>S1: Forward prompt + vision/audio embeddings
        S1->>TPU: Run prefill & decode with PagedAttention (TP/EP)
        TPU-->>S1: Generated Thinker hidden states / text tokens
        S1-->>Orch: Intermediate conditioning tensor stream
    end

    rect rgb(255, 250, 240)
        note over Orch,S2: Stage 2 - Audio Token Generation (Talker)
        Orch->>S2: Stream intermediate conditioning (Async Chunking)
        S2->>TPU: Autoregressive decode on TPU
        TPU-->>S2: Codec tokens & text output
        S2-->>Orch: Streamed audio codec chunks
    end

    rect rgb(245, 255, 245)
        note over Orch,S3: Stage 3 - Audio Waveform Synthesis (Code2Wav)
        Orch->>S3: Dispatch codec chunks
        S3->>TPU: Decode codec to waveform
        TPU-->>S3: Audio PCM / Waveform buffer
        S3-->>Engine: Stream audio packets (Low TTFP)
    end

    Engine-->>User: Stream real-time text & audio packets
```

---

## 🧩 3. Key Components & Implementation Mapping

| Layer / Component | Code Reference | Function / Responsibility |
| :--- | :--- | :--- |
| **Plugin Entry Point** | [`pyproject.toml:L102-103`](file:///usr/local/google/home/rishabhbaghel/vllm-torchtpu/pyproject.toml#L102-L103) | Registers `vllm_omni.platform_plugins` pointing to `register_omni_tpu_platform` |
| **Platform Plugin** | [`src/vllm_torchtpu/omni/platform.py`](file:///usr/local/google/home/rishabhbaghel/vllm-torchtpu/src/vllm_torchtpu/omni/platform.py) | Implements `OmniTpuPlatform` combining `OmniPlatform` and `TpuPlatform` |
| **Device & Memory Setup** | `OmniTpuPlatform.set_device()`, `get_device_total_memory()` | Manages `torch.device("tpu")`, HBM queries via `hardware.get_hbm_bytes_per_device()`, and synchronization |
| **Diffusion SDPA Backend** | [`src/vllm_torchtpu/omni/attention.py`](file:///usr/local/google/home/rishabhbaghel/vllm-torchtpu/src/vllm_torchtpu/omni/attention.py) | Implements `TpuSDPABackend` & `TpuSDPAImpl` using PyTorch `scaled_dot_product_attention` for DiT/Diffusion |
| **Stage Worker Class** | `OmniTpuPlatform.get_omni_ar_worker_cls()` | Returns `vllm_torchtpu.worker.tpu_worker.TPUWorker` for AR & generation stages |
| **Orchestrator & Engine** | `AsyncOmniEngine`, `StageRuntime` | Manages request lifecycle, stage placement, and async chunk transport between TPU workers |
