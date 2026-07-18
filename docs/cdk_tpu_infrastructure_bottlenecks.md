# GKE TPU Workload Bottlenecks, Causes & Workarounds

This document outlines the three primary infrastructure bottlenecks encountered when running massive model inference (such as DeepSeek-V4) on GKE TPU clusters, explaining the underlying causes, temporary developer workarounds, and permanent solutions.

---

## 1. Slow Model Weights Loading (NFS Bottleneck)

*   **The Issue**: Loading raw model checkpoints (e.g., 200GB+) from a shared network NFS mount to TPU HBM takes **over 25 minutes** during startup.
*   **The Underlying Cause**: 
    *   TPU worker processes perform parallel read operations on the checkpoint files.
    *   This parallel read load hammers the shared NFS volume over a single network TCP socket channel. The shared network bandwidth saturates, forcing the processes into an **uninterruptible wait state (`D` state)** while waiting for network packet arrival.
*   **Temporary Workaround**:
    *   **Use Truncated Model Configurations**: For early-stage validation, shape testing, and kernel verification runs, use a truncated/toy model slice (e.g., a 4-layer configuration). This reduces weight data size from 200GB to ~20GB, minimizing load times.
*   **Long-Term Solution**:
    *   Archive the model checkpoints to a Google Cloud Storage (GCS) bucket.
    *   Configure GKE pod init containers to download checkpoint files directly from GCS to the node's local NVMe SSD storage (`/mnt/local-cache/`) upon container startup. Local NVMe reads bypass the network share entirely.

---

## 2. Long JIT Compilation Times (Cold Boot Cache)

*   **The Issue**: Tracing and compiling the graph takes **10 to 12 minutes** on cold starts before inference begins.
*   **The Underlying Cause**: 
    *   The XLA compiler must trace the model architecture and custom Pallas kernels to generate High-Level Optimizer (HLO) graphs.
    *   It must then calculate optimal operator fusions, layout memory space assignments (MSA) on the on-chip SRAM (VMEM), and compile kernels to device machine instructions.
    *   Without a persistent compiler cache directory, these graph optimization passes must be computed from scratch on the host CPU for every container execution.
*   **Temporary Workaround**:
    *   **Pre-Compile on Dedicated TPU VMs**: Perform the initial graph tracing and compilation pass on a dedicated TPU VM (which has instant local SSD access), and copy the resulting JAX cache folder directly from the VM to GCS to seed the cluster queue.
*   **Long-Term Solution**:
    *   Enable persistent JAX compiler caching inside the container:
        ```bash
        export JAX_COMPILER_CACHE_DIR=/mnt/local-cache/jax_cache
        ```
    *   Configure GKE wrapper scripts to archive this compiler cache directory to a persistent GCS path upon successful completion, and download it to local SSD on subsequent pod startups.

---

## 3. TPU Hardware Scale-Up Latency (Autoscaler Loops)

*   **The Issue**: Submitted JobSets sit in a `Pending` state for **10 to 15 minutes** before container execution begins.
*   **The Underlying Cause**: 
    *   To minimize idle compute resource costs, GKE TPU node pools scale down to 0 nodes when inactive.
    *   Submitting a job forces the GKE Autoscaler to trigger GCP API requests to provision physical TPU virtual machines, wait for the host OS to boot, and register nodes with the cluster before GKE can schedule the pods.
*   **Temporary Workaround**:
    *   **Validate on Local VMs First**: Run shape compilation tests, syntax parsing checks, and layer-by-layer verifications on dedicated persistent debug VMs (e.g., local host VM slices) to bypass cluster queue delays during rapid iteration loops.
*   **Long-Term Solution**:
    *   Configure GKE node pool rules to maintain a minimum node count of 1 (a pre-warmed node slice) during active development hours.
    *   Utilize GKE `ProvisioningRequests` to pre-reserve hardware instances before executing debugging runs.
