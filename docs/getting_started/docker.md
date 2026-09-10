# Docker Guide for vLLM TPU

This guide provides instructions on how to use pre-built Docker container images and build custom images for `vllm-torchtpu`.

---

## 🚀 Pre-built Nightly Images

Pre-built Docker images are automatically generated on a nightly cadence and hosted on Google Artifact Registry.

### 1. Authentication

To pull images, authenticate Docker with Google Artifact Registry:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 2. Available Image Targets

You can pull the following pre-built image targets using the `:latest` tag:

* **Production Image** (`prod:latest`): Recommended for serving and production workloads.

  ```bash
  docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-prod:latest
  ```

* **Developer Image** (`dev:latest`): Includes the repository installed in editable mode (`-e ".[test,benchmarking]"`), recommended for debugging and development.

```bash
  docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-dev:latest
  ```

* **CI Image** (`ci:latest`): Minimal environment containing build and test tools for CI/CD runners.

  ```bash
docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-ci:latest

  ```

### 3. Date-Specific Tags

To reproduce results or pin to a specific build date, use the `nightly-YYYYMMDD` tag format:

```bash
docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-<target>:nightly-YYYYMMDD
```

Example:

```bash
docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-prod:nightly-20260523
```

---

## 🏃 Running Containers on Cloud TPUs

When running a container on a Cloud TPU VM or GKE node, ensure you grant privileged access and host networking to attach TPU devices:

```bash
docker run --privileged --net=host --ipc=host \
  --name vllm-tpu-server \
  -v /dev:/dev \
  us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-prod:latest \
  vllm serve Qwen/Qwen3-0.6B
```

---

## 🛠️ Building Images Locally

The repository includes a build script `./docker/build_image.sh` to construct image targets locally.

### Build Targets

1. **Build Developer Image**:

   ```bash
   ./docker/build_image.sh --target dev -t dev:latest
   ```

2. **Build Production Image**:

```bash
   ./docker/build_image.sh --target prod -t prod:latest
   ```

1. **Build CI Image**:

   ```bash
./docker/build_image.sh --target ci -t ci:latest

   ```

### Advanced Build Arguments

* `BASE_IMAGE`: Override the base runtime container image.
* `VLLM_SOURCE`: When set, compiles and installs `vllm` directly from a local checked-out source tree.
