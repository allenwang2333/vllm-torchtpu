# Docker images

This folder contains the Dockerfile used to build the main image variants for torchtpu-vllm.

## Image targets

### `ci:latest`
CI image target built from `--target ci`.

Use this image for CI and test infrastructure. It includes extra tooling on top of the base runtime, such as:

- `git`
- `google-cloud-cli`
- `gcsfuse`
- `lsof`
- `psmisc` / `fuser`

Example build:

```bash
docker build -f docker/Dockerfile --target ci -t ci:latest .
```

### `dev:latest`
Developer image target built from `--target dev`.

This image is based on the `ci` stage, copies the repository into the image, and installs the project in editable mode with test and benchmarking extras:

- editable install: `-e ".[test,benchmarking]"`
- intended for local development, debugging, and running tests

Example build:

```bash
docker build -f docker/Dockerfile --target dev -t dev:latest .
```

### `prod:latest`
Production image target built from `--target prod`.

This image is intended for regular runtime usage. It performs a standard install of the project and is the best target for production-style execution.

Example build:

```bash
docker build -f docker/Dockerfile --target prod -t prod:latest .
```

## Useful build arguments

The Dockerfile also supports a few build arguments that may be useful in advanced workflows:

- `ARTIFACT_SOURCE`: source image used to copy prebuilt `torch-tpu` wheels
- `BASE_IMAGE`: base runtime image
- `USE_TORCH_TPU_REGISTRY`: when set, installs `torch-tpu` from the registry instead of the artifact image
- `TORCH_TPU_VERSION`: explicit `torch-tpu` version override
- `VLLM_SOURCE`: when set, installs `vllm` from the copied local source tree

Example with build args:

```bash
docker build \
  -f docker/Dockerfile \
  --target prod \
  --build-arg USE_TORCH_TPU_REGISTRY=1 \
  --build-arg TORCH_TPU_VERSION=<version> \
  -t prod:latest .
```
