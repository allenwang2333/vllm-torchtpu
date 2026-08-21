# Dockerfile for vllm-torchtpu.
# This Dockerfile builds images for running vLLM on Google Cloud TPUs.
# It pre-installs all dependencies from pyproject.toml (including torch-tpu) in the base variant.

ARG BASE_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu-base:nightly-latest"

# Variant 1: The Runtime (Base Image with pre-installed dependencies)
# This variant provides the core runtime environment with all external dependencies
# (including torch-tpu) pre-installed.
FROM $BASE_IMAGE AS base

# Install curl for health checks (minimal for base)
RUN apt-get update && apt-get install -y curl git build-essential gcc g++ cmake ninja-build && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/vllm-torchtpu

# Copy pyproject.toml to extract dependencies and read settings
COPY pyproject.toml .

# Which upstream vLLM commit to build against.
#
# Empty -- the default, and what every normal build uses -- keeps the last
# known good (LKG) ref already pinned in pyproject.toml. Only the integration
# pipeline passes a ref here, so it can test vLLM main HEAD without committing
# a new pin. vllm_pin.sh rewrites pyproject.toml in-image so every
# `uv pip install` below resolves the same vLLM.
ARG VLLM_COMMIT_HASH=""
COPY docker/vllm_pin.sh docker/vllm_pin.sh
RUN bash docker/vllm_pin.sh write "${VLLM_COMMIT_HASH}"

# Install all dependencies from pyproject.toml. We mount the gcloud token secret
# to authenticate with the private torch-tpu registry, falling back to PyPI for other packages.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=gcloud_token \
    set -eu; \
    TOKEN=""; \
    if [ -f /run/secrets/gcloud_token ]; then TOKEN="$(cat /run/secrets/gcloud_token)"; fi; \
    if [ -z "${TOKEN}" ]; then echo "Missing BuildKit secret gcloud_token for registry mode" >&2; exit 1; fi; \
    export UV_INDEX_TORCH_TPU_REGISTRY_USERNAME=oauth2accesstoken; \
    export UV_INDEX_TORCH_TPU_REGISTRY_PASSWORD="${TOKEN}"; \
    export CXX=g++; \
    export CC=gcc; \
    export VLLM_TARGET_DEVICE=empty; \
    uv pip install --system \
      --index https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/ \
      --extra-index-url https://pypi.org/simple \
      --index-strategy unsafe-best-match \
      -r pyproject.toml

# Record how to identify the vLLM that actually landed in the image, so CI can
# prove the image holds the vLLM it asked for instead of trusting the tag
# (verify_image_vllm). Per PEP 610 an installer records a git dependency's
# origin in direct_url.json, but whether it keeps the requested ref, the
# resolved commit, or both is installer-specific -- so write every identifier,
# one per line, and let the verifier match any of them.
RUN python3 -c "import importlib.metadata as m, json; d = m.distribution('vllm'); r = d.read_text('direct_url.json'); i = json.loads(r).get('vcs_info', {}) if r else {}; print(chr(10).join(filter(None, [i.get('requested_revision'), i.get('commit_id'), d.version])))" \
      > /etc/vllm-revision && cat /etc/vllm-revision


# Variant 2: The CI Environment (CI/CD Variant)
# This variant builds on top of the runtime base, installing additional packages
# and testing tools needed for CI/CD pipelines.
FROM base AS ci

# Copy bktec binary from Buildkite Test Engine client image
COPY --from=buildkite/test-engine-client /usr/local/bin/bktec /usr/local/bin/bktec

# Install tools needed for CI (git, gcloud, lsof, fuser)
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg git lsof psmisc lsb-release && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" | tee -a /etc/apt/sources.list.d/gcsfuse.list && \
    apt-get update && apt-get install -y google-cloud-cli gcsfuse && \
    rm -rf /var/lib/apt/lists/*

# Clear PYTHONPATH inherited from base image if any
ENV PYTHONPATH=""

WORKDIR /workspace/vllm-torchtpu

# Mark mounted directory as safe for git and maintain backward compatibility
RUN git config --global --add safe.directory /workspace/vllm-torchtpu && \
    git config --global --add safe.directory /root/torchtpu-vllm && \
    mkdir -p /root /workspace && \
    ln -sfnT /workspace/vllm-torchtpu /root/torchtpu-vllm && \
    ln -sfnT /workspace/vllm-torchtpu /workspace/torchtpu-vllm

# Note: we **intentionally** set VLLM_TARGET_DEVICE="empty", see README.
ENV VLLM_TARGET_DEVICE="empty"

# Variant 3: Developer Image (Dev Variant with test/benchmark tools)
# This variant contains all source code and is intended for active development.
# It installs testing and benchmarking tools in editable mode.
FROM ci AS dev
COPY . .
# `COPY . .` restores the repo's own pyproject.toml, undoing the base-stage pin
# rewrite. Re-apply it, or the editable install below would resolve vLLM back
# to the committed LKG ref and quietly downgrade an upstream-HEAD build.
# No-op for normal builds, where VLLM_COMMIT_HASH is empty.
ARG VLLM_COMMIT_HASH=""
RUN bash docker/vllm_pin.sh write "${VLLM_COMMIT_HASH}"
# Scoped to the vllm_torchtpu dist: the unscoped SETUPTOOLS_SCM_PRETEND_VERSION
# also applies to any other scm-versioned package built in this step. With a
# cold uv cache, resolving ".[...]" rebuilds metadata for the vllm git
# dependency, which then gets stamped 0.0.0 and reinstalled over the correctly
# versioned wheel from the base stage.
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.0.0 \
    uv pip install --system -e ".[test,benchmarking]"

# Install vLLM from source if requested.
# If VLLM_SOURCE is provided, we build it in editable mode.
ARG VLLM_SOURCE
# Copy vllm source (might be empty if not provided)
COPY vllm /workspace/vllm-torchtpu/vllm

# Install vllm from source if requested.
# If VLLM_SOURCE is provided, we build it in editable mode.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$VLLM_SOURCE" ]; then \
      uv pip install --system --no-deps \
        -e /workspace/vllm-torchtpu/vllm ; \
    fi

ENV PYTHONPATH="/workspace/vllm-torchtpu/src:/workspace/vllm-torchtpu"
CMD ["/bin/bash"]


# Variant 4: Production/Full Image (Production/Minimal Variant)
# This variant copies all source code and installs the package regularly
# without extra dev/testing tools, starting directly from the runtime base.
FROM base AS prod
WORKDIR /workspace/vllm-torchtpu
COPY . .
# See the dev stage: `COPY . .` undoes the base-stage pin rewrite.
ARG VLLM_COMMIT_HASH=""
RUN bash docker/vllm_pin.sh write "${VLLM_COMMIT_HASH}"
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.0.0 \
    uv pip install --system .

RUN mkdir -p /root /workspace && \
    ln -sfnT /workspace/vllm-torchtpu /root/torchtpu-vllm && \
    ln -sfnT /workspace/vllm-torchtpu /workspace/torchtpu-vllm
ENV PYTHONPATH="/workspace/vllm-torchtpu/src:/workspace/vllm-torchtpu"
CMD ["/bin/bash"]
