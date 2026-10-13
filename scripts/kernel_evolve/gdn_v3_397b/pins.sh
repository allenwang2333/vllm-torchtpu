# The frozen pins of the gdn/v3 Qwen3.5-397B scorer, sourced by score.sh and trace.sh.
GOLDEN=79bd30e0aa156b5fb125d6935463367f89b6b0e6
# vllm-torchtpu CI's image for $GOLDEN (tag ${GOLDEN}-v0.29.0), pinned by digest.
export IMAGE=us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu@sha256:918da23c0c92f2226c7c46cf576c3d9d1258a55f423a24e65a0ccc2b34559762
# pyproject.toml's jax/jaxlib and libtpu pins at $GOLDEN; the pod refuses others.
export EXPECT_JAX=0.10.2 EXPECT_LIBTPU=0.0.47
export WEIGHTS_BUCKET=cloud-devkit WEIGHTS_PREFIX=ckpt/hf/Qwen/Qwen3.5-397B-A17B-FP8
SCORER_FLAGS="--tolerance 1.5 --error-stat mean --reps 3 --warmup 1"
