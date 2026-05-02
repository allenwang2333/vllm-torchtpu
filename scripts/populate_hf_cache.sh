#!/bin/bash
# Script to populate GCS cache with HuggingFace models for CI/CD.
# Run this on a machine with internet access (like a TPU VM) and gcloud configured.

set -e

# 1. Configuration via environment variables with defaults
BUCKET="${HF_CACHE_BUCKET:-gs://hf-llm-model-checkpoints/}"
CACHE_DIR="${HF_CACHE_DIR:-/mnt/pd/tmp_hf_cache}"

# Parse options
FORCE=false
if [ "$1" = "--force" ]; then
    FORCE=true
    shift
fi

# Default models if none specified as arguments
DEFAULT_MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-Coder-30B-A3B-Instruct"
)

# Use arguments as models, or fallback to defaults
MODELS=("${@:-${DEFAULT_MODELS[@]}}")

echo "=== Configuration ==="
echo "GCS Bucket:  $BUCKET"
echo "Local Cache: $CACHE_DIR"
echo "Force:        $FORCE"
echo "Models:      ${MODELS[*]}"
echo "======================"

# 2. Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN is not set. Gated models may fail to download,"
    echo "and you may hit lower rate limits."
fi

# 3. Ensure hf or huggingface-cli is available
if [ -f ".venv/bin/huggingface-cli" ]; then
    HF_CLI=".venv/bin/huggingface-cli"
    echo "Using Hugging Face CLI from .venv: $HF_CLI"
elif command -v hf &> /dev/null; then
    HF_CLI="hf"
    echo "Using hf CLI from PATH"
elif command -v huggingface-cli &> /dev/null; then
    HF_CLI="huggingface-cli"
    echo "Using huggingface-cli from PATH"
else
    echo "Hugging Face CLI not found. Installing..."
    curl -LsSf https://hf.co/cli/install.sh | bash
    HF_CLI="hf"
fi

mkdir -p "$CACHE_DIR"

# 4. Download and package models
for model in "${MODELS[@]}"; do
    # Apply naming rule to get HF cache folder name
    folder_name="models--$(echo "$model" | sed 's/\//--/g')"
    tarball_name="${folder_name}.tar"

    # Check if file exists in GCS
    if [ "$FORCE" = "false" ] && /usr/bin/gcloud storage ls "$BUCKET$tarball_name" &> /dev/null; then
        echo "===> Model $model already exists in GCS (tarball: $tarball_name), skipping..."
        continue
    fi

    echo "===> Processing model: $model..."

    echo "Downloading $model..."
    $HF_CLI download "$model" --cache-dir "$CACHE_DIR"

    echo "Tarring cache directory: $folder_name"
    tar -cf "/tmp/$tarball_name" -C "$CACHE_DIR" "$folder_name"

    echo "Uploading to GCS as: $tarball_name"
    # Use full path to gcloud to avoid Python 'imp' module error
    /usr/bin/gcloud storage cp "/tmp/$tarball_name" "$BUCKET$tarball_name"

    # Clean up local tarball to save disk space
    rm "/tmp/$tarball_name"
done

echo "===> Cache population complete!"
