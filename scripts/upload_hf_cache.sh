#!/bin/bash
# ==============================================================================
# Script to upload local HuggingFace cache to GCS bucket for CI/CD use.
#
# This script can be used by developers to manually update or populate the
# model checkpoints stored in the GCS bucket used by CI tests.
# It uses 'gcloud storage rsync' to only upload new or modified files.
#
# Usage:
#   ./scripts/upload_hf_cache.sh <local_hf_cache_dir> [gcs_bucket]
#
# Example:
#   ./scripts/upload_hf_cache.sh ~/.cache/huggingface/hub
# ==============================================================================
set -e

LOCAL_CACHE_DIR="$1"
BUCKET="${2:-gs://tpu-inference-hf-llm-model-checkpoints/}"

if [ -z "$LOCAL_CACHE_DIR" ]; then
    echo "Usage: $0 <local_hf_cache_dir> [gcs_bucket]"
    echo "Example: $0 /home/user/.cache/huggingface/hub"
    exit 1
fi

# Models used in tests and perf workflows
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    "Qwen/Qwen3-Coder-480B-A35B-Instruct"
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
    "Qwen/Qwen3.5-397B-A17B"
)

echo "=== Configuration ==="
echo "Local Cache Dir: $LOCAL_CACHE_DIR"
echo "GCS Bucket:      $BUCKET"
echo "======================"

for model in "${MODELS[@]}"; do
    # Convert "Qwen/Qwen3-0.6B" to "models--Qwen--Qwen3-0.6B"
    folder_name="models--$(echo "$model" | sed 's/\//--/g')"
    local_folder="$LOCAL_CACHE_DIR/$folder_name"

    if [ -d "$local_folder" ]; then
        echo "===> Uploading $model from $local_folder to $BUCKET..."
        gcloud storage rsync -r "$local_folder" "$BUCKET/$folder_name"
    else
        echo "===> Warning: Model folder $folder_name not found in $LOCAL_CACHE_DIR. Skipping."
    fi
done

echo "Done!"
