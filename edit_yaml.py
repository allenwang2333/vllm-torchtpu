# SPDX-License-Identifier: Apache-2.0
import os

yaml_path = os.path.expanduser("~/cloud-devkit/recipes/experimental/torchtpu-vllm/dsv4-inference-v7x-8-patemotter.yml")
if not os.path.exists(yaml_path):
    print(f"Error: {yaml_path} does not exist.")
    exit(1)

with open(yaml_path, "r") as f:
    content = f.read()

# Locate the block we want to replace
target_block = """# Recover JAX cache from GCS
mkdir -p /mnt/local-cache/jax_cache
gcloud storage cp -r gs://cloud-tpu-inference-test-patemotter-archive/workspace/jax_cache/* /mnt/local-cache/jax_cache/ || true

mkdir -p /mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash

if [ ! -d "/mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/blobs" ] || [ ! "$(ls -A /mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/blobs 2>/dev/null)" ]; then
  echo "=== [GCS PULL] Syncing large blobs from GCS to local NVMe SSD ==="
  gcloud storage cp -r gs://cloud-tpu-inference-test-patemotter-archive/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/blobs /mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/
fi

if [ ! -d "/mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots" ]; then
  echo "=== [NFS PULL] Copying symlinks and refs from NFS ==="
  cp -a /mnt/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots /mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/
  cp -a /mnt/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/refs /mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/
  if [ -d "/mnt/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/.no_exist" ]; then
    cp -a /mnt/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/.no_exist /mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/
  fi
  echo "=== [NFS PULL] Copy finished ==="
fi"""

# Normalize block formatting in case of YAML escaping backslashes
target_normalized = target_block.replace("\\\n", "").replace("\n", " ").replace("  ", " ")
content_normalized = content.replace("\\\n", "").replace("\n", " ").replace("  ", " ")

# Since it might have YAML backslash escapes, let's do a direct search in the original file
# using string matching with wildcards or direct search.
# Let's inspect if target_block is in content
if target_block in content:
    replacement_block = """# Recover JAX cache from GCS
mkdir -p /mnt/local-cache/jax_cache
gcloud storage cp -r gs://cloud-tpu-inference-test-patemotter-archive/workspace/jax_cache/* /mnt/local-cache/jax_cache/ || true

mkdir -p /mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1

if [ ! -d "/mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1" ] || [ ! "$(ls -A /mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1 2>/dev/null)" ]; then
  echo "=== [GCS PULL] Syncing model snapshots directly from GCS to local SSD ==="
  gcloud storage cp -r gs://cloud-tpu-inference-test-patemotter-archive/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1/* /mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1/
fi"""
    new_content = content.replace(target_block, replacement_block)
    with open(yaml_path, "w") as f:
        f.write(new_content)
    print("YAML file successfully updated.")
else:
    # Let's do a more robust substring check using regex or direct string replacement for blobs/snapshots block
    print("Direct block match failed. Attempting fallback replacement.")
    # Fallback replacement based on the specific keys
    lines = content.splitlines()
    new_lines = []
    skip = False
    added_replacement = False
    
    for line in lines:
        if "# Recover JAX cache from GCS" in line:
            new_lines.append('                # Recover JAX cache from GCS')
            new_lines.append('                mkdir -p /mnt/local-cache/jax_cache')
            new_lines.append('                gcloud storage cp -r gs://cloud-tpu-inference-test-patemotter-archive/workspace/jax_cache/* /mnt/local-cache/jax_cache/ || true')
            new_lines.append('')
            new_lines.append('                mkdir -p /mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1')
            new_lines.append('                if [ ! -d "/mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1" ] || [ ! "$(ls -A /mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1 2>/dev/null)" ]; then')
            new_lines.append('                  echo "=== [GCS PULL] Syncing model snapshots directly from GCS to local SSD ==="')
            new_lines.append('                  gcloud storage cp -r gs://cloud-tpu-inference-test-patemotter-archive/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1/* /mnt/local-cache/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1/')
            new_lines.append('                fi')
            skip = True
            added_replacement = True
            continue
        if skip:
            # We skip until we hit the cache fp4 check
            if 'mkdir -p /mnt/local-cache/dsv4-flash-fp4' in line or 'mkdir -p\n                /mnt/local-cache/dsv4-flash-fp4' in line:
                skip = False
                # Do NOT continue here, so we append the dsv4-flash-fp4 block normally
            else:
                continue
        new_lines.append(line)
        
    if added_replacement:
        with open(yaml_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")
        print("YAML file successfully updated via fallback parser.")
    else:
        print("Error: Could not locate model pull block in YAML file.")
        exit(1)
