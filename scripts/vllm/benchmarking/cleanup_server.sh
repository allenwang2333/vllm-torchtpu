#!/bin/bash
# Cleanup vLLM server and ensure port and TPU resources are released.
#
# Note: vLLM V1 uses a multi-process architecture where the API Server (CPU)
# and Engine Core (TPU) run in separate processes. Killing the API Server
# might leave the Engine Core running as an orphan, holding the TPU devices
# (/dev/vfio/*). This script ensures both are cleaned up.

PORT="${PORT:-8000}"

echo "Cleaning up vLLM server on port $PORT..."

# 1. Kill processes by name
pkill -f "vllm" 2>/dev/null || true
sleep 5
pkill -9 -f "vllm" 2>/dev/null || true

# 2. Force kill processes occupying devices
fuser -k -9 /dev/vfio/* 2>/dev/null || true
fuser -k -9 /dev/accel* 2>/dev/null || true

# 3. Remove caches and lock files
rm -f /tmp/libtpu_lockfile* 2>/dev/null || true

# 4. Check if port is still occupied (informational only)
echo "Checking if port $PORT is still occupied..."
if python3 -c "import socket; s=socket.socket(); s.settimeout(1); exit(0 if s.connect_ex((\"localhost\", $PORT)) == 0 else 1)" 2>/dev/null; then
  echo "Warning: Port $PORT is still occupied after cleanup!"
  if command -v lsof >/dev/null; then
    echo "Processes occupying port $PORT:"
    lsof -i:"$PORT" || true
  fi
else
  echo "vLLM server cleanup complete and port $PORT is free."
fi
