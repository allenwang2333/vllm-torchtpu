import sys
import os
import traceback

# Setup paths
sys.path.insert(0, '/home/patemotter_google_com/workspace/dsv4/torchtpu-vllm/src')
sys.path.insert(0, '/home/patemotter_google_com/workspace/dsv4/vllm')

os.environ['VLLM_PLUGINS'] = ''
os.environ['TPU_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import vllm.platforms

try:
    print("Attempting getattr...")
    platform = getattr(vllm.platforms, "current_platform")
    print("Getattr succeeded!", platform)
except Exception as e:
    print("EXCEPTION:")
    traceback.print_exc()
