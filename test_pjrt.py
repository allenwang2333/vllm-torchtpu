import sys

import torch

print("Testing torch.tpu.Stream()...")
try:
    stream = torch.tpu.Stream()
    print("SUCCESSFULLY CREATED TPU STREAM ON CPU!!!", file=sys.stderr)
except Exception as e:
    print(f"FAILED to create TPU stream: {e}", file=sys.stderr)
