import os
from transformers import AutoTokenizer

model_path = "/mnt/workspace/dsv4-flash-4layer-real-fp8"
if not os.path.exists(model_path):
    model_path = "deepseek-ai/DeepSeek-V4-Flash"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

prompt = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer.encode(prompt)
print(f"Prompt: {prompt!r}")
print(f"Prompt token IDs: {input_ids}")
print(f"Decoded Prompt: {[tokenizer.decode([i]) for i in input_ids]}")

test_ids = [6560, 3900, 23097, 1493, 53457, 19834, 25622, 69146, 59566, 33031]
print(f"Test token IDs: {test_ids}")
print(f"Decoded Test tokens: {[tokenizer.decode([i]) for i in test_ids]}")
print(f"Decoded Test string: {tokenizer.decode(test_ids)}")
