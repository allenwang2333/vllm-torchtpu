import os
os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["MODEL_IMPL_TYPE"] = "vllm"
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    os.getenv("DSV4_MODEL_PATH", "/home/patemotter_google_com/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1"),
    trust_remote_code=True
)
print("CHAT TEMPLATE:")
print(tokenizer.chat_template)
