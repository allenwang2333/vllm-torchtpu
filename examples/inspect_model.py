import os
from vllm.entrypoints.llm import LLM

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.getenv("DSV4_MODEL_PATH", "/mnt/workspace/dsv4-flash-4layer-real-fp8")
    
    print("Initializing LLM...")
    llm = LLM(
        model=model_path,
        quantization="deepseek_v4_fp8",
        load_format="auto",
        tensor_parallel_size=8,
        max_model_len=128,
        max_num_batched_tokens=128,
        trust_remote_code=True,
        kv_cache_dtype="fp8",
        gpu_memory_util_opts=0.45,
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        enable_expert_parallel=True,
        safetensors_load_strategy="lazy",
        additional_config={
            "compilation_config": {
                "compile_sizes": [16, 128],
            },
            "sharding": {
                "sharding_strategy": {
                    "enable_dp_attention": True,
                    "expert_parallelism": 8,
                }
            }
        },
    )
    
    # Access the model parameters of layer 0 ffn experts
    model_executor = llm.llm_engine.engine_core.model_executor
    # Print on rank 0
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print("\n=== Model Parameters for layer 0 FFN experts ===")
        # Get the worker model
        worker = model_executor.collective_rpc("get_worker")
        print("Worker:", worker)
        
if __name__ == "__main__":
    main()
