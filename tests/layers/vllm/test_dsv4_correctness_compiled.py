import torch
import pytest
import numpy as np
import os
import sys
import multiprocessing
import traceback
import time
import portpicker
import torch.distributed as dist
import torch_tpu
from torch_tpu._internal.utils import hardware
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Ensure src/ is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

class DummyAttnSinkModule(torch.nn.Module):
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
        # Global shape [64] with values 0 to 63
        self.attn_sink = torch.nn.Parameter(torch.arange(64, dtype=torch.bfloat16))
        # Shard the parameter with P("model")
        self.attn_sink.data.sharding = NamedSharding(mesh, P("model"))
        
    def forward(self):
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        # Slicing using rank
        local_sink = self.attn_sink[rank * 8 : (rank + 1) * 8].clone()
        return local_sink


def run_tp8_worker(rank, group_env):
    try:
        # Set env vars
        os.environ.update(group_env)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        
        # Initialize distributed group
        init_method = f"tcp://{group_env['MASTER_ADDR']}:{group_env['MASTER_PORT']}"
        dist.init_process_group(
            backend="tpu_dist",
            init_method=init_method,
            rank=rank,
            world_size=8,
        )
        
        # Touch TPU and set default device
        torch.empty((1,), device="tpu")
        torch.set_default_device("tpu")
        device = torch.device("tpu")
        
        # Initialize JAX Device Mesh over local devices
        local_devices = jax.devices()
        devices_arr = np.array(local_devices).reshape(8)
        mesh = Mesh(devices_arr, axis_names=("model",))
        
        # Instantiate Module
        module = DummyAttnSinkModule(mesh).to(device)
        
        # 1. Run Eager Mode
        sink_eager = module()
        
        # 2. Run Compiled Mode
        compiled_fn = torch.compile(module, backend="tpu")
        sink_compiled = compiled_fn()
        
        # Fetch values to CPU for verification
        eager_cpu = sink_eager.cpu()
        compiled_cpu = sink_compiled.cpu()
        
        # Print results on each rank
        print(f"[RANK {rank}] Eager: {eager_cpu.tolist()} | Compiled: {compiled_cpu.tolist()}", flush=True)
        
        # Verify correctness
        # Eager values must match expected slice: [rank*8 : (rank+1)*8]
        expected = list(range(rank * 8, (rank + 1) * 8))
        assert eager_cpu.tolist() == expected, f"Rank {rank} Eager values {eager_cpu.tolist()} do not match expected {expected}!"
        
        # Compiled values must match eager values exactly
        assert compiled_cpu.tolist() == eager_cpu.tolist(), f"Rank {rank} Compiled values {compiled_cpu.tolist()} do not match Eager {eager_cpu.tolist()}!"
        
        if rank == 0:
            print("\n--- Attention Sink Correctness Comparison (Rank 0) ---", flush=True)
            print("Eager vs Compiled match successfully across all ranks!", flush=True)
            
    except BaseException as e:
        print(f"[WORKER ERROR] Rank {rank} failed with exception:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_attn_sink_suite():
    print("Preparing TP=8 distributed execution environment...", flush=True)
    
    # 1. Fetch TPU topology dynamically
    topology = hardware.get_tpu_topology(8)
    if topology is None:
        pytest.skip("Could not query TPU topology. Skipping distributed TPU-only test.")
    print(f"Detected TPU topology: {topology}", flush=True)
    
    # 2. Allocate ports and set up env
    sb_ports = [portpicker.pick_unused_port() for _ in range(8)]
    slicebuilder_addresses = ",".join(f"localhost:{p}" for p in sb_ports)
    master_port = portpicker.pick_unused_port()
    
    group_env = {
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "8",
        "LOCAL_RANK_BASE": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(master_port),
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": slicebuilder_addresses,
        "TORCH_TPU_TOPOLOGY": topology,
        "TORCH_TPU_XPROF_SESSION_ID": str(time.time_ns()),
    }
    
    # 3. Spawn child processes
    ctx = multiprocessing.get_context("spawn")
    processes = []
    for r in range(8):
        p = ctx.Process(target=run_tp8_worker, args=(r, group_env), name=f"tp8-worker-{r}")
        p.start()
        processes.append(p)
        
    # 4. Wait for all processes to complete
    success = True
    for r, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            print(f"Process {r} failed with exit code {p.exitcode}", file=sys.stderr, flush=True)
            success = False
            
    assert success, "One or more TP=8 worker processes failed!"
    print("All TP=8 worker processes completed successfully!", flush=True)

if __name__ == "__main__":
    test_attn_sink_suite()
