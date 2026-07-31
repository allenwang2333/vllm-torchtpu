import json
import subprocess
import sys
import time
import requests

cmd = [
    sys.executable,
    "-m",
    "vllm.entrypoints.cli.main",
    "serve",
    "/mnt/pds/xfgu-pd-v7x/kimik26/Kimi-K3-test",
    "--tensor-parallel-size",
    "8",
    "--port",
    "8000",
    "--trust-remote-code",
    "--gpu-memory-utilization",
    "0.4",
    "--compilation-config",
    '{"compile_sizes": [16], "compile_ranges_endpoints": [16]}',
]

proc = subprocess.Popen(cmd)

server_url = "http://localhost:8000"
health_url = f"{server_url}/health"
completion_url = f"{server_url}/v1/completions"

start_time = time.time()
print("Waiting for server to become healthy...")
while True:
    try:
        r = requests.get(health_url, timeout=2)
        if r.status_code == 200:
            elapsed = time.time() - start_time
            print(f"Server is HEALTHY after {elapsed:.1f}s! "
                  "Sending completion request...")
            break
    except Exception:
        pass

    if proc.poll() is not None:
        print(f"ERROR: vllm serve exited prematurely with code "
              f"{proc.returncode}!")
        sys.exit(1)

    time.sleep(5)

payload = {
    "model": "/mnt/pds/xfgu-pd-v7x/kimik26/Kimi-K3-test",
    "prompt": "Hello, this is a test prompt.",
    "max_tokens": 10,
    "temperature": 0.0,
}

try:
    res = requests.post(completion_url, json=payload, timeout=30)
    print("=== HTTP COMPLETED ===")
    print("Status Code:", res.status_code)
    print("Response JSON:", json.dumps(res.json(), indent=2))
    print("=== SERVER TEST PASSED ===")
finally:
    print("Shutting down server...")
    proc.terminate()
    proc.wait()
