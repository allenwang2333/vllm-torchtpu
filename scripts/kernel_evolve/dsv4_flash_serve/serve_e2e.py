"""Served-model scorer for DeepSeek-V4-Flash, DP8 + EP on one tpu7x 2x2x1 node.

Each arm boots `vllm serve` from its own tree with the exact server command and
environment that `scripts/vllm/benchmarking/run_benchmarks.sh --config
deepseek-v4-flash-dp8-ep-in1024-out8192-c2048` builds at the golden commit,
runs the same `vllm bench serve` sweep against it, and reports output-token
throughput as a latency (microseconds per output token), so control / arm > 1
means faster.

It is a `perfkit scorer run` benchmark (`perfkit/scorer_arm.py` defines the
interface) and is meant to run in a `perfkit cluster score` pod, where each arm
root holds only `src/vllm_torchtpu`. The serve command and the client command
live here, not in the arm, so a candidate can change the code behind the server
and nothing about what is served or how it is measured.

Correctness half: a served model has no fp32 reference. Each arm sends a fixed
prompt suite serially, greedy, once with top-5 logprobs and once without, and
reports a digest of the generated text as `checksum`; equal digests mean the
same tokens. `worst_error` is 0.0, or 1.0 when the sweep lost requests or the
suite could not be read, which fails any tolerance below 1.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
PORT = 8000
CONFIG = "deepseek-v4-flash-dp8-ep-in1024-out8192-c2048"

#: One config: the recipe's single shape. `sweeps` timed sweeps run against one
#: boot; the score is their median.
CONFIGS = {
    CONFIG: {"isl": 1024, "osl": 8192, "concurrency": 2048, "num_prompts": 2048,
             "sweeps": 1},
}

#: The checkpoint, copied once per pod from the cluster's GCS mirror into the
#: Hugging Face hub cache layout, so the server resolves the bare model id offline.
CHECKPOINT_URI = "gs://cloud-devkit/ckpt/hf/deepseek-ai/DeepSeek-V4-Flash-0731"
HF_HOME = Path("/dev/shm/hf")
HUB_REVISION = "0000000000000000000000000000000000000731"

#: Server environment, as run_benchmarks.sh exports it for this config
#: (deepseek-v4-flash-dp8-ep.sh plus the harness's own exports).
SERVER_ENV = {
    "NEW_MODEL_DESIGN": "1",
    "MODEL_IMPL_TYPE": "vllm",
    "MOE_REQUANTIZE_WEIGHT_DTYPE": "fp4",
    "MOE_REQUANTIZE_BLOCK_SIZE": "512",
    "TPU_ROPE_CACHE_ROW_MAJOR": "1",
    "TPU_MOE_HASH_TABLE_ROW_MAJOR": "1",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
    "VLLM_USE_AOT_COMPILE": "0",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
    "TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE": "1",
    "CAPTURE_PROFILE": "1",
    "USE_PHASED_PROFILER": "1",
    "PYTHONUNBUFFERED": "1",
}

#: Variables the recipe runs without; any inherited value is dropped.
UNSET_ENV = ("ASYNC_SCHEDULING", "ATTENTION_BACKEND", "BENCHMARK_EXTRA_BODY",
             "BENCHMARK_TEMPERATURE", "BENCHMARK_WARMUP_RUNS", "DATASET_NAME",
             "DATASET_PATH", "ENABLE_PREFIX_CACHING", "EXTRA_BENCH_ARGS",
             "EXTRA_LM_EVAL_MODEL_ARGS", "EXTRA_SERVE_ARGS", "LM_EVAL_GEN_KWARGS",
             "MODEL_URI", "PORT", "PROFILE_NUM_PROMPTS", "TOKENIZER", "XLA_FLAGS")

READY_TIMEOUT_S = 2700
SWEEP_TIMEOUT_S = 2400

#: Fixed prompts for the output fingerprint: sent one at a time, greedy.
SUITE = (
    "The capital of France is",
    "Write a Python function that returns the n-th Fibonacci number.\n\ndef fib(n):",
    "Q: A train travels 120 km in 1.5 hours. What is its average speed in km/h?\nA:",
    "Translate to German: The weather is nice today, so we will walk to the station.",
    "List the first ten prime numbers, separated by commas:",
    "Explain in one paragraph why the sky appears blue during the day.",
    "def quicksort(arr):\n    \"\"\"Sort a list of integers.\"\"\"\n",
    "In 1969, the first humans to walk on the Moon were",
)
SUITE_TOKENS = 64


def say(msg: str) -> None:
    print(f"dsv4_flash_serve: {msg}", file=sys.stderr, flush=True)


def server_cmd(profile_dir: Path) -> list[str]:
    """The `[cmd]` line run_benchmarks.sh prints for this config, as argv."""
    gen = json.dumps({"do_sample": False, "temperature": 0.0, "model": MODEL},
                     separators=(",", ":"))
    return ["vllm", "serve", MODEL, "--tensor-parallel-size=1", "--data-parallel-size=8",
            "--max-model-len=9216", "--max-num-batched-tokens=1024", "--max-num-seqs=256",
            "--port", str(PORT), "--async-scheduling", "--no-enable-prefix-caching",
            "--gpu-memory-utilization=0.9", "--kv-cache-dtype=fp8_e4m3",
            "--enable-expert-parallel", "--quantization", "deepseek_v4_fp8",
            "--attention-backend", "CUSTOM", "--api-server-count=1", "--trust-remote-code",
            "--generation-config=vllm", f"--override-generation-config={gen}",
            "--profiler-config.profiler", "torch",
            "--profiler-config.torch_profiler_dir", str(profile_dir),
            "--profiler-config.ignore_frontend", "true"]


def client_cmd(cfg: dict, result: Path) -> list[str]:
    """The `vllm bench serve` line run_benchmarks.sh runs for the measured sweep."""
    return ["vllm", "bench", "serve", "--backend", "vllm", "--model", MODEL,
            "--trust-remote-code", "--host", "localhost", "--port", str(PORT),
            "--dataset-name", "random", "--random-input-len", str(cfg["isl"]),
            "--random-output-len", str(cfg["osl"]), "--random-range-ratio", "0.0",
            "--num-prompts", str(cfg["num_prompts"]),
            "--max-concurrency", str(cfg["concurrency"]), "--request-rate", "inf",
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--save-result", "--ignore-eos",
            "--result-filename", str(result), "--temperature", "0", "--seed", "42"]


class Handle:
    def __init__(self, root: Path, tree: str, scratch: Path):
        self.root = root
        self.tree = tree
        self.scratch = scratch


def _scratch(arm: str) -> Path:
    """Per-arm files, outside the arm root: anything written there changes the
    tree the driver hashes after the arm exits."""
    base = Path(os.environ.get("WORK_DIR") or os.environ.get("TMPDIR") or "/tmp")
    job = os.environ.get("PERFKIT_JOB_ID", "job").replace("/", "-")
    d = base / "dsv4_flash_serve" / job / arm
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def _copy_checkpoint() -> None:
    """Copy the checkpoint into the hub cache once per pod; later arms reuse it."""
    hub = HF_HOME / "hub" / ("models--" + MODEL.replace("/", "--"))
    snap = hub / "snapshots" / HUB_REVISION
    done = hub / ".complete"
    if done.exists():
        say(f"checkpoint already at {snap}")
        return
    snap.mkdir(parents=True, exist_ok=True)
    (hub / "refs").mkdir(exist_ok=True)
    (hub / "refs" / "main").write_text(HUB_REVISION)
    t0 = time.time()
    if shutil.which("gcloud"):
        cmd = ["gcloud", "storage", "rsync", "-r", CHECKPOINT_URI, str(snap)]
    else:
        cmd = ["gsutil", "-q", "-m", "rsync", "-r", CHECKPOINT_URI, str(snap)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"checkpoint copy failed: {r.stderr[-2000:]}")
    size = sum(f.stat().st_size for f in snap.rglob("*") if f.is_file())
    say(f"checkpoint copied to {snap}: {size / 1e9:.1f} GB in {time.time() - t0:.0f}s")
    done.write_text("ok\n")


def setup():
    from perfkit import scorer as pk_scorer

    root = Path.cwd().resolve()
    tree = pk_scorer.git_tree(root)
    arm = os.environ.get("PERFKIT_ARM", "arm")
    say(f"arm {arm} root {root} tree {tree or '(not a git tree)'}")
    _copy_checkpoint()
    return Handle(root, tree, _scratch(arm))


def origin(handle) -> str:
    """The vllm_torchtpu the server will import, resolved on this process's path
    (the arm's src first) without importing it: an import here would open the TPU
    the server needs."""
    spec = importlib.util.find_spec("vllm_torchtpu")
    return spec.origin if spec and spec.origin else ""


def _env(handle) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in UNSET_ENV}
    env.update(SERVER_ENV)
    home = handle.scratch / "home"
    # Short: vLLM puts its zmq IPC sockets here, and a socket path is capped at
    # 107 bytes.
    tag = hashlib.sha1(str(handle.scratch).encode()).hexdigest()[:8]
    tmp = Path(os.environ.get("TMPDIR") or "/tmp") / f"a{tag}"
    for d in (home, tmp):
        d.mkdir(parents=True, exist_ok=True)
    # Every compile cache starts empty in every arm, as it does in the recipe's
    # fresh container, and no arm can load another arm's executables.
    env.update({
        "HOME": str(home), "TMPDIR": str(tmp), "XDG_CACHE_HOME": str(home / ".cache"),
        "TORCH_TPU_CACHE_DIR": str(home / "torch_tpu_cache"),
        "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": str(home / "torch_tpu_cache"),
        "HF_HOME": str(HF_HOME), "HF_HUB_OFFLINE": "1",
    })
    return env


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    for pat in ("vllm serve", "VLLM::", "vllm.entrypoints"):
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
    time.sleep(5)
    for f in Path("/tmp").glob("libtpu_lockfile*"):
        f.unlink(missing_ok=True)


def _complete(prompt: str, logprobs: bool) -> str:
    body = {"model": MODEL, "prompt": prompt, "max_tokens": SUITE_TOKENS,
            "temperature": 0.0, "seed": 0}
    if logprobs:
        body["logprobs"] = 5
    req = urllib.request.Request(f"http://localhost:{PORT}/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    choice = out["choices"][0]
    text = choice["text"]
    if logprobs:
        text += "\x00" + json.dumps(choice["logprobs"]["tokens"])
    return text


def _fingerprint(art: Path) -> str:
    """Digest of the suite's greedy outputs, with and without logprobs."""
    outs = {"logprobs": [_complete(p, True) for p in SUITE],
            "nologprobs": [_complete(p, False) for p in SUITE]}
    (art / "fingerprint.json").write_text(json.dumps(outs, indent=1))
    return "sha256:" + hashlib.sha256(json.dumps(outs, sort_keys=True).encode()).hexdigest()


def _sweep(cfg: dict, env: dict, art: Path, k: int) -> dict:
    result = art / f"sweep{k}.json"
    cmd = client_cmd(cfg, result)
    say(f"sweep {k}: {shlex.join(cmd)}")
    t0 = time.time()
    with open(art / f"sweep{k}.log", "w") as log:
        rc = subprocess.call(cmd, env=env, cwd=art, stdout=log, stderr=subprocess.STDOUT,
                             timeout=SWEEP_TIMEOUT_S)
    if rc != 0 or not result.exists():
        raise RuntimeError(f"sweep {k} exited {rc}; see {art / f'sweep{k}.log'}")
    d = json.loads(result.read_text())
    say(f"sweep {k}: {time.time() - t0:.0f}s, {d.get('completed')} completed, "
        f"{d.get('output_throughput', 0):.1f} output tok/s, median TTFT "
        f"{d.get('median_ttft_ms', 0):.0f} ms, median TPOT {d.get('median_tpot_ms', 0):.2f} ms")
    return d


def measure(handle, name, cfg):
    art = handle.scratch / name
    art.mkdir(parents=True, exist_ok=True)
    env = _env(handle)
    probe = subprocess.run([sys.executable, "-c", "import importlib.util as u; "
                            "print(u.find_spec('vllm_torchtpu').origin)"],
                           env=env, cwd=art, capture_output=True, text=True)
    served = probe.stdout.strip()
    if not served.startswith(str(handle.root) + os.sep):
        raise RuntimeError(f"the server would import vllm_torchtpu from {served or '?'}, "
                           f"not from this arm's root {handle.root}")
    cmd = server_cmd(art / "profile")
    say(f"[cmd] {shlex.join(cmd)}")
    t0 = time.time()
    log = open(art / "server.log", "w")
    proc = subprocess.Popen(cmd, env=env, cwd=art, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        while not _healthy():
            if proc.poll() is not None:
                tail = (art / "server.log").read_text()[-3000:]
                raise RuntimeError(f"server exited {proc.returncode} during startup:\n{tail}")
            if time.time() - t0 > READY_TIMEOUT_S:
                raise RuntimeError(f"server not ready after {READY_TIMEOUT_S}s")
            time.sleep(5)
        boot_s = time.time() - t0
        say(f"server ready in {boot_s:.0f}s")
        sweeps = [_sweep(cfg, env, art, k) for k in range(1, int(cfg["sweeps"]) + 1)]
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode} during the sweeps")
        try:
            checksum = _fingerprint(art)
        except (OSError, ValueError, KeyError) as e:
            say(f"fingerprint suite failed: {e}")
            checksum = ""
    finally:
        _stop(proc)
        log.close()

    def med(key):
        return statistics.median(float(s.get(key) or 0.0) for s in sweeps)

    tput = med("output_throughput")
    completed = min(int(s.get("completed") or 0) for s in sweeps)
    lost = completed < int(cfg["num_prompts"])
    if lost or not checksum:
        say(f"marking incorrect: completed {completed} of {cfg['num_prompts']}, "
            f"fingerprint {'read' if checksum else 'missing'}")
    metrics = {k: med(k) for k in ("median_ttft_ms", "median_tpot_ms", "p99_ttft_ms",
                                   "p99_tpot_ms", "output_throughput", "duration")}
    metrics.update({"completed": completed, "boot_s": boot_s})
    if len(sweeps) > 1:
        t = [float(s["output_throughput"]) for s in sweeps]
        metrics["sweep_spread"] = (max(t) - min(t)) / tput
    return {
        "latency_us": 1e6 / tput if tput > 0 else 0.0,
        "worst_error": 1.0 if (lost or not checksum) else 0.0,
        "checksum": checksum,
        "metrics": metrics,
        "dispatched": f"serve/{handle.tree}/{name}",
        "binary_hash": "sha256:" + (handle.tree or "unknown"),
    }
