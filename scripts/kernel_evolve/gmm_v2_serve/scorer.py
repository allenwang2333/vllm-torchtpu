#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Served-model fitness function for the gmm_v2 (kernels/megablox) campaign.

One cdk job is one measurement: a single tpu7x 2x2x1 pod that boots `vllm serve`
once per arm, from that arm's own tree, with the serve command and environment of
the recipe of record (Qwen3.5-397B-A17B-FP8, TP8 + EP, MTP k=3), and drives it
with a fixed closed-loop load. The pod prints ONE JSON value in the campaign
scorer contract; `submit` prints that value verbatim on stdout and everything
human-readable on stderr.

On the host (needs cdk, gcloud and git; run from the repository):

    scorer.py submit --arm golden=<rev>:control --arm golden_aa=<rev> --arm cand=<rev>
    scorer.py submit ... --no-wait          # prints the job id
    scorer.py wait <job>; scorer.py fetch <job>
    scorer.py tree <dir>                    # the tree hash this file reports for <dir>

In the pod (the recipe runs this):

    scorer.py run --stage /root/gmmv2_score --out $CDK_OUTPUT_DIR/gmmv2_score

Per arm and config:
  latency_us   wall-clock microseconds per output token over one repetition of
               the config's fixed requests, median of REPS.
  correct      every known-answer probe right (short questions and needles in
               long uncached contexts), every load request exactly as long as
               the config asks. The logprob tolerance is the campaign's, applied
               to worst_error by the harness.
  worst_error  the worst prompt's next-token logprob error against the control
               arm of the same job: for each reference prompt whose length puts
               the forward pass in this config's token bucket, the largest
               |logprob| difference over the control's top-5 tokens (the
               control itself reports 0).
  checksum     sha256 of every probe's greedy reply and every reference
               prompt's top token.
  dispatched   digest of every Mosaic kernel the arm's server lowered (name and
               serialized config, which carries the kernel's lowered IR) and of
               the LIBTPU_INIT_ARGS it was lowered under. A candidate that leaves
               both unchanged has not changed what runs on the device.
  declined     the arm's server could not be measured at all; latency_us is 0.
Per arm:
  tree         git tree hash of the arm's whole tree as deployed in the pod.
  binary_hash  digest of the arm's Python sources the server processes imported.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import gzip
import hashlib
import http.client
import json
import os
import re
import secrets
import shutil
import signal
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import zoneinfo
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOOK = HERE / "hook" / "sitecustomize.py"
#: The hook is part of this fitness function; `run` refuses any other bytes.
HOOK_SHA256 = "8609bb60f808d65269ad3e32e26dfbb61a53b4f16cdb57071f10305663cf3ccb"

KERNEL_REL = "src/vllm_torchtpu/kernels/megablox"
MARKER = "GMMV2_MARKER"

# --- the server, as the recipe of record runs it ---------------------------------

#: The image the job of record pulled, by the digest its tag held at the time.
IMAGE = (
    "us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/"
    "vllm-torchtpu-caojx_google_com"
    "@sha256:78108de22c693ad17d2547c82076f34863752feda979e670d5a2230993e24cb2"
)
DEPLOY_ROOT = Path("/root/cloud-devkit/vllm-torchtpu")
CACHE_ROOT = Path("/root/vllm-torchtpu-cache")
WORK_ROOT = Path("/root/.cache/huggingface/gmmv2_score")
MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
MODEL_DIR = Path("/root/.cache/huggingface") / MODEL
CKPT_URI = f"gs://cloud-devkit/ckpt/hf/{MODEL}"
PORT = 8000

#: As the recipe's shell assembles it: the defaults, the hierarchical all-reduce
#: (TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS=1024 -> 1024*4096*2 bytes) and the
#: async row gather (TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER=1).
LIBTPU_INIT_ARGS = (
    " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
    " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"
    " --xla_tpu_ars_combiner_threshold_in_bytes=0"
    " --xla_tpu_enable_async_collective_merger=false"
    " --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true"
    " --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=8388608"
    " --xla_enable_async_all_gather=true"
    " --xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
    " --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144"
)

SERVER_ENV = {
    "BASE_RESULT_DIR": "/root/cloud-devkit/results",
    "CACHE_DIR": "/root/vllm-torchtpu-cache",
    "CLOUD_TPU_TASK_ID": "0",
    "DP_SCHED_BATCH_PREFILL": "1",
    "HF_HOME": "/vllm-cache",
    "HF_HUB_OFFLINE": "1",
    "JAX_COMPILATION_CACHE_DIR": "/root/vllm-torchtpu-cache/jax_cache",
    "JAX_COMPILER_CACHE_DIR": "/root/vllm-torchtpu-cache/jax_cache",
    "LIBTPU_INIT_ARGS": LIBTPU_INIT_ARGS,
    "MEGASCALE_COORDINATOR_ADDRESS": "",
    "MEGASCALE_NUM_SLICES": "",
    "MEGASCALE_PORT": "",
    "MEGASCALE_SLICE_ID": "",
    "MODEL_IMPL_TYPE": "vllm",
    "MODEL_NAME": MODEL,
    "MOE_FUSED_EP_KERNEL_MIN_TOKENS": "1024",
    "ONEHOT_MOE_PERMUTE_THRESHOLD": "2048",
    "RAGGED_GATHER_REDUCE_VERSION": "v3",
    "RAIDEN_NUMA_POLICY": "preferred",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "TMPDIR": "/root/vllm-torchtpu-cache/tmp",
    "TORCH_TPU_CACHE_DIR": "/root/vllm-torchtpu-cache/torch_tpu_cache",
    "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "/root/vllm-torchtpu-cache/torch_tpu_cache",
    "TPU_ACCELERATOR_TYPE": "tpu7x",
    "TPU_EVICT_WEIGHTS_PAGE_CACHE": "true",
    "TPU_LOG_DIR": "/root/vllm-torchtpu-cache/tpu_logs",
    "TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER": "1",
    "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS": "33",
    "TPU_MOE_ROUTER_TOPK": "rowmax",
    "TPU_MOE_SKIP_PADDED_TOKENS": "1",
    "TPU_PARALLEL_PRECOMPILE": "1",
    "TPU_PJRT_EVENT_TRACE_ENABLE": "1",
    "TPU_PREMAPPED_BUFFER_SIZE": "4294967296",
    "TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES": "4294967296",
    "TPU_PROCESS_PORT": "8471",
    "TPU_ROPE_CACHE_TRUNCATE": "1",
    "TPU_STDERR_LOG_LEVEL": "0",
    "TPU_TOKEN_BUCKET_EXTRA": "4,8,48,3072",
    "TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS": "1024",
    "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "1",
    "TPU_VMODULE": "tpu_pjrt_client=1,pjrt_stream_executor_client=1,tpu_pjrt_compiler_utils=1",
    "TPU_WORKER_ID": "0",
    "TRITON_CACHE_DIR": "/root/vllm-torchtpu-cache/triton",
    "USE_BATCHED_RPA_LONGCTX": "1",
    "USE_MOE_FUSED_EP_KERNEL": "0",
    "USE_MOE_SPARSE_CORE": "1",
    "VLLM_CACHE_ROOT": "/root/vllm-torchtpu-cache/vllm",
    "VLLM_CONFIG_ROOT": "/root/vllm-torchtpu-cache/vllm_config",
    "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
    "VLLM_KV_CACHE_LAYOUT": "HND",
    "VLLM_LOGGING_LEVEL": "INFO",
    "VLLM_TPU_BLOCK_MAJOR_KV": "0",
    "VLLM_XLA_CACHE_PATH": "/root/vllm-torchtpu-cache/vllm/xla_cache",
    "VLLM_XLA_CHECK_RECOMPILATION": "0",
}

SERVE_CMD = [
    "vllm", "serve", str(MODEL_DIR),
    f"--served-model-name={MODEL}",
    "--tensor-parallel-size=8",
    "--data-parallel-size=1",
    "--max-model-len=262144",
    "--quantization=fp8",
    "--max-num-batched-tokens=16384",
    "--max-num-seqs=8",
    "--host", "0.0.0.0",
    "--port", str(PORT),
    "--async-scheduling",
    "--prefill-schedule-interval=1",
    "--enable-prefix-caching",
    "--gpu-memory-utilization=0.92",
    "--kv-cache-dtype=fp8",
    "--language-model-only",
    "--enable-expert-parallel",
    "--attention-backend", "CUSTOM",
    "--block-size", "256",
    "--limit-mm-per-prompt", '{"image":0,"video":0}',
    "--default-chat-template-kwargs", '{"enable_thinking":false}',
    "--tool-call-parser=qwen3_coder",
    "--reasoning-parser=qwen3",
    "--mamba-cache-mode", "align",
    "--trust-remote-code",
    "--enable-prompt-tokens-details",
    "--kv-cache-memory-bytes", "42467328000",
    "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}',
    "--profiler-config.profiler=torch",
    "--profiler-config.torch_profiler_dir=/dev/shm/xprof",
    "--profiler-config.ignore_frontend=true",
]  # fmt: skip

#: Directories the recipe creates before vLLM starts; vLLM creates none of them.
SERVER_DIRS = ("CACHE_DIR", "TMPDIR", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT",
               "VLLM_CONFIG_ROOT", "VLLM_XLA_CACHE_PATH", "TORCH_TPU_CACHE_DIR",
               "JAX_COMPILER_CACHE_DIR", "TPU_LOG_DIR")  # fmt: skip

#: Printed in their millions under TPU_VMODULE; kept out of the saved log.
LOG_NOISE = re.compile(
    rb"tpu_pjrt_client\.cc|pjrt_stream_executor|tpu_pjrt_compiler_utils|GET /metrics"
)

READY_TIMEOUT_S = 3000
REQUEST_TIMEOUT_S = 900

# --- the load --------------------------------------------------------------------
#
# The job of record replayed an agentic-coding trace at concurrency 8: ~96k-token
# prompts, 93% prefix-cache hits, ~7.7 uncached prompt tokens per output token,
# and, by its own 10 s engine stats, 1 request running in 20% of the busy
# intervals' generated tokens, 2 in 30%, 3-4 in 35% and 5-8 in 15%. The load
# reproduces that regime with fixed greedy closed-loop work: every request shares
# one long cached prefix, adds an uncached suffix no other request uses, and
# decodes a fixed number of tokens with EOS ignored. A decode config is one
# concurrency level; its weight in the win rule is that level's share of
# generated tokens. `p16k` is the prefill column: requests whose uncached part
# fills the 16384-token chunk, so the large-M kernel buckets are timed too.

PREFIX_CHARS = 300_000
REPS = 7
#: Untimed passes over every config before the timed ones: the server compiles small
#: host-side programs the first time it meets a batch shape, for minutes after boot.
WARMUP_PASSES = 2
CONFIGS = {
    "c1": {"concurrency": 1, "requests_per_stream": 16, "suffix_chars": 7_500, "output_tokens": 256},
    "c2": {"concurrency": 2, "requests_per_stream": 12, "suffix_chars": 7_500, "output_tokens": 256},
    "c4": {"concurrency": 4, "requests_per_stream": 8, "suffix_chars": 7_500, "output_tokens": 256},
    "c8": {"concurrency": 8, "requests_per_stream": 6, "suffix_chars": 7_500, "output_tokens": 256},
    "p16k": {"concurrency": 1, "requests_per_stream": 3, "suffix_chars": 64_000, "output_tokens": 16},
}  # fmt: skip
MAX_SUFFIX_CHARS = max(c["suffix_chars"] for c in CONFIGS.values())

#: (question, a string the greedy reply must contain). Asked one at a time.
KNOWN_ANSWER_PROBES = [
    ("What is 47 + 38? Reply with only the number.", "85"),
    ("What is 93 - 57? Reply with only the number.", "36"),
    ("What is 12 * 11? Reply with only the number.", "132"),
    ("What is 144 / 12? Reply with only the number.", "12"),
    ("What is 25 + 76? Reply with only the number.", "101"),
    ("What is 9 * 8? Reply with only the number.", "72"),
    ("What is 300 - 123? Reply with only the number.", "177"),
    ("What is 15 squared? Reply with only the number.", "225"),
    ("What is the capital of France? Reply with one word.", "Paris"),
    ("What is the capital of Japan? Reply with one word.", "Tokyo"),
    ("What is the capital of Italy? Reply with one word.", "Rome"),
    ("What is the chemical symbol for gold? Reply with only the symbol.", "Au"),
    ("How many days are in a week? Reply with only the number.", "7"),
    ("What planet is known as the Red Planet? Reply with one word.", "Mars"),
    ("What is 10 + 10? Reply with only the number.", "20"),
    ("What is the opposite of 'hot'? Reply with one word.", "cold"),
]
KNOWN_ANSWER_MAX_TOKENS = 16
#: (secret, characters of uncached context around it): the answer sits in the middle
#: of text no other request uses, so the reply depends on a full 16384-token chunk.
NEEDLE_PROBES = [("7341", 70_000), ("2958", 70_000)]
#: Greedy open-ended replies: checksummed, not graded.
OPEN_PROBES = [
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain in three sentences how a hash table works.",
    "List the first ten prime numbers, separated by commas.",
    "Describe the water cycle in two sentences.",
]
OPEN_MAX_TOKENS = 64

#: Reference prompts for worst_error, per config: (prompts, max tokens, min tokens). A
#: prompt of n tokens runs its whole forward pass, MoE included, in the smallest token
#: bucket >= n, which for the decode configs is the bucket their decode steps use. The
#: logprobs are the prefill step's: with speculative decoding on, this stack returns
#: logprobs only on a step that verifies no drafts.
REFERENCE = {"c1": (12, 4, 1), "c2": (12, 8, 5), "c4": (12, 16, 9), "c8": (12, 32, 17),
             "p16k": (3, 32_768, 16_385)}  # fmt: skip
REFERENCE_TOPK = 5

#: Profiled windows per config, when an arm asks for them (untimed, after the reps).
PROFILE_LEAD_S = 4
PROFILE_SPAN_S = 3


def say(msg: str) -> None:
    print(
        f"[gmmv2-score {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- git tree hash, without git ----------------------------------------------------


def _git_obj(kind: bytes, body: bytes) -> bytes:
    return hashlib.sha1(kind + b" " + str(len(body)).encode() + b"\0" + body).digest()


def _tree_body(root: str) -> bytes | None:
    entries = []
    for child in os.scandir(root):
        name = child.name.encode()
        if child.is_symlink():
            target = os.readlink(child.path).encode()
            entries.append((name, b"120000", _git_obj(b"blob", target)))
        elif child.is_dir():
            if child.name == ".git":
                continue
            sub = _tree_body(child.path)
            if sub is not None:  # git does not record empty directories
                entries.append((name + b"/", b"40000", _git_obj(b"tree", sub)))
        else:
            mode = b"100755" if os.stat(child.path).st_mode & 0o111 else b"100644"
            entries.append(
                (name, mode, _git_obj(b"blob", Path(child.path).read_bytes()))
            )
    if not entries:
        return None
    entries.sort(key=lambda e: e[0])  # git sorts a directory as if named "<name>/"
    return b"".join(m + b" " + n.rstrip(b"/") + b"\0" + h for n, m, h in entries)


def git_tree_hash(root: Path) -> str:
    """What `git add -A && git write-tree` would print for a directory."""
    return _git_obj(b"tree", _tree_body(str(root)) or b"").hex()


# =================================================================================
# In the pod
# =================================================================================


def _conn(timeout: float = REQUEST_TIMEOUT_S) -> http.client.HTTPConnection:
    return http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)


def http_get(path: str, timeout: float = 10) -> tuple[int, str]:
    c = _conn(timeout)
    try:
        c.request("GET", path)
        r = c.getresponse()
        return r.status, r.read().decode(errors="replace")
    finally:
        c.close()


def http_post_json(path: str, body: dict | None = None) -> dict:
    c = _conn()
    try:
        c.request(
            "POST", path, json.dumps(body or {}), {"Content-Type": "application/json"}
        )
        r = c.getresponse()
        data = r.read().decode(errors="replace")
        if r.status != 200:
            raise RuntimeError(f"POST {path} -> {r.status}: {data[:500]}")
        return json.loads(data) if data.strip() else {}
    finally:
        c.close()


def stream_completion(prompt: str, max_tokens: int) -> dict:
    """One greedy streamed completion: its timings and token counts."""
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}  # fmt: skip
    c = _conn()
    t0 = time.perf_counter()
    t_first = usage = None
    try:
        c.request(
            "POST",
            "/v1/completions",
            json.dumps(body),
            {"Content-Type": "application/json"},
        )
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"completion -> {r.status}: {r.read()[:500]!r}")
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            msg = json.loads(payload)
            if t_first is None and msg.get("choices") and msg["choices"][0].get("text"):
                t_first = time.perf_counter()
            if msg.get("usage"):
                usage = msg["usage"]
    finally:
        c.close()
    t_end = time.perf_counter()
    if t_first is None or usage is None:
        raise RuntimeError("stream ended without text or usage")
    tokens = usage["completion_tokens"]
    return {"ttft_s": t_first - t0, "tokens": tokens, "prompt_tokens": usage["prompt_tokens"],
            "tpot_s": (t_end - t_first) / max(tokens - 1, 1)}  # fmt: skip


COUNTERS = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total",
            "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total")  # fmt: skip


def scrape_counters() -> dict[str, float]:
    status, text = http_get("/metrics")
    if status != 200:
        raise RuntimeError(f"/metrics -> {status}")
    out = dict.fromkeys(COUNTERS, 0.0)
    for line in text.splitlines():
        name = re.split(r"[{ ]", line, maxsplit=1)[0]
        if name in out:
            out[name] += float(line.rsplit(" ", 1)[1])
    return out


class Prompts:
    """Natural text the image fixes: its own Python stdlib, in sorted file order."""

    def __init__(self):
        files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
        self.corpus = "".join(Path(f).read_text(errors="replace") for f in files)
        if len(self.corpus) < PREFIX_CHARS + 20 * MAX_SUFFIX_CHARS:
            raise RuntimeError(f"corpus too small: {len(self.corpus)} chars")
        self.sha256 = hashlib.sha256(self.corpus.encode()).hexdigest()
        self.prefix = self.corpus[:PREFIX_CHARS]
        self.span = len(self.corpus) - PREFIX_CHARS - MAX_SUFFIX_CHARS

    def excerpt(self, index: int, chars: int) -> str:
        start = PREFIX_CHARS + (index * 104_729) % self.span
        return self.corpus[start : start + chars]

    def get(self, index: int, suffix_chars: int) -> str:
        """The index-th request: the shared prefix plus a suffix only it uses."""
        suffix = self.excerpt(index, suffix_chars)
        return f"{self.prefix}\n\n# ---- excerpt {index} ----\n{suffix}\n# Continue this file:\n"

    def needle(self, index: int, secret: str, chars: int) -> str:
        text = self.excerpt(index, chars)
        half = len(text) // 2
        return (
            f"{text[:half]}\n# The secret code is {secret}.\n{text[half:]}\n\n"
            "What is the secret code mentioned in the text above? Reply with only the number."
        )


def server_env(rec_dir: Path | None = None) -> dict:
    env = dict(os.environ)
    env.update(SERVER_ENV)
    if rec_dir is not None:
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(HOOK.parent), env.get("PYTHONPATH")) if p
        )
        env["GMMV2_SCORE_RECORD_DIR"] = str(rec_dir)
        env["GMMV2_SCORE_ARM_ROOT"] = str(DEPLOY_ROOT)
    return env


def kill_strays() -> list[str]:
    """SIGKILL every python/vllm process but this one, its parents and the pod's shell."""
    keep = {1, os.getpid(), os.getppid()}
    killed = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in keep:
            continue
        try:
            cmd = (
                (d / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except OSError:
            continue
        if cmd.startswith("/bin/bash") or "scorer.py" in cmd:
            continue
        if "python" in cmd or "vllm" in cmd:
            try:
                os.kill(int(d.name), signal.SIGKILL)
                killed.append(cmd.strip()[:120])
            except ProcessLookupError:
                pass
    return killed


class Server:
    """One `vllm serve` from the deployed arm; its log filtered as it is written."""

    def __init__(self, work: Path, rec_dir: Path):
        self.log_path = work / "server.log"
        self.markers: list[str] = []
        self.n_markers = 0
        self.strays: list[str] = []
        self.proc = subprocess.Popen(SERVE_CMD, cwd="/root", env=server_env(rec_dir),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     start_new_session=True)  # fmt: skip
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        with open(self.log_path, "wb") as log:
            for raw in self.proc.stdout:
                if LOG_NOISE.search(raw):
                    continue
                log.write(raw)
                line = raw.decode(errors="replace")
                if MARKER in line:
                    self.n_markers += 1
                    if len(self.markers) < 60:
                        self.markers.append(line.rstrip())

    def wait_ready(self) -> float:
        t0 = time.time()
        while time.time() - t0 < READY_TIMEOUT_S:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited with {self.proc.returncode} during boot"
                )
            try:
                if http_get("/health", timeout=5)[0] == 200:
                    return time.time() - t0
            except OSError:
                pass
            time.sleep(10)
        raise RuntimeError(f"server not ready after {READY_TIMEOUT_S}s")

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=60)
        # A worker that left the session would keep the chips for the next arm.
        self.strays = kill_strays()
        self.reader.join(timeout=60)
        time.sleep(10)


def deploy(src: Path) -> None:
    """Make the serving checkout hold exactly the arm's tree."""
    DEPLOY_ROOT.mkdir(parents=True, exist_ok=True)
    for child in DEPLOY_ROOT.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    shutil.copytree(src, DEPLOY_ROOT, symlinks=True, dirs_exist_ok=True)


def reset_runtime_state() -> None:
    """Empty every compile cache and TPU scratch: each arm builds its own programs."""
    if CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
    for name in SERVER_DIRS:
        Path(SERVER_ENV[name]).mkdir(parents=True, exist_ok=True)
    shutil.rmtree("/dev/shm/xprof", ignore_errors=True)
    Path("/dev/shm/xprof").mkdir(parents=True)
    for pat in ("/tmp/libtpu*", "/dev/shm/libtpu*"):
        for f in glob.glob(pat):
            Path(f).unlink()


def package_origin() -> str:
    out = subprocess.run(
        [sys.executable, "-c", "import vllm_torchtpu; print(vllm_torchtpu.__file__)"],
        env=server_env(), capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    if out.returncode != 0:
        raise RuntimeError(f"cannot import vllm_torchtpu: {out.stderr[-1000:]}")
    return out.stdout.strip().splitlines()[-1]


def run_probes(prompts: Prompts) -> dict:
    def ask(question: str, max_tokens: int) -> str:
        r = http_post_json("/v1/chat/completions", {
            "model": MODEL, "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens, "temperature": 0.0})  # fmt: skip
        msg = r["choices"][0]["message"]
        return (msg.get("reasoning_content") or msg.get("reasoning") or "") + (
            msg.get("content") or ""
        )

    graded = []
    questions = list(KNOWN_ANSWER_PROBES) + [
        (prompts.needle(500_000 + i, secret, chars), secret)
        for i, (secret, chars) in enumerate(NEEDLE_PROBES)
    ]
    for question, want in questions:
        got = ask(question, KNOWN_ANSWER_MAX_TOKENS)
        graded.append({"q": question[-160:], "want": want, "got": got,
                       "ok": want.lower() in got.lower()})  # fmt: skip
    open_replies = [ask(p, OPEN_MAX_TOKENS) for p in OPEN_PROBES]
    texts = [g["got"] for g in graded] + open_replies
    wrong = sum(not g["ok"] for g in graded)
    return {
        "graded": graded,
        "open": open_replies,
        "worst_error": wrong / len(graded),
        "checksum": "sha256:" + hashlib.sha256(json.dumps(texts).encode()).hexdigest(),
    }


def count_tokens(text: str) -> int:
    return int(http_post_json("/tokenize", {"model": MODEL, "prompt": text})["count"])


def reference_prompts(prompts: Prompts) -> list[tuple[str, str]]:
    """(config, prompt) pairs, chosen by the server's own tokenizer, the same in every arm."""
    chosen = []
    for k, (cfg, (want, hi, lo)) in enumerate(REFERENCE.items()):
        found, index = 0, 600_000 + 10_000 * k
        while found < want:
            index += 1
            if index > 600_000 + 10_000 * k + 2_000:
                raise RuntimeError(f"no {lo}-{hi} token prompts found for {cfg}")
            for chars in (max(hi // 2, 2), hi, 2 * hi, 3 * hi, 4 * hi):
                text = prompts.excerpt(index, chars)
                n = count_tokens(text)
                if lo <= n <= hi:
                    chosen.append((cfg, text))
                    found += 1
                    break
                if n > hi:
                    break
    return chosen


def next_token_logprobs(refs: list[tuple[str, str]]) -> list[dict]:
    out = []
    for cfg, text in refs:
        r = http_post_json("/v1/completions", {
            "model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0.0,
            "logprobs": REFERENCE_TOPK})  # fmt: skip
        top = r["choices"][0]["logprobs"]["top_logprobs"][0]
        out.append({"config": cfg, "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
                    "tokens": r["usage"]["prompt_tokens"], "top": top})  # fmt: skip
    return out


def logprob_error(ref: dict, got: dict) -> float:
    """Largest |logprob| gap over the reference's top tokens; a token missing from `got`
    is charged at least its distance above got's lowest listed logprob."""
    floor = min(got.values())
    return max(
        abs(got[t] - lp) if t in got else max(lp - floor, 0.0) for t, lp in ref.items()
    )


def reference_errors(mine: list[dict], control: list[dict]) -> dict[str, float]:
    by_sha = {r["sha"]: r for r in control}
    if [r["sha"] for r in mine] != [r["sha"] for r in control]:
        raise RuntimeError("reference prompts differ from the control arm's")
    worst = dict.fromkeys(REFERENCE, 0.0)
    for r in mine:
        err = logprob_error(by_sha[r["sha"]]["top"], r["top"])
        worst[r["config"]] = max(worst[r["config"]], err)
    return worst


def run_rep(prompts: Prompts, cfg: dict, base: int) -> dict:
    """One closed-loop repetition: `concurrency` streams issuing their requests in turn."""
    results: list[dict] = []
    errors: list[str] = []
    lock = threading.Lock()
    per_stream, want = cfg["requests_per_stream"], cfg["output_tokens"]

    def stream(s: int) -> None:
        for j in range(per_stream):
            try:
                prompt = prompts.get(base + s * per_stream + j, cfg["suffix_chars"])
                r = stream_completion(prompt, want)
            except Exception as e:  # collected and raised below: the rep is void
                with lock:
                    errors.append(f"stream {s} request {j}: {e!r}")
                return
            with lock:
                results.append(r)

    t0 = time.perf_counter()
    threads = [
        threading.Thread(target=stream, args=(s,)) for s in range(cfg["concurrency"])
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    if errors:
        raise RuntimeError("; ".join(errors[:3]))
    tokens = sum(r["tokens"] for r in results)
    return {
        "us_per_token": wall * 1e6 / tokens,
        "requests": len(results),
        "exact_lengths": all(r["tokens"] == want for r in results),
        "prompt_tokens": [r["prompt_tokens"] for r in results],
        "ttft_s": [r["ttft_s"] for r in results],
        "tpot_s": [r["tpot_s"] for r in results],
    }


def run_load(prompts: Prompts) -> dict:
    """REPS rounds, each running every config once, so drift lands on all of them alike."""
    reps = {name: [] for name in CONFIGS}
    deltas = {name: [] for name in CONFIGS}
    index = 1_000
    for rep in range(REPS):
        for name, cfg in CONFIGS.items():
            before = scrape_counters()
            r = run_rep(prompts, cfg, index)
            after = scrape_counters()
            index += cfg["concurrency"] * cfg["requests_per_stream"]
            reps[name].append(r)
            deltas[name].append({k: after[k] - before[k] for k in COUNTERS})
            say(f"  rep {rep} {name}: {r['us_per_token']:.1f} us/token")
    out = {}
    for name in CONFIGS:
        lat = [r["us_per_token"] for r in reps[name]]
        med = statistics.median(lat)

        def total(key, name=name):
            return sum(d[key] for d in deltas[name])

        drafts = total("vllm:spec_decode_num_drafts_total")
        queries = total("vllm:prefix_cache_queries_total")
        out[name] = {
            "latency_us": med,
            "exact_lengths": all(r["exact_lengths"] for r in reps[name]),
            "reps_us_per_token": lat,
            "metrics": {
                "median_ttft_ms": 1e3 * statistics.median(t for r in reps[name] for t in r["ttft_s"]),
                "median_tpot_ms": 1e3 * statistics.median(t for r in reps[name] for t in r["tpot_s"]),
                "output_tok_s": 1e6 / med,
                "completed": float(sum(r["requests"] for r in reps[name])),
                "rep_spread": (max(lat) - min(lat)) / med,
                "spec_accept_len": (
                    1 + total("vllm:spec_decode_num_accepted_tokens_total") / drafts if drafts else 0.0
                ),
                "prefix_hit_rate": total("vllm:prefix_cache_hits_total") / queries if queries else 0.0,
                "median_prompt_tokens": float(
                    statistics.median(p for r in reps[name] for p in r["prompt_tokens"])
                ),
            },
        }  # fmt: skip
    return out


def capture_profiles(prompts: Prompts, arm_out: Path) -> dict:
    """One short trace per config, inside an untimed repetition of it."""
    got = {}
    for name, cfg in CONFIGS.items():
        shutil.rmtree("/dev/shm/xprof", ignore_errors=True)
        Path("/dev/shm/xprof").mkdir(parents=True)
        rep = threading.Thread(target=run_rep, args=(prompts, cfg, 900_000))
        rep.start()
        time.sleep(PROFILE_LEAD_S)
        http_post_json("/start_profile")
        time.sleep(PROFILE_SPAN_S)
        http_post_json("/stop_profile")
        rep.join()
        time.sleep(30)  # the trace is flushed asynchronously
        dest = arm_out / "xprof" / name
        dest.mkdir(parents=True, exist_ok=True)
        files = sorted(glob.glob("/dev/shm/xprof/**/*.xplane.pb", recursive=True))
        kept = [f for f in files if Path(f).name.startswith("rank0_")]
        for f in kept:
            shutil.copy(f, dest / Path(f).name)
        got[name] = {"files": len(files), "kept": [Path(f).name for f in kept]}
        say(f"  profile {name}: kept {len(kept)} of {len(files)} trace files")
    return got


def read_records(rec_dir: Path) -> dict:
    modules, mosaic, chained, libtpu_args = set(), set(), set(), set()
    for f in sorted(rec_dir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            kind = f.name.split(".", 1)[0]
            if kind == "module":
                modules.add((r["path"], r["sha256"]))
            elif kind == "mosaic":
                mosaic.add((r["name"], r["sha256"]))
                libtpu_args.add(r["libtpu_init_args"])
            elif kind == "chained":
                chained.add(r["path"])
    by_path: dict[str, set] = {}
    for path, digest in modules:
        by_path.setdefault(path, set()).add(digest)
    torn = sorted(p for p, d in by_path.items() if len(d) > 1)
    if torn:
        raise RuntimeError(f"processes imported different bytes for {torn[:5]}")
    if len(libtpu_args) > 1:
        raise RuntimeError(
            f"processes lowered under different LIBTPU_INIT_ARGS: {libtpu_args}"
        )
    return {
        "libtpu_init_args": next(iter(libtpu_args), ""),
        "modules": sorted(modules),
        "mosaic": sorted(mosaic),
        "chained": sorted(chained),
    }


def digest_lines(prefix: str, lines) -> str:
    return prefix + hashlib.sha256("\n".join(lines).encode()).hexdigest()


def unmeasured_configs() -> dict:
    """The arm's server could not be measured: its code did not run on any config."""
    return {name: {"latency_us": 0.0, "correct": False, "worst_error": 1.0, "checksum": "",
                   "dispatched": "", "declined": True} for name in CONFIGS}  # fmt: skip


def run_arm(
    spec: dict, stage: Path, out: Path, prompts: Prompts, control_ref: list[dict] | None
) -> dict:
    name = spec["name"]
    arm_out = out / "arms" / name
    arm_out.mkdir(parents=True, exist_ok=True)
    work = WORK_ROOT / name
    shutil.rmtree(work, ignore_errors=True)
    rec_dir = work / "records"
    rec_dir.mkdir(parents=True)
    arm = {"name": name, "control": bool(spec["control"]), "rev": spec["rev"]}
    say(f"arm {name}: deploying {spec['rev'][:12]}")
    deploy(stage / "arms" / name)
    arm["tree"] = git_tree_hash(DEPLOY_ROOT)
    arm["kernel_tree"] = git_tree_hash(DEPLOY_ROOT / KERNEL_REL)
    if arm["tree"] != spec["tree"]:
        # Staging failed; no number can be vouched for, so none is printed.
        arm["error"] = f"deployed tree {arm['tree']} != manifest {spec['tree']}"
        say(f"arm {name}: {arm['error']}")
        return arm
    reset_runtime_state()
    arm["origin"] = package_origin()
    if not arm["origin"].startswith(f"{DEPLOY_ROOT}/"):
        arm["error"] = (
            f"vllm_torchtpu resolves to {arm['origin']}, not the deployed arm"
        )
        say(f"arm {name}: {arm['error']}")
        return arm
    server = load = probes = None
    try:
        t0 = time.time()
        server = Server(work, rec_dir)
        arm["boot_s"] = server.wait_ready()
        say(f"arm {name}: ready after {arm['boot_s']:.0f}s")
        arm["prefix_prompt_tokens"] = stream_completion(prompts.get(0, 0), 32)[
            "prompt_tokens"
        ]
        for warm in range(WARMUP_PASSES):
            for cfg in CONFIGS.values():
                run_rep(prompts, cfg, 700_000 + 1_000 * warm)
        probes = run_probes(prompts)
        say(f"arm {name}: probes worst_error={probes['worst_error']:.3f}")
        load = run_load(prompts)
        arm["reference"] = next_token_logprobs(reference_prompts(prompts))
        if spec.get("profile"):
            arm["profiles"] = capture_profiles(prompts, arm_out)
        arm["arm_wall_s"] = time.time() - t0
    except Exception as e:  # reported as an unmeasured arm, never as a number
        arm["error"] = repr(e)
        arm["load_before_failure"] = load  # for diagnosis only; no config is printed
        say(f"arm {name}: FAILED {e!r}")
        load = None
    finally:
        if server is not None:
            server.stop()
            arm["markers"] = server.markers
            arm["n_markers"] = server.n_markers
            arm["strays_killed"] = server.strays
            with (
                open(server.log_path, "rb") as src,
                gzip.open(arm_out / "server.log.gz", "wb") as dst,
            ):
                shutil.copyfileobj(src, dst)
    try:
        rec = read_records(rec_dir)
    except RuntimeError as e:
        arm["error"] = f"{arm.get('error', '')} records: {e}"
        return arm  # imported bytes disagree across processes: no number is printed
    mosaic = [f"{n} {h}" for n, h in rec["mosaic"]]
    arm["binary_hash"] = (
        digest_lines("py:", (f"{p} {h}" for p, h in rec["modules"]))
        if rec["modules"]
        else ""
    )
    arm["records"] = {
        "n_modules": len(rec["modules"]),
        "megablox_modules": [f"{p} {h[:16]}" for p, h in rec["modules"] if p.startswith(KERNEL_REL)],
        "n_mosaic": len(mosaic),
        "gmm_v2_kernels": [m for m in mosaic if m.startswith("gmm_v2")],
        "mosaic_digest": digest_lines("mosaic:", mosaic),
        "libtpu_init_args": rec["libtpu_init_args"],
        "chained_sitecustomize": rec["chained"],
    }  # fmt: skip
    if load is None:
        arm["configs"] = unmeasured_configs()
        return arm
    dispatched = (
        digest_lines(
            "mosaic:", [*mosaic, f"LIBTPU_INIT_ARGS={rec['libtpu_init_args']}"]
        )
        if mosaic
        else ""
    )
    arm["probes"] = probes
    reference = arm["reference"]
    if control_ref is None and not spec["control"]:
        arm["error"] = "the control arm left no reference to compare against"
        return arm  # nothing to vouch for: no number is printed
    try:
        errors = reference_errors(
            reference, reference if spec["control"] else control_ref
        )
    except RuntimeError as e:
        arm["error"] = repr(e)
        return arm
    top1 = [max(r["top"], key=r["top"].get) for r in reference]
    checksum = (
        "sha256:"
        + hashlib.sha256(json.dumps([probes["checksum"], top1]).encode()).hexdigest()
    )
    arm["configs"] = {
        cfg: {
            "latency_us": r["latency_us"],
            "correct": probes["worst_error"] == 0.0 and r["exact_lengths"],
            "worst_error": errors[cfg],
            "checksum": checksum,
            "dispatched": dispatched,
            "declined": False,
            "metrics": r["metrics"],
            "reps_us_per_token": r["reps_us_per_token"],
        }
        for cfg, r in load.items()
    }
    return arm


def fetch_checkpoint() -> float:
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcloud", "storage", "rsync", "-r", CKPT_URI, str(MODEL_DIR)],
                   check=True, stdout=subprocess.DEVNULL)  # fmt: skip
    return time.time() - t0


def package_versions() -> dict:
    from importlib import metadata

    out = {}
    for pkg in ("jax", "jaxlib", "libtpu", "torch", "torch_tpu", "vllm"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def cmd_run(args) -> int:
    stage, out = Path(args.stage), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if file_sha256(HOOK) != HOOK_SHA256:
        say(f"{HOOK} is not the hook this scorer was frozen with; refusing to measure")
        return 1
    manifest = json.loads((stage / "arms.json").read_text())
    if sum(bool(a["control"]) for a in manifest["arms"]) != 1:
        say("the manifest must name exactly one control arm")
        return 1
    prompts = Prompts()
    job = {"job_id": os.environ.get("CDK_JOB_ID", ""), "pod": os.environ.get("HOSTNAME", ""),
           "corpus_sha256": prompts.sha256, "versions": package_versions()}  # fmt: skip
    say(
        f"job {job['job_id']} pod {job['pod']}: {len(manifest['arms'])} arms {job['versions']}"
    )
    job["checkpoint_s"] = fetch_checkpoint()
    say(f"checkpoint in place after {job['checkpoint_s']:.0f}s")
    arms = []
    if not manifest["arms"][0]["control"]:
        say(
            "the control arm must be measured first: the others are compared against it"
        )
        return 1
    control_ref = None
    for spec in manifest["arms"]:
        arm = run_arm(spec, stage, out, prompts, control_ref)
        if spec["control"]:
            control_ref = arm.get("reference")
        (out / "arms" / spec["name"] / "arm.json").write_text(json.dumps(arm, indent=1))
        arms.append(arm)
    payload = {"scorer_hash": "sha256:" + file_sha256(Path(__file__)), "job_id": job["job_id"],
               "arms": arms, "job": job}  # fmt: skip
    text = json.dumps(payload, sort_keys=True)
    (out / "score.json").write_text(text)
    print(text)
    return 0


# =================================================================================
# On the host
# =================================================================================

CDK_RECIPES = Path.home() / "cloud-devkit"
RECIPE_SUBDIR = "recipes/experimental/gmmv2-serve-score"
STAGE_MOUNT = "/root/gmmv2_score"
SCORE_GCS = "gs://cloud-devkit/jobs/{job}/outputs/*/*/scorer/gmmv2_score/score.json"
PERSONAL_CAP = 5

RECIPE = """\
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: {name}
spec:
  replicatedJobs:
  - name: scorer
    replicas: 1
    template:
      spec:
        backoffLimit: 0
        completions: 1
        parallelism: 1
        template:
          metadata:
            annotations:
              kueue.x-k8s.io/podset-unconstrained-topology: "true"
          spec:
            restartPolicy: Never
            terminationGracePeriodSeconds: 180
            nodeSelector:
              cloud.google.com/gke-accelerator-count: "4"
              cloud.google.com/gke-tpu-accelerator: tpu7x
              cloud.google.com/gke-tpu-topology: 2x2x1
            containers:
            - name: scorer
              image: {image}
              imagePullPolicy: IfNotPresent
              command: ["/bin/bash", "-c"]
              args:
              - |
                set -uo pipefail
                S={mount}
                OUT="${{CDK_OUTPUT_DIR}}/gmmv2_score"
                mkdir -p "$OUT"
                ( while sleep 60; do
                    if grep -rlsF 'Stall detected' /root/vllm-torchtpu-cache/tpu_logs >/dev/null; then
                      echo "GMMV2_SCORE_STALL pod=$HOSTNAME"; break
                    fi
                  done ) &
                echo "gmmv2-score: scorer sha256 $(sha256sum $S/scorer/scorer.py)"
                python3 $S/scorer/scorer.py run --stage $S --out "$OUT" > /tmp/score.json
                rc=$?
                echo GMMV2_SCORE_BEGIN; cat /tmp/score.json; echo; echo GMMV2_SCORE_END
                echo "gmmv2-score: exit=$rc"
                exit $rc
              env:
              - name: MEGASCALE_COORDINATOR_ADDRESS
              - name: MEGASCALE_NUM_SLICES
              - name: MEGASCALE_PORT
              - name: MEGASCALE_SLICE_ID
              resources:
                limits: {{cpu: "200", ephemeral-storage: 100Gi, google.com/tpu: "4", memory: 900Gi}}
                requests: {{cpu: "200", ephemeral-storage: 100Gi, google.com/tpu: "4", memory: 900Gi}}
              volumeMounts:
              - {{mountPath: /root/.cache/huggingface/, name: huggingface-storage}}
              - {{mountPath: /root/.cache/huggingface/token, name: hf-token-volume, readOnly: true, subPath: token}}
              - {{mountPath: /tmp, name: tmp-volume}}
              - {{mountPath: /root/.cache, name: cache-volume}}
              - {{mountPath: /root/.triton, name: triton-volume}}
              - {{mountPath: /var/tmp, name: var-tmp-volume}}
              - {{mountPath: /dev/shm, name: dshm}}
              - {{mountPath: /vllm-cache, name: vllm-shared-cache, readOnly: true}}
              - {{mountPath: /root/cloud-devkit/vllm-torchtpu, name: deploy-root}}
            volumes:
            - name: hf-token-volume
              secret: {{defaultMode: 420, optional: true, secretName: vllm-hf-token, items: [{{key: token, path: token}}]}}
            - name: huggingface-storage
              ephemeral:
                volumeClaimTemplate:
                  spec:
                    accessModes: [ReadWriteOnce]
                    resources: {{requests: {{storage: 1Ti}}}}
                    storageClassName: premium-rwo
            - {{name: dshm, emptyDir: {{medium: Memory}}}}
            - {{name: tmp-volume, emptyDir: {{medium: Memory}}}}
            - {{name: cache-volume, emptyDir: {{medium: Memory}}}}
            - {{name: triton-volume, emptyDir: {{medium: Memory}}}}
            - {{name: var-tmp-volume, emptyDir: {{medium: Memory}}}}
            - name: vllm-shared-cache
              persistentVolumeClaim: {{claimName: vllm-shared-cache-rox-pvc, readOnly: true}}
            - {{name: deploy-root, emptyDir: {{}}}}
"""


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def cdk(*args: str) -> str:
    """One cdk call with ack codes read fresh from the agent letter."""
    letter = subprocess.run(
        ["cdk", "agent-letter"], capture_output=True, text=True
    ).stdout
    codes = "-".join(re.findall(r"ack \d+/\d+: ([0-9A-F]+)", letter))
    if not codes:
        raise RuntimeError("cdk agent-letter printed no ack codes")
    out = subprocess.run(
        ["cdk", f"--agent-code={codes}", *args], capture_output=True, text=True
    )
    if out.returncode != 0:
        raise RuntimeError(f"cdk {args[0]} {args[1] if len(args) > 1 else ''} failed: "
                           f"{out.stderr.strip()[-800:]}")  # fmt: skip
    return out.stdout


def cdk_json(*args: str) -> list:
    rows = json.loads(cdk(*args, "-o", "json"))
    if not isinstance(rows, list):
        raise RuntimeError(f"cdk {' '.join(args)} returned no JSON array")
    return rows


def my_users() -> set[str]:
    user = os.environ["USER"].split("_")[0]
    return {user, f"{user}_google_com"}


def my_active_jobs() -> list[dict]:
    rows = cdk_json("job", "list", "-a", "-A", "-n", "300")
    return [r for r in rows
            if r.get("job_status") in ("Running", "Pending") and r.get("user") in my_users()]  # fmt: skip


def check_caps(budget_tag: str | None, max_inflight: int) -> None:
    """Refuse a submit that would exceed the cluster, personal or run budget."""
    nodes = [n for n in cdk_json("usage")
             if "singlehost" in n.get("node_pool", "") and n.get("status") == "Ready"]  # fmt: skip
    if not nodes:
        raise RuntimeError("cdk usage lists no Ready single-host node")
    busy = sum(1 for n in nodes if any(p.get("job_id") for p in n.get("pods") or []))
    if busy >= len(nodes):
        raise RuntimeError(f"cluster full: {busy}/{len(nodes)} single-host nodes busy")
    mine = my_active_jobs()
    now = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
    off_peak = now.weekday() >= 5 or now.hour >= 20 or now.hour < 8
    lifted = off_peak and (busy - len(mine)) < 0.7 * len(nodes)
    if len(mine) >= PERSONAL_CAP and not lifted:
        raise RuntimeError(
            f"personal cap: {len(mine)}/{PERSONAL_CAP} jobs running or pending"
        )
    if budget_tag:
        tagged = [r for r in mine if budget_tag in (r.get("tags") or [])]
        if len(tagged) >= max_inflight:
            raise RuntimeError(f"run budget: {len(tagged)}/{max_inflight} jobs tagged "
                               f"{budget_tag} in flight")  # fmt: skip
    say(f"caps: cluster {busy}/{len(nodes)}, mine {len(mine)}, personal cap "
        f"{'lifted' if lifted else PERSONAL_CAP}")  # fmt: skip


def parse_arm(text: str) -> dict:
    m = re.fullmatch(r"([A-Za-z0-9_\-]+)=([^:]+)(:control)?", text)
    if not m:
        raise SystemExit(f"--arm {text!r}: expected NAME=REV or NAME=REV:control")
    return {"name": m.group(1), "rev": m.group(2), "control": bool(m.group(3))}


def stage_arms(arms: list[dict], profile: set[str], dest: Path) -> list[dict]:
    """`git archive` each arm into dest/arms/<name> and prove the tree survived."""
    manifest = []
    for a in arms:
        sha = git("rev-parse", "--verify", f"{a['rev']}^{{commit}}")
        want = git("rev-parse", f"{sha}^{{tree}}")
        arm_dir = dest / "arms" / a["name"]
        arm_dir.mkdir(parents=True)
        archive = subprocess.run(
            ["git", "archive", sha], check=True, capture_output=True
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(arm_dir)], input=archive, check=True)
        got = git_tree_hash(arm_dir)
        if got != want:
            raise RuntimeError(
                f"arm {a['name']}: staged tree {got} != commit tree {want}"
            )
        manifest.append({"name": a["name"], "control": a["control"], "rev": sha, "tree": want,
                         "profile": a["name"] in profile})  # fmt: skip
        say(f"staged {a['name']}: {sha[:12]} tree {want}")
    return manifest


def register_recipe(name: str, k8s_file: str) -> None:
    """Add the recipe to recipes.yml unless present; other sessions rewrite that file."""
    registry = CDK_RECIPES / "recipes.yml"
    if re.search(rf"^- name: {re.escape(name)}$", registry.read_text(), re.M):
        return
    with open(registry, "a") as f:
        f.write(f"- name: {name}\n  owner: {os.environ['USER']}\n  k8s_file: {k8s_file}\n"
                f"  require_gcs_mount: true\n")  # fmt: skip


def cmd_submit(args) -> int:
    arms = sorted((parse_arm(a) for a in args.arm), key=lambda a: not a["control"])
    if sum(a["control"] for a in arms) != 1 or len({a["name"] for a in arms}) != len(
        arms
    ):
        raise SystemExit("name every arm once and exactly one of them :control")
    name = f"gmmv2-score-{time.strftime('%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    stage = Path(tempfile.mkdtemp(prefix=f"{name}-"))
    manifest = stage_arms(arms, set(args.profile), stage)
    (stage / "arms.json").write_text(json.dumps({"arms": manifest}, indent=1))
    shutil.copytree(
        HERE, stage / "scorer", ignore=shutil.ignore_patterns("__pycache__")
    )
    recipe_dir = CDK_RECIPES / RECIPE_SUBDIR
    recipe_dir.mkdir(parents=True, exist_ok=True)
    k8s_file = f"{RECIPE_SUBDIR}/{name}.yaml"
    (CDK_RECIPES / k8s_file).write_text(
        RECIPE.format(name=name, image=IMAGE, mount=STAGE_MOUNT)
    )
    deadline = 2400 + 3600 * len(arms)
    tags = [
        "agent-jobs",
        os.environ.get("AI_AGENT", "agent"),
        "gmmv2-serve-score",
        *args.tag,
    ]
    cmd = ["job", "create", name, "-t", ",".join(tags), "--active-deadline-seconds", str(deadline),
           "--ttl", "1800", "--map-dir", f"{stage}:scorer:{STAGE_MOUNT}"]  # fmt: skip
    if args.dry_run:
        say(f"dry run: staged {stage}; would run cdk {' '.join(cmd)}")
        return 0
    check_caps(args.budget_tag, args.max_inflight)
    register_recipe(name, k8s_file)
    created = cdk(*cmd)
    m = re.search(r"\bj-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}\b", created)
    if not m:
        raise RuntimeError(f"no job id in cdk output: {created[-500:]}")
    job = m.group(0)
    say(f"submitted {job} (recipe {name}, stage {stage})")
    if args.no_wait:
        print(job)
        return 0
    if wait_job(job) != "Succeeded":
        return 1
    return fetch_score(job)


def job_status(job: str) -> str:
    for r in cdk_json("job", "list", "-a", "-n", "300"):
        if r.get("id") == job:
            return r.get("job_status", "")
    return ""


def wait_job(job: str, poll_s: int = 180) -> str:
    last = None
    while True:
        status = job_status(job)
        if status != last:
            say(f"{job}: {status or '(not listed)'}")
            last = status
        if status in ("Succeeded", "Failed"):
            return status
        time.sleep(poll_s)


def fetch_score(job: str) -> int:
    """Print the pod's score.json verbatim; nothing when there is not exactly one."""
    ls = subprocess.run(["gcloud", "storage", "ls", SCORE_GCS.format(job=job)],
                        capture_output=True, text=True)  # fmt: skip
    found = [line for line in ls.stdout.split() if line.startswith("gs://")]
    if len(found) != 1:
        say(f"{job}: expected one score.json, found {found}")
        return 1
    text = subprocess.run(["gcloud", "storage", "cat", found[0]], check=True,
                          capture_output=True, text=True).stdout  # fmt: skip
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="in the pod: measure every staged arm")
    r.add_argument("--stage", required=True)
    r.add_argument("--out", required=True)
    s = sub.add_parser(
        "submit", help="stage the arms, submit one cdk job, print its score"
    )
    s.add_argument("--arm", action="append", required=True, help="NAME=REV[:control]")
    s.add_argument("--profile", action="append", default=[], help="arm NAME to trace")
    s.add_argument("--tag", action="append", default=[])
    s.add_argument("--budget-tag", help="count my in-flight jobs carrying this tag ...")
    s.add_argument(
        "--max-inflight", type=int, default=3, help="... and refuse at this many"
    )
    s.add_argument("--no-wait", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    w = sub.add_parser("wait")
    w.add_argument("job")
    f = sub.add_parser("fetch")
    f.add_argument("job")
    t = sub.add_parser("tree")
    t.add_argument("dir")
    args = ap.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "submit":
        return cmd_submit(args)
    if args.cmd == "wait":
        return 0 if wait_job(args.job) == "Succeeded" else 1
    if args.cmd == "fetch":
        return fetch_score(args.job)
    print(git_tree_hash(Path(args.dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
