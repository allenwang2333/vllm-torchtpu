"""Served-model scorer for Qwen3.5-397B-A17B-FP8, TP1 x DP8 + EP on one tpu7x 2x2x1 node.

Each arm boots `vllm serve` from its own tree with the server command and
environment of the reproduce recipe (the `qwen3.5-397b-fp8-tp1-dp8-ep` serve
config at 1024 tokens in, 8192 out, range ratio 0.8, 512 requests at once, 640
in total), runs that sweep against it, and reports output-token throughput as
a latency (microseconds per output token), so control / arm > 1 means faster.

It is a `perfkit scorer run` benchmark (`perfkit/scorer_arm.py` defines the
interface) meant to run in a `perfkit cluster score` pod, where each arm root
holds only `src/vllm_torchtpu`. The serve and client commands live here, not in
the arm, so a candidate changes the code behind the server and nothing about
what is served or how it is measured.

Two departures from the recipe's literal text, both outside what is timed:
- Weights stream from the cluster's GCS copy of the checkpoint
  (`--load-format runai_streamer`, served under the model's own name, each
  rank fetching only its own experts) because the pod has no disk that holds
  the 406 GB checkpoint. Loading is not part of any sweep.
- The recipe's client samples lengths in [0.8 * len, len]. `vllm bench serve`
  samples symmetrically around its length, so it is given the midpoint and the
  ratio that span the same range, as run_benchmarks.sh does for
  RANGE_RATIO_STYLE=min. Requests are greedy, as the recipe's client is.

Timing: one short warmup sweep, then SWEEPS timed sweeps on the same server.
The control reports their median and any other arm their lower median; the
sweeps at least that fast are reported as `timing_rounds`, the harness's timing
samples.

Correctness half: a served model has no fp32 reference. Each arm sends a fixed
prompt suite serially and greedily, once with logprobs and once without: short
prompts, and long ones cut to fixed token counts so that the attention and
linear-attention (GDN) paths run past short contexts. `checksum` digests the
generated token ids, so equal checksums mean the same tokens, and
`first_tokens` gives each prompt's first token and its logprob for the
harness's first-token check. `worst_error` is 0.0, or 1.0 when a sweep lost
requests or the suite could not be read, which fails any tolerance below 1.

A job that measures golden alone (scoping, and the job that profiles the base
of a campaign's next segment) boots its server with the phased profiler
configured but unarmed, and after the suite takes a profile run: the sweep's
first wave again with `--profile`. The arm's artifacts (server log, sweep
results, fingerprint and the phased capture) go to the job's results bucket on
every arm. Nothing measured runs after the profile run.
"""

from __future__ import annotations

import contextlib
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

MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
PORT = 8000
CONFIG = "qwen3.5-397b-fp8-tp1-dp8-ep"

#: The recipe's single shape. `ratio` is the recipe client's: lengths in
#: [ratio * len, len]. `warmup` is the short sweep run and discarded before the
#: timed ones; `sweeps` timed sweeps then run against the same boot.
CONFIGS = {
    CONFIG: {
        "isl": 1024,
        "osl": 8192,
        "ratio": 0.8,
        "concurrency": 512,
        "num_prompts": 640,
        "sweeps": 2,
        "warmup": {
            "isl": 1024,
            "osl": 128,
            "ratio": 0.8,
            "concurrency": 512,
            "num_prompts": 512,
        },
    },
}

#: The cluster's copy of the checkpoint. The server streams the weights from it;
#: the client and the suite read the tokenizer from TOKENIZER_DIR, pulled once
#: per pod without the weight files.
CHECKPOINT_URI = "gs://cloud-devkit/ckpt/hf/Qwen/Qwen3.5-397B-A17B-FP8"
TOKENIZER_DIR = Path("/dev/shm/qwen35_397b_tokenizer")
#: Host memory each engine's streamer may buffer: eight engines stream at once.
STREAM_MEMORY_LIMIT = 16 * 1024**3

#: Server environment: the recipe's variables. RAGGED_GATED_DELTA_RULE_IMPL is
#: read by nothing in vllm_torchtpu at the recipe's commit; it stays because the
#: recipe sets it and a candidate tree may read it.
SERVER_ENV = {
    "TPU_ACCELERATOR_TYPE": "tpu7x",
    "MODEL_IMPL_TYPE": "vllm",
    "USE_MOE_SPARSE_CORE": "1",
    "ONEHOT_MOE_PERMUTE_THRESHOLD": "32768",
    "RAGGED_GATED_DELTA_RULE_IMPL": "chunked_kernel_v3_pd",
    "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
    "VLLM_NO_USAGE_STATS": "1",
    "PYTHONUNBUFFERED": "1",
}

#: Variables the recipe runs without; any inherited value is dropped.
UNSET_ENV = (
    "ASYNC_SCHEDULING",
    "ATTENTION_BACKEND",
    "BENCHMARK_EXTRA_BODY",
    "BENCHMARK_TEMPERATURE",
    "BENCHMARK_WARMUP_RUNS",
    "CAPTURE_PROFILE",
    "DATASET_NAME",
    "DATASET_PATH",
    "ENABLE_PREFIX_CACHING",
    "EXTRA_BENCH_ARGS",
    "EXTRA_LM_EVAL_MODEL_ARGS",
    "EXTRA_SERVE_ARGS",
    "LM_EVAL_GEN_KWARGS",
    "MODEL_URI",
    "PORT",
    "PROFILE_NUM_PROMPTS",
    "TOKENIZER",
    "USE_PHASED_PROFILER",
    "XLA_FLAGS",
    "LIBTPU_INIT_ARGS",
)

#: Set by `perfkit scorer run` when the job measures golden alone
#: (perfkit/scorer.py's GOLDEN_ALONE_ENV): only then is the profiler configured
#: and the profile run taken.
GOLDEN_ALONE_ENV = "PERFKIT_GOLDEN_ALONE"
#: Added to the server's environment in a golden-alone job only.
PROFILE_ENV = {
    "USE_PHASED_PROFILER": "1",
    "HOST_TRACER_LEVEL": "0",
    "PYTHON_TRACER_LEVEL": "0",
}
#: Steps captured per phase. At the phased profiler's default of 15, an 8-engine
#: DeepSeek-V4 server hung in trace parsing (each even rank hit its 1M-event cap).
PROFILE_STEPS = 5
#: decode_only is captured once every sequence of the wave holds this many
#: tokens: about the middle of a 1k-in / 6.5k-8k-out request, so the capture's
#: context length is the sweep's typical one, not its first decode step's.
PROFILE_DECODE_MIN_KV = 4096

READY_TIMEOUT_S = 4500
SWEEP_TIMEOUT_S = 3000
REQUEST_TIMEOUT_S = 600

#: Fixed short prompts for the output fingerprint: sent one at a time, greedy.
SUITE = (
    "The capital of France is",
    "Write a Python function that returns the n-th Fibonacci number.\n\ndef fib(n):",
    "Q: A train travels 120 km in 1.5 hours. What is its average speed in km/h?\nA:",
    "Translate to German: The weather is nice today, so we will walk to the station.",
    "List the first ten prime numbers, separated by commas:",
    "Explain in one paragraph why the sky appears blue during the day.",
    'def quicksort(arr):\n    """Sort a list of integers."""\n',
    "In 1969, the first humans to walk on the Moon were",
    "The chemical symbol for gold is",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Q: If x + 3 = 11, what is 2x?\nA:",
    "A haiku about autumn leaves:\n",
)
SUITE_TOKENS = 64
#: Long prompts, as token counts: a haystack of numbered records with a question
#: about one of them, cut to each length by the server's own tokenizer. The
#: largest leaves room for SUITE_TOKENS under --max-model-len=10240.
LONG_TOKENS = (2048, 4096, 6144, 10000)


def say(msg: str) -> None:
    print(f"qwen35_397b_serve: {msg}", file=sys.stderr, flush=True)


def server_cmd(profile_dir: Path | None) -> list[str]:
    """The recipe's `vllm serve` line as argv, loading from CHECKPOINT_URI, plus
    the phased profiler's settings when `profile_dir` is given."""
    cmd = [
        "vllm",
        "serve",
        f"--model={CHECKPOINT_URI}",
        "--served-model-name",
        MODEL,
        "--load-format",
        "runai_streamer",
        "--enable-ep-weight-filter",
        "--model-loader-extra-config",
        json.dumps({"memory_limit": STREAM_MEMORY_LIMIT}),
        "--tensor-parallel-size=1",
        "--data-parallel-size=8",
        "--max-model-len=10240",
        "--max-num-batched-tokens=1024",
        "--max-num-seqs=64",
        "--port",
        str(PORT),
        "--async-scheduling",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization=0.92",
        "--kv-cache-dtype=fp8",
        "--enable-expert-parallel",
        "--quantization",
        "fp8",
        "--attention-backend",
        "CUSTOM",
        "--block-size",
        "256",
        "--language-model-only",
        '--limit-mm-per-prompt={"image":0,"video":0}',
        '--default-chat-template-kwargs={"enable_thinking":false}',
    ]
    if profile_dir is not None:
        cmd += [
            "--profiler-config.profiler",
            "torch",
            "--profiler-config.torch_profiler_dir",
            str(profile_dir),
            "--profiler-config.ignore_frontend",
            "true",
            "--profiler-config.max_iterations",
            str(PROFILE_STEPS),
            "--additional-config",
            json.dumps(
                {"phased_profiler_decode_only_kv_len_threshold": PROFILE_DECODE_MIN_KV}
            ),
        ]
    return cmd


def client_cmd(shape: dict, result: Path) -> list[str]:
    """The recipe's client as `vllm bench serve`: [r * len, len] is the symmetric
    range around len * (1 + r) / 2 with ratio (1 - r) / (1 + r)."""
    r = float(shape["ratio"])
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--model",
        MODEL,
        "--tokenizer",
        str(TOKENIZER_DIR),
        "--trust-remote-code",
        "--host",
        "localhost",
        "--port",
        str(PORT),
        "--dataset-name",
        "random",
        "--random-input-len",
        str(round(shape["isl"] * (1 + r) / 2)),
        "--random-output-len",
        str(round(shape["osl"] * (1 + r) / 2)),
        "--random-range-ratio",
        f"{(1 - r) / (1 + r):.6f}",
        "--num-prompts",
        str(shape["num_prompts"]),
        "--max-concurrency",
        str(shape["concurrency"]),
        "--request-rate",
        "inf",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--save-result",
        "--ignore-eos",
        "--result-filename",
        str(result),
        "--temperature",
        "0",
        "--seed",
        "42",
    ]


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
    d = base / "qwen35_397b_serve" / job / arm
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def _pull_tokenizer() -> None:
    """Copy the checkpoint's config and tokenizer files, not its weights, once
    per pod; later arms reuse them."""
    done = TOKENIZER_DIR / ".complete"
    if done.exists():
        return
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gsutil",
        "-q",
        "-m",
        "rsync",
        "-r",
        "-x",
        r".*\.safetensors$",
        CHECKPOINT_URI,
        str(TOKENIZER_DIR),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"tokenizer pull failed: {r.stderr[-2000:]}")
    say(
        f"tokenizer files at {TOKENIZER_DIR}: {sorted(p.name for p in TOKENIZER_DIR.iterdir())}"
    )
    done.write_text("ok\n")


def setup():
    from perfkit import scorer as pk_scorer

    root = Path.cwd().resolve()
    tree = pk_scorer.git_tree(root)
    arm = os.environ.get("PERFKIT_ARM", "arm")
    say(f"arm {arm} root {root} tree {tree or '(not a git tree)'}")
    _pull_tokenizer()
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
    env.update(
        {
            "HOME": str(home),
            "TMPDIR": str(tmp),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "TORCH_TPU_CACHE_DIR": str(home / "torch_tpu_cache"),
            "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": str(home / "torch_tpu_cache"),
            "HF_HOME": str(home / "hf"),
            "HF_HUB_OFFLINE": "1",
        }
    )
    return env


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
    with contextlib.suppress(OSError):
        os.killpg(proc.pid, signal.SIGKILL)
    for pat in ("vllm serve", "VLLM::", "vllm.entrypoints"):
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
    time.sleep(5)
    for f in Path("/tmp").glob("libtpu_lockfile*"):
        f.unlink(missing_ok=True)


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://localhost:{PORT}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
        return json.loads(r.read())


def _long_prompts() -> dict:
    """id -> prompt token ids: one haystack, cut to each of LONG_TOKENS by the
    server's tokenizer and closed with the same question."""
    words = (
        "amber",
        "basalt",
        "cobalt",
        "dune",
        "ember",
        "fjord",
        "granite",
        "harbor",
        "indigo",
        "juniper",
        "kelp",
        "lagoon",
        "meadow",
        "nickel",
        "onyx",
        "pumice",
    )
    lines = [
        f"Record {i:04d}: the code word for locker {i} is {words[(i * 7) % 16]}-{i % 97}."
        for i in range(1, 1000)
    ]
    question = (
        "\n\nQuestion: according to the records above, what is the code word for "
        "locker 17? Answer:"
    )
    head = _post(
        "/tokenize",
        {"model": MODEL, "prompt": "\n".join(lines), "add_special_tokens": True},
    )["tokens"]
    tail = _post(
        "/tokenize", {"model": MODEL, "prompt": question, "add_special_tokens": False}
    )["tokens"]
    out = {}
    for n in LONG_TOKENS:
        if len(head) + len(tail) < n:
            raise RuntimeError(
                f"the haystack is {len(head) + len(tail)} tokens, short of {n}"
            )
        out[f"long{n}"] = head[: n - len(tail)] + tail
    return out


def _complete(prompt, logprobs: bool) -> dict:
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": SUITE_TOKENS,
        "temperature": 0.0,
        "seed": 0,
        "return_tokens_as_token_ids": True,
    }
    if logprobs:
        body["logprobs"] = 5
    choice = _post("/v1/completions", body)["choices"][0]
    rec = {"text": choice["text"]}
    if logprobs:
        lp = choice["logprobs"]
        rec["tokens"] = lp["tokens"]
        rec["token_logprobs"] = lp["token_logprobs"]
    return rec


def _token_id(tok: str) -> float:
    """`token_id:<n>`, as return_tokens_as_token_ids spells a token, to n."""
    if not tok.startswith("token_id:"):
        raise ValueError(f"token {tok!r} is not spelled as a token id")
    return float(tok.split(":", 1)[1])


def _capture(art: Path) -> tuple[str, dict]:
    """(checksum, first_tokens) of the suite's serial greedy outputs, with and
    without logprobs. first_tokens is {set: {prompt id: [token id, logprob]}}."""
    sets = {
        "default": {f"s{i:02d}": p for i, p in enumerate(SUITE)},
        "long-context": _long_prompts(),
    }
    outs: dict = {"logprobs": {}, "nologprobs": {}}
    first: dict = {}
    for mode in outs:
        for set_name, prompts in sets.items():
            for pid, prompt in prompts.items():
                rec = _complete(prompt, mode == "logprobs")
                outs[mode][f"{set_name}/{pid}"] = rec
                if mode == "logprobs":
                    first.setdefault(set_name, {})[pid] = [
                        _token_id(rec["tokens"][0]),
                        float(rec["token_logprobs"][0]),
                    ]
    (art / "fingerprint.json").write_text(json.dumps(outs, indent=1))
    digest = hashlib.sha256(json.dumps(outs, sort_keys=True).encode()).hexdigest()
    return "sha256:" + digest, first


def _sweep(shape: dict, env: dict, art: Path, label: str, extra: tuple = ()) -> dict:
    result = art / f"{label}.json"
    cmd = client_cmd(shape, result) + list(extra)
    say(f"{label}: {shlex.join(cmd)}")
    t0 = time.time()
    with open(art / f"{label}.log", "w") as log:
        rc = subprocess.call(
            cmd,
            env=env,
            cwd=art,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=SWEEP_TIMEOUT_S,
        )
    if rc != 0 or not result.exists():
        raise RuntimeError(f"{label} exited {rc}; see {art / f'{label}.log'}")
    d = json.loads(result.read_text())
    say(
        f"{label}: {time.time() - t0:.0f}s, {d.get('completed')} completed, "
        f"{d.get('output_throughput', 0):.1f} output tok/s, median TTFT "
        f"{d.get('median_ttft_ms', 0):.0f} ms, median TPOT {d.get('median_tpot_ms', 0):.2f} ms"
    )
    return d


def _profile_run(cfg: dict, env: dict, art: Path) -> bool:
    """The profile run: one full wave of the sweep (`concurrency` prompts, as
    run_benchmarks.sh's profile run sends) with `--profile`. Whether the phased
    profiler wrote a capture."""
    shape = dict(cfg, num_prompts=cfg["concurrency"])
    try:
        _sweep(shape, env, art, "profile", ("--profile",))
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        say(f"PROFILE RUN FAILED: {e}")
        return False
    traces = sorted(
        str(p.relative_to(art)) for p in (art / "profile").rglob("*.xplane.pb")
    )
    say(f"profile run: {len(traces)} traces: {traces}")
    return bool(traces)


def _publish(art: Path) -> bool:
    """Copy the arm's artifacts to the job's results bucket, beside the pod's
    score.json. Whether they arrived."""
    bucket, pod = os.environ.get("RESULTS_URI"), os.environ.get("POD_NAME")
    if not (bucket and pod):
        say(f"no results bucket in this environment; the artifacts stay in {art}")
        return False
    job = os.environ.get("PERFKIT_JOB_ID", "job").replace("/", "-")
    dest = f"{bucket}/{pod}/artifacts/{job}/{art.parent.name}/"
    r = subprocess.run(
        ["gsutil", "-q", "-m", "cp", "-r", str(art), dest],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        say(f"ARTIFACT UPLOAD TO {dest} FAILED: {r.stderr[-2000:]}")
        return False
    say(f"artifacts at {dest}{art.name}/")
    return True


def _timed(tputs: list) -> tuple[float, int]:
    """(the throughput this arm reports, its timing samples): the control's
    median, any other arm's lower median, and the sweeps at least that fast."""
    control = bool(os.environ.get("PERFKIT_ARM_CONTROL"))
    reported = statistics.median(tputs) if control else statistics.median_low(tputs)
    return reported, sum(1 for t in tputs if t >= reported)


def measure(handle, name, cfg):
    art = handle.scratch / name
    art.mkdir(parents=True, exist_ok=True)
    env = _env(handle)
    profiling = bool(os.environ.get(GOLDEN_ALONE_ENV))
    if profiling:
        env.update(PROFILE_ENV)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util as u; print(u.find_spec('vllm_torchtpu').origin)",
        ],
        env=env,
        cwd=art,
        capture_output=True,
        text=True,
    )
    served = probe.stdout.strip()
    if not served.startswith(str(handle.root) + os.sep):
        raise RuntimeError(
            f"the server would import vllm_torchtpu from {served or '?'}, "
            f"not from this arm's root {handle.root}"
        )
    cmd = server_cmd(art / "profile" if profiling else None)
    say(f"[cmd] {shlex.join(cmd)}")
    t0 = time.time()
    log = open(art / "server.log", "w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=art,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        while not _healthy():
            if proc.poll() is not None:
                tail = (art / "server.log").read_text()[-3000:]
                raise RuntimeError(
                    f"server exited {proc.returncode} during startup:\n{tail}"
                )
            if time.time() - t0 > READY_TIMEOUT_S:
                raise RuntimeError(f"server not ready after {READY_TIMEOUT_S}s")
            time.sleep(5)
        boot_s = time.time() - t0
        say(f"server ready in {boot_s:.0f}s")
        _sweep(cfg["warmup"], env, art, "warmup")
        sweeps = [
            _sweep(cfg, env, art, f"sweep{k}") for k in range(1, int(cfg["sweeps"]) + 1)
        ]
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode} during the sweeps")
        try:
            checksum, first = _capture(art)
        except (
            OSError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            RuntimeError,
        ) as e:
            say(f"fingerprint suite failed: {e}")
            checksum, first = "", {}
        profiled = _profile_run(cfg, env, art) if profiling else False
    finally:
        _stop(proc)
        log.close()
        published = _publish(art)

    tputs = [float(s.get("output_throughput") or 0.0) for s in sweeps]
    tput, samples = _timed(tputs)
    completed = min(int(s.get("completed") or 0) for s in sweeps)
    lost = completed < int(cfg["num_prompts"])
    if lost or not checksum:
        say(
            f"marking incorrect: completed {completed} of {cfg['num_prompts']}, "
            f"fingerprint {'read' if checksum else 'missing'}"
        )

    def med(key):
        return statistics.median(float(s.get(key) or 0.0) for s in sweeps)

    metrics = {
        k: med(k)
        for k in (
            "median_ttft_ms",
            "median_tpot_ms",
            "p99_ttft_ms",
            "p99_tpot_ms",
            "duration",
            "total_input_tokens",
            "total_output_tokens",
        )
    }
    metrics.update(
        {
            "output_throughput": tput,
            "completed": completed,
            "boot_s": boot_s,
            "timing_rounds": samples,
            "artifacts_published": float(published),
        }
    )
    for k, t in enumerate(tputs, 1):
        metrics[f"sweep{k}_output_throughput"] = t
    if profiling:
        metrics["profile_captured"] = float(profiled)
    if len(tputs) > 1:
        metrics["sweep_spread"] = (max(tputs) - min(tputs)) / statistics.median(tputs)
    say(
        f"timed sweeps {tputs}: reported {tput:.1f} output tok/s, {samples} timing samples"
    )
    return {
        "latency_us": 1e6 / tput if tput > 0 else 0.0,
        "worst_error": 1.0 if (lost or not checksum) else 0.0,
        "checksum": checksum,
        "first_tokens": first,
        "metrics": metrics,
        "dispatched": f"serve/{handle.tree}/{name}",
        "binary_hash": "sha256:" + (handle.tree or "unknown"),
    }
