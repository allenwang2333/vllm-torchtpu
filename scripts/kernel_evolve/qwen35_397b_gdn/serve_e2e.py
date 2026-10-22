"""Served-model campaign scorer: Qwen3.5-397B-A17B-FP8, TP1 x DP8 + EP, one tpu7x 2x2x1 node.

A `perfkit scorer run` benchmark (`perfkit/scorer_arm.py` is the interface) for
`perfkit cluster submit`, whose pod holds each arm's `src/vllm_torchtpu` and
puts it first on the path. Each arm boots `vllm serve` from its own tree with
the reproduce recipe's server flags and environment, runs the recipe's sweep
(1024 in / 8192 out, lengths in [0.8 len, len], 512 at once, 640 requests,
greedy, ignore-eos, seed 42) and reports output throughput as microseconds per
output token, so control / arm > 1 is faster. The serve and client commands
live here, not in the arm: a candidate changes the code behind the server and
nothing about what is served or how it is measured.

Departures from the recipe's text, none of them inside a timed sweep:

- The model is the cluster's copy of the checkpoint, streamed
  (`--load-format runai_streamer`, each rank fetching only its experts with
  `--enable-ep-weight-filter`): the pod has no disk that holds 406 GB. The
  server answers to the model's own name. Each of the eight API servers copies
  the checkpoint's config and tokenizer files into one shared directory before
  reading them, and they overwrite each other's files while others read them;
  `SITECUSTOMIZE` makes the copy happen once, under a lock.
- The recipe's client draws lengths in [0.8 len, len]. `vllm bench serve` draws
  symmetrically, so it gets the midpoint and the ratio that span the same range,
  as the repo's run_benchmarks.sh does for RANGE_RATIO_STYLE=min.

Timing: a short warmup sweep, then `sweeps` timed sweeps on the same server.
The control reports their median, any other arm their lower median, and the
sweeps at least that fast are its `timing_rounds`.

Correctness: a served model has no fp32 reference. After the sweeps each arm
sends a fixed prompt suite one request at a time, greedily: short prompts, and
long ones cut to fixed token counts so the GDN and attention paths run past
short contexts. `checksum` digests the generated tokens and `first_tokens`
carries each prompt's first token and logprob for the harness's first-token
check. `worst_error` is 1.0 when a sweep lost requests or the suite could not
be read, else 0.0.

Accuracy: an arm whose output may differ from golden's runs the serving
config's eval tasks (`_eval_why`) on the server that produced it and reports
their scores as `accuracy:<task>` metrics, which the harness compares with
golden's.

A job that measures golden alone (scoping) arms the phased profiler after
everything above and takes a profile run: one wave of the sweep with
`--profile`. Each arm's artifacts (server log, sweeps, fingerprint, evals,
capture) go to the job's results bucket.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

#: The kernel's directory relative to the arm's `src`: the tree `perfkit
#: cluster` records as the kernel's.
KERNEL_REL = "vllm_torchtpu/kernels/gdn/v3"

MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
PORT = 8000
CONFIG = "qwen3.5-397b-fp8-dp8-ep-isl1024-osl8192-c512"

#: The recipe's one shape. `ratio` is its client's: lengths in [ratio*len, len].
CONFIGS = {
    CONFIG: {
        "isl": 1024,
        "osl": 8192,
        "ratio": 0.8,
        "concurrency": 512,
        "num_prompts": 640,
        "sweeps": 2,
        "warmup": {"isl": 1024, "osl": 128, "ratio": 0.8, "concurrency": 512,
                   "num_prompts": 512},
    },
}

CHECKPOINT_URI = "gs://cloud-devkit/ckpt/hf/Qwen/Qwen3.5-397B-A17B-FP8"
#: The checkpoint's config and tokenizer files, copied once per pod, for the
#: benchmark client's tokenizer.
MODEL_DIR = Path("/dev/shm/qwen35_397b_model_files")

#: Put first on the server's PYTHONPATH only. vLLM's ObjectStorageModel.pull_files
#: re-downloads into a directory every process of the server shares, so a
#: process can read config.json while another truncates it ("is not a valid
#: JSON file", which stopped an API server on this config). This serializes the
#: pull and lets each later process find it done, as runai_model_streamer's own
#: ObjectStorageModel does. Nothing it touches runs after the server is ready.
SITECUSTOMIZE = """\
import fcntl, hashlib, importlib.abc, importlib.util, os, sys

_MODULE = "vllm.transformers_utils.runai_utils"


def _patch(module):
    cls = module.ObjectStorageModel
    pull = cls.pull_files

    def pull_files(self, model_path="", allow_pattern=None, ignore_pattern=None):
        key = repr((model_path, allow_pattern, ignore_pattern)).encode()
        done = os.path.join(self.dir, ".pulled-" + hashlib.sha1(key).hexdigest()[:12])
        with open(self.dir + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not os.path.exists(done):
                pull(self, model_path, allow_pattern, ignore_pattern)
                open(done, "w").close()

    cls.pull_files = pull_files


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _MODULE:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())
"""
#: Host memory each engine's streamer may buffer; eight engines stream at once.
STREAM_MEMORY_LIMIT = 16 * 1024**3

#: The recipe's server environment.
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
#: Inherited variables the recipe does not set; dropped from every arm.
UNSET_ENV = (
    "ASYNC_SCHEDULING", "ATTENTION_BACKEND", "BENCHMARK_EXTRA_BODY", "BENCHMARK_TEMPERATURE",
    "BENCHMARK_WARMUP_RUNS", "CAPTURE_PROFILE", "DATASET_NAME", "DATASET_PATH",
    "ENABLE_PREFIX_CACHING", "EXTRA_BENCH_ARGS", "EXTRA_LM_EVAL_MODEL_ARGS", "EXTRA_SERVE_ARGS",
    "LM_EVAL_GEN_KWARGS", "MODEL_URI", "PORT", "PROFILE_NUM_PROMPTS", "TOKENIZER",
    "USE_PHASED_PROFILER", "XLA_FLAGS", "LIBTPU_INIT_ARGS",
)

#: `perfkit scorer run` sets these (perfkit/scorer.py); the harness may set
#: GOLDEN_ENV. See `_eval_why`.
GOLDEN_ALONE_ENV = "PERFKIT_GOLDEN_ALONE"
ACCURACY_ENV = "PERFKIT_ACCURACY"
GOLDEN_ENV = "PERFKIT_GOLDEN"

#: The phased profiler, in a golden-alone job only.
PROFILE_ENV = {"USE_PHASED_PROFILER": "1", "HOST_TRACER_LEVEL": "0", "PYTHON_TRACER_LEVEL": "0"}
#: Steps per phase. At the profiler's default of 15 an 8-engine server has hung
#: in trace collection on this cluster.
PROFILE_STEPS = 5
#: decode_only is captured once every sequence holds this many tokens: mid-way
#: through a 1k-in / 6.5k-8k-out request, the sweep's typical context.
PROFILE_DECODE_MIN_KV = 4096

#: The serving config's eval tasks and allowed drop (run_eval_flow.sh's
#: defaults, and EVAL_TOLERANCE and MMLU_PRO_DISABLE_MULTITURN_ARGS in
#: scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp1-dp8-ep.sh).
EVAL_TASKS = ("mmlu_llama", "mmlu_pro")
EVAL_TOLERANCE = 0.02
EVAL_CONFIG_ENV = {"MMLU_PRO_DISABLE_MULTITURN_ARGS": "true"}
EVAL_TIMEOUT_S = 3600

READY_TIMEOUT_S = 4500
SWEEP_TIMEOUT_S = 3000
REQUEST_TIMEOUT_S = 600

#: Short prompts for the output fingerprint, sent one at a time, greedy.
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
#: Long prompts, as token counts: numbered records and a question about one of
#: them, cut by the server's tokenizer. The largest leaves room for
#: SUITE_TOKENS under --max-model-len=10240.
LONG_TOKENS = (2048, 4096, 6144, 10000)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] qwen35_397b_gdn: {msg}", file=sys.stderr, flush=True)


def server_cmd(profile_dir: Path | None) -> list[str]:
    """The recipe's `vllm serve` line, loading as the module docstring says, plus
    the phased profiler's settings when `profile_dir` is given."""
    cmd = [
        "vllm", "serve",
        f"--model={CHECKPOINT_URI}",
        "--served-model-name", MODEL,
        "--load-format", "runai_streamer",
        "--enable-ep-weight-filter",
        "--model-loader-extra-config", json.dumps({"memory_limit": STREAM_MEMORY_LIMIT}),
        "--tensor-parallel-size=1",
        "--data-parallel-size=8",
        "--max-model-len=10240",
        "--max-num-batched-tokens=1024",
        "--max-num-seqs=64",
        "--port", str(PORT),
        "--async-scheduling",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization=0.92",
        "--kv-cache-dtype=fp8",
        "--enable-expert-parallel",
        "--quantization", "fp8",
        "--attention-backend", "CUSTOM",
        "--block-size", "256",
        "--language-model-only",
        '--limit-mm-per-prompt={"image":0,"video":0}',
        '--default-chat-template-kwargs={"enable_thinking":false}',
    ]
    if profile_dir is not None:
        cmd += [
            "--profiler-config.profiler", "torch",
            "--profiler-config.torch_profiler_dir", str(profile_dir),
            "--profiler-config.ignore_frontend", "true",
            "--profiler-config.max_iterations", str(PROFILE_STEPS),
            "--additional-config",
            json.dumps({"phased_profiler_decode_only_kv_len_threshold": PROFILE_DECODE_MIN_KV}),
        ]
    return cmd


def client_cmd(shape: dict, result: Path) -> list[str]:
    """The recipe's client as `vllm bench serve`: [r*len, len] is the symmetric
    range around len*(1+r)/2 with ratio (1-r)/(1+r)."""
    r = float(shape["ratio"])
    return [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--model", MODEL,
        "--tokenizer", str(MODEL_DIR),
        "--host", "localhost",
        "--port", str(PORT),
        "--dataset-name", "random",
        "--random-input-len", str(round(shape["isl"] * (1 + r) / 2)),
        "--random-output-len", str(round(shape["osl"] * (1 + r) / 2)),
        "--random-range-ratio", f"{(1 - r) / (1 + r):.6f}",
        "--num-prompts", str(shape["num_prompts"]),
        "--max-concurrency", str(shape["concurrency"]),
        "--request-rate", "inf",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--save-result",
        "--ignore-eos",
        "--result-filename", str(result),
        "--temperature", "0",
        "--seed", "42",
    ]


class Handle:
    def __init__(self, root: Path, tree: str, scratch: Path):
        self.root = root
        self.tree = tree
        self.scratch = scratch


def _job_dir() -> Path:
    """Per job, outside every arm root: an arm root must not change while the
    driver hashes it."""
    base = Path(os.environ.get("WORK_DIR") or os.environ.get("TMPDIR") or "/tmp")
    job = re.sub(r"[^A-Za-z0-9._-]+", "-", os.environ.get("PERFKIT_JOB_ID", "job"))
    return base / "qwen35_397b_gdn" / job


def _pull_model_files() -> None:
    """Copy the checkpoint's files other than its weights to MODEL_DIR, once per
    pod and under a lock, so every later arm and process reads a finished copy."""
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{MODEL_DIR}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        done = MODEL_DIR / ".complete"
        if done.exists():
            return
        MODEL_DIR.mkdir(exist_ok=True)
        cmd = ["gsutil", "-q", "-m", "rsync", "-r", "-x", r".*\.safetensors$|^\.cache/.*",
               CHECKPOINT_URI, str(MODEL_DIR)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"model files pull failed: {r.stderr[-2000:]}")
        json.loads((MODEL_DIR / "config.json").read_text())
        say(f"model files at {MODEL_DIR}: {sorted(p.name for p in MODEL_DIR.iterdir())}")
        done.write_text("ok\n")


def setup():
    from perfkit import scorer as pk_scorer

    root = Path.cwd().resolve()
    tree = pk_scorer.git_tree(root)
    arm = os.environ.get("PERFKIT_ARM", "arm")
    say(f"arm {arm} root {root} tree {tree or '(not a git tree)'}")
    _pull_model_files()
    scratch = _job_dir() / arm
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    return Handle(root, tree, scratch)


def origin(handle) -> str:
    """The vllm_torchtpu the server will import, found on this process's path
    (the arm's src first) without importing it: an import would open the TPU."""
    spec = importlib.util.find_spec("vllm_torchtpu")
    return spec.origin if spec and spec.origin else ""


def _env(handle) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in UNSET_ENV}
    env.update(SERVER_ENV)
    home = handle.scratch / "home"
    # Short: vLLM puts its zmq IPC sockets under TMPDIR, capped at 107 bytes.
    tmp = Path("/tmp") / ("q" + hashlib.sha1(str(handle.scratch).encode()).hexdigest()[:8])
    for d in (home, tmp):
        d.mkdir(parents=True, exist_ok=True)
    # Every compile cache starts empty in every arm, as in the recipe's fresh
    # container, so no arm loads another arm's executables and all boot alike.
    env.update({
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "VLLM_CACHE_ROOT": str(home / ".cache" / "vllm"),
        "HF_HOME": str(home / "hf"),
    })
    return env


def _server_env(env: dict, handle) -> dict:
    """`env` with SITECUSTOMIZE first on the path, for the server's processes only."""
    shim = handle.scratch / "pyshim"
    shim.mkdir(exist_ok=True)
    (shim / "sitecustomize.py").write_text(SITECUSTOMIZE)
    return {**env, "PYTHONPATH": os.pathsep.join(filter(None, [str(shim), env.get("PYTHONPATH")]))}


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
            proc.wait(timeout=90)
    with contextlib.suppress(OSError):
        os.killpg(proc.pid, signal.SIGKILL)
    for pat in ("vllm serve", "VLLM::", "vllm.entrypoints"):
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
    time.sleep(10)
    for f in Path("/tmp").glob("libtpu_lockfile*"):
        f.unlink(missing_ok=True)


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://localhost:{PORT}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
        return json.loads(r.read())


def _long_prompts() -> dict:
    """id -> prompt token ids: one haystack cut to each of LONG_TOKENS by the
    server's tokenizer, closed by the same question."""
    words = ("amber", "basalt", "cobalt", "dune", "ember", "fjord", "granite", "harbor",
             "indigo", "juniper", "kelp", "lagoon", "meadow", "nickel", "onyx", "pumice")
    lines = [f"Record {i:04d}: the code word for locker {i} is {words[(i * 7) % 16]}-{i % 97}."
             for i in range(1, 1000)]
    question = ("\n\nQuestion: according to the records above, what is the code word for "
                "locker 17? Answer:")
    head = _post("/tokenize", {"model": MODEL, "prompt": "\n".join(lines),
                               "add_special_tokens": True})["tokens"]
    tail = _post("/tokenize", {"model": MODEL, "prompt": question,
                               "add_special_tokens": False})["tokens"]
    out = {}
    for n in LONG_TOKENS:
        if len(head) + len(tail) < n:
            raise RuntimeError(f"the haystack is {len(head) + len(tail)} tokens, short of {n}")
        out[f"long{n}"] = head[: n - len(tail)] + tail
    return out


def _complete(prompt, logprobs: bool) -> dict:
    body = {"model": MODEL, "prompt": prompt, "max_tokens": SUITE_TOKENS, "temperature": 0.0,
            "seed": 0, "return_tokens_as_token_ids": True}
    if logprobs:
        body["logprobs"] = 5
    choice = _post("/v1/completions", body)["choices"][0]
    rec = {"text": choice["text"]}
    if logprobs:
        rec["tokens"] = choice["logprobs"]["tokens"]
        rec["token_logprobs"] = choice["logprobs"]["token_logprobs"]
    return rec


def _token_id(tok: str) -> float:
    """`token_id:<n>`, as return_tokens_as_token_ids spells a token, to n."""
    if not tok.startswith("token_id:"):
        raise ValueError(f"token {tok!r} is not spelled as a token id")
    return float(tok.split(":", 1)[1])


def _fingerprint(art: Path) -> tuple[str, dict]:
    """(checksum, first_tokens) of the suite's serial greedy outputs, with and
    without logprobs; first_tokens is {set: {prompt id: [token id, logprob]}}.
    The checksum covers the generated tokens only, not the logprobs."""
    sets = {"default": {f"s{i:02d}": p for i, p in enumerate(SUITE)},
            "long-context": _long_prompts()}
    outs: dict = {"logprobs": {}, "nologprobs": {}}
    first: dict = {}
    for mode in outs:
        for set_name, prompts in sets.items():
            for pid, prompt in prompts.items():
                rec = _complete(prompt, mode == "logprobs")
                outs[mode][f"{set_name}/{pid}"] = rec
                if mode == "logprobs":
                    first.setdefault(set_name, {})[pid] = [
                        _token_id(rec["tokens"][0]), float(rec["token_logprobs"][0])]
    (art / "fingerprint.json").write_text(json.dumps(outs, indent=1))
    tokens = {m: {k: r.get("tokens") or r["text"] for k, r in v.items()} for m, v in outs.items()}
    digest = hashlib.sha256(json.dumps(tokens, sort_keys=True).encode()).hexdigest()
    return "sha256:" + digest, first


def _sweep(shape: dict, env: dict, art: Path, label: str, extra: tuple = ()) -> dict:
    result = art / f"{label}.json"
    cmd = client_cmd(shape, result) + list(extra)
    say(f"{label}: {shlex.join(cmd)}")
    t0 = time.time()
    with open(art / f"{label}.log", "w") as log:
        rc = subprocess.call(cmd, env=env, cwd=art, stdout=log, stderr=subprocess.STDOUT,
                             timeout=SWEEP_TIMEOUT_S)
    if rc != 0 or not result.exists():
        raise RuntimeError(f"{label} exited {rc}; see {art / f'{label}.log'}")
    d = json.loads(result.read_text())
    say(f"{label}: {time.time() - t0:.0f}s, {d.get('completed')} completed, "
        f"{d.get('output_throughput', 0):.1f} output tok/s, median TTFT "
        f"{d.get('median_ttft_ms', 0):.0f} ms, median TPOT {d.get('median_tpot_ms', 0):.2f} ms")
    return d


def _golden_env() -> dict:
    """What the harness says golden did on CONFIG (GOLDEN_ENV), {} when it says nothing."""
    try:
        known = json.loads(os.environ.get(GOLDEN_ENV) or "null")
    except ValueError:
        return {}
    entry = known.get(CONFIG) if isinstance(known, dict) else None
    return entry if isinstance(entry, dict) else {}


def _known_golden_outputs() -> list | None:
    """Golden's output checksums on CONFIG, from the harness or from the control
    arm that ran earlier in this job. None when neither says."""
    outs = _golden_env().get("outputs")
    found = [o for o in outs if isinstance(o, str)] if isinstance(outs, list) else []
    control = _job_dir() / "control_checksum"
    if control.exists():
        found.append(control.read_text().strip())
    return found or None


def _eval_why(checksum: str) -> str:
    """Why this arm runs the eval tasks, "" when it does not.

    Asked for, every arm does. The control does unless the harness says golden's
    scores are already known, so a job that carries golden refreshes them. Any
    other arm does unless its output is one of golden's, and does when golden's
    output is not known in this job. An arm with no checksum is refused the
    record whatever it scores, and runs none."""
    if os.environ.get(ACCURACY_ENV):
        return "asked for"
    if not checksum:
        return ""
    if os.environ.get("PERFKIT_ARM_CONTROL"):
        return "" if _golden_env().get("scored") else "golden's eval scores for this job"
    golden = _known_golden_outputs()
    if golden is None:
        return "golden's output is not known in this job"
    return "" if checksum in golden else "its output differs from golden's"


def _evals(art: Path) -> dict:
    """The eval tasks against this arm's live server, as `accuracy:` metrics. A
    task that fails is said and left out, which the harness reads as not measured."""
    from perfkit import lmeval
    from perfkit.targets.vllm_torchtpu import eval_plan

    parsed: dict = {}
    for task in EVAL_TASKS:
        out = art / "eval" / task
        out.mkdir(parents=True, exist_ok=True)
        plan = eval_plan(task, model=MODEL, base_url=f"http://localhost:{PORT}", output_dir=out,
                         config_env=EVAL_CONFIG_ENV)
        cmd = lmeval.argv(plan, python=sys.executable)
        say(f"eval {task}: {shlex.join(cmd)}")
        t0 = time.time()
        with open(out / "lm_eval.log", "w") as log:
            try:
                rc = subprocess.call(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, **plan.extra_env},
                                     timeout=EVAL_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                rc = "timeout"
        if rc != 0:
            say(f"EVAL {task} FAILED ({rc}) after {time.time() - t0:.0f}s; "
                f"see {out / 'lm_eval.log'}")
            continue
        parsed.update(lmeval.parse_dir(out))
        say(f"eval {task}: {time.time() - t0:.0f}s")
    m = lmeval.accuracy_metrics(parsed, EVAL_TASKS, EVAL_TOLERANCE)
    say("accuracy: " + (", ".join(f"{k} {v:.4f}" for k, v in m.items()) or "none"))
    return m


def _profile_run(cfg: dict, env: dict, art: Path) -> bool:
    """One wave of the sweep (`concurrency` prompts) with `--profile`. Whether
    the phased profiler wrote a capture."""
    shape = dict(cfg, num_prompts=cfg["concurrency"])
    try:
        _sweep(shape, env, art, "profile", ("--profile",))
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        say(f"PROFILE RUN FAILED: {e}")
    time.sleep(30)  # the trace is written after stop_profile returns
    traces = sorted(str(p.relative_to(art)) for p in (art / "profile").rglob("*.xplane.pb"))
    say(f"profile run: {len(traces)} traces: {traces}")
    return bool(traces)


def _publish(art: Path) -> bool:
    """Copy the arm's artifacts beside the pod's results. Whether they arrived."""
    bucket, pod = os.environ.get("RESULTS_URI"), os.environ.get("POD_NAME")
    if not (bucket and pod):
        say(f"no results bucket in this environment; the artifacts stay in {art}")
        return False
    dest = f"{bucket}/{pod}/artifacts/{_job_dir().name}/{art.parent.name}/"
    r = subprocess.run(["gsutil", "-q", "-m", "cp", "-r", str(art), dest],
                       capture_output=True, text=True)
    if r.returncode != 0:
        say(f"ARTIFACT UPLOAD TO {dest} FAILED: {r.stderr[-2000:]}")
        return False
    say(f"artifacts at {dest}{art.name}/")
    return True


def _timed(tputs: list) -> tuple[float, int]:
    """(reported throughput, timing samples): the control's median, any other
    arm's lower median, and the sweeps at least that fast."""
    control = bool(os.environ.get("PERFKIT_ARM_CONTROL"))
    reported = statistics.median(tputs) if control else statistics.median_low(tputs)
    return reported, sum(1 for t in tputs if t >= reported)


def _kv_cache_tokens(log: Path) -> float:
    """The KV cache size the server logged, in tokens; 0 when absent."""
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log.read_text(errors="replace"))
    return float(m[-1].replace(",", "")) if m else 0.0


def measure(handle, name, cfg):
    art = handle.scratch / name
    art.mkdir(parents=True, exist_ok=True)
    env = _env(handle)
    profiling = bool(os.environ.get(GOLDEN_ALONE_ENV))
    if profiling:
        env.update(PROFILE_ENV)
    probe = subprocess.run(
        [sys.executable, "-c",
         "import importlib.util as u; print(u.find_spec('vllm_torchtpu').origin)"],
        env=env, cwd=art, capture_output=True, text=True)
    served = probe.stdout.strip()
    if not served.startswith(str(handle.root) + os.sep):
        raise RuntimeError(f"the server would import vllm_torchtpu from {served or '?'}, "
                           f"not from this arm's root {handle.root}")
    cmd = server_cmd(art / "profile" if profiling else None)
    say(f"[cmd] {shlex.join(cmd)}")
    t0 = time.time()
    checksum, first, accuracy, eval_why, profiled = "", {}, {}, "", False
    log = open(art / "server.log", "w")
    proc = subprocess.Popen(cmd, env=_server_env(env, handle), cwd=art, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while not _healthy():
            if proc.poll() is not None:
                tail = (art / "server.log").read_text(errors="replace")[-3000:]
                raise RuntimeError(f"server exited {proc.returncode} during startup:\n{tail}")
            if time.time() - t0 > READY_TIMEOUT_S:
                raise RuntimeError(f"server not ready after {READY_TIMEOUT_S}s")
            time.sleep(5)
        boot_s = time.time() - t0
        say(f"server ready in {boot_s:.0f}s")
        _sweep(cfg["warmup"], env, art, "warmup")
        sweeps = [_sweep(cfg, env, art, f"sweep{k}") for k in range(1, int(cfg["sweeps"]) + 1)]
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode} during the sweeps")
        try:
            checksum, first = _fingerprint(art)
        except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as e:
            say(f"FINGERPRINT SUITE FAILED: {e}")
        say(f"output checksum {checksum or '(none)'}")
        if checksum and os.environ.get("PERFKIT_ARM_CONTROL"):
            (_job_dir() / "control_checksum").write_text(checksum + "\n")
        eval_why = _eval_why(checksum)
        if eval_why:
            say(f"running the eval tasks: {eval_why}")
            accuracy = _evals(art)
        if profiling:
            profiled = _profile_run(cfg, env, art)
    finally:
        _stop(proc)
        log.close()
        published = _publish(art)

    tputs = [float(s.get("output_throughput") or 0.0) for s in sweeps]
    tput, samples = _timed(tputs)
    completed = min(int(s.get("completed") or 0) for s in sweeps)
    lost = completed < int(cfg["num_prompts"])
    if lost or not checksum:
        say(f"marking incorrect: completed {completed} of {cfg['num_prompts']}, "
            f"fingerprint {'read' if checksum else 'missing'}")

    def med(key):
        return statistics.median(float(s.get(key) or 0.0) for s in sweeps)

    metrics = {k: med(k) for k in ("median_ttft_ms", "median_tpot_ms", "p99_ttft_ms",
                                   "p99_tpot_ms", "median_itl_ms", "duration",
                                   "total_input_tokens", "total_output_tokens")}
    metrics.update({
        "output_throughput": tput,
        "completed": completed,
        "boot_s": boot_s,
        "timing_rounds": samples,
        "kv_cache_tokens": _kv_cache_tokens(art / "server.log"),
        "artifacts_published": float(published),
        "eval_ran": float(bool(eval_why)),
    })
    metrics.update(accuracy)
    for k, t in enumerate(tputs, 1):
        metrics[f"sweep{k}_output_throughput"] = t
    if len(tputs) > 1:
        metrics["sweep_spread"] = (max(tputs) - min(tputs)) / statistics.median(tputs)
    if profiling:
        metrics["profile_captured"] = float(profiled)
    say(f"timed sweeps {tputs}: reported {tput:.1f} output tok/s, {samples} timing samples")
    return {
        "latency_us": 1e6 / tput if tput > 0 else 0.0,
        "worst_error": 1.0 if (lost or not checksum) else 0.0,
        "checksum": checksum,
        "first_tokens": first,
        "metrics": metrics,
        "dispatched": f"serve/{handle.tree}/{name}",
        "binary_hash": "sha256:" + (handle.tree or "unknown"),
    }
