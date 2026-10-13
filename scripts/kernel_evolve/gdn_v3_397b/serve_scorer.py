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
"""Campaign fitness function for ``kernels/gdn/v3``, scored on the served
Qwen3.5-397B-A17B-FP8.

Each arm is a source tree. The arm's ``vllm serve`` runs the frozen DP8/EP
serve line below and ``vllm bench serve`` replays the frozen random
1024-in / 8192-out, concurrency-512 workload against it. The score is output
token throughput, reported as a latency (``1e6 / tok_per_s``) so that
``control / arm`` above 1 is faster.

Correctness is measured on the same server after the timed sweeps. A fixed
prompt set is decoded greedily with top-5 logprobs, one prompt at a time, and
compared with the control arm of the same job. While two arms have produced
the same tokens their contexts are identical, so the logprobs of the tokens
they chose must agree; at the first token where they part, each arm's choice
must have been a near-tie in the other's distribution. That gap is computed
per prompt in nats; ``worst_error`` is the mean over the prompts
(``--error-stat mean``) or the worst prompt's gap (``--error-stat worst``), and
both are reported in ``metrics``. The mean exists because two boots of the
same tree need not decode identically at this serve line even one prompt at a
time, so a single prompt can part from the control on noise. An arm is also
incorrect when its sweeps complete a different number of requests than the
control's: the workload's length draws are seeded, so that count is fixed.

The prompts are then decoded again with several copies in flight, reaching
multi-sequence decode tiles. Every request must complete; the logprob gap is
reported as ``concurrent_logprob_gap`` and does not decide correctness,
because batch composition differs from boot to boot.

What the server loaded is recorded, not assumed: ``loadlog/sitecustomize.py``
logs the origin of every ``vllm_torchtpu`` module each server process imports.
An arm whose server imported a ``vllm_torchtpu`` file from outside the arm's
tree is refused, and the files it did import are hashed into ``binary_hash``
and ``dispatched``.

A TPU log reporting ``Stall detected`` ends the arm at once: a hung device
otherwise holds the node until the job's deadline.

With ``--profile-dir`` (scoping only) each arm's server carries the phased
profiler, armed for one extra sweep after the correctness capture; it writes
``decode_only/`` and ``prefill_only/`` traces under ``<profile-dir>/<arm>``.

stdout carries exactly one JSON value; everything else goes to stderr. An arm
that cannot be measured is left out, and without the control nothing is
printed at all.

    serve_scorer.py --arm golden=/w/golden:control --arm aa=/w/golden_copy \\
        --arm cand=/w/cand --tolerance <nats> --error-stat mean --out-dir /scratch/out
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOADLOG_DIR = HERE / "loadlog"

MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
PORT = 8000
CONFIG = "qwen35_397b_dp8ep_1k8k_c512"

# The reproduce recipe's environment and serve line, verbatim.
SERVE_ENV = {
    "TPU_ACCELERATOR_TYPE": "tpu7x",
    "MODEL_IMPL_TYPE": "vllm",
    "USE_MOE_SPARSE_CORE": "1",
    "ONEHOT_MOE_PERMUTE_THRESHOLD": "32768",
    "RAGGED_GATED_DELTA_RULE_IMPL": "chunked_kernel_v3_pd",
    "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
}
SERVE_ARGS = [
    f"--model={MODEL}",
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
NUM_PROMPTS = 640
RESULT_FILENAME = "bench.json"
# The recipe's client line verbatim, plus where to send it and what to save.
BENCH_ARGS = [
    "--backend",
    "vllm",
    "--model",
    MODEL,
    "--host",
    "localhost",
    "--port",
    str(PORT),
    "--dataset-name",
    "random",
    "--random-input-len",
    "1024",
    "--random-output-len",
    "8192",
    "--random-range-ratio",
    "0.8",
    "--num-prompts",
    str(NUM_PROMPTS),
    "--max-concurrency",
    "512",
    "--request-rate",
    "inf",
    "--ignore-eos",
    "--seed",
    "42",
    "--percentile-metrics",
    "ttft,tpot,itl,e2el",
    "--save-result",
    "--result-filename",
    RESULT_FILENAME,
]

CONFIG_ARGS = {
    "serve": " ".join(f"{k}={v}" for k, v in SERVE_ENV.items())
    + " vllm serve "
    + " ".join(SERVE_ARGS),
    "bench": "vllm bench serve " + " ".join(BENCH_ARGS),
}

# Medians over the timed sweeps, for the harness to guard or report.
SECONDARY = (
    "median_ttft_ms",
    "median_tpot_ms",
    "median_itl_ms",
    "median_e2el_ms",
    "p99_ttft_ms",
    "p99_tpot_ms",
)

READY_TIMEOUT_S = 3600
BENCH_TIMEOUT_S = 2400
# Where the TPU runtime writes its logs; a stall there ends the arm at once.
TPU_LOG_DIR = Path("/tmp/tpu_logs")
PROFILE_FLUSH_S = 120
REQUEST_TIMEOUT_S = 900
CAPTURE_MAX_TOKENS = 64
CAPTURE_TOP_LOGPROBS = 5


def say(msg: str) -> None:
    print(
        f"[{time.strftime('%H:%M:%S')}] serve_scorer: {msg}",
        file=sys.stderr,
        flush=True,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_tree_hash(root: Path) -> str:
    """The git tree id of a directory's contents, computed without git.

    For an export of a commit (``git archive``, no ignored files) this equals
    ``git rev-parse <commit>^{tree}``.
    """

    def blob(data: bytes) -> bytes:
        return hashlib.sha1(b"blob %d\0" % len(data) + data).digest()

    def tree(d: Path) -> bytes | None:
        entries = []
        for p in d.iterdir():
            if p.name == ".git":
                continue
            if p.is_symlink():
                mode, sort_key, oid = b"120000", p.name, blob(os.readlink(p).encode())
            elif p.is_dir():
                sub = tree(p)
                if sub is None:
                    continue
                mode, sort_key, oid = b"40000", p.name + "/", sub
            else:
                mode = b"100755" if os.access(p, os.X_OK) else b"100644"
                sort_key, oid = p.name, blob(p.read_bytes())
            entries.append(
                (sort_key.encode(), mode + b" " + p.name.encode() + b"\0" + oid)
            )
        if not entries:
            return None
        body = b"".join(e for _, e in sorted(entries))
        return hashlib.sha1(b"tree %d\0" % len(body) + body).digest()

    return tree(root).hex()


# --------------------------------------------------------------------------
# The fixed correctness prompts.

_WORD_TEXT = (
    "river stone market winter signal garden copper engine harbor letter "
    "forest window theory silver bridge orbit lantern meadow pattern cotton "
    "thunder planet cabinet violin glacier compass ladder pepper canyon "
    "festival harvest marble voyage anchor blanket chimney desert falcon "
    "granite island jungle kettle lemon mirror needle oyster pillow quarry "
    "saddle tunnel valley wagon yellow zebra archive beacon candle dolphin"
)
_WORDS = tuple(_WORD_TEXT.split())


def _long_prompt(seed: int, n_words: int) -> str:
    rng = random.Random(seed)
    sentences = []
    while sum(len(s.split()) for s in sentences) < n_words:
        k = rng.randint(6, 14)
        sentences.append(
            " ".join(rng.choice(_WORDS) for _ in range(k)).capitalize() + "."
        )
    return (
        "Read the following notes and then summarize them in one paragraph.\n\n"
        + " ".join(sentences)
        + "\n\nSummary:"
    )


PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the n-th Fibonacci number.\n\ndef fib(n):",
    "Q: A train travels 120 km in 1.5 hours. What is its average speed?\nA:",
    "Explain in two sentences why the sky is blue.",
    "List five prime numbers greater than 100:",
    "Translate to German: 'The weather is nice today and we are going for a walk.'",
    "Once upon a time, in a small village by the sea,",
    "The three laws of thermodynamics are",
    "SELECT name, COUNT(*) FROM orders",
    'def quicksort(arr):\n    """Sort a list in place."""\n',
    "In linear algebra, an eigenvector of a matrix is",
    "Here is a haiku about autumn:\n",
    _long_prompt(1, 300),
    _long_prompt(2, 700),
    _long_prompt(3, 1200),
    _long_prompt(4, 2400),
]


# --------------------------------------------------------------------------
# The server.


def _post(path: str, body: dict | None, timeout: float) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else {}


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(
            f"http://localhost:{PORT}/health", timeout=5
        ) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _arm_pids(marker: str) -> list[int]:
    """Every live process started for this arm, found by its environment marker."""
    pids = []
    needle = f"KE_ARM_MARKER={marker}".encode()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if needle in (d / "environ").read_bytes().split(b"\0"):
                pids.append(int(d.name))
        except OSError:
            continue
    return pids


_STALL_SEEN: dict[Path, int] = {}


def tpu_stalled(since: float) -> bool:
    """Has a TPU log written since `since` reported a stalled device?

    Each file is read from where the previous call stopped."""
    if not TPU_LOG_DIR.is_dir():
        return False
    for f in TPU_LOG_DIR.rglob("*"):
        try:
            if not f.is_file() or f.is_symlink() or f.stat().st_mtime < since:
                continue
            with open(f, "rb") as fh:
                fh.seek(max(0, _STALL_SEEN.get(f, 0) - 64))
                chunk = fh.read()
                _STALL_SEEN[f] = fh.tell()
            if b"Stall detected" in chunk:
                return True
        except OSError:
            continue
    return False


class Server:
    def __init__(self, root: Path, out: Path, name: str, profile_dir: Path | None):
        self.root, self.out, self.name = root, out, name
        self.marker = f"{name}-{os.getpid()}-{time.time_ns()}"
        self.cache = Path(os.environ.get("KE_CACHE_ROOT", "/tmp/ke-cache")) / name
        self.loadlog = out / "loadlog.tsv"
        self.profile_dir = profile_dir
        self.proc: subprocess.Popen | None = None
        self.ready_s = 0.0
        self.started = time.time()

    def env(self) -> dict:
        env = dict(os.environ)
        rest = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        # The image puts its own tree on the path; the arm's tree replaces it.
        rest = [
            p
            for p in rest
            if not (Path(p) / "vllm_torchtpu").exists()
            and not (Path(p) / "src" / "vllm_torchtpu").exists()
        ]
        env["PYTHONPATH"] = os.pathsep.join(
            [str(LOADLOG_DIR), str(self.root / "src"), str(self.root)] + rest
        )
        env["KE_LOADLOG"] = str(self.loadlog)
        env["KE_ARM_MARKER"] = self.marker
        # Every arm compiles cold into its own caches, so no arm inherits
        # another's programs and the control boots exactly as a candidate does.
        for var, sub in (
            ("VLLM_CACHE_ROOT", "vllm"),
            ("VLLM_XLA_CACHE_PATH", "xla"),
            ("JAX_COMPILATION_CACHE_DIR", "jax"),
            ("TORCH_TPU_CACHE_DIR", "torch_tpu"),
            ("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE", "torch_tpu"),
            ("TMPDIR", "tmp"),
        ):
            (self.cache / sub).mkdir(parents=True, exist_ok=True)
            env[var] = str(self.cache / sub)
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.update(SERVE_ENV)
        if self.profile_dir is not None:
            env.update(
                USE_PHASED_PROFILER="true",
                HOST_TRACER_LEVEL="0",
                PYTHON_TRACER_LEVEL="0",
            )
        return env

    def start(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.loadlog.unlink(missing_ok=True)
        cmd = ["vllm", "serve", *SERVE_ARGS]
        if self.profile_dir is not None:
            cmd += [
                "--profiler-config.profiler",
                "torch",
                "--profiler-config.torch_profiler_dir",
                str(self.profile_dir / self.name),
                "--profiler-config.ignore_frontend",
                "true",
            ]
        say(f"{self.name}: starting {' '.join(cmd)}")
        with open(self.out / "server.log", "w") as log:
            self.proc = subprocess.Popen(
                cmd,
                cwd=self.out,
                env=self.env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        t0 = self.started = time.time()
        while not _healthy():
            if tpu_stalled(t0):
                raise RuntimeError("the TPU reported a stall during boot")
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited with {self.proc.returncode} before ready; "
                    f"see {self.out / 'server.log'}"
                )
            if time.time() - t0 > READY_TIMEOUT_S:
                raise RuntimeError(f"server not ready after {READY_TIMEOUT_S}s")
            time.sleep(10)
        self.ready_s = time.time() - t0
        say(f"{self.name}: ready in {self.ready_s:.0f}s")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=180)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass
        # Engine workers may have left the process group; the marker finds them.
        deadline = time.time() + 180
        while (pids := _arm_pids(self.marker)) and time.time() < deadline:
            for pid in pids:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
            time.sleep(2)
        if _arm_pids(self.marker):
            raise RuntimeError(f"{self.name}: server processes survived SIGKILL")
        shutil.rmtree(self.cache, ignore_errors=True)
        time.sleep(20)  # let the TPU runtime release the chips

    def loaded_modules(self) -> dict[str, Path]:
        """module -> origin, over every server process, refusing foreign files."""
        mods: dict[str, Path] = {}
        root = self.root.resolve()
        for line in self.loadlog.read_text().splitlines():
            _, mod, origin = line.split("\t")
            if origin in ("None", "namespace", "built-in"):
                continue
            path = Path(origin).resolve()
            if root not in path.parents:
                raise RuntimeError(
                    f"{self.name}: the server imported {mod} from {path}, "
                    f"outside the arm's tree {root}"
                )
            if mods.setdefault(mod, path) != path:
                raise RuntimeError(f"{self.name}: {mod} imported from two files")
        if "vllm_torchtpu.kernels.gdn.v3.wrapper" not in mods:
            raise RuntimeError(f"{self.name}: the server never imported gdn/v3")
        return mods


# --------------------------------------------------------------------------
# Measurement.


def run_bench(out: Path, since: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["vllm", "bench", "serve", *BENCH_ARGS, "--result-dir", str(out)]
    t0 = time.time()
    with open(out / "bench.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT)
        while (rc := proc.poll()) is None:
            stalled = tpu_stalled(since)
            if stalled or time.time() - t0 > BENCH_TIMEOUT_S:
                proc.kill()
                proc.wait()
                raise RuntimeError(
                    "the TPU reported a stall during a sweep"
                    if stalled
                    else f"sweep still running after {BENCH_TIMEOUT_S}s"
                )
            time.sleep(5)
    result = out / RESULT_FILENAME
    if rc != 0 or not result.exists():
        raise RuntimeError(f"bench rc={rc}; see {out / 'bench.log'}")
    r = json.loads(result.read_text())
    if not r.get("completed", 0) > 0 or not r.get("output_throughput", 0) > 0:
        raise RuntimeError(
            f"bench completed {r.get('completed')} of {NUM_PROMPTS} "
            f"at {r.get('output_throughput')} tok/s"
        )
    return r


def profile_sweep(out: Path, since: float) -> None:
    """Scoping only: one extra sweep with the phased profiler armed throughout."""
    _post("/start_profile", None, 600)
    try:
        run_bench(out, since)
    finally:
        _post("/stop_profile", None, 1800)
    time.sleep(PROFILE_FLUSH_S)
    say(f"profiled one sweep; traces flushed for {PROFILE_FLUSH_S}s")


def _complete(prompt: str) -> list[list]:
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": CAPTURE_MAX_TOKENS,
        "temperature": 0.0,
        "ignore_eos": True,
        "logprobs": CAPTURE_TOP_LOGPROBS,
        "return_tokens_as_token_ids": True,
    }
    lp = _post("/v1/completions", body, REQUEST_TIMEOUT_S)["choices"][0]["logprobs"]
    steps = [
        [tok, float(lpv), {k: float(v) for k, v in top.items()}]
        for tok, lpv, top in zip(lp["tokens"], lp["token_logprobs"], lp["top_logprobs"])
    ]
    if len(steps) != CAPTURE_MAX_TOKENS:
        raise RuntimeError(
            f"capture returned {len(steps)} of {CAPTURE_MAX_TOKENS} tokens"
        )
    return steps


# In-flight copies of each prompt in the concurrent pass: 64 sequences, about
# eight per data-parallel rank, so every rank builds multi-sequence tiles.
CONCURRENT_COPIES = 4


def capture() -> dict:
    """Greedy decodes of the fixed prompts, one at a time and all at once."""
    sequential = [_complete(p) for p in PROMPTS]
    batch = PROMPTS * CONCURRENT_COPIES
    with concurrent.futures.ThreadPoolExecutor(len(batch)) as pool:
        concurrent_ = list(pool.map(_complete, batch))
    return {"sequential": sequential, "concurrent": concurrent_}


def capture_digest(cap: dict) -> str:
    toks = [[s[0] for s in seq] for seq in cap["sequential"]]
    return "sha256:" + hashlib.sha256(json.dumps(toks).encode()).hexdigest()


def sequence_error(ref: list[list], arm: list[list]) -> float:
    """Largest logprob gap between two greedy decodes of one prompt, in nats.

    Over the shared prefix the contexts are identical, so the chosen token's
    logprob is compared directly. At the first differing token each arm's pick
    is scored against the other's distribution: how far below that
    distribution's top choice it sat (a pick outside the other's top-k is
    charged at least the top-1 to top-k spread). Nothing after it is
    comparable.
    """
    worst = 0.0
    for (rt, rlp, rtop), (at, alp, atop) in zip(ref, arm):
        if rt == at:
            worst = max(worst, abs(rlp - alp))
            continue
        gap_ref = rlp - rtop.get(at, min(rtop.values()))
        gap_arm = alp - atop.get(rt, min(atop.values()))
        return max(worst, gap_ref, gap_arm)
    return worst


def capture_errors(ref: dict, arm: dict, mode: str) -> list[float]:
    if len(ref[mode]) != len(arm[mode]):
        raise ValueError("captures of different prompt sets")
    return [sequence_error(r, a) for r, a in zip(ref[mode], arm[mode])]


def binary_digest(mods: dict[str, Path], root: Path) -> str:
    rows = sorted(
        f"{p.relative_to(root.resolve())}\t{sha256_file(p)}" for p in mods.values()
    )
    return "sha256:" + hashlib.sha256("\n".join(rows).encode()).hexdigest()


def measure_arm(
    name: str, root: Path, out: Path, reps: int, warmup: int, profile_dir: Path | None
) -> dict:
    tree = git_tree_hash(root)  # before the server can write into the tree
    srv = Server(root, out, name, profile_dir)
    try:
        srv.start()
        mods = srv.loaded_modules()
        say(f"{name}: {len(mods)} vllm_torchtpu modules imported from {root}")
        for i in range(warmup):
            r = run_bench(out / f"warmup{i}", srv.started)
            say(f"{name}: warmup {i}: {r['output_throughput']:.2f} tok/s (discarded)")
        sweeps = []
        for i in range(reps):
            r = run_bench(out / f"rep{i}", srv.started)
            sweeps.append(r)
            say(
                f"{name}: rep {i}: {r['output_throughput']:.2f} tok/s  "
                f"completed {r['completed']}  ttft {r['median_ttft_ms']:.1f} ms  "
                f"tpot {r['median_tpot_ms']:.3f} ms"
            )
        cap = capture()
        (out / "capture.json").write_text(json.dumps(cap))
        if profile_dir is not None:
            profile_sweep(out / "profiled", srv.started)
        mods = srv.loaded_modules()  # again: workers may import lazily
    finally:
        srv.stop()
    tput = [r["output_throughput"] for r in sweeps]
    med = statistics.median(tput)
    completed = sorted({r["completed"] for r in sweeps})
    say(
        f"{name}: median {med:.2f} tok/s over {tput}, spread "
        f"{(max(tput) - min(tput)) / med * 100:.2f}%, completed {completed}"
    )
    digest = binary_digest(mods, root)
    return {
        "tree": tree,
        "binary_hash": digest,
        "latency_us": 1e6 / med,
        "capture": cap,
        "completed": completed,
        "metrics": {k: statistics.median(r[k] for r in sweeps) for k in SECONDARY}
        | {"completed": float(min(completed))},
        "diag": {
            "output_throughput": tput,
            "ready_s": round(srv.ready_s, 1),
            "modules": len(mods),
        },
        "dispatched": f"serve:{CONFIG}:{digest.split(':', 1)[1]}",
    }


def parse_arm(spec: str) -> tuple[str, Path, bool]:
    name, _, rest = spec.partition("=")
    control = rest.endswith(":control")
    path = Path(rest[: -len(":control")] if control else rest).resolve()
    if not name or not (path / "src" / "vllm_torchtpu").is_dir():
        raise SystemExit(f"bad --arm {spec!r}: want name=/path/to/tree[:control]")
    return name, path, control


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument(
        "--candidate", default="", help="recorded only; the harness names it"
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        required=True,
        help="largest worst_error (nats, see --error-stat) of a correct arm",
    )
    ap.add_argument(
        "--error-stat",
        choices=("mean", "worst"),
        required=True,
        help="worst_error is the mean over prompts or the worst prompt's gap",
    )
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/serve_scorer"))
    ap.add_argument("--job-id", default=os.environ.get("CDK_JOB_ID", ""))
    ap.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="scoping only: profile windows of one extra sweep per arm",
    )
    args = ap.parse_args()

    arms = [parse_arm(a) for a in args.arm]
    controls = [a for a in arms if a[2]]
    if len(controls) != 1 or len({a[0] for a in arms}) != len(arms):
        raise SystemExit("need unique arm names and exactly one control")
    scorer_hash = "sha256:" + sha256_file(Path(__file__))
    say(
        f"scorer {scorer_hash} job {args.job_id or '-'} arms "
        + ", ".join(f"{n}{'(control)' if c else ''}" for n, _, c in arms)
    )

    results: dict[str, dict] = {}
    for name, root, _ in arms:
        try:
            results[name] = measure_arm(
                name,
                root,
                args.out_dir / name,
                args.reps,
                args.warmup,
                args.profile_dir,
            )
        except Exception as exc:  # the arm is left out; see the module docstring
            say(f"{name}: NOT MEASURED: {type(exc).__name__}: {exc}")

    ctrl_name = controls[0][0]
    if ctrl_name not in results:
        say("the control arm was not measured; printing nothing")
        return 1
    ref = results[ctrl_name]
    out_arms = []
    for name, _, is_control in arms:
        if name not in results:
            continue
        r = results[name]
        cap = r["capture"]
        ref_cap = ref["capture"]
        seq = [0.0] if is_control else capture_errors(ref_cap, cap, "sequential")
        conc = [0.0] if is_control else capture_errors(ref_cap, cap, "concurrent")
        r["metrics"]["mean_prompt_gap"] = statistics.fmean(seq)
        r["metrics"]["worst_prompt_gap"] = max(seq)
        r["metrics"]["concurrent_logprob_gap"] = statistics.fmean(conc)
        err = r["metrics"][f"{args.error_stat}_prompt_gap"]
        same_count = r["completed"] == ref["completed"] and len(r["completed"]) == 1
        correct = err <= args.tolerance and same_count
        say(
            f"{name}: worst_error {err:.4f} nats ({args.error_stat} over prompts, "
            f"tolerance {args.tolerance}), "
            f"completed {r['completed']} vs control {ref['completed']}: "
            f"{'correct' if correct else 'INCORRECT'}; latency {r['latency_us']:.3f} us"
        )
        out_arms.append(
            {
                "name": name,
                "control": is_control,
                "tree": r["tree"],
                "binary_hash": r["binary_hash"],
                "configs": {
                    CONFIG: {
                        "latency_us": r["latency_us"],
                        "correct": correct,
                        "worst_error": err,
                        "checksum": capture_digest(cap),
                        "dispatched": r["dispatched"],
                        "metrics": r["metrics"],
                    }
                },
                "diag": r["diag"],
            }
        )
    print(
        json.dumps(
            {
                "scorer_hash": scorer_hash,
                "job_id": args.job_id,
                "candidate": args.candidate,
                "config_args": {CONFIG: CONFIG_ARGS},
                "arms": out_arms,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
