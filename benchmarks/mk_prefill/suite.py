"""Run several bench / test commands in one job, one process each (one TPU holder at a time).

    python3 suite.py 'tests/kernels/qwen35_prefill_mk/test_stage_pooled_cpu.py --tpu --real' \
        'benchmarks/mk_prefill/mk_bench.py stage --xprof'

Each argument is a python script path (relative to the checkout) plus its arguments; a leading
NAME=VALUE word sets an environment variable for that command.
"""

import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKOUT = os.path.abspath(os.path.join(HERE, "..", ".."))

rc = 0
for cmd in sys.argv[1:]:
    words = shlex.split(cmd)
    env = dict(os.environ)
    while words and "=" in words[0] and not words[0].startswith("-"):
        k, _, v = words.pop(0).partition("=")
        env[k] = v
    print(f"[suite] >>> {cmd}", flush=True)
    p = subprocess.run([sys.executable, os.path.join(CHECKOUT, words[0])] + words[1:], env=env,
                       cwd=CHECKOUT)
    print(f"[suite] <<< exit {p.returncode}: {cmd}", flush=True)
    rc = rc or p.returncode
sys.exit(rc)
