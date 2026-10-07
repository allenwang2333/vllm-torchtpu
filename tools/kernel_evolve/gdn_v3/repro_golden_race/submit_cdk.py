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
"""Run race_check.py on one tpu7x pod through cdk, for one or more git revisions.

    python3 submit_cdk.py --image IMAGE --tag TAG --arm golden=REV [--arm b=REV ...]

Each arm is `git archive`d from this repository and run in its own python
process, one after another, in a single pod (one TPU boot per job). The job
writes `<arm>.jsonl` (one record per call) and `<arm>.log` to its output
directory and prints every `CALL` line to the job log. Prints the job id.

Needs `cdk` with a cloud-devkit recipes checkout (CDK_DIR, default
~/cloud-devkit), `git`, and an IMAGE that has python3 with jax, jaxlib and
libtpu at the versions pyproject.toml pins plus numpy and ml_dtypes.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CDK_DIR = Path(os.environ.get("CDK_DIR", Path.home() / "cloud-devkit"))

RUN_SH = """#!/bin/bash
set -u
OUT="$CDK_OUTPUT_DIR"
echo "pod $HOSTNAME job $CDK_JOB_ID"
sha256sum /mapped/stage/race_check.py
( while sleep 60; do
    grep -rlsF 'Stall detected' /tmp/tpu_logs && { echo "STALL pod=$HOSTNAME"; break; }
  done ) &
rc_all=0
for arm in @ARMS@; do
  d=/scratch/$arm; mkdir -p $d; tar -xf /mapped/stage/$arm.tar -C $d
  echo "=== ARM $arm commit $(cat /mapped/stage/$arm.commit) tree $(cat /mapped/stage/$arm.tree)"
  ( cd $d && timeout @TIMEOUT@ python3 /mapped/stage/race_check.py $d --label $arm \\
      --out "$OUT/$arm.jsonl" @ARGS@ ) > "$OUT/$arm.log" 2>&1
  rc=$?; [ $rc -ne 0 ] && rc_all=$rc
  echo "RESULT arm=$arm rc=$rc"
  grep -E "^(versions|devices|WARNING|CALL|DONE)|Error" "$OUT/$arm.log"
done
echo "=== RACE CHECK DONE"
exit $rc_all
"""


def git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def cdk(*args):
    """Run cdk, passing the agent ack codes when an agent harness requires them."""
    cmd = ["cdk"]
    if os.environ.get("AI_AGENT"):
        letter = subprocess.run(
            ["cdk", "agent-letter"], capture_output=True, text=True, check=True
        ).stdout
        codes = re.findall(r"ack \d+/10: ([0-9A-F]+)", letter)
        if len(codes) != 10:
            raise SystemExit("could not read the ten ack codes from `cdk agent-letter`")
        cmd.append(f"--agent-code={'-'.join(codes)}")
    res = subprocess.run([*cmd, *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"cdk {' '.join(args[:2])} failed ({res.returncode}): {res.stderr}")
    return res.stdout


def jobset(name, image, deadline_s):
    """One single-host tpu7x pod (2x2x1, 4 chips); no model weights."""
    resources = {"cpu": "192", "memory": "900Gi", "google.com/tpu": "4"}
    pod = {
        "spec": {
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 60,
            "nodeSelector": {
                "cloud.google.com/gke-tpu-accelerator": "tpu7x",
                "cloud.google.com/gke-tpu-topology": "2x2x1",
            },
            "containers": [
                {
                    "name": "tpu-job",
                    "image": image,
                    "command": ["/bin/bash", "/mapped/stage/run.sh"],
                    "env": [
                        # A single-host job: no multi-slice runtime.
                        {"name": "MEGASCALE_NUM_SLICES", "value": ""},
                        {"name": "MEGASCALE_SLICE_ID", "value": ""},
                        {"name": "MEGASCALE_COORDINATOR_ADDRESS", "value": ""},
                        {"name": "MEGASCALE_PORT", "value": ""},
                        {"name": "PYTHONUNBUFFERED", "value": "1"},
                    ],
                    "resources": {"limits": resources, "requests": resources},
                    "volumeMounts": [
                        {"name": "dshm", "mountPath": "/dev/shm"},
                        {"name": "scratch", "mountPath": "/scratch"},
                    ],
                }
            ],
            "volumes": [
                {"name": "dshm", "emptyDir": {"medium": "Memory"}},
                {"name": "scratch", "emptyDir": {}},
            ],
        },
    }
    return {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {"name": name},
        "spec": {
            "replicatedJobs": [
                {
                    "name": "tpu-job",
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "backoffLimit": 0,
                            "activeDeadlineSeconds": deadline_s,
                            "template": pod,
                        }
                    },
                }
            ]
        },
    }


def register_recipe(name, doc):
    """Write the recipe and register it in recipes.yml, right before use."""
    owner = getpass.getuser()
    rel = Path("recipes/experimental") / owner / "gdn-race-check" / f"{name}.yaml"
    (CDK_DIR / rel).parent.mkdir(parents=True, exist_ok=True)
    (CDK_DIR / rel).write_text(json.dumps(doc, indent=1) + "\n")  # JSON is YAML
    index = CDK_DIR / "recipes.yml"
    text = index.read_text()
    if f"- name: {name}\n" not in text:
        entry = (
            f"- name: {name}\n  owner: {owner}\n  k8s_file: {rel}\n"
            "  require_gcs_mount: true\n"  # --map-dir reads back through it
        )
        index.write_text(text.rstrip("\n") + "\n" + entry)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", required=True, help="NAME=REV")
    ap.add_argument("--image", required=True)
    ap.add_argument("--tag", required=True, help="job tag")
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument("--suite", default="full", choices=("full", "quick", "boundary"))
    ap.add_argument("--deadline-s", type=int, default=3600)
    ap.add_argument("--dry-run", action="store_true", help="stage and print, submit nothing")
    args = ap.parse_args()

    repo = Path(git("rev-parse", "--show-toplevel", cwd=HERE))
    stage = Path(tempfile.mkdtemp(prefix="gdn_race_stage_"))
    arms = []
    for spec in args.arm:
        name, _, rev = spec.partition("=")
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise SystemExit(f"arm name {name!r}: use [a-z0-9_]")
        sha = git("rev-parse", "--verify", f"{rev}^{{commit}}", cwd=repo)
        git("archive", "--format=tar", "-o", str(stage / f"{name}.tar"), sha, cwd=repo)
        (stage / f"{name}.commit").write_text(sha)
        (stage / f"{name}.tree").write_text(git("rev-parse", f"{sha}^{{tree}}", cwd=repo))
        arms.append(name)
        print(f"arm {name}: commit {sha}", file=sys.stderr)
    shutil.copy(HERE / "race_check.py", stage / "race_check.py")
    run = (
        RUN_SH.replace("@ARMS@", " ".join(arms))
        .replace("@TIMEOUT@", str(max(60, args.deadline_s // len(arms) - 60)))
        .replace("@ARGS@", shlex.join(["--calls", str(args.calls), "--suite", args.suite]))
    )
    (stage / "run.sh").write_text(run)
    user = re.sub(r"[^a-z0-9-]", "-", getpass.getuser().lower())
    name = f"{user}-gdn-race-{time.strftime('%m%d-%H%M%S')}"[:63].rstrip("-")
    doc = jobset(name, args.image, args.deadline_s)
    if args.dry_run:
        print(f"stage {stage}\n{json.dumps(doc, indent=1)}\n{run}")
        return 0
    register_recipe(name, doc)
    tags = ",".join(t for t in (args.tag, "agent-jobs", os.environ.get("AI_AGENT")) if t)
    out = cdk(
        "job", "create", name, "-t", tags, "--ttl", "3600",
        "--active-deadline-seconds", str(args.deadline_s),
        "--map-dir", f"{stage}:tpu-job:/mapped/stage",
    )  # fmt: skip
    m = re.search(r"\bj-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}\b", out)
    if not m:
        raise SystemExit(f"no job id in cdk output: {out}")
    print(m.group(0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
