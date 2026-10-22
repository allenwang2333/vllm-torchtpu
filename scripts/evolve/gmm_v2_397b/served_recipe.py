"""Rebuild the served recipe of record for the gmm_v2 campaign, as a cdk recipe.

The recipe of record is the JobSet CDK job j-e8281eaa-7238-47b0-be0c ran
(`cdk job recipe <job>` / the synced `recipe.yaml`). Re-submitting
`infx-qwen35-agentx` by name renders today's jsonnet, whose defaults have moved,
so this takes the job's rendered JobSet and strips what `cdk job create` adds
on every submit (namespace, Kueue labels, annotations, the CDK_* env, the
/cdk-outputs mount and the --map-dir init container), leaving what the recipe
itself rendered. cdk adds them back when the result is submitted.

    python served_recipe.py <job recipe.yaml> --variant record  > ror.yml
    python served_recipe.py <job recipe.yaml> --variant phased  > phased.yml

`record` changes nothing else. `phased` is the profiling arm of scoping §1.2:
the server runs vLLM-TPU's phased profiler (one capture per batch composition)
with host and Python tracing off, the copy loop keeps each phase's directory,
each of the bench's three profiler windows lasts 90 s instead of 10 so a
prefill step is likely to land in one, and the replay runs 900 s (the windows
open at 180, 510 and 630 s). Its timings are not a measurement.
"""

from __future__ import annotations

import argparse
import copy
import sys

import yaml

CDK_ENV = {"CDK_JOB_ID", "CDK_POD_NAME", "CDK_JOB_NAME", "CDK_CONTAINER_NAME",
           "CDK_OUTPUT_DIR", "CDK_LOG_DIR", "CDK_PROFILE"}
CDK_VOLUMES = {"cdk-outputs"}
CDK_INIT = {"cdk-bootstrap", "cdk-map-dir"}
CDK_LABELS = {"cloud-devkit/log-transport"}

FLAT_COPY = "find /dev/shm/xprof -name '*.xplane.pb' -exec cp -n {} \"$XPROF_OUT/\" \\; 2>/dev/null || true"
RECORD_DURATION = 'export DURATION="3600"'
PHASED_DURATION = 'export DURATION="900"'
TREE_COPY = ("(cd /dev/shm/xprof && find . \\( -name '*.xplane.pb' -o -name '*.json' \\) "
             "-exec cp -n --parents {} \"$XPROF_OUT/\" \\;) 2>/dev/null || true")


def strip(doc: dict, name: str) -> dict:
    doc = copy.deepcopy(doc)
    doc["metadata"] = {"name": name}
    for rj in doc["spec"]["replicatedJobs"]:
        rj["template"].get("metadata", {}).pop("annotations", None)
        if not rj["template"].get("metadata"):
            rj["template"].pop("metadata", None)
        jspec = rj["template"]["spec"]
        jspec.pop("activeDeadlineSeconds", None)
        pod = jspec["template"]
        meta = pod.get("metadata", {})
        meta.pop("annotations", None)
        labels = {k: v for k, v in meta.get("labels", {}).items() if k not in CDK_LABELS}
        if labels:
            meta["labels"] = labels
        else:
            meta.pop("labels", None)
        if not meta:
            pod.pop("metadata", None)
        ps = pod["spec"]
        ps["volumes"] = [v for v in ps.get("volumes", [])
                         if v["name"] not in CDK_VOLUMES and not v["name"].startswith("cdk-map-")]
        init = [c for c in ps.get("initContainers", []) if c["name"] not in CDK_INIT]
        if init:
            ps["initContainers"] = init
        else:
            ps.pop("initContainers", None)
        for c in ps["containers"]:
            c["env"] = [e for e in c.get("env", []) if e["name"] not in CDK_ENV]
            c["volumeMounts"] = [m for m in c.get("volumeMounts", [])
                                 if m["name"] not in CDK_VOLUMES and not m["name"].startswith("cdk-map-")]
    return doc


def container(doc: dict, job: str) -> dict:
    for rj in doc["spec"]["replicatedJobs"]:
        if rj["name"] == job:
            return rj["template"]["spec"]["template"]["spec"]["containers"][0]
    raise SystemExit(f"no replicatedJob {job}")


def set_env(c: dict, key: str, value: str) -> None:
    for e in c["env"]:
        if e["name"] == key:
            e["value"] = value
            e.pop("valueFrom", None)
            return
    c["env"].append({"name": key, "value": value})


def phased(doc: dict) -> dict:
    server = container(doc, "vllm-server")
    script = server["args"][0]
    if script.count(FLAT_COPY) != 1:
        raise SystemExit("the server script's xprof copy line is not the one this variant replaces")
    server["args"][0] = script.replace(FLAT_COPY, TREE_COPY, 1)
    for key, value in (("USE_PHASED_PROFILER", "true"), ("HOST_TRACER_LEVEL", "0"),
                       ("PYTHON_TRACER_LEVEL", "0")):
        set_env(server, key, value)
    bench = container(doc, "bench")
    set_env(bench, "PROFILE_DURATION_SECONDS", "90")
    # The bench script exports its own DURATION, so the env entry alone does not move it.
    if bench["args"][0].count(RECORD_DURATION) != 1:
        raise SystemExit("the bench script's DURATION export is not the one this variant replaces")
    bench["args"][0] = bench["args"][0].replace(RECORD_DURATION, PHASED_DURATION, 1)
    set_env(bench, "DURATION", "900")
    return doc


class _Dumper(yaml.SafeDumper):
    """Scripts as `|` blocks, so the committed recipe reads like the job's."""


def _str(dumper, value):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("recipe", help="the job's rendered JobSet (synced recipe.yaml)")
    p.add_argument("--variant", choices=("record", "phased"), default="record")
    p.add_argument("--name", required=True, help="the recipe name to register")
    a = p.parse_args(argv)
    with open(a.recipe) as f:
        doc = strip(yaml.safe_load(f), a.name)
    if a.variant == "phased":
        doc = phased(doc)
    yaml.dump(doc, sys.stdout, Dumper=_Dumper, sort_keys=False, width=1000, allow_unicode=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
