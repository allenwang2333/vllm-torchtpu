# SPDX-License-Identifier: Apache-2.0
"""Turn a cdk recipe into a JobSet that plain kubectl will accept.

A cdk recipe is not quite a Kubernetes manifest. It carries `{{ .VAR | default
`value` }}` placeholders, and it leaves out the namespace, the Kueue labels and
the ownership annotation that the cdk CLI fills in when it submits. That is
fine when you have the CLI. When you do not, this fills in the same things, so
the recipe in the tree stays the one that ran rather than diverging from a
hand-edited copy.

Only the `default` form of the placeholder is understood, because that is the
only form these recipes use. A placeholder without a default is an error rather
than an empty string: silently dropping the image name would produce a JobSet
that fails ten minutes later for no visible reason.

Defaults are overridable, so the same recipe can be pointed at a different
tarball without editing it:

    python3 render_cdk_recipe.py recipe.yml --name dsv4-pd-alynie \
        --set PR_TARBALL=gs://bucket/prs/pr684.tar.gz | kubectl apply -f -
"""

from __future__ import annotations

import argparse
import re
import sys

import yaml

_PLACEHOLDER = re.compile(r"\{\{\s*\.(\w+)\s*(?:\|\s*default\s*`([^`]*)`\s*)?\}\}")


def render(text: str, overrides: dict[str, str]) -> str:

    def sub(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        if name in overrides:
            return overrides[name]
        if default is None:
            raise SystemExit(
                f"{name} has no default in the recipe; pass --set {name}=...")
        return default

    return _PLACEHOLDER.sub(sub, text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe")
    ap.add_argument("--name", help="JobSet name; the recipe's own if omitted")
    ap.add_argument("--namespace", default="cloud-devkit-jobs")
    ap.add_argument("--queue", default="cdk-local-queue")
    ap.add_argument("--priority-class", default="cdk-p2")
    ap.add_argument("--username", help="value for the cdk ownership annotation")
    ap.add_argument("--set",
                    action="append",
                    default=[],
                    metavar="VAR=VALUE",
                    help="override a placeholder default")
    args = ap.parse_args()

    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        if not _:
            raise SystemExit(f"--set wants VAR=VALUE, got {item!r}")
        overrides[key] = value

    with open(args.recipe) as fh:
        doc = yaml.safe_load(render(fh.read(), overrides))

    meta = doc.setdefault("metadata", {})
    if args.name:
        meta["name"] = args.name
    meta["namespace"] = args.namespace
    labels = meta.setdefault("labels", {})
    labels["kueue.x-k8s.io/queue-name"] = args.queue
    labels["kueue.x-k8s.io/priority-class"] = args.priority_class
    if args.username:
        annotations = meta.setdefault("annotations", {})
        annotations["cloud-devkit.google.com/username"] = args.username

    yaml.safe_dump(doc, sys.stdout, sort_keys=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
