# Automatic vLLM "Last Known Good" (LKG) promotion for `vllm-torchtpu`

**Goal:** replace today's manual `vllm==<version>` bump with the same automated LKG
promotion loop that `tpu-inference` already runs.

Read Part 1 first — it explains the mechanism in the repo where it already works.
Part 2 is what we have today. Part 3 is the actual step-by-step implementation.
Part 4 is the runbook you'll use after it ships.

Written against `vllm-torchtpu` `main` @ `2a0e1000` and `tpu-inference` `main` @ `7ecc401e6`
(both 2026-07-27). CI layouts drift — if a snippet below doesn't match what you see, trust
the repo and please fix this doc.

---

## Part 1 — High-level overview: how `tpu-inference` does it today

### 1.1 The problem this solves

`tpu-inference` is a plugin that lives on top of upstream `vllm-project/vllm`. Upstream
merges dozens of commits a day, and any of them can break us. That leaves two bad options:

- **Track upstream HEAD in CI.** Every PR author gets blamed for upstream breakage. CI is
  red for reasons that have nothing to do with the PR.
- **Pin a version by hand.** CI is stable, but the pin rots. When someone finally bumps it,
  they're bisecting weeks of upstream changes at once.

The LKG pattern gets both: **day-to-day CI runs against a known-good pinned vLLM commit,
and a separate scheduled pipeline continuously tries upstream HEAD.** When HEAD passes the
full test suite, the pin is automatically advanced to it. When HEAD fails, nothing moves —
the pin stays where it was and only the integration pipeline is red.

### 1.2 The moving parts

All file links below point at
[`vllm-project/tpu-inference`](https://github.com/vllm-project/tpu-inference) on `main`.

| Thing | Where | Role |
|---|---|---|
| [`vllm_lkg.version`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/vllm_lkg.version) | file in repo, single line, 40-char SHA | **The pin.** The current last-known-good upstream vLLM commit. |
| `tpu-vllm-integration` | a *separate Buildkite pipeline*, scheduled | Runs the test suite against upstream vLLM **HEAD**. |
| main CI pipeline | Buildkite, on PR / push / nightly | Runs the test suite against the **pin**. |
| [`scripts/bootstrap.sh`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/bootstrap.sh) | the one command both pipelines run | Decides HEAD-vs-pin and publishes it as build metadata. |
| `VLLM_COMMIT_HASH` | Buildkite build metadata | The single source of truth for the rest of the build. |
| [`integration_promote.yml`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/integration_promote.yml) | pipeline fragment | The promotion step. Only uploaded by the integration pipeline. |
| [`scripts/update_lkg_version.sh`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/update_lkg_version.sh) | script | Commits + pushes the new pin to `main`. |
| [`verified_commit_hashes.csv`](https://github.com/vllm-project/tpu-inference/blob/main/verified_commit_hashes.csv) | file in repo | Append-only audit log of `(vllm commit, tpu-inference commit)` pairs that passed nightly. |

Note that **both pipelines run the exact same `bootstrap.sh`**. There is no forked copy of
the test definitions. The only difference is which vLLM commit gets selected, and whether
the promote step is appended.

### 1.3 The flow

```
                       ┌──────────────────────────────────────────────┐
                       │  Buildkite pipeline: tpu-vllm-integration     │
                       │  (scheduled, e.g. daily)                     │
                       └──────────────────────────────────────────────┘
                                          │
                          bootstrap.sh sees slug == integration
                                          │
                  VLLM_COMMIT_HASH := git ls-remote <vllm> HEAD   ← upstream HEAD
                                          │
                    uploads: promote fragment, then the test pipelines
                                          │
                       ┌──────────────────┴───────────────────┐
                       │  build image with that vLLM commit   │
                       │  run v6e + v7x test suites           │
                       └──────────────────┬───────────────────┘
                                          │
                            all green? ───┴─── no ──▶ stop. pin unchanged.
                                          │              (integration build is red,
                                         yes              oncall gets notified)
                                          │
                     update_lkg_version.sh writes the hash into
                     .buildkite/vllm_lkg.version and pushes to main
                     ("[skip ci] Update vLLM LKG to <hash>")
                                          │
                                          ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │  Every subsequent PR / nightly build reads that file and tests        │
    │  against the newly promoted commit.                                   │
    └───────────────────────────────────────────────────────────────────────┘
```

### 1.4 Walking the code

**a) Selecting the vLLM commit** — [`bootstrap.sh`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/bootstrap.sh),
near the bottom:

```bash
if [[ $BUILDKITE_PIPELINE_SLUG == "tpu-vllm-integration" ]]; then
    # Note: Integration pipeline always fetch latest vllm version
    VLLM_COMMIT_HASH=$(git ls-remote https://github.com/vllm-project/vllm.git HEAD | awk '{ print $1}')
    buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
    # Note: upload are inserted in reverse order, so promote LKG should upload before tests
    upload_with_priority .buildkite/integration_promote.yml "$JOB_PRIORITY"
    ... upload the jax test pipelines ...
else
    # PR and Nightly pipelines load VLLM_COMMIT_HASH from vllm_lkg.version
    VLLM_COMMIT_HASH=$(get_vllm_commit_hash)
    buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
    ...
fi
```

and `get_vllm_commit_hash()` in
[`pipeline_config.sh`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/configs/pipeline_config.sh)
is just "read the file, fall back to upstream HEAD if it's missing or empty":

```bash
get_vllm_commit_hash() {
  local commit_hash="" version_file=".buildkite/vllm_lkg.version"
  if [ -f "$version_file" ]; then commit_hash="$(cat "$version_file")"; fi
  if [ -z "${commit_hash:-}" ]; then
    commit_hash=$(git ls-remote https://github.com/vllm-project/vllm.git HEAD | awk '{ print $1}')
  fi
  echo "$commit_hash"
}
```

**b) Making the build actually use it.**
[`setup_docker_env.sh`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/setup_docker_env.sh)
reads the metadata and passes it to Docker; the
[Dockerfile](https://github.com/vllm-project/tpu-inference/blob/main/docker/Dockerfile)
clones vLLM and checks the commit out:

```dockerfile
ARG VLLM_COMMIT_HASH
RUN git clone $VLLM_REPO . && \
    if [ -n "$VLLM_COMMIT_HASH" ]; then git checkout $VLLM_COMMIT_HASH; fi
```

Two safety details here that we should copy, because both were learned the hard way:

- **The vLLM commit is part of the image cache tag**: `${TPU_INFERENCE_HASH}-${VLLM_COMMIT_HASH}-${TPU_VERSION}`.
  Without it, a CI build (pin) and an integration build (HEAD) of the *same* repo commit
  produce the same tag and silently overwrite each other in the registry — you'd promote a
  hash that was never actually tested. `setup_docker_env.sh` hard-fails if the metadata is
  empty inside Buildkite rather than falling back to a vLLM-agnostic tag.
- **`verify_image_vllm`** runs `git rev-parse HEAD` *inside the built image* and aborts if
  it doesn't equal the expected hash. Cheap, and it catches a whole class of "we tested the
  wrong thing" bugs.

**c) Promoting.** [`integration_promote.yml`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/integration_promote.yml)
is tiny:

```yaml
steps:
  - wait: ~
  - label: "Promote to Last Known Good (LKG)"
    key: "promote_lkg"
    agents: { queue: "cpu" }
    secrets:
      - GITHUB_PAT     # see the credentials note under (d)
    commands:
      - |
        VLLM_LKG_HASH=$(buildkite-agent meta-data get "VLLM_COMMIT_HASH")
        bash .buildkite/scripts/update_lkg_version.sh "$$VLLM_LKG_HASH"
        buildkite-agent annotate ":white_check_mark: LKG Updated: $$VLLM_LKG_HASH" --style "success"
```

The leading `- wait: ~` is the whole gating mechanism, and it depends on upload ordering:

> Buildkite inserts uploaded steps immediately after the currently running step, so **the
> last thing you upload appears first**. `bootstrap.sh` therefore uploads the promote
> fragment *before* the test pipelines so that it lands *after* them.

A bare `wait` blocks until every preceding step finishes and — unless you set
`continue_on_failure: true` — is skipped if any of them failed. So "promote only when
everything is green" is expressed in two characters of YAML.

**d) Writing the pin back.**
[`update_lkg_version.sh`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/update_lkg_version.sh)
is a plain git push: reset to `origin/main`, write the hash to the file, and if the content
changed, commit `[skip ci] Update vLLM LKG to <hash>` and push. The `[skip ci]` matters —
without it the push retriggers CI in a loop.

> **Credentials — don't copy the PAT plumbing.** This script (and the two `commit_*.sh`
> scripts below) injects a long-lived `GITHUB_PAT` into the remote URL. That is legacy, and
> on our current agents it is **redundant**: our Buildkite agent images already authenticate
> every git operation to GitHub via a **GitHub App**, so a plain `git push origin main`
> just works. See §1.5 — Step 6 relies on it and provisions no secret at all.

**e) The audit trail.** Separately from promotion, the
[nightly pipeline](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/nightly_verify.yml)
appends to
[`verified_commit_hashes.csv`](https://github.com/vllm-project/tpu-inference/blob/main/verified_commit_hashes.csv)
via
[`commit_verified_commit_hashes.sh`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/commit_verified_commit_hashes.sh):

```
timestamp,vllm_commit_hash,tpu_inference_commit_hash
2026-07-05 08:41:21,34b560b725b401f33477464163244b4744d1c40f,d7983f16361878912854c256224c9b67540ba44b
```

This is the "which pair of commits did we last see fully working" record used when
debugging a regression or cutting a release. It's gated to `NIGHTLY=1 && branch == main` —
originally because that's where the PAT was provisioned, though as §1.5 explains that
constraint no longer really applies.

**f) Consumers of the pin.** Beyond CI, the file is the published answer to "which vLLM do
I build against?" — the
[install docs](https://github.com/vllm-project/tpu-inference/blob/main/docs/getting_started/installation.md)
tell users `export VLLM_COMMIT_HASH="$(cat tpu-inference/.buildkite/vllm_lkg.version)"`,
and local/multihost scripts opt in with `USE_VLLM_LKG=true`.

### 1.5 How the bot actually authenticates (and why you need no secret)

This is the part that isn't visible from either repo's `.buildkite/` directory, and it's the
single biggest simplification available to us.

Our Buildkite agents are provisioned by
[`vllm-project/ci-infra`](https://github.com/vllm-project/ci-infra), under
[`terraform/gcp_old/tpu-inference/`](https://github.com/vllm-project/ci-infra/tree/main/terraform/gcp_old/tpu-inference).
Each agent module's VM startup script installs **GitHub App authentication into the agent
image itself**
([`modules/ci_cpu/main.tf`](https://github.com/vllm-project/ci-infra/blob/main/terraform/gcp_old/tpu-inference/modules/ci_cpu/main.tf),
"Setup In-Memory GitHub App Authentication"):

1. `/etc/buildkite-agent/get_github_token.py` — signs a JWT for App ID `4156238` and
   exchanges it for an installation access token on installation `142868369`.
2. `/etc/buildkite-agent/git-credential-github-app` — a **git credential helper**. On every
   `get`, it pulls the App private key with `buildkite-agent secret get <pem-secret>`, mints
   a fresh token, prints `username=x-access-token` / `password=<token>`, and unsets the key.
3. It's registered in the **system** gitconfig plus root's and `buildkite-agent`'s globals,
   along with `insteadOf` rewrites mapping `git@github.com:` and `ssh://git@github.com/` →
   `https://github.com/`.

What that means in practice:

- **Any `git push`/`fetch` to GitHub from a job on these agents is authenticated
  automatically.** No `secrets:` block, no `x-access-token:${TOKEN}@` URL rewriting, no
  `export`ing anything.
- **Nothing expires.** A token is minted per git operation and lives only for that call, so
  there is no PAT rotation to track and no refresh logic to write.
- It works regardless of whether the remote is an SSH-style or HTTPS URL.

Installed on the `ci_cpu` (`queue=cpu`), `ci_cpu_64_core`, `ci_v5`, `ci_v6`, `ci_v6e`, and
`ci_v7x` modules — i.e. every queue this repo uses (`cpu_64_core`, `tpu_v7x_2_queue`,
`tpu_v7x_8_queue`, all instantiated in
[`cloud-ullm-inference-ci-cd`](https://github.com/vllm-project/ci-infra/blob/main/terraform/gcp_old/tpu-inference/cloud-ullm-inference-ci-cd/main.tf)).
The one exception is the `benchmark` module, which does **not** install the helper — don't
schedule a pushing step there.

So `tpu-inference`'s `GITHUB_PAT` and its newer
[`export_ci_bot_token.sh`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/export_ci_bot_token.sh)
(App-token exchange in-pipeline, used only by the kernel-autotune pipeline) are both
carrying weight the agent already carries. **Don't port either one.**

### 1.6 The properties worth preserving

1. One bootstrap, two pipelines. Test definitions are never duplicated.
2. Everything downstream reads a single build-metadata key, not the file directly.
3. The pin only advances through a fully green run — no partial promotion.
4. The promotion commit can't retrigger CI.
5. The artifact under test is provably built from the commit being promoted.
6. A human can always override the selection and always revert the pin with one commit.

---

## Part 2 — Where `vllm-torchtpu` is today

As of `main` @ `2a0e1000` (2026-07-27):

```
pyproject.toml:    "vllm==0.25.1",          ← the pin, edited by hand
README.md:         git clone --depth 1 --branch v0.25.1 ...   ← and again, twice
docker/Dockerfile: base stage installs deps from pyproject.toml (so: vLLM from PyPI)
.buildkite/:       bootstrap.sh → build → tests → integration → perf
```

The pin has moved by hand from `0.23.0` → `0.25.1` over the last few months, one release at
a time. That cadence is exactly what this change automates.

Differences from `tpu-inference` that shape the work:

| | `tpu-inference` | `vllm-torchtpu` |
|---|---|---|
| How vLLM is installed in CI | cloned from git, `git checkout <sha>` | installed from **PyPI** as `vllm==0.25.1` |
| Pin granularity | any upstream commit | only published releases |
| Integration pipeline vs upstream HEAD | exists | **does not exist** |
| `VLLM_COMMIT_HASH` build metadata | exists | **does not exist** |
| CI image tag | includes the vLLM commit | `<repo commit>` only |
| A credential that can push to the repo | yes — and the agents can push regardless (§1.5) | **same agents, so also yes** |

The one genuinely new capability we need is the first row: **CI has to be able to install
vLLM at an arbitrary commit**, otherwise there is nothing to promote. The good news is the
Dockerfile already has that path — the `ci` stage takes a `VLLM_SOURCE` build arg, strips
the version pin, and installs the copied tree in editable mode:

```dockerfile
ARG VLLM_SOURCE
COPY vllm /root/torchtpu-vllm/vllm
RUN if [ -n "$VLLM_SOURCE" ]; then \
      sed -i 's/vllm==[^"]*/vllm/' pyproject.toml && \
      VLLM_TARGET_DEVICE="tpu" uv pip install --system -e /root/torchtpu-vllm/vllm ; \
    fi
```

and `docker/build_image.sh -s <dir>` wires it up. CI just never passes `-s` today.

---

## Part 3 — Step-by-step implementation

Work through these in order. Steps 1–5 are "test against a pinned commit" and are useful on
their own; steps 6–9 add the automatic promotion; 10–12 are polish. Land them as separate
PRs — each is independently reviewable and revertable.

### Step 0 — Access you need before you start

- [ ] Buildkite access to the `tpu-commons` org with permission to **create a pipeline** and
      **add a schedule**. Ask the CI oncall if you don't have it.
- [ ] **No token to mint** — see §1.5. What you *do* need to confirm is that GitHub App
      installation `142868369` grants **contents: write** on `vllm-project/vllm-torchtpu`
      (it's an org-level install, so it very likely already does). Quickest check: run a
      throwaway Buildkite job on the `cpu_64_core` queue that pushes an empty commit to a
      scratch branch. If that works, promotion will work.
- [ ] Confirm with the repo admins whether `main` in `vllm-project/vllm-torchtpu` has branch
      protection. If it does, the App needs a push allowance, **or** you switch promotion to
      "open a PR" instead of "push to main" (see Step 7, variant B).
- [ ] A TPU queue you can burn on a daily extra run (`tpu_v7x_2_queue` / `tpu_v7x_8_queue`).
      Sanity-check capacity with the team — the integration pipeline roughly doubles daily
      TPU consumption.

### Step 1 — Introduce the pin file

Create `.buildkite/vllm_lkg.version` containing exactly one line: the full 40-character
commit SHA of the vLLM we're on today, so the first state is a no-op change in behavior.
The current pin is `vllm==0.25.1`, and `v0.25.1` is a lightweight tag, so it resolves
directly to a commit:

```bash
git ls-remote --tags https://github.com/vllm-project/vllm.git v0.25.1
```

which today gives `752a3a504485790a2e8491cacbb35c137339ad34`. Write it:

```bash
echo 752a3a504485790a2e8491cacbb35c137339ad34 > .buildkite/vllm_lkg.version
```

(Re-run the `ls-remote` when you actually do this — if someone bumps `pyproject.toml` in
the meantime, seed from *that* version's tag so the first commit changes nothing.)

Keep the file to a bare SHA with a trailing newline and nothing else — several scripts will
`cat` it, and a comment line would break them.

### Step 2 — Add the selector helper

In `.buildkite/scripts/configs/pipeline_config.sh`, add the same helper `tpu-inference`
uses (read the file, fall back to upstream HEAD):

```bash
VLLM_REPO_URL="https://github.com/vllm-project/vllm.git"

get_vllm_commit_hash() {
  # Load the vLLM commit from the LKG file; fall back to upstream HEAD if absent.
  local commit_hash="" version_file=".buildkite/vllm_lkg.version"
  if [ -f "$version_file" ]; then
    commit_hash="$(tr -d '[:space:]' < "$version_file")"
  fi
  if [ -z "${commit_hash:-}" ]; then
    commit_hash=$(git ls-remote "$VLLM_REPO_URL" HEAD | awk '{ print $1 }')
  fi
  echo "$commit_hash"
}
```

### Step 3 — Make CI build vLLM from that commit

This is the substantive change. Two decisions to make first:

**Decision 3a — where the clone happens.** Cloning inside `setup_docker_env.sh` and passing
`-s` to `build_image.sh` reuses the existing, tested `VLLM_SOURCE` path and needs no
Dockerfile change. The alternative (clone inside the Dockerfile like `tpu-inference` does)
is a bigger diff and loses Docker layer caching on the clone. **Recommend: clone on the
host, pass `-s`.**

**Decision 3b — the double install.** The `base` stage installs `vllm==0.25.1` from PyPI
via `-r pyproject.toml`; the `ci` stage then reinstalls vLLM from source over it. That
works (and keeps the transitive dependency set resolved by the PyPI metadata), at the cost
of one wasted wheel download. **Recommend: accept it for v1**, and only add a build arg to
skip the PyPI vLLM if image build time actually becomes a problem. If you do skip it,
you must verify that vLLM's own transitive deps still get installed.

Then, in `.buildkite/scripts/setup_docker_env.sh`:

```bash
# Resolve the vLLM commit chosen by bootstrap.sh. Inside Buildkite this must be set.
VLLM_COMMIT_HASH="$(buildkite-agent meta-data get "VLLM_COMMIT_HASH" --default "")"
if [ -z "${VLLM_COMMIT_HASH}" ]; then
  echo "[FATAL] VLLM_COMMIT_HASH metadata is empty; bootstrap.sh did not set it." >&2
  exit 1
fi

# Shallow-fetch just the commit we need.
VLLM_SRC="$(mktemp -d)/vllm"
git init "${VLLM_SRC}"
git -C "${VLLM_SRC}" remote add origin https://github.com/vllm-project/vllm.git
git -C "${VLLM_SRC}" fetch --depth 1 origin "${VLLM_COMMIT_HASH}"
git -C "${VLLM_SRC}" checkout FETCH_HEAD

# Match the local dev instructions in README.md: don't drag in the upstream
# tpu-inference plugin alongside vllm-torchtpu.
sed -i '/tpu-inference/d' "${VLLM_SRC}/requirements/tpu.txt"

# The vLLM commit is part of the tag. Without this, a CI build (LKG) and an
# integration build (upstream HEAD) of the same repo commit collide in the
# registry and one silently overwrites the other.
IMAGE_TAG="${IMAGE_REPO}:${COMMIT_HASH}-${VLLM_COMMIT_HASH}"

./docker/build_image.sh --target dev -t "${IMAGE_TAG}" -s "${VLLM_SRC}"

# Publish the tag so every downstream step runs the *same* image.
buildkite-agent meta-data set "CI_IMAGE_TAG" "${IMAGE_TAG}"
```

Add the equivalent of `verify_image_vllm` right after the build — it's ~10 lines and it is
what stops us from ever promoting an untested hash:

```bash
actual="$(docker run --rm --entrypoint git "${IMAGE_TAG}" \
  -C /root/torchtpu-vllm/vllm -c safe.directory=/root/torchtpu-vllm/vllm rev-parse HEAD)"
if [ "${actual}" != "${VLLM_COMMIT_HASH}" ]; then
  echo "[FATAL] image contains vLLM ${actual}, expected ${VLLM_COMMIT_HASH}" >&2
  exit 1
fi
```

> ⚠️ `build_image.sh` copies `$VLLM_SOURCE` into `${REPO_ROOT}/vllm` and `rm -rf`s it
> afterwards. A full vLLM tree is not small — check the CPU builder's disk headroom, and
> keep the fetch shallow (`--depth 1` on the specific SHA, as above).

### Step 4 — Teach `bootstrap.sh` to select and publish the hash

In `.buildkite/scripts/bootstrap.sh`, after the priority/skip logic and before the pipeline
uploads:

```bash
INTEGRATION_SLUG="vllm-torchtpu-integration"   # must match the Buildkite pipeline slug

if [[ "${BUILDKITE_PIPELINE_SLUG:-}" == "${INTEGRATION_SLUG}" ]]; then
  # Integration pipeline always tests upstream vLLM HEAD.
  VLLM_COMMIT_HASH=$(git ls-remote "$VLLM_REPO_URL" HEAD | awk '{ print $1 }')
else
  # PR / post-merge / nightly builds test the pinned LKG commit.
  VLLM_COMMIT_HASH=$(get_vllm_commit_hash)
fi
# Allow a human to override on a manual build via the VLLM_COMMIT_HASH env var.
VLLM_COMMIT_HASH="${VLLM_COMMIT_HASH_OVERRIDE:-$VLLM_COMMIT_HASH}"
buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
echo "Using vLLM commit: ${VLLM_COMMIT_HASH}"
```

Nothing else in the repo should read `.buildkite/vllm_lkg.version` during a build — always
go through the metadata, so an override applies everywhere at once.

### Step 5 — Point the test steps at the same image

`.buildkite/scripts/run_in_docker.sh` currently recomputes the tag as
`${IMAGE_REPO}:${BUILDKITE_COMMIT}`. Since the tag now includes the vLLM hash, change it to
consume the metadata published in Step 3:

```bash
IMAGE_TAG="$(buildkite-agent meta-data get "CI_IMAGE_TAG" --default "")"
if [ -z "${IMAGE_TAG}" ]; then
  # Local / non-Buildkite use.
  IMAGE_TAG="${IMAGE_REPO}:${BUILDKITE_COMMIT:-latest}"
fi
```

**Land Steps 1–5 as one PR and stop here for a few days.** At this point CI builds vLLM
from a pinned commit instead of PyPI, but nothing promotes yet. Confirm PR builds, nightly,
and perf runs are all still green before adding automation on top.

### Step 6 — Add the promotion script

Create `.buildkite/scripts/update_lkg_version.sh`. This is `tpu-inference`'s script with
the repo URL changed and **all of the credential plumbing deleted** — no `GITHUB_PAT` check,
no `git remote set-url` with an embedded token. The agent's credential helper (§1.5) handles
authentication for the `git push` on its own:

```bash
#!/bin/bash
set -euo pipefail

NEW_LKG_HASH="${1:-}"
[ -n "${NEW_LKG_HASH}" ] || { echo "Error: no hash provided." >&2; exit 1; }

TARGET_BRANCH="main"
VERSION_FILE=".buildkite/vllm_lkg.version"

# Push credentials come from the agent-level GitHub App credential helper
# (ci-infra: terraform/gcp_old/tpu-inference/modules/*/main.tf). Nothing to inject here.
git config user.name  "Buildkite Bot"
git config user.email "buildkite-bot@users.noreply.github.com"

git fetch origin "${TARGET_BRANCH}"
git checkout -f "${TARGET_BRANCH}"
git reset --hard "origin/${TARGET_BRANCH}"

echo "${NEW_LKG_HASH}" > "${VERSION_FILE}"
git add "${VERSION_FILE}"

if git diff --cached --quiet; then
  echo "No change in LKG version. Skipping push."
else
  # [skip ci] is required: bootstrap.sh honors it, and without it this push
  # retriggers CI on every promotion.
  git commit -s -m "[skip ci] Update vLLM LKG to ${NEW_LKG_HASH}"
  git push origin "${TARGET_BRANCH}"
  echo "Promoted LKG to ${NEW_LKG_HASH}"
fi
```

`chmod +x` it, and note the `-s` (DCO sign-off) — this repo requires it.

> If the push ever fails with an auth error, the fix is **not** to add a token here. Either
> the step landed on a queue whose agents lack the credential helper (see §1.5), or the App
> installation doesn't grant write on this repo. Fix it at that level.

### Step 7 — Add the promote pipeline fragment

Create `.buildkite/pipeline_promote.yml`:

```yaml
steps:
  - wait: ~          # blocks until every preceding step passes; skipped if any failed
  - label: ":arrow_up: Promote vLLM to Last Known Good (LKG)"
    key: "promote_lkg"
    agents:
      queue: cpu
    commands:
      - |
        VLLM_LKG_HASH=$(buildkite-agent meta-data get "VLLM_COMMIT_HASH")
        bash .buildkite/scripts/update_lkg_version.sh "$$VLLM_LKG_HASH"
        buildkite-agent annotate ":white_check_mark: LKG updated to $$VLLM_LKG_HASH" --style "success"
```

Note what's *absent*: no `secrets:` block. Authentication is the agent's job (§1.5).

Two details that are easy to get wrong:

- The `$$` is required — Buildkite interpolates `$VAR` at *upload* time, and we want the
  shell to expand it at *run* time.
- The queue must be one whose agents carry the credential helper. `cpu` and `cpu_64_core`
  both do; the `benchmark` pool does not.

**Variant B (if `main` is protected and the bot can't push):** have the script create a
branch `bot/vllm-lkg-<short-sha>` and open a PR, auto-merge enabled. Note that `gh` needs an
explicit `GH_TOKEN` — the git credential helper covers `git`, not the GitHub REST API — so
this variant is the one case where you'd mint a token in-pipeline (mirror
[`export_ci_bot_token.sh`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/export_ci_bot_token.sh)).
A `git push` of the branch plus a PR opened by a human is the cheaper middle ground. Same
trigger and gating either way; only the last few lines of Step 6's script change.

### Step 8 — Wire it into `bootstrap.sh`

In the upload section:

```bash
if [[ "${BUILDKITE_PIPELINE_SLUG:-}" == "${INTEGRATION_SLUG}" ]]; then
  # Uploaded steps are inserted in reverse order, so the promote fragment must be
  # uploaded FIRST in order to run LAST — after all tests.
  upload_with_priority .buildkite/pipeline_promote.yml "$JOB_PRIORITY"
fi

upload_with_priority .buildkite/pipeline_perf.yml        "$JOB_PRIORITY"
upload_with_priority .buildkite/pipeline_integration.yml "$JOB_PRIORITY"
upload_with_priority .buildkite/pipeline_tests.yml       "$JOB_PRIORITY"
upload_with_priority .buildkite/pipeline_build.yml       "$JOB_PRIORITY"
```

Also check the `if:` conditions in the step files. Several steps today are gated on
`build.env("NIGHTLY") == "1"` — for example every step in `pipeline_integration.yml` and
the nightly unit-test variants. **The integration pipeline must run the full suite, or you
will promote a commit that only passed the PR subset.** Either set `NIGHTLY=1` on the
integration pipeline's schedule, or widen those conditions to
`build.env("NIGHTLY") == "1" || build.pipeline.slug == "vllm-torchtpu-integration"`.
Decide this explicitly and write the choice into the PR description.

### Step 9 — Create the Buildkite pipeline and schedule

In the Buildkite UI (org `tpu-commons`):

1. **New pipeline**, slug **`vllm-torchtpu-integration`** — it must match `INTEGRATION_SLUG`
   in `bootstrap.sh` exactly.
2. Repository: `git@github.com:vllm-project/vllm-torchtpu.git`. Steps — the same single
   bootstrap step the main pipeline uses:
   ```yaml
   steps:
     - label: ":pipeline: Upload Pipeline"
       agents:
         queue: cpu
       command: "bash .buildkite/scripts/bootstrap.sh"
   ```
3. **Disable the GitHub webhook trigger** on this pipeline. It should run on a schedule
   only, never on push — otherwise every merge doubles TPU usage.
4. **Schedule:** daily, off-peak, branch `main`, with `NIGHTLY=1` in the env if you took
   that route in Step 8. Start with daily; tighten only if upstream churn justifies it.
5. **Secrets: none.** Nothing to add — the agents authenticate to GitHub themselves (§1.5).
   The only thing to verify is that App installation `142868369` has **contents: write** on
   this repo (Step 0). If it doesn't, that's a change in
   [`ci-infra`](https://github.com/vllm-project/ci-infra) / the App's GitHub settings, not
   in this repo.
6. **Notifications:** copy the `tpu-inference` pattern — notify oncall/Slack when the build
   was triggered by a schedule, and only the build creator on manual runs, so debugging
   runs don't page anyone.

### Step 10 — (Optional, recommended) the audit trail

Mirror `verified_commit_hashes.csv`: a nightly step that appends
`timestamp,vllm_commit_hash,vllm_torchtpu_commit_hash` and pushes — again with no
credential handling of its own. It costs almost nothing and it's the first thing you'll want
when someone asks "when did this last work?". Gate it to `NIGHTLY=1 && branch == main`, not
for credential reasons but because that's the only run whose result the row is claiming.

### Step 11 — Update the human-facing docs

The pin now lives in one place; make sure nothing contradicts it:

- `README.md` — the two `git clone --depth 1 --branch v0.25.1 ...` blocks should become
  something like:
  ```bash
  export VLLM_COMMIT_HASH="$(cat .buildkite/vllm_lkg.version)"
  git clone https://github.com/vllm-project/vllm.git ../vllm
  git -C ../vllm checkout "${VLLM_COMMIT_HASH}"
  ```
  Also revisit `SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1` and the "Currently `vllm==0.25.1` is
  supported" note — off a post-release commit, the version is no longer exactly `0.25.1`.
- `pyproject.toml` — decide what `vllm==0.25.1` means now. It is still the right pin for
  users installing from PyPI, but it is **no longer what CI tests**. At minimum add a
  comment saying so and pointing at `.buildkite/vllm_lkg.version`. Agree with the team on
  when the PyPI pin gets bumped (suggestion: at each upstream release, verified against the
  LKG loop).
- `.buildkite/README.md` currently says Buildkite "is not currently configured or active".
  That's stale; replace it with a short description of the pipelines and a pointer to this
  document.

### Step 12 — Validate before trusting it

1. **Dry run the promotion script locally**: run it with a bogus hash against a scratch fork
   and confirm the commit message, sign-off, and no-op path (`git diff --cached --quiet`).
2. **Manual integration build** from the Buildkite UI, on a branch, with the promote step's
   push replaced by an `echo`. Verify: the metadata hash is upstream HEAD, the image tag
   contains it, the in-image `git rev-parse` check passes, and the promote step lands last
   in the step list.
3. **Force a red run**: point it at a known-broken vLLM commit and confirm the promote step
   is skipped and the pin file is untouched.
4. **Enable for real**, then watch the first three scheduled runs before announcing it.

---

## Part 4 — Runbook (after it ships)

**What vLLM are we on?**
```bash
cat .buildkite/vllm_lkg.version
```
The Buildkite build log also prints `Using vLLM commit: <sha>` in the bootstrap step.

**What changed upstream since our pin?**
```bash
git -C ../vllm log --oneline "$(cat .buildkite/vllm_lkg.version)"..origin/main
```

**Reproduce a CI environment locally:**
```bash
export VLLM_COMMIT_HASH="$(cat .buildkite/vllm_lkg.version)"
git -C ../vllm fetch origin "${VLLM_COMMIT_HASH}" && git -C ../vllm checkout "${VLLM_COMMIT_HASH}"
./docker/build_image.sh --target dev -t torchtpu-vllm-local -s ../vllm
```

**Pin to a specific commit by hand / roll back a bad promotion.** Edit the file, commit,
merge. That's it — it's one line, and reverting the bot's commit is a valid rollback.

**Test one PR against a different vLLM without moving the pin.** Trigger a manual build
with `VLLM_COMMIT_HASH_OVERRIDE=<sha>` in the build's env (Step 4).

**Integration has been red for days.** The pin is *safe* — PR CI is unaffected — but it is
also *rotting*, and the longer it sits the more upstream commits you'll have to bisect.
Triage:
1. Is the failure upstream's or ours? Re-run the same test against the current LKG commit;
   if it passes, it's upstream's change.
2. Bisect with manual integration builds using `VLLM_COMMIT_HASH_OVERRIDE`.
3. If upstream broke us intentionally (an API change), fix forward in this repo. If it's an
   upstream bug, file it there and leave integration red — that redness *is* the signal.

**Promotion stopped happening but integration is green.** Usual suspects, in order:
the promote step silently no-op'ing because the hash didn't change (upstream had no new
commits); the schedule got disabled; branch protection rejecting the bot push; the App
installation losing write access to this repo; the promote step landing on a queue whose
agents lack the credential helper. Check the promote step's log and its Buildkite annotation
first. For auth failures specifically, the credential helper fails *silently* — it just
prints nothing and git falls back to unauthenticated — so the symptom is a bare
`403`/`could not read Username`, not a helpful error. Reproduce with a scratch-branch push
on the same queue before touching anything in this repo.

**The pin moved and now a PR is failing.** Look at the promotion commit
(`git log --oneline -- .buildkite/vllm_lkg.version`) to get the vLLM range, then the
upstream diff. Reverting the promotion commit is an acceptable stop-the-bleeding move;
open an issue so the pin doesn't stay frozen.

---

## Appendix — Files to be created or touched

| File | Action |
|---|---|
| `.buildkite/vllm_lkg.version` | **new** — the pin |
| `.buildkite/scripts/configs/pipeline_config.sh` | add `get_vllm_commit_hash()`, `VLLM_REPO_URL` |
| `.buildkite/scripts/bootstrap.sh` | select HEAD vs pin, publish `VLLM_COMMIT_HASH`, upload promote fragment on the integration slug |
| `.buildkite/scripts/setup_docker_env.sh` | fetch vLLM at the commit, `-s` into the image build, vLLM hash in the tag, verify image, publish `CI_IMAGE_TAG` |
| `.buildkite/scripts/run_in_docker.sh` | consume `CI_IMAGE_TAG` |
| `.buildkite/scripts/update_lkg_version.sh` | **new** — commit + push the pin (no credential handling; see §1.5) |
| `.buildkite/pipeline_promote.yml` | **new** — `wait` + promote step |
| `.buildkite/pipeline_tests.yml`, `pipeline_integration.yml` | widen `NIGHTLY`-gated conditions to cover the integration pipeline |
| `.buildkite/README.md` | replace the stale placeholder |
| `README.md`, `pyproject.toml` | reconcile the human-facing vLLM version references |
| Buildkite UI | new `vllm-torchtpu-integration` pipeline + schedule (no secrets) |

Reference implementation to read alongside this doc — everything above is adapted from it:

- [`.buildkite/scripts/bootstrap.sh`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/bootstrap.sh)
- [`.buildkite/scripts/configs/pipeline_config.sh`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/configs/pipeline_config.sh)
- [`.buildkite/scripts/setup_docker_env.sh`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/setup_docker_env.sh)
- [`.buildkite/scripts/update_lkg_version.sh`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/update_lkg_version.sh)
- [`.buildkite/scripts/commit_verified_commit_hashes.sh`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/commit_verified_commit_hashes.sh)
- [`.buildkite/scripts/export_ci_bot_token.sh`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/export_ci_bot_token.sh) — in-pipeline App token exchange. Only needed if you call the GitHub REST API (e.g. `gh pr create`); plain `git` doesn't need it.

And the agent infrastructure that makes the pushes work — [`vllm-project/ci-infra`](https://github.com/vllm-project/ci-infra):

- [`terraform/gcp_old/tpu-inference/modules/ci_cpu/main.tf`](https://github.com/vllm-project/ci-infra/blob/main/terraform/gcp_old/tpu-inference/modules/ci_cpu/main.tf) — the GitHub App credential helper baked into the agent image ("Setup In-Memory GitHub App Authentication"). The `ci_cpu_64_core`, `ci_v6e`, and `ci_v7x` modules carry the same block.
- [`terraform/gcp_old/tpu-inference/cloud-ullm-inference-ci-cd/main.tf`](https://github.com/vllm-project/ci-infra/blob/main/terraform/gcp_old/tpu-inference/cloud-ullm-inference-ci-cd/main.tf) — the fleet this repo's queues (`cpu_64_core`, `tpu_v7x_2_queue`, `tpu_v7x_8_queue`) are instantiated from.
- [`.buildkite/integration_promote.yml`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/integration_promote.yml)
- [`.buildkite/nightly_verify.yml`](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/nightly_verify.yml)
- [`docker/Dockerfile`](https://github.com/vllm-project/tpu-inference/blob/main/docker/Dockerfile)
