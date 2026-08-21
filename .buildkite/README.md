# Buildkite CI/CD Pipelines

## The vLLM last known good (LKG) pin

This repo tracks upstream vLLM, so every build has to answer: *which vLLM?*

The answer lives in exactly one place — the `vllm` dependency in
`pyproject.toml`:

```toml
"vllm @ git+https://github.com/vllm-project/vllm.git@<ref>",
```

That ref is the **last known good** revision: the newest upstream commit that
passed our full suite. Because it is an ordinary dependency, CI, the prod
image, and a plain `pip install` all get the same vLLM with no extra steps.
`docker/vllm_pin.sh` is the only thing that reads or writes it.

### The promotion loop

The `vllm-torchtpu-integration` pipeline is the one that tracks upstream:

```text
integration build
  └─ bootstrap resolves vLLM main HEAD, publishes it as VLLM_COMMIT_HASH
     └─ image built against that commit, tagged <repo-sha>-<vllm-sha>
        └─ unit + integration + perf steps run
           └─ all green?  promote_lkg advances the pin on main
                          └─ every other pipeline now builds that vLLM
```

Anything that fails leaves the pin alone, so `main` only ever points at a vLLM
that has been through the suite end to end.

**The promotion step is report-only today.** It resolves and validates the
promotion, rewrites `pyproject.toml` in the build workspace, prints the exact
diff that would land, and stops — nothing is committed or pushed. That lets the
whole loop run before CI has write access to this repo. Flip it on by setting
`LKG_PROMOTE_PUSH=1` (see [Enabling the push](#enabling-the-push)).

### Which vLLM a build uses

`resolve_vllm_ref` in `scripts/configs/pipeline_config.sh` decides, in order:

| Condition | vLLM ref |
| --- | --- |
| `VLLM_COMMIT_HASH_OVERRIDE` is set | that ref, verbatim |
| pipeline is `vllm-torchtpu-integration` | upstream `main` HEAD, resolved at bootstrap |
| otherwise (default) | the LKG pin in `pyproject.toml` |

The bootstrap step resolves this **once** and publishes it as the
`VLLM_COMMIT_HASH` build metadata. Every later step reads that metadata rather
than re-resolving, so the build step and the test steps cannot disagree.

### Image tags

CI images are tagged `<vllm-torchtpu-commit>-<vllm-ref>`. Both halves matter:
the LKG and upstream-HEAD pipelines can build the *same* repo commit against
*different* vLLMs, and a tag naming only the repo commit would let them
overwrite each other in the registry — a test could then silently run a vLLM it
was never meant to.

As a second line of defence, each image records its installed vLLM revision at
`/etc/vllm-revision`, and `verify_image_vllm` fails the build if that does not
match what was requested.

### Running against a different vLLM

* One-off CI build: set `VLLM_COMMIT_HASH_OVERRIDE` in the build's env.
* Locally: `./docker/build_image.sh --vllm-ref <sha>`.
* Permanently: land a PR editing the pin (`docker/vllm_pin.sh write <ref>`).

### Enabling the push

To make promotions real, set `LKG_PROMOTE_PUSH=1` on the integration pipeline.
Two prerequisites first:

1. The GitHub App used by the `cpu`-queue agents
   (ci-infra, `terraform/gcp_old/tpu-inference/modules/ci_cpu`) must have write
   access to this repo — it provides the git credentials `promote_lkg` pushes
   with, so no PAT is involved.
2. If `main` is protected, that App needs permission to push to it, or the
   promotion step will fail on every green run.

Until then the step still runs on every green integration build and logs the
promotion it would have made, which is enough to confirm the pin resolution,
the image tagging and the gating all behave.
