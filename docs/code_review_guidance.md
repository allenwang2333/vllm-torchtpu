# vllm-torchtpu Code Review Guidance

How code review works in `vllm-project/vllm-torchtpu`, for both audiences:

- **Human reviewers** — what to look for, in what order, and how to phrase
  and route findings.
- **Review bots and coding agents** — treat the priority order and the output
  format below as the review protocol; emit findings in the table format of
  §4 and cite the rule each finding enforces.

This is the third document of the engineering-standard set. It says *how to
review*; the rules being enforced live in the other two:

| Document | Carries |
|---|---|
| [coding_guidance.md](coding_guidance.md) | `R-1.x`–`R-8.x` — style, architecture, reuse, derive-don't-declare, fail-fast, surface area, diff discipline, tests |
| [optimization_best_practices.md](optimization_best_practices.md) | `PR-x` (PR discipline), `P-x` (performance), `Q-x` (quantization) |

Cite those IDs in findings. A finding no rule covers is still valid — say so
and argue it on its merits.

---

## 1. Two facts that set the priority order

**Bugs here usually do not crash.** A wrong sharding annotation, a bad
quantization scale, a dtype cast that papers over a mismatch: the model runs
and quietly produces wrong numbers. There is no stack trace — only an eval
score a few points lower, found weeks later. So silent-numerics risk
outranks everything else in review, and "it ran and the output looked
reasonable" is not evidence.

**Upstream moves fast.** vLLM and JAX refactor constantly. Code that depends
on a *name* — a string-matched model id, a `getattr` with a default, a
hard-coded module path, a copied helper — breaks silently the day upstream
renames something. Code that depends on *structure* — shapes, dtypes, typed
configs, registries, hardware APIs — survives. So duplication and
name-based dispatch are review findings, not style preferences.

## 2. Priority order for findings

Check in this order. Report in this order. The default severity assumes no
other information; adjust with a stated reason.

1. **Silent numeric corruption / kernel breakage.** Unasserted quantization
   assumptions; dtype casts hiding a mismatch; scale-layout errors (a weight
   transpose does not imply a scale transpose); calls outside a kernel's
   supported envelope (mesh rank, EP-only, dtype packing, JAX version).
   Rules: R-5.2, R-5.5, P-17, Q-5.
   *Default: blocker.*
2. **Hot-path cost.** Work that belongs at weight-loading time; hidden
   barriers (`jax.named_scope`, `lax.cond` in a kernel's main loop);
   unpipelined DMA (`start(); wait()` back-to-back); silent upcasts to 16-bit
   compute; a "no-op on the hot path" claim with no large-batch trace.
   Rules: P-1, P-7–P-14.
   *Default: major; blocker once a regression is demonstrated.*
3. **Duplication / wrong home.** vLLM upstream already has it and should be
   subclassed; layering violation against the R-2.2 table; model-specific
   logic in a shared file; a kernel that imports vLLM or torch outside its
   adapter. Rules: R-2.2, R-2.3, R-3.1, R-3.3.
   *Default: major.*
4. **Fragility and scalability.** String or `hasattr` dispatch; magic
   constants that a shape, the mesh, or `config` could supply; env reads
   inside jitted code; a new knob where a derived default would do; a fix
   gated on the single config where the bug was observed instead of the
   variable that drives the behavior; "works for the official checkpoint".
   Rules: R-4.x, R-6.1, R-6.2, R-6.5, R-6.6.
   *Default: major; minor for one isolated instance.*
5. **Tests and evidence.** Does a new test fail without the fix (R-8.2); is
   it actually wired into CI and not skipped by a pipeline ignore list
   (R-8.4); does it exercise the forward pass rather than import the module
   (R-8.1); are perf and numeric claims backed by a trace or an eval with a
   mechanism explanation (P-1, Q-3).
   *Default: blocker for a missing required test; major for an unbacked
   claim.*
6. **Diagnosability.** New log lines are self-contained: subsystem tag, the
   relevant shapes and values, the failure mode named, so a reader with only
   the log can find the root cause and the file to open (R-1.7). Swallowed
   exceptions and bare `except` are findings here as well as under §2.1
   (R-5.1). Comments state what the code does now — a constraint, an
   invariant, a non-obvious why — not why the change is being made today
   ("previously X", "this fixes…"); that belongs in the PR description
   (R-1.8).
   *Default: minor; major when a failure would be undiagnosable.*
7. **Nits last, labeled as nits**, and deferred explicitly.

## 3. Severity rubric

| Severity | Meaning |
|---|---|
| `blocker` | Correctness bug, silent-numerics risk, missing required test, or a change that will break production or CI |
| `major` | Significant duplication, wrong layer or wrong home, perf-regression risk, unproven perf claim, substantially missing coverage |
| `minor` | Readability, naming, dead code — worth doing, does not block merge |
| `nit` | Stylistic preference; say it is a nit and defer it |

## 4. Output format

One row per distinct site, ordered by severity:

| ID | Severity | Rule | Location | Description |
|---|---|---|---|---|
| 1 | blocker | R-5.1 | `src/vllm_torchtpu/layers/vllm/fused_moe.py:142` | Bare `except` swallows the shape-mismatch error raised during weight processing; catch the specific exception or let it propagate. |

- **ID** — sequential integer from 1.
- **Rule** — the rule ID the finding enforces; leave empty when no rule
  covers it.
- **Location** — `path:line`, or `path:line-line` for a range. When the same
  issue recurs, list the primary site and add "(also at L120, L155)" to the
  description instead of repeating rows.
- **Description** — what is wrong and what to do, in one sentence. Name the
  exact reuse target or fix; "duplicated" alone is not actionable.

After the table, add a short paragraph for any cross-cutting concern that
does not fit one row (for example: the whole `process_weights` path is
duplicated across two quant methods and should be lifted).

## 5. Standard probes

Questions that repeatedly find real problems:

- *What happens if this line does not exist?* — the universal probe (R-7.4).
- *Can this be derived from the shapes or the config instead of passed in?*
  (R-4.x)
- *Is there an existing vLLM function to subclass instead?* (R-3.1)
- *Can you share a trace?* — for any perf claim (P-1).
- *What is the mechanism behind the gain?* — a win the mechanism does not
  predict is usually a measurement artifact (P-4).
- *What happens with an odd block size, or when the checkpoint is not the
  official one?* (Q-5, R-6.6)
- *Does this test fail if you revert the fix?* (R-8.2)

## 6. Review conduct

- **Ask more than you direct.** A question that makes the author explain the
  mechanism finds more bugs than an instruction, and is often how the
  reviewer discovers they were wrong.
- **State your confidence.** "I may be wrong here, happy to hear the
  counterargument" is a normal thing to write. Reverse yourself publicly
  when the author's answer is right.
- **Escalate hard only for numeric safety and modularity.** "Please roll
  this back" is reserved for those; everything else is a conversation.
- **Ask for the context you need.** A summary of the core issue and how the
  change fixes it is a reasonable thing to request before reviewing, and
  produces better feedback than guessing.
- **Verify the fix landed.** Re-read the diff after the author says done; a
  thread is not resolved because someone replied to it.
- **Redirect work that conflicts with the roadmap.** A contribution to a
  component scheduled for deletion should be pointed at the successor
  instead.
- **Say what you approve.** When a PR carries deferred scope, name it in the
  thread rather than letting it disappear on merge.

## 7. Routing and merge gating

- **Route approval by ownership.** Changes under a kernel directory need
  that kernel's owner in addition to a general LGTM; tag the original author
  of the code being touched. For a large PR, delegate the first pass and
  reserve the second.
- **Oversized diffs may be refused.** "Oversized" is reviewer judgment, not
  a line count (R-7.1). A very large design starts as an RFC, not a PR.
- **Nothing merges until CI is green on the current head.** A green run on
  an older head is not a green PR. For an external contributor the reviewer
  triggers CI and merges only on verified green.
- **One escape valve.** A trivial fix that unblocks CI or `main` may merge
  before its test or cleanup — but only with an explicit follow-up
  commitment recorded in the thread, naming what was deferred. Deferred
  scope is named, never silently dropped.
- **A revert is a first-class mitigation.** When a landed change turns out
  to be numerically wrong, revert first and root-cause offline (PR-2).

## 8. Reviewer quick checklist

Run top to bottom on any PR:

1. Does the description state the mechanism and the evidence, and does the
   title name the capability rather than the model that prompted it?
   (R-7.11)
2. Anything under §2.1 — asserts, dtype casts, scale layout, kernel
   envelope?
3. Does anything on the inference path belong at weight-loading time?
4. Does vLLM upstream already have this, and is each file in the layer
   R-2.2 assigns it?
5. Any string/`hasattr` dispatch, magic constant, or new knob?
6. Is there a test, does it fail without the fix, and does CI actually run
   it?
7. Do the new log lines identify a root cause on their own?
8. Is CI green on the current head, and is every deferred item named in a
   thread?
