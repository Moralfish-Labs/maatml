# MaatML Roadmap

MaatML is where a small, task-specific model is built, weighed, and packaged.
The runtime is where it answers. The lifecycle runs validator-checked seed
generation and ingestion → training → evaluation → export → serving, with
plugins that distill models and export them to the runtimes that load them.
Purpose: task-specific domain language models with a small footprint, from a
multi-gigabyte base model to a small subject-matter expert. Correctness is
checked outside the model via validators (the Maat weighing thesis).

**Design rule:** core owns architectures (`causal_sft`, `seq2seq`,
`multi_head_classifier`, `dpo` / `orpo`); examples own task semantics
(validators, metrics, tokenizers, generators, sanitizer rules, seed builders).
Plugin-owned architectures (`vision_multitask`, `vlm_sft`) register from their
model folders.

**Positioning:** MaatML runs its own **fixed** validator-gated lifecycle. It is
not a general-purpose workflow scheduler or arbitrary DAG engine (no
triggers/cron, no shell/python steps, no remote executors). Source-mutating
operations (`datagen`, `ingest`, `distill`, `mint`, reviewed capture, and rollout
collection) stay explicit and outside
the default runner.

**Claims rule:** the guarantee register ("gates", "cannot", "never") ships only
together with the mechanism that enforces it and a test that proves it. Exit
criteria in this file are testable assertions; wording that cannot be tested
does not belong in them.

## Status

Version numbers are assigned only when a tranche ships. Planned tranches below
are ordered but unversioned.

Current package version: **0.15.0**. Streaming preparation, corrective training
inputs, population safeguards, rollouts and tool-call chat training are released. Historical tranches below describe completed work;
they are not the current task queue.

| Tranche | Theme | Status |
|---|---|---|
| Phase 0 | Rename to MaatML + examples-first restructure | Done |
| v0.2 | Experiment layer | Done |
| v0.3 | Methods + scale | Done |
| v0.4 | Product surface (export / deploy / flywheel) | Done |
| v0.5 | Serving + multimodal | Done |
| v0.5.1 | Truth and safety | Done |
| v0.6 | Truth and safety II: gates tell the truth | Done |
| v0.7 | Silent-failure hardening + test floor | Done |
| v0.8 | Fixed lifecycle runner: `maatml run` | Done |
| v0.8 | Distill + reviewed flywheel + serve contract | Done |
| v0.9 | Serving-bundle exports (MLX / ONNX), loud distill, content-level benchmark leakage check, lint/type gate | Done |
| v0.9.1 | Pluggable compile / serve runtimes, actionable `vision_scene` feedback, `maatml audit`, slimmer examples | Done |
| v0.10 | Gated device compiles (`compile --require-gated`), shared capture path, `ingest --video`, Anthropic Messages wire, `diffusion_lora` | Done |
| v0.11 | Evidence layer: derived gates, operating points, named populations, licensed sources, portable runs | Done |
| v0.11.1–0.11.2 | Attribution fields and recorded evaluation overrides; operating-point config validation | Done |
| v0.12 | Batched evaluation with per-row evidence and amortized latency reporting | Done |
| v0.13 | Lifecycle evaluation-schema parity and CLI version reporting | Done |
| v0.14 | Rollout environment protocol, DAgger labels, streamed progress and Wilson intervals; train-only rows; tool-call chat training and multi-turn masking; inline gate rewrite fix | Done |
| — | Slim artifact distribution | Planned |

## Current ownership and next work

- **Framework:** the fixed lifecycle, plugin dispatch, dataset isolation,
  training, evaluation, gate evidence, export, serving, and portable run records.
- **Model folder:** task semantics, validators, metrics, architectures supplied
  by plugins, datasets, recipes, and benchmarks.
- **External orchestrator:** scheduling, notebook/VM provisioning, credentials,
  object-storage transfers, monitoring, retry policy, and resource cleanup.
  These remain outside the framework; see [orchestration](docs/orchestration.md).

Before the next package release, run the lint, formatting, type, test,
standalone-wheel and CPU lifecycle CI gates against the same revision, build
the docs with `mkdocs build --strict`, and move the accepted Unreleased entries into versioned release
notes. A passing unit suite does not establish every architecture/device
combination or a model's production quality.

The next planned product tranche remains slim artifact distribution. There is
no `publish`/`pull` command or signing guarantee in the current CLI. Portable
`runs --pack` / `--adopt` already exist for moving training evidence between
machines; they are distinct from a public model distribution service.

## Non-goals

- Arbitrary shell or Python pipeline steps
- Broad model zoo competing with Axolotl / LLaMA-Factory / Unsloth

Completed release detail lives in [CHANGELOG.md](CHANGELOG.md). This file
carries outcomes, dependencies, and measurable exit criteria.

## v0.2: Experiment layer

Infrastructure everything else consumes.

- **Run registry**: `output/runs.jsonl`, per-run checkpoint dirs, `maatml runs`,
  replace mtime-based `_latest_checkpoint`
- **Tracking passthrough**: `training.report_to` / `run_name` → HF
  `TrainingArguments` (W&B / TensorBoard if installed)
- **Per-head loss logging**: generic `loss_<head>` from `multi_head`
- **Adapter-only artifacts**: `training.lora.save_mode: merged|adapter|both`
- **Tokenize-once cache** + `group_by_length`
- **Format adapters**: `alpaca` / `sharegpt` as format plugins; multi-turn
  loss masking in `causal_sft`
- **Eval gates**: `evaluation.gates:` + `maatml evaluate --gate` (non-zero exit)
- **Hygiene**: wire or remove dead `dataset.group_by`

## v0.3: Methods + scale

- QLoRA / quantized bases (`[cuda]` extra, bitsandbytes; CUDA-only)
- DPO / ORPO via TRL (`[pref]` extra); preference pairs mintable from validator
  outcomes (`mint_preference_pairs` library helper)
- CUDA profile maturation (wired `weights_dtype_policy`, flash-attn passthrough,
  `dataloader_workers` override)
- Multi-GPU via accelerate / torchrun (rank-0 run-registry writes)
- HPO / sweep hooks (`--set` overrides, `maatml sweep`)

## v0.4: Product surface

- **Export**: `maatml export --format gguf|mlx|safetensors` + manifest from
  `PackagingSpec`; post-export parity gate on pinned benchmarks
- **Checksums / revision pinning**: `maatml verify` (sha256 integrity);
  `training.model_revision`. Cryptographic *signing* is deferred (see Later)
- **Data flywheel**: `maatml datagen` / `ingest`; shared gated builder;
  optional OpenAI-compatible teacher behind the validator gate (`[teacher]`)
- **Docs site**: mkdocs-material (`[docs]`), plugin-author guide

## v0.5: Serving + multimodal

Shipped in 0.5.0 (see [CHANGELOG.md](CHANGELOG.md)).

- **Serving**: `maatml serve` (`/health`, `/info`, `/predict`;
  `/predict?validate=1` re-runs the registered validator when configured)
- **Vision**: `vision_multitask` + ONNX export (`[vision]`)
- **Vision-language**: `vlm_sft` (SmolVLM; vLLM-servable; `[vllm]` Linux-only)
- **Captioning**: `examples/vision-describer` (seq2seq over vision JSON)
- **Trusted Publishing**: PyPI OIDC publish workflow

## v0.5.1: Truth and safety (Done)

The thesis is "ships with a contract"; make the public claims true before
building on them.

**Depends on:** v0.5.

- Support-ticket-triage gained a real validator plugin (`triage_plugin`:
  JSON → schema → `category → team` routing contract → summary quality) +
  `evaluation.gates` + a fixed benchmark + tests. It was the only reference
  example without a validator 
- jcl-validator and spool-interpreter promoted their README target tables into
  enforced `evaluation.gates` in `model.yml`, so **every** example now gates
  before the strict `--gate` change and the lifecycle runner depend on them 
- `evaluate --gate` with no gates configured now fails (exit non-zero) instead
  of passing vacuously, a behavior change (exit code 0 → non-zero), noted in
  CHANGELOG 
- Manifest `weights_dtype` verified from the actual exported tensors, with the
  declared value kept as `weights_dtype_declared` and a `weights_dtype_verified`
  flag 
- Serve: dropped the wildcard CORS default (opt-in `--cors` /
  `MAATML_SERVE_CORS`); added a request-body size cap (`--max-body-bytes`) 
- CI: Python 3.13 in the test matrix; a `[ml]` CPU `ml-smoke` job running
  `prepare → train --smoke → evaluate` on triage . **Follow-up (maintainer):**
  publish-workflow rehearsal against TestPyPI needs a TestPyPI trusted-publisher
  configured first (OIDC), so it is left as a maintainer setup step.

## v0.6: Truth and safety II: gates tell the truth (Done)

**Tracking:** #13

Every place the validator contract can quietly
degrade, every command that can destroy user data, a security floor for serve
and plugin execution, and config honesty. Small diffs, high stakes. This
tranche is what makes the README's three-stage gating sentence true as written.

**Depends on:** v0.5.1.

**Gating is real at all three stages**

- `datagen` fails closed when no `evaluation.validator` is configured
  (reuse the `GateConfigError` pattern); explicit `--allow-ungated` escape
  hatch marks the summary line and dataset card `UNGATED` (test)
- `serve --enforce`: validation failures return HTTP 422; `?validate=1`
  remains as annotation mode. README / docs index / lifecycle keep the
  "gates serving" wording only once this ships (test)
- Sanitize tags apply on **all** format paths (`alpaca` / `sharegpt` /
  preference) or the command errors on an unsupported combination; the
  dataset card records what actually ran, not what was declared (test)
- `ingest`: rows missing the gold field are rejected or separately counted
  (`skipped_unvalidated`); `--map` errors when a source column matches zero
  input rows (test)
- Serve validator call shape resolved once at startup (no TypeError fallback
  that silently weakens validation to a raw-output call)
- A configured `evaluation.validator` that does not resolve to a registered
  validator is an error at evaluate / datagen time (`GateConfigError`
  pattern), not a silent downgrade to JSON-parse-only scoring

**No command destroys user data**

- `scaffold` refuses to overwrite an existing `model.yml` or seed corpus
  without `--force` (test)
- `datagen` writes seed files atomically (temp + `os.replace`) and refuses to
  truncate a non-empty seed file on a zero-accepted run (test)
- `runs.jsonl`: single-write appends; readers skip-and-warn unparseable lines
  (quarantine to `runs.jsonl.corrupt`) instead of failing every consumer (test)

**Broken as shipped → working**

- `--resume auto|RUN_ID` resolves the newest `checkpoint-*`
  (`get_last_checkpoint`) instead of the run root

**Security floor**

- serve 500s return a generic message; tracebacks go to the server log only
  (client tracebacks solely under `--debug`); warn on non-loopback bind
- Trust boundary documented in README + SECURITY.md + docs/plugins.md:
  running any command against a model folder executes its `plugins:`, including `validate` and `plan`; add `validate --no-plugins` for schema and
  path checks without code execution
- Tokenized cache loads with `weights_only=True` (or moves to JSON), no
  pickle sink
- GGUF converter resolved from explicit config/env only (drop the generic
  `convert.py` PATH / cwd lookup)
- Vision predictors reject absolute and `..` image paths and confine reads to
  the model directory

**Config honesty**

- `--set` overrides re-validate (`validate_assignment` or re-parse after
  apply), the CLI can no longer bypass schema constraints it advertises (test)
- Known-keys warning pass for `dataset:` / `evaluation:` sections; `evaluate`
  prints "no validator configured, scoring JSON parse only" when that is
  what is happening
- docs/serving.md describes `verify` as corruption detection (listed files
  match their recorded sha256), not tamper evidence, guarantee wording
  returns when injected-file detection and signing ship

**Exit criteria:** the README three-stage gating sentence is true as worded,
each stage backed by a test (datagen fail-closed; serve `--enforce` → 422;
eval `--gate` already covered); `scaffold` and `datagen` cannot remove user
content without `--force` (tests); a torn last line in `runs.jsonl` is
quarantined and `maatml runs` still lists prior runs (test); `--resume auto`
resumes to completion from an interrupted smoke run that saved a mid-run
checkpoint (CI test; the smoke overlay sets `save_steps` low enough to
checkpoint within the smoke budget); `--set` of a schema-invalid value exits
non-zero (test); serve 500 response bodies contain no traceback (test).

## v0.7: Silent-failure hardening + test floor (Done)

**Tracking:** #14

The remaining "looks green, did nothing" paths, and tests for the parts of the
codebase that had none, the trainers (4 of 6 architectures) and the CLI.

**Depends on:** Truth and safety II.

**Honest numbers**

- Per-class eval numbers are `pass_rate` / `passed` / `n`. The old shape
  reported a real precision alongside a literal `recall: 1.0` and `f1: 0.0`
  for every category, so two of four numbers were invented
- seq2seq brace repair is opt-in (`evaluation.repair_braces`), counted, and
  recorded in `Report.extras`; both seq2seq examples set it
- `evaluate` defaults its token budget to `packaging.max_input_tokens` (parity
  with serve and `export --parity`) and counts truncated inputs
- `evaluation.metrics` as a list runs every entry and merges results; two
  plugins claiming one metric key is an error (test)

**Coercion, now loud failure**

- Fractional epochs honoured in seq2seq / multi_head; `precision` validated
  where it is parsed (test)
- Unknown gold labels are scanned before training and fail with the offending
  values counted; boolean gold maps through the declared label order (test)
- seq2seq rows with a missing or empty target are dropped and counted, never
  serialised to the literal `"{}"` (test)
- Preference rows serialise structured chosen/rejected as JSON (test); an
  identical pair warns; DPO / ORPO save through `_save_sft_artifacts`, so
  `lora.save_mode` means the same thing everywhere
- alpaca / sharegpt drop and count rows with no user or assistant content; an
  all-degenerate corpus fails (test)
- `multi_head` legacy JCL fallback fires only when the legacy keys are present;
  absent or malformed `training.heads` is an error (test)
- Teacher failures are counted, the first error is reported on the datagen
  card, and five consecutive failures abort instead of burning the attempts
  cap (test)
- A group key covering ~the whole corpus splits per row with a loud warning;
  teacher and ingest rows carry a per-row family; an empty split is reported
  (test)
- `prepare` refuses a benchmark whose rows share a group key with the training
  splits (test). The vision examples were copying the first N seed rows as
  their benchmark, so the pinned score measured memorisation; both builders now
  generate held-out rows in a `bench_*` family namespace
- Trainer bodies mark the run `aborted` for fallible work that used to sit
  outside the finish handler; `--resume auto` skips `running` records with no
  checkpoint (test)
- The SFT tokenize cache key includes `val.jsonl` content
- `sweep` records a failed trial, continues, and exits non-zero at the end;
  ranking compares only trials reporting the chosen metric, with the direction
  taken from the metric name (test)
- Sanitizer warns once per rule when length-preserving truncation cuts a
  redaction short, and rejects a fixed replacement that cannot fit its
  pattern's shortest match at load (test)
- `datagen` append dedups by content / `sample_id` and clears a stale reject
  report (test)
- Plugin lifecycle: `load_model_plugins` is the single idempotent owner (test);
  per-source failures are recorded, listed by `maatml plugins`, and named in
  `Unknown … plugin` errors (test); `discover_plugins` no longer wipes the
  registries when the trainer registry looks empty (test); the `Validator`
  protocol matches the real call sites
- Known CLI user errors print one actionable line; `maatml --debug` restores
  the traceback (test)

**Test floor**

- `typer.testing.CliRunner` suite: `evaluate --gate` exit codes, `verify` on a
  tampered manifest, `scaffold` refusal, per-command argument parsing, and a
  torn `runs.jsonl` (torch-free)
- Torch-gated unit tests for all four previously untested trainers (SFT label
  masking, seq2seq label padding, multi_head per-head targets, preference row
  loading), plus torch-free config-parsing tests
- Shared `tests/conftest.py` snapshots and restores registries through a public
  API (`snapshot_registries` / `restore_registries` / `reset_registries` /
  `Registry.unregister`); tests no longer touch `_entries`
- The ml job installs `[vision]` and runs `pytest tests/ examples/`, so the
  torch-gated tests and the vision end-to-end test execute; the matrix runs
  with `-rs`
- `macos-latest` torch-free job. The optional weekly scheduled MPS `[ml]` smoke
  was not added: it is an optimization rather than a claims item, and local
  development exercises MPS continuously
- `gh-action-pypi-publish` pinned to a commit SHA (Dependabot keeps it bumped)

**Exit criteria (met):** each of the four previously-untested trainer modules
has a torch-gated unit test exercising config parsing or label masking; the
CliRunner suite covers `evaluate --gate` exit codes, `verify` on a tampered
manifest, and `scaffold` refusal; the ml job runs pytest with the vision
end-to-end test green; preparing a datagen-produced corpus yields non-empty
val/test splits (regression test for the degenerate-group fix); CI includes a
macOS job.

## v0.8: Fixed lifecycle runner: `maatml run` (Done)

**Tracking:** #15

```
seed corpus
    │
    ▼
prepare ──► train ──► evaluate --gate ──► export ──► verify
                           │
                           └── gates fail → stop (non-zero)
```

Explicit source ops (`datagen` / `ingest` / `distill`) append to the seed
corpus and stay **outside** the default runner; changing seeds makes `prepare`
stale.

**Depends on:** Truth and safety II + Silent-failure hardening (every step the
runner chains must fail loud first; `runs.jsonl` must tolerate a torn line).

- **`maatml run <model-dir>`**: executes stale steps in order:
  prepare → train → evaluate (gates enforced) → export → verify.
  Flags: `--smoke`, `--force`, `--from STEP`, `--until STEP`, `--dry-run`,
  `--device`, `--set` (overrides feed the fingerprint)
- **Smoke-tier gates**: the `smoke:` overlay may override `evaluation.gates`;
  a smoke-gated run is marked as such in the run record and manifest, so a
  green smoke line stays distinguishable from a real gate pass
- **Fingerprints = idempotence, not speed.** Each step hashes effective config
  after smoke/overrides, declared input assets, prepared/eval data, checkpoint
  or manifest content, MaatML version / git SHA, plugin source hashes,
  device/profile, and exporter identity. Stored atomically in
  `output/pipeline.json`. Skip only when the fingerprint matches, the prior
  step completed, and expected outputs still verify. Train's fingerprint
  populates `RunRecord.spec_hash`. `--set` values are validated (Truth and
  safety II) before they enter a fingerprint
- **`maatml plan` becomes `run --dry-run`**: per-step fresh/stale status with
  the reason (which hash changed)
- Typed config where the runner fingerprints it (step sections), rather than a
  big-bang config rewrite. The typed surface includes `evaluation.validator`
  and gate keys

**Exit criteria (met):** `maatml run examples/support-ticket-triage/ --smoke`
goes seeds → verified export in one command in CI using smoke-tier gates, with
the run record and the export manifest marked smoke-gated; a re-run with no
changes does no work (CI asserts it); a run with a misspelled
`evaluation.validator` exits non-zero before any step runs; `run --set` with a
schema-invalid value exits non-zero and leaves `output/pipeline.json`
unchanged.

The smoke tier gates on `output_nonempty_rate`, a coverage metric maatml always
reports: a rehearsal cannot earn the production thresholds, and a 0.0 gate
would pass vacuously, but "the checkpoint saved, reloaded, and produced output"
is a real claim a smoke run can make.

## v0.8: Distill + reviewed flywheel + serve contract (Done)

**Tracking:** #16

**Depends on:** Truth and safety II + Silent-failure hardening (hardened
`TeacherClient` / `build_gated_corpus` / ingest), and preferably the lifecycle
runner (staleness).

- **`maatml distill`**: teacher responses for an unlabeled prompt pool; every
  row validator-gated before entering the seed corpus (reuse `TeacherClient`,
  `build_gated_corpus`, ingest field-mapping). Typed stage config (prompt
  source, teacher model/revision, retry/budget limits, output, provenance), no new untyped `dict[str, Any]` surface. Accepted rows carry teacher
  model/revision, prompt hash, source, family; rejection reports retained;
  offline replay supported. Worked example on triage
- **Serve capture**: opt-in, sanitized, size/retention-capped, local.
  Captured model predictions are **not** automatically gold; human/teacher
  correction and explicit approval are required before ingestion. Approved
  rows fold into seeds and make `prepare` stale, the retrain loop is
  `serve --capture` → review → `ingest` → `run`. Capture requires the serve
  auth token (below)
- **Serve contract** (promoted from Later: `--capture` and non-loopback binds
  need auth, and `--enforce` from Truth and safety II pairs with enforcement
  during generation):
  - serve auth token (required for `--capture`, recommended for non-loopback
    binds)
  - jsonschema-constrained decoding at serve time via a constrained-decoding
    logits processor (outlines or lm-format-enforcer, behind a serve extra);
    generative architectures only; structure enforced *during* generation
    instead of only rejected after
  - bounded retry-with-feedback: on validation failure, feed the error back
    to the model and re-ask once; retries counted and reported, never silent
- **Preference minting CLI**: surface `mint_preference_pairs` as an optional
  source operation (not a default `run` step) feeding `dpo` / `orpo`

**Exit criteria (met):** a validator-rejected teacher row is absent from the
produced seed corpus (test); `ingest` rejects a captured row lacking explicit
approval (test); `serve --capture` without the auth token refuses to start
(test); distill replay with the network disabled reproduces the accepted corpus
(a torch-free CLI step in CI plus a test against the shipped triage cache); a
serve request failing validation under `--enforce` returns 422 with the retry
count in the response (test).

jsonschema-constrained decoding is deferred rather than dropped: it needs a
serve extra (outlines / lm-format-enforcer) and only applies to generative
architectures, whereas bounded retry-with-feedback (shipped) enforces the
contract for every predictor today. Tracked for a serving follow-up.

## Slim artifact distribution (Planned)

**Tracking:** #17

**Depends on:** v0.5.1 (verified manifest dtype), and preferably the lifecycle
runner (verified export as a runner step). `verify`'s own blind spots close
inside this tranche, before publish/pull builds on it.

- **`verify` closes its blind spots first**: distribution multiplies bundles
  in the wild, so the verb it rests on must be exact:
  - files present in the export dir but absent from `manifest.json` are
    verification errors (injected-file detection)
  - `export --parity` writes its artifacts outside the bundle or re-emits the
    manifest afterwards, a documented flag must not invalidate the bundle it
    documents
  - with these shipped, docs may describe `verify` as integrity checking of
    the full bundle contents; tamper evidence still waits for signing (Later)
- **Canonical bundle digest**: stable digest from identity + formats + sorted
  file hashes in `manifest.json`
- **`maatml publish` / `pull`** with two backends: `local` (network-free
  reference, tested in CPU-free CI) and `hf` (Hub, `[hub]` extra); verify
  before upload and after download; immutable references; credentials
  delegated to native Hub auth
- **Auto `MODEL_CARD.md` at export**: recorded eval/gates + *explicitly
  declared* license metadata only; export must not network-lookup or infer a
  license

**Deferred until demand:** OCI backend, signing / SBOM, promotion channels /
ledger, `verify --require-signature`.

**Exit criteria:** `export → verify → publish → pull → verify` round-trips
through the local backend in CPU-free CI preserving the digest; a file
injected into a bundle after export fails `verify` (test).

## v0.11: Evidence layer: derived gates, named populations, portable runs (Done)

Shipped in 0.11.0 (#34). Two exit criteria need hardware and stay open as
noted below: a real `checkpoint-*` loading under `training.select_by`, and
the vision example on CPU-free CI.

v0.10 ends at "the run finished and the floors passed". Everything between a
finished run and an honest floor — deriving the floor, choosing the operating
point, naming which population a number came from, carrying the run record
home — is left to each model folder. The evidence that this is a gap rather
than a preference: the SIP detector folders (a production user of
`object_detection` through plugins) carry about fifty scripts across three
folders, at least eight of them in two or three near-identical copies
(`derive_gates.py`, `derive_operating_point.py`, `pack_colab.py`,
`write_model_card.py`, `check_split_hygiene.py`, …), plus twelve Colab
runners; and `docs/skill/` itself instructs a folder to carry
`scripts/derive_gates.py --write`. An external review round of that project's
training report asked for exactly the mechanisms below (clustered floors,
val-derived thresholds, a benchmark version, a blind population, seed
statistics, an environment manifest). Everything here is lifecycle; nothing is
an executor or a DAG.

**Depends on:** the v0.8 runner (every new step fingerprints its inputs and
records into `runs.jsonl`) and v0.8 distill (the KD cache gains provenance).
The sources attribution block extends the auto `MODEL_CARD.md` from Slim
artifact distribution and lands after it. The report schema lands first;
every other item writes into it.

**Eval reports have a schema and keep their predictions**

- `output/eval/<run>.json` carries `report_version`; a report missing a
  required field fails schema validation instead of being read as zeros
  (test)
- `evaluate --cache` persists per-row predictions keyed by run id and split
  hash; `gates derive` and `operating-point derive` read the cache instead of
  re-running inference (test: a derive from cache and a derive from a fresh
  evaluate of the same checkpoint produce identical floors)
- `evaluation.slices: [field, …]` — the harness emits `n`, rate metrics and
  intervals per value of every declared row field; plugins supply only the
  primary score and match (test: a declared slice with zero rows reports
  `n: 0` and no rate, never `0.0`)
- Detection metric helpers as a core library module plugins import — IoU
  matching with ignore regions, VOC11 / COCO-101 AP selected by config and
  named in the report, size buckets, `empty_clean`, `fp_per_frame`;
  architectures stay plugin-owned per the design rule (test: two plugins
  using the helpers agree on a fixture; a report without the AP protocol
  field fails validation)

**Gates are derived, tiered and population-stamped**

- `maatml gates derive --run R [--run R2 …] --write`: Wilson 95% lower bound
  at each metric's own denominator; per-metric minimum across the runs given;
  refuses thin denominators (`min_n`) and few-group sources (`min_groups`)
  and prints the refusal beside the metric rather than writing a floor (test)
- Group-cluster bootstrap for rate floors when rows carry the group key,
  replacing row-level Wilson on clustered populations; AP floors gain an
  image-level bootstrap interval (test: on a fixture whose rows cluster
  perfectly within groups, the clustered bound is wider than Wilson)
- Every written floor carries its derivation comment and the **benchmark
  version** it came from; `evaluate --gate` warns when the current benchmark
  version differs from the floors' and refuses under `--strict-population`
  (test)
- `evaluation.gates` entries may carry a tier, `blocking` (default) or
  `advisory`; an advisory miss is reported and recorded in `gate_evidence`
  and never fails the step; `compile --require-gated` reads the tier (test)
- `maatml ship-check CANDIDATE BASELINE`: absolute (floors), delta (no
  gated metric regresses beyond one row at n ≥ 30), and controlled replay
  (both checkpoints over identical rows when the benchmark version differs)
  in one verdict (test: a benchmark version change alone never yields
  "regressed")

**Operating points are chosen on val and spent once on test**

- `evaluation.operating_point: {threshold_key, objective, budget: {metric,
  max}, sources}`; `maatml operating-point derive --run R --write` sweeps the
  threshold over val predictions, takes the best objective under the budget,
  and writes `evaluation.<threshold_key>` with the sweep artifact as
  provenance; a val split with no rows in a required source refuses (test)
- Sweep points carry intervals; the sweep artifact records `n` per source
  (test)
- `--confirm-on-test` evaluates once at the written threshold and appends a
  test-spend record to `runs.jsonl`; `maatml runs` lists spends per benchmark
  version; a second confirm on the same benchmark version warns and is
  listed (test)
- The report records val-to-benchmark transfer: the metrics at the written
  threshold on val and on the benchmark, per source (test)

**Populations beyond one group key**

- `dataset.isolation: [clip, camera, site]` declares the hierarchy from row
  fields and the policy each population satisfies; `dataset.pins: {val:
  [...], benchmark: [...]}` assigns groups; `prepare` and `audit` refuse a
  group that spans populations under the declared policy (test)
- A benchmark has a version: hash of its rows plus the pins, written to the
  eval report and the floor block; `prepare` refuses to change
  `benchmark_samples` in place and requires a new version (test)
- `dataset.blind_samples`: a population that never enters train, val or the
  benchmark; `maatml evaluate --blind` runs once for a checkpoint whose
  train, decode and threshold fingerprints are unchanged since its last gated
  evaluate, records the spend, and refuses a second run for the same
  fingerprint without `--force` (test)
- `maatml train --seeds N`: one fixed recipe, N seeds, mean / sd / min / max
  per metric in one record, consumable by `gates derive` as the per-metric
  minimum (test)

**Sources carry their licence**

- `dataset.attribution`: a sidecar Markdown table (one row per `source`:
  licence, commercial-use `yes` / `no` / `unknown`, consent basis, sign-off;
  `dataset.sources[]` already names input paths, and licence rows are long
  and shared across folders, so they live beside the corpus). `prepare`
  refuses rows whose source has no entry or whose sign-off is blocked or
  unsigned, and refuses `no` / `unknown` commercial-use unless the sign-off
  itself records an accepted risk (`accepted-risk — <name> <date>`), which
  the corpus lock then carries; acceptance is signed in the table, not
  passed as a flag (test)
- Corpus lock: hash of every seed file that entered `prepare` plus the
  sources table, written under `output/prepared/` and into the export
  manifest (test)
- The auto `MODEL_CARD.md` (Slim artifact distribution) gains a per-source
  attribution block from the sources table; still declared metadata only, no
  inference or lookup. Until that card exists the block ships as the
  bundle's `ATTRIBUTION.md`, listed in the manifest (test)

**Training tells the truth earlier**

- `training.select_by: <val metric>` selects the checkpoint: every saved
  `checkpoint-*` and the final weights evaluated on val with the evaluate
  harness, the best recorded on the run and resolved by the run id from then
  on; `keep_checkpoints` sets how many survive to compare (test). There is no
  KD term in maatml yet; when top-k logit KD lands (Later) `select_by` is
  required with it, since val loss is not comparable across arms once a
  distillation term enters it
- `distill` refuses a prompt pool that overlaps a held-out population (by
  content against `benchmark_samples`, `blind_samples` and the prepared val /
  test split; by `dataset.pins` against a tagged pool row) before any teacher
  call, and the teacher cache carries a provenance header (teacher, prompt
  pools by sha256, benchmark version) (test). The label cache is keyed by
  prompt hash and teacher, so a cache cannot be "from another split"; the
  split-fingerprint refusal belongs to the logit KD cache under Later
- Named pathology signatures at evaluate — "never fires" (no output, or
  recall near zero with precision high or absent), "one class", "identical
  output across inputs", plus a plugin's `__pathologies__` — reported as
  `pathologies[]` and failing the smoke tier (test)

**Runs travel**

- `maatml runs <dir> --pack RUN` / `--adopt BUNDLE` (options on `runs`, since
  its positional model dir leaves no room for a subcommand): the run
  directory (without `checkpoint-*` unless `--with-checkpoints`), its
  `runs.jsonl` record, eval reports and caches, and an environment manifest,
  every file hashed; `--adopt` refuses a bundle whose model-folder
  fingerprint (`spec_hash` of the effective `model.yml`) or identity does not
  match without `--force` (test). Records move; jobs do not — no remote
  executor
- Environment manifest on every `train` record and every `evaluate` report
  and gate payload: git SHA, Python, torch, maatml, CUDA / cuDNN, GPU and
  driver, OS, determinism settings (test)
- `maatml report`: Markdown / CSV of runs, floors with derivation, slices,
  seed statistics and test spends, generated from the report schema alone
  (test: output regenerates from `output/eval/*.json` and `runs.jsonl` with
  no other input)

**Exit criteria:** a model folder with no `scripts/` directory derives its
floors, selects its operating point, pins a camera-disjoint benchmark and a
site-disjoint blind population, and produces a report, using only `model.yml`
and the CLI (CI test on a vision example); `gates derive` on a clustered
fixture writes a floor below row-level Wilson (test); floors carry a benchmark
version and `evaluate --gate --strict-population` exits non-zero when it
differs (test); `operating-point derive --write` followed by a second
`--confirm-on-test` on the same benchmark version prints a warning and lists
both spends (test); `evaluate --blind` on an unchanged fingerprint exits
non-zero without `--force` (test); `prepare` exits non-zero for a source with
`commercial-use: no` and no accepted-risk sign-off (test); `distill` with a
benchmark prompt in its pool exits non-zero before any teacher call (test); a
"never fires" fixture fails the smoke tier (test); `runs --adopt` of a packed
run reproduces `maatml runs` output on the receiving machine (test);
`maatml report` regenerates byte-identically from reports and `runs.jsonl`
(test).

## Later

- Precomputed top-k logit KD (teacher is training-only; eval/serve keep
  trusting validators); online KD CUDA-only and optional. Lands with
  `training.select_by` required and a logit cache that records the split
  fingerprint it was built on, refused by `train` from another split
- QAT (torchao) for the edge path: post-hoc GGUF/MLX/ONNX quantization loses
  accuracy that quantization-aware training recovers
- Signing (`verify --require-signature`, fail closed), OCI backend, promotion
  channels
- OpenAI-compatible `/v1/chat/completions`
- lm-eval-harness bridge: declare a public-benchmark smoke suite in
  `model.yml` next to task gates, for regression numbers alongside the
  validators
- Validator adapters: run Guardrails Hub or plain Pydantic validators inside
  the MaatML validator contract
- SetFit-style few-shot classifier path: ~8 labels/class at low compute (per
  published SetFit results), as a cheap entry point before a full
  `multi_head_classifier` fine-tune
- `maatml render-config`: effective config with all defaults and overrides
  materialized (pairs with the known-keys warnings from Truth and safety II)
- Quantized export (GGUF `--outtype` levels, ONNX int8); generic core ONNX
  exporter (promote out of the vision plugin)
- 1.0: frozen config / plugin / manifest APIs, migration guarantees,
  reproducible benchmarks
