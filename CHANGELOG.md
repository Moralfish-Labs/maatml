# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for the Python package and per-model versions under `examples/`.

## [Unreleased]

### Added

- **`maatml rollout` and the rollout environment protocol
  (`maatml.rollout_env/1`).** A checkpoint acts in an environment that lists
  the legal candidates, applies the chosen one, and verifies the outcome
  itself. The driver records every decision with its log-probabilities and,
  with `--label`, the valid set the environment's teacher accepts at each
  state the policy reached, which is what DAgger relabeling trains on.
  Environments plug in as the new `environment` plugin kind or as any program
  speaking JSON lines over stdio (`--env-command`). `line_world` ships as the
  example environment. See `docs/rollouts.md`.
- **Train-only rows in `prepare_rows`.** `train_rows` join train after the
  hash split and never reach val or the benchmark, so rows a policy collected
  from its own rollouts train the next round without moving the numbers
  earlier runs were measured on. A train-only row whose group is already in
  val or test is refused.

## [0.13.0] - 2026-08-28

### Fixed

- **`run` accepts every evaluation key `config` does.** `EvaluationSpec`
  forbids extra keys so a typo is caught before training spends the compute,
  but it knew five keys where `config._EVALUATION_KNOWN_KEYS` knows eleven. A
  folder declaring `slices`, `cache_predictions`, `gates_benchmark`,
  `operating_point` or `batch_size` passed `validate`, passed `evaluate`, and
  was refused by `run` with `evaluation: section is invalid: Extra inputs are
  not permitted` — every stage individually, nothing end to end, which left the
  evidence layer unreachable through the runner. The remaining known keys are
  now declared; the runner does not read them, it carries them so the section
  it validates is the section `config.py` accepts. The extra-forbid typo check
  is unchanged, and a test asserts `EvaluationSpec` covers
  `_EVALUATION_KNOWN_KEYS` so the two cannot drift apart again.

### Added

- **`maatml --version`.** Prints `maatml <version>` and exits, so a tool probe
  can read the version without loading a model folder. `maatml.__version__`
  comes from the installed package metadata and falls back to
  `0.0.0+unknown` in a source tree with no install.

## [0.12.0] - 2026-08-23

### Added

- **Batched evaluate.** A predictor may implement
  `predict_batch(rows) -> list[str]` (one output per row, in order);
  `evaluation.batch_size: N` or `maatml evaluate --batch-size N` then feeds
  the split through it in chunks of N with one device sync per chunk, so a
  GPU predictor stops idling between rows. Outputs, validation, slices, the
  prediction cache and every metric are per row as before. The report
  records `extras.batch_size`, and at N > 1 the per-row latency is the
  chunk's wall time divided by its rows, flagged `extras.latency_amortized`.
  `--batch-size` is a device knob, not an override: it combines with
  `--gate`. A predictor without `predict_batch` runs one row at a time with a
  warning; a chunk returning the wrong number of outputs is an error.
  `training.select_by` and every other caller of the harness inherit the
  file's `batch_size`.

## [0.11.2] - 2026-08-23

### Fixed

- `maatml validate` no longer warns that the key named by
  `evaluation.operating_point.threshold_key` is unrecognized: it is the key
  `operating-point derive --write` sets.

## [0.11.1] - 2026-08-23

Two gaps found while a real model folder adopted the evidence layer.

### Added

- **`dataset.attribution_field`**. The attribution table is keyed on each
  row's `source` by default; a corpus whose `source` is finer than its
  licence (one value per annotation file, `benchmark:holdout` for pinned
  rows) names the licence-level field instead, typically `dataset`. The
  corpus lock records the field; an empty or non-string value is refused.
- **`maatml evaluate --set K=V`**. Overrides `model.yml` for that one
  evaluate (repeatable, same syntax as `train --set`), recorded on the report
  as `extras.overrides`, refused together with `--gate` or `--blind`. This is
  how the val prediction cache for `operating-point derive` is written at a
  low decode cut without editing the file: the predictor receives the
  overridden `model_def`, and `extras.decode_threshold` records the cut the
  cache was decoded at.

## [0.11.0] - 2026-08-23

The Evidence layer (#34): everything between a finished run and an honest
floor, as lifecycle rather than per-folder scripts. Versioned eval reports
with slices, counts and a prediction cache; floors derived by `maatml gates
derive` with tiers and a population stamp; `maatml ship-check`; the operating
point derived on val with recorded test spends; named populations
(isolation, pins, benchmark version, a once-only blind evaluate, seed
studies); a sidecar licence table gating `prepare` and a corpus lock in the
export manifest; checkpoint selection on val, pathology signatures failing
the smoke tier, `distill` refusing held-out prompts; and runs that travel
(`runs --pack` / `--adopt`, an environment manifest on every record,
`maatml report` from the records alone).

### Added

- **`report_version`** on every eval report. `Report.read(strict=True)`
  refuses a report that predates the field or lacks a required one, so a
  derivation never reads a missing number as zero; lenient reads still load
  older reports as version 0.
- **`evaluation.slices`**. A list of row fields (or `{field, values}`); the
  harness reports `n`, `pass_rate` and the Wilson 95 % lower bound per value,
  `(absent)` for rows without the field, and `n: 0` with no rate for a
  declared value that has no rows. Rendered in the markdown summary.
- **`evaluate --cache` / `evaluation.cache_predictions`**. Writes
  `<run>.predictions.jsonl` beside the report: a `maatml.predictions/1`
  header (split, content sha256, checkpoint, report name) and one line per
  row with the row's metadata, the raw output, the validator verdict and the
  parsed object. `read_predictions` refuses a torn or foreign file. The
  report's `extras` now always carry `split_sha256` and, when cached,
  `predictions_cache`.
- `maatml.evaluation.stats`: `wilson_interval` / `wilson_lower`, exact at the
  0 and n boundaries, raising on `n <= 0`; `cluster_bootstrap_lower` and
  `floor2`.
- **`maatml gates derive`**. Floors from one or more runs' reports: Wilson
  95 % lower bound at each metric's own denominator, the per-metric minimum
  across runs, `--min-n` refusal for thin denominators, and a group-cluster
  bootstrap (`--cluster-by`, `--min-groups`) when the run carries a
  predictions cache. `--write` rewrites `<section>.gates` in `model.yml`
  textually, keeping comments and tiers, and stamps
  `evaluation.gates_benchmark` with the split hash.
- **Report `counts`**: `{metric: {k, n}}` behind every harness rate and
  slice; a metrics plugin adds its own through a `__counts__` entry, which is
  lifted out of `metrics`.
- **Gate tiers**: a gate value may be `{min, tier}`; `advisory` misses are
  recorded under `gates.advisory_failed` and never fail evaluate, the
  lifecycle runner, or `compile --require-gated`.
- **Slice gates**: `"slice:<field>=<value>"` gates that slice's pass rate; an
  empty slice reads `None` and fails.
- **`maatml ship-check CANDIDATE BASELINE`**. The release decision in one
  verdict: absolute (every blocking floor met at production tier), delta
  (no gated metric drops more than one row at n >= 30, or `--max-regression`),
  and population (same split, else `--replay` re-evaluates both checkpoints
  over the current test split under `output/eval/replay/` without touching
  either run's own evidence). Exit 1 on DO NOT SHIP; `--json` for the
  verdict. `evaluate_model` gained `out_dir` and `record_gates` for this.
- **`maatml operating-point derive`** and `evaluation.operating_point`
  (`threshold_key`, `objective`, `budget: {metric, max}`, `sources`, `grid`).
  Sweeps the predictor's new `rescore(rows, threshold)` over a val prediction
  cache, skips cuts below the one the cache was decoded at, picks the best
  objective under the budget, writes `<run>.<split>.operating_point.json`,
  and with `--write` sets `evaluation.<threshold_key>` in `model.yml` with
  the sweep as provenance. `--confirm-on-test` evaluates once on test at the
  written cut and records a **test spend** on the run (`RunRecord.test_spends`;
  listed by `maatml runs`; a repeat on the same benchmark warns). Deriving on
  test is refused.
- **Populations** (`dataset.isolation`, `dataset.pins`, `dataset.blind_samples`).
  `isolation: {fields, policy}` declares the row hierarchy (fine → coarse) and
  the level each of val / benchmark / blind must be disjoint from training at;
  `pins: {val: [field:value], benchmark: [...]}` move whole groups after the
  hash split, and a pinned population is exactly its pins. `prepare` refuses
  to write splits that violate the policy, refuses a pin matching nothing,
  checks the blind manifest for leakage without writing it, and `audit`
  re-checks the prepared splits.
- **Benchmark version**: `output/prepared/benchmark.json` records an
  order-insensitive hash of the test rows plus the pins; reports carry it as
  `extras.benchmark_version`. An in-place edit of `benchmark_samples` is
  refused — version the file instead.
- **`evaluate --blind`**: spends `dataset.blind_samples` once on a candidate
  whose production gate pass is current (`RunRecord.gated_fingerprint`:
  evaluation config + weights, recorded on every non-smoke test gate pass);
  gates enforced; `<run>.blind.json`; the spend is recorded on the run
  (`blind_spends`, listed by `maatml runs`) and a repeat needs `--force`.
- **`maatml train --seeds N`**: one recipe, N seeds (from `--seed`, default
  0), each its own run; `output/seeds/<first-run>-xN.json` records the runs
  and mean / sd / min / max per metric every run reported. `gates derive
  --seed-study FILE` takes those runs (the per-metric minimum) alongside
  `--run`.
- Reports for non-test splits are named `<run>.<split>.json` (and their
  caches likewise), so a val evaluate no longer overwrites the test report.
  `extras.decode_threshold` records the cut the predictor decoded at.
- **Population stamp**: with `--gate`, the gates payload records the split's
  `benchmark_sha256`; when `evaluation.gates_benchmark` names another split
  evaluate warns (`population_mismatch`), and `--strict-population` refuses.
- **Sources carry their licence** (`dataset.attribution`). A sidecar Markdown
  table with one row per `source` (`source`, `licence`, `commercial-use`,
  `sign-off` required; `provenance` / `consent` and `attribution` read when
  present; long headers and extra columns accepted). `prepare` refuses a row
  without a `source` or without a table row, a blocked or unsigned sign-off,
  and a `no` / `unknown` commercial-use whose sign-off does not record an
  `accepted-risk — <name> <date>`; the acceptance is recorded in the lock
  rather than passed as a flag.
- **Corpus lock**: every prepare writes `output/prepared/corpus.lock.json` —
  input files by sha256 and row count, the attribution rows used, the risk
  accepted, and a `lock_sha256` over all of it. `export` copies it into
  `manifest.json` as `corpus_lock`, and a lock that names sources also ships
  as the bundle's `ATTRIBUTION.md` (listed in the manifest). The auto
  `MODEL_CARD.md` from the Slim tranche will embed the same block.
- **`training.select_by`**: after training, every saved `checkpoint-<step>`
  and the final weights are evaluated on val with the evaluate harness; the
  best by the named metric is recorded on the run (`selection`,
  `selected_checkpoint`) and the run id resolves to it for evaluate, export
  and serve, with evidence still recorded against the run
  (`run_id_for_checkpoint`). Per-candidate reports under
  `output/eval/select/<run>/`; a metric the harness does not report is an
  error. `training.keep_checkpoints` sets `save_total_limit` (default 2) for
  every trainer. `evaluate_model` gained `label`.
- **Pathologies**: every report carries `pathologies[]` — `never_fires`
  (no output, or recall ≤ 0.01 with precision ≥ 0.9 / absent),
  `identical_output`, `one_class` — plus a metrics plugin's own
  `__pathologies__`. The gates payload lists them; at the smoke tier each is a
  failing blocking gate (`pathology:<name>`), so a never-firing fixture fails
  `run --smoke` whatever the floor. Rendered in the markdown summary.
- **`distill` refuses held-out prompts**: a pool prompt found in
  `benchmark_samples`, `blind_samples` or the prepared val / test split, or a
  pool row carrying a pinned group, stops the run before any teacher call. The
  teacher cache now starts with a `maatml.teacher_cache/1` provenance header
  (teacher, prompt pools by sha256, benchmark version), repeated on the
  distill card; a cache that recorded nothing still leaves no file.
- **Runs travel**: `maatml runs <dir> --pack RUN` writes
  `output/bundles/<run>.maatml-run.tar.gz` — the run directory (without
  `checkpoint-*` unless `--with-checkpoints`), the run's eval reports, caches
  and selection reports, the registry record, the environment, and the
  model-folder fingerprint (`spec_hash`), every file hashed. `--adopt BUNDLE`
  unpacks it into the receiving folder, verifies every file, refuses another
  identity or `model.yml` fingerprint and an existing run without `--force`,
  and appends the record with its paths rewritten (`RunRecord.adopted_from`).
  Records move; jobs do not.
- **Environment manifest** (`maatml.environment/1`: git SHA, Python, OS,
  maatml / torch / transformers / peft / safetensors, CUDA, cuDNN, GPUs and
  driver, MPS, determinism settings) on every train record
  (`RunRecord.environment`) and every evaluate (`extras.environment`, and
  `gates.environment` when gated). Every train record now also carries
  `spec_hash` (`spec_fingerprint`).
- **`maatml report`**: Markdown (default, `output/REPORT.md`) or CSV of
  runs, every eval report's metrics with their derivation (k / n and Wilson
  95 % from `counts`) beside the floor that judged them, slice and pathology
  gates, slices, seed statistics and spends — generated from `runs.jsonl`,
  `output/eval/*.json` and `output/seeds/*.json` alone, so it regenerates
  byte-identically and never reads `model.yml`.

## [0.10.0] - 2026-08-17

Gated device compiles, a shared capture path for custom servers, video-frame
ingest, an Anthropic Messages serve wire, and `diffusion_lora` through kohya
sd-scripts (#33).

### Added

- **`maatml compile --require-gated`**. Refuses before the compiler plugin
  runs when `manifest.gate_evidence` is missing, `passed` is not true, or
  `smoke_gated` is true. `target_manifest.json` now records
  `promotion_eligible` and `promotion_reason` on every compile so a serving
  host can refuse an ungated device artifact without reading SIP-specific
  sidecars.
- **`open_capture` / `LifecycleServer.record_capture`**. Custom servers
  (DeepStream/UDS, …) can append the same `source: serve_capture` /
  `approved: false` rows the HTTP backend writes. `--capture` still requires
  `--auth-token`.
- **`maatml ingest --video`**. A sidecar JSONL (`frame` / `timestamp_ms` /
  `t` plus gold) plus a video file becomes validator-gated image rows;
  ffmpeg extracts PNGs under `datasets/samples/images/`. Annotation dialects
  stay in the sidecar.
- **`maatml serve --server anthropic`**. A translating proxy that speaks
  Anthropic's Messages API in front of an OpenAI-compatible upstream
  (llama.cpp). Options cover `upstream`, `model`, `timeout`, `tool_style`
  (`native` / `inline`), and `call_retries`. `--enforce` / `--max-retries`
  gate replies with the model folder's validator the same way they gate
  `http`.
- **`architecture: diffusion_lora`**. Image LoRA training driven through a
  kohya sd-scripts checkout. `format: image_caption_folder` prepares a
  flat image+caption folder into the usual splits; the trainer materializes
  kohya's repeats layout, records the run, and stops the child on SIGTERM
  so an abort does not leave an orphan holding the GPU.
- **`distill.target_format: text`**. Hands the teacher's reply to the
  validator as it stands instead of requiring JSON. A string gold target is
  the assistant text itself — `render_assistant_target` no longer wraps it
  in quotes.
- **Eval progress**. The shared eval loop prints `eval N/total` with a
  remaining-time estimate while it generates, so a long split is not a
  silent hang.
- **Agent skill docs** under `docs/skill/` (model.yml, plugins, flywheel).

## [0.9.1] - 2026-08-10

Pluggable compile/serve runtimes, actionable `vision_scene` validator feedback,
`maatml audit` (renamed from `doctor`), and a slimmer example set.

### Added

- **`ValidationError.hint`** and actionable `vision_scene` errors (#6). Every
  validator failure can carry a short "how to fix it" string; eval
  `sample_failures`, the eval markdown summary, serve 422 payloads / retry
  feedback, and datagen reject rows (`_validation_errors`) all surface it.
  `vision_scene` now names what failed, where (`location`), and the allowed
  values; it reports every bad detection (capped) instead of stopping at the
  first. The vision example README documents the four layers, error codes, and
  how they map to `evaluation.gates`.
- **`maatml audit` pre-flight checks** (renamed from `doctor`; see Changed):
  configured `evaluation.metrics` / `evaluation.predictor` /
  `dataset.generator` must resolve in the registries; seed / benchmark corpus
  row counts; gate keys compared against the latest `output/eval/*.json`; a
  quarantined `runs.jsonl.corrupt`; QLoRA configured on a non-CUDA auto device;
  optional Hub cache probe for `base_model`.
- **Compiler and server registries.** `@register_compiler` / `@register_server`
  plus `maatml compile <export-dir> --target NAME --out DIR [--option K=V]` and
  `maatml serve … --server NAME [--server-option K=V]`. The built-in `http`
  server remains the default; plugins own TensorRT, vLLM, llama.cpp, and other
  native backends without forcing ONNX or an in-process predictor. Core writes
  `target_manifest.json` and provides a shared lifecycle envelope
  (verify → warmup → serve → close with SIGINT/SIGTERM drain).
- **`maatml manifest amend <export-dir> <files...> [--format NAME]`**: adds
  out-of-band artifacts to an existing manifest with sha256 entries, so
  `maatml verify` covers them and a serving catalog's checksum traces back to
  a gate-evidenced export. Gate evidence and every other field are untouched.
- **GGUF quantization in the exporter.** `extensions.gguf.quantize_binary`
  (or `MAATML_LLAMA_QUANTIZE`) names the llama.cpp `llama-quantize`
  executable and `extensions.gguf.quant_levels` the levels (default
  `Q4_K_M`); `export --format gguf` then emits `<name>-<level>.gguf` beside
  the f16, all in the manifest's file list. Explicit-only like the convert
  script: no PATH search.
- **`maatml evaluate --baseline X --max-regression 0.03`**: fails the run
  when a gated metric drops more than the allowance against the baseline
  report. Repeatable; `metric=0.05` overrides one metric and may name an
  ungated metric. The default allowance judges gated metrics only, since
  ungated keys like `eval_loss` improve downward.

### Changed

- **`maatml doctor` → `maatml audit`** (breaking, no alias). Same exit-code
  contract (errors → 1, warnings → 0) and `--json` shape; call sites and docs
  updated.

### Removed

- The JCL Validator and Spool Interpreter example model folders and their
  task-specific plugins, datasets, tests, and documentation.

## [0.9.0] - 2026-08-04

Serving-bundle export paths for MLX and ONNX, a distill/teacher path that fails
loudly instead of quietly, benchmark leakage caught at the content level, and a
lint/type gate that actually runs.

### Added

- **MLX export for seq2seq.** `mlx_lm.convert` only handles decoder-only
  models, so a seq2seq checkpoint is assembled directly into `<name>.mlx/`:
  the safetensors bundle plus `spiece.model` and a `serving.json` declaring the
  `maatml.serving/1` contract (`kind`, token budgets, decoder start and EOS ids,
  input prefix, and the brace-repair note when `evaluation.repair_braces` is
  set). `tie_word_embeddings` is set false when the checkpoint carries its own
  `lm_head.weight`, so a downstream loader does not project logits through the
  embedding matrix the fine-tune untied.
- **ONNX serving-bundle exporter for the JCL multi-head classifier**
  (`jcl_plugin/export_onnx.py`): `model.onnx` with external weights, the
  tokenizer, and a `serving.json`, with the pooled and per-token heads baked
  into the graph, so the model can be served without torch.
- **`distill.request_params`**: merged into the chat-completions payload. A
  reasoning teacher spends the default 1024-token budget on hidden reasoning
  before any content arrives, and switches like `chat_template_kwargs` had no
  other way in. A `timeout` in that block is routed to the client rather than
  the payload. `TeacherClient.generate_row` takes the same parameter.
- **`temperature=None` omits the field** from the chat-completions payload
  instead of sending the default: some endpoints reject the parameter itself,
  not just particular values.
- **`serve --allow-unauthenticated`**: the explicit opt-in for a non-loopback
  bind with no token.
- **`reject_unknown_training_keys`**: the dataclass-backed trainer configs
  (built with `d.get(...)`) now fail on a `training:` key the architecture does
  not read, matching the `extra="forbid"` the pydantic ones already had.
- **`maatml.training.schedule`**: one derivation of optimizer step counts and
  mixed-precision flags, shared by causal SFT, seq2seq, multi-head, and
  preference training instead of four near-copies.
- **`realign_special_token_ids`** and one shared special-token resolver used by
  both the multi-head trainer and the predictor, so the two cannot drift.
- **CI:** a gitleaks secret scan (with the rule set enabled rather than
  declared), an example-invariant check the test job runs, an explicit ruff
  rule set plus `ruff format --check`, and a real `[tool.mypy]` section.
- **Docs:** the gate tables in every example README are generated from
  `model.yml` and checked in CI, so a gate change cannot leave the README
  stating the old threshold.
- **`MANIFEST.in`**: the sdist file list is declared rather than left to
  setuptools' defaults. It carries the license, README, changelog, and the
  security, contributing, and conduct docs, and prunes `tests/`, which cannot
  pass without the ~23 MB `examples/` tree that the sdist does not ship.

### Changed

- **Behavior change: the teacher base URL must be set explicitly.**
  `TeacherClient` no longer defaults to `https://api.openai.com/v1`. Set
  `MAATML_TEACHER_BASE_URL` or pass `base_url`; the scheme must be `http` or
  `https`. `datagen` and `distill` send the prompt pool to whatever this points
  at, and for the shipped domains the prompt pool is the sensitive asset, so
  the destination is always a stated choice.
- **Behavior change: `serve` refuses an empty auth token** (the usual cause is
  an unset variable expanding to `""` in a unit file), and an empty bind host
  is no longer treated as loopback: `--host ""` binds every interface, so it
  needs a token or `--allow-unauthenticated`.
- **Behavior change: `maatml distill` exits non-zero** when a non-empty prompt
  pool produced no accepted and no duplicate rows, and aborts after five
  consecutive teacher failures rather than walking the whole pool. A scheduled
  flywheel no longer reports success against an unreachable teacher.
- **Behavior change: a declared but missing eval asset is an error.** An asset
  named explicitly or declared in `model.yml` raises `DeclaredAssetMissing`
  instead of resolving to `None`; only genuinely optional assets may be absent.
  A typo in `dataset.contracts` used to surface as a `TypeError` from the
  validator on the first evaluated row.
- **Behavior change: `training.lora` rejects unknown keys.** A typo was dropped
  in silence, and since `training_config` is hashed into the lifecycle
  fingerprint the run still looked fresh.
- **Behavior change: the train fingerprint includes `--limit` and `--seed`.**
  Both change the checkpoint, and leaving them out let the next plain
  `maatml run` report "all fresh" over a truncated run, with evaluate, export,
  and verify inheriting the skip.
- The lifecycle environment fingerprint uses maatml's own checkout SHA and is
  `None` for an installed package, so an unrelated repository's commits no
  longer invalidate every step.
- `export --format mlx` accepts seq2seq architectures (served as a direct-load
  bundle); `gguf` stays gated to causal and preference architectures.
- **Dependency floors raised past known advisories:** `torch>=2.6`
  (GHSA-53q9-r3pm-6pq6 is an RCE in `torch.load` despite `weights_only=True`,
  which is the call the SFT tokenized cache makes), `transformers>=5.5`,
  `pillow>=12.3`, `onnx>=1.22`, `sentencepiece>=0.2.1`. The unused `evaluate`
  dependency is dropped.
- **Example corpora regenerated and gates re-derived from measured runs.** The
  JCL corpus is real decks judged by the MVS 3.8j converter, with jobnames
  decorrelated from the label (earlier corpora leaked it through the name), and
  its gates are the Wilson 95% lower bound of the measured rate with the
  structurally guaranteed layers at 1.0. Triage gates are set the same way, and
  its splits, batch size, epochs, and warmup were retuned.
- The spool `hostname_fqdn` sanitizer rule is anchored on a lowercase TLD. The
  looser pattern also matched the shape of a z/OS dataset name, so it rewrote
  every dataset name to `HOST.REDACTED` while the labels still named them. The
  trade-off is that an all-uppercase FQDN is no longer redacted.
- The sanitizer's length-preserving warning states what actually happens: the
  original value is fully replaced and only the marker is clipped.
- `ruff format` applied across the tree.

### Fixed

- **`ingest` no longer destroys the seed corpus.** A run that accepts nothing
  (an empty input, or a validator that rejected everything) leaves an existing
  corpus untouched and reports `protected_existing`, including under
  `append=False`. The seed file is written atomically.
- **`serve --capture` identifies rows by the (request, output) pair.** Hashing
  the output alone collided across distinct requests, and `ingest` then dropped
  the later ones as duplicates: exactly the reviewed rows the flywheel exists
  to keep. Captured requests now pass through the model's declared
  `dataset.sanitize` tags before being written.
- **Content-level benchmark leakage is caught.** A benchmark drawn from the
  same generator as the seeds was invisible to the group check once it carried
  its own family namespace; the rendered request (or the row minus its identity
  fields) is now compared against the seed corpus. The triage benchmark is
  drawn disjoint from the seeds.
- **Valid JSON that is not an object** (a bare list, string, or number) is
  scored as a failed row instead of raising `AttributeError` and aborting the
  whole evaluate or distill run before the report is written.
- **A validator that raises during distill** is counted (`validator_errors`),
  reported, and rejects the row, rather than ending the run.
- **Model plugin modules are keyed on the resolved folder path**, so two model
  folders sharing a directory name no longer collide in `sys.modules`.
- **A plugin name claimed by two different sources warns**, naming both, so the
  registration that silently loses is visible. Re-binding the same source (a
  registry wipe, `discover_plugins(force=True)`) stays quiet.
- `scripts/evaluate_all.py` and the CLI share one eval-key resolver instead of
  two copies that had drifted.
- The `distill` teacher cache flushes every 25 rows, so a crash in a long
  local-teacher run does not lose hours of paid-for responses.
- Docs: the validator example, the stale quickstart numbers, and the statement
  that `serve` validates on `--enforce` or `?validate=1` rather than by
  default. `tests/test_docs_truth.py` checks these against the code.

### Security

- Dependency floors raised past known advisories (see Changed).
- Gitleaks runs in CI and pre-commit with its rule set enabled.
- The teacher client has no implicit third-party endpoint (see Changed).

## [0.8.0] - 2026-07-24

The fixed lifecycle runner, the reviewed data flywheel and serve contract, plus
the hygiene backlog that had been carried outside the tranches.

### Added

- **`maatml distill`: validator-gated teacher labels over a prompt pool.**
  Asks a teacher only for the label on prompts you already have; every response
  is gated by `evaluation.validator` before it enters the corpus, so a wrong
  label is dropped. Accepted rows carry provenance (teacher model / revision,
  prompt hash, source, family), rejections are kept, and teacher responses are
  cached so `--replay` reproduces the accepted corpus with no network. Typed
  `distill:` config; worked example on triage.
- **`maatml mint`: preference pairs for DPO / ORPO.** Splits each prompt's
  candidate completions into pass / fail with the registered validator and
  emits `{prompt, chosen, rejected}` pairs. An explicit source op, never a
  default `run` step.
- **`serve --auth-token`** (or `MAATML_SERVE_TOKEN`): bearer token required on
  `/predict`, compared in constant time, mandatory for `--capture`, and warned
  about when a non-loopback bind has none.
- **`serve --capture PATH`**: append served predictions for review. Captured
  rows are not gold (`approved: false`, row/byte capped); `maatml ingest`
  refuses a `serve_capture` row until a reviewer sets `approved: true`. The
  loop is `serve --capture` → review → `ingest` → `run`.
- **`serve --max-retries N`**: on a validation failure under `--enforce`, feed
  the error back and re-ask up to N times. Responses report `attempts` /
  `retries`; a request still failing returns 422 with the retry count.
- **The data-flywheel docs page** (mkdocs) covering datagen, distill, ingest,
  mint, and reviewed capture as gated source operations.
- **`maatml run`: the fixed lifecycle in one command.** Walks prepare, train,
  evaluate (gates enforced), export, verify in order and stops non-zero at the
  first failure, so one green line means every stage passed. Flags: `--smoke`,
  `--force`, `--from`, `--until`, `--dry-run`, `--device`, `--seed`, `--limit`,
  `--format`, `--set`. `datagen` / `ingest` stay outside the runner: they
  change the seed corpus, which is what makes `prepare` stale.
- **Step fingerprints in `output/pipeline.json`** (written atomically). Each
  step records the effective config after `--smoke` / `--set`, its declared
  input files, the upstream fingerprint, the maatml version and git SHA, the
  plugin sources, the device profile, and the exporter. A step is skipped only
  when the fingerprint matches, it completed last time, and its outputs are
  still present, so this is idempotence rather than a cache.
- **Smoke-tier gates:** a `smoke:` block may declare its own `gates:`, which
  `--smoke` runs enforce instead of the production thresholds. The pass is
  recorded as smoke-gated in the run record (`smoke_gated`) and in the export
  manifest (`gate_evidence`), so a rehearsal never reads as a real gate pass.
- **`output_nonempty_rate`** is reported alongside a model's own metrics: it
  says the checkpoint saved, reloaded, and produced output, which is what a
  smoke tier can gate on honestly. A metrics plugin claiming that key is an
  error.
- **CI:** the ml job runs `maatml run --smoke` end to end on triage, re-runs it
  to prove nothing further executes, and asserts the manifest records the smoke
  tier.
- **`maatml doctor`:** read-only diagnostics for "why did that not work here?".
  Reports the installed optional stacks, the device the CLI would pick and its
  profile, the registry contents plus anything that failed to load, and (given
  a model folder) its declared paths, splits, runs, validator, and gates.
  `--json` for machines; exits 1 when something is broken rather than absent.
- **`maatml runs --compare`:** run-by-metric table across recorded runs, with
  `--metric` to select keys, `--limit` for the most recent N, and
  `--all-metrics` to include trainer timing. A metric a run never reported
  shows as `-`, never as `0`, and hidden timing metrics are named.
- **`maatml scaffold --plugin`:** loads a plugin before resolving
  `--architecture` and records it in the new `model.yml`, so plugin-owned
  architectures (`vision_multitask`, `vlm_sft`, third-party) can be scaffolded
  at all. Scaffold hooks may contribute `model_yml` sections, `seed_rows`, and
  extra `files`; the vision and vision-vlm plugins now ship theirs.
- **`py.typed`** in the wheel (PEP 561), so a downstream mypy reads maatml's
  annotations instead of treating the package as untyped.
- **CI:** a torch-free `windows-latest` job covering path, encoding, and
  console assumptions that only fail off POSIX.

### Changed

- **Behavior change:** `maatml plan` prints per-step fresh/stale status with
  the reason a step is stale (which fingerprint component changed), instead of
  a static list of commands. It is now an alias for `run --dry-run`.
- The `evaluation:` section is typed where the runner depends on it, so an
  unknown key, a non-numeric gate, a misspelled validator, or an unregistered
  metrics plugin fails before any step runs rather than after training.
- `maatml evaluate` and the runner share one implementation, so both enforce
  gates, resolve the token budget, and record results on the run identically.
- Scaffolded `model.yml` omits `base_model` when the architecture has no base
  model id, instead of leaving a `CHANGE_ME` placeholder in the file.
- `pyproject.toml` is the only dependency manifest: the `requirements*.txt`
  files are removed (`requirements-base.txt` had already lost `jsonschema`),
  and `jsonschema` stays a core dependency with the reason recorded, it is the
  contract every shipped validator is written against.
- Quickstart says what step 4 actually is: a rehearsal on eight seed rows with
  `max_steps: 4`, where `--gate` is expected to fail until the corpus and
  training budget are real.
- `docs/index.md` installs the vision extra as `maatml[ml,vision]`; the vision
  examples need the training stack too.
- The orphaned root `maatml.md` is gone; README, AGENTS.md, and the docs site
  carry that content.

## [0.7.0] - 2026-07-24

Silent-failure hardening and a test floor: the paths that reported success
while doing nothing, plus tests for the four trainer architectures and the CLI,
which had none.

### Added

- **Global `--debug`:** `maatml --debug <command>` prints full tracebacks for
  user errors (missing file, unparseable `model.yml`, unknown plugin). Without
  it those errors print a single actionable line.
- **`evaluation.repair_braces`** (default `false`): opt-in re-adding of the
  outer `{` / `}` that flan-t5 tokenizers drop. Every repair is counted in
  `Report.extras.brace_repairs` and warned about, so a pass rate stays a
  measurement of the model rather than of maatml's repair.
- **Multi-metric evaluation:** `evaluation.metrics` as a list runs every entry
  and merges the results (two plugins reporting the same metric key is an
  error). It previously ran only the first entry.
- **Report extras:** `max_input_tokens`, `truncated_inputs`, and brace-repair
  counts are recorded in the eval report.
- **Public registry reset API:** `reset_registries`, `snapshot_registries`,
  `restore_registries`, and `Registry.unregister`, so tests and embedding hosts
  stop reaching into registry internals.
- **Plugin failure surfacing:** `maatml plugins` lists sources that failed to
  load, and `Unknown … plugin` errors name them.
- **Tests:** a `typer.testing.CliRunner` suite (gate exit codes, `verify` on a
  tampered manifest, `scaffold` refusal, argument parsing) and torch-gated unit
  tests for all four trainer architectures.
- **CI:** a torch-free `macos-latest` job; the ml job runs `pytest tests/
  examples/` so the torch-gated tests and the vision end-to-end test execute;
  the matrix runs with `-rs` so skips are visible.

### Changed

- **Behavior change:** `maatml evaluate` defaults its token budget to
  `packaging.max_input_tokens` (previously a fixed 2048), matching serve and
  `export --parity`. Pass `--max-input-tokens` to override. Inputs the budget
  truncated are counted and reported.
- **Behavior change:** per-class eval numbers are `pass_rate` / `passed` / `n`.
  The old shape reported `precision` alongside a literal `recall: 1.0` and
  `f1: 0.0` for every category: two of the four numbers were invented.
- **Behavior change:** seq2seq brace repair is off unless
  `evaluation.repair_braces: true` (both seq2seq examples set it).
- **Behavior change:** `training.precision` is validated where it is parsed;
  an unsupported value (`bfloat16`, a typo) fails instead of silently training
  in fp32.
- **Behavior change:** an absent or malformed `training.heads` is an error. The
  legacy JCL head shape (`heads.error_codes` / `heads.severities` /
  `head_loss_weights`) only applies when those keys are present.
- **Behavior change:** `maatml prepare` refuses a `benchmark_samples` file whose
  rows share a group key with the training splits (a benchmark is pinned to
  test, so the overlap inflates the number it exists to protect).
- **Behavior change:** `maatml sweep` records a failed trial and continues,
  exiting non-zero at the end, instead of aborting and discarding the trials
  that already trained. Ranking compares only trials reporting the chosen
  metric, and the direction comes from the metric name.
- `maatml datagen` appending to an existing corpus skips rows already present
  (by content / `sample_id`) and reports `duplicates`; a stale rejection report
  from a previous run is removed rather than left in place.
- `maatml evaluate` resolves validator, metrics, and gates before touching a
  checkpoint, so a config error is reported instead of "no checkpoints found".
- `discover_plugins` no longer wipes the registries when the trainer registry
  looks empty (that heuristic dropped whatever a model folder had registered),
  and model-folder plugins execute once per process instead of once per caller.
- The `Validator` protocol describes the call shape the harness, serve,
  datagen, and ingest actually use (a callable with optional keywords).
- The vision and vision-vlm seed builders generate held-out benchmark rows in a
  `bench_*` family namespace; they previously copied the first N seed rows, so
  the pinned benchmark was a subset of train. Committed corpora regenerated.

### Fixed

- **Splits:** a corpus from `datagen` / `ingest` (one source, no families)
  hashed into a single group, so every row landed in one split and val/test
  came out empty while prepare reported success. A group covering ~the whole
  corpus is now split per row with a loud warning, teacher and ingest rows
  carry a per-row family, and an empty split is called out.
- **seq2seq targets:** rows with a missing or empty target serialised to the
  literal string `"{}"` and were trained on. They are dropped and counted, and
  an all-empty corpus fails.
- **Gold labels:** `multi_head` mapped any unrecognised label to index 0 /
  `none`. Labels are scanned before training and unmappable values fail with
  the offending values counted. Boolean gold maps through the declared labels,
  so `labels: [valid, invalid]` no longer means `True` → `invalid`.
- **Fractional epochs:** `seq2seq` and `multi_head` floored `epochs` to an int,
  so `epochs: 0.5` trained for zero epochs and reported success.
- **Preference rows:** structured `chosen` / `rejected` values serialise as JSON
  instead of a Python `repr`; identical pairs warn; DPO / ORPO honour
  `training.lora.save_mode` through the same saver as causal SFT.
- **Degenerate rows:** the alpaca / sharegpt adapters emitted rows with no user
  or assistant content (a mis-mapped field name produced a corpus of empty
  strings); they are dropped and counted, and an all-degenerate corpus fails.
- **Teacher datagen:** request failures were swallowed into "no row", so a dead
  endpoint burned the attempts cap and reported "0 accepted" with no reason.
  Failures are counted, the first error is reported on the datagen card, and
  five consecutive failures abort the run.
- **Run records:** trainers mark a run `aborted` for fallible work that used to
  sit outside the finish handler (tokenizer load, reading the splits), so a
  crashed run no longer stays `running` forever.
- **Resume:** `--resume auto` skips `running` records with no checkpoint, which
  used to hide an older run that could actually be resumed.
- **Tokenize cache:** the SFT cache key includes `val.jsonl` content, so a new
  val split is not evaluated against a stale cache.
- **Sanitizer:** length-preserving truncation warns once per rule instead of
  silently emitting a cut-short redaction, and a fixed replacement that cannot
  fit its pattern's shortest match is rejected when the rules load.

### Security

- `pypa/gh-action-pypi-publish` is pinned to a commit SHA (that job holds an
  OIDC token); Dependabot keeps the pin current.

## [0.6.0] - 2026-07-23

Truth and safety II: make the "the same validator gates your data, evaluation,
and live inference" claim true, stop data-destroying commands, add a serve and
plugin security floor, and config honesty.

### Added

- **Serve enforcement:** `maatml serve --enforce` returns HTTP 422 when the
  configured validator rejects a prediction (gates live inference).
  `/predict?validate=1` stays a non-blocking annotation.
- **Ungated datagen escape hatch:** `maatml datagen --allow-ungated` runs
  without a validator and marks the run and a new `*.datagen_card.md` as
  UNGATED; the summary line reports GATED / UNGATED.
- **Untrusted-folder linting:** `maatml validate --no-plugins` checks schema and
  paths without importing model-folder plugin code.
- **Serve debug:** `maatml serve --debug` includes the exception and traceback
  in 500 responses (off by default).
- **Trust boundary** documented in README, SECURITY.md, and docs/plugins.md: a
  model folder is executable code and every command that reads `model.yml`
  (including `validate` and `plan`) runs its plugins.

### Changed

- **Behavior change:** `maatml datagen` now **fails** when no
  `evaluation.validator` is configured, instead of silently accepting every
  generated row. Pass `--allow-ungated` to keep the old accept-all behavior.
- **Behavior change:** `maatml scaffold` **refuses** to overwrite an existing
  `model.yml` or seed corpus; pass `--force` to regenerate.
- **Behavior change:** `--set` overrides are now validated (semver, `gt=0`,
  types); an invalid override exits non-zero instead of being applied silently.
- `maatml ingest` counts rows missing the gold field as `skipped_unvalidated`
  (instead of accepting them unvalidated) and errors when a `--map` source
  column matches zero input rows.
- `maatml evaluate` prints a notice when no validator is configured; a
  configured-but-unresolvable `evaluation.validator` now errors instead of
  silently degrading to JSON-parse-only scoring.
- Declaring `dataset.sanitize` with the alpaca / sharegpt / preference formats
  now errors (those paths cannot sanitize) rather than the dataset card falsely
  claiming a sanitizer ran; the card reports only tags actually applied.
- `maatml validate` warns on unrecognized `dataset:` / `evaluation:` keys.

### Fixed

- **Resume:** `maatml train --resume auto|<run_id>` now resolves to the newest
  `checkpoint-*` directory (previously it passed the run root, which current
  transformers rejects).
- **Run registry:** a torn or unparseable line in `runs.jsonl` is skipped with a
  warning and quarantined to `runs.jsonl.corrupt` instead of failing every
  command that reads the registry; records are written in a single append.
- **Seed safety:** `maatml datagen` writes seed files atomically and never
  truncates a non-empty seed file when nothing was accepted.

### Security

- **Serve:** 500 responses no longer leak the exception message or traceback to
  the client (server-side log only; opt in with `--debug`); a warning is printed
  when binding a non-loopback host.
- **Tokenized cache** loads with `torch.load(weights_only=True)`, closing a
  pickle code-execution sink under `output/cache/`.
- **GGUF export** resolves the convert script only from `MAATML_LLAMA_CONVERT`
  or `extensions.gguf.convert_script`; it no longer searches `PATH` or the cwd
  for a generic `convert.py`.
- **Vision predictors** confine request-supplied image paths to the model
  directory, rejecting absolute paths, `..` segments, and symlink escapes
  (closes a serve-time arbitrary-file-read).

## [0.5.1] - 2026-07-23

### Added

- **Triage contract:** `examples/support-ticket-triage` ships a real validator
  (`triage_plugin`) with a JSON → schema → **routing contract** (`category →
  team`) → summary-quality pipeline, plus enforced `evaluation.gates`, a fixed
  benchmark (`test_prompt_set.jsonl`), and CPU-free tests. It was the only
  reference example without a validator.
- **Gates everywhere:** `jcl-validator` and `spool-interpreter` promote their
  README target tables into enforced `evaluation.gates`, so every bundled
  example now gates.
- **Serve hardening:** opt-in CORS via `--cors` / `MAATML_SERVE_CORS` and a
  request-body size cap via `--max-body-bytes` (default 1 MiB → `413`).
- **Honest manifests:** export `manifest.json` records `weights_dtype` read from
  the exported safetensors tensors (`weights_dtype_verified: true`) alongside the
  declared `weights_dtype_declared` hint; mixed-precision exports list every
  observed dtype.
- **CI:** Python 3.13 in the test matrix; a CPU `ml-smoke` job that runs
  `prepare → train --smoke → evaluate` on triage through the real `[ml]` stack.

### Changed

- **Behavior change:** `maatml evaluate --gate` now **fails (exit non-zero)**
  when a model declares no `evaluation.gates`, instead of passing vacuously.
  Scripts relying on the old exit-0 must add a `gates:` block or drop `--gate`.
- **Security:** `maatml serve` no longer sends a wildcard
  `Access-Control-Allow-Origin: *` by default, cross-origin access is now
  opt-in.

## [0.5.0] - 2026-07-22

### Added

- **Serving:** `maatml serve` runs a dependency-light HTTP inference API
  (`/health`, `/info`, `/predict`); `/predict?validate=1` re-runs the registered
  validator inline. Light enough for edge / Jetson.
- **Vision:** `vision_multitask` architecture (MobileNetV3: scene classification,
  shape detection, pose) with ONNX export. Extra: `pip install maatml[vision]`.
- **Vision-language:** `vlm_sft` architecture; `examples/vision-vlm` fine-tunes
  SmolVLM-256M and exports HF-format checkpoints servable by vLLM. Extra:
  `pip install maatml[vllm]` (Linux-only).
- **Captioning:** `examples/vision-describer` (flan-t5 seq2seq) turns the
  multitask vision output into a short description.
- `maatml export --format onnx`; VLM processor assets bundled for vLLM serving.

### Fixed

- CPU-free CI: move SFT config models to `training/sft_config.py` so
  `tests/test_quantization.py` collects without torch; mypy assignment /
  `TrainingArguments` stub mismatches.
- `maatml serve` builds a serve context without torch (falls back to a plain
  device string), so its tests pass on CPU-free CI.

### Changed

- Docs overhaul: README and site now lead with the validator-gated
  *data → eval → serving* differentiator across text / vision / VLM; new
  `docs/lifecycle.md` and `docs/serving.md`; CONTRIBUTING gains a
  "Where to start" good-first-issues section.
- Docs and README lead with `pip install maatml` (PyPI); canonical site is
  [maatml.pages.dev](https://maatml.pages.dev).
- PyPI Trusted Publishing workflow (`.github/workflows/publish.yml`) + GitHub
  Environment `pypi`.

## [0.4.0] - 2026-07-21

### Added

- **Product surface (v0.4):**
  - `maatml export`: safetensors bundle (+ optional `gguf` / `mlx` when
    tooling is installed) with `manifest.json` from `PackagingSpec`
  - `maatml verify`: recompute sha256 of manifest-listed files
  - Post-export `--parity` gate against `dataset.benchmark_samples` +
    `evaluation.gates`
  - `training.model_revision` passed to `from_pretrained` / tokenizer load;
    recorded in `run_metadata.extra`
  - Data flywheel: `maatml.data.gated.build_gated_corpus`,
    `maatml datagen` (registered generators or `--teacher`),
    `maatml ingest` (map / sanitize / validate / dedupe)
  - Optional teacher client (`MAATML_TEACHER_BASE_URL` /
    `MAATML_TEACHER_API_KEY`); extra `pip install maatml[teacher]`
  - Registries: `EXPORTERS` / `GENERATORS` (`register_exporter`,
    `register_generator`); jcl / spool example generators
  - Lightweight docs site: `mkdocs.yml` + `docs/`; extra
    `pip install maatml[docs]`

## [0.3.0] - 2026-07-21

### Added

- **Methods + scale (v0.3):**
  - QLoRA / quantized bases for `causal_sft` via `training.quantization:`
    (bitsandbytes; CUDA-only). Extra: `pip install maatml[cuda]` (with `[ml]`).
  - `DeviceProfile.allow_quantized_load` (True only for `cuda`); hard error on
    mps/cpu when quantization is requested
  - Wired `weights_dtype_policy`: `fp32_master` (mps/cpu) vs `native` bf16/fp16
    master weights on CUDA when `training.precision` matches
  - `training.attn_implementation` passthrough (`flash_attention_2` / `sdpa` /
    `eager`) and `training.dataloader_workers` override across trainers
  - Multi-GPU via accelerate / torchrun: distributed detection, HF Trainer owns
    placement, rank-0-only run-registry / `run_metadata` writes
  - Architectures `dpo` / `orpo` (TRL); format `preference_jsonl`; helper
    `mint_preference_pairs`. Extra: `pip install maatml[pref]` (with `[ml]`)
  - `maatml train --set KEY=VALUE` and offline `maatml sweep --param KEY=a,b`
    (cartesian grid, no Optuna)

## [0.2.0] - 2026-07-21

### Added

- **Experiment layer (v0.2):**
  - Run registry (`output/runs.jsonl`, `maatml runs`) with per-run checkpoint
    dirs under `output/checkpoints/<run_id>/`
  - `maatml train --resume [auto|PATH]` wired into all trainers
  - `training.report_to` / `run_name` passthrough to HuggingFace TrainingArguments
  - Per-head `loss_<name>` logging for `multi_head_classifier`
  - `training.lora.save_mode: merged|adapter|both` (adapter-aware CausalSFTPredictor)
  - Tokenize-once dataset cache (`output/cache/`) + `training.group_by_length` (causal_sft)
  - Format adapters `alpaca` / `sharegpt`; multi-turn loss masking in causal SFT
  - `evaluation.gates` + `maatml evaluate --gate` (non-zero exit on failure)
  - `dataset.group_by` wired into group-aware splits

### Changed

- **Renamed the project from flow-ml to MaatML.** Package / CLI / entry-point
  group are now `maatml` / `maatml` / `maatml.plugins`. Resolves the name
  collision with [MLflow](https://mlflow.org/). GitHub repo:
  [moralfish/maatml](https://github.com/moralfish/maatml).
- **Examples-first restructure:** `jcl-validator` and `spool-interpreter` live
  under `examples/` with folder-local plugins (`jcl_plugin` / `spool_plugin`).
  Core owns architectures (`causal_sft`, `seq2seq`, `multi_head_classifier`);
  examples own validators, metrics, tokenizers, generators, and sanitizer rules.
- Sanitizers and text transforms are registries (`register_sanitizer`,
  `register_transform`); `load_model_plugins` loads package directories.

## [0.1.0] - 2026-07-21

### Added

- Public contribution surface: Apache-2.0 LICENSE, CONTRIBUTING, CODE_OF_CONDUCT,
  SECURITY, CHANGELOG, CODEOWNERS, issue/PR templates, Dependabot, pre-commit
- Plugin registry (`maatml.registry`) with trainers, validators, metrics,
  predictors, formats, and scaffold hooks; entry-point group `maatml.plugins`
- Registry-driven CLI: `prepare`, `train`, `evaluate`, `scaffold`, `validate`,
  `plugins`, `plan`
- Standalone model folders: all paths resolve relative to the model dir; no
  repo-relative fallbacks; wheel package-data for sanitization rules and fixtures
- Device profiles (`mps` / `cuda` / `cpu`) and training guards (NaN abort,
  tokenizer↔embedding contract, `run_metadata.json` provenance)
- Shared validation base + generic evaluation harness
- Reference contrib plugins: `maatml.contrib.jcl`, `maatml.contrib.spool`
- Example model: `examples/support-ticket-triage` (`causal_sft`, ticket → triage JSON)
- Reference models: JCL Validator (ModernBERT multi-head classifier) and Spool
  Interpreter (flan-t5 seq2seq), versioned at `0.1.0`
- Group-aware (`family`) dataset splits and family-stamped seed corpora
- CI: lint (ruff/mypy), Python 3.10-3.12 test matrix, wheel standalone install job
