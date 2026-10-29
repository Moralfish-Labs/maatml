# CLI reference

Run `maatml --version` to identify the installed package and
`maatml COMMAND --help` for its full option list. These docs describe
**0.15.0**, including corrective training inputs and evaluation population safeguards. `maatml --debug COMMAND` includes tracebacks for input errors.

Most commands take a folder containing `model.yml`. Paths declared in that
file resolve relative to the model folder. Replace placeholders such as
`MODEL`, `RUN_ID`, `BASELINE` and `EXPORT` with your own paths or run identifiers.

## Setup and inspection

| Command | Purpose |
| --- | --- |
| `maatml scaffold MODEL --architecture causal_sft` | Create a standalone model folder; `--plugin` loads a plugin-owned architecture. |
| `maatml validate MODEL` | Validate configuration, declared paths and registrations; `--no-plugins` avoids plugin imports. |
| `maatml audit MODEL` | Inspect environment and model health; `--json` emits structured results. |
| `maatml plugins` | List discovered registrations and unavailable plugins. |
| `maatml plan MODEL` | Show freshness decisions without running lifecycle stages. |

## Lifecycle

| Command | Purpose |
| --- | --- |
| `maatml run MODEL` | Prepare, train, gated evaluate, export and verify; skip fresh steps. |
| `maatml prepare MODEL` | Write train, val and test splits, benchmark identity and corpus lock. |
| `maatml train MODEL` | Train; `--smoke`, `--device`, `--seed`, `--resume` and `--set` configure the run. |
| `maatml sweep MODEL --param training.learning_rate=1e-4,3e-4` | Run a bounded grid of training recipes; use `--max-trials` to cap work. |
| `maatml evaluate MODEL --checkpoint RUN_ID --gate` | Score a checkpoint and fail on blocking gates; `--cache` preserves per-row predictions. |
| `maatml export MODEL --checkpoint RUN_ID --format safetensors` | Write a portable bundle and manifest. |
| `maatml verify EXPORT` | Check the manifest's file hashes. |
| `maatml compile EXPORT --target NAME --out TARGET --require-gated` | Invoke a registered target compiler with production gate evidence required. |
| `maatml manifest amend EXPORT EXPORT/model.gguf --format gguf` | Add an artifact's checksum while preserving existing gate evidence. |
| `maatml serve MODEL --checkpoint RUN_ID --enforce` | Start inference with the selected backend; the default HTTP backend rejects invalid predictions. |

`run --smoke` uses `smoke.gates` when declared, otherwise the production gates.
`--from`, `--until` and `--force` control the pipeline walk. Serving, compilation
and data collection are separate operations. See [lifecycle](lifecycle.md).

## Data collection

| Command | Purpose |
| --- | --- |
| `maatml datagen MODEL --target 100` | Generate rows with the registered generator; `--teacher` uses a teacher endpoint. |
| `maatml distill MODEL --prompts prompts.jsonl` | Label prompts through a teacher; `--replay` consumes the recorded cache offline. |
| `maatml ingest MODEL --input rows.jsonl` | Map, sanitize and optionally validate imported rows. |
| `maatml mint MODEL --input candidates.jsonl` | Create chosen/rejected preference pairs from validator verdicts. |
| `maatml rollout MODEL --env NAME --label` | Collect verified episodes and teacher labels using a compatible policy predictor. |

These commands can change source data. See [the data flywheel](flywheel.md)
and [rollouts](rollouts.md) for provenance and held-out population boundaries.

## Evidence and transfer

| Command | Purpose |
| --- | --- |
| `maatml gates derive MODEL --run RUN_ID --write` | Derive and record evaluation floors from a measured report. |
| `maatml operating-point derive MODEL --run RUN_ID --write` | Choose a threshold from a val prediction cache using a predictor's `rescore`. |
| `maatml ship-check MODEL RUN_ID BASELINE` | Compare absolute gates, regression and populations; `--replay` uses the current test split for both. |
| `maatml runs MODEL --compare` | Inspect recorded runs and metrics. |
| `maatml runs MODEL --pack RUN_ID --with-checkpoints` | Bundle a run and its evidence, including intermediate checkpoints. |
| `maatml runs MODEL --adopt BUNDLE` | Verify and import a bundle into a compatible model folder. |
| `maatml report MODEL` | Render recorded evidence as Markdown or CSV. |

See [evidence](evidence.md) for gate tiers, blind evaluation and test spends,
and [orchestration](orchestration.md) for recovery and portable runs.
