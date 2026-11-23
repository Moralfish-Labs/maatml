# CLI reference

Run `maatml --version` to identify the installed package and
`maatml COMMAND --help` for its full option list. These docs describe
**0.20.0**, including served-model parity, free-text rollouts, environment-verified GRPO, spec-checked adapter export, multimodal GGUF companions, checkpoint snapshots, corrective training inputs and evaluation population safeguards. `maatml --debug COMMAND` includes tracebacks for input errors.

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
| `maatml run MODEL` | Prepare, train, gated evaluate, export and verify; skip fresh steps; accepts `--resume auto\|PATH` and `--progress-file PATH`. |
| `maatml prepare MODEL` | Write train, val and test splits (including optional `dataset.train_samples`), benchmark identity and corpus lock. |
| `maatml train MODEL` | Train (`causal_sft`, `seq2seq`, `multi_head_classifier`, `dpo`, `orpo`, `diffusion_lora`, `filtered_imitation`, `grpo`, or plugin architectures); `--smoke`, `--device`, `--seed`, `--seeds`, `--resume`, `--progress-file` and `--set` configure the run. |
| `maatml sweep MODEL --param training.learning_rate=1e-4,3e-4` | Run a bounded grid of training recipes; use `--max-trials` to cap work. |
| `maatml evaluate MODEL --checkpoint RUN_ID --gate` | Score a checkpoint or live OpenAI-compatible server (`served_http` / `serve_parity`) and fail on blocking gates; `--cache` preserves per-row predictions and `prediction_context`, `--out DIR` writes reports separately, `--batch-size N` enables batched prediction, `--strict-population` refuses mismatched gate populations, `--blind` spends the blind split once, and `--set` overrides config (refused with `--gate`/`--blind`). |
| `maatml export MODEL --checkpoint RUN_ID --format safetensors` | Write a portable bundle and manifest (`safetensors`, `gguf`, `mlx`, `adapter`, or plugin formats like `onnx`); `--parity` re-checks `evaluation.gates` on `benchmark_samples`. |
| `maatml verify EXPORT` | Check listed manifest file hashes and flag unlisted files under the export directory. |
| `maatml compile EXPORT --target NAME --out TARGET --require-gated` | Invoke a registered target compiler with production gate evidence required. |
| `maatml manifest amend EXPORT EXPORT/model.gguf --format gguf` | Add an artifact's checksum while preserving existing gate evidence. |
| `maatml serve MODEL --checkpoint RUN_ID --enforce` | Start inference with the selected backend (`http`, `anthropic`, or a plugin); the default HTTP backend rejects invalid predictions with HTTP 422. |

`run --smoke` uses `smoke.gates` when declared, otherwise the production gates.
`--from`, `--until`, `--force`, `--resume` and `--progress-file` control the
pipeline walk. Serving, compilation, snapshots and data collection are separate
operations. See [lifecycle](lifecycle.md).

## Data collection

| Command | Purpose |
| --- | --- |
| `maatml datagen MODEL --target 100` | Generate rows with the registered generator; `--teacher` uses a teacher endpoint. |
| `maatml distill MODEL --prompts prompts.jsonl` | Label prompts through a teacher; `--replay` consumes the recorded cache offline. |
| `maatml ingest MODEL --input rows.jsonl` | Map, sanitize and optionally validate imported rows (`--video` extracts frames via ffmpeg). |
| `maatml mint MODEL --input candidates.jsonl` | Create chosen/rejected preference pairs from validator verdicts. |
| `maatml rollout MODEL --env NAME --label` | Collect verified episodes and optional teacher labels in candidate-ranking (`line_world`) or free-text (`free_text` / `arithmetic_world`) environments, in-process (`--env`) or over stdio (`--env-command`). |

These commands can change source data. See [the data flywheel](flywheel.md)
and [rollouts](rollouts.md) for provenance and held-out population boundaries.

## Evidence and transfer

| Command | Purpose |
| --- | --- |
| `maatml gates derive MODEL --run RUN_ID --write` | Derive and record evaluation floors from a measured report or `--seed-study`. |
| `maatml operating-point derive MODEL --run RUN_ID --write` | Choose a threshold from a val prediction cache using a predictor's `rescore`; `--confirm-on-test` spends test once. |
| `maatml ship-check MODEL RUN_ID BASELINE` | Compare absolute gates, regression and populations; `--replay` uses the current test split for both. |
| `maatml snapshot MODEL --started-after FILE --writer-lock PATH --min-step N --out BUNDLE` | Wait for a complete policy or SFT checkpoint (`>= --min-step`) and write a gzip-compressed backup bundle (`--compress-level`, default 1) without stopping training. |
| `maatml runs MODEL --compare` | Inspect recorded runs and metrics. |
| `maatml runs MODEL --pack RUN_ID --with-checkpoints` | Bundle a run and its evidence, including intermediate checkpoints. |
| `maatml runs MODEL --adopt BUNDLE` | Verify and import a bundle into a compatible model folder. |
| `maatml report MODEL` | Render recorded evidence as Markdown or CSV. |

See [evidence](evidence.md) for gate tiers, blind evaluation and test spends,
and [orchestration](orchestration.md) for recovery and portable runs.
