# MaatML

[![PyPI](https://img.shields.io/pypi/v/maatml.svg)](https://pypi.org/project/maatml/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![CI](https://github.com/Moralfish-Labs/maatml/actions/workflows/ci.yml/badge.svg)](https://github.com/Moralfish-Labs/maatml/actions/workflows/ci.yml)

**Build task-specific models with checks you can inspect.** MaatML prepares
data, trains models, measures their outputs against task contracts, and packages
them for inference across **text, vision, and vision-language**. One declarative
`model.yml` drives **prepare → train → evaluate → export → serve**.

**What makes it different:** correctness is checked *outside* the model by
**validators**. The same validator gates your synthetic **data** and your
**evaluation**, and can guard your **live inference**: `maatml serve` runs it
per request on `/predict?validate=1`, and on every response under `--enforce`,
where a failing output is rejected with HTTP 422. The contract defines what is
checked; passing it establishes those checks, not correctness beyond them.

Site: [maatml.pages.dev](https://maatml.pages.dev) ·
PyPI: [`maatml`](https://pypi.org/project/maatml/) ·
Source: [github.com/Moralfish-Labs/maatml](https://github.com/Moralfish-Labs/maatml)

## Implementation and release status

The package version is **0.20.1**. It resumes a run restored on another host.
0.20.0 added label-only logits for causal SFT (`training.logits: labels`). The
0.19 line added a served-model parity predictor, reasoning-span loss masking
for SFT, free-text rollouts, the
`filtered_imitation` objective and an environment-verified GRPO trainer. It also
includes regular-expression LoRA targets, the `adapter` export format with a
layout check, a refusal of over-long SFT rows instead of truncation,
processor-aware SFT trainers, frozen-component checks, multimodal GGUF
companions, `maatml snapshot` checkpoint
backups, per-prediction evidence, streaming preparation, `dataset.train_samples` for corrective examples, stricter
population checks, rollout/DAgger collection, tool-call chat training, and
multi-turn masking. See [CHANGELOG.md](CHANGELOG.md) and [ROADMAP.md](ROADMAP.md).

MaatML owns the model lifecycle and its evidence. Model folders own task
contracts and recipes; an external orchestrator owns remote machines,
scheduling, storage transfers, monitoring, and recovery policy. The
[orchestration guide](docs/orchestration.md) describes how to keep those
responsibilities separate while carrying checkpoints and run records between
machines.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

# Library + CLI (no torch)
pip install maatml

# Training / evaluation stack
pip install "maatml[ml]"

# Optional extras
pip install "maatml[ml,cuda]"    # QLoRA on NVIDIA CUDA (bitsandbytes)
pip install "maatml[ml,pref]"    # DPO / ORPO (TRL >= 0.9)
pip install "maatml[ml,vision]"  # torchvision + ONNX (examples/vision)
pip install "maatml[vllm]"       # Linux-only vLLM serving (examples/vision-vlm)
pip install "maatml[teacher]"    # OpenAI-compatible teacher for datagen / distill
pip install "maatml[docs]"       # site tooling; also requires a source checkout
```

Then:

```bash
maatml --help
maatml scaffold ~/models/my-task --architecture causal_sft --name my-task
maatml validate ~/models/my-task
```

For contributing to this repository (editable install), see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Example models

Four reference models share the identical folder layout and CLI, from a
one-command support-ticket triage to a vLLM-servable vision-language model:

| Model | Task | Architecture | Base |
|-------|------|--------------|------|
| [Support Ticket Triage](examples/support-ticket-triage/) | triage → JSON | `causal_sft` (LoRA) | Qwen3-0.6B |
| [Vision VLM](examples/vision-vlm/) | describe a scene image | `vlm_sft` (vLLM-servable) | SmolVLM-256M-Instruct |
| [Vision](examples/vision/) | scene + detect + pose | `vision_multitask` | MobileNetV3-Large |
| [Vision Describer](examples/vision-describer/) | caption from vision JSON | `seq2seq` | flan-t5-small |

Any directory with a valid `model.yml` works the same way: install maatml from
PyPI and point the CLI at the folder. Scaffold a new model folder with
`maatml scaffold`.

## Where MaatML fits

MaatML **builds on** Hugging Face `transformers` / `peft` / `trl` and does the
one thing those building blocks leave to you: it wraps them in an opinionated,
validator-gated lifecycle for **small** task-specific models. Train on a suitable
local device or cloud GPU, then export for the runtime your model supports.

- **Complements** general fine-tuning tools (Axolotl, LLaMA-Factory, Unsloth,
  TRL) rather than competing on scale. Reach for those for large models,
  multi-node training, large-scale RL, or broad model coverage. MaatML includes
  supervised fine-tuning, DPO/ORPO, `filtered_imitation`, and
  environment-verified `grpo` scoped to small task-specific models.
- **Runs its own fixed lifecycle** (`prepare → train → evaluate → export →
  verify` / `serve`), as one command: `maatml run`, which skips steps that are
  already fresh and stops non-zero at the first failure. It is **not** a
  general-purpose workflow scheduler: no triggers, no arbitrary shell/Python
  steps, no remote executors. An external orchestrator can invoke its CLI on
  local machines, notebook GPUs, or GPU VMs; see
  [orchestration and recovery](docs/orchestration.md).
- **Its niche:** local-first, multimodal, structured-output models with
  correctness gated *outside* the model, from data generation through serving.

## Requirements

- **Python** 3.10+ (developed against 3.13)
- **OS** macOS and Linux; Windows has a torch-free CLI/test CI job, not a
  demonstrated GPU training path
- **Disk / memory** depend on the selected extras, base model, context length
  and batch size. Reserve space for downloads, checkpoints and exports; use
  a smoke run to check the recipe on your hardware.

## CLI overview

Most commands take a model folder (containing `model.yml`) as their first
argument. Outputs land under `<model-folder>/output/` (gitignored). Run
`maatml <command> --help` for the full flag list. Errors in your input (a
missing file, an unparseable `model.yml`, an unregistered plugin) print one
line; `maatml --debug <command>` prints the traceback.

| Command | What it does |
| --- | --- |
| `run` | The whole lifecycle in one command: prepare, train, evaluate (gated), export, verify. Skips fresh steps; accepts `--resume auto\|PATH` and `--progress-file PATH` |
| `prepare` | Build `train`/`val`/`test` splits from the seed corpus (plus optional `dataset.train_samples`); enforces `dataset.isolation` / `pins`, records the benchmark version and corpus lock (including supporting assets), and checks source sign-offs when `dataset.attribution` is declared |
| `train` | Fine-tune the model (`causal_sft`, `seq2seq`, `multi_head_classifier`, `dpo`, `orpo`, `diffusion_lora`, `filtered_imitation`, `grpo`, or plugin architectures); supports `--smoke`, `--resume auto\|PATH`, `--progress-file PATH`, `--set K=V`, `--seeds N`; `training.select_by` picks the checkpoint on val |
| `sweep` | Offline grid HPO over `--param K=a,b` |
| `evaluate` | Score a local checkpoint or live OpenAI-compatible endpoint (`served_http` / `serve_parity`); `--gate` exits non-zero on a gate miss, `--cache` keeps per-row predictions and `prediction_context`, `--out DIR` separates reports from historical results, `--batch-size N` uses a predictor's `predict_batch` when available, `--set K=V` overrides configuration (never with `--gate`), `--blind` spends the blind manifest once, `--strict-population` refuses mismatched gate populations before loading the predictor. The token budget defaults to `packaging.max_input_tokens` |
| `gates derive` | Floors from a run's report: Wilson 95 % at each metric's own denominator, cluster bootstrap from a cache, `--seed-study`; `--write` rewrites `evaluation.gates` with the derivation beside each floor |
| `ship-check` | `CANDIDATE BASELINE`: absolute, delta and population verdict in one; `--replay` re-evaluates both over the current test split |
| `operating-point derive` | Sweep the predictor's `rescore` over a val cache under a budget; `--write` the cut, `--confirm-on-test` spends test once |
| `export` | Deployable bundle + `manifest.json` (`--format safetensors\|gguf\|mlx\|adapter\|onnx`, `--parity`); `adapter` checks `extensions.adapter.spec`, `gguf` supports `extensions.gguf.mmproj` multimodal companions |
| `compile` | Dispatch an export to a registered target compiler; `--require-gated` rejects missing, failed or smoke-tier gate evidence |
| `manifest amend` | Add files inside an export to its checksummed manifest without changing gate evidence |
| `verify` | Recompute sha256 of listed manifest files and flag unlisted files under the export directory |
| `serve` | JSON inference API (`http` default, or `anthropic` / plugin backends); `--enforce` (422), `--max-retries`, `--auth-token`, `--capture` |
| `snapshot` | Wait for a complete policy or SFT checkpoint (`>= --min-step`) and write a gzip-compressed portable backup bundle without stopping training |
| `datagen` | Validator-gated seed generation (`--teacher`, `--allow-ungated`) |
| `distill` | Validator-gated teacher labels over a prompt pool (`--replay` offline) |
| `mint` | Preference pairs (`chosen`/`rejected`) from validator-scored candidates |
| `ingest` | Import external samples (`--map field=col`, `--sanitize tag`, `--video FILE`) |
| `runs` | List recorded training runs (`--compare` tabulates their metrics); `--pack RUN` / `--adopt BUNDLE` carry a run, its evidence and its record between machines |
| `report` | Runs, floors with their derivation, slices, pathologies, seed statistics and spends, regenerated from `output/` alone (`--format md\|csv`) |
| `plan` | Show which lifecycle steps are stale (alias for `run --dry-run`) |
| `plugins` | List registered trainers, validators, metrics, predictors, formats, generators, exporters, compilers, servers and environments |
| `audit` | Check the environment, plugins, and a model folder; exits 1 on problems |
| `rollout` | Run a checkpoint in a verifying environment (candidate-ranking `line_world` or free-text `free_text` / `arithmetic_world`); `--label` collects teacher labels for DAgger, `--teacher-mix` mixes teacher decisions |
| `scaffold` | Create a new model folder (`--architecture`, `--plugin`, `--force`) |
| `validate` | Check `model.yml` and paths (`--no-plugins` skips plugin code) |

Multi-GPU (CUDA): `accelerate launch -m maatml.cli train <model-dir>/` or
`torchrun --nproc_per_node=N -m maatml.cli train <model-dir>/`.

QLoRA (CUDA + `[cuda]`): set `training.quantization.load_in_4bit: true` in
`model.yml`. Preference data: `dataset.format: preference_jsonl` with
`{prompt, chosen, rejected}` rows; scaffold with `--architecture dpo`.
Verified-output post-training: use `architecture: filtered_imitation` for
environment-verified expert iteration, or `architecture: grpo` for
environment-verified GRPO with a frozen anchor checkpoint KL penalty (see
[Rollouts](docs/rollouts.md) and [Lifecycle](docs/lifecycle.md)).

Export defaults to a safetensors bundle + `manifest.json`. `adapter` exports
`adapters/<name>/` (`adapter_config.json` + `adapter_model.safetensors`) with
header-only rank and optional `extensions.adapter.spec` shape checks. GGUF/MLX
need external tooling (`llama.cpp` convert / `mlx_lm`), and
`extensions.gguf.mmproj: true` emits an f16 `mmproj-<name>.gguf` multimodal
companion. Pin base-model revisions with `training.model_revision`, confine LoRA
to one tower with a regex `training.lora.target_modules`, and guard frozen
submodules with `training.frozen_module_prefixes`.

The causal SFT path preserves per-row tool definitions, assistant tool calls
(ChatML `<tool_call>` and Gemma 4 `<|tool_call>`), and tool results. It refuses
rows exceeding `training.max_input_tokens` (`RowTooLongError`) instead of
silently truncating prompts, supervises every assistant turn while masking other
roles, and supports selective reasoning-span loss masking via
`training.reasoning_loss: include | mask | reasoning_only`; see
[tool-call chat training](docs/lifecycle.md#tool-call-chat-training) and
[reasoning-span loss masking](docs/lifecycle.md#reasoning-span-loss-masking).

Docs: [maatml.pages.dev](https://maatml.pages.dev) · Roadmap: [ROADMAP.md](ROADMAP.md) ·
In-repo docs: `docs/` (from the repository root, `pip install -e ".[docs]"`
then `mkdocs serve`). See the [CLI reference](docs/cli.md) and
[configuration reference](docs/skill/references/model-yml.md).

## End-to-end example (Support Ticket Triage)

The quickest model to run: a LoRA fine-tune of Qwen3-0.6B that turns a raw
support ticket into `{priority, category, team, summary}` JSON, gated by a
schema validator plus a `category → team` routing contract enforced *outside*
the model. Every reference model now registers a validator and declares
`evaluation.gates`.

```bash
git clone https://github.com/Moralfish-Labs/maatml.git
cd maatml
pip install -e ".[ml]"   # match the framework to the checked-out examples

maatml prepare  examples/support-ticket-triage/
maatml train    examples/support-ticket-triage/ --smoke   # fast pipeline check
maatml train    examples/support-ticket-triage/
maatml evaluate examples/support-ticket-triage/ --gate    # enforce eval gates
maatml serve    examples/support-ticket-triage/           # JSON inference API
```

For a **multimodal** walkthrough (image → description, servable by vLLM) see
[examples/vision-vlm/](examples/vision-vlm/).

## From a passed gate to a claim

A green `evaluate --gate` is the start of the evidence, not the end of it.
The same CLI derives the floors, chooses the operating point, names the
populations, and carries the run home:

```bash
maatml gates derive <model-dir> --run RUN --write      # Wilson floors, derivation beside each
maatml operating-point derive <model-dir> --run RUN --write --confirm-on-test
maatml ship-check <model-dir> CANDIDATE BASELINE        # absolute + delta + population
maatml evaluate <model-dir> --blind                     # once per frozen candidate
maatml runs <model-dir> --pack RUN                      # → --adopt on the machine that exports
maatml report <model-dir>                               # everything above, from the records alone
```

`model.yml` declares the rest: `dataset.isolation` / `pins` / `blind_samples`
for populations, `dataset.attribution` for the licence table every source
must be signed in, `training.select_by` for checkpoint selection on val.
[docs/evidence.md](docs/evidence.md) walks through it.

Keep acceptance floors fixed while measuring a corrective candidate. Deriving
new floors from that candidate is a separate calibration decision, not a way
to turn a failed experiment into a pass. `ship-check` rejects failed aggregate
gate evidence, missing or mismatched comparison populations, and a candidate
population that differs from a declared `evaluation.gates_benchmark`.
`--replay` must produce reports over the same rows; it does not waive those checks.

## Corrective data without changing the test population

For the `jsonl_seed` format, add reviewed corrective examples through
`dataset.train_samples` instead of mixing them into the seed population:

```yaml
dataset:
  format: jsonl_seed
  seed_samples: datasets/samples/seed_samples.jsonl
  train_samples: datasets/samples/corrections.jsonl
  benchmark_samples: datasets/samples/benchmark.jsonl
```

Train-only rows enter after the seed split. Preparation checks held-out group
collisions, applies configured isolation and attribution checks, and records
the extra input in `corpus.lock.json`. This preserves the seed split assignment;
keep the seed data, grouping, split settings, and benchmark fixed when comparing
experiments. Source attribution records provenance, not model quality.

## Cloud training, progress, and recovery

Use the same model folder and CLI on the GPU runtime. Transfer the prepared
data, plugin, recipe, and any local base checkpoint before running:

```bash
maatml validate <model-dir>
maatml train <model-dir> --device cuda \
  --progress-file /tmp/progress.json \
  --set training.logging_steps=1 --set training.disable_tqdm=true \
  --set training.checkpoint_snapshots=true
# Optionally back up a completed mid-run checkpoint while training continues:
maatml snapshot <model-dir> --started-after /tmp/launch.marker \
  --writer-lock /tmp/train.lock --min-step 200 --out /tmp/ckpt-200.tar.gz
# After an interruption (SIGTERM/SIGINT flushes a checkpoint and records status="interrupted"):
maatml run <model-dir> --device cuda --resume auto \
  --progress-file /tmp/progress.json \
  --set training.logging_steps=1 --set training.disable_tqdm=true
maatml evaluate <model-dir> --checkpoint RUN --cache --out <report-dir>
maatml runs <model-dir> --pack RUN
```

`--progress-file` (or `FLOW_JOB_PROGRESS_FILE`) writes atomic `JobProgress` JSON
snapshots (`heartbeatAt`, `stage`, `epoch`, `step`, `runId`, `metricName`,
`metricValue`, `checkpointAgeSecs`) from rank 0 so supervisors can monitor
progress without scraping logs. `disable_tqdm` replaces the SFT progress bar
with ordinary Trainer log records; `logging_steps` controls their frequency.
On `SIGTERM` or `SIGINT` inside the training loop, MaatML flushes a checkpoint
at the step boundary and records run status `interrupted` in `runs.jsonl`.
`training.checkpoint_snapshots: true` publishes a hash-bound completion marker
beside each saved single-process SFT checkpoint (single-file or sharded) so
`maatml snapshot` captures only complete checkpoints at gzip level 1.
Evaluation prints row progress and estimated time remaining. Set
`PYTHONUNBUFFERED=1` when an external orchestrator streams Python output.
Resume requires a saved checkpoint; work since the last save may need to run
again.

The orchestrator provisions and stops the GPU, streams logs, and transfers
checkpoints. On the receiving machine, `maatml runs <model-dir> --adopt BUNDLE`
imports the completed run and its evidence. A completed training run does not
by itself establish an acceptance or release pass. See
[orchestration and recovery](docs/orchestration.md).

## Batch scripts

```bash
# Deterministic seed corpora (no API calls)
python examples/support-ticket-triage/scripts/build_seeds.py
python examples/vision/scripts/build_seeds.py
python examples/vision-vlm/scripts/build_seeds.py
python examples/vision-describer/scripts/build_seeds.py

# Train / evaluate example models
python scripts/train_all.py --smoke
python scripts/train_all.py
python scripts/evaluate_all.py
```

## Apple Silicon / MPS notes

- Default precision is **bf16** autocast with fp32 master weights.
- The MPS device profile disables mid-training evaluation; setting
  `eval_steps` does not override that profile.
- `grad_checkpointing` defaults to `false`; `dataloader_num_workers=0`
  everywhere (multi-worker + MPS can deadlock via fork pickling).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the CLI for unsupported ops.

## Repository layout

```
examples/                   # reference task models (plugins + data)
  support-ticket-triage/    # causal LoRA SFT
  vision/                   # multitask vision
  vision-vlm/               # vision-language LoRA SFT
  vision-describer/         # seq2seq captioning

src/maatml/                 # core framework (architectures, CLI, harnesses)
scripts/                    # batch train/eval/validate
tests/                      # core unit tests
```

## Trust boundary

A model folder is executable code, not just data. Any `plugins:` entry in
`model.yml` is imported as Python the moment the folder is loaded, and every
command that reads `model.yml`, including `maatml validate` and `maatml plan`,
loads the folder. Running any `maatml` command against a folder therefore runs
that folder's code with your privileges. Only run `maatml` on model folders you
trust, the same way you would only run a script you trust. Use
`maatml validate --no-plugins` to check the schema and paths without importing
plugin code.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, PR expectations, DCO
sign-off, and versioning policy. AI coding agents: [AGENTS.md](AGENTS.md).

Community: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) · Security:
[SECURITY.md](SECURITY.md) · Changes: [CHANGELOG.md](CHANGELOG.md)

## Licensing

- **maatml** is licensed under the [Apache License 2.0](LICENSE).
- This repository **does not redistribute base-model weights**, only Hugging
  Face Hub IDs. **Your fine-tuned checkpoints inherit the base model's license
  terms**.
- Seed corpora are **fully synthetic**, produced by deterministic builders under
  `examples/*/scripts/`, with no proprietary source data shipped.
