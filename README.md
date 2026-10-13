# maatml

[![PyPI](https://img.shields.io/pypi/v/maatml.svg)](https://pypi.org/project/maatml/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![CI](https://github.com/moralfish/maatml/actions/workflows/ci.yml/badge.svg)](https://github.com/moralfish/maatml/actions/workflows/ci.yml)

**MaatML** fine-tunes small, task-specific models across **text, vision, and
vision-language**, and takes them from experimentation to production through a
single declarative `model.yml`: **prepare → train → evaluate → export → serve**.
Licensed under **Apache-2.0**.

**What makes it different:** correctness is checked *outside* the model by
**validators**. The same validator gates your synthetic **data** and your
**evaluation**, and can guard your **live inference**: `maatml serve` runs it
per request on `/predict?validate=1`, and on every response under `--enforce`,
where a failing output is rejected with HTTP 422. So a MaatML model ships with a
contract, not just weights. That validator-gated *data → eval → serving* loop,
now across modalities, is what general fine-tuning tools leave out.

Site: [maatml.pages.dev](https://maatml.pages.dev) ·
PyPI: [`maatml`](https://pypi.org/project/maatml/) ·
Source: [github.com/moralfish/maatml](https://github.com/moralfish/maatml)

## Implementation and release status

The package version is **0.13.0**. The source checkout also contains unreleased
rollout/DAgger collection, train-only dataset rows, tool-call chat training,
and multi-turn masking fixes. These are implemented features, not guarantees
about an installed 0.13.0 wheel; see [CHANGELOG.md](CHANGELOG.md) for the release
boundary and [ROADMAP.md](ROADMAP.md) for remaining work.

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
pip install "maatml[ml,pref]"    # DPO / ORPO (TRL)
pip install "maatml[ml,vision]"  # torchvision + ONNX (examples/vision)
pip install "maatml[vllm]"       # Linux-only vLLM serving (examples/vision-vlm)
pip install "maatml[teacher]"    # OpenAI-compatible teacher for datagen
pip install "maatml[docs]"       # mkdocs site
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
validator-gated lifecycle for **small** task-specific models you can train on a
laptop and deploy to the edge or vLLM.

- **Complements** general fine-tuning tools (Axolotl, LLaMA-Factory, Unsloth,
  TRL) rather than competing on scale. Reach for those for large models,
  multi-node training, RL, or broad model coverage.
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
- **Disk / memory** ~3 GB for the ML stack; 16 GB unified memory is the design
  target for local training

## CLI overview

Most commands take a model folder (containing `model.yml`) as their first
argument. Outputs land under `<model-folder>/output/` (gitignored). Run
`maatml <command> --help` for the full flag list. Errors in your input (a
missing file, an unparseable `model.yml`, an unregistered plugin) print one
line; `maatml --debug <command>` prints the traceback.

| Command | What it does |
| --- | --- |
| `run` | The whole lifecycle in one command: prepare, train, evaluate (gated), export, verify. Skips steps that are already fresh |
| `prepare` | Build `train`/`val`/`test` splits from the seed corpus; enforces `dataset.isolation` / `pins`, records the benchmark version and the corpus lock, refuses unsigned sources (`dataset.attribution`) |
| `train` | Fine-tune the model (`--smoke`, `--resume auto\|PATH`, `--set K=V`, `--seeds N`); `training.select_by` picks the checkpoint on val |
| `sweep` | Offline grid HPO over `--param K=a,b` |
| `evaluate` | Score a checkpoint; `--gate` exits non-zero on a gate miss, `--cache` keeps per-row predictions, `--batch-size N` feeds a predictor's `predict_batch` in chunks, `--set K=V` overrides model.yml for that evaluate (recorded, never with `--gate`), `--blind` spends the blind manifest once, `--strict-population` refuses floors from another split. The token budget defaults to `packaging.max_input_tokens` |
| `gates derive` | Floors from a run's report: Wilson 95 % at each metric's own denominator, cluster bootstrap from a cache, `--seed-study`; `--write` rewrites `evaluation.gates` with the derivation beside each floor |
| `ship-check` | `CANDIDATE BASELINE`: absolute, delta and population verdict in one; `--replay` re-evaluates both over the current test split |
| `operating-point derive` | Sweep the predictor's `rescore` over a val cache under a budget; `--write` the cut, `--confirm-on-test` spends test once |
| `export` | Deployable bundle + `manifest.json` (`--format`, `--parity`) |
| `verify` | Recompute sha256 of an export against its `manifest.json` |
| `serve` | JSON inference API; `--enforce` (422), `--max-retries`, `--auth-token`, `--capture` |
| `datagen` | Validator-gated seed generation (`--teacher`, `--allow-ungated`) |
| `distill` | Validator-gated teacher labels over a prompt pool (`--replay` offline) |
| `mint` | Preference pairs (`chosen`/`rejected`) from validator-scored candidates |
| `ingest` | Import external samples (`--map field=col`, `--sanitize tag`) |
| `runs` | List recorded training runs (`--compare` tabulates their metrics); `--pack RUN` / `--adopt BUNDLE` carry a run, its evidence and its record between machines |
| `report` | Runs, floors with their derivation, slices, pathologies, seed statistics and spends, regenerated from `output/` alone (`--format md\|csv`) |
| `plan` | Show which lifecycle steps are stale (alias for `run --dry-run`) |
| `plugins` | List discovered trainers, validators, and metrics |
| `audit` | Check the environment, plugins, and a model folder; exits 1 on problems |
| `rollout` (unreleased) | Run a checkpoint in a verifying environment; `--label` collects teacher labels for DAgger, and `--teacher-mix` mixes teacher decisions |
| `scaffold` | Create a new model folder (`--architecture`, `--plugin`, `--force`) |
| `validate` | Check `model.yml` and paths (`--no-plugins` skips plugin code) |

Multi-GPU (CUDA): `accelerate launch -m maatml.cli train <model-dir>/` or
`torchrun --nproc_per_node=N -m maatml.cli train <model-dir>/`.

QLoRA (CUDA + `[cuda]`): set `training.quantization.load_in_4bit: true` in
`model.yml`. Preference data: `dataset.format: preference_jsonl` with
`{prompt, chosen, rejected}` rows; scaffold with `--architecture dpo`.

Export defaults to a safetensors bundle + `manifest.json`. GGUF/MLX need
external tooling (`llama.cpp` convert / `mlx_lm`). Pin base-model revisions
with `training.model_revision`.

The unreleased causal SFT path also preserves per-row tool definitions,
assistant tool calls and tool results. It supervises every assistant turn and
masks the other roles; see [tool-call chat training](docs/lifecycle.md#tool-call-chat-training-unreleased).

Docs: [maatml.pages.dev](https://maatml.pages.dev) · Roadmap: [ROADMAP.md](ROADMAP.md) ·
In-repo docs: `docs/` (`pip install "maatml[docs]"` then `mkdocs serve`).

## End-to-end example (Support Ticket Triage)

The quickest model to run: a LoRA fine-tune of Qwen3-0.6B that turns a raw
support ticket into `{priority, category, team, summary}` JSON, gated by a
schema validator plus a `category → team` routing contract enforced *outside*
the model. Every reference model now registers a validator and declares
`evaluation.gates`.

```bash
git clone https://github.com/moralfish/maatml.git
cd maatml
pip install "maatml[ml]"

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
- Trainers set `eval_steps: 9999` to disable mid-training eval on MPS (unified
  memory does not release val-set tensors between eval and training).
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
