# AGENTS.md

Guidance for AI coding agents working in this repository.

## Commands

**Users** install from PyPI (`pip install maatml` / `pip install "maatml[ml]"`).
This file is for **contributors** working in a checkout.

All commands assume the project venv (`.venv/bin/...`). Training needs the
`[ml]` extra (torch, transformers, peft, tokenizers). Torch-free contributions
can use `pip install -e ".[dev]"`; unit tests run without torch.

```bash
# Install (editable, for framework development)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ml]"
# Optional: QLoRA (CUDA Linux) / preference trainers / teacher / docs / vision / vllm
# pip install -e ".[dev,ml,cuda]"   # bitsandbytes
# pip install -e ".[dev,ml,pref]"   # trl (DPO/ORPO)
# pip install -e ".[teacher]"       # httpx for maatml datagen --teacher
# pip install -e ".[docs]"          # mkdocs-material
# pip install -e ".[dev,ml,vision]" # torchvision + ONNX for examples/vision
# pip install -e ".[vllm]"          # Linux-only OpenAI-compatible VLM serving

# Docs: pip install -e ".[docs]" then mkdocs build --strict
# Test / lint (run from repo root)
.venv/bin/python -m pytest tests/ examples/ -q
ruff check src tests scripts examples
ruff format --check src tests scripts examples
mypy src/maatml --ignore-missing-imports

# Per-model lifecycle (any standalone folder with model.yml)
.venv/bin/maatml scaffold ~/models/my-task --architecture causal_sft
.venv/bin/maatml scaffold ~/models/my-dpo --architecture dpo
.venv/bin/maatml scaffold ~/models/my-vision --architecture vision_multitask \
    --plugin examples/vision/vision_plugin   # plugin-owned architectures
.venv/bin/maatml validate ~/models/my-task
.venv/bin/maatml run      examples/<name>/ [--smoke] [--dry-run] [--force] [--from STEP] [--until STEP] [--set K=V]
.venv/bin/maatml prepare  examples/<name>/
.venv/bin/maatml train    examples/<name>/ [--smoke] [--device mps|cpu|cuda] [--seed N] [--limit N] [--resume auto|PATH] [--set training.learning_rate=1e-4]
.venv/bin/maatml sweep    examples/<name>/ --param training.learning_rate=1e-4,3e-4 --param training.lora.r=8,16 [--metric eval_loss] [--smoke] [--max-trials N]
.venv/bin/maatml evaluate examples/<name>/ [--checkpoint X] [--split test|val] [--gate]
.venv/bin/maatml export   examples/<name>/ [--checkpoint X] [--format safetensors|gguf|mlx|onnx] [--out PATH] [--parity]
.venv/bin/maatml compile  <export-dir> --target NAME --out DIR [--option K=V] [--require-gated]
.venv/bin/maatml verify   examples/<name>/output/export/<run_id>
.venv/bin/maatml serve    examples/<name>/ [--server http|NAME] [--server-option K=V] [--enforce] [--max-retries N] [--auth-token T] [--capture PATH]
.venv/bin/maatml datagen  examples/<name>/ [--target N] [--seed S] [--teacher] [--out PATH]
.venv/bin/maatml distill  examples/<name>/ [--prompts PATH] [--replay] [--offline]
.venv/bin/maatml mint     examples/<name>/ --input candidates.jsonl   # dpo/orpo pairs
.venv/bin/maatml ingest   examples/<name>/ --input PATH [--map field=col] [--sanitize tag] [--append] [--video FILE]
.venv/bin/maatml runs     examples/<name>/ [--compare] [--metric K] [--limit N]
.venv/bin/maatml audit    [examples/<name>/] [--json]
.venv/bin/maatml plugins
.venv/bin/maatml rollout ~/models/my-policy --env line_world --tasks '{"count": 50, "seed": 1}' --label
# Global: maatml --debug <command> prints tracebacks for user errors
# (missing file, unparseable model.yml, unknown plugin), which otherwise
# print a single line.

# Multi-GPU (CUDA): HF Trainer / accelerate owns placement; rank-0 writes runs.jsonl
accelerate launch -m maatml.cli train examples/<name>/
# or:
torchrun --nproc_per_node=N -m maatml.cli train examples/<name>/

# Seed corpus regen (deterministic, validator-gated)
.venv/bin/python examples/support-ticket-triage/scripts/build_seeds.py
.venv/bin/python examples/vision/scripts/build_seeds.py --target 2000
.venv/bin/python examples/vision-vlm/scripts/build_seeds.py --target 300
.venv/bin/python examples/vision-describer/scripts/build_seeds.py --target 400
```

See [ROADMAP.md](ROADMAP.md) for the 0.13.0 release boundary, implemented
Unreleased work and planned tranches. See [docs/orchestration.md](docs/orchestration.md)
for worker and checkpoint ownership; external orchestration stays outside core.

## Architecture

MaatML is a **plugin-based training/fine-tuning framework** (experimentation →
production). Core owns architectures, registries, device profiles, generic
prepare/train/eval harnesses, and guards. **Examples own task semantics**: validators, metrics, predictors, tokenizers, generators, sanitizer rules, and
seed builders live under `examples/*/…_plugin/` and register via
`model.yml` `plugins:`.

| Location | Architecture | Base |
|---|---|---|
| `examples/support-ticket-triage/` | `causal_sft` (LoRA SFT) | Qwen3-0.6B |
| `examples/vision/` | `vision_multitask` (scene + detect + pose) | MobileNetV3-Large |
| `examples/vision-vlm/` | `vlm_sft` (SmolVLM LoRA; vLLM-servable) | SmolVLM-256M-Instruct |
| `examples/vision-describer/` | `seq2seq` (vision JSON → short description) | flan-t5-small |

Built-in architectures also include `dpo` / `orpo` (TRL preference; `maatml[pref]`).

**Standalone model folders.** Every path in `model.yml` resolves relative to
the model folder via `ModelDefinition.resolve()`. Core never falls back to
repo-relative paths. `pip install maatml` + `maatml scaffold ~/anywhere/...`
must work with no checkout present.

**Single source of truth.** `model.yml` carries `architecture`, `dataset:`,
`training:`, `smoke:`, `evaluation:`, `packaging:`, `plugins:`. Version is
semver (`0.1.0`); identity is `name@version`.

### Dispatch (registry)

```
architecture → trainer registry (causal_sft / seq2seq / multi_head_classifier / dpo / orpo / vlm_sft / …)
dataset.format → format registry (jsonl_seed / alpaca / sharegpt / preference_jsonl → prepare)
dataset.generator → generator registry (synthetic_scenes / described_scenes / custom → datagen)
evaluation.validator / metrics / predictor → plugin registries
export --format → exporter registry (safetensors / gguf / mlx / onnx)
compile --target → compiler registry (device-specific plans / packages)
serve --server → server registry (http default; DeepStream / vLLM / … plugins)
```

`model.yml` may also list `plugins: [./triage_plugin]` for folder-local packages.

### Device profiles (`maatml.device`)

- `mps`: no mid-train eval, workers=0, no grad checkpointing, fp32 master weights,
  `allow_quantized_load=False`
- `cuda`: mid-train eval on, workers>0, checkpointing allowed, native bf16/fp16
  master weights when `training.precision` matches, `allow_quantized_load=True`
- `cpu`: conservative defaults, no quantized load

QLoRA (`training.quantization:`) is CUDA-only; mps/cpu raise a hard error.
Optional `training.attn_implementation` and `training.dataloader_workers`.

### Guards (`maatml.training.guards`)

- `NanGuardCallback` aborts on non-finite loss/grad_norm
- `ensure_tokenizer_model_contract` + `training.embedding_strategy`
- `run_metadata.json` written into each checkpoint dir (rank-0 only when distributed)

### Out-of-model validators

Shared fence stripping / result types are in `maatml.validation.base`.
Task validators register from example plugins (e.g. `triage_plugin`, `vision_plugin`).

## Data flow

```
datasets/samples/seed_samples.jsonl  ← build_seeds / maatml datagen / ingest
        │
        │  maatml prepare
        ▼
output/prepared/{train,val,test}.jsonl
        │
        │  maatml train / sweep  →  output/runs.jsonl
        ▼
output/checkpoints/<run_id>/{*.safetensors, tokenizer, run_metadata.json, ...}
        │
        │  maatml evaluate [--gate]
        ▼
output/eval/<run>.{json,md}
        │
        │  maatml export [--parity] → output/export/<run_id>/ + manifest.json
        │  maatml verify <export-dir>
```

## Operational notes

- **Seed regen stamps `family`.** Group-aware splits hash `dataset.group_by`
  when set, else `family` → `source` → `sample_id`. Re-run builders after
  schema changes. Prefer `maatml datagen` when `dataset.generator` is set.
- **Checkpoint selection.** `resolve_checkpoint` prefers the latest completed
  run in `output/runs.jsonl`, then falls back to newest mtime under
  `output/checkpoints/`. Pass `--checkpoint <run_id|path>` explicitly when
  needed. Resume via `maatml train --resume auto|PATH`.
- **Export.** Default format is safetensors (works for all architectures).
  `gguf` / `mlx` are causal/preference-only and need external tooling
  (llama.cpp convert / `mlx_lm`). `training.model_revision` pins the HF
  revision for base model loads.
- **Overrides / sweep.** `--set training.learning_rate=1e-4` mutates the loaded
  model def; `maatml sweep` runs a cartesian product of `--param` axes (cap with
  `--max-trials`).
- **Tests** live under `tests/` (core) and `examples/*/tests/` (task plugins).
  Torch-gated tests (`pytest.importorskip("torch")`) run in the `ml-smoke` CI
  job; the torch-free matrix runs with `-rs` so those skips stay visible.
  `tests/conftest.py` snapshots and restores the plugin registries around every
  test through the public API (`snapshot_registries` / `restore_registries`).
