# MaatML

MaatML fine-tunes small, task-specific models across **text, vision, and
vision-language**, and takes them from experimentation to production through one
declarative `model.yml`: **prepare → train → evaluate → export → serve**.

**What makes it different:** correctness is checked *outside* the model by
[validators](lifecycle.md). The same validator gates your synthetic **data** and
your **evaluation**, and can guard your **live inference**: `maatml serve` runs
it per request on `/predict?validate=1`, and on every response under
`--enforce`, where a failing output is rejected with HTTP 422. So a MaatML model
ships with a contract, not just weights. That validator-gated
*data → eval → serving* loop, now across modalities, is what general
fine-tuning tools leave out.

**Install from PyPI** (not from source):

```bash
pip install maatml
pip install "maatml[ml]"       # training stack
pip install "maatml[ml,vision]"  # + torchvision and ONNX (vision / VLM examples)
pip install "maatml[docs]"     # this site: mkdocs serve
```

- Site: [maatml.pages.dev](https://maatml.pages.dev)
- PyPI: [pypi.org/project/maatml](https://pypi.org/project/maatml/)
- Source: [github.com/moralfish/maatml](https://github.com/moralfish/maatml)

## Documentation

- [Get started](getting-started.md): install, smoke-test, train and serve a model
- [The validator-gated lifecycle](lifecycle.md): the core idea, end to end
- [Evidence](evidence.md): derived floors, the ship decision, operating points, named populations, licences, portable runs
- [Orchestration and recovery](orchestration.md): CLI ownership, remote workers, checkpoint continuation and portable evidence
- [Rollouts](rollouts.md): environment verification and DAgger collection (unreleased)
- [Serving & deployment](serving.md): `maatml serve`, ONNX/edge, and vLLM (VLMs)
- [Plugin author guide](plugins.md): register trainers, validators, metrics, exporters, generators
- [Examples](examples/index.md): four reference models, from support-ticket triage to a vLLM-servable VLM

Canonical references in the repo:

- [README.md](https://github.com/moralfish/maatml/blob/main/README.md): install, CLI overview, examples
- [ROADMAP.md](https://github.com/moralfish/maatml/blob/main/ROADMAP.md): tranche status
- [CHANGELOG.md](https://github.com/moralfish/maatml/blob/main/CHANGELOG.md): release notes

## Quick CLI map

| Command | Purpose |
|---------|---------|
| `maatml prepare` | Build train/val/test splits |
| `maatml train` | Fine-tune (LoRA / QLoRA / full / DPO / ORPO / vision / VLM) |
| `maatml evaluate` | Validator + metrics + eval gates (`--gate` fails CI; `--cache`, `--blind`) |
| `maatml gates derive` / `ship-check` / `operating-point derive` | Floors from reports, the release verdict, the val-derived threshold ([evidence](evidence.md)) |
| `maatml runs` / `report` | The run registry (`--pack` / `--adopt` to move a run); the evidence document from the records alone |
| `maatml export` | Bundle checkpoint (safetensors / gguf / mlx / onnx) + `manifest.json` |
| `maatml verify` | Recompute sha256 of manifest files |
| `maatml serve` | JSON inference API, validator inline (`/predict?validate=1`) |
| `maatml datagen` / `distill` / `ingest` / `mint` | Validator-gated [data flywheel](flywheel.md) |
| `maatml run` / `plan` | Fixed lifecycle and freshness planning |
| `maatml rollout` (unreleased) | Environment-verified episodes and optional teacher labels |

See the [Plugin author guide](plugins.md) for registering generators, exporters,
validators, and metrics.

## Release boundary

The package version is **0.13.0**. These docs also cover unreleased source
features: rollouts, train-only rows, tool-call chat training, and corrected
multi-turn masking. Check the command's availability in your installed version;
the repository changelog distinguishes shipped releases from Unreleased work.
Model recipes and task plugins remain independent of the framework, while
external orchestrators manage compute, storage and monitoring.
