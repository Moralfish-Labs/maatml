# Plugin author guide

MaatML core owns architectures and harnesses. **Examples / model folders own
task semantics** and register them via decorators.

## Registries

| Kind | Decorator | Typical use |
|------|-----------|-------------|
| trainer | `@register_trainer` | Architecture training loop |
| validator | `@register_validator` | Out-of-model JSON / contract gate |
| metrics | `@register_metrics` | Eval scoring |
| predictor | `@register_predictor` | Checkpoint → text / structured output; optional `predict_batch(rows)` for batched evaluate |
| format | `@register_format` | Dataset prepare adapters |
| sanitizer | `@register_sanitizer` | Regex PII / domain scrubbing |
| transform | `@register_transform` | Text pre-tokenization |
| generator | `@register_generator` | `maatml datagen` candidate factories |
| exporter | `@register_exporter` | `maatml export --format …` |
| compiler | `@register_compiler` | `maatml compile --target …` (TensorRT, vLLM package, GGUF quantize, …) |
| server | `@register_server` | `maatml serve --server …` (HTTP, DeepStream, vLLM, llama.cpp, …) |
| environment | `@register_environment` | Legal candidates, teacher labels, state transitions and independent verification for [rollouts](rollouts.md) |

`maatml plugins` lists built-ins and discoverable installed plugins; it does
not load an arbitrary model folder. Use `maatml audit <model-dir>` to check
that folder's registrations. Compilers and servers are **format-agnostic**:
core dispatches and writes a thin `target_manifest.json`; the plugin owns the
engine. Do not assume ONNX or an in-process predictor.

## Folder-local plugins

In `model.yml`:

```yaml
plugins: [./triage_plugin]
evaluation:
  validator: triage
  metrics: triage
  predictor: causal_sft
```

`load_model_plugins` imports the package (or `.py` file); side-effect
registrations run at import time. It is the single owner of that import and is
idempotent, so a folder's plugin code runs once per process no matter how many
commands (or library calls) ask for it. Pass `force=True` to re-execute it.

A plugin source that fails to import is recorded rather than skipped in
silence: `maatml plugins` lists the failures under `unavailable`, and an
`Unknown … plugin` error names them, which is usually why a name is missing.

`jsonschema` is a core maatml dependency for this reason: `dataset.schema` is a
JSON Schema document and every shipped validator calls `jsonschema.validate`, so
the documented validator shape works without a second install step. Core itself
does not import it.

> **Trust boundary.** These imports run arbitrary Python at load time. Because
> every command reads `model.yml`, even `maatml validate` and `maatml plan`
> execute a folder's plugins. Only point maatml at model folders you trust, or
> use `maatml validate --no-plugins` to check schema and paths without importing
> plugin code.

## Per-prediction evidence

A predictor may implement `prediction_context(row) -> dict` to expose host-side
observations, such as retrieval receipts, separately from its generated text.
The evaluation harness calls it immediately after `predict(row)` and snapshots
finite JSON values. This hook requires `batch_size: 1`; a batched predictor with
the hook is rejected before generation because a single last-result trace cannot
unambiguously describe a batch.

A validator explicitly opts in with a named `prediction_context` parameter:

```python
def validate(text, *, user_prompt=None, schema_path=None, prediction_context=None):
    # Validate the answer against observed evidence, not row-supplied claims.
    ...
```

Existing validator signatures remain supported. Accepting `**kwargs` alone does
not opt in. The validator receives its own copy; mutations do not overwrite the
snapshot available to metrics as `RowEval.prediction_context` or saved beside
the output in `evaluate --cache`. This top-level cache field is distinct from
any similarly named field inside the original dataset row. Missing context is
an empty dictionary, not proof of successful retrieval. Task plugins remain
responsible for checking receipt identity, grounding and answer correctness.

## Generators (`maatml datagen`)

A generator is a factory:

```python
from maatml.registry import register_generator

@register_generator("my_task")
def my_generator(model_def, *, seed: int = 0, **kwargs):
    def generate_fn():
        return {"sample_id": "...", "request": "...", "expected": {...}}
    return generate_fn
```

Match the row keys to `dataset.request_field` and `dataset.target_field`;
use a distinct sample ID for each candidate.

Core runs `build_gated_corpus(generate_fn, validate_fn, target_n=…)` and
appends accepted rows to `dataset.seed_samples`, skipping rows already in the
corpus (matched by `sample_id` or content). Returning `None` means "no
candidate this time" and costs one attempt; raising is recorded as a rejected
row. Raise `maatml.data.gated.GenerationAbort` to stop the run immediately
(the teacher client does this after five consecutive request failures).

Stamp a `family` (or whatever `dataset.group_by` names) on generated rows.
Rows that share one group key cannot be split, so a corpus where every row
carries the same key is split per row with a warning instead.

Optional teacher path: `maatml datagen --teacher` uses
`MAATML_TEACHER_BASE_URL` / `MAATML_TEACHER_API_KEY` (`pip install maatml[teacher]`).
`MAATML_TEACHER_BASE_URL` is required and has no default, because your prompts
are sent to whatever it names; point it at a local server
(`http://127.0.0.1:8000/v1`) or at a hosted endpoint deliberately.

## Scaffolding a plugin-owned architecture

Core cannot scaffold an architecture it has never heard of, so point
`maatml scaffold` at the plugin that owns it:

```bash
maatml scaffold ~/models/my-vision --architecture vision_multitask \
    --plugin examples/vision/vision_plugin
```

`--plugin` (repeatable; a folder, a `.py` file, or an installed module) is
loaded before the architecture is resolved and recorded in the new `model.yml`,
so every later command finds the trainer too.

A `@register_scaffold_hook(<architecture>)` supplies the defaults core cannot
guess. It may return a mapping with any of:

| Key | Effect |
|-----|--------|
| `model_yml` | Top-level sections that **replace** core's defaults (`dataset`, `training`, `evaluation`, …) |
| `seed_rows` | Rows written to `seed_samples.jsonl`; `[]` means "this corpus is generated" |
| `files` | Extra files, keyed by path relative to the model folder |

Core stays the only writer, so a hook cannot half-create a folder. Ship the
same schema and prompt spec your validator and generator were written against:
a lookalike copy makes `maatml datagen` reject every row it generates.

## Processor-aware SFT trainers

A plugin-owned trainer can reuse `maatml.training.sft_base.train_sft` with
`data_collator_factory=MyCollator`. The factory receives `(model_def, processor,
training_config)` and returns a callable over raw prepared rows. This opts into
revision-pinned `AutoProcessor` and `AutoModelForImageTextToText` loading. The
plugin owns media decoding, asset hash verification, assistant loss masks,
padding and expanded-token budget checks. Media rows bypass the text token cache.

The shared Trainer still owns optimization, resume, callbacks and run records.
It receives the processor as `processing_class`, preserving processor assets in
intermediate and final saves. `training.frozen_module_prefixes` optionally names
model submodules that must remain frozen: trainable parameters or changed tensor
bytes, including persistent buffers, abort the run. Hashes are recorded in final
run metadata. Submodule paths must exist before LoRA is attached; freezing and
correct target selection remain the trainer's responsibility.

A plugin-owned architecture can register an exporter name that delegates to a
shared exporter, as other plugin formats do. Register an explicit name for GGUF;
the built-in `gguf` format remains restricted to its supported architectures.

## Exporters (`maatml export`)

Built-ins: `safetensors`, `gguf` / `mlx` (optional tooling). The vision example
registers `onnx` / `vision_onnx` from its plugin. Custom:

```python
from maatml.registry import register_exporter


@register_exporter("my_fmt")
def export_my_fmt(model_def, checkpoint_dir, out_dir, *, run_id=None):
    ...
    return out_dir
```

Always write / update `manifest.json` via `maatml.export.manifest`.

## Compilers and servers

`maatml compile --require-gated` refuses an export whose `gate_evidence` is
missing, failed, or `smoke_gated`. Every compile writes `promotion_eligible`
and `promotion_reason` on `target_manifest.json`.

A long-lived backend should use `LifecycleServer`. To join the reviewed
flywheel, call `maatml.serve.open_capture` with the `capture_path` /
`auth_token` kwargs `dispatch_server` already forwards, attach the writer as
`LifecycleServer.capture`, and `record_capture(row, output, raw)` on each
prediction. Ingest then sees the same `source: serve_capture` rows the HTTP
server writes.

## Testing plugins

Registries are process-global. Snapshot and restore them around a test through
the public API instead of touching registry internals:

```python
from maatml.registry import restore_registries, snapshot_registries

snapshot = snapshot_registries()
try:
    ...  # register, load a model folder, assert
finally:
    restore_registries(snapshot)
```

`reset_registries()` wipes every registry (and forgets which model-folder
plugins have run) for a blank slate; `reset_registries(rediscover=True)`
re-imports the built-ins afterwards. `REGISTRY.unregister(name)` drops one
entry. `discover_plugins()` only adds registrations, so it never removes what a
model folder registered.

For validator, metrics, predictor and generator implementations, see the
[plugin contracts reference](skill/references/plugins.md).

## Deprecation policy

- Semver for the `maatml` package; model folders version independently in
  `model.yml`.
- Registry names are sticky once published in an example; rename with a
  temporary dual-register period.
- CLI flags may gain aliases; removals land in a minor with a CHANGELOG note.
