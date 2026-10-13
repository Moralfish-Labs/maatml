# The validator-gated lifecycle

MaatML's organizing idea is the **Maat weighing thesis**: a model's correctness
is judged *outside* the model, by a **validator** that checks its output against
a contract: a JSON schema, a grammar, node contracts, or task rules. The same
validator is reused at every stage. That reuse is what ties the lifecycle
together, and it is the main thing general fine-tuning tools leave to you.

## One validator, three jobs

```
                    ┌───────────────┐
                    │   validator   │   (per task, registered by a plugin)
                    └───────┬───────┘
          ┌─────────────────┼─────────────────┐
     gates data        gates eval        guards serving
          │                 │                 │
          ▼                 ▼                 ▼
       datagen           evaluate      serve ?validate=1
        ingest            --gate         /predict
```

- **Data:** `datagen` and `distill` validate accepted targets; `ingest` validates
  when `evaluation.validator` is configured. `datagen --allow-ungated` explicitly
  bypasses that requirement. A validator checks its declared contract; it does
  not establish that every accepted label is factually correct.
- **Evaluation:** `maatml evaluate` scores predictions and reports
  `evaluation.gates`; adding `--gate` exits non-zero on failure, so it drops straight
  into CI.
- **Serving:** `maatml serve` re-runs the *same* validator inline when a
  request hits `/predict?validate=1`. It annotates each response with the
  validator result by default, and with `--enforce` it rejects failing outputs
  (HTTP 422), so the contract can hold in production too.

The payoff: a MaatML model ships with a contract, not just weights.

## Registering a validator

A validator is a plugin registration in your model folder (see the
[plugin author guide](plugins.md)):

`ok` is not something you set. It is derived: a result is `ok` once every
required layer has passed, which is what makes a partial pass reportable rather
than a bare true/false.

```python
import json

from maatml.registry import register_validator
from maatml.validation.base import ValidationError, ValidationResult

@register_validator("my_task")
def validate_my_task(raw_output, *, schema_path=None, **kwargs):
    result = ValidationResult(raw_output=raw_output, required_layers={1, 2})

    # Layer 1: it parses.
    try:
        result.parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        result.errors.append(
            ValidationError(layer=1, code="invalid_json", message=str(exc))
        )
        return result
    result.passed_layers.add(1)

    # Layer 2: it satisfies the contract.
    if isinstance(result.parsed, dict) and result.parsed.get("answer"):
        result.passed_layers.add(2)
    else:
        result.errors.append(
            ValidationError(layer=2, code="missing_answer", message="answer required")
        )
    return result
```

Point `model.yml` at it:

```yaml
plugins: [./my_plugin]
evaluation:
  validator: my_task
  metrics: my_task
```

The `my_task` metrics registration is separate from the validator; implement it
as shown in the [plugin contracts](skill/references/plugins.md#metrics), then
derive gates from a measured report.

## Even free text can be gated

Gating is not limited to strict JSON. The
[`vision-vlm`](serving.md) example gates a natural-language image description
with *proxy* metrics (scene-mention rate, shape-mention F1, and brevity),
proving the pattern extends to open-ended output. The validator still decides
what "correct enough to ship" means; only the checks change.

## The lifecycle commands

```bash
maatml prepare   <model-dir>   # build train/val/test splits
maatml train     <model-dir>   # fine-tune (LoRA/QLoRA/full/DPO/ORPO/vision/VLM)
maatml evaluate  <model-dir> --gate   # score and enforce gates
maatml export    <model-dir>   # safetensors/gguf/mlx/onnx + manifest.json
maatml verify    <export-dir>  # sha256 check vs manifest
maatml serve     <model-dir>   # JSON inference API (validator inline)
maatml audit     [model-dir]   # environment + model-folder health check
```

Between evaluate and export sits the evidence the gate rests on: derived
floors, the ship verdict, the operating point, named populations, the run
record that travels. Those are lifecycle steps too; see [Evidence](evidence.md).

## One command for the whole lifecycle

An external orchestrator can invoke this command or individual stages on a
remote worker. Provisioning, transfers and recovery policy remain outside
maatml; see [orchestration and recovery](orchestration.md).

```bash
maatml run <model-dir>            # prepare -> train -> evaluate -> export -> verify
maatml run <model-dir> --smoke    # the same walk at the smoke tier
maatml run <model-dir> --dry-run  # what is stale, and why
```

`maatml run` walks the fixed pipeline in order and stops non-zero at the first
failure, so one green line means every stage passed rather than "the last
command I happened to type passed". Gates are enforced at the evaluate step,
which is what makes the line worth anything.

Steps that are already fresh are skipped. Freshness is decided by a fingerprint
over what the step consumed: effective config after `--smoke` and `--set`, the
declared input files, the upstream step's fingerprint, the maatml version and
git SHA, the plugin sources, the device profile, and the exporter. It lives in
`output/pipeline.json` and exists for **idempotence, not speed**: a step is
skipped only when its fingerprint matches, the step completed last time, and
its outputs are still there. `--dry-run` prints which component changed.

```
step      action   reason
prepare   skip     up to date
train     run      changed: training_config
evaluate  run      changed: upstream
```

The source operations (`datagen`, `distill`, `ingest`, `mint`, and reviewed
`serve --capture`) stay outside the runner. They change the seed corpus, and
that is precisely what makes `prepare` stale on the next run. See
[the data flywheel](flywheel.md) for how each one gates what it adds.

`--from` / `--until` restrict the walk, `--force` re-runs everything selected,
and `--set` overrides feed the fingerprint (an invalid one exits non-zero
before any step runs and leaves `output/pipeline.json` untouched).

### Smoke-tier gates

A `--smoke` rehearsal cannot meet production thresholds, and holding it to them
would only teach people to ignore a red gate. A `smoke:` block may declare its
own `gates:`, which `--smoke` runs enforce instead. Without that block,
evaluation falls back to the production gates. The pass is recorded as
smoke-gated in the run record and in the export manifest's `gate_evidence`, so
a rehearsal never reads as a production gate pass later.

maatml always reports `output_nonempty_rate` alongside a model's own metrics,
which is what a smoke tier can honestly gate on: it says the checkpoint saved,
reloaded, and produced output, without claiming the output was any good.

Every evaluate also reports **pathologies** (`never_fires`, `identical_output`,
`one_class`), and at the smoke tier each one is a failing gate of its own, so a
rehearsal floor loose enough for a broken model cannot be cleared by one.

`maatml plan <model-dir>` is the same view as `run --dry-run`.

<span id="tool-call-chat-training-unreleased"></span>

## Tool-call chat training

Causal SFT rows can carry `messages`, per-row `tools`, and
`chat_template_kwargs`. Assistant messages preserve `tool_calls`; tool-result
messages preserve their role, name and call ID. The template receives the same
tool definitions during training and prediction.

Every assistant turn is supervised, including tool calls. System messages,
user messages, tool results and the tool-definition preamble are masked from
the loss. The trainer locates assistant spans in the complete rendered chat
and checks their token decoding, including templates that render intermediate
assistant turns differently from the final turn. If a tool row cannot be
matched exactly, training rejects it rather than accepting shifted labels.

`scripts/check_chat_masks.py --help` describes the offline masking check using
a cached tokenizer. Tool-call chat training is available in 0.14.0 and later;
upgrade older installations before using it.

## What each stage refuses to do quietly

- `audit` is the read-only pre-flight: optional extras, the device the CLI
  would pick, registry contents, and (given a model folder) whether declared
  paths, registered plugins (`validator` / `metrics` / `predictor` /
  `generator`), seed corpora, gate keys vs the latest eval report, a
  quarantined `runs.jsonl.corrupt`, and QLoRA-on-non-CUDA conflicts are in
  place. Warnings stay exit 0; anything broken exits 1. `--json` is for
  machines.
- `prepare` splits by group key (`dataset.group_by`, else `family` → `source` →
  `sample_id`). A key covering nearly the whole corpus cannot be split, so
  those rows are split individually with a warning: group-level leakage
  protection does not apply to them. An empty split is reported, and a
  `benchmark_samples` row sharing a group key with the training splits is an
  error, because a benchmark is pinned to test. With `dataset.isolation`
  declared it refuses to write splits that violate the policy; with
  `dataset.attribution` declared it refuses a row whose source is unlisted,
  unsigned, blocked, or non-commercial without a signed risk acceptance; and
  an in-place edit of `benchmark_samples` is refused, because a benchmark is
  versioned by file.
- `distill` refuses a prompt pool that overlaps `benchmark_samples`,
  `blind_samples`, the prepared val / test split, or a pinned group, before
  any teacher call.
- `train` fails on gold labels no head declares, on a seq2seq corpus with no
  targets, and on an unsupported `training.precision`. Any failure marks the
  run `aborted` in `runs.jsonl`.
- `evaluate` uses `packaging.max_input_tokens` as its token budget (the same
  budget serve and `export --parity` enforce) and records how many inputs it
  truncated. Per-class output is a pass rate with its sample count, not a
  confusion matrix. Floors derived on another split are reported as a
  population mismatch (refused under `--strict-population`); `--blind` is
  refused on a candidate that has no current production gate pass or has
  already spent the blind population.
- `runs --adopt` refuses a bundle packed under another `model.yml` or identity,
  and any file whose hash differs from the bundle's manifest.
- `sweep` records a failed trial and keeps going, then exits non-zero. It ranks
  only trials that reported the metric being ranked.
