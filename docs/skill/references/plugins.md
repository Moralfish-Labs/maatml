# Plugins: teaching maatml your task

A model folder registers what is specific to its task and inherits everything
else. Registration happens in a folder-local package named in `model.yml`
(`plugins: [./my_plugin]`), loaded before any stage runs.

## Contents

- [The registries](#the-registries)
- [Writing a validator](#writing-a-validator)
- [Layers, and why `ok` is derived](#layers-and-why-ok-is-derived)
- [Metrics](#metrics)
- [Predictors](#predictors)
- [Generators](#generators)
- [Exporters, compilers, servers](#exporters-compilers-servers)
- [Testing a plugin](#testing-a-plugin)

## The registries

| Registry | Decorator | What it supplies |
|---|---|---|
| trainer | `@register_trainer` | An architecture's training loop |
| validator | `@register_validator` | The out-of-model contract gate |
| metrics | `@register_metrics` | Eval scoring |
| predictor | `@register_predictor` | Checkpoint to text or structured output |
| format | `@register_format` | Dataset prepare adapters |
| sanitizer | `@register_sanitizer` | PII / domain scrubbing |
| transform | `@register_transform` | Text pre-tokenization |
| generator | `@register_generator` | `maatml datagen` candidate factories |
| exporter | `@register_exporter` | `maatml export --format ...` |
| compiler | `@register_compiler` | `maatml compile --target ...` |
| server | `@register_server` | `maatml serve --server ...` |
| environment | `@register_environment` | Rollout state transitions and verification |
| scaffold hook | `@register_scaffold_hook` | Defaults for a plugin-owned architecture |

`maatml plugins` lists built-ins and discoverable installed plugins; `maatml audit <dir>` says whether
what `model.yml` names is actually there.

## Writing a validator

The validator is the reason to use maatml at all: it gates the synthetic data,
grades the evaluation, and can guard live inference, so one definition of
correct serves all three. Write it against the **contract**, not against the
model's current failure modes.

```python
import json

from maatml.registry import register_validator
from maatml.validation.base import ValidationError, ValidationResult


@register_validator("my_task")
def validate_my_task(raw_output, *, schema_path=None, user_prompt=None, **kwargs):
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
            ValidationError(
                layer=2,
                code="missing_answer",
                message="answer required",
                hint="return {\"answer\": ...}",
            )
        )
    return result
```

Every keyword is optional so one implementation satisfies all four call sites
(harness, serve, datagen, ingest). Accept `**kwargs` and read only what you
need.

`message` is not just for humans. Under `serve --enforce --max-retries N` it is
fed back to the model as the correction to make, so a message naming the exact
fault (`'stage' should be string`) repairs a reply where "invalid output" cannot.

**Derive the graded family from the gold target, never from the prediction.**
If a validator decides which contract applies by looking at what the model
produced, a broken answer falls into whichever family it accidentally resembles
and leaves the denominator of the metric that was supposed to catch it — so the
metric reads 0 of 0 at exactly the moment it stops working.

## Layers, and why `ok` is derived

`ok` is not something you set. A result is `ok` once every required layer has
passed — `required_layers`, or `n_layers` for "layers 1..n". That is what makes
a partial pass reportable: "parsed but broke the contract" is different from
"did not parse", and both are different from "passed", and a bare boolean loses
that.

Order layers the way a wrong answer goes wrong: shape, then parse, then the
domain contract, then the finer semantic checks. The first failing layer is the
one worth reporting.

## Metrics

A metrics function is called once with every evaluated row and returns
`{name: float}`. Each row is a `RowEval`: `.row` is the gold sample, `.gen_text`
the model's output, `.result` the `ValidationResult`, `.latency_ms` the time it
took.

```python
from maatml.registry import register_metrics


@register_metrics("my_task")
def metrics_my_task(rows):
    # The contract a row is held to comes from the GOLD sample, never from
    # what came back — see below for why that distinction decides whether the
    # metric can observe its own failure.
    graded = [r for r in rows if r.row.get("family") == "my_family"]
    passed = sum(1 for r in graded if r.result.ok)
    return {"my_family_pass_rate": passed / len(graded) if graded else 0.0}
```

Report each rate at **its own denominator** and name the metric after what it
measures. maatml adds `output_nonempty_rate` alongside whatever you return.

Two reserved keys are lifted out of `metrics` before the report is written.
`__counts__` is the evidence behind each rate — `{"my_family_pass_rate":
{"k": passed, "n": len(graded)}}` — and is what `maatml gates derive` floors
on. Without counts, an existing gate uses the observed value and labels it
`observed`; it has no derived uncertainty bound. `__pathologies__` is a list of
names or `{name, evidence}` dicts for output shapes no floor should have to
catch (a detector that never fires, one class for everything); they join the
harness's own `never_fires` / `identical_output` / `one_class` and fail the
smoke tier.

Resist collapsing everything into one aggregate. A pooled rate stays flat while
the composition underneath it moves, and it can read highest at the arm where a
safety metric is worst.

## Predictors

A predictor turns a checkpoint into output. Register one only when the
architecture's default does not fit — for example when the prompt must be
assembled from a `prompt_spec`, or when the served protocol differs from the
trainer's default chat template.

```python
from maatml.registry import register_predictor


@register_predictor("my_task")
class MyPredictor:
    def setup(self, checkpoint_dir, *, model_def, device, max_input_tokens,
              schema_path=None, contracts_path=None, prompt_spec_path=None):
        ...

    def predict(self, row):
        """Return the raw string the validator will judge."""

    def predict_batch(self, rows):
        """Optional. One string per row, same order; evaluation.batch_size > 1
        feeds chunks through this instead of predict, one device sync per chunk."""

    def report_extras(self):
        """Optional. Counts the report should carry, e.g. truncated_inputs."""
        return {}
```

A predictor whose output carries a score may also implement
`rescore(rows, threshold) -> dict[str, float]`: given the rows of a prediction
cache (`evaluate --cache`; each row has `row`, `output`, `parsed`, `ok`), return
the metrics that hold when predictions below `threshold` are dropped, with
`__counts__` for the rates. `maatml operating-point derive` sweeps it over a
val cache without running inference, so it must be callable on a freshly
instantiated predictor that never saw `setup()`.

`predict_batch(rows) -> list[str]` is how a GPU predictor stops idling during
`evaluate`: decode the chunk's inputs, run one forward, return one output per
row in order. With `evaluation.batch_size: N` (or `evaluate --batch-size N`,
a device knob rather than an override) the harness calls it in chunks of N;
outputs, validation, slices and the prediction cache are per row as before,
and the report's per-row latency is the chunk's time divided by its rows,
flagged `latency_amortized` in extras. A predictor without it runs one row at
a time whatever the batch size, with a warning; a chunk that returns the
wrong number of outputs is an error, never padded.

Whatever the predictor does to raw output — repairing braces, stripping fences —
belongs in `report_extras` too, so the report says the pass rate includes a
repair rather than hiding it.

## Generators

For `maatml datagen`: produce candidate rows, and let the validator decide which
survive.

```python
from maatml.registry import register_generator


@register_generator("my_task")
def my_generator(model_def, *, seed=0, **kwargs):
    counter = 0

    def generate_one():
        nonlocal counter
        counter += 1
        # Replace with task-specific generation; match dataset field names.
        return {"sample_id": f"sample-{seed}-{counter}",
                "request": "Example question", "expected": {"answer": "Example"}}

    return generate_one
```

`datagen` fails closed when no validator is configured, unless you pass
`--allow-ungated`. Ungated synthetic data is the fastest way to teach a model a
contract nobody checks.

## Exporters, compilers, servers

```python
from maatml.registry import register_exporter


@register_exporter("my_fmt")
def export_my_fmt(model_def, checkpoint_dir, out_dir, *, run_id=None):
    """Write artifacts and manifest.json into out_dir; return that directory."""
    ...  # conversion and manifest creation are the exporter's responsibility
    return out_dir
```

Everything written must end up in `manifest.json` so `maatml verify` can check
it. If a bundle gains a file out of band — a quantized GGUF produced separately
— add it with `maatml manifest amend <export-dir> <file> --format gguf` rather
than leaving it unlisted.

A backend that accepts `**kwargs` it does not honour will silently swallow flags
like `--enforce`. If a flag is meant to apply to your backend, consume it
explicitly and say so in the startup banner, so an operator can see the mode
they asked for. `--capture` is one of those flags: call
`maatml.serve.open_capture` and `LifecycleServer.record_capture` rather than
ignoring `capture_path`. `maatml compile --require-gated` is the matching
compiler claim: refuse an ungated or smoke-gated export before the plugin runs.

## Testing a plugin

Plugins are ordinary Python: import the module, call the validator with a
handful of known-good and known-bad strings, and assert on layers rather than on
`ok` alone.

```python
def test_a_broken_call_fails_at_the_layer_that_checks_it():
    result = validate_my_task('{"answer": null}')
    assert not result.ok
    assert 1 in result.passed_layers          # it parsed
    assert result.errors[0].code == "missing_answer"
```

Test the validator against **real artifacts the task has produced** — approved
outputs should pass and rejected ones should fail. A validator that has never
seen a real success is a validator nobody has calibrated.
