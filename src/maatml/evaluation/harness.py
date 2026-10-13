"""Shared evaluation harness used by task evaluators and the CLI.

Predictors emit raw text; validators gate structure; metrics plugins score
semantics. Asset paths (schema, contracts, prompt_spec, tokenizer) resolve
from ``model_def`` / explicit kwargs / ``checkpoint_dir``: never from a
hardcoded repo-relative fallback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

from ..config import ModelDefinition, get_dataset_cfg
from ..device import resolve_device
from ..registry import METRICS, PREDICTORS, VALIDATORS, discover_plugins
from ..utils.io import iter_jsonl, read_json, sha256_file, write_json
from ..validation.base import ValidationResult
from .pathologies import PATHOLOGIES_KEY, detect_pathologies, normalize_plugin_pathologies
from .predictions import prediction_row, predictions_path, write_predictions
from .stats import wilson_lower

console = Console()


class LatencyStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    p50: float
    p95: float
    mean: float
    n: int


# Bumped when a field a consumer relies on changes shape. A report that does
# not carry the key predates the versioned schema and reads back as 0.
REPORT_VERSION = 1
_REPORT_REQUIRED = ("report_version", "model_id", "dataset", "n", "metrics")


class ReportSchemaError(ValueError):
    """A report is missing a field this schema version requires."""


def validate_report_payload(payload: Any, *, strict: bool = False) -> dict[str, Any]:
    """Check a raw report object before it becomes a :class:`Report`.

    Lenient by default so reports written before ``report_version`` still read
    (as version 0). ``strict`` is for consumers that derive from a report
    (floors, sweeps): a missing field there is a wrong number later, not a
    missing key now.
    """
    if not isinstance(payload, dict):
        raise ReportSchemaError("report is not a JSON object")
    if "report_version" not in payload:
        payload = {**payload, "report_version": 0}
    if strict:
        missing = [key for key in _REPORT_REQUIRED if key not in payload]
        if missing:
            raise ReportSchemaError(
                f"report is missing required field(s) {missing}; "
                f"re-run evaluate with maatml >= report_version {REPORT_VERSION}"
            )
        if int(payload["report_version"]) < 1:
            raise ReportSchemaError(
                "report predates report_version 1 (no version field); "
                "re-run evaluate before deriving from it"
            )
    return payload


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_version: int = REPORT_VERSION
    model_id: str = ""
    name: str = ""
    version: str = ""
    task: str = ""
    dataset: str = ""
    n: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)
    # evaluation.slices: field -> value -> {n, passed, pass_rate, pass_rate_w95}
    slices: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    # Numerator / denominator behind every rate a floor can be derived from:
    # metric name -> {k, n}. Harness rates and slices always; plugin rates when
    # the plugin reports ``__counts__``.
    counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    # Named output shapes no floor should have to catch: {name, evidence}.
    # Reported always; a non-empty list fails the smoke tier.
    pathologies: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: Optional[LatencyStats] = None
    baseline_delta: Optional[dict[str, float]] = None
    sample_failures: list[dict] = Field(default_factory=list)
    extras: dict[str, Any] = Field(default_factory=dict)
    # Eval gates (evaluation.gates + --gate)
    gates: Optional[dict[str, Any]] = None
    passed: Optional[bool] = None

    def write(self, path: str | Path) -> Path:
        return write_json(path, self.model_dump(mode="json"))

    @classmethod
    def read(cls, path: str | Path, *, strict: bool = False) -> "Report":
        payload = validate_report_payload(read_json(path), strict=strict)
        return cls.model_validate(payload)


class GateConfigError(ValueError):
    """Raised when gate enforcement is requested but no gates are configured."""


def effective_gates(model_def: Any, *, smoke: bool = False) -> dict[str, float]:
    """Gate minima for this run, empty when none are configured.

    A ``smoke:`` overlay may declare its own ``gates:``. A smoke run is a
    rehearsal on a fraction of the data, so holding it to the production
    thresholds would only teach people to ignore the gate; a smoke tier keeps
    the gate meaningful at that budget. Runs gated this way are marked
    smoke-gated everywhere they are recorded, so they never read as a real
    gate pass.
    """
    evaluation = getattr(model_def, "evaluation", None)
    spec = evaluation.get("gates") if isinstance(evaluation, dict) else None
    if smoke:
        smoke_section = getattr(model_def, "smoke", None)
        smoke_spec = smoke_section.get("gates") if isinstance(smoke_section, dict) else None
        if isinstance(smoke_spec, dict) and smoke_spec:
            spec = smoke_spec
    if not (isinstance(spec, dict) and spec):
        return {}
    out: dict[str, float] = {}
    for key, value in spec.items():
        minimum = value.get("min") if isinstance(value, dict) else value
        try:
            if minimum is None:
                raise TypeError("missing min")
            out[str(key)] = float(minimum)
        except (TypeError, ValueError) as exc:
            raise GateConfigError(
                f"evaluation.gates[{key!r}] must be a number or {{min, tier}}; got {value!r}"
            ) from exc
    return out


GATE_TIERS = ("blocking", "advisory")


def gate_tiers(model_def: Any, *, smoke: bool = False) -> dict[str, str]:
    """Tier per gate: ``blocking`` (default) fails the step, ``advisory`` is recorded."""
    evaluation = getattr(model_def, "evaluation", None)
    spec = evaluation.get("gates") if isinstance(evaluation, dict) else None
    if smoke:
        smoke_section = getattr(model_def, "smoke", None)
        smoke_spec = smoke_section.get("gates") if isinstance(smoke_section, dict) else None
        if isinstance(smoke_spec, dict) and smoke_spec:
            spec = smoke_spec
    if not (isinstance(spec, dict) and spec):
        return {}
    out: dict[str, str] = {}
    for key, value in spec.items():
        tier = value.get("tier", "blocking") if isinstance(value, dict) else "blocking"
        if tier not in GATE_TIERS:
            raise GateConfigError(
                f"evaluation.gates[{key!r}].tier must be one of {GATE_TIERS}; got {tier!r}"
            )
        out[str(key)] = str(tier)
    return out


def uses_smoke_gates(model_def: Any) -> bool:
    """Does the ``smoke:`` overlay declare its own gates?"""
    smoke_section = getattr(model_def, "smoke", None)
    spec = smoke_section.get("gates") if isinstance(smoke_section, dict) else None
    return isinstance(spec, dict) and bool(spec)


def resolve_gate_spec(model_def: Any, *, smoke: bool = False) -> dict[str, float]:
    """Return the configured gate minima, or raise if none are set.

    ``evaluate --gate`` (and ``enforce_gates=True``) must not pass vacuously: a
    model with no ``evaluation.gates`` has nothing to enforce, so requesting
    enforcement against an empty spec is a configuration error rather than a
    silent success.
    """
    gates = effective_gates(model_def, smoke=smoke)
    if not gates:
        raise GateConfigError(
            "gate enforcement requested but no evaluation.gates are configured. "
            "Add a gates: block to model.yml (see any example) or drop --gate."
        )
    return gates


SLICE_GATE_PREFIX = "slice:"


def parse_slice_ref(name: str) -> Optional[tuple[str, str]]:
    """``slice:<field>=<value>`` -> ``(field, value)``; anything else -> None."""
    if not name.startswith(SLICE_GATE_PREFIX) or "=" not in name:
        return None
    field_name, value = name[len(SLICE_GATE_PREFIX) :].split("=", 1)
    return field_name, value


def gate_actual(
    name: str,
    metrics: dict[str, float],
    slices: Optional[dict[str, dict[str, dict[str, float]]]] = None,
) -> Optional[float]:
    """The value a gate compares against: a metric, or a slice's pass rate.

    A slice with no rows has no rate, so its gate reads ``None`` and fails
    rather than passing on an invented 0.0 or 1.0.
    """
    if name in metrics:
        return float(metrics[name])
    ref = parse_slice_ref(name)
    if ref is None or not slices:
        return None
    field_name, value = ref
    stats = (slices.get(field_name) or {}).get(value)
    if not stats or not stats.get("n"):
        return None
    return float(stats["pass_rate"])


def check_gates(
    metrics: dict[str, float],
    gates: dict[str, float],
    *,
    tiers: Optional[dict[str, str]] = None,
    slices: Optional[dict[str, dict[str, dict[str, float]]]] = None,
) -> dict[str, Any]:
    """Compare metrics against minimum thresholds.

    Returns ``passed`` (every *blocking* gate met), ``results`` mapping each
    gate to ``{minimum, actual, passed, tier}``, and ``advisory_failed``: the
    advisory gates that missed, recorded but never fatal.
    """
    results: dict[str, dict[str, Any]] = {}
    all_ok = True
    advisory_failed: list[str] = []
    for name, minimum in gates.items():
        actual = gate_actual(name, metrics, slices)
        ok = actual is not None and float(actual) >= float(minimum)
        tier = (tiers or {}).get(name, "blocking")
        results[name] = {
            "minimum": float(minimum),
            "actual": None if actual is None else float(actual),
            "passed": ok,
            "tier": tier,
        }
        if not ok:
            if tier == "advisory":
                advisory_failed.append(name)
            else:
                all_ok = False
    return {"passed": all_ok, "results": results, "advisory_failed": advisory_failed}


@dataclass
class RowEval:
    """One evaluated row: gold sample, model text, validator outcome."""

    row: dict
    gen_text: str
    result: ValidationResult
    latency_ms: float = 0.0


@dataclass
class _EvalCtx:
    schema_path: Optional[Path] = None
    contracts_path: Optional[Path] = None
    prompt_spec_path: Optional[Path] = None
    extras: dict[str, Any] = field(default_factory=dict)


PredictorLike = Union[str, Any]
ValidatorLike = Union[str, Callable[..., ValidationResult]]
MetricsEntry = Union[str, Callable[..., dict[str, float]]]
# ``evaluation.metrics`` may name one metrics plugin or a list of them.
MetricsLike = Union[MetricsEntry, Sequence[MetricsEntry]]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def latency_stats(samples_ms: list[float]) -> LatencyStats:
    return LatencyStats(
        p50=percentile(samples_ms, 0.5),
        p95=percentile(samples_ms, 0.95),
        mean=sum(samples_ms) / len(samples_ms) if samples_ms else 0.0,
        n=len(samples_ms),
    )


def binary_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "support": float(tp + fn)}


def per_class_prf(
    true: list[str], pred: list[str], labels: list[str]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(true, pred, strict=False) if t == label and p == label)
        fp = sum(1 for t, p in zip(true, pred, strict=False) if t != label and p == label)
        fn = sum(1 for t, p in zip(true, pred, strict=False) if t == label and p != label)
        out[label] = binary_prf(tp, fp, fn)
    return out


def baseline_delta(
    metrics: dict[str, float], baseline_path: Optional[str | Path]
) -> Optional[dict[str, float]]:
    if not baseline_path:
        return None
    base = Report.read(baseline_path)
    delta: dict[str, float] = {}
    for k, v in metrics.items():
        if k in base.metrics:
            delta[k] = v - base.metrics[k]
    return delta


def regression_failures(
    delta: dict[str, float],
    gate_keys: set[str],
    default_max: Optional[float],
    overrides: dict[str, float],
) -> list[str]:
    """Deltas that fall further than the caller allows, formatted for a report.

    The default ceiling applies to gated metrics only: they are the rates the
    model is accountable for, and they share a direction (higher is better).
    Ungated keys like ``eval_loss`` improve downward, so judging them by the
    same rule would flag improvements; they participate only when named in an
    override.
    """
    failures: list[str] = []
    for metric, change in sorted(delta.items()):
        limit = overrides.get(metric)
        if limit is None and metric in gate_keys:
            limit = default_max
        if limit is None:
            continue
        if change < -limit:
            failures.append(f"{metric}: {change:+.4f} (allowed -{limit:g})")
    return failures


def default_eval_keys(
    model_def: ModelDefinition,
) -> tuple[Optional[str], Optional[str], Any]:
    """Infer predictor / validator / metrics from ``evaluation:`` or architecture.

    ``evaluation.metrics`` may be a single name or a list; every entry runs and
    the results are merged (the harness rejects two plugins claiming the same
    metric key), so a list is never truncated to its first entry. Validator and
    metrics come from ``evaluation:`` or the model's plugins; core keeps no
    hardcoded task-name fallbacks.
    """
    from ..scaffold import normalize_architecture

    ev = model_def.evaluation or {}
    predictor = ev.get("predictor")
    validator = ev.get("validator")
    metrics = ev.get("metrics")
    if isinstance(metrics, list) and not metrics:
        metrics = None

    arch = normalize_architecture(model_def.architecture)
    if predictor is None:
        if arch in PREDICTORS.names() or PREDICTORS.get(model_def.architecture):
            predictor = model_def.architecture if PREDICTORS.get(model_def.architecture) else arch
        elif arch in ("multi_head_classifier", "seq2seq", "causal_sft"):
            predictor = arch

    return predictor, validator, metrics


class DeclaredAssetMissing(FileNotFoundError):
    """An asset the caller named explicitly, or model.yml declares, is absent.

    Distinct from a plain FileNotFoundError, which only means an *optional*
    asset could not be discovered. Callers treat the optional case as "no
    asset" and must not do the same here: a typo in ``dataset.contracts``
    would otherwise resolve to None and surface much later as an opaque
    TypeError from the validator, on the first evaluated row.
    """


def resolve_eval_asset(
    key: str,
    *,
    model_def: Optional[ModelDefinition] = None,
    checkpoint_dir: Path,
    filenames: Sequence[str] = (),
    explicit: Optional[str | Path] = None,
) -> Path:
    """Resolve schema/contracts/prompt_spec/tokenizer without repo fallbacks.

    Order: explicit path → ``model_def`` dataset/data key → file under checkpoint.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise DeclaredAssetMissing(f"{key} not found at explicit path: {path}")
        return path.resolve()

    if model_def is not None:
        cfg = get_dataset_cfg(model_def)
        if key in cfg and isinstance(cfg[key], str):
            path = model_def.resolve(cfg[key])
            if not path.is_file():
                raise DeclaredAssetMissing(
                    f"model.yml declares {key}={cfg[key]!r} but file missing: {path}"
                )
            return path

    checkpoint_dir = Path(checkpoint_dir)
    for name in filenames:
        path = checkpoint_dir / name
        if path.is_file():
            return path.resolve()

    hints = list(filenames) if filenames else ["(none)"]
    raise FileNotFoundError(
        f"Could not resolve {key!r}. Provide model_def with data/dataset.{key}, "
        f"pass an explicit path, or place one of {hints} under {checkpoint_dir}."
    )


def _resolve_callable(kind: str, value: Any, registry) -> Any:
    if isinstance(value, str):
        return registry.require(value)
    if value is None:
        raise KeyError(f"No {kind} provided")
    return value


def _noop_validate(
    raw_output: str,
    *,
    schema_path: str | Path | None = None,
    contracts_path: str | Path | None = None,
    user_prompt: Optional[str] = None,
    strip_fences: bool = True,
) -> ValidationResult:
    del schema_path, contracts_path, user_prompt, strip_fences
    import json

    from ..validation.base import strip_fences as _strip

    text = _strip(raw_output)
    result = ValidationResult(raw_output=raw_output, n_layers=1, required_layers={1})
    try:
        result.parsed = json.loads(text)
        result.passed_layers.add(1)
    except json.JSONDecodeError as exc:
        from ..validation.base import ValidationError

        result.errors.append(ValidationError(layer=1, code="invalid_json", message=str(exc)))
    return result


def resolve_validator(validator: Any) -> Callable[..., ValidationResult]:
    """Resolve a configured validator, or fall back to JSON-parse-only.

    ``validator=None`` is the ONLY path to the no-contract ``_noop_validate``
    scorer (a causal-SFT model that declares no validator). A configured name
    that does not resolve to a registered validator is a configuration error,
    not a reason to silently degrade to bare JSON parsing.
    """
    if validator is None:
        return _noop_validate
    if isinstance(validator, str) and VALIDATORS.get(validator) is None:
        known = ", ".join(VALIDATORS.names()) or "(none)"
        raise GateConfigError(
            f"evaluation.validator={validator!r} does not resolve to a registered "
            f"validator (known: {known}). Fix the plugins: list in model.yml, or "
            "remove evaluation.validator to score JSON-parse-only."
        )
    return _resolve_callable("validator", validator, VALIDATORS)


def _category_buckets(row_results: list[RowEval]) -> dict[str, dict[str, float]]:
    """Validator pass rate per ``category``.

    This is a pass/fail bucket count, not a classification confusion matrix:
    there is no per-category prediction to compare against a gold category, so
    the report carries ``pass_rate`` / ``n`` and nothing else. Real per-class
    P/R/F1 comes from :func:`per_class_prf`, which metrics plugins call with
    actual (true, predicted) label pairs.
    """
    per_category: dict[str, dict[str, int]] = {}
    for item in row_results:
        category = str(item.row.get("category") or "unknown")
        bucket = per_category.setdefault(category, {"n": 0, "passed_all": 0})
        bucket["n"] += 1
        if item.result.ok:
            bucket["passed_all"] += 1
    return {
        cat: {
            "pass_rate": b["passed_all"] / max(1, b["n"]),
            "passed": float(b["passed_all"]),
            "n": float(b["n"]),
        }
        for cat, b in per_category.items()
    }


@dataclass(frozen=True)
class SliceSpec:
    """One ``evaluation.slices`` entry: a row field, optionally with declared values."""

    field: str
    values: Optional[tuple[str, ...]] = None


SLICE_ABSENT = "(absent)"


def resolve_slices(model_def: Any) -> list[SliceSpec]:
    """Parse ``evaluation.slices``: a list of field names or ``{field, values}``."""
    evaluation = getattr(model_def, "evaluation", None)
    spec = evaluation.get("slices") if isinstance(evaluation, dict) else None
    if spec is None:
        return []
    if not isinstance(spec, list):
        raise GateConfigError("evaluation.slices must be a list of field names or {field, values}")
    out: list[SliceSpec] = []
    for entry in spec:
        if isinstance(entry, str) and entry.strip():
            out.append(SliceSpec(field=entry.strip()))
            continue
        if isinstance(entry, dict) and isinstance(entry.get("field"), str):
            values = entry.get("values")
            if values is not None and not isinstance(values, list):
                raise GateConfigError(
                    f"evaluation.slices[{entry['field']!r}].values must be a list"
                )
            out.append(
                SliceSpec(
                    field=entry["field"].strip(),
                    values=tuple(str(v) for v in values) if values is not None else None,
                )
            )
            continue
        raise GateConfigError(
            f"evaluation.slices entry {entry!r} is not a field name or {{field, values}}"
        )
    seen: set[str] = set()
    for item in out:
        if item.field in seen:
            raise GateConfigError(f"evaluation.slices names {item.field!r} twice")
        seen.add(item.field)
    return out


def slice_buckets(
    row_results: list[RowEval], specs: Sequence[SliceSpec]
) -> dict[str, dict[str, dict[str, float]]]:
    """Validator pass rate per value of each declared row field.

    A value with rows reports ``n`` / ``passed`` / ``pass_rate`` and the Wilson
    95 % lower bound; a declared value with no rows reports ``n: 0`` and no
    rate, so an empty slice can never read as ``0.0``. Rows lacking the field
    land under ``(absent)`` rather than vanishing from the denominator.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for spec in specs:
        counts: dict[str, dict[str, int]] = {}
        if spec.values:
            for value in spec.values:
                counts[value] = {"n": 0, "passed": 0}
        for item in row_results:
            raw = item.row.get(spec.field)
            value = SLICE_ABSENT if raw is None else str(raw)
            bucket = counts.setdefault(value, {"n": 0, "passed": 0})
            bucket["n"] += 1
            if item.result.ok:
                bucket["passed"] += 1
        stats: dict[str, dict[str, float]] = {}
        for value, bucket in counts.items():
            n = bucket["n"]
            if n == 0:
                stats[value] = {"n": 0.0}
                continue
            stats[value] = {
                "n": float(n),
                "passed": float(bucket["passed"]),
                "pass_rate": bucket["passed"] / n,
                "pass_rate_w95": wilson_lower(bucket["passed"], n),
            }
        out[spec.field] = stats
    return out


def _resolve_metrics(metrics_fn: Any) -> list[Callable[..., dict[str, float]]]:
    """Resolve one metrics entry, or a list of them, to callables.

    ``evaluation.metrics`` may be a single name or a list. Every entry runs and
    the results are merged; two entries claiming the same metric key is a
    configuration error rather than a silent last-writer-wins.
    """
    if metrics_fn is None:
        return []
    entries = list(metrics_fn) if isinstance(metrics_fn, (list, tuple)) else [metrics_fn]
    return [_resolve_callable("metrics", entry, METRICS) for entry in entries]


# Always computed, whatever metrics plugin runs: it measures the harness end of
# the contract (did the checkpoint load and emit anything at all?) rather than
# the model's quality, which is what a smoke tier can gate on honestly.
COVERAGE_METRIC = "output_nonempty_rate"


def coverage_metrics(row_results: list[RowEval]) -> dict[str, float]:
    """Built-in metrics no plugin owns: prediction coverage."""
    n = len(row_results)
    if not n:
        return {COVERAGE_METRIC: 0.0}
    nonempty = sum(1 for item in row_results if item.gen_text and item.gen_text.strip())
    return {COVERAGE_METRIC: nonempty / n}


COUNTS_KEY = "__counts__"


def _merge_metrics(
    callables: list[Callable[..., dict[str, float]]],
    row_results: list[RowEval],
    names: list[str],
    *,
    counts_out: Optional[dict[str, dict[str, int]]] = None,
    pathologies_out: Optional[list[dict[str, Any]]] = None,
) -> dict[str, float]:
    """Run every metrics plugin and merge; ``__counts__`` / ``__pathologies__`` are lifted out.

    A plugin reports the evidence behind a rate as
    ``{"__counts__": {"recall": [k, n], ...}}`` (or ``{"k", "n"}`` dicts). It
    never lands in ``metrics``; it is what ``gates derive`` reads.
    """
    merged: dict[str, float] = {}
    owner: dict[str, str] = {}
    for name, fn in zip(names, callables, strict=False):
        produced = dict(fn(row_results) or {})
        raw_pathologies = produced.pop(PATHOLOGIES_KEY, None)
        if raw_pathologies is not None and pathologies_out is not None:
            try:
                pathologies_out.extend(normalize_plugin_pathologies(raw_pathologies, plugin=name))
            except ValueError as exc:
                raise GateConfigError(str(exc)) from exc
        raw_counts = produced.pop(COUNTS_KEY, None)
        if raw_counts is not None and counts_out is not None:
            if not isinstance(raw_counts, dict):
                raise GateConfigError(
                    f"evaluation.metrics: {name!r} {COUNTS_KEY} must be a mapping"
                )
            for metric, kn in raw_counts.items():
                if isinstance(kn, dict):
                    k, n = kn.get("k"), kn.get("n")
                else:
                    k, n = (list(kn) + [None, None])[:2]
                if k is None or n is None:
                    raise GateConfigError(
                        f"evaluation.metrics: {name!r} {COUNTS_KEY}[{metric!r}] needs k and n"
                    )
                if metric in counts_out:
                    raise GateConfigError(
                        f"evaluation.metrics: {name!r} reports counts for {metric!r} twice"
                    )
                counts_out[str(metric)] = {"k": int(k), "n": int(n)}
        for key, value in produced.items():
            if key in merged:
                raise GateConfigError(
                    f"evaluation.metrics: {name!r} and {owner[key]!r} both report "
                    f"{key!r}. Rename one metric key or configure a single metrics "
                    "plugin."
                )
            owner[key] = name
            merged[key] = value
    return merged


def run_evaluation(
    *,
    checkpoint_dir: Path,
    dataset_dir: Path,
    out_path: Path,
    model_def: Optional[ModelDefinition] = None,
    predictor: PredictorLike,
    validator: Optional[ValidatorLike] = None,
    metrics_fn: Optional[MetricsLike] = None,
    device: str = "auto",
    split: str = "test",
    max_input_tokens: int = 2048,
    baseline_path: Optional[Path] = None,
    failures_to_keep: int = 20,
    limit: Optional[int] = None,
    schema_path: Optional[Path] = None,
    contracts_path: Optional[Path] = None,
    prompt_spec_path: Optional[Path] = None,
    task: Optional[str] = None,
    enforce_gates: bool = False,
    gate_spec: Optional[dict[str, float]] = None,
    smoke_gated: bool = False,
    slices: Optional[Sequence[SliceSpec]] = None,
    cache_predictions: bool = False,
    strict_population: bool = False,
    rows_path: Optional[Path] = None,
    overrides: Optional[dict[str, Any]] = None,
    batch_size: int = 1,
) -> Report:
    """Run the shared eval loop and write a :class:`Report` JSON.

    ``gate_spec`` overrides the configured minima (the lifecycle runner passes
    the smoke tier); ``smoke_gated`` records that the pass came from that tier.
    ``slices`` adds per-value pass rates for the named row fields;
    ``cache_predictions`` keeps every row's output beside the report, keyed to
    the split's content hash, for derivations that must not re-run inference.
    ``strict_population`` refuses to enforce floors derived on a different
    split (``evaluation.gates_benchmark``) instead of warning. ``rows_path``
    evaluates a manifest that is not one of the prepared splits (a blind
    population); ``split`` then only names it. ``overrides`` are the
    ``--set`` values already applied to ``model_def``, recorded on the report
    so a number measured under one is never read as the file's. ``batch_size``
    > 1 feeds rows through the predictor's ``predict_batch(rows)`` in chunks
    (one device sync per chunk); per-row latency is then the chunk's time
    divided by its rows and the report says so (``latency_amortized``). A
    predictor without ``predict_batch`` runs one row at a time whatever the
    batch size, with a warning.
    """
    discover_plugins()

    checkpoint_dir = Path(checkpoint_dir)
    dataset_dir = Path(dataset_dir)
    out_path = Path(out_path)

    split_path = Path(rows_path) if rows_path is not None else dataset_dir / f"{split}.jsonl"
    rows = list(iter_jsonl(split_path))
    if not rows:
        raise ValueError(f"No rows in {split_path}")
    split_sha256 = sha256_file(split_path)
    if limit is not None and limit > 0:
        rows = rows[:limit]

    target_device = resolve_device(device)

    pred_obj = _resolve_callable("predictor", predictor, PREDICTORS)
    if isinstance(pred_obj, type):
        pred_obj = pred_obj()

    resolved_schema = None
    resolved_contracts = None
    resolved_prompt = None
    try:
        resolved_schema = resolve_eval_asset(
            "schema",
            model_def=model_def,
            checkpoint_dir=checkpoint_dir,
            filenames=(
                "schema.json",
                "jcl_validation_schema.json",
                "spool_interpretation_schema.json",
            ),
            explicit=schema_path,
        )
    except DeclaredAssetMissing:
        # Declared and missing is a config error, not an absent optional asset.
        raise
    except FileNotFoundError:
        # Causal-SFT / no-validator paths may omit schema.
        resolved_schema = None

    try:
        resolved_contracts = resolve_eval_asset(
            "contracts",
            model_def=model_def,
            checkpoint_dir=checkpoint_dir,
            filenames=("node_contracts.json",),
            explicit=contracts_path,
        )
    except DeclaredAssetMissing:
        raise
    except FileNotFoundError:
        resolved_contracts = None

    try:
        resolved_prompt = resolve_eval_asset(
            "prompt_spec",
            model_def=model_def,
            checkpoint_dir=checkpoint_dir,
            filenames=("prompt_spec.json",),
            explicit=prompt_spec_path,
        )
    except DeclaredAssetMissing:
        raise
    except FileNotFoundError:
        resolved_prompt = None

    setup = getattr(pred_obj, "setup", None)
    if callable(setup):
        setup(
            checkpoint_dir,
            model_def=model_def,
            device=target_device,
            max_input_tokens=max_input_tokens,
            schema_path=resolved_schema,
            contracts_path=resolved_contracts,
            prompt_spec_path=resolved_prompt,
        )

    validate_fn: Callable[..., ValidationResult] = resolve_validator(validator)
    # A real validator (name or callable) needs a schema when declared; the
    # noop path does not. Contracts are optional and task-specific.
    if validator is not None and resolved_schema is None:
        raise FileNotFoundError(
            "Evaluator requires a schema file. Set data/dataset.schema in "
            "model.yml, pass schema_path=, or place schema.json under the checkpoint."
        )

    metrics_names = (
        [str(m) for m in metrics_fn]
        if isinstance(metrics_fn, (list, tuple))
        else ([str(metrics_fn)] if metrics_fn is not None else [])
    )
    metrics_callables = _resolve_metrics(metrics_fn)

    row_results: list[RowEval] = []
    failures: list[dict] = []
    timings: list[float] = []

    request_field = "request"
    if model_def is not None:
        cfg = get_dataset_cfg(model_def)
        request_field = str(cfg.get("request_field") or cfg.get("raw_field") or "request")

    predict = pred_obj.predict if hasattr(pred_obj, "predict") else pred_obj
    predict_batch = getattr(pred_obj, "predict_batch", None)
    batch_size = max(1, int(batch_size))
    if batch_size > 1 and not callable(predict_batch):
        console.print(
            f"[yellow]warning[/] batch_size={batch_size} but the predictor has no "
            "predict_batch(rows); evaluating one row at a time"
        )
        batch_size = 1

    def _sync() -> None:
        if target_device.type == "mps":
            import torch

            torch.mps.synchronize()
        elif target_device.type == "cuda":
            import torch

            torch.cuda.synchronize()

    # Evaluation generates once per row and writes its report at the end, so
    # without this it is a silent half hour that looks identical to a hang -
    # and the only honest answer to "how much longer" is to wait and see.
    # Printed on its own line rather than redrawn in place: this output is read
    # through a pipe as often as at a terminal, and a carriage-return bar
    # arrives there as one unbroken line that says nothing until it ends.
    total = len(rows)
    every = max(1, total // 20)

    def _chunks() -> Any:
        """(index of the chunk's last row, rows, outputs, ms per row)."""
        for start in range(0, total, batch_size):
            chunk = rows[start : start + batch_size]
            t0 = time.perf_counter()
            if batch_size > 1:
                outputs = list(predict_batch(chunk))  # type: ignore[misc]
                if len(outputs) != len(chunk):
                    raise ValueError(
                        f"predict_batch returned {len(outputs)} outputs for "
                        f"{len(chunk)} rows; one string per row is the contract"
                    )
            else:
                outputs = [predict(chunk[0])]
            _sync()
            per_row = (time.perf_counter() - t0) * 1000.0 / len(chunk)
            yield start + len(chunk), chunk, outputs, per_row

    for last_index, chunk, outputs, elapsed in _chunks():
        timings.extend([elapsed] * len(chunk))
        index = last_index

        if (index // every) != ((index - len(chunk)) // every) or index == total:
            # Remaining time from the rows this run has actually generated, not
            # from a rate measured elsewhere: the same split takes minutes on
            # one machine and an hour on another, and a figure carried over
            # from the faster one is worse than no figure at all.
            mean_s = sum(timings) / len(timings) / 1000.0
            left_s = mean_s * (total - index)
            console.print(
                f"[cyan]eval[/] {index}/{total}  {elapsed / 1000.0:.1f}s "
                f"(mean {mean_s:.1f}s)  ~{left_s / 60.0:.0f}m left"
            )

        for row, gen_text in zip(chunk, outputs, strict=True):
            user_prompt = row.get(request_field)
            if not isinstance(user_prompt, str):
                # Fall back to classic text field when request_field is an image path.
                alt = row.get("request")
                user_prompt = alt if isinstance(alt, str) else None

            if resolved_schema is None and resolved_contracts is None:
                result = validate_fn(gen_text, user_prompt=user_prompt)
            else:
                kwargs: dict[str, Any] = {"user_prompt": user_prompt}
                if resolved_schema is not None:
                    kwargs["schema_path"] = resolved_schema
                if resolved_contracts is not None:
                    kwargs["contracts_path"] = resolved_contracts
                result = validate_fn(gen_text, **kwargs)

            item = RowEval(row=row, gen_text=gen_text, result=result, latency_ms=elapsed)
            row_results.append(item)

            if not result.ok and len(failures) < failures_to_keep:
                input_val = row.get(request_field, row.get("request"))
                if isinstance(input_val, str):
                    input_preview: Any = input_val[:500]
                else:
                    input_preview = input_val
                failures.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "category": row.get("category"),
                        "request": input_preview,
                        request_field: input_preview,
                        "raw_output": gen_text[:1500],
                        "errors": [
                            {
                                "layer": e.layer,
                                "code": e.code,
                                "message": e.message,
                                "location": e.location,
                                "hint": e.hint,
                            }
                            for e in result.errors
                        ],
                    }
                )

    n = len(row_results)
    counts: dict[str, dict[str, int]] = {}
    pathologies: list[dict[str, Any]] = []
    if metrics_callables:
        metrics = _merge_metrics(
            metrics_callables,
            row_results,
            metrics_names,
            counts_out=counts,
            pathologies_out=pathologies,
        )
    else:
        # Layer-pass rates only.
        layer_pass: dict[int, int] = {}
        all_ok = 0
        for item in row_results:
            for layer in item.result.passed_layers:
                layer_pass[layer] = layer_pass.get(layer, 0) + 1
            if item.result.ok:
                all_ok += 1
        metrics = {f"layer_{k}_pass_rate": layer_pass.get(k, 0) / n for k in sorted(layer_pass)}
        metrics["all_layers_pass_rate"] = all_ok / n if n else 0.0
        for k in sorted(layer_pass):
            counts[f"layer_{k}_pass_rate"] = {"k": layer_pass[k], "n": n}
        counts["all_layers_pass_rate"] = {"k": all_ok, "n": n}

    for key, value in coverage_metrics(row_results).items():
        if key in metrics:
            raise GateConfigError(
                f"metrics plugin reports {key!r}, which maatml computes itself "
                "(prediction coverage). Rename the plugin's metric."
            )
        metrics[key] = value
    counts[COVERAGE_METRIC] = {
        "k": sum(1 for item in row_results if item.gen_text and item.gen_text.strip()),
        "n": n,
    }

    per_class = _category_buckets(row_results)
    pathologies.extend(detect_pathologies(row_results, metrics, per_class))
    slice_stats = slice_buckets(row_results, slices or [])
    for field_name, values in slice_stats.items():
        for slice_value, stats in values.items():
            if stats.get("n"):
                counts[f"{SLICE_GATE_PREFIX}{field_name}={slice_value}"] = {
                    "k": int(stats["passed"]),
                    "n": int(stats["n"]),
                }

    # Anything the predictor did to the raw model output (brace repair) or to
    # the input (truncation at the token budget) is reported, so a pass rate is
    # never quietly a measurement of a repaired output.
    from ..environment import environment_manifest

    environment = environment_manifest(
        getattr(model_def, "model_dir", None) if model_def is not None else None
    )
    extras: dict[str, Any] = {
        "max_input_tokens": max_input_tokens,
        "split_sha256": split_sha256,
        "environment": environment,
    }
    if limit is not None and limit > 0:
        extras["limit"] = int(limit)
    if overrides:
        extras["overrides"] = dict(overrides)
    extras["batch_size"] = batch_size
    if batch_size > 1:
        # A chunk's wall time spread over its rows, not a per-row measurement.
        extras["latency_amortized"] = True
    if split == "test" and rows_path is None:
        from ..data.populations import read_benchmark_state

        state = read_benchmark_state(dataset_dir)
        if state and state.get("version"):
            extras["benchmark_version"] = str(state["version"])
    # The cut the predictor decoded at, so a later sweep knows the cache cannot
    # be re-filtered below it.
    evaluation_cfg = getattr(model_def, "evaluation", None) if model_def is not None else None
    if isinstance(evaluation_cfg, dict):
        op = evaluation_cfg.get("operating_point")
        op_key = op.get("threshold_key") if isinstance(op, dict) else None
        if isinstance(op_key, str) and op_key in evaluation_cfg:
            extras["decode_threshold"] = {"key": op_key, "value": evaluation_cfg[op_key]}
    report_extras = getattr(pred_obj, "report_extras", None)
    if callable(report_extras):
        extras.update(report_extras() or {})
    truncated = extras.get("truncated_inputs")
    if truncated:
        console.print(
            f"[yellow]warning[/] {truncated}/{n} inputs exceeded "
            f"max_input_tokens={max_input_tokens} and were truncated"
        )
    repaired = extras.get("brace_repairs")
    if repaired:
        console.print(
            f"[yellow]warning[/] brace repair rewrote {repaired}/{n} model outputs "
            "(evaluation.repair_braces); pass rates include the repair"
        )

    identity_name = model_def.name if model_def else checkpoint_dir.name
    identity_version = model_def.version if model_def else ""
    identity_id = model_def.model_id if model_def else str(checkpoint_dir)
    report_task = task or (model_def.task if model_def else "")

    if cache_predictions:
        cache_path = predictions_path(out_path)
        write_predictions(
            cache_path,
            header={
                "checkpoint": str(checkpoint_dir),
                "split": split,
                "split_path": str(split_path),
                "split_sha256": split_sha256,
                "report": out_path.name,
                "limit": int(limit) if limit is not None and limit > 0 else None,
            },
            rows=(
                prediction_row(
                    row=item.row,
                    gen_text=item.gen_text,
                    result=item.result,
                    latency_ms=item.latency_ms,
                    drop_fields=(request_field, "request"),
                )
                for item in row_results
            ),
        )
        extras["predictions_cache"] = cache_path.name

    gates_payload: Optional[dict[str, Any]] = None
    passed: Optional[bool] = None
    if enforce_gates:
        # Raises GateConfigError when no gates are configured, enforcement must
        # never pass vacuously.
        spec = gate_spec or resolve_gate_spec(model_def, smoke=smoke_gated)
        tiers = gate_tiers(model_def, smoke=smoke_gated) if model_def is not None else {}
        gates_payload = check_gates(metrics, spec, tiers=tiers, slices=slice_stats)
        # A smoke-tier pass is recorded as such wherever it is stored, so it
        # can never be read as a production gate pass later.
        gates_payload["smoke"] = bool(smoke_gated)
        gates_payload["environment"] = environment
        # Floors describe the population they were derived on. Enforcing them
        # against another split mixes distribution shift into the regression
        # signal, so the mismatch is always recorded and, strictly, refused.
        gates_payload["benchmark_sha256"] = split_sha256
        evaluation_cfg = getattr(model_def, "evaluation", None) if model_def is not None else None
        declared = (
            evaluation_cfg.get("gates_benchmark") if isinstance(evaluation_cfg, dict) else None
        )
        if isinstance(declared, str) and declared:
            gates_payload["floors_benchmark_sha256"] = declared
            if declared != split_sha256:
                message = (
                    f"evaluation.gates_benchmark {declared[:16]} is not this split "
                    f"({split_sha256[:16]}); the floors were derived on a different "
                    "population. Re-derive with `maatml gates derive` or evaluate the "
                    "split they came from."
                )
                if strict_population:
                    raise GateConfigError(message)
                console.print(f"[yellow]warning[/] {message}")
                gates_payload["population_mismatch"] = True
        if gates_payload.get("advisory_failed"):
            console.print(
                "[yellow]advisory[/] below floor: " + ", ".join(gates_payload["advisory_failed"])
            )
        if pathologies:
            names = sorted({str(p["name"]) for p in pathologies})
            gates_payload["pathologies"] = names
            # A rehearsal exists to prove the model works at all; a floor set
            # for the smoke tier is loose enough for a model that never fires
            # to clear it, so the signature itself is the failing gate there.
            if smoke_gated:
                for name in names:
                    gates_payload["results"][f"pathology:{name}"] = {
                        "minimum": None,
                        "actual": name,
                        "passed": False,
                        "tier": "blocking",
                    }
                gates_payload["passed"] = False
            console.print(
                "[yellow]pathology[/] "
                + "; ".join(f"{p['name']}: {p['evidence']}" for p in pathologies)
            )
        passed = bool(gates_payload["passed"])

    report = Report(
        model_id=identity_id,
        name=identity_name,
        version=identity_version,
        task=report_task,
        dataset=str(dataset_dir / f"{split}.jsonl"),
        n=n,
        metrics=metrics,
        per_class=per_class,
        slices=slice_stats,
        counts=counts,
        pathologies=pathologies,
        latency_ms=latency_stats(timings),
        baseline_delta=baseline_delta(metrics, baseline_path),
        sample_failures=failures,
        extras=extras,
        gates=gates_payload,
        passed=passed,
    )
    report.write(out_path)
    console.print(
        f"[green]eval[/] {report.task or report.name}: n={n} "
        + " ".join(f"{k}={v:.3f}" for k, v in list(metrics.items())[:4])
    )
    return report
