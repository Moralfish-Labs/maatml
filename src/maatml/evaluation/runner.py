"""Evaluation entry points and report helpers.

Task evaluation goes through :func:`maatml.evaluation.harness.run_evaluation`
(via the CLI). Shared report types live in ``harness`` and are re-exported
here for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from .harness import (
    LatencyStats,
    Report,
    baseline_delta,
    binary_prf,
    latency_stats,
    per_class_prf,
    percentile,
    run_evaluation,
)

# Backward-compatible private aliases used by tests.
_percentile = percentile
_latency_stats = latency_stats
_binary_prf = binary_prf
_per_class_prf = per_class_prf
_baseline_delta = baseline_delta

console = Console()

__all__ = [
    "LatencyStats",
    "Report",
    "evaluate_model",
    "run_evaluation",
    "write_markdown_summary",
    "_percentile",
    "_latency_stats",
    "_binary_prf",
    "_per_class_prf",
    "_baseline_delta",
]


def evaluate_model(
    model_def,
    *,
    checkpoint: Optional[str] = None,
    split: str = "test",
    device: str = "auto",
    baseline: Optional[Path] = None,
    max_input_tokens: Optional[int] = None,
    limit: Optional[int] = None,
    gate: bool = False,
    smoke: bool = False,
    cache_predictions: Optional[bool] = None,
    strict_population: bool = False,
    out_dir: Optional[Path] = None,
    record_gates: bool = True,
    blind: bool = False,
    force: bool = False,
    label: Optional[str] = None,
    overrides: Optional[dict[str, Any]] = None,
    batch_size: Optional[int] = None,
) -> tuple["Report", Path]:
    """Evaluate a checkpoint of ``model_def`` and write report.{json,md}.

    The single implementation behind ``maatml evaluate`` and the lifecycle
    runner's evaluate step, so both enforce gates, resolve the token budget,
    and record results on the run identically. Configuration problems raise
    before the checkpoint is resolved or loaded. ``cache_predictions`` falls
    back to ``evaluation.cache_predictions`` when not given. ``out_dir`` redirects
    the report (a controlled replay must not overwrite the run's own evidence)
    and ``record_gates=False`` keeps it off the run record. ``blind`` evaluates
    ``dataset.blind_samples`` once for a candidate whose gate evidence is
    current; ``force`` allows a repeat spend. ``overrides`` (already applied
    to ``model_def`` by ``--set``) are recorded on the report and refused
    with ``gate`` or ``blind``: a measurement under an override is never
    gate evidence. ``batch_size`` falls back to ``evaluation.batch_size``
    (default 1) and only takes effect on a predictor with ``predict_batch``.
    """
    from ..runs import (
        evidence_fingerprint,
        get_run,
        record_blind_spend,
        resolve_checkpoint,
        run_id_for_checkpoint,
        update_run_gates,
    )
    from . import predictors as _predictors  # noqa: F401  register built-ins
    from .harness import (
        GateConfigError,
        _resolve_metrics,
        resolve_gate_spec,
        resolve_slices,
        resolve_validator,
        run_evaluation,
        uses_smoke_gates,
    )

    evaluation = model_def.evaluation or {}
    if overrides and (gate or blind):
        raise GateConfigError(
            "--set overrides model.yml for this evaluate only; a report measured under "
            "an override is not gate evidence, so --set cannot be combined with --gate "
            "or --blind"
        )
    predictor = evaluation.get("predictor")
    validator = evaluation.get("validator")
    metrics = evaluation.get("metrics")
    if isinstance(metrics, list) and not metrics:
        metrics = None

    if predictor is None:
        from ..registry import PREDICTORS
        from ..scaffold import normalize_architecture

        arch = normalize_architecture(model_def.architecture)
        if PREDICTORS.get(model_def.architecture):
            predictor = model_def.architecture
        elif PREDICTORS.get(arch):
            predictor = arch
    if predictor is None:
        raise KeyError(
            f"No predictor for architecture={model_def.architecture!r}; "
            "set evaluation.predictor in model.yml"
        )

    gate_spec = None
    smoke_gated = False
    if gate:
        gate_spec = resolve_gate_spec(model_def, smoke=smoke)
        smoke_gated = smoke and uses_smoke_gates(model_def)
    if validator is not None:
        resolve_validator(validator)
    _resolve_metrics(metrics)
    slices = resolve_slices(model_def)
    if cache_predictions is None:
        cache_predictions = bool(evaluation.get("cache_predictions", False))
    if batch_size is None:
        raw_batch = evaluation.get("batch_size", 1)
        if isinstance(raw_batch, bool) or not isinstance(raw_batch, int) or raw_batch < 1:
            raise GateConfigError("evaluation.batch_size must be an integer >= 1")
        batch_size = raw_batch
    elif batch_size < 1:
        raise GateConfigError("--batch-size must be >= 1")

    rows_path: Optional[Path] = None
    blind_sha256: Optional[str] = None
    current_fp: Optional[str] = None
    if blind:
        from ..config import get_dataset_cfg
        from ..utils.io import sha256_file

        blind_rel = get_dataset_cfg(model_def).get("blind_samples")
        if not blind_rel:
            raise GateConfigError("--blind needs dataset.blind_samples in model.yml")
        rows_path = model_def.resolve(str(blind_rel))
        if not rows_path.is_file():
            raise GateConfigError(f"dataset.blind_samples not found: {rows_path}")
        blind_sha256 = sha256_file(rows_path)

    ckpt = resolve_checkpoint(model_def, checkpoint)
    run_id = run_id_for_checkpoint(model_def, ckpt)
    from .operating_point import report_name

    if blind:
        split = "blind"
        gate = True
        if gate_spec is None:
            gate_spec = resolve_gate_spec(model_def, smoke=False)
        rec = get_run(model_def, run_id)
        current_fp = evidence_fingerprint(model_def, ckpt)
        if rec is None:
            raise GateConfigError(
                f"{run_id} is not in runs.jsonl; a blind evaluate needs a recorded, gated run"
            )
        if not (rec.gates or {}).get("passed") or rec.smoke_gated:
            raise GateConfigError(
                f"{run_id} has no production gate pass on record; run evaluate --gate "
                "on the benchmark first. The blind population is spent only on a candidate "
                "that already passed."
            )
        if rec.gated_fingerprint != current_fp:
            raise GateConfigError(
                f"{run_id} changed since its gate pass (evaluation config or weights); "
                "re-run evaluate --gate before spending the blind population"
            )
        prior = [
            s
            for s in (rec.blind_spends or [])
            if s.get("fingerprint") == current_fp and s.get("blind_sha256") == blind_sha256
        ]
        if prior and not force:
            raise GateConfigError(
                f"{run_id} already spent this blind population at this fingerprint "
                f"({len(prior)} time(s), last report {prior[-1].get('report')}); "
                "a blind test is run once per frozen candidate. Pass --force to repeat, "
                "and say so in the record."
            )

    target_dir = Path(out_dir) if out_dir is not None else model_def.eval_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    # A val report is named for its split so it never overwrites the test one.
    out_path = target_dir / f"{report_name(label or run_id, split)}.json"
    budget = (
        max_input_tokens if max_input_tokens is not None else model_def.packaging.max_input_tokens
    )

    report = run_evaluation(
        checkpoint_dir=ckpt,
        dataset_dir=model_def.prepared_dir,
        out_path=out_path,
        model_def=model_def,
        predictor=predictor,
        validator=validator,
        metrics_fn=metrics,
        device=device,
        split=split,
        max_input_tokens=budget,
        baseline_path=baseline,
        limit=limit,
        task=model_def.task,
        enforce_gates=gate,
        gate_spec=gate_spec,
        smoke_gated=smoke_gated,
        slices=slices,
        cache_predictions=cache_predictions,
        strict_population=strict_population,
        rows_path=rows_path,
        overrides=overrides,
        batch_size=batch_size,
    )
    write_markdown_summary(report, out_path.with_suffix(".md"))

    if blind:
        spend = {
            "fingerprint": current_fp,
            "blind_sha256": blind_sha256,
            "report": out_path.name,
            "passed": report.passed,
            "forced": bool(force),
        }
        record_blind_spend(model_def, run_id, spend)
        return report, out_path

    run_rec = get_run(model_def, run_id) if record_gates else None
    if run_rec is not None and report.gates is not None:
        update_run_gates(
            model_def,
            run_rec.run_id,
            report.gates,
            metrics=report.metrics,
            smoke_gated=smoke_gated,
            gated_fingerprint=evidence_fingerprint(model_def, ckpt) if split == "test" else None,
        )
    return report, out_path


_COUNT_KEYS = frozenset({"n", "support", "passed"})


def _format_class_stats(vals: dict[str, float]) -> str:
    """Render whatever per-class keys the report carries.

    Category buckets report ``pass_rate`` / ``passed`` / ``n``; metrics plugins
    that compute real per-class P/R/F1 report those instead. Neither shape is
    padded with invented keys.
    """
    parts = []
    for key in sorted(vals):
        value = vals[key]
        if key in _COUNT_KEYS:
            parts.append(f"{key}={int(value)}")
        else:
            parts.append(f"{key}={value:.3f}")
    return " ".join(parts)


def write_markdown_summary(report: Report, path: str | Path) -> Path:
    title = report.task or report.name or "eval"
    lines = [
        f"# {title} eval report",
        "",
        f"- model: `{report.model_id}`",
    ]
    if report.name:
        lines.append(f"- name: `{report.name}`")
    if report.version:
        lines.append(f"- version: `{report.version}`")
    lines.extend(
        [
            f"- dataset: `{report.dataset}`",
            f"- n: {report.n}",
            "",
            "## Metrics",
            "",
        ]
    )
    for k, v in sorted(report.metrics.items()):
        lines.append(f"- {k}: {v:.4f}")
    if report.gates is not None:
        lines.extend(["", "## Gates", ""])
        if report.passed is not None:
            lines.append(f"- passed: {report.passed}")
        results = report.gates.get("results") or {}
        for name, info in sorted(results.items()):
            tier = info.get("tier", "blocking")
            suffix = "" if tier == "blocking" else f" tier={tier}"
            lines.append(
                f"- {name}: actual={info.get('actual')} "
                f"minimum={info.get('minimum')} passed={info.get('passed')}{suffix}"
            )
        if report.gates.get("benchmark_sha256"):
            lines.append(f"- benchmark: `{report.gates['benchmark_sha256']}`")
        if report.gates.get("population_mismatch"):
            lines.append(
                f"- floors derived on `{report.gates.get('floors_benchmark_sha256')}` "
                "(population mismatch)"
            )
    if report.latency_ms:
        lines.extend(
            [
                "",
                "## Latency (ms)",
                f"- p50: {report.latency_ms.p50:.2f}",
                f"- p95: {report.latency_ms.p95:.2f}",
                f"- mean: {report.latency_ms.mean:.2f}",
                f"- n: {report.latency_ms.n}",
            ]
        )
    if report.pathologies:
        lines.extend(["", "## Pathologies", ""])
        for entry in report.pathologies:
            lines.append(f"- {entry.get('name')}: {entry.get('evidence')}")
    if report.per_class:
        lines.extend(["", "## Per-class", ""])
        for label, vals in sorted(report.per_class.items()):
            lines.append(f"- {label}: {_format_class_stats(vals)}")
    if report.slices:
        lines.extend(["", "## Slices", ""])
        for field_name, values in sorted(report.slices.items()):
            for value, stats in sorted(values.items()):
                n = int(stats.get("n", 0))
                if n == 0:
                    lines.append(f"- {field_name}={value}: n=0 (no rate)")
                    continue
                lines.append(
                    f"- {field_name}={value}: n={n} "
                    f"pass_rate={stats['pass_rate']:.3f} w95={stats['pass_rate_w95']:.3f}"
                )
    if report.baseline_delta:
        lines.extend(["", "## Baseline delta", ""])
        for k, v in sorted(report.baseline_delta.items()):
            sign = "+" if v >= 0 else ""
            lines.append(f"- {k}: {sign}{v:.4f}")
    if report.extras:
        lines.extend(["", "## Extras", ""])
        for k, v in sorted(report.extras.items()):
            lines.append(f"- {k}: {v}")
    if report.sample_failures:
        lines.extend(["", "## Sample failures", ""])
        # Cap the markdown list so a long eval stays readable; the JSON report
        # already keeps the full failures_to_keep budget.
        for failure in report.sample_failures[:20]:
            sample_id = failure.get("sample_id") or "(no sample_id)"
            lines.append(f"- `{sample_id}`")
            for err in failure.get("errors") or []:
                layer = err.get("layer")
                code = err.get("code")
                location = err.get("location")
                message = err.get("message") or ""
                hint = err.get("hint")
                where = f" at `{location}`" if location else ""
                lines.append(f"  - L{layer}/{code}{where}: {message}")
                if hint:
                    lines.append(f"    - fix: {hint}")
    out = Path(path)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
