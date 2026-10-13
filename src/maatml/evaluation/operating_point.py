"""``maatml operating-point derive``: a decision threshold chosen on val, spent once on test.

The predictor owns re-scoring: ``rescore(rows, threshold)`` takes the rows of a
prediction cache and returns the metrics that hold at ``threshold`` (with
``__counts__`` where they are rates). The sweep never re-runs inference; it
reads the cache an ``evaluate --split val --cache`` run left behind, picks the
best objective under the budget, and writes the threshold into ``model.yml``
with the sweep artifact as its provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..registry import PREDICTORS
from ..utils.io import write_json_atomic
from .harness import COUNTS_KEY, GateConfigError, Report, ReportSchemaError
from .predictions import PredictionsError, read_predictions
from .stats import wilson_lower

OPERATING_POINT_KIND = "maatml.operating_point/1"
DEFAULT_GRID = tuple(round(x / 20.0, 2) for x in range(1, 20))


class OperatingPointError(ValueError):
    """The sweep cannot be run or the budget cannot be met."""


@dataclass(frozen=True)
class OperatingPointSpec:
    threshold_key: str
    objective: str
    budget_metric: Optional[str] = None
    budget_max: Optional[float] = None
    sources: tuple[str, ...] = ()
    source_field: str = "dataset"
    grid: tuple[float, ...] = DEFAULT_GRID

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold_key": self.threshold_key,
            "objective": self.objective,
            "budget": (
                {"metric": self.budget_metric, "max": self.budget_max}
                if self.budget_metric
                else None
            ),
            "sources": list(self.sources),
            "source_field": self.source_field,
            "grid": list(self.grid),
        }


def resolve_operating_point(model_def: Any) -> OperatingPointSpec:
    """Parse ``evaluation.operating_point``; every malformed key is a config error."""
    evaluation = getattr(model_def, "evaluation", None)
    spec = evaluation.get("operating_point") if isinstance(evaluation, dict) else None
    if not isinstance(spec, dict):
        raise GateConfigError(
            "evaluation.operating_point is not configured: set threshold_key, objective "
            "and optionally budget: {metric, max}, sources, grid"
        )
    key = spec.get("threshold_key")
    objective = spec.get("objective")
    if not isinstance(key, str) or not key.strip():
        raise GateConfigError(
            "evaluation.operating_point.threshold_key must be a metric-setting key"
        )
    if not isinstance(objective, str) or not objective.strip():
        raise GateConfigError("evaluation.operating_point.objective must name a rescore metric")
    budget = spec.get("budget")
    budget_metric: Optional[str] = None
    budget_max: Optional[float] = None
    if budget is not None:
        if not isinstance(budget, dict) or "metric" not in budget or "max" not in budget:
            raise GateConfigError("evaluation.operating_point.budget must be {metric, max}")
        budget_metric = str(budget["metric"])
        try:
            budget_max = float(budget["max"])
        except (TypeError, ValueError) as exc:
            raise GateConfigError("evaluation.operating_point.budget.max must be a number") from exc
    sources = spec.get("sources") or []
    if not isinstance(sources, list):
        raise GateConfigError("evaluation.operating_point.sources must be a list")
    grid_raw = spec.get("grid")
    grid: tuple[float, ...]
    if grid_raw is None:
        grid = DEFAULT_GRID
    elif isinstance(grid_raw, list) and grid_raw:
        try:
            grid = tuple(sorted({round(float(v), 6) for v in grid_raw}))
        except (TypeError, ValueError) as exc:
            raise GateConfigError("evaluation.operating_point.grid must be numbers") from exc
    elif isinstance(grid_raw, dict):
        try:
            start, stop, step = (
                float(grid_raw["start"]),
                float(grid_raw["stop"]),
                float(grid_raw["step"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GateConfigError(
                "evaluation.operating_point.grid must be a list or {start, stop, step}"
            ) from exc
        if step <= 0 or stop < start:
            raise GateConfigError("evaluation.operating_point.grid: step > 0 and stop >= start")
        values = []
        v = start
        while v <= stop + 1e-9:
            values.append(round(v, 6))
            v += step
        grid = tuple(values)
    else:
        raise GateConfigError(
            "evaluation.operating_point.grid must be a list or {start, stop, step}"
        )
    return OperatingPointSpec(
        threshold_key=key.strip(),
        objective=objective.strip(),
        budget_metric=budget_metric,
        budget_max=budget_max,
        sources=tuple(str(s) for s in sources),
        source_field=str(spec.get("source_field") or "dataset"),
        grid=grid,
    )


RescoreFn = Callable[[list[dict[str, Any]], float], dict[str, Any]]


def resolve_rescore(model_def: Any) -> RescoreFn:
    """The predictor's ``rescore(rows, threshold)``; instantiated without ``setup``."""
    from ..scaffold import normalize_architecture

    evaluation = getattr(model_def, "evaluation", None) or {}
    name = evaluation.get("predictor") if isinstance(evaluation, dict) else None
    if name is None:
        arch = normalize_architecture(model_def.architecture)
        name = model_def.architecture if PREDICTORS.get(model_def.architecture) else arch
    pred = PREDICTORS.get(str(name))
    if pred is None:
        raise GateConfigError(
            f"evaluation.predictor={name!r} is not a registered predictor "
            f"(known: {', '.join(PREDICTORS.names()) or 'none'})"
        )
    obj = pred() if isinstance(pred, type) else pred
    rescore = getattr(obj, "rescore", None)
    if not callable(rescore):
        raise OperatingPointError(
            f"predictor {name!r} has no rescore(rows, threshold); the sweep needs the "
            "predictor to re-score cached predictions at a threshold without inference"
        )
    return rescore


def _filter_rows(rows: Sequence[dict[str, Any]], spec: OperatingPointSpec) -> list[dict[str, Any]]:
    if not spec.sources:
        return list(rows)
    wanted = set(spec.sources)
    return [r for r in rows if str((r.get("row") or {}).get(spec.source_field)) in wanted]


def sweep(
    rows: Sequence[dict[str, Any]],
    rescore: RescoreFn,
    spec: OperatingPointSpec,
    *,
    floor_threshold: Optional[float] = None,
) -> list[dict[str, Any]]:
    """One point per grid threshold: the objective, the budget metric, and the evidence.

    Thresholds below ``floor_threshold`` (the cut the cache was decoded at)
    are skipped: a cache cannot be re-filtered downward.
    """
    points: list[dict[str, Any]] = []
    rows = list(rows)
    for threshold in spec.grid:
        if floor_threshold is not None and threshold < floor_threshold - 1e-9:
            points.append({"threshold": threshold, "skipped": "below the cache's decode cut"})
            continue
        produced = dict(rescore(rows, float(threshold)) or {})
        counts = produced.pop(COUNTS_KEY, None) or {}
        if spec.objective not in produced:
            raise OperatingPointError(
                f"rescore did not report objective {spec.objective!r} at {threshold}"
            )
        if spec.budget_metric and spec.budget_metric not in produced:
            raise OperatingPointError(
                f"rescore did not report budget metric {spec.budget_metric!r} at {threshold}"
            )
        point: dict[str, Any] = {
            "threshold": threshold,
            "objective": float(produced[spec.objective]),
            "metrics": {k: float(v) for k, v in produced.items()},
            "n_rows": len(rows),
        }
        if spec.budget_metric:
            point["budget"] = float(produced[spec.budget_metric])
        kn = counts.get(spec.objective)
        if kn is not None:
            k, n = (kn.get("k"), kn.get("n")) if isinstance(kn, dict) else (kn[0], kn[1])
            if k is not None and n:
                point["objective_w95"] = wilson_lower(int(k), int(n))
                point["objective_counts"] = {"k": int(k), "n": int(n)}
        points.append(point)
    return points


def pick(points: Sequence[dict[str, Any]], spec: OperatingPointSpec) -> Optional[dict[str, Any]]:
    """Highest objective within budget; ties go to the lower budget, then the higher cut."""
    usable = [p for p in points if "objective" in p]
    if spec.budget_metric and spec.budget_max is not None:
        usable = [p for p in usable if p["budget"] <= spec.budget_max + 1e-12]
    if not usable:
        return None
    return max(
        usable,
        key=lambda p: (p["objective"], -p.get("budget", 0.0), p["threshold"]),
    )


@dataclass
class OperatingPointResult:
    run: str
    split: str
    split_sha256: str
    spec: OperatingPointSpec
    points: list[dict[str, Any]]
    chosen: Optional[dict[str, Any]]
    cache_path: Path
    n_rows: int
    floor_threshold: Optional[float] = None
    refusals: list[str] = field(default_factory=list)

    def comment(self) -> str:
        assert self.chosen is not None
        parts = [f"{self.spec.objective} {self.chosen['objective']:.3f}"]
        if "objective_w95" in self.chosen:
            parts[-1] += f" (w95 {self.chosen['objective_w95']:.3f})"
        if self.spec.budget_metric:
            parts.append(f"@ {self.spec.budget_metric} {self.chosen['budget']:.3f}")
        parts.append(f"on {self.split} n={self.n_rows} bench {self.split_sha256[:16]}")
        return " ".join(parts)

    def artifact(self) -> dict[str, Any]:
        return {
            "kind": OPERATING_POINT_KIND,
            "run": self.run,
            "split": self.split,
            "split_sha256": self.split_sha256,
            "cache": self.cache_path.name,
            "n_rows": self.n_rows,
            "floor_threshold": self.floor_threshold,
            "spec": self.spec.as_dict(),
            "points": self.points,
            "chosen": self.chosen,
            "refusals": list(self.refusals),
        }


def report_name(run: str, split: str) -> str:
    """``<run>`` for the test split, ``<run>.<split>`` otherwise: a val report never
    masquerades as the test one."""
    return run if split == "test" else f"{run}.{split}"


def artifact_path(model_def: Any, run: str, split: str) -> Path:
    return Path(model_def.eval_dir) / f"{report_name(run, split)}.operating_point.json"


def derive_operating_point(
    model_def: Any,
    *,
    run: str,
    split: str = "val",
    grid: Optional[Sequence[float]] = None,
) -> OperatingPointResult:
    spec = resolve_operating_point(model_def)
    if grid:
        spec = OperatingPointSpec(
            **{**spec.__dict__, "grid": tuple(sorted({round(float(v), 6) for v in grid}))}
        )
    if split == "test":
        raise OperatingPointError(
            "an operating point is chosen on val and spent once on test; "
            "derive on --split val (or another non-test split)"
        )
    report_path = Path(model_def.eval_dir) / f"{report_name(run, split)}.json"
    if not report_path.is_file():
        raise OperatingPointError(
            f"no {split} report for {run!r} at {report_path}; run "
            f"`maatml evaluate --checkpoint {run} --split {split} --cache` first"
        )
    try:
        report = Report.read(report_path, strict=True)
    except ReportSchemaError as exc:
        raise OperatingPointError(f"{report_path.name}: {exc}") from exc
    cache_name = report.extras.get("predictions_cache")
    if not cache_name:
        raise OperatingPointError(
            f"{report_path.name} has no predictions cache; re-run evaluate with --cache"
        )
    cache_path = report_path.with_name(str(cache_name))
    try:
        header, rows = read_predictions(cache_path)
    except (PredictionsError, FileNotFoundError) as exc:
        raise OperatingPointError(f"{cache_path.name}: {exc}") from exc
    split_sha256 = str(report.extras.get("split_sha256") or "")
    if header.get("split_sha256") != split_sha256 or header.get("split") != split:
        raise OperatingPointError(
            f"{cache_path.name} was written for another split than {report_path.name}"
        )

    decode = report.extras.get("decode_threshold")
    floor_threshold: Optional[float] = None
    if isinstance(decode, dict) and decode.get("key") == spec.threshold_key:
        try:
            floor_threshold = float(decode["value"])
        except (TypeError, ValueError, KeyError):
            floor_threshold = None

    rescore = resolve_rescore(model_def)
    selected = _filter_rows(rows, spec)
    if not selected:
        raise OperatingPointError(
            f"no cached rows with {spec.source_field} in {list(spec.sources)}; the {split} "
            "split holds none of the sources the operating point is tuned on"
        )
    points = sweep(selected, rescore, spec, floor_threshold=floor_threshold)
    chosen = pick(points, spec)
    result = OperatingPointResult(
        run=run,
        split=split,
        split_sha256=split_sha256,
        spec=spec,
        points=points,
        chosen=chosen,
        cache_path=cache_path,
        n_rows=len(selected),
        floor_threshold=floor_threshold,
    )
    skipped = [p["threshold"] for p in points if "skipped" in p]
    if skipped:
        result.refusals.append(
            f"{len(skipped)} grid point(s) below the cache's decode cut {floor_threshold} "
            "were skipped; re-evaluate with a lower threshold to sweep them"
        )
    if chosen is None:
        result.refusals.append(
            f"no grid point meets the budget {spec.budget_metric} <= {spec.budget_max}"
        )
    return result


def write_artifact(model_def: Any, result: OperatingPointResult) -> Path:
    path = artifact_path(model_def, result.run, result.split)
    write_json_atomic(path, result.artifact())
    return path


def rewrite_threshold(text: str, key: str, value: float, comment: str) -> str:
    """Set ``evaluation.<key>`` in model.yml text, keeping every other line."""
    import re

    line = f"  {key}: {value:g}  # {comment}\n"
    section = re.compile(r"(^evaluation:\n)((?:^  .*\n|^\n(?=  ))*)", re.MULTILINE)
    match = section.search(text)
    if match is None:
        raise OperatingPointError("could not find the evaluation: section in model.yml")
    body = match.group(2)
    key_line = re.compile(rf"^  {re.escape(key)}:.*\n", re.MULTILINE)
    body = key_line.sub(line, body, count=1) if key_line.search(body) else line + body
    return text[: match.start()] + match.group(1) + body + text[match.end() :]


def write_threshold(model_yml: Path, result: OperatingPointResult, artifact: Path) -> Path:
    if result.chosen is None:
        raise OperatingPointError("refusing to write: no grid point met the budget")
    comment = f"{result.comment()}; sweep {artifact.name}"
    text = model_yml.read_text(encoding="utf-8")
    model_yml.write_text(
        rewrite_threshold(
            text, result.spec.threshold_key, float(result.chosen["threshold"]), comment
        ),
        encoding="utf-8",
    )
    return model_yml


def render(result: OperatingPointResult) -> list[str]:
    lines = [
        f"operating point for {result.run} on {result.split} "
        f"(n={result.n_rows}, bench {result.split_sha256[:16]})"
    ]
    for p in result.points:
        if "skipped" in p:
            lines.append(f"  {p['threshold']:.2f}  skipped: {p['skipped']}")
            continue
        budget = (
            f"  {result.spec.budget_metric}={p['budget']:.3f}" if result.spec.budget_metric else ""
        )
        w95 = f" (w95 {p['objective_w95']:.3f})" if "objective_w95" in p else ""
        lines.append(
            f"  {p['threshold']:.2f}  {result.spec.objective}={p['objective']:.3f}{w95}{budget}"
        )
    if result.chosen is not None:
        lines.append(
            f"{result.spec.threshold_key}: {result.chosen['threshold']:g}  # {result.comment()}"
        )
    for refusal in result.refusals:
        lines.append(f"refused {refusal}")
    return lines


def load_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != OPERATING_POINT_KIND:
        raise OperatingPointError(f"{path}: not a {OPERATING_POINT_KIND} artifact")
    return payload
