"""``maatml report``: the folder's evidence, regenerated from the records alone.

Reads ``output/runs.jsonl``, ``output/eval/*.json`` and ``output/seeds/*.json``
and nothing else — not ``model.yml``, not the weights — so the document says
only what was recorded, and regenerating it is byte-identical until a record
changes. Floors are shown with their derivation (k / n and the Wilson 95 %
lower bound from the report's ``counts``), beside the value that met them.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Optional

from .evaluation.stats import wilson_lower
from .runs import RunRecord

_SKIP_REPORT_DIRS = ("select", "replay")


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _runs(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "runs.jsonl"
    if not path.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = RunRecord.model_validate_json(line)
        except ValueError:
            continue
        if rec.run_id not in latest:
            order.append(rec.run_id)
        latest[rec.run_id] = rec.model_dump(mode="json")
    return [latest[r] for r in order]


def _reports(output_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    eval_dir = output_dir / "eval"
    if not eval_dir.is_dir():
        return []
    found = []
    for path in sorted(eval_dir.glob("*.json")):
        payload = _read_json(path)
        if payload is not None and "metrics" in payload:
            found.append((path.name, payload))
    return found


def _seed_studies(output_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    seeds_dir = output_dir / "seeds"
    if not seeds_dir.is_dir():
        return []
    found = []
    for path in sorted(seeds_dir.glob("*.json")):
        payload = _read_json(path)
        if payload is not None and "stats" in payload:
            found.append((path.name, payload))
    return found


def _derivation(counts: dict[str, Any], metric: str) -> Optional[dict[str, Any]]:
    kn = counts.get(metric)
    if not isinstance(kn, dict):
        return None
    k, n = kn.get("k"), kn.get("n")
    if k is None or n is None or int(n) <= 0:
        return None
    return {"k": int(k), "n": int(n), "w95": wilson_lower(int(k), int(n))}


def build_report(output_dir: Path) -> dict[str, Any]:
    """Everything the document says, as data; only ``output/`` is read."""
    output_dir = Path(output_dir)
    runs = _runs(output_dir)
    reports = []
    for name, payload in _reports(output_dir):
        counts = payload.get("counts") or {}
        gates = payload.get("gates") or {}
        results = gates.get("results") or {}
        metrics = []
        for metric, value in sorted((payload.get("metrics") or {}).items()):
            gate = results.get(metric)
            metrics.append(
                {
                    "metric": metric,
                    "value": value,
                    "derivation": _derivation(counts, metric),
                    "floor": gate.get("minimum") if isinstance(gate, dict) else None,
                    "passed": gate.get("passed") if isinstance(gate, dict) else None,
                    "tier": gate.get("tier", "blocking") if isinstance(gate, dict) else None,
                }
            )
        slice_gates = [
            {"gate": g, **info}
            for g, info in sorted(results.items())
            if g.startswith(("slice:", "pathology:"))
        ]
        extras = payload.get("extras") or {}
        reports.append(
            {
                "file": name,
                "model_id": payload.get("model_id"),
                "dataset": payload.get("dataset"),
                "n": payload.get("n"),
                "report_version": payload.get("report_version", 0),
                "passed": payload.get("passed"),
                "smoke": bool(gates.get("smoke")) if gates else None,
                "benchmark_sha256": gates.get("benchmark_sha256") if gates else None,
                "benchmark_version": extras.get("benchmark_version"),
                "population_mismatch": bool(gates.get("population_mismatch")) if gates else False,
                "metrics": metrics,
                "slice_gates": slice_gates,
                "slices": payload.get("slices") or {},
                "pathologies": payload.get("pathologies") or [],
            }
        )
    return {
        "runs": runs,
        "reports": reports,
        "seed_studies": [{"file": name, **payload} for name, payload in _seed_studies(output_dir)],
    }


def _kn(derivation: Optional[dict[str, Any]]) -> str:
    return f"{derivation['k']}/{derivation['n']}" if derivation else "-"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(data: dict[str, Any]) -> str:
    lines = ["# Evidence report", ""]
    runs = data["runs"]
    lines.append(f"## Runs ({len(runs)})")
    lines.append("")
    if runs:
        lines.append(
            "| run | status | smoke | gates | selected | test spends | blind spends | environment |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for rec in runs:
            gates = rec.get("gates") or {}
            gate = "-" if not gates else ("pass" if gates.get("passed") else "fail")
            if gates and gates.get("smoke"):
                gate += " (smoke)"
            env = rec.get("environment") or {}
            env_text = "-"
            if env:
                pk = env.get("packages") or {}
                env_text = (
                    f"git {str(env.get('git_sha'))[:8]} torch {pk.get('torch')} "
                    f"cuda {env.get('cuda')}"
                )
            lines.append(
                f"| {rec['run_id']} | {rec['status']} | {_fmt(rec.get('smoke'))} | {gate} | "
                f"{rec.get('selected_checkpoint') or 'final'} | "
                f"{len(rec.get('test_spends') or [])} | {len(rec.get('blind_spends') or [])} | "
                f"{env_text} |"
            )
    for report in data["reports"]:
        lines.extend(["", f"## {report['file']}", ""])
        lines.append(
            f"- n: {report['n']}  passed: {_fmt(report['passed'])}"
            + ("  tier: smoke" if report["smoke"] else "")
        )
        if report["benchmark_version"]:
            lines.append(f"- benchmark version: `{report['benchmark_version']}`")
        if report["benchmark_sha256"]:
            lines.append(f"- split: `{report['benchmark_sha256']}`")
        if report["population_mismatch"]:
            lines.append("- **floors derived on another population**")
        lines.append("")
        lines.append("| metric | value | k / n | w95 | floor | tier | met |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for m in report["metrics"]:
            d = m["derivation"]
            lines.append(
                f"| {m['metric']} | {_fmt(m['value'])} | "
                f"{_kn(d)} | {_fmt(d['w95']) if d else '-'} | "
                f"{_fmt(m['floor'])} | {m['tier'] or '-'} | {_fmt(m['passed'])} |"
            )
        if report["slice_gates"]:
            lines.append("")
            for g in report["slice_gates"]:
                lines.append(
                    f"- {g['gate']}: actual {_fmt(g.get('actual'))} minimum "
                    f"{_fmt(g.get('minimum'))} met {_fmt(g.get('passed'))} ({g.get('tier')})"
                )
        if report["slices"]:
            lines.append("")
            for field_name, values in sorted(report["slices"].items()):
                for value, stats in sorted(values.items()):
                    n = int(stats.get("n", 0))
                    if n == 0:
                        lines.append(f"- slice {field_name}={value}: n=0")
                    else:
                        lines.append(
                            f"- slice {field_name}={value}: n={n} pass_rate "
                            f"{_fmt(stats.get('pass_rate'))} w95 {_fmt(stats.get('pass_rate_w95'))}"
                        )
        if report["pathologies"]:
            lines.append("")
            for p in report["pathologies"]:
                lines.append(f"- pathology {p.get('name')}: {p.get('evidence')}")
    for study in data["seed_studies"]:
        lines.extend(["", f"## seed study {study.get('label') or study['file']}", ""])
        lines.append(f"- seeds: {study.get('seeds')}  runs: {len(study.get('runs') or [])}")
        lines.append("")
        lines.append("| metric | mean | sd | min | max |")
        lines.append("| --- | --- | --- | --- | --- |")
        for metric, stat in sorted((study.get("stats") or {}).items()):
            lines.append(
                f"| {metric} | {_fmt(stat.get('mean'))} | {_fmt(stat.get('sd'))} | "
                f"{_fmt(stat.get('min'))} | {_fmt(stat.get('max'))} |"
            )
    return "\n".join(lines) + "\n"


def render_csv(data: dict[str, Any]) -> str:
    """One row per (report, metric): value, derivation, floor, verdict."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ["report", "n", "smoke", "metric", "value", "k", "n_metric", "w95", "floor", "tier", "met"]
    )
    for report in data["reports"]:
        for m in report["metrics"]:
            d = m["derivation"] or {}
            writer.writerow(
                [
                    report["file"],
                    report["n"],
                    _fmt(report["smoke"]),
                    m["metric"],
                    _fmt(m["value"]),
                    d.get("k", ""),
                    d.get("n", ""),
                    _fmt(d.get("w95")) if d else "",
                    _fmt(m["floor"]),
                    m["tier"] or "",
                    _fmt(m["passed"]),
                ]
            )
    return buffer.getvalue()
