"""``maatml gates derive``: floors measured from reports, never typed.

A floor is the lower bound of the observed rate at that metric's own
denominator, floored to two places, with the measurement written beside it.
Where the rows cluster (a camera, a family) and a prediction cache exists, the
bound is a cluster bootstrap rather than row-level Wilson. Thin denominators
and few clusters are refused, not floored, and every floor names the split it
came from so it is never enforced on another.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .harness import (
    COVERAGE_METRIC,
    Report,
    effective_gates,
    gate_actual,
    parse_slice_ref,
)
from .predictions import PredictionsError, predictions_path, read_predictions
from .stats import cluster_bootstrap_lower, floor2, wilson_lower

DEFAULT_MIN_N = 30
DEFAULT_MIN_GROUPS = 3
DEFAULT_BOOTSTRAP_ITERS = 1000

_LAYER_RATE = re.compile(r"layer_(\d+)_pass_rate")


class DeriveError(ValueError):
    """The reports cannot support a derivation (missing, mixed, or unversioned)."""


@dataclass(frozen=True)
class Floor:
    name: str
    value: float
    comment: str
    method: str  # wilson | cluster_bootstrap | observed
    run: str


@dataclass
class DeriveResult:
    section: str
    runs: list[str]
    benchmark_sha256: str
    floors: dict[str, Floor] = field(default_factory=dict)
    refusals: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        return [f"{f.name}: {f.value:.2f}  # {f.comment}" for f in self.floors.values()]


def _load_report(model_def: Any, run: str) -> tuple[str, Path, Report]:
    candidate = Path(run)
    if candidate.is_file():
        path, label = candidate, candidate.stem
    else:
        path, label = Path(model_def.eval_dir) / f"{run}.json", run
    if not path.is_file():
        raise DeriveError(f"no eval report for {run!r} at {path}")
    return label, path, Report.read(path, strict=True)


def _row_ok(metric: str, rec: dict[str, Any]) -> Optional[bool]:
    """Whether one cached row counts as a success for a harness-computed rate."""
    if metric == "all_layers_pass_rate":
        return bool(rec.get("ok"))
    if metric == COVERAGE_METRIC:
        return bool(str(rec.get("output") or "").strip())
    layer = _LAYER_RATE.fullmatch(metric)
    if layer:
        return int(layer.group(1)) in set(rec.get("passed_layers") or [])
    if parse_slice_ref(metric) is not None:
        return bool(rec.get("ok"))
    return None


def harness_rate_groups(
    rows: Sequence[dict[str, Any]], metric: str, cluster_by: str
) -> Optional[list[tuple[int, int]]]:
    """``(k, n)`` per ``cluster_by`` value for a harness rate, or None for plugin metrics."""
    ref = parse_slice_ref(metric)
    groups: dict[str, list[int]] = {}
    for rec in rows:
        row = rec.get("row") or {}
        if ref is not None:
            field_name, value = ref
            raw = row.get(field_name)
            if ("(absent)" if raw is None else str(raw)) != value:
                continue
        ok = _row_ok(metric, rec)
        if ok is None:
            return None
        key = row.get(cluster_by)
        bucket = groups.setdefault("(absent)" if key is None else str(key), [0, 0])
        bucket[1] += 1
        if ok:
            bucket[0] += 1
    return [(k, n) for k, n in groups.values()]


def _load_cache(path: Path, split_sha256: str, refusals: list[str], label: str) -> list[dict]:
    cache_path = predictions_path(path)
    if not cache_path.is_file():
        return []
    try:
        header, rows = read_predictions(cache_path)
    except PredictionsError as exc:
        refusals.append(f"{label}: predictions cache unusable ({exc}); row-level bounds only")
        return []
    if header.get("split_sha256") != split_sha256:
        refusals.append(
            f"{label}: predictions cache was written for another split; row-level bounds only"
        )
        return []
    return rows


def derive_gates(
    model_def: Any,
    *,
    runs: Sequence[str],
    metrics: Optional[Sequence[str]] = None,
    section: str = "evaluation",
    min_n: int = DEFAULT_MIN_N,
    min_groups: int = DEFAULT_MIN_GROUPS,
    cluster_by: str = "family",
    bootstrap_iters: int = DEFAULT_BOOTSTRAP_ITERS,
    seed: int = 42,
) -> DeriveResult:
    """Floors for ``section.gates`` from one or more runs' reports.

    With several runs the per-metric minimum is taken, so a lucky seed cannot
    set the contract. Metrics default to the gates already configured, then to
    every rate the reports carry counts for.
    """
    if not runs:
        raise DeriveError("at least one --run is required")
    if section not in ("evaluation", "smoke"):
        raise DeriveError(f"section must be evaluation or smoke; got {section!r}")
    loaded = [_load_report(model_def, run) for run in runs]

    hashes = {report.extras.get("split_sha256") for _l, _p, report in loaded}
    if None in hashes or "" in hashes:
        raise DeriveError(
            "a report lacks extras.split_sha256; re-run evaluate so floors can name their split"
        )
    if len(hashes) > 1:
        raise DeriveError(
            "reports were measured on different splits "
            f"({', '.join(sorted(str(h)[:16] for h in hashes))}); floors must share one population"
        )
    benchmark = str(next(iter(hashes)))

    names: list[str]
    if metrics:
        names = list(dict.fromkeys(metrics))
    else:
        names = list(effective_gates(model_def, smoke=section == "smoke").keys())
        if not names:
            union: set[str] = set()
            for _l, _p, report in loaded:
                union.update(report.counts.keys())
            names = sorted(union)
    if not names:
        raise DeriveError("nothing to derive: no gates configured and no counts in the reports")

    result = DeriveResult(
        section=section, runs=[label for label, _p, _r in loaded], benchmark_sha256=benchmark
    )
    per_run: dict[str, dict[str, Floor]] = {}
    for label, path, report in loaded:
        floors: dict[str, Floor] = {}
        cache_rows = _load_cache(path, benchmark, result.refusals, label)
        for name in names:
            kn = report.counts.get(name)
            if kn is None:
                actual = gate_actual(name, report.metrics, report.slices)
                if actual is None:
                    result.refusals.append(f"{label}: {name}: not in the report")
                    continue
                floors[name] = Floor(
                    name,
                    floor2(actual),
                    f"observed {actual:.3f}; no k/n recorded, so not a derived bound "
                    "(report __counts__ from the metrics plugin to bound it)",
                    "observed",
                    label,
                )
                continue
            k, n = int(kn["k"]), int(kn["n"])
            if n < min_n:
                result.refusals.append(f"{label}: {name}: {k}/{n} — n < {min_n}, too thin to floor")
                continue
            observed = k / n
            groups = harness_rate_groups(cache_rows, name, cluster_by) if cache_rows else None
            if groups is not None:
                if len(groups) < min_groups:
                    result.refusals.append(
                        f"{label}: {name}: {k}/{n} spans {len(groups)} {cluster_by} group(s) "
                        f"< {min_groups}; a row-level bound is meaningless under clustering"
                    )
                    continue
                bound = cluster_bootstrap_lower(
                    groups, iters=bootstrap_iters, seed=f"{seed}:{name}"
                )
                floors[name] = Floor(
                    name,
                    floor2(bound),
                    f"{k}/{n} = {observed:.3f}, cluster bootstrap p5 over "
                    f"{len(groups)} {cluster_by} groups {bound:.3f}",
                    "cluster_bootstrap",
                    label,
                )
            else:
                bound = wilson_lower(k, n)
                floors[name] = Floor(
                    name,
                    floor2(bound),
                    f"{k}/{n} = {observed:.3f}, w95 {bound:.3f}",
                    "wilson",
                    label,
                )
        per_run[label] = floors

    multi = len(per_run) > 1
    for name in names:
        present = [(label, floors[name]) for label, floors in per_run.items() if name in floors]
        if len(present) != len(per_run):
            continue  # the per-run refusal already says why
        label, chosen = min(present, key=lambda item: (item[1].value, item[0]))
        comment = chosen.comment
        if multi:
            comment = f"min over {len(present)} runs ({label}): {comment}"
        comment += f" @ bench {benchmark[:16]}"
        result.floors[name] = Floor(name, chosen.value, comment, chosen.method, label)
    return result


_UNQUOTED_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def render_gate_lines(result: DeriveResult, tiers: dict[str, str]) -> list[str]:
    lines: list[str] = []
    for name, floor in result.floors.items():
        key = name if _UNQUOTED_KEY.match(name) else f'"{name}"'
        value = f"{floor.value:.2f}"
        if tiers.get(name) == "advisory":
            lines.append(f"    {key}: {{min: {value}, tier: advisory}}  # {floor.comment}")
        else:
            lines.append(f"    {key}: {value}  # {floor.comment}")
    return lines


def _section_pattern(section: str) -> re.Pattern[str]:
    # The section head up to and including its `  gates:` line, then the
    # indented block beneath it.
    return re.compile(
        rf"(^{section}:\n(?:^  (?!gates:).*\n)*^  gates:.*\n)((?:^    .*\n|^\n(?=    ))*)",
        re.MULTILINE,
    )


def rewrite_gates_block(text: str, result: DeriveResult, tiers: dict[str, str]) -> str:
    """Replace ``<section>.gates`` in model.yml text and stamp ``gates_benchmark``.

    Textual, so every other line and comment in the file survives; a YAML
    round-trip would drop the comments the floors are documented with.
    """
    match = _section_pattern(result.section).search(text)
    if match is None:
        raise DeriveError(f"could not find {result.section}.gates in model.yml")
    block = "".join(line + "\n" for line in render_gate_lines(result, tiers))
    head = match.group(1)
    # An empty inline mapping (`gates: {}`) cannot take a block beneath it.
    head = re.sub(r"^  gates:.*\n\Z", "  gates:\n", head, flags=re.MULTILINE)
    stamp = f"  gates_benchmark: {result.benchmark_sha256}\n"
    if re.search(r"^  gates_benchmark:.*\n", head, re.MULTILINE):
        head = re.sub(r"^  gates_benchmark:.*\n", stamp, head, count=1, flags=re.MULTILINE)
    else:
        head = head.replace("  gates:", stamp + "  gates:", 1)
    return text[: match.start()] + head + block + text[match.end() :]


def write_gates(model_yml: Path, result: DeriveResult, tiers: dict[str, str]) -> Path:
    if not result.floors:
        raise DeriveError("refusing to write: no floor survived derivation")
    text = model_yml.read_text(encoding="utf-8")
    model_yml.write_text(rewrite_gates_block(text, result, tiers), encoding="utf-8")
    return model_yml
