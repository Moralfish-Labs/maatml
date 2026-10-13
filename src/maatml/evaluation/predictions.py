"""Per-row prediction cache written beside an eval report.

``evaluate --cache`` keeps what the predictor said for every row, with the
validator's verdict and the row's own metadata, keyed to the split it was
measured on. Floor derivation and threshold sweeps read this file instead of
re-running inference, so a derived number always comes from the same
predictions the report did.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..utils.io import write_jsonl_atomic

PREDICTIONS_KIND = "maatml.predictions/1"


class PredictionsError(ValueError):
    """The cache file is not one this version of maatml wrote, or is torn."""


def predictions_path(report_path: str | Path) -> Path:
    report_path = Path(report_path)
    return report_path.with_name(report_path.stem + ".predictions.jsonl")


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return None
    return value


def prediction_row(
    *,
    row: dict[str, Any],
    gen_text: str,
    result: Any,
    latency_ms: float,
    drop_fields: Iterable[str] = (),
) -> dict[str, Any]:
    """One cache line: the row minus its request payload, plus the verdict."""
    dropped = set(drop_fields)
    kept = {k: v for k, v in row.items() if k not in dropped}
    return {
        "sample_id": row.get("sample_id"),
        "row": kept,
        "output": gen_text,
        "ok": bool(result.ok),
        "passed_layers": sorted(int(layer) for layer in result.passed_layers),
        "errors": [
            {"layer": e.layer, "code": e.code, "message": e.message, "location": e.location}
            for e in result.errors
        ],
        "parsed": _json_safe(getattr(result, "parsed", None)),
        "latency_ms": float(latency_ms),
    }


def write_predictions(
    path: str | Path, *, header: dict[str, Any], rows: Iterable[dict[str, Any]]
) -> Path:
    rows = list(rows)
    head = {"kind": PREDICTIONS_KIND, "n": len(rows), **header}

    def _lines() -> Iterator[dict[str, Any]]:
        yield head
        yield from rows

    return write_jsonl_atomic(path, _lines())


def read_predictions(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Header and rows, refusing a file that is not a complete predictions cache."""
    path = Path(path)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise PredictionsError(f"{path}: empty predictions cache")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise PredictionsError(f"{path}: unreadable header: {exc}") from exc
    if not isinstance(header, dict) or header.get("kind") != PREDICTIONS_KIND:
        raise PredictionsError(
            f"{path}: not a {PREDICTIONS_KIND} file (kind={header.get('kind')!r})"
            if isinstance(header, dict)
            else f"{path}: header is not an object"
        )
    rows = [json.loads(line) for line in lines[1:]]
    expected = header.get("n")
    if expected != len(rows):
        raise PredictionsError(
            f"{path}: header says n={expected} but {len(rows)} rows follow; "
            "the cache is torn or was edited"
        )
    return header, rows
