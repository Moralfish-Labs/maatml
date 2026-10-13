"""Pathology signatures: reported on every evaluate, failing the smoke tier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maatml.evaluation.harness import GateConfigError, Report, RowEval, run_evaluation
from maatml.evaluation.pathologies import detect_pathologies, normalize_plugin_pathologies
from maatml.evaluation.runner import write_markdown_summary
from maatml.validation.base import ValidationResult


def _rows(n: int = 6, *, classes: tuple[str, ...] = ("a", "b")) -> list[dict]:
    return [
        {
            "sample_id": f"s{i}",
            "request": f"ask {i}",
            "expected": {"class": classes[i % len(classes)]},
            "category": classes[i % len(classes)],
        }
        for i in range(n)
    ]


def _dataset(tmp_path: Path, rows: list[dict]) -> Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir(exist_ok=True)
    with (prepared / "test.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return prepared


def _json_validator(raw_output, **_kw) -> ValidationResult:
    result = ValidationResult(raw_output=raw_output, n_layers=1, required_layers={1})
    try:
        result.parsed = json.loads(raw_output)
        result.passed_layers.add(1)
    except (json.JSONDecodeError, TypeError):
        pass
    return result


def _evaluate(tmp_path: Path, predictor, **kwargs) -> Report:
    schema = tmp_path / "schema.json"
    schema.write_text('{"type": "object"}')
    return run_evaluation(
        checkpoint_dir=tmp_path / "ckpt",
        dataset_dir=_dataset(tmp_path, _rows()),
        out_path=tmp_path / "eval" / "r.json",
        predictor=predictor,
        validator=_json_validator,
        schema_path=schema,
        device="cpu",
        **kwargs,
    )


# --- detection --------------------------------------------------------------------------


def _row_eval(text: str, parsed=None) -> RowEval:
    return RowEval(row={}, gen_text=text, result=ValidationResult(raw_output=text, parsed=parsed))


def test_never_fires_on_no_output_or_recall_zero_with_high_precision() -> None:
    rows = [_row_eval("") for _ in range(6)]
    found = detect_pathologies(rows, {"output_nonempty_rate": 0.0}, {})
    assert [p["name"] for p in found] == ["never_fires"]

    rows = [_row_eval(f"out {i}") for i in range(6)]
    found = detect_pathologies(rows, {"recall": 0.0, "precision": 1.0}, {})
    assert [p["name"] for p in found] == ["never_fires"]
    assert "recall 0.000 with precision 1.000" in found[0]["evidence"]
    assert detect_pathologies(rows, {"person_recall": 0.0}, {}) != []
    assert detect_pathologies(rows, {"recall": 0.0, "precision": 0.3}, {}) == []
    assert detect_pathologies(rows, {"recall": 0.4, "precision": 1.0}, {}) == []


def test_identical_output_and_one_class() -> None:
    rows = [_row_eval("same") for _ in range(6)]
    assert [p["name"] for p in detect_pathologies(rows, {}, {})] == ["identical_output"]

    rows = [_row_eval(f'{{"class": "a", "i": {i}}}', {"class": "a", "i": i}) for i in range(6)]
    found = detect_pathologies(rows, {}, {"a": {}, "b": {}})
    assert [p["name"] for p in found] == ["one_class"]
    assert "class='a' across 2 gold classes" in found[0]["evidence"]
    # One gold class cannot show a one-class pathology.
    assert detect_pathologies(rows, {}, {"a": {}}) == []
    rows = [_row_eval(f"x{i}", {"class": "a" if i % 2 else "b"}) for i in range(6)]
    assert detect_pathologies(rows, {}, {"a": {}, "b": {}}) == []


def test_too_few_rows_report_nothing() -> None:
    assert detect_pathologies([_row_eval("") for _ in range(4)], {}, {}) == []


def test_plugin_pathologies_are_names_or_dicts() -> None:
    out = normalize_plugin_pathologies(
        ["never_fires", {"name": "one_class", "evidence": "e"}], plugin="p"
    )
    assert out == [
        {"name": "never_fires", "evidence": "reported by p"},
        {"name": "one_class", "evidence": "e"},
    ]
    with pytest.raises(ValueError):
        normalize_plugin_pathologies("never_fires", plugin="p")
    with pytest.raises(ValueError):
        normalize_plugin_pathologies([{"evidence": "no name"}], plugin="p")


# --- evaluate -----------------------------------------------------------------------------


def test_evaluate_reports_pathologies_and_fails_the_smoke_tier(tmp_path: Path) -> None:
    report = _evaluate(
        tmp_path,
        lambda row: '{"class": "a"}',
        enforce_gates=True,
        gate_spec={"output_nonempty_rate": 0.5},
        smoke_gated=True,
    )
    names = sorted(p["name"] for p in report.pathologies)
    assert names == ["identical_output", "one_class"]
    assert report.passed is False
    results = report.gates["results"]
    assert results["output_nonempty_rate"]["passed"] is True
    assert results["pathology:identical_output"] == {
        "minimum": None,
        "actual": "identical_output",
        "passed": False,
        "tier": "blocking",
    }
    assert report.gates["pathologies"] == ["identical_output", "one_class"]
    md = write_markdown_summary(report, tmp_path / "r.md").read_text()
    assert "## Pathologies" in md and "identical_output:" in md

    production = _evaluate(
        tmp_path,
        lambda row: '{"class": "a"}',
        enforce_gates=True,
        gate_spec={"output_nonempty_rate": 0.5},
        smoke_gated=False,
    )
    assert production.passed is True
    assert production.gates["pathologies"] == ["identical_output", "one_class"]
    assert "pathology:identical_output" not in production.gates["results"]


def test_never_fires_fixture_fails_smoke_even_with_a_loose_floor(tmp_path: Path) -> None:
    report = _evaluate(
        tmp_path,
        lambda row: "",
        enforce_gates=True,
        gate_spec={"output_nonempty_rate": 0.0},
        smoke_gated=True,
    )
    assert [p["name"] for p in report.pathologies] == ["never_fires"]
    assert report.passed is False and "pathology:never_fires" in report.gates["results"]


def test_plugin_pathologies_are_lifted_out_of_metrics(tmp_path: Path) -> None:
    def metrics(row_results):
        return {"recall": 0.5, "__pathologies__": [{"name": "one_class", "evidence": "plugin"}]}

    report = _evaluate(
        tmp_path, lambda row: json.dumps({"class": "a", "r": row["sample_id"]}), metrics_fn=metrics
    )
    assert "__pathologies__" not in report.metrics
    assert {"name": "one_class", "evidence": "plugin"} in report.pathologies

    def bad(row_results):
        return {"recall": 0.5, "__pathologies__": "one_class"}

    with pytest.raises(GateConfigError, match="__pathologies__"):
        _evaluate(tmp_path, lambda row: json.dumps({"r": row["sample_id"]}), metrics_fn=bad)


def test_reports_without_the_field_still_read(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps({"report_version": 1, "model_id": "m", "dataset": "d", "n": 1, "metrics": {}})
    )
    assert Report.read(path, strict=True).pathologies == []
