"""Evidence layer, first slice: versioned reports, slices, the prediction cache."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from maatml.evaluation.harness import (
    REPORT_VERSION,
    Report,
    ReportSchemaError,
    RowEval,
    SliceSpec,
    resolve_slices,
    run_evaluation,
    slice_buckets,
)
from maatml.evaluation.predictions import (
    PREDICTIONS_KIND,
    PredictionsError,
    predictions_path,
    read_predictions,
)
from maatml.evaluation.runner import write_markdown_summary
from maatml.evaluation.stats import wilson_interval, wilson_lower
from maatml.utils.io import sha256_file
from maatml.validation.base import ValidationError, ValidationResult

# --- stats -----------------------------------------------------------------


def test_wilson_lower_matches_known_values() -> None:
    # The floors a production folder derived by hand, reproduced here.
    assert wilson_lower(69087, 219079) == pytest.approx(0.313, abs=5e-4)
    assert wilson_lower(77, 146) == pytest.approx(0.447, abs=5e-4)
    assert wilson_lower(16, 16) == pytest.approx(0.806, abs=5e-4)


def test_wilson_interval_is_bounded_and_refuses_no_evidence() -> None:
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = wilson_interval(10, 10)
    assert 0.0 < lo < 1.0 and hi == 1.0
    with pytest.raises(ValueError, match="n > 0"):
        wilson_lower(0, 0)
    with pytest.raises(ValueError, match="\\[0, n\\]"):
        wilson_lower(5, 3)


# --- slices ----------------------------------------------------------------


def _row_eval(fields: dict, ok: bool) -> RowEval:
    result = ValidationResult(raw_output="{}", n_layers=1)
    if ok:
        result.passed_layers.add(1)
    else:
        result.errors.append(ValidationError(layer=1, code="bad", message="nope"))
    return RowEval(row=fields, gen_text="{}", result=result)


def test_slice_buckets_report_rate_and_bound_per_value() -> None:
    rows = [
        _row_eval({"camera": "G339"}, True),
        _row_eval({"camera": "G339"}, False),
        _row_eval({"camera": "G341"}, True),
    ]
    out = slice_buckets(rows, [SliceSpec(field="camera")])
    assert out["camera"]["G339"]["n"] == 2.0
    assert out["camera"]["G339"]["pass_rate"] == 0.5
    assert out["camera"]["G339"]["pass_rate_w95"] == pytest.approx(wilson_lower(1, 2))
    assert out["camera"]["G341"]["pass_rate"] == 1.0


def test_declared_slice_value_with_no_rows_reports_n_zero_and_no_rate() -> None:
    rows = [_row_eval({"spectrum": "rgb"}, True)]
    out = slice_buckets(rows, [SliceSpec(field="spectrum", values=("rgb", "ir"))])
    assert out["spectrum"]["ir"] == {"n": 0.0}
    assert "pass_rate" not in out["spectrum"]["ir"]
    assert out["spectrum"]["rgb"]["pass_rate"] == 1.0


def test_rows_without_the_field_are_counted_under_absent() -> None:
    rows = [_row_eval({"camera": "G339"}, True), _row_eval({}, False)]
    out = slice_buckets(rows, [SliceSpec(field="camera")])
    assert out["camera"]["(absent)"]["n"] == 1.0
    assert out["camera"]["(absent)"]["pass_rate"] == 0.0


def test_resolve_slices_accepts_names_and_declared_values() -> None:
    md = SimpleNamespace(
        evaluation={"slices": ["camera", {"field": "spectrum", "values": ["rgb", "ir"]}]}
    )
    specs = resolve_slices(md)
    assert specs == [
        SliceSpec(field="camera"),
        SliceSpec(field="spectrum", values=("rgb", "ir")),
    ]
    assert resolve_slices(SimpleNamespace(evaluation={})) == []


@pytest.mark.parametrize(
    "spec",
    [
        "camera",
        [{"values": ["a"]}],
        [{"field": "x", "values": "a"}],
        ["camera", "camera"],
    ],
)
def test_resolve_slices_rejects_malformed_entries(spec) -> None:
    from maatml.evaluation.harness import GateConfigError

    with pytest.raises(GateConfigError):
        resolve_slices(SimpleNamespace(evaluation={"slices": spec}))


# --- report version ----------------------------------------------------------


def test_new_reports_carry_the_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    Report(model_id="m", dataset="d", n=1, metrics={"a": 1.0}).write(path)
    assert json.loads(path.read_text())["report_version"] == REPORT_VERSION
    assert Report.read(path, strict=True).report_version == REPORT_VERSION


def test_legacy_report_reads_as_version_zero_and_fails_strict(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"model_id": "m", "dataset": "d", "n": 3, "metrics": {"a": 0.5}}))
    assert Report.read(path).report_version == 0
    with pytest.raises(ReportSchemaError, match="predates"):
        Report.read(path, strict=True)


def test_strict_read_names_the_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"report_version": 1, "model_id": "m"}))
    with pytest.raises(ReportSchemaError, match="dataset"):
        Report.read(path, strict=True)


# --- prediction cache ----------------------------------------------------------


def _dataset(tmp_path: Path, rows: list[dict]) -> Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    with (prepared / "test.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return prepared


def test_evaluate_cache_writes_predictions_keyed_to_the_split(tmp_path: Path) -> None:
    rows = [
        {"sample_id": "a", "request": "ask a", "expected": "ok", "camera": "G339"},
        {"sample_id": "b", "request": "ask b", "expected": "ok", "camera": "G341"},
    ]
    dataset_dir = _dataset(tmp_path, rows)
    out = tmp_path / "eval" / "run1.json"
    report = run_evaluation(
        checkpoint_dir=tmp_path / "ckpt",
        dataset_dir=dataset_dir,
        out_path=out,
        predictor=lambda row: '{"answer": "%s"}' % row["sample_id"],
        device="cpu",
        slices=[SliceSpec(field="camera")],
        cache_predictions=True,
    )
    split_hash = sha256_file(dataset_dir / "test.jsonl")
    assert report.extras["split_sha256"] == split_hash
    assert report.extras["predictions_cache"] == "run1.predictions.jsonl"
    assert report.slices["camera"]["G339"]["n"] == 1.0

    header, cached = read_predictions(predictions_path(out))
    assert header["kind"] == PREDICTIONS_KIND
    assert header["n"] == 2
    assert header["split"] == "test"
    assert header["split_sha256"] == split_hash
    assert header["report"] == "run1.json"
    assert [c["sample_id"] for c in cached] == ["a", "b"]
    assert cached[0]["ok"] is True
    assert cached[0]["parsed"] == {"answer": "a"}
    assert cached[0]["row"]["camera"] == "G339"
    # The request payload is what can be large; everything else stays.
    assert "request" not in cached[0]["row"]
    assert cached[0]["row"]["expected"] == "ok"


def test_evaluate_without_cache_writes_no_predictions_file(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path, [{"request": "x", "expected": "ok"}])
    out = tmp_path / "eval" / "run1.json"
    report = run_evaluation(
        checkpoint_dir=tmp_path / "ckpt",
        dataset_dir=dataset_dir,
        out_path=out,
        predictor=lambda row: "{}",
        device="cpu",
    )
    assert "predictions_cache" not in report.extras
    assert "split_sha256" in report.extras
    assert not predictions_path(out).exists()


def test_torn_or_foreign_predictions_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.predictions.jsonl"
    path.write_text(json.dumps({"kind": PREDICTIONS_KIND, "n": 2}) + "\n" + "{}\n")
    with pytest.raises(PredictionsError, match="torn"):
        read_predictions(path)
    path.write_text(json.dumps({"kind": "something/else", "n": 0}) + "\n")
    with pytest.raises(PredictionsError, match="not a"):
        read_predictions(path)
    path.write_text("")
    with pytest.raises(PredictionsError, match="empty"):
        read_predictions(path)


# --- markdown ------------------------------------------------------------------


def test_markdown_renders_slices_including_empty_ones(tmp_path: Path) -> None:
    report = Report(
        model_id="m",
        dataset="d",
        n=2,
        slices={
            "spectrum": {
                "rgb": {"n": 2.0, "passed": 1.0, "pass_rate": 0.5, "pass_rate_w95": 0.095},
                "ir": {"n": 0.0},
            }
        },
    )
    body = write_markdown_summary(report, tmp_path / "r.md").read_text()
    assert "## Slices" in body
    assert "- spectrum=rgb: n=2 pass_rate=0.500 w95=0.095" in body
    assert "- spectrum=ir: n=0 (no rate)" in body


def test_known_keys_accept_slices_and_cache(tmp_path: Path) -> None:
    from maatml.config import config_key_warnings

    md = SimpleNamespace(
        dataset={},
        evaluation={"slices": ["camera"], "cache_predictions": True, "bogus": 1},
    )
    warnings = config_key_warnings(md)  # type: ignore[arg-type]
    assert warnings == ["evaluation.bogus: unrecognized key, ignored by known stages"]

    # The key operating_point.threshold_key names is one maatml itself writes.
    md.evaluation["operating_point"] = {"threshold_key": "bogus", "objective": "recall"}
    assert config_key_warnings(md) == []  # type: ignore[arg-type]
