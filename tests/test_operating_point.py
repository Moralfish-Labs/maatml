"""The operating point: swept on val over the predictor's rescore, spent once on test."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from maatml.cli import app
from maatml.config import load_model_def
from maatml.evaluation.harness import GateConfigError, Report
from maatml.evaluation.operating_point import (
    OperatingPointError,
    OperatingPointSpec,
    derive_operating_point,
    pick,
    report_name,
    resolve_operating_point,
    rewrite_threshold,
    sweep,
)
from maatml.evaluation.predictions import write_predictions
from maatml.registry import PREDICTORS

runner = CliRunner()


class ScorePredictor:
    """Cached rows carry ``parsed.score``; a row is a hit when score >= threshold."""

    def predict(self, row):  # pragma: no cover - never run here
        return "{}"

    def rescore(self, rows, threshold):
        positives = [r for r in rows if r["row"]["expected"] == "pos"]
        negatives = [r for r in rows if r["row"]["expected"] == "neg"]
        tp = sum(1 for r in positives if r["parsed"]["score"] >= threshold)
        fp = sum(1 for r in negatives if r["parsed"]["score"] >= threshold)
        return {
            "recall": tp / len(positives) if positives else 0.0,
            "fp_rate": fp / len(negatives) if negatives else 0.0,
            "__counts__": {"recall": [tp, len(positives)]},
        }


@pytest.fixture(autouse=True)
def _register_predictor():
    PREDICTORS.register("score_test", ScorePredictor, source="test")
    yield
    PREDICTORS.unregister("score_test")


def _cached(expected: str, score: float, dataset: str = "cam") -> dict:
    return {
        "sample_id": f"{expected}-{score}-{dataset}",
        "row": {"expected": expected, "dataset": dataset, "family": dataset},
        "output": "{}",
        "ok": True,
        "passed_layers": [1],
        "errors": [],
        "parsed": {"score": score},
        "latency_ms": 1.0,
    }


ROWS = [
    _cached("pos", 0.9),
    _cached("pos", 0.6),
    _cached("pos", 0.3),
    _cached("neg", 0.5),
    _cached("neg", 0.2),
    _cached("pos", 0.95, dataset="other"),
]

SPEC = OperatingPointSpec(
    threshold_key="score_thresh",
    objective="recall",
    budget_metric="fp_rate",
    budget_max=0.0,
    sources=("cam",),
    grid=(0.1, 0.4, 0.55, 0.7),
)


# --- spec ----------------------------------------------------------------------


def test_resolve_spec_parses_budget_sources_and_grid_forms() -> None:
    md = SimpleNamespace(
        evaluation={
            "operating_point": {
                "threshold_key": "score_thresh",
                "objective": "recall",
                "budget": {"metric": "fp_per_frame", "max": 1.0},
                "sources": ["meva", "virat"],
                "grid": {"start": 0.1, "stop": 0.3, "step": 0.1},
            }
        }
    )
    spec = resolve_operating_point(md)
    assert spec.budget_metric == "fp_per_frame" and spec.budget_max == 1.0
    assert spec.sources == ("meva", "virat")
    assert spec.grid == (0.1, 0.2, 0.3)
    plain = SimpleNamespace(
        evaluation={"operating_point": {"threshold_key": "t", "objective": "r"}}
    )
    assert resolve_operating_point(plain).grid[0] == 0.05
    assert resolve_operating_point(plain).budget_metric is None


@pytest.mark.parametrize(
    "op",
    [
        None,
        {"objective": "r"},
        {"threshold_key": "t"},
        {"threshold_key": "t", "objective": "r", "budget": {"metric": "x"}},
        {"threshold_key": "t", "objective": "r", "grid": "abc"},
        {"threshold_key": "t", "objective": "r", "grid": {"start": 1, "stop": 0, "step": 0.1}},
    ],
)
def test_malformed_spec_is_a_config_error(op) -> None:
    with pytest.raises(GateConfigError):
        resolve_operating_point(SimpleNamespace(evaluation={"operating_point": op}))


# --- sweep and pick --------------------------------------------------------------------


def test_sweep_reports_objective_budget_and_evidence_per_threshold() -> None:
    cam_rows = [r for r in ROWS if r["row"]["dataset"] == "cam"]
    points = sweep(cam_rows, ScorePredictor().rescore, SPEC)
    by_t = {p["threshold"]: p for p in points}
    assert by_t[0.1]["objective"] == 1.0 and by_t[0.1]["budget"] == 1.0
    assert by_t[0.55]["objective"] == pytest.approx(2 / 3) and by_t[0.55]["budget"] == 0.0
    assert by_t[0.55]["objective_counts"] == {"k": 2, "n": 3}
    assert 0 < by_t[0.55]["objective_w95"] < 2 / 3


def test_pick_takes_the_best_objective_under_budget_and_prefers_the_higher_cut() -> None:
    cam_rows = [r for r in ROWS if r["row"]["dataset"] == "cam"]
    points = sweep(cam_rows, ScorePredictor().rescore, SPEC)
    chosen = pick(points, SPEC)
    assert chosen is not None and chosen["threshold"] == 0.55
    loose = OperatingPointSpec(**{**SPEC.__dict__, "budget_max": 1.0})
    assert pick(points, loose)["threshold"] == 0.1
    tight = OperatingPointSpec(**{**SPEC.__dict__, "budget_max": -1.0})
    assert pick(points, tight) is None


def test_sweep_skips_thresholds_below_the_cache_decode_cut() -> None:
    points = sweep(ROWS, ScorePredictor().rescore, SPEC, floor_threshold=0.5)
    assert points[0] == {"threshold": 0.1, "skipped": "below the cache's decode cut"}
    assert "objective" in points[2]


def test_sweep_refuses_when_rescore_omits_the_objective() -> None:
    with pytest.raises(OperatingPointError, match="objective"):
        sweep(ROWS, lambda rows, t: {"fp_rate": 0.0}, SPEC)


# --- derive on a model folder --------------------------------------------------------

_MODEL_YML = """name: op-test
model_id: op-test
version: 0.1.0
architecture: causal_sft
dataset:
  format: jsonl_seed
  seed_samples: datasets/samples/seed_samples.jsonl
evaluation:
  predictor: score_test
  score_thresh: 0.05  # low decode for the sweep
  operating_point:
    threshold_key: score_thresh
    objective: recall
    budget: {metric: fp_rate, max: 0.0}
    sources: [cam]
    grid: [0.1, 0.4, 0.55, 0.7]
  gates:
    output_nonempty_rate: 0.5
training:
  base_model: sshleifer/tiny-gpt2
"""


def _model_dir(tmp_path: Path, yml: str = _MODEL_YML) -> Path:
    mdir = tmp_path / "model"
    (mdir / "datasets" / "samples").mkdir(parents=True)
    (mdir / "datasets" / "samples" / "seed_samples.jsonl").write_text(
        json.dumps({"request": "x", "expected": "y", "family": "f"}) + "\n"
    )
    (mdir / "model.yml").write_text(yml)
    return mdir


def _val_report_with_cache(
    mdir: Path, run: str, *, decode: float = 0.05, split: str = "val"
) -> Path:
    eval_dir = mdir / "output" / "eval"
    name = report_name(run, split)
    path = eval_dir / f"{name}.json"
    Report(
        model_id="op-test",
        dataset=f"output/prepared/{split}.jsonl",
        n=len(ROWS),
        metrics={"output_nonempty_rate": 1.0},
        extras={
            "split_sha256": "v" * 64,
            "predictions_cache": f"{name}.predictions.jsonl",
            "decode_threshold": {"key": "score_thresh", "value": decode},
        },
    ).write(path)
    write_predictions(
        eval_dir / f"{name}.predictions.jsonl",
        header={"split": split, "split_sha256": "v" * 64},
        rows=ROWS,
    )
    return path


def test_derive_reads_the_val_cache_filters_sources_and_picks(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    _val_report_with_cache(mdir, "run-a")
    result = derive_operating_point(load_model_def(mdir, load_plugins=False), run="run-a")
    assert result.n_rows == 5  # the "other" source is excluded
    assert result.chosen is not None and result.chosen["threshold"] == 0.55
    assert result.split_sha256 == "v" * 64
    assert result.floor_threshold == 0.05
    assert result.refusals == []
    assert "recall 0.667" in result.comment() and "fp_rate 0.000" in result.comment()


def test_derive_refuses_test_split_missing_cache_and_foreign_cache(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    md = load_model_def(mdir, load_plugins=False)
    with pytest.raises(OperatingPointError, match="spent once on test"):
        derive_operating_point(md, run="run-a", split="test")
    with pytest.raises(OperatingPointError, match="no val report"):
        derive_operating_point(md, run="run-a")
    path = _val_report_with_cache(mdir, "run-b")
    payload = json.loads(path.read_text())
    payload["extras"].pop("predictions_cache")
    path.write_text(json.dumps(payload))
    with pytest.raises(OperatingPointError, match="no predictions cache"):
        derive_operating_point(md, run="run-b")


def test_derive_names_skipped_grid_points_below_the_decode_cut(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    _val_report_with_cache(mdir, "run-a", decode=0.5)
    result = derive_operating_point(load_model_def(mdir, load_plugins=False), run="run-a")
    assert any("below the cache's decode cut 0.5" in r for r in result.refusals)
    assert result.chosen is not None and result.chosen["threshold"] == 0.55


def test_derive_needs_a_predictor_with_rescore(tmp_path: Path) -> None:
    mdir = _model_dir(
        tmp_path, _MODEL_YML.replace("predictor: score_test", "predictor: causal_sft")
    )
    _val_report_with_cache(mdir, "run-a")
    with pytest.raises(OperatingPointError, match="no rescore"):
        derive_operating_point(load_model_def(mdir, load_plugins=False), run="run-a")


# --- write ------------------------------------------------------------------------------


def test_rewrite_threshold_replaces_or_inserts_under_evaluation() -> None:
    text = (
        "evaluation:\n  predictor: p\n  score_thresh: 0.05  # old\n"
        "  gates:\n    a: 0.1\ntraining:\n  x: 1\n"
    )
    out = rewrite_threshold(text, "score_thresh", 0.55, "new")
    assert "  score_thresh: 0.55  # new\n" in out and "# old" not in out
    assert out.endswith("training:\n  x: 1\n")
    inserted = rewrite_threshold(
        "evaluation:\n  predictor: p\ntraining:\n  x: 1\n", "cut", 0.4, "c"
    )
    assert inserted.startswith("evaluation:\n  cut: 0.4  # c\n  predictor: p\n")


def test_report_name_keeps_test_bare_and_suffixes_other_splits() -> None:
    assert report_name("run-a", "test") == "run-a"
    assert report_name("run-a", "val") == "run-a.val"


# --- CLI ----------------------------------------------------------------------------------


def test_cli_derive_writes_the_threshold_with_provenance(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    _val_report_with_cache(mdir, "run-a")
    result = runner.invoke(app, ["operating-point", "derive", str(mdir), "--run", "run-a"])
    assert result.exit_code == 0, result.output
    assert "score_thresh: 0.55" in result.output
    assert (mdir / "output" / "eval" / "run-a.val.operating_point.json").is_file()
    assert "score_thresh: 0.05" in (mdir / "model.yml").read_text()

    result = runner.invoke(
        app, ["operating-point", "derive", str(mdir), "--run", "run-a", "--write"]
    )
    assert result.exit_code == 0, result.output
    text = (mdir / "model.yml").read_text()
    assert "  score_thresh: 0.55  # recall 0.667" in text
    assert "sweep run-a.val.operating_point.json" in text
    assert "# low decode for the sweep" not in text
    reloaded = load_model_def(mdir, load_plugins=False)
    assert reloaded.evaluation["score_thresh"] == 0.55


def test_cli_confirm_on_test_spends_once_and_warns_on_the_second(
    tmp_path: Path, monkeypatch
) -> None:
    from maatml.runs import RunRecord, _append_record, list_runs

    mdir = _model_dir(tmp_path)
    _val_report_with_cache(mdir, "run-a")
    md = load_model_def(mdir, load_plugins=False)
    _append_record(
        md,
        RunRecord(
            run_id="run-a",
            identity="op-test@0.1.0",
            architecture="causal_sft",
            status="completed",
            started_at="2026-01-01T00:00:00Z",
            out_dir="output/checkpoints/run-a",
        ),
    )
    seen: list[tuple[str, float]] = []

    def fake_evaluate_model(md, *, checkpoint, split, **kwargs):
        seen.append((split, md.evaluation["score_thresh"]))
        report = Report(model_id="op-test", dataset="t", n=1, extras={"split_sha256": "t" * 64})
        path = md.eval_dir / f"{checkpoint}.json"
        report.write(path)
        return report, path

    import maatml.evaluation.runner as runner_mod

    monkeypatch.setattr(runner_mod, "evaluate_model", fake_evaluate_model)
    args = ["operating-point", "derive", str(mdir), "--run", "run-a", "--confirm-on-test"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert seen == [("test", 0.55)]
    assert "confirmed on test" in result.output
    spends = list_runs(md)[0].test_spends
    assert spends and spends[0]["benchmark_sha256"] == "t" * 64
    assert spends[0]["threshold"] == 0.55

    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "already spent 1 time(s)" in result.output
    assert len(list_runs(md)[0].test_spends) == 2

    listing = runner.invoke(app, ["runs", str(mdir)])
    assert "test-spends=2" in listing.output
