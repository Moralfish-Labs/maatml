"""evaluate batch_size: predict_batch is fed chunks, outputs stay per-row, latency is amortized."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from maatml.cli import app
from maatml.config import load_model_def
from maatml.evaluation.harness import GateConfigError
from maatml.evaluation.runner import evaluate_model
from maatml.registry import PREDICTORS
from maatml.runs import RunRecord, _append_record

runner = CliRunner()

_MODEL_YML = """name: batch-test
model_id: batch-test
version: 0.1.0
architecture: batch_test
dataset:
  format: jsonl_seed
  seed_samples: datasets/samples/seed_samples.jsonl
evaluation:
  predictor: {predictor}
  gates:
    output_nonempty_rate: 0.5
{extra}training:
  base_model: sshleifer/tiny-gpt2
"""

CALLS: list[int] = []


class BatchPredictor:
    def setup(self, checkpoint_dir, **kwargs):
        pass

    def predict(self, row):
        CALLS.append(1)
        return json.dumps({"id": row["sample_id"]})

    def predict_batch(self, rows):
        CALLS.append(len(rows))
        return [json.dumps({"id": row["sample_id"]}) for row in rows]


class ShortBatchPredictor(BatchPredictor):
    def predict_batch(self, rows):
        return [json.dumps({"id": rows[0]["sample_id"]})]


class RowOnlyPredictor:
    def setup(self, checkpoint_dir, **kwargs):
        pass

    def predict(self, row):
        CALLS.append(1)
        return json.dumps({"id": row["sample_id"]})


@pytest.fixture(autouse=True)
def _registry():
    CALLS.clear()
    PREDICTORS.register("batch_pred", BatchPredictor, source="test")
    PREDICTORS.register("short_pred", ShortBatchPredictor, source="test")
    PREDICTORS.register("row_pred", RowOnlyPredictor, source="test")
    yield
    for name in ("batch_pred", "short_pred", "row_pred"):
        PREDICTORS.unregister(name)


def _model_dir(tmp_path: Path, *, predictor: str = "batch_pred", extra: str = "") -> Path:
    mdir = tmp_path / "model"
    (mdir / "datasets" / "samples").mkdir(parents=True)
    (mdir / "datasets" / "samples" / "seed_samples.jsonl").write_text("")
    prepared = mdir / "output" / "prepared"
    prepared.mkdir(parents=True)
    with (prepared / "test.jsonl").open("w") as handle:
        for i in range(10):
            handle.write(
                json.dumps({"sample_id": f"t{i}", "request": "ask", "expected": "ok"}) + "\n"
            )
    (mdir / "model.yml").write_text(_MODEL_YML.format(predictor=predictor, extra=extra))
    run = mdir / "output" / "checkpoints" / "run-b"
    run.mkdir(parents=True)
    md = load_model_def(mdir, load_plugins=False)
    _append_record(
        md,
        RunRecord(
            run_id="run-b",
            identity="batch-test@0.1.0",
            architecture="batch_test",
            status="completed",
            started_at="2026-01-01T00:00:00Z",
            out_dir=str(run),
        ),
    )
    return mdir


def test_batch_size_feeds_chunks_and_keeps_outputs_per_row(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path, extra="  batch_size: 4\n")
    md = load_model_def(mdir, load_plugins=False)
    report, path = evaluate_model(md, device="cpu", cache_predictions=True)
    assert CALLS == [4, 4, 2]
    assert report.metrics["output_nonempty_rate"] == 1.0
    assert report.extras["batch_size"] == 4 and report.extras["latency_amortized"] is True
    assert report.latency_ms is not None and report.latency_ms.n == 10
    cache = path.with_name("run-b.predictions.jsonl").read_text().splitlines()[1:]
    assert [json.loads(line)["sample_id"] for line in cache] == [f"t{i}" for i in range(10)]
    assert all(
        json.loads(json.loads(line)["output"])["id"] == json.loads(line)["sample_id"]
        for line in cache
    )


def test_cli_batch_size_overrides_the_file_and_is_not_an_override(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    result = runner.invoke(
        app, ["evaluate", str(mdir), "--gate", "--device", "cpu", "--batch-size", "3"]
    )
    assert result.exit_code == 0, result.output
    assert CALLS == [3, 3, 3, 1]
    report = json.loads((mdir / "output" / "eval" / "run-b.json").read_text())
    assert report["extras"]["batch_size"] == 3 and "overrides" not in report["extras"]
    assert report["passed"] is True


def test_default_is_one_row_per_call_without_amortized_latency(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    report, _ = evaluate_model(load_model_def(mdir, load_plugins=False), device="cpu")
    assert CALLS == [1] * 10
    assert report.extras["batch_size"] == 1 and "latency_amortized" not in report.extras


def test_predictor_without_predict_batch_falls_back_with_a_warning(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path, predictor="row_pred", extra="  batch_size: 8\n")
    result = runner.invoke(app, ["evaluate", str(mdir), "--device", "cpu"])
    assert result.exit_code == 0, result.output
    assert "no predict_batch" in result.output.replace("\n", " ")
    assert CALLS == [1] * 10
    report = json.loads((mdir / "output" / "eval" / "run-b.json").read_text())
    assert report["extras"]["batch_size"] == 1


def test_short_batch_output_is_refused_and_bad_sizes_are_config_errors(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path, predictor="short_pred", extra="  batch_size: 4\n")
    with pytest.raises(ValueError, match="one string per row"):
        evaluate_model(load_model_def(mdir, load_plugins=False), device="cpu")

    mdir = _model_dir(tmp_path / "b", extra="  batch_size: 0\n")
    with pytest.raises(GateConfigError, match="batch_size"):
        evaluate_model(load_model_def(mdir, load_plugins=False), device="cpu")
    with pytest.raises(GateConfigError, match="batch-size"):
        evaluate_model(
            load_model_def(_model_dir(tmp_path / "c"), load_plugins=False),
            device="cpu",
            batch_size=0,
        )
