"""evaluate --set: an override reaches the predictor, is recorded, and is never gate evidence."""

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

_MODEL_YML = """name: ov-test
model_id: ov-test
version: 0.1.0
architecture: ov_test
dataset:
  format: jsonl_seed
  seed_samples: datasets/samples/seed_samples.jsonl
evaluation:
  predictor: ov_pred
  score_thresh: 0.4
  operating_point:
    threshold_key: score_thresh
    objective: output_nonempty_rate
  gates:
    output_nonempty_rate: 0.5
training:
  base_model: sshleifer/tiny-gpt2
"""


class ThresholdEchoPredictor:
    """Answers with the cut it was configured with, read from the model_def object."""

    def setup(self, checkpoint_dir, *, model_def=None, **kwargs):
        self.cut = float((model_def.evaluation or {}).get("score_thresh", 0.0))

    def predict(self, row):
        return json.dumps({"decode": {"score_thresh": self.cut}})


@pytest.fixture(autouse=True)
def _registry():
    PREDICTORS.register("ov_pred", ThresholdEchoPredictor, source="test")
    yield
    PREDICTORS.unregister("ov_pred")


def _model_dir(tmp_path: Path) -> Path:
    mdir = tmp_path / "model"
    (mdir / "datasets" / "samples").mkdir(parents=True)
    (mdir / "datasets" / "samples" / "seed_samples.jsonl").write_text("")
    prepared = mdir / "output" / "prepared"
    prepared.mkdir(parents=True)
    for split in ("val", "test"):
        with (prepared / f"{split}.jsonl").open("w") as handle:
            for i in range(6):
                handle.write(
                    json.dumps({"sample_id": f"{split}{i}", "request": "ask", "expected": "ok"})
                    + "\n"
                )
    (mdir / "model.yml").write_text(_MODEL_YML)
    run = mdir / "output" / "checkpoints" / "run-o"
    run.mkdir(parents=True)
    md = load_model_def(mdir, load_plugins=False)
    _append_record(
        md,
        RunRecord(
            run_id="run-o",
            identity="ov-test@0.1.0",
            architecture="ov_test",
            status="completed",
            started_at="2026-01-01T00:00:00Z",
            out_dir=str(run),
        ),
    )
    return mdir


def test_set_reaches_the_predictor_and_is_recorded_on_the_report(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(mdir),
            "--split",
            "val",
            "--cache",
            "--device",
            "cpu",
            "--set",
            "evaluation.score_thresh=0.05",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((mdir / "output" / "eval" / "run-o.val.json").read_text())
    assert report["extras"]["overrides"] == {"evaluation.score_thresh": 0.05}
    assert report["extras"]["decode_threshold"] == {"key": "score_thresh", "value": 0.05}
    cache = (mdir / "output" / "eval" / "run-o.val.predictions.jsonl").read_text().splitlines()
    assert json.loads(json.loads(cache[1])["output"])["decode"]["score_thresh"] == 0.05
    # model.yml itself is untouched.
    assert "score_thresh: 0.4" in (mdir / "model.yml").read_text()

    report, _path = evaluate_model(load_model_def(mdir, load_plugins=False), device="cpu")
    assert "overrides" not in report.extras
    assert report.extras["decode_threshold"]["value"] == 0.4


def test_set_is_refused_with_gate_and_blind(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    result = runner.invoke(
        app,
        ["evaluate", str(mdir), "--gate", "--device", "cpu", "--set", "evaluation.gates.x=0"],
    )
    assert result.exit_code != 0
    assert "not gate evidence" in result.output.replace("\n", " ")
    assert not (mdir / "output" / "eval").exists()

    md = load_model_def(mdir, load_plugins=False)
    with pytest.raises(GateConfigError, match="gate evidence"):
        evaluate_model(md, device="cpu", blind=True, overrides={"evaluation.score_thresh": 0.1})


def test_set_with_an_unknown_path_exits_before_loading(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    result = runner.invoke(app, ["evaluate", str(mdir), "--set", "nonsense.key=1"])
    assert result.exit_code == 2
    assert "nonsense" in result.output
