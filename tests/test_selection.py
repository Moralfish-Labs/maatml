"""training.select_by: the shipped checkpoint is chosen on val, recorded, and resolved."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from maatml.cli import app
from maatml.config import load_model_def
from maatml.evaluation.runner import evaluate_model
from maatml.registry import PREDICTORS, TRAINERS
from maatml.runs import (
    RunRecord,
    _append_record,
    get_run,
    resolve_checkpoint,
    run_id_for_checkpoint,
)
from maatml.training.selection import candidate_checkpoints, select_checkpoint

runner = CliRunner()

_MODEL_YML = """name: sel-test
model_id: sel-test
version: 0.1.0
architecture: sel_test
dataset:
  format: jsonl_seed
  seed_samples: datasets/samples/seed_samples.jsonl
evaluation:
  predictor: sel_pred
  gates:
    output_nonempty_rate: 0.5
training:
  base_model: sshleifer/tiny-gpt2
{select}
"""


class QualityPredictor:
    """Answers a fraction of rows set by a marker file in the checkpoint dir."""

    def setup(self, checkpoint_dir, **kwargs):
        marker = Path(checkpoint_dir) / "quality.txt"
        self.quality = float(marker.read_text()) if marker.is_file() else 0.0
        self.seen = 0

    def predict(self, row):
        self.seen += 1
        return '{"ok": true}' if self.seen <= round(self.quality * 8) else ""


class FakeTrainer:
    def __call__(self, md, *, smoke, limit, device, seed, resume=None, trial=None):
        out = Path(md.output_dir) / "checkpoints" / "run-x"
        for name, quality in (("checkpoint-100", 0.5), ("checkpoint-200", 1.0)):
            (out / name).mkdir(parents=True, exist_ok=True)
            (out / name / "quality.txt").write_text(str(quality))
        (out / "quality.txt").write_text("0.75")
        _append_record(
            md,
            RunRecord(
                run_id="run-x",
                identity="sel-test@0.1.0",
                architecture="sel_test",
                status="completed",
                started_at="2026-01-01T00:00:00Z",
                out_dir=str(out),
                metrics={"eval_loss": 1.0},
            ),
        )
        return SimpleNamespace(out_dir=str(out), metrics={"eval_loss": 1.0})


@pytest.fixture(autouse=True)
def _registry():
    TRAINERS.register("sel_test", FakeTrainer(), source="test")
    PREDICTORS.register("sel_pred", QualityPredictor, source="test")
    yield
    TRAINERS.unregister("sel_test")
    PREDICTORS.unregister("sel_pred")


def _model_dir(tmp_path: Path, *, select: str = "  select_by: output_nonempty_rate") -> Path:
    mdir = tmp_path / "model"
    (mdir / "datasets" / "samples").mkdir(parents=True)
    (mdir / "datasets" / "samples" / "seed_samples.jsonl").write_text("")
    prepared = mdir / "output" / "prepared"
    prepared.mkdir(parents=True)
    for split in ("val", "test"):
        with (prepared / f"{split}.jsonl").open("w") as handle:
            for i in range(8):
                handle.write(
                    json.dumps({"sample_id": f"{split}{i}", "request": "ask", "expected": "ok"})
                    + "\n"
                )
    (mdir / "model.yml").write_text(_MODEL_YML.replace("{select}", select))
    return mdir


def test_candidates_are_steps_in_order_then_final(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for name in ("checkpoint-1000", "checkpoint-200", "notes", "checkpoint-x"):
        (run / name).mkdir(parents=True)
    assert [n for n, _ in candidate_checkpoints(run)] == [
        "checkpoint-200",
        "checkpoint-1000",
        "final",
    ]


def test_train_selects_the_best_val_checkpoint_and_the_run_resolves_to_it(
    tmp_path: Path,
) -> None:
    mdir = _model_dir(tmp_path)
    result = runner.invoke(app, ["train", str(mdir), "--device", "cpu"])
    assert result.exit_code == 0, result.output
    assert "selected checkpoint-200" in result.output
    md = load_model_def(mdir, load_plugins=False)
    rec = get_run(md, "run-x")
    assert rec is not None and rec.selected_checkpoint == "checkpoint-200"
    assert rec.selection["metric"] == "output_nonempty_rate"
    assert [c["name"] for c in rec.selection["candidates"]] == [
        "checkpoint-100",
        "checkpoint-200",
        "final",
    ]
    assert [c["value"] for c in rec.selection["candidates"]] == [0.5, 1.0, 0.75]
    select_dir = md.eval_dir / "select" / "run-x"
    assert {p.name for p in select_dir.glob("*.val.json")} == {
        "checkpoint-100.val.json",
        "checkpoint-200.val.json",
        "final.val.json",
    }

    chosen = resolve_checkpoint(md, "run-x")
    assert chosen.name == "checkpoint-200"
    assert resolve_checkpoint(md).name == "checkpoint-200"
    assert run_id_for_checkpoint(md, chosen) == "run-x"

    # Evidence lands on the run, not on the subdirectory.
    report, path = evaluate_model(md, gate=True, device="cpu")
    assert path.name == "run-x.json" and report.passed is True
    assert report.metrics["output_nonempty_rate"] == 1.0
    assert (get_run(md, "run-x").gates or {}).get("passed") is True
    assert not (md.eval_dir / "test.jsonl").exists()


def test_ties_go_to_the_later_checkpoint_and_final_needs_no_subdir(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    md = load_model_def(mdir, load_plugins=False)
    FakeTrainer()(md, smoke=False, limit=None, device="cpu", seed=0)
    run = Path(get_run(md, "run-x").out_dir)
    (run / "quality.txt").write_text("1.0")
    selection = select_checkpoint(md, "run-x", device="cpu")
    assert selection is not None and selection["selected"] == "final"
    rec = get_run(md, "run-x")
    assert rec.selected_checkpoint is None
    assert resolve_checkpoint(md, "run-x") == run.resolve()


def test_select_by_is_optional_and_must_name_a_reported_metric(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path, select="")
    md = load_model_def(mdir, load_plugins=False)
    FakeTrainer()(md, smoke=False, limit=None, device="cpu", seed=0)
    assert select_checkpoint(md, "run-x", device="cpu") is None
    assert get_run(md, "run-x").selection is None

    mdir = _model_dir(tmp_path / "b", select="  select_by: map50")
    md = load_model_def(mdir, load_plugins=False)
    FakeTrainer()(md, smoke=False, limit=None, device="cpu", seed=0)
    with pytest.raises(ValueError, match="map50.*not a metric"):
        select_checkpoint(md, "run-x", device="cpu")


def test_seed_study_carries_the_selected_value(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    result = runner.invoke(app, ["train", str(mdir), "--seeds", "2", "--device", "cpu"])
    assert result.exit_code == 0, result.output
    study = json.loads((mdir / "output" / "seeds" / "run-x-x2.json").read_text())
    assert study["stats"]["select:output_nonempty_rate"]["min"] == 1.0


def test_select_by_is_not_a_trainer_key() -> None:
    from maatml.training.sft_config import reject_unknown_training_keys

    reject_unknown_training_keys(
        {"model_id": "x", "select_by": "m"}, frozenset({"model_id"}), architecture="t"
    )
    with pytest.raises(ValueError, match="keep_me"):
        reject_unknown_training_keys(
            {"model_id": "x", "keep_me": 1}, frozenset({"model_id"}), architecture="t"
        )
