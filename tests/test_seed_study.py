"""train --seeds N: one recipe, the spread in one record, consumable by gates derive."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from maatml.cli import app
from maatml.evaluation.harness import Report
from maatml.registry import TRAINERS
from maatml.training.seeds import load_seed_study, summarize_seed_runs

runner = CliRunner()

_MODEL_YML = """name: seed-test
model_id: seed-test
version: 0.1.0
architecture: seed_test
dataset:
  format: jsonl_seed
  seed_samples: datasets/samples/seed_samples.jsonl
evaluation:
  predictor: causal_sft
  gates:
    all_layers_pass_rate: 0.5
training:
  base_model: sshleifer/tiny-gpt2
"""


def _model_dir(tmp_path: Path) -> Path:
    mdir = tmp_path / "model"
    (mdir / "datasets" / "samples").mkdir(parents=True)
    (mdir / "datasets" / "samples" / "seed_samples.jsonl").write_text(
        json.dumps({"request": "x", "expected": "y", "family": "f"}) + "\n"
    )
    (mdir / "model.yml").write_text(_MODEL_YML)
    return mdir


class FakeTrainer:
    calls: list[dict] = []

    def __call__(self, md, *, smoke, limit, device, seed, resume=None, trial=None):
        self.calls.append({"seed": seed, "smoke": smoke})
        out = Path(md.output_dir) / "checkpoints" / f"run-s{seed}"
        out.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            out_dir=str(out), metrics={"eval_loss": 1.0 + seed / 10, "epoch": 1.0}
        )


@pytest.fixture(autouse=True)
def _trainer():
    FakeTrainer.calls = []
    TRAINERS.register("seed_test", FakeTrainer(), source="test")
    yield
    TRAINERS.unregister("seed_test")


def test_summary_uses_metrics_every_run_reports_and_sample_sd() -> None:
    runs = [
        {"metrics": {"a": 1.0, "b": 2.0}},
        {"metrics": {"a": 3.0}},
        {"metrics": {"a": 2.0, "b": 4.0, "c": float("nan")}},
    ]
    stats = summarize_seed_runs(runs)
    assert set(stats) == {"a"}
    assert stats["a"]["mean"] == 2.0 and stats["a"]["min"] == 1.0 and stats["a"]["max"] == 3.0
    assert stats["a"]["sd"] == pytest.approx(1.0)
    assert summarize_seed_runs([{"metrics": {"a": 5.0}}])["a"]["sd"] == 0.0
    assert summarize_seed_runs([]) == {}


def test_train_seeds_runs_the_recipe_n_times_and_records_the_spread(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    result = runner.invoke(
        app, ["train", str(mdir), "--seeds", "3", "--seed", "7", "--device", "cpu"]
    )
    assert result.exit_code == 0, result.output
    assert [c["seed"] for c in FakeTrainer.calls] == [7, 8, 9]
    study_path = mdir / "output" / "seeds" / "run-s7-x3.json"
    study = load_seed_study(study_path)
    assert [r["run_id"] for r in study["runs"]] == ["run-s7", "run-s8", "run-s9"]
    assert study["stats"]["eval_loss"]["min"] == pytest.approx(1.7)
    assert study["stats"]["eval_loss"]["max"] == pytest.approx(1.9)
    assert "eval_loss: mean 1.8000" in result.output

    result = runner.invoke(app, ["train", str(mdir), "--seeds", "1"])
    assert result.exit_code != 0 and "at least 2" in result.output


def test_gates_derive_takes_runs_from_a_seed_study(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    runner.invoke(app, ["train", str(mdir), "--seeds", "2", "--device", "cpu"])
    for run, k in (("run-s0", 90), ("run-s1", 70)):
        Report(
            model_id="seed-test",
            dataset="d",
            n=100,
            metrics={"all_layers_pass_rate": k / 100},
            counts={"all_layers_pass_rate": {"k": k, "n": 100}},
            extras={"split_sha256": "a" * 64},
        ).write(mdir / "output" / "eval" / f"{run}.json")
    study = mdir / "output" / "seeds" / "run-s0-x2.json"
    result = runner.invoke(app, ["gates", "derive", str(mdir), "--seed-study", str(study)])
    assert result.exit_code == 0, result.output
    assert "min over 2 runs (run-s1)" in result.output

    result = runner.invoke(app, ["gates", "derive", str(mdir)])
    assert result.exit_code != 0
    # typer's rich error box wraps and colours the message at the CI terminal width.
    plain = re.sub(r"\x1b\[[0-9;]*m|[│╭╮╰╯─\s]", "", result.output)
    assert "--seed-study" in plain
