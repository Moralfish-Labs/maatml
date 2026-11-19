"""JobProgress writer schema compatibility and CLI/lifecycle integration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from maatml.cli import app
from maatml.progress import (
    PROGRESS_FILE_ENV,
    JobProgressReporter,
    make_progress_callback,
    pick_progress_metric,
    resolve_progress_file,
)
from maatml.registry import TRAINERS


def test_resolve_progress_file_explicit_wins_over_env(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / "env.json"
    cli_path = tmp_path / "cli.json"
    monkeypatch.setenv(PROGRESS_FILE_ENV, str(env_path))
    assert resolve_progress_file(cli_path) == cli_path
    assert resolve_progress_file(None) == env_path
    monkeypatch.delenv(PROGRESS_FILE_ENV, raising=False)
    assert resolve_progress_file(None) is None


def test_job_progress_reporter_writes_camel_case_schema_atomically(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "progress.json"
    reporter = JobProgressReporter(out, run_id="run-123", artifact_uri="/tmp/out")
    reporter.record_checkpoint(at=100.0)
    # Non-finite metrics must be ignored so serde_json never sees NaN/Inf.
    reporter.update(
        stage="train", epoch=1.8, step=42, metric_name="loss", metric_value=float("nan")
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["stage"] == "train"
    assert data["epoch"] == 1 and isinstance(data["epoch"], int)
    assert data["step"] == 42 and isinstance(data["step"], int)
    assert data["runId"] == "run-123"
    assert data["artifactUri"] == "/tmp/out"
    assert data["heartbeatAt"].endswith("Z")
    assert "metricName" not in data and "metricValue" not in data
    assert isinstance(data["checkpointAgeSecs"], int) and data["checkpointAgeSecs"] >= 0
    assert not list(out.parent.glob("*.tmp"))

    reporter.update(stage="evaluate", metric_name="eval_loss", metric_value=0.42)
    data2 = json.loads(out.read_text(encoding="utf-8"))
    assert data2["metricName"] == "eval_loss"
    assert math.isclose(data2["metricValue"], 0.42)


def test_pick_progress_metric_priority() -> None:
    assert pick_progress_metric({"loss": 1.2, "eval_loss": 0.9}) == ("eval_loss", 0.9)
    assert pick_progress_metric({"learning_rate": 1e-4, "loss": 0.5}) == ("loss", 0.5)
    assert pick_progress_metric({"epoch": 1.0, "accuracy": 0.85}) == ("accuracy", 0.85)
    assert pick_progress_metric({"loss": float("inf"), "grad_norm": 1.5}) == ("grad_norm", 1.5)


def test_make_progress_callback_updates_progress_file(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    out = tmp_path / "progress.json"
    reporter = JobProgressReporter(out, run_id="r1")
    cb = make_progress_callback(reporter)

    state = SimpleNamespace(epoch=2.0, global_step=15)
    cb.on_train_begin(None, state, None)
    assert json.loads(out.read_text())["stage"] == "train"

    cb.on_log(None, state, None, logs={"loss": 0.33, "epoch": 2.0})
    cb.on_save(None, state, None)
    saved = json.loads(out.read_text())
    assert saved["stage"] == "checkpoint"
    assert saved["metricName"] == "loss"
    assert saved["metricValue"] == pytest.approx(0.33)
    assert saved["checkpointAgeSecs"] == 0

    cb.on_evaluate(None, state, None, metrics={"eval_loss": 0.25})
    assert json.loads(out.read_text())["stage"] == "evaluate"


def test_cli_train_passes_progress_file_and_reads_env(tmp_path: Path, monkeypatch) -> None:
    mdir = tmp_path / "model"
    mdir.mkdir()
    (mdir / "model.yml").write_text(
        "name: prog\nmodel_id: prog\narchitecture: prog_arch\nversion: 0.1.0\n",
        encoding="utf-8",
    )
    seen: list[Path | None] = []

    def _fake_trainer(md, **kwargs):
        seen.append(kwargs.get("progress_file"))
        return SimpleNamespace(out_dir=mdir / "output" / "checkpoints" / "r1", metrics={})

    TRAINERS.register("prog_arch", _fake_trainer, source="test")
    monkeypatch.setattr("maatml.training.selection.select_checkpoint", lambda *a, **kw: None)
    runner = CliRunner()
    try:
        p1 = tmp_path / "p1.json"
        res1 = runner.invoke(app, ["train", str(mdir), "--progress-file", str(p1)])
        assert res1.exit_code == 0, res1.output

        p2 = tmp_path / "p2.json"
        monkeypatch.setenv(PROGRESS_FILE_ENV, str(p2))
        res2 = runner.invoke(app, ["train", str(mdir)])
        assert res2.exit_code == 0, res2.output
    finally:
        TRAINERS.unregister("prog_arch")

    assert seen == [p1, p2]
