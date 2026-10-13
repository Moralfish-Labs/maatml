"""Unit tests for the append-only run registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from maatml.config import ModelDefinition
from maatml.runs import (
    finish_run,
    get_run,
    latest_completed_run,
    list_runs,
    resolve_checkpoint,
    resolve_resume_checkpoint,
    runs_path,
    start_run,
)


def _md(tmp_path: Path) -> ModelDefinition:
    md = ModelDefinition(
        name="run-test",
        model_id="run-test",
        architecture="causal_sft",
        version="0.1.0",
    )
    object.__setattr__(md, "model_dir", tmp_path)
    return md


def test_start_finish_list_latest(tmp_path: Path) -> None:
    md = _md(tmp_path)
    r1 = start_run(md, smoke=True, device="cpu", profile="cpu")
    assert r1.status == "running"
    assert Path(r1.out_dir).is_dir()
    assert (tmp_path / "output" / "runs.jsonl").is_file()

    finish_run(md, r1.run_id, "completed", metrics={"loss": 0.5})
    runs = list_runs(md)
    assert len(runs) == 1
    assert runs[0].status == "completed"
    assert runs[0].metrics == {"loss": 0.5}

    r2 = start_run(md, smoke=False, device="mps", profile="mps")
    finish_run(md, r2.run_id, "aborted", error="boom")
    latest = latest_completed_run(md)
    assert latest is not None
    assert latest.run_id == r1.run_id


def test_resolve_checkpoint_by_run_id_and_path(tmp_path: Path) -> None:
    md = _md(tmp_path)
    rec = start_run(md)
    finish_run(md, rec.run_id, "completed")

    by_id = resolve_checkpoint(md, rec.run_id)
    assert by_id == Path(rec.out_dir)

    by_path = resolve_checkpoint(md, rec.out_dir)
    assert by_path == Path(rec.out_dir).resolve()

    default = resolve_checkpoint(md, None)
    assert default == Path(rec.out_dir)


def test_resolve_checkpoint_legacy_mtime(tmp_path: Path) -> None:
    md = _md(tmp_path)
    ckpt = md.checkpoints_dir / "legacy@version"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_text("x", encoding="utf-8")
    resolved = resolve_checkpoint(md, None)
    assert resolved == ckpt


def test_finish_unknown_run_raises(tmp_path: Path) -> None:
    md = _md(tmp_path)
    with pytest.raises(KeyError):
        finish_run(md, "missing-id", "completed")


# --- D3: torn-line tolerance -------------------------------------------------


def test_list_runs_skips_corrupt_line(tmp_path: Path) -> None:
    md = _md(tmp_path)
    rec = start_run(md)
    finish_run(md, rec.run_id, "completed")
    # Simulate a torn/partial record from a crash mid-write.
    with open(runs_path(md), "a", encoding="utf-8") as f:
        f.write('{"run_id": "torn\n')

    with pytest.warns(RuntimeWarning):
        runs = list_runs(md)
    assert len(runs) == 1
    assert runs[0].run_id == rec.run_id
    corrupt = runs_path(md).with_name("runs.jsonl.corrupt")
    assert corrupt.is_file()
    assert '{"run_id": "torn' in corrupt.read_text(encoding="utf-8")


def test_append_writes_single_line(tmp_path: Path) -> None:
    md = _md(tmp_path)
    start_run(md)
    text = runs_path(md).read_text(encoding="utf-8")
    assert text.count("\n") == 1
    assert text.endswith("\n")


# --- B1: resume resolves to the newest checkpoint-* --------------------------


def test_resolve_resume_auto_returns_newest_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    md = _md(tmp_path)
    rec = start_run(md)  # status running
    (Path(rec.out_dir) / "checkpoint-5").mkdir(parents=True)
    (Path(rec.out_dir) / "checkpoint-40").mkdir(parents=True)
    assert resolve_resume_checkpoint(md, "auto") == Path(rec.out_dir) / "checkpoint-40"


def test_resolve_resume_by_run_id_returns_newest_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    md = _md(tmp_path)
    rec = start_run(md)
    (Path(rec.out_dir) / "checkpoint-5").mkdir(parents=True)
    (Path(rec.out_dir) / "checkpoint-40").mkdir(parents=True)
    assert resolve_resume_checkpoint(md, rec.run_id) == Path(rec.out_dir) / "checkpoint-40"


def test_resolve_resume_auto_no_checkpoint_raises(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    md = _md(tmp_path)
    start_run(md)  # running, but no checkpoint-* saved yet
    with pytest.raises(FileNotFoundError):
        resolve_resume_checkpoint(md, "auto")


def test_resume_skips_running_records_with_no_checkpoint(tmp_path: Path) -> None:
    """A run that died before its first save must not hide a resumable one."""
    pytest.importorskip("transformers")
    from maatml.runs import latest_incomplete_run

    md = _md(tmp_path)
    resumable = start_run(md)
    (Path(resumable.out_dir) / "checkpoint-10").mkdir(parents=True)
    stale = start_run(md)  # running, killed before any checkpoint-*

    assert Path(stale.out_dir).exists()
    assert latest_incomplete_run(md).run_id == resumable.run_id
    assert resolve_resume_checkpoint(md, "auto") == Path(resumable.out_dir) / "checkpoint-10"


def _orphan(tmp_path: Path, run_id: str, *, report: dict | None = None) -> ModelDefinition:
    """A checkpoint on disk whose registry line never travelled with it."""
    import json

    md = _md(tmp_path)
    out_dir = md.checkpoints_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "architecture": "causal_sft",
                "identity": "run-test@0.1.0",
                "created_at": "2026-08-13T09:51:27.171928+00:00",
                "spec_hash": "abc123",
                "extra": {"run_id": run_id, "device": "cuda", "profile": "cuda", "smoke": False},
            }
        )
    )
    if report is not None:
        md.eval_dir.mkdir(parents=True, exist_ok=True)
        (md.eval_dir / f"{run_id}.json").write_text(json.dumps(report))
    return md


def test_a_checkpoint_without_a_registry_line_is_adopted(tmp_path: Path) -> None:
    md = _orphan(tmp_path, "20260813-082031-83f868")
    rec = get_run(md, "20260813-082031-83f868")
    assert rec is not None
    assert rec.status == "completed"
    assert rec.identity == "run-test@0.1.0"
    assert rec.spec_hash == "abc123"
    assert rec.device == "cuda"


def test_an_adopted_run_carries_the_gate_result_its_report_recorded(tmp_path: Path) -> None:
    # The manifest reads `gated` off the registry, so a run whose line was lost
    # exports as never-gated unless the report is read back onto it.
    md = _orphan(
        tmp_path,
        "20260813-082031-83f868",
        report={
            "gates": {"passed": True, "smoke": False, "results": {}},
            "metrics": {"all_layers_pass_rate": 0.9216, "note": "text"},
        },
    )
    rec = get_run(md, "20260813-082031-83f868")
    assert rec is not None
    assert rec.gates is not None and rec.gates["passed"] is True
    assert rec.smoke_gated is False
    assert rec.metrics == {"all_layers_pass_rate": 0.9216}


def test_an_adopted_run_joins_the_registry_once(tmp_path: Path) -> None:
    md = _orphan(tmp_path, "20260813-082031-83f868")
    first = get_run(md, "20260813-082031-83f868")
    second = get_run(md, "20260813-082031-83f868")
    assert first is not None and second is not None
    assert [r.run_id for r in list_runs(md)] == ["20260813-082031-83f868"]


def test_an_unknown_run_id_stays_unknown(tmp_path: Path) -> None:
    md = _md(tmp_path)
    assert get_run(md, "20260813-000000-nosuch") is None
