"""Lifecycle runner: staleness, state, and orchestration (torch-free).

The executors are swapped for fakes so the ordering, skipping, and failure
behaviour can be tested without training anything; the real end-to-end path
runs in the ml CI job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maatml.config import load_model_def
from maatml.lifecycle import (
    STEPS,
    RunOptions,
    _selected_steps,
    compute_components,
    fingerprint,
    load_state,
    plan_pipeline,
    record_step,
    run_pipeline,
    state_path,
    validate_run_config,
)
from maatml.utils.io import write_jsonl

MODEL_YML = """name: lifecycle-test
model_id: lifecycle-test
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: datasets/samples/seed_samples.jsonl
  target_field: expected_output
evaluation:
  predictor: causal_sft
  gates:
    all_layers_pass_rate: 0.5
smoke:
  max_steps: 2
  gates:
    output_nonempty_rate: 0.5
training:
  model_id: tiny
  epochs: 1
"""


def _model(tmp_path: Path, *, body: str = MODEL_YML):
    mdir = tmp_path / "model"
    (mdir / "datasets" / "samples").mkdir(parents=True, exist_ok=True)
    (mdir / "model.yml").write_text(body, encoding="utf-8")
    write_jsonl(
        mdir / "datasets" / "samples" / "seed_samples.jsonl",
        [{"sample_id": "a", "request": "r", "expected_output": {"ok": True}}],
    )
    return load_model_def(mdir)


def _mark_prepared(md) -> None:
    for split in ("train", "val", "test"):
        write_jsonl(md.prepared_dir / f"{split}.jsonl", [{"request": "r"}])


def _plans(md, **kwargs) -> dict:
    return {plan.name: plan for plan in plan_pipeline(md, **kwargs)}


# --- step selection --------------------------------------------------------


def test_selected_steps_ranges() -> None:
    assert _selected_steps(None, None) == STEPS
    assert _selected_steps("train", "evaluate") == ("train", "evaluate")
    assert _selected_steps(None, "prepare") == ("prepare",)


def test_selected_steps_rejects_unknown_and_inverted_ranges() -> None:
    with pytest.raises(ValueError, match="--from must be one of"):
        _selected_steps("nonsense", None)
    with pytest.raises(ValueError, match="--until must be one of"):
        _selected_steps(None, "nonsense")
    with pytest.raises(ValueError, match="comes after"):
        _selected_steps("export", "train")


# --- staleness -------------------------------------------------------------


def test_everything_is_stale_before_the_first_run(tmp_path: Path) -> None:
    md = _model(tmp_path)
    plans = _plans(md)
    assert [p.name for p in plans.values()] == list(STEPS)
    assert all(not p.fresh and p.reason == "never run" for p in plans.values())


def test_a_completed_step_with_outputs_is_fresh(tmp_path: Path) -> None:
    md = _model(tmp_path)
    _mark_prepared(md)
    components = compute_components(md, smoke=False, device="cpu")["prepare"]
    record_step(md, "prepare", components=components, status="completed")

    plan = _plans(md, device="cpu")["prepare"]
    assert plan.fresh, plan.reason
    assert plan.reason == "up to date"


def test_missing_outputs_make_a_matching_fingerprint_stale(tmp_path: Path) -> None:
    """A fingerprint match is not enough: someone may have deleted output/."""
    md = _model(tmp_path)
    _mark_prepared(md)
    components = compute_components(md, smoke=False, device="cpu")["prepare"]
    record_step(md, "prepare", components=components, status="completed")
    (md.prepared_dir / "test.jsonl").unlink()

    plan = _plans(md, device="cpu")["prepare"]
    assert not plan.fresh
    assert plan.reason == "outputs missing"


def test_changed_config_names_the_component(tmp_path: Path) -> None:
    md = _model(tmp_path)
    _mark_prepared(md)
    record_step(
        md,
        "prepare",
        components=compute_components(md, smoke=False, device="cpu")["prepare"],
        status="completed",
    )
    md.dataset["split_ratios"] = [0.5, 0.25, 0.25]

    plan = _plans(md, device="cpu")["prepare"]
    assert not plan.fresh
    assert "dataset_config" in plan.reason


def test_changed_seed_corpus_makes_prepare_stale(tmp_path: Path) -> None:
    """datagen / ingest live outside the runner; their output is what it sees."""
    md = _model(tmp_path)
    _mark_prepared(md)
    record_step(
        md,
        "prepare",
        components=compute_components(md, smoke=False, device="cpu")["prepare"],
        status="completed",
    )
    write_jsonl(
        md.resolve("datasets/samples/seed_samples.jsonl"),
        [
            {"sample_id": "a", "request": "r", "expected_output": {"ok": True}},
            {"sample_id": "b", "request": "r2", "expected_output": {"ok": False}},
        ],
    )

    plan = _plans(md, device="cpu")["prepare"]
    assert not plan.fresh
    assert "input_assets" in plan.reason


def test_force_and_failed_status_make_steps_stale(tmp_path: Path) -> None:
    md = _model(tmp_path)
    _mark_prepared(md)
    components = compute_components(md, smoke=False, device="cpu")["prepare"]
    record_step(md, "prepare", components=components, status="completed")

    assert _plans(md, device="cpu", force=True)["prepare"].reason == "--force"

    record_step(md, "prepare", components=components, status="failed")
    assert "failed" in _plans(md, device="cpu")["prepare"].reason


def test_smoke_changes_the_train_fingerprint(tmp_path: Path) -> None:
    md = _model(tmp_path)
    plain = compute_components(md, smoke=False, device="cpu")
    smoke = compute_components(md, smoke=True, device="cpu")
    assert plain["prepare"] == smoke["prepare"]
    assert fingerprint(plain["train"]) != fingerprint(smoke["train"])
    assert fingerprint(plain["evaluate"]) != fingerprint(smoke["evaluate"])


# --- state file ------------------------------------------------------------


def test_state_is_written_atomically_and_survives_corruption(tmp_path: Path) -> None:
    md = _model(tmp_path)
    record_step(
        md,
        "prepare",
        components={"a": "1"},
        status="completed",
        detail="splits",
        smoke=True,
    )
    state = json.loads(state_path(md).read_text(encoding="utf-8"))
    assert state["steps"]["prepare"]["status"] == "completed"
    assert state["steps"]["prepare"]["smoke"] is True
    assert state["maatml_version"]
    assert not list(md.output_dir.glob("*.tmp"))

    state_path(md).write_text("{not json", encoding="utf-8")
    # A corrupt file re-runs everything rather than bricking the runner.
    assert load_state(md) == {"version": 1, "steps": {}}


# --- config validation -----------------------------------------------------


def test_run_config_rejects_an_unregistered_validator(tmp_path: Path) -> None:
    md = _model(tmp_path)
    md.evaluation["validator"] = "not_registered"
    with pytest.raises(Exception, match="not_registered"):
        validate_run_config(md)


def test_run_config_rejects_unknown_evaluation_keys(tmp_path: Path) -> None:
    md = _model(tmp_path)
    md.evaluation["gatez"] = {"x": 1}
    with pytest.raises(Exception, match="evaluation"):
        validate_run_config(md)


def test_run_config_requires_gates(tmp_path: Path) -> None:
    md = _model(tmp_path)
    md.evaluation.pop("gates")
    with pytest.raises(Exception, match="gates"):
        validate_run_config(md)
    # The smoke tier satisfies a --smoke run on its own.
    validate_run_config(md, smoke=True)


# --- orchestration ---------------------------------------------------------


@pytest.fixture
def fake_steps(monkeypatch):
    """Replace the executors; record the order they run in."""
    import maatml.lifecycle as lifecycle

    calls: list[str] = []

    def _make(name: str):
        def _fn(model_def, options):
            calls.append(name)
            if name == "prepare":
                _mark_prepared(model_def)
            return f"{name} done"

        return _fn

    monkeypatch.setattr(lifecycle, "_EXECUTORS", {s: _make(s) for s in STEPS})
    monkeypatch.setattr(lifecycle, "_resolve_paths", lambda md: (None, None, None))
    monkeypatch.setattr(
        lifecycle,
        "outputs_present",
        lambda md, step, **kwargs: True,
    )
    return calls


def test_run_pipeline_runs_every_step_in_order(tmp_path: Path, fake_steps) -> None:
    md = _model(tmp_path)
    result = run_pipeline(md, RunOptions(device="cpu"))
    assert result.ok
    assert fake_steps == list(STEPS)
    assert [o.status for o in result.outcomes] == ["ran"] * len(STEPS)


def test_second_run_does_no_work(tmp_path: Path, fake_steps) -> None:
    md = _model(tmp_path)
    run_pipeline(md, RunOptions(device="cpu"))
    fake_steps.clear()

    result = run_pipeline(md, RunOptions(device="cpu"))
    assert result.ok
    assert fake_steps == [], "a re-run with no changes must not execute anything"
    assert [o.status for o in result.outcomes] == ["skipped"] * len(STEPS)


def test_a_failing_step_stops_the_pipeline(tmp_path: Path, fake_steps, monkeypatch) -> None:
    import maatml.lifecycle as lifecycle

    def _boom(model_def, options):
        raise RuntimeError("gates failed")

    executors = dict(lifecycle._EXECUTORS)
    executors["evaluate"] = _boom
    monkeypatch.setattr(lifecycle, "_EXECUTORS", executors)

    md = _model(tmp_path)
    result = run_pipeline(md, RunOptions(device="cpu"))
    assert not result.ok
    assert result.failed == "evaluate"
    # export and verify never ran on the output of a step that did not pass.
    assert fake_steps == ["prepare", "train"]
    assert load_state(md)["steps"]["evaluate"]["status"] == "failed"
    assert "export" not in load_state(md)["steps"]


def test_upstream_rerun_invalidates_downstream(tmp_path: Path, fake_steps) -> None:
    md = _model(tmp_path)
    run_pipeline(md, RunOptions(device="cpu"))
    fake_steps.clear()

    # Only the training config changed, but everything after train re-runs.
    md.training["learning_rate"] = 5e-5
    result = run_pipeline(md, RunOptions(device="cpu"))
    assert result.ok
    assert fake_steps == ["train", "evaluate", "export", "verify"]


def test_from_and_until_limit_the_steps(tmp_path: Path, fake_steps) -> None:
    md = _model(tmp_path)
    result = run_pipeline(md, RunOptions(device="cpu", from_step="train", until_step="evaluate"))
    assert result.ok
    assert fake_steps == ["train", "evaluate"]
    statuses = {o.name: o.status for o in result.outcomes}
    assert statuses["prepare"] == "not selected"
    assert statuses["export"] == "not selected"


def test_run_pipeline_validates_config_before_running_anything(tmp_path: Path, fake_steps) -> None:
    md = _model(tmp_path)
    md.evaluation["validator"] = "not_registered"
    with pytest.raises(Exception, match="not_registered"):
        run_pipeline(md, RunOptions(device="cpu"))
    assert fake_steps == [], "nothing should run when the config cannot be used"
    assert not state_path(md).exists()


def test_limit_and_seed_change_the_train_fingerprint(tmp_path: Path) -> None:
    """A run truncated with --limit trained on a fraction of the corpus. Leaving
    limit/seed out of the fingerprint let the next plain run report 'all fresh'
    over that checkpoint, and evaluate/export/verify inherited the skip."""
    md = _model(tmp_path)
    base = compute_components(md, smoke=False, device="cpu")["train"]
    limited = compute_components(md, smoke=False, device="cpu", limit=50)["train"]
    seeded = compute_components(md, smoke=False, device="cpu", seed=7)["train"]

    assert base["limit"] == "none" and base["seed"] == "none"
    assert limited["limit"] == "50"
    assert seeded["seed"] == "7"
    assert limited != base, "--limit must not look fresh against a full run"
    assert seeded != base, "--seed must not look fresh against another seed"
    # Downstream steps inherit train's fingerprint, so they go stale too.
    assert (
        compute_components(md, smoke=False, device="cpu", limit=50)["evaluate"]["upstream"]
        != compute_components(md, smoke=False, device="cpu")["evaluate"]["upstream"]
    )


def test_environment_sha_ignores_an_unrelated_repository(monkeypatch) -> None:
    """The environment component may only track maatml's own checkout. Asking
    git from the package directory otherwise returns the SHA of whatever repo
    contains site-packages, forcing spurious full retrains."""
    from maatml.training import guards

    calls: list[list[str]] = []

    def _fake_check_output(cmd, **kwargs):
        calls.append(list(cmd))
        if "--show-toplevel" in cmd:
            # An unrelated repository that does not hold src/maatml.
            return "/some/other/repo\n"
        return "deadbeef\n"

    monkeypatch.setattr(guards.subprocess, "check_output", _fake_check_output)
    assert guards.maatml_checkout_sha() is None
    assert any("--show-toplevel" in c for c in calls)


def test_run_options_resume_forces_train_and_passes_resume_arg(
    tmp_path: Path, fake_steps, monkeypatch
) -> None:
    import json as _json

    from typer.testing import CliRunner

    import maatml.lifecycle as lifecycle
    from maatml.cli import app

    md = _model(tmp_path)
    assert run_pipeline(md, RunOptions(device="cpu")).ok
    fake_steps.clear()

    # Without resume, everything is skipped.
    assert [o.status for o in run_pipeline(md, RunOptions(device="cpu")).outcomes] == [
        "skipped"
    ] * len(STEPS)

    # With resume="auto", prepare stays skipped while train + downstream re-run.
    plans = _plans(md, device="cpu", resume="auto")
    assert plans["prepare"].fresh is True
    assert plans["train"].fresh is False and plans["train"].reason == "--resume auto"
    assert plans["evaluate"].fresh is False

    seen_calls: list[dict[str, object]] = []

    def _resumed_train(model_def, options):
        seen_calls.append({"resume": options.resume, "progress_file": options.progress_file})
        return "resumed"

    executors = dict(lifecycle._EXECUTORS)
    executors["train"] = _resumed_train
    monkeypatch.setattr(lifecycle, "_EXECUTORS", executors)

    progress_path = tmp_path / "pipeline-progress.json"
    res = run_pipeline(
        md,
        RunOptions(device="cpu", resume="auto", progress_file=progress_path),
    )
    assert res.ok
    assert [o.status for o in res.outcomes] == ["skipped", "ran", "ran", "ran", "ran"]
    assert seen_calls == [{"resume": "auto", "progress_file": progress_path}]
    assert _json.loads(progress_path.read_text(encoding="utf-8"))["stage"] == "completed"

    # CLI --resume auto --dry-run surfaces the resume reason.
    dry = CliRunner().invoke(
        app, ["run", str(md.model_dir), "--device", "cpu", "--resume", "auto", "--dry-run"]
    )
    assert dry.exit_code == 0, dry.output
    assert "--resume auto" in dry.output
    assert fake_steps == ["evaluate", "export", "verify"]


def test_step_train_forwards_resume_to_registered_trainer(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from maatml.lifecycle import _step_train
    from maatml.registry import TRAINERS

    md = _model(tmp_path)
    object.__setattr__(md, "architecture", "resume_test_arch")
    seen: dict[str, object] = {}

    def _trainer(model_def, **kwargs):
        del model_def
        seen.update(kwargs)
        return SimpleNamespace(out_dir=tmp_path / "ckpt", metrics={})

    TRAINERS.register("resume_test_arch", _trainer, source="test")
    monkeypatch.setattr("maatml.training.selection.select_checkpoint", lambda *a, **kw: None)
    _step_train(md, RunOptions(device="cpu", resume="auto"))
    assert seen["resume"] == "auto"
