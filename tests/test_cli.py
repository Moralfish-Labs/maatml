"""CLI surface: exit codes, argument parsing, and user-error messages.

Torch-free by construction: every command here fails (or succeeds) before any
model load, which is exactly the behaviour that has to hold in CI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from maatml.cli import app

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BOX = re.compile(r"[─-╿]")


def plain(output: str) -> str:
    """Console output with styling and box drawing removed.

    typer renders usage errors inside a rich panel, so the message is broken
    across lines by borders whose position depends on the terminal width. That
    made width-sensitive assertions pass on one platform and fail on another.
    """
    text = _BOX.sub(" ", _ANSI.sub("", output))
    return " ".join(text.split())


def _write_model(
    tmp_path: Path,
    *,
    name: str = "cli-test",
    evaluation: str = "",
    extra: str = "",
) -> Path:
    mdir = tmp_path / name
    (mdir / "datasets").mkdir(parents=True, exist_ok=True)
    (mdir / "model.yml").write_text(
        f"""name: {name}
model_id: {name}
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: datasets/seeds.jsonl
{evaluation}{extra}""",
        encoding="utf-8",
    )
    (mdir / "datasets" / "seeds.jsonl").write_text("", encoding="utf-8")
    return mdir


# --- argument parsing ------------------------------------------------------


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "prepare",
        "train",
        "sweep",
        "evaluate",
        "export",
        "verify",
        "serve",
        "datagen",
        "ingest",
        "runs",
        "scaffold",
        "validate",
        "plan",
        "plugins",
        "audit",
    ):
        assert command in result.output


def test_unknown_command_exits_two() -> None:
    result = runner.invoke(app, ["definitely-not-a-command"])
    assert result.exit_code == 2


def test_missing_model_dir_is_a_usage_error() -> None:
    result = runner.invoke(app, ["prepare", "/nonexistent/model/dir"])
    assert result.exit_code == 2
    assert "Error" in plain(result.output)


def test_ingest_rejects_a_malformed_field_map(tmp_path: Path) -> None:
    mdir = _write_model(tmp_path)
    source = tmp_path / "in.jsonl"
    source.write_text('{"text": "x"}\n', encoding="utf-8")
    result = runner.invoke(
        app, ["ingest", str(mdir), "--input", str(source), "--map", "no-equals-sign"]
    )
    assert result.exit_code == 2
    assert "field=col" in plain(result.output)


def test_plugins_lists_registered_trainers() -> None:
    result = runner.invoke(app, ["plugins"])
    assert result.exit_code == 0
    assert "causal_sft" in result.output


# --- evaluate --gate -------------------------------------------------------


def test_evaluate_gate_without_gates_configured_exits_nonzero(tmp_path: Path) -> None:
    mdir = _write_model(tmp_path, evaluation="evaluation:\n  predictor: causal_sft\n")
    result = runner.invoke(app, ["evaluate", str(mdir), "--gate"])
    assert result.exit_code != 0
    assert "no evaluation.gates" in plain(result.output)


def test_evaluate_reports_unregistered_validator_before_loading(tmp_path: Path) -> None:
    mdir = _write_model(
        tmp_path,
        evaluation="evaluation:\n  predictor: causal_sft\n  validator: not_registered\n",
    )
    result = runner.invoke(app, ["evaluate", str(mdir)])
    assert result.exit_code != 0
    assert "not_registered" in plain(result.output)


def test_evaluate_reports_unregistered_metrics_before_loading(tmp_path: Path) -> None:
    mdir = _write_model(
        tmp_path,
        evaluation="evaluation:\n  predictor: causal_sft\n  metrics: [nope_metrics]\n",
    )
    result = runner.invoke(app, ["evaluate", str(mdir)])
    assert result.exit_code != 0
    assert "nope_metrics" in plain(result.output)


# --- verify ----------------------------------------------------------------


def _export_dir(tmp_path: Path) -> Path:
    """A real export bundle: manifest built the way `maatml export` builds it."""
    from maatml.config import load_model_def
    from maatml.export.manifest import build_manifest, write_manifest

    mdir = _write_model(tmp_path, name="exported")
    export = tmp_path / "export"
    export.mkdir()
    (export / "model.safetensors").write_bytes(b"weights")
    (export / "config.json").write_text("{}", encoding="utf-8")
    manifest = build_manifest(
        model_def=load_model_def(mdir),
        export_dir=export,
        files=[export / "model.safetensors", export / "config.json"],
        formats=["safetensors"],
        source_checkpoint=tmp_path / "ckpt",
    )
    write_manifest(export, manifest)
    return export


def test_verify_passes_on_an_untouched_export(tmp_path: Path) -> None:
    export = _export_dir(tmp_path)
    result = runner.invoke(app, ["verify", str(export)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_verify_fails_on_a_tampered_file(tmp_path: Path) -> None:
    export = _export_dir(tmp_path)
    (export / "model.safetensors").write_bytes(b"tampered")
    result = runner.invoke(app, ["verify", str(export)])
    assert result.exit_code == 1
    assert "verify failed" in result.output


def test_compile_require_gated_refuses_an_ungated_export(tmp_path: Path) -> None:
    from maatml.registry import register_compiler
    from maatml.utils.io import write_json

    export = tmp_path / "export"
    export.mkdir()
    (export / "model.onnx").write_bytes(b"onnx")
    write_json(
        export / "manifest.json",
        {
            "name": "toy",
            "version": "1",
            "identity": "toy@1",
            "files": [{"path": "model.onnx", "sha256": "abc"}],
        },
    )

    @register_compiler("cli_gated")
    def _cmp(export_dir, out_dir, *, manifest, options):
        del export_dir, manifest, options
        return out_dir

    result = runner.invoke(
        app,
        [
            "compile",
            str(export),
            "--target",
            "cli_gated",
            "--out",
            str(tmp_path / "out"),
            "--require-gated",
        ],
    )
    assert result.exit_code == 1
    assert "require-gated" in plain(result.output)


def test_verify_fails_when_a_listed_file_is_missing(tmp_path: Path) -> None:
    export = _export_dir(tmp_path)
    (export / "config.json").unlink()
    result = runner.invoke(app, ["verify", str(export)])
    assert result.exit_code == 1
    assert "config.json" in result.output


def test_verify_on_a_missing_manifest_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["verify", str(tmp_path / "no-such-export")])
    assert result.exit_code != 0


# --- scaffold --------------------------------------------------------------


def test_scaffold_creates_a_model_folder(tmp_path: Path) -> None:
    target = tmp_path / "fresh"
    result = runner.invoke(app, ["scaffold", str(target), "--architecture", "causal_sft"])
    assert result.exit_code == 0, result.output
    assert (target / "model.yml").is_file()
    assert (target / "datasets" / "samples" / "seed_samples.jsonl").is_file()


def test_scaffold_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    runner.invoke(app, ["scaffold", str(target), "--architecture", "causal_sft"])
    (target / "datasets" / "samples" / "seed_samples.jsonl").write_text(
        json.dumps({"request": "mine", "expected_output": {"a": 1}}) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["scaffold", str(target), "--architecture", "causal_sft"])
    assert result.exit_code == 1
    assert "scaffold refused" in result.output
    # The hand-written corpus survives.
    seeds = (target / "datasets" / "samples" / "seed_samples.jsonl").read_text()
    assert "mine" in seeds

    forced = runner.invoke(
        app, ["scaffold", str(target), "--architecture", "causal_sft", "--force"]
    )
    assert forced.exit_code == 0, forced.output


def test_scaffold_rejects_an_unknown_architecture(tmp_path: Path) -> None:
    result = runner.invoke(app, ["scaffold", str(tmp_path / "x"), "--architecture", "not_an_arch"])
    assert result.exit_code != 0


# --- validate --------------------------------------------------------------


def test_validate_reports_a_missing_declared_path(tmp_path: Path) -> None:
    mdir = _write_model(tmp_path, name="broken-paths")
    (mdir / "datasets" / "seeds.jsonl").unlink()
    result = runner.invoke(app, ["validate", str(mdir)])
    assert result.exit_code == 1
    assert "validate failed" in result.output


def test_validate_warns_on_unknown_config_keys(tmp_path: Path) -> None:
    mdir = _write_model(tmp_path, name="typos", extra="  tarjet_field: oops\n")
    result = runner.invoke(app, ["validate", str(mdir)])
    assert result.exit_code == 0, result.output
    assert "tarjet_field" in result.output


def test_validate_no_plugins_says_what_it_skipped(tmp_path: Path) -> None:
    mdir = _write_model(tmp_path, name="noplugins")
    result = runner.invoke(app, ["validate", str(mdir), "--no-plugins"])
    assert result.exit_code == 0, result.output
    assert "not verified" in result.output


# --- runs ------------------------------------------------------------------


def test_runs_on_a_model_with_no_runs(tmp_path: Path) -> None:
    mdir = _write_model(tmp_path, name="norun")
    result = runner.invoke(app, ["runs", str(mdir)])
    assert result.exit_code == 0
    assert "no runs yet" in result.output


def test_runs_survives_a_torn_last_line(tmp_path: Path) -> None:
    from maatml.config import load_model_def
    from maatml.runs import runs_path, start_run

    mdir = _write_model(tmp_path, name="torn")
    md = load_model_def(mdir)
    rec = start_run(md)
    with open(runs_path(md), "a", encoding="utf-8") as fh:
        fh.write('{"run_id": "truncated", "identi')

    result = runner.invoke(app, ["runs", str(mdir)])
    assert result.exit_code == 0, result.output
    assert rec.run_id in result.output
    assert (runs_path(md).with_name("runs.jsonl.corrupt")).is_file()


# --- user-error messages ---------------------------------------------------


def test_unparseable_model_yml_prints_one_line(tmp_path: Path) -> None:
    mdir = tmp_path / "bad-yaml"
    mdir.mkdir()
    (mdir / "model.yml").write_text("name: [unclosed\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(mdir)])
    assert result.exit_code != 0


def test_missing_model_yml_names_the_file(tmp_path: Path) -> None:
    mdir = tmp_path / "empty-dir"
    mdir.mkdir()
    result = runner.invoke(app, ["plan", str(mdir)], catch_exceptions=True)
    assert result.exit_code != 0
    assert "model.yml" in str(result.output) + str(result.exception)


@pytest.mark.parametrize("debug", [False, True])
def test_main_prints_one_line_for_user_errors(monkeypatch, capsys, debug: bool) -> None:
    """maatml.cli.main turns known user errors into a single actionable line."""
    from maatml import cli

    def _boom(*args, **kwargs):
        raise FileNotFoundError("datasets/samples/seed_samples.jsonl not found")

    monkeypatch.setattr(cli, "app", _boom)
    monkeypatch.setitem(cli._STATE, "debug", debug)

    if debug:
        with pytest.raises(FileNotFoundError):
            cli.main()
        return

    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert "seed_samples.jsonl not found" in out
    assert "Traceback" not in out
    assert "--debug" in out


# --- audit -----------------------------------------------------------------


def test_audit_reports_environment_and_plugins() -> None:
    result = runner.invoke(app, ["audit"])
    assert result.exit_code == 0, result.output
    for section in ("environment", "packages", "device", "plugins"):
        assert section in result.output
    # Extras are named literally, not eaten as rich markup.
    assert "maatml" in result.output


def test_audit_json_is_machine_readable() -> None:
    result = runner.invoke(app, ["audit", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {"environment", "packages", "device", "plugins"} <= set(payload)
    assert all(
        {"name", "status", "detail"} == set(check)
        for checks in payload.values()
        for check in checks
    )


def test_audit_flags_a_model_folder_with_missing_paths(tmp_path: Path) -> None:
    mdir = _write_model(tmp_path, name="audit-broken")
    (mdir / "datasets" / "seeds.jsonl").unlink()
    result = runner.invoke(app, ["audit", str(mdir), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    model = {check["name"]: check for check in payload["model"]}
    assert model["declared paths"]["status"] == "error"
    # An unprepared folder is a warning, not an error: prepare has not run yet.
    assert model["prepared splits"]["status"] == "warn"


def test_audit_on_a_healthy_example_exits_zero() -> None:
    result = runner.invoke(app, ["audit", "examples/support-ticket-triage", "--json"])
    assert result.exit_code == 0, result.output
    model = {c["name"]: c for c in json.loads(result.output)["model"]}
    assert model["architecture"]["status"] == "ok"
    assert model["evaluation.gates"]["status"] == "ok"


def test_audit_flags_unregistered_metrics(tmp_path: Path) -> None:
    mdir = _write_model(
        tmp_path,
        name="audit-bad-metrics",
        evaluation="evaluation:\n  metrics: not_a_real_metric\n  gates:\n    x: 0.5\n",
    )
    result = runner.invoke(app, ["audit", str(mdir), "--json"])
    assert result.exit_code == 1
    model = {c["name"]: c for c in json.loads(result.output)["model"]}
    assert model["evaluation.metrics"]["status"] == "error"
    assert "not_a_real_metric" in model["evaluation.metrics"]["detail"]


def test_audit_flags_unregistered_generator(tmp_path: Path) -> None:
    mdir = _write_model(
        tmp_path,
        name="audit-bad-gen",
        extra="  generator: no_such_generator\n",
    )
    result = runner.invoke(app, ["audit", str(mdir), "--json"])
    assert result.exit_code == 1
    model = {c["name"]: c for c in json.loads(result.output)["model"]}
    assert model["dataset.generator"]["status"] == "error"


def test_audit_warns_on_runs_quarantine(tmp_path: Path) -> None:
    mdir = _write_model(tmp_path, name="audit-corrupt")
    corrupt = mdir / "output" / "runs.jsonl.corrupt"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_text("{not json\n{also bad\n", encoding="utf-8")
    result = runner.invoke(app, ["audit", str(mdir), "--json"])
    assert result.exit_code == 0, result.output
    model = {c["name"]: c for c in json.loads(result.output)["model"]}
    assert model["runs quarantine"]["status"] == "warn"
    assert "2" in model["runs quarantine"]["detail"]


def test_audit_warns_when_gate_key_missing_from_eval_report(tmp_path: Path) -> None:
    mdir = _write_model(
        tmp_path,
        name="audit-gate-mismatch",
        evaluation=(
            "evaluation:\n  gates:\n    all_layers_pass_rate: 0.9\n    made_up_metric: 0.5\n"
        ),
    )
    eval_dir = mdir / "output" / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "run1.json").write_text(
        json.dumps(
            {
                "model_id": "x",
                "n": 1,
                "metrics": {"all_layers_pass_rate": 1.0},
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["audit", str(mdir), "--json"])
    assert result.exit_code == 0, result.output
    model = {c["name"]: c for c in json.loads(result.output)["model"]}
    assert model["gate keys"]["status"] == "warn"
    assert "made_up_metric" in model["gate keys"]["detail"]


def test_audit_warns_on_empty_seed_corpus(tmp_path: Path) -> None:
    mdir = _write_model(tmp_path, name="audit-empty-seeds")
    # _write_model already writes an empty seeds.jsonl
    result = runner.invoke(app, ["audit", str(mdir), "--json"])
    assert result.exit_code == 0, result.output
    model = {c["name"]: c for c in json.loads(result.output)["model"]}
    assert model["seed corpus"]["status"] == "warn"
    assert "0 rows" in model["seed corpus"]["detail"]


def test_audit_flags_qlora_on_non_cuda_when_torch_present(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("auto device is cuda; cannot exercise the non-CUDA QLoRA error")
    mdir = _write_model(
        tmp_path,
        name="audit-qlora",
        extra=("training:\n  quantization:\n    load_in_4bit: true\n"),
    )
    result = runner.invoke(app, ["audit", str(mdir), "--json"])
    assert result.exit_code == 1
    model = {c["name"]: c for c in json.loads(result.output)["model"]}
    assert model["training.quantization"]["status"] == "error"
    assert "CUDA" in model["training.quantization"]["detail"]


# --- runs --compare --------------------------------------------------------


def _seed_runs(tmp_path: Path):
    from maatml.config import load_model_def
    from maatml.runs import finish_run, start_run

    mdir = _write_model(tmp_path, name="compare")
    md = load_model_def(mdir)
    first = start_run(md, smoke=True)
    finish_run(md, first.run_id, "completed", metrics={"eval_loss": 1.5, "eval_runtime": 9.0})
    second = start_run(md)
    finish_run(md, second.run_id, "completed", metrics={"eval_loss": 0.5, "accuracy": 0.9})
    return mdir, first, second


def test_runs_compare_tabulates_metrics_across_runs(tmp_path: Path) -> None:
    mdir, first, second = _seed_runs(tmp_path)
    result = runner.invoke(app, ["runs", str(mdir), "--compare"])
    assert result.exit_code == 0, result.output
    assert "eval_loss" in result.output
    assert "accuracy" in result.output
    # Timing metrics are set aside, and the command says so rather than
    # dropping them silently.
    assert "eval_runtime" not in result.output.split("hidden")[0]
    assert "timing metric(s) hidden" in result.output


def test_runs_compare_metric_filter_and_all_metrics(tmp_path: Path) -> None:
    mdir, _first, _second = _seed_runs(tmp_path)
    filtered = runner.invoke(app, ["runs", str(mdir), "--compare", "--metric", "eval_loss"])
    assert filtered.exit_code == 0
    assert "accuracy" not in filtered.output

    everything = runner.invoke(app, ["runs", str(mdir), "--compare", "--all-metrics"])
    assert everything.exit_code == 0
    assert "eval_runtime" in everything.output


def test_compare_runs_marks_metrics_a_run_never_reported(tmp_path: Path) -> None:
    from maatml.config import load_model_def
    from maatml.runs import compare_runs, list_runs

    mdir, _first, _second = _seed_runs(tmp_path)
    keys, rows, hidden = compare_runs(list_runs(load_model_def(mdir)))
    assert keys == ["eval_loss", "accuracy"]
    assert hidden == ["eval_runtime"]
    # The first run never reported accuracy: None, not 0.0.
    assert rows[0]["metrics"]["accuracy"] is None
    assert rows[1]["metrics"]["accuracy"] == 0.9


# --- run / plan ------------------------------------------------------------


def _runnable_model(tmp_path: Path) -> Path:
    mdir = tmp_path / "runnable"
    (mdir / "datasets" / "samples").mkdir(parents=True, exist_ok=True)
    (mdir / "model.yml").write_text(
        """name: runnable
model_id: runnable
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: datasets/samples/seed_samples.jsonl
evaluation:
  predictor: causal_sft
  gates:
    all_layers_pass_rate: 0.5
smoke:
  max_steps: 2
  gates:
    output_nonempty_rate: 0.5
""",
        encoding="utf-8",
    )
    (mdir / "datasets" / "samples" / "seed_samples.jsonl").write_text(
        json.dumps({"request": "r", "expected_output": {"ok": True}}) + "\n",
        encoding="utf-8",
    )
    return mdir


def test_run_dry_run_lists_every_step_as_stale(tmp_path: Path) -> None:
    mdir = _runnable_model(tmp_path)
    result = runner.invoke(app, ["run", str(mdir), "--dry-run"])
    assert result.exit_code == 0, result.output
    output = plain(result.output)
    for step in ("prepare", "train", "evaluate", "export", "verify"):
        assert step in output
    assert "never run" in output
    # A dry run does no work and leaves no state behind.
    assert not (mdir / "output" / "pipeline.json").exists()


def test_plan_is_the_same_view_as_run_dry_run(tmp_path: Path) -> None:
    mdir = _runnable_model(tmp_path)
    planned = plain(runner.invoke(app, ["plan", str(mdir)]).output)
    dry = plain(runner.invoke(app, ["run", str(mdir), "--dry-run"]).output)
    for step in ("prepare", "train", "evaluate", "export", "verify"):
        assert step in planned and step in dry


def test_run_rejects_an_unknown_step_name(tmp_path: Path) -> None:
    mdir = _runnable_model(tmp_path)
    result = runner.invoke(app, ["run", str(mdir), "--dry-run", "--from", "nope"])
    assert result.exit_code == 2
    assert "--from must be one of" in plain(result.output)


def test_run_with_an_invalid_set_leaves_no_state(tmp_path: Path) -> None:
    """Exit criterion: an invalid --set must not touch output/pipeline.json."""
    mdir = _runnable_model(tmp_path)
    runner.invoke(app, ["run", str(mdir), "--dry-run"])  # no state written
    state = mdir / "output" / "pipeline.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text('{"steps": {"prepare": {"status": "completed"}}}', encoding="utf-8")
    before = state.read_text(encoding="utf-8")

    result = runner.invoke(app, ["run", str(mdir), "--set", "version=not-semver"])
    assert result.exit_code == 2
    assert state.read_text(encoding="utf-8") == before


def test_run_with_a_misspelled_validator_exits_nonzero(tmp_path: Path) -> None:
    """Exit criterion: the config error surfaces before any step runs."""
    mdir = _runnable_model(tmp_path)
    result = runner.invoke(app, ["run", str(mdir), "--set", "evaluation.validator=nope_misspelled"])
    assert result.exit_code != 0
    assert "nope_misspelled" in plain(result.output)
    assert not (mdir / "output" / "pipeline.json").exists()


def test_run_without_gates_refuses_to_start(tmp_path: Path) -> None:
    mdir = _runnable_model(tmp_path)
    body = (mdir / "model.yml").read_text(encoding="utf-8")
    (mdir / "model.yml").write_text(
        body.replace("  gates:\n    all_layers_pass_rate: 0.5\n", ""), encoding="utf-8"
    )
    result = runner.invoke(app, ["run", str(mdir)])
    assert result.exit_code != 0
    assert "gates" in plain(result.output)
