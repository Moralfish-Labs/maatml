"""Evidence layer, second slice: tiers, slice gates, counts, population stamp, derive."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from maatml.cli import app
from maatml.config import load_model_def
from maatml.evaluation.gates import (
    DeriveError,
    derive_gates,
    harness_rate_groups,
    rewrite_gates_block,
    write_gates,
)
from maatml.evaluation.harness import (
    GateConfigError,
    Report,
    check_gates,
    effective_gates,
    gate_actual,
    gate_tiers,
    run_evaluation,
)
from maatml.evaluation.predictions import predictions_path, write_predictions
from maatml.evaluation.stats import cluster_bootstrap_lower, floor2, wilson_lower

runner = CliRunner()

# --- tiers -------------------------------------------------------------------


def test_gate_values_accept_a_number_or_min_and_tier() -> None:
    md = SimpleNamespace(
        evaluation={"gates": {"a": 0.9, "b": {"min": 0.5, "tier": "advisory"}, "c": {"min": 0.7}}},
        smoke={},
    )
    assert effective_gates(md) == {"a": 0.9, "b": 0.5, "c": 0.7}
    assert gate_tiers(md) == {"a": "blocking", "b": "advisory", "c": "blocking"}


def test_unknown_tier_is_a_config_error() -> None:
    md = SimpleNamespace(evaluation={"gates": {"a": {"min": 0.5, "tier": "soft"}}}, smoke={})
    with pytest.raises(GateConfigError, match="tier"):
        gate_tiers(md)


def test_advisory_miss_is_recorded_but_never_fails() -> None:
    out = check_gates(
        {"a": 0.95, "b": 0.1},
        {"a": 0.9, "b": 0.5},
        tiers={"b": "advisory"},
    )
    assert out["passed"] is True
    assert out["advisory_failed"] == ["b"]
    assert out["results"]["b"] == {
        "minimum": 0.5,
        "actual": 0.1,
        "passed": False,
        "tier": "advisory",
    }
    blocking = check_gates({"a": 0.95, "b": 0.1}, {"a": 0.9, "b": 0.5})
    assert blocking["passed"] is False
    assert blocking["advisory_failed"] == []


# --- slice gates ---------------------------------------------------------------


def test_slice_gate_reads_the_slice_rate_and_an_empty_slice_fails() -> None:
    slices = {
        "camera": {
            "G339": {"n": 4.0, "passed": 3.0, "pass_rate": 0.75, "pass_rate_w95": 0.3},
            "G341": {"n": 0.0},
        }
    }
    assert gate_actual("slice:camera=G339", {}, slices) == 0.75
    assert gate_actual("slice:camera=G341", {}, slices) is None
    out = check_gates({}, {"slice:camera=G339": 0.5, "slice:camera=G341": 0.5}, slices=slices)
    assert out["results"]["slice:camera=G339"]["passed"] is True
    assert out["results"]["slice:camera=G341"]["actual"] is None
    assert out["passed"] is False


# --- counts ----------------------------------------------------------------------


def _dataset(tmp_path: Path, rows: list[dict]) -> Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir(exist_ok=True)
    with (prepared / "test.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return prepared


def _fake_model_def(evaluation: dict) -> SimpleNamespace:
    return SimpleNamespace(
        evaluation=evaluation,
        smoke={},
        dataset={},
        data={},
        name="m",
        version="0.1.0",
        model_id="m",
        task="t",
    )


def test_harness_records_counts_and_lifts_plugin_counts(tmp_path: Path) -> None:
    rows = [
        {"request": "a", "family": "f1"},
        {"request": "b", "family": "f1"},
        {"request": "c", "family": "f2"},
    ]

    def metrics_plugin(row_results):
        return {"acc": 2 / 3, "__counts__": {"acc": [2, 3]}}

    report = run_evaluation(
        checkpoint_dir=tmp_path / "ckpt",
        dataset_dir=_dataset(tmp_path, rows),
        out_path=tmp_path / "eval" / "r.json",
        predictor=lambda row: "{}",
        metrics_fn=metrics_plugin,
        device="cpu",
    )
    assert "__counts__" not in report.metrics
    assert report.counts["acc"] == {"k": 2, "n": 3}
    assert report.counts["output_nonempty_rate"] == {"k": 3, "n": 3}


def test_harness_counts_layer_rates_and_slices(tmp_path: Path) -> None:
    from maatml.evaluation.harness import SliceSpec

    rows = [{"request": "a", "camera": "G339"}, {"request": "b", "camera": "G339"}]
    report = run_evaluation(
        checkpoint_dir=tmp_path / "ckpt",
        dataset_dir=_dataset(tmp_path, rows),
        out_path=tmp_path / "eval" / "r.json",
        predictor=lambda row: "{}" if row["request"] == "a" else "not json",
        device="cpu",
        slices=[SliceSpec(field="camera")],
    )
    assert report.counts["all_layers_pass_rate"] == {"k": 1, "n": 2}
    assert report.counts["slice:camera=G339"] == {"k": 1, "n": 2}


# --- population stamp -----------------------------------------------------------


def test_gate_enforcement_records_the_split_and_flags_a_foreign_floor(tmp_path: Path) -> None:
    rows = [{"request": "a"}, {"request": "b"}]
    md = _fake_model_def({"gates": {"all_layers_pass_rate": 0.0}, "gates_benchmark": "0" * 64})
    report = run_evaluation(
        checkpoint_dir=tmp_path / "ckpt",
        dataset_dir=_dataset(tmp_path, rows),
        out_path=tmp_path / "eval" / "r.json",
        model_def=md,
        predictor=lambda row: "{}",
        device="cpu",
        enforce_gates=True,
    )
    assert report.gates is not None
    assert report.gates["benchmark_sha256"] == report.extras["split_sha256"]
    assert report.gates["floors_benchmark_sha256"] == "0" * 64
    assert report.gates["population_mismatch"] is True
    assert report.passed is True  # a warning, unless strict

    class NeverLoadPredictor:
        def setup(self, *args, **kwargs):
            pytest.fail("population mismatch must fail before loading the model")

    with pytest.raises(GateConfigError, match="different population"):
        run_evaluation(
            checkpoint_dir=tmp_path / "ckpt",
            dataset_dir=_dataset(tmp_path, rows),
            out_path=tmp_path / "eval" / "r2.json",
            model_def=md,
            predictor=NeverLoadPredictor(),
            device="cpu",
            enforce_gates=True,
            strict_population=True,
        )


# --- derive ----------------------------------------------------------------------

_MODEL_YML = """name: derive-test
model_id: derive-test
version: 0.1.0
architecture: causal_sft
dataset:
  format: jsonl_seed
  seed_samples: datasets/samples/seed_samples.jsonl
evaluation:
  predictor: causal_sft
  gates:
    all_layers_pass_rate: 0.5   # typed by hand
    precision: {min: 0.5, tier: advisory}
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


def _write_report(
    mdir: Path,
    run: str,
    *,
    split_sha256: str = "a" * 64,
    metrics: dict | None = None,
    counts: dict | None = None,
) -> Path:
    path = mdir / "output" / "eval" / f"{run}.json"
    Report(
        model_id="derive-test",
        dataset="output/prepared/test.jsonl",
        n=100,
        metrics=metrics or {},
        counts=counts or {},
        extras={"split_sha256": split_sha256},
    ).write(path)
    return path


def test_derive_floors_from_counts_with_wilson_and_stamps_the_split(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    _write_report(
        mdir,
        "run-a",
        metrics={"all_layers_pass_rate": 0.9, "precision": 0.8},
        counts={"all_layers_pass_rate": {"k": 90, "n": 100}, "precision": {"k": 40, "n": 50}},
    )
    result = derive_gates(load_model_def(mdir, load_plugins=False), runs=["run-a"])
    assert result.benchmark_sha256 == "a" * 64
    floor = result.floors["all_layers_pass_rate"]
    assert floor.method == "wilson"
    assert floor.value == floor2(wilson_lower(90, 100))
    assert "90/100 = 0.900, w95" in floor.comment
    assert floor.comment.endswith("@ bench aaaaaaaaaaaaaaaa")
    assert result.floors["precision"].value == floor2(wilson_lower(40, 50))
    assert result.refusals == []


def test_derive_takes_the_minimum_across_runs_and_refuses_thin_denominators(
    tmp_path: Path,
) -> None:
    mdir = _model_dir(tmp_path)
    _write_report(
        mdir,
        "run-a",
        metrics={"all_layers_pass_rate": 0.9, "precision": 0.8},
        counts={"all_layers_pass_rate": {"k": 90, "n": 100}, "precision": {"k": 8, "n": 10}},
    )
    _write_report(
        mdir,
        "run-b",
        metrics={"all_layers_pass_rate": 0.7, "precision": 0.8},
        counts={"all_layers_pass_rate": {"k": 70, "n": 100}, "precision": {"k": 8, "n": 10}},
    )
    result = derive_gates(load_model_def(mdir, load_plugins=False), runs=["run-a", "run-b"])
    floor = result.floors["all_layers_pass_rate"]
    assert floor.run == "run-b"
    assert floor.value == floor2(wilson_lower(70, 100))
    assert floor.comment.startswith("min over 2 runs (run-b):")
    assert "precision" not in result.floors
    assert any("precision: 8/10" in r and "too thin" in r for r in result.refusals)


def test_derive_floors_a_metric_without_counts_at_its_observed_value(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    _write_report(
        mdir,
        "run-a",
        metrics={"all_layers_pass_rate": 0.9, "precision": 0.8, "map_50": 0.213},
        counts={"all_layers_pass_rate": {"k": 90, "n": 100}, "precision": {"k": 80, "n": 100}},
    )
    result = derive_gates(
        load_model_def(mdir, load_plugins=False), runs=["run-a"], metrics=["map_50"]
    )
    floor = result.floors["map_50"]
    assert floor.method == "observed"
    assert floor.value == 0.21
    assert "not a derived bound" in floor.comment


def test_derive_refuses_mixed_splits_and_unversioned_reports(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    _write_report(mdir, "run-a", counts={"all_layers_pass_rate": {"k": 9, "n": 100}})
    _write_report(
        mdir, "run-b", split_sha256="b" * 64, counts={"all_layers_pass_rate": {"k": 9, "n": 100}}
    )
    md = load_model_def(mdir, load_plugins=False)
    with pytest.raises(DeriveError, match="different splits"):
        derive_gates(md, runs=["run-a", "run-b"])
    legacy = mdir / "output" / "eval" / "old.json"
    legacy.write_text(json.dumps({"model_id": "m", "dataset": "d", "n": 1, "metrics": {}}))
    with pytest.raises(Exception, match="predates"):
        derive_gates(md, runs=["old"])
    with pytest.raises(DeriveError, match="no eval report"):
        derive_gates(md, runs=["missing"])


def _clustered_cache(path: Path, split_sha256: str) -> None:
    # Ten families of twenty rows; seven pass entirely, three fail entirely.
    rows = []
    for fam in range(10):
        for i in range(20):
            rows.append(
                {
                    "sample_id": f"{fam}-{i}",
                    "row": {"family": f"fam{fam}"},
                    "output": "{}",
                    "ok": fam < 7,
                    "passed_layers": [1] if fam < 7 else [],
                    "errors": [],
                    "parsed": None,
                    "latency_ms": 1.0,
                }
            )
    write_predictions(path, header={"split": "test", "split_sha256": split_sha256}, rows=rows)


def test_cluster_bootstrap_is_below_row_level_wilson_on_clustered_rows(tmp_path: Path) -> None:
    groups = [(20, 20)] * 7 + [(0, 20)] * 3
    boot = cluster_bootstrap_lower(groups, seed="t")
    assert boot < wilson_lower(140, 200)
    assert cluster_bootstrap_lower(groups, seed="t") == boot  # deterministic


def test_derive_uses_the_cluster_bootstrap_when_a_cache_exists(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    report_path = _write_report(
        mdir,
        "run-a",
        metrics={"all_layers_pass_rate": 0.7, "precision": 0.8},
        counts={"all_layers_pass_rate": {"k": 140, "n": 200}, "precision": {"k": 80, "n": 100}},
    )
    _clustered_cache(predictions_path(report_path), "a" * 64)
    result = derive_gates(load_model_def(mdir, load_plugins=False), runs=["run-a"])
    floor = result.floors["all_layers_pass_rate"]
    assert floor.method == "cluster_bootstrap"
    assert "over 10 family groups" in floor.comment
    assert floor.value < floor2(wilson_lower(140, 200))
    # A plugin rate has no per-row verdict in the cache: Wilson, as before.
    assert result.floors["precision"].method == "wilson"


def test_derive_refuses_a_clustered_rate_with_too_few_groups(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    report_path = _write_report(
        mdir, "run-a", counts={"all_layers_pass_rate": {"k": 140, "n": 200}}
    )
    _clustered_cache(predictions_path(report_path), "a" * 64)
    result = derive_gates(
        load_model_def(mdir, load_plugins=False),
        runs=["run-a"],
        metrics=["all_layers_pass_rate"],
        min_groups=50,
    )
    assert result.floors == {}
    assert any("spans 10 family group" in r for r in result.refusals)


def test_harness_rate_groups_returns_none_for_plugin_metrics() -> None:
    rows = [{"row": {"family": "f"}, "ok": True, "output": "x", "passed_layers": [1]}]
    assert harness_rate_groups(rows, "precision", "family") is None
    assert harness_rate_groups(rows, "all_layers_pass_rate", "family") == [(1, 1)]
    assert harness_rate_groups(rows, "output_nonempty_rate", "family") == [(1, 1)]
    assert harness_rate_groups(rows, "layer_1_pass_rate", "family") == [(1, 1)]
    assert harness_rate_groups(rows, "layer_2_pass_rate", "family") == [(0, 1)]


# --- write -------------------------------------------------------------------------


def test_write_rewrites_the_block_keeps_tiers_and_stamps_the_benchmark(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    _write_report(
        mdir,
        "run-a",
        metrics={"all_layers_pass_rate": 0.9, "precision": 0.8},
        counts={"all_layers_pass_rate": {"k": 90, "n": 100}, "precision": {"k": 80, "n": 100}},
    )
    md = load_model_def(mdir, load_plugins=False)
    result = derive_gates(md, runs=["run-a"])
    write_gates(mdir / "model.yml", result, gate_tiers(md))
    text = (mdir / "model.yml").read_text()
    assert "typed by hand" not in text
    assert f"  gates_benchmark: {'a' * 64}\n  gates:\n" in text
    assert "    precision: {min: " in text and "tier: advisory}" in text
    expected = f"{floor2(wilson_lower(90, 100)):.2f}"
    assert f"    all_layers_pass_rate: {expected}  # 90/100 = 0.900, w95" in text
    # The file still loads, and the floors are what was derived.
    reloaded = load_model_def(mdir, load_plugins=False)
    assert effective_gates(reloaded) == {
        "all_layers_pass_rate": float(expected),
        "precision": floor2(wilson_lower(80, 100)),
    }
    assert gate_tiers(reloaded) == {"all_layers_pass_rate": "blocking", "precision": "advisory"}
    assert reloaded.evaluation["gates_benchmark"] == "a" * 64
    # Everything outside the block is untouched.
    assert "training:\n  base_model: sshleifer/tiny-gpt2" in text


def test_write_refuses_with_no_surviving_floor(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    _write_report(mdir, "run-a", counts={"all_layers_pass_rate": {"k": 1, "n": 2}})
    md = load_model_def(mdir, load_plugins=False)
    result = derive_gates(md, runs=["run-a"], metrics=["all_layers_pass_rate"])
    with pytest.raises(DeriveError, match="no floor survived"):
        write_gates(mdir / "model.yml", result, {})


def test_rewrite_quotes_slice_keys_and_replaces_an_existing_stamp() -> None:
    from maatml.evaluation.gates import DeriveResult, Floor

    text = (
        "evaluation:\n  gates_benchmark: old\n  gates:\n    x: 0.1\nsmoke:\n  gates:\n    y: 0.2\n"
    )
    result = DeriveResult(section="evaluation", runs=["r"], benchmark_sha256="new")
    result.floors["slice:camera=G339"] = Floor("slice:camera=G339", 0.4, "c", "wilson", "r")
    out = rewrite_gates_block(text, result, {})
    assert out == (
        'evaluation:\n  gates_benchmark: new\n  gates:\n    "slice:camera=G339": 0.40  # c\n'
        "smoke:\n  gates:\n    y: 0.2\n"
    )


# --- CLI -----------------------------------------------------------------------------


def test_cli_gates_derive_prints_floors_and_writes(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    _write_report(
        mdir,
        "run-a",
        metrics={"all_layers_pass_rate": 0.9, "precision": 0.8},
        counts={"all_layers_pass_rate": {"k": 90, "n": 100}, "precision": {"k": 8, "n": 10}},
    )
    result = runner.invoke(app, ["gates", "derive", str(mdir), "--run", "run-a"])
    assert result.exit_code == 0, result.output
    assert f"all_layers_pass_rate: {floor2(wilson_lower(90, 100)):.2f}" in result.output
    assert "refused" in result.output and "precision" in result.output

    result = runner.invoke(app, ["gates", "derive", str(mdir), "--run", "run-a", "--write"])
    assert result.exit_code == 0, result.output
    assert "wrote 1 floors" in result.output
    assert "gates_benchmark" in (mdir / "model.yml").read_text()


def test_cli_gates_derive_reports_a_missing_run(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    result = runner.invoke(app, ["gates", "derive", str(mdir), "--run", "nope"])
    assert result.exit_code != 0
    assert "no eval report" in result.output


def test_rewrite_opens_an_inline_empty_gates_mapping() -> None:
    import yaml

    from maatml.evaluation.gates import DeriveResult, Floor

    text = "evaluation:\n  slices: [family]\n  gates: {}\n\ntraining:\n  epochs: 1\n"
    result = DeriveResult(section="evaluation", runs=["r"], benchmark_sha256="b" * 64)
    result.floors["decision_accuracy_rate"] = Floor(
        "decision_accuracy_rate", 0.97, "c", "wilson", "r"
    )
    out = rewrite_gates_block(text, result, {})
    loaded = yaml.safe_load(out)
    assert loaded["evaluation"]["gates"] == {"decision_accuracy_rate": 0.97}
    assert loaded["evaluation"]["gates_benchmark"] == "b" * 64
    assert loaded["training"] == {"epochs": 1}
    assert "gates: {}" not in out
