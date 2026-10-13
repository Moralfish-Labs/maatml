"""Populations: isolation hierarchy, pins, benchmark version, blind evaluate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from maatml.cli import app
from maatml.config import load_model_def
from maatml.data.pipeline import prepare
from maatml.data.populations import (
    Isolation,
    Pin,
    PopulationError,
    apply_pins,
    benchmark_version,
    check_isolation,
    check_prepared_isolation,
    read_benchmark_state,
    resolve_isolation,
    resolve_pins,
)
from maatml.evaluation.harness import GateConfigError
from maatml.evaluation.runner import evaluate_model
from maatml.registry import PREDICTORS
from maatml.runs import RunRecord, _append_record, list_runs
from maatml.utils.io import iter_jsonl

runner = CliRunner()

# --- config ----------------------------------------------------------------------


def test_isolation_and_pins_parse_and_validate() -> None:
    cfg = {
        "isolation": {
            "fields": ["clip", "camera", "site"],
            "policy": {"val": "camera", "blind": "site"},
        },
        "pins": {"val": ["camera:G339"], "benchmark": ["camera:G341"]},
    }
    iso = resolve_isolation(cfg)
    assert iso == Isolation(
        fields=("clip", "camera", "site"), policy={"val": "camera", "blind": "site"}
    )
    pins = resolve_pins(cfg, iso)
    assert pins == [Pin("val", "camera", "G339"), Pin("benchmark", "camera", "G341")]
    assert resolve_isolation({"isolation": ["family"]}) == Isolation(fields=("family",))
    assert resolve_isolation({}) is None and resolve_pins({}, None) == []


@pytest.mark.parametrize(
    "cfg",
    [
        {"isolation": "camera"},
        {"isolation": {"fields": []}},
        {"isolation": {"fields": ["camera"], "policy": {"train": "camera"}}},
        {"isolation": {"fields": ["camera"], "policy": {"val": "site"}}},
        {"pins": {"train": ["camera:G1"]}},
        {"pins": {"val": ["G1"]}},
        {"isolation": ["camera"], "pins": {"val": ["site:S1"]}},
        {"pins": {"val": ["camera:G1"], "benchmark": ["camera:G1"]}},
    ],
)
def test_malformed_isolation_or_pins_are_errors(cfg) -> None:
    with pytest.raises(PopulationError):
        resolve_pins(cfg, resolve_isolation(cfg))


def test_apply_pins_moves_whole_groups_and_refuses_an_empty_pin() -> None:
    rows = {
        "train": [{"camera": "G1"}, {"camera": "G2"}, {"camera": "G3"}],
        "val": [{"camera": "G2"}],
        "test": [],
    }
    moved = apply_pins(rows, [Pin("val", "camera", "G2"), Pin("benchmark", "camera", "G3")])
    assert moved == {"camera:G2": 2, "camera:G3": 1}
    assert [r["camera"] for r in rows["train"]] == ["G1"]
    assert all(r["split"] == "val" for r in rows["val"]) and len(rows["val"]) == 2
    assert rows["test"][0]["camera"] == "G3" and rows["test"][0]["split"] == "test"
    with pytest.raises(PopulationError, match="matches no row"):
        apply_pins(rows, [Pin("val", "camera", "G9")])


def test_check_isolation_names_shared_levels_and_missing_fields() -> None:
    iso = Isolation(fields=("camera", "site"), policy={"val": "camera", "blind": "site"})
    populations = {
        "train": [{"camera": "G1", "site": "A"}],
        "val": [{"camera": "G1", "site": "A"}, {"site": "B"}],
        "blind": [{"camera": "G9", "site": "A"}],
    }
    problems = check_isolation(populations, iso)
    assert any("val is camera-disjoint" in p and "G1" in p for p in problems)
    assert any("lack isolation field 'camera'" in p for p in problems)
    assert any("blind is site-disjoint" in p and "'A'" in p for p in problems)
    clean = {"train": [{"camera": "G1", "site": "A"}], "val": [{"camera": "G2", "site": "A"}]}
    assert check_isolation(clean, iso) == []


def test_benchmark_version_is_order_insensitive_and_pin_sensitive() -> None:
    rows = [{"sample_id": "a", "split": "test"}, {"sample_id": "b", "split": "test"}]
    assert benchmark_version(rows, []) == benchmark_version(list(reversed(rows)), [])
    assert benchmark_version(rows, []) != benchmark_version(
        rows, [Pin("benchmark", "camera", "G1")]
    )
    assert benchmark_version(rows, []) != benchmark_version(rows[:1], [])


# --- prepare -----------------------------------------------------------------------

_MODEL_YML = """name: pop-test
model_id: pop-test
version: 0.1.0
architecture: causal_sft
dataset:
  format: jsonl_seed
  seed: 7
  seed_samples: datasets/samples/seed_samples.jsonl
  group_by: family
  split_ratios: [0.6, 0.2, 0.2]
  isolation:
    fields: [family, camera, site]
    policy: {val: camera, benchmark: camera, blind: site}
  pins:
    val: ["camera:G339"]
    benchmark: ["camera:G341"]
{extra}
evaluation:
  predictor: blind_test
  gates:
    output_nonempty_rate: 0.5
training:
  base_model: sshleifer/tiny-gpt2
"""


def _row(i: int, camera: str, site: str, family: str | None = None) -> dict:
    return {
        "sample_id": f"s{i}",
        "request": f"ask {i}",
        "expected": "ok",
        "family": family or f"{camera}-clip{i % 3}",
        "camera": camera,
        "site": site,
    }


def _seed_rows() -> list[dict]:
    rows = []
    i = 0
    for camera, site in (
        ("G331", "bus"),
        ("G506", "bus"),
        ("G339", "school"),
        ("G341", "hospital"),
    ):
        for _ in range(9):
            rows.append(_row(i, camera, site))
            i += 1
    return rows


def _model_dir(tmp_path: Path, *, extra: str = "", blind: list[dict] | None = None) -> Path:
    mdir = tmp_path / "model"
    samples = mdir / "datasets" / "samples"
    samples.mkdir(parents=True)
    with (samples / "seed_samples.jsonl").open("w") as handle:
        for row in _seed_rows():
            handle.write(json.dumps(row) + "\n")
    if blind is not None:
        with (samples / "blind_v001.jsonl").open("w") as handle:
            for row in blind:
                handle.write(json.dumps(row) + "\n")
        extra = extra + "  blind_samples: datasets/samples/blind_v001.jsonl\n"
    (mdir / "model.yml").write_text(_MODEL_YML.replace("{extra}", extra))
    return mdir


def _cameras(path: Path) -> set[str]:
    return {row["camera"] for row in iter_jsonl(path)}


def test_prepare_pins_cameras_and_writes_the_benchmark_version(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    md = load_model_def(mdir, load_plugins=False)
    summary = prepare(md)
    prepared = md.prepared_dir
    assert _cameras(prepared / "val.jsonl") == {"G339"}
    assert _cameras(prepared / "test.jsonl") == {"G341"}
    assert _cameras(prepared / "train.jsonl") == {"G331", "G506"}
    assert summary["pins"] == {"camera:G339": 9, "camera:G341": 9}
    state = read_benchmark_state(prepared)
    assert state is not None and state["version"] == summary["benchmark_version"]
    assert state["n"] == 9 and state["pins"] == ["benchmark=camera:G341", "val=camera:G339"]
    assert "Benchmark version:" in (prepared / "dataset_card.md").read_text()
    assert check_prepared_isolation(md) == []


def test_prepare_refuses_a_benchmark_camera_that_also_trains(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path, extra="  benchmark_samples: datasets/samples/bench_v001.jsonl\n")
    bench = mdir / "datasets" / "samples" / "bench_v001.jsonl"
    bench.write_text(json.dumps(_row(900, "G331", "bus", family="bench-G331")) + "\n")
    with pytest.raises(ValueError, match="benchmark is camera-disjoint from train"):
        prepare(load_model_def(mdir, load_plugins=False))


def test_prepare_checks_blind_rows_and_never_writes_them(tmp_path: Path) -> None:
    leaking = [_row(950, "G999", "bus", family="blind-1")]
    mdir = _model_dir(tmp_path, blind=leaking)
    with pytest.raises(ValueError, match="blind is site-disjoint from train"):
        prepare(load_model_def(mdir, load_plugins=False))

    disjoint = [
        _row(960, "G999", "admin", family="blind-1"),
        _row(961, "G998", "admin", family="blind-2"),
    ]
    mdir = _model_dir(tmp_path / "ok", blind=disjoint)
    md = load_model_def(mdir, load_plugins=False)
    summary = prepare(md)
    assert summary["blind_rows"] == 2
    for split in ("train", "val", "test"):
        assert "admin" not in {
            row["site"] for row in iter_jsonl(md.prepared_dir / f"{split}.jsonl")
        }


def test_prepare_refuses_an_in_place_benchmark_edit_but_accepts_a_new_version(
    tmp_path: Path,
) -> None:
    mdir = _model_dir(tmp_path, extra="  benchmark_samples: datasets/samples/bench_v001.jsonl\n")
    bench = mdir / "datasets" / "samples" / "bench_v001.jsonl"
    bench.write_text(json.dumps(_row(900, "G421", "school", family="bench-G421")) + "\n")
    md = load_model_def(mdir, load_plugins=False)
    first = prepare(md)["benchmark_version"]
    bench.write_text(
        bench.read_text() + json.dumps(_row(901, "G421", "school", family="bench-G421")) + "\n"
    )
    with pytest.raises(PopulationError, match="changed in place"):
        prepare(load_model_def(mdir, load_plugins=False))
    v2 = bench.with_name("bench_v002.jsonl")
    v2.write_text(bench.read_text())
    yml = mdir / "model.yml"
    yml.write_text(yml.read_text().replace("bench_v001", "bench_v002"))
    second = prepare(load_model_def(mdir, load_plugins=False))["benchmark_version"]
    assert second != first


def test_audit_reports_a_prepared_split_that_breaks_isolation(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    md = load_model_def(mdir, load_plugins=False)
    prepare(md)
    with (md.prepared_dir / "train.jsonl").open("a") as handle:
        handle.write(json.dumps({**_row(999, "G339", "school"), "split": "train"}) + "\n")
    problems = check_prepared_isolation(md)
    assert problems and "G339" in problems[0]
    result = runner.invoke(app, ["audit", str(mdir)])
    assert "isolation" in result.output and "G339" in result.output


# --- blind evaluate -------------------------------------------------------------------


class BlindPredictor:
    def setup(self, checkpoint_dir, **kwargs):
        return None

    def predict(self, row):
        return "{}"


@pytest.fixture(autouse=True)
def _register_predictor():
    PREDICTORS.register("blind_test", BlindPredictor, source="test")
    yield
    PREDICTORS.unregister("blind_test")


def _gated_candidate(tmp_path: Path) -> tuple[Path, object]:
    blind = [
        _row(960, "G999", "admin", family="blind-1"),
        _row(961, "G998", "admin", family="blind-2"),
    ]
    mdir = _model_dir(tmp_path, blind=blind)
    md = load_model_def(mdir, load_plugins=False)
    prepare(md)
    ckpt = md.output_dir / "checkpoints" / "run-a"
    ckpt.mkdir(parents=True)
    (ckpt / "weights.bin").write_bytes(b"w")
    _append_record(
        md,
        RunRecord(
            run_id="run-a",
            identity="pop-test@0.1.0",
            architecture="causal_sft",
            status="completed",
            started_at="2026-01-01T00:00:00Z",
            out_dir=str(ckpt),
        ),
    )
    return mdir, md


def test_blind_is_spent_once_on_a_gated_unchanged_candidate(tmp_path: Path) -> None:
    mdir, md = _gated_candidate(tmp_path)
    with pytest.raises(GateConfigError, match="no production gate pass"):
        evaluate_model(md, checkpoint="run-a", device="cpu", blind=True)

    report, path = evaluate_model(md, checkpoint="run-a", device="cpu", gate=True)
    assert report.passed is True
    assert report.extras["benchmark_version"] == read_benchmark_state(md.prepared_dir)["version"]
    rec = list_runs(md)[0]
    assert rec.gated_fingerprint

    blind_report, blind_path = evaluate_model(md, checkpoint="run-a", device="cpu", blind=True)
    assert blind_path.name == "run-a.blind.json"
    assert blind_report.n == 2 and blind_report.passed is True
    assert blind_report.gates is not None
    spends = list_runs(md)[0].blind_spends
    assert spends and spends[0]["report"] == "run-a.blind.json" and spends[0]["forced"] is False

    with pytest.raises(GateConfigError, match="already spent"):
        evaluate_model(md, checkpoint="run-a", device="cpu", blind=True)
    evaluate_model(md, checkpoint="run-a", device="cpu", blind=True, force=True)
    assert len(list_runs(md)[0].blind_spends) == 2
    assert list_runs(md)[0].blind_spends[1]["forced"] is True

    listing = runner.invoke(app, ["runs", str(mdir)])
    assert "blind-spends=2" in listing.output


def test_blind_refuses_a_candidate_changed_since_its_gate_pass(tmp_path: Path) -> None:
    _mdir, md = _gated_candidate(tmp_path)
    evaluate_model(md, checkpoint="run-a", device="cpu", gate=True)
    md.evaluation["score_thresh"] = 0.9
    with pytest.raises(GateConfigError, match="changed since its gate pass"):
        evaluate_model(md, checkpoint="run-a", device="cpu", blind=True)


def test_blind_needs_a_blind_manifest(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    md = load_model_def(mdir, load_plugins=False)
    with pytest.raises(GateConfigError, match="dataset.blind_samples"):
        evaluate_model(md, checkpoint="nope", device="cpu", blind=True)
