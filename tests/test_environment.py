"""The environment manifest is on every train record and every evaluate."""

from __future__ import annotations

import json
from pathlib import Path

from maatml.config import load_model_def
from maatml.environment import ENVIRONMENT_KIND, environment_manifest, render_environment
from maatml.evaluation.harness import run_evaluation
from maatml.runs import get_run, start_run


def test_manifest_has_every_declared_field_and_never_raises(tmp_path: Path) -> None:
    env = environment_manifest(tmp_path)
    assert env["kind"] == ENVIRONMENT_KIND
    for key in ("git_sha", "python", "os", "packages", "cuda", "cudnn", "gpus", "driver", "mps"):
        assert key in env
    assert env["packages"]["maatml"] is not None
    assert set(env["determinism"]) >= {
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONHASHSEED",
        "deterministic_algorithms",
    }
    # A directory that is not a git checkout records no SHA rather than a guess.
    assert env["git_sha"] is None or len(env["git_sha"]) == 40
    assert render_environment(env)[0].startswith("git ")


def test_train_record_carries_the_environment_and_spec_fingerprint(tmp_path: Path) -> None:
    mdir = tmp_path / "model"
    (mdir / "datasets").mkdir(parents=True)
    (mdir / "datasets" / "s.jsonl").write_text("")
    (mdir / "model.yml").write_text(
        "name: m\nmodel_id: m\nversion: 0.1.0\narchitecture: causal_sft\n"
        "dataset:\n  seed_samples: datasets/s.jsonl\ntraining:\n  base_model: x\n"
    )
    md = load_model_def(mdir, load_plugins=False)
    run = start_run(md, device="cpu")
    rec = get_run(md, run.run_id)
    assert rec is not None
    assert rec.environment["kind"] == ENVIRONMENT_KIND
    assert rec.environment["python"] == environment_manifest()["python"]
    assert rec.spec_hash and len(rec.spec_hash) == 64


def test_evaluate_records_its_environment_on_the_report_and_the_gates(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    with (prepared / "test.jsonl").open("w") as handle:
        for i in range(3):
            handle.write(json.dumps({"sample_id": f"s{i}", "request": "a", "expected": "b"}) + "\n")
    report = run_evaluation(
        checkpoint_dir=tmp_path / "ckpt",
        dataset_dir=prepared,
        out_path=tmp_path / "eval" / "r.json",
        predictor=lambda row: "ok",
        device="cpu",
        enforce_gates=True,
        gate_spec={"output_nonempty_rate": 0.5},
    )
    assert report.extras["environment"]["kind"] == ENVIRONMENT_KIND
    assert report.gates["environment"]["python"] == report.extras["environment"]["python"]
