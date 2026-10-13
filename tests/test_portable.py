"""runs --pack / --adopt: a run record travels with its evidence; jobs do not."""

from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from maatml.cli import app
from maatml.config import load_model_def
from maatml.portable import BundleError, adopt_bundle, pack_run, read_bundle_manifest
from maatml.runs import RunRecord, _append_record, get_run, list_runs, resolve_checkpoint

runner = CliRunner()

_MODEL_YML = """name: travel
model_id: travel
version: 0.1.0
architecture: causal_sft
dataset:
  format: jsonl_seed
  seed_samples: datasets/samples/seed_samples.jsonl
evaluation:
  gates:
    output_nonempty_rate: 0.5
training:
  base_model: sshleifer/tiny-gpt2
  learning_rate: {lr}
"""


def _folder(tmp_path: Path, name: str, *, lr: str = "1e-4") -> Path:
    mdir = tmp_path / name
    (mdir / "datasets" / "samples").mkdir(parents=True)
    (mdir / "datasets" / "samples" / "seed_samples.jsonl").write_text("")
    (mdir / "model.yml").write_text(_MODEL_YML.replace("{lr}", lr))
    return mdir


def _trained(mdir: Path, run_id: str = "run-a") -> Path:
    md = load_model_def(mdir, load_plugins=False)
    run = md.checkpoints_dir / run_id
    (run / "checkpoint-100").mkdir(parents=True)
    (run / "checkpoint-100" / "optimizer.pt").write_bytes(b"o" * 64)
    (run / "adapter_model.safetensors").write_bytes(b"w" * 32)
    (run / "chat_template.jinja").write_text("{{ messages }}")
    (run / "run_metadata.json").write_text(json.dumps({"identity": md.identity}))
    md.eval_dir.mkdir(parents=True)
    (md.eval_dir / f"{run_id}.json").write_text(json.dumps({"metrics": {"a": 1.0}}))
    (md.eval_dir / f"{run_id}.predictions.jsonl").write_text("{}\n")
    (md.eval_dir / "other.json").write_text("{}")
    (md.eval_dir / "select" / run_id).mkdir(parents=True)
    (md.eval_dir / "select" / run_id / "final.val.json").write_text("{}")
    _append_record(
        md,
        RunRecord(
            run_id=run_id,
            identity=md.identity,
            architecture="causal_sft",
            status="completed",
            started_at="2026-01-01T00:00:00Z",
            out_dir=str(run),
            metrics={"eval_loss": 0.5},
            gates={"passed": True, "results": {}, "smoke": False},
            selected_checkpoint=None,
            test_spends=[{"benchmark_sha256": "ab" * 32}],
        ),
    )
    return run


def test_pack_carries_the_run_its_evidence_and_the_record_but_not_checkpoints(
    tmp_path: Path,
) -> None:
    mdir = _folder(tmp_path, "origin")
    _trained(mdir)
    md = load_model_def(mdir, load_plugins=False)
    bundle = pack_run(md, "run-a")
    assert bundle == md.output_dir / "bundles" / "run-a.maatml-run.tar.gz"
    with tarfile.open(bundle) as tar:
        names = sorted(tar.getnames())
    assert names == [
        "bundle.json",
        "eval/run-a.json",
        "eval/run-a.predictions.jsonl",
        "eval/select/run-a/final.val.json",
        "run/adapter_model.safetensors",
        "run/chat_template.jinja",
        "run/run_metadata.json",
    ]
    manifest = read_bundle_manifest(bundle)
    assert manifest["run_id"] == "run-a" and manifest["identity"] == "travel@0.1.0"
    assert manifest["record"]["test_spends"] == [{"benchmark_sha256": "ab" * 32}]
    assert manifest["packed_environment"]["kind"] == "maatml.environment/1"
    assert len(manifest["files"]) == 6

    with_ckpt = pack_run(md, "run-a", out=tmp_path, with_checkpoints=True)
    with tarfile.open(with_ckpt) as tar:
        assert "run/checkpoint-100/optimizer.pt" in tar.getnames()

    with pytest.raises(BundleError, match="not in runs.jsonl"):
        pack_run(md, "run-zzz")


def test_adopt_reproduces_the_runs_listing_on_the_receiving_folder(tmp_path: Path) -> None:
    origin = _folder(tmp_path, "origin")
    _trained(origin)
    origin_md = load_model_def(origin, load_plugins=False)
    bundle = pack_run(origin_md, "run-a", out=tmp_path)
    before = runner.invoke(app, ["runs", str(origin)]).output

    home = _folder(tmp_path, "home")
    result = runner.invoke(app, ["runs", str(home), "--adopt", str(bundle)])
    assert result.exit_code == 0, result.output
    home_md = load_model_def(home, load_plugins=False)
    rec = get_run(home_md, "run-a")
    assert rec is not None
    assert rec.out_dir == str((home_md.checkpoints_dir / "run-a").resolve())
    assert rec.adopted_from == {
        "bundle": bundle.name,
        "spec_hash": rec.spec_hash or read_bundle_manifest(bundle)["spec_hash"],
        "forced": False,
    }
    assert rec.gates == {"passed": True, "results": {}, "smoke": False}
    assert (home_md.checkpoints_dir / "run-a" / "chat_template.jinja").is_file()
    assert not (home_md.checkpoints_dir / "run-a" / "checkpoint-100").exists()
    assert (home_md.eval_dir / "run-a.json").is_file()
    assert (home_md.eval_dir / "select" / "run-a" / "final.val.json").is_file()
    assert not (home_md.eval_dir / "other.json").exists()
    assert resolve_checkpoint(home_md, "run-a") == (home_md.checkpoints_dir / "run-a").resolve()

    after = runner.invoke(app, ["runs", str(home)]).output

    def normal(text: str, folder: Path) -> str:
        # rich wraps long lines at the terminal width, mid-path; only the folder differs.
        return re.sub(r"\s+", "", text).replace(str(folder.resolve()), "<dir>")

    assert normal(before, origin) == normal(after, home)


def test_adopt_refuses_another_recipe_or_identity_without_force(tmp_path: Path) -> None:
    origin = _folder(tmp_path, "origin")
    _trained(origin)
    bundle = pack_run(load_model_def(origin, load_plugins=False), "run-a", out=tmp_path)

    other = _folder(tmp_path, "other", lr="5e-5")
    other_md = load_model_def(other, load_plugins=False)
    with pytest.raises(BundleError, match="recipe differs"):
        adopt_bundle(other_md, bundle)
    assert list_runs(other_md) == []
    rec = adopt_bundle(other_md, bundle, force=True)
    assert rec.adopted_from["forced"] is True

    renamed = _folder(tmp_path, "renamed")
    (renamed / "model.yml").write_text(
        (renamed / "model.yml").read_text().replace("name: travel", "name: elsewhere")
    )
    with pytest.raises(BundleError, match="packed for travel@0.1.0"):
        adopt_bundle(load_model_def(renamed, load_plugins=False), bundle)

    home = _folder(tmp_path, "home")
    home_md = load_model_def(home, load_plugins=False)
    adopt_bundle(home_md, bundle)
    with pytest.raises(BundleError, match="already"):
        adopt_bundle(home_md, bundle)
    assert adopt_bundle(home_md, bundle, force=True).run_id == "run-a"


def test_adopt_checks_every_file_against_the_manifest(tmp_path: Path) -> None:
    origin = _folder(tmp_path, "origin")
    _trained(origin)
    bundle = pack_run(load_model_def(origin, load_plugins=False), "run-a", out=tmp_path)
    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(bundle) as src, tarfile.open(tampered, "w:gz") as dst:
        for member in src.getmembers():
            data = src.extractfile(member).read() if member.isfile() else b""
            if member.name == "run/chat_template.jinja":
                data = b"{{ tampered }}"
                member.size = len(data)
            dst.addfile(member, __import__("io").BytesIO(data))
    home = _folder(tmp_path, "home")
    with pytest.raises(BundleError, match="does not match its manifest hash"):
        adopt_bundle(load_model_def(home, load_plugins=False), tampered)

    not_a_bundle = tmp_path / "x.tar.gz"
    with tarfile.open(not_a_bundle, "w:gz"):
        pass
    with pytest.raises(BundleError, match="not a maatml run bundle"):
        adopt_bundle(load_model_def(home, load_plugins=False), not_a_bundle)
