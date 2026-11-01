"""Snapshots carry consistent recovery evidence and never bless incomplete state."""

import hashlib
import os
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from maatml.cli import app
from maatml.config import load_model_def
from maatml.portable import BundleError, adopt_bundle, read_bundle_manifest
from maatml.runs import RunRecord, _append_record, spec_fingerprint
from maatml.snapshot import snapshot_checkpoint

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX writer-lock protocol")


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    (root / "model.yml").write_text(
        "name: snapshot-test\nmodel_id: snapshot-test\nversion: 0.1.0\narchitecture: causal_sft\n"
        "training:\n  base_model: test\n"
    )
    md = load_model_def(root, load_plugins=False)
    marker = root / "launch.marker"
    marker.touch()
    os.utime(marker, (1, 1))
    run = md.checkpoints_dir / "run-a"
    checkpoint = run / "checkpoint-500"
    checkpoint.mkdir(parents=True)
    for name in ["model.safetensors", "config.json", "training_state.pt"]:
        (checkpoint / name).write_bytes(b"immutable " + name.encode())
    (run / "validation-baseline.json").write_text("{}")
    (run / "mining-epoch-0.json.gz").write_bytes(b"frozen mining")
    (run / "training-progress.jsonl").write_text('{"step":500}\n')
    rec = RunRecord(
        run_id="run-a",
        identity=md.identity,
        architecture="causal_sft",
        status="running",
        started_at="2026-01-01T00:00:00Z",
        out_dir=str(run),
        spec_hash=spec_fingerprint(md),
    )
    _append_record(md, rec)
    monkeypatch.setattr("maatml.snapshot.environment_manifest", lambda _: {})
    return md, marker, run, checkpoint, rec


def call(md, marker, out, **kwargs):
    return snapshot_checkpoint(
        md,
        started_after=marker,
        writer_lock=marker.parent / "writer.lock",
        min_step=500,
        out=out,
        poll_seconds=0.001,
        **kwargs,
    )


def test_snapshot_hashes_adopt_and_contains_optimizer(tmp_path, monkeypatch):
    md, marker, run, checkpoint, rec = fixture(tmp_path, monkeypatch)
    out = call(md, marker, tmp_path / "backup.tar.gz")
    manifest = read_bundle_manifest(out)
    assert manifest["snapshot_step"] == 500
    assert manifest["record"]["status"] == "running"
    with tarfile.open(out) as tar:
        for item in manifest["files"]:
            assert (
                hashlib.sha256(tar.extractfile(item["path"]).read()).hexdigest() == item["sha256"]
            )
        assert "run/checkpoint-500/training_state.pt" in tar.getnames()
        assert "run/validation-baseline.json" in tar.getnames()
        assert "run/mining-epoch-0.json.gz" in tar.getnames()
    # Native adoption verifies all hashes and reconstructs resumable checkpoint paths.
    import shutil

    shutil.rmtree(run)
    adopted = adopt_bundle(md, out, force=True)
    assert (
        Path(adopted.out_dir, "checkpoint-500/training_state.pt")
        .read_bytes()
        .startswith(b"immutable")
    )


def test_incomplete_checkpoint_or_dead_writer_fails_closed(tmp_path, monkeypatch):
    md, marker, run, checkpoint, rec = fixture(tmp_path, monkeypatch)
    (checkpoint / "training_state.pt").unlink()
    with pytest.raises(BundleError, match="trainer stopped"):
        call(md, marker, tmp_path / "backup.tar.gz")
    assert not (tmp_path / "backup.tar.gz").exists()


def test_wrong_recipe_is_refused(tmp_path, monkeypatch):
    md, marker, run, checkpoint, rec = fixture(tmp_path, monkeypatch)
    rec.spec_hash = "wrong"
    _append_record(md, rec)
    with pytest.raises(BundleError, match="fingerprint"):
        call(md, marker, tmp_path / "backup.tar.gz")


def test_old_run_is_not_selected(tmp_path, monkeypatch):
    md, marker, run, checkpoint, rec = fixture(tmp_path, monkeypatch)
    os.utime(marker, (4102444800, 4102444800))
    with pytest.raises(BundleError, match="did not register"):
        call(md, marker, tmp_path / "backup.tar.gz", startup_seconds=0)


def test_pruning_race_does_not_replace_previous_good_bundle(tmp_path, monkeypatch):
    md, marker, run, checkpoint, rec = fixture(tmp_path, monkeypatch)
    out = tmp_path / "backup.tar.gz"
    out.write_bytes(b"last good backup")

    def disappear(*args):
        raise FileNotFoundError("pruned")

    monkeypatch.setattr("maatml.snapshot._publish", disappear)
    with pytest.raises(BundleError, match="trainer stopped"):
        call(md, marker, out)
    assert out.read_bytes() == b"last good backup"


def test_cli_exposes_native_snapshot_command():
    result = CliRunner().invoke(app, ["snapshot", "--help"])
    assert result.exit_code == 0
    assert "--min-step" in result.output


def test_waits_for_atomic_optimizer_completion(tmp_path, monkeypatch):
    import fcntl
    import threading
    import time

    md, marker, run, checkpoint, rec = fixture(tmp_path, monkeypatch)
    state = checkpoint / "training_state.pt"
    state.unlink()
    lock = marker.parent / "writer.lock"
    with lock.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)

        def complete():
            time.sleep(0.03)
            temporary = checkpoint / "training_state.pt.tmp"
            temporary.write_bytes(b"optimizer")
            temporary.replace(state)

        worker = threading.Thread(target=complete)
        worker.start()
        try:
            result = call(md, marker, tmp_path / "complete.tar.gz")
        finally:
            worker.join()
        assert read_bundle_manifest(result)["snapshot_step"] == 500


def test_symlink_checkpoint_is_not_backed_up(tmp_path, monkeypatch):
    import shutil

    md, marker, run, checkpoint, rec = fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    shutil.move(str(checkpoint), outside)
    checkpoint.symlink_to(outside, target_is_directory=True)
    with pytest.raises(BundleError, match="trainer stopped"):
        call(md, marker, tmp_path / "backup.tar.gz")
