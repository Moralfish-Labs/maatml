"""Consistent, optimizer-bearing snapshots for native workflow backup nodes."""

from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .config import ModelDefinition
from .environment import environment_manifest
from .portable import BUNDLE_KIND, MANIFEST, BundleError
from .runs import RunRecord, list_runs, spec_fingerprint
from .utils.io import sha256_file


def _writer_active(lock: Path) -> bool:
    if os.name != "posix":
        raise BundleError("live checkpoint observation requires POSIX writer locks")
    import fcntl

    with lock.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False


def snapshot_checkpoint(
    model_def: ModelDefinition,
    *,
    started_after: Path,
    writer_lock: Path,
    min_step: int,
    out: Path,
    poll_seconds: float = 2.0,
    startup_seconds: float = 120.0,
) -> Path:
    """Wait for a complete policy checkpoint, then atomically publish its bundle.

    The policy trainer publishes training_state.pt last and never modifies a
    completed checkpoint. Copying first isolates the archive from log updates
    and checkpoint pruning. No model or optimizer files are deserialized here.
    The startup timeout only bounds this observer; it never stops the trainer.
    """
    if min_step < 1 or poll_seconds <= 0:
        raise BundleError("min_step and poll_seconds must be positive")
    since = started_after.stat().st_mtime
    startup_deadline = time.monotonic() + startup_seconds
    fingerprint = spec_fingerprint(model_def)
    while True:
        records = [
            rec
            for rec in list_runs(model_def)
            if datetime.fromisoformat(rec.started_at.replace("Z", "+00:00")).timestamp() >= since
        ]
        if len(records) > 1:
            raise BundleError("multiple runs started after the launch marker; refusing ambiguity")
        if not records:
            if time.monotonic() >= startup_deadline:
                raise BundleError("trainer did not register a run after the launch marker")
            time.sleep(poll_seconds)
            continue
        record = records[0]
        if record.spec_hash != fingerprint:
            raise BundleError("live run recipe fingerprint differs from the receiving model")
        if Path(record.run_id).name != record.run_id or record.run_id in {".", ".."}:
            raise BundleError("invalid run identifier")
        run_dir = Path(model_def.checkpoints_dir) / record.run_id
        if run_dir.is_symlink():
            raise BundleError("snapshot refuses symbolic run directories")
        candidates = []
        for checkpoint in run_dir.glob("checkpoint-*"):
            if checkpoint.is_symlink():
                continue
            suffix = checkpoint.name.removeprefix("checkpoint-")
            if (
                suffix.isdigit()
                and int(suffix) >= min_step
                and (checkpoint / "training_state.pt").is_file()
                and (checkpoint / "model.safetensors").is_file()
                and (checkpoint / "config.json").is_file()
            ):
                candidates.append((int(suffix), checkpoint))
        if candidates:
            step, checkpoint = max(candidates)
            try:
                return _publish(model_def, record, run_dir, checkpoint, step, out)
            except FileNotFoundError:
                # A checkpoint may be pruned during the copy. Never publish a
                # partial archive; retry against the next complete checkpoint.
                pass
        if record.status != "running" or not _writer_active(writer_lock):
            raise BundleError(f"trainer stopped before a complete checkpoint >= {min_step}")
        time.sleep(poll_seconds)


def _publish(
    model_def: ModelDefinition,
    record: RunRecord,
    run_dir: Path,
    checkpoint: Path,
    step: int,
    out: Path,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=out.parent) as scratch:
        root = Path(scratch)
        files = [p for p in run_dir.iterdir() if p.is_file() and not p.name.endswith(".tmp")]
        files += [p for p in checkpoint.rglob("*") if p.is_file()]
        entries = []
        for source in sorted(files):
            if source.is_symlink():
                raise BundleError("snapshot refuses symbolic links in run evidence")
            relative = source.relative_to(run_dir)
            target = root / "run" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            entries.append(("run/" + relative.as_posix(), target))
        manifest = {
            "kind": BUNDLE_KIND,
            "run_id": record.run_id,
            "identity": model_def.identity,
            "spec_hash": record.spec_hash,
            "record": record.model_dump(mode="json"),
            "packed_environment": environment_manifest(model_def.model_dir),
            "with_checkpoints": True,
            "snapshot_step": step,
            "files": [],
        }
        archive = root / "snapshot.tar.gz"
        for name, target in entries:
            digest = sha256_file(target)
            manifest["files"].append({"path": name, "sha256": digest})
        with tarfile.open(archive, "w:gz") as tar:
            body = json.dumps(manifest, sort_keys=True).encode()
            info = tarfile.TarInfo(MANIFEST)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
            for name, target in entries:
                tar.add(target, arcname=name, recursive=False)
        with archive.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(archive, out)
    return out
