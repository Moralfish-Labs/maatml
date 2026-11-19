"""Publish hash-bound completion evidence for single-process SFT checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from ..utils.io import sha256_file

READY_FILE = "maatml-checkpoint-ready.json"
READY_KIND = "maatml.sft-checkpoint-ready/1"


def _has_regular_file(checkpoint: Path, name: str, files: Optional[set[str]]) -> bool:
    if files is not None and name not in files:
        return False
    target = checkpoint / name
    return not target.is_symlink() and target.is_file()


def _has_sharded_weights(checkpoint: Path, index_name: str, files: Optional[set[str]]) -> bool:
    if not _has_regular_file(checkpoint, index_name, files):
        return False
    try:
        data = json.loads((checkpoint / index_name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shards: set[str] = set()
    for shard in weight_map.values():
        if (
            not isinstance(shard, str)
            or not shard
            or Path(shard).name != shard
            or not shard.endswith(".safetensors")
        ):
            return False
        shards.add(shard)
    return all(_has_regular_file(checkpoint, shard, files) for shard in shards)


def has_model_weights(checkpoint: Path, files: Optional[set[str]] = None) -> bool:
    """True when ``config.json`` and single-file or sharded safetensors exist."""
    if not _has_regular_file(checkpoint, "config.json", files):
        return False
    return _has_regular_file(checkpoint, "model.safetensors", files) or _has_sharded_weights(
        checkpoint, "model.safetensors.index.json", files
    )


def has_adapter_weights(checkpoint: Path, files: Optional[set[str]] = None) -> bool:
    """True when ``adapter_config.json`` and single-file or sharded adapter weights exist."""
    if not _has_regular_file(checkpoint, "adapter_config.json", files):
        return False
    return _has_regular_file(
        checkpoint, "adapter_model.safetensors", files
    ) or _has_sharded_weights(checkpoint, "adapter_model.safetensors.index.json", files)


def checkpoint_files(checkpoint: Path) -> dict[str, str]:
    """Hash regular checkpoint files without following links."""
    files = {}
    if checkpoint.is_symlink():
        raise ValueError("SFT checkpoint must not be a symbolic link")
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise ValueError("SFT checkpoint contains a symbolic link")
        if path.is_file() and path.name not in {READY_FILE, READY_FILE + ".tmp"}:
            files[path.relative_to(checkpoint).as_posix()] = sha256_file(path)
    return files


def publish_sft_checkpoint(checkpoint: Path, step: int, spec_hash: str) -> None:
    """Called only after HF Trainer completes its checkpoint save."""
    checkpoint = Path(checkpoint)
    files = checkpoint_files(checkpoint)
    required = {"trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"}
    file_names = set(files.keys())
    if not required <= file_names or not (
        has_adapter_weights(checkpoint, file_names) or has_model_weights(checkpoint, file_names)
    ):
        raise ValueError("SFT snapshot requires complete single-process optimizer and weights")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state.get("global_step") != step or checkpoint.name != f"checkpoint-{step}":
        raise ValueError("SFT checkpoint step mismatch")
    payload = {"kind": READY_KIND, "step": step, "spec_hash": spec_hash, "files": files}
    raw = json.dumps(payload, sort_keys=True).encode()
    target = checkpoint / READY_FILE
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError("Refusing to change a published SFT checkpoint")
        return
    temporary = checkpoint / (READY_FILE + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def verify_sft_checkpoint(checkpoint: Path, step: int, spec_hash: str) -> None:
    """Verify completion evidence before and after copying a live checkpoint."""
    marker = checkpoint / READY_FILE
    if marker.is_symlink():
        raise ValueError("SFT completion marker must not be a symbolic link")
    payload = json.loads(marker.read_text())
    if (
        payload.get("kind") != READY_KIND
        or payload.get("step") != step
        or payload.get("spec_hash") != spec_hash
        or not payload.get("files")
        or payload["files"] != checkpoint_files(checkpoint)
    ):
        raise ValueError("SFT checkpoint completion evidence differs from saved files")
    required = {"trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"}
    files = payload["files"]
    file_names = set(files.keys())
    if not required <= file_names or not (
        has_adapter_weights(checkpoint, file_names) or has_model_weights(checkpoint, file_names)
    ):
        raise ValueError("SFT completion evidence lacks resumable state")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state.get("global_step") != step or checkpoint.name != f"checkpoint-{step}":
        raise ValueError("SFT checkpoint step mismatch")


def make_snapshot_ready_callback(spec_hash: str) -> Any:
    """Load the optional Trainer API only for an opted-in SFT trainer."""
    from transformers import TrainerCallback

    class SnapshotReadyCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                publish_sft_checkpoint(checkpoint, state.global_step, spec_hash)
            return control

    return SnapshotReadyCallback()
