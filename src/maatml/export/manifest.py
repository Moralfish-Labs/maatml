"""Export ``manifest.json`` build / verify helpers."""

from __future__ import annotations

import json
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import ModelDefinition, PackagingSpec
from ..utils.io import read_json, sha256_file, write_json


def _file_entries(root: Path, files: list[Path]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        entries.append({"path": rel, "sha256": sha256_file(path)})
    return entries


def _read_safetensors_header(path: str | Path) -> Optional[dict[str, Any]]:
    """Parse the JSON header of a ``.safetensors`` file, or None on failure.

    Reads only the header (a u64 length prefix + that many header bytes), so
    it needs neither ``torch`` nor the ``safetensors`` package and works in
    the CPU-free environment.
    """
    try:
        file_size = Path(path).stat().st_size
        with open(path, "rb") as fh:
            size_bytes = fh.read(8)
            if len(size_bytes) < 8:
                return None
            (header_len,) = struct.unpack("<Q", size_bytes)
            # The header must fit within the file after its 8-byte length
            # prefix. Bounding here keeps a dummy/corrupt file from triggering a
            # multi-gigabyte read (MemoryError) before we can reject it.
            if header_len <= 0 or header_len > file_size - 8:
                return None
            header_bytes = fh.read(header_len)
        if len(header_bytes) < header_len:
            return None
        header = json.loads(header_bytes.decode("utf-8"))
    except (OSError, struct.error, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return header if isinstance(header, dict) else None


def read_safetensors_dtypes(path: str | Path) -> list[str]:
    """Return the dtype of every tensor in a ``.safetensors`` file.

    Header-only read; returns ``[]`` for anything it cannot parse.
    """
    header = _read_safetensors_header(path)
    if header is None:
        return []
    dtypes: list[str] = []
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        if isinstance(spec, dict) and isinstance(spec.get("dtype"), str):
            dtypes.append(spec["dtype"])
    return dtypes


def read_safetensors_tensor_names(path: str | Path) -> list[str]:
    """Return every tensor name in a ``.safetensors`` file.

    Header-only read; returns ``[]`` for anything it cannot parse.
    """
    header = _read_safetensors_header(path)
    if header is None:
        return []
    return [key for key in header if key != "__metadata__"]


def _observed_weights_dtype(files: list[Path]) -> tuple[Optional[str], list[str]]:
    """Most-common tensor dtype across exported safetensors, and the full set.

    Returns ``(dominant_dtype, sorted_unique_dtypes)`` normalised to lowercase
    (``"F16"`` → ``"f16"``). ``dominant_dtype`` is ``None`` when no safetensors
    file is present, i.e. the dtype could not be verified from tensors.
    """
    counts: Counter[str] = Counter()
    for path in files:
        if path.suffix == ".safetensors":
            counts.update(dt.lower() for dt in read_safetensors_dtypes(path))
    if not counts:
        return None, []
    dominant = counts.most_common(1)[0][0]
    return dominant, sorted(counts)


def build_manifest(
    *,
    model_def: ModelDefinition,
    export_dir: Path,
    files: list[Path],
    formats: list[str],
    source_checkpoint: str | Path,
    run_id: Optional[str] = None,
    packaging: Optional[PackagingSpec] = None,
    extra_runtime_hints: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble an export manifest (inspired by legacy ``.fm`` manifests)."""
    pkg = packaging or model_def.packaging
    observed_dtype, observed_all = _observed_weights_dtype(files)
    hints: dict[str, Any] = {
        "formats": list(formats),
        "max_input_tokens": pkg.max_input_tokens,
        "expected_latency_ms": pkg.expected_latency_ms,
        # `weights_dtype` is the verified dtype when it can be read from the
        # exported tensors, and falls back to the declared hint otherwise.
        # `weights_dtype_declared` always records what packaging claimed, and
        # `weights_dtype_verified` says whether the value came from the tensors.
        "weights_dtype": observed_dtype or pkg.weights_dtype,
        "weights_dtype_declared": pkg.weights_dtype,
        "weights_dtype_verified": observed_dtype is not None,
    }
    if len(observed_all) > 1:
        # Mixed-precision export, surface every dtype rather than hide it.
        hints["weights_dtypes_observed"] = observed_all
    if extra_runtime_hints:
        hints.update(extra_runtime_hints)

    manifest: dict[str, Any] = {
        "name": model_def.name,
        "version": model_def.version,
        "identity": model_def.identity,
        "architecture": model_def.architecture,
        "base_model": model_def.base_model or model_def.training.get("model_id"),
        "runtime_hints": hints,
        "packaging": pkg.model_dump(mode="json"),
        "files": _file_entries(export_dir, files),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source_checkpoint),
    }
    if run_id:
        manifest["run_id"] = run_id
        manifest["gate_evidence"] = _gate_evidence(model_def, run_id)
    lock = _corpus_lock(model_def)
    if lock is not None:
        manifest["corpus_lock"] = lock
    return manifest


def _corpus_lock(model_def: ModelDefinition) -> Optional[dict[str, Any]]:
    """The prepare-time corpus lock, so a bundle names the sources that trained it."""
    try:
        from ..data.attribution import read_corpus_lock

        lock = read_corpus_lock(Path(model_def.prepared_dir))
    except Exception:  # noqa: BLE001  a manifest must not fail on a torn lock
        return None
    if lock is None:
        return None
    return {k: v for k, v in lock.items() if k != "kind"}


def _gate_evidence(model_def: ModelDefinition, run_id: str) -> dict[str, Any]:
    """What gate result backs this bundle, and at which tier.

    A bundle exported from a smoke-tier pass has to say so: the thresholds a
    rehearsal met are not the production ones, and a manifest that recorded
    only "passed" would let the two be mistaken for each other later.
    """
    evidence: dict[str, Any] = {"run_id": run_id, "gated": False, "smoke_gated": False}
    try:
        from ..runs import get_run

        record = get_run(model_def, run_id)
    except Exception:  # noqa: BLE001  a manifest must not fail on registry issues
        return evidence
    if record is None:
        return evidence
    gates = record.gates or {}
    evidence["gated"] = bool(gates)
    evidence["passed"] = gates.get("passed")
    evidence["smoke_gated"] = bool(record.smoke_gated or gates.get("smoke"))
    return evidence


def write_manifest(export_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write ``manifest.json``; a lock with sources also ships as ``ATTRIBUTION.md``."""
    from ..data.attribution import ATTRIBUTION_FILE, write_attribution_file

    export_dir = Path(export_dir)
    lock = manifest.get("corpus_lock")
    if isinstance(lock, dict) and lock.get("sources"):
        path = write_attribution_file(export_dir, lock)
        entries = [e for e in manifest.get("files", []) if e.get("path") != ATTRIBUTION_FILE]
        entries.extend(_file_entries(export_dir, [path]))
        manifest["files"] = entries
    return write_json(export_dir / "manifest.json", manifest)


def amend_manifest(
    export_dir: str | Path,
    files: list[Path],
    *,
    formats: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Add files (and optionally format names) to an existing manifest.

    For artifacts produced outside the exporter, a quantized GGUF being the
    canonical case: without an entry here they sit beside the bundle unverifiable,
    and a catalog checksum can never be traced back to a gate-evidenced export.
    Existing entries for the same path are replaced; ``gate_evidence`` and every
    other field are left untouched, since amending adds artifacts to a bundle
    rather than re-arguing what the bundle proved.
    """
    root, manifest = load_manifest(export_dir)
    entries: list[dict[str, str]] = list(manifest.get("files", []))
    by_path = {entry["path"]: index for index, entry in enumerate(entries)}
    for file in files:
        path = Path(file).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"cannot amend manifest with missing file: {path}")
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"{path} is outside the export directory {root}; the manifest "
                "only describes files the bundle actually ships"
            ) from exc
        entry = {"path": rel, "sha256": sha256_file(path)}
        if rel in by_path:
            entries[by_path[rel]] = entry
        else:
            by_path[rel] = len(entries)
            entries.append(entry)
    manifest["files"] = entries
    if formats:
        hints = manifest.setdefault("runtime_hints", {})
        known = list(hints.get("formats", []))
        for fmt in formats:
            if fmt not in known:
                known.append(fmt)
        hints["formats"] = known
    write_json(root / "manifest.json", manifest)
    return manifest


def load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load a manifest from a file path or an export directory."""
    path = Path(path).resolve()
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}")
    data = read_json(manifest_path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest (expected object): {manifest_path}")
    return manifest_path.parent, data


def verify_manifest(path: str | Path) -> list[str]:
    """Recompute sha256 for listed files; return a list of mismatch messages.

    Empty list means OK. Missing files are reported as mismatches.
    """
    root, data = load_manifest(path)
    files = data.get("files") or []
    errors: list[str] = []
    if not isinstance(files, list):
        return ["manifest.files must be a list"]
    for entry in files:
        if not isinstance(entry, dict):
            errors.append(f"invalid file entry: {entry!r}")
            continue
        rel = entry.get("path")
        expected = entry.get("sha256")
        if not rel or not expected:
            errors.append(f"incomplete file entry: {entry!r}")
            continue
        target = root / rel
        if not target.is_file():
            errors.append(f"missing file: {rel}")
            continue
        actual = sha256_file(target)
        if actual != expected:
            errors.append(f"checksum mismatch: {rel} (expected {expected}, got {actual})")
    return errors
