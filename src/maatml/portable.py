"""``maatml runs --pack RUN`` / ``--adopt BUNDLE``: a run record that travels.

``runs.jsonl`` is written where training happened and does not follow the
weights. A run carried home as a bare checkpoint directory loses its gate
pass, its spends, its selection and the environment it ran in. A bundle is
the run directory (without ``checkpoint-*`` unless asked), the run's eval
reports and caches, the registry record, the environment manifest, and the
model-folder fingerprint it was produced under, in one tar. ``adopt``
refuses a bundle from a different model folder without ``--force``, places
the files where the receiving folder expects them, and appends the record
with its paths rewritten. Records move; jobs do not.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any, Optional

from .config import ModelDefinition
from .runs import RunRecord, _append_record, get_run, spec_fingerprint
from .utils.io import sha256_file

BUNDLE_KIND = "maatml.run_bundle/1"
BUNDLE_SUFFIX = ".maatml-run.tar.gz"
MANIFEST = "bundle.json"


class BundleError(ValueError):
    """A bundle cannot be built or adopted as asked."""


def _run_files(run_dir: Path, *, with_checkpoints: bool) -> list[Path]:
    files: list[Path] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir)
        if not with_checkpoints and rel.parts and rel.parts[0].startswith("checkpoint-"):
            continue
        files.append(path)
    return files


def _eval_files(model_def: ModelDefinition, run_id: str) -> list[Path]:
    eval_dir = Path(model_def.eval_dir)
    if not eval_dir.is_dir():
        return []
    files = [
        p for p in sorted(eval_dir.iterdir()) if p.is_file() and p.name.startswith(f"{run_id}.")
    ]
    select = eval_dir / "select" / run_id
    if select.is_dir():
        files.extend(p for p in sorted(select.rglob("*")) if p.is_file())
    return files


def bundle_path(model_def: ModelDefinition, run_id: str, out: Optional[Path] = None) -> Path:
    if out is not None:
        out = Path(out)
        return out / f"{run_id}{BUNDLE_SUFFIX}" if out.is_dir() else out
    return Path(model_def.output_dir) / "bundles" / f"{run_id}{BUNDLE_SUFFIX}"


def pack_run(
    model_def: ModelDefinition,
    run_id: str,
    *,
    out: Optional[Path] = None,
    with_checkpoints: bool = False,
) -> Path:
    """Write ``<run_id>.maatml-run.tar.gz``; returns its path."""
    rec = get_run(model_def, run_id)
    if rec is None:
        raise BundleError(f"{run_id} is not in runs.jsonl")
    run_dir = Path(rec.out_dir)
    if not run_dir.is_dir():
        run_dir = Path(model_def.checkpoints_dir) / run_id
    if not run_dir.is_dir():
        raise BundleError(f"{run_id}: run directory {rec.out_dir} is missing")
    from .environment import environment_manifest

    entries: list[tuple[str, Path]] = []
    for path in _run_files(run_dir, with_checkpoints=with_checkpoints):
        entries.append((f"run/{path.relative_to(run_dir).as_posix()}", path))
    eval_dir = Path(model_def.eval_dir)
    for path in _eval_files(model_def, run_id):
        entries.append((f"eval/{path.relative_to(eval_dir).as_posix()}", path))
    manifest: dict[str, Any] = {
        "kind": BUNDLE_KIND,
        "run_id": run_id,
        "identity": model_def.identity,
        "spec_hash": rec.spec_hash or spec_fingerprint(model_def),
        "record": rec.model_dump(mode="json"),
        "packed_environment": environment_manifest(model_def.model_dir),
        "with_checkpoints": bool(with_checkpoints),
        "files": [{"path": name, "sha256": sha256_file(path)} for name, path in entries],
    }
    target = bundle_path(model_def, run_id, out)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    with tarfile.open(target, "w:gz") as tar:
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
        for name, path in entries:
            tar.add(path, arcname=name, recursive=False)
    return target


def read_bundle_manifest(bundle: Path) -> dict[str, Any]:
    with tarfile.open(bundle, "r:gz") as tar:
        try:
            member = tar.extractfile(MANIFEST)
        except KeyError as exc:
            raise BundleError(f"{bundle.name}: no {MANIFEST}; not a maatml run bundle") from exc
        if member is None:
            raise BundleError(f"{bundle.name}: {MANIFEST} is not a file")
        payload = json.loads(member.read().decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != BUNDLE_KIND:
        raise BundleError(f"{bundle.name}: not a {BUNDLE_KIND} bundle")
    return payload


def _safe_member(member: tarfile.TarInfo) -> bool:
    name = member.name
    return (
        member.isfile()
        and not name.startswith("/")
        and ".." not in Path(name).parts
        and name.startswith(("run/", "eval/"))
    )


def adopt_bundle(model_def: ModelDefinition, bundle: Path, *, force: bool = False) -> RunRecord:
    """Unpack a bundle into this model folder and append its record."""
    bundle = Path(bundle)
    if not bundle.is_file():
        raise BundleError(f"bundle not found: {bundle}")
    manifest = read_bundle_manifest(bundle)
    run_id = str(manifest["run_id"])
    here = spec_fingerprint(model_def)
    if manifest.get("identity") != model_def.identity and not force:
        raise BundleError(
            f"{bundle.name} was packed for {manifest.get('identity')}, this folder is "
            f"{model_def.identity}; pass --force to adopt it anyway"
        )
    if manifest.get("spec_hash") != here and not force:
        raise BundleError(
            f"{bundle.name} was packed under model.yml {str(manifest.get('spec_hash'))[:12]}, "
            f"this folder is {here[:12]}: the recipe differs, so the evidence would describe "
            "another model. Pass --force to adopt it anyway; the record keeps the packed hash."
        )
    run_dir = Path(model_def.checkpoints_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise BundleError(f"{run_dir} already exists; pass --force to overwrite its files")
    existing = get_run(model_def, run_id)
    if existing is not None and not force:
        raise BundleError(f"{run_id} is already in runs.jsonl; pass --force to replace it")

    expected = {f["path"]: f["sha256"] for f in manifest.get("files", [])}
    eval_dir = Path(model_def.eval_dir)
    with tarfile.open(bundle, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name == MANIFEST or not _safe_member(member):
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            data = source.read()
            digest = hashlib.sha256(data).hexdigest()
            if expected.get(member.name) != digest:
                raise BundleError(f"{bundle.name}: {member.name} does not match its manifest hash")
            rel = Path(member.name)
            dest = (
                (run_dir / Path(*rel.parts[1:]))
                if rel.parts[0] == "run"
                else (eval_dir / Path(*rel.parts[1:]))
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)

    payload = dict(manifest["record"])
    payload["out_dir"] = str(run_dir.resolve())
    payload["adopted_from"] = {
        "bundle": bundle.name,
        "spec_hash": manifest.get("spec_hash"),
        "forced": bool(force),
    }
    record = RunRecord(**payload)
    _append_record(model_def, record)
    return record
