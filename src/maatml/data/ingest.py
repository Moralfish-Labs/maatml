"""Ingest external JSON/JSONL into a model's seed corpus with validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..config import ModelDefinition, get_dataset_cfg
from ..registry import SANITIZERS, VALIDATORS
from ..utils.io import (
    iter_jsonl,
    sha256_file,
    stable_hash,
    write_json,
    write_jsonl_atomic,
)


def _read_input(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return list(iter_jsonl(path))
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    if isinstance(raw, dict):
        # Allow {"samples": [...]} wrappers.
        for key in ("samples", "data", "rows"):
            if isinstance(raw.get(key), list):
                return [r for r in raw[key] if isinstance(r, dict)]
        return [raw]
    raise ValueError(f"Unsupported JSON shape in {path}")


def _apply_field_map(row: dict[str, Any], field_map: dict[str, str]) -> dict[str, Any]:
    """Map ``target_field=source_col`` into a new row (unmapped keys kept)."""
    if not field_map:
        return dict(row)
    out = dict(row)
    for dest, src in field_map.items():
        if src in row:
            out[dest] = row[src]
    return out


def _row_id(row: dict[str, Any]) -> str:
    if row.get("sample_id"):
        return str(row["sample_id"])
    return stable_hash(json.dumps(row, sort_keys=True, default=str))[:16]


def _is_unapproved_capture(row: dict[str, Any]) -> bool:
    """A serve-captured row that has not been reviewed and approved.

    Capture writes ``source: serve_capture`` with ``approved: false``. A
    reviewer sets ``approved: true`` (and typically fixes the target) before
    ingesting; anything still marked unapproved, or a capture with the approval
    flag stripped out, is refused.
    """
    if row.get("source") != "serve_capture":
        return False
    return row.get("approved") is not True


def ingest_samples(
    model_def: ModelDefinition,
    input_path: str | Path,
    *,
    field_map: Optional[dict[str, str]] = None,
    sanitize_tag: Optional[str] = None,
    append: bool = True,
    out_path: Optional[str | Path] = None,
    video: Optional[str | Path] = None,
    images_rel: str = "datasets/samples/images",
    fps: Optional[float] = None,
) -> dict[str, Any]:
    """Ingest rows into ``seed_samples``, validating gold targets when configured.

    Writes a rejection report JSON next to the seed file
    (``<stem>.ingest_rejected.json``).

    The seed corpus is written atomically, and only when at least one row was
    accepted: a run that accepts nothing leaves an existing corpus untouched and
    reports ``protected_existing``, including under ``append=False``.
    """
    input_path = Path(input_path).resolve()
    cfg = get_dataset_cfg(model_def)
    seed_rel = cfg.get("seed_samples") or "datasets/samples/seed_samples.jsonl"
    seeds_path = Path(out_path) if out_path else model_def.resolve(str(seed_rel))
    seeds_path.parent.mkdir(parents=True, exist_ok=True)

    request_field = str(cfg.get("request_field") or cfg.get("raw_field") or "request")
    target_field = str(cfg.get("target_field") or "target")

    fmap = dict(field_map or {})
    rows_in = _read_input(input_path)
    for dest, src in fmap.items():
        if not any(src in r for r in rows_in):
            raise ValueError(f"--map {dest}={src}: source column {src!r} matches zero input rows")
    mapped = [_apply_field_map(r, fmap) for r in rows_in]
    if video is not None:
        from .video import materialize_video_rows

        mapped = materialize_video_rows(
            mapped,
            video,
            images_dir=model_def.resolve(images_rel),
            images_rel=images_rel,
            request_field=request_field,
            fps=fps,
        )

    sanitizer = None
    if sanitize_tag:
        sanitizer = SANITIZERS.get(sanitize_tag)
        if sanitizer is None:
            raise KeyError(
                f"Unknown sanitizer {sanitize_tag!r}. Known: "
                f"{', '.join(SANITIZERS.names()) or '(none)'}"
            )

    validator_name = (model_def.evaluation or {}).get("validator")
    validate_fn = VALIDATORS.get(validator_name) if validator_name else None
    schema_path = None
    contracts_path = None
    if validate_fn is not None:
        if isinstance(cfg.get("schema"), str):
            schema_path = model_def.resolve(cfg["schema"])
        if isinstance(cfg.get("contracts"), str):
            contracts_path = model_def.resolve(cfg["contracts"])

    existing: list[dict[str, Any]] = []
    seen: set[str] = set()
    if append and seeds_path.is_file():
        for row in iter_jsonl(seeds_path):
            existing.append(row)
            seen.add(_row_id(row))

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    skipped_unvalidated: list[dict[str, Any]] = []
    provenance = f"ingest:{input_path.name}"

    unapproved_capture = 0
    for row in mapped:
        sample = dict(row)

        # A serve-captured row is a model proposal, not a label. It only becomes
        # a seed after a human (or teacher) has corrected and explicitly approved
        # it; an unapproved capture is refused rather than trusted.
        if _is_unapproved_capture(sample):
            unapproved_capture += 1
            rejected.append({**sample, "_reject_reason": "unapproved_capture"})
            continue
        # A row that was reviewed and approved sheds its review markers so it
        # reads as an ordinary seed downstream.
        sample.pop("approved", None)
        sample.pop("needs_review", None)

        if sanitizer is not None and request_field in sample:
            sample[request_field] = sanitizer(str(sample[request_field]))

        sid = _row_id(sample)
        sample.setdefault("sample_id", sid)
        sample["source"] = provenance
        # Every ingested row shares one source, so without a per-row family
        # they would hash into a single split group and land in one split.
        sample.setdefault("family", str(sample["sample_id"]))

        if sid in seen:
            rejected.append({**sample, "_reject_reason": "duplicate"})
            continue

        if validate_fn is not None:
            # A validator is configured: a row with no gold target cannot be
            # gated, so it is counted and excluded, never silently accepted.
            if target_field not in sample:
                skipped_unvalidated.append({**sample, "_skip_reason": "missing_target"})
                continue
            gold = sample[target_field]
            raw = gold if isinstance(gold, str) else json.dumps(gold)
            try:
                result = validate_fn(
                    raw,
                    schema_path=schema_path,
                    contracts_path=contracts_path,
                    user_prompt=sample.get(request_field),
                )
                ok = bool(getattr(result, "ok", False))
            except Exception as exc:  # noqa: BLE001
                ok = False
                sample["_validate_error"] = str(exc)
            if not ok:
                rejected.append({**sample, "_reject_reason": "validator"})
                continue

        seen.add(sid)
        accepted.append(sample)

    # Never truncate or rewrite a non-empty seed file when nothing was accepted.
    # A validator that rejected everything (or an empty input) must not destroy a
    # hand-curated corpus, and --no-append must not turn a zero-accept run into
    # an empty file.
    seed_existing_nonempty = seeds_path.is_file() and seeds_path.stat().st_size > 0
    if accepted:
        out_rows = existing + accepted if append else accepted
        write_jsonl_atomic(seeds_path, out_rows)
        seed_written = True
        protected_existing = False
    else:
        seed_written = False
        protected_existing = seed_existing_nonempty
        out_rows = (
            existing if append else (list(iter_jsonl(seeds_path)) if seeds_path.is_file() else [])
        )

    reject_path = seeds_path.with_name(seeds_path.stem + ".ingest_rejected.json")
    report = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path) if input_path.is_file() else None,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "skipped_unvalidated": len(skipped_unvalidated),
        "unapproved_capture": unapproved_capture,
        "total_seeds": len(out_rows),
        "seed_written": seed_written,
        "protected_existing": protected_existing,
        "rejected_rows": rejected,
        "skipped_unvalidated_rows": skipped_unvalidated,
    }
    write_json(reject_path, report)
    return {
        "seeds_path": str(seeds_path),
        "reject_path": str(reject_path),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "skipped_unvalidated": len(skipped_unvalidated),
        "unapproved_capture": unapproved_capture,
        "total_seeds": len(out_rows),
        "seed_written": seed_written,
        "protected_existing": protected_existing,
    }
