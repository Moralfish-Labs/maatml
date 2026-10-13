"""Append-only run registry (`output/runs.jsonl`) per model folder.

Each training run gets a unique ``run_id`` and checkpoint directory under
``output/checkpoints/<run_id>/``. The registry records status, device profile,
metrics, and optional eval-gate results.
"""

from __future__ import annotations

import json
import secrets
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, ValidationError

from .config import ModelDefinition
from .device import is_main_process
from .utils.io import write_json

RunStatus = Literal["running", "completed", "aborted"]

_RUNS_FILENAME = "runs.jsonl"


class RunRecord(BaseModel):
    """One training (or evaluated) run entry."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    identity: str
    architecture: str
    status: RunStatus
    started_at: str
    finished_at: Optional[str] = None
    smoke: bool = False
    device: Optional[str] = None
    profile: Optional[str] = None
    out_dir: str
    spec_hash: Optional[str] = None
    metrics: Optional[dict[str, float]] = None
    error: Optional[str] = None
    gates: Optional[dict[str, Any]] = None
    # True when the recorded gate pass came from the smoke tier (`smoke.gates`)
    # rather than the production thresholds, so the two never look alike.
    smoke_gated: Optional[bool] = None
    # Optional HPO / sweep trial metadata (rank-0 writes only).
    trial: Optional[dict[str, Any]] = None
    # Every time the test split was spent to confirm a val-chosen operating
    # point: {benchmark_sha256, split, threshold_key, threshold, report, at}.
    test_spends: Optional[list[dict[str, Any]]] = None
    # Evaluation config + checkpoint contents at the last production gate
    # pass; a blind evaluate requires the candidate to be unchanged since.
    gated_fingerprint: Optional[str] = None
    # Every blind evaluate: {fingerprint, blind_sha256, report, passed, at}.
    blind_spends: Optional[list[dict[str, Any]]] = None
    # training.select_by: every candidate's val score and the one chosen.
    selection: Optional[dict[str, Any]] = None
    # Subdirectory of out_dir the run id resolves to (None = the final weights).
    selected_checkpoint: Optional[str] = None
    # The machine this run trained on (maatml.environment/1); evaluate records
    # its own under gates.environment.
    environment: Optional[dict[str, Any]] = None
    # Set by `runs --adopt`: {bundle, spec_hash, forced}.
    adopted_from: Optional[dict[str, Any]] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(*, when: Optional[datetime] = None) -> str:
    """``YYYYMMDD-HHMMSS-<short>`` unique-enough run id."""
    ts = when or datetime.now(timezone.utc)
    short = secrets.token_hex(3)
    return f"{ts.strftime('%Y%m%d-%H%M%S')}-{short}"


def runs_path(model_def: ModelDefinition) -> Path:
    return model_def.output_dir / _RUNS_FILENAME


def _append_record(model_def: ModelDefinition, record: RunRecord) -> None:
    """Append to ``runs.jsonl`` on the main process only (multi-GPU safe)."""
    if not is_main_process():
        return
    path = runs_path(model_def)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build the full line first and write once, so a crash or a concurrent
    # append cannot leave a torn record (body with no newline).
    line = json.dumps(record.model_dump(mode="json"), sort_keys=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _quarantine_corrupt(path: Path, lines: list[str]) -> None:
    """Append never-before-seen unparseable lines to a sidecar corrupt file.

    Dedup-guarded because ``list_runs`` runs on every read: it must not append
    the same bad line repeatedly, and it must never rewrite ``runs.jsonl``.
    """
    corrupt_path = path.with_name(path.name + ".corrupt")
    try:
        seen: set[str] = set()
        if corrupt_path.is_file():
            with open(corrupt_path, "r", encoding="utf-8") as f:
                seen = {ln.strip() for ln in f}
        new = [ln for ln in lines if ln not in seen]
        if not new:
            return
        with open(corrupt_path, "a", encoding="utf-8") as f:
            for ln in new:
                f.write(ln + "\n")
    except OSError:
        pass


def list_runs(model_def: ModelDefinition) -> list[RunRecord]:
    """Return all run records (latest entry per ``run_id`` wins).

    A line that cannot be parsed (a torn record from a crash mid-write, or
    manual corruption) is skipped with a warning and recorded in
    ``runs.jsonl.corrupt`` rather than raising and bricking every consumer.
    """
    path = runs_path(model_def)
    if not path.is_file():
        return []
    latest: dict[str, RunRecord] = {}
    order: list[str] = []
    corrupt: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = RunRecord.model_validate_json(line)
            except (ValidationError, ValueError):
                corrupt.append(line)
                warnings.warn(
                    f"skipping unparseable run record at {path}:{lineno}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if rec.run_id not in latest:
                order.append(rec.run_id)
            latest[rec.run_id] = rec
    if corrupt:
        _quarantine_corrupt(path, corrupt)
    return [latest[rid] for rid in order]


# Trainer telemetry: real metrics, but they measure the machine rather than the
# model, and there are enough of them to bury the ones being compared.
_TELEMETRY_SUFFIXES = ("_runtime", "_samples_per_second", "_steps_per_second")
_TELEMETRY_KEYS = frozenset({"epoch", "total_flos", "train_loss"})


def is_telemetry(key: str) -> bool:
    return key in _TELEMETRY_KEYS or key.endswith(_TELEMETRY_SUFFIXES)


def compare_runs(
    records: list[RunRecord],
    *,
    metrics: Optional[list[str]] = None,
    include_telemetry: bool = False,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Build a run-by-metric comparison table.

    Returns ``(metric_keys, rows, hidden_keys)``. Every row carries the run
    identity plus one entry per metric key (``None`` when that run never
    reported it, so a missing metric reads as missing rather than as zero).
    Explicitly requested ``metrics`` are never hidden; otherwise trainer
    telemetry is set aside and returned in ``hidden_keys`` so the caller can
    say what it left out.
    """
    keys: list[str] = []
    hidden: list[str] = []
    if metrics:
        keys = list(dict.fromkeys(metrics))
    else:
        for rec in records:
            for key in rec.metrics or {}:
                if key in keys or key in hidden:
                    continue
                if not include_telemetry and is_telemetry(key):
                    hidden.append(key)
                else:
                    keys.append(key)

    rows: list[dict[str, Any]] = []
    for rec in records:
        values = rec.metrics or {}
        gates = rec.gates or {}
        rows.append(
            {
                "run_id": rec.run_id,
                "status": rec.status,
                "smoke": rec.smoke,
                "device": rec.device,
                "gates_passed": gates.get("passed"),
                "metrics": {key: values.get(key) for key in keys},
            }
        )
    return keys, rows, hidden


def get_run(model_def: ModelDefinition, run_id: str) -> Optional[RunRecord]:
    for rec in list_runs(model_def):
        if rec.run_id == run_id:
            return rec
    return adopt_run(model_def, run_id)


def adopt_run(model_def: ModelDefinition, run_id: str) -> Optional[RunRecord]:
    """Rebuild a lost registry line from what the run left beside its weights.

    A run trained on one machine and exported on another arrives as a
    checkpoint directory and nothing else: ``runs.jsonl`` is written where the
    training happened, and does not travel with the weights. The registry is
    the only thing missing, and everything it holds was already written twice -
    ``run_metadata.json`` carries the identity, architecture and spec hash, and
    an eval report carries the metrics and the gate result. Without this, a
    checkpoint that trained and passed its gates exports a manifest saying
    ``gated: false``, which reads as a run that was never gated rather than as
    a record that did not survive the trip.

    Returns ``None`` when the checkpoint or its metadata is absent, so a
    genuinely unknown run id stays unknown.
    """
    out_dir = model_def.checkpoints_dir / run_id
    meta_path = out_dir / "run_metadata.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    extra = meta.get("extra") or {}
    created = meta.get("created_at") or _utc_now()
    record = RunRecord(
        run_id=extra.get("run_id") or run_id,
        identity=meta.get("identity") or model_def.identity,
        architecture=meta.get("architecture") or model_def.architecture,
        status="completed",
        started_at=created,
        finished_at=created,
        smoke=bool(extra.get("smoke")),
        device=extra.get("device"),
        profile=extra.get("profile"),
        out_dir=str(out_dir),
        spec_hash=meta.get("spec_hash"),
    )

    report_path = model_def.eval_dir / f"{run_id}.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = {}
        gates = report.get("gates")
        if isinstance(gates, dict):
            record.gates = gates
            record.smoke_gated = bool(gates.get("smoke"))
        metrics = report.get("metrics")
        if isinstance(metrics, dict):
            record.metrics = {
                k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
            }

    # Appended rather than returned alone: the next reader gets it from the
    # registry like any other run, and `maatml runs` stops hiding a run whose
    # weights are on disk.
    _append_record(model_def, record)
    return record


def spec_fingerprint(model_def: ModelDefinition) -> str:
    """sha256 of the effective model definition: the model-folder fingerprint
    a run is produced under (the same hash ``run_metadata.json`` records)."""
    import hashlib

    spec = model_def.model_dump(mode="json", exclude={"model_dir"})
    body = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def start_run(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    device: Optional[str] = None,
    profile: Optional[str] = None,
    spec_hash: Optional[str] = None,
    run_id: Optional[str] = None,
    out_dir: Optional[Path] = None,
    trial: Optional[dict[str, Any]] = None,
) -> RunRecord:
    """Create a new ``running`` run and append it to ``runs.jsonl``."""
    from .environment import environment_manifest

    rid = run_id or make_run_id()
    ckpt = Path(out_dir) if out_dir else (model_def.checkpoints_dir / rid)
    ckpt.mkdir(parents=True, exist_ok=True)
    record = RunRecord(
        run_id=rid,
        identity=model_def.identity,
        architecture=model_def.architecture,
        status="running",
        started_at=_utc_now(),
        smoke=smoke,
        device=device,
        profile=profile,
        out_dir=str(ckpt.resolve()),
        spec_hash=spec_hash or spec_fingerprint(model_def),
        trial=trial,
        environment=environment_manifest(model_def.model_dir),
    )
    _append_record(model_def, record)
    return record


def finish_run(
    model_def: ModelDefinition,
    run_id: str,
    status: RunStatus,
    *,
    metrics: Optional[dict[str, float]] = None,
    error: Optional[str] = None,
    gates: Optional[dict[str, Any]] = None,
) -> Optional[RunRecord]:
    """Mark a run completed/aborted by appending an updated record.

    On non-main ranks (multi-GPU), returns ``None`` without writing.
    """
    if status == "running":
        raise ValueError("finish_run status must be 'completed' or 'aborted'")
    if not is_main_process():
        return None
    rec = get_run(model_def, run_id)
    if rec is None:
        raise KeyError(f"Unknown run_id {run_id!r} for {model_def.identity}")
    payload = rec.model_dump()
    payload["status"] = status
    payload["finished_at"] = _utc_now()
    if metrics is not None:
        payload["metrics"] = metrics
    if error is not None:
        payload["error"] = error
    if gates is not None:
        payload["gates"] = gates
    updated = RunRecord(**payload)
    _append_record(model_def, updated)
    return updated


def update_run_gates(
    model_def: ModelDefinition,
    run_id: str,
    gates: dict[str, Any],
    *,
    metrics: Optional[dict[str, float]] = None,
    smoke_gated: bool = False,
    gated_fingerprint: Optional[str] = None,
) -> Optional[RunRecord]:
    """Attach eval-gate results to a known run (no-op if run_id unknown)."""
    rec = get_run(model_def, run_id)
    if rec is None:
        return None
    payload = rec.model_dump()
    payload["gates"] = gates
    payload["smoke_gated"] = bool(smoke_gated)
    if gated_fingerprint is not None and not smoke_gated:
        payload["gated_fingerprint"] = gated_fingerprint
    if metrics is not None:
        payload["metrics"] = {**(payload.get("metrics") or {}), **metrics}
    updated = RunRecord(**payload)
    _append_record(model_def, updated)
    return updated


def record_selection(
    model_def: ModelDefinition,
    run_id: str,
    selection: dict[str, Any],
    *,
    selected_checkpoint: Optional[str],
) -> Optional[RunRecord]:
    """Record a ``training.select_by`` outcome on a run (no-op if unknown)."""
    rec = get_run(model_def, run_id)
    if rec is None:
        return None
    payload = rec.model_dump()
    payload["selection"] = selection
    payload["selected_checkpoint"] = selected_checkpoint
    updated = RunRecord(**payload)
    _append_record(model_def, updated)
    return updated


def run_id_for_checkpoint(model_def: ModelDefinition, path: Path) -> str:
    """The run a checkpoint path belongs to: its own name, or its parent run's.

    A selected ``checkpoint-<step>`` lives inside the run directory; evidence
    is recorded against the run, never against the subdirectory.
    """
    path = Path(path)
    if get_run(model_def, path.name) is not None:
        return path.name
    parent = get_run(model_def, path.parent.name)
    if parent is not None and parent.selected_checkpoint == path.name:
        return parent.run_id
    return path.name


def _selected(rec: RunRecord, out: Path) -> Path:
    if rec.selected_checkpoint:
        chosen = out / rec.selected_checkpoint
        if chosen.is_dir():
            return chosen
    return out


def record_test_spend(
    model_def: ModelDefinition, run_id: str, spend: dict[str, Any]
) -> tuple[Optional[RunRecord], int]:
    """Append a test spend to a run; returns the record and how many spends the
    same benchmark already had (so a caller can say the test was spent twice)."""
    rec = get_run(model_def, run_id)
    if rec is None:
        return None, 0
    spends = list(rec.test_spends or [])
    prior = sum(1 for s in spends if s.get("benchmark_sha256") == spend.get("benchmark_sha256"))
    spends.append({**spend, "at": _utc_now()})
    payload = rec.model_dump()
    payload["test_spends"] = spends
    updated = RunRecord(**payload)
    _append_record(model_def, updated)
    return updated, prior


def evidence_fingerprint(model_def: ModelDefinition, checkpoint: Path) -> str:
    """What a gate pass was measured on: the evaluation config and the weights.

    A blind evaluate is spent against exactly this; change a threshold or the
    checkpoint and the candidate must be re-gated first.
    """
    from .lifecycle import _dir_signature
    from .utils.io import stable_hash

    evaluation = model_def.evaluation or {}
    return stable_hash(
        sorted(evaluation.items(), key=lambda kv: kv[0]),
        model_def.packaging.model_dump(),
        _dir_signature(Path(checkpoint)),
    )


def record_blind_spend(
    model_def: ModelDefinition, run_id: str, spend: dict[str, Any]
) -> tuple[Optional[RunRecord], int]:
    """Append a blind spend; returns the record and how many prior spends share
    the same fingerprint and blind population."""
    rec = get_run(model_def, run_id)
    if rec is None:
        return None, 0
    spends = list(rec.blind_spends or [])
    prior = sum(
        1
        for s in spends
        if s.get("fingerprint") == spend.get("fingerprint")
        and s.get("blind_sha256") == spend.get("blind_sha256")
    )
    spends.append({**spend, "at": _utc_now()})
    payload = rec.model_dump()
    payload["blind_spends"] = spends
    updated = RunRecord(**payload)
    _append_record(model_def, updated)
    return updated, prior


def latest_completed_run(model_def: ModelDefinition) -> Optional[RunRecord]:
    """Most recently finished completed run (by ``finished_at``, then order)."""
    completed = [r for r in list_runs(model_def) if r.status == "completed"]
    if not completed:
        return None

    def _key(r: RunRecord) -> str:
        return r.finished_at or r.started_at

    return max(completed, key=_key)


def _has_hf_checkpoints(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(p.is_dir() and p.name.startswith("checkpoint-") for p in path.iterdir())


def latest_incomplete_run(model_def: ModelDefinition) -> Optional[RunRecord]:
    """Latest resumable run: ``running`` with checkpoints, else any with them.

    A ``running`` record with no ``checkpoint-*`` has nothing to resume from
    (the run died before its first save, or the process was killed). Such a
    record used to win anyway and make ``--resume auto`` fail, hiding an older
    run that could actually be resumed.
    """
    runs = list_runs(model_def)
    resumable = [r for r in runs if _has_hf_checkpoints(Path(r.out_dir))]
    running = [r for r in resumable if r.status == "running"]
    if running:
        return running[-1]
    return resumable[-1] if resumable else None


def _last_trainer_checkpoint(root: Path) -> Optional[Path]:
    """Newest ``checkpoint-*`` dir under a run root (HF Trainer resume target).

    transformers.Trainer.train only auto-discovers the newest checkpoint when
    ``resume_from_checkpoint`` is the bool ``True``; a string is treated as the
    exact checkpoint dir. A run root holds only ``checkpoint-*`` subdirs, so we
    must descend to the newest one ourselves.
    """
    from transformers.trainer_utils import get_last_checkpoint

    found = get_last_checkpoint(str(root))
    return Path(found) if found else None


def resolve_resume_checkpoint(
    model_def: ModelDefinition,
    resume: Optional[str],
) -> Optional[Path]:
    """Resolve ``--resume auto|PATH`` to a checkpoint path for ``trainer.train``.

    ``None`` / empty → fresh run (no resume). ``auto`` and a run_id both resolve
    to the newest ``checkpoint-*`` directory under that run's out_dir. An
    explicit path is used as-is.
    """
    if resume is None or resume == "":
        return None
    if resume == "auto":
        rec = latest_incomplete_run(model_def)
        if rec is None:
            raise FileNotFoundError(f"No incomplete run to resume under {model_def.output_dir}")
        root = Path(rec.out_dir)
        ckpt = _last_trainer_checkpoint(root)
        if ckpt is None:
            raise FileNotFoundError(
                f"Run {rec.run_id!r} at {root} has no checkpoint-* to resume from"
            )
        return ckpt
    path = Path(resume)
    if not path.is_absolute():
        # Allow run_id or path relative to model dir / checkpoints.
        as_run = get_run(model_def, resume)
        if as_run is not None:
            root = Path(as_run.out_dir)
            ckpt = _last_trainer_checkpoint(root)
            if ckpt is None:
                raise FileNotFoundError(
                    f"Run {resume!r} at {root} has no checkpoint-* to resume from"
                )
            return ckpt
        cand = model_def.checkpoints_dir / resume
        if cand.exists():
            return cand
        cand = model_def.model_dir / resume
        if cand.exists():
            return cand
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
    return path.resolve()


def resolve_checkpoint(
    model_def: ModelDefinition,
    checkpoint: str | Path | None = None,
) -> Path:
    """Resolve an eval/train checkpoint: run_id, path, or latest completed.

    Falls back to the newest directory under ``output/checkpoints/`` when the
    registry is empty (legacy name@version / smoke dirs).
    """
    if checkpoint is not None and str(checkpoint).strip():
        raw = str(checkpoint).strip()
        path = Path(raw)
        if path.exists():
            return path.resolve()
        # Model-dir-relative path (e.g. output/export/<run_id>).
        cand = model_def.model_dir / raw
        if cand.exists():
            return cand.resolve()
        rec = get_run(model_def, raw)
        if rec is not None:
            out = Path(rec.out_dir)
            if out.exists():
                return _selected(rec, out).resolve()
            # A run trained elsewhere records that machine's absolute path. The
            # weights travel home; the path does not. `output/checkpoints/
            # <run_id>` is where they land, so looking there lets a relocated
            # run export under its own id - and a bundle exported from an
            # anonymous directory carries no gate evidence, because the
            # manifest looks the run up by exactly this id.
            moved = model_def.checkpoints_dir / raw
            if moved.exists():
                return _selected(rec, moved).resolve()
            return out
        cand = model_def.checkpoints_dir / raw
        if cand.exists():
            return cand.resolve()
        raise FileNotFoundError(
            f"Checkpoint {raw!r} not found as path or run_id under {model_def.checkpoints_dir}"
        )

    completed = latest_completed_run(model_def)
    if completed is not None:
        out = Path(completed.out_dir)
        if out.exists():
            return _selected(completed, out)

    # Legacy fallback: newest mtime under checkpoints/
    ckpt_root = model_def.checkpoints_dir
    if not ckpt_root.exists():
        raise FileNotFoundError(
            f"No checkpoints under {ckpt_root}. Run `maatml train {model_def.model_dir}` first."
        )
    candidates = [p for p in ckpt_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint directories in {ckpt_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def write_run_sidecar(out_dir: Path, record: RunRecord) -> Path:
    """Optional small sidecar next to weights (debug); not the registry."""
    return write_json(Path(out_dir) / "run_record.json", record.model_dump(mode="json"))


def normalize_report_to(raw: Any) -> list[str] | str:
    """Normalize ``training.report_to`` for HuggingFace TrainingArguments."""
    if raw is None or raw == "none" or raw == []:
        return []
    if isinstance(raw, str):
        return [] if raw.lower() == "none" else [raw]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw if str(x).lower() != "none"]
    return []


def begin_training_run(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    device: Optional[str] = None,
    profile: Optional[str] = None,
    out_dir: Optional[Path] = None,
    resume: Optional[str] = None,
    trial: Optional[dict[str, Any]] = None,
) -> tuple[RunRecord, Path, Optional[Path]]:
    """Start (or resume) a training run.

    Returns ``(run, out_dir, resume_checkpoint_path)``.
    """
    resume_path = resolve_resume_checkpoint(model_def, resume) if resume else None

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        run = start_run(
            model_def,
            smoke=smoke,
            device=device,
            profile=profile,
            out_dir=out,
            trial=trial,
        )
        return run, out, resume_path

    if resume_path is not None:
        resume_root = resume_path
        if resume_root.name.startswith("checkpoint-"):
            resume_root = resume_root.parent
        existing = get_run(model_def, resume_root.name)
        if existing is None:
            for rec in list_runs(model_def):
                if Path(rec.out_dir).resolve() == resume_root.resolve():
                    existing = rec
                    break
        if existing is not None:
            out = Path(existing.out_dir)
            run = start_run(
                model_def,
                smoke=smoke,
                device=device,
                profile=profile,
                run_id=existing.run_id,
                out_dir=out,
                trial=trial,
            )
            return run, out, resume_path
        run = start_run(
            model_def,
            smoke=smoke,
            device=device,
            profile=profile,
            out_dir=resume_root,
            trial=trial,
        )
        return run, Path(run.out_dir), resume_path

    run = start_run(
        model_def,
        smoke=smoke,
        device=device,
        profile=profile,
        trial=trial,
    )
    return run, Path(run.out_dir), None
