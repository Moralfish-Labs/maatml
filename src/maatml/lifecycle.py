"""The fixed lifecycle runner behind ``maatml run``.

One command walks the pipeline in order:

    prepare -> train -> evaluate (gates enforced) -> export -> verify

and stops non-zero at the first failure, so a green line means every stage
passed rather than "the last command I happened to run passed".

Source-mutating operations (``datagen`` / ``ingest``) stay outside the runner:
they change the seed corpus, which makes ``prepare`` stale, and that staleness
is exactly what the runner is here to notice.

**Fingerprints are for idempotence, not speed.** Every step records what it
consumed (effective config, input files, upstream fingerprints, maatml version
and git SHA, plugin sources, device profile, exporter identity) in
``output/pipeline.json``. A step is skipped only when its fingerprint matches,
the prior step completed, and its declared outputs still exist. Anything else
re-runs, and the plan says which component changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from .config import ModelDefinition, get_dataset_cfg
from .utils.io import read_json, sha256_file, stable_hash, write_json_atomic

STEPS: tuple[str, ...] = ("prepare", "train", "evaluate", "export", "verify")

_STATE_FILENAME = "pipeline.json"
_STATE_VERSION = 1


class StepError(RuntimeError):
    """A lifecycle step failed; the runner stops here."""


@dataclass
class StepPlan:
    """What the runner intends to do with one step."""

    name: str
    fingerprint: str
    components: dict[str, str]
    fresh: bool
    reason: str
    selected: bool = True

    @property
    def action(self) -> str:
        if not self.selected:
            return "not selected"
        return "skip" if self.fresh else "run"


@dataclass
class StepOutcome:
    name: str
    status: str  # ran | skipped | failed | not selected
    detail: str = ""


@dataclass
class PipelineResult:
    outcomes: list[StepOutcome] = field(default_factory=list)
    failed: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.failed is None


# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------


def state_path(model_def: ModelDefinition) -> Path:
    return model_def.output_dir / _STATE_FILENAME


def load_state(model_def: ModelDefinition) -> dict[str, Any]:
    """Read ``output/pipeline.json``; an unreadable file is treated as empty.

    A corrupt state file must not brick the runner: the worst case of ignoring
    it is that every step re-runs, which is the safe direction.
    """
    path = state_path(model_def)
    if not path.is_file():
        return {"version": _STATE_VERSION, "steps": {}}
    try:
        data = read_json(path)
    except (ValueError, OSError):
        return {"version": _STATE_VERSION, "steps": {}}
    if not isinstance(data, dict) or not isinstance(data.get("steps"), dict):
        return {"version": _STATE_VERSION, "steps": {}}
    return data


def save_state(model_def: ModelDefinition, state: dict[str, Any]) -> Path:
    from .training.guards import _pkg_version

    state["version"] = _STATE_VERSION
    state["maatml_version"] = _pkg_version("maatml")
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    return write_json_atomic(state_path(model_def), state)


# ---------------------------------------------------------------------------
# Fingerprint components
# ---------------------------------------------------------------------------


def _file_hash(path: Optional[Path]) -> str:
    if path is None:
        return "absent"
    path = Path(path)
    if not path.is_file():
        return "absent"
    return sha256_file(path)


def _dir_signature(path: Optional[Path]) -> str:
    """Cheap change detector for a directory of large artifacts.

    Checkpoints and export bundles hold multi-gigabyte weights, so this hashes
    (relative name, size, mtime) rather than contents. It can only cause extra
    work, never a wrong skip in the dangerous direction: ``verify`` still
    recomputes real sha256 sums from the manifest.
    """
    if path is None:
        return "absent"
    path = Path(path)
    if not path.is_dir():
        return "absent"
    entries = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            stat = item.stat()
            entries.append((str(item.relative_to(path)), stat.st_size, stat.st_mtime_ns))
    if not entries:
        return "empty"
    return stable_hash(entries)


def plugin_sources_hash(model_def: ModelDefinition) -> str:
    """Hash the Python sources of the model's declared plugins.

    A model folder's plugins are executable code that decides validators,
    metrics, and trainers, so editing them changes what every step means.
    """
    parts: list[tuple[str, str]] = []
    for entry in model_def.plugins or []:
        from .registry import looks_like_plugin_path

        if not looks_like_plugin_path(entry, model_def.model_dir):
            parts.append((entry, "module"))
            continue
        path = model_def.resolve(entry)
        if path.is_file():
            parts.append((entry, _file_hash(path)))
        elif path.is_dir():
            files = sorted(p for p in path.rglob("*.py") if p.is_file())
            parts.append(
                (
                    entry,
                    stable_hash([(str(p.relative_to(path)), sha256_file(p)) for p in files]),
                )
            )
        else:
            parts.append((entry, "missing"))
    return stable_hash(parts)


def _declared_assets_hash(model_def: ModelDefinition) -> str:
    """Hash every path-like asset the config declares (schema, seeds, ...)."""
    from .config import _PATH_KEYS

    parts: list[tuple[str, str]] = []
    for section_name in ("dataset", "data", "evaluation"):
        section = getattr(model_def, section_name, None)
        if not isinstance(section, dict):
            continue
        for key, value in sorted(section.items()):
            if key not in _PATH_KEYS or not isinstance(value, str):
                continue
            parts.append((f"{section_name}.{key}", _file_hash(model_def.resolve(value))))
    return stable_hash(parts)


def _prepared_hash(model_def: ModelDefinition, splits: tuple[str, ...]) -> str:
    return stable_hash(
        [(split, _file_hash(model_def.prepared_dir / f"{split}.jsonl")) for split in splits]
    )


def _environment_component() -> str:
    from .training.guards import _pkg_version, maatml_checkout_sha

    return stable_hash(
        _pkg_version("maatml"),
        # A checkout's SHA distinguishes "same version string, different code".
        # None for an installed package, so an unrelated repository's commits
        # cannot invalidate every step.
        maatml_checkout_sha(),
    )


def _effective_training(model_def: ModelDefinition, *, smoke: bool) -> dict[str, Any]:
    return model_def.merged_smoke() if smoke else dict(model_def.training)


def compute_components(
    model_def: ModelDefinition,
    *,
    smoke: bool,
    device: str,
    checkpoint: Optional[Path] = None,
    export_dir: Optional[Path] = None,
    export_format: Optional[str] = None,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict[str, dict[str, str]]:
    """Fingerprint components per step, keyed by component name.

    Named components (rather than one opaque hash) are what lets ``--dry-run``
    say *why* a step is stale.

    ``limit`` and ``seed`` belong to the train fingerprint because both change
    the checkpoint: a run truncated with ``--limit`` trained on a fraction of
    the corpus, and leaving them out let the next plain ``maatml run`` report
    "all fresh" over it, with evaluate/export/verify inheriting the skip.
    """
    from .evaluation.harness import effective_gates
    from .scaffold import normalize_architecture

    env = _environment_component()
    plugins = plugin_sources_hash(model_def)
    ds_cfg = get_dataset_cfg(model_def)
    identity = stable_hash(model_def.identity, normalize_architecture(model_def.architecture))

    prepare = {
        "environment": env,
        "plugins": plugins,
        "identity": identity,
        "dataset_config": stable_hash(sorted(ds_cfg.items(), key=lambda kv: kv[0])),
        "input_assets": _declared_assets_hash(model_def),
    }
    prepare_fp = stable_hash(sorted(prepare.items()))

    train = {
        "upstream": prepare_fp,
        "environment": env,
        "plugins": plugins,
        "identity": identity,
        "training_config": stable_hash(
            sorted(_effective_training(model_def, smoke=smoke).items(), key=lambda kv: kv[0])
        ),
        "smoke": str(bool(smoke)),
        "device": str(device),
        "limit": "none" if limit is None else str(int(limit)),
        "seed": "none" if seed is None else str(int(seed)),
        "prepared_data": _prepared_hash(model_def, ("train", "val")),
    }
    train_fp = stable_hash(sorted(train.items()))

    evaluation = {
        "upstream": train_fp,
        "environment": env,
        "plugins": plugins,
        "evaluation_config": stable_hash(
            sorted((model_def.evaluation or {}).items(), key=lambda kv: kv[0])
        ),
        "gates": stable_hash(sorted(effective_gates(model_def, smoke=smoke).items())),
        "packaging": stable_hash(model_def.packaging.model_dump()),
        "test_data": _prepared_hash(model_def, ("test",)),
        "checkpoint": _dir_signature(checkpoint),
        "device": str(device),
    }
    evaluation_fp = stable_hash(sorted(evaluation.items()))

    export = {
        "upstream": evaluation_fp,
        "environment": env,
        "plugins": plugins,
        "packaging": stable_hash(model_def.packaging.model_dump()),
        "checkpoint": _dir_signature(checkpoint),
        "exporter": str(export_format or "safetensors"),
    }
    export_fp = stable_hash(sorted(export.items()))

    verify = {
        "upstream": export_fp,
        "environment": env,
        "export_bundle": _dir_signature(export_dir),
    }

    return {
        "prepare": prepare,
        "train": train,
        "evaluate": evaluation,
        "export": export,
        "verify": verify,
    }


def fingerprint(components: dict[str, str]) -> str:
    return stable_hash(sorted(components.items()))


# ---------------------------------------------------------------------------
# Output checks
# ---------------------------------------------------------------------------


def _has_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = {"model.safetensors", "adapter_model.safetensors", "classifier_heads.safetensors"}
    if any((path / name).is_file() for name in names):
        return True
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def outputs_present(
    model_def: ModelDefinition,
    step: str,
    *,
    checkpoint: Optional[Path],
    export_dir: Optional[Path],
    eval_report: Optional[Path],
) -> bool:
    """Do this step's declared outputs still exist?

    A matching fingerprint is not enough to skip: someone may have deleted
    ``output/`` since, and skipping then would report success over nothing.
    """
    if step == "prepare":
        return all(
            (model_def.prepared_dir / f"{split}.jsonl").is_file()
            for split in ("train", "val", "test")
        )
    if step == "train":
        return checkpoint is not None and _has_weights(checkpoint)
    if step == "evaluate":
        return eval_report is not None and eval_report.is_file()
    if step in ("export", "verify"):
        return export_dir is not None and (export_dir / "manifest.json").is_file()
    return True


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _selected_steps(from_step: Optional[str], until_step: Optional[str]) -> tuple[str, ...]:
    start = 0
    end = len(STEPS) - 1
    if from_step:
        if from_step not in STEPS:
            raise ValueError(f"--from must be one of {', '.join(STEPS)}; got {from_step!r}")
        start = STEPS.index(from_step)
    if until_step:
        if until_step not in STEPS:
            raise ValueError(f"--until must be one of {', '.join(STEPS)}; got {until_step!r}")
        end = STEPS.index(until_step)
    if start > end:
        raise ValueError(f"--from {from_step!r} comes after --until {until_step!r}; nothing to run")
    return STEPS[start : end + 1]


def _stale_reason(stored: Optional[dict[str, Any]], components: dict[str, str]) -> Optional[str]:
    """None when the step is fresh, else why it is not."""
    if not stored:
        return "never run"
    if stored.get("status") != "completed":
        return f"last run {stored.get('status', 'unknown')}"
    stored_components = stored.get("components") or {}
    changed = [
        name for name, value in sorted(components.items()) if stored_components.get(name) != value
    ]
    if changed:
        return "changed: " + ", ".join(changed)
    if stored.get("fingerprint") != fingerprint(components):
        return "fingerprint mismatch"
    return None


def plan_pipeline(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    device: str = "auto",
    force: bool = False,
    from_step: Optional[str] = None,
    until_step: Optional[str] = None,
    checkpoint: Optional[Path] = None,
    export_dir: Optional[Path] = None,
    export_format: Optional[str] = None,
    eval_report: Optional[Path] = None,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> list[StepPlan]:
    """Per-step fresh/stale decision, with the reason a step is stale."""
    selected = _selected_steps(from_step, until_step)
    state = load_state(model_def)
    components = compute_components(
        model_def,
        smoke=smoke,
        device=device,
        checkpoint=checkpoint,
        export_dir=export_dir,
        export_format=export_format,
        limit=limit,
        seed=seed,
    )

    plans: list[StepPlan] = []
    upstream_ran = False
    for step in STEPS:
        step_components = components[step]
        stored = (state.get("steps") or {}).get(step)
        reason = _stale_reason(stored, step_components)
        fresh = reason is None
        if fresh and not outputs_present(
            model_def,
            step,
            checkpoint=checkpoint,
            export_dir=export_dir,
            eval_report=eval_report,
        ):
            fresh, reason = False, "outputs missing"
        if fresh and force:
            fresh, reason = False, "--force"
        # A step whose upstream re-ran cannot be fresh: its inputs are about to
        # change, and the stored fingerprint describes the old ones.
        if fresh and upstream_ran:
            fresh, reason = False, "upstream step re-ran"
        if step in selected and not fresh:
            upstream_ran = True
        plans.append(
            StepPlan(
                name=step,
                fingerprint=fingerprint(step_components),
                components=step_components,
                fresh=fresh,
                reason=reason or "up to date",
                selected=step in selected,
            )
        )
    return plans


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


StepFn = Callable[[], str]


class EvaluationSpec(BaseModel):
    """The evaluation section, typed where the runner depends on it.

    Not a full config rewrite: this is the slice the runner fingerprints and
    gates on, so a typo here would otherwise be discovered only after training
    finished. Unknown keys are rejected for the same reason ``model.yml``'s
    top level rejects them.

    Rejecting unknown keys means carrying every *known* one. The fields below
    the divider are not read here - they are declared so that a section
    ``config.py`` accepts, ``validate`` passes and ``evaluate`` honours is not
    refused by ``run`` alone. A folder that configures slices or a prediction
    cache could otherwise run every stage individually and nothing end to end,
    which is the opposite of the check this class exists to make.
    """

    model_config = ConfigDict(extra="forbid")

    predictor: Optional[str] = None
    validator: Optional[str] = None
    metrics: Optional[Union[str, list[str]]] = None
    gates: dict[str, float] = Field(default_factory=dict)
    repair_braces: bool = False

    # Known to config.py, not depended on by the runner. Keep in step with
    # ``config._EVALUATION_KNOWN_KEYS``.
    gates_benchmark: Optional[str] = None
    operating_point: Optional[dict[str, Any]] = None
    slices: Optional[list[Any]] = None
    cache_predictions: bool = False
    batch_size: Optional[int] = None


def validate_run_config(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    steps: tuple[str, ...] = STEPS,
) -> None:
    """Check everything the run depends on before any step executes.

    A misspelled validator, an unregistered metrics plugin, or a gate value
    that is not a number used to surface only when evaluate ran, which is
    after training had already spent the compute. Gates are only required when
    the evaluate step is actually selected.
    """
    from .evaluation.harness import (
        GateConfigError,
        _resolve_metrics,
        effective_gates,
        resolve_validator,
    )

    try:
        spec = EvaluationSpec(**(model_def.evaluation or {}))
    except PydanticValidationError as exc:
        raise GateConfigError(f"evaluation: section is invalid: {exc.errors()[0]['msg']}") from exc

    if spec.validator is not None:
        resolve_validator(spec.validator)
    _resolve_metrics(spec.metrics)
    if "evaluate" not in steps:
        return
    gates = effective_gates(model_def, smoke=smoke)
    if not gates:
        raise GateConfigError(
            "maatml run enforces evaluation.gates, and none are configured. "
            "Add a gates: block to model.yml"
            + (" (or a smoke.gates tier for --smoke runs)." if smoke else ".")
        )


@dataclass
class RunOptions:
    """Everything a lifecycle run needs beyond the model definition."""

    smoke: bool = False
    device: str = "auto"
    force: bool = False
    from_step: Optional[str] = None
    until_step: Optional[str] = None
    seed: Optional[int] = None
    limit: Optional[int] = None
    export_format: Optional[str] = None


def _resolve_paths(
    model_def: ModelDefinition,
) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Current checkpoint, export dir, and eval report, when they exist."""
    from .runs import resolve_checkpoint, run_id_for_checkpoint

    try:
        checkpoint = resolve_checkpoint(model_def)
    except FileNotFoundError:
        return None, None, None

    run_id = run_id_for_checkpoint(model_def, checkpoint)
    export_dir = model_def.output_dir / "export" / run_id
    report = model_def.eval_dir / f"{run_id}.json"
    return checkpoint, export_dir, report


def _step_prepare(model_def: ModelDefinition, options: RunOptions) -> str:
    from .registry import FORMATS

    fmt = str(get_dataset_cfg(model_def).get("format", "jsonl_seed"))
    summary = FORMATS.require(fmt)(model_def)
    counts = (summary or {}).get("split_counts", {})
    return f"format={fmt} splits={counts}"


def _step_train(model_def: ModelDefinition, options: RunOptions) -> str:
    from .registry import TRAINERS
    from .scaffold import normalize_architecture

    arch = normalize_architecture(model_def.architecture)
    trainer = TRAINERS.get(model_def.architecture) or TRAINERS.require(arch)
    result = trainer(
        model_def,
        smoke=options.smoke,
        limit=options.limit,
        device=options.device,
        seed=options.seed,
    )
    from .training.selection import select_checkpoint

    selection = select_checkpoint(
        model_def, Path(result.out_dir).name, device=options.device, smoke=options.smoke
    )
    chosen = f" selected={selection['selected']}" if selection else ""
    return f"out_dir={Path(result.out_dir).name} metrics={result.metrics}{chosen}"


def _step_evaluate(model_def: ModelDefinition, options: RunOptions) -> str:
    from .evaluation.runner import evaluate_model

    report, out_path = evaluate_model(
        model_def,
        device=options.device,
        gate=True,
        smoke=options.smoke,
    )
    tier = "smoke-tier" if (report.gates or {}).get("smoke") else "gates"
    if report.passed is False:
        failed = [
            name
            for name, info in ((report.gates or {}).get("results") or {}).items()
            if not info.get("passed") and info.get("tier", "blocking") != "advisory"
        ]
        raise StepError(f"{tier} failed: {', '.join(sorted(failed))} (report {out_path})")
    return f"{tier} passed, report={out_path.name}"


def _step_export(model_def: ModelDefinition, options: RunOptions) -> str:
    from .export.bundle import export_model
    from .runs import resolve_checkpoint, run_id_for_checkpoint

    checkpoint = resolve_checkpoint(model_def)
    run_id = run_id_for_checkpoint(model_def, checkpoint)
    out_dir = model_def.output_dir / "export" / run_id
    export_model(model_def, checkpoint, out_dir, format=options.export_format, run_id=run_id)
    return f"export={out_dir.name}"


def _step_verify(model_def: ModelDefinition, options: RunOptions) -> str:
    from .export.manifest import verify_manifest

    _checkpoint, export_dir, _report = _resolve_paths(model_def)
    if export_dir is None or not (export_dir / "manifest.json").is_file():
        raise StepError("nothing to verify: no export bundle with a manifest")
    errors = verify_manifest(export_dir)
    if errors:
        raise StepError("manifest verification failed: " + "; ".join(errors[:3]))
    return f"verified={export_dir.name}"


_EXECUTORS: dict[str, Callable[[ModelDefinition, RunOptions], str]] = {
    "prepare": _step_prepare,
    "train": _step_train,
    "evaluate": _step_evaluate,
    "export": _step_export,
    "verify": _step_verify,
}


def run_pipeline(
    model_def: ModelDefinition,
    options: RunOptions,
    *,
    on_step: Optional[Callable[[str, str, str], None]] = None,
) -> PipelineResult:
    """Execute the lifecycle, skipping steps that are already fresh.

    Stops at the first failure: a later step must never run on the output of a
    step that did not pass. ``on_step(name, status, detail)`` reports progress.
    """
    validate_run_config(
        model_def,
        smoke=options.smoke,
        steps=_selected_steps(options.from_step, options.until_step),
    )
    result = PipelineResult()
    for step in STEPS:
        checkpoint, export_dir, report = _resolve_paths(model_def)
        # Re-planned each step: training produces the checkpoint the later
        # fingerprints are computed from.
        plans = {
            plan.name: plan
            for plan in plan_pipeline(
                model_def,
                smoke=options.smoke,
                device=options.device,
                force=options.force,
                from_step=options.from_step,
                until_step=options.until_step,
                checkpoint=checkpoint,
                export_dir=export_dir,
                export_format=options.export_format,
                eval_report=report,
                limit=options.limit,
                seed=options.seed,
            )
        }
        plan = plans[step]
        if not plan.selected:
            result.outcomes.append(StepOutcome(step, "not selected"))
            continue
        if plan.fresh:
            result.outcomes.append(StepOutcome(step, "skipped", plan.reason))
            if on_step:
                on_step(step, "skipped", plan.reason)
            continue

        if on_step:
            on_step(step, "running", plan.reason)
        try:
            detail = _EXECUTORS[step](model_def, options)
        except Exception as exc:  # noqa: BLE001  recorded, then re-raised as a result
            record_step(
                model_def,
                step,
                components=plan.components,
                status="failed",
                detail=f"{type(exc).__name__}: {exc}",
                smoke=options.smoke,
            )
            result.outcomes.append(StepOutcome(step, "failed", f"{type(exc).__name__}: {exc}"))
            result.failed = step
            if on_step:
                on_step(step, "failed", f"{type(exc).__name__}: {exc}")
            return result

        # Recompute components after the step: its own outputs (checkpoint,
        # export bundle) are part of what the next run compares against.
        checkpoint, export_dir, report = _resolve_paths(model_def)
        components = compute_components(
            model_def,
            smoke=options.smoke,
            device=options.device,
            checkpoint=checkpoint,
            export_dir=export_dir,
            export_format=options.export_format,
            limit=options.limit,
            seed=options.seed,
        )[step]
        record_step(
            model_def,
            step,
            components=components,
            status="completed",
            detail=detail,
            smoke=options.smoke,
        )
        result.outcomes.append(StepOutcome(step, "ran", detail))
        if on_step:
            on_step(step, "ran", detail)
    return result


def record_step(
    model_def: ModelDefinition,
    step: str,
    *,
    components: dict[str, str],
    status: str,
    detail: str = "",
    smoke: bool = False,
) -> None:
    """Persist one step's outcome atomically."""
    state = load_state(model_def)
    steps = state.setdefault("steps", {})
    steps[step] = {
        "status": status,
        "fingerprint": fingerprint(components),
        "components": components,
        "detail": detail,
        "smoke": bool(smoke),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    save_state(model_def, state)
