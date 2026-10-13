"""Read-only environment and model-folder diagnostics behind ``maatml audit``.

Answers "why did that not work here?" without running training: which optional
extras are installed, which device the CLI would pick, what the registries
hold (and what failed to load), and, for a model folder, whether its declared
paths, architecture, splits, and gates are actually in place.

Nothing here mutates state or imports a model folder's plugins beyond what
``load_model_def`` already does.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

# (import name, why it matters) for the optional stacks maatml dispatches to.
_OPTIONAL_PACKAGES: tuple[tuple[str, str], ...] = (
    ("torch", "training and inference ([ml])"),
    ("transformers", "trainers and tokenizers ([ml])"),
    ("peft", "LoRA / QLoRA adapters ([ml])"),
    ("datasets", "preference trainers ([ml])"),
    ("safetensors", "checkpoint and export weights ([ml])"),
    ("trl", "DPO / ORPO ([pref])"),
    ("bitsandbytes", "4/8-bit quantized bases ([cuda])"),
    ("torchvision", "vision examples ([vision])"),
    ("onnxruntime", "ONNX export checks ([vision])"),
    ("httpx", "teacher-backed datagen ([teacher])"),
    ("jsonschema", "validator schema checks (core dependency)"),
)

OK = "ok"
WARN = "warn"
ERROR = "error"


@dataclass
class Check:
    """One diagnostic line: ``name``, ``status``, and a human-readable detail."""

    name: str
    status: str
    detail: str


@dataclass
class Diagnostics:
    sections: dict[str, list[Check]] = field(default_factory=dict)

    def add(self, section: str, name: str, status: str, detail: str) -> None:
        self.sections.setdefault(section, []).append(Check(name, status, detail))

    @property
    def errors(self) -> list[Check]:
        return [c for checks in self.sections.values() for c in checks if c.status == ERROR]

    def as_dict(self) -> dict[str, Any]:
        return {
            section: [{"name": c.name, "status": c.status, "detail": c.detail} for c in checks]
            for section, checks in self.sections.items()
        }


def _version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _environment(diag: Diagnostics) -> None:
    diag.add(
        "environment",
        "maatml",
        OK,
        _version("maatml") or "not installed (running from a checkout?)",
    )
    diag.add("environment", "python", OK, f"{platform.python_version()} at {sys.executable}")
    diag.add(
        "environment",
        "platform",
        OK,
        f"{platform.system()} {platform.release()} ({platform.machine()})",
    )


def _packages(diag: Diagnostics) -> None:
    for package, why in _OPTIONAL_PACKAGES:
        version = _version(package)
        if version is None:
            diag.add("packages", package, WARN, f"not installed: {why}")
        else:
            diag.add("packages", package, OK, version)


def _device(diag: Diagnostics) -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        diag.add(
            "device",
            "torch",
            WARN,
            'not installed, so training and evaluation are unavailable (pip install "maatml[ml]")',
        )
        return

    from .device import get_profile, is_distributed, resolve_device

    device = resolve_device("auto")
    profile = get_profile(device)
    diag.add("device", "auto resolves to", OK, str(device))
    diag.add(
        "device",
        f"profile {profile.name}",
        OK,
        f"mid_train_eval={profile.allow_mid_train_eval} "
        f"workers={profile.dataloader_workers} "
        f"grad_checkpointing={profile.allow_grad_checkpointing} "
        f"weights={profile.weights_dtype_policy} "
        f"quantized_load={profile.allow_quantized_load}",
    )
    diag.add("device", "distributed", OK, "yes" if is_distributed() else "no")


def _plugins(diag: Diagnostics) -> None:
    from .registry import discover_plugins, list_all_plugins, load_errors

    discover_plugins()
    # Predictors and exporters register on import of their modules.
    from .evaluation import predictors as _predictors  # noqa: F401

    for kind, entries in list_all_plugins().items():
        names = ", ".join(e.name for e in entries) or "(none)"
        diag.add("plugins", kind, OK, f"{len(entries)}: {names}")

    for source, error in load_errors():
        # An optional extra that is simply absent is a warning; anything else
        # is a plugin that meant to load and did not.
        status = WARN if source.startswith("module:maatml.") else ERROR
        diag.add("plugins", source, status, error)


def _count_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _check_registered(
    diag: Diagnostics,
    section: str,
    *,
    label: str,
    name: Any,
    registry: Any,
) -> None:
    """ERROR when a configured plugin name is not in the given registry."""
    if name is None:
        return
    if not isinstance(name, str):
        diag.add(
            section,
            label,
            ERROR,
            f"{label} must be a string name; got {type(name).__name__}",
        )
        return
    if registry.get(name) is not None:
        diag.add(section, label, OK, f"{name} resolves")
        return
    known = ", ".join(registry.names()) or "(none)"
    diag.add(
        section,
        label,
        ERROR,
        f"{name!r} is not registered (known: {known})",
    )


def _latest_eval_report(eval_dir: Path) -> Optional[Path]:
    if not eval_dir.is_dir():
        return None
    reports = sorted(
        (p for p in eval_dir.glob("*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


def _is_hub_model_id(base_model: str) -> bool:
    """Heuristic: Hub ids look like ``org/name``; skip local / torchvision paths."""
    if not base_model or "/" not in base_model:
        return False
    if base_model.startswith(("torchvision/", ".", "/", "~")):
        return False
    return not Path(base_model).exists()


def _check_base_model_cache(diag: Diagnostics, section: str, base_model: Optional[str]) -> None:
    if not base_model or not _is_hub_model_id(base_model):
        return
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        diag.add(
            section,
            "base_model cache",
            WARN,
            f"{base_model}: huggingface_hub not installed; cannot check local cache",
        )
        return
    # config.json is the cheapest marker that a snapshot is present locally.
    # try_to_load_from_cache returns None (or a sentinel) when the file is absent.
    cached = try_to_load_from_cache(base_model, "config.json")
    if isinstance(cached, (str, Path)):
        diag.add(section, "base_model cache", OK, f"{base_model} cached at {cached}")
    else:
        diag.add(
            section,
            "base_model cache",
            WARN,
            f"{base_model} not in local Hub cache; first train will download it",
        )


def _model_folder(diag: Diagnostics, model_dir: Path) -> None:
    from .config import config_key_warnings, get_dataset_cfg, load_model_def
    from .device import get_profile, resolve_device
    from .evaluation.harness import GateConfigError, resolve_gate_spec, resolve_validator
    from .registry import FORMATS, GENERATORS, METRICS, PREDICTORS, TRAINERS
    from .runs import list_runs
    from .scaffold import normalize_architecture

    section = "model"
    try:
        md = load_model_def(model_dir)
    except Exception as exc:  # noqa: BLE001  audit reports, never raises
        diag.add(section, str(model_dir), ERROR, f"failed to load model.yml: {exc}")
        return

    diag.add(section, "identity", OK, f"{md.identity} ({md.architecture})")

    arch = normalize_architecture(md.architecture)
    if TRAINERS.get(md.architecture) or TRAINERS.get(arch):
        diag.add(section, "architecture", OK, f"{md.architecture} is registered")
    else:
        diag.add(
            section,
            "architecture",
            ERROR,
            f"{md.architecture!r} has no registered trainer "
            f"(known: {', '.join(TRAINERS.names()) or 'none'})",
        )

    cfg = get_dataset_cfg(md)
    fmt = str(cfg.get("format", "jsonl_seed"))
    if FORMATS.get(fmt):
        diag.add(section, "dataset.format", OK, f"{fmt} is registered")
    else:
        diag.add(section, "dataset.format", ERROR, f"{fmt!r} is not registered")

    generator = cfg.get("generator")
    if generator is not None:
        _check_registered(
            diag, section, label="dataset.generator", name=generator, registry=GENERATORS
        )

    try:
        md.validate_paths()
        diag.add(section, "declared paths", OK, "all present")
    except FileNotFoundError as exc:
        diag.add(section, "declared paths", ERROR, str(exc).replace("\n", " "))

    for warning in config_key_warnings(md):
        diag.add(section, "config keys", WARN, warning)

    # Seed / benchmark corpus counts (independent of prepare).
    for key, label in (
        ("seed_samples", "seed corpus"),
        ("benchmark_samples", "benchmark corpus"),
    ):
        rel = cfg.get(key)
        if not isinstance(rel, str):
            continue
        path = md.resolve(rel)
        if not path.is_file():
            diag.add(section, label, ERROR, f"{rel} missing at {path}")
            continue
        n = _count_rows(path)
        empty_hint = "  (empty; run maatml datagen or a seed builder)" if n == 0 else ""
        diag.add(
            section,
            label,
            WARN if n == 0 else OK,
            f"{n} rows in {rel}{empty_hint}",
        )

    splits = []
    for split in ("train", "val", "test"):
        path = md.prepared_dir / f"{split}.jsonl"
        splits.append(f"{split}={_count_rows(path) if path.is_file() else 'missing'}")
    empty = [s for s in splits if s.endswith(("=0", "=missing"))]
    try:
        from .data.populations import PopulationError, check_prepared_isolation

        problems = check_prepared_isolation(md)
    except PopulationError as exc:
        problems = [str(exc)]
    if problems:
        diag.add(section, "isolation", ERROR, "; ".join(problems))
    elif (get_dataset_cfg(md) or {}).get("isolation"):
        diag.add(section, "isolation", OK, "prepared populations satisfy dataset.isolation")
    diag.add(
        section,
        "prepared splits",
        WARN if empty else OK,
        " ".join(splits) + ("  (run maatml prepare)" if empty else ""),
    )

    runs = list_runs(md)
    completed = [r for r in runs if r.status == "completed"]
    diag.add(
        section,
        "runs",
        OK if completed else WARN,
        f"{len(runs)} recorded, {len(completed)} completed"
        + ("" if completed else "  (run maatml train)"),
    )

    corrupt = md.output_dir / "runs.jsonl.corrupt"
    if corrupt.is_file():
        n_corrupt = _count_rows(corrupt)
        diag.add(
            section,
            "runs quarantine",
            WARN,
            f"{corrupt.name} has {n_corrupt} unparseable line(s); "
            "maatml runs skips them but the file should be inspected",
        )

    validator = (md.evaluation or {}).get("validator")
    if validator is None:
        diag.add(
            section,
            "evaluation.validator",
            WARN,
            "not configured: evaluate scores JSON parse only and datagen refuses "
            "to run without --allow-ungated",
        )
    else:
        try:
            resolve_validator(validator)
            diag.add(section, "evaluation.validator", OK, f"{validator} resolves")
        except GateConfigError as exc:
            diag.add(section, "evaluation.validator", ERROR, str(exc).replace("\n", " "))

    metrics_cfg = (md.evaluation or {}).get("metrics")
    if metrics_cfg is None:
        pass
    elif isinstance(metrics_cfg, (list, tuple)):
        for entry in metrics_cfg:
            _check_registered(
                diag,
                section,
                label=f"evaluation.metrics[{entry!r}]",
                name=entry,
                registry=METRICS,
            )
    else:
        _check_registered(
            diag, section, label="evaluation.metrics", name=metrics_cfg, registry=METRICS
        )

    predictor = (md.evaluation or {}).get("predictor")
    if predictor is None:
        inferred = arch if PREDICTORS.get(arch) or PREDICTORS.get(md.architecture) else None
        if inferred is not None:
            diag.add(
                section,
                "evaluation.predictor",
                OK,
                f"not set; evaluate will infer {inferred!r} from architecture",
            )
    else:
        _check_registered(
            diag, section, label="evaluation.predictor", name=predictor, registry=PREDICTORS
        )

    gates: dict[str, float] = {}
    try:
        gates = resolve_gate_spec(md)
        diag.add(
            section,
            "evaluation.gates",
            OK,
            ", ".join(f"{k}>={v:g}" for k, v in sorted(gates.items())),
        )
    except GateConfigError:
        diag.add(
            section,
            "evaluation.gates",
            WARN,
            "none configured: `maatml evaluate --gate` fails rather than passing vacuously",
        )

    if gates:
        latest = _latest_eval_report(md.eval_dir)
        if latest is not None:
            try:
                report = json.loads(latest.read_text(encoding="utf-8"))
                reported = set((report.get("metrics") or {}).keys())
            except (OSError, json.JSONDecodeError) as exc:
                diag.add(
                    section,
                    "gate keys",
                    WARN,
                    f"could not read latest eval report {latest.name}: {exc}",
                )
            else:
                missing = sorted(k for k in gates if k not in reported)
                if missing:
                    diag.add(
                        section,
                        "gate keys",
                        WARN,
                        f"{', '.join(missing)} gated but absent from {latest.name} "
                        "metrics (typo or metrics change; these gates can only fail)",
                    )
                else:
                    diag.add(
                        section,
                        "gate keys",
                        OK,
                        f"all gate keys present in {latest.name}",
                    )

    # QLoRA / quantization is CUDA-only; flag the conflict before train fails.
    quant = (md.training or {}).get("quantization")
    quant_on = False
    if isinstance(quant, dict):
        quant_on = bool(quant.get("load_in_4bit") or quant.get("load_in_8bit"))
    elif quant:
        quant_on = True
    if quant_on:
        try:
            import torch  # noqa: F401

            device = resolve_device("auto")
            profile = get_profile(device)
            if not profile.allow_quantized_load:
                diag.add(
                    section,
                    "training.quantization",
                    ERROR,
                    f"QLoRA / bitsandbytes requires CUDA; auto device is {device} "
                    f"(profile {profile.name}). Train with --device cuda or remove "
                    "training.quantization",
                )
            else:
                diag.add(
                    section,
                    "training.quantization",
                    OK,
                    f"enabled and device {device} allows quantized load",
                )
        except ImportError:
            diag.add(
                section,
                "training.quantization",
                WARN,
                "configured but torch is not installed; cannot verify device support",
            )

    _check_base_model_cache(diag, section, md.base_model)


def collect_diagnostics(model_dir: Optional[Path] = None) -> Diagnostics:
    """Gather environment (and optionally model-folder) diagnostics."""
    diag = Diagnostics()
    _environment(diag)
    _packages(diag)
    _device(diag)
    _plugins(diag)
    if model_dir is not None:
        _model_folder(diag, Path(model_dir))
    return diag
