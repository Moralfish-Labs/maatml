"""PEFT adapter export: ``adapters/<name>/`` holding the two files a server loads.

``maatml export --format adapter`` reads a checkpoint trained with
``training.lora.save_mode: adapter`` (or ``both``) and writes::

    <out>/adapters/<name>/adapter_config.json
    <out>/adapters/<name>/adapter_model.safetensors
    <out>/manifest.json

The adapter folder holds exactly those two files, so it can be copied as it
is into a runtime that loads LoRA modules by folder. Tokenizer, prompt spec
and run metadata stay in the checkpoint.

```yaml
extensions:
  adapter:
    name: my-adapter           # folder name; defaults to the model name
    spec: adapter-spec.json    # optional: {"rank": 16, "tensors": {name: [dims]}}
```

With ``spec`` set, the export fails unless the adapter has exactly the spec's
tensors with the spec's shapes and rank. The spec is the layout a serving
stack was shown to load; tensors are compared by their safetensors header, so
no weights are read and torch is not needed.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..config import ModelDefinition
from ..registry import register_exporter
from ..utils.io import sha256_file
from .manifest import build_manifest, read_safetensors_shapes, write_manifest

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_FILES = (ADAPTER_CONFIG, ADAPTER_WEIGHTS)

_NAME_RX = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_LISTED = 5


@dataclass(frozen=True)
class AdapterSpec:
    """The adapter layout an export must match."""

    rank: int
    tensors: dict[str, list[int]]


def load_adapter_spec(path: Path) -> AdapterSpec:
    """Read ``{"rank": int, "tensors": {name: [dims]}}``; other keys are ignored."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"adapter spec {path} is unreadable: {exc}") from exc
    rank = raw.get("rank") if isinstance(raw, dict) else None
    tensors = raw.get("tensors") if isinstance(raw, dict) else None
    shapes_ok = isinstance(tensors, dict) and all(
        isinstance(shape, list) and all(isinstance(d, int) for d in shape)
        for shape in tensors.values()
    )
    if not isinstance(rank, int) or isinstance(rank, bool) or not tensors or not shapes_ok:
        raise ValueError(
            f"adapter spec {path} must hold an integer rank and tensors: {{name: [dims]}}"
        )
    return AdapterSpec(rank=rank, tensors={str(k): list(v) for k, v in tensors.items()})


def _adapter_settings(model_def: ModelDefinition) -> tuple[str, Optional[Path]]:
    """``(folder name, spec path or None)`` from ``extensions.adapter``."""
    cfg = (model_def.extensions or {}).get("adapter", {})
    if not isinstance(cfg, dict):
        raise ValueError("extensions.adapter must be a mapping")
    unknown = sorted(set(cfg) - {"name", "spec"})
    if unknown:
        raise ValueError(f"extensions.adapter has unknown key(s): {', '.join(unknown)}")
    name = cfg.get("name", model_def.name)
    if not isinstance(name, str) or not _NAME_RX.fullmatch(name):
        raise ValueError(
            f"extensions.adapter.name {name!r} must be a plain folder name "
            "(letters, digits, dot, underscore, hyphen)"
        )
    spec = cfg.get("spec")
    if spec is None:
        return name, None
    if not isinstance(spec, str) or not spec:
        raise ValueError("extensions.adapter.spec must be a path")
    path = Path(spec)
    return name, path if path.is_absolute() else model_def.resolve(spec)


def find_adapter(checkpoint_dir: Path) -> Path:
    """The folder of a checkpoint that holds the adapter's two files."""
    for folder in (checkpoint_dir, checkpoint_dir / "adapter"):
        if all((folder / name).is_file() for name in ADAPTER_FILES):
            return folder
    raise FileNotFoundError(
        f"No adapter under {checkpoint_dir} (expected {' and '.join(ADAPTER_FILES)}); "
        "train with training.lora.save_mode: adapter or both"
    )


def _some(names: list[str]) -> str:
    more = f" and {len(names) - _LISTED} more" if len(names) > _LISTED else ""
    return ", ".join(names[:_LISTED]) + more


def check_adapter(adapter_dir: Path, spec: Optional[AdapterSpec] = None) -> dict[str, Any]:
    """Validate an adapter folder; return ``{"rank", "tensors"}`` or raise ``ValueError``.

    Always: the config states a rank and every ``lora_A`` / ``lora_B`` tensor
    has it. With ``spec``: the same rank, the same tensor names, the same
    shapes.
    """
    try:
        config = json.loads((adapter_dir / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{ADAPTER_CONFIG} is unreadable: {exc}") from exc
    rank = config.get("r") if isinstance(config, dict) else None
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError(f"{ADAPTER_CONFIG} states no rank (r={rank!r})")
    shapes = read_safetensors_shapes(adapter_dir / ADAPTER_WEIGHTS)
    if not shapes:
        raise ValueError(f"{ADAPTER_WEIGHTS} has no readable tensor header")

    problems: list[str] = []
    if not config.get("rank_pattern"):
        off_rank = sorted(
            name
            for name, shape in shapes.items()
            if (".lora_A." in name and shape[:1] != [rank])
            or (".lora_B." in name and shape[1:2] != [rank])
        )
        if off_rank:
            problems.append(f"{len(off_rank)} tensor(s) are not rank {rank}: {_some(off_rank)}")
    if spec is not None:
        if rank != spec.rank:
            problems.append(f"rank {rank}, the spec has {spec.rank}")
        missing = sorted(set(spec.tensors) - set(shapes))
        extra = sorted(set(shapes) - set(spec.tensors))
        reshaped = sorted(
            f"{name} {shapes[name]} (spec {spec.tensors[name]})"
            for name in set(shapes) & set(spec.tensors)
            if shapes[name] != spec.tensors[name]
        )
        if missing:
            problems.append(f"{len(missing)} spec tensor(s) missing: {_some(missing)}")
        if extra:
            problems.append(f"{len(extra)} tensor(s) not in the spec: {_some(extra)}")
        if reshaped:
            problems.append(f"{len(reshaped)} tensor(s) differ in shape: {_some(reshaped)}")
    if problems:
        raise ValueError("adapter does not match: " + "; ".join(problems))
    return {"rank": rank, "tensors": len(shapes)}


@register_exporter("adapter")
def export_adapter(
    model_def: ModelDefinition,
    checkpoint_dir: Path,
    out_dir: Path,
    *,
    run_id: Optional[str] = None,
) -> Path:
    """Write ``adapters/<name>/`` with the adapter's two files, and ``manifest.json``."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
    name, spec_path = _adapter_settings(model_def)
    source = find_adapter(checkpoint_dir)
    spec = load_adapter_spec(spec_path) if spec_path is not None else None
    # Checked before anything is written: a refused adapter leaves no folder behind.
    summary = check_adapter(source, spec)

    target = out_dir / "adapters" / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    files = [Path(shutil.copy2(source / item, target / item)) for item in ADAPTER_FILES]
    found = sorted(path.name for path in target.iterdir())
    if found != sorted(ADAPTER_FILES):
        raise RuntimeError(f"{target} holds {found}, expected exactly {sorted(ADAPTER_FILES)}")

    hints = {
        "adapter": {
            "name": name,
            **summary,
            "spec": (
                {"path": str(spec_path), "sha256": sha256_file(spec_path)}
                if spec_path is not None
                else None
            ),
        }
    }
    manifest = build_manifest(
        model_def=model_def,
        export_dir=out_dir,
        files=files,
        formats=["adapter"],
        source_checkpoint=checkpoint_dir,
        run_id=run_id,
        extra_runtime_hints=hints,
    )
    write_manifest(out_dir, manifest)
    return out_dir
