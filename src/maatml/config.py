"""Model definition schema (`models/<name>/model.yml`).

A single `model.yml` per model is the source of truth for that model's
lifecycle: data preparation, training, smoke training, and evaluation.
This module owns the schema and the loader.

Nested `data:` / `dataset:`, `training:`, `smoke:`, `evaluation:`, and
`packaging:` sections are plain dicts so each pipeline stage can validate
them against its own typed config without a rigid super-schema.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .utils.io import read_yaml

_SEMVER_RX = re.compile(r"^\d+\.\d+\.\d+$")

# Path-like keys commonly declared under data:/dataset:/evaluation:.
_PATH_KEYS = frozenset(
    {
        "schema",
        "prompt_spec",
        "seed_samples",
        "benchmark_samples",
        "contracts",
        "tokenizer",
        "template_dir",
        "attribution",
    }
)


class PackagingSpec(BaseModel):
    """Packaging knobs from ``model.yml``'s ``packaging:`` section.

    Consumed by ``maatml export`` and written into the export ``manifest.json``.
    ``confidence_thresholds`` is a plain dict so the YAML stays light.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_input_tokens: int = Field(gt=0, default=2048)
    expected_latency_ms: int = Field(gt=0, default=2000)
    confidence_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"high": 0.9, "low": 0.6}
    )
    # Half-precision export knob: `"f32"` (default) | `"f16"` | `"bf16"`.
    weights_dtype: str = "f32"


class ModelDefinition(BaseModel):
    """Top-level schema for `models/<name>/model.yml`."""

    # validate_assignment so CLI --set overrides (setattr) run the same
    # validators as load, instead of silently bypassing semver / gt=0 / types.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(..., description="Folder name; e.g. 'my-model'")
    model_id: str = Field(..., description="Stable model identifier; e.g. 'my-model'")
    # Free-form metadata; trainers dispatch primarily on ``architecture``.
    task: str = ""
    runtime: Optional[str] = None
    # Architecture dispatch key for the train registry / CLI.
    # Built-ins: causal_sft, seq2seq, multi_head_classifier (and legacy
    # aliases classifier / generative still accepted by the transitional CLI).
    architecture: str = "causal_sft"
    version: str = "0.1.0"
    description: str = ""
    base_model: Optional[str] = None

    # Nested sections, dicts so each stage typechecks its own subset.
    data: dict[str, Any] = Field(default_factory=dict)
    # Preferred over ``data:`` going forward; empty falls back to ``data:``.
    dataset: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    smoke: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    # Optional source-op stage for `maatml distill` (teacher labels over a
    # prompt pool). A plain dict here, parsed into the typed DistillConfig by
    # the command, like the other stage sections.
    distill: dict[str, Any] = Field(default_factory=dict)
    packaging: PackagingSpec = Field(default_factory=PackagingSpec)
    plugins: list[str] = Field(default_factory=list)
    extensions: dict[str, Any] = Field(default_factory=dict)

    # ----- Filled in by load_model_def, not present in YAML -----
    model_dir: Path = Field(default_factory=Path, exclude=True)

    @field_validator("version")
    @classmethod
    def _check_semver(cls, v: str) -> str:
        if not _SEMVER_RX.match(v):
            raise ValueError(
                f"version must match semver MAJOR.MINOR.PATCH (e.g. '0.1.0'); got {v!r}"
            )
        return v

    # --- helpers ---------------------------------------------------------

    @property
    def identity(self) -> str:
        """``name@version``: used as default checkpoint run name."""
        return f"{self.name}@{self.version}"

    def resolve(self, rel: str | Path) -> Path:
        """Resolve a YAML-declared path relative to the model folder."""
        p = Path(rel)
        return p if p.is_absolute() else (self.model_dir / p).resolve()

    @property
    def output_dir(self) -> Path:
        """`models/<name>/output/`: root of all generated artifacts."""
        return self.model_dir / "output"

    @property
    def prepared_dir(self) -> Path:
        """`models/<name>/output/prepared/`: train/val/test JSONL splits."""
        return self.output_dir / "prepared"

    @property
    def checkpoints_dir(self) -> Path:
        """`models/<name>/output/checkpoints/`."""
        return self.output_dir / "checkpoints"

    @property
    def eval_dir(self) -> Path:
        """`models/<name>/output/eval/`."""
        return self.output_dir / "eval"

    # Keys a `smoke:` overlay may carry that are not training knobs, so they
    # never reach a trainer config that forbids unknown fields.
    _SMOKE_ONLY_KEYS = ("gates",)

    def merged_smoke(self) -> dict[str, Any]:
        """Return ``training`` overlaid with ``smoke`` overrides.

        Used by `--smoke` to run a fast variant of the same training loop without
        a separate config file. ``smoke.gates`` (the smoke tier the lifecycle
        runner enforces) is not a training knob and is left out.
        """
        merged = dict(self.training)
        merged.update(self.smoke or {})
        # `smoke.base_model` overrides `training.model_id` if present
        if "base_model" in (self.smoke or {}):
            merged["model_id"] = self.smoke["base_model"]
            merged.pop("base_model", None)
        for key in self._SMOKE_ONLY_KEYS:
            merged.pop(key, None)
        return merged

    def validate_paths(self) -> None:
        """Raise ``FileNotFoundError`` if declared path fields are missing.

        Checks path-like keys under ``dataset:``, ``data:``, and ``evaluation:``
        (and ``data.sources`` / ``data.template_dir`` when present).
        """
        missing: list[str] = []
        sections = (
            ("dataset", self.dataset),
            ("data", self.data),
            ("evaluation", self.evaluation),
        )
        for section_name, section in sections:
            if not isinstance(section, dict):
                continue
            for key, val in section.items():
                if key not in _PATH_KEYS:
                    continue
                if not isinstance(val, str):
                    continue
                path = self.resolve(val)
                if not path.exists():
                    missing.append(f"{section_name}.{key}={val!r} -> {path}")
            for source in section.get("sources") or []:
                if isinstance(source, str):
                    path = self.resolve(source)
                    if not path.exists():
                        missing.append(f"{section_name}.sources:{source!r} -> {path}")
        if missing:
            raise FileNotFoundError(
                "ModelDefinition path check failed for "
                f"{self.identity}:\n  - " + "\n  - ".join(missing)
            )


# Backward-compat alias (prefer ModelDefinition).
ModelSpec = ModelDefinition


def get_dataset_cfg(md: ModelDefinition) -> dict[str, Any]:
    """Return dataset knobs with ``data:`` fallback for migration.

    Keys present in ``dataset:`` win; missing keys are filled from ``data:``.
    Target-field / request-field / sanitize / seed paths may live in either.
    """
    merged = dict(md.data or {})
    merged.update(md.dataset or {})
    return merged


def load_model_def(model_dir: str | Path, *, load_plugins: bool = True) -> ModelDefinition:
    """Read `<model_dir>/model.yml` and return a populated ModelDefinition.

    The returned object's `model_dir` attribute is set to the absolute path of
    the folder so `resolve(...)` and `output_dir` etc. work correctly.

    When ``load_plugins`` is False, any ``plugins:`` declared in model.yml are
    NOT imported. A model folder is executable code, so this is how a command
    can read the schema without running the folder's Python.
    """
    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    yml_path = model_dir / "model.yml"
    if not yml_path.is_file():
        raise FileNotFoundError(f"{yml_path} not found - each model folder must contain model.yml")

    raw = read_yaml(yml_path)
    if not isinstance(raw, dict):
        raise ValueError(f"{yml_path}: top-level must be a mapping")

    md = ModelDefinition(**raw)
    # Pydantic doesn't see `model_dir` from the YAML (it's not in the file);
    # set it post-construction.
    object.__setattr__(md, "model_dir", model_dir)

    # Load any folder-local / module plugins declared in model.yml.
    if load_plugins and md.plugins:
        from .registry import load_model_plugins

        load_model_plugins(model_dir, md.plugins)

    return md


# Known keys for the untyped dataset:/evaluation: sections. Kept in sync with
# the readers in data/, training/, export/ and scaffold's section builders. A
# typo produces a warning (never a hard failure), so plugins can still extend.
_DATASET_KNOWN_KEYS = frozenset(
    {
        "format",
        "request_field",
        "raw_field",
        "target_field",
        "target_key_order",
        "group_by",
        "seed_samples",
        "schema",
        "split_ratios",
        "seed",
        "sanitize",
        "prompt_spec",
        "source_prefix",
        "user_placeholder",
        "text_transform",
        "tokenizer",
        "generator",
        "benchmark_samples",
        "contracts",
        "template_dir",
        "sources",
        "base_model_name_or_path",
        # image_caption_folder (diffusion corpora)
        "source_dir",
        "val_fraction",
        "test_fraction",
        # populations
        "isolation",
        "pins",
        "blind_samples",
        "attribution",
        "attribution_field",
    }
)
_EVALUATION_KNOWN_KEYS = frozenset(
    {
        "predictor",
        "validator",
        "metrics",
        "gates",
        "gates_benchmark",
        "operating_point",
        "repair_braces",
        "slices",
        "cache_predictions",
        "batch_size",
    }
)


def config_key_warnings(md: ModelDefinition) -> list[str]:
    """Warn on unrecognized dataset:/evaluation: keys. Never fails validation."""
    warns: list[str] = []
    for key in md.dataset or {}:
        if key not in _DATASET_KNOWN_KEYS:
            warns.append(f"dataset.{key}: unrecognized key, ignored by known stages")
    ev = md.evaluation if isinstance(md.evaluation, dict) else {}
    # The key operating-point derive writes is whatever threshold_key names.
    op = ev.get("operating_point")
    threshold_key = op.get("threshold_key") if isinstance(op, dict) else None
    for key in ev:
        if key not in _EVALUATION_KNOWN_KEYS and key != threshold_key:
            warns.append(f"evaluation.{key}: unrecognized key, ignored by known stages")
    return warns
