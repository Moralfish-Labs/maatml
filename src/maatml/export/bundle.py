"""Safetensors (HF-style) export bundle + top-level ``export_model`` dispatcher."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from ..config import ModelDefinition, get_dataset_cfg
from ..registry import EXPORTERS, register_exporter
from ..scaffold import normalize_architecture
from .manifest import build_manifest, write_manifest

# Artifact globs / names commonly present in HF checkpoint dirs.
_WEIGHT_NAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "adapter_model.safetensors",
    "adapter_config.json",
    "classifier_heads.safetensors",
)
_CONFIG_NAMES = (
    "config.json",
    "generation_config.json",
    "training_args.bin",
    "run_metadata.json",
    # Multimodal processor assets (VLM checkpoints; needed for vLLM serving).
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "chat_template.json",
    "chat_template.jinja",
)
_TOKENIZER_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "spm.model",
    "sentencepiece.bpe.model",
)


def resolve_export_format(
    architecture: str,
    requested: Optional[str] = None,
) -> str:
    """Pick export format; enforce architecture constraints.

    Known formats come from the ``EXPORTERS`` registry (built-ins + plugins).
    ``gguf`` stays gated to causal / preference architectures; ``mlx`` also
    accepts seq2seq (served as a direct-load bundle, no mlx_lm conversion).
    """
    arch = normalize_architecture(architecture)
    if requested is None or requested == "auto":
        return "safetensors"

    fmt = requested.lower().strip()
    known = set(EXPORTERS.names()) or {"safetensors", "gguf", "mlx"}
    if fmt not in known:
        raise ValueError(f"Unknown export format {requested!r}; known: {', '.join(sorted(known))}")

    if fmt == "gguf" and arch not in ("causal_sft", "dpo", "orpo"):
        raise ValueError(
            f"Format {fmt!r} is only supported for causal / preference architectures; "
            f"architecture={architecture!r} must use safetensors"
        )
    if fmt == "mlx" and arch not in ("causal_sft", "dpo", "orpo", "seq2seq"):
        raise ValueError(
            f"Format {fmt!r} is only supported for causal / preference / seq2seq "
            f"architectures; architecture={architecture!r} must use safetensors"
        )
    return fmt


def _copy_if_exists(src: Path, dest_dir: Path) -> Optional[Path]:
    if not src.is_file():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    return dest


def _copy_weight_shards(ckpt: Path, dest_dir: Path) -> list[Path]:
    """Copy safetensors / bin shards referenced by an index, or single-file weights."""
    copied: list[Path] = []
    for name in _WEIGHT_NAMES:
        path = ckpt / name
        if not path.is_file():
            continue
        dest = _copy_if_exists(path, dest_dir)
        if dest is not None:
            copied.append(dest)
        # Follow index → shard listing when present.
        if name.endswith(".index.json"):
            try:
                import json

                index = json.loads(path.read_text(encoding="utf-8"))
                weight_map = index.get("weight_map") or {}
                for shard in sorted(set(weight_map.values())):
                    shard_path = ckpt / shard
                    shard_dest = _copy_if_exists(shard_path, dest_dir)
                    if shard_dest is not None:
                        copied.append(shard_dest)
            except Exception:  # noqa: BLE001  best-effort shard copy
                pass
    # Catch remaining *.safetensors not listed above (multi-shard without index match).
    for path in sorted(ckpt.glob("*.safetensors")):
        if any(c.name == path.name for c in copied):
            continue
        dest = _copy_if_exists(path, dest_dir)
        if dest is not None:
            copied.append(dest)
    return copied


def _copy_sidecar_assets(model_def: ModelDefinition, dest_dir: Path) -> list[Path]:
    """Copy schema / contracts / prompt_spec declared in model.yml when present."""
    cfg = get_dataset_cfg(model_def)
    copied: list[Path] = []
    for key in ("schema", "contracts", "prompt_spec", "tokenizer"):
        rel = cfg.get(key)
        if not isinstance(rel, str):
            continue
        src = model_def.resolve(rel)
        if not src.is_file():
            continue
        # Keep original basename so predictors / eval can find them.
        dest = _copy_if_exists(src, dest_dir)
        if dest is not None:
            copied.append(dest)
    return copied


@register_exporter("safetensors")
def export_safetensors_bundle(
    model_def: ModelDefinition,
    checkpoint_dir: Path,
    out_dir: Path,
    *,
    run_id: Optional[str] = None,
) -> Path:
    """Copy checkpoint weights + tokenizer + sidecars; write ``manifest.json``."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    files: list[Path] = []
    files.extend(_copy_weight_shards(checkpoint_dir, out_dir))
    for name in _CONFIG_NAMES + _TOKENIZER_NAMES:
        dest = _copy_if_exists(checkpoint_dir / name, out_dir)
        if dest is not None:
            files.append(dest)
    files.extend(_copy_sidecar_assets(model_def, out_dir))

    if not files:
        raise FileNotFoundError(
            f"No exportable artifacts found under {checkpoint_dir} "
            "(expected model.safetensors / tokenizer / config)"
        )

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[Path] = []
    for f in files:
        key = f.resolve().as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)

    manifest = build_manifest(
        model_def=model_def,
        export_dir=out_dir,
        files=unique,
        formats=["safetensors"],
        source_checkpoint=checkpoint_dir,
        run_id=run_id,
    )
    write_manifest(out_dir, manifest)
    return out_dir


def export_model(
    model_def: ModelDefinition,
    checkpoint_dir: Path,
    out_dir: Path,
    *,
    format: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Path:
    """Dispatch to a registered exporter for ``format``."""
    fmt = resolve_export_format(model_def.architecture, format)
    exporter = EXPORTERS.get(fmt)
    if exporter is None:
        raise KeyError(
            f"No exporter registered for format {fmt!r}. "
            f"Known: {', '.join(EXPORTERS.names()) or '(none)'}"
        )
    return exporter(model_def, checkpoint_dir, out_dir, run_id=run_id)


def run_parity_check(
    model_def: ModelDefinition,
    export_dir: Path,
    *,
    device: str = "auto",
    max_input_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Re-run evaluation gates against ``dataset.benchmark_samples`` if present.

    Lightweight: uses the same predictor as ``evaluate`` with
    ``checkpoint_dir=export_dir``. Returns a dict with ``passed``, ``gates``,
    and ``metrics``. Skips (passed=True, skipped=True) when no benchmark path.
    """
    from ..evaluation.harness import check_gates, run_evaluation
    from ..utils.io import iter_jsonl, write_jsonl

    cfg = get_dataset_cfg(model_def)
    bench_rel = cfg.get("benchmark_samples")
    if not bench_rel:
        return {"passed": True, "skipped": True, "reason": "no benchmark_samples"}

    bench_path = model_def.resolve(str(bench_rel))
    if not bench_path.is_file():
        return {
            "passed": False,
            "skipped": False,
            "reason": f"benchmark_samples missing: {bench_path}",
        }

    # Materialise benchmark as a temporary split for the harness.
    tmp_dir = Path(export_dir) / "_parity_data"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rows = list(iter_jsonl(bench_path))
    write_jsonl(tmp_dir / "test.jsonl", rows)

    from ..registry import PREDICTORS
    from ..scaffold import normalize_architecture

    ev = model_def.evaluation or {}
    predictor = ev.get("predictor")
    validator = ev.get("validator")
    metrics = ev.get("metrics")
    if isinstance(metrics, list) and not metrics:
        metrics = None
    arch = normalize_architecture(model_def.architecture)
    if predictor is None:
        if PREDICTORS.get(model_def.architecture):
            predictor = model_def.architecture
        elif PREDICTORS.get(arch):
            predictor = arch
    if predictor is None:
        return {
            "passed": False,
            "skipped": False,
            "reason": "no predictor configured for parity",
        }

    tokens = max_input_tokens or model_def.packaging.max_input_tokens
    out_path = Path(export_dir) / "parity_report.json"
    report = run_evaluation(
        checkpoint_dir=Path(export_dir),
        dataset_dir=tmp_dir,
        out_path=out_path,
        model_def=model_def,
        predictor=predictor,
        validator=validator,
        metrics_fn=metrics,
        device=device,
        split="test",
        max_input_tokens=tokens,
        task=model_def.task,
        enforce_gates=False,
    )

    gates_cfg = (model_def.evaluation or {}).get("gates") or {}
    result: dict[str, Any] = {
        "skipped": False,
        "metrics": report.metrics,
        "report": str(out_path),
    }
    if gates_cfg:
        gate_out = check_gates(report.metrics, gates_cfg)
        result["gates"] = gate_out
        result["passed"] = bool(gate_out["passed"])
    else:
        result["passed"] = True
        result["gates"] = None
    return result
