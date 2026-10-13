"""Target compilation: portable export → device-specific runtime artifact.

Compilers are registered plugins (``@register_compiler``). Core owns the
dispatch, option parsing, and the derived-manifest envelope; plugins own the
engine (TensorRT, llama.cpp quantize, vLLM packaging, …).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .export.manifest import load_manifest
from .registry import COMPILERS, discover_plugins
from .utils.io import sha256_file, write_json


def promotion_status(manifest: dict[str, Any]) -> tuple[bool, str]:
    """Whether this export may be compiled for production.

    A rehearsal (``smoke_gated``) or an ungated / failed evaluate must not
    become a device artifact that later reads as a gate pass. Core owns that
    claim so every compiler plugin sees the same refusal.
    """
    evidence = manifest.get("gate_evidence")
    if not isinstance(evidence, dict) or not evidence:
        return False, "manifest has no gate_evidence"
    if evidence.get("smoke_gated"):
        return False, "export is smoke-gated; run a production evaluate --gate"
    if evidence.get("passed") is not True:
        return False, "gate did not pass"
    return True, "production gate passed"


def parse_options(raw: list[str] | None) -> dict[str, str]:
    """Parse ``KEY=VALUE`` strings into a dict. Empty value is allowed."""
    options: dict[str, str] = {}
    for item in raw or []:
        if "=" not in item:
            raise ValueError(
                f"compile/serve option {item!r} must be KEY=VALUE "
                "(repeat --option / --server-option for each)"
            )
        key, _, value = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"compile/serve option {item!r} has an empty key")
        options[key] = value
    return options


def compile_export(
    export_dir: str | Path,
    *,
    target: str,
    out_dir: str | Path,
    options: Optional[dict[str, str]] = None,
    require_gated: bool = False,
) -> Path:
    """Run a registered compiler against an export (or deployment) directory.

    The compiler receives the source export path, the destination directory,
    the loaded ``manifest.json``, and free-form ``options``. It must return the
    populated ``out_dir`` (or a path inside it). Core then writes a thin
    ``target_manifest.json`` that records the source identity, compiler name,
    option keys, and ``promotion_eligible`` so a serving host can refuse a
    mismatched or ungated device artifact.

    ``require_gated`` refuses before the plugin runs when ``gate_evidence`` is
    missing, failed, or ``smoke_gated``.
    """
    discover_plugins()
    export_dir = Path(export_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    root, manifest = load_manifest(export_dir)
    eligible, reason = promotion_status(manifest)
    if require_gated and not eligible:
        raise ValueError(f"compile --require-gated: {reason}")
    compiler = COMPILERS.require(target)
    opts = dict(options or {})
    result = compiler(root, out_dir, manifest=manifest, options=opts)
    result_path = Path(result).resolve() if result is not None else out_dir
    if not result_path.exists():
        result_path = out_dir

    source_files = manifest.get("files") or []
    primary = next(
        (
            f
            for f in source_files
            if isinstance(f, dict)
            and str(f.get("path", "")).endswith((".onnx", ".gguf", ".safetensors"))
        ),
        source_files[0] if source_files else None,
    )
    target_manifest: dict[str, Any] = {
        "kind": "maatml.target/1",
        "compiler": target,
        "source": {
            "name": manifest.get("name"),
            "version": manifest.get("version"),
            "identity": manifest.get("identity"),
            "architecture": manifest.get("architecture"),
            "run_id": manifest.get("run_id"),
            "gate_evidence": manifest.get("gate_evidence"),
            "export_dir": str(root),
            "manifest_sha256": sha256_file(root / "manifest.json")
            if (root / "manifest.json").is_file()
            else None,
            "primary_artifact": primary,
        },
        "options": opts,
        "out_dir": str(result_path if result_path.is_dir() else result_path.parent),
        "promotion_eligible": eligible,
        "promotion_reason": reason,
    }
    write_json(out_dir / "target_manifest.json", target_manifest)
    return result_path
