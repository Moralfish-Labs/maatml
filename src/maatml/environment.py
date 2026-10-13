"""The environment a record was produced in, captured on every train and evaluate.

A number without the machine behind it cannot be reproduced or explained: a
cuDNN change, a driver change or non-deterministic kernels move a metric as
surely as a recipe change. Everything here is best-effort and never raises —
provenance capture must not fail a run — and a field that cannot be read is
``None``, never a guess.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

ENVIRONMENT_KIND = "maatml.environment/1"
_DETERMINISM_ENV = ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "TOKENIZERS_PARALLELISM")


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return None


def _git_sha(cwd: Optional[Path]) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _nvidia_driver() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        first = out.strip().splitlines()
        return first[0].strip() if first else None
    except Exception:  # noqa: BLE001
        return None


def _accelerator() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda": None,
        "cudnn": None,
        "gpus": [],
        "driver": None,
        "mps": False,
    }
    try:
        import torch
    except Exception:  # noqa: BLE001
        return info
    try:
        info["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["cudnn"] = torch.backends.cudnn.version()
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["driver"] = _nvidia_driver()
        mps = getattr(torch.backends, "mps", None)
        info["mps"] = bool(mps is not None and mps.is_available())
    except Exception:  # noqa: BLE001
        pass
    return info


def _determinism() -> dict[str, Any]:
    settings: dict[str, Any] = {name: os.environ.get(name) for name in _DETERMINISM_ENV}
    try:
        import torch

        settings["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
        settings["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        settings["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except Exception:  # noqa: BLE001
        settings.setdefault("deterministic_algorithms", None)
    return settings


def environment_manifest(model_dir: Optional[Path] = None) -> dict[str, Any]:
    """Git SHA, interpreter, packages, accelerator, OS and determinism settings."""
    accelerator = _accelerator()
    return {
        "kind": ENVIRONMENT_KIND,
        "git_sha": _git_sha(Path(model_dir) if model_dir else None),
        "python": platform.python_version(),
        "executable": sys.executable,
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "packages": {
            name: _pkg_version(name)
            for name in ("maatml", "torch", "transformers", "peft", "safetensors")
        },
        **accelerator,
        "determinism": _determinism(),
    }


def render_environment(env: dict[str, Any]) -> list[str]:
    packages = env.get("packages") or {}
    lines = [
        f"git {str(env.get('git_sha'))[:12]}  python {env.get('python')}  {env.get('os')}",
        "  " + "  ".join(f"{k}={v}" for k, v in packages.items()),
    ]
    gpus = env.get("gpus") or []
    if gpus:
        lines.append(
            f"  cuda {env.get('cuda')}  cudnn {env.get('cudnn')}  driver {env.get('driver')}  "
            f"gpu {', '.join(gpus)}"
        )
    elif env.get("mps"):
        lines.append("  mps")
    det = env.get("determinism") or {}
    lines.append("  determinism " + " ".join(f"{k}={v}" for k, v in det.items()))
    return lines
