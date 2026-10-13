"""Device resolution and per-backend training profiles.

Apple Silicon / MPS has historically needed conservative settings (no mid-train
eval, zero dataloader workers, no grad checkpointing, fp32 master weights).
CUDA can run more aggressively; CPU sits in between.

Torch is imported lazily so CPU-only installs (``pip install maatml`` without
the ``[ml]`` extra) can still import this module for CLI plugins / validate.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Union


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    allow_mid_train_eval: bool
    dataloader_workers: int
    allow_grad_checkpointing: bool
    weights_dtype_policy: str  # "fp32_master" | "native"
    allow_quantized_load: bool  # bitsandbytes / QLoRA, CUDA only

    def empty_cache(self) -> None:
        """Release allocator caches when the backend supports it."""
        torch = _torch()
        if self.name == "mps" and hasattr(torch, "mps"):
            # A backend that refuses to drop its cache is not a failure.
            with contextlib.suppress(Exception):
                torch.mps.empty_cache()
        elif self.name == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def apply_env(self) -> None:
        """Set backend-specific environment knobs (idempotent)."""
        if self.name == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


_PROFILES: dict[str, DeviceProfile] = {
    "mps": DeviceProfile(
        name="mps",
        allow_mid_train_eval=False,
        dataloader_workers=0,
        allow_grad_checkpointing=False,
        weights_dtype_policy="fp32_master",
        allow_quantized_load=False,
    ),
    "cuda": DeviceProfile(
        name="cuda",
        allow_mid_train_eval=True,
        dataloader_workers=2,
        allow_grad_checkpointing=True,
        weights_dtype_policy="native",
        allow_quantized_load=True,
    ),
    "cpu": DeviceProfile(
        name="cpu",
        allow_mid_train_eval=True,
        dataloader_workers=0,
        allow_grad_checkpointing=False,
        weights_dtype_policy="fp32_master",
        allow_quantized_load=False,
    ),
}


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by wheel smoke
        raise ImportError("torch is required for device resolution; install maatml[ml]") from exc
    return torch


def is_distributed() -> bool:
    """True when launched under torchrun / accelerate (or torch.distributed init)."""
    if os.environ.get("LOCAL_RANK") is not None or os.environ.get("RANK") is not None:
        return True
    if os.environ.get("WORLD_SIZE") is not None:
        try:
            if int(os.environ["WORLD_SIZE"]) > 1:
                return True
        except ValueError:
            pass
    try:
        import torch.distributed as dist

        return bool(dist.is_available() and dist.is_initialized())
    except ImportError:
        return False


def is_main_process() -> bool:
    """Rank-0 (or single-process): use for run-registry / metadata writes."""
    if not is_distributed():
        return True
    for key in ("RANK", "LOCAL_RANK"):
        raw = os.environ.get(key)
        if raw is not None:
            try:
                return int(raw) == 0
            except ValueError:
                continue
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except ImportError:
        pass
    return True


def resolve_device(device: str) -> Any:
    """Map ``auto|mps|cpu|cuda|<torch device str>`` to a ``torch.device``.

    ``cpu`` is resolved without torch so a callable predictor can still run
    the eval loop on a torch-free install.
    """
    if device == "cpu":
        try:
            return _torch().device("cpu")
        except ImportError:
            return SimpleNamespace(type="cpu")
    torch = _torch()
    if device == "mps":
        return torch.device("mps")
    if device == "cuda":
        return torch.device("cuda")
    if device == "auto":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def get_profile(device: Union[Any, str]) -> DeviceProfile:
    """Return the training profile for a device (or device type / ``auto``)."""
    if isinstance(device, str):
        if device in _PROFILES:
            name = device
        elif device == "auto":
            name = resolve_device("auto").type
        else:
            name = resolve_device(device).type
    else:
        name = device.type

    profile = _PROFILES.get(name, _PROFILES["cpu"])
    profile.apply_env()
    return profile


def resolve_training_placement(
    device: str = "auto",
) -> tuple[Any, DeviceProfile, bool]:
    """Resolve device + profile for training.

    When distributed, return a cuda device placeholder and let HuggingFace
    Trainer / accelerate own placement (``use_cpu=False``).

    Returns ``(torch_device, profile, distributed)``.
    """
    if is_distributed():
        # Prefer cuda for multi-GPU launches; fall back to requested/auto.
        if device in ("auto", "cuda") or device.startswith("cuda"):
            profile = get_profile("cuda")
            return resolve_device("cuda"), profile, True
        target = resolve_device(device)
        return target, get_profile(target), True

    target = resolve_device(device)
    return target, get_profile(target), False


def resolve_load_dtype(
    profile: DeviceProfile, precision: str, *, frozen_base: bool = False
) -> Any | None:
    """``torch.dtype`` for ``from_pretrained``, or ``None`` (default / fp32).

    ``fp32_master`` (mps/cpu): keep default fp32 master weights.
    ``native`` (cuda): honour ``training.precision`` bf16/fp16.

    ``frozen_base`` overrides the ``fp32_master`` policy, and only that policy.
    Master weights are fp32 so optimizer updates stay stable, and a LoRA's base
    takes no gradients and no optimizer state: there is nothing for the extra
    precision to protect, and it is paid twice, once on the weights and again
    on every activation that follows their dtype. On mps the cost is larger
    than it looks, because autocast falls back to fp32 for many ops there.

    The adapter is unaffected: PEFT creates its parameters in fp32 regardless
    of the base dtype, so the trainable weights keep master precision.
    """
    if profile.weights_dtype_policy != "native" and not frozen_base:
        return None
    torch = _torch()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def effective_dataloader_workers(profile: DeviceProfile, override: Optional[int] = None) -> int:
    """Profile default, overridable via ``training.dataloader_workers``."""
    if override is not None:
        return int(override)
    return profile.dataloader_workers


def ensure_quantization_allowed(profile: DeviceProfile) -> None:
    """Raise if quantization / QLoRA is requested on a non-CUDA profile."""
    if not profile.allow_quantized_load:
        raise RuntimeError(
            f"Quantized model load (QLoRA / bitsandbytes) requires CUDA; "
            f"profile={profile.name!r} has allow_quantized_load=False. "
            "Install maatml[cuda] and train with --device cuda."
        )
