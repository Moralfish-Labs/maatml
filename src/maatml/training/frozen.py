"""Exact tensor invariants for explicitly frozen model components."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def component_digest(module: Any) -> str:
    """Hash names, types, shapes and bytes of parameters and persistent buffers."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


class FrozenComponents:
    """Bind existing component objects before adaptation and refuse later changes."""

    def __init__(self, model: Any, prefixes: list[str]) -> None:
        if any(not prefix for prefix in prefixes) or len(set(prefixes)) != len(prefixes):
            raise ValueError("Frozen component prefixes must be nonempty and unique")
        self.modules = {prefix: model.get_submodule(prefix) for prefix in prefixes}
        if any(not module.state_dict() for module in self.modules.values()):
            raise ValueError("Frozen components must contain tensors")
        self.hashes = {prefix: component_digest(module) for prefix, module in self.modules.items()}

    def verify(self) -> None:
        """Reject trainable parameters, inserted adapters, or changed component tensors."""
        for prefix, module in self.modules.items():
            if any(parameter.requires_grad for parameter in module.parameters()):
                raise ValueError(f"Frozen component has trainable parameters: {prefix}")
            if component_digest(module) != self.hashes[prefix]:
                raise ValueError(f"Frozen component tensors changed: {prefix}")


def make_frozen_guard_callback(components: FrozenComponents) -> Any:
    """Check frozen tensors at saved checkpoints before publishing snapshot readiness."""
    from transformers import TrainerCallback

    class FrozenGuardCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            components.verify()
            return control

    return FrozenGuardCallback()
