"""Frozen modality invariants include buffers and reject adapter insertion."""

import pytest

from maatml.training.frozen import FrozenComponents, component_digest

torch = pytest.importorskip("torch")


def test_frozen_tensors_and_buffers_are_bound():
    model = torch.nn.Module()
    model.vision = torch.nn.Linear(2, 2)
    model.vision.register_buffer("calibration", torch.tensor(1.0))
    model.vision.requires_grad_(False)
    guard = FrozenComponents(model, ["vision"])
    guard.verify()
    assert guard.hashes["vision"] == component_digest(model.vision)
    model.vision.calibration.add_(1)
    with pytest.raises(ValueError, match="tensors changed"):
        guard.verify()


def test_trainable_adapter_in_frozen_tower_is_rejected():
    model = torch.nn.Module()
    model.audio = torch.nn.Linear(2, 2).requires_grad_(False)
    guard = FrozenComponents(model, ["audio"])
    model.audio.adapter = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="trainable"):
        guard.verify()
    model.audio.adapter.requires_grad_(False)
    with pytest.raises(ValueError, match="tensors changed"):
        guard.verify()


def test_empty_prefix_cannot_freeze_the_whole_model_by_accident():
    with pytest.raises(ValueError, match="nonempty"):
        FrozenComponents(torch.nn.Linear(2, 2), [""])
