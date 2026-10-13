"""Device profile + distributed / quantization guard tests."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import pytest

from maatml.device import (
    effective_dataloader_workers,
    ensure_quantization_allowed,
    get_profile,
    is_distributed,
    is_main_process,
    resolve_device,
    resolve_load_dtype,
)


def test_mps_profile() -> None:
    p = get_profile("mps")
    assert p.name == "mps"
    assert p.allow_mid_train_eval is False
    assert p.dataloader_workers == 0
    assert p.allow_grad_checkpointing is False
    assert p.weights_dtype_policy == "fp32_master"
    assert p.allow_quantized_load is False


def test_cuda_profile() -> None:
    p = get_profile("cuda")
    assert p.name == "cuda"
    assert p.allow_mid_train_eval is True
    assert p.dataloader_workers == 2
    assert p.allow_grad_checkpointing is True
    assert p.weights_dtype_policy == "native"
    assert p.allow_quantized_load is True


def test_cpu_profile() -> None:
    p = get_profile("cpu")
    assert p.name == "cpu"
    assert p.allow_mid_train_eval is True
    assert p.dataloader_workers == 0
    assert p.allow_grad_checkpointing is False
    assert p.weights_dtype_policy == "fp32_master"
    assert p.allow_quantized_load is False


def test_get_profile_from_device_like_object() -> None:
    """Profiles accept any object with a ``.type`` attribute (no torch needed)."""
    p = get_profile(SimpleNamespace(type="mps"))
    assert p.name == "mps"


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_get_profile_from_torch_device() -> None:
    torch = pytest.importorskip("torch")
    p = get_profile(torch.device("cpu"))
    assert p.name == "cpu"


def test_quantization_guard_rejects_mps_and_cpu() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        ensure_quantization_allowed(get_profile("mps"))
    with pytest.raises(RuntimeError, match="CUDA"):
        ensure_quantization_allowed(get_profile("cpu"))
    ensure_quantization_allowed(get_profile("cuda"))  # no raise


def test_effective_dataloader_workers_override() -> None:
    p = get_profile("cuda")
    assert effective_dataloader_workers(p, None) == 2
    assert effective_dataloader_workers(p, 0) == 0
    assert effective_dataloader_workers(p, 8) == 8


def test_resolve_load_dtype_policies() -> None:
    pytest.importorskip("torch")
    import torch

    assert resolve_load_dtype(get_profile("mps"), "bf16") is None
    assert resolve_load_dtype(get_profile("cpu"), "fp16") is None
    assert resolve_load_dtype(get_profile("cuda"), "bf16") is torch.bfloat16
    assert resolve_load_dtype(get_profile("cuda"), "fp16") is torch.float16
    assert resolve_load_dtype(get_profile("cuda"), "fp32") is None


def test_a_frozen_base_skips_the_fp32_master_policy() -> None:
    """A LoRA's base takes no optimizer updates, so on mps/cpu it loads at the
    requested precision. PEFT creates the adapter in fp32 either way."""
    pytest.importorskip("torch")
    import torch

    assert resolve_load_dtype(get_profile("mps"), "bf16", frozen_base=True) is torch.bfloat16
    assert resolve_load_dtype(get_profile("cpu"), "fp16", frozen_base=True) is torch.float16
    # fp32 asked for is fp32 given, frozen or not.
    assert resolve_load_dtype(get_profile("mps"), "fp32", frozen_base=True) is None
    # And a full fine-tune on mps still keeps its master weights.
    assert resolve_load_dtype(get_profile("mps"), "bf16", frozen_base=False) is None


def test_is_distributed_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert is_distributed() is False
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert is_distributed() is True
    assert is_main_process() is False
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert is_main_process() is True


def test_is_main_process_single() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE"):
            os.environ.pop(key, None)
        assert is_main_process() is True


def test_a_cache_release_callback_fires_on_its_cadence() -> None:
    """The allocator is released during a run, not only after it: on mps the
    cache is never reclaimed on its own and the run slides into swap."""
    pytest.importorskip("transformers")
    from maatml.training.guards import make_cache_release_callback

    class Spy:
        def __init__(self) -> None:
            self.calls = 0

        def empty_cache(self) -> None:
            self.calls += 1

    spy = Spy()
    callback = make_cache_release_callback(spy, every=4)
    state = SimpleNamespace(global_step=0)
    for step in range(1, 13):
        state.global_step = step
        callback.on_step_end(None, state, None)
    assert spy.calls == 3


def test_a_cadence_of_zero_never_releases() -> None:
    pytest.importorskip("transformers")
    from maatml.training.guards import make_cache_release_callback

    class Spy:
        calls = 0

        def empty_cache(self) -> None:
            type(self).calls += 1

    callback = make_cache_release_callback(Spy(), every=0)
    for step in range(1, 9):
        callback.on_step_end(None, SimpleNamespace(global_step=step), None)
    assert Spy.calls == 0
