"""QLoRA / quantization config parsing (no bitsandbytes required)."""

from __future__ import annotations

from unittest import mock

import pytest

from maatml.device import get_profile
from maatml.training.load import from_pretrained_kwargs
from maatml.training.sft_config import QuantizationSettings, SFTTrainConfig


def test_quantization_settings_parse() -> None:
    cfg = SFTTrainConfig(
        quantization={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bf16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
    )
    assert cfg.quantization is not None
    assert cfg.quantization.enabled()
    assert cfg.quantization.load_in_4bit is True
    assert cfg.quantization.bnb_4bit_quant_type == "nf4"


def test_quantization_disabled_by_default() -> None:
    cfg = SFTTrainConfig()
    assert cfg.quantization is None


def test_from_pretrained_kwargs_rejects_quant_on_mps() -> None:
    q = QuantizationSettings(load_in_4bit=True)
    with pytest.raises(RuntimeError, match="CUDA"):
        from_pretrained_kwargs(get_profile("mps"), quantization=q)


def test_from_pretrained_kwargs_quant_builds_when_bnb_present() -> None:
    q = QuantizationSettings(load_in_4bit=True)
    fake_cfg = object()
    with mock.patch("maatml.training.load.build_bitsandbytes_config", return_value=fake_cfg):
        kwargs = from_pretrained_kwargs(get_profile("cuda"), quantization=q)
    assert kwargs["quantization_config"] is fake_cfg
    assert kwargs.get("device_map") == "auto"


def test_from_pretrained_kwargs_attn_passthrough() -> None:
    kwargs = from_pretrained_kwargs(
        get_profile("cpu"),
        precision="bf16",
        attn_implementation="sdpa",
    )
    assert kwargs["attn_implementation"] == "sdpa"
    assert "dtype" not in kwargs  # fp32_master
    assert "torch_dtype" not in kwargs


@pytest.mark.parametrize("device,frozen_base", [("cuda", False), ("cpu", True), ("mps", True)])
def test_from_pretrained_kwargs_native_dtype(device: str, frozen_base: bool) -> None:
    pytest.importorskip("torch")
    import torch

    kwargs = from_pretrained_kwargs(get_profile(device), precision="bf16", frozen_base=frozen_base)
    assert kwargs["dtype"] is torch.bfloat16
    assert "torch_dtype" not in kwargs
