"""LoRA target selection on a base whose towers share projection names.

A multimodal base repeats ``q_proj`` and friends in its vision and audio
towers, so a suffix list adapts those too. One regular expression, or the
full module names, selects the language model alone.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from peft.utils import get_peft_model_state_dict  # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from maatml.training.sft_base import _maybe_attach_lora  # noqa: E402
from maatml.training.sft_config import LoraSettings  # noqa: E402

LANGUAGE_ONLY = (
    r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
)
ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


class _Projections(torch.nn.Module):
    def __init__(self, names: tuple[str, ...]) -> None:
        super().__init__()
        for name in names:
            setattr(self, name, torch.nn.Linear(4, 4, bias=False))


class _Layer(torch.nn.Module):
    def __init__(self, attention: tuple[str, ...]) -> None:
        super().__init__()
        self.self_attn = _Projections(attention)
        self.mlp = _Projections(MLP)


class _Tower(torch.nn.Module):
    def __init__(self, *layers: _Layer) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


def _model() -> GPT2LMHeadModel:
    """A causal LM carrying the module names of a multimodal base."""
    model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_embd=8, n_layer=1, n_head=1, n_positions=16))
    model.model = torch.nn.Module()
    # The second language layer shares keys and values, as late layers of some bases do.
    model.model.language_model = _Tower(_Layer(ATTENTION), _Layer(("q_proj", "o_proj")))
    model.model.vision_tower = _Tower(_Layer(ATTENTION))
    model.model.audio_tower = _Tower(_Layer(ATTENTION))
    return model


def _adapted(target_modules) -> dict[str, tuple[int, ...]]:
    wrapped = _maybe_attach_lora(
        _model(), LoraSettings(r=2, alpha=4, target_modules=target_modules)
    )
    return {name: tuple(t.shape) for name, t in get_peft_model_state_dict(wrapped).items()}


def _language_modules() -> list[str]:
    names = [f"model.language_model.layers.0.self_attn.{p}" for p in ATTENTION]
    names += [f"model.language_model.layers.1.self_attn.{p}" for p in ("q_proj", "o_proj")]
    names += [f"model.language_model.layers.{i}.mlp.{p}" for i in (0, 1) for p in MLP]
    return names


def test_a_regular_expression_selects_the_language_model_only() -> None:
    tensors = _adapted(LANGUAGE_ONLY)
    modules = {name.split(".lora_")[0].removeprefix("base_model.model.") for name in tensors}
    assert modules == set(_language_modules())
    assert len(tensors) == 2 * len(_language_modules()) == 24
    assert all(shape in ((2, 4), (4, 2)) for shape in tensors.values())


def test_full_module_names_select_the_same_tensors() -> None:
    assert _adapted(_language_modules()) == _adapted(LANGUAGE_ONLY)


def test_a_suffix_list_also_adapts_the_other_towers() -> None:
    tensors = _adapted([*ATTENTION, *MLP])
    towers = {name.split("base_model.model.model.")[1].split(".")[0] for name in tensors}
    assert towers == {"language_model", "vision_tower", "audio_tower"}
    assert len(tensors) > len(_adapted(LANGUAGE_ONLY))


def test_an_expression_that_matches_nothing_fails() -> None:
    with pytest.raises(ValueError, match="[Tt]arget modules"):
        _adapted(r".*no_such_tower\.layers\.\d+\.q_proj")
