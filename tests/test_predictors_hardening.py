"""Evaluation hardening: precision fidelity, base revision, batching, image-text loading."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from maatml.config import ModelDefinition
from maatml.evaluation.predictors import (
    CausalSFTPredictor,
    Seq2SeqPredictor,
    _is_image_text_checkpoint,
    _load_causal_or_image_text,
    _resolve_base_revision,
    _resolve_eval_dtype,
    _resolve_eval_precision,
)


def _model_def(tmp_path: Path, **training_extra: object) -> ModelDefinition:
    md = ModelDefinition(
        name="eval-hardening",
        model_id="eval-hardening",
        architecture="causal_sft",
        version="0.1.0",
        training={"model_id": "base/model", **training_extra},
    )
    object.__setattr__(md, "model_dir", tmp_path)
    return md


def test_resolve_eval_precision_prefers_run_metadata_then_spec_then_model_def(
    tmp_path: Path,
) -> None:
    md = _model_def(tmp_path, precision="fp16")
    meta_extra = {"extra": {"precision": "fp32"}, "spec": {"training": {"precision": "bf16"}}}
    assert _resolve_eval_precision(meta_extra, md) == "fp32"

    meta_smoke = {
        "extra": {"smoke": True},
        "spec": {"smoke": {"precision": "bf16"}, "training": {"precision": "fp16"}},
    }
    assert _resolve_eval_precision(meta_smoke, md) == "bf16"
    assert _resolve_eval_precision({}, md) == "fp16"
    assert _resolve_eval_precision({}, None) == "bf16"


def test_resolve_eval_dtype_honors_device_and_precision() -> None:
    torch = pytest.importorskip("torch")
    assert _resolve_eval_dtype(torch, SimpleNamespace(type="cuda"), "bf16") is torch.bfloat16
    assert _resolve_eval_dtype(torch, SimpleNamespace(type="mps"), "bf16") is torch.bfloat16
    assert _resolve_eval_dtype(torch, SimpleNamespace(type="cuda"), "fp16") is torch.float16
    assert _resolve_eval_dtype(torch, SimpleNamespace(type="cuda"), "fp32") is torch.float32
    assert _resolve_eval_dtype(torch, SimpleNamespace(type="cpu"), "bf16") is torch.float32


def test_resolve_base_revision_and_image_text_detection(tmp_path: Path) -> None:
    md = _model_def(tmp_path, model_revision="rev-model")
    assert _resolve_base_revision({"extra": {"model_revision": "rev-run"}}, md) == "rev-run"
    assert (
        _resolve_base_revision({"spec": {"training": {"model_revision": "rev-spec"}}}, md)
        == "rev-spec"
    )
    assert _resolve_base_revision({}, md) == "rev-model"
    assert _resolve_base_revision({}, None, {"revision": "rev-adapter"}) == "rev-adapter"

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.json").write_text('{"architectures":["Qwen3ForCausalLM"]}', encoding="utf-8")
    assert not _is_image_text_checkpoint(plain, {})

    vlm_meta = {"extra": {"processor_class": "Gemma4Processor"}}
    assert _is_image_text_checkpoint(plain, vlm_meta)

    vlm_proc = tmp_path / "vlm_proc"
    vlm_proc.mkdir()
    (vlm_proc / "processor_config.json").write_text("{}", encoding="utf-8")
    assert _is_image_text_checkpoint(vlm_proc, {})

    vlm_cfg = tmp_path / "vlm_cfg"
    vlm_cfg.mkdir()
    (vlm_cfg / "config.json").write_text('{"vision_config": {}}', encoding="utf-8")
    assert _is_image_text_checkpoint(vlm_cfg, {})


def test_load_causal_or_image_text_falls_back_on_value_error(monkeypatch) -> None:
    transformers = pytest.importorskip("transformers")

    calls: list[tuple[str, object, dict]] = []

    class _CausalStub:
        @classmethod
        def from_pretrained(cls, ref, **kwargs):
            calls.append(("causal", ref, kwargs))
            raise ValueError("Unrecognized configuration class for AutoModelForCausalLM")

    class _ImageTextStub:
        @classmethod
        def from_pretrained(cls, ref, **kwargs):
            calls.append(("image_text", ref, kwargs))
            return "vlm-model"

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _CausalStub)
    monkeypatch.setattr(transformers, "AutoModelForImageTextToText", _ImageTextStub)

    out = _load_causal_or_image_text(
        "google/gemma-4-E2B-it", dtype="bf16", revision="rev1", prefer_image_text=False
    )
    assert out == "vlm-model"
    assert calls[0] == ("causal", "google/gemma-4-E2B-it", {"dtype": "bf16", "revision": "rev1"})
    assert calls[1] == (
        "image_text",
        "google/gemma-4-E2B-it",
        {"dtype": "bf16", "revision": "rev1"},
    )


def test_causal_predictor_setup_passes_revision_precision_and_image_text(
    tmp_path: Path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    import transformers

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "prompt_spec.json").write_text(
        json.dumps({"system": "S", "user_template": "<<USER_REQUEST>>"}),
        encoding="utf-8",
    )
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "org/vlm-base"}),
        encoding="utf-8",
    )
    (ckpt / "run_metadata.json").write_text(
        json.dumps(
            {
                "extra": {
                    "base_model_id": "org/vlm-base",
                    "model_revision": "pinned-rev",
                    "precision": "bf16",
                    "processor_class": "VLMProcessor",
                }
            }
        ),
        encoding="utf-8",
    )

    seen: dict[str, object] = {}

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

    class _Proc:
        tokenizer = _Tok()

    class _FakeModel:
        def to(self, device):
            seen["to_device"] = device
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        transformers,
        "AutoProcessor",
        SimpleNamespace(from_pretrained=lambda *a, **kw: _Proc()),
    )
    monkeypatch.setattr(
        transformers,
        "AutoModelForImageTextToText",
        SimpleNamespace(
            from_pretrained=lambda ref, **kw: (seen.update({"base_ref": ref, "base_kw": kw}), "B")[
                1
            ]
        ),
    )

    fake_peft = SimpleNamespace(
        PeftModel=SimpleNamespace(
            from_pretrained=lambda base, adir: (
                seen.update({"peft_base": base, "peft_dir": Path(adir)}),
                _FakeModel(),
            )[1]
        )
    )
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    pred = CausalSFTPredictor()
    pred.setup(ckpt, device=SimpleNamespace(type="mps"))

    assert seen["base_ref"] == "org/vlm-base"
    assert seen["base_kw"] == {"dtype": torch.bfloat16, "revision": "pinned-rev"}
    assert seen["peft_base"] == "B"
    assert seen["peft_dir"] == ckpt


def test_causal_predictor_predict_batch_left_pads_and_uses_attention_mask() -> None:
    torch = pytest.importorskip("torch")

    captured: list[tuple[list[list[int]], list[list[int]], int]] = []

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "".join(chr(int(i)) for i in ids if int(i) != 0)

    class _BatchModel:
        def generate(self, input_ids, attention_mask, pad_token_id=None, **kwargs):
            del kwargs
            captured.append(
                (
                    input_ids.tolist(),
                    attention_mask.tolist(),
                    -1 if pad_token_id is None else int(pad_token_id),
                )
            )
            batch_n = input_ids.shape[0]
            replies = torch.tensor([[ord("x"), ord("y"), 0], [ord("z"), 0, 0]], dtype=torch.long)[
                :batch_n
            ]
            return torch.cat([input_ids, replies], dim=-1)

    pred = CausalSFTPredictor()
    pred._torch = torch
    pred._tokenizer = _Tok()
    pred._model = _BatchModel()
    pred._device = torch.device("cpu")
    pred._max_input_tokens = 4
    pred._render = lambda req, *_a, **_kw: [ord(c) for c in req]

    # Second row triggers left-tail truncation (5 > max_input_tokens=4).
    out = pred.predict_batch([{"request": "hi"}, {"request": "abcde"}])
    assert out == ["xy", "z"]
    assert pred.report_extras()["truncated_inputs"] == 1
    assert captured == [
        (
            [[0, 0, ord("h"), ord("i")], [ord("b"), ord("c"), ord("d"), ord("e")]],
            [[0, 0, 1, 1], [1, 1, 1, 1]],
            0,
        )
    ]


def test_seq2seq_predictor_batch_and_precision() -> None:
    torch = pytest.importorskip("torch")

    class _SeqTok:
        def __call__(
            self,
            sources,
            max_length=16,
            padding=False,
            truncation=True,
            return_tensors="pt",
        ):
            items = [sources] if isinstance(sources, str) else list(sources)
            encoded = [
                ([ord(c) for c in s][:max_length] if truncation else [ord(c) for c in s])
                for s in items
            ]
            if return_tensors is None:
                return {"input_ids": encoded[0] if isinstance(sources, str) else encoded}
            width = max(len(x) for x in encoded) if padding else len(encoded[0])
            ids = [x + [0] * (width - len(x)) for x in encoded]
            mask = [[1] * len(x) + [0] * (width - len(x)) for x in encoded]

            class _Batch(dict):
                def to(self, device):
                    del device
                    return self

            return _Batch(
                {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(mask, dtype=torch.long),
                }
            )

        def decode(self, seq, skip_special_tokens=True):
            del skip_special_tokens
            return "".join(chr(int(i)) for i in seq if int(i) != 0)

    class _SeqModel:
        def generate(self, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            return torch.tensor(
                [[ord('"'), ord("a"), ord('"'), ord(":"), ord("1")]] * input_ids.shape[0],
                dtype=torch.long,
            )

    pred = Seq2SeqPredictor()
    pred._torch = torch
    pred._tokenizer = _SeqTok()
    pred._model = _SeqModel()
    pred._device = torch.device("cpu")
    pred._repair_braces = True
    pred._max_input_tokens = 4

    out = pred.predict_batch([{"request": "ok"}, {"request": "toolong"}])
    assert out == ['{"a":1}', '{"a":1}']
    assert pred.report_extras()["brace_repairs"] == 2
    assert pred.report_extras()["truncated_inputs"] == 1
