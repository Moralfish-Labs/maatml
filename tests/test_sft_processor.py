"""Processor-aware plugins reuse native Trainer saves and run bookkeeping."""

import json

import pytest


def test_processor_batches_train_and_save_through_shared_driver(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
        ProcessorMixin,
    )

    from maatml.config import ModelDefinition
    from maatml.runs import list_runs
    from maatml.training.sft_base import train_sft

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2, "b": 3}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="a",
        eos_token="b",
    )

    class TinyProcessor(ProcessorMixin):
        attributes = ["tokenizer"]
        tokenizer_class = "PreTrainedTokenizerFast"

    processor = TinyProcessor(tokenizer=tokenizer)

    class TinyModel(GPT2LMHeadModel):
        def __init__(self):
            super().__init__(
                GPT2Config(vocab_size=4, n_embd=8, n_layer=1, n_head=1, n_positions=16)
            )
            self.visual = torch.nn.Linear(2, 2).requires_grad_(False)
            self.saw_pixels = False

        def forward(self, pixel_values=None, **kwargs):
            self.saw_pixels = pixel_values is not None
            return super().forward(**kwargs)

    model = TinyModel()
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *a, **kw: processor)
    monkeypatch.setattr(AutoModelForImageTextToText, "from_pretrained", lambda *a, **kw: model)
    definition = ModelDefinition(
        name="processor-test",
        model_id="processor-test",
        architecture="causal_sft",
        version="0.1.0",
        training={
            "model_id": "fixture",
            "precision": "fp32",
            "batch_size": 1,
            "grad_accum": 1,
            "max_steps": 1,
            "save_steps": 1,
            "disable_tqdm": True,
            "lora": {"enabled": False},
            "frozen_module_prefixes": ["visual"],
        },
        dataset={"prompt_spec": "prompt.json"},
    )
    object.__setattr__(definition, "model_dir", tmp_path)
    (tmp_path / "prompt.json").write_text("{}")
    definition.prepared_dir.mkdir(parents=True)
    (definition.prepared_dir / "train.jsonl").write_text('{"sample_id":"fixture"}\n')
    (definition.prepared_dir / "val.jsonl").write_text("")

    def factory(model_def, actual_processor, cfg):
        assert actual_processor is processor and model_def is definition and cfg.batch_size == 1

        def collate(rows):
            assert rows == [{"sample_id": "fixture"}]
            return {
                "input_ids": torch.tensor([[2, 3, 2]]),
                "labels": torch.tensor([[-100, 3, 2]]),
                "pixel_values": torch.ones(1, 2),
            }

        return collate

    result = train_sft(definition, device="cpu", data_collator_factory=factory)
    assert model.saw_pixels
    assert (result.out_dir / "processor_config.json").is_file()
    assert (result.out_dir / "checkpoint-1" / "processor_config.json").is_file()
    metadata = json.loads((result.out_dir / "run_metadata.json").read_text())
    assert metadata["extra"]["processor_class"] == "TinyProcessor"
    assert set(metadata["extra"]["frozen_component_sha256"]) == {"visual"}
    assert list_runs(definition)[0].status == "completed"
