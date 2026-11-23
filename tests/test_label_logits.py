"""`training.logits: labels` gives the default causal-LM loss without full logits."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from maatml.config import ModelDefinition  # noqa: E402
from maatml.training.label_logits import (  # noqa: E402
    accepts_logits_to_keep,
    label_logits_loss,
    label_logits_trainer,
    label_positions,
)
from maatml.training.sft_config import SFTTrainConfig  # noqa: E402


def _tiny_llama(vocab: int = 32):
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=vocab,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return transformers.LlamaForCausalLM(cfg).float()


def _batch():
    torch.manual_seed(1)
    input_ids = torch.randint(0, 32, (2, 12))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, -3:] = 0
    labels = input_ids.clone()
    labels[0, :7] = -100
    labels[1, :4] = -100
    labels[1, 6:] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _grads(model):
    return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]


def test_label_positions_predict_the_next_labelled_token() -> None:
    labels = torch.tensor([[-100, -100, 5, 6, -100], [-100, 7, -100, -100, 8]])
    assert label_positions(labels).tolist() == [0, 1, 2, 3]
    assert label_positions(torch.full((1, 4), -100)).tolist() == []


@pytest.mark.parametrize("num_items", [None, 9])
def test_loss_and_gradients_match_the_full_logits_loss(num_items) -> None:
    batch = _batch()
    reference = _tiny_llama()
    kwargs = {} if num_items is None else {"num_items_in_batch": torch.tensor(num_items)}
    expected = reference(**batch, **kwargs).loss
    expected.backward()

    model = _tiny_llama()
    loss, outputs = label_logits_loss(
        model, dict(batch), None if num_items is None else torch.tensor(num_items)
    )
    loss.backward()

    assert torch.allclose(loss, expected, atol=1e-6)
    assert outputs.logits.shape[1] == label_positions(batch["labels"]).numel()
    for got, want in zip(_grads(model), _grads(reference), strict=True):
        assert torch.allclose(got, want, atol=1e-6)


def test_a_batch_with_no_labels_steps_with_zero_loss() -> None:
    batch = _batch()
    batch["labels"] = torch.full_like(batch["labels"], -100)
    model = _tiny_llama()
    loss, _ = label_logits_loss(model, batch)
    loss.backward()
    assert loss.item() == 0.0


def test_a_peft_wrapped_model_is_recognised_and_trains() -> None:
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(
        _tiny_llama(),
        peft.LoraConfig(
            r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
        ),
    )
    assert accepts_logits_to_keep(model)
    loss, _ = label_logits_loss(model, _batch())
    loss.backward()
    assert any(p.grad is not None for n, p in model.named_parameters() if "lora_" in n)


def test_a_forward_without_logits_to_keep_is_detected() -> None:
    class NoKeep(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None):
            return input_ids

    assert not accepts_logits_to_keep(NoKeep())


def test_the_trainer_subclass_returns_the_label_loss() -> None:
    class Base:
        pass

    trainer = label_logits_trainer(Base)()
    model = _tiny_llama()
    batch = _batch()
    loss, outputs = trainer.compute_loss(model, dict(batch), return_outputs=True)
    expected, _ = label_logits_loss(_tiny_llama(), dict(batch))
    assert torch.allclose(loss, expected, atol=1e-6)
    assert outputs.logits.shape[1] == label_positions(batch["labels"]).numel()


def test_config_accepts_only_known_logits_modes() -> None:
    assert SFTTrainConfig().logits == "all"
    assert SFTTrainConfig(logits="labels").logits == "labels"
    with pytest.raises(ValueError, match="training.logits must be one of all, labels"):
        SFTTrainConfig(logits="assistant")


def test_train_sft_with_label_logits_runs_a_step(tmp_path, monkeypatch) -> None:
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from maatml.training import sft_base

    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2, "b": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", bos_token="a", eos_token="b"
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    tiny = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_embd=8, n_layer=1, n_head=1, n_positions=32))
    monkeypatch.setattr(sft_base.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(sft_base.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: tiny)

    seen: list[type] = []
    real_init = transformers.Trainer.__init__

    def record_init(self, *args, **kwargs):
        seen.append(type(self))
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(transformers.Trainer, "__init__", record_init)

    model = ModelDefinition(
        name="label-logits-sft",
        model_id="label-logits-sft",
        architecture="causal_sft",
        version="0.1.0",
        training={
            "model_id": "local-test-model",
            "max_input_tokens": 24,
            "precision": "fp32",
            "lora": {"enabled": False},
            "logits": "labels",
            "max_steps": 1,
            "batch_size": 2,
            "grad_accum": 1,
            "save_strategy": "no",
        },
        dataset={"prompt_spec": "prompt.json"},
    )
    object.__setattr__(model, "model_dir", tmp_path)
    (tmp_path / "prompt.json").write_text(
        json.dumps({"system": "a", "user_template": "<<USER_REQUEST>>"})
    )
    model.prepared_dir.mkdir(parents=True)
    rows = [
        {"sample_id": "one", "request": "a b a", "expected_output": "b a"},
        {"sample_id": "two", "request": "b a", "expected_output": "a b b"},
    ]
    (model.prepared_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (model.prepared_dir / "val.jsonl").write_text("")

    result = sft_base.train_sft(model, device="cpu")

    assert seen and seen[0].__name__ == "LabelLogitsTrainer"
    assert result.train_runtime > 0
