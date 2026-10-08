"""Registered evaluation predictors (model load + per-row generation).

Each predictor is a small class with ``setup(...)`` then ``predict(row) -> str``.
Asset resolution uses ``maatml.evaluation.harness.resolve_eval_asset``:
never a hardcoded repo-root fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..config import ModelDefinition, get_dataset_cfg
from ..device import resolve_device
from ..registry import TRANSFORMS, register_predictor
from ..utils.tokenizers import resolve_special_tokens as _tokenizer_specials
from .harness import resolve_eval_asset


def _count_truncation(tokenizer, text: str, max_len: int, encoded_len: int) -> bool:
    """Did ``max_len`` actually cut this input?

    Only pays for a second tokenization when the encoded length sits at the cap,
    so the common (untruncated) row costs nothing extra.
    """
    if not max_len or encoded_len < max_len:
        return False
    try:
        full = tokenizer(text, truncation=False, return_tensors=None)["input_ids"]
    except Exception:  # noqa: BLE001  a tokenizer that refuses is not a failure
        return False
    if full and isinstance(full[0], list):
        full = full[0]
    return len(full) > max_len


def _load_head_specs(checkpoint_dir: Path, model_def: Optional[ModelDefinition]) -> list[dict]:
    """Load head configs from run_metadata.json, falling back to model.yml."""
    meta_path = Path(checkpoint_dir) / "run_metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        heads = (meta.get("extra") or {}).get("heads") or meta.get("heads")
        if heads:
            return list(heads)
    if model_def is not None:
        from ..training.multi_head import parse_heads

        return [h.to_dict() for h in parse_heads(dict(model_def.training or {}))]
    return []


class MultiHeadClassifierPredictor:
    """Generic multi-head encoder predictor.

    Emits JSON ``{"<head>": {"label": ..., "confidence": ...}, ...}`` for
    classification heads and ``{"line": N|null, "confidence": ...}`` for
    ``line_pointer`` heads. Task-specific assembly (e.g. JclValidationResult)
    lives in example plugins.
    """

    def __init__(self) -> None:
        self._encoder: Any = None
        self._tokenizer: Any = None
        self._heads: dict[str, dict[str, Any]] = {}
        self._head_specs: list[dict] = []
        self._device: Any = None
        self._max_input_tokens = 1024
        self._text_transform = None
        self._request_field = "request"
        self._torch: Any = None
        self._truncated = 0

    def report_extras(self) -> dict[str, Any]:
        return {"truncated_inputs": self._truncated}

    def setup(
        self,
        checkpoint_dir: Path,
        *,
        model_def: Optional[ModelDefinition] = None,
        device: Any = "auto",
        max_input_tokens: int = 1024,
        schema_path: Optional[Path] = None,
        contracts_path: Optional[Path] = None,
        prompt_spec_path: Optional[Path] = None,
        **_kwargs: Any,
    ) -> None:
        del schema_path, contracts_path, prompt_spec_path
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModel, PreTrainedTokenizerFast

        self._torch = torch
        self._max_input_tokens = max_input_tokens
        self._device = device if hasattr(device, "type") else resolve_device(str(device))
        self._head_specs = _load_head_specs(checkpoint_dir, model_def)

        if model_def is not None:
            cfg = get_dataset_cfg(model_def)
            self._request_field = cfg.get("request_field") or cfg.get("raw_field") or "request"
            transform_name = cfg.get("text_transform")
            if transform_name:
                self._text_transform = TRANSFORMS.get(str(transform_name))

        checkpoint_dir = Path(checkpoint_dir)
        tokenizer_path = resolve_eval_asset(
            "tokenizer",
            model_def=model_def,
            checkpoint_dir=checkpoint_dir,
            filenames=("tokenizer.json",),
        )
        specials = _tokenizer_specials(Path(tokenizer_path))
        self._tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            model_max_length=max_input_tokens,
            **specials,
        )

        encoder = AutoModel.from_pretrained(checkpoint_dir).to(self._device)
        encoder.eval()
        self._encoder = encoder

        head_state = load_file(checkpoint_dir / "classifier_heads.safetensors")
        self._heads = {}
        for spec in self._head_specs:
            name = spec["name"]
            self._heads[name] = {
                "weight": head_state[f"heads.{name}.weight"].to(self._device),
                "bias": head_state[f"heads.{name}.bias"].to(self._device),
                "kind": spec.get("kind", "classification"),
                "labels": list(spec.get("labels") or []),
            }

    def _head_forward(self, name: str, x):
        h = self._heads[name]
        return self._torch.nn.functional.linear(x, h["weight"], h["bias"])

    def _first_error_line(self, line_logits, encoding, pre: str) -> Optional[int]:
        cls_per_tok = line_logits.squeeze(0).argmax(dim=-1).tolist()
        tokens = encoding.tokens()
        offsets = encoding.offsets
        for i, label in enumerate(cls_per_tok):
            if label != 1:
                continue
            tok = tokens[i] if i < len(tokens) else ""
            if tok.startswith("<") and tok.endswith(">"):
                continue
            char_offset = offsets[i][0] if i < len(offsets) else 0
            # Count line markers if present; else count newlines.
            prefix = pre[: min(char_offset, len(pre))]
            if "<COL1>" in pre:
                return max(1, prefix.count("<COL1>"))
            return max(1, prefix.count("\n") + 1)
        return None

    def predict_heads(self, row: dict) -> dict[str, Any]:
        """Return structured per-head predictions (not JSON-serialised)."""
        assert self._encoder is not None and self._tokenizer is not None
        torch = self._torch
        text = row.get(self._request_field, "") or ""
        pre = self._text_transform(text) if self._text_transform else text
        encoding = self._tokenizer(
            pre,
            max_length=self._max_input_tokens,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors=None,
        )
        if _count_truncation(
            self._tokenizer, pre, self._max_input_tokens, len(encoding["input_ids"])
        ):
            self._truncated += 1
        input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long, device=self._device)
        attention_mask = torch.tensor(
            [encoding["attention_mask"]], dtype=torch.long, device=self._device
        )

        class _Enc:
            def __init__(self, e):
                self._e = e

            def tokens(self):
                return self._e.tokens(0) if hasattr(self._e, "tokens") else []

            @property
            def offsets(self):
                return self._e["offset_mapping"]

        with torch.inference_mode():
            out = self._encoder(input_ids=input_ids, attention_mask=attention_mask)
            seq = out.last_hidden_state
            pooled = seq[:, 0, :]

        result: dict[str, Any] = {}
        for name, h in self._heads.items():
            if h["kind"] == "line_pointer":
                line_logits = self._head_forward(name, seq)
                probs = torch.softmax(line_logits.squeeze(0), dim=-1)
                conf = float(probs[:, 1].max().item()) if probs.numel() else 0.0
                line_no = self._first_error_line(line_logits, _Enc(encoding), pre)
                result[name] = {"line": line_no, "confidence": conf}
            else:
                logits = self._head_forward(name, pooled)
                probs = torch.softmax(logits.squeeze(0), dim=-1)
                idx = int(probs.argmax().item())
                conf = float(probs[idx].item())
                labels = h["labels"]
                label = labels[idx] if idx < len(labels) else str(idx)
                result[name] = {"label": label, "confidence": conf, "index": idx}
        return result

    def predict(self, row: dict) -> str:
        return json.dumps(self.predict_heads(row), ensure_ascii=False)


class Seq2SeqPredictor:
    """Encoder-decoder generate (T5 / flan-t5) with optional task prefix.

    ``evaluation.repair_braces: true`` re-adds the outer ``{`` / ``}`` that T5
    SentencePiece tokenizers map to ``<unk>``. It is off by default: repairing
    output before scoring turns a pass rate into a measurement of maatml's
    repair, so it must be opted into and it is counted in ``Report.extras``.
    """

    def __init__(self, task_prefix: str = "") -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._max_input_tokens = 1024
        self._task_prefix = task_prefix
        self._max_new_tokens = 512
        self._request_field = "request"
        self._repair_braces = False
        self._repairs = 0
        self._truncated = 0

    def report_extras(self) -> dict[str, Any]:
        return {
            "truncated_inputs": self._truncated,
            "brace_repairs": self._repairs,
            "repair_braces": self._repair_braces,
        }

    def setup(
        self,
        checkpoint_dir: Path,
        *,
        model_def: Optional[ModelDefinition] = None,
        device: Any = "auto",
        max_input_tokens: int = 1024,
        schema_path: Optional[Path] = None,
        contracts_path: Optional[Path] = None,
        prompt_spec_path: Optional[Path] = None,
        **_kwargs: Any,
    ) -> None:
        del schema_path, contracts_path, prompt_spec_path
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self._task_prefix = ""
        self._max_input_tokens = max_input_tokens
        self._device = device if hasattr(device, "type") else resolve_device(str(device))
        if model_def is not None:
            cfg = get_dataset_cfg(model_def)
            self._task_prefix = cfg.get("source_prefix") or ""
            self._request_field = cfg.get("request_field") or cfg.get("raw_field") or "request"
            gen = (model_def.training or {}).get("generation") or {}
            if "max_new_tokens" in gen:
                self._max_new_tokens = int(gen["max_new_tokens"])
            self._repair_braces = bool((model_def.evaluation or {}).get("repair_braces", False))

        checkpoint_dir = Path(checkpoint_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        inference_dtype = torch.float16 if self._device.type in ("mps", "cuda") else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir, dtype=inference_dtype).to(
            self._device
        )
        model.eval()
        self._model = model

    def predict(self, row: dict) -> str:
        assert self._model is not None and self._tokenizer is not None
        torch = self._torch
        source = self._task_prefix + (row.get(self._request_field, "") or "")
        enc = self._tokenizer(
            source,
            max_length=self._max_input_tokens,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)
        if _count_truncation(
            self._tokenizer,
            source,
            self._max_input_tokens,
            int(enc["input_ids"].shape[-1]),
        ):
            self._truncated += 1

        with torch.inference_mode():
            generated = self._model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=self._max_new_tokens,
                num_beams=1,
                do_sample=False,
            )

        gen_text = self._tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        if not self._repair_braces or not gen_text:
            return gen_text
        # T5 SentencePiece maps `{`/`}` to <unk>; skip_special_tokens strips them.
        repaired = gen_text
        if not repaired.startswith("{"):
            repaired = "{" + repaired
        if not repaired.endswith("}"):
            repaired = repaired + "}"
        if repaired != gen_text:
            self._repairs += 1
        return repaired


class CausalSFTPredictor:
    """Causal LM generate with chat template / prompt_spec."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._max_input_tokens = 2048
        self._max_new_tokens = 512
        self._prompt_spec: dict = {}
        self._user_placeholder = "<<USER_REQUEST>>"
        self._request_field = "request"
        self._truncated = 0

    def report_extras(self) -> dict[str, Any]:
        return {"truncated_inputs": self._truncated}

    def setup(
        self,
        checkpoint_dir: Path,
        *,
        model_def: Optional[ModelDefinition] = None,
        device: Any = "auto",
        max_input_tokens: int = 2048,
        schema_path: Optional[Path] = None,
        contracts_path: Optional[Path] = None,
        prompt_spec_path: Optional[Path] = None,
        **_kwargs: Any,
    ) -> None:
        del schema_path, contracts_path
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..training.sft_base import render_chat_prompt, render_inference_prompt
        from ..utils.io import read_json

        self._torch = torch
        self._render = render_inference_prompt
        self._render_chat = render_chat_prompt
        self._max_input_tokens = max_input_tokens
        self._device = device if hasattr(device, "type") else resolve_device(str(device))

        checkpoint_dir = Path(checkpoint_dir)
        if prompt_spec_path is None:
            prompt_spec_path = resolve_eval_asset(
                "prompt_spec",
                model_def=model_def,
                checkpoint_dir=checkpoint_dir,
                filenames=("prompt_spec.json",),
            )
        self._prompt_spec = read_json(prompt_spec_path)

        if model_def is not None:
            cfg = get_dataset_cfg(model_def)
            self._request_field = cfg.get("request_field") or cfg.get("raw_field") or "request"
            self._user_placeholder = cfg.get("user_placeholder") or "<<USER_REQUEST>>"
            gen = (model_def.training or {}).get("generation") or {}
            if "max_new_tokens" in gen:
                self._max_new_tokens = int(gen["max_new_tokens"])

        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        inference_dtype = torch.float16 if self._device.type in ("mps", "cuda") else torch.float32
        adapter_cfg = checkpoint_dir / "adapter_config.json"
        adapter_subdir = checkpoint_dir / "adapter" / "adapter_config.json"
        if adapter_cfg.is_file() or adapter_subdir.is_file():
            from peft import PeftModel

            adapter_dir = checkpoint_dir if adapter_cfg.is_file() else checkpoint_dir / "adapter"
            base_id = None
            meta_path = checkpoint_dir / "run_metadata.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                extra = meta.get("extra") or {}
                base_id = extra.get("base_model_id") or (meta.get("spec") or {}).get("base_model")
            if base_id is None and model_def is not None:
                base_id = (model_def.training or {}).get("model_id") or model_def.base_model
            if not base_id:
                # PEFT adapter_config.json stores base_model_name_or_path.
                acfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
                base_id = acfg.get("base_model_name_or_path")
            if not base_id:
                raise FileNotFoundError(
                    f"Adapter checkpoint at {adapter_dir} but no base model id found "
                    "(set run_metadata.extra.base_model_id or training.model_id)"
                )
            base = AutoModelForCausalLM.from_pretrained(base_id, dtype=inference_dtype)
            model = PeftModel.from_pretrained(base, adapter_dir).to(self._device)
        else:
            model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, dtype=inference_dtype).to(
                self._device
            )
        model.eval()
        self._model = model

    def _uses_messages(self, row: dict) -> bool:
        """A ``messages`` row is rendered from its own turns when it carries
        tools or has no request field; request rows keep the prompt_spec path."""
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            return False
        return bool(row.get("tools")) or self._request_field not in row

    def prompt_ids(self, row: dict) -> list[int]:
        """Prompt token ids for ``row``, rendered the way training rendered it:
        a tool row gets its ``tools`` and ``chat_template_kwargs``."""
        if self._uses_messages(row):
            return self._render_chat(row, self._tokenizer)
        return self._render(
            row[self._request_field],
            self._prompt_spec,
            self._tokenizer,
            user_placeholder=self._user_placeholder,
        )

    def predict_structured(self, row: dict) -> dict[str, Any]:
        """``predict`` plus the output parsed into prose and ``tool_calls``."""
        from ..utils.chat import parse_tool_calls

        raw = self.predict(row)
        parsed = parse_tool_calls(raw)
        return {
            "raw": raw,
            "content": parsed.content,
            "tool_calls": parsed.tool_calls,
            "errors": parsed.errors,
        }

    def predict(self, row: dict) -> str:
        assert self._model is not None and self._tokenizer is not None
        torch = self._torch
        input_ids = self.prompt_ids(row)
        if self._max_input_tokens and len(input_ids) > self._max_input_tokens:
            self._truncated += 1
            input_ids = input_ids[-self._max_input_tokens :]
        tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
        with torch.inference_mode():
            generated = self._model.generate(
                input_ids=tensor,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        new_tokens = generated[0, tensor.shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


register_predictor("multi_head_classifier")(MultiHeadClassifierPredictor)
register_predictor("classifier")(MultiHeadClassifierPredictor)
register_predictor("seq2seq")(Seq2SeqPredictor)
register_predictor("causal_sft")(CausalSFTPredictor)
