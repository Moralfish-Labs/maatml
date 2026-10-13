"""Training-time guards: NaN abort, run provenance, tokenizer/model contract."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import ModelDefinition
from ..device import is_main_process
from ..utils.io import sha256_file, write_json


class NanGuardCallback:
    """Abort training when loss or grad_norm becomes non-finite.

    Instantiated lazily so importing this module does not require torch /
    transformers (keeps unit tests light). Use :func:`make_nan_guard_callback`
    or construct after ``from transformers import TrainerCallback``.
    """

    @staticmethod
    def create():
        from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

        class _NanGuardCallback(TrainerCallback):
            def on_log(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                logs: Optional[dict[str, float]] = None,
                **kwargs,
            ):
                if not logs:
                    return
                for key in ("loss", "grad_norm", "train_loss", "eval_loss"):
                    val = logs.get(key)
                    if val is None:
                        continue
                    try:
                        fval = float(val)
                    except (TypeError, ValueError):
                        continue
                    if fval != fval or fval in (float("inf"), float("-inf")):  # noqa: PLR0124
                        raise RuntimeError(
                            f"Non-finite {key}={val!r} at step "
                            f"{state.global_step}; aborting training"
                        )

        return _NanGuardCallback()


def make_nan_guard_callback():
    """Return a transformers ``TrainerCallback`` that aborts on non-finite loss."""
    return NanGuardCallback.create()


class CacheReleaseCallback:
    """Release the backend allocator's cache during training, not only after.

    On MPS the cache is never reclaimed on its own, so a run's footprint grows
    step over step until unified memory is exhausted and the machine pages.
    The cost is not a stall but a slide: a 1.7B LoRA measured 11.5 s/step at
    step 1 and 225 s/step by step 25, with swap climbing throughout.

    Releasing has its own cost, so it is periodic rather than per step, and
    ``empty_cache`` is a no-op on backends that manage their own.
    """

    @staticmethod
    def create(profile, every: int = 8):
        from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

        class _CacheReleaseCallback(TrainerCallback):
            def on_step_end(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs,
            ):
                if every > 0 and state.global_step % every == 0:
                    profile.empty_cache()

        return _CacheReleaseCallback()


def make_cache_release_callback(profile, every: int = 8):
    """Return a ``TrainerCallback`` that drops the allocator cache periodically."""
    return CacheReleaseCallback.create(profile, every)


def _git_sha(cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
            # Bounded: git on a pathological or network filesystem must not
            # hang a training run behind provenance capture.
            timeout=5,
        )
        return out.strip() or None
    except Exception:  # noqa: BLE001
        return None


def maatml_checkout_sha() -> Optional[str]:
    """SHA of maatml's own source checkout, or None when pip-installed.

    Asking git for HEAD from the package directory returns the SHA of whatever
    repository happens to contain it. For a pip install under a path that sits
    inside an unrelated repo, every commit there changed maatml's environment
    fingerprint and forced a full retrain, which is the opposite of the
    documented idempotence. Only a checkout whose root holds this very file at
    ``src/maatml/__init__.py`` counts.
    """
    pkg_dir = Path(__file__).resolve().parent.parent  # .../src/maatml
    try:
        toplevel = subprocess.check_output(
            ["git", "-C", str(pkg_dir), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:  # noqa: BLE001
        return None
    if not toplevel:
        return None
    expected = Path(toplevel).resolve() / "src" / "maatml" / "__init__.py"
    if not expected.is_file() or expected != pkg_dir / "__init__.py":
        return None
    return _git_sha(pkg_dir)


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return None


def write_run_metadata(
    out_dir: str | Path,
    model_def: ModelDefinition,
    dataset_paths: dict[str, str | Path],
    extra: Optional[dict[str, Any]] = None,
) -> Optional[Path]:
    """Write ``run_metadata.json`` with spec snapshot, hashes, and env provenance.

    Rank-0 only under multi-GPU; non-main ranks return ``None``.
    """
    if not is_main_process():
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = model_def.model_dump(mode="json", exclude={"model_dir"})
    spec_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    import hashlib

    spec_hash = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()

    ds_hashes: dict[str, str] = {}
    for key, path in dataset_paths.items():
        p = Path(path)
        if p.is_file():
            ds_hashes[key] = sha256_file(p)

    payload: dict[str, Any] = {
        "identity": model_def.identity,
        "model_id": model_def.model_id,
        "architecture": model_def.architecture,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(model_def.model_dir),
        "spec_hash": spec_hash,
        "spec": spec,
        "dataset_hashes": ds_hashes,
        "package_versions": {
            "maatml": _pkg_version("maatml"),
            "torch": _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
            "peft": _pkg_version("peft"),
            "safetensors": _pkg_version("safetensors"),
        },
    }
    if extra:
        payload["extra"] = extra

    return write_json(out_dir / "run_metadata.json", payload)


def ensure_tokenizer_model_contract(
    model: Any,
    tokenizer: Any,
    *,
    embedding_strategy: Optional[str] = None,
) -> None:
    """Align model embedding size with tokenizer vocab when they diverge.

    Strategies:
      - ``resize`` / ``reinit``: call ``model.resize_token_embeddings(len(tokenizer))``
      - ``reuse``: allow only when tokenizer vocab ≤ model vocab (no resize)
    """
    try:
        tok_size = len(tokenizer)
    except TypeError:
        tok_size = tokenizer.vocab_size

    model_vocab = None
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            model_vocab = int(emb.weight.shape[0])
    if model_vocab is None and hasattr(model, "config"):
        model_vocab = getattr(model.config, "vocab_size", None)
    if model_vocab is None:
        return
    model_vocab = int(model_vocab)

    if tok_size == model_vocab:
        return

    allowed = {"resize", "reinit", "reuse"}
    if embedding_strategy not in allowed:
        raise ValueError(
            f"Tokenizer vocab ({tok_size}) != model vocab ({model_vocab}). "
            f"Set training.embedding_strategy to one of {sorted(allowed)}."
        )

    if embedding_strategy == "reuse":
        if tok_size > model_vocab:
            raise ValueError(
                f"embedding_strategy=reuse requires tokenizer vocab ({tok_size}) "
                f"<= model vocab ({model_vocab})"
            )
        return

    # resize / reinit
    if hasattr(model, "resize_token_embeddings"):
        _refuse_resize_that_discards_output_head(model, tok_size, model_vocab)
        model.resize_token_embeddings(tok_size)
        realign_special_token_ids(model, tokenizer, tok_size)


def _has_untied_output_head(model: Any) -> bool:
    """Does the model carry an output projection separate from its embeddings?"""
    get_out = getattr(model, "get_output_embeddings", None)
    get_in = getattr(model, "get_input_embeddings", None)
    if not callable(get_out) or not callable(get_in):
        return False
    out, inp = get_out(), get_in()
    if out is None or inp is None:
        return False
    out_w, in_w = getattr(out, "weight", None), getattr(inp, "weight", None)
    if out_w is None or in_w is None:
        return False
    return out_w.data_ptr() != in_w.data_ptr()


def _refuse_resize_that_discards_output_head(model: Any, tok_size: int, model_vocab: int) -> None:
    """Block a shrink that would throw away a separately trained output head.

    ``resize_token_embeddings`` re-ties the output projection to the input
    embeddings on models that ship them untied (flan-t5), discarding trained
    weights and wrecking the logit scale. A shrink is never required anyway: an
    embedding matrix larger than the tokenizer's vocabulary just has unused
    rows, which ``reuse`` already permits.
    """
    if tok_size >= model_vocab:
        return
    if not _has_untied_output_head(model):
        return
    raise ValueError(
        f"embedding_strategy=resize would shrink the vocabulary "
        f"({model_vocab} -> {tok_size}) on a model whose output head is trained "
        "separately from its input embeddings. Resizing re-ties the two and "
        "discards the trained head, which silently destroys the pretrained "
        "model while training still reports success. The embedding matrix is "
        "already larger than the tokenizer needs, so set "
        "training.embedding_strategy: reuse instead."
    )


# Config fields holding a token id that must index into the embedding matrix.
_SPECIAL_ID_FIELDS = (
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "cls_token_id",
    "sep_token_id",
    "unk_token_id",
    "mask_token_id",
    "decoder_start_token_id",
)


def realign_special_token_ids(model: Any, tokenizer: Any, vocab_size: int) -> list[str]:
    """Point the config's special-token ids at the tokenizer's, post-resize.

    ``resize_token_embeddings`` leaves the config's token ids alone, so after a
    shrink they can sit outside the embedding matrix. The live model still
    trains; rebuilding it from that config does not.

    Returns the names of the fields that were changed.
    """
    config = getattr(model, "config", None)
    if config is None:
        return []

    changed: list[str] = []
    for field in _SPECIAL_ID_FIELDS:
        if not hasattr(config, field):
            continue
        current = getattr(config, field)
        if current is None:
            continue
        # The tokenizer is the authority on what these ids are now.
        replacement = getattr(tokenizer, field, None)
        if not isinstance(replacement, int) or replacement >= vocab_size:
            replacement = None
        if isinstance(current, int) and current < vocab_size and replacement is None:
            continue  # already in range and the tokenizer offers nothing better
        if replacement == current:
            continue
        setattr(config, field, replacement)
        changed.append(f"{field}: {current} -> {replacement}")

    pad = getattr(config, "pad_token_id", None)
    if isinstance(pad, int) and pad >= vocab_size:
        raise ValueError(
            f"pad_token_id={pad} is outside the resized vocab ({vocab_size}) and "
            "the tokenizer does not define one. Give the tokenizer a pad token "
            "before training, or the checkpoint cannot be reloaded."
        )

    # Generation config carries its own copies and is what generate() reads.
    gen = getattr(model, "generation_config", None)
    if gen is not None:
        for field in ("pad_token_id", "bos_token_id", "eos_token_id", "decoder_start_token_id"):
            if not hasattr(gen, field):
                continue
            value = getattr(gen, field)
            if isinstance(value, int) and value >= vocab_size:
                setattr(gen, field, getattr(config, field, None))
    return changed
