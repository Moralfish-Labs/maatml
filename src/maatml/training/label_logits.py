"""Causal-LM loss scored only at the positions that predict a trained token.

A causal LM's default loss scores every position against the whole
vocabulary, then masks all but the labelled ones. With a large vocabulary and
a long prompt that sequence-by-vocabulary tensor dominates training memory
(for a 262,144-entry vocabulary, about 1 MiB per position in fp32). Here the
model is asked only for the logits of positions whose next token carries a
label (``logits_to_keep`` with a position tensor), and the summed cross
entropy is divided exactly as transformers' ``ForCausalLMLoss`` divides it, so
the loss and its gradients are unchanged.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def label_positions(labels: torch.Tensor) -> torch.Tensor:
    """Positions ``p`` whose logits predict a labelled token ``labels[:, p + 1]``.

    Taken across the batch, so a padded batch scores the union and the rows
    without a label at a kept position contribute nothing.
    """
    targets = labels[:, 1:]
    return (targets != IGNORE_INDEX).any(dim=0).nonzero(as_tuple=True)[0]


def accepts_logits_to_keep(model: Any) -> bool:
    """Whether the model's forward (under any PEFT wrapper) takes ``logits_to_keep``."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return "logits_to_keep" in inspect.signature(base.forward).parameters


def label_logits_loss(
    model: Any,
    inputs: dict[str, Any],
    num_items_in_batch: Optional[torch.Tensor | int] = None,
) -> tuple[torch.Tensor, Any]:
    """The causal-LM loss over labelled positions, and the model outputs.

    ``num_items_in_batch`` is the accumulation-wide count of labelled targets
    the Trainer passes; without it the loss is the mean over this batch's
    labelled targets, as the default loss is.
    """
    labels = inputs["labels"]
    keep = label_positions(labels)
    model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
    outputs = model(**model_inputs, logits_to_keep=keep)
    logits = outputs.logits
    targets = labels[:, 1:].index_select(1, keep)
    if keep.numel() == 0:
        # Keeps the graph connected so a fully masked batch steps cleanly.
        return logits.sum() * 0.0, outputs
    total = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1).to(logits.device),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    if num_items_in_batch is None:
        count = (targets != IGNORE_INDEX).sum().clamp(min=1)
    else:
        count = num_items_in_batch
    if torch.is_tensor(count):
        count = count.to(total.device)
    return total / count, outputs


def label_logits_trainer(base: type) -> type:
    """A ``base`` Trainer subclass whose loss uses :func:`label_logits_loss`."""

    class LabelLogitsTrainer(base):
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs
        ):
            loss, outputs = label_logits_loss(model, inputs, num_items_in_batch)
            return (loss, outputs) if return_outputs else loss

    return LabelLogitsTrainer
