"""Prepare an image+caption folder into train/val/test jsonl splits.

``format: image_caption_folder`` reads ``dataset.source_dir``: a flat folder
where every image sits beside a caption file of the same stem
(``00017.jpg`` + ``00017.txt``), the layout captioning tools already write.
Rows come out as ``{image, caption, sample_id}`` with image paths kept
folder-relative, so the prepared jsonl survives the model folder moving.

Splits are deterministic by content hash of the image file name, not by
position: re-running prepare over a grown corpus keeps every old row in the
split it was in, which is what keeps eval comparable across corpus growth.
"""

from __future__ import annotations

import hashlib
from typing import Any

from rich.console import Console

from ..config import ModelDefinition, get_dataset_cfg
from ..registry import register_format
from ..utils.io import write_jsonl

console = Console()

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _bucket(name: str, val_fraction: float, test_fraction: float) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    point = int.from_bytes(digest[:4], "big") / 2**32
    if point < test_fraction:
        return "test"
    if point < test_fraction + val_fraction:
        return "val"
    return "train"


@register_format("image_caption_folder")
def prepare_image_caption_folder(model_def: ModelDefinition) -> dict[str, Any]:
    ds_cfg = get_dataset_cfg(model_def)
    source = ds_cfg.get("source_dir")
    if not source:
        raise ValueError("format image_caption_folder needs dataset.source_dir in model.yml")
    source_dir = model_def.resolve(str(source))
    if not source_dir.is_dir():
        raise FileNotFoundError(f"dataset.source_dir does not exist: {source_dir}")

    caption_ext = str(ds_cfg.get("caption_extension", ".txt"))
    val_fraction = float(ds_cfg.get("val_fraction", 0.05))
    test_fraction = float(ds_cfg.get("test_fraction", 0.05))

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    missing_captions = 0
    for image in sorted(source_dir.iterdir()):
        if image.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        caption_path = image.with_suffix(caption_ext)
        if not caption_path.is_file():
            missing_captions += 1
            continue
        row = {
            "sample_id": image.stem,
            "image": str(image.resolve()),
            "caption": caption_path.read_text(encoding="utf-8").strip(),
        }
        splits[_bucket(image.name, val_fraction, test_fraction)].append(row)

    if not splits["train"]:
        raise ValueError(
            f"no trainable rows in {source_dir}: every image needs a {caption_ext} beside it "
            f"({missing_captions} images had none)"
        )
    # A tiny corpus can hash every row into train; eval needs something to
    # measure, so steal the tail rather than report empty splits.
    for name in ("val", "test"):
        if not splits[name]:
            splits[name].append(splits["train"].pop())

    for name, rows in splits.items():
        write_jsonl(model_def.prepared_dir / f"{name}.jsonl", rows)
    if missing_captions:
        console.print(f"[yellow]{missing_captions} images skipped: no caption file beside them[/]")

    counts = {name: len(rows) for name, rows in splits.items()}
    return {"split_counts": counts}
