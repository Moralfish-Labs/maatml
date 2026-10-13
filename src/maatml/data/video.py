"""Extract frames from a video for ``maatml ingest --video``.

Core stays format-agnostic: a sidecar JSONL names which frames to keep and
what the gold target is. ffmpeg does the decode. Annotation dialects (MEVA
KPF, COCO VID, …) stay in model-folder plugins that write the sidecar.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional


def resolve_frame_index(row: dict[str, Any], *, fps: Optional[float] = None) -> int:
    """Return a 0-based frame index from a sidecar row.

    Accepts ``frame`` (int), ``timestamp_ms``, or ``t`` (seconds). Timestamp
    conversion uses ``fps`` when given, otherwise assumes 30.
    """
    if "frame" in row and row["frame"] is not None:
        idx = int(row["frame"])
        if idx < 0:
            raise ValueError(f"frame index must be >= 0, got {idx}")
        return idx
    rate = float(fps) if fps is not None and fps > 0 else 30.0
    if row.get("timestamp_ms") is not None:
        return max(0, int(round(float(row["timestamp_ms"]) / 1000.0 * rate)))
    if row.get("t") is not None:
        return max(0, int(round(float(row["t"]) * rate)))
    raise ValueError(
        "video sidecar row needs frame, timestamp_ms, or t so a frame can be extracted"
    )


def extract_video_frame(
    video: str | Path,
    out_path: str | Path,
    *,
    frame: int,
) -> Path:
    """Write one PNG frame with ffmpeg. Raises FileNotFoundError / RuntimeError."""
    video = Path(video)
    out_path = Path(out_path)
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError(
            "ffmpeg is required for maatml ingest --video; install it and retry"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select=eq(n\\,{frame})",
        "-vframes",
        "1",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "").strip() or str(exc)
        raise RuntimeError(f"ffmpeg failed extracting frame {frame} from {video}: {err}") from exc
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg wrote no frame {frame} from {video} → {out_path}")
    return out_path


def materialize_video_rows(
    rows: list[dict[str, Any]],
    video: str | Path,
    *,
    images_dir: Path,
    images_rel: str,
    request_field: str,
    extract_fn: Any = None,
    fps: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Copy sidecar rows and point ``request_field`` at extracted PNGs."""
    grab = extract_fn or extract_video_frame
    video = Path(video)
    stem = video.stem
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    rel_root = images_rel.rstrip("/")
    out: list[dict[str, Any]] = []
    for row in rows:
        sample = dict(row)
        idx = resolve_frame_index(sample, fps=fps)
        name = f"{stem}-f{idx:06d}.png"
        dest = images_dir / name
        grab(video, dest, frame=idx)
        sample[request_field] = f"{rel_root}/{name}"
        sample.setdefault("sample_id", f"{stem}-f{idx:06d}")
        sample.setdefault("family", stem)
        sample.setdefault("source", f"ingest:video:{video.name}")
        out.append(sample)
    return out
