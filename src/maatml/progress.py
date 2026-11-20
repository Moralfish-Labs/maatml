"""Atomic job-progress reporter aligned with Flow's ``JobProgress`` schema."""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .device import is_main_process
from .utils.io import write_json_atomic

PROGRESS_FILE_ENV = "FLOW_JOB_PROGRESS_FILE"

_PREFERRED_METRICS = ("eval_loss", "loss", "train_loss")
_SECONDARY_KEYS = frozenset(
    {
        "epoch",
        "grad_norm",
        "learning_rate",
        "total_flos",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
    }
)


def resolve_progress_file(explicit: Optional[str | Path] = None) -> Optional[Path]:
    """Return ``explicit`` progress path, else ``$FLOW_JOB_PROGRESS_FILE`` if set."""
    if explicit is not None:
        text = str(explicit).strip()
        if text:
            return Path(text)
    raw_env = os.environ.get(PROGRESS_FILE_ENV, "").strip()
    return Path(raw_env) if raw_env else None


def _utc_rfc3339_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    val = float(value)
    return val if math.isfinite(val) else None


def _nonneg_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    fval = float(value)
    if not math.isfinite(fval) or fval < 0:
        return None
    return int(fval)


def pick_progress_metric(logs: Optional[dict[str, Any]]) -> tuple[Optional[str], Optional[float]]:
    """Extract one ``(metric_name, finite_value)`` pair from Trainer logs."""
    if not logs:
        return None, None
    for key in _PREFERRED_METRICS:
        if key in logs:
            val = _finite_float(logs[key])
            if val is not None:
                return key, val
    for key in sorted(logs):
        if key in _SECONDARY_KEYS:
            continue
        val = _finite_float(logs[key])
        if val is not None:
            return str(key), val
    for key in sorted(logs):
        if key == "epoch":
            continue
        val = _finite_float(logs[key])
        if val is not None:
            return str(key), val
    return None, None


class JobProgressReporter:
    """Write atomic camelCase ``JobProgress`` snapshots for external supervisors.

    Only rank-0 / single-process writers touch the progress file so multi-GPU
    ranks never race on the target path.
    """

    def __init__(
        self,
        path: Optional[str | Path],
        *,
        run_id: Optional[str] = None,
        artifact_uri: Optional[str] = None,
    ) -> None:
        self.path: Optional[Path] = Path(path) if path is not None else None
        self.run_id = run_id
        self.artifact_uri = artifact_uri
        self._last_checkpoint_at: Optional[float] = None
        self._last_stage: Optional[str] = None
        self._last_epoch: Optional[int] = None
        self._last_step: Optional[int] = None
        self._last_metric_name: Optional[str] = None
        self._last_metric_value: Optional[float] = None

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def record_checkpoint(self, *, at: Optional[float] = None) -> None:
        self._last_checkpoint_at = time.monotonic() if at is None else float(at)

    def checkpoint_age_secs(self, *, now: Optional[float] = None) -> Optional[int]:
        if self._last_checkpoint_at is None:
            return None
        current = time.monotonic() if now is None else float(now)
        return max(0, int(current - self._last_checkpoint_at))

    def snapshot(
        self,
        *,
        stage: Optional[str] = None,
        epoch: Optional[int | float] = None,
        step: Optional[int] = None,
        run_id: Optional[str] = None,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        artifact_uri: Optional[str] = None,
    ) -> dict[str, Any]:
        """Merge provided fields into the reporter state and return the wire dict."""
        if stage is not None:
            self._last_stage = str(stage)
        if run_id is not None:
            self.run_id = run_id
        if artifact_uri is not None:
            self.artifact_uri = artifact_uri

        parsed_epoch = _nonneg_int(epoch)
        if parsed_epoch is not None:
            self._last_epoch = parsed_epoch

        parsed_step = _nonneg_int(step)
        if parsed_step is not None:
            self._last_step = parsed_step

        if metric_name is not None and metric_value is not None:
            parsed_val = _finite_float(metric_value)
            if parsed_val is not None:
                self._last_metric_name = str(metric_name)
                self._last_metric_value = parsed_val

        payload: dict[str, Any] = {"heartbeatAt": _utc_rfc3339_now()}
        if self._last_stage is not None:
            payload["stage"] = self._last_stage
        if self._last_epoch is not None:
            payload["epoch"] = self._last_epoch
        if self._last_step is not None:
            payload["step"] = self._last_step
        if self.run_id is not None:
            payload["runId"] = self.run_id
        if self._last_metric_name is not None and self._last_metric_value is not None:
            payload["metricName"] = self._last_metric_name
            payload["metricValue"] = self._last_metric_value
        age = self.checkpoint_age_secs()
        if age is not None:
            payload["checkpointAgeSecs"] = age
        if self.artifact_uri is not None:
            payload["artifactUri"] = self.artifact_uri
        return payload

    def update(
        self,
        *,
        stage: Optional[str] = None,
        epoch: Optional[int | float] = None,
        step: Optional[int] = None,
        run_id: Optional[str] = None,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        artifact_uri: Optional[str] = None,
    ) -> Optional[Path]:
        """Update state and atomically write ``JobProgress`` JSON if enabled."""
        payload = self.snapshot(
            stage=stage,
            epoch=epoch,
            step=step,
            run_id=run_id,
            metric_name=metric_name,
            metric_value=metric_value,
            artifact_uri=artifact_uri,
        )
        if self.path is None or not is_main_process():
            return None
        return write_json_atomic(self.path, payload, sort_keys=True)


def make_progress_callback(reporter: JobProgressReporter) -> Any:
    """Return a transformers ``TrainerCallback`` driving ``reporter``."""
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    class _JobProgressCallback(TrainerCallback):
        def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ) -> None:
            del args, control, kwargs
            reporter.update(stage="train", epoch=state.epoch, step=state.global_step)

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: Optional[dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            del args, control, kwargs
            metric_name, metric_value = pick_progress_metric(logs)
            reporter.update(
                stage="train",
                epoch=state.epoch,
                step=state.global_step,
                metric_name=metric_name,
                metric_value=metric_value,
            )

        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ) -> None:
            del args, control, kwargs
            reporter.record_checkpoint()
            reporter.update(stage="checkpoint", epoch=state.epoch, step=state.global_step)

        def on_evaluate(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            metrics: Optional[dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            del args, control, kwargs
            metric_name, metric_value = pick_progress_metric(metrics)
            reporter.update(
                stage="evaluate",
                epoch=state.epoch,
                step=state.global_step,
                metric_name=metric_name,
                metric_value=metric_value,
            )

    return _JobProgressCallback()
