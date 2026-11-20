"""CPU-safe GRPO configuration, group advantages, environment reward bridge,
and promotion guard (no torch / transformers import).

Design invariants:
  - Rewards are ALWAYS computed by an environment implementing
    ``maatml.rollout_env/1`` with the ``free_text`` capability (in-process or
    over stdio JSON lines), never by the learner.
  - Training anchors KL divergence to an explicit reference checkpoint (the
    last promoted checkpoint, or the initial model snapshot when unset).
  - Promotion requires held-out non-regression and improved task completion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..rollout.protocol import (
    FREE_TEXT_CAPABILITY,
    FreeTextEnvironment,
    RolloutEnvError,
    Verification,
)
from .sft_config import LoraSettings, QuantizationSettings, validate_precision


class GRPOTrainConfig(BaseModel):
    """Configuration for Group-Relative Policy Optimization (GRPO)."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = "Qwen/Qwen3-0.6B"
    # Anchor checkpoint (last promoted checkpoint) for KL regularization.
    # Defaults to a frozen copy of `model_id` at step 0 when omitted.
    anchor_checkpoint: Optional[str] = None
    # Environment plugin name (in ENVIRONMENTS) or stdio command argv list.
    environment: Union[str, list[str]] = "arithmetic_world"
    environment_options: dict[str, Any] = Field(default_factory=dict)
    task_query: dict[str, Any] = Field(default_factory=dict)

    # GRPO rollout + objective knobs.
    group_size: int = Field(default=4, ge=2)
    kl_beta: float = Field(default=0.04, ge=0.0)
    clip_eps: float = Field(default=0.2, gt=0.0, lt=1.0)
    max_input_tokens: int = Field(default=2048, gt=0)
    max_new_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.8, gt=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)

    # Optimization knobs.
    batch_size: int = Field(default=2, gt=0)
    grad_accum: int = Field(default=4, gt=0)
    learning_rate: float = Field(default=1e-5, gt=0.0)
    epochs: float = Field(default=1.0, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float = Field(default=1.0, ge=0.0)
    max_steps: int = -1
    seed: int = 7331
    # Optional multi-seed list for seed-study GRPO runs.
    seeds: Optional[list[int]] = None
    precision: str = "bf16"
    lora: LoraSettings = Field(default_factory=LoraSettings)
    quantization: Optional[QuantizationSettings] = None
    attn_implementation: Optional[str] = None
    model_revision: Optional[str] = None
    logging_steps: int = Field(default=5, ge=1)
    save_steps: int = Field(default=50, ge=1)

    # Promotion guard thresholds.
    min_completion_gain: float = Field(default=0.0, ge=0.0)
    max_heldout_regression: float = Field(default=0.0, ge=0.0)

    @field_validator("precision")
    @classmethod
    def _check_precision(cls, v: str) -> str:
        return validate_precision(v)

    @field_validator("environment")
    @classmethod
    def _check_environment(cls, v: Union[str, list[str]]) -> Union[str, list[str]]:
        if isinstance(v, str) and not v.strip():
            raise ValueError("training.environment must be a non-empty name or command list")
        if isinstance(v, list) and (not v or not all(isinstance(a, str) and a for a in v)):
            raise ValueError("training.environment command list must contain non-empty strings")
        return v

    @field_validator("seeds")
    @classmethod
    def _check_seeds(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is not None and len(v) < 2:
            raise ValueError("training.seeds must contain at least 2 seeds when provided")
        return v


@dataclass
class GRPOTrainResult:
    """Output summary of a GRPO run."""

    out_dir: Path
    metrics: dict[str, float]
    train_runtime: float
    anchor_checkpoint: str
    seed_results: list[dict[str, Any]] = field(default_factory=list)


def compute_group_advantages(
    rewards: Sequence[float],
    *,
    eps: float = 1e-8,
) -> list[float]:
    """Compute group-relative advantages ``(r_i - mean) / (std + eps)``.

    Zero-variance groups (where every completion earned the exact same reward)
    yield all-zero advantages so no spurious gradient noise is injected.
    Non-finite rewards raise ``ValueError``.
    """
    values = [float(r) for r in rewards]
    if not values:
        return []
    for v in values:
        if not math.isfinite(v):
            raise ValueError(f"non-finite reward from environment: {v!r}")
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    if std <= eps:
        return [0.0] * n
    return [(v - mean) / (std + eps) for v in values]


@dataclass(frozen=True)
class EpisodeScore:
    """Environment-verified reward and diagnostics for one completion."""

    episode_id: str
    task_id: str
    completion: str
    reward: float
    success: bool
    status: str
    detail: dict[str, Any] = field(default_factory=dict)


class EnvironmentRewardBridge:
    """Evaluates completions strictly inside a ``maatml.rollout_env/1`` process.

    The learner calls :meth:`score_group` with a task and ``G`` candidate
    completions; each completion runs in an isolated episode via
    ``reset -> step_text -> verify -> close``. Rewards are never computed in
    the learner.
    """

    def __init__(
        self,
        environment: FreeTextEnvironment,
        *,
        options: Optional[dict[str, Any]] = None,
        prefix: str = "grpo",
    ) -> None:
        hello = environment.hello()
        if not hello.supports(FREE_TEXT_CAPABILITY):
            raise RolloutEnvError(
                f"GRPO requires an environment with {FREE_TEXT_CAPABILITY!r} capability; "
                f"{hello.name!r} advertises {list(hello.capabilities)!r}"
            )
        self._env = environment
        self._hello = hello
        self._options = dict(options or {})
        self._prefix = prefix
        self._counter = 0

    @property
    def environment_identity(self) -> dict[str, str]:
        return {
            "name": self._hello.name,
            "version": self._hello.version,
            "protocol": self._hello.protocol,
        }

    def score_one(self, task: dict[str, Any], completion: str) -> EpisodeScore:
        """Run one isolated episode for ``completion`` and return the
        environment's verified reward."""
        self._counter += 1
        episode_id = f"{self._prefix}-{self._counter:08d}"
        self._env.reset(episode_id, task, self._options)
        verdict: Optional[Verification] = None
        try:
            step = self._env.step_text(episode_id, completion)
            verdict = self._env.verify(episode_id)
            reward = float(verdict.reward)
            if not math.isfinite(reward):
                raise RolloutEnvError(
                    f"environment {self._hello.name!r} returned non-finite reward {reward!r}"
                )
            detail = dict(verdict.detail)
            if step.info:
                detail.setdefault("step_info", dict(step.info))
        finally:
            self._env.close(episode_id)
        assert verdict is not None
        return EpisodeScore(
            episode_id=episode_id,
            task_id=str(task.get("id") or ""),
            completion=completion,
            reward=reward,
            success=bool(verdict.success),
            status=str(verdict.status),
            detail=detail,
        )

    def score_group(
        self, task: dict[str, Any], completions: Sequence[str]
    ) -> tuple[list[EpisodeScore], list[float]]:
        """Score a group of completions for ``task`` and return
        ``(scores, advantages)``."""
        scores = [self.score_one(task, text) for text in completions]
        advantages = compute_group_advantages([s.reward for s in scores])
        return scores, advantages


@dataclass(frozen=True)
class PromotionDecision:
    """Verdict from :func:`evaluate_grpo_promotion`."""

    promote: bool
    completion_gain: float
    regressions: dict[str, float]
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "promote": self.promote,
            "completion_gain": self.completion_gain,
            "regressions": dict(self.regressions),
            "reasons": list(self.reasons),
        }


def evaluate_grpo_promotion(
    baseline_metrics: Mapping[str, float],
    candidate_metrics: Mapping[str, float],
    *,
    completion_metric: str = "task_completion",
    heldout_metrics: Optional[Sequence[str]] = None,
    min_completion_gain: float = 0.0,
    max_heldout_regression: float = 0.0,
) -> PromotionDecision:
    """Promote a GRPO candidate only when held-out metrics do not regress and
    ``completion_metric`` strictly improves by at least ``min_completion_gain``.
    """
    reasons: list[str] = []
    if completion_metric not in baseline_metrics or completion_metric not in candidate_metrics:
        reasons.append(f"missing {completion_metric!r} in baseline or candidate metrics")
        return PromotionDecision(
            promote=False,
            completion_gain=0.0,
            regressions={},
            reasons=tuple(reasons),
        )

    gain = float(candidate_metrics[completion_metric]) - float(baseline_metrics[completion_metric])
    if gain <= 0.0 or gain < min_completion_gain:
        reasons.append(
            f"{completion_metric} gain {gain:+.4f} did not beat required >{min_completion_gain:.4f}"
        )

    checked = (
        list(heldout_metrics)
        if heldout_metrics is not None
        else [k for k in baseline_metrics if k in candidate_metrics]
    )
    regressions: dict[str, float] = {}
    for key in checked:
        if key not in baseline_metrics or key not in candidate_metrics:
            reasons.append(f"held-out metric {key!r} missing from candidate or baseline")
            continue
        delta = float(candidate_metrics[key]) - float(baseline_metrics[key])
        if delta < -max_heldout_regression:
            regressions[key] = delta
            reasons.append(
                f"held-out metric {key!r} regressed by {delta:+.4f} "
                f"(allowed >= {-max_heldout_regression:+.4f})"
            )

    return PromotionDecision(
        promote=len(reasons) == 0,
        completion_gain=gain,
        regressions=regressions,
        reasons=tuple(reasons),
    )
