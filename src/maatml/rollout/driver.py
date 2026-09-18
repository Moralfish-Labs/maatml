"""Drive a policy through an environment and record what happened.

One driver serves three uses: plain rollouts, DAgger collection (the policy
acts, the environment's teacher labels every visited state), and closed-loop
evaluation (no labels, frozen tasks, verified success per family).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Protocol, Sequence

from .protocol import Environment, RolloutEnvError

EPISODE_KIND = "maatml.rollout_episode/1"


class Policy(Protocol):
    """Scores every candidate at an observation; higher is better."""

    def score(
        self, observation: dict[str, Any], candidates: Sequence[dict[str, Any]]
    ) -> Sequence[float]: ...


@dataclass
class RolloutConfig:
    """How episodes are collected."""

    # Episode provenance: `rollout`, `dagger`, `eval`, ...
    source: str = "rollout"
    policy_version: str = ""
    round: int = 0
    # Ask the environment's teacher for the valid set at every state.
    label: bool = False
    # Probability of acting with the teacher's first valid action instead of
    # the policy's choice (DAgger's beta). Needs `label`.
    teacher_mix: float = 0.0
    # End an episode at the first state the teacher cannot label.
    stop_unlabelled: bool = True
    # Guard against an environment that never terminates.
    max_decisions: int = 256
    # Passed to `reset` untouched, e.g. a relaxed step budget.
    options: dict[str, Any] = field(default_factory=dict)
    seed: int = 0


def log_softmax(scores: Sequence[float]) -> list[float]:
    top = max(scores)
    total = math.log(sum(math.exp(value - top) for value in scores)) + top
    return [value - total for value in scores]


def select(candidates: Sequence[dict[str, Any]], scores: Sequence[float]) -> Optional[str]:
    """Highest-scoring valid candidate, or ``None`` when none is valid.

    A candidate is valid unless the environment marks it ``"valid": false``.
    Ties go to the smaller id, so selection is deterministic.
    """
    if len(candidates) != len(scores):
        raise RolloutEnvError(f"{len(scores)} scores for {len(candidates)} candidates")
    best: Optional[tuple[float, str]] = None
    for candidate, score in zip(candidates, scores, strict=True):
        value = float(score)
        if not math.isfinite(value):
            raise RolloutEnvError(f"non-finite score for candidate {candidate.get('id')!r}")
        if candidate.get("valid") is False:
            continue
        ident = str(candidate["id"])
        if best is None or value > best[0] or (value == best[0] and ident < best[1]):
            best = (value, ident)
    return None if best is None else best[1]


def run_episode(
    environment: Environment,
    policy: Policy,
    task: dict[str, Any],
    *,
    episode_id: str,
    config: RolloutConfig,
    rng: random.Random,
    identity: dict[str, str],
) -> dict[str, Any]:
    """One episode, from ``reset`` to the environment's verdict."""
    observation = environment.reset(episode_id, task, config.options)
    decisions: list[dict[str, Any]] = []
    truncated: Optional[str] = None
    try:
        while True:
            if len(decisions) >= config.max_decisions:
                truncated = "max_decisions"
                break
            candidates = environment.candidates(episode_id)
            if not candidates:
                truncated = "no_candidates"
                break
            teacher_valid: Optional[list[str]] = None
            if config.label:
                labels = environment.label(episode_id)
                if labels.labelled:
                    teacher_valid = list(labels.valid_ids or ())
                elif config.stop_unlabelled:
                    truncated = f"unlabelled:{labels.reason or 'unknown'}"
                    break
            scores = [float(value) for value in policy.score(observation, candidates)]
            chosen = select(candidates, scores)
            if chosen is None:
                truncated = "no_valid_candidate"
                break
            top = max(range(len(scores)), key=lambda i: (scores[i], -i))
            actor = "policy"
            if teacher_valid and rng.random() < config.teacher_mix:
                chosen, actor = teacher_valid[0], "teacher"
            result = environment.step(episode_id, chosen)
            decisions.append(
                {
                    "observation": observation,
                    "candidates": candidates,
                    "chosen": chosen,
                    "actor": actor,
                    "logp": log_softmax(scores),
                    # Whether the policy's first choice was valid before masking.
                    "top_valid": candidates[top].get("valid") is not False,
                    "teacher_valid": teacher_valid,
                    "applied": result.applied,
                    "reward": result.reward,
                }
            )
            observation = result.observation
            if result.terminal is not None:
                break
        verdict = environment.verify(episode_id)
    finally:
        environment.close(episode_id)
    return {
        "kind": EPISODE_KIND,
        "episode_id": episode_id,
        "task_id": str(task.get("id") or ""),
        "task_family": str(task.get("family") or "unknown"),
        "task": task,
        "source": config.source,
        "policy_version": config.policy_version,
        "round": config.round,
        "environment": identity,
        "decisions": decisions,
        "outcome": {
            "status": verdict.status,
            "success": verdict.success,
            "reward": verdict.reward,
            "decisions": len(decisions),
            "truncated": truncated,
            "detail": verdict.detail,
        },
    }


def rollout(
    environment: Environment,
    policy: Policy,
    tasks: Iterable[dict[str, Any]],
    *,
    config: Optional[RolloutConfig] = None,
) -> Iterator[dict[str, Any]]:
    """Episodes for ``tasks``, one at a time, in task order."""
    config = config or RolloutConfig()
    if config.teacher_mix and not config.label:
        raise ValueError("teacher_mix needs label=True")
    hello = environment.hello()
    if config.label and not hello.supports("label"):
        raise RolloutEnvError(f"environment {hello.name!r} cannot label states")
    identity = {"name": hello.name, "version": hello.version, "protocol": hello.protocol}
    rng = random.Random(config.seed)
    for index, task in enumerate(tasks):
        episode_id = f"{config.source}-r{config.round}-{index:06d}"
        yield run_episode(
            environment,
            policy,
            task,
            episode_id=episode_id,
            config=config,
            rng=rng,
            identity=identity,
        )


def summarize(episodes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Verified success, overall and per task family, plus label coverage."""
    families: dict[str, dict[str, int]] = {}
    decisions = labelled = agreed = truncated = 0
    for episode in episodes:
        outcome = episode["outcome"]
        bucket = families.setdefault(episode["task_family"], {"episodes": 0, "success": 0})
        bucket["episodes"] += 1
        bucket["success"] += 1 if outcome["success"] else 0
        truncated += 1 if outcome["truncated"] else 0
        for decision in episode["decisions"]:
            decisions += 1
            if decision["teacher_valid"] is not None:
                labelled += 1
                by_policy = decision["actor"] == "policy"
                if by_policy and decision["chosen"] in decision["teacher_valid"]:
                    agreed += 1
    episodes_total = sum(bucket["episodes"] for bucket in families.values())
    success_total = sum(bucket["success"] for bucket in families.values())
    return {
        "episodes": episodes_total,
        "success": success_total,
        "success_rate": success_total / episodes_total if episodes_total else 0.0,
        "truncated": truncated,
        "decisions": decisions,
        "labelled_decisions": labelled,
        "policy_agrees_with_teacher": agreed / labelled if labelled else None,
        "families": {
            name: {**bucket, "success_rate": bucket["success"] / bucket["episodes"]}
            for name, bucket in sorted(families.items())
        },
    }


def write_episodes(
    episodes: Iterable[dict[str, Any]], out_dir: Path, *, name: str = "episodes"
) -> dict[str, Any]:
    """Stream episodes to ``<out_dir>/<name>.jsonl`` and write their summary beside it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    kept: list[dict[str, Any]] = []
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, separators=(",", ":"), sort_keys=True) + "\n")
            # The summary needs outcomes and labels, not the observations.
            kept.append(
                {
                    "task_family": episode["task_family"],
                    "outcome": episode["outcome"],
                    "decisions": [
                        {key: d[key] for key in ("teacher_valid", "actor", "chosen")}
                        for d in episode["decisions"]
                    ],
                }
            )
    summary = {"episodes_file": path.name, **summarize(kept)}
    (out_dir / f"{name}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
