"""``filtered_imitation`` objective: expert iteration on environment-verified outputs.

Filters rollout episodes strictly to trajectories verified as successful by the
environment process (never rewarding raw syntax or format passes alone),
deduplicates by ``(task_id, trace_digest)``, enforces split isolation so
held-out test tasks never leak into training data, writes provenance metadata,
and hands the verified dataset to the standard SFT trainer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..config import ModelDefinition, get_dataset_cfg
from ..rollout.driver import FreeTextPolicy, RolloutConfig, rollout_free_text, trace_digest
from ..rollout.protocol import FreeTextEnvironment
from ..utils.io import iter_jsonl, write_json_atomic


class FilterConfig(BaseModel):
    """Selection and hygiene rules for ``filtered_imitation``."""

    model_config = ConfigDict(extra="forbid")

    # Require environment verification (`outcome.success == True`).
    require_verified: bool = True
    # Optional reward floor in addition to `outcome.success`.
    min_reward: Optional[float] = None
    # Reject episodes whose split matches any excluded split (prevents leakage).
    exclude_splits: tuple[str, ...] = ("test", "blind", "held_out")
    # Deduplicate identical verified trajectories per task.
    deduplicate: bool = True
    # Cap verified episodes kept per task id to prevent easy tasks dominating.
    max_per_task: Optional[int] = Field(default=None, ge=1)
    # Require at least this many decisions in an episode.
    min_decisions: int = Field(default=1, ge=1)
    # Optional system prompt attached when building single-turn chat rows.
    system_prompt: Optional[str] = None
    # Anchor checkpoint recorded in dataset provenance.
    anchor_checkpoint: Optional[str] = None


@dataclass(frozen=True)
class FilterStats:
    """Accounting for every episode seen by :func:`filter_verified_episodes`."""

    total_episodes: int = 0
    kept_episodes: int = 0
    kept_rows: int = 0
    rejected_unverified: int = 0
    rejected_low_reward: int = 0
    rejected_split_leakage: int = 0
    rejected_empty: int = 0
    rejected_duplicate: int = 0
    rejected_task_cap: int = 0
    families: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "kept_episodes": self.kept_episodes,
            "kept_rows": self.kept_rows,
            "rejected_unverified": self.rejected_unverified,
            "rejected_low_reward": self.rejected_low_reward,
            "rejected_split_leakage": self.rejected_split_leakage,
            "rejected_empty": self.rejected_empty,
            "rejected_duplicate": self.rejected_duplicate,
            "rejected_task_cap": self.rejected_task_cap,
            "families": dict(self.families),
        }


@dataclass(frozen=True)
class FilterResult:
    """Verified training rows and their provenance accounting."""

    rows: list[dict[str, Any]]
    stats: FilterStats
    dataset_sha256: str


def _episode_trace_digest(episode: dict[str, Any]) -> str:
    existing = episode.get("trace_digest")
    if isinstance(existing, str) and existing:
        return existing
    actions = [str(d.get("text", d.get("chosen", ""))) for d in episode.get("decisions") or []]
    return trace_digest(actions)


def _episode_to_sft_row(
    episode: dict[str, Any],
    *,
    system_prompt: Optional[str],
    request_field: str = "request",
    target_field: str = "expected",
) -> Optional[dict[str, Any]]:
    """Convert one verified episode into a chat-SFT row with full provenance."""
    decisions = episode.get("decisions") or []
    if not decisions:
        return None
    raw_task = episode.get("task")
    task: dict[str, Any] = raw_task if isinstance(raw_task, dict) else {}
    raw_outcome = episode.get("outcome")
    outcome: dict[str, Any] = raw_outcome if isinstance(raw_outcome, dict) else {}
    digest = _episode_trace_digest(episode)

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    elif isinstance(task.get("system"), str) and task["system"]:
        messages.append({"role": "system", "content": task["system"]})

    first_prompt = ""
    last_answer = ""
    for step_idx, decision in enumerate(decisions):
        raw_obs = decision.get("observation")
        obs: dict[str, Any] = raw_obs if isinstance(raw_obs, dict) else {}
        if step_idx == 0:
            user_text = str(
                obs.get("prompt") or task.get("prompt") or task.get(request_field) or ""
            )
            first_prompt = user_text
        else:
            user_text = str(obs.get("feedback") or obs.get("prompt") or json.dumps(obs))
        assistant_text = str(decision.get("text", decision.get("chosen", "")))
        last_answer = assistant_text
        if user_text:
            messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})

    row: dict[str, Any] = {
        "sample_id": f"{episode.get('episode_id', 'ep')}:{digest[:12]}",
        "family": str(episode.get("task_family") or task.get("family") or "unknown"),
        request_field: first_prompt,
        target_field: last_answer,
        "messages": messages,
        "provenance": {
            "objective": "filtered_imitation",
            "episode_id": str(episode.get("episode_id") or ""),
            "task_id": str(episode.get("task_id") or task.get("id") or ""),
            "task_family": str(episode.get("task_family") or task.get("family") or "unknown"),
            "source": str(episode.get("source") or "rollout"),
            "policy_version": str(episode.get("policy_version") or ""),
            "round": int(episode.get("round") or 0),
            "verified": bool(outcome.get("success")),
            "reward": float(outcome.get("reward") or 0.0),
            "trace_digest": digest,
        },
    }
    if isinstance(task.get("tools"), list) and task["tools"]:
        row["tools"] = task["tools"]
    if isinstance(task.get("chat_template_kwargs"), dict):
        row["chat_template_kwargs"] = task["chat_template_kwargs"]
    return row


def filter_verified_episodes(
    episodes: Iterable[dict[str, Any]],
    *,
    config: Optional[FilterConfig] = None,
    request_field: str = "request",
    target_field: str = "expected",
) -> FilterResult:
    """Keep only environment-verified passing trajectories as SFT rows."""
    cfg = config or FilterConfig()
    excluded = {s.strip().lower() for s in cfg.exclude_splits if s.strip()}

    total = 0
    kept_ep = 0
    rej_unverified = 0
    rej_low_reward = 0
    rej_split = 0
    rej_empty = 0
    rej_dup = 0
    rej_cap = 0
    seen_traces: set[tuple[str, str]] = set()
    per_task_counts: dict[str, int] = {}
    families: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    for episode in episodes:
        total += 1
        raw_outcome = episode.get("outcome")
        outcome: dict[str, Any] = raw_outcome if isinstance(raw_outcome, dict) else {}
        raw_task = episode.get("task")
        task: dict[str, Any] = raw_task if isinstance(raw_task, dict) else {}
        split = str(episode.get("split") or task.get("split") or "").strip().lower()
        if split and split in excluded:
            rej_split += 1
            continue

        verified = bool(outcome.get("success") is True)
        if cfg.require_verified and not verified:
            rej_unverified += 1
            continue

        reward = float(outcome.get("reward") or 0.0)
        if cfg.min_reward is not None and reward < cfg.min_reward:
            rej_low_reward += 1
            continue

        decisions = episode.get("decisions") or []
        if len(decisions) < cfg.min_decisions:
            rej_empty += 1
            continue

        task_id = str(episode.get("task_id") or task.get("id") or "")
        digest = _episode_trace_digest(episode)
        if cfg.deduplicate:
            key = (task_id, digest)
            if key in seen_traces:
                rej_dup += 1
                continue
            seen_traces.add(key)

        if cfg.max_per_task is not None and task_id:
            count = per_task_counts.get(task_id, 0)
            if count >= cfg.max_per_task:
                rej_cap += 1
                continue
            per_task_counts[task_id] = count + 1

        row = _episode_to_sft_row(
            episode,
            system_prompt=cfg.system_prompt,
            request_field=request_field,
            target_field=target_field,
        )
        if row is None:
            rej_empty += 1
            continue
        rows.append(row)
        kept_ep += 1
        family = str(row["family"])
        families[family] = families.get(family, 0) + 1

    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return FilterResult(
        rows=rows,
        stats=FilterStats(
            total_episodes=total,
            kept_episodes=kept_ep,
            kept_rows=len(rows),
            rejected_unverified=rej_unverified,
            rejected_low_reward=rej_low_reward,
            rejected_split_leakage=rej_split,
            rejected_empty=rej_empty,
            rejected_duplicate=rej_dup,
            rejected_task_cap=rej_cap,
            families=families,
        ),
        dataset_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def collect_and_filter(
    environment: FreeTextEnvironment,
    policy: FreeTextPolicy,
    tasks: Iterable[dict[str, Any]],
    *,
    rollout_config: Optional[RolloutConfig] = None,
    filter_config: Optional[FilterConfig] = None,
    samples_per_task: int = 1,
) -> tuple[list[dict[str, Any]], FilterResult]:
    """Sample free-text trajectories from ``policy`` in ``environment``,
    verify every trajectory inside the environment, and filter strictly to
    verified-passing outputs."""
    if samples_per_task < 1:
        raise ValueError(f"samples_per_task must be >= 1, got {samples_per_task}")
    base_cfg = rollout_config or RolloutConfig(source="filtered_imitation")
    task_list = list(tasks)
    episodes: list[dict[str, Any]] = []
    for sample_idx in range(samples_per_task):
        sample_cfg = RolloutConfig(
            source=base_cfg.source,
            policy_version=base_cfg.policy_version,
            round=base_cfg.round,
            label=False,
            teacher_mix=0.0,
            stop_unlabelled=base_cfg.stop_unlabelled,
            max_decisions=base_cfg.max_decisions,
            options=dict(base_cfg.options),
            seed=base_cfg.seed + sample_idx * 1009,
        )
        for ep in rollout_free_text(environment, policy, task_list, config=sample_cfg):
            if samples_per_task > 1:
                ep["episode_id"] = f"{ep['episode_id']}-s{sample_idx}"
            episodes.append(ep)
    filtered = filter_verified_episodes(episodes, config=filter_config)
    return episodes, filtered


def write_filtered_dataset(
    result: FilterResult,
    out_dir: Path,
    *,
    val_rows: Optional[list[dict[str, Any]]] = None,
    seed_rows: Optional[list[dict[str, Any]]] = None,
    anchor_checkpoint: Optional[str] = None,
) -> Path:
    """Write filtered verified rows + optional seed mix as an SFT-ready folder."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = list(result.rows)
    if seed_rows:
        combined.extend(seed_rows)

    train_path = out_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    val_path = out_dir / "val.jsonl"
    with val_path.open("w", encoding="utf-8") as handle:
        for row in val_rows or []:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    write_json_atomic(
        out_dir / "filtered_imitation_manifest.json",
        {
            "objective": "filtered_imitation",
            "dataset_sha256": result.dataset_sha256,
            "stats": result.stats.to_dict(),
            "seed_rows_mixed": len(seed_rows or []),
            "total_train_rows": len(combined),
            "val_rows": len(val_rows or []),
            "anchor_checkpoint": anchor_checkpoint,
        },
    )
    return out_dir


def train_filtered_imitation(
    model_def: ModelDefinition,
    *,
    episodes_path: Optional[str | Path] = None,
    dataset_dir: Optional[Path] = None,
    filter_config: Optional[FilterConfig] = None,
    **sft_kwargs: Any,
) -> Any:
    """Train with ``filtered_imitation``: filter verified episodes then run SFT."""
    from .sft_base import train_sft

    ext = (model_def.extensions or {}).get("filtered_imitation") or {}
    cfg = filter_config or FilterConfig(
        **{k: v for k, v in ext.items() if k in FilterConfig.model_fields}
    )
    ds_cfg = get_dataset_cfg(model_def)
    request_field = ds_cfg.get("request_field") or ds_cfg.get("raw_field") or "request"
    target_field = ds_cfg.get("target_field") or "expected"

    source_path = Path(episodes_path or ext.get("episodes_path") or "")
    if not str(source_path) or not source_path.is_file():
        prepared_root = Path(dataset_dir) if dataset_dir else model_def.prepared_dir
        default_ep = prepared_root / "episodes.jsonl"
        if default_ep.is_file():
            source_path = default_ep
        else:
            raise FileNotFoundError(
                "filtered_imitation requires episodes.jsonl (pass episodes_path or set "
                "extensions.filtered_imitation.episodes_path)"
            )

    episodes = list(iter_jsonl(source_path))
    result = filter_verified_episodes(
        episodes,
        config=cfg,
        request_field=str(request_field),
        target_field=str(target_field),
    )
    if not result.rows:
        raise ValueError(
            f"filtered_imitation found 0 verified episodes in {source_path} "
            f"(stats={result.stats.to_dict()})"
        )

    base_prepared = Path(dataset_dir) if dataset_dir else model_def.prepared_dir
    val_file = base_prepared / "val.jsonl"
    val_rows = list(iter_jsonl(val_file)) if val_file.is_file() else []
    seed_rows: list[dict[str, Any]] = []
    if ext.get("mix_seed_train") and (base_prepared / "train.jsonl").is_file():
        seed_rows = list(iter_jsonl(base_prepared / "train.jsonl"))

    filtered_dir = write_filtered_dataset(
        result,
        model_def.output_dir / "filtered_imitation" / result.dataset_sha256[:12],
        val_rows=val_rows,
        seed_rows=seed_rows,
        anchor_checkpoint=cfg.anchor_checkpoint,
    )
    return train_sft(
        model_def,
        dataset_dir=filtered_dir,
        log_label="filtered_imitation",
        **sft_kwargs,
    )
