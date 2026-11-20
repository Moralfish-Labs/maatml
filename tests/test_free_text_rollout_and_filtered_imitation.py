"""Free-text rollout protocol extension and verified-output filtered_imitation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest

from maatml.registry import ENVIRONMENTS, TRAINERS, discover_plugins
from maatml.rollout import (
    FREE_TEXT_CAPABILITY,
    GeneratedAction,
    RolloutConfig,
    RolloutEnvError,
    StdioEnvironment,
    rollout_free_text,
    summarize,
    write_episodes,
)
from maatml.rollout.example import ArithmeticWorld, LineWorld
from maatml.training.filtered_imitation import (
    FilterConfig,
    collect_and_filter,
    filter_verified_episodes,
    write_filtered_dataset,
)


class _MixedArithmeticPolicy:
    """Solves addition tasks correctly, returns a well-formed wrong integer on multiply."""

    def generate(
        self,
        observation: dict[str, Any],
        *,
        n: int = 1,
        seed: Optional[int] = None,
    ) -> Sequence[GeneratedAction]:
        del n, seed
        prompt = str(observation["prompt"])
        if "+" in prompt:
            nums = [int(tok.rstrip(".")) for tok in prompt.split() if tok.rstrip(".").isdigit()]
            return [GeneratedAction(text=f"<think>add</think>\n{nums[0] + nums[1]}", logp=-0.2)]
        # Well-formed JSON answer that is semantically wrong: format alone must NOT get reward.
        return [GeneratedAction(text='{"answer": 999999}', logp=-0.5)]


def test_free_text_rollout_works_in_process_and_over_stdio_and_never_rewards_format_only() -> None:
    env = ArithmeticWorld()
    assert env.hello().supports(FREE_TEXT_CAPABILITY)
    tasks = env.tasks({"count": 4, "seed": 7, "split": "train"})
    policy = _MixedArithmeticPolicy()
    cfg = RolloutConfig(source="expert_iter", policy_version="v1", round=1, seed=11)

    local_eps = list(rollout_free_text(env, policy, tasks, config=cfg))
    assert len(local_eps) == 4
    # Even indices are add (verified success), odd indices are mul (wrong integer -> reward 0).
    assert [ep["outcome"]["success"] for ep in local_eps] == [True, False, True, False]
    assert [ep["outcome"]["reward"] for ep in local_eps] == [1.0, 0.0, 1.0, 0.0]
    assert all(ep["action_mode"] == FREE_TEXT_CAPABILITY for ep in local_eps)
    assert all(len(ep["trace_digest"]) == 64 for ep in local_eps)
    assert local_eps[1]["decisions"][0]["info"]["format_valid"] is True
    assert local_eps[1]["decisions"][0]["info"]["correct"] is False

    # Identical behavior over real stdio JSON-lines subprocess carrier.
    server = [
        sys.executable,
        "-c",
        "from maatml.rollout import serve; from maatml.rollout.example import ArithmeticWorld; "
        "serve(ArithmeticWorld())",
    ]
    with StdioEnvironment(server) as remote:
        assert remote.hello().supports(FREE_TEXT_CAPABILITY)
        remote_eps = list(rollout_free_text(remote, policy, tasks, config=cfg))
    assert remote_eps == local_eps


def test_rollout_free_text_rejects_candidate_only_environment(tmp_path: Path) -> None:
    with pytest.raises(RolloutEnvError, match="free_text"):
        list(rollout_free_text(LineWorld(), _MixedArithmeticPolicy(), []))

    env = ArithmeticWorld()
    tasks = env.tasks({"count": 2, "seed": 1})
    summary = write_episodes(rollout_free_text(env, _MixedArithmeticPolicy(), tasks), tmp_path)
    assert summary["episodes"] == 2
    assert summary["success"] == 1
    assert summarize(rollout_free_text(env, _MixedArithmeticPolicy(), tasks))["success_rate"] == 0.5


def test_filtered_imitation_keeps_only_verified_non_leaking_deduped_outputs(tmp_path: Path) -> None:
    env = ArithmeticWorld()
    train_tasks = env.tasks({"count": 4, "seed": 3, "split": "train"})
    test_tasks = env.tasks({"count": 2, "seed": 99, "split": "test"})

    episodes, filtered = collect_and_filter(
        env,
        _MixedArithmeticPolicy(),
        train_tasks + test_tasks,
        samples_per_task=2,
        filter_config=FilterConfig(
            require_verified=True,
            min_reward=0.5,
            deduplicate=True,
            system_prompt="Solve arithmetic accurately.",
            anchor_checkpoint="ckpt-v0",
        ),
    )
    assert len(episodes) == 12
    # 2 test tasks x 2 samples = 4 rejected for split leakage.
    assert filtered.stats.rejected_split_leakage == 4
    # 2 wrong train tasks x 2 samples = 4 rejected as unverified.
    assert filtered.stats.rejected_unverified == 4
    # 2 correct train tasks x 2 identical samples -> 2 kept, 2 deduplicated.
    assert filtered.stats.kept_rows == 2
    assert filtered.stats.rejected_duplicate == 2

    for row in filtered.rows:
        assert row["provenance"]["objective"] == "filtered_imitation"
        assert row["provenance"]["verified"] is True
        assert row["provenance"]["reward"] == 1.0
        assert row["messages"][0] == {"role": "system", "content": "Solve arithmetic accurately."}

    # Direct filter_verified_episodes also enforces max_per_task and min_reward.
    high_floor = filter_verified_episodes(episodes, config=FilterConfig(min_reward=1.5))
    assert high_floor.stats.kept_rows == 0
    assert high_floor.stats.rejected_low_reward == 4

    out_dir = write_filtered_dataset(
        filtered,
        tmp_path / "filtered_out",
        seed_rows=[{"sample_id": "seed-1", "request": "1+1", "expected": "2"}],
        anchor_checkpoint="ckpt-v0",
    )
    manifest = json.loads((out_dir / "filtered_imitation_manifest.json").read_text())
    assert manifest["objective"] == "filtered_imitation"
    assert manifest["total_train_rows"] == 3
    assert manifest["seed_rows_mixed"] == 1
    assert manifest["anchor_checkpoint"] == "ckpt-v0"
    assert len((out_dir / "train.jsonl").read_text().splitlines()) == 3


def test_arithmetic_world_and_filtered_imitation_are_registered() -> None:
    discover_plugins(force=True)
    assert ENVIRONMENTS.require("arithmetic_world")().hello().name == "arithmetic_world"
    assert "filtered_imitation" in TRAINERS.names()
