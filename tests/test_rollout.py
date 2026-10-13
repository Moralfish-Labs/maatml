"""Rollout driver, the environment protocol, and its stdio carrier."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from maatml.rollout import (
    PROTOCOL,
    RolloutConfig,
    RolloutEnvError,
    StdioEnvironment,
    rollout,
    select,
    summarize,
    write_episodes,
)
from maatml.rollout.example import LineWorld


class TowardTarget:
    """Scores like the teacher would."""

    def score(self, observation, candidates):
        gap = observation["target"] - observation["position"]
        want = "finish" if gap == 0 else ("right" if gap > 0 else "left")
        return [1.0 if item["id"] == want else 0.0 for item in candidates]


class AlwaysRight:
    def score(self, observation, candidates):
        return [1.0 if item["id"] == "right" else 0.0 for item in candidates]


def test_select_is_deterministic_and_rejects_bad_scores() -> None:
    candidates = [{"id": "b"}, {"id": "a"}, {"id": "c"}]
    assert select(candidates, [1.0, 1.0, 0.5]) == "a"
    with pytest.raises(RolloutEnvError):
        select(candidates, [1.0, float("nan"), 0.0])
    with pytest.raises(RolloutEnvError):
        select(candidates, [1.0])


def test_a_good_policy_succeeds_and_the_environment_verifies_it() -> None:
    env = LineWorld()
    tasks = env.tasks({"count": 8, "seed": 3})
    episodes = list(rollout(env, TowardTarget(), tasks))

    assert [e["task_id"] for e in episodes] == [t["id"] for t in tasks]
    assert all(e["outcome"]["success"] for e in episodes)
    first = episodes[0]["decisions"][0]
    assert first["teacher_valid"] is None
    assert len(first["logp"]) == len(first["candidates"])
    assert sum(math.exp(value) for value in first["logp"]) == pytest.approx(1.0)
    assert env._episodes == {}


def test_labels_record_the_teacher_on_states_the_policy_reached() -> None:
    env = LineWorld()
    task = {"id": "t", "family": "near", "start": 0, "target": -2, "max_steps": 3}
    config = RolloutConfig(source="dagger", label=True, round=1)
    (episode,) = rollout(env, AlwaysRight(), [task], config=config)

    assert episode["outcome"]["status"] == "exhausted"
    assert not episode["outcome"]["success"]
    assert [d["chosen"] for d in episode["decisions"]] == ["right"] * 3
    assert all(d["teacher_valid"] == ["left"] for d in episode["decisions"])
    assert [d["observation"]["position"] for d in episode["decisions"]] == [0, 1, 2]
    summary = summarize([episode])
    assert summary["policy_agrees_with_teacher"] == 0.0
    assert summary["families"]["near"]["success_rate"] == 0.0
    assert summary["families"]["near"]["wilson95"][0] == 0.0


def test_wilson_interval_matches_known_values() -> None:
    from maatml.rollout.driver import wilson95

    low, high = wilson95(49, 200)
    assert low == pytest.approx(0.1905, abs=1e-3) and high == pytest.approx(0.3090, abs=1e-3)
    assert wilson95(0, 0) == [0.0, 1.0]


def test_reset_options_relax_the_budget() -> None:
    env = LineWorld()
    task = {"id": "t", "family": "near", "start": 0, "target": 1, "max_steps": 1}
    tight = next(rollout(env, TowardTarget(), [task]))
    relaxed = next(
        rollout(env, TowardTarget(), [task], config=RolloutConfig(options={"extra_steps": 1}))
    )
    assert tight["outcome"]["status"] == "exhausted"
    assert relaxed["outcome"]["success"]


def test_teacher_mix_acts_with_the_teacher() -> None:
    env = LineWorld()
    task = {"id": "t", "family": "near", "start": 0, "target": -2, "max_steps": 4}
    config = RolloutConfig(label=True, teacher_mix=1.0)
    (episode,) = rollout(env, AlwaysRight(), [task], config=config)
    assert episode["outcome"]["success"]
    assert {d["actor"] for d in episode["decisions"]} == {"teacher"}
    with pytest.raises(ValueError):
        list(rollout(env, AlwaysRight(), [task], config=RolloutConfig(teacher_mix=0.5)))


def test_invalid_candidates_are_masked_and_the_miss_is_recorded() -> None:
    class Jumper:
        def score(self, observation, candidates):
            rank = {"jump": 2.0, "finish": 1.0}
            return [rank.get(item["id"], 0.0) for item in candidates]

    task = {"id": "t", "family": "near", "start": 0, "target": 0, "max_steps": 2}
    (episode,) = rollout(LineWorld(), Jumper(), [task])
    (decision,) = episode["decisions"]
    assert decision["chosen"] == "finish" and decision["top_valid"] is False
    assert episode["outcome"]["success"]
    assert select([{"id": "a", "valid": False}], [1.0]) is None


def test_episodes_are_written_with_a_summary(tmp_path: Path) -> None:
    env = LineWorld()
    summary = write_episodes(
        rollout(env, TowardTarget(), env.tasks({"count": 4, "seed": 1})), tmp_path
    )
    lines = (tmp_path / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == 4 and summary["episodes"] == 4 and summary["success_rate"] == 1.0
    assert json.loads(lines[0])["kind"] == "maatml.rollout_episode/1"
    assert json.loads((tmp_path / "episodes.summary.json").read_text()) == summary


def test_stdio_carrier_matches_the_in_process_environment() -> None:
    server = [
        sys.executable,
        "-c",
        "from maatml.rollout import serve; from maatml.rollout.example import LineWorld; "
        "serve(LineWorld())",
    ]
    local = LineWorld()
    tasks = local.tasks({"count": 5, "seed": 9, "slack": 1})
    config = RolloutConfig(label=True)
    expected = list(rollout(local, AlwaysRight(), tasks, config=config))
    with StdioEnvironment(server) as remote:
        assert remote.hello().protocol == PROTOCOL
        assert remote.tasks({"count": 5, "seed": 9, "slack": 1}) == tasks
        got = list(rollout(remote, AlwaysRight(), tasks, config=config))
        with pytest.raises(RolloutEnvError, match="unknown episode"):
            remote.candidates("missing")
    assert got == expected


def test_line_world_is_a_registered_environment() -> None:
    from maatml.registry import ENVIRONMENTS, discover_plugins

    discover_plugins()
    # Registry resets reload the module, so compare behaviour, not the class.
    assert ENVIRONMENTS.require("line_world")().hello().name == "line_world"


def _policy_folder(tmp_path: Path, body: str) -> Path:
    model_dir = tmp_path / "walker"
    plugin = model_dir / "walker_plugin"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(
        "from maatml.registry import register_predictor\n"
        "@register_predictor('walker')\n"
        "class Walker:\n"
        "    def setup(self, checkpoint_dir, **kwargs):\n"
        "        self.checkpoint = checkpoint_dir\n" + body,
        encoding="utf-8",
    )
    (model_dir / "model.yml").write_text(
        "name: walker\nmodel_id: walker\narchitecture: causal_sft\nversion: 0.1.0\n"
        "plugins: [walker_plugin]\nevaluation:\n  predictor: walker\n",
        encoding="utf-8",
    )
    (model_dir / "ckpt").mkdir()
    return model_dir


def test_cli_rollout_stores_labelled_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from maatml.cli import app

    # This fake predictor needs no tensor runtime; device discovery is covered
    # separately by the device tests and the ML smoke job.
    monkeypatch.setattr("maatml.device.resolve_device", lambda device: "cpu")

    model_dir = _policy_folder(
        tmp_path,
        "    def score(self, observation, candidates):\n"
        "        gap = observation['target'] - observation['position']\n"
        "        want = 'finish' if gap == 0 else ('right' if gap > 0 else 'left')\n"
        "        return [1.0 if c['id'] == want else 0.0 for c in candidates]\n",
    )
    result = CliRunner().invoke(
        app,
        [
            "rollout", str(model_dir), "--env", "line_world", "--checkpoint", "ckpt",
            "--tasks", '{"count": 3, "seed": 2}', "--label", "--round", "1",
            "--option", "extra_steps=2",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "loading checkpoint" in result.output
    for completed in range(4):
        assert f"{completed}/3 episodes completed" in result.output
    assert result.output.index("1/3 episodes completed") < result.output.index("rollout complete")
    out_dir = model_dir / "output" / "episodes" / "dagger-r1"
    summary = json.loads((out_dir / "episodes.summary.json").read_text())
    assert summary["episodes"] == 3 and summary["success_rate"] == 1.0
    assert summary["policy_agrees_with_teacher"] == 1.0
    episode = json.loads((out_dir / "episodes.jsonl").read_text().splitlines()[0])
    assert episode["source"] == "dagger" and episode["round"] == 1


def test_cli_rollout_refuses_a_predictor_that_cannot_rank(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from maatml.cli import app

    model_dir = _policy_folder(tmp_path, "    def predict(self, row):\n        return ''\n")
    result = CliRunner().invoke(
        app, ["rollout", str(model_dir), "--env", "line_world", "--checkpoint", "ckpt"]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError) and "score(" in str(result.exception)
