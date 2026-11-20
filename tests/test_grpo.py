"""GRPO config, group advantages, environment reward bridge, promotion guard, and CPU smoke."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from maatml.config import load_model_def
from maatml.registry import TRAINERS, discover_plugins
from maatml.rollout import RolloutEnvError, StdioEnvironment
from maatml.rollout.example import ArithmeticWorld, LineWorld
from maatml.training.grpo_config import (
    EnvironmentRewardBridge,
    GRPOTrainConfig,
    compute_group_advantages,
    evaluate_grpo_promotion,
)


def test_grpo_config_validation_and_registration() -> None:
    cfg = GRPOTrainConfig()
    assert cfg.group_size == 4
    assert cfg.kl_beta == 0.04

    with pytest.raises(ValueError, match="greater than or equal to 2"):
        GRPOTrainConfig(group_size=1)
    with pytest.raises(ValueError, match="at least 2 seeds"):
        GRPOTrainConfig(seeds=[42])
    with pytest.raises(ValueError, match="non-empty"):
        GRPOTrainConfig(environment="   ")

    multi = GRPOTrainConfig(seeds=[7, 13], anchor_checkpoint="/tmp/anchor")
    assert multi.seeds == [7, 13]

    discover_plugins(force=True)
    assert "grpo" in TRAINERS.names()


def test_compute_group_advantages_handles_mixed_and_zero_variance_groups() -> None:
    assert compute_group_advantages([]) == []
    assert compute_group_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    assert compute_group_advantages([0.0, 0.0]) == [0.0, 0.0]

    adv = compute_group_advantages([1.0, 0.0, 1.0, 0.0])
    assert adv[0] > 0.0 and adv[1] < 0.0
    assert adv[0] == pytest.approx(1.0, abs=1e-6)
    assert adv[1] == pytest.approx(-1.0, abs=1e-6)
    assert sum(adv) == pytest.approx(0.0, abs=1e-6)

    with pytest.raises(ValueError, match="non-finite"):
        compute_group_advantages([1.0, float("nan")])


def test_environment_reward_bridge_enforces_free_text_and_runs_over_stdio() -> None:
    with pytest.raises(RolloutEnvError, match="free_text"):
        EnvironmentRewardBridge(LineWorld())

    env = ArithmeticWorld()
    task = {"id": "t1", "family": "add", "prompt": "Compute 6 + 7.", "expected": 13, "max_steps": 1}

    bridge = EnvironmentRewardBridge(env)
    scores, advantages = bridge.score_group(task, ["13", "12", "not a number", '{"answer": 13}'])
    assert [s.success for s in scores] == [True, False, False, True]
    assert [s.reward for s in scores] == [1.0, 0.0, 0.0, 1.0]
    assert advantages[0] > 0 and advantages[1] < 0 and advantages[3] > 0
    assert env._episodes == {}

    server = [
        sys.executable,
        "-c",
        "from maatml.rollout import serve; from maatml.rollout.example import ArithmeticWorld; "
        "serve(ArithmeticWorld())",
    ]
    with StdioEnvironment(server) as remote:
        remote_bridge = EnvironmentRewardBridge(remote)
        r_scores, r_adv = remote_bridge.score_group(task, ["13", "99"])
    assert [s.reward for s in r_scores] == [1.0, 0.0]
    assert r_adv[0] > 0 and r_adv[1] < 0


def test_evaluate_grpo_promotion_enforces_completion_gain_and_heldout_non_regression() -> None:
    baseline = {"task_completion": 0.60, "safety": 1.00, "citation_precision": 0.90}

    ok = evaluate_grpo_promotion(
        baseline,
        {"task_completion": 0.72, "safety": 1.00, "citation_precision": 0.91},
        min_completion_gain=0.05,
    )
    assert ok.promote is True
    assert ok.completion_gain == pytest.approx(0.12)
    assert ok.regressions == {}

    # Rejects when a held-out metric regresses even if task completion jumped.
    regressed = evaluate_grpo_promotion(
        baseline,
        {"task_completion": 0.85, "safety": 0.95, "citation_precision": 0.92},
    )
    assert regressed.promote is False
    assert "safety" in regressed.regressions

    # Rejects when task completion did not improve.
    flat = evaluate_grpo_promotion(
        baseline,
        {"task_completion": 0.60, "safety": 1.00, "citation_precision": 0.95},
    )
    assert flat.promote is False
    assert flat.to_dict()["promote"] is False


def test_grpo_loss_and_tiny_cpu_multiseed_smoke(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from maatml.training.grpo import grpo_loss, train_grpo

    # 1. Exact GRPO loss properties: zero KL when policy == ref, finite grad.
    pol = torch.tensor([[-1.0, -0.5, -0.2], [-1.2, -0.8, -0.4]], requires_grad=True)
    old = pol.detach().clone()
    ref = pol.detach().clone()
    adv = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    loss, diag = grpo_loss(pol, old, ref, adv, mask, clip_eps=0.2, kl_beta=0.05)
    assert diag["kl"] == pytest.approx(0.0, abs=1e-6)
    assert diag["ratio"] == pytest.approx(1.0, abs=1e-6)
    loss.backward()
    assert pol.grad is not None and torch.isfinite(pol.grad).all()

    # Non-zero KL penalty when policy drifts from anchor ref.
    drifted = pol.detach() + 0.5
    _, diag_drift = grpo_loss(drifted, old, ref, adv, mask, clip_eps=0.2, kl_beta=0.05)
    assert diag_drift["kl"] > 0.0

    # 2. End-to-end tiny CPU GRPO smoke run with anchor checkpoint and 2 seeds.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "Compute": 3,
        "1": 4,
        "+": 5,
        "2": 6,
        "3": 7,
        "4": 8,
    }
    raw_tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    raw_tok.pre_tokenizer = Whitespace()
    fast_tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw_tok,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )

    tiny_cfg = transformers.GPT2Config(
        vocab_size=len(vocab),
        n_positions=64,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    tiny_model = transformers.GPT2LMHeadModel(tiny_cfg)
    base_dir = tmp_path / "tiny-base"
    base_dir.mkdir()
    tiny_model.save_pretrained(base_dir)
    fast_tok.save_pretrained(base_dir)

    model_dir = tmp_path / "grpo-demo"
    model_dir.mkdir()
    (model_dir / "model.yml").write_text(
        "\n".join(
            [
                "name: grpo-demo",
                "model_id: grpo-demo",
                "version: 0.1.0",
                "architecture: grpo",
                "training:",
                f"  model_id: {base_dir}",
                f"  anchor_checkpoint: {base_dir}",
                "  environment: arithmetic_world",
                "  group_size: 2",
                "  kl_beta: 0.04",
                "  max_new_tokens: 4",
                "  max_steps: 1",
                "  grad_accum: 1",
                "  precision: fp32",
                "  seeds: [11, 22]",
                "  lora:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    md = load_model_def(model_dir)
    prepared = md.prepared_dir
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "tasks.jsonl").write_text(
        json.dumps({"id": "t1", "family": "add", "prompt": "Compute 1 + 2", "expected": 3}) + "\n",
        encoding="utf-8",
    )

    result = train_grpo(
        md,
        device="cpu",
        out_dir=tmp_path / "grpo-out",
        environment_override=ArithmeticWorld(),
    )
    assert result.metrics["seed_count"] == 2.0
    assert result.anchor_checkpoint == str(base_dir)
    assert len(result.seed_results) == 2
    assert (tmp_path / "grpo-out" / "grpo_seed_study.json").is_file()
    meta = json.loads((result.out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["extra"]["trainer"] == "grpo"
    assert meta["extra"]["anchor_checkpoint"] == str(base_dir)
    assert meta["extra"]["environment"]["name"] == "arithmetic_world"
