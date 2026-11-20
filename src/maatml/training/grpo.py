"""Group-Relative Policy Optimization (GRPO) trainer backed by
``maatml.rollout_env/1`` environments.

Rewards are always computed by the environment process via
:class:`maatml.training.grpo_config.EnvironmentRewardBridge`, never inside the
learner. The policy updates against a frozen anchor checkpoint (KL penalty)
and supports multi-seed runs without requiring a newer TRL release beyond the
package's ``trl>=0.9`` floor.
"""

from __future__ import annotations

import contextlib
import copy
import math
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
import torch.nn.functional as F
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from ..config import ModelDefinition
from ..device import resolve_training_placement
from ..progress import JobProgressReporter, resolve_progress_file
from ..registry import ENVIRONMENTS
from ..rollout.protocol import FreeTextEnvironment
from ..rollout.stdio import StdioEnvironment
from ..runs import begin_training_run, finish_run
from ..utils.io import iter_jsonl, write_json_atomic
from .grpo_config import (
    EnvironmentRewardBridge,
    GRPOTrainConfig,
    GRPOTrainResult,
)
from .guards import ensure_tokenizer_model_contract, write_run_metadata
from .load import from_pretrained_kwargs, maybe_prepare_kbit
from .seeds import summarize_seed_runs
from .sft_base import _maybe_attach_lora, _refuse_quantized_merge, _save_sft_artifacts
from .signals import TrainingInterrupted, TrainingStopController

console = Console()


def sequence_token_logprobs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-token log probabilities for ``input_ids[:, 1:]``."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def grpo_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute clipped GRPO surrogate loss + non-negative anchor KL penalty.

    Shapes:
      - ``policy_logps``, ``old_logps``, ``ref_logps``, ``completion_mask``:
        ``[G, T]`` over completion token positions.
      - ``advantages``: ``[G]`` group-normalized rewards from the environment.
    """
    mask = completion_mask.float()
    denom = mask.sum().clamp_min(1.0)

    log_ratio = (policy_logps - old_logps).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    adv = advantages.float().unsqueeze(-1)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_obj = torch.minimum(surr1, surr2)

    # Schulman / DeepSeek unbiased non-negative KL estimator to anchor checkpoint.
    ref_diff = (ref_logps - policy_logps).clamp(-20.0, 20.0)
    per_tok_kl = torch.exp(ref_diff) - ref_diff - 1.0

    loss_per_tok = -(policy_obj - kl_beta * per_tok_kl)
    total_loss = (loss_per_tok * mask).sum() / denom
    mean_kl = float(((per_tok_kl * mask).sum() / denom).detach().item())
    mean_ratio = float(((ratio * mask).sum() / denom).detach().item())
    return total_loss, {"kl": mean_kl, "ratio": mean_ratio}


@contextlib.contextmanager
def _open_environment(spec: str | list[str]) -> Iterator[FreeTextEnvironment]:
    if isinstance(spec, list):
        with StdioEnvironment(spec) as stdio_env:
            yield stdio_env
        return
    env_cls = ENVIRONMENTS.require(spec)
    env = env_cls()
    try:
        yield env
    finally:
        shutdown = getattr(env, "shutdown", None)
        if callable(shutdown):
            shutdown()


def _load_grpo_tasks(
    model_def: ModelDefinition,
    cfg: GRPOTrainConfig,
    environment: FreeTextEnvironment,
    *,
    dataset_dir: Optional[Path],
    limit: Optional[int],
) -> list[dict[str, Any]]:
    source_dir = Path(dataset_dir) if dataset_dir else model_def.prepared_dir
    tasks_file = source_dir / "tasks.jsonl"
    if tasks_file.is_file():
        tasks = list(iter_jsonl(tasks_file))
    elif (source_dir / "train.jsonl").is_file():
        rows = list(iter_jsonl(source_dir / "train.jsonl"))
        tasks = [
            row
            if "prompt" in row
            else {
                "id": str(row.get("sample_id") or f"row-{i}"),
                "family": str(row.get("family") or "unknown"),
                "prompt": str(row.get("request") or ""),
                "expected": row.get("expected"),
            }
            for i, row in enumerate(rows)
        ]
    else:
        tasks = environment.tasks(dict(cfg.task_query))
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("GRPO trainer received 0 tasks from dataset or environment")
    return tasks


def _render_prompt_text(tokenizer: Any, task: dict[str, Any]) -> str:
    prompt = str(task.get("prompt") or "")
    messages = task.get("messages")
    if isinstance(messages, list) and messages and hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if isinstance(rendered, str) and rendered:
                return rendered
        except (TypeError, ValueError):
            pass
    return prompt


def _run_single_grpo_seed(
    model_def: ModelDefinition,
    cfg: GRPOTrainConfig,
    *,
    seed: int,
    smoke: bool,
    dataset_dir: Optional[Path],
    out_dir: Optional[Path],
    limit: Optional[int],
    device: str,
    resume: Optional[str],
    trial: Optional[dict[str, Any]],
    progress_file: Optional[str | Path],
    environment_override: Optional[FreeTextEnvironment],
) -> GRPOTrainResult:
    set_seed(seed)
    torch.manual_seed(seed)

    quant = cfg.quantization if cfg.quantization and cfg.quantization.enabled() else None
    _refuse_quantized_merge(
        cfg.lora.save_mode,
        lora_enabled=cfg.lora.enabled,
        quantized=quant is not None,
    )
    target_device, profile, distributed = resolve_training_placement(device)
    if distributed:
        raise ValueError("GRPO trainer currently runs in single-process mode")

    run, resolved_out, _resume_path = begin_training_run(
        model_def,
        smoke=smoke,
        device=str(target_device),
        profile=profile.name,
        out_dir=out_dir,
        resume=resume,
        trial=trial,
    )
    reporter = JobProgressReporter(
        resolve_progress_file(progress_file),
        run_id=run.run_id,
        artifact_uri=str(resolved_out),
    )
    stop_controller = TrainingStopController()
    anchor_ref = cfg.anchor_checkpoint or cfg.model_id
    started = time.perf_counter()

    try:
        with stop_controller.installed():
            reporter.update(stage="train")
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            load_kwargs = from_pretrained_kwargs(
                profile,
                precision=cfg.precision,
                attn_implementation=cfg.attn_implementation,
                quantization=quant,
                revision=cfg.model_revision,
                frozen_base=bool(cfg.lora and cfg.lora.enabled),
                distributed=False,
            )
            policy_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **load_kwargs)
            policy_model = maybe_prepare_kbit(policy_model, quant)
            ensure_tokenizer_model_contract(policy_model, tokenizer)
            policy_model = _maybe_attach_lora(policy_model, cfg.lora)
            policy_model.to(target_device)

            if cfg.anchor_checkpoint:
                ref_model = AutoModelForCausalLM.from_pretrained(
                    cfg.anchor_checkpoint, **load_kwargs
                )
            else:
                ref_model = copy.deepcopy(
                    policy_model.get_base_model()
                    if hasattr(policy_model, "get_base_model")
                    else policy_model
                )
            ref_model.to(target_device)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad_(False)

            optimizer = torch.optim.AdamW(
                [p for p in policy_model.parameters() if p.requires_grad],
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )

            env_cm = (
                contextlib.nullcontext(environment_override)
                if environment_override is not None
                else _open_environment(cfg.environment)
            )
            with env_cm as env:
                bridge = EnvironmentRewardBridge(
                    env,
                    options=cfg.environment_options,
                    prefix=f"grpo-s{seed}",
                )
                tasks = _load_grpo_tasks(model_def, cfg, env, dataset_dir=dataset_dir, limit=limit)

                total_steps = 0
                total_episodes = 0
                total_successes = 0
                reward_sum = 0.0
                kl_sum = 0.0
                loss_sum = 0.0
                optimizer.zero_grad()

                epochs_int = max(1, math.ceil(cfg.epochs))
                for _epoch in range(epochs_int):
                    for task in tasks:
                        stop_controller.raise_if_interrupted()
                        if cfg.max_steps > 0 and total_steps >= cfg.max_steps:
                            break

                        prompt_text = _render_prompt_text(tokenizer, task)
                        enc = tokenizer(
                            prompt_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=cfg.max_input_tokens,
                        )
                        prompt_ids = enc["input_ids"].to(target_device)
                        prompt_mask = enc["attention_mask"].to(target_device)
                        prompt_len = int(prompt_ids.shape[1])

                        # Sample a group of G completions from the current policy.
                        policy_model.eval()
                        with torch.no_grad():
                            gen_ids = policy_model.generate(
                                input_ids=prompt_ids,
                                attention_mask=prompt_mask,
                                max_new_tokens=cfg.max_new_tokens,
                                do_sample=True,
                                temperature=cfg.temperature,
                                top_p=cfg.top_p,
                                num_return_sequences=cfg.group_size,
                                pad_token_id=tokenizer.pad_token_id,
                            )

                        full_mask = (gen_ids != tokenizer.pad_token_id).long()
                        full_mask[:, :prompt_len] = 1
                        completions: list[str] = []
                        comp_mask = torch.zeros(
                            (gen_ids.shape[0], max(1, gen_ids.shape[1] - 1)),
                            dtype=torch.float32,
                            device=target_device,
                        )
                        for row_idx in range(gen_ids.shape[0]):
                            comp_ids = gen_ids[row_idx, prompt_len:]
                            decoded = str(
                                tokenizer.decode(comp_ids, skip_special_tokens=True)
                            ).strip()
                            completions.append(decoded)
                            if gen_ids.shape[1] > prompt_len:
                                comp_mask[row_idx, prompt_len - 1 :] = full_mask[
                                    row_idx, prompt_len:
                                ].float()

                        # Rewards come strictly from the environment process.
                        scores, advantages_list = bridge.score_group(task, completions)
                        total_episodes += len(scores)
                        total_successes += sum(1 for s in scores if s.success)
                        reward_sum += sum(s.reward for s in scores)

                        if gen_ids.shape[1] <= prompt_len:
                            continue

                        with torch.no_grad():
                            old_logps = sequence_token_logprobs(policy_model, gen_ids, full_mask)
                            ref_logps = sequence_token_logprobs(ref_model, gen_ids, full_mask)

                        policy_model.train()
                        pol_logps = sequence_token_logprobs(policy_model, gen_ids, full_mask)
                        adv_tensor = torch.tensor(
                            advantages_list, dtype=torch.float32, device=target_device
                        )
                        loss, diag = grpo_loss(
                            pol_logps,
                            old_logps,
                            ref_logps,
                            adv_tensor,
                            comp_mask,
                            clip_eps=cfg.clip_eps,
                            kl_beta=cfg.kl_beta,
                        )
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"GRPO encountered non-finite loss: {loss.item()}")

                        (loss / float(cfg.grad_accum)).backward()
                        total_steps += 1
                        loss_sum += float(loss.detach().item())
                        kl_sum += diag["kl"]

                        if total_steps % cfg.grad_accum == 0:
                            if cfg.max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    policy_model.parameters(), cfg.max_grad_norm
                                )
                            optimizer.step()
                            optimizer.zero_grad()

                        if total_steps % cfg.logging_steps == 0:
                            reporter.update(
                                stage="train",
                                step=total_steps,
                                metric_name="task_completion",
                                metric_value=total_successes / max(1, total_episodes),
                            )
                    if cfg.max_steps > 0 and total_steps >= cfg.max_steps:
                        break

                if total_steps % cfg.grad_accum != 0:
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                resolved_out.mkdir(parents=True, exist_ok=True)
                _save_sft_artifacts(
                    policy_model,
                    tokenizer,
                    resolved_out,
                    save_mode=cfg.lora.save_mode,
                    base_model_id=cfg.model_id,
                    quantized=quant is not None,
                )
                runtime = time.perf_counter() - started
                metrics = {
                    "grpo_steps": float(total_steps),
                    "episodes": float(total_episodes),
                    "mean_reward": reward_sum / max(1, total_episodes),
                    "task_completion": total_successes / max(1, total_episodes),
                    "mean_kl": kl_sum / max(1, total_steps),
                    "train_loss": loss_sum / max(1, total_steps),
                    "train_runtime": float(runtime),
                }
                write_run_metadata(
                    resolved_out,
                    model_def,
                    {},
                    extra={
                        "trainer": "grpo",
                        "run_id": run.run_id,
                        "seed": seed,
                        "anchor_checkpoint": anchor_ref,
                        "environment": bridge.environment_identity,
                        "group_size": cfg.group_size,
                        "kl_beta": cfg.kl_beta,
                        "clip_eps": cfg.clip_eps,
                        "precision": cfg.precision,
                    },
                )
                finish_run(model_def, run.run_id, "completed", metrics=metrics)
                reporter.update(
                    stage="done",
                    step=total_steps,
                    metric_name="task_completion",
                    metric_value=metrics["task_completion"],
                )
                return GRPOTrainResult(
                    out_dir=resolved_out,
                    metrics=metrics,
                    train_runtime=runtime,
                    anchor_checkpoint=anchor_ref,
                )
    except TrainingInterrupted:
        finish_run(model_def, run.run_id, "interrupted", metrics={})
        raise
    except Exception as exc:
        finish_run(model_def, run.run_id, "aborted", metrics={}, error=str(exc))
        raise


def train_grpo(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    dataset_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
    resume: Optional[str] = None,
    trial: Optional[dict[str, Any]] = None,
    progress_file: Optional[str | Path] = None,
    environment_override: Optional[FreeTextEnvironment] = None,
) -> GRPOTrainResult:
    """Train a language model with GRPO against an environment verifier.

    When ``training.seeds`` lists multiple seeds (and no explicit single
    ``seed`` override is passed), runs every seed against the same anchor
    checkpoint and aggregates per-metric mean/sd/min/max into
    ``grpo_seed_study.json``.
    """
    training_dict = dict(model_def.merged_smoke() if smoke else model_def.training)
    for drop in ("generation", "heads", "head_loss_weights", "embedding_strategy", "select_by"):
        training_dict.pop(drop, None)
    cfg = GRPOTrainConfig(**training_dict)

    if seed is not None or not cfg.seeds:
        run_seed = cfg.seed if seed is None else int(seed)
        return _run_single_grpo_seed(
            model_def,
            cfg,
            seed=run_seed,
            smoke=smoke,
            dataset_dir=dataset_dir,
            out_dir=out_dir,
            limit=limit,
            device=device,
            resume=resume,
            trial=trial,
            progress_file=progress_file,
            environment_override=environment_override,
        )

    # Multi-seed GRPO run.
    started = time.perf_counter()
    results: list[GRPOTrainResult] = []
    seed_records: list[dict[str, Any]] = []
    for run_seed in cfg.seeds:
        seed_out = (Path(out_dir) / f"seed-{run_seed}") if out_dir is not None else None
        res = _run_single_grpo_seed(
            model_def,
            cfg,
            seed=int(run_seed),
            smoke=smoke,
            dataset_dir=dataset_dir,
            out_dir=seed_out,
            limit=limit,
            device=device,
            resume=resume,
            trial=trial,
            progress_file=progress_file,
            environment_override=environment_override,
        )
        results.append(res)
        seed_records.append(
            {"seed": int(run_seed), "out_dir": str(res.out_dir), "metrics": dict(res.metrics)}
        )

    stats = summarize_seed_runs(seed_records)
    best = max(results, key=lambda r: r.metrics.get("task_completion", 0.0))
    summary_dir = Path(out_dir) if out_dir is not None else best.out_dir.parent
    write_json_atomic(
        summary_dir / "grpo_seed_study.json",
        {
            "seeds": list(cfg.seeds),
            "anchor_checkpoint": best.anchor_checkpoint,
            "runs": seed_records,
            "stats": stats,
        },
    )
    agg_metrics = {
        **best.metrics,
        "seed_count": float(len(results)),
        "task_completion_mean": stats.get("task_completion", {}).get("mean", 0.0),
        "task_completion_min": stats.get("task_completion", {}).get("min", 0.0),
    }
    return GRPOTrainResult(
        out_dir=best.out_dir,
        metrics=agg_metrics,
        train_runtime=time.perf_counter() - started,
        anchor_checkpoint=best.anchor_checkpoint,
        seed_results=seed_records,
    )
