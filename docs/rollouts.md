# Rollouts

Candidate-ranking rollouts are available in **0.14.0** and later; **0.19.0**
adds the `free_text` protocol extension (`step_text`, `arithmetic_world`),
`filtered_imitation`, and environment-verified `grpo`. Check
`maatml rollout --help` in the environment that will run the job.

A rollout is a trained policy acting in an environment, one decision at a
time, until the environment says the episode is over. `maatml rollout` drives
it and stores what happened. The same driver serves three uses:

| Use | Flags | What it gives you |
| --- | --- | --- |
| Closed-loop check | none | Verified success per task family, which offline accuracy cannot show |
| DAgger collection | `--label` | The teacher's valid actions at every state the policy reached, ready to train on |
| Mixed collection | `--label --teacher-mix 0.3` | The same, with the teacher acting on a share of the decisions |

The environment decides what is legal, what a teacher accepts, and whether an
episode succeeded. The driver and the learner never compute a reward.

```bash
maatml rollout ~/models/my-policy --env line_world \
    --tasks '{"count": 50, "seed": 1}' --label --option extra_steps=4
```

Episodes land in `<model-dir>/output/episodes/<source>-r<round>/episodes.jsonl`
with `episodes.summary.json` beside them.

Episode completions stream progress during collection. Summaries include Wilson
intervals for measured success rates; offline decision accuracy and closed-loop
task success remain separate evidence.

## The environment protocol

`maatml.rollout_env/1` is small on purpose. Observations, candidates and tasks
are JSON objects the driver passes through without reading, apart from a
candidate's `id` and optional `valid`.

| Request | Reply |
| --- | --- |
| `hello` | `protocol`, `name`, `version`, `capabilities` (`label`, `tasks`, `free_text`) |
| `tasks` with a `query` | `tasks`, each with an `id` and a `family` |
| `reset` with `episode`, `task`, `options` | the first `observation` |
| `candidates` | `candidates`, each with an `id`; `"valid": false` masks one |
| `label` | `valid_ids`, or `null` with a `reason` when the teacher has none |
| `step` with `id` | `observation`, `applied`, `terminal`, `reward`, optional `info` |
| `step_text` with `text` and optional `metadata` | `observation`, `applied`, `terminal`, `reward`, optional `info` (requires `free_text` capability) |
| `verify` | `success`, `status`, `reward`, `detail` |
| `close` | nothing |

One environment hosts many episodes; the driver names each. `options` is how
a caller relaxes a task, for example a step budget, without editing the task.

There are two carriers.

**In process.** Register a factory and pass its name to `--env`:

```python
from maatml.registry import register_environment
from maatml.rollout import Hello, Labels, StepResult, Verification

@register_environment("my_env")
class MyEnv:
    def hello(self) -> Hello: ...
    def tasks(self, query): ...
    def reset(self, episode, task, options): ...
    def candidates(self, episode): ...
    def label(self, episode) -> Labels: ...
    def step(self, episode, candidate_id) -> StepResult: ...
    def verify(self, episode) -> Verification: ...
    def close(self, episode) -> None: ...
```

**Over stdio.** Any program, in any language, that reads one JSON request per
line and writes one reply per line: `{"ok": true, ...}` or
`{"ok": false, "error": "..."}`. Pass its command to `--env-command`. stderr is
left alone, so the program can log there. `maatml.rollout.serve(env)` is the
reference server for a Python environment.

## The policy

A rollout policy is the model folder's predictor with one extra method:

```python
def score(self, observation: dict, candidates: list[dict]) -> list[float]: ...
```

The driver takes the highest-scoring candidate the environment did not mark
invalid, and breaks ties on the smaller id, so a rollout is deterministic for
a fixed checkpoint. It records the log-probabilities over all candidates and
whether the policy's first choice was valid before masking (`top_valid`).

## What an episode records

`maatml.rollout_episode/1`: the task, the environment's name and version, the
policy's run id, the round, and per decision the observation, the candidates,
the chosen id, who acted (`policy` or `teacher`), `logp`, `teacher_valid` when
labelled, and whether the action applied. The outcome is the environment's
verdict. An episode the teacher could not label to the end is cut at that
state and says so in `outcome.truncated`.

## Training on what was collected

Turning episodes into rows is the format plugin's job, because only it knows
how an observation becomes model input. Hand the rows to `prepare_rows` as
`train_rows`: they join train after the split and never reach val or the
benchmark, so every round is measured on the same held-out rows as the first.
Collect them only from tasks your held-out sets exclude; an environment can
say so on the task (for example a `split` field), and the format plugin should
refuse episodes that do not carry it.

## Free-text rollouts

Tasks whose actions are generated text (tool calls, structured answers, code or
patches) use the `free_text` capability on `maatml.rollout_env/1`. An
environment advertising `"free_text"` in `hello().capabilities` implements
`FreeTextEnvironment` (`step_text(episode, text, metadata=None)`) over either
the in-process carrier or stdio JSON lines (`{"op": "step_text", "episode":
..., "text": ..., "metadata": ...}`).

A generative policy implements `FreeTextPolicy`:

```python
from maatml.rollout import GeneratedAction, rollout_free_text

class MyGenerativePolicy:
    def generate(self, observation: dict, *, n: int = 1, seed=None) -> list[GeneratedAction]:
        ...
```

`rollout_free_text(env, policy, tasks, config=...)` drives each episode through
`reset -> generate -> step_text -> verify -> close`. Every emitted episode
carries `action_mode: "free_text"`, `split` (from `task["split"]`), and a
deterministic SHA-256 `trace_digest` over the sequence of chosen actions.
Rewards and terminal verification remain strictly inside the environment.

## Verified-output training: filtered imitation and GRPO

Verified free-text rollouts feed two built-in trainers:

- **`filtered_imitation`** (`maatml.training.filtered_imitation`):
  `collect_and_filter` and `filter_verified_episodes` retain strictly episodes
  verified as passing by the environment (`outcome.success == True`), reject
  tasks from held-out splits (`exclude_splits: ("test", "blind", "held_out")`),
  deduplicate identical trajectories by `(task_id, trace_digest)`, optionally
  cap per-task counts (`max_per_task`), write `filtered_imitation_manifest.json`
  and hand the verified corpus (plus optional `mix_seed_train` rows) to
  `train_sft`. Configure with `architecture: filtered_imitation` and
  `extensions.filtered_imitation`.
- **`grpo`** (`maatml.training.grpo`, `maatml.training.grpo_config`):
  Group-Relative Policy Optimization backed by `EnvironmentRewardBridge`. For
  each task the trainer samples `training.group_size` completions, evaluates
  every completion in an isolated `reset -> step_text -> verify -> close`
  episode inside the environment process, normalizes rewards via
  `compute_group_advantages` (zero-variance groups emit zero advantage), and
  updates the policy with a clipped surrogate loss plus an unbiased non-negative
  KL penalty against a frozen `anchor_checkpoint` (`kl_beta`). Multi-seed runs
  accept `training.seeds` (at least 2 seeds), and `evaluate_grpo_promotion`
  gates promotion on held-out non-regression plus a strict gain on the
  completion metric.

## Built-in reference environments

- **`line_world`** (candidate-ranking): walk an integer line to a target, then
  finish. Includes a scripted teacher (`label`), a step budget, an invalid
  candidate, and an independent verifier.
- **`arithmetic_world`** (`free_text`): asks for the exact integer result of
  `a + b` or `a * b` (plain integer or `{"answer": <int>}`, stripping optional
  `<think>` / `<thought>` blocks). Only verified correct answers earn reward
  `1.0`; well-formed but wrong answers earn `0.0`, so format compliance alone
  is never rewarded. Both environments work in-process and over stdio for tests
  and wiring checks.
