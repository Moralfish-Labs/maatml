# Orchestration and recovery

MaatML is the training component in a larger execution stack. Its CLI runs on
the machine that holds the model folder, whether that is a workstation,
notebook GPU, or GPU VM. The framework does not provision those machines or
schedule arbitrary jobs.

| Owner | Responsibility |
| --- | --- |
| MaatML | Prepare, train, evaluate, gate, export, verify, serve, and record model evidence |
| Model folder | `model.yml`, datasets, task plugins, validators, metrics and benchmarks |
| External orchestrator | Scheduling, worker provisioning, credentials, transfers, logs, retries, monitoring and cleanup |
| Storage service | Durable copies of source bundles, checkpoints and result bundles |

## Invoke the lifecycle

For a complete local lifecycle, use `maatml run`. For a remote job, the
orchestrator stages a model folder and its declared inputs, installs the
required framework revision and extras, then invokes the same CLI on the
worker. Recipes belong in `model.yml`; task-specific orchestration should not
reimplement trainers, gates, or run records.

```bash
maatml validate /workspace/models/my-task
maatml audit /workspace/models/my-task
maatml plan /workspace/models/my-task
maatml run /workspace/models/my-task --device cuda \
  --progress-file /workspace/progress.json
```

`validate`, `audit` and `plan` do not train. `run` can spend compute: it walks
prepare, train, gated evaluation, export and verification. Source-mutating
operations such as teacher generation and rollout collection are separate,
explicit commands. An orchestrator should preserve non-zero exit codes and
stop dependent stages on failure.

Before allocating a worker, freeze the recipe, framework/plugin revisions,
input hashes, seed and benchmark. Preserve these with the run's evidence.
Smoke checks establish pipeline operation; their gate evidence is marked as
smoke-tier and does not establish production quality.

## Structured progress reporting

Pass `--progress-file PATH` to `maatml train` or `maatml run`, or export
`FLOW_JOB_PROGRESS_FILE=PATH` in the job environment. MaatML writes atomic
camelCase `JobProgress` JSON snapshots from rank 0 only (so multi-GPU ranks
never race on the target file):

```json
{
  "checkpointAgeSecs": 42,
  "epoch": 1,
  "heartbeatAt": "2026-11-20T16:00:00Z",
  "metricName": "loss",
  "metricValue": 0.412,
  "runId": "20261120-154500-my-task",
  "stage": "train",
  "step": 250
}
```

Supervisors can poll this file directly for heartbeats, active stage
(`prepare`, `train`, `checkpoint`, `evaluate`, `export`, `verify`), step,
latest finite metric, checkpoint age and `artifactUri` without scraping
terminal output. Set `training.disable_tqdm: true` and `PYTHONUNBUFFERED=1`
alongside `--progress-file` when streaming stdout logs.

## Signal handling and clean interruption

`TrainingStopController` coordinates `SIGTERM` and `SIGINT` across training
runs:

- Inside the training loop, the first `SIGTERM` or `SIGINT` requests a deferred
  stop so the Trainer flushes a checkpoint at the step boundary, exits cleanly
  and records run status `interrupted` in `output/runs.jsonl`.
- A second signal interrupts immediately.
- Outside the training loop (during model load or setup), `SIGTERM` interrupts
  immediately so startup cannot hang until an orchestrator kill timeout.

## Resume an interrupted trainer

Recovery requires the trainer's checkpoint state, not just exported weights.
Keep the interrupted model folder's `output/runs.jsonl`, original run directory,
mid-run checkpoint, recipe, tokenizer and prepared splits. Checkpoint contents
and resume support depend on the registered trainer; verify its optimizer,
scheduler, scaler and progress state as applicable.

```bash
maatml train /workspace/models/my-task --device cuda --seed 44 \
  --resume /workspace/models/my-task/output/checkpoints/RUN_ID/checkpoint-33000

# Or resume through the fixed lifecycle runner:
maatml run /workspace/models/my-task --device cuda --resume auto \
  --progress-file /workspace/progress.json
```

`RUN_ID` and `checkpoint-33000` are placeholders for a verified checkpoint.
For an exact recovery request, choose an explicit path rather than `auto`,
which selects the latest incomplete (`running` or `interrupted`) run carrying a
saved checkpoint. Verify recipe and data compatibility before launch and
confirm the restored step in the trainer log. Do not silently replace a missing
checkpoint with fresh training or infer successful continuation from the
absence of errors. Exact stochastic replay depends on the trainer's saved
state; `--resume` alone is not a universal bit-for-bit guarantee.

## Live checkpoint snapshots

A remote snapshot used for recovery must retain complete mid-run checkpoints.
`runs --pack` omits nested `checkpoint-*` directories by default;
`--with-checkpoints` includes them, and neither option establishes that a live
checkpoint is complete while training is still writing.

For live backups on POSIX hosts, enable `training.checkpoint_snapshots: true`
for single-process SFT runs (which writes a hash-bound
`maatml-checkpoint-ready.json` marker after `on_save`, supporting both
single-file and sharded safetensors weights) or use a policy trainer that
writes `training_state.pt` last, then run `maatml snapshot` alongside training:

```bash
maatml snapshot /workspace/models/my-task \
  --started-after /workspace/launch.marker \
  --writer-lock /workspace/train.lock \
  --min-step 200 \
  --out /workspace/backups/ckpt-200.tar.gz \
  --compress-level 1
```

`maatml snapshot` waits for a complete checkpoint at or after `--min-step`,
copies and verifies the files before archiving (so concurrent checkpoint
pruning cannot yield a torn archive), and writes the portable bundle atomically
at gzip level 1 by default for fast backups inside billed shutdown windows. It
never stops training, uploads data, deserializes optimizer state or weakens
gates. Snapshots require POSIX writer locks and are refused on Windows.

## Bring a completed run home

After training, evaluate the explicit run against the intended populations and
gates. Preserve prediction caches when downstream analysis needs per-row
evidence. Use the CLI's portable run bundle to move the checkpoint, reports,
caches, record and environment together:

```bash
# On the worker, with measured production gates configured:
maatml evaluate /workspace/models/my-task --checkpoint RUN_ID --split test --cache --gate
maatml runs /workspace/models/my-task --pack RUN_ID

# Transfer the emitted bundle using the orchestrator/storage service, then:
maatml runs /local/models/my-task --adopt /downloads/run-bundle.tar.gz
maatml report /local/models/my-task
```

Adoption checks bundle integrity and receiving recipe compatibility. A mismatch
needs investigation; forcing adoption is not the normal transfer path. Where a
baseline exists, `ship-check` adds regression and population checks to absolute
gates. Completing training or transferring a bundle is not a release verdict.

The orchestrator decides when durable results are confirmed and a worker can
be stopped. Monitoring should report stage, checkpoint, exit status and gate
verdict separately so evaluation failures cannot be mistaken for lost training.

See [Evidence](evidence.md) for measured floors and release decisions, and
[Rollouts](rollouts.md) for environment-verified policy behavior.
