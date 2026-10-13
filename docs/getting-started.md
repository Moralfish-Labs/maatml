# Get started

The fastest path from a clean checkout to a served model uses
[support-ticket-triage](examples/support-ticket-triage.md): a LoRA fine-tune of
Qwen3-0.6B that turns a raw support ticket into `{priority, category, team,
summary}` JSON. Runtime depends on downloads and hardware; full training is
best run on a GPU. Use the smoke tier to check the pipeline first.

## 1. Install

```bash
git clone https://github.com/Moralfish-Labs/maatml.git
cd maatml
python -m venv .venv
source .venv/bin/activate
pip install -e ".[ml]"
```

This walkthrough installs the checkout so its examples and framework agree.
For a released installation with your own model folder, use
`pip install "maatml[ml]"` instead. Use 0.14.0 or later for rollouts and tool-call chat training.
On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

`[ml]` pulls in the training stack (`torch`, `transformers`, `peft`, …). If you
only need the CLI and library, no training, `pip install maatml` is enough.

## 2. Build the train / val / test splits

```bash
maatml --version
maatml validate examples/support-ticket-triage/
maatml prepare examples/support-ticket-triage/
```

This reads the seed data already committed at
`datasets/samples/seed_samples.jsonl` and writes
`output/prepared/{train,val,test}.jsonl` under the model folder. Nothing is
downloaded yet.

## 3. Smoke-train the pipeline

```bash
maatml train examples/support-ticket-triage/ --smoke
```

`--smoke` runs a couple of steps on a slice of data so you can confirm the
tokenizer, base model, LoRA adapter, and trainer all wire up correctly before
spending real compute. This step downloads the base model
(`Qwen/Qwen3-0.6B`, ~1.2 GB) from the Hugging Face Hub on first run.

## 4. Train the example as configured

```bash
maatml train examples/support-ticket-triage/
```

Checkpoints land under `output/checkpoints/<run_id>/`. List every run with
`maatml runs examples/support-ticket-triage/`, and compare their metrics with
`maatml runs examples/support-ticket-triage/ --compare`.

Unlike step 3, this is a real run: the example ships 600 seed rows and trains a
LoRA adapter for `epochs: 3` with `max_steps: -1`, so expect minutes rather than
seconds, and a GPU or Apple silicon rather than a laptop CPU. Step 3's `--smoke`
tier stays available whenever you only want to prove the wiring.

## 5. Evaluate against the gates

```bash
maatml evaluate examples/support-ticket-triage/ --gate
```

`--gate` exits non-zero if `evaluation.gates` in `model.yml` aren't met, the
same check you'd wire into CI. Drop `--gate` to see the scores on their own.

A red gate is the contract working, not a broken quickstart: it says this
checkpoint has not earned the threshold yet. Train longer, grow the corpus
(`maatml datagen` or `maatml ingest`), and inspect failures by metric and population before changing the recipe. The thresholds shipped with this example are marked
pending re-measurement in `model.yml`, so treat them as a starting point rather
than a target to tune against. When you have a run you trust, let
`maatml gates derive <model-dir> --run RUN --write` set them from its report
instead of typing numbers (see [Evidence](evidence.md)).

## 6. Export and verify

```bash
maatml export examples/support-ticket-triage/ --format safetensors
maatml verify examples/support-ticket-triage/output/export/RUN_ID
```

Replace `RUN_ID` with the export directory reported by the command. Verification
checks file integrity; it does not establish model quality. Use explicit
`--checkpoint RUN_ID` on evaluate, export and serve when comparing several runs.

## 7. Serve it

```bash
maatml serve examples/support-ticket-triage/
```

In another terminal:

```bash
curl -s localhost:8080/predict \
  -H 'content-type: application/json' \
  -d '{"request": "Cannot log in since this morning, urgent, paying customer"}' | jq
```

Add `?validate=1` to the URL to also run the task's validator inline on the
response.

## Run the fixed pipeline

Once the individual stages are familiar, `maatml run examples/support-ticket-triage/`
walks prepare, train, gated evaluate, export and verify, skipping fresh stages.
`maatml run examples/support-ticket-triage/ --smoke` uses the example's smoke
gates to check that pipeline. Serving is started separately.

## What just happened

One `model.yml` drove every stage above (prepare, train, evaluate, export, verify, serve) through the same CLI. See [the validator-gated lifecycle](lifecycle.md) for why
that matters, and the [plugin author guide](plugins.md) for how to point this
at your own task instead of support-ticket triage.

## Next steps

- When a command misbehaves, start with
  `maatml audit examples/support-ticket-triage/` (add `--json` for machines).
  It reports missing extras, device/profile mismatches, unregistered plugins,
  empty seeds, and gate-key typos without running training.
- Browse the [other examples](examples/index.md): vision, a
  vLLM-servable vision-language model, and a seq2seq describer share this
  same folder layout.
- Scaffold your own task:
  `maatml scaffold ~/models/my-task --architecture causal_sft --name my-task`.
- `maatml --help` and `maatml <command> --help` document every flag.
