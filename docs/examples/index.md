# Examples

Four reference models share the identical folder layout and CLI, from a
one-command support-ticket triage to a vLLM-servable vision-language model.
Each is a standalone folder under `examples/` with its own `model.yml`. Clone
the repository and install its matching source revision for these walkthroughs.
The [quickstart](../getting-started.md) covers the virtual environment. Commands
below run from the repository root; replace `RUN_ID` or `<run_id>` with a real
run listed by `maatml runs`.

| Model | Task | Architecture | Base |
|-------|------|--------------|------|
| [Support Ticket Triage](support-ticket-triage.md) | triage → JSON | `causal_sft` (LoRA) | Qwen3-0.6B |
| [Vision VLM](vision-vlm.md) | describe a scene image | `vlm_sft` (vLLM-servable) | SmolVLM-256M-Instruct |
| [Vision](vision.md) | scene + detect + pose | `vision_multitask` | MobileNetV3-Large |
| [Vision Describer](vision-describer.md) | caption from vision JSON | `seq2seq` | flan-t5-small |

New to MaatML? [Support Ticket Triage](support-ticket-triage.md) is the
shortest path end to end, see the [quickstart](../getting-started.md).
For the multimodal path, start with [Vision](vision.md) (trains a checkpoint)
and [Vision Describer](vision-describer.md) (reads its output and captions it).
[Vision VLM](vision-vlm.md) folds both into one vision-language model, servable
directly by vLLM.

Any directory with a valid `model.yml` works the same way. Scaffold a new one
with:

```bash
maatml scaffold ~/models/my-task --architecture causal_sft --name my-task
```

See the [plugin author guide](../plugins.md) for wiring up a validator,
metrics, and a predictor for your own task.
