# Troubleshooting

Start with the installed version and the model folder's configuration:

```bash
maatml --version
maatml validate /path/to/model
maatml audit /path/to/model
maatml --debug validate /path/to/model
```

Validation and audit load trusted model plugins. `validate --no-plugins` checks
schema and paths without importing those plugins. `audit --json` emits machine
readable findings; warnings alone do not fail the command.

| Symptom | Check and next step |
| --- | --- |
| A documented command is missing | Compare `maatml --help` with the release boundary. `served_http`/`serve_parity`, `reasoning_loss`, `free_text` rollouts, `filtered_imitation`, `grpo`, `--progress-file`, and `run --resume` require **0.19.0** (`adapter` export requires 0.18.0+, `snapshot` requires 0.16.0+, candidate rollouts and tool-call training require 0.14.0+). |
| `RowTooLongError` during causal SFT | A rendered row exceeded `training.max_input_tokens`. MaatML refuses over-long rows instead of left-truncating system prompts and tool declarations. Drop or shorten the row, or raise `training.max_input_tokens`. |
| `training.frozen_module_prefixes` aborts training | A submodule declared frozen had trainable parameters or mutated tensors/buffers. Narrow `training.lora.target_modules` (for example with a regex confined to the language tower) and verify that frozen modules stay in eval mode without buffer updates. |
| `export --format adapter` fails layout check | The adapter rank, tensor names or tensor shapes differ from `extensions.adapter.spec`, or the checkpoint was not trained with `training.lora.save_mode: adapter` or `both`. |
| `maatml verify` reports `unlisted file` | A file exists inside the export directory without a `manifest.json` entry. Remove stray files or record intentional out-of-band artifacts with `maatml manifest amend`. |
| `served_http` / `serve_parity` refuses server identity | Check that the running server's `/props` `model_path` and `build_info` match `extensions.serve_parity` and that `<export_dir>/manifest.json` carries passing non-smoke `gate_evidence`. |
| `maatml snapshot` refused on Windows | Live snapshots require POSIX `fcntl` writer locks and are supported on macOS and Linux. |
| `Unknown ... plugin` | Check `plugins:` paths relative to the model folder and `maatml plugins` import failures. Install the plugin's dependencies in the same environment. |
| Missing image files | Generate the vision or VLM corpus with its `scripts/build_seeds.py`; image assets are not tracked. |
| Empty split or group warning | Inspect `dataset.group_by`, group sizes, pins and isolation. A fallback to row splitting loses group-level protection. |
| QLoRA rejected on CPU or MPS | Quantized training loads require CUDA and the `[ml,cuda]` extras. |
| A production gate fails after smoke training | Smoke checks establish execution only. Use declared `smoke.gates` for a rehearsal; assess production quality on a full run. |
| A full run misses a gate | Inspect `output/eval/` failures, counts and slices. Check the population and configured floors before changing training. |
| Benchmark changed in place | Preserve the existing benchmark, introduce a versioned file, and re-evaluate before deriving new floors. |
| The wrong checkpoint is evaluated | Use `maatml runs` and pass an explicit `--checkpoint RUN_ID` or path. |
| A run bundle lacks intermediate checkpoints | Pack with `--with-checkpoints`, or use `maatml snapshot` with `training.checkpoint_snapshots: true` for live recovery backups. |
| Non-loopback serving is refused | Configure `MAATML_SERVE_TOKEN` or `--auth-token` and send a bearer token on prediction requests. |
| Direct vLLM responses are not validated | Direct vLLM does not run the MaatML validator. Evaluate via the configured VLM predictor or `served_http` / `serve_parity`, or integrate validation into the serving path. |

For recovery and checkpoint ownership, see [orchestration](orchestration.md).
For validation, retries and captures, see [serving](serving.md). For gate
provenance and changed populations, see [evidence](evidence.md).
