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
| A documented command is missing | Compare `maatml --help` with the release boundary. Rollouts and tool-call training require 0.14.0 or later. |
| `Unknown ... plugin` | Check `plugins:` paths relative to the model folder and `maatml plugins` import failures. Install the plugin's dependencies in the same environment. |
| Missing image files | Generate the vision or VLM corpus with its `scripts/build_seeds.py`; image assets are not tracked. |
| Empty split or group warning | Inspect `dataset.group_by`, group sizes, pins and isolation. A fallback to row splitting loses group-level protection. |
| QLoRA rejected on CPU or MPS | Quantized training loads require CUDA and the `[ml,cuda]` extras. |
| A production gate fails after smoke training | Smoke checks establish execution only. Use declared `smoke.gates` for a rehearsal; assess production quality on a full run. |
| A full run misses a gate | Inspect `output/eval/` failures, counts and slices. Check the population and configured floors before changing training. |
| Benchmark changed in place | Preserve the existing benchmark, introduce a versioned file, and re-evaluate before deriving new floors. |
| The wrong checkpoint is evaluated | Use `maatml runs` and pass an explicit `--checkpoint RUN_ID` or path. |
| A run bundle lacks intermediate checkpoints | Pack with `--with-checkpoints`; use a consistent snapshot for interrupted-training recovery. |
| Non-loopback serving is refused | Configure `MAATML_SERVE_TOKEN` or `--auth-token` and send a bearer token on prediction requests. |
| Direct vLLM responses are not validated | Direct vLLM does not run the MaatML validator. Evaluate via the configured VLM predictor or integrate validation into the serving path. |

For recovery and checkpoint ownership, see [orchestration](orchestration.md).
For validation, retries and captures, see [serving](serving.md). For gate
provenance and changed populations, see [evidence](evidence.md).
