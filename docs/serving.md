# Serving & deployment

Deployment depends on the architecture and installed backend. The built-in
HTTP server loads a model-folder predictor; the Anthropic proxy forwards to
an external runtime. ONNX and vLLM consume compatible exports. A direct
external runtime does not automatically apply MaatML validators.

Serving and target compilation are pluggable: `maatml serve --server NAME` and
`maatml compile --target NAME` dispatch through the `server` / `compiler`
registries. The built-in `http` server stays the default for development; native
backends can register as plugins and keep their hot path in the native engine.
`http` and `anthropic` are bundled servers. The `sip_tensorrt` compiler and
`sip_deepstream` server below are illustrative external plugin names; install
and register those plugins before using them. `maatml plugins` lists available
registrations.

## 0. Target compilation

```bash
maatml compile <export-dir> --target sip_tensorrt --out /opt/sip/models/person \
    --option precision=fp16 --option profile=4cam --require-gated
```

Compilers receive the portable export (ONNX, GGUF, HF, …) and write a
device-specific bundle plus `target_manifest.json` (source identity, options,
`promotion_eligible` / `promotion_reason` from `manifest.gate_evidence`). Core
never assumes ONNX.

`--require-gated` refuses before the plugin runs when `gate_evidence` is
missing, `passed` is not true, or `smoke_gated` is true, so a rehearsal cannot
become a device artifact.

## 1. `maatml serve`, built-in HTTP API

```bash
maatml serve <model-dir> --server http          # default
maatml serve <model-dir> --server sip_deepstream \
    --server-option socket=/run/sip/perception.sock \
    --server-option deployment=/opt/sip/models/person
```

A dependency-light server (Python stdlib, no FastAPI/uvicorn) that loads the
predictor once and exposes:

| Route | Purpose |
|-------|---------|
| `GET /health` | liveness + identity |
| `GET /info` | model summary + packaging hints, and which of enforce / auth / capture / retries are on |
| `POST /predict` | dataset-shaped JSON row → prediction |
| `POST /predict?validate=1` | prediction **plus** the inline validator result |

```bash
maatml serve examples/support-ticket-triage/ --port 8080
```

It is intentionally simple, a single model, one request at a time, which keeps
it light enough for edge / single-instance use (including Jetson/JetPack). For
higher throughput, use the vLLM path below.

### Gating live inference

`--enforce` turns the validator into a gate: a `/predict` whose output fails
validation returns **HTTP 422** instead of 200, so the same contract that gated
training and evaluation also gates production. Without `--enforce`, `?validate=1`
annotates the response but never blocks it.

```bash
maatml serve examples/support-ticket-triage/ --enforce
```

`--max-retries N` softens that: on a validation failure the server feeds the
error back to the model and re-asks, up to N times, before giving up. Every
response reports `attempts` and `retries`, and a request still failing after the
budget returns 422 with the retry count, so retries are always visible, never
silent. (Retry-with-feedback works for any predictor. Constrained decoding that
enforces the schema *during* generation is a planned follow-up: it needs a serve
extra and only applies to generative architectures.)

### Auth

`--auth-token TOKEN` (or `MAATML_SERVE_TOKEN`) requires `Authorization: Bearer
TOKEN` on `/predict`. It is compared in constant time, and it is mandatory both
for `--capture` (below) and for any non-loopback bind: serving on `0.0.0.0`
without a token is refused before the socket is opened, because anyone who can
reach the port could query the model. An empty token is refused too, so an
unset variable cannot silently produce a server whose auth is a formality.

Pass `--allow-unauthenticated` to bind a non-loopback address anyway on a
trusted network; it downgrades the refusal to a warning.

```bash
maatml serve examples/support-ticket-triage/ --host 0.0.0.0 --auth-token "$TOKEN"
```

### Capture and the reviewed flywheel

`--capture PATH` appends served predictions to a JSONL for later review. A
captured row is **not** gold: it carries `approved: false` / `needs_review:
true`, and the file is row/byte capped so an unattended server cannot fill the
disk. Capture requires `--auth-token`.

Custom servers (`--server sip_deepstream`, …) use the same writer:
`maatml.serve.open_capture(model_def, capture_path, auth_token=…)` then
`LifecycleServer.record_capture(row, output, raw)` (or `writer.record`). The
CLI still passes `--capture` and `--auth-token` through `dispatch_server`.

The request is written through the model's declared `dataset.sanitize` tags, so
a model that sanitizes its corpus sanitizes its captures by the same rules. A
model that declares no tags captures the request verbatim, which is worth
knowing before pointing `--capture` at live traffic.

The retrain loop is deliberate at every step:

```bash
maatml serve <model> --auth-token "$TOKEN" --capture captures.jsonl
# ... traffic accumulates in captures.jsonl ...
# review: fix the target and set "approved": true on the rows worth keeping
maatml ingest <model> --input captures.jsonl   # refuses any row not approved
maatml run <model>                              # the new seeds make prepare stale
```

`maatml ingest` refuses a `serve_capture` row unless a reviewer set `approved:
true` (dropping the flag does not sneak it through), so a raw model prediction
can never become training data without a human or teacher approving it.

### `--server anthropic`, the Messages API in front of llama.cpp

A translating proxy rather than a runtime: llama.cpp holds the weights and
renders the chat template, and this supplies only the protocol, so any Anthropic
client reaches a fine-tuned local model without knowing it is one.

```bash
llama-server --jinja -m model.gguf --port 8081 &
maatml serve <model-dir> --server anthropic \
    --server-option upstream=http://127.0.0.1:8081 --port 8100
```

| option | default | meaning |
|---|---|---|
| `upstream` | `http://127.0.0.1:8081` | the OpenAI-compatible server holding the weights |
| `model` | the folder's `model_id` | the name echoed back in the response |
| `timeout` | `600` | seconds to wait on the upstream |
| `tool_style` | `native` | `native` declares tools upstream; `inline` carries them in message text |
| `call_retries` | unset | how many times a turn owing a tool call is re-asked |

`tool_style=inline` serves a model fine-tuned on the text protocol in
`maatml.wire.inline_tools` rather than on a chat template's tool syntax: the
catalogue travels as one line per tool in the last user turn, a call is a JSON
object ending the assistant turn, and a result returns as
`<tool_response>…</tool_response>`. The upstream then never sees a tool at all,
so `--jinja` buys nothing under it. Because one module renders both the corpus
and the wire, the string trained on is the string served.

`call_retries=N` makes a turn that follows a user message owe a call: a reply
carrying none is re-asked up to N times and then says plainly that nothing ran.
It never applies to a turn answering a tool result, because summarising one is
how a loop ends, and unset leaves prose allowed.

`--enforce` and `--max-retries` gate this backend the way they gate `http`.
Every reply is collected before any of it is sent, for the same reason
`call_retries` collects one: whether a reply is valid is only known once it
ends. The model folder's validator judges the collected text; a rejection goes
back to the model as the failed reply plus the validator's own message, re-asked
up to `--max-retries` times, and a reply that never passes is replaced under
`--enforce` by a plain statement of what was refused and that nothing ran.
Without `--enforce` the retries still run and the last reply stands. Requires
`tool_style=inline`, because the text the validator was trained against is the
inline transcript. The Anthropic proxy does not write capture rows; attach
`open_capture` in a custom backend if that flywheel is needed there.

## 2. ONNX / edge (vision)

`vision_multitask` exports to **ONNX** for `onnxruntime` (CPU on Mac) or the
TensorRT execution provider on Jetson:

```bash
maatml export examples/vision/ --format onnx
```

## 3. vLLM, vision-language models

VLM checkpoints (`vlm_sft`) export as an HF-format directory that **vLLM loads
directly** for OpenAI-compatible, higher-throughput serving:

```bash
vllm serve examples/vision-vlm/output/export/<run_id>
```

The [`vision-vlm` example](https://github.com/Moralfish-Labs/maatml/tree/main/examples/vision-vlm)
ships `serve_vllm.sh` (Linux / Jetson container) and an OpenAI-compatible client.
Its evaluator can even score against a live vLLM endpoint, set
`MAATML_VLLM_ENDPOINT` and the predictor switches to the chat-completions API
(the image is sent as a base64 data URL).

```bash
pip install "maatml[vllm]"     # Linux-only
export MAATML_VLLM_ENDPOINT=http://127.0.0.1:8000
maatml evaluate examples/vision-vlm --gate
```

## Adding files to an export

If an external tool writes an additional artifact inside the export directory,
register its checksum before verifying the bundle:

```bash
maatml manifest amend /path/to/export /path/to/export/model-q5.gguf --format gguf
maatml verify /path/to/export
```

Amendment preserves existing gate evidence. It does not evaluate or certify the
added artifact; evaluate the deployed representation separately.

## Verifying an export

Every export writes a `manifest.json`. `maatml verify <export-dir>` recomputes
the sha256 of each file listed in the manifest and checks it against the
recorded value, so you can catch truncated, corrupted, or partially copied
artifacts before you ship. This detects accidental corruption, not tampering:
anyone who can rewrite a file can also recompute its hash in `manifest.json`, so
treat `verify` as an integrity check, not a signature.

```bash
maatml export examples/support-ticket-triage/
maatml verify examples/support-ticket-triage/output/export/<run_id>
```
