# The data flywheel

Each source operation changes the corpus explicitly. Configure
`evaluation.validator` to check labels against the same contract used in
evaluation and serving. `datagen` fails closed unless `--allow-ungated` is set;
`ingest` validates only when a validator is configured. `mint` deliberately
retains a failing completion as the rejected half of each preference pair.

These are deliberate operations, not steps of `maatml run`. They change the seed
corpus, and that is exactly what makes `prepare` stale on the next run:

```
datagen / ingest / distill / mint / (reviewed capture)
        │  append validator-gated rows to the seed corpus
        ▼
maatml run   # prepare is now stale, so the loop retrains
```

## Policy-collected rows

`maatml rollout --label` collects teacher labels at states reached by a
checkpoint. The task's format plugin turns these episodes into training rows.
`prepare_rows(..., train_rows=...)` adds them only to train and refuses groups
already present in val or test. This keeps collected experience from silently
moving the evaluation population. See [Rollouts](rollouts.md) for the protocol,
teacher mixing and independently verified outcomes.

## `maatml datagen`: generate rows

Runs a registered generator (or a teacher) and keeps only rows the validator
accepts. Fails closed when no validator is configured unless you pass
`--allow-ungated`. See [the lifecycle](lifecycle.md) for the gating contract.

## `maatml distill`: label a prompt pool

Where `datagen` invents whole rows, `distill` starts from prompts you already
have and asks a teacher only for the label. Every response is gated before it
enters the corpus, so a wrong label is dropped rather than trusted.

```bash
maatml distill <model> --prompts prompts.jsonl
maatml distill <model> --replay          # reproduce the corpus offline
```

Accepted rows carry provenance (teacher model and revision, prompt hash,
source, family), and rejections are kept in a report. Teacher responses are
recorded in a cache keyed on the prompt hash plus the teacher's model and
revision, so `--replay` reproduces exactly the same accepted corpus with no
network, and a different teacher never silently reuses another's labels. The
cache opens with a provenance header (teacher, prompt pools by sha256, the
benchmark version it was recorded against). A pool prompt that already sits
in `benchmark_samples`, `blind_samples` or the prepared val / test split is
refused before any teacher call: its label would enter training as a
memorised answer to a question the benchmark asks. Point it at a pool with
`--prompts`, or declare a `distill:` section in `model.yml`:

```yaml
distill:
  prompt_source: datasets/distill/prompts.jsonl
  teacher_model: gpt-4o-mini
  teacher_revision: "2026-07"
  cache: datasets/distill/cache.jsonl
  system_prompt_file: datasets/distill/brief.txt   # instead of an inline system_prompt
  target_format: text                              # json (default) parses; text does not
```

The teacher is any OpenAI-compatible endpoint, named by
`MAATML_TEACHER_BASE_URL` and `MAATML_TEACHER_API_KEY`, so serving one locally
is a base URL rather than a code path.

`target_format` decides what a reply must be *before* the validator sees it. The
default `json` parses the response and drops what will not parse, which is right
when the target is a document. Set `text` when it is not, such as a sentence and
then a call object, or a rendered patch: the raw reply is handed to the
validator to judge instead. Left at `json`, such a label is refused as unparseable and the
gate that decides never sees it. `system_prompt_file` is for a briefing too long
to sit inline in `model.yml`, such as one carrying a real system prompt and tool
catalogue.

The [triage example](examples/support-ticket-triage.md) ships a prompt pool and
a recorded cache, so `maatml distill examples/support-ticket-triage --replay`
runs offline. One recorded label routes a billing ticket to the wrong team; the
routing contract rejects it, so it never reaches the seeds.

## `maatml ingest`: import external rows

Maps external columns into the seed shape, optionally sanitizes, and validates
gold targets when a validator is configured. It also guards the capture loop
below: a `serve_capture` row is refused unless a reviewer approved it.

`--video PATH` turns a sidecar JSONL plus a video file into image rows: each
sidecar object names a frame (`frame`, `timestamp_ms`, or `t`) and the gold
target; ffmpeg writes PNGs under `datasets/samples/images/` and the request
field becomes that relative path. Annotation dialects (MEVA KPF, COCO VID)
stay in the sidecar a plugin writes — core does not parse them.

```bash
maatml ingest <model> --input boxes.jsonl --video clip.mp4 --map expected=boxes
```

## `maatml mint`: preference pairs for DPO / ORPO

Turns candidate completions into `{prompt, chosen, rejected}` pairs. For each
prompt the validator splits the candidates into pass and fail; a prompt with
both yields one pair. So a minted pair means "this completion passes the
contract and that one does not", not a hand-labelled guess.

```bash
maatml mint <dpo-model> --input candidates.jsonl
```

Input is JSONL of `{prompt, candidates: [completion, ...]}`. Pairs append to the
preference seed corpus, stamped `source: mint`.

## Reviewed capture: learn from production

`maatml serve --capture` records real traffic for review. A captured prediction
is **not** automatically training data, it is a proposal a human or teacher must
correct and approve first:

```bash
maatml serve <model> --auth-token "$TOKEN" --capture captures.jsonl
# review captures.jsonl: fix the target, set "approved": true on keepers
maatml ingest <model> --input captures.jsonl   # refuses unapproved rows
maatml run <model>                              # retrains on the new seeds
```

Capture requires the serve auth token (an open capture endpoint is an unbounded
write sink and a way to poison the corpus), the file is size-capped, and
`ingest` refuses any row still marked unapproved. See
[serving](serving.md#capture-and-the-reviewed-flywheel) for the endpoint side.
