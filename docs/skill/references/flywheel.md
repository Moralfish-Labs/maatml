# The data flywheel: growing a corpus that stays gated

Each source operation changes the corpus explicitly. Configure
`evaluation.validator` to check labels against the same contract used in
evaluation and serving. `datagen` fails closed unless `--allow-ungated` is set;
`ingest` validates only when a validator is configured. `mint` deliberately
retains a failing completion as the rejected half of each preference pair.

None of these are steps of `maatml run`. They change the seed corpus, which is
exactly what makes `prepare` stale on the next run:

```
datagen / distill / ingest / mint / (reviewed capture)
        |  append validator-gated rows to the seed corpus
        v
maatml run          # prepare is stale, so the loop retrains
```

## Contents

- [Choosing an operation](#choosing-an-operation)
- [datagen: generate whole rows](#datagen-generate-whole-rows)
- [distill: label a prompt pool](#distill-label-a-prompt-pool)
- [ingest: import external rows](#ingest-import-external-rows)
- [mint: preference pairs](#mint-preference-pairs)
- [Reviewed capture](#reviewed-capture)
- [Diagnosing teacher rejections](#diagnosing-teacher-rejections)

## Choosing an operation

| You have | Use |
|---|---|
| nothing but a contract | `datagen` |
| prompts, but no labels | `distill` |
| rows from somewhere else | `ingest` |
| candidate completions to rank | `mint` |
| real traffic worth learning from | `serve --capture`, then review, then `ingest` |

## `datagen`: generate whole rows

Runs a registered generator (or a teacher) and keeps only rows the validator
accepts. It **fails closed** when no validator is configured unless you pass
`--allow-ungated` — ungated synthetic data teaches a contract nobody checks.

## `distill`: label a prompt pool

Where `datagen` invents whole rows, `distill` starts from prompts you already
have and asks a teacher only for the label. Every response is gated before it
enters the corpus, so a wrong label is dropped rather than trusted.

```bash
maatml distill <model-dir> --prompts prompts.jsonl
maatml distill <model-dir> --replay      # reproduce the corpus offline, no network
```

Accepted rows carry provenance (teacher model and revision, prompt hash, source,
family); rejections are kept in a report next to the corpus, which is worth
reading — a cluster of identical rejections is usually a briefing problem, not a
teacher problem.

Responses are cached on prompt hash **plus teacher model and revision**, so
`--replay` reproduces exactly the same accepted corpus and a different teacher
never silently reuses another's labels. The cache's first line is a
provenance header — teacher, every prompt pool it labelled (by sha256), and
the benchmark version current when it was recorded — and the distill card
repeats it.

`distill` refuses a pool that overlaps a held-out population before any
teacher call: a prompt whose text is in `benchmark_samples`, `blind_samples`
or the prepared val / test split, or a pool row tagged with a pinned group
(`dataset.pins`). A teacher label for one of those would enter training as
a memorised answer to a question the benchmark asks. Fix the pool; the
held-out manifests are not prompt sources.

The teacher is any OpenAI-compatible endpoint, named by
`MAATML_TEACHER_BASE_URL` and `MAATML_TEACHER_API_KEY` — so serving one locally
is a base URL, not a code path. Export credentials in the process environment; MaatML does not automatically
load a `.env` file. Do not commit credentials.

Configure it in `model.yml` under `distill:` rather than on the command line, so
the run record can say what labelled the corpus. See `model-yml.md`, especially
`target_format`: left at `json`, a non-JSON label is refused as unparseable and
the gate that decides never sees it.

## `ingest`: import external rows

Maps external columns into the seed shape, optionally sanitizes, and validates
gold targets when a validator is configured. It also guards the capture loop: a
`serve_capture` row is refused unless a reviewer approved it.

`--video PATH` extracts sidecar-named frames (`frame` / `timestamp_ms` / `t`)
with ffmpeg into `datasets/samples/images/` and sets the request field to each
PNG. Core does not parse MEVA KPF or COCO VID; a plugin writes the sidecar.

## `mint`: preference pairs

Turns candidate completions into `{prompt, chosen, rejected}` pairs for DPO or
ORPO. For each prompt the validator splits candidates into pass and fail; a
prompt with both yields one pair.

```bash
maatml mint <dpo-model-dir> --input candidates.jsonl
```

So a minted pair means "this completion passes the contract and that one does
not" — not a hand-labelled guess. Input is JSONL of
`{prompt, candidates: [completion, ...]}`; pairs append stamped `source: mint`.

Preference training needs volume and balance. A hundred approvals against nine
declines is too thin and too imbalanced to mint from; filtering supervised
training to approved actions is the better use of that signal.

## Reviewed capture

`maatml serve --capture` records real traffic. A captured prediction is **not**
training data — it is a proposal a human or teacher must correct and approve.

```bash
maatml serve <model-dir> --auth-token "$TOKEN" --capture captures.jsonl
# review captures.jsonl: fix the target, set "approved": true on keepers
maatml ingest <model-dir> --input captures.jsonl    # refuses unapproved rows
maatml run <model-dir>                               # retrains on the new seeds
```

Capture requires the auth token (an open capture endpoint is an unbounded write
sink and a way to poison the corpus), the file is size-capped, and `ingest`
refuses any row still marked unapproved.

## Diagnosing teacher rejections

Before changing the teacher or increasing its token budget, inspect the
rejection report and cached responses:

- Empty or truncated replies can indicate that reasoning consumed the output
  budget. Check the endpoint's returned usage and supported generation options.
- Valid JSON with the wrong argument shape points to a contract or prompt
  mismatch. Include a worked example that the task validator accepts.
- Repeated request errors point to endpoint configuration or availability;
  increasing the corpus target does not resolve them.
- Acceptance rates depend on the task, prompt pool, teacher revision and
  validator. Record those inputs with the distillation report before comparing
  runs.
