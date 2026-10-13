# `model.yml`: the whole configuration surface

`model.yml` is the recipe shared by the lifecycle. Supported CLI overrides
are recorded with the operation that uses them; environment variables and
plugin defaults also affect execution. Put durable recipe choices in this
file and preserve the effective configuration with the run evidence.

Start from `maatml scaffold <dir> --architecture causal_sft --name my-task`
rather than a blank file; it writes the sections below with defaults that
already agree with each other.

## Contents

- [Identity](#identity)
- [Dataset](#dataset)
- [Evaluation and gates](#evaluation-and-gates)
- [Training](#training)
- [Packaging and extensions](#packaging-and-extensions)
- [Distillation](#distillation)
- [The smoke tier](#the-smoke-tier)
- [Overrides that still get recorded](#overrides-that-still-get-recorded)

## Identity

```yaml
name: ticket-triage
model_id: ticket-triage
task: support_ticket_routing
architecture: causal_sft          # decides the trainer, the predictor, the scaffold
version: 0.1.0
description: "One line. It reaches the export manifest."
base_model: Qwen/Qwen3-4B
plugins: [./triage_plugin]        # folder-local registrations, loaded before any stage
```

`version` is not decoration: the export manifest carries it, and a release
decision reads it.

## Dataset

```yaml
dataset:
  format: jsonl_seed            # a registered format adapter
  seed: 7                       # the split is deterministic; changing this reshuffles it
  seed_samples: datasets/samples/seed_samples.jsonl
  benchmark_samples: datasets/samples/benchmark_samples.jsonl
  schema: datasets/schema.json        # handed to the validator
  prompt_spec: datasets/prompt_spec.json
  request_field: request
  target_field: expected
  user_placeholder: "<<USER_REQUEST>>"
  group_by: scenario            # see below
  split_ratios: [0.85, 0.075, 0.075]
  sanitize: []                  # registered sanitizers, run before training
```

**`group_by` is leakage protection, and choosing it wrongly is the classic
mistake.** All rows sharing the key land in the same split, so paraphrases of
one situation cannot straddle train and test. Group by the *situation*
(`scenario`), not by the *contract* (`family`) — a family key sends every row of
a family into one split, which is both useless as a split and confusing later,
because `family` usually also names the contract a row is graded under.

Without `group_by`, prepare falls back to `family` -> `source` -> `sample_id`.
A key covering nearly the whole corpus cannot be split, so those rows are split
individually **with a warning**: group-level leakage protection does not apply
to them. Read that warning rather than scrolling past it.

### Populations: isolation, pins, a blind manifest

`group_by` keeps correlated rows on one side of a split; it cannot say which
side a camera lands on or that a site is absent from training. Declare the
hierarchy a row carries and the level each held-out population is disjoint at:

```yaml
dataset:
  isolation:
    fields: [clip, camera, site]                  # row fields, fine -> coarse
    policy: {val: camera, benchmark: camera, blind: site}
  pins:
    val: ["camera:G339"]                          # whole groups, field:value
    benchmark: ["camera:G341", "camera:G421"]
  blind_samples: datasets/samples/blind_v001.jsonl
```

`prepare` moves pinned groups after the hash split; a pinned population is
exactly its pins (plus `benchmark_samples` for the benchmark), so unpinned
groups the hash dealt it return to train. It then asserts the policy over
train / val / benchmark / blind and refuses to write splits on a violation; a
pin matching no row is an error. `maatml audit` re-checks the prepared splits.
The blind manifest is checked for leakage but never written into a split.

Every prepare writes `output/prepared/benchmark.json`: the **benchmark
version** (an order-insensitive hash of the test rows plus the pins), which
eval reports carry as `extras.benchmark_version`. Editing `benchmark_samples`
under the same filename is refused — write the rows to a new file and point
the key at it, so floors keep naming the population they came from.

`maatml evaluate --blind` spends the blind manifest once per frozen candidate:
the run must hold a production gate pass whose recorded fingerprint
(evaluation config + weights) is unchanged, the gates are enforced on the blind
rows, the report is `<run>.blind.json`, and the spend is recorded on the run;
a repeat at the same fingerprint needs `--force` and is recorded as forced.

A `benchmark_samples` row sharing a group key with the training splits is an
error, not a warning. A benchmark is pinned to test on purpose: it is the
population release decisions are made against, and it should be able to grow
without retraining.

### Sources carry their licence

```yaml
dataset:
  attribution: datasets/ATTRIBUTION.md   # a sidecar Markdown table, one row per `source`
  attribution_field: dataset             # optional: the row field the table is keyed on
```

The table is the execute-time licence record, kept beside the corpus rather
than inside `model.yml` because licence rows are long and shared across model
folders. The first Markdown table whose first column is `source` is read;
columns are matched by the casefolded prefix of their header, so a real table
with long headers and extra columns parses as it is:

```markdown
| source     | licence   | commercial-use | provenance/consent | attribution         | sign-off                           |
| ---        | ---       | ---            | ---                | ---                 | ---                                |
| meva       | CC BY 4.0 | yes            | consented actors   | © Kitware and IARPA | cleared — A. Name 2026-08-17       |
| crowdhuman | research  | no             | third-party photos | CrowdHuman          | accepted-risk — A. Name 2026-08-17 |
| dukemtmc   | withdrawn | no             | non-consented      | —                   | do not use                         |
```

`source`, `licence` (or `license`), `commercial-use` and `sign-off` are
required; `provenance` / `consent` and `attribution` are read when present.
`commercial-use` is read from the start of the cell — `yes`, `no`, anything
else is `unknown` — so a hedge like `unknown until filtered` stays unknown.
The table is keyed on each row's `source` unless `attribution_field` names
another field: a corpus whose `source` is finer than its licence (one value
per annotation file, `benchmark:holdout` for pinned rows) keys the table on
the licence-level field it carries, typically `dataset`. The lock records the
field.

With the key declared, `prepare` checks every row that enters (seed,
benchmark and blind) and refuses to write splits when:

- a row carries no `source`, or its source has no table row;
- the sign-off is blocked (`do not use`, `blocked`, `fixtures only`) or
  unsigned (empty or `*unsigned*`);
- commercial-use is `no` or `unknown` and the sign-off does not record an
  accepted risk.

Risk is accepted in the table, not on the command line: a sign-off such as
`accepted-risk — A. Name 2026-08-17` admits a `no` / `unknown` source and is
recorded against it in the corpus lock, so the acceptance carries a name and
a date rather than a flag nobody can trace later. A blocked row is never
admitted by an acceptance.

Every prepare — with or without a table — writes
`output/prepared/corpus.lock.json`: each input file by path, sha256 and row
count, the table rows the corpus actually used with their row counts, the
risk accepted, and a `lock_sha256` over all of it. `maatml export` copies the
lock into `manifest.json` as `corpus_lock`, and when it names sources the
bundle also ships an `ATTRIBUTION.md` rendered from those rows (listed in the
manifest, so `verify` covers it). Nothing is looked up or inferred at export:
the block is the declared table and nothing else. When the auto
`MODEL_CARD.md` lands it embeds the same block.

## Evaluation and gates

```yaml
evaluation:
  predictor: ticket_triage   # registered; defaults to the architecture's own
  validator: ticket_triage   # the contract, reused by datagen/distill/ingest and serve
  metrics: ticket_triage
  gates:
    routing_json_parse_rate:  0.85  # 117/128 = 0.914, w95 0.853
    routing_team_known_rate:  0.85  # 117/128 = 0.914, w95 0.853
    all_layers_pass_rate:     0.94  # 401/414 = 0.969, w95 0.947
  slices: [family, {field: spectrum, values: [rgb, ir]}]   # per-value pass rates
  cache_predictions: true   # keep <run>.predictions.jsonl beside the report
  batch_size: 32           # rows per predict_batch call (default 1; needs the predictor's predict_batch)
```

`slices` names row fields; the report carries `n`, `pass_rate` and the Wilson
95% lower bound per value, with `(absent)` for rows lacking the field and
`n: 0` (no rate) for a declared value with no rows. `cache_predictions` (or
`evaluate --cache`) keeps every row's output, verdict and metadata keyed to the
split's content hash, so floors and sweeps derive from the predictions the
report measured instead of re-running inference. Reports carry
`report_version`; `Report.read(strict=True)` refuses one that predates it.

Each floor is the **Wilson 95% lower bound of the observed rate at that
metric's own denominator**, floored to two places, with the measurement written
beside it so the number can be audited without rerunning anything. Derive them,
never type them:

```bash
maatml gates derive <model-dir> --run <run_id> [--run <run_id2>] --write
```

`maatml train --seeds N [--seed S]` trains the same recipe at seeds S..S+N-1 and
writes `output/seeds/<first-run>-xN.json` with mean / sd / min / max per
metric; `gates derive --seed-study <that file>` floors on those runs.

`gates derive` reads each run's eval report (`report_version >= 1`), takes the
per-metric **minimum across runs** so a lucky seed cannot set the contract,
refuses a rate with fewer than `--min-n` rows, and rewrites `evaluation.gates`
in place with the measurement as the comment. It also stamps
`evaluation.gates_benchmark` with the split's content hash: `evaluate --gate`
warns when it is asked to enforce those floors on a different split and
refuses under `--strict-population`, because floors describe the population
they were derived on.

Where rows cluster (frames within a camera, paraphrases within a scenario) a
row-level bound overstates the evidence. If the run was evaluated with
`--cache`, `gates derive --cluster-by family` resamples whole groups of the
prediction cache instead (harness rates and `slice:` gates; a plugin rate needs
its own per-row verdict), and refuses a rate spanning fewer than
`--min-groups` groups.

A metrics plugin makes its rates derivable by reporting the evidence behind
them: `{"recall": 0.315, "__counts__": {"recall": [69087, 219079]}}`. Without
counts a metric is floored at its observed value and the comment says so.

### Operating point

A decision threshold is chosen on val and spent once on test:

```yaml
evaluation:
  score_thresh: 0.40   # written by operating-point derive; comment carries the sweep
  operating_point:
    threshold_key: score_thresh     # the evaluation key the predictor reads
    objective: recall               # maximised
    budget: {metric: fp_per_frame, max: 1.0}   # must hold at the chosen cut
    sources: [meva, virat]          # rows whose `dataset` is one of these
    grid: {start: 0.05, stop: 0.95, step: 0.05}
```

```bash
maatml evaluate <dir> --checkpoint R --split val --cache \
    --set evaluation.score_thresh=0.05                         # cache val at a low cut
maatml operating-point derive <dir> --run R --write            # sweep, pick, write
maatml operating-point derive <dir> --run R --confirm-on-test  # spend test once
```

The sweep calls the predictor's `rescore(rows, threshold)` over the cached val
predictions (see the plugins reference), skips grid points below the cut the
cache was decoded at (so the val evaluate that feeds it runs under
`--set evaluation.<threshold_key>=<low cut>`; the override is recorded as
`extras.overrides`, model.yml is untouched, and `--set` is refused with
`--gate` or `--blind` because a measurement under an override is never gate
evidence), picks the best objective under the budget (ties to the
lower budget, then the higher cut), writes `<run>.<split>.operating_point.json`
and, with `--write`, the threshold line with its provenance. `--confirm-on-test`
evaluates once on test at the written cut and records the spend on the run;
`maatml runs` lists spends per benchmark, and a second spend on the same
benchmark warns. Deriving on the test split is refused.

Gate values are a number or `{min, tier}`; `tier: advisory` is reported and
recorded but never fails a step. A gate may name a slice:
`"slice:camera=G339": 0.45` gates that slice's pass rate, and a slice with no
rows fails rather than passing on an invented number.

### Why Wilson

You cannot gate at the rate you measured. Gate `all_layers_pass` at its observed
0.969 and an identical model fails about half the time, because that gate is
reading noise. So the floor has to sit below the observation, and the only
question is how far below, decided by something other than taste.

A round number cannot answer it: 0.90 is trivially clearable for a metric
measured on 414 rows and unreachable in practice for one measured on 16.

Wald — the normal approximation, `p̂ ± z·sqrt(p̂(1-p̂)/n)` — answers it wrongly at
exactly the rates a good model produces. At a perfect rate `p̂(1-p̂)` is zero, so
the interval collapses and the "lower bound" is 1.000. These are real
measurements from one multi-family model, chosen because the denominators span
16 to 414:

| | observed | Wald floor | Wilson floor |
|---|---|---|---|
| `all_layers_pass` | 401/414 | 0.952 | 0.947 |
| `routing_refusal_recall` | 66/71 | 0.870 | 0.846 |
| `grounded_injection_resistance` | 26/27 | 0.892 | 0.817 |
| `conversation_honesty` | 16/16 | **1.000** | 0.806 |
| `state_refusal_recall` | 21/21 | **1.000** | 0.845 |

A floor of 1.000 claims sixteen successes prove a hundred-percent rate, and it
makes the first single failure a red gate forever. Wald also wanders outside
[0,1] and has poor coverage at small n, which is where four of these families
live.

Wilson inverts the score test instead, so it stays inside [0,1], behaves at 0
and 1, and keeps coverage near nominal at small n. It is also closed-form
arithmetic — a floor can be checked in a spreadsheet, which matters for a number
committed to a file as an auditable claim.

```python
import math


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return centre - half
```

The property that earns it: the slack is set by the evidence. `n` perfect rows
give a floor of 0.722 at n=10, 0.886 at n=30, 0.963 at n=100 and 0.990 at
n=400 — so the gate tightens as the benchmark grows, with nobody editing it.
That is also the arithmetic reason an aggregate is treacherous: pooled over
everything it has a huge denominator, so its floor sits tight against its
observation while the small families it averages over carry floors twenty points
loose.

Clopper-Pearson, the exact binomial, guarantees *at least* 95% coverage and is
therefore systematically wider — floors lower than the evidence warrants, which
lets real regressions through. Wilson trades that guarantee for calibration.

### The caveat these floors carry

Wilson assumes independent Bernoulli trials. Test rows are grouped by scenario,
so paraphrases of one situation are not independent: the effective sample size
can be below the row count, so row-level bounds can overstate the
evidence. The size of that effect depends on the correlation. Twenty-seven rows drawn from nine scenarios is weaker evidence
than twenty-seven unrelated ones.

This does not change the choice — every alternative assumes the same thing — but
it does mean a floor derived from a small, heavily grouped family should be read
as approximate, and that widening the *variety* of a benchmark buys more than
adding paraphrases to it.

`evaluate` uses `packaging.max_input_tokens` as its token budget — the same
budget `serve` and `export --parity` enforce — and records how many inputs it
truncated. If that count is not zero, the measurement is of a truncated
population.

## Training

```yaml
training:
  model_id: Qwen/Qwen3-4B
  max_input_tokens: 8192
  batch_size: 1
  grad_accum: 16
  learning_rate: 1.0e-4
  epochs: 6
  eval_steps: 20
  save_steps: 100
  logging_steps: 5
  warmup_ratio: 0.03
  weight_decay: 0.0
  precision: bf16
  grad_checkpointing: true
  seed: 7
  generation:
    max_new_tokens: 4096      # the ceiling evaluation generates under
  lora:
    enabled: true
    r: 16
    alpha: 32
    dropout: 0.05
    save_mode: adapter        # adapter | merged
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

### Selecting the checkpoint

```yaml
training:
  select_by: all_layers_pass_rate   # a metric the eval harness reports
  keep_checkpoints: 4               # save_total_limit (default 2)
```

Without `select_by` a run ships its last weights, and "last" is not a
selection. With it, after training every saved `checkpoint-<step>` plus the
final weights is evaluated on the **val** split with the same predictor,
validator and metrics as `evaluate`, the best by the named metric wins (ties
go to the later one), and the choice is recorded on the run (`selection`:
every candidate's score and report; `selected_checkpoint`). From then on the
run id resolves to that checkpoint for evaluate, export and serve, while
evidence is still recorded against the run. The per-candidate reports live
under `output/eval/select/<run>/`; the test split is never read. A metric the
harness does not report is an error, not a silent fallback to the last
checkpoint. `keep_checkpoints` is what makes the comparison worth anything —
at the default of 2 only the last two survive. A seed study records the
selected value as `select:<metric>`.

`generation.max_new_tokens` is a frequent silent cause of bad scores: a ceiling
below what the contract needs truncates every long answer, and the validator
correctly reports the truncation as a contract failure. If a metric collapses
while spot-checked outputs look right up to the point they stop, check this
first.

`save_mode: adapter` keeps the checkpoint small enough to carry between
machines; merge locally before exporting GGUF.

## Packaging and extensions

```yaml
packaging:
  max_input_tokens: 8192      # the budget eval, serve and export --parity share
  expected_latency_ms: 8000
  weights_dtype: f16
  confidence_thresholds: { low: 0.6, high: 0.9 }

extensions:
  gguf:
    quant_levels: [Q5_K_M]
    llama_cpp_tag: b9829
    convert_script: null      # or a path; never searched on PATH
    quantize_binary: null
```

maatml never searches `PATH` for the llama.cpp convert script or quantize
binary — that would execute whatever `convert.py` happened to be found first.
Name them explicitly here or via `MAATML_LLAMA_CONVERT` /
`MAATML_LLAMA_QUANTIZE`, otherwise `--format gguf` produces no GGUF.

Prefer **Q5_K_M over Q4_K_M** for anything citation-shaped: quantization costs
citation fidelity specifically, and an aggregate pass rate hides it.

## Distillation

```yaml
distill:
  prompt_source: datasets/distill/prompts.jsonl
  teacher_model: qwen/qwen3.6-35b-a3b       # any OpenAI-compatible endpoint
  teacher_revision: "2026-08-b"             # part of the cache key
  cache: datasets/distill/cache.jsonl
  system_prompt_file: datasets/distill/brief.txt
  target_format: text                        # json (default) parses; text does not
  family: routing
  request_params:
    max_tokens: 8192
    reasoning_effort: none
    timeout: 600
```

`target_format` decides what a reply must be *before* the validator sees it.
Default `json` parses and drops what will not parse — right when the target is a
document. Set `text` when the target is not JSON (a sentence then a call object,
a rendered patch): the raw reply goes to the validator instead. Left at `json`,
such a label is refused as unparseable and the gate that decides never sees it.

`teacher_revision` is part of the cache key, so a different teacher never
silently reuses another's labels and `--replay` reproduces the accepted corpus
offline.

## The smoke tier

```yaml
smoke:
  max_steps: 2
  epochs: 1
  gates:
    output_nonempty_rate: 0.5
```

A rehearsal cannot meet production thresholds, and holding it to them only
teaches people to ignore a red gate. `--smoke` enforces these instead, and the
pass is recorded as smoke-gated in the run record and in the export manifest's
`gate_evidence` — so a rehearsal never reads later as a production gate pass.

`output_nonempty_rate` is always reported and is the honest thing for a smoke
tier to gate on: it says the checkpoint saved, reloaded and produced output,
without claiming the output was any good.

### Pathologies

Some shapes of output no floor should have to catch. Every evaluate reports
them as `pathologies[]` (`{name, evidence}`), the gates payload lists their
names, and at the smoke tier each one is a failing blocking gate
(`pathology:<name>`), since a rehearsal floor is loose enough for a broken
model to clear:

- `never_fires` — no output on any row, or a recall-like metric at or below
  0.01 while its precision counterpart is 0.9 or better (or absent): nothing
  is being predicted and precision is flattering it.
- `identical_output` — the same non-empty output on every row.
- `one_class` — every parsed output carries the same `class` / `category` /
  `label` / `family` / `intent` / `kind` while the gold rows span two or more
  categories.

Fewer than five rows report nothing. At the production tier the floors decide,
with the signature beside them. A metrics plugin contributes its own through a
`__pathologies__` entry (names, or `{name, evidence}` dicts), lifted out of
`metrics` like `__counts__`.

## Overrides that still get recorded

```bash
maatml run <dir> --set training.epochs=9
```

`--set` feeds the pipeline fingerprint, so an override makes the right steps
stale and is visible in `maatml plan`. An invalid one exits non-zero **before**
any step runs and leaves `output/pipeline.json` untouched. This is the only
sanctioned way to deviate for one run; anything you want to keep belongs in the
file.
