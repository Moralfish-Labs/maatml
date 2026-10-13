# support-ticket-triage

Causal LoRA SFT example: given a customer support ticket, emit triage JSON.

```json
{
  "priority": "p1|p2|p3|p4",
  "category": "billing|access|bug|how_to|other",
  "team": "payments|identity|platform|docs|general",
  "summary": "one-line agent summary"
}
```

## Validator (the contract)

`triage_plugin/` registers an out-of-model validator with four layers:

1. **JSON parse**: output is a JSON object
2. **Schema**: structure, required fields, and enums (`datasets/schema.json`)
3. **Routing contract**: `category` must route to the mandated `team`
   (`billing→payments`, `access→identity`, `bug→platform`, `how_to→docs`,
   `other→general`)
4. **Summary quality**: non-empty, single line, ≤ 20 words

Layer 3 is the point: it ties two fields together by a task rule a plain JSON
schema alone does not check in this example. The same validator gates the seed data, scores
`maatml evaluate`, and can re-check live output at `maatml serve
/predict?validate=1`.

## Lifecycle

Run from the repository root after `pip install -e ".[ml]"` in a virtual
environment, as described in the quickstart.

```bash
maatml prepare  examples/support-ticket-triage/
maatml train    examples/support-ticket-triage/ --smoke
maatml train    examples/support-ticket-triage/
maatml evaluate examples/support-ticket-triage/ --gate
```

## Quality gates

| Metric | Gate | Meaning |
|---|---|---|
| `all_layers_pass_rate` | >= 0.95 | every validator layer passed |
| `category_accuracy` | >= 0.86 | predicted category matches gold |
| `json_parse_rate` | >= 0.97 | output is valid JSON |
| `priority_accuracy` | >= 0.64 | predicted priority matches gold |
| `routing_consistency_rate` | >= 0.95 | `category -> team` contract holds |
| `schema_conformance_rate` | >= 0.97 | matches `datasets/schema.json` |
| `summary_quality_rate` | >= 0.97 | summary is one line within the word cap |
| `team_accuracy` | >= 0.88 | predicted team matches gold |

Add more tickets under `datasets/samples/seed_samples.jsonl`, then re-prepare.
The committed corpus is intentionally small; it demonstrates the `causal_sft`
path, the routing contract, and the standalone model-folder layout, not
production triage quality. The configured gates are explicitly **pending
re-measurement**: their historical measurements used a contaminated split.
Re-evaluate on the corrected population and derive floors from the resulting
report with `maatml gates derive examples/support-ticket-triage --run RUN_ID --write`.
