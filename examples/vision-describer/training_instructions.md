# Vision Result Describer: training instructions

## 1. Purpose

Turn a `VisionMultitaskResult` JSON blob (scene + detections + pose) into one
short, grounded English sentence for UIs, logs, and accessibility captions.

## 2. Base model

`google/flan-t5-small` (~80M params). Encoder-decoder fits “structured text
in → short JSON out” without chat-template overhead. Upgrade path is a
`model_id` swap to `google/flan-t5-base` if gates miss on small.

## 3. Training objective

Standard seq2seq cross-entropy. Source = `dataset.source_prefix` + linearized
vision JSON. Target = compact JSON `{"description":"..."}` (wrapper required
so `Seq2SeqPredictor`’s brace repair stays valid).

## 4. Expected output

```json
{"description": "A striped scene contains two circles and a square, with the centered figure raising both arms."}
```

Constraints: one sentence, ≤ 30 words, mention scene label, summarize objects,
include a pose cue when geometry is clear.

## 5. Input contract

Linearized vision JSON after `linearize_vision_result()`:

- drop detections / keypoints with `confidence < 0.3`
- round floats to 2 decimals
- sort detections by label then position
- always emit all 12 keypoint slots (zeros for filtered points)

## 6. Dataset

Synthetic rows from `scripts/build_seeds.py` / `maatml datagen`: prediction-like
noise on confidences and coordinates, gated by the task validator. Split by
`family` (0.8 / 0.1 / 0.1). Benchmark anchors pin one scene × pose variant set
to the test split.

## 7. Dataset size by stage

| Stage | Rows |
|---|---|
| Committed starter | 120 seed + 15 benchmark |
| Recommended train | ≥ 400 (`--target 400`) |
| Stretch | 1000+ via `maatml datagen` |

## 8. System / prefix

`describe vision result: ` source prefix (see `datasets/prompt_spec.json`).

## 9. Method

Full fine-tune (no LoRA). Small base + short targets → cheap on CPU/MPS.

## 10. Training block

See `model.yml` `training:` / `smoke:`.

## 11. Environment

From the repository root in an activated virtual environment, run
`pip install -e ".[ml]"`. No vision extra is required for this example.

## 12. Tokenizer

Stock flan-t5 tokenizer; `embedding_strategy: resize` for version skew.

## 13. Validation layers

1. JSON parse  
2. JSON Schema  
3. Non-empty `description`  
4. Conciseness (≤ 30 words)  
5. Scene grounding  
6. Object grounding  

## 14. Evaluation gates

See `model.yml` `evaluation.gates`.

## 15. Test prompt set

`datasets/samples/benchmark_samples.jsonl`: fixed anchors per scene label.

## 16. Composition

Two servers + `scripts/compose_client.py`. Describer never imports the vision
plugin; only the linearized JSON contract is shared (duplicated constants).

## 17. Artifacts

Checkpoints under `output/checkpoints/`; safetensors export only (no GGUF/MLX
for seq2seq).

## 18. Versioning

Use the current SemVer in `model.yml`. Bump on schema or caption-template changes;
re-run `build_seeds.py`.

## 19. Validation checklist

- [ ] `maatml validate examples/vision-describer`
- [ ] `maatml prepare` + `train --smoke`
- [ ] `maatml evaluate --gate` on a full train
- [ ] Compose client returns a description for a vision sample image

## 20. Recommended commands

```bash
python examples/vision-describer/scripts/build_seeds.py --target 400
maatml prepare examples/vision-describer
maatml train examples/vision-describer --device mps
maatml evaluate examples/vision-describer --gate
maatml serve examples/vision-describer --port 8081
```

## 21. Closing the train/serve gap

Seeds use clean synthetic vision JSON. After the vision model is trained,
optionally run it over images, linearize predictions, and
`maatml ingest examples/vision-describer --input …` so the describer sees
realistic confidence noise.
