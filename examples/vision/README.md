# Vision multitask

One MobileNetV3-Large checkpoint that jointly predicts:

- **scene** classification (background style)
- **object detection** (colored shapes, CenterNet-style, NMS-free ONNX)
- **pose** estimation (single stick figure, 12 keypoints)

Scene generation is deterministic and offline. Full training uses a pretrained
MobileNet backbone and may download its weights on first use; the smoke
configuration disables pretrained weights.
Deploy with `maatml export --format onnx` and `maatml serve`.

## Output shape

```json
{
  "scene": {"label": "striped", "confidence": 0.94},
  "detections": [
    {"label": "circle", "box": [0.1, 0.2, 0.3, 0.4], "confidence": 0.88}
  ],
  "pose": {
    "keypoints": [
      {"name": "head", "x": 0.5, "y": 0.2, "confidence": 1.0}
    ]
  }
}
```

## Layout

```
examples/vision/
├── model.yml
├── vision_plugin/          # trainer, predictor, metrics, ONNX exporter, synth
├── datasets/
│   ├── schema.json
│   └── samples/            # seed_samples.jsonl + images/ + benchmark
├── scripts/build_seeds.py
└── tests/
```

## Lifecycle

Run these commands from the repository root in an activated virtual
environment. The editable install matches the example to this source revision.

```bash
# Install (once)
pip install -e ".[ml,vision]"

# Generate the images and matching rows before preparing (images are not tracked)
python examples/vision/scripts/build_seeds.py --target 2000
# or: maatml datagen examples/vision --target 500

maatml validate examples/vision
maatml prepare examples/vision
maatml train examples/vision --smoke --device mps   # or cpu|cuda
maatml train examples/vision --device mps
maatml evaluate examples/vision --gate
maatml export examples/vision --format onnx --parity
maatml verify examples/vision/output/export/<run_id>
```

## Serve (makes "→ deploy" real)

Same command on a Mac (onnxruntime CPU) and a Jetson (TensorRT/CUDA EP):

```bash
maatml serve examples/vision --checkpoint output/export/<run_id> --port 8080

# health / info
curl -s localhost:8080/health | jq
curl -s localhost:8080/info | jq

# predict (path relative to model dir, or base64 / data-URI)
python examples/vision/output/export/<run_id>/deploy/client.py \
  examples/vision/datasets/samples/images/<sample>.png
```

Endpoints: `GET /health`, `GET /info`, `POST /predict` (`?validate=1` optional).

## Deploy to Jetson

1. Copy the export directory to the device.
2. Install NVIDIA's `onnxruntime-gpu` wheel for your JetPack.
3. Run `maatml serve … --checkpoint <export-dir>`: providers prefer TensorRT → CUDA → CPU.
4. Optional power-user path: `./deploy/build_engine.sh` (`trtexec --fp16`) for a standalone engine.

Int8 calibration is out of scope for this example.

## Bring real data

Point `dataset.seed_samples` at your own JSONL (same schema: `image` + `expected`)
or use `maatml ingest examples/vision --input PATH --map …`. Prefer a small COCO
subset via ingest rather than streaming 20 GB in core.

## Quality gates

| Metric | Gate | Meaning |
|---|---|---|
| `all_layers_pass_rate` | >= 0.97 | every validator layer passed |
| `map_50` | >= 0.18 | VOC-style mAP @ IoU 0.5 |
| `pck_0_2` | >= 0.33 | pose PCK @ 0.2 x person diagonal |
| `scene_accuracy` | >= 0.97 | background-style classification |

The table reflects the configured example gates, not a fresh measurement.
After changing the corpus, evaluate an explicit run and derive supported
floors with `maatml gates derive`; retain the report and population hash.

## Validator (`vision_scene`)

The out-of-model contract is a 4-layer gate. Eval, serve (`?validate=1` /
`--enforce`), and datagen all call the same function. Failures carry
`layer` / `code` / `location` / `message` / `hint`, and show up in:

- `output/eval/<run>.json` → `sample_failures` (and the markdown summary)
- `serve --enforce` → HTTP 422 payload (and retry feedback when `--max-retries` > 0)
- `*.datagen_rejected.jsonl` → `_validation_errors` on each rejected row

| Layer | What it checks |
|---|---|
| 1 | JSON parse; root must be an object |
| 2 | JSON Schema (`datasets/schema.json`) when declared |
| 3 | Required shapes: `scene.label`, `detections[].{label,box}`, `pose.keypoints` |
| 4 | Enum membership: scene labels, shape labels, the 12 keypoint names |

### Error codes

| Code | Layer | Meaning | Fix |
|---|---|---|---|
| `invalid_json` | 1 | output is not parseable JSON | return bare JSON (no fences / trailing commas) |
| `not_object` | 1 | root is an array or scalar | wrap as `{scene, detections, pose}` |
| `schema` | 2 | fails `datasets/schema.json` | match required keys and field types |
| `scene_shape` | 3 | `scene` missing or has no `label` | set `scene: {label, confidence?}` |
| `detections_shape` | 3 | `detections` is not a list | set `detections` to a JSON array |
| `detection_item` | 3 | one detection is not `{label, box}` | fix the indexed item |
| `box_shape` | 3 | `box` is not `[x1,y1,x2,y2]` | four normalized floats in `[0,1]` |
| `pose_shape` | 3 | `pose.keypoints` missing or not a list | set `pose: {keypoints: [...]}` |
| `scene_label` | 4 | unknown scene label | one of `plain/gradient/striped/noisy/checker` |
| `shape_label` | 4 | unknown detection label | one of `circle/square/triangle/star` |
| `keypoints_missing` | 4 | not all 12 keypoint names present | include every name from `KEYPOINT_NAMES` |

`all_layers_pass_rate` is the validator verdict (every required layer passed).
`scene_accuracy`, `map_50`, and `pck_0_2` come from the metrics plugin and
compare predictions against gold — a row can pass the validator and still
lose on accuracy.
