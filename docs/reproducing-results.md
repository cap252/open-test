# Reproduce the original measurements

This guide is for holders of the original assets: the FP32/INT8 ONNX files, records and data manifests of the original measurements. These assets are not distributed with this repository. Without them, use `run` to create and evaluate new INT8 models, starting from the [quick start](../README.md#quick-start).

## What this workflow does

`reproduce` re-evaluates the original models as they are. Nothing is exported, calibrated or trained, and an AdaRound condition selects a model that was already trained. Each model has seven conditions: FP32 plus RTN and AdaRound for Asym/PC float, Asym/PC PoT and Sym/PT PoT. Here `float` means unconstrained INT8 scales, not FP32.

The packaged selection (`selected.json`) supplies each condition's percentile, evaluation profile, GPU memory limit and AdaRound settings. The workflow has four steps: `reproduce prepare` checks the assets, `reproduce run` evaluates, `reproduce report` writes tables and `reproduce compare` compares two runs.

## Bind the assets

`reproduce` does not download anything. The packaged selection identifies each required file by ID and SHA256 hash, and a bindings file tells the tool where those files live on your server.

```bash
mkdir -p workspace/repro
cp docs/reproduction-bindings.example.json workspace/repro/bindings.json
```

Edit the path values in `bindings.json`. For example, `"research-runs": "/data/frozen/research-runs"` resolves `artifact://research-runs/…` under that directory. Keep the root IDs, the relative paths inside each root and the file contents unchanged.

| Root | Contents |
|---|---|
| `research-repo`, `research-runs` | FP32 and INT8 ONNX files with any external weight files, CPU host graphs, QDQ audits and the original `result.json` records |
| `data-manifests`, `final-data-manifests` | Fixed data manifests that define each split and its order |
| `dataset-imagenet`, `dataset-coco`, `dataset-coco_keypoints`, `dataset-widerface`, `dataset-voc` | Image and annotation roots |
| `dataset-lfw` | The official LFW pair list and verified original/flipped input caches |
| `reference-metrics-v1` | Reference records of the MobileFaceNet conditions, stored with the current metric names |
| `gtx-package` | Saved raw GPU outputs that serve as comparison references |

Checking ResNet-18 alone needs `research-repo`, `research-runs`, `data-manifests` and `dataset-imagenet`. The MobileFaceNet conditions also need `reference-metrics-v1`. Conditions without comparison references still run, but their comparison with the original raw outputs stays `pending`. The tool reads files from `research-repo` and never imports Python code from it.

## Check one model

Start with FP32 and PoT RTN for ResNet-18 on three images:

```bash
opennpu-quant reproduce prepare --bindings workspace/repro/bindings.json \
  --models resnet18 --conditions fp32,asym_pc_pot_rtn --limit 3 \
  --output workspace/repro/resnet18-3
opennpu-quant reproduce run --package workspace/repro/resnet18-3 \
  --bindings workspace/repro/bindings.json --device cuda:0 \
  --output workspace/repro/resnet18-3-run
opennpu-quant reproduce report --run workspace/repro/resnet18-3-run \
  --package workspace/repro/resnet18-3 --output workspace/repro/resnet18-3-tables \
  --table-profile opennpu_raw
```

Preparation should report `counts: {"ready": 2}`, and the run summary should show `completed: 2`, `failed: 0` and `pending: 0`. Both evaluations appear in `workspace/repro/resnet18-3-tables/raw_data.partial.csv`.

At startup, `reproduce run` checks package versions and runs a real CUDA convolution. If this probe fails, read `gpu_probe.json` and `run.json` or `attempt.json`, and see [Troubleshooting](run-pipeline.md#troubleshooting).

## Preparation states

`reproduce prepare` checks the models, records, selected data and data order. It neither downloads nor copies anything, and it records each condition's state in `manifest.json`.

- `ready`: every required asset is available.
- `pending`: assets are missing. Their paths are listed under `missing`.
- `fail`: an asset does not match its recorded hash or data declaration. Diagnostics are listed under `missing`.

After fixing assets, prepare into a new `--output` directory because `prepare` never overwrites one.

## Choose a model set

`reproduce prepare` and `reproduce configs` default to `extended` for the packaged selection.

| Selection | Models | Conditions |
|---|---:|---:|
| `--model-set core10` | 10 | 70 |
| `--model-set extended` (default) | 34, including core10 and PoseNet | 238 |
| `--model-set all` | 38, including the four additional YOLO models | 266 |

A model set selects names only. Percentiles, profiles and other per-condition settings still come from the packaged selection, so `--model-set core10` gives seven conditions per model even though the `core10` preset of `run` has five. Each condition keeps its own percentile. YOLOv9-T, for example, uses P99.99 for the float conditions and P99.9 for the PoT conditions ([Original per-model settings](model-reference.md#original-per-model-settings)).

- `--models resnet18,yolov9_s` picks models by name instead of a set, including a single additional model. It cannot be combined with `--model-set`.
- `--conditions` narrows the selection further and does not add FP32 automatically.
- `--limit` sets the samples per condition. It defaults to 3, `full` evaluates each complete validation set ([dataset sizes](models-and-data.md#prepare-datasets)), and for LFW it counts pairs ([LFW](models-and-data.md#lfw)).
- `--selection` uses a custom selection file instead of the packaged one. Omitting both model options then selects every model in that file, and an explicit `core10` or `extended` requires all of its models to be present. A different selection or data order needs new prepared and run directories.

`reproduce run`, `compare` and `report` work on an existing prepared package or run, and they keep its scope.

## Scale up from one model

Prepare a new package for each scope with the options below. Then run `reproduce run` and `reproduce report` as in [Check one model](#check-one-model), with new directories.

| Scope | `reproduce prepare` options | Conditions | Expected report |
|---|---|---:|---|
| Every `extended` condition on three samples | `--model-set extended --limit 3` | 238 | `full: 0 / partial: 238 / excluded: 0` |
| Full evaluation of `extended` | `--model-set extended --limit full` | 238 | `full: 238 / partial: 0 / excluded: 0` |
| Full evaluation of all 38 models | `--model-set all --limit full` | 266 | `full: 266 / partial: 0 / excluded: 0` |
| YOLOv9-T PoT conditions with FP32 | `--models yolov9_t --limit full --conditions fp32,asym_pc_pot_rtn,asym_pc_pot_adaround,sym_pt_pot_rtn,sym_pt_pot_adaround` | 5 | `full: 5 / partial: 0 / excluded: 0` |

For example, the full evaluation of `extended`:

```bash
opennpu-quant reproduce prepare --bindings workspace/repro/bindings.json \
  --model-set extended --limit full --output workspace/repro/extended-full
opennpu-quant reproduce run --package workspace/repro/extended-full \
  --bindings workspace/repro/bindings.json --device cuda:0 --storage predictions \
  --output workspace/repro/extended-full-run
```

In every scope, preparation should report every condition as `ready` and the run every condition as `completed`, with nothing failed or pending. The report counts depend on the sample limit. A three-sample scope writes every condition to the partial tables, and a full scope writes every condition to the full tables, as the last column shows. If the report counts differ, `raw_data.metadata.json` lists the reasons. To open the tables in Excel, see [Import into Excel](reporting.md#import-into-excel).

Without a condition filter, each model's FP32 and six INT8 conditions share one run directory, which lets the report find the FP32 denominator. Every network input runs twice to confirm bitwise repeatability on the same GPU, so a full run takes much longer than the three-sample scope. Run the three-sample scope first to make sure every asset is in place.

`--storage predictions` keeps the final predictions together with the hash, shape and dtype of each raw output. It needs much less space than the default `--storage raw`, which also keeps the raw tensors as NPZ files. Without the saved tensors, the size of a raw-output difference cannot be computed later, so use `raw` from the three-sample scope onward if you plan to compare GPUs.

## Check agreement with the original measurements

In the run directory, `run.json` records each condition's state and `<model>__<condition>/result.json` records its metrics and comparisons, for example `resnet18__fp32/result.json`. This script summarizes the saved JSON without running inference:

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

root = Path('workspace/repro/extended-full-run')
run = json.loads((root / 'run.json').read_text())
print('Execution states:', dict(Counter(row['state'] for row in run['conditions'])))
metrics, predictions = Counter(), Counter()
for row in run['conditions']:
    if row['state'] != 'completed':
        continue
    result = json.loads((root / row['result']['path']).read_text())
    comparison = result.get('original_accuracy', {})
    metrics[comparison.get('metrics_state', 'not_recorded')] += 1
    predictions[comparison.get('prediction_stream', 'unavailable')] += 1
print('Original metric comparison:', dict(metrics))
print('Original prediction-stream comparison:', dict(predictions))
PY
```

- `original_accuracy.metrics_state` compares the metrics with the original values within numerical tolerance.
- `bitwise_equal` in `original_accuracy.metrics` records exact equality for each metric. `metrics_bitwise_equal` requires all expected metrics to be present, non-conflicting and bitwise equal. A numerical-tolerance pass alone does not establish this.
- Metric checks compare metrics with the same name and record `actual_paths` and `expected_paths`. The one documented exception is segmentation `pixel_accuracy`, which the evaluator stores in `metric_details` and the check reads there when the reference keeps it under `metrics`. Conflicting duplicate fields fail.
- Tasks that support it also compare the hash of the complete prediction stream. Classification, semantic segmentation and face verification may leave this `pending` because of the original prediction format.
- `not_recorded` means that the field was not saved, not that the values differ.

A `completed` state or the right number of CSV rows does not establish agreement ([How to read a result](validation.md#how-to-read-a-result)). When a condition fails or differs, read `original_accuracy` in its `result.json` or `error` in `run.json`. If FP32 also differs, compare the model, inputs, environment and profile first. If only INT8 differs, also check the selected model files, percentiles and QDQ audits.

## Resume an interrupted run

Repeat only `reproduce run` with the same package, bindings, device, storage mode and package version. There is no need to run `prepare` again. Completed conditions are reused after their files and identity are verified.

Within an interrupted condition, completed samples are fed to the evaluator again. With `--storage predictions`, those samples are also inferred again to verify their raw hashes, because the tensors were not kept.

Use new prepared and run directories when the sample count, model files, profiles, runtime environment or code change. A package prepared from an earlier packaged selection also keeps that selection's identity and cannot be resumed with the current one. If only the asset roots move and the files and their relative paths stay the same, update the bindings. Never edit JSON files or hashes to force reuse.

## Compare two servers

Prepare the same package and assets on the second server, update the binding paths and run the same small example. Then put both run directories side by side and compare them:

```bash
opennpu-quant reproduce compare --actual workspace/repro/server-b-resnet18-3-run \
  --expected workspace/repro/resnet18-3-run --output workspace/repro/compare-3
```

`comparison.md` and `comparison.json` report raw-output, prediction, metric and environment verdicts separately, using the rules in the next section. CPU-to-GPU equality is not a completion criterion.

## GPU environment and numerical verdicts

The reference environment uses CUDA runtime **12.4.127**, cuBLAS **12.4.5.8**, cuDNN **9.1.0.70** and ONNX Runtime **1.20.2**, with batch size 1, one thread, `ORT_DISABLE_ALL` and TF32 disabled. Each run records the package versions together with the SHA256 hashes of the libraries actually loaded. Comparisons also include the GPU model and UUID, the driver, the CPU postprocessing libraries, the provider options and any CPU fallback.

The environment identity is recorded after the Torchvision libraries used for CPU postprocessing are loaded, so model order and lazy loading do not change it. In these records, `not_loaded` and `content_changed` are different states.

| Comparison | Verdict |
|---|---|
| Two executions on the same GPU | Bitwise equality |
| Raw outputs across GPUs | Matching shape, dtype and finite values, and elementwise tolerance `1e-5 + 1e-4 × abs(reference)` |
| Final predictions | Matching detection counts, classes, order, labels and masks, with numerical tolerance for coordinates and scores |
| Metrics | Per-metric numerical comparison, with bitwise results recorded separately |

A raw-output pass does not replace the prediction and metric verdicts. `within_tolerance_nonbitwise` marks tensors that pass the tolerance but differ in bytes, and records the model, condition, sample, tensor and maximum error. Without reference outputs, or without two distinct GPUs, cross-GPU reproduction cannot be established.

## Maintainer tools

The scripts in `tools/` support maintainers who hold the original assets. Normal use of `run` and `reproduce` does not need them.

| Script | Purpose |
|---|---|
| `check_calibration_reproduction.py` | Collects calibration statistics again and compares them bit for bit with the original statistics |
| `check_auxiliary_metrics.py` | Re-scores saved full LFW or VOC predictions against the current references, without inference |
| `migrate_reference_metrics.py` | Converts older reference records to the current metric names, keeping every value and rebuilding the hashes |
| `build_reproduction_manifest.py` | Builds the packaged selection from the original records |
| `check_reproduction.py` | Checks frozen tensors or ONNX files of a prepared package through explicit bindings |
| `smoke.py` | Runs the `cnn38_smoke` preset model by model on the CPU |

### Recollect calibration statistics

`check_calibration_reproduction.py` runs on the GPU. Its [input plan](../tools/validation/calibration-plan.json) lists the artifact IDs and SHA256 hashes of the original statistics for all 38 models. Start with one model:

```bash
PYTHONPATH=src python tools/check_calibration_reproduction.py \
  --models resnet18 --bindings workspace/repro/bindings.json \
  --device 0 --output workspace/calibration-check
```

The command uses each model's full calibration set, so it takes longer than a small evaluation. To check more models, pass a comma-separated `--models` list.

### Re-score older runs

The current reference records use the metric names that the evaluator writes, and older records are converted only by `migrate_reference_metrics.py`. To check a completed older run that saved predictions for MobileFaceNet or DeepLabV3-MobileNetV2 against the current references, re-score it:

```bash
PYTHONPATH=src python tools/check_auxiliary_metrics.py \
  --run workspace/repro/older-run \
  --bindings workspace/repro/bindings.json \
  --migrate-metadata \
  --output workspace/repro/older-run-metric-check.json
```

`--migrate-metadata` allows only a name conversion that keeps every value bit for bit. A changed value or a different model or recipe identity still fails, and the saved run is not modified.

## Files written by reproduce

| Location | Contents |
|---|---|
| Prepared `manifest.json`, `manifest.sha256.json` | Selected conditions, data order, profiles, asset states and hashes |
| Run `run.json`, `gpu_probe.json` | Condition states and GPU availability |
| Per-condition `identity.json`, `progress.json` | Settings and asset identity, and progress for resume |
| Per-sample NPZ, prediction JSON and completion marker | Raw outputs (with `--storage raw`), final predictions, input/output hashes and repeatability checks |
| Per-condition `result.json` and SHA sidecar | Metrics, QDQ audit, placement, environment and comparisons with the original measurements |
| `comparison.json`, `comparison.md` | Raw-output, prediction, metric and environment verdicts and bitwise differences |

The packaged `resources/reproduction/selected.json` keeps the artifact IDs and SHA256 hashes of the graphs, original records and audits. It also holds the recipes, output aliases, data and per-condition percentile, GPU and training settings. The original values under `measurement` serve as comparison references, and a run never overwrites them with its own `metrics`.
