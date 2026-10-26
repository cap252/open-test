# Reproduce the original measurements

You can download the ONNX files from GitHub Releases.

This guide uses the FP32/INT8 ONNX files, records and data manifests of the original measurements. Obtain evaluation datasets using the [dataset download guide](models-and-data.md#prepare-datasets), and prepare the files and path settings listed below. With prepared data and an FP32 model, you can use `run` to create and evaluate new INT8 models independently of the original measurement assets ([create new INT8 models](../README.md#create-new-int8-models)).

## What this workflow does

`reproduce` re-evaluates the original models as they are. Nothing is exported, calibrated or trained, and an AdaRound condition selects a model that was already trained. Each model has seven conditions: FP32 plus RTN and AdaRound for Asym/PC float, Asym/PC PoT and Sym/PT PoT. Here `float` means unconstrained INT8 scales, not FP32.

The selection manifest supplies each condition's percentile, evaluation profile, GPU memory limit and AdaRound settings. The workflow has four steps: `reproduce prepare` checks the assets, `reproduce run` evaluates, `reproduce report` writes tables and `reproduce compare` compares two runs.

## Bind the assets

Extract the model release into `bundle/` and prepare the evaluation datasets. The examples below use `bundle/selections/selected-210.json`, which identifies the release files by artifact ID and SHA256. The packaged selection retains the original records and uses different bindings. Do not substitute it for the release selection. Copy the template below, then edit its path values.

`reproduce prepare` verifies these existing assets and creates an execution package. It is separate from `data prepare`, which creates dataset manifests for new experiments.

```bash
mkdir -p workspace/repro
cp docs/reproduction-bindings.example.json workspace/repro/bindings.json
```

Edit the path values in `bindings.json`. Set `model-bundle` to the absolute path of the extracted `bundle/` directory and `lfw-cache` to the absolute path of the cache you will generate below. Keep the root IDs, relative paths within the bundle and file contents unchanged.

| Root | Contents |
|---|---|
| `model-bundle` | ONNX files, external weights, CPU host graphs, QDQ audits, data manifests and reference records from the release |
| `dataset-imagenet`, `dataset-coco`, `dataset-widerface`, `dataset-voc` | Local image and annotation roots |
| `lfw-cache` | Locally generated `manifest.json`, `tensor_0.npy` and `verification.json` |

Checking ResNet-18 alone needs `model-bundle` and `dataset-imagenet`. MobileFaceNet needs `model-bundle` and the verified `lfw-cache`. A condition can run without a raw GPU output reference, but that comparison remains `pending`.

An ONNX delivery must include its external weights, required CPU host graphs, input/output mapping and per-model execution conventions, QDQ audits, ordered data manifests and hashes, reference records, and a bindings example. A network file alone is insufficient. Packaging and uploading such a delivery is a separate step from this repository workflow.

### External ONNX weights

For a graph with external tensor data, its selection record must include an `external_data` mapping from every referenced filename to its expected SHA256. Paths are relative to the ONNX file and must remain inside its directory. This applies to the network and both CPU host graphs. External data in sparse ONNX tensors is not supported.

```json
{
  "id": "artifact://models/example/network.onnx",
  "sha256": "<ONNX SHA256>",
  "external_data": {
    "weights.bin": "<weight file SHA256>"
  }
}
```

Obtain these expected hashes from the model provider's verified manifest. Hashing whatever files happen to be present does not establish agreement with the original model. Missing files are `pending`, and missing or mismatched expected hashes are `fail`. A raw-output reference for this graph must carry the same `external_data` mapping in its reference manifest.

Preparation checks these files and includes their expected hashes in the condition identity. Execution checks them again before reusing a completed result or interrupted samples. An older external-weight package without these hashes needs a new selection and preparation. Existing self-contained ONNX packages keep their format. Moving unchanged files together only requires updated bindings.

## Generate the LFW input cache

LFW images and input tensors are not included in the model release. Generate the input cache locally with `reproduce prepare-lfw`. The command verifies the generated inputs against the recorded reference before they can be used for reproduction.

```bash
opennpu-quant reproduce prepare-lfw \
  --selection bundle/selections/selected-210.json \
  --bindings workspace/repro/bindings.json \
  --root /data/lfw --download \
  --output workspace/repro/lfw-cache
```

Set the `lfw-cache` binding to the absolute path corresponding to `--output` before running this command. Keep the cache outside `--root`. With `--download`, the command obtains the pinned original images, pair list and landmarks. Without it, provide `pairs.txt`, `lfw_funneled.json` and the original JPEGs under `data/lfw_funneled/<person>/` within `--root`.

The command performs CPU alignment and normalization for all 6,000 pairs, preserving the original and flipped views in their recorded order. It does not run ONNX inference, calibration or AdaRound. The cache needs about 2.2 GiB in addition to the originals. A successful result reports `state: verified` and `sample_count: 15402`. Every input digest, tensor file hash and the manifest hash must match the selection. An interrupted or mismatched build is not published as a complete cache. Existing output is reused only after verification and is never overwritten.

`verification.json` records the local generation environment and code identity separately from the unchanged reference cache manifest. If a source file or generated input differs, the command fails rather than accepting a numerical tolerance. Keep originals and generated tensors local, outside Git and model release archives.

`data prepare lfw_mobilefacenet` prepares aligned images and a dataset manifest for ordinary `run` experiments. `reproduce prepare-lfw` instead creates the verified input cache required by the fixed-model release selection. After it succeeds, use `reproduce prepare --limit full` below.

## Check one model

Start with FP32 and PoT RTN for ResNet-18 on three images:

```bash
opennpu-quant reproduce prepare --selection bundle/selections/selected-210.json \
  --bindings workspace/repro/bindings.json \
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

`reproduce prepare` and `reproduce configs` default to `cnn30` for the packaged selection. The table below describes those original records. A release selection contains only its listed models, so the CNN30 release supports 210 conditions. Additional-model examples require a selection and bindings that include those models.

| Selection | Models | Conditions |
|---|---:|---:|
| `--model-set core10` | 10 | 70 |
| `--model-set cnn30` (default) | 30, including core10 | 210 |
| `--model-set all` | 38, including the eight models outside cnn30 | 266 |

Existing prepared packages retain their complete scope even when their models are outside the current presets. A package prepared with an earlier model set does not need that preset to run or produce reports.

A model set selects names only. Percentiles, profiles and other per-condition settings still come from the packaged selection, so `--model-set core10` gives seven conditions per model even though the `core10` preset of `run` has five. Each condition keeps its own percentile. MobileFaceNet, for example, uses P99.99 for the float conditions and P99.9 for the PoT conditions ([Original per-model settings](model-reference.md#original-per-model-settings)).

- `--models resnet18,yolov9_s` picks models by name instead of a set, including a single additional model. It cannot be combined with `--model-set`.
- `--conditions` narrows the selection further and does not add FP32 automatically.
- `--limit` sets the samples per condition. It defaults to 3, `full` evaluates each complete validation set ([dataset sizes](models-and-data.md#prepare-datasets)), and for LFW it counts pairs ([LFW](models-and-data.md#lfw)). LFW requires the full manifest and at least 6,000 pairs. A smaller request is rejected before preparation writes a package, including mixed-model requests. With the default `cnn30` model set, explicitly use `--limit full` or select non-LFW representative models.
- `--selection` uses a custom selection file instead of the packaged one. Omitting both model options then selects every model in that file, and an explicit `core10` or `cnn30` requires all of its models to be present. A different selection or data order needs new prepared and run directories.

`reproduce run`, `compare` and `report` work on an existing prepared package or run, and they keep its scope.

## Scale up from one model

Prepare a new package for each scope with the options below. Then run `reproduce run` and `reproduce report` as in [Check one model](#check-one-model), with new directories.

| Scope | `reproduce prepare` options | Conditions | Expected report |
|---|---|---:|---|
| Representative FP32 and PoT RTN on three images | `--models resnet18,yolov5s --conditions fp32,asym_pc_pot_rtn --limit 3` | 4 | `full: 0 / partial: 4 / excluded: 0` |
| Full evaluation of `cnn30` | `--model-set cnn30 --limit full` | 210 | `full: 210 / partial: 0 / excluded: 0` |
| Full evaluation of all 38 models | `--model-set all --limit full` | 266 | `full: 266 / partial: 0 / excluded: 0` |
| Additional support: YOLOv9-T PoT conditions with FP32 | `--models yolov9_t --limit full --conditions fp32,asym_pc_pot_rtn,asym_pc_pot_adaround,sym_pt_pot_rtn,sym_pt_pot_adaround` | 5 | `full: 5 / partial: 0 / excluded: 0` |


For example, the full evaluation of `cnn30`:

```bash
opennpu-quant reproduce prepare --selection bundle/selections/selected-210.json \
  --bindings workspace/repro/bindings.json \
  --model-set cnn30 --limit full --output workspace/repro/cnn30-full
opennpu-quant reproduce run --package workspace/repro/cnn30-full \
  --bindings workspace/repro/bindings.json --device cuda:0 --storage predictions \
  --output workspace/repro/cnn30-full-run
```

In every scope, preparation should report every condition as `ready` and the run every condition as `completed`, with nothing failed or pending. The report counts depend on the sample limit. The representative three-image scope writes its four conditions to the partial tables, and a full scope writes every condition to the full tables, as the last column shows. If the report counts differ, `raw_data.metadata.json` lists the reasons. To open the tables in Excel, see [Import into Excel](reporting.md#import-into-excel).

Without a condition filter, each model's FP32 and six INT8 conditions share one run directory, which lets the report find the FP32 denominator. Every network input runs twice to confirm bitwise repeatability on the same GPU, so a full run takes much longer than the three-sample scope. Run the representative scope first to check the pipeline. Full preparation separately verifies the assets of every selected model.

`--storage predictions` keeps the final predictions together with the hash, shape and dtype of each raw output. It needs much less space than the default `--storage raw`, which also keeps the raw tensors as NPZ files. Without the saved tensors, the size of a raw-output difference cannot be computed later, so use `raw` from the three-sample scope onward if you plan to compare GPUs.

## Check agreement with the original measurements

In the run directory, `run.json` records each condition's state and `<model>__<condition>/result.json` records its metrics and comparisons, for example `resnet18__fp32/result.json`. This script summarizes the saved JSON without running inference:

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

root = Path('workspace/repro/cnn30-full-run')
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
- Tasks that support it also compare the hash of the complete prediction stream. A differing hash leaves `prediction_stream` as `pending`, even if metrics pass, because a hash cannot show the size of a numerical difference. Classification, semantic segmentation and face verification may also leave this `pending` because of the original prediction format.
- `not_recorded` means that the field was not saved, not that the values differ.

A `completed` state or the right number of CSV rows does not establish agreement ([How to read a result](validation.md#how-to-read-a-result)). When a condition fails or differs, read `original_accuracy` in its `result.json` or `error` in `run.json`. If FP32 also differs, compare the model, inputs, environment and profile first. If only INT8 differs, also check the selected model files, percentiles and QDQ audits.

## Resume an interrupted run

Repeat only `reproduce run` with the same package, bindings, device, storage mode and package version. There is no need to run `prepare` again. Completed conditions are reused after their files and identity are verified.

Within an interrupted condition, completed samples are fed to the evaluator again. With `--storage predictions`, those samples are also inferred again to verify their raw hashes, because the tensors were not kept. A changed output hash fails this resume check, even for a 1 ULP difference; it checks continuity with the saved execution.

Which directories must be new depends on what changed:

- **Code or runtime environment:** run into a new output directory. The existing prepared package stays usable while its models, settings and data are unchanged.
- **Data that no longer matches the package:** the affected conditions fail with "Dataset/order changed since prepare". If the change is intended, prepare again. To reproduce the original measurements, first restore the original data and its order instead of preparing on the changed data.
- **Model files, profiles or the sample count:** prepare again and run into a new output directory.
- **Packaged selection:** an existing prepared package keeps the selection it was prepared with. Prepare again to apply the current selection.
- **Old partial LFW packages or results:** this version rejects them for execution, reporting and comparison. Keep them unchanged and prepare a new package with the full manifest and `--limit full`. Existing full LFW packages remain usable.
- **Asset locations only:** if the files and their relative paths stay the same, update the bindings.

A reusable package says nothing about whether the new results match the earlier ones. `reproduce compare` reports the raw-output, prediction and metric verdicts separately. In comparisons between the same device type, an implementation mismatch fails the overall comparison even when the numerical checks pass. CPU-to-GPU comparisons are diagnostic and stay pending. Never edit JSON files or hashes to force reuse.

## Compare two servers

Prepare the same package and assets on the second server, update the binding paths and run the same small example. Then put both run directories side by side and compare them:

```bash
opennpu-quant reproduce compare --actual workspace/repro/server-b-resnet18-3-run \
  --expected workspace/repro/resnet18-3-run --output workspace/repro/compare-3
```

`comparison.md` and `comparison.json` report raw-output, prediction, metric and environment verdicts separately, using the rules in the next section. CPU-to-GPU equality is not a completion criterion.

## GPU environment and numerical verdicts

The reference environment uses CUDA runtime **12.4.127**, cuBLAS **12.4.5.8**, cuDNN **9.1.0.70** and ONNX Runtime **1.20.2**, with batch size 1, one thread, `ORT_DISABLE_ALL` and TF32 disabled. Each run records the package versions together with the SHA256 hashes of the libraries actually loaded. Comparisons also include the GPU model and UUID, the driver, the CPU postprocessing libraries, the provider options and any CPU fallback.

Hardware and loaded libraries can differ even with the same execution settings. Missing required environment evidence leaves environment agreement `pending`.

Hashes establish that the recorded model files, inputs and saved artifacts are unchanged. Output agreement uses the separate rules below, so a hash check is not itself a numerical tolerance test.

| Comparison | Verdict |
|---|---|
| Recorded model and input identity | Exact hash match; changed files or input tensors do not represent the same measurement conditions |
| The same input run twice in the same ONNX Runtime session | Bitwise equality; a 1 ULP output difference fails the repeatability check |
| Raw outputs across GPUs, with both tensors retained | Matching shape, dtype and finite values, and elementwise tolerance `1e-5 + 1e-4 × abs(reference)` |
| Raw-output comparison with only hashes available | Equal hashes establish byte equality. Different hashes leave the raw verdict `pending` because tensors are needed to assess tolerance |
| Final predictions | Matching detection counts, classes, order, labels and masks, with numerical tolerance for coordinates and scores |
| Metrics | Per-metric numerical comparison, with bitwise results recorded separately |

A 1 ULP difference in FP32 outputs across GPUs can pass the raw-output tolerance check. `within_tolerance_nonbitwise` marks tensors that pass the tolerance but differ in bytes, and records the model, condition, sample, tensor and maximum error. Predictions, metrics and environment compatibility are checked separately: changed labels or masks, or an environment mismatch such as differing software versions, can still fail the comparison. Cross-GPU acceptance requires reference evidence and two verified, distinct GPUs.

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
