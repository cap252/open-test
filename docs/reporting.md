# Reports and Excel export

Report commands turn saved evaluations into tables. They check the provenance and completion evidence of each result, and they never run inference, calibration or training. `run` already writes its default reports, so use these commands to add the Excel-compatible tables or to rebuild reports later.

## Choose a report command

```bash
opennpu-quant report workspace/example-runs/quick --table-profile opennpu_raw
opennpu-quant results workspace/example-runs/quick/records.json \
  --output workspace/tables --table-profile opennpu_raw
opennpu-quant reproduce report --run workspace/repro/resnet18-3-run \
  --package workspace/repro/resnet18-3 --output workspace/repro/resnet18-3-tables \
  --table-profile opennpu_raw
```

| You have | Command | What it reads |
|---|---|---|
| An experiment directory from `run` | `report` | The saved configuration, status and completed evaluation files |
| An exported `records.json` | `results` | The portable records only, without the original result files |
| A `reproduce` run and its prepared package (original assets) | `reproduce report` | The saved run results and their prepared model and data bindings |

The [code map](architecture.md#from-saved-evaluations-to-tables) shows how the three commands share one implementation.

By default, `run` and `report` write the 15-column `accuracy.csv` and `results.csv` with internal names. `--table-profile opennpu_raw` adds the Excel-compatible `raw_data.*` tables described below, with display names, a full/partial split and metadata. It is optional for `report` and `results` and the default for `reproduce report`. With `results`, the option also requires `--output`, so that the full and partial tables and their metadata stay together. With `--stdout csv` or `--stdout tsv` and this profile, stdout shows only the full-evaluation table, and partial rows stay in the files.

## Output files

| File | Contents |
|---|---|
| `raw_data.csv`, `raw_data.tsv` | Verified full evaluations: `full_evaluation` and `complete` are true and the sample count equals the full expected count |
| `raw_data.partial.csv`, `raw_data.partial.tsv` | Valid metrics from evaluations that do not meet the full-evaluation requirements, such as `--limit` runs |
| `raw_data.metadata.json` | Model, condition and run IDs, settings, evaluation identity, counts, units and views, provenance, row locations, and pairing and exclusion reasons |

Every file gets its header, even when a group is empty. CSV uses UTF-8 with a BOM, and TSV uses plain UTF-8. `reproduce report` accepts `--models` and `--conditions` to restrict the selection. If a condition is missing from the tables, look up its reason in `raw_data.metadata.json`.

## The 15-column table

The column names and order follow the `raw data` sheet of the OpenNPU results workbook.

| Order | Column | Value |
|---:|---|---|
| 1 | `model` | Model display name |
| 2 | `task` | Task name |
| 3 | `dataset` | Dataset name |
| 4 | `metric` | Primary metric of this evaluation |
| 5 | `fp32_accuracy` | FP32 metric × 100. INT8 rows show their matching FP32 value |
| 6 | `int8_accuracy` | INT8 metric × 100, blank for FP32 rows |
| 7 | `recovery_percent` | 100 × INT8 / FP32, blank when unavailable |
| 8 | `scheme` | `FP32` or the policy string, without the percentile value |
| 9 | `percentile` | Percentile value when the Percentile method is used |
| 10 | `model_source_url` | Source URL from the recipe |
| 11 | `model_source_revision` | Source commit or framework version |
| 12 | `top5_percent` | top5 × 100, blank if absent |
| 13 | `AP50_percent` | AP50 × 100, blank if absent |
| 14 | `AP75_percent` | AP75 × 100, blank if absent |
| 15 | `metrics_json` | Original value and unit of every metric |

Accuracy and recovery are plain numbers without a `%` sign or extra rounding. For example, the fraction `0.69684` appears as `69.684`, while `metrics_json` keeps the fraction. Metrics keep their own names in `metrics_json`, so instance segmentation reports `mask_AP50` rather than `AP50`.

A typical `scheme` value:

```text
scope=all | activation=asymmetric | weight=per_channel | activation_scale=pot_ceil | weight_scale=pot_ceil | calibration=percentile | rounding=AdaRound
```

The table profile changes only these display names, to match the workbook's lookups:

| Internal ID or name | Display name |
|---|---|
| `yolov3_full`, `yolov4_full`, `yolov7_full` | YOLOv3, YOLOv4, YOLOv7 |
| `deeplabv3plus_r50_voc` | DeepLabV3+ R50-D8 |
| classification, detection, face | image_classification, object_detection, face_detection |
| segmentation, keypoints | semantic_segmentation, pose_estimation |
| coco_keypoints, lfw_mobilefacenet | coco, lfw |

Other models keep their recipe titles. Internal IDs, evaluation profiles and metrics are unchanged. The [format fixture](../tests/fixtures/opennpu_raw_format.json) stores the columns, names and schemes, so the tests do not need the workbook.

## FP32 pairing and recovery

Recovery needs a successful FP32 result for the same model in the same run. The two results must match in model, preprocessing and evaluation identity, metric, data split, sample count, full or partial scope, profile, sample unit, view count and full expected count. For `reproduce`, the prepared selection, recipe and data identities and the CPU postprocessing graph hashes are part of this evidence.

Without a compatible FP32 result, the denominator and recovery stay blank, and the INT8 measurement is still reported. If FP32 measures 0, its zero is kept and only the recovery is blank. FP32 rows show a recovery of about 100, and the value is not rounded. The filters of `reproduce report` do not add FP32 automatically, so include it whenever you need recovery, for example `--conditions fp32,asym_pc_pot_rtn`.

Reports use only the current result's `metrics` and `metric_units`. They never copy the original `measurement.metrics`, fill missing values with zero or borrow FP32 from another run directory. A current measurement that differs from the original is reported as measured.

In `run`, the summary also shows a cohort mean per condition. It is computed only when every model has a full evaluation with a matching FP32 result, so partial runs show `mean_recovery: null` even though each row has its own recovery.

For LFW, `view_count` records the actual input views. `run` stores `samples` as views, so the metadata adds `sample_unit: views` and the evaluator's `pair_count`. `reproduce` stores `samples` as pairs and records the views separately ([LFW](models-and-data.md#lfw)).

## Exclusions and verification

Failed, unexecuted, unverifiable or metric-less results get no numerical row, and `raw_data.metadata.json` records why. A valid INT8 measurement without a compatible FP32 denominator still appears, with blank recovery. Partial evaluations go to the partial files instead of being excluded.

- `report` checks the saved configuration, completion markers and attested file hashes.
- `reproduce report` also checks the sealed run and prepared manifests and each result's input, recipe, profile and graph bindings. A tampered result or a mismatched package stops the export.
- `results` checks the structure, units and identity stored in the portable records. Its `result_hash_verified` field is evidence recorded by the original export and does not authenticate edited records. Use `report` or `reproduce report` when you need the source files checked.

`reproduce report` checks result metadata without rereading ONNX files, images or raw tensors. Use `reproduce compare` to compare inference outputs. A table shows what was measured, and [How to read a result](validation.md#how-to-read-a-result) explains what it does and does not establish.

## Import into Excel

1. Import `raw_data.csv` with Excel's text/CSV import, or `raw_data.tsv` as tab-separated text.
2. Keep the accuracy and recovery columns as plain numbers such as `69.684`. Do not multiply them by 100 or apply percentage formatting.
3. Paste the rows into your workbook and update any summary sheets yourself. A report has one row per executed condition and never edits workbook formulas or ranges.

## Examples

The [raw data examples](examples/raw-data/README.md) contain a full table and a partial table in CSV and TSV, exported from saved evaluations. They illustrate the format and are not evidence of a full reproduction.
