# Run experiments

`opennpu-quant run` collects activation statistics from FP32 models, creates INT8 models and evaluates both. This guide covers presets, run options, paths, progress, reuse and the output layout. It continues from the [quick start](../README.md#quick-start), and re-evaluating the original fixed models is covered in [Reproduce the original measurements](reproducing-results.md).

## Choose an experiment

Select a packaged preset or your own YAML/JSON experiment file with `-c`:

| Preset | Models | Conditions | Data |
|---|---|---|---|
| `quick` | ResNet-18, YOLOv5s | `fp32`, `pot_rtn`, `float_rtn` | 8 calibration and 32 evaluation samples. Needs ImageNet and COCO unless you select one model |
| `core10` (default) | 10 core models | `fp32`, `pot_*`, `float_*` | Full validation sets |
| `extended` | 34 models | `fp32`, `pot_*`, `float_*`, `sym_pot_*` | Full validation sets |
| `cnn38` | All 38 models | Same as `extended` | Full validation sets |
| `cnn38_smoke` | All 38 models | `fp32`, `pot_rtn`, `pot_adaround` | CPU, 2 calibration and 2 evaluation samples, 2 AdaRound steps, percentile 99.99, functional AdaRound validation and the LFW smoke subset |

Only presets or experiment files with `evaluation.limit: null` evaluate full validation sets. `quick` and `cnn38_smoke` stay capped even without `--limit`. The [model reference](model-reference.md#model-groups) lists which models belong to `core10` and `extended`.

Each INT8 condition combines a scheme with RTN (`_rtn`) or AdaRound (`_adaround`):

| Scheme | Activations | Weights | Scales | `reproduce` name |
|---|---|---|---|---|
| `pot` | Asymmetric, per-tensor | Per-channel | `pot_ceil` | `asym_pc_pot_*` |
| `float` | Asymmetric, per-tensor | Per-channel | `float` | `asym_pc_float_*` |
| `sym_pot` | Symmetric, per-tensor | Per-tensor | `pot_ceil` | `sym_pt_pot_*` |

`float` means INT8 with unconstrained scales, not FP32.

To change percentiles, conditions or other settings, copy a preset and edit it, as shown in [Example: change a percentile](configuration.md#example-change-a-percentile). The presets use their own percentiles, and [Original per-model settings](model-reference.md#original-per-model-settings) lists how they differ from the original measurements. [`reproduce configs`](configuration.md#use-the-original-per-model-settings) writes those settings as experiment files.

## Select models, conditions and samples

- `--models` and `--conditions` select entries that the experiment already defines. `--conditions` puts FP32 first when the experiment defines it.
- `--limit` sets the number of evaluation samples. For LFW it counts pairs ([LFW](models-and-data.md#lfw)).
- `--calibration-samples` sets the number of calibration inputs separately.
- `--dry-run` prints the effective settings and paths. It validates the configuration but does not check whether models, data or the output directory are ready.

[Command-line overrides](configuration.md#command-line-overrides) lists the other options, such as AdaRound steps and cache sizes.

## Paths and devices

Use the same model and data directories for preparation and execution. `--home` defaults to `./workspace`, and individual path options override its subdirectories. For `run`, `--output` is the parent of the run directory, so `--output workspace/example-runs` with the `quick` preset writes to `workspace/example-runs/quick`. [Workspace paths](configuration.md#workspace-paths) lists every path option and its precedence.

The presets run on `cuda:0`, except `cnn38_smoke`, which runs on the CPU. An experiment file sets the device with `runtime.device`, and `--device` overrides it. CPU host graphs and postprocessing always run on the CPU, also during GPU execution.

## Evaluate an existing ONNX model

If `<models-dir>/resnet18/network.onnx` and the ImageNet manifest already exist, skip the export. Models with CPU host graphs also need those files and their recipe mappings. Replace the paths with your bundle and data locations.

```bash
opennpu-quant run -c quick --models resnet18 --conditions fp32 \
  --limit 2 --device cuda:0 \
  --models-dir workspace/models --data-dir workspace/data \
  --output workspace/fp32-check
```

With FP32 as the only condition, the command runs inference, postprocessing and evaluation without calibration or quantization. If the same command already completed in that output directory, the tool verifies the stored result and reuses it without running inference again.

| Goal | Path |
|---|---|
| Evaluate an existing FP32 model | The `run --conditions fp32` command above |
| Create a new INT8 model from an existing FP32 model | Add INT8 conditions to `run`. Valid statistics caches are reused |
| Evaluate the original INT8 models (requires the original assets) | [`reproduce prepare` and `reproduce run`](reproducing-results.md) |
| Rebuild tables from completed evaluations | [`report` or `reproduce report`](reporting.md) |

An ONNX file by itself does not tell the tool which dataset, preprocessing or decoder to use. To evaluate your own model, give it a recipe as described in [Add a custom model](models-and-data.md#add-a-custom-model).

## Monitor progress and read results

A run first verifies its settings, model bundle and dataset. FP32 conditions go straight to evaluation. INT8 conditions reuse or collect statistics, build RTN QDQ, optionally train AdaRound, audit the graph and evaluate it.

`status.json` tracks each model and condition as `pending`, `running`, `completed`, `reused`, `failed` or `interrupted`, together with any error. A directory lock keeps two runs from writing to the same run directory. Reports are refreshed after each model, at the end and after a failure.

When the run finishes, the console shows each condition's metric, accuracy, recovery, sample count and `partial` or `full` scope. `run --json` prints `output`, `mean_recovery`, `device` and `rows` as JSON and sends progress messages to stderr. The console format does not change which report files are written.

To write the Excel-compatible `raw_data.*` tables or rebuild reports later, use `report` as described in [Reports and Excel export](reporting.md). [How to read a result](validation.md#how-to-read-a-result) explains what partial and full results can show.

## Reuse and caching

Reuse depends on an **identity**: a content hash of the code, model graphs, input mappings, data and its order, evaluation profiles, and quantization and training settings. A matching identity lets the tool reuse completed work. A different identity stops the run with an error, and a damaged artifact is recomputed. Moving files without changing their contents keeps the identity. Use a new `--output` whenever inputs, evaluation scope or settings change.

| Layer | Location | Lookup and verification | `--force` | `--recalibrate` | Disable with |
|---|---|---|---|---|---|
| Configuration guard | `<run>/effective_config.json` | Complete settings | Bypassed and overwritten | No effect | — |
| Completed results | `<run>/<model>/<cond>/complete.json` | Condition identity and file hashes | Ignored and recomputed | INT8 recomputed with a new generation | — |
| Input disk cache | `<cache>/inputs/<model>/<identity>/` | Preprocessing and data identity | Kept | Kept | `--no-input-cache` |
| Statistics cache | `<cache>/calibration/<model>/<key>/` | Model, input digest, bins, ORT environment and statistics-code hash | Kept | Collected again under `requests/<generation>/` | — |
| AdaRound checkpoints | `<run>/<model>/<cond>/adaround/<identity>[/<generation>]` | Condition identity | Kept and resumed | New generation, started over | — |
| AdaRound activation cache | RAM and GPU memory | — | — | — | `--no-cache` |

- **`--force`** skips completed-result reuse and the configuration guard, and overwrites `effective_config.json`. Caches and checkpoints are kept. Results that other models or conditions produced with the previous settings drop out of the reports.
- **`--recalibrate`** writes a per-model `recalibration.json` whose generation is the current execution ID. Later runs keep using that generation while the file exists, even without the flag.
- **Code changes** also change the identity, because it covers every packaged `.py` file. After updating the package, even for a comment, earlier results and AdaRound checkpoints are no longer reused. Use a new `--output` or rerun with `--force`, and never edit stored hashes to force reuse.
- **Memory budgets** such as `--host-cache-gib` are part of the identity, so changing them reports "Run configuration changed".
- There is no cache-cleaning command. Delete a cache directory when no run is using it.

## Output files

```text
<runs>/<name>/
├── .lock                    Execution lock
├── effective_config.json    Settings that produced these results
├── status.json              Run/model/condition states and errors
├── summary.md, summary.json Per-condition results and cohort means
├── records.json             Portable measurement records, input to results
├── accuracy.csv/tsv          15-column table, identical to results.csv/tsv
├── results.csv/tsv
├── metrics.csv/tsv           One row per metric
└── <model>/
    ├── recalibration.json   Recalibration request, if present
    ├── input_cache.json     Input-cache metadata, when enabled
    └── <condition>/
        ├── result.json          Measurements and identity
        ├── complete.json        Hashes of result.json and additional files
        ├── quantization.json    Quantization audit for INT8 conditions
        ├── model.onnx           --save-models
        ├── predictions.jsonl.gz --save-predictions
        └── adaround/            AdaRound checkpoints
```

## Troubleshooting

| Problem | What to do |
|---|---|
| GPU execution does not start | Run `nvidia-smi` and `opennpu-quant info`. If `nvidia-smi` itself reports an NVML error, the installed driver libraries do not match the running kernel driver, which is separate from the Python CUDA packages. The tool never modifies drivers or falls back to the CPU on its own |
| "Run configuration changed" | The settings differ from the ones stored in the output directory. Use a new `--output`, or `--force` to recompute |
| Missing dataset manifest | Prepare the dataset with the same `--home` or `--data-dir` as the run |
| Unknown model or condition | Check `opennpu-quant models` and the conditions defined in the selected experiment |
