# Configuration reference

Start with a supplied experiment YAML, select the models and conditions you want, and change only the settings your experiment needs. The examples below use `run`; settings for fixed-model `reproduce` are explained under [Use the original per-model settings](#use-the-original-per-model-settings).

<a id="use-a-preset-or-an-experiment-file"></a>

## Start with a preset

The repository includes complete experiment YAML files:

| Preset | YAML file | Models | Conditions per model |
|---|---|---|---|
| `core10` | [core10.yaml](../src/opennpu_quant/resources/experiments/core10.yaml) | 10 | 5: FP32 and RTN/AdaRound for `pot` and `float` |
| `cnn30` | [cnn30.yaml](../src/opennpu_quant/resources/experiments/cnn30.yaml) | 30 | 7: the core10 conditions plus RTN/AdaRound for `sym_pot` |

`run` defaults to `core10`. Choose `-c cnn30` to use the 30-model preset. Both presets use `runtime.device: cuda:0` and `evaluation.limit: null` for full evaluation. You can inspect a preset without starting an experiment:

```bash
opennpu-quant run -c cnn30 --dry-run
```

For settings with CLI options, combine the preset with those options; no new YAML or JSON is needed. The [common settings](#common-settings) and [multiple-model example](#run-multiple-models-and-conditions) show this approach. To save your own settings or change an option that has no CLI flag, copy a preset and edit it. From the repository root:

```bash
cp src/opennpu_quant/resources/experiments/cnn30.yaml experiment.yaml
opennpu-quant run -c experiment.yaml --dry-run
```

The file selected with `-c` must contain the required experiment settings. A file containing only changed keys is not automatically filled from a preset. You can keep editing your own copy; a new file is not required for each change. Use a new `name` or `--output` when changing a completed experiment so its saved results remain separate.

If you installed only the package, [Example: change a percentile](#example-change-a-percentile) shows how to load and edit the full preset through Python. The same approach works for other supported experiment settings.

## Common settings

An experiment file can describe several models and conditions. These are useful starting points; the [detailed settings](#detailed-settings) cover less common options.

| What to change | Experiment key | Command-line option |
|---|---|---|
| Models to evaluate | `models` | `--models resnet18,resnet50` |
| Conditions to apply | `conditions` | `--conditions fp32,pot_rtn`, using names defined in the experiment |
| Execution device | `runtime.device` | `--device cuda:0` or `--device cpu` |
| Calibration input count | `calibration.samples.<dataset>` | `--calibration-samples 8` sets the count for every dataset |
| Evaluation sample count | `evaluation.limit` | `--limit 3` for an image-based check; full evaluation uses `null` in YAML |
| Save generated INT8 ONNX files | `output.save_qdq_models` | `--save-models` |

Calibration inputs determine quantization ranges; evaluation inputs measure accuracy. Set these counts separately. LFW evaluates face pairs and requires all 6,000 pairs, so a three-image limit is not applicable to that model. A small evaluation checks execution; use the full evaluation set to measure accuracy.

The keys control concrete stages of execution: `models` selects the outer loop, `conditions` selects the work for each model, and the remaining settings configure calibration, quantization and evaluation. Model-specific input and output rules come from model recipes, so a separate experiment file for every model is unnecessary.

## Run multiple models and conditions

This example selects two models and two conditions, giving four model/condition combinations:

```bash
opennpu-quant run -c cnn30 --models resnet18,resnet50 \
  --conditions fp32,pot_rtn --output workspace/two-model-runs --dry-run
```

`--dry-run` validates and prints the effective configuration and paths. It does not check whether model files, datasets or the output directory are ready. After [preparing models and data](models-and-data.md), remove `--dry-run` to execute the four combinations. This example retains the preset's full evaluation scope.

Execution visits each selected model in order and then each selected condition. Results are stored under `<output>/<experiment name>/<model>/<condition>`; here, for example, `workspace/two-model-runs/cnn30/resnet18/pot_rtn`. When `--conditions` is used, the CLI includes the FP32 baseline automatically if it was omitted and is defined in the experiment.

An error stops `run`. Running the same configuration again can reuse verified completed results; it does not necessarily repeat every measurement. See [Reuse and caching](run-pipeline.md#reuse-and-caching) for rerunning or changing an experiment.

<a id="where-settings-live"></a>

## Where to change settings

Use this table to choose the file or option for the change you want to make.

| What to change | Where to change it |
|---|---|
| Models, quantization conditions, calibration counts and evaluation scope | Experiment YAML/JSON selected with `-c`, or a supported `run` option |
| A model's input size, preprocessing or output interpretation | Model recipe; the packaged files are in `resources/models/*.yaml`, and an exported bundle can have a local `recipe.yaml` |
| Model, data, cache and result directories | Path options or the optional `opennpu_quant.local.yaml` file; see [Workspace paths](#workspace-paths) |
| Local locations of the supplied files used by `reproduce` | `bindings.json`, which connects recorded asset locations to your server |

For example, changing a percentile in an experiment file changes the INT8 model created by `run`. Fixed-model `reproduce` uses the supplied files and recorded settings. Editing the experiment YAML does not change those files or settings. Reports use the settings saved at execution time, so editing a preset also leaves existing evaluations unchanged.

## Detailed settings

Use the following sections when you need an option beyond the common settings. Tables distinguish preset values from defaults supplied by configuration objects. Unqualified scheme defaults refer to `Scheme`; they may differ from a preset's explicit values. A listed default does not mean every YAML key can be omitted, which is why copying a complete preset is the starting point.

### Command-line overrides

The experiment file is validated when loaded. Supported CLI options replace the corresponding values, and the effective configuration is validated again. Options that select several models or datasets can update more than one key.

A local `recipe.yaml` supplies the model bundle's input and graph settings. Packaged models still use their packaged evaluation profile by default; choose another profile explicitly with `evaluation.profiles.<model>` or `--evaluation-profile`. See [Precedence and local recipes](evaluation-profiles.md#precedence-and-local-recipes).

| Option | Configuration key |
|---|---|
| `--device` | `runtime.device` |
| `--limit` | `evaluation.limit` |
| `--evaluation-profile` | `evaluation.profiles` for all selected models (comparison only) |
| `--calibration-samples` | All `calibration.samples.*` |
| `--adaround-steps` | `adaround.steps` |
| `--save-models` | `output.save_qdq_models` |
| `--save-predictions` | `output.save_predictions` |
| `--host-cache-gib`, `--gpu-cache-mib`, `--no-cache` | `activation_cache.*` |
| `--no-input-cache` | `calibration.input_cache.enabled` |

`--models`, `--conditions` and `--dry-run` are described in [Select models, conditions and samples](run-pipeline.md#select-models-conditions-and-samples).

Environment variables such as `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` control CPU numerical-library thread counts. The CLI sets unset variables to 1 and preserves existing values. Python API calls leave these variables unchanged. To compare Python and CLI runs with matching thread settings, set the variables before importing NumPy, ONNX Runtime or `opennpu_quant`, because libraries can read them during initialization.

### Workspace paths

A local path file is optional. Paths resolve per key in this order: CLI option, environment variable, local YAML, then the default derived from home. The default local filename is `opennpu_quant.local.yaml`; `--local-config` selects another file. Relative paths resolve from the working directory. Local `datasets.<name>` settings override only the image root recorded in a dataset manifest.

| Directory | CLI | Environment variable | Default |
|---|---|---|---|
| home | `--home` | `OPENNPU_QUANT_HOME` | `./workspace` |
| models | `--models-dir` | `OPENNPU_QUANT_MODELS` | `<home>/models` |
| data | `--data-dir` | `OPENNPU_QUANT_DATA` | `<home>/data` |
| cache | `--cache-dir` | `OPENNPU_QUANT_CACHE` | `<home>/cache` |
| runs | `--output` (run) | `OPENNPU_QUANT_RUNS` | `<home>/runs` |
| weights | `--weights-dir` | `OPENNPU_QUANT_WEIGHTS` | `<home>/weights` |

Precedence applies to each key separately. For example, `models:` in the local YAML wins over a default derived from `--home`.

`--output` means something different for each command. For `run` it is the parent of `<output>/<experiment name>`, and for `results` it is the export directory. `reproduce` resolves assets through its bindings instead of these paths.

### Preprocessing and calibration

`input`, `network_input_map`, `graphs` and `preprocess.*` belong to the model recipe or bundle. The `calibration.*` keys belong to the experiment file.

| Key | Default and allowed values | Effect of a change |
| --- | --- | --- |
| `input`, `network_input_map`, `graphs` | Model-specific shape, layout and names. Network feeds are float32 | Export boundaries and input shapes are revalidated, and the input, statistics and evaluation identities change |
| `preprocess.kind`, resize, crop, size, mean, std | See the [model reference](model-reference.md). Each kind supports its own keys | Inputs, statistics and evaluations are revalidated. Editing a declared shape does not change the ONNX file |
| `calibration.samples.<dataset>` | ImageNet 1,000 and others 128 in both presets. Positive integer | The input prefix and statistics identity change, including the order |
| `calibration.expected_samples_identity.<model>` | Optional SHA256 | Collection fails early if the input order or bytes differ from the original |
| `calibration.histogram_bins` | Default 2048. Integer of at least 2 and at least `quantized_bins` | New histograms are collected |
| `calibration.device` | Default `run`. Allowed: `run`, `cpu`, `cuda:N` | The statistics runtime identity changes. CPU and GPU statistics are not bitwise equal |
| `calibration.input_cache.enabled` | `true` in presets and `false` if omitted | Changes caching only, not the numerical policy |
| `calibration.input_cache.max_bytes` | Default 2 GiB. Positive integer number of bytes | Storage budget. Fixed and per-sample variable shapes are both kept |

The manifest written by `data prepare` defines the files, annotations, calibration lists and split order ([Prepare datasets](models-and-data.md#prepare-datasets)).

### Quantization and AdaRound

These keys belong to the experiment file. Some keys abbreviate a repeated prefix with a leading dot; for example, `.weight_granularity` means `schemes.<family>.weight_granularity`.

| Key | Default and allowed values | Effect of a change |
| --- | --- | --- |
| `conditions.<name>.scheme`, `.adaround` | Scheme name and boolean. `fp32: {}` is the FP32 baseline | Creates a separate QDQ, training and evaluation condition |
| `schemes.<family>.scope` | Default `all`. Allowed: `basic`, `all` | Feature boundaries change, so the statistics plan and audit are revalidated |
| `.activation_symmetric` | Default `false`. Boolean | Statistics can be reused, but QDQ is rebuilt and reevaluated |
| `.weight_granularity` | Default `per_channel`. Allowed: `per_channel`, `per_tensor` | QDQ, biases and AdaRound are revalidated |
| `.method`, `.percentile` | Default `percentile` with 99.99. `percentile` is `null` for `minmax` and `entropy` | New ranges come from the same statistics, and the result and training identity change |
| `percentiles.<model>.<family>` | Used when the scheme percentile is `per_model`. Range (0, 100] | Per-model value. Generated original settings use per-condition family keys |
| `.activation_scale`, `.weight_scale` | Default `float`. Allowed: `float`, `pot_nearest`, `pot_ceil`, and `pot_mse` for weights | QDQ is rebuilt and reevaluated. The original PoT conditions use `pot_ceil` |
| `.quantized_bins` | Default 128. Integer from 2 to `histogram_bins` | Ranges and QDQ are revalidated |
| `adaround.steps` | Default 10,000. Positive integer | The training and checkpoint identity changes |
| `adaround.learning_rate`, `.warmup_fraction`, `.regularization` | 0.001, 0.2 and 0.01. Ranges: >0, [0, 1) and ≥0 | The training trajectory changes |
| `adaround.beta_start`, `.beta_end`, `.seed` | 20, 2 and 20260928. Start ≥ end > 0, integer seed ≥ 0 | Settings for reproducible training |
| `adaround.batch_size`, `.window_samples`, `.window_steps` | 8, 32 and 100. Positive integers | Sample grouping, order and checkpoint identity change |
| `adaround.window_bytes` | 512 MiB, or 4 GiB for some original settings. Positive integer number of bytes | The window composition can change. This is separate from the activation cache budget |
| `adaround.device` | Default `auto`. Allowed: `auto`, `cpu`, `cuda` | The Torch device and numerical path change |
| `adaround.validation`, `.atol`, `.rtol` | `strict`, 1e-5 and 1e-4. Allowed: `strict` or `functional`, and non-negative tolerances | Functional mode records finite numerical differences. Shape and dtype errors still fail |
| `activation_cache.*` | `enabled` true, host 4 GiB, device 512 MiB and reserve 2 GiB. Integer bytes ≥ 0 | Part of the completion and training identity. Disable with `--no-cache` |

`--host-cache-gib` and `--gpu-cache-mib` set `activation_cache.host_bytes` and `device_bytes`. `--gpu-window-mib` is an alias of `--gpu-cache-mib`, and neither changes `adaround.window_bytes`. These cache settings apply only to AdaRound. The Python API supports more detailed `QuantizationConfig` policies, and not every API field is an experiment key ([Quantization logic](quantization.md)).

### Postprocessing, evaluation and reporting

`decode.*` belongs to the model recipe. Other configuration keys below belong to the experiment; `--table-profile` is a report command option.

`run` and `reproduce` use the same model-specific evaluation profiles by default. `--evaluation-profile` is a comparison option for `run`, explained in [Evaluation profiles](evaluation-profiles.md).

| Key | Default and allowed values | Effect of a change |
| --- | --- | --- |
| `decode.*` | Model-specific confidence, IoU, maximum detections and output mapping | Identical network outputs can give different predictions and metrics. Reevaluate FP32 and INT8 |
| `evaluation.profiles.<model>` | Default of the packaged model. See [Evaluation profiles](evaluation-profiles.md) | The profile identity changes. Unsupported combinations are rejected |
| `evaluation.limit` | `null` (full) in both presets. Positive integer or `null`. LFW requires all 6,000 pairs | The evaluation input identity and FP32 pairing change |
| `runtime.device`, `runtime.cuda.gpu_mem_limit` | Presets use `cuda:0` and the CPU must be selected explicitly. Memory limit 6 GiB by default, 6 or 8 GiB in the original settings | The runtime identity changes. The memory limit is part of the reproducible settings |
| `runtime.intra_op_threads`, `.inter_op_threads` | 1 and 1. Non-negative integers, where 0 lets ORT choose | `reproduce` fixes both to 1 |
| `output.save_qdq_models`, `.save_predictions` | `false` in presets. Boolean | Part of the completed-result identity |
| `report.primary_metric.<task>` | Recipe or preset metric | Selects among saved metrics. Only ratio units are allowed, and no inference runs |
| `report.target_mean_recovery` | 98 in `core10`, `null` in `cnn30`. Positive number or `null` | The summary marks whether each condition's cohort mean recovery meets this target ([FP32 pairing and recovery](reporting.md#fp32-pairing-and-recovery)) |
| `--table-profile opennpu_raw` | Optional for `report` and `results`, default for `reproduce report` | Adds the `raw_data.*` tables with display names and a full/partial split. The evaluation identity is unchanged |

The `report.*` settings are part of the `effective_config.json` identity, and editing the configuration of a completed directory breaks result verification. Use `--format` or `--table-profile` to change presentation, and treat any other setting change as a new experiment.

## Example: change a percentile

`load_config()` loads the full preset from an installed package. Edit the returned configuration and save it to change any supported experiment setting, including model selection, `calibration.histogram_bins` or `adaround.learning_rate`. This example changes a percentile and validates the configuration without running the experiment:

```bash
python - <<'PY'
from opennpu_quant.configuration import load_config
import json
config = load_config('core10')
config['name'] = 'resnet18_p999'
config['models'] = ['resnet18']
config['conditions'] = {key: config['conditions'][key] for key in ('fp32', 'pot_rtn')}
config['calibration']['samples'] = dict.fromkeys(config['calibration']['samples'], 8)
config['evaluation']['limit'] = 32
config['percentiles']['resnet18']['pot'] = 99.9
with open('experiment.json', 'w') as stream:
    json.dump(config, stream, indent=2)
PY
opennpu-quant run -c experiment.json --dry-run
```

## Example: add an AdaRound condition

`core10` already defines AdaRound conditions. `--adaround-steps` changes their training length but does not select them. This example copies the preset, defines a short AdaRound condition and selects it explicitly:

```bash
python - <<'PY'
import json
from opennpu_quant.configuration import load_config

config = load_config('core10')
config['name'] = 'resnet18_adaround_smoke'
config['models'] = ['resnet18']
config['conditions']['pot_adaround'] = {'scheme': 'pot', 'adaround': True}
config['adaround'].update(batch_size=1, window_samples=1, window_steps=1)
with open('adaround-smoke.json', 'w', encoding='utf-8') as stream:
    json.dump(config, stream, indent=2)
PY
opennpu-quant run -c adaround-smoke.json --conditions fp32,pot_adaround \
  --calibration-samples 2 --limit 2 --adaround-steps 2 --device cuda:0 \
  --save-models --output workspace/adaround-example-runs --dry-run
```

Without `--dry-run`, the command calibrates, trains two AdaRound steps per target layer and evaluates FP32 and INT8 in the same run, so the report has a recovery denominator. These settings check that training works. They are not the original training settings.

## Use the original per-model settings

Fixed-model reproduction uses `resources/reproduction/selected.json`, which records the selected graphs and their hashes, input mappings, evaluation profiles, percentiles, and per-condition GPU and AdaRound settings. Experiment presets and local recipes do not override these recorded conditions. `bindings.json` maps the recorded `artifact://root/...` locations to local directories independently of the workspace path settings. Select the models to reproduce with the options in [Choose a model set](reproducing-results.md#choose-a-model-set).

To create new models with the recorded settings, use `reproduce configs` to write experiment files for `run`. Generating these settings needs no original assets.

```bash
opennpu-quant reproduce configs --models vgg16,yolov9_t \
  --output workspace/selected-settings
opennpu-quant run -c workspace/selected-settings/yolov9_t__asym_pc_pot_adaround.json --dry-run
```

Without `--models` or `--model-set`, `reproduce configs` covers the `cnn30` set ([Choose a model set](reproducing-results.md#choose-a-model-set)).

Each file describes one model and one condition, including its percentile, GPU memory limit, AdaRound window, evaluation profile and calibration input hash. To compute recovery in the same run, add `fp32: {}` to the file and select `--conditions fp32,...`, because the generated file contains only its own condition. You still need a matching ONNX bundle, data and environment.

These files drive `run`, which creates and trains new models. `reproduce prepare` and `reproduce run` evaluate the fixed original files instead. In `reproduce`, `--limit full` corresponds to `evaluation.limit: null` here.
