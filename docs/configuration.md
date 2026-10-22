# Configuration reference

Settings come from four places: model recipes, experiment files, local path settings and, for `reproduce`, the packaged selection. Defaults below are those of the general `Scheme` objects or the named preset unless noted. The settings of the original measurements are listed in [Original per-model settings](model-reference.md#original-per-model-settings).

## Where settings live

| Location | Responsibility | Selection and precedence |
|---|---|---|
| `resources/models/*.yaml` | Inputs, preprocessing, graphs, decoder and source checkpoint | A local `recipe.yaml` overrides the bundle settings ([Precedence and local recipes](evaluation-profiles.md#precedence-and-local-recipes)) |
| Experiment YAML/JSON or preset (`-c`) | Models, conditions, statistics, quantization, evaluation and storage | The file is validated, CLI overrides are applied, and the result is validated again. Presets are never merged with user files |
| `run` options | Model and condition subsets, device, limit and other overrides | Override only the keys listed in [Command-line overrides](#command-line-overrides) |
| `evaluation.profiles.<model>` | Evaluation profile for the recipe's decoder | Comparison override ([Precedence and local recipes](evaluation-profiles.md#precedence-and-local-recipes)) |
| `opennpu_quant.local.yaml`, `--local-config` | home, models, data, cache, runs and weights directories, and dataset roots | Per path: CLI, then environment variable, then local YAML, then the default derived from home |
| `resources/reproduction/selected.json` (`reproduce` only) | Model and host-graph hashes, input mapping, profile, percentile and per-condition GPU and AdaRound settings | Authoritative for `reproduce`. Presets and local recipes do not override it |
| `--model-set`, `--models` of `reproduce prepare` and `reproduce configs` (`reproduce` only) | Model subset | See [Choose a model set](reproducing-results.md#choose-a-model-set) |
| `bindings.json` (`reproduce` only) | Local root directories for `artifact://root/...` | Passed with `reproduce --bindings`, independently of the path settings above |

Relative paths resolve from the working directory. `datasets.<name>` in the local YAML overrides only the image root recorded in a dataset manifest. Reports use the settings saved at execution time, so editing a preset does not change existing evaluations.

## Workspace paths

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

## Use a preset or an experiment file

Presets are complete YAML files. Copy one, edit it and pass the copy with `-c`. Loading a preset never runs a generator or merges it with another file.

For the original per-condition measurement settings, use `reproduce configs` as described in [Use the original per-model settings](#use-the-original-per-model-settings). Ordinary presets and the selected measurement manifest serve different purposes.

## Command-line overrides

These `run` options override values in the loaded experiment. The command validates the effective configuration before it starts.

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

Unless `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` or similar variables are already set, the CLI sets numerical-library threads to 1. Python API calls do not change these variables. To match the CLI from Python, set them before importing NumPy or ONNX Runtime.

## Preprocessing and calibration

| Key | Set in | Default and allowed values | Stage | Effect of a change |
|---|---|---|---|---|
| `input`, `network_input_map`, `graphs` | Model recipe or bundle | Model-specific shape, layout and names. Network feeds are float32 | Preprocessing and network boundary | Export boundaries and input shapes are revalidated, and the input, statistics and evaluation identities change |
| `preprocess.kind`, resize, crop, size, mean, std | Model recipe | See the [model reference](model-reference.md). Each kind supports its own keys | Preprocessing | Inputs, statistics and evaluations are revalidated. Editing a declared shape does not change the ONNX file |
| `calibration.samples.<dataset>` | Experiment, or `--calibration-samples` for all datasets | ImageNet 1,000 and others 128 in both presets. Positive integer | Calibration | The input prefix and statistics identity change, including the order |
| `calibration.expected_samples_identity.<model>` | Experiment or generated original settings | Optional SHA256 | Preprocessing and calibration | Collection fails early if the input order or bytes differ from the original |
| `calibration.histogram_bins` | Experiment | Default 2048. Integer of at least 2 and at least `quantized_bins` | Calibration | New histograms are collected |
| `calibration.device` | Experiment | Default `run`. Allowed: `run`, `cpu`, `cuda:N` | Calibration and RTN | The statistics runtime identity changes. CPU and GPU statistics are not bitwise equal |
| `calibration.input_cache.enabled` | Experiment, or `--no-input-cache` | `true` in presets and `false` if omitted | Preprocessing cache | Changes caching only, not the numerical policy |
| `calibration.input_cache.max_bytes` | Experiment | Default 2 GiB. Positive integer number of bytes | Input cache | Storage budget. Fixed and per-sample variable shapes are both kept |

The manifest written by `data prepare` defines the files, annotations, calibration lists and split order ([Prepare datasets](models-and-data.md#prepare-datasets)).

## Quantization and AdaRound

| Key | Set in | Default and allowed values | Stage | Effect of a change |
|---|---|---|---|---|
| `conditions.<name>.scheme`, `.adaround` | Experiment | Scheme name and boolean. `fp32: {}` is the FP32 baseline | Condition selection | Creates a separate QDQ, training and evaluation condition |
| `schemes.<family>.scope` | Experiment | Default `all`. Allowed: `basic`, `all` | QDQ placement | Feature boundaries change, so the statistics plan and audit are revalidated |
| `.activation_symmetric` | Experiment | Default `false`. Boolean | Scale and zero point | Statistics can be reused, but QDQ is rebuilt and reevaluated |
| `.weight_granularity` | Experiment | Default `per_channel`. Allowed: `per_channel`, `per_tensor` | Weight encoding | QDQ, biases and AdaRound are revalidated |
| `.method`, `.percentile` | Experiment | Default `percentile` with 99.99. `percentile` is `null` for `minmax` and `entropy` | Range selection | New ranges come from the same statistics, and the result and training identity change |
| `percentiles.<model>.<family>` | Experiment | Used when the scheme percentile is `per_model`. Range (0, 100] | Range selection | Per-model value. Generated original settings use per-condition family keys |
| `.activation_scale`, `.weight_scale` | Experiment | Default `float`. Allowed: `float`, `pot_nearest`, `pot_ceil`, and `pot_mse` for weights | Scale | QDQ is rebuilt and reevaluated. The original PoT conditions use `pot_ceil` |
| `.quantized_bins` | Experiment | Default 128. Integer from 2 to `histogram_bins` | Entropy range selection | Ranges and QDQ are revalidated |
| `adaround.steps` | Experiment, or `--adaround-steps` | Default 10,000. Positive integer | AdaRound | The training and checkpoint identity changes |
| `adaround.learning_rate`, `.warmup_fraction`, `.regularization` | Experiment | 0.001, 0.2 and 0.01. Ranges: >0, [0, 1) and ≥0 | AdaRound | The training trajectory changes |
| `adaround.beta_start`, `.beta_end`, `.seed` | Experiment | 20, 2 and 20260928. Start ≥ end > 0, integer seed ≥ 0 | AdaRound | Settings for reproducible training |
| `adaround.batch_size`, `.window_samples`, `.window_steps` | Experiment | 8, 32 and 100. Positive integers | AdaRound | Sample grouping, order and checkpoint identity change |
| `adaround.window_bytes` | Experiment | 512 MiB, or 4 GiB for some original settings. Positive integer number of bytes | AdaRound | The window composition can change. This is separate from the activation cache budget |
| `adaround.device` | Experiment | Default `auto`. Allowed: `auto`, `cpu`, `cuda` | AdaRound | The Torch device and numerical path change |
| `adaround.validation`, `.atol`, `.rtol` | Experiment | `strict`, 1e-5 and 1e-4. Allowed: `strict` or `functional`, and non-negative tolerances | Layer-output parity | Functional mode records finite numerical differences. Shape and dtype errors still fail |
| `activation_cache.*` | Experiment or cache options | `enabled` true, host 4 GiB, device 512 MiB and reserve 2 GiB. Integer bytes ≥ 0 | AdaRound cache | Part of the completion and training identity. Disable with `--no-cache` |

`--host-cache-gib` and `--gpu-cache-mib` set `activation_cache.host_bytes` and `device_bytes`. `--gpu-window-mib` is an alias of `--gpu-cache-mib`, and neither changes `adaround.window_bytes`. These cache settings apply only to AdaRound. The Python API supports more detailed `QuantizationConfig` policies, and not every API field is an experiment key ([Quantization logic](quantization.md)).

## Postprocessing, evaluation and reporting

`run` and `reproduce` use the same model-specific evaluation profiles by default. `--evaluation-profile` is a comparison option for `run`, explained in [Evaluation profiles](evaluation-profiles.md).

| Key | Set in | Default and allowed values | Stage | Effect of a change |
|---|---|---|---|---|
| `decode.*` | Model recipe | Model-specific confidence, IoU, maximum detections and output mapping | Postprocessing | Identical network outputs can give different predictions and metrics. Reevaluate FP32 and INT8 |
| `evaluation.profiles.<model>` | Experiment | Default of the packaged model. See [Evaluation profiles](evaluation-profiles.md) | Postprocessing and evaluation | The profile identity changes. Unsupported combinations are rejected |
| `evaluation.limit` | Experiment, or `--limit` | `null` (full) in both presets. Positive integer or `null`. LFW requires all 6,000 pairs | Evaluation | The evaluation input identity and FP32 pairing change |
| `runtime.device`, `runtime.cuda.gpu_mem_limit` | Experiment, or `--device` | Presets use `cuda:0` and the CPU must be selected explicitly. Memory limit 6 GiB by default, 6 or 8 GiB in the original settings | Inference | The runtime identity changes. The memory limit is part of the reproducible settings |
| `runtime.intra_op_threads`, `.inter_op_threads` | Experiment | 1 and 1. Non-negative integers, where 0 lets ORT choose | Inference | `reproduce` fixes both to 1 |
| `output.save_qdq_models`, `.save_predictions` | Experiment, or `--save-models` and `--save-predictions` | `false` in presets. Boolean | Storage | Part of the completed-result identity |
| `report.primary_metric.<task>` | Experiment | Recipe or preset metric | Reporting | Selects among saved metrics. Only ratio units are allowed, and no inference runs |
| `report.target_mean_recovery` | Experiment | 98 in `core10`, `null` in `cnn30`. Positive number or `null` | Reporting | The summary marks whether each condition's cohort mean recovery meets this target ([FP32 pairing and recovery](reporting.md#fp32-pairing-and-recovery)) |
| `--table-profile opennpu_raw` | `report`, `results` and `reproduce report` | Optional for `report` and `results`, default for `reproduce report` | Reporting | Adds the `raw_data.*` tables with display names and a full/partial split. The evaluation identity is unchanged |

The `report.*` settings are part of the `effective_config.json` identity, and editing the configuration of a completed directory breaks result verification. Use `--format` or `--table-profile` to change presentation, and treat any other setting change as a new experiment.

## Example: change a percentile

Presets can be loaded and copied from an installed package. This example creates a percentile experiment and validates it without running it:

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

`reproduce configs` writes the settings of the original measurements as experiment files for `run`. It needs no original assets.

```bash
opennpu-quant reproduce configs --models vgg16,yolov9_t \
  --output workspace/selected-settings
opennpu-quant run -c workspace/selected-settings/yolov9_t__asym_pc_pot_adaround.json --dry-run
```

Without `--models` or `--model-set`, `reproduce configs` covers the `cnn30` set ([Choose a model set](reproducing-results.md#choose-a-model-set)).

Each file describes one model and one condition, including its percentile, GPU memory limit, AdaRound window, evaluation profile and calibration input hash. To compute recovery in the same run, add `fp32: {}` to the file and select `--conditions fp32,...`, because the generated file contains only its own condition. You still need a matching ONNX bundle, data and environment.

These files drive `run`, which creates and trains new models. `reproduce prepare` and `reproduce run` evaluate the fixed original files instead. In `reproduce`, `--limit full` corresponds to `evaluation.limit: null` here.
