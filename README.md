# opennpu-quant

opennpu-quant turns FP32 ONNX models into INT8 QDQ models, with optional power-of-two scales and AdaRound, and measures their accuracy against FP32. The default `core10` group covers 10 models, and `cnn30` covers 30 across classification, object and face detection, semantic and instance segmentation, and face verification.

Preprocessing prepares network inputs. Calibration collects FP32 activation statistics, and quantization builds the INT8 graph. Network outputs pass through postprocessing and task evaluation, then a report pairs INT8 with FP32 to calculate recovery. Fixed-model `reproduce` starts from existing ONNX graphs and skips calibration and training. [Architecture](docs/architecture.md) explains the stages.

## Requirements

- Linux x86_64 and Python 3.11.13 for the tested setup.
- An NVIDIA GPU with a compatible driver for CUDA execution. A system CUDA toolkit is not needed.
- CPU execution is available for installation and pipeline checks. Its numbers can differ from GPU results.

## Install

```bash
./setup.sh --device cuda
source .venv/bin/activate
opennpu-quant info
```

`setup.sh` installs Python 3.11.13 and pinned dependencies into `.venv`, using `uv` from `PATH` or a checksum-verified copy in `.tools/`. For CPU checks, use `./setup.sh --device cpu` and pass `--device cpu` below. Re-running setup re-syncs the environment, so repeat the intended device profile. Add `--data` when preparing ImageNet from Hugging Face. The base profiles include LFW alignment dependencies. Model-specific export dependencies and their installation commands are listed in [Extra export environments](docs/models-and-data.md#extra-export-environments). The CPU profile uses the CPU provider in the pinned GPU-capable ORT package.

## Quick start

### 1. Check the installation

```bash
opennpu-quant demo --device cuda:0
```

The demo calibrates, quantizes and evaluates a small built-in model without downloading assets.

<a id="2-check-supplied-onnx-models-on-a-few-images"></a>

### 2. Check supplied ONNX models

You can download the ONNX files from GitHub Releases.

Prepare the evaluation datasets using the [dataset guide](docs/models-and-data.md#prepare-datasets). Before running the commands below, follow [Set up model and data paths](docs/reproducing-results.md#bind-the-assets) to prepare `workspace/repro/bindings.json`. This configuration file specifies where the downloaded assets are stored.

Start with ResNet-18 and YOLOv5s from `core10`. The example uses three images per model for a quick execution check of their saved FP32 and PoT RTN graphs. You can also evaluate the full dataset for each model:

```bash
opennpu-quant reproduce prepare --selection bundle/selections/selected-210.json \
  --bindings workspace/repro/bindings.json \
  --models resnet18,yolov5s --conditions fp32,asym_pc_pot_rtn --limit 3 \
  --output workspace/repro/representative-3
opennpu-quant reproduce run --package workspace/repro/representative-3 \
  --bindings workspace/repro/bindings.json --device cuda:0 \
  --output workspace/repro/representative-3-run
opennpu-quant reproduce report --run workspace/repro/representative-3-run \
  --package workspace/repro/representative-3 --output workspace/repro/representative-3-tables \
  --table-profile opennpu_raw
```

Check for four ready conditions, four completed conditions and `full: 0 / partial: 4 / excluded: 0`. The results are in `workspace/repro/representative-3-tables/raw_data.partial.csv` and `.tsv`. These scores describe only the three-image subsets.

For full evaluation of the same models and conditions, set `--limit full` in `reproduce prepare` and replace `representative-3` with `representative-full` in all three commands. Use the new package and output directories throughout. A completed full evaluation reports `full: 4 / partial: 0 / excluded: 0` and writes `raw_data.csv` and `.tsv`. The next section evaluates all 30 models.

### 3. Evaluate all selected conditions

First [generate and verify the LFW input cache locally](docs/reproducing-results.md#generate-the-lfw-input-cache). LFW images and input tensors are not included in the model release. With all assets available, evaluate `cnn30`: 30 models with FP32 and six selected INT8 conditions each, or **210 conditions**. This reuses the supplied RTN and AdaRound graphs without training them again.

```bash
opennpu-quant reproduce prepare --selection bundle/selections/selected-210.json \
  --bindings workspace/repro/bindings.json \
  --model-set cnn30 --limit full --output workspace/repro/cnn30-full
opennpu-quant reproduce run --package workspace/repro/cnn30-full \
  --bindings workspace/repro/bindings.json --device cuda:0 --storage predictions \
  --output workspace/repro/cnn30-full-run
opennpu-quant reproduce report --run workspace/repro/cnn30-full-run \
  --package workspace/repro/cnn30-full --output workspace/repro/cnn30-full-tables \
  --table-profile opennpu_raw
```

Check **210 ready**, **210 completed**, and **`full: 210 / partial: 0 / excluded: 0`** separately. Import `workspace/repro/cnn30-full-tables/raw_data.csv` or `.tsv` into Excel. The 15-column report uses this execution's measured values. Original metric comparisons are in each condition's `result.json` under `original_accuracy`. Execution completion and numerical agreement are separate verdicts.

LFW requires all 6,000 pairs. A small request that includes MobileFaceNet stops before any model executes, so specify the representative models for small checks. The [reproduction guide](docs/reproducing-results.md) covers asset checks, storage, resume and comparisons between servers.

<a id="create-new-models-when-needed"></a>

## Create new INT8 models

Use this workflow to create an INT8 model from an existing trained checkpoint: prepare the dataset, export FP32 ONNX, collect calibration statistics, then quantize and evaluate. `export resnet18` downloads the pinned pretrained checkpoint and writes the FP32 bundle to `workspace/models/resnet18/`. `run` evaluates FP32 and creates and evaluates the selected INT8 model.

First download and extract ImageNet using the [dataset guide](docs/models-and-data.md#prepare-datasets). The `data prepare` command below checks the local files and writes a manifest; it does not download images. This example uses two calibration images and two evaluation images for a quick execution check:

```bash
opennpu-quant data prepare imagenet --root /data/imagenet --source folder
opennpu-quant export resnet18
opennpu-quant run -c core10 --models resnet18 --conditions fp32,pot_rtn \
  --calibration-samples 2 --limit 2 --device cuda:0 \
  --save-models --output workspace/example-runs
opennpu-quant report workspace/example-runs/core10 --table-profile opennpu_raw
```

Preparation requires all 50,000 ImageNet validation images and the calibration inputs, even when this example evaluates only two images. With `--save-models`, the generated INT8 QDQ model is saved to `workspace/example-runs/core10/resnet18/pot_rtn/model.onnx`. The report at `workspace/example-runs/core10/raw_data.partial.csv` records FP32 and INT8 accuracy and recovery for the evaluated subset, so you can inspect the effect of quantization. See [model and data preparation](docs/models-and-data.md) and [run experiments](docs/run-pipeline.md) for full evaluation and AdaRound training.

| Preset for `run` | Models | Conditions | Evaluation scope |
|---|---|---|---|
| `core10` (default) | 10 | FP32, Asym/PC PoT and float RTN/AdaRound | Full datasets |
| `cnn30` | 30 | FP32, Asym/PC float and PoT, Sym/PT PoT RTN/AdaRound | Full datasets |

A successful fixed-model evaluation verifies the supplied graphs and evaluation path. Recreating their quantization statistics and trained weights requires separate regeneration checks. The [configuration reference](docs/configuration.md) explains how to change an experiment or generate original settings with `reproduce configs`.

## Validation scope

Assess execution completion, reference agreement and benchmark accuracy separately. Benchmark accuracy requires full evaluation of each selected model and condition. For LFW, use all 6,000 pairs. Rescoring saved outputs checks postprocessing and metric calculation; verifying inference on a new GPU or newly generated INT8 models requires a new evaluation. See [validation scope](docs/validation.md) for the evidence and limits.

## Python API

The Python API is available when you need to integrate quantization into your own Python workflow. With an FP32 model and a calibration input function prepared:

```python
from opennpu_quant import OrtConfig, Scheme, calibrate, quantize, save_model

ort = OrtConfig.cpu()
stats = calibrate(model, feeds, ort=ort)
scheme = Scheme(activation_scale="pot_ceil", weight_scale="pot_ceil")
result = quantize(model, stats, scheme, ort=ort)
save_model(result.model, "model_int8.onnx")
```

`feeds` supplies preprocessed calibration inputs as dictionaries mapping model input names to NumPy arrays. Pass the function itself; each call must return a fresh iterator with the same values and order, because calibration reads the inputs twice. See the [Python API guide](docs/api.md) for a complete example.

## Documentation

- **Guides:** [Prepare models and data](docs/models-and-data.md), [Run experiments](docs/run-pipeline.md), [Reports and Excel export](docs/reporting.md), [Python API](docs/api.md), [Reproduce the original measurements](docs/reproducing-results.md)
- **Reference:** [Configuration reference](docs/configuration.md), [Model reference](docs/model-reference.md), [Evaluation profiles](docs/evaluation-profiles.md), [Validation scope and limitations](docs/validation.md)
- **Internals:** [How opennpu-quant works](docs/architecture.md), [Quantization logic](docs/quantization.md), [Development checks](tests/README.md)

The [documentation index](docs/README.md) describes each page.

## License

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY.md).
