# opennpu-quant

opennpu-quant quantizes ONNX models to INT8 and measures how much accuracy they keep relative to FP32. It targets NPU-style integer inference, with optional power-of-two scales and AdaRound.

- **INT8 QDQ models:** signed INT8 activations (per-tensor, symmetric or asymmetric), symmetric INT8 weights (per-tensor or per-channel) and INT32 biases, written as standard ONNX `QuantizeLinear`/`DequantizeLinear` graphs.
- **Scale policies:** MinMax, Percentile or Entropy ranges with unconstrained (`float`) or power-of-two (`pot_nearest`, `pot_ceil`, and `pot_mse` for weights) scales.
- **AdaRound:** learns only the INT8 weight rounding. Scales, zero points and biases stay as RTN set them.
- **Packaged CNN recipes:** the default `core10` set has 10 models and `extended` has 34. They cover classification, detection, face detection, pose estimation, semantic and instance segmentation, and face verification, each with its original preprocessing and postprocessing.
- **Comparable results:** FP32 and INT8 run through the same pipeline, completed work is reused by content hash, and reports export Excel-compatible tables.

## Requirements

- Linux x86_64.
- An NVIDIA GPU with a compatible driver for CUDA runs. A system CUDA toolkit is not needed.
- CPU-only machines work for functional checks, and their numbers can differ from GPU runs.

`setup.sh` installs Python 3.11.13 and the pinned dependencies into `.venv`, using `uv` from `PATH` or a checksum-verified copy in `.tools/`.

## Install

```bash
./setup.sh --device cuda
source .venv/bin/activate
opennpu-quant info
```

On a CPU-only machine, use `./setup.sh --device cpu` and pass `--device cpu` to the commands below.

To download ImageNet from Hugging Face, add `--data` to your setup command, for example `./setup.sh --device cpu --data`. Each run of `setup.sh` re-syncs `.venv` to the selected device profile, and the default profile is CUDA. Always repeat the `--device` you installed with.

## Quick start

### 1. Run the demo

```bash
opennpu-quant demo --device cuda:0
```

The demo calibrates, quantizes and evaluates a tiny built-in model without downloading anything. Its output shows the `fp32` score and the INT8 `results`.

### 2. Prepare data and an FP32 model

```bash
opennpu-quant data prepare imagenet --root /data/imagenet
opennpu-quant export resnet18
```

Replace `/data/imagenet` with your ImageNet folder (`train/<wnid>/` and `val/<wnid>/`). `data prepare` indexes the images without copying them. `export` downloads the pinned ResNet-18 checkpoint, converts it to ONNX and checks the result against PyTorch. Other datasets and models are covered in [Prepare models and data](docs/models-and-data.md).

### 3. Quantize and evaluate

```bash
opennpu-quant run -c quick --models resnet18 --conditions fp32,pot_rtn \
  --calibration-samples 2 --limit 2 --device cuda:0 \
  --save-models --output workspace/example-runs
```

This collects activation statistics from two images, builds a PoT RTN model and evaluates FP32 and INT8 on two images each. The INT8 model is saved as `workspace/example-runs/quick/resnet18/pot_rtn/model.onnx`. The accuracy is computed on two images only, so it does not represent ImageNet accuracy.

### 4. Export the report

```bash
opennpu-quant report workspace/example-runs/quick --table-profile opennpu_raw
```

Because this run evaluated only two images, its rows go to `workspace/example-runs/quick/raw_data.partial.csv`. Full evaluations go to `raw_data.csv`. Both use the same 15 columns, described in [Reports and Excel export](docs/reporting.md), and [How to read a result](docs/validation.md#how-to-read-a-result) explains what partial and full results show.

## Presets and full evaluation

The `quick` preset caps every run at 8 calibration and 32 evaluation samples, even without `--limit` and `--calibration-samples`. A full evaluation needs a preset that evaluates whole validation sets:

| Preset (`-c`) | Models | Conditions | Typical use |
|---|---|---|---|
| `quick` | ResNet-18, YOLOv5s | FP32, PoT RTN, float RTN | First experiments, 32 evaluation samples |
| `core10` (default) | 10 core models | FP32 and RTN/AdaRound for Asym/PC PoT and Asym/PC float | Full evaluation of the core set |
| `extended` | 34 models | FP32 and RTN/AdaRound for Asym/PC float, Asym/PC PoT and Sym/PT PoT | Broad coverage |

Switch to such a preset, drop the two options and write to a new output directory:

```bash
opennpu-quant run -c core10 --models resnet18 --conditions fp32,pot_rtn \
  --device cuda:0 --output workspace/resnet18-full
```

This calibrates on 1,000 ImageNet images and evaluates ResNet-18 on all 50,000 validation images. To run every model and condition of the preset, also drop `--models` and `--conditions`:

```bash
opennpu-quant run -c core10 --device cuda:0 --output workspace/core10-full
```

The AdaRound conditions of `core10` and `extended` train for 10,000 steps per layer and can take hours per model. To try AdaRound on a small scale first, follow [Add an AdaRound condition](docs/configuration.md#example-add-an-adaround-condition).

`opennpu-quant models` lists every packaged model, including a few larger YOLO variants outside `extended`. [Run experiments](docs/run-pipeline.md) covers all presets, run options, caching and output files, and the [model reference](docs/model-reference.md) gives each model's input, preprocessing, decoder and group.

## Python API

The quantization core works without the CLI:

```python
from opennpu_quant import OrtConfig, Scheme, calibrate, quantize, save_model

ort = OrtConfig.cpu()
stats = calibrate(model, feeds, ort=ort)
scheme = Scheme(activation_scale="pot_ceil", weight_scale="pot_ceil")
result = quantize(model, stats, scheme, ort=ort)
save_model(result.model, "model_int8.onnx")
```

`feeds` is a function that returns a fresh iterator of input dictionaries. See the [Python API guide](docs/api.md) for a complete example.

## Reproduce the original measurements

`reproduce` re-evaluates the fixed FP32 and INT8 models of the original measurements without exporting, calibrating or training anything. It requires the original model files, records and data manifests, which are not distributed with this repository. If you have them, follow [Reproduce the original measurements](docs/reproducing-results.md). `reproduce configs` works without these assets and writes the original per-model settings as experiment files for `run`.

## Documentation

- **Guides:** [Prepare models and data](docs/models-and-data.md), [Run experiments](docs/run-pipeline.md), [Reports and Excel export](docs/reporting.md), [Python API](docs/api.md), [Reproduce the original measurements](docs/reproducing-results.md)
- **Reference:** [Configuration reference](docs/configuration.md), [Model reference](docs/model-reference.md), [Evaluation profiles](docs/evaluation-profiles.md), [Validation scope and limitations](docs/validation.md)
- **Internals:** [How opennpu-quant works](docs/architecture.md), [Quantization logic](docs/quantization.md), [Development checks](tests/README.md)

The [documentation index](docs/README.md) describes each page.

## License

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY.md).
