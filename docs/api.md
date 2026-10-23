# Python API

Use the Python API when you need to integrate quantization into your own Python workflow. Start with the quantization example below. The optional postprocessing section is for users who want to change how model outputs become predictions while keeping the existing accuracy calculation.

Run the examples from the repository root after [installing and activating the environment](../README.md#install). For CLI workflows, see [Run experiments](run-pipeline.md), and for the numerical details, see [Quantization logic](quantization.md).

## Quantize a model

The workflow is **prepare an FP32 model and calibration inputs → collect statistics → create an INT8 QDQ model → save it**. This example builds a tiny model in memory and runs on the CPU, so no model or dataset downloads are needed.

```python
import numpy as np
from onnx import helper as h, numpy_helper as n
from opennpu_quant import OrtConfig, Scheme, calibrate, quantize, save_model

model = h.make_model(
    h.make_graph(
        [h.make_node("Gemm", ["images", "w", "b"], ["logits"], name="projection")],
        "tiny",
        [h.make_tensor_value_info("images", 1, [1, 4])],
        [h.make_tensor_value_info("logits", 1, [1, 6])],
        [
            n.from_array(np.arange(24, dtype=np.float32).reshape(4, 6) / 40 - 0.3, "w"),
            n.from_array(np.arange(6, dtype=np.float32) / 10, "b"),
        ],
    ),
    opset_imports=[h.make_opsetid("", 18)],
    ir_version=10,
)


def feeds():
    for i in range(8):
        yield {"images": np.asarray([[i / 8, -0.5, 0.3, 0.1]], np.float32)}


ort = OrtConfig.cpu()
stats = calibrate(model, feeds, ort=ort)
result = quantize(
    model, stats,
    Scheme(activation_scale="pot_ceil", weight_scale="pot_ceil"), ort=ort,
)
save_model(result.model, "workspace/api-demo.onnx")
```

`calibrate` returns activation statistics in `stats`. `Scheme` selects the quantization settings, and `quantize` uses the model and statistics to create the QDQ model in `result.model`. `save_model` writes it to `workspace/api-demo.onnx` and creates the parent directory if needed. This example creates a model; measuring its accuracy is a separate evaluation step.

## Prepare calibration inputs

`feeds` supplies the input data used for calibration. Each item is a dictionary mapping model input names to NumPy arrays. In the example, `"images"` matches the model's input name, and each array has shape `[1, 4]` and dtype `float32`. For your own model, provide representative calibration data with the preprocessing, input names, shapes and dtypes that the model expects. Calibration does not need ground-truth labels.

Calibration reads the data twice: first to collect minimum and maximum activation values, then to build histograms. Pass the `feeds` function itself, as in `calibrate(model, feeds, ort=ort)`, so the library can call it again to start a new pass. Each call must supply the same values in the same order. The `feeds()` expression in this example creates a one-pass iterator and is not the argument expected by `calibrate`.

AdaRound also reads inputs repeatedly, so prepare its input function in the same way. If preprocessing includes random transformations or shuffling, prepare a fixed sequence before calibration so repeated reads do not change the data.

## Choose an API

| API | Purpose |
|---|---|
| `calibrate`, `CalibrationConfig`, `CalibrationStats` | Collect activation statistics and control how they are collected |
| `quantize`, `Scheme`, `QuantizationConfig`, `QuantizationResult` | Create a QDQ model with the selected quantization settings |
| `apply_adaround`, `AdaroundConfig`, `ActivationCacheConfig` | Optimize weight rounding with AdaRound and configure its training and caches |
| `evaluate`, `EvaluationSample`, `TopKClassification` | Measure model accuracy using prepared samples and an evaluator, such as Top-K classification |
| `OrtConfig`, `save_model`, `prepare_feed_cache` | Choose the execution device, save a model, or cache prepared calibration inputs |

`Scheme()` and `QuantizationConfig()` have different defaults, listed in [Policy objects and defaults](quantization.md#9-policy-objects-and-defaults). AdaRound requires Torch. When enabling AdaRound in `quantize` with previously collected statistics, also supply its input function through `reconstruction_feeds`.

<a id="replace-python-postprocessing"></a>

## Optional: customize postprocessing

Use this section when you want to modify predictions before they are scored. For example, you can filter detections by confidence while continuing to use the model's existing accuracy metric. This option supports detection, face detection, keypoints and semantic/instance segmentation.

### Prepare the model and dataset

The following example uses YOLOv5s and COCO. Prepare COCO using [Download and prepare COCO 2017 or VOC2012](models-and-data.md#download-and-prepare-coco-2017-or-voc2012), then run these commands from the repository root. Replace `/data/coco` with your dataset location. `data prepare` checks those local files; `export` downloads the pinned model checkpoint and creates the ONNX files.

```bash
opennpu-quant data prepare coco --root /data/coco --home workspace
opennpu-quant export yolov5s --home workspace
```

These steps create the dataset manifest at `workspace/data/coco/manifest.json` and the model bundle at `workspace/models/yolov5s/`. The bundle contains `network.onnx` and `postprocess.onnx`. If these assets are already prepared at those paths, continue with the Python example.

### Change postprocessing and evaluate

`ModelSpec` loads the model settings, `ModelBundle` opens its ONNX files, and `Dataset` reads the prepared data manifest. Together they provide the `bundle` and `dataset` used below.

`resolve` selects the model's existing evaluation behavior; its `.postprocess` attribute gives us the default Python postprocessing function. `ModelEvaluator` connects the replacement function to the model's existing metric. Import these helpers from the module paths shown in the example.

This example keeps detections with a score of at least `0.25`. Adjust that filtering step to try your own postprocessing.

```python
from pathlib import Path

from opennpu_quant import evaluate, OrtConfig
from opennpu_quant.data.samples import Dataset
from opennpu_quant.models.adapter import ModelEvaluator
from opennpu_quant.models.bundle import ModelBundle
from opennpu_quant.models.evaluation_contracts import resolve
from opennpu_quant.models.spec import ModelSpec

model_dir = Path("workspace/models/yolov5s")
spec = ModelSpec.for_run("yolov5s", model_dir)
bundle = ModelBundle.from_directory(spec, model_dir)
dataset = Dataset("workspace/data/coco/manifest.json")
built_in = resolve(bundle.spec.recipe).postprocess


def postprocess(host_outputs, recipe, metadata, categories):
    predictions = built_in(host_outputs, recipe, metadata, categories)
    return [row for row in predictions if row["score"] >= 0.25]


evaluator = ModelEvaluator(bundle, dataset, postprocess=postprocess)
result = evaluate(
    bundle.paths["network"],
    dataset.samples(bundle, limit=3),
    evaluator=evaluator,
    ort=OrtConfig.cpu(),
    expected_samples=len(dataset.rows("validation")),
    limit=3,
)
print(result.metrics)
print(result.samples, result.complete)
```

The example prints the accuracy metrics and `3 False`: three images were evaluated, so this is not a complete COCO evaluation. To evaluate all validation images, remove `limit=3` from both `dataset.samples` and `evaluate`, keeping `expected_samples` unchanged. Filtering detections can change accuracy; use a full evaluation to measure that effect.

### Inputs and results of your postprocessing function

Your `postprocess` function runs after the model's CPU postprocessing graph, when one is present. It receives four arguments:

| Argument | What it contains |
|---|---|
| `host_outputs` | Output tensors after the model's CPU postprocessing graph, or the network outputs when there is no such graph |
| `recipe` | The model's input, preprocessing, postprocessing and evaluation settings |
| `metadata` | Per-image information from preprocessing, including the information needed to restore coordinates to the original image |
| `categories` | The mapping from model classes to COCO category IDs for COCO detection tasks |

Keep these arguments unchanged and return the modified predictions. Calling the default function first, as above, gives you predictions in the format the evaluator expects. When writing a different decoder, preserve the task's output format: detection, keypoint or mask rows; face rows of `[x, y, width, height, score]`; or a semantic label array. The ground truth, metric and scoring settings continue to come from the model's [evaluation profile](evaluation-profiles.md).

Classification ranking and LFW pair verification use dedicated evaluators, so they do not support this postprocessing replacement. To change their behavior, provide your own evaluator to `evaluate(..., evaluator=...)`. An evaluator starts a measurement with `reset`, processes each output and sample with `update`, and returns the metrics with `finalize`.

Each evaluation with custom postprocessing is treated as a new experiment. The tool cannot automatically verify that user-written code is unchanged, so it does not reuse these results or treat them as reproduction of the original measurements. The result records `reusable: false` and `original_reproduction_eligible: false`. Pass the function from Python; CLI and YAML configuration do not load custom postprocessing functions.
