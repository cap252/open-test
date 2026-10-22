# Python API

The quantization core can be used from Python without the CLI. This example builds a tiny model in memory and quantizes it on the CPU. For CLI workflows, see [Run experiments](run-pipeline.md), and for the numerical details, see [Quantization logic](quantization.md).

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

Pass `feeds`, not `feeds()`. Calibration and AdaRound require a zero-argument factory that returns the same ordered inputs on every call.

| API | Purpose |
|---|---|
| `calibrate`, `CalibrationConfig`, `CalibrationStats` | Collect reusable calibration statistics |
| `quantize`, `Scheme`, `QuantizationConfig`, `QuantizationResult` | Create a QDQ model with audit data. When quantizing from stored statistics, `reconstruction_feeds` supplies AdaRound inputs |
| `apply_adaround`, `AdaroundConfig`, `ActivationCacheConfig` | Optimize rounding with bounded caches and checkpoint resume |
| `evaluate`, `EvaluationSample`, `TopKClassification` | Evaluate samples. Custom evaluators implement `reset`, `update` and `finalize` |
| `OrtConfig`, `save_model`, `prepare_feed_cache` | Configure the runtime, save models with identity information, and prepare verified input caches |

`Scheme()` and `QuantizationConfig()` have different defaults, listed in [Policy objects and defaults](quantization.md#9-policy-objects-and-defaults). Importing the package does not load the numerical runtime. AdaRound requires Torch.

## Replace Python postprocessing

For detection, face detection, keypoints and semantic/instance segmentation, reuse the model's metric while replacing its Python decoder:

```python
from opennpu_quant import evaluate, OrtConfig
from opennpu_quant.models.adapter import ModelEvaluator
from opennpu_quant.models.evaluation_contracts import resolve

# bundle and dataset are the prepared ModelBundle and Dataset.
built_in = resolve(bundle.spec.recipe).postprocess

def postprocess(host_outputs, recipe, metadata, categories):
    predictions = built_in(host_outputs, recipe, metadata, categories)
    # Return new predictions here, preserving host_outputs and metadata.
    return predictions

evaluator = ModelEvaluator(bundle, dataset, postprocess=postprocess)
result = evaluate(
    bundle.paths["network"],
    dataset.samples(bundle, limit=3),
    evaluator=evaluator,
    ort=OrtConfig.cpu(),
    limit=3,
)
assert not result.runtime["evaluation_contract"]["reusable"]
```

`ModelEvaluator` and `resolve` come from internal modules rather than the public names. Their signatures are kept as an extension contract for this hook, and the test suite fails if they change.

The callable runs after the CPU host graph and receives `(host_outputs, recipe, metadata, categories)`. `categories` maps model classes to COCO category IDs, and `metadata` holds the coordinate restoration information from preprocessing. Leave these inputs unchanged.

Return predictions in the task's existing format: detection, keypoint or mask rows, face rows of `[x, y, width, height, score]`, or a semantic label array. The metric, ground truth and scoring settings such as the COCO mask-area policy still come from the model's evaluation profile ([Evaluation profiles](evaluation-profiles.md)).

Classification ranking and LFW pair verification use dedicated evaluators and do not accept this hook. To change them, pass your own evaluator to `evaluate(..., evaluator=...)`.

A custom callable has no trusted implementation identity, so its result records `reusable: false` and `original_reproduction_eligible: false`. It is never reused automatically, never compared with the original measurements and cannot be loaded from the CLI or YAML. Each run with a custom callable is a new experiment.
