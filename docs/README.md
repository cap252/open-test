# Documentation

Start with the [README](../README.md) to install the tool, run the demo and quantize a first model. The pages below are grouped by what you want to do.

## Guides

| Page | Use it to |
|---|---|
| [Prepare models and data](models-and-data.md) | Prepare datasets, export FP32 models and add your own model |
| [Run experiments](run-pipeline.md) | Choose presets and options, follow a run and reuse completed work |
| [Reports and Excel export](reporting.md) | Rebuild reports, export the 15-column tables and import them into Excel |
| [Python API](api.md) | Use the quantization core and evaluator from Python |
| [Reproduce the original measurements](reproducing-results.md) | Re-evaluate the original fixed models (requires the original assets) |

## Reference

| Page | Contents |
|---|---|
| [Configuration reference](configuration.md) | Every experiment key, its default and what changing it affects |
| [Model reference](model-reference.md) | Inputs, preprocessing, decoders, model groups and original settings for every packaged model |
| [Evaluation profiles](evaluation-profiles.md) | Model-specific postprocessing behavior and how to compare alternatives |
| [Validation scope and limitations](validation.md) | How to read a result, what has been verified and known limitations |

## Internals

| Page | Contents |
|---|---|
| [How opennpu-quant works](architecture.md) | The six stages from an image to a report row, and a map of the code |
| [Quantization logic](quantization.md) | Calibration, range selection, scales, QDQ construction and AdaRound |
| [Development checks](../tests/README.md) | The self-contained test suite and what each file covers |

The `tools/` directory holds maintainer scripts, described in [Maintainer tools](reproducing-results.md#maintainer-tools). Normal use does not need them.
