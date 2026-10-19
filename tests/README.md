# Development checks

The test suite builds small models and datasets in temporary directories, so it needs no downloads or external assets. Real-model results are summarized in [Validation scope and limitations](../docs/validation.md#what-has-been-verified).

## Run the checks

Run from the repository root after setup:

```bash
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/ruff check src tests tools
.venv/bin/ruff format --check src tests tools
.venv/bin/python -m unittest discover -s tests -t . -v
```

Use `./.tools/uv` if setup installed uv locally. Tests that need Torch, Torchvision or pycocotools run when those packages are installed, as they are in the `setup.sh` environments.

## Fixtures

| File | Contents |
|---|---|
| `fixtures/pot/golden_40.json` | Expected graph and output hashes for a demo model |
| `fixtures/public_api_signatures.json` | The public API contract |
| `fixtures/opennpu_raw_format.json` | Columns, display names and schemes of the 15-column table |
| `library_fixtures.py` | A toy model and feeds shared by tests |

## What each file covers

### Quantization core

| File | Covers |
|---|---|
| `core/test_pot_scales.py` | PoT scale math: exponents, ties, non-finite values, zero-point clipping, MSE candidates (NumPy only) |
| `ort/test_pot_quantization.py` | 480 scale/policy combinations, golden hashes, shared weights, bias scales, INT32 limits, tamper detection |
| `ort/test_adaround.py` | AdaRound changes only weight codes. Conv/Gemm/MatMul variants, checkpoints and resume, budgets, empty bias names, CUDA checks |
| `ort/test_adaround_cache.py` | AdaRound activation cache: LRU, budgets, GPU fallback |
| `ort/test_adaround_memory.py` | Chunked regularizer gradients equal the dense computation |
| `ort/test_input_cache.py` | `prepare_feed_cache` contract, tampering, locks, budgets, cached feeds in AdaRound, CLI cache switches |
| `ort/test_safe_fetch.py` | `ort.fetch.run_tensors` copies and pass-through inputs |

### API, CLI and configuration

| File | Covers |
|---|---|
| `test_api_usability.py` | Public names and signatures, docstrings and doctests, light import, configuration-change messages |
| `test_cli_usability.py` | CLI help, input errors, demo output, local recipes, device validation |
| `test_input_validation.py` | Experiment validation with exact key paths, YAML error locations, filesystem errors, error messages |
| `test_recipe_settings.py` | Removed recipe keys are rejected by YAML loading and direct construction, while active settings and custom preprocessing are kept |
| `test_release_inputs.py` | Local path settings, dataset roots and precedence |
| `test_interface_reduction.py` | Segmentation metrics, range selection, recipe keys, removed CLI options, version |

### Models, export and evaluation

| File | Covers |
|---|---|
| `test_cnn38.py` | 38-model catalog and presets, strict/functional validation, export rewrites (face grids, fixed upsample, DFL split, constant Floor), dynamic input caches and AdaRound windows, decoders, float/PoT scale audits, DepthToSpace |
| `test_tensorflow_export.py` | Mocked SSD/DeepLab converter calls forward the recipe opset (no real TensorFlow conversion) |
| `test_preprocessing_reliability.py` | Custom preprocessing never reuses cached feeds, and built-in caches invalidate on setting or dependency changes |
| `test_metrics.py` | Exact integer Top-K ordering, ties at top-1/top-5, input preservation, WIDER FACE AP edge cases, COCO undefined values |
| `test_segmentation_predictions.py` | A VOC-style run whose saved predictions replay to the same metrics |
| `test_lfw.py` | LFW pair order, ten folds, flip views, smoke subset and metric, pair-based limits, manifest tampering |
| `test_auxiliary_metrics.py` | Fixed-angle normalization bits, FP64 angle threshold boundary, percentage conversion, canonical metric names compared without runtime renaming, diagnostic field paths, conflicting values and bitwise verdicts |
| `test_evaluation_contracts.py` | Profile resolution for every saved recipe, descriptive profile names in experiments, the generic `current_v1` behavior, and the Python postprocessing hook |
| `test_evaluation_defaults.py` | Packaged defaults matching the selection, local bundles, comparison precedence and classifier tie order |
| `test_evaluation_profiles.py` | YOLO profile behavior (NMS, clipping, mask arithmetic, metric area), real evaluator routing and reuse rejection. Needs Torch and Torchvision |

### Runs, reuse and reports

| File | Covers |
|---|---|
| `test_portable.py` | End-to-end local run with resume, reuse and tampering, statistics relocation, downloads and archives, checkout cleanup and ImageNet preparation |
| `test_result_reliability.py` | Result reuse evidence, invalidation by graph or data changes, forced recompute, stale files, original errors, undefined primary metric |
| `test_recalibrate.py` | `--recalibrate` generations, partial and interrupted runs |
| `test_results.py` | `results` tables: units, FP32 pairing, identity mismatches, historical-origin rejection before export, Excel/Unicode, partial denominators |
| `test_final_evidence.py` | Evaluation-identity mismatches |
| `test_raw_exports.py` | 15-column raw tables of run exports, `results --stdout` and `report` rebuilds, including the percentile and scheme columns |
| `test_raw_report.py` | Excel-compatible raw tables for `report` and `reproduce report`: formats, units, full/partial split, denominators, exclusions and tampering |
| `test_review_fixes.py` | Report formats and policies, export Top-5 parity, recipe layout checks, audit block size, gate provenance, optional Torch suite, COCO details |

### The reproduce workflow

| File | Covers |
|---|---|
| `test_gpu_reproduction.py` | Fixed-graph CLI contracts on a small real ONNX model: replay, repeats, comparison, missing references, model/input/profile tampering, resume, relocation, CUDA wheel lookup and separate GPU identities. The installed-wheel checks also load all 266 selected conditions |
| `test_final_reproduction.py` | YOLOv9-T/S family binding, source condition settings and calibration input guards, the independent PoseNet split, original classification ties without changing the public API, mixed full-dataset limits, compact resume and tamper detection, and separate metric and prediction-stream verdicts |
| `test_reproduction_model_sets.py` | The `extended` default, explicit core10/all/model selection, unchanged original entries and condition settings, custom selections, conflicting options, and CPU replay/resume/reporting of packages with additional models |
| `test_reproduction_checks.py` | Raw tensor tolerances, prediction ordering and artifact tampering |
| `test_calibration_reproduction.py` | Bitwise statistics comparisons (including signed zero and one-ULP differences), histogram count drift, missing or non-finite tensors, and direct comparison against QDQ initializers on a demo model |
| `test_reference_metric_migration.py` | Maintainer conversion of reference records to the current metric names: bit-preserving values, hashes, tampering and conflicts, binding collisions, and new packages that run, report, resume and relocate |
| `test_research.py` | Imported measurement records: units, missing measurements, selected percentiles, portable exports and atomic failure handling |
