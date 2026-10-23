# Development checks

The test suite builds small models and datasets in temporary directories, so it needs no downloads or external assets. Real-model results are summarized in [Validation scope and limitations](../docs/validation.md#what-has-been-verified).

## Run the checks

Run from the repository root after setup:

```bash
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/ruff check src tests tools
.venv/bin/ruff format --check src tests tools
.venv/bin/python tools/build_presets.py --check
.venv/bin/python -m unittest discover -s tests -t . -v
```

Use `./.tools/uv` if setup installed uv locally. Tests that need Torch, Torchvision or pycocotools run when those packages are installed, as they are in the `setup.sh` environments.

## Fixtures

| File | Contents |
|---|---|
| `fixtures/pot/golden_40.json` | Expected graph and output hashes for a demo model |
| `fixtures/public_api_signatures.json` | The public API contract |
| `fixtures/preset_contracts.json` | Ordered effective configurations and condition lists for the two standalone presets. Update only for an intentional settings change |
| `fixtures/documented_internals.json` | Signatures of the non-public names that README and docs import |
| `fixtures/opennpu_raw_format.json` | Columns, display names and schemes of the 15-column table |
| `library_fixtures.py` | A toy model and feeds shared by tests |

## Documented API compatibility

The API guide imports model and dataset helpers from their implementation modules. These names are not exported by the top-level `opennpu_quant` package, but the documented imports are part of the examples users rely on. `test_documented_internal_imports_keep_their_signatures` checks both the complete set of these imports and their signatures against `fixtures/documented_internals.json`. Keep the fixture in sync when intentionally adding a documented helper, and review compatibility before changing a listed signature.

The Python postprocessing example uses `ModelEvaluator` and `resolve` as extension points. Behavioral tests in `test_evaluation_contracts.py` also check execution order, argument forwarding, reuse of the existing metric, and the flags that exclude custom postprocessing from automatic result reuse and original-measurement reproduction. The basic package import is checked separately to ensure that it creates no files and does not load numerical runtimes.

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
| `test_api_usability.py` | Public names and signatures, signatures of non-public names used in docs, docstrings and doctests, light import, configuration-change messages |
| `test_cli_usability.py` | CLI help, input errors, demo output, local recipes, device validation |
| `test_input_validation.py` | Experiment validation with exact key paths, YAML error locations, filesystem errors, error messages |
| `test_recipe_settings.py` | Removed recipe keys are rejected by YAML loading and direct construction, while active settings and custom preprocessing are kept |
| `test_release_inputs.py` | Local path settings, dataset roots and precedence |
| `test_interface_reduction.py` | Segmentation metrics, range selection, recipe keys, removed CLI options, version |

### Models, export and evaluation

| File | Covers |
|---|---|
| `test_cnn38.py` | The exact cnn30 model order and task counts, unchanged core10 and cnn30 settings, the 38-model catalog, strict/functional validation, export rewrites (face grids, fixed upsample, DFL split, constant Floor), dynamic input caches and AdaRound windows, decoders, float/PoT scale audits, DepthToSpace |
| `test_tensorflow_export.py` | Mocked SSD/DeepLab converter calls forward the recipe opset (no real TensorFlow conversion) |
| `test_model_bundle.py` | Exported folders with and without `model.json`, mapping precedence, tampering, explicit graph paths and output aliases |
| `test_preprocessing_reliability.py` | Custom preprocessing never reuses cached feeds, and built-in caches invalidate on setting or dependency changes, or on ORT code changes when a CPU graph runs |
| `test_metrics.py` | Exact integer Top-K ordering, ties at top-1/top-5, input preservation, WIDER FACE AP edge cases, COCO undefined values |
| `test_segmentation_predictions.py` | A VOC-style run whose saved predictions replay to the same metrics |
| `test_recovery.py` | Common recovery operation order, exact FP32 self-comparison, mixed units, missing denominators and serialized precision |
| `test_lfw.py` | Full LFW pair order, ten folds, flip views, unchanged calibration input selection, early rejection of partial mixed-model requests and archived artifacts, manifest tampering |
| `test_auxiliary_metrics.py` | Fixed-angle normalization bits, high-precision arccos references, same-environment repeatability, FP64 angle threshold boundary, percentage conversion, canonical metric names, diagnostic field paths, conflicting values and bitwise verdicts |
| `test_evaluation_contracts.py` | Profile resolution for every saved recipe, descriptive profile names in experiments, the generic `current_v1` behavior, and the Python postprocessing hook |
| `test_evaluation_defaults.py` | Packaged defaults matching the selection, local bundles, comparison precedence and classifier tie order |
| `test_evaluation_profiles.py` | YOLO profile behavior (NMS, clipping, mask arithmetic, metric area), real evaluator routing and reuse rejection. Needs Torch and Torchvision |

The fixed-angle math check uses [20 high-precision reference cases](fixtures/face_angle_reference.json), including endpoints and inputs near the 74.18-degree threshold. The references were computed from exact FP32 inputs at 100 and 200 decimal digits, with matching stored digits. The test requires arccos error within one FP32 spacing of the mathematical reference. Its degree-conversion budget includes that error propagated by 180/pi and one additional FP32 spacing. This is a numeric test contract for these cases, not a guarantee about every input or CPU. Normalization, threshold semantics and same-environment repeatability are checked separately. Runtime metric comparisons and bitwise reproduction verdicts are unchanged.

### Runs, reuse and reports

| File | Covers |
|---|---|
| `test_portable.py` | End-to-end local run with resume, reuse and tampering, statistics relocation, downloads and archives, checkout cleanup and ImageNet preparation |
| `test_dataset_download.py` | COCO/VOC downloads from small local archives, CLI preparation, keypoint inputs, VOC layout, reuse and missing-file repair, checksums, conflicting files, unsafe archives and existing manifest validation |
| `test_result_reliability.py` | Result reuse evidence, invalidation by graph or data changes, forced recompute, stale files, original errors, undefined primary metric |
| `test_cache_concurrency.py` | Concurrent input and statistics readers, one input-cache builder, waiting writers, failure recovery, process exit, interruption and integrity checks after waiting |
| `test_recalibrate.py` | `--recalibrate` generations, partial and interrupted runs, the source files in the statistics cache key |
| `test_results.py` | `results` tables: units, FP32 pairing, identity mismatches, historical-origin rejection before export, Excel/Unicode, partial denominators |
| `test_final_evidence.py` | Evaluation-identity mismatches |
| `test_raw_exports.py` | 15-column raw tables of run exports, `results --stdout` and `report` rebuilds, including the percentile and scheme columns |
| `test_raw_report.py` | Excel-compatible raw tables for `report` and `reproduce report`: formats, units, full/partial split, denominators, exclusions and tampering |
| `test_review_fixes.py` | Report formats and policies, export Top-5 parity, recipe layout checks, audit block size, gate provenance, optional Torch suite, COCO details |

### The reproduce workflow

| File | Covers |
|---|---|
| `test_gpu_reproduction.py` | Fixed-graph CLI contracts on a small real ONNX model: replay, repeats, comparison, missing references, model/input/profile tampering, resume, relocation, CUDA wheel lookup and separate GPU identities, an FP32/INT8 pair with QDQ audits and recovery, output aliases and restored Torch settings. The installed-wheel checks also load all 266 selected conditions |
| `test_final_reproduction.py` | YOLOv9-T/S family binding, source condition settings and calibration input guards, the independent PoseNet split, original classification ties without changing the public API, mixed full-dataset limits, compact resume and tamper detection, and separate metric and prediction-stream verdicts |
| `test_reproduction_model_sets.py` | The `cnn30` default, explicit core10/cnn30/all/model selection, unchanged original entries and condition settings, custom selections, conflicting options, and CPU replay/resume/reporting of packages with additional models |
| `test_repeat_outputs.py` | Replay repeats through `run_tensors`: requested output order, independent copies of pass-through outputs, missing inputs and compact resume |
| `test_reproduction_checks.py` | Raw tensor tolerances, prediction ordering and artifact tampering |
| `test_calibration_reproduction.py` | Bitwise statistics comparisons (including signed zero and one-ULP differences), histogram count drift, missing or non-finite tensors, and direct comparison against QDQ initializers on a demo model |
| `test_reference_metric_migration.py` | Maintainer conversion of reference records to the current metric names: bit-preserving values, hashes, tampering and conflicts, binding collisions, and new packages that run, report, resume and relocate |
| `test_preset_generation.py` | Standalone preset generation, read-only drift detection, independent configs, policy conversion equivalence and the runtime/tool import boundary |
