# Maintenance checks

These source-checkout tools inspect original experiment evidence. They are separate from the public `run` and `reproduce` commands and are not included in the wheel. Ordinary fixed-model evaluation does not require them.

## Calibration reproduction

`check_calibration_reproduction.py` recollects the original ordered calibration inputs and compares statistics and QDQ activation encodings bit for bit. It requires the original model graphs, source records, dataset manifests, calibration images, statistics and histograms. It does not train AdaRound or generate a new QDQ model.

The default [calibration plan](validation/calibration-plan.json) covers the 38-model catalog. Its artifact IDs refer to original asset layouts, not the downloadable model bundle. The release [bindings example](../docs/reproduction-bindings.example.json) does not supply these assets.

Create a local JSON bindings file for the original assets. The plan uses these roots:

| Root ID | Required local contents |
|---|---|
| `research-repo` | Original graphs, statistics, histograms and records at the relative paths recorded in the plan/catalog |
| `research-runs` | Original experiment output trees at their recorded relative paths |
| `final-data-manifests` | The selected LFW calibration data manifest |

The tool also reads the packaged selection, so include every additional binding needed by the chosen model's dataset and graphs. Root values must identify the actual local assets. Adding a placeholder path does not provide the original files. Keep relative paths and recorded SHA256 values intact. Obtain the original data from its maintainer before using this check.

From a source checkout with the required GPU environment and assets:

```bash
python tools/check_calibration_reproduction.py \
  --bindings /absolute/path/to/original-assets-bindings.json \
  --models resnet18 --device 0 --output workspace/calibration-validation
```

Use a fresh output directory. `--plan` can select another plan matching the available original evidence. The tool saves newly collected statistics and comparison details in each model directory. A bit mismatch yields `state: fail` and a nonzero process exit status, but the completed statistics and `result.json` remain available. Missing assets, incompatible inputs or collection errors can stop a condition before final comparison results exist.

The self-contained tests in [test_calibration_reproduction.py](../tests/test_calibration_reproduction.py) exercise comparison logic with small models. They do not attest that the original assets are available or that this complete plan has been rerun.
