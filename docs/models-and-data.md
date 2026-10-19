# Prepare models and data

A new experiment needs a prepared dataset and an FP32 ONNX model bundle. This guide covers both, the files that make up a bundle and how to add your own model. Running experiments is covered in [Run experiments](run-pipeline.md).

## Prepare datasets

`data prepare` checks the dataset layout, image counts, ground truth and the separation of calibration and evaluation images. It then writes `<data>/<dataset>/manifest.json`. Image files are referenced in place, not copied. Run only the commands for the datasets you need, and replace the paths with your own.

```bash
opennpu-quant data prepare imagenet --root /data/imagenet --source folder
opennpu-quant data prepare coco --root /data/coco
opennpu-quant data prepare coco_keypoints --root /data/coco
opennpu-quant data prepare widerface --root /data/widerface
opennpu-quant data prepare voc --root /data/VOCdevkit/VOC2012
opennpu-quant data prepare lfw_mobilefacenet --root /data/lfw --download
```

Use the same `--home` or `--data-dir` for preparation and execution. Preparation always indexes the full validation set, even if a later run evaluates only a few samples with `--limit`. `--limit` uses a prefix of the manifest.

| Dataset | Required layout | Full evaluation | Calibration |
|---|---|---:|---|
| ImageNet folder | `train/<wnid>/`, `val/<wnid>/` or `validation/` | 50,000 images | 1,000 training images selected with seed 20260917 |
| ImageNet HF | `train_images/`, `validation_images/` and label information. Use `--source hf` | 50,000 images | Packaged fixed list of 1,000 images |
| COCO | `train2017/`, `val2017/`, `annotations/instances_val2017.json` | 5,000 images | Packaged fixed list of 128 training images |
| COCO keypoints | The COCO images and `annotations/person_keypoints_val2017.json` | 5,000 images | 128 images from the PoseNet list `coco_keypoints.txt` |
| WIDER FACE | `WIDER_train/images/`, `WIDER_val/images/` and the official `.mat` ground truth | 3,226 images | Packaged fixed list of 128 training images |
| VOC2012 | `JPEGImages/`, `SegmentationClass/`, `ImageSets/Segmentation/val.txt` | 1,449 images | Packaged fixed list of 128 training images |
| LFW | `pairs.txt`, `data/lfw_funneled/`, `lfw_funneled.json` landmarks | 6,000 pairs (15,402 views) | Packaged 128 images that do not appear in any evaluation pair |

The two ImageNet layouts select different calibration images. Matching statistics from another run requires the same list, order and preprocessing, not just the same dataset. `--download` is available for LFW and for ImageNet with `--source hf`. The ImageNet download needs authorized Hugging Face access and the data packages described in [Install](../README.md#install).

### LFW

Preparation writes aligned 112×112 face crops into the data directory. `--download` fetches the pair list, images and landmarks from fixed URLs and checks their SHA256 hashes. If any image fails to align, preparation records the failures and writes no manifest.

A full evaluation scores all 6,000 pairs from 15,402 original and horizontally flipped image views. It reports `verification_accuracy`, with thresholds selected over ten folds, and `fixed_angle_accuracy` ([MobileFaceNet on LFW](model-reference.md#notes-on-specific-models)). A limit counts pairs and keeps both views of every image in them.

`--smoke` prepares a fixed subset of two pairs (one positive and one negative) and two calibration images. Scores on this subset use `smoke_pair_accuracy`, a fixed squared-L2 threshold of 1.0, and are never paired with `verification_accuracy`. A limited LFW evaluation in `run` needs this subset and `report.primary_metric.face_verification: smoke_pair_accuracy`, which the `cnn38_smoke` preset sets. In `reproduce`, a limit below 6,000 takes the first pairs of the packaged manifest and also reports `smoke_pair_accuracy`.

## Export FP32 ONNX

```bash
opennpu-quant export resnet18
```

Export downloads the checkpoint and upstream code pinned in the recipe, verifies their SHA256 hash or commit, converts the model and compares the ONNX outputs with the source framework. Validation defaults to `strict`. `--validation functional` records finite numerical differences and continues, but still rejects shape and dtype errors, non-finite values and graph-partition failures. `model.json` records the export result and the numerical parity separately.

Running `export` again converts the model again. To evaluate a bundle you already have, skip export and follow [Evaluate an existing ONNX model](run-pipeline.md#evaluate-an-existing-onnx-model). Use `--weights FILE` to reuse an identical source checkpoint, or `--source-dir DIR` for a clean checkout at the pinned commit.

### Extra export environments

You need these environments only to export models.

| Models | Environment |
|---|---|
| torchvision classifiers and other base Torch models | Included in the `setup.sh` environment |
| timm models, additional YOLO models, NanoDet and others | [cnn38-linux-py311.txt](../requirements/cnn38-linux-py311.txt) |
| The three MMSeg models | A separate interpreter. Follow the Torch 2.1.0 and compiled MMCV 2.1.0 installation order in [cnn38-mmseg.txt](../requirements/cnn38-mmseg.txt) |
| TF SSD and DeepLab | The TensorFlow/tf2onnx packages in the setup environment, or a separate interpreter selected with `--python` if they conflict |

```bash
python -m pip install -r requirements/cnn38-linux-py311.txt
opennpu-quant export deeplabv3_r50_voc --python /path/to/mmseg/bin/python
```

With `--python`, export runs in that interpreter, while quantization and evaluation stay in the main environment. Do not use `mmcv-lite` for MMSeg. The TF DeepLab export checks its output on VOC validation images, so prepare VOC first.

## Model bundles and recipes

```text
<models-dir>/<model>/
├── network.onnx          Quantization and inference target
├── preprocess.onnx       Optional CPU input graph
├── postprocess.onnx      Optional CPU output graph
├── model.json            Export validation and file hashes
└── recipe.yaml           Optional local recipe
```

Include the host graphs a model needs, using the file names in its recipe. If `model.json` exists, loading the bundle verifies the graph hashes. Quantization changes only `network.onnx`, and the CPU host graphs stay in FP32. For some models the recipe input differs from the actual network input, and the [model reference](model-reference.md) lists both.

A recipe defines the inputs, preprocessing, graph boundaries, decoder, metric and source checkpoint. See the packaged [ResNet-18 recipe](../src/opennpu_quant/resources/models/resnet18.yaml) for an example. A local recipe overrides the bundle settings of a packaged model, as described in [Precedence and local recipes](evaluation-profiles.md#precedence-and-local-recipes).

Models with variable input sizes are checked at three different resolutions during export, and their input caches keep each sample's shape.

## Add a custom model

1. Place the ONNX file and any host graphs in `<models-dir>/<name>/`.
2. Copy a similar packaged recipe to `recipe.yaml` and edit the name, input, preprocessing, decoder and graph mappings. When the copy gets a different model name, remove its `decode.profile` or set it to `current_v1`.
3. Copy an experiment file, add the model and its percentile, and give the experiment a new name ([Configuration reference](configuration.md#example-change-a-percentile)).
4. Check the settings with `opennpu-quant run -c my_experiment.yaml --models my_model --dry-run`, then verify inputs, outputs and evaluation on a small run.

If no built-in decoder fits the model's outputs, write a custom evaluator with the [Python API](api.md). Custom `module:function` preprocessing turns off input caching and result reuse, because its identity cannot be determined automatically. Unsupported operators and other graph limits are listed in [Common limitations for custom models](quantization.md#10-common-limitations-for-custom-models).
