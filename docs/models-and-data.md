# Prepare models and data

A new experiment needs a prepared dataset and an FP32 ONNX model bundle. This guide covers both, the files that make up a bundle and how to add your own model. Running experiments is covered in [Run experiments](run-pipeline.md).

## Prepare datasets

`data prepare` prepares data for execution: it checks the dataset layout, image counts, ground truth and the separation of calibration and evaluation images, then writes `<data>/<dataset>/manifest.json`. With `--download`, it also downloads the inputs for COCO 2017, COCO keypoints, VOC2012, ImageNet HF or LFW. Without this option it uses local files. LFW preparation also writes aligned face crops.

For `run`, follow the order **download and extract → prepare the manifest → run the evaluation**. `--download` combines the first two steps for supported datasets. For fixed-model `reproduce`, use the original manifests and path bindings described in [Bind the assets](reproducing-results.md#bind-the-assets); newly prepared manifests do not replace those fixed inputs.

### Download and prepare COCO 2017 or VOC2012

Run the command for the dataset you need. It downloads the archives, checks their SHA256 hashes, extracts the files into the required layout and prepares the manifest:

```bash
opennpu-quant data prepare coco --root /data/coco --download
opennpu-quant data prepare voc --root /data/voc --download
```

COCO uses its official S3 bucket. VOC2012 uses the [archive mirror](https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar) listed by TensorFlow Datasets, with a fixed checksum. For VOC, `--root` is the final dataset directory: the command places `JPEGImages/`, `SegmentationClass/` and `ImageSets/` directly under `/data/voc`. No manual extraction or directory renaming is needed.

A fresh COCO download includes the full training and validation archives plus annotations, about 20.4 GB in total; VOC2012 is about 2 GB. Allow additional disk space for the extracted files and temporary extraction. Calibration still uses the packaged list of 128 images, and preparation indexes the full evaluation set.

Existing required files are reused. Archives are cached under `<root>/.downloads` and checked before extraction; rerunning the command can restore missing files from the cache. If an archive conflicts with an existing file, preparation stops without replacing that file. An interrupted archive download restarts on the next attempt.

For PoseNet, use `opennpu-quant data prepare coco_keypoints --root /data/coco --download`. It reuses the COCO images and prepares the keypoint annotations with PoseNet's calibration list.

### Download COCO, WIDER FACE and VOC2012

WIDER FACE requires a separate download. The following sources also support manual preparation of COCO and VOC2012 if you already manage your own datasets. The training images supply calibration inputs, and the validation images and annotations supply evaluation inputs.

| Dataset | Download location | Files to obtain and extract |
|---|---|---|
| COCO 2017 | [Official downloads](https://cocodataset.org/#download) | `train2017.zip`, `val2017.zip`, `annotations_trainval2017.zip` (2017 Train/Val annotations) |
| WIDER FACE | [Official project page](https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html), [CUHK image archives](https://huggingface.co/datasets/CUHK-CSE/wider_face/tree/main/data) | `WIDER_train.zip`, `WIDER_val.zip`, and the official evaluation tools archive `eval_tools.zip` containing the validation `.mat` ground truth |
| VOC2012 | [Official development kit and data](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/#devkit) | `VOCtrainval_11-May-2012.tar`, which includes images, segmentation masks and split lists |

Extract the COCO archives under one root so that it contains `train2017/`, `val2017/` and `annotations/instances_val2017.json` directly. Use the instance annotations for the detection and instance-segmentation models in `cnn30`.

For WIDER FACE, keep `WIDER_train/images/` and `WIDER_val/images/` under the dataset root. Place the evaluation tools at `<root>/eval_tools/`; its `ground_truth/` directory must contain `wider_face_val.mat`, `wider_easy_val.mat`, `wider_medium_val.mat` and `wider_hard_val.mat`. The text annotations in `wider_face_split.zip` do not replace these `.mat` files.

Extract VOC2012 so that `VOCdevkit/VOC2012/` contains `JPEGImages/`, `SegmentationClass/` and `ImageSets/Segmentation/val.txt`. Pass the `VOC2012` directory itself as `--root`.

### Prepare local datasets

After downloading and extracting the files, run only the commands for the datasets you need and replace the paths with your own. These commands check existing files and write manifests; they do not download datasets.

```bash
opennpu-quant data prepare imagenet --root /data/imagenet --source folder
opennpu-quant data prepare coco --root /data/coco
opennpu-quant data prepare widerface --root /data/widerface
opennpu-quant data prepare voc --root /data/VOCdevkit/VOC2012
```

cnn30 uses ImageNet, COCO, WIDER FACE, VOC2012 and LFW. COCO keypoints is needed only for the additional PoseNet model, prepared with `opennpu-quant data prepare coco_keypoints --root /data/coco`.

Use the same `--home` or `--data-dir` for preparation and execution. Preparation always indexes the full validation set, even if a later run evaluates only a few samples with `--limit`. `--limit` uses a prefix of the manifest, except for LFW ([LFW](#lfw)).

| Dataset | Download through this CLI | Required layout | Full evaluation | Calibration |
|---|---|---|---:|---|
| ImageNet folder | No | `train/<wnid>/`, `val/<wnid>/` or `validation/` | 50,000 images | 1,000 training images selected with seed 20260917 |
| ImageNet HF | `--source hf --download` | `train_images/`, `validation_images/` and label information. Use `--source hf` | 50,000 images | Packaged fixed list of 1,000 images |
| COCO | `--download` | `train2017/`, `val2017/`, `annotations/instances_val2017.json` | 5,000 images | Packaged fixed list of 128 training images |
| COCO keypoints | `--download` | The COCO images and `annotations/person_keypoints_val2017.json` | 5,000 images | 128 images from the PoseNet list `coco_keypoints.txt` |
| WIDER FACE | No | `WIDER_train/images/`, `WIDER_val/images/` and the official `.mat` ground truth | 3,226 images | Packaged fixed list of 128 training images |
| VOC2012 | `--download` | `JPEGImages/`, `SegmentationClass/`, `ImageSets/Segmentation/val.txt` | 1,449 images | Packaged fixed list of 128 training images |
| LFW | `--download` | `pairs.txt`, `data/lfw_funneled/`, `lfw_funneled.json` landmarks | 6,000 pairs (15,402 views) | Packaged 128 images that do not appear in any evaluation pair |

The two ImageNet layouts select different calibration images. Matching statistics from another run requires the same list, order and preprocessing, not just the same dataset.

### Download ImageNet or LFW

For ImageNet HF, first obtain access on the [ImageNet dataset page](https://huggingface.co/datasets/ILSVRC/imagenet-1k). From the repository root, install the data dependencies, activate the environment and authenticate with an access token using [huggingface-cli login](https://huggingface.co/docs/huggingface_hub/v0.29.3/en/guides/cli#huggingface-cli-login). Use `--device cpu` for a CPU environment.

```bash
./setup.sh --device cuda --data
source .venv/bin/activate
huggingface-cli login
opennpu-quant data prepare imagenet --root /data/imagenet-hf --source hf --download
```

This command downloads the inputs and prepares their manifest. The downloader saves the fixed 1,000 calibration images from the training split and all 50,000 validation images. An existing ImageNet HF download can be prepared again with the same command without `--download`.

For LFW, the base setup already includes the preparation dependencies. The following command downloads the source files, aligns the faces and prepares the manifest:

```bash
opennpu-quant data prepare lfw_mobilefacenet --root /data/lfw --download
```

### LFW

The CPU and CUDA `setup.sh` profiles include `scikit-image==0.25.2` for alignment. It is also part of the package's `images` extra. Preparation checks this dependency before downloading or reading the full dataset.

Preparation writes aligned 112×112 face crops into the data directory. `--download` fetches the pair list, images and landmarks from fixed URLs and checks their SHA256 hashes. If any image fails to align, preparation records the failures and writes no manifest.

A full evaluation scores all 6,000 pairs from 15,402 original and horizontally flipped image views. It reports `verification_accuracy`, with thresholds selected over ten folds, and `fixed_angle_accuracy` ([MobileFaceNet on LFW](model-reference.md#notes-on-specific-models)). LFW accuracy always needs all pairs.

### Full LFW evaluation only

Use the full manifest with `run --limit 6000` or `evaluation.limit: null`, or use `reproduce prepare --limit full`. A numeric limit of 6,000 or more includes all pairs. A smaller effective limit is rejected before any model calibrates or infers, including mixed-model requests. In `run`, omitting `--limit` uses the experiment's `evaluation.limit`.

The complete pair order, ten folds and both image views are preserved. Calibration still uses the separate 128-image split. For small checks, select `--models resnet18,yolov5s`.

LFW smoke manifests and partial results are not accepted for execution, reporting or comparison in this version. Keep existing artifacts unchanged and prepare a new full evaluation using the commands above. The removed `--functional-check` and `data prepare --smoke` options are not supported.

## Export FP32 ONNX

```bash
opennpu-quant export resnet18
```

Export verifies the recipe's pinned checkpoint and source, then creates the ONNX bundle. Both `strict` (default) and `functional` report finite framework-to-ONNX differences in the console and `model.json` without rejecting export. Shape, dtype, non-finite output, graph conversion, partition and Release contract errors still stop export.

Export checks use CPU fixtures and do not establish GPU execution or full-dataset accuracy. For GPU evaluation, run the exported network with `--device cuda:0`. Host graphs remain on CPU.

Running `export` again converts the model again. To evaluate a bundle you already have, skip export and follow [Evaluate an existing ONNX model](run-pipeline.md#evaluate-an-existing-onnx-model). Use `--weights FILE` to reuse an identical source checkpoint, or `--source-dir DIR` for a clean checkout at the pinned commit.

### Extra export environments

You need these environments only to export models.

| Models | Environment |
|---|---|
| torchvision classifiers and other base Torch models | Included in the `setup.sh` environment |
| timm models, additional YOLO models, NanoDet and others | [cnn38-linux-py311.txt](../requirements/cnn38-linux-py311.txt) |
| The three MMSeg models | A separate interpreter. Follow the Torch 2.1.0 and compiled MMCV 2.1.0 installation order in [cnn38-mmseg.txt](../requirements/cnn38-mmseg.txt) |
| TF SSD and DeepLab | The TensorFlow/tf2onnx packages in the setup environment, or a separate interpreter selected with `--python` if they conflict |

The setup environment does not require or install pip. From the repository root, select the same uv executable that setup uses and specify the destination interpreter explicitly. If uv was downloaded by setup, it is at `.tools/uv` and may not be on `PATH`.

```bash
if command -v uv >/dev/null 2>&1; then
  UV_BIN=uv
else
  UV_BIN="$PWD/.tools/uv"
fi
"$UV_BIN" pip install --python "$PWD/.venv/bin/python" \
  -r requirements/cnn38-linux-py311.txt
```

For MMSeg, create a separate environment and use the ordered install commands in [cnn38-mmseg.txt](../requirements/cnn38-mmseg.txt). Every install command must target that environment, not the main `.venv`.

```bash
"$UV_BIN" venv --python 3.11 workspace/mmseg-venv
"$UV_BIN" pip install --python workspace/mmseg-venv/bin/python \
  torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
"$UV_BIN" pip install --python workspace/mmseg-venv/bin/python \
  mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html
"$UV_BIN" pip install --python workspace/mmseg-venv/bin/python \
  -r requirements/cnn38-mmseg.txt
opennpu-quant export deeplabv3_r50_voc --python "$PWD/workspace/mmseg-venv/bin/python"
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

Graphs with symbolic spatial dimensions are checked at three resolutions during export. Models with variable input sizes keep each sample's shape in their input caches.

## Add a custom model

1. Place the ONNX file and any host graphs in `<models-dir>/<name>/`.
2. Copy a similar packaged recipe to `recipe.yaml` and edit the name, input, preprocessing, decoder and graph mappings. When the copy gets a different model name, remove its `decode.profile` so that the evaluation behavior is selected from `decode.kind`.
3. Copy an experiment file, add the model and its percentile, and give the experiment a new name ([Configuration reference](configuration.md#example-change-a-percentile)).
4. Check the settings with `opennpu-quant run -c my_experiment.yaml --models my_model --dry-run`, then verify inputs, outputs and evaluation on a small run.

If no built-in decoder fits the model's outputs, write a custom evaluator with the [Python API](api.md). Custom `module:function` preprocessing turns off input caching and result reuse, because its identity cannot be determined automatically. Unsupported operators and other graph limits are listed in [Common limitations for custom models](quantization.md#10-common-limitations-for-custom-models).
