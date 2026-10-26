# Evaluation profiles

Models differ in how they break classification ties, run NMS, restore face coordinates and rebuild masks. An **evaluation profile** selects one version of this postprocessing and scoring behavior. The evaluator resolves it once when the model is loaded.

`run` and `reproduce` use the same profile for each model by default, so you do not need to choose one. `run` reads it from the packaged model recipe, and `reproduce` reads it from the recipe snapshot in the packaged selection (`selected.json`). The same profile still gives different numbers when weights, calibration, percentiles, training, inputs or devices differ.

## Default profile per model

| Models | Profile | Behavior it preserves |
|---|---|---|
| 13 classifiers | `numpy_topk_v1` | Tie ordering of `np.argsort(logits)[::-1]` |
| YOLOv9-T/S | `yolov9_torch_nms_v1` | Torch CPU NMS, FP32 coordinate operation order, and bbox/score serialization with 3 and 5 decimal places |
| YOLOX | `yolox_torch_nms_v1` | Original NMS and score handling, including out-of-image coordinates and more than 300 detections |
| YOLOv7-face | `yolov7_face_clip_first4_v1` | Original coordinate restoration, integer conversion, score serialization and clipping of only the first four boxes |
| YOLOv5s-seg | `yolov5_seg_scaled_coeff_bbox_area_v1` | Objectness-scaled mask coefficients, mask reconstruction order and bbox-area evaluation |
| Other models | Decoder-specific names, such as `retinaface_v1`, `posenet_multi_pose_v1` or `segmentation_resize_v1` | Each model's decoder and evaluator as implemented |

Use the names in this table when setting `evaluation.profiles.<model>` in an experiment file.

For classification, `stable_topk_v1` is also available. It uses stable tie ordering, which differs from the default `numpy_topk_v1`. The public `TopKClassification` API also sorts stably by default.

## What a profile changes

A profile changes only the postprocessing and scoring behavior. Preprocessing, ONNX files, host graphs and input mappings stay the same. Graph partitions and tensor names belong to the exported model bundle, and `reproduce` connects the original partitions through the packaged selection.

The default evaluation profiles for the YOLO models listed above use Torch/Torchvision for CPU postprocessing. `setup.sh` installs the required packages.

To replace a model's Python decoder entirely, use the Python API hook described in [Replace Python postprocessing](api.md#replace-python-postprocessing).

## Precedence and local recipes

For a packaged model, `run` picks the profile in this order:

1. `--evaluation-profile`, which applies to every selected model.
2. `evaluation.profiles.<model>` in the experiment file.
3. The default profile of the packaged recipe.

A local `<models-dir>/<model>/recipe.yaml` still controls the bundle settings, such as inputs, preprocessing, graphs and output names. The profile default comes from the package, and the tool warns when a local recipe names a different profile. It does not edit the file. Custom models that are not in the package use the profile in their own recipe.

<a id="changing-preserved-behavior"></a>

## Compare against another profile

Changing a profile can change predictions and evaluation scores even for the same model outputs. Use the same profile for FP32 and INT8 accuracy comparisons, and keep each model's default profile when reproducing the supplied reference results.

Use a new output directory and set the profile per model in the experiment file. This excerpt changes only face clipping and mask area:

```yaml
evaluation:
  limit: 32
  profiles:
    yolov7_face: yolov7_face_clip_all_v1
    yolov5s_seg: yolov5_seg_scaled_coeff_mask_area_v1
```

YOLOv7-face clips only the first four face boxes with its default profile. `yolov7_face_clip_all_v1` applies clipping to every face box. `yolov5_seg_scaled_coeff_mask_area_v1` keeps the objectness-scaled mask coefficients in YOLOv5s-seg and changes only the COCO area policy. These alternatives apply to their respective models. Set a supported profile for each model. The tool validates the model/profile combination before execution. `reproduce run` has no profile override.

Profiles are part of the stored results and their identities. A profile change does not affect calibration inputs or statistics, so statistics caches remain valid. See [Reuse and caching](run-pipeline.md#reuse-and-caching) for what triggers recomputation.
