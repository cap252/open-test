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

Results record these names. Recipes and `selected.json` store the shorter IDs `numpy_argsort_v1` (classifiers), `historical_cnn33_v1` (the four YOLO rows) and `current_v1` (the others), which resolve to the names above. Experiment files accept either form. A recipe without a profile uses `current_v1`.

An explicit `current_v1` always selects the generic decoder, not the model's default. For a classifier it resolves to `stable_topk_v1`, whose tie order differs from the default. The public `TopKClassification` API also sorts stably by default.

## What a profile changes

A profile changes only the postprocessing and scoring behavior. Preprocessing, ONNX files, host graphs and input mappings stay the same. Graph partitions and tensor names belong to the exported model bundle, and `reproduce` connects the original partitions through the packaged selection.

The original YOLO behavior uses Torch/Torchvision NMS on the CPU. The environments installed by `setup.sh` include these packages. A minimal wheel installation needs the `reproduction` extra.

To replace a model's Python decoder entirely, use the Python API hook described in [Replace Python postprocessing](api.md#replace-python-postprocessing).

## Precedence and local recipes

For a packaged model, `run` picks the profile in this order:

1. `--evaluation-profile`, which applies to every selected model.
2. `evaluation.profiles.<model>` in the experiment file.
3. The default profile of the packaged recipe.

A local `<models-dir>/<model>/recipe.yaml` still controls the bundle settings, such as inputs, preprocessing, graphs and output names. The profile default comes from the package, and the tool warns when a local recipe names a different profile. It does not edit the file. Custom models that are not in the package use the profile in their own recipe.

## Compare against another profile

Use a new output directory and set the profile per model in the experiment file. This excerpt changes only face clipping and mask area:

```yaml
evaluation:
  limit: 32
  profiles:
    yolov7_face: yolov7_face_clip_all_v1
    yolov5s_seg: yolov5_seg_scaled_coeff_mask_area_v1
```

`yolov7_face_clip_all_v1` clips every face box. `yolov5_seg_scaled_coeff_mask_area_v1` keeps the objectness-scaled mask coefficients and changes only the COCO area policy. The shorter names `face_all_boxes_v1` and `mask_area_v1` are also accepted. To compare a classifier or one of the YOLO models above with the generic behavior, select `current_v1` for that model. Unsupported model/profile combinations fail, so set profiles per model rather than for every task at once. `reproduce run` has no profile override.

Profiles are part of the stored results and their identities. FP32 and INT8 results scored with different profiles are not paired for recovery. A profile change does not affect calibration inputs or statistics, so statistics caches remain valid. See [Reuse and caching](run-pipeline.md#reuse-and-caching) for what triggers recomputation.

## Changing preserved behavior

To reproduce the original evaluation, the operation order of pre- and postprocessing must stay as it is, along with the model files. Treat any change below as a separate experiment with its own profile or output directory, and measure it on the full dataset.

| Change | Why it needs a separate evaluation |
|---|---|
| Classification tie ordering | Equal logits rank differently under another sorting rule, which changes Top-K |
| YOLOv9-T/S NMS, coordinate arithmetic or decimal serialization | Predictions near score, IoU or rounding boundaries change, and so does AP. Compare full COCO evaluations |
| YOLOX clipping or detection-count limits | Bounds and caps change the prediction set even when network outputs are identical |
| YOLOv7-face clipping of all boxes | The default clips only the first four boxes. This is a known upstream defect, kept so that results stay comparable with the original measurements. A comparison with `yolov7_face_clip_all_v1` on the same network outputs is complete for all seven conditions and all 3,226 WIDER FACE images, but its results are not promoted to the default profile or to the original reference values |
| YOLOv5s-seg mask arithmetic or area definition | The pinned YOLOv5 v7.0 source multiplies mask coefficients by objectness. Its COCO JSON includes boxes, so COCO loads bbox area, while mask IoU still uses masks. Changing the coefficient scaling or the area used for size filtering requires separate evaluations |
| Segmentation restoration | Crop, resize, softmax and argmax order, including `align_corners`, change pixel labels and mIoU |
| Image preparation | Recompression, orientation correction, normalization or reordering changes the input bytes and therefore the input identity |

The existing measurements do not validate such a change. A small AP difference after a change also does not show that the change is correct. See [Validation scope](validation.md) for what has been verified.
