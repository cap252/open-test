# Model reference

This page lists the input, preprocessing, decoder and evaluation profile of every packaged model, followed by the settings of the original measurements.

## Model groups

| Group | Models | Where it is used |
|---|---|---|
| core10 | The 10 models marked **core10** below, also included in cnn30 | Default preset of `run` |
| cnn30 | core10 plus the 20 models marked **cnn30 additional**, 30 in total | `cnn30` preset and default model set of `reproduce` |
| additional support | EfficientNet-Lite0, YOLOv9-T, PoseNet, DeepLabV3+ ResNet50, YOLOv3-full, YOLOv4-full, YOLOv7-full and YOLOv9-S | Explicit `--models` selection for `reproduce`, `reproduce prepare --model-set all`, or a custom experiment file for `run` |

The eight additional models keep their recipes, execution support and original records. They do not have a separate packaged experiment preset. Use `reproduce configs --models <names>` to create experiment files with their original settings.

The cnn30 task counts are 12 classification, 10 object detection, 2 face detection, 4 semantic segmentation, 1 instance segmentation and 1 face verification models. `yolox` uses YOLOX-S (S-640). RetinaFace MobileNet0.25 at 640×640 is a face detector, while MobileFaceNet ReLU (foamliu) performs face verification.

## How to read the tables

The shapes come from the FP32 ONNX files referenced by the packaged selection ([`selected.json`](../src/opennpu_quant/resources/reproduction/selected.json)), read after verifying their SHA256 hashes. [`model-inputs.json`](model-inputs.json) records the graph IDs and hashes. For a model you export yourself, check its `model.json` and ONNX inputs.

- **Recipe input** is the output of Python preprocessing. **Network input** is the actual feed after any CPU pre graph and input mapping. All network feeds are float32, and shapes include batch 1.
- `f32` is float32, `u8` is uint8, and H/W are symbolic spatial dimensions. When a recipe does not declare a dtype, the external dtype comes from the preprocessing implementation.
- In the host column, `pre` and `post` are CPU ONNX graphs. Python decoders and evaluators run in addition to them.
- The last column lists the CPU host graphs, decoder kind and default evaluation profile. [Evaluation profiles](evaluation-profiles.md#default-profile-per-model) explains the behavior selected by each profile.
- ImageNet mean/std is `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`. MMSeg uses `[123.675,116.28,103.53]` / `[58.395,57.12,57.375]`. NanoDet uses BGR `[103.53,116.28,123.675]` / `[57.375,57.12,58.395]`, divided by 255 in the original order.

## Classification

Dataset ImageNet, metrics top1 and top5.

| Model ID | Group | Recipe input shape / dtype / layout | Actual network shape / dtype / layout | Spatial dimensions / feed | Preprocessing | CPU host / decoder / profile |
|---|---|---|---|---|---|---|
| `resnet18` | core10 | 1×3×224×224 / f32 / NCHW | 1×3×224×224 / f32 / NCHW | Fixed | short side 256 → crop224, bilinear, /255 + ImageNet mean/std | none / `logits` / `numpy_topk_v1` |
| `resnet50` | core10 | 1×3×224×224 / f32 / NCHW | 1×3×224×224 / f32 / NCHW | Fixed | short side 256 → crop224, bilinear, /255 + ImageNet mean/std | none / `logits` / `numpy_topk_v1` |
| `resnet101` | cnn30 additional | 1×3×224×224 / f32 / NCHW | 1×3×224×224 / f32 / NCHW | Fixed | short side 256 → crop224, bilinear, /255 + ImageNet mean/std | none / `logits` / `numpy_topk_v1` |
| `mobilenet_v1` | cnn30 additional | 1×3×256×256 / f32 / NCHW | 1×3×256×256 / f32 / NCHW | Fixed | short side 269 → crop256, bicubic, /255 + mean/std=0.5 | none / `logits` / `numpy_topk_v1` |
| `mobilenet_v2` | core10 | 1×3×224×224 / f32 / NCHW | 1×3×224×224 / f32 / NCHW | Fixed | short side 256 → crop224, bilinear, /255 + ImageNet mean/std | none / `logits` / `numpy_topk_v1` |
| `mobilenet_v3_large` | core10 | 1×3×224×224 / f32 / NCHW | 1×3×224×224 / f32 / NCHW | Fixed | short side 256 → crop224, bilinear, /255 + ImageNet mean/std | none / `logits` / `numpy_topk_v1` |
| `efficientnet_b0` | core10 | 1×3×224×224 / f32 / NCHW | 1×3×224×224 / f32 / NCHW | Fixed | short side 256 → crop224, bicubic, /255 + ImageNet mean/std | none / `logits` / `numpy_topk_v1` |
| `efficientnet_lite4` | cnn30 additional | 1×3×380×380 / f32 / NCHW | 1×3×380×380 / f32 / NCHW | Fixed | short side 413 → crop380, bilinear, /255 + mean/std=0.5 | none / `logits` / `numpy_topk_v1` |
| `squeezenet1_1` | cnn30 additional | 1×3×224×224 / f32 / NCHW | 1×3×224×224 / f32 / NCHW | Fixed | short side 256 → crop224, bilinear, /255 + ImageNet mean/std | none / `logits` / `numpy_topk_v1` |
| `shufflenet_v2_x1_0` | cnn30 additional | 1×3×224×224 / f32 / NCHW | 1×3×224×224 / f32 / NCHW | Fixed | short side 256 → crop224, bilinear, /255 + ImageNet mean/std | none / `logits` / `numpy_topk_v1` |
| `vgg16` | cnn30 additional | 1×3×224×224 / f32 / NCHW | 1×3×224×224 / f32 / NCHW | Fixed | short side 256 → crop224, bilinear, /255 + ImageNet mean/std | none / `logits` / `numpy_topk_v1` |
| `inception_v3` | core10 | 1×3×299×299 / f32 / NCHW | 1×3×299×299 / f32 / NCHW | Fixed | short side 342 → crop299, bilinear, /255 + ImageNet mean/std | pre / `logits` / `numpy_topk_v1` |
| `efficientnet_lite0` | additional support | 1×3×224×224 / f32 / NCHW | 1×3×224×224 / f32 / NCHW | Fixed | short side 256 → crop224, bicubic, /255 + mean/std=0.5 | none / `logits` / `numpy_topk_v1` |

## Object detection

Dataset COCO, metrics AP, AP50 and AP75.

| Model ID | Group | Recipe input shape / dtype / layout | Actual network shape / dtype / layout | Spatial dimensions / feed | Preprocessing | CPU host / decoder / profile |
|---|---|---|---|---|---|---|
| `yolov3_tiny` | cnn30 additional | 1×3×416×416 / f32 / NCHW | 1×3×416×416 / f32 / NCHW | Fixed | RGB letterbox, pad114, /255 | post / `yolo` / `yolo_opencv_nms_v1` |
| `yolov4_tiny` | cnn30 additional | 1×3×416×416 / f32 / NCHW | 1×3×416×416 / f32 / NCHW | Fixed | RGB 416² stretch resize, /255 | post / `yolov4` / `yolov4_opencv_nms_v1` |
| `yolov5n` | cnn30 additional | 1×3×640×640 / f32 / NCHW | 1×3×640×640 / f32 / NCHW | Fixed | RGB letterbox, pad114, /255 | post / `yolo` / `yolo_opencv_nms_v1` |
| `yolov5s` | core10 | 1×3×640×640 / f32 / NCHW | 1×3×640×640 / f32 / NCHW | Fixed | RGB letterbox, pad114, /255 | post / `yolo` / `yolo_opencv_nms_v1` |
| `yolov7_tiny` | cnn30 additional | 1×3×640×640 / f32 / NCHW | 1×3×640×640 / f32 / NCHW | Fixed | RGB letterbox, pad114, /255 | post / `yolo` / `yolo_opencv_nms_v1` |
| `yolov8n` | cnn30 additional | 1×3×640×640 / f32 / NCHW | 1×3×640×640 / f32 / NCHW | Fixed | RGB letterbox, pad114, /255 | post / `yolo_no_objectness` / `yolo_scores_opencv_nms_v1` |
| `ssd_mobilenet_v2_320` | core10 | 1×320×320×3 / u8 / NHWC | 1×3×300×300 / f32 / NCHW | Fixed | RGB resize, uint8 → CPU pre graph | pre/post / `tf_od_api` / `tf_od_api_v1` |
| `nanodet_plus_m_320` | cnn30 additional | 1×3×320×320 / f32 / NCHW | 1×3×320×320 / f32 / NCHW | Fixed | BGR warpPerspective, /255, then normalize with mean/255 and std/255 | post / `nanodet` / `nanodet_v1` |
| `retinanet_resnet50_fpn` | cnn30 additional | 1×3×800×800 / f32 / NCHW | 1×3×800×800 / f32 / NCHW | Fixed | RGB 800² stretch resize, /255 | pre/post / `retinanet` / `retinanet_host_nms_v1` |
| `yolov9_t` | additional support | 1×3×H×W / f32 / NCHW | 1×3×H×W / f32 / NCHW | Variable ONNX / variable feed | RGB stride32 letterbox, /255, rectangular | post / `yolo_no_objectness` / `yolov9_torch_nms_v1` |
| `yolox` | cnn30 additional | 1×3×640×640 / f32 / NCHW | 1×3×H×W / f32 / NCHW | Variable ONNX / fixed 640 feed | BGR keep-ratio, top-left placement, pad114, 0–255 float32 | post / `yolo` / `yolox_torch_nms_v1` |
| `yolov3_full` | additional support | 1×3×416×416 / f32 / NCHW | 1×3×416×416 / f32 / NCHW | Fixed | RGB letterbox, pad114, /255 | post / `yolo` / `yolo_opencv_nms_v1` |
| `yolov4_full` | additional support | 1×3×416×416 / f32 / NCHW | 1×3×416×416 / f32 / NCHW | Fixed | RGB 416² stretch resize, /255 | post / `yolov4` / `yolov4_opencv_nms_v1` |
| `yolov7_full` | additional support | 1×3×640×640 / f32 / NCHW | 1×3×640×640 / f32 / NCHW | Fixed | RGB letterbox, pad114, /255 | post / `yolo` / `yolo_opencv_nms_v1` |
| `yolov9_s` | additional support | 1×3×H×W / f32 / NCHW | 1×3×H×W / f32 / NCHW | Variable ONNX / variable feed | RGB stride32 letterbox, /255, rectangular | post / `yolo_no_objectness` / `yolov9_torch_nms_v1` |

## Face detection

Dataset WIDER FACE, primary metric AP_medium.

| Model ID | Group | Recipe input shape / dtype / layout | Actual network shape / dtype / layout | Spatial dimensions / feed | Preprocessing | CPU host / decoder / profile |
|---|---|---|---|---|---|---|
| `retinaface_mnet025_640` | core10 | 1×3×640×640 / f32 / NCHW | 1×3×640×640 / f32 / NCHW | Fixed | BGR resize, subtract mean[104,117,123] | post / `retinaface` / `retinaface_v1` |
| `yolov7_face` | cnn30 additional | 1×3×H×W / f32 / NCHW | 1×3×H×W / f32 / NCHW | Variable ONNX / variable feed | RGB stride32 letterbox, /255, auto padding | post / `yolo_face` / `yolov7_face_clip_first4_v1` |

## Pose estimation

Dataset COCO keypoints, metric keypoint AP.

| Model ID | Group | Recipe input shape / dtype / layout | Actual network shape / dtype / layout | Spatial dimensions / feed | Preprocessing | CPU host / decoder / profile |
|---|---|---|---|---|---|---|
| `posenet_101` | additional support | 1×3×513×513 / f32 / NCHW | 1×3×513×513 / f32 / NCHW | Fixed | 513² resize, RGB ×(2/255)−1 | post / `posenet` / `posenet_multi_pose_v1` |

## Semantic segmentation

Dataset VOC2012, metric mIoU.

| Model ID | Group | Recipe input shape / dtype / layout | Actual network shape / dtype / layout | Spatial dimensions / feed | Preprocessing | CPU host / decoder / profile |
|---|---|---|---|---|---|---|
| `deeplabv3_mnv2_voc` | core10 | 1×513×513×3 / f32 / NHWC | 1×513×513×3 / f32 / NHWC | Fixed | long side 513, (RGB−127.5)/127.5, bottom/right 0 pad | post / `segmentation_labels` / `segmentation_labels_v1` |
| `unet_resnet50_voc` | cnn30 additional | 1×3×512×512 / f32 / NCHW | 1×3×512×512 / f32 / NCHW | Fixed | BICUBIC letterbox512, pad128, /255 | none / `segmentation_resize` / `segmentation_resize_v1` |
| `deeplabv3_r50_voc` | cnn30 additional | 1×3×H×W / f32 / NCHW | 1×3×H×W / f32 / NCHW | Variable ONNX / variable feed | RGB keep-ratio ≤2048×512, mean/std(0–255) | none / `segmentation_resize` / `segmentation_resize_v1` |
| `fcn_r50_voc` | cnn30 additional | 1×3×H×W / f32 / NCHW | 1×3×H×W / f32 / NCHW | Variable ONNX / variable feed | RGB keep-ratio ≤2048×512, mean/std(0–255) | none / `segmentation_resize` / `segmentation_resize_v1` |
| `deeplabv3plus_r50_voc` | additional support | 1×3×H×W / f32 / NCHW | 1×3×H×W / f32 / NCHW | Variable ONNX / variable feed | RGB keep-ratio ≤2048×512, mean/std(0–255) | none / `segmentation_resize` / `segmentation_resize_v1` |

## Instance segmentation

Dataset COCO, metric mask_AP. Box AP and mask AP are separate metrics.

| Model ID | Group | Recipe input shape / dtype / layout | Actual network shape / dtype / layout | Spatial dimensions / feed | Preprocessing | CPU host / decoder / profile |
|---|---|---|---|---|---|---|
| `yolov5s_seg` | cnn30 additional | 1×3×H×W / f32 / NCHW | 1×3×H×W / f32 / NCHW | Variable ONNX / variable feed | RGB stride32 letterbox, /255, rectangular | post / `yolo_seg` / `yolov5_seg_scaled_coeff_bbox_area_v1` |

## Face verification

Dataset LFW, metric verification_accuracy.

| Model ID | Group | Recipe input shape / dtype / layout | Actual network shape / dtype / layout | Spatial dimensions / feed | Preprocessing | CPU host / decoder / profile |
|---|---|---|---|---|---|---|
| `mobilefacenet_relu_foamliu` | cnn30 additional | 1×3×112×112 / f32 / NCHW | 1×3×112×112 / f32 / NCHW | Fixed | aligned 112², RGB /255 → ImageNet mean/std | none / `face_verification` / `lfw_flip_ten_fold_v1` |

## Decoder and evaluation settings

| Model/family | Confidence / NMS IoU / count | Restoration, scoring and where it is set |
|---|---|---|
| YOLOv3/Tiny, YOLOv5n/s, YOLOv7/Tiny, YOLOv8n | 0.001 / 0.6 / 300, 30000 candidates | Recipe `decode`. Contiguous COCO 80-class indices map to annotation category IDs. v8 has no objectness |
| YOLOv9-T/S | 0.001 / 0.7 / 300, 30000 candidates | Recipe `decode` + `yolov9_torch_nms_v1`. Preserve original bbox/score serialization |
| YOLOX | 0.001 / 0.65 / no global cap with the default profile | `yolox_torch_nms_v1` follows original behavior even with max300 in the recipe. Cap/clipping changes define a separate convention |
| YOLOv7-face | 0.01 / 0.5 / no final cap with the default profile, 30000 candidates | Recipe `decode` + `yolov7_face_clip_first4_v1`. Original NMS/clipping without the recipe's max300, landmark separation and WIDER xywh restoration |
| YOLOv5s-seg | 0.001 / 0.6 / 300, 30000 candidates | Recipe `decode` + `yolov5_seg_scaled_coeff_bbox_area_v1`. Mask coefficient/prototype reconstruction and bbox-area COCO mask AP |
| YOLOv4/Tiny | 0.001 / 0.6 / 300 | Fixed in `extended_decode.py`. Restore normalized boxes from 416 coordinates using FP64 coordinate arithmetic |
| NanoDet | 0.05 / 0.6 / 100 | Fixed in `extended_decode.py`. DFL with 8 bins and strides 8/16/32/64, restored from 320 coordinates |
| RetinaFace | 0.02 / 0.4 / 750 | Recipe priors/min_sizes/steps/variance and thresholds. Medium is the primary WIDER easy/medium/hard metric |
| SSD / RetinaNet | TopK/NMS inside the host graph | Python validates/restores coordinates and valid COCO categories. It does not add arbitrary NMS or fill missing slots |
| PoseNet | score 0.25 / up to 10 poses | Vendor decoder, stride 16, restore original coordinates. COCO keypoint AP |
| U-Net | softmax → padding crop → bilinear → argmax | Fixed in the Python decoder. VOC mIoU with the ignore label preserved |
| 3 MMSeg models | bilinear to input size, then to original size → argmax | `align_corners=False`, fixed in the Python decoder. VOC mIoU |
| TF DeepLab | host labels → crop valid area → restore labels | Do not replace label-map resizing with logit resizing |
| MobileFaceNet ReLU | sum original+flip embeddings → FP32 L2 normalization | Ten-fold threshold selection for `verification_accuracy`, plus `fixed_angle_accuracy` at 74.18°. Requires all 6,000 pairs ([LFW](models-and-data.md#lfw)) |

Values described as fixed cannot be overridden from YAML or the CLI. The NMS details and profile exceptions are part of the decoder code as well as the recipe numbers. See [Compare against another profile](evaluation-profiles.md#compare-against-another-profile) for how profile choices affect evaluation comparisons.

## Notes on specific models

- **SSD:** the recipe input differs from the network input, as the detection table shows. The TF host graphs handle resizing, normalization and detection output restoration, so keep their connections intact.
- **InceptionV3 and RetinaNet:** the bundles include CPU pre graphs. Equal external and network shapes do not mean the graph does nothing, so keep the input mapping.
- **Variable inputs:** YOLOv9-T/S and YOLOv5s-seg round to multiples of the stride with rectangular padding, so the longer side can exceed 640. YOLOv7-face uses its own automatic padding, and MMSeg input sizes depend on the aspect ratio. Eight networks have symbolic dimensions, including YOLOX, and seven of them receive variable-size feeds.
- **Preprocessing arithmetic:** NanoDet divides before normalizing, PoseNet multiplies before subtracting and YOLOv4 promotes coordinates to FP64. These orders preserve the original input bytes and predictions, so keep them even where an algebraically equal form exists.
- **MobileFaceNet on LFW:** besides the primary `verification_accuracy`, the full LFW evaluation reports `fixed_angle_accuracy`, which predicts the same identity when the angle between the two embeddings is at most 74.18°. The threshold is recorded as `metric_details.fixed_angle_threshold_degrees`. The calculation normalizes rows in FP32 and compares the angles in FP64, matching the original evaluation order.
- **Published accuracy:** FP32 accuracy is measured with each model's packaged preprocessing and evaluation, so it can differ from the number published for a checkpoint. Published numbers are not acceptance targets, and recovery always compares INT8 with this FP32 result. Most models come close to their published numbers. Known differences in the evaluation conditions can contribute to the larger gaps: RetinaNet and RetinaFace resize to a fixed square instead of the source evaluation geometry, TF DeepLab-MobileNetV2 uses PIL resizing and its own label restoration, and MobileFaceNet pins a state dict and uses its own alignment and threshold selection. The effect of each difference on accuracy has not been measured separately. The published YOLOv7-Tiny number belongs to a different variant (SiLU, while the checkpoint used here is LeakyReLU), and no matching published number was found for YOLOv3-Tiny and PoseNet.

## Original per-model settings

These are the settings of the original measurements, which `reproduce` applies and `reproduce configs` writes as experiment files ([Use the original per-model settings](configuration.md#use-the-original-per-model-settings)). The presets (`core10` and `cnn30`) use the same calibration counts and AdaRound hyperparameters. They differ in the percentiles (99.99 for every model in `cnn30`, a separate per-model list in `core10`), the 6 GiB GPU memory limit, the 512 MiB AdaRound window and AdaRound device `auto`, and the original settings also pin the calibration input hash. `run` uses the graph partitions of your exported bundle, while `reproduce` uses the packaged graphs.

The P column lists the percentiles for **Asym/PC float / Asym/PC PoT / Sym/PT PoT**, and RTN and AdaRound share each family's percentile. Calibration counts are inputs, so LFW uses 128 images rather than pairs. The window column is `adaround.window_bytes`, which is separate from the ONNX Runtime GPU limit and the activation cache.

All selected AdaRound conditions use 10,000 steps, learning rate 0.001, warm-up 0.2, regularization 0.01, beta 20→2, seed 20260928, batch 8, 32 window samples, 100 window steps, device `cuda`, atol 1e-5 and rtol 1e-4.

| Model ID | Group | Calibration inputs | P(float / asym PoT / sym PoT) | ORT GPU limit GiB | AdaRound window MiB |
|---|---|---:|---|---|---:|
| `resnet18` | core10 | 1000 | 99.999 / 99.99 / 99.99 | 6 (all conditions) | 512 |
| `resnet50` | core10 | 1000 | 99.999 / 99.99 / 99.99 | 6 (all conditions) | 512 |
| `resnet101` | cnn30 additional | 1000 | 99.99 / 99.99 / 99.99 | 6 (all conditions) | 512 |
| `mobilenet_v1` | cnn30 additional | 1000 | 99.9 / 99.9 / 99.999 | 6 (all conditions) | 512 |
| `mobilenet_v2` | core10 | 1000 | 99.99 / 99.9 / 99.9 | 6 (all conditions) | 512 |
| `mobilenet_v3_large` | core10 | 1000 | 99.99 / 99.9 / 99.9 | 6 (all conditions) | 512 |
| `efficientnet_b0` | core10 | 1000 | 99.9 / 99.9 / 99.9 | 6 (all conditions) | 512 |
| `efficientnet_lite4` | cnn30 additional | 1000 | 99.99 / 99.99 / 99.99 | 6 (all conditions) | 512 |
| `squeezenet1_1` | cnn30 additional | 1000 | 99.999 / 99.99 / 99.99 | 6 (all conditions) | 512 |
| `shufflenet_v2_x1_0` | cnn30 additional | 1000 | 99.999 / 99.99 / 99.99 | 6 (all conditions) | 512 |
| `vgg16` | cnn30 additional | 1000 | 99.999 / 99.999 / 99.999 | 8 for float and Sym/PT PoT AdaRound, 6 for the others | 512 |
| `inception_v3` | core10 | 1000 | 99.999 / 99.99 / 99.99 | 6 (all conditions) | 512 |
| `efficientnet_lite0` | additional support | 1000 | 99.99 / 99.99 / 99.99 | 6 for FP32, 8 for all INT8 | 512 |
| `yolov3_tiny` | cnn30 additional | 128 | 99.99 / 99.99 / 99.99 | 6 (all conditions) | 512 |
| `yolov4_tiny` | cnn30 additional | 128 | 99.99 / 99.9 / 99.99 | 6 (all conditions) | 512 |
| `yolov5n` | cnn30 additional | 128 | 99.99 / 99.99 / 99.9 | 6 (all conditions) | 512 |
| `yolov5s` | core10 | 128 | 99.999 / 99.99 / 99.99 | 6 (all conditions) | 512 |
| `yolov7_tiny` | cnn30 additional | 128 | 99.99 / 99.99 / 99.99 | 6 (all conditions) | 512 |
| `yolov8n` | cnn30 additional | 128 | 99.999 / 99.99 / 99.9 | 6 (all conditions) | 512 |
| `ssd_mobilenet_v2_320` | core10 | 128 | 99.999 / 99.99 / 99.99 | 6 (all conditions) | 512 |
| `nanodet_plus_m_320` | cnn30 additional | 128 | 99.99 / 99.99 / 99.9 | 6 (all conditions) | 512 |
| `retinanet_resnet50_fpn` | cnn30 additional | 128 | 99.999 / 99.999 / 99.99 | 6 (all conditions) | 512 |
| `yolov9_t` | additional support | 128 | 99.99 / 99.9 / 99.9 | 8 (all conditions) | 4096 |
| `yolox` | cnn30 additional | 128 | 99.99 / 99.99 / 99.99 | 8 (all conditions) | 4096 |
| `retinaface_mnet025_640` | core10 | 128 | 99.9 / 99.9 / 99.9 | 6 (all conditions) | 512 |
| `yolov7_face` | cnn30 additional | 128 | 99.99 / 99.99 / 99.99 | 8 (all conditions) | 4096 |
| `posenet_101` | additional support | 128 | 99.9 / 99.9 / 99.999 | 6 (all conditions) | 512 |
| `deeplabv3_mnv2_voc` | core10 | 128 | 99.9 / 99.99 / 99.99 | 6 (all conditions) | 512 |
| `unet_resnet50_voc` | cnn30 additional | 128 | 99.99 / 99.99 / 99.99 | 8 (all conditions) | 4096 |
| `deeplabv3_r50_voc` | cnn30 additional | 128 | 99.99 / 99.99 / 99.99 | 8 for AdaRound, 6 for FP32 and RTN | 4096 |
| `fcn_r50_voc` | cnn30 additional | 128 | 99.99 / 99.99 / 99.99 | 8 (all conditions) | 4096 |
| `yolov5s_seg` | cnn30 additional | 128 | 99.99 / 99.99 / 99.99 | 8 (all conditions) | 4096 |
| `yolov3_full` | additional support | 128 | 99.99 / 99.99 / 99.99 | 8 (all conditions) | 512 |
| `yolov4_full` | additional support | 128 | 99.99 / 99.9 / 99.99 | 8 (all conditions) | 512 |
| `yolov7_full` | additional support | 128 | 99.99 / 99.99 / 99.99 | 8 (all conditions) | 512 |
| `yolov9_s` | additional support | 128 | 99.99 / 99.99 / 99.99 | 8 (all conditions) | 512 |
| `mobilefacenet_relu_foamliu` | cnn30 additional | 128 | 99.99 / 99.9 / 99.9 | 6 (all conditions) | 512 |
| `deeplabv3plus_r50_voc` | additional support | 128 | 99.99 / 99.99 / 99.99 | 6 (all conditions) | 4096 |
