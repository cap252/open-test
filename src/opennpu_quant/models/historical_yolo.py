"""Original evaluation arithmetic used by the affected models' default profiles.

Torchvision CPU NMS is intentional: OpenCV differs at ties and IoU boundaries.
Source bindings and independent comparison evidence are recorded with research data.
"""

import numpy as np
from .evaluation_profiles import decoder_family


def corners(center):
    import torch

    return torch.cat(
        (center[:, :2] - center[:, 2:] / 2, center[:, :2] + center[:, 2:] / 2), dim=1
    )


def candidates(prediction, model, cfg):
    import torch
    from torchvision.ops import nms, batched_nms

    model = decoder_family(model)
    expected = (
        84
        if model == "yolov9"
        else 21
        if model == "yolov7_face"
        else 117
        if model == "yolov5s_seg"
        else 85
    )
    x = torch.from_numpy(np.array(prediction, copy=True))
    if (
        x.ndim != 3
        or x.shape[0] != 1
        or x.dtype != torch.float32
        or not torch.isfinite(x).all()
    ):
        raise ValueError("Expected finite batch-one FP32 decoded predictions")
    x = x[0].T if model == "yolov9" else x[0]
    if x.shape[1] != expected:
        raise ValueError("Unexpected historical prediction channels")
    threshold, overlap = cfg["confidence_threshold"], cfg["nms"]["iou"]
    if model == "yolox":
        score, label = x[:, 5:85].max(1)
        combined = x[:, 4] * score
        selected = combined >= threshold
        boxes, combined, label = (
            corners(x[:, :4])[selected],
            combined[selected],
            label[selected],
        )
        keep = batched_nms(boxes, combined, label.float(), overlap)
        return torch.cat(
            (boxes[keep], combined[keep, None], label[keep, None].float()), 1
        )
    if model == "yolov9":
        x = x[x[:, 4:84].amax(1) > threshold]
        scores = x[:, 4:84]
        extra = x[:, 84:]
    else:
        x = x[x[:, 4] > threshold]
        if model == "yolov5s_seg":
            # The pinned source scales ALL columns after objectness, including mask coefficients.
            x[:, 5:] *= x[:, 4:5]
            scores, extra = x[:, 5:85], x[:, 85:]
        else:
            scores, extra = x[:, 5:6] * x[:, 4:5], x[:, 6:]
    boxes = corners(x[:, :4])
    rows, labels = torch.where(scores > threshold)
    selected = torch.cat(
        (boxes[rows], scores[rows, labels, None], labels[:, None].float(), extra[rows]),
        1,
    )
    if not len(selected):
        return selected
    if model != "yolov7_face" or len(selected) > cfg["max_nms_candidates"]:
        selected = selected[
            selected[:, 4].argsort(descending=True)[: cfg["max_nms_candidates"]]
        ]
    offset = 4096 if model == "yolov7_face" else 7680
    keep = nms(selected[:, :4] + selected[:, 5:6] * offset, selected[:, 4], overlap)
    if model != "yolov7_face":
        keep = keep[: cfg["max_detections"]]
    return selected[keep]


def restored_boxes(boxes, meta, model, profile):
    value = boxes.clone()
    width, height = meta["original_size"]
    if model == "yolox":
        return value / meta["scale"][0]
    pad, gain = meta["pad"], meta["scale"][0]
    if model == "yolov7_face":
        iw, ih = meta["input_size"]
        gain = min(ih / height, iw / width)
        pad = ((iw - width * gain) / 2, (ih - height * gain) / 2)
    value[:, [0, 2]] -= pad[0]
    value[:, [1, 3]] -= pad[1]
    if model == "yolov7_face":
        value[:, [0, 2]] /= gain
        value[:, [1, 3]] /= gain
    else:
        value /= gain
    # Reference behavior clips rows, not columns. Comparison profiles clip all rows.
    clipped = (
        value[:4]
        if model == "yolov7_face" and profile == "historical_cnn33_v1"
        else value
    )
    clipped[:, 0].clamp_(0, width)
    clipped[:, 1].clamp_(0, height)
    clipped[:, 2].clamp_(0, width)
    clipped[:, 3].clamp_(0, height)
    return value


def masks(prototypes, coefficients, boxes, meta):
    import cv2
    import torch
    import torch.nn.functional as F
    from pycocotools import mask as mask_api

    proto = torch.from_numpy(np.array(prototypes[0], copy=True))
    if proto.ndim != 3 or proto.shape[0] != 32 or not torch.isfinite(proto).all():
        raise ValueError("Expected 32 finite mask prototypes")
    iw, ih = meta["input_size"]
    probability = (
        (coefficients @ proto.float().reshape(32, -1))
        .sigmoid()
        .reshape(-1, *proto.shape[1:])
    )
    value = F.interpolate(
        probability[None], (ih, iw), mode="bilinear", align_corners=False
    )[0]
    xx = torch.arange(iw, dtype=boxes.dtype)[None, None, :]
    yy = torch.arange(ih, dtype=boxes.dtype)[None, :, None]
    value *= (
        (xx >= boxes[:, 0, None, None])
        & (xx < boxes[:, 2, None, None])
        & (yy >= boxes[:, 1, None, None])
        & (yy < boxes[:, 3, None, None])
    )
    value = value.gt_(0.5).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    px, py = meta["pad"]
    value = value[int(py) : int(ih - py), int(px) : int(iw - px)]
    value = cv2.resize(value, tuple(meta["original_size"]))
    if value.ndim == 2:
        value = value[:, :, None]
    result = []
    for i in range(value.shape[2]):
        rle = mask_api.encode(np.asfortranarray(value[:, :, i].astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        result.append(rle)
    return result


def postprocess(outputs, recipe, meta, categories, profile):
    model, cfg = decoder_family(recipe["name"]), recipe["decode"]
    prediction = outputs[cfg["output"]]
    selected = candidates(prediction, model, cfg)
    boxes = restored_boxes(selected[:, :4], meta, model, profile)
    if model == "yolov7_face":
        result = []
        for box, row in zip(boxes, selected):
            x, y, x2, y2 = [int(float(v) + 0.5) for v in box]
            result.append([x, y, x2 - x, y2 - y, float(f"{min(float(row[4]), 1):.3f}")])
        return result
    rles = (
        masks(outputs["prototypes"], selected[:, 6:], selected[:, :4], meta)
        if model == "yolov5s_seg" and len(selected)
        else []
    )
    result = []
    for i, (box, row) in enumerate(zip(boxes, selected)):
        label, score = int(row[5]), float(row[4])
        if not 0 <= label < len(categories):
            raise ValueError("Invalid COCO category")
        x1, y1, x2, y2 = map(float, box)
        bbox = [x1, y1, x2 - x1, y2 - y1]
        if model in ("yolov9", "yolov5s_seg"):
            # Preserve FP32 center->top-left arithmetic before decimal serialization.
            xywh = box.clone()
            xywh[:2] = (box[:2] + box[2:]) / 2
            xywh[2:] = box[2:] - box[:2]
            xywh[:2] -= xywh[2:] / 2
            bbox, score = [round(v, 3) for v in xywh.tolist()], round(score, 5)
        item = dict(category_id=categories[label], bbox=bbox, score=score)
        if model == "yolov5s_seg":
            item["segmentation"] = rles[i]
        result.append(item)
    return result
