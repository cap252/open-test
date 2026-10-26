"""Built-in image protocols of the extended model recipes; no upstream imports."""

import numpy as np
from PIL import Image

KINDS = {
    "aligned_face",
    "resize_normalize",
    "nanodet",
    "posenet",
    "unet_letterbox",
    "mmseg_resize",
    "yolox",
    "letterbox_dynamic",
}


def preprocess(image, cfg):
    import cv2

    width, height = image.size
    meta = dict(original_size=[width, height], scale=[1.0, 1.0], pad=[0, 0])
    pixels = np.asarray(image.convert("RGB"))
    kind = cfg["kind"]
    size = cfg.get("size", 640)
    if kind == "aligned_face":
        if (width, height) != (112, 112):
            raise ValueError(
                "Face recognition requires a prepared aligned 112x112 crop"
            )
        pixels = pixels.astype(np.float32) / np.float32(cfg["divide"])
        pixels = (pixels - np.asarray(cfg["mean"], np.float32)) / np.asarray(
            cfg["std"], np.float32
        )
        return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None]), meta
    if kind in ("resize_normalize", "nanodet", "posenet"):
        if kind == "nanodet":
            # NanoDet's validation warpPerspective uses an origin-aligned scale.
            matrix = np.diag([size / width, size / height, 1.0])
            pixels = cv2.warpPerspective(pixels[..., ::-1], matrix, (size, size))
        else:
            pixels = cv2.resize(pixels, (size, size), interpolation=cv2.INTER_LINEAR)
        if kind == "nanodet":
            # Preserve the source's divide-before-normalize FP32 order.
            pixels = pixels.astype(np.float32) / 255
            pixels = (pixels - np.asarray(cfg["mean"], np.float32) / 255) / (
                np.asarray(cfg["std"], np.float32) / 255
            )
        elif kind == "posenet":
            pixels = pixels.astype(np.float32) * (2.0 / 255.0) - 1.0
        else:
            pixels = (
                pixels.astype(np.float32) - np.asarray(cfg["mean"], np.float32)
            ) / np.asarray(cfg["std"], np.float32)
        meta["scale"] = [size / width, size / height]
        meta["output_scale_yx"] = [height / size, width / size]
    elif kind == "unet_letterbox":
        ratio = min(size / width, size / height)
        nw, nh = int(width * ratio), int(height * ratio)
        resized = image.resize((nw, nh), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (size, size), (128, 128, 128))
        left, top = (size - nw) // 2, (size - nh) // 2
        canvas.paste(resized, (left, top))
        pixels = np.asarray(canvas, np.float32) / 255
        meta.update(pad=[left, top], valid_size=[nw, nh])
    elif kind == "mmseg_resize":
        ratio = min(2048 / max(width, height), 512 / min(width, height))
        pixels = cv2.resize(
            pixels,
            (int(width * ratio + 0.5), int(height * ratio + 0.5)),
            interpolation=cv2.INTER_LINEAR,
        )
        pixels = (
            pixels.astype(np.float32) - np.array([123.675, 116.28, 103.53], np.float32)
        ) / np.array([58.395, 57.12, 57.375], np.float32)
    elif kind == "yolox":
        ratio = min(size / width, size / height)
        resized = cv2.resize(
            pixels[..., ::-1],
            (int(width * ratio), int(height * ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas = np.full((size, size, 3), 114, np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        pixels = canvas.astype(np.float32)
        meta["scale"] = [ratio, ratio]
    elif kind == "letterbox_dynamic":
        stride = cfg.get("stride", 32)
        ratio = min(size / width, size / height)
        if cfg.get("rectangular"):
            rw, rh = int(width * ratio), int(height * ratio)
            pixels = cv2.resize(
                pixels,
                (rw, rh),
                interpolation=cv2.INTER_LINEAR if ratio > 1 else cv2.INTER_AREA,
            )
            outw = int(np.ceil(width * ratio / stride + 0.5) * stride)
            outh = int(np.ceil(height * ratio / stride + 0.5) * stride)
            # The pinned validation loader passes its height ratio as the
            # scalar gain consumed by scale_boxes for both coordinate axes.
            meta["scale"] = [rh / height, rh / height]
        else:
            rw, rh = round(width * ratio), round(height * ratio)
            pixels = cv2.resize(pixels, (rw, rh), interpolation=cv2.INTER_LINEAR)
            outw = rw + (size - rw) % stride
            outh = rh + (size - rh) % stride
            meta["scale"] = [ratio, ratio]
        dw, dh = (outw - rw) / 2, (outh - rh) / 2
        left, right = round(dw - 0.1), round(dw + 0.1)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        pixels = (
            cv2.copyMakeBorder(
                pixels,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            ).astype(np.float32)
            / 255
        )
        meta["pad"] = [dw, dh]
        if not cfg.get("rectangular"):
            # YOLOv7-face's scale_coords recomputes padding from the original
            # image shape, before rounding resized pixel dimensions.
            meta["pad"] = [(outw - width * ratio) / 2, (outh - height * ratio) / 2]
    else:
        raise ValueError("Unknown extended preprocessing: " + kind)
    meta["input_size"] = [pixels.shape[1], pixels.shape[0]]
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32), meta
