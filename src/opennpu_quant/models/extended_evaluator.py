import numpy as np
from ..evaluation.coco import CocoMetric
from ..evaluation.metrics import SegmentationEvaluator
from ..evaluation.dataset_metrics import wider_face_ap
from .._io import read_json
from .evaluation_contracts import resolve

KINDS = {
    "yolo_no_objectness",
    "retinanet",
    "nanodet",
    "yolov4",
    "yolo_face",
    "yolo_seg",
    "posenet",
    "segmentation_resize",
}


class ExtendedEvaluator:
    def __init__(self, bundle, dataset, *, contract=None):
        self.bundle, self.dataset = bundle, dataset
        self.recipe = bundle.spec.recipe
        self.contract = contract or resolve(self.recipe)
        self.cfg = self.recipe["decode"]
        self.kind = self.cfg["kind"]
        self.metric = None
        self.bbox_metric = None
        self.categories = []
        if self.kind in (
            "yolo_no_objectness",
            "retinanet",
            "nanodet",
            "yolov4",
            "yolo_seg",
            "posenet",
        ):
            annotation = dataset.annotations["coco"]
            self.categories = sorted(
                r["id"] for r in read_json(annotation)["categories"]
            )
            self.metric = CocoMetric(
                annotation,
                mask_area=self.contract.mask_area,
                iou_type="keypoints"
                if self.kind == "posenet"
                else "segm"
                if self.kind == "yolo_seg"
                else "bbox",
            )
            if self.contract.bbox_metric:
                self.bbox_metric = CocoMetric(annotation)
        elif self.kind == "segmentation_resize":
            self.metric = SegmentationEvaluator(output_name="labels")
        self.reset()

    def reset(self):
        if self.metric:
            self.metric.reset()
        if self.bbox_metric:
            self.bbox_metric.reset()
        self.prediction = None
        self.faces = {}
        self.truth = []

    def update(self, outputs, sample):
        out = self.bundle.finish(outputs, sample.metadata)
        meta = sample.metadata
        if any(not np.isfinite(v).all() for v in out.values()):
            raise ValueError("Nonfinite host output")
        prediction = self.contract.postprocess(out, self.recipe, meta, self.categories)
        if self.kind == "segmentation_resize":
            self.metric.update({"labels": prediction}, sample)
            self.prediction = {"labels": prediction.tolist()}
            return
        self.prediction = prediction
        if self.kind == "yolo_face":
            self.faces[int(sample.sample_id)] = prediction
            self.truth.append(meta["dataset_row"]["truth"])
        else:
            self.metric.update(prediction, sample)
            if self.bbox_metric:
                self.bbox_metric.update(prediction, sample)

    def finalize(self):
        if self.kind == "yolo_face":
            return wider_face_ap(self.faces, self.truth)
        values = self.metric.finalize()
        if self.kind == "yolo_seg":
            result = {"mask_" + k: v for k, v in values.items()}
            if self.bbox_metric:
                result.update(
                    {"bbox_" + k: v for k, v in self.bbox_metric.finalize().items()}
                )
            return result
        return values

    @property
    def metric_details(self):
        if self.bbox_metric:
            return dict(mask=self.metric.details, bbox=self.bbox_metric.details)
        return getattr(
            self.metric, "metric_details", getattr(self.metric, "details", {})
        )
