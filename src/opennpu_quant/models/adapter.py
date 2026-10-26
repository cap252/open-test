import numpy as np
from ..evaluation.metrics import TopKClassification, SegmentationEvaluator
from ..evaluation.coco import CocoMetric
from ..evaluation.dataset_metrics import wider_face_ap
from .evaluation_contracts import resolve


class ModelEvaluator:
    """Evaluator for a recipe: run host postprocessing, decode by decode.kind, score.

    Core decoders are handled here; extended tasks and face verification delegate
    to their evaluators. The last prediction is kept for --save-predictions.
    """

    def __init__(self, bundle, dataset, *, postprocess=None):
        self.bundle, self.dataset = bundle, dataset
        self.recipe = bundle.spec.recipe
        self.kind = self.recipe["decode"]["kind"]
        self.contract = resolve(self.recipe, postprocess=postprocess)
        self.categories = []
        from .extended_evaluator import KINDS, ExtendedEvaluator

        self.delegate = (
            ExtendedEvaluator(bundle, dataset, contract=self.contract)
            if self.kind in KINDS
            else None
        )
        if self.kind == "face_verification":
            from ..evaluation.face_verification import FaceVerificationEvaluator

            self.delegate = FaceVerificationEvaluator(bundle, dataset)
        if self.delegate is not None:
            self.metric = None
        elif self.kind == "logits":
            self.metric = TopKClassification(
                output_name=self.recipe["decode"]["output"],
            )
            self.metric._ranking = self.contract.ranking
        elif self.kind == "segmentation_labels":
            self.metric = SegmentationEvaluator(output_name="labels")
        elif self.kind in ("yolo", "tf_od_api"):
            self.metric = CocoMetric(dataset.annotations["coco"])
            from .._io import read_json

            self.categories = sorted(
                x["id"] for x in read_json(dataset.annotations["coco"])["categories"]
            )
        elif self.kind == "retinaface":
            self.metric = None
        else:
            raise ValueError("Unknown decoder: " + self.kind)
        self.reset()

    def reset(self):
        if self.delegate is not None:
            self.delegate.reset()
        if self.metric:
            self.metric.reset()
        self.detections = {}
        self.truth = []
        self.prediction = None

    def update(self, outputs, sample):
        if self.delegate is not None:
            self.delegate.update(outputs, sample)
            self.prediction = self.delegate.prediction
            return
        out = self.bundle.finish(outputs, sample.metadata)
        meta = sample.metadata
        if any(not np.isfinite(v).all() for v in out.values()):
            raise ValueError("Nonfinite host output")
        if self.kind == "logits":
            self.metric.update(out, sample)
            self.prediction = self.metric.prediction
            return
        prediction = self.contract.postprocess(out, self.recipe, meta, self.categories)
        if self.kind == "segmentation_labels":
            self.metric.update({"labels": prediction}, sample)
            self.prediction = {"labels": prediction.tolist()}
        elif self.kind == "retinaface":
            self.detections[int(sample.sample_id)] = prediction
            self.truth.append(meta["dataset_row"]["truth"])
            self.prediction = prediction
        else:
            self.metric.update(prediction, sample)
            self.prediction = prediction

    @property
    def metric_details(self):
        if self.delegate is not None:
            return self.delegate.metric_details
        return getattr(
            self.metric, "metric_details", getattr(self.metric, "details", {})
        )

    @property
    def metric_units(self):
        if self.delegate is not None:
            return getattr(self.delegate, "metric_units", {})
        return getattr(self.metric, "metric_units", {})

    def finalize(self):
        if self.delegate is not None:
            return self.delegate.finalize()
        if self.kind == "retinaface":
            return wider_face_ap(self.detections, self.truth)
        return self.metric.finalize()
