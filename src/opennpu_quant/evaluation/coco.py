import copy
import contextlib
import sys


def evaluate_predictions(
    ground_truth, predictions, image_ids, iou_type="bbox", *, mask_area="mask"
):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if mask_area not in ("mask", "bbox"):
        raise ValueError("COCO mask_area must be mask or bbox")
    if iou_type == "segm" and mask_area == "mask":
        # COCO loadRes otherwise uses bbox area even when scoring masks.
        predictions = [
            {k: v for k, v in row.items() if k != "bbox"} for row in predictions
        ]
    # loadRes annotates rows in place; callers may replay the same predictions.
    predictions = copy.deepcopy(predictions)
    if predictions:
        detected = ground_truth.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = dict(
            images=ground_truth.dataset["images"],
            categories=ground_truth.dataset["categories"],
            annotations=[],
        )
        detected.createIndex()
    if iou_type not in ("bbox", "segm", "keypoints"):
        raise ValueError("Unsupported COCO evaluation type")
    evaluator = COCOeval(ground_truth, detected, iou_type)
    evaluator.params.imgIds = list(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    if iou_type == "keypoints":
        names = [
            "AP",
            "AP50",
            "AP75",
            "AP_medium",
            "AP_large",
            "AR",
            "AR50",
            "AR75",
            "AR_medium",
            "AR_large",
        ]
    # pycocotools reports -1 for a range without ground truth (e.g. no small objects).
    metrics = {name: float(v) for name, v in zip(names, evaluator.stats) if v >= 0}
    per_class = []
    for i, category in enumerate(evaluator.params.catIds):
        precision = evaluator.eval["precision"][:, :, i, 0, -1]
        values = precision[precision > -1]
        per_class.append(
            dict(
                category_id=int(category),
                name=ground_truth.cats[category]["name"],
                AP=float(values.mean()) if values.size else None,
            )
        )
    return dict(
        metrics=metrics,
        mask_area=mask_area if iou_type == "segm" else None,
        per_class=per_class,
        image_count=len(image_ids),
        prediction_count=len(predictions),
        iou_thresholds=evaluator.params.iouThrs.tolist(),
        max_detections=evaluator.params.maxDets,
        evaluator=f"pycocotools.cocoeval.COCOeval {iou_type}; standard 101 recall thresholds and crowd/area handling",
    )


class CocoMetric:
    """Collect COCO bbox detections per image and score them with pycocotools."""

    def __init__(self, annotation, *, iou_type="bbox", mask_area="mask"):
        self.iou_type = iou_type
        self.mask_area = mask_area
        self.annotation = str(annotation)
        self.reset()

    def reset(self):
        self.predictions = []
        self.image_ids = []
        self.details = {}

    def update(self, predictions, sample):
        image_id = int(sample.sample_id)
        self.image_ids.append(image_id)
        self.predictions.extend((dict(image_id=image_id, **row) for row in predictions))

    def finalize(self):
        from pycocotools.coco import COCO

        # pycocotools prints progress; keep stdout for the CLI's JSON result.
        with contextlib.redirect_stdout(sys.stderr):
            self.details = evaluate_predictions(
                COCO(self.annotation),
                self.predictions,
                self.image_ids,
                self.iou_type,
                mask_area=self.mask_area,
            )
        return self.details["metrics"]
