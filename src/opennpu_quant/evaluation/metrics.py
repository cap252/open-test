import numpy as np


class TopKClassification:
    """Accumulate per-image Top-K classification accuracy for evaluate().

    Parameters
    ----------
    output_name : str
        Logit tensor key, default 'logits'; must have shape [1, classes].
    topk : iterable of int
        Distinct positive K values, default (1, 5); logits need at least max(K)
        classes. EvaluationSample.target is the zero-based class index.
    Built-in adapters can select the recorded research tie order by profile.

    reset() clears counts, update(outputs, sample) adds one observation and
    finalize() returns fraction-valued topK metrics. Empty observations and
    nonfinite logits are rejected. Tied logits rank the lower class index first.

    Examples
    --------
    >>> from opennpu_quant import TopKClassification
    >>> evaluator = TopKClassification(output_name='logits', topk=(1, 5))
    >>> assert evaluator.metric_units == {'top1': 'fraction', 'top5': 'fraction'}
    """

    def __init__(self, *, output_name="logits", topk=(1, 5)):
        self._ranking = "stable"
        topk = tuple(topk)
        if (
            not topk
            or len(set(topk)) != len(topk)
            or any(type(k) is not int or k < 1 for k in topk)
        ):
            raise ValueError("topk must be distinct positive integers, e.g. (1, 5)")
        self.output_name = output_name
        self.topk = topk
        self.reset()

    def reset(self):
        self.count = 0
        self.prediction = None
        self.correct = {k: 0 for k in self.topk}

    def update(self, outputs, sample):
        logits = np.asarray(outputs[self.output_name])
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] < max(self.topk):
            raise ValueError(
                "One sample requires [1, classes] logits with sufficient classes"
            )
        if (
            not np.isfinite(logits).all()
            or not 0 <= int(sample.target) < logits.shape[1]
        ):
            raise ValueError("Invalid logits or target")
        # Reverse both the input and sorted indices so ties keep the lowest class
        # first, without negating integers or losing precision through float casts.
        if self._ranking == "numpy_argsort":
            prediction = np.argsort(logits[0])[::-1]
        else:
            reversed_order = np.argsort(logits[0, ::-1], kind="stable")
            prediction = logits.shape[1] - 1 - reversed_order[::-1]
        for k in self.topk:
            self.correct[k] += int(int(sample.target) in prediction[:k])
        self.count += 1
        self.prediction = dict(
            image_id=sample.sample_id,
            label=int(sample.target),
            top5=prediction[:5].tolist(),
            top5_logits=logits[0, prediction[:5]].tolist(),
        )

    @property
    def metric_units(self):
        return {f"top{k}": "fraction" for k in self.topk}

    def finalize(self):
        if not self.count:
            raise ValueError("No classification observations")
        return {f"top{k}": v / self.count for k, v in self.correct.items()}


class SegmentationEvaluator:
    def __init__(self, *, output_name, classes=21, ignore_label=255):
        self.output_name = output_name
        self.classes = classes
        self.ignore_label = ignore_label
        self.reset()

    def reset(self):
        from .segmentation import _ConfusionMatrix

        self._state = _ConfusionMatrix(self.classes, self.ignore_label)
        self.metric_details = {}

    @property
    def confusion(self):
        return self._state.matrix

    @property
    def metric_units(self):
        return {"mIoU": "fraction"}

    def update(self, outputs, sample):
        logits = np.asarray(outputs[self.output_name])
        if logits.ndim == 4:
            if logits.shape[:2] != (1, self.classes) or not np.isfinite(logits).all():
                raise ValueError(
                    "Expected finite [1, classes, H, W] segmentation logits"
                )
            predicted = logits.argmax(1)[0]
        else:
            predicted = logits
        self._state.add(predicted, sample.target)

    def finalize(self):
        result = self._state.metrics()
        self.metric_details = {k: v for k, v in result.items() if k != "mIoU"}
        return {"mIoU": result["mIoU"]}
