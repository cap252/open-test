"""Full LFW evaluation and rejection of archived subset artifacts."""

FULL_PAIRS = 6000
GUIDANCE = (
    "LFW requires the full 6000-pair manifest and an evaluation limit of at least 6000 "
    "(or no limit). Use reproduce prepare --limit full for replay, or run --limit 6000. "
    "For a small check, select --models resnet18,yolov5s. "
    "Partial LFW packages and results are not supported by this version."
)


def require_full(info=None, limit=None):
    """Validate the requested scope without altering the manifest or limit."""
    if limit not in (None, "full") and (type(limit) is not int or limit < FULL_PAIRS):
        raise ValueError(GUIDANCE)
    if info is not None and (
        info.get("smoke")
        or len(info.get("evaluation_pairs", info.get("pairs", []))) != FULL_PAIRS
    ):
        raise ValueError(GUIDANCE)


def reject_partial_result(record):
    """Reject legacy diagnostics before execution, reporting, or comparison."""
    metadata = record.get("model_metadata", {})
    is_lfw = (
        record.get("task", metadata.get("task")) == "face_verification"
        or record.get("dataset", metadata.get("dataset")) == "lfw_mobilefacenet"
        or record.get("unit") == "pairs"
    )
    if (
        record.get("evaluation_mode") == "functional_check"
        or record.get("functional_check")
        or record.get("primary_metric", record.get("metric")) == "smoke_pair_accuracy"
        or "smoke_pair_accuracy" in record.get("metrics", {})
    ):
        raise ValueError(GUIDANCE)
    if is_lfw:
        require_full(limit=record.get("limit"))
        count = record.get("metric_details", {}).get("pair_count")
        if count is not None and count != FULL_PAIRS:
            raise ValueError(GUIDANCE)
