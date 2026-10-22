"""Resolve model evaluation once, independently of host-graph execution."""

from dataclasses import dataclass, replace
from functools import partial
from typing import Callable

from .evaluation_profiles import (
    CURRENT,
    HISTORICAL,
    NUMPY_TOPK,
    CLASSIFIERS,
    FAMILIES,
    SEMANTIC,
    SPECIAL,
    GENERIC,
    NAMES,
)


def legacy_profile(model, name, kind=None):
    """Validate a compatibility input without changing its original meaning."""
    if name in NAMES:
        if name in ("numpy_topk_v1", "stable_topk_v1"):
            valid = model in CLASSIFIERS or kind == "logits"
        elif name in SPECIAL.values():
            valid = SPECIAL.get(model) == name
        elif name == "yolov7_face_clip_all_v1":
            valid = model == "yolov7_face"
        elif name == "yolov5_seg_scaled_coeff_mask_area_v1":
            valid = model == "yolov5s_seg"
        else:
            valid = GENERIC.get(kind) == name
        if not valid:
            raise ValueError(f"Evaluation contract {name} is not verified for {model}")
        return SEMANTIC.get(name, CURRENT)
    if name not in (
        CURRENT,
        HISTORICAL,
        NUMPY_TOPK,
        "face_all_boxes_v1",
        "mask_area_v1",
    ):
        raise ValueError("Unknown evaluation profile: " + str(name))
    if name == NUMPY_TOPK:
        if model not in CLASSIFIERS:
            raise ValueError("numpy_argsort_v1 requires a selected classifier")
    elif name != CURRENT and model not in FAMILIES:
        raise ValueError(f"Evaluation profile {name} is not verified for {model}")
    if name == "face_all_boxes_v1" and model != "yolov7_face":
        raise ValueError("face_all_boxes_v1 requires yolov7_face")
    if name == "mask_area_v1" and model != "yolov5s_seg":
        raise ValueError("mask_area_v1 requires yolov5s_seg")
    return name


@dataclass(frozen=True)
class EvaluationContract:
    """A decoder plus scoring policy. Custom callables are never reference evidence.

    postprocess receives (host_outputs, recipe, metadata, categories). Detection,
    face and keypoint decoders return their existing prediction rows. Semantic
    segmentation decoders return a label array. Classification/verification use
    their dedicated evaluators and do not accept this hook.
    """

    name: str
    postprocess: Callable | None
    ranking: str = "stable"
    mask_area: str = "mask"
    bbox_metric: bool = False
    custom: bool = False

    def identity(self):
        return dict(
            contract=self.name,
            ranking=self.ranking,
            mask_area=self.mask_area,
            bbox_metric=self.bbox_metric,
            reusable=not self.custom,
            original_reproduction_eligible=not self.custom,
        )


def resolve(recipe, *, postprocess=None):
    """Resolve aliases and numerical behavior once when constructing an evaluator."""
    from . import model_postprocess as decoders
    from .historical_yolo import postprocess as torch_decode

    model, kind = recipe.get("name", ""), recipe["decode"]["kind"]
    legacy = legacy_profile(model, recipe["decode"].get("profile", CURRENT), kind)
    if legacy == NUMPY_TOPK:
        result = EvaluationContract("numpy_topk_v1", None, ranking="numpy_argsort")
    elif legacy != CURRENT:
        name = SPECIAL[model]
        if legacy == "face_all_boxes_v1":
            name = "yolov7_face_clip_all_v1"
        elif legacy == "mask_area_v1":
            name = "yolov5_seg_scaled_coeff_mask_area_v1"
        result = EvaluationContract(
            name,
            partial(torch_decode, profile=legacy),
            mask_area="bbox" if legacy == HISTORICAL else "mask",
            bbox_metric=kind == "yolo_seg",
        )
    else:
        result = EvaluationContract(GENERIC[kind], decoders.for_recipe(recipe))
    if postprocess is not None:
        if not callable(postprocess):
            raise TypeError("postprocess must be a callable")
        if result.postprocess is None:
            raise ValueError(
                "Use a dedicated evaluator for classification or face verification"
            )
        result = replace(
            result, name="custom_postprocess", postprocess=postprocess, custom=True
        )
    return result
