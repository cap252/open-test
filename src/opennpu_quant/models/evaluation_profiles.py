"""Versioned evaluation contracts selected by model recipes or explicit overrides."""

import copy
from importlib.metadata import version, PackageNotFoundError

CURRENT = "current_v1"
HISTORICAL = "historical_cnn33_v1"
NUMPY_TOPK = "numpy_argsort_v1"
PROFILES = (CURRENT, HISTORICAL, NUMPY_TOPK, "face_all_boxes_v1", "mask_area_v1")
CLASSIFIERS = frozenset(
    (
        "resnet18",
        "resnet50",
        "resnet101",
        "mobilenet_v1",
        "mobilenet_v2",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "efficientnet_lite0",
        "efficientnet_lite4",
        "inception_v3",
        "shufflenet_v2_x1_0",
        "squeezenet1_1",
        "vgg16",
    )
)
FAMILIES = {
    "yolox": "yolox",
    "yolov9_t": "yolov9",
    "yolov9_s": "yolov9",
    "yolov7_face": "yolov7_face",
    "yolov5s_seg": "yolov5s_seg",
}
MODELS = frozenset(FAMILIES)


# Compatibility inputs describe an implementation, never a moving default.
SEMANTIC = {
    "numpy_topk_v1": NUMPY_TOPK,
    "stable_topk_v1": CURRENT,
    "yolov9_torch_nms_v1": HISTORICAL,
    "yolox_torch_nms_v1": HISTORICAL,
    "yolov7_face_clip_first4_v1": HISTORICAL,
    "yolov7_face_clip_all_v1": "face_all_boxes_v1",
    "yolov5_seg_scaled_coeff_bbox_area_v1": HISTORICAL,
    "yolov5_seg_scaled_coeff_mask_area_v1": "mask_area_v1",
}
SPECIAL = {
    "yolov9_t": "yolov9_torch_nms_v1",
    "yolov9_s": "yolov9_torch_nms_v1",
    "yolox": "yolox_torch_nms_v1",
    "yolov7_face": "yolov7_face_clip_first4_v1",
    "yolov5s_seg": "yolov5_seg_scaled_coeff_bbox_area_v1",
}
GENERIC = {
    "logits": "stable_topk_v1",
    "yolo": "yolo_opencv_nms_v1",
    "yolo_no_objectness": "yolo_scores_opencv_nms_v1",
    "yolo_face": "yolo_face_opencv_nms_v1",
    "yolo_seg": "yolo_seg_unscaled_coeff_mask_area_v1",
    "tf_od_api": "tf_od_api_v1",
    "retinaface": "retinaface_v1",
    "retinanet": "retinanet_host_nms_v1",
    "nanodet": "nanodet_v1",
    "yolov4": "yolov4_opencv_nms_v1",
    "posenet": "posenet_multi_pose_v1",
    "segmentation_labels": "segmentation_labels_v1",
    "segmentation_resize": "segmentation_resize_v1",
    "face_verification": "lfw_flip_ten_fold_v1",
}
NAMES = tuple(dict.fromkeys((*SEMANTIC, *GENERIC.values())))

PROFILES = (*PROFILES, *NAMES)


def decoder_family(model):
    return FAMILIES.get(model, model)


def validate_profile(model, profile, kind=None):
    from .evaluation_contracts import legacy_profile

    if kind is None and profile in NAMES:
        from .spec import ModelSpec, model_names

        if model in model_names():
            kind = ModelSpec.load(model).recipe["decode"]["kind"]
    legacy_profile(model, profile, kind)


def apply_profile(recipe, profile):
    validate_profile(recipe["name"], profile, recipe["decode"]["kind"])
    result = copy.deepcopy(recipe)
    result["decode"]["profile"] = profile
    return result


def profile_name(recipe):
    name = recipe["decode"].get("profile", CURRENT)
    validate_profile(recipe["name"], name, recipe["decode"]["kind"])
    return name


def mask_area(recipe):
    from .evaluation_contracts import resolve

    return resolve(recipe).mask_area


def identity(recipe):
    dependencies = {}
    for name in (
        "numpy",
        "torch",
        "torchvision",
        "opencv-python",
        "opencv-python-headless",
        "pycocotools",
    ):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            pass
    from .evaluation_contracts import resolve

    return dict(
        contract=resolve(recipe).identity(),
        profile=profile_name(recipe),
        mask_area=mask_area(recipe),
        dependencies=dependencies,
    )


def decode(outputs, recipe, meta, categories):
    """Compatibility entry point. Evaluators resolve the callable at construction."""
    from .evaluation_contracts import resolve

    contract = resolve(recipe)
    if contract.postprocess is None:
        raise ValueError("This task uses a dedicated evaluator")
    return contract.postprocess(outputs, recipe, meta, categories)
