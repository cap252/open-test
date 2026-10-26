from dataclasses import dataclass
from pathlib import Path
from importlib.resources import files
from .._io import read_yaml


@dataclass(frozen=True)
class ModelSpec:
    recipe: dict

    def __post_init__(self):
        validate_recipe_policy(self.recipe)

    @property
    def name(self):
        return self.recipe["name"]

    @classmethod
    def for_run(cls, name, directory, *, profile=None):
        """Use packaged evaluation defaults with the local bundle's graph contract."""
        from .evaluation_profiles import apply_profile, profile_name

        local = Path(directory) / "recipe.yaml"
        spec = cls.load(local if local.is_file() else name)
        if spec.name != name:
            raise ValueError(f"Local recipe name {spec.name!r} does not match {name!r}")
        if profile is None and name in model_names():
            profile = profile_name(cls.load(name).recipe)
            previous = profile_name(spec.recipe)
            if local.is_file() and previous != profile:
                import warnings

                warnings.warn(
                    f"{name}: local profile {previous} is replaced by the packaged "
                    f"default {profile}. To compare another convention, set "
                    f"evaluation.profiles.{name} explicitly.",
                    UserWarning,
                    stacklevel=2,
                )
        return cls(apply_profile(spec.recipe, profile)) if profile is not None else spec

    @classmethod
    def load(cls, name):
        path = Path(name)
        if path.is_file():
            data = read_yaml(path)
        else:
            if str(name) not in model_names():
                raise ValueError(
                    f"Unknown model: {name}. Packaged models: {', '.join(model_names())}"
                )
            data = read_yaml(
                files("opennpu_quant.resources.models").joinpath(str(name) + ".yaml")
            )
        required = {
            "name",
            "title",
            "task",
            "dataset",
            "metric",
            "input",
            "graphs",
            "preprocess",
            "decode",
            "source",
            "export",
        }
        allowed = required | {"network_input_map"}
        if not isinstance(data, dict):
            raise ValueError(f"Model recipe must be a mapping: {name}")
        missing, unknown = required - set(data), set(data) - allowed
        if missing or unknown:
            _reject_removed_settings(data)
            raise ValueError(
                f"Invalid model recipe keys in {name}: "
                f"missing {sorted(missing)}, unknown {sorted(map(str, unknown))}"
            )
        if (
            set(data["graphs"]) - {"network", "preprocess", "postprocess"}
            or "network" not in data["graphs"]
        ):
            raise ValueError("Expected a network and optional host graphs")
        for value in data["graphs"].values():
            if Path(value).is_absolute() or ".." in Path(value).parts:
                raise ValueError("Unsafe graph path")
        return cls(data)


def model_names():
    return sorted(
        p.name[:-5]
        for p in files("opennpu_quant.resources.models").iterdir()
        if p.name.endswith(".yaml")
    )


# Built-in preprocessing kinds that fix the network input layout.
PREPROCESS_LAYOUT = {
    "center_crop": "NCHW",
    "letterbox": "NCHW",
    "resize_bgr_mean": "NCHW",
    "resize_uint8": "NHWC",
    **dict.fromkeys(
        (
            "resize_normalize",
            "nanodet",
            "posenet",
            "unet_letterbox",
            "mmseg_resize",
            "yolox",
            "letterbox_dynamic",
            "aligned_face",
        ),
        "NCHW",
    ),
}


def _reject_removed_settings(recipe):
    """Reject removed keys without restricting plugin-owned preprocessing settings."""
    paths = [
        "outputs",
        "network_outputs",
        "metric_params",
        "input.dtype",
        "export.framework_parity.rtol",
        "export.framework_parity.atol",
        "export.framework_parity.require_top5_order",
    ]
    kind = recipe.get("preprocess", {}).get("kind")
    if kind in ("resize_uint8", "resize_bgr_mean"):
        paths.append("preprocess.interpolation")
    elif kind == "longer_side_pad":
        paths.append("preprocess.pad")
    decode_keys = {
        "retinaface": ("record", "rescale", "priors.clip"),
        "tf_od_api": ("boxes_to",),
        "yolo": (
            "score",
            "boxes",
            "category_ids",
            "nms.impl",
            "nms.per_class",
            "nms.score_threshold",
        ),
    }
    paths.extend(
        "decode." + key
        for key in decode_keys.get(recipe.get("decode", {}).get("kind"), ())
    )
    removed = []
    for path in sorted(paths):
        node = recipe
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                break
            node = node[key]
        else:
            removed.append(path)
    if removed:
        raise ValueError(
            "Unknown setting(s): "
            + ", ".join(removed)
            + ". Remove these keys; their behavior is fixed or the setting is no longer supported."
        )


def validate_recipe_policy(recipe):
    """Check that recipe settings agree with each other and with the preprocessing."""
    _reject_removed_settings(recipe)
    from .evaluation_profiles import profile_name

    profile_name(recipe)
    layout = recipe.get("input", {}).get("layout")
    if layout not in ("NCHW", "NHWC"):
        raise ValueError("Recipe input.layout must be NCHW or NHWC")
    kind = recipe.get("preprocess", {}).get("kind")
    expected = PREPROCESS_LAYOUT.get(kind)
    if expected and layout != expected:
        raise ValueError(f"Preprocessing {kind} produces {expected}; set input.layout")
    if not isinstance(recipe.get("export", {}).get("framework_parity", {}), dict):
        raise ValueError("export.framework_parity must be a mapping")
    if recipe.get("export", {}).get("family") == "tf_deeplab":
        for key in ("graph_input", "graph_output"):
            value = recipe["export"].get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("Invalid DeepLab graph tensor name: " + key)
