"""Load and validate experiment YAML: a packaged preset (core10, quick, extended, cnn38, cnn38_smoke) or a user file.

An experiment names the models, scheme families, conditions and run settings.
validate() rejects unknown keys and wrong types before any work starts, and every
error names the YAML key path, e.g. "adaround.steps must be an integer >= 1".
"""

from dataclasses import fields
from importlib.resources import files
import math
from pathlib import Path
import re

from ._io import read_yaml
from .ort.config import DEFAULT_GPU_MEMORY_LIMIT
from .quant.config import AdaroundConfig
from .quant.scheme import Scheme

PRESETS = ("cnn38", "cnn38_smoke", "core10", "extended", "quick")
DEFAULT_INPUT_CACHE_BYTES = 2 * 1024**3
# Top-level sections; activation_cache is the only optional one.
SECTIONS = {
    "name",
    "models",
    "runtime",
    "calibration",
    "schemes",
    "conditions",
    "percentiles",
    "adaround",
    "activation_cache",
    "evaluation",
    "output",
    "report",
}
# Section -> (allowed keys, required keys).
SECTION_KEYS = {
    "runtime": (
        {"device", "intra_op_threads", "inter_op_threads", "cuda"},
        {"device", "intra_op_threads", "inter_op_threads"},
    ),
    "calibration": (
        {
            "device",
            "histogram_bins",
            "samples",
            "input_cache",
            "expected_samples_identity",
        },
        {"device", "histogram_bins", "samples"},
    ),
    "evaluation": ({"limit", "profiles"}, {"limit"}),
    "output": (
        {"save_qdq_models", "save_predictions"},
        {"save_qdq_models", "save_predictions"},
    ),
    "report": ({"primary_metric", "target_mean_recovery"}, set()),
    "activation_cache": (
        {"enabled", "host_bytes", "device_bytes", "device_reserve_bytes"},
        set(),
    ),
}


def _mapping(value, path, *, allowed=None, required=()):
    if not isinstance(value, dict):
        raise ValueError(path + " must be a mapping")
    missing = set(required) - value.keys()
    unknown = value.keys() - allowed if allowed is not None else set()
    if missing or unknown:
        kind, keys = ("Missing", missing) if missing else ("Unknown", unknown)
        raise ValueError(
            kind
            + " setting(s): "
            + ", ".join(f"{path}.{k}" for k in sorted(keys, key=str))
        )
    return value


def check_device(value, path, *, allow_run=False):
    if not isinstance(value, str) or not (
        value == "cpu"
        or allow_run
        and value == "run"
        or value.startswith("cuda:")
        and value[5:].isdigit()
    ):
        choices = "run, cpu or cuda:N" if allow_run else "cpu or cuda:N"
        raise ValueError(f"{path} must be {choices}")


def _integer(value, path, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")


def _boolean(value, path):
    if type(value) is not bool:
        raise ValueError(f"{path} must be boolean (true or false)")


def _number(value, path):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{path} must be a finite number")


def load_config(name="core10"):
    """Load a packaged preset (core10, quick, extended, cnn38, cnn38_smoke) or an experiment YAML/JSON file and validate it."""
    path = Path(name)
    if not path.is_file():
        if str(name) not in PRESETS:
            raise ValueError(
                f"Experiment file not found: {name}. Pass an existing YAML file "
                f"or a packaged preset: {', '.join(PRESETS)}"
            )
        path = files("opennpu_quant.resources.experiments").joinpath(
            str(name) + ".yaml"
        )
    if path.name.endswith(".json"):
        from ._io import read_json

        value = read_json(path)
    else:
        value = read_yaml(path)
    validate(value)
    return value


def validate(value, *, models_dir=None):
    """Check an experiment mapping. With models_dir, local recipes are checked too."""
    _mapping(
        value, "experiment", allowed=SECTIONS, required=SECTIONS - {"activation_cache"}
    )
    _validate_names(value)
    _validate_settings(value)
    _validate_models(value, models_dir)
    _validate_conditions(value)
    _mapping(
        value["adaround"],
        "adaround",
        allowed={field.name for field in fields(AdaroundConfig)},
    )
    try:
        AdaroundConfig(**value["adaround"])
    except ValueError as error:
        raise ValueError(f"adaround.{error}") from None


def _validate_names(value):
    if not isinstance(value["models"], (list, tuple)):
        raise ValueError("models must be a list of model names")
    _mapping(value["conditions"], "conditions")
    if not value["conditions"]:
        raise ValueError("conditions must contain at least one condition")
    for section in ("schemes", "percentiles", "adaround"):
        _mapping(value[section], section)
    for path, names in (
        ("name", [value["name"]]),
        ("models", value["models"]),
        ("conditions", value["conditions"]),
    ):
        if any(
            not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", n)
            for n in names
        ):
            raise ValueError(
                path + " must use simple names (letters, digits, '_' or '-')"
            )
    if not value["models"]:
        raise ValueError("models must list at least one model")
    if len(value["models"]) != len(set(value["models"])):
        raise ValueError("Unique model names required")


def _validate_settings(value):
    from .results import report_policy

    for section, (allowed, required) in SECTION_KEYS.items():
        _mapping(value.get(section, {}), section, allowed=allowed, required=required)
    samples = _mapping(value["calibration"]["samples"], "calibration.samples")
    if not samples:
        raise ValueError("calibration.samples must contain dataset sample counts")
    for dataset, count in samples.items():
        _integer(count, f"calibration.samples.{dataset}", 1)
    expected = _mapping(
        value["calibration"].get("expected_samples_identity", {}),
        "calibration.expected_samples_identity",
    )
    for model, digest in expected.items():
        if (
            model not in value["models"]
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(
                "calibration.expected_samples_identity must map selected models to SHA256 digests"
            )
    for field in ("intra_op_threads", "inter_op_threads"):
        _integer(value["runtime"][field], "runtime." + field, 0)
    limit = value["evaluation"]["limit"]
    if limit is not None:
        _integer(limit, "evaluation.limit", 1)
    profiles = _mapping(value["evaluation"].get("profiles", {}), "evaluation.profiles")
    from .models.evaluation_profiles import validate_profile

    for model, profile in profiles.items():
        validate_profile(model, profile)
    for field, flag in value["output"].items():
        _boolean(flag, "output." + field)
    for field, item in value.get("activation_cache", {}).items():
        if field == "enabled":
            _boolean(item, "activation_cache.enabled")
        else:
            _integer(item, "activation_cache." + field, 0)
    check_device(value["runtime"]["device"], "runtime.device")
    check_device(value["calibration"]["device"], "calibration.device", allow_run=True)
    _integer(value["calibration"]["histogram_bins"], "calibration.histogram_bins", 2)
    cache = value["calibration"].get("input_cache", {})
    _mapping(cache, "calibration.input_cache", allowed={"enabled", "max_bytes"})
    _boolean(cache.get("enabled", False), "calibration.input_cache.enabled")
    _integer(
        cache.get("max_bytes", DEFAULT_INPUT_CACHE_BYTES),
        "calibration.input_cache.max_bytes",
        1,
    )
    report_policy(value.get("report"))
    cuda = value["runtime"].get("cuda", {})
    _mapping(cuda, "runtime.cuda", allowed={"gpu_mem_limit"})
    _integer(
        cuda.get("gpu_mem_limit", DEFAULT_GPU_MEMORY_LIMIT),
        "runtime.cuda.gpu_mem_limit",
        1,
    )


def _validate_models(value, models_dir):
    from .models.spec import ModelSpec, model_names

    packaged = model_names()
    for model in value["models"]:
        local = (
            Path(models_dir) / model / "recipe.yaml" if models_dir is not None else None
        )
        source = local if local is not None and local.is_file() else model
        if models_dir is not None and (source != model or model in packaged):
            dataset = ModelSpec.load(source).recipe["dataset"]
            if dataset not in value["calibration"]["samples"]:
                raise ValueError(
                    f"Missing setting: calibration.samples.{dataset} (model '{model}')"
                )
        for family in value["schemes"]:
            scheme_for(value, model, family)


def _validate_conditions(value):
    for name, row in value["conditions"].items():
        _mapping(row, "conditions." + name, allowed={"scheme", "adaround"})
        if name == "fp32" and row:
            raise ValueError("conditions.fp32 must be an empty mapping: fp32: {}")
        if "adaround" in row:
            _boolean(row["adaround"], f"conditions.{name}.adaround")
        if row and (
            not isinstance(row.get("scheme"), str)
            or row["scheme"] not in value["schemes"]
        ):
            raise ValueError(
                f"Invalid conditions.{name}.scheme; scheme families: "
                + ", ".join(map(str, value["schemes"]))
            )
        if not row and name != "fp32":
            raise ValueError(f"conditions.{name} needs a scheme; only fp32 may omit it")


def scheme_for(config, model, family):
    """Build one scheme family's Scheme for a model (resolves percentile: per_model)."""
    fields = dict(_mapping(config["schemes"][family], f"schemes.{family}"))
    _boolean(
        fields.get("activation_symmetric", False),
        f"schemes.{family}.activation_symmetric",
    )
    bins = fields.get("quantized_bins", 128)
    _integer(bins, f"schemes.{family}.quantized_bins", 2)
    if bins > config["calibration"]["histogram_bins"]:
        raise ValueError(
            f"calibration.histogram_bins must be >= schemes.{family}.quantized_bins ({bins})"
        )
    if fields.get("percentile") == "per_model":
        path = f"percentiles.{model}"
        if model not in config["percentiles"]:
            raise ValueError("Missing setting: " + path)
        per_model = _mapping(config["percentiles"][model], path)
        if family not in per_model:
            raise ValueError("Missing setting: " + path + "." + family)
        percentile = per_model[family]
        _number(percentile, f"{path}.{family}")
        if not 0 < percentile <= 100:
            raise ValueError(f"{path}.{family} must be in (0, 100]")
        fields["percentile"] = percentile
    elif fields.get("percentile") is not None:
        _number(fields["percentile"], f"schemes.{family}.percentile")
        if not 0 < fields["percentile"] <= 100:
            raise ValueError(f"schemes.{family}.percentile must be in (0, 100]")
    fields["histogram_bins"] = config["calibration"]["histogram_bins"]
    try:
        return Scheme(**fields)
    except TypeError as error:
        raise ValueError(f"Invalid schemes.{family} setting: {error}") from None


def runtime(config, device=None):
    """OrtConfig for the experiment's runtime settings (device defaults to runtime.device)."""
    from .ort.config import OrtConfig

    policy = config["runtime"]
    device = device or policy["device"]
    check_device(device, "runtime.device")
    opts = dict(
        intra_op_threads=policy["intra_op_threads"],
        inter_op_threads=policy["inter_op_threads"],
    )
    if device == "cpu":
        return OrtConfig.cpu(**opts)
    return OrtConfig.cuda(
        int(device[5:]),
        memory_limit=policy.get("cuda", {}).get(
            "gpu_mem_limit", DEFAULT_GPU_MEMORY_LIMIT
        ),
        **opts,
    )
