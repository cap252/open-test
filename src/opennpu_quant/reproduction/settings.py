"""Generate ordinary runner configurations from the final recorded selection."""

from copy import deepcopy
from pathlib import Path

from .._io import atomic_json, object_hash
from ..configuration import validate
from ..models.evaluation_profiles import profile_name
from .assets import select_catalog, seal, file_record


def training_config(entry, template):
    config = deepcopy(template)
    model, condition = entry["model"], entry["condition"]
    config["name"] = "selected_" + entry["id"]
    config["models"] = [model]
    config["conditions"] = {condition: config["conditions"][condition]}
    config["runtime"]["device"] = "cuda:0"
    config["runtime"]["cuda"] = {"gpu_mem_limit": entry["execution"]["gpu_mem_limit"]}
    config["evaluation"]["profiles"] = {model: profile_name(entry["recipe"])}
    config["evaluation"]["limit"] = None
    if condition != "fp32":
        config["percentiles"][model][condition] = entry["percentile"]
    if entry.get("adaround"):
        config["adaround"] = deepcopy(entry["adaround"])
    validate(config)
    return config


def configs(*, output, selection=None, model_set=None, models=None, conditions=None):
    source, entries = select_catalog(
        selection=selection, model_set=model_set, models=models, conditions=conditions
    )
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    templates = source.get("training_templates")
    if not isinstance(templates, dict):
        raise ValueError(
            "Selection has no training_templates. Config generation requires the "
            "original training settings; use reproduce prepare/run to evaluate fixed models."
        )
    values = []
    for entry in entries:
        model, condition = entry["model"], entry["condition"]
        template = templates.get(model)
        if not isinstance(template, dict):
            raise ValueError("Missing training template for model: " + model)
        available_conditions = template.get("conditions")
        if (
            not isinstance(available_conditions, dict)
            or condition not in available_conditions
        ):
            raise ValueError(
                "Missing training template condition: " + model + "/" + condition
            )
        values.append((entry, training_config(entry, template)))
    output.mkdir(parents=True)
    rows = []
    for entry, config in values:
        path = output / (entry["id"] + ".json")
        atomic_json(path, config)
        rows.append(
            dict(
                id=entry["id"],
                config=file_record(path, output),
                source_result=entry["source_result"],
                selected_model=entry["graphs"]["network"],
                retraining="not_run",
            )
        )
    seal(
        output / "manifest.json",
        dict(
            selection_identity=object_hash(source),
            conditions=rows,
            scope="Ordinary runner configs; fixed-model replay uses reproduce run",
            retraining="not_run",
        ),
    )
    return dict(output=str(output), conditions=len(rows), retraining="not_run")
