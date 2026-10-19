"""Bind the selected research results once; runtime replay needs no research Python."""

import argparse
import tempfile
from pathlib import Path
import onnx
from opennpu_quant._io import read_json, atomic_json, sha256
from opennpu_quant.models.spec import ModelSpec, model_names
from opennpu_quant.models.evaluation_profiles import (
    apply_profile,
    MODELS,
    HISTORICAL,
    CLASSIFIERS,
    NUMPY_TOPK,
)
from opennpu_quant.research import IDS
from opennpu_quant.reproduction.assets import catalog

if __package__:
    from .migrate_reference_metrics import migrate
else:
    from migrate_reference_metrics import migrate


def build(a):
    roots = {
        "research-runs": a.runs,
        "research-repo": a.research,
        "data-manifests": a.data,
        "gtx-package": a.frozen,
    }
    roots = {k: Path(v).resolve() for k, v in roots.items()}

    def uri(path):
        path = Path(path).absolute()
        for k, base in roots.items():
            if path.is_relative_to(base):
                return "artifact://" + k + "/" + str(path.relative_to(base))
        raise ValueError("Unbound path: " + str(path))

    def asset(path, digest=None):
        return dict(id=uri(path), sha256=digest or sha256(path))

    source = read_json(a.snapshot)["tables"]["All_Selected_Current"]
    rows = [r for r in source if IDS.get(r["model_id"], r["model_id"]) in model_names()]

    def source_path(r):
        path = Path(r["evidence"])
        if path.is_absolute():
            return path
        for base in (a.runs, a.research / "results"):
            path = base / r["source_run"] / r["evidence"]
            if path.is_file():
                return path
        raise FileNotFoundError(r["evidence"])

    source_results = {}
    for row in rows:
        path = source_path(row)
        if sha256(path) != row["evidence_sha256"]:
            raise ValueError("Source result changed: " + str(path))
        source_results[id(row)] = (path, read_json(path))
    frozen = read_json(a.frozen / "manifest.json")
    sealed = read_json(a.frozen / "CONTENTS-SHA256.json")
    recipes, fp_paths, datasets = {}, {}, {}
    for row in rows:
        if row["method"] != "fp32":
            continue
        name = IDS.get(row["model_id"], row["model_id"])
        path, result = source_results[id(row)]
        network = Path(result.get("model_path") or path.parent / "model.onnx")
        recipe = ModelSpec.load(name).recipe
        if name in MODELS:
            recipe = apply_profile(recipe, HISTORICAL)
        elif name in CLASSIFIERS:
            recipe = apply_profile(recipe, NUMPY_TOPK)
        parts = {"network": network}
        for kind in ("preprocess", "postprocess"):
            p = network.parent / (kind + ".onnx")
            if p.is_file():
                parts[kind] = p
        contract = network.parent / "contract.json"
        if contract.is_file():
            recipe["network_input_map"] = read_json(contract)["network_input_map"]
        recipe["graphs"] = {k: k + ".onnx" for k in parts}
        network_model = onnx.load(network, load_external_data=False)
        inputs = [v.name for v in network_model.graph.input]
        outputs = [v.name for v in network_model.graph.output]
        if "preprocess" not in parts:
            if len(inputs) != 1:
                raise ValueError("Explicit input map required: " + name)
            recipe["network_input_map"] = {inputs[0]: recipe["input"]["name"]}
        if "postprocess" in parts:
            outputs = [
                v.name
                for v in onnx.load(
                    parts["postprocess"], load_external_data=False
                ).graph.output
            ]
        aliases = {}
        cfg = recipe["decode"]
        wanted = cfg.get("output")
        if wanted and wanted not in outputs and len(outputs) == 1:
            aliases[wanted] = outputs[0]
        if name == "yolov5s_seg":
            aliases = {"predictions": "prediction"}
        recipes[name] = (recipe, aliases)
        fp_paths[name] = parts
        ds = recipe["dataset"]
        if ds not in datasets:
            if ds == "lfw_mobilefacenet":
                base = a.runs / "mobilefacenet_relu_20261014/datasets/lfw"
                datasets[ds] = dict(
                    kind="frozen_lfw",
                    samples=asset(base / "samples.json"),
                    cache=asset(base / "validation_inputs/manifest.json"),
                )
            else:
                manifest = a.data / ds / "manifest.json"
                info = read_json(manifest)
                if ds == "coco_keypoints":
                    from opennpu_quant.data.prepare import identifiers

                    expected = [int(row[0]) for row in identifiers(ds)]
                    if [row["id"] for row in info["calibration"]] != expected:
                        raise ValueError(
                            "Reprepare COCO keypoints with the pinned PoseNet calibration split"
                        )
                key = "dataset-" + ds
                roots[key] = Path(info["root"]).resolve()
                datasets[ds] = dict(
                    kind="images",
                    manifest=asset(manifest),
                    root="artifact://" + key + "/",
                )
        print(
            name,
            "inputs",
            inputs,
            "outputs",
            outputs[:5],
            "aliases",
            aliases,
            flush=True,
        )
    entries = []
    for row in rows:
        name = IDS.get(row["model_id"], row["model_id"])
        condition = (
            "fp32" if row["method"] == "fp32" else row["family"] + "_" + row["method"]
        )
        path, result = source_results[id(row)]
        network = Path(result.get("model_path") or path.parent / "model.onnx")
        recipe, aliases = recipes[name]
        graphs = {k: asset(v) for k, v in fp_paths[name].items() if k != "network"}
        graphs["network"] = asset(
            network, result.get("model_sha256") or row["model_sha256_recorded"]
        )
        audit = None
        if condition != "fp32":
            ap = Path(result.get("audit_path") or path.parent / "audit.json")
            audit = asset(
                ap,
                result.get("audit_sha256") or row.get("audit_sha256_recorded") or None,
            )
        observed = result.get("result", result)
        env = observed.get("environment", {})
        options = (env.get("provider_options") or [{}])[0]
        reference = None
        for g in frozen["graphs"]:
            if (
                g["model"] == name
                and g["files"][g["network"]] == graphs["network"]["sha256"]
            ):
                rel = "expected/" + g["id"] + "/manifest.json"
                reference = dict(
                    manifest=asset(a.frozen / rel, sealed[rel]),
                    samples=g["samples"],
                    root="artifact://gtx-package/",
                    kind="GTX raw; delivery decoder is not a research oracle",
                )
                break
        entry = dict(
            id=name + "__" + condition,
            model=name,
            condition=condition,
            recorded_condition=row["condition"],
            percentile=row.get("percentile"),
            recipe=recipe,
            output_aliases=aliases,
            graphs=graphs,
            audit=audit,
            dataset=datasets[recipe["dataset"]],
            source_result=asset(path, row["evidence_sha256"]),
            measurement=dict(
                metrics=observed.get("metrics", {}),
                samples=observed.get("sample_count"),
                recovery_percent=row.get("recovery_percent"),
                environment=env,
                samples_identity=observed.get("samples_identity"),
            ),
            execution=dict(
                gpu_mem_limit=int(options.get("gpu_mem_limit", 6 * 1024**3)),
                provider_options={k: v for k, v in options.items() if k != "device_id"},
            ),
            adaround=(result.get("scheme") or {}).get("adaround"),
            reference=reference,
        )
        entries.append(entry)
    if len(entries) != 266 or len({e["id"] for e in entries}) != 266:
        raise ValueError("Expected 38 x 7 unique conditions")
    selection = dict(
        schema_version=1,
        source_sha256=sha256(a.snapshot),
        expected_conditions=266,
        stages=[3, 32, 200],
        entries=entries,
        training_templates=catalog()["training_templates"],
        training_input_references=catalog().get("training_input_references", {}),
    )
    with tempfile.TemporaryDirectory() as temp:
        raw = Path(temp) / "selected.json"
        atomic_json(raw, selection)
        reference_output = a.output.with_suffix(".reference-metrics")
        migrate(
            selection=raw,
            bindings={k: str(v) for k, v in roots.items()},
            output=reference_output,
        )
        atomic_json(a.output, read_json(reference_output / "selected.json"))
        atomic_json(a.bindings, read_json(reference_output / "bindings.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "snapshot",
        "runs",
        "research",
        "data",
        "frozen",
        "output",
        "bindings",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    build(p.parse_args())
