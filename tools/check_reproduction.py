"""Check frozen tensors or frozen ONNX using explicit external asset bindings."""

import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
from opennpu_quant._io import read_json, atomic_json, sha256
from opennpu_quant.models.spec import ModelSpec
from opennpu_quant.models import evaluation_profiles as profiles
from opennpu_quant.models.preprocess import preprocess
from opennpu_quant.ort.config import OrtConfig
from opennpu_quant.ort.session import create_session, environment
from opennpu_quant.ort.cuda_libs import loaded_libraries
from reproduction_checks import arrays, predictions
from opennpu_quant.reproduction.execution import (
    load_arrays as load_values,
    repeat_outputs,
)


def checked(root, relative, digest):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Missing/unsafe artifact: " + str(relative))
    if sha256(path) != digest:
        raise ValueError("Artifact changed: " + str(relative))
    return path


def current_decode(full, recipe, meta, categories):
    from opennpu_quant.models.decode import yolo
    from opennpu_quant.models.extended_decode import detection, yolo_extra

    cfg = recipe["decode"]
    if cfg["kind"] == "yolo":
        return yolo(
            full[cfg["output"]],
            meta,
            categories,
            dict(cfg, nms_iou_threshold=cfg["nms"]["iou"]),
        )
    if cfg["kind"] in ("yolo_face", "yolo_seg"):
        return yolo_extra(full, cfg, meta, categories)
    return detection(full, cfg, meta, categories)


def original(a):
    """Import the independently pinned research functions only in this validation tool."""
    if not all((a.research_run, a.research_code, a.datasets)):
        raise ValueError(
            "Postprocessing comparison requires --research-run, --research-code and --datasets"
        )
    proof = read_json(a.package.parent / ("protocol_" + a.model + "_200.json"))
    target = a.research_code / "examples/models/cnn33_new/protocols.py"
    if proof["source_hashes"].get(str(target)) != sha256(target):
        raise ValueError("Research protocol source differs from measured binding")
    sys.path.extend(
        [
            str(a.research_code / "src"),
            str(a.research_code),
            str(a.research_run / "python_deps"),
        ]
    )
    from examples.models.cnn33_new import protocols, sources

    source = sources.source_path(
        a.research_run, a.model
    )  # verifies original .py file hashes
    # Upstream optional user config must never be written into the reference run.
    os.environ["YOLOV5_CONFIG_DIR"] = str(a.output.parent / "upstream_config")
    source_proof = read_json(source.with_suffix(".json"))
    return protocols, dict(protocol_sha256=sha256(target), source=source_proof)


def run(a):
    import torch

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = a.package.resolve()
    seal = read_json(base / "PACKAGE_READY.json")
    checked(base, "manifest.json", seal["manifest_sha256"])
    manifest = read_json(base / "manifest.json")
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    selected = [
        g
        for g in manifest["graphs"]
        if (not a.model or g["model"] == a.model)
        and (not a.graph or g["id"] == a.graph)
    ]
    if a.mode == "postprocess":
        selected = [g for g in selected if g["condition"] == "fp32"]
    if not selected:
        raise ValueError("No matching frozen graph")
    implementation = Path(profiles.__file__).parent
    result = dict(
        checker_sha256=sha256(Path(__file__)),
        comparison_sha256=sha256(Path(__file__).with_name("reproduction_checks.py")),
        implementation_hashes={
            name: sha256(implementation / name)
            for name in (
                "historical_yolo.py",
                "evaluation_profiles.py",
                "decode.py",
                "extended_decode.py",
                "extended_preprocess.py",
            )
        },
        mode=a.mode,
        state="running",
        full_accuracy_reproduction=False,
        retraining=False,
        rtx_200="pending",
        package_manifest_sha256=seal["manifest_sha256"],
        graphs=[],
    )
    original_functions = bindings = dataset_rows = None
    if a.mode == "postprocess":
        original_functions, bindings = original(a)
        result["research_source_binding"] = bindings
        dataset = read_json(a.datasets)
        split = (
            "widerface_validation" if a.model == "yolov7_face" else "coco_validation"
        )
        dataset_rows = {str(r["image_id"]): r for r in dataset["splits"][split]}
    cfg = (
        OrtConfig.cpu(strict_versions=True)
        if a.device == "cpu"
        else OrtConfig.cuda(int(a.device.split(":")[1]), strict_versions=True)
    )
    for entry in selected:
        for rel, digest in entry["files"].items():
            checked(base, rel, digest)
        ref_path = base / "expected" / entry["id"] / "manifest.json"
        sealed = read_json(base / "CONTENTS-SHA256.json")
        # Package seal covers expected manifests as well as graph/input payloads.
        checked(
            base,
            str(ref_path.relative_to(base)),
            sealed[str(ref_path.relative_to(base))],
        )
        refs = read_json(ref_path)
        if refs["model_sha256"] != entry["files"][entry["network"]]:
            raise ValueError("Reference is bound to a different model")
        post = (
            create_session(base / entry["postprocess"], OrtConfig.cpu())
            if entry.get("postprocess")
            else None
        )
        record = dict(
            graph=entry["id"],
            model=entry["model"],
            state="running",
            samples=[],
            profile=a.profile,
        )
        result["graphs"].append(record)
        predictions_path = a.output.with_name(
            a.output.stem + "_" + entry["id"] + ".predictions.jsonl.gz"
        )
        with (
            tempfile.TemporaryDirectory(dir=a.output.parent) as temp,
            gzip.open(predictions_path, "wt") as stream,
        ):
            network = (
                create_session(
                    base / entry["network"], cfg, profile=Path(temp) / "placement"
                )
                if a.mode == "frozen"
                else None
            )
            if network and entry["condition"] != "fp32":
                from opennpu_quant.quant.parameters import verify_parameters
                from opennpu_quant.quant.engine import audit_runtime

                audit_rel = (
                    "evidence/"
                    + entry["model"]
                    + "/"
                    + entry["condition"]
                    + ".audit.json.gz"
                )
                audit_path = checked(base, audit_rel, sealed[audit_rel])
                with gzip.open(audit_path, "rt") as audit_file:
                    audit = json.load(audit_file)
                record["parameter_audit"] = verify_parameters(
                    base / entry["network"], audit
                )
                executed = Path(temp) / "executed.onnx"
                optimized_session = create_session(
                    base / entry["network"], cfg, optimized=executed
                )
                del optimized_session
                record["runtime_audit"] = audit_runtime(
                    base / entry["network"], executed, audit
                )
                record["detailed_scale_audit_available"] = (
                    audit.get("scale_audit") is not None
                )
            recipe = ModelSpec.load(entry["model"]).recipe
            if a.mode == "postprocess":
                recipe = profiles.apply_profile(recipe, a.profile)
            for index, sample in enumerate(entry["samples"][: a.limit]):
                stored = refs["samples"][index]
                if (
                    sample["id"] != stored["id"]
                    or sample["inputs_sha256"] != stored["inputs_sha256"]
                ):
                    raise ValueError("Reference sample/order changed")
                feeds = load_values(
                    checked(base, sample["inputs"], sample["inputs_sha256"])
                )
                expected = load_values(
                    checked(base, stored["outputs"], stored["outputs_sha256"])
                )
                values = {
                    n: expected["output_" + str(i)]
                    for i, n in enumerate(refs["output_names"])
                }
                row = dict(id=sample["id"])
                if network:
                    names = [x.name for x in network.get_outputs()]
                    if names != refs["output_names"]:
                        raise ValueError("Output names/order changed")
                    observed, row["repeat"] = repeat_outputs(network, names, feeds)
                    row["raw"] = {n: arrays(observed[n], values[n]) for n in names}
                    # The package expected decoder is NOT a research-protocol oracle.
                    row["reference_kind"] = "package GTX raw outputs"
                else:
                    context = {**feeds, **values}
                    full = (
                        dict(
                            zip(
                                [x.name for x in post.get_outputs()],
                                post.run(
                                    None,
                                    {
                                        x.name: context[x.name]
                                        for x in post.get_inputs()
                                    },
                                ),
                            )
                        )
                        if post
                        else values
                    )
                    if "prediction" in full:
                        full["predictions"] = full["prediction"]
                    raw = dict(
                        full, prediction=full.get("prediction", full.get("predictions"))
                    )
                    data_row = dataset_rows[sample["id"]]
                    if sha256(data_row["path"]) != sample["image_sha256"]:
                        raise ValueError("Original image changed")
                    original_feeds, original_meta, _ = original_functions.preprocess(
                        a.research_run, a.model, data_row
                    )
                    from PIL import Image

                    with Image.open(data_row["path"]) as image:
                        tensor, meta = preprocess(image, recipe["preprocess"])
                    row["input"] = arrays(tensor, original_feeds["images"])
                    row["frozen_input"] = arrays(tensor, feeds["images"])
                    native = original_functions.postprocess(
                        a.research_run,
                        a.model,
                        raw,
                        original_meta,
                        manifest["categories"],
                    )
                    before = current_decode(full, recipe, meta, manifest["categories"])
                    after = (
                        before
                        if a.profile == profiles.CURRENT
                        else profiles.decode(full, recipe, meta, manifest["categories"])
                    )
                    row["before"] = predictions(before, native)
                    row["after"] = predictions(after, native)
                    stream.write(
                        json.dumps(
                            dict(
                                id=sample["id"],
                                research=native,
                                current=before,
                                profile=after,
                            )
                        )
                        + "\n"
                    )
                record["samples"].append(row)
                if (index + 1) in (3, 32, 200):
                    atomic_json(a.output, result)
                    print(entry["id"], index + 1, flush=True)
            if network:
                from opennpu_quant.ort.profile import (
                    summarize_profile,
                    assert_cuda_observed,
                )

                record["placement"] = summarize_profile(
                    read_json(network.end_profiling())
                )
                if a.device != "cpu":
                    assert_cuda_observed(network.get_providers(), record["placement"])
            record["state"] = "completed"
        record["prediction_stream_sha256"] = sha256(predictions_path)
        if len(record["samples"]) != a.limit:
            raise ValueError("Incomplete fixed sample coverage")
        atomic_json(a.output, result)
    if a.mode == "postprocess":
        result["postprocessing_parity_passed"] = all(
            r["after"]["passed"]
            and r["input"]["bitwise_equal"]
            and r["frozen_input"]["bitwise_equal"]
            for g in result["graphs"]
            for r in g["samples"]
        )
    else:
        result["numeric_passed"] = all(
            c["numeric_passed"]
            for g in result["graphs"]
            for r in g["samples"]
            for c in r["raw"].values()
        )
        result["repeat_bitwise_equal"] = all(
            c["bitwise_equal"]
            for g in result["graphs"]
            for r in g["samples"]
            for c in r["repeat"].values()
        )
    result.update(
        state="completed",
        environment=environment(cfg),
        loaded_libraries=loaded_libraries(),
    )
    atomic_json(a.output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--mode", choices=("postprocess", "frozen"), required=True)
    p.add_argument("--model")
    p.add_argument("--graph")
    p.add_argument("--profile", choices=profiles.PROFILES)
    p.add_argument("--limit", type=int, choices=(3, 32, 200), default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--research-run", type=Path)
    p.add_argument("--research-code", type=Path)
    p.add_argument("--datasets", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "postprocess":
        if a.model not in profiles.MODELS:
            p.error(
                "Postprocessing requires one of: " + ", ".join(sorted(profiles.MODELS))
            )
        a.profile = a.profile or profiles.HISTORICAL
        profiles.validate_profile(a.model, a.profile)
    elif a.profile is not None:
        p.error(
            "--profile applies only to postprocess; frozen mode compares raw tensors"
        )
    if a.output.exists() or a.output.with_suffix(".failure.json").exists():
        p.error("Output already exists; use a new output path")
    try:
        result = run(a)
        if (
            a.mode == "postprocess"
            and a.profile == profiles.HISTORICAL
            and not result["postprocessing_parity_passed"]
        ):
            raise SystemExit(1)
        if a.mode == "frozen" and not (
            result["numeric_passed"] and result["repeat_bitwise_equal"]
        ):
            raise SystemExit(1)
    except Exception as e:
        atomic_json(
            a.output.with_suffix(".failure.json"), dict(state="failed", error=repr(e))
        )
        raise


if __name__ == "__main__":
    main()
