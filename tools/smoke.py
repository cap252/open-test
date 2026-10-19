#!/usr/bin/env python3
"""Sequential CPU functional checks of the cnn38_smoke preset, one run per model.

Optional assets JSON maps model names to original weights/source/python paths.
Prepared dataset manifests belong under --home/data/<dataset>/manifest.json; LFW
needs the fixed pair subset (opennpu-quant data prepare lfw_mobilefacenet --smoke).
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from opennpu_quant._io import atomic_json, read_json
from opennpu_quant.configuration import load_config, validate
from opennpu_quant.models.bundle import ModelBundle
from opennpu_quant.models.spec import ModelSpec
from opennpu_quant.paths import Paths

PRESET = "cnn38"
# One model per task; --representatives adds float and symmetric PoT for these.
REPRESENTATIVES = {
    "efficientnet_lite0",
    "yolov9_t",
    "yolov7_face",
    "posenet_101",
    "deeplabv3_r50_voc",
    "yolov5s_seg",
    "mobilefacenet_relu_foamliu",
}


def dynamic_samples(name, paths):
    """Make a smoke-only two-image split with two actual network input shapes."""
    from PIL import Image
    from opennpu_quant.models.preprocess import preprocess

    recipe = ModelSpec.load(name).recipe
    if recipe["task"] == "face_verification":
        manifest = read_json(paths.data / recipe["dataset"] / "manifest.json")
        if not manifest.get("smoke"):
            raise ValueError(
                "Prepare the LFW smoke subset: opennpu-quant data prepare "
                + recipe["dataset"]
                + " --smoke"
            )
        return paths.data, {}
    if all(isinstance(d, int) for d in recipe["input"]["shape"]):
        return paths.data, {}
    dataset = recipe["dataset"]
    manifest = read_json(paths.data / dataset / "manifest.json")
    shapes = {}
    for split in ("calibration", "validation"):
        chosen, observed = [], []
        for row in manifest[split]:
            if split == "validation" and dataset == "widerface":
                difficulty = recipe["metric"].removeprefix("AP_")
                if not row["truth"]["keep"][difficulty]:
                    continue
            with Image.open(Path(manifest["root"]) / row["path"]) as image:
                value, _ = preprocess(
                    image, recipe["preprocess"], recipe["input"]["layout"]
                )
            shape = list(value.shape)
            if shape not in observed:
                chosen.append(row)
                observed.append(shape)
            if len(chosen) == 2:
                break
        if len(chosen) != 2:
            raise ValueError(
                "Dynamic smoke requires two distinct input shapes: "
                + name
                + "/"
                + split
            )
        # A distinct smoke manifest prevents accidental use of this subset for a full run.
        manifest[split] = chosen
        shapes[split] = observed
    destination = paths.home / "smoke_data" / name
    manifest["source"] = str(manifest.get("source", "folder")) + ":smoke"
    manifest["reference_bytes_verified"] = False
    atomic_json(destination / dataset / "manifest.json", manifest)
    return destination, shapes


def qdq_audit(directory):
    """Detailed scale audit, runtime audit and Conv/Gemm weight coverage of one QDQ."""
    audit = read_json(directory / "quantization.json")
    runtime = read_json(directory / "result.json")["runtime_audit"]
    unquantized = [
        row["name"]
        for row in audit["operator_coverage"]
        if row["op_type"] in ("Conv", "Gemm")
        and 1 not in row["quantized_weight_inputs"]
    ]
    if not (
        audit["passed"]
        and audit.get("scale_audit") is not None
        and runtime["passed"]
        and runtime["parameter_verification"]["passed"]
        and not unquantized
    ):
        raise ValueError("QDQ audit failed: " + str(directory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("workspace") / PRESET)
    parser.add_argument(
        "--assets", type=Path, help="Optional JSON: model -> weights, source, python"
    )
    parser.add_argument("--models", nargs="+", help="Subset; defaults to all 38 models")
    parser.add_argument(
        "--stage", choices=["export", "run", "all", "report"], default="all"
    )
    parser.add_argument(
        "--reuse-export",
        action="store_true",
        help="Reuse content-verified model bundles",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute run results after implementation changes",
    )
    parser.add_argument(
        "--representatives",
        action="store_true",
        help="Also test float and symmetric PoT on task representatives",
    )
    a = parser.parse_args()
    a.home = a.home.resolve()
    paths = Paths.resolve(home=a.home)
    logs = a.home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    assets = read_json(a.assets) if a.assets else {}
    base = load_config(PRESET + "_smoke")
    full = load_config(PRESET)
    names = a.models or base["models"]
    unknown = set(names) - set(base["models"])
    if unknown:
        parser.error("Models outside preset: " + ", ".join(sorted(unknown)))
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    summary_path = a.home / "smoke_results.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    failures = 0
    for name in names:
        summary = read_json(summary_path) if summary_path.exists() else {}
        row = summary.setdefault(name, {})
        row["full_evaluation"] = False
        started = time.monotonic()
        asset = assets.get(name, {})
        print(name + ": " + a.stage, flush=True)
        try:
            if a.stage in ("export", "all"):
                reused = False
                if a.reuse_export and (paths.models / name / "model.json").exists():
                    ModelBundle(ModelSpec.load(name), paths.models / name)
                    reused = True
                if not reused:
                    cmd = [
                        sys.executable,
                        "-m",
                        "opennpu_quant",
                        "export",
                        name,
                        "--home",
                        str(a.home),
                        "--validation",
                        "functional",
                    ]
                    for key, option in [
                        ("weights", "--weights"),
                        ("source", "--source-dir"),
                        ("python", "--python"),
                    ]:
                        if asset.get(key):
                            cmd += [option, str(asset[key])]
                    with (logs / (name + "_export.log")).open("w") as log:
                        subprocess.run(
                            cmd,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                proof = read_json(paths.models / name / "model.json")
                row.update(
                    export_passed=True,
                    run_passed=False,
                    export_numerical_parity=proof["validation"][
                        "framework_parity_passed"
                    ],
                    export_validation=proof["validation"],
                )
            if a.stage in ("run", "all"):
                data_dir, input_shapes = dynamic_samples(name, paths)
                row["input_shapes"] = input_shapes
                config = load_config(PRESET + "_smoke")
                config["name"] = PRESET + "_smoke_" + name
                config["models"] = [name]
                if a.representatives and name in REPRESENTATIVES:
                    config["conditions"] = full["conditions"]
                validate(config)
                config_path = a.home / "configs" / (name + ".json")
                atomic_json(config_path, config)
                cmd = [
                    sys.executable,
                    "-m",
                    "opennpu_quant",
                    "run",
                    "--config",
                    str(config_path),
                    "--home",
                    str(a.home),
                ]
                cmd += ["--data-dir", str(data_dir)]
                if a.force:
                    cmd.append("--force")
                with (logs / (name + "_run.log")).open("w") as log:
                    subprocess.run(
                        cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
            if a.stage in ("run", "all", "report"):
                from opennpu_quant.report import report
                from opennpu_quant.graph.model import model_identity

                bundle = ModelBundle(ModelSpec.load(name), paths.models / name)
                graphs = {
                    key: model_identity(path) for key, path in bundle.paths.items()
                }
                proof = read_json(paths.models / name / "model.json")
                row.update(
                    export_passed=True,
                    export_numerical_parity=proof["validation"][
                        "framework_parity_passed"
                    ],
                    export_validation=proof["validation"],
                )
                directory = paths.runs / (PRESET + "_smoke_" + name)
                result = report(directory)
                expected = set(
                    full["conditions"]
                    if a.representatives and name in REPRESENTATIVES
                    else base["conditions"]
                )
                completed = {
                    r["condition"] for r in result["rows"] if r["state"] == "completed"
                }
                if not expected <= completed:
                    raise ValueError(
                        "Missing verified smoke conditions: "
                        + ", ".join(sorted(expected - completed))
                    )
                mismatches = []
                audited = 0
                for condition in completed:
                    measured = read_json(directory / name / condition / "result.json")
                    if measured["graph_identities"] != graphs:
                        raise ValueError(
                            "Smoke result belongs to a different exported bundle: "
                            + condition
                        )
                    if measured.get("full_evaluation"):
                        raise ValueError("Smoke result claims a full evaluation")
                    if measured.get("scheme"):
                        qdq_audit(directory / name / condition)
                        audited += 1
                    for layer in (measured.get("adaround") or {}).get("layers", []):
                        mismatches.extend(
                            dict(condition=condition, **entry)
                            for entry in layer.get("parity_mismatches", [])
                        )
                row.update(
                    run_passed=True,
                    qdq_audited_conditions=audited,
                    conditions=result["rows"],
                    adaround_numerical_parity=not mismatches,
                    adaround_parity_mismatches=mismatches,
                    full_evaluation=False,
                )
            row["numerical_parity"] = bool(
                row.get("export_numerical_parity")
                and row.get("adaround_numerical_parity", True)
            )
            row.update(error=None, elapsed_s=time.monotonic() - started)
        except Exception as error:
            failures += 1
            row.update(error=str(error), elapsed_s=time.monotonic() - started)
            if a.stage == "export":
                row["export_passed"] = False
            else:
                row["run_passed"] = False
            print(name + ": FAILED: " + str(error), flush=True)
        row["functional_passed"] = bool(
            row.get("export_passed") and row.get("run_passed") and not row["error"]
        )
        atomic_json(summary_path, summary)
        with (a.home / "smoke_results.csv").open("w", newline="") as stream:
            fields = [
                "model",
                "export_passed",
                "run_passed",
                "functional_passed",
                "numerical_parity",
                "export_numerical_parity",
                "adaround_numerical_parity",
                "qdq_audited_conditions",
                "full_evaluation",
                "error",
            ]
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for model, entry in summary.items():
                writer.writerow(
                    dict(model=model, **{k: entry.get(k) for k in fields[1:]})
                )
    print(
        json.dumps(
            {"models": len(names), "failures": failures, "summary": str(summary_path)}
        ),
        flush=True,
    )
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
