"""Re-score sealed full LFW/VOC predictions without inference or reference edits."""

import argparse
import hashlib
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from opennpu_quant._identity import code_identity
from opennpu_quant._io import object_hash, read_json, sha256
from opennpu_quant.evaluation.face_verification import (
    _fixed_angle_verification,
    verification,
)
from opennpu_quant.evaluation.metrics import SegmentationEvaluator
from opennpu_quant.reproduction.assets import (
    Resolver,
    catalog,
    checked_file,
    seal,
    unseal,
)
from opennpu_quant.reproduction.checks import arrays
from opennpu_quant.reproduction.execution import raw_digest
from opennpu_quant.reproduction.scoring import original_accuracy


if __package__:
    from .migrate_reference_metrics import canonicalize, exact
else:
    from migrate_reference_metrics import canonicalize, exact


MODELS = {"mobilefacenet_relu_foamliu", "deeplabv3_mnv2_voc"}


def lfw(entry, previous, folder, resolver, source):
    data = resolver.json(entry["dataset"]["samples"])
    pairs = data["pairs"]
    rows = data["validation"]
    if [s["id"] for s in previous["samples"]] != [r["image_id"] for r in rows]:
        raise ValueError("LFW sample order/count differs")
    embeddings = {}
    for marker, row in zip(previous["samples"], rows):
        saved = read_json(checked_file(folder, marker["prediction"]))
        if (saved["image_key"], saved["flip"]) != (row["image_key"], row["flip"]):
            raise ValueError("LFW image/view differs")
        value = np.asarray(saved["embedding"], np.float32)
        if raw_digest({"embedding": value[None]}) != marker["raw_digest"]:
            raise ValueError("Saved embedding differs from the GPU tensor digest")
        views = embeddings.setdefault(row["image_key"], {})
        if row["flip"] in views:
            raise ValueError("Duplicate LFW view")
        views[row["flip"]] = value
    metrics, details = verification(embeddings, pairs, dimensions=128)
    summed = {key: views[0] + views[1] for key, views in embeddings.items()}
    labels = np.array([p["same"] for p in pairs], bool)
    _, angles = _fixed_angle_verification(
        np.stack([summed[p["first"]] for p in pairs]),
        np.stack([summed[p["second"]] for p in pairs]),
        labels,
    )
    reference = source["result"]["details"]["fixed_angle_reference"]
    expected = np.asarray(reference["angles"], np.float64)
    actual = angles.astype(np.float64)
    check = arrays(actual, expected)
    if reference["threshold_degrees"] != details["fixed_angle_threshold_degrees"]:
        raise ValueError("Fixed-angle threshold differs")
    predicted = actual <= details["fixed_angle_threshold_degrees"]
    expected_predicted = expected <= reference["threshold_degrees"]
    return (
        metrics,
        details,
        dict(
            pairs=len(pairs),
            network_views=len(rows),
            embedding_tensor_digests_verified=True,
            angles=check,
            angles_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
            pair_decisions_equal=bool(np.array_equal(predicted, expected_predicted)),
            correct_pairs=int(np.count_nonzero(predicted == labels)),
            reference_evaluator_sha256=reference["source_sha256"],
            folds_unchanged=details["folds"] == previous["metric_details"]["folds"],
        ),
    )


def voc(entry, previous, folder, resolver):
    data = resolver.json(entry["dataset"]["manifest"])
    rows = data["validation"]
    if [s["id"] for s in previous["samples"]] != [r["id"] for r in rows]:
        raise ValueError("VOC sample order/count differs")
    root = resolver.path(entry["dataset"]["root"])
    evaluator = SegmentationEvaluator(output_name="labels")
    targets = {}
    for marker, row in zip(previous["samples"], rows):
        saved = read_json(checked_file(folder, marker["prediction"]))
        target = (root / row["target"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Target escapes dataset root")
        targets[row["id"]] = sha256(target)
        with Image.open(target) as image:
            truth = np.asarray(image)
        evaluator.update(
            {"labels": np.asarray(saved["labels"])}, SimpleNamespace(target=truth)
        )
    metrics = evaluator.finalize()
    return (
        metrics,
        evaluator.metric_details,
        dict(images=len(rows), target_files_sha256=object_hash(targets)),
    )


def run(run_dir, bindings, output, *, migrate_metadata=False):
    if output.exists():
        raise FileExistsError(output)
    entries = {e["id"]: e for e in catalog()["entries"] if e["model"] in MODELS}
    resolver = Resolver(bindings)
    summary = unseal(run_dir / "run.json")
    results = []
    for record in summary["conditions"]:
        if record["id"] not in entries or record["state"] != "completed":
            continue
        entry = entries[record["id"]]
        path = checked_file(run_dir, record["result"])
        previous = unseal(path)
        if previous["stages"]["full_accuracy"] != "executed":
            raise ValueError("This check requires a full evaluation")
        measurement = previous["measurement"]
        migrations = []
        if migrate_metadata:
            measurement = canonicalize(measurement, migrations)
        if (
            exact(measurement) != exact(entry["measurement"])
            or previous["source_graphs"] != entry["graphs"]
            or previous["recipe_identity"] != object_hash(entry["recipe"])
        ):
            raise ValueError(
                "Run and selected reference identities differ. "
                "For a verified name-only reference migration, use --migrate-metadata."
            )
        source = resolver.json(entry["source_result"])
        if source["result"]["metrics"] != entry["measurement"]["metrics"]:
            raise ValueError("Selected metrics differ from the source result")
        print("Re-scoring", entry["id"], flush=True)
        if entry["model"] == "mobilefacenet_relu_foamliu":
            metrics, details, diagnostics = lfw(
                entry, previous, path.parent, resolver, source
            )
        else:
            metrics, details, diagnostics = voc(entry, previous, path.parent, resolver)
        candidate = dict(
            previous, measurement=measurement, metrics=metrics, metric_details=details
        )
        comparison = original_accuracy(candidate, source, path.parent)
        primary = previous["primary_metric"]
        primary_check = arrays(
            np.array([metrics[primary]], np.float64),
            np.array([previous["metrics"][primary]], np.float64),
        )
        result = dict(
            id=entry["id"],
            previous_result_sha256=sha256(path),
            metadata_migrations=migrations,
            previous_implementation=previous["environment"]["implementation"],
            source_result=entry["source_result"],
            recipe_identity=previous["recipe_identity"],
            input_identity=previous["input_identity"],
            metrics=metrics,
            metric_details=details,
            primary_metric_unchanged=primary_check["bitwise_equal"],
            original_accuracy=comparison,
            diagnostics=diagnostics,
        )
        resolver.check(entry["source_result"])
        checked_file(run_dir, record["result"])
        results.append(result)
        print(
            entry["id"],
            comparison["metrics_state"],
            comparison["metrics_bitwise_equal"],
            flush=True,
        )
    if not results:
        raise ValueError("No completed LFW/VOC evaluations in this run")
    passed = all(
        r["original_accuracy"]["metrics_state"] == "pass"
        and r["original_accuracy"]["metrics_bitwise_equal"]
        and r["primary_metric_unchanged"]
        and r["diagnostics"].get("angles", {}).get("bitwise_equal", True)
        and r["diagnostics"].get("pair_decisions_equal", True)
        and r["diagnostics"].get("folds_unchanged", True)
        for r in results
    )
    evidence = dict(
        schema_version=1,
        scope="CPU metric re-scoring of saved full GPU predictions, no new inference",
        implementation=code_identity(),
        selection_sha256=sha256(
            files("opennpu_quant.resources.reproduction") / "selected.json"
        ),
        tool_sha256=sha256(__file__),
        run_sha256=sha256(run_dir / "run.json"),
        conditions=results,
        passed=passed,
        gpu_execution="not_run",
        calibration="not_run",
        retraining="not_run",
    )
    seal(output, evidence)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--migrate-metadata",
        action="store_true",
        help="Explicitly convert saved reference metric names in memory for re-scoring. Does not edit or resume the saved run.",
    )
    args = parser.parse_args()
    if not run(
        args.run, args.bindings, args.output, migrate_metadata=args.migrate_metadata
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
