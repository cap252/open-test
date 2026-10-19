"""Recollect complete calibration sets and compare numerical payloads bit for bit."""

import argparse
import copy
from dataclasses import fields
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np
import onnx
from onnx import numpy_helper

from opennpu_quant import CalibrationConfig, OrtConfig, calibrate
from opennpu_quant._io import atomic_json, object_hash, read_json, sha256
from opennpu_quant.quant.calibration import load_statistics, statistics_identity
from opennpu_quant.quant.engine import activation_params
from opennpu_quant.quant.parameters import verify_parameters
from opennpu_quant.quant.ranges import select_ranges
from opennpu_quant.quant.scales import activation_parameters
from opennpu_quant.reproduction.assets import Resolver, catalog, implementation_identity
from opennpu_quant.reproduction.data import dataset_for
from opennpu_quant.reproduction.environment import capture
from opennpu_quant.reproduction.execution import ReplayBundle

CHECKER_SHA256 = sha256(__file__)


def compare_arrays(expected, actual):
    """Preserve dtype, shape and signed zero; allclose is not bitwise equality."""
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    differences = []
    matched = 0
    for name in sorted(set(expected) & set(actual)):
        left, right = np.asarray(expected[name]), np.asarray(actual[name])
        reason = None
        if left.dtype != right.dtype or left.shape != right.shape:
            reason = "shape_or_dtype"
        elif not np.isfinite(left).all() or not np.isfinite(right).all():
            reason = "nonfinite"
        elif left.tobytes() != right.tobytes():
            reason = "bits"
        if reason:
            row = dict(name=name, reason=reason)
            if left.shape == right.shape and left.size and reason != "nonfinite":
                row["max_abs_error"] = float(
                    np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
                )
            differences.append(row)
        else:
            matched += 1
    return dict(
        state="pass" if not (missing or extra or differences) else "fail",
        compared=len(set(expected) & set(actual)),
        bitwise_equal=matched,
        missing=missing,
        extra=extra,
        differences=differences,
    )


def statistics_payload(stats):
    arrays = {}
    for name, limits in stats.minmax.items():
        for key, value in limits.items():
            arrays[f"{name}/minmax/{key}"] = np.asarray(value, np.float64)
    for name, row in stats.histograms.rows.items():
        for key, value in row.items():
            arrays[f"{name}/histogram/{key}"] = np.asarray(value)
    return arrays


def encoding_payload(ranges, scheme):
    arrays = {}
    for name, limits in ranges.items():
        symmetric = scheme["activation_symmetric"]
        scale, zero = activation_params(limits, symmetric)
        scale, zero, _ = activation_parameters(
            limits,
            scale,
            zero,
            symmetric,
            scheme.get("activation_scale_policy", "float"),
        )
        arrays[name + "/range"] = np.array(
            [limits["lowest"], limits["highest"]], np.float64
        )
        arrays[name + "/scale"] = np.asarray(scale)
        arrays[name + "/zero"] = np.asarray(zero)
    return arrays


def compare_encodings(entries, resolver, reference, actual):
    rows = []
    for entry in entries:
        if entry["condition"] == "fp32":
            continue
        audit = resolver.json(entry["audit"])
        model_path = resolver.check(entry["graphs"]["network"])
        verify_parameters(model_path, audit)
        scheme = audit["scheme"]
        config = CalibrationConfig(
            method=scheme["method"],
            percentile_value=scheme.get("percentile"),
            histogram_bins=reference.histogram_bins,
        )
        selected_ranges = []
        for stats in (reference, actual):
            selected_ranges.append(
                select_ranges(
                    stats.histograms,
                    stats.minmax,
                    config=config,
                    activation_symmetric=scheme["activation_symmetric"],
                )
            )
        old_ranges, new_ranges = selected_ranges
        graph = onnx.load(model_path)
        tensors = {v.name: numpy_helper.to_array(v) for v in graph.graph.initializer}
        aliases = {v: k for k, v in audit["raw_aliases"].items()}
        recorded = {}
        for node in graph.graph.node:
            if node.op_type != "QuantizeLinear":
                continue
            name = aliases.get(node.input[0], node.input[0])
            if name not in audit["quantized_tensors"] or name + "/scale" in recorded:
                raise ValueError("Unexpected or duplicate activation Q node: " + name)
            recorded[name + "/scale"] = tensors[node.input[1]]
            recorded[name + "/zero"] = tensors[node.input[2]]
        expected = encoding_payload(old_ranges, scheme)
        regenerated = encoding_payload(new_ranges, scheme)
        linkage = compare_arrays(
            recorded, {k: v for k, v in expected.items() if not k.endswith("/range")}
        )
        comparison = compare_arrays(
            recorded, {k: v for k, v in regenerated.items() if not k.endswith("/range")}
        )
        range_match = object_hash(old_ranges) == audit["range_identity"]
        fresh_range_match = object_hash(new_ranges) == audit["range_identity"]
        rows.append(
            dict(
                condition=entry["condition"],
                percentile=entry["percentile"],
                audit=entry["audit"],
                source_range_identity_matches=range_match,
                fresh_range_identity_matches=fresh_range_match,
                reference_to_original_qdq=linkage,
                fresh_to_original_qdq=comparison,
                fresh_to_reference=compare_arrays(expected, regenerated),
                state="pass"
                if range_match
                and fresh_range_match
                and linkage["state"] == comparison["state"] == "pass"
                else "fail",
            )
        )
    return rows


def run_model(row, resolver, output, device):
    started = time.monotonic()
    name = row["model"]
    folder = output / name
    folder.mkdir(parents=True, exist_ok=False)
    source = catalog()
    entries = [e for e in source["entries"] if e["model"] == name]
    entry = copy.deepcopy(next(e for e in entries if e["condition"] == "fp32"))
    reference_paths = {k: resolver.check(v) for k, v in row["reference"].items()}
    reference = load_statistics(reference_paths["statistics.json"].parent)
    if reference.statistics_identity != statistics_identity(
        reference.minmax, reference.histograms
    ):
        raise ValueError("Source statistics identity is inconsistent")
    network = resolver.check(entry["graphs"]["network"])
    training = source["training_input_references"][name]
    if (
        reference.model_identity != sha256(network)
        or reference.samples_identity != training["samples_identity"]
        or reference.sample_count != training["sample_count"]
    ):
        raise ValueError("Reference model or ordered input contract differs")
    if row.get("dataset_override"):
        entry["dataset"] = row["dataset_override"]
    dataset = dataset_for(entry, resolver, 1)
    before_data = dataset.identity("calibration", reference.sample_count)
    source_options = reference.environment["provider_options"]
    options = [dict(v) for v in source_options]
    options[0]["device_id"] = str(device)
    cfg = OrtConfig(
        providers=tuple(reference.environment["providers"]),
        provider_options=tuple(tuple(v.items()) for v in options),
        strict_versions=True,
    )
    env = capture()
    if env["version_mismatches"] or not env["gpu"]["available"]:
        raise RuntimeError("Required GPU software stack is unavailable")
    shapes = set()
    progress = dict(model=name, state="collecting", sample_count=reference.sample_count)
    atomic_json(folder / "progress.json", progress)
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        bundle = ReplayBundle(entry, resolver, tmp)
        passes = 0

        def feeds():
            nonlocal passes
            passes += 1
            for index, sample in enumerate(
                dataset.samples(
                    bundle, split="calibration", limit=reference.sample_count
                )
            ):
                shapes.add(
                    tuple((k, tuple(v.shape)) for k, v in sorted(sample.feeds.items()))
                )
                if index % 16 == 0:
                    progress.update(
                        pass_index=passes,
                        samples=index,
                        seconds=time.monotonic() - started,
                    )
                    atomic_json(folder / "progress.json", progress)
                    print(name, "pass", passes, "sample", index, flush=True)
                yield sample.feeds

        actual = calibrate(
            network,
            feeds,
            config=CalibrationConfig(histogram_bins=reference.histogram_bins),
            ort=cfg,
        )
    elapsed = time.monotonic() - started
    actual.histograms.save(folder / "histograms.npz")
    atomic_json(
        folder / "statistics.json",
        {
            f.name: getattr(actual, f.name)
            for f in fields(actual)
            if f.name != "histograms"
        },
    )
    invariants = dict(
        model=actual.model_identity == reference.model_identity,
        ordered_inputs=actual.samples_identity == reference.samples_identity,
        sample_count=actual.sample_count == reference.sample_count,
        bins=actual.histogram_bins == reference.histogram_bins,
        source_data_unchanged=before_data
        == dataset.identity("calibration", reference.sample_count),
    )
    statistics = compare_arrays(
        statistics_payload(reference), statistics_payload(actual)
    )
    encodings = compare_encodings(entries, resolver, reference, actual)
    result = dict(
        model=name,
        dataset=row["dataset"],
        state="pass"
        if all(invariants.values())
        and statistics["state"] == "pass"
        and all(r["state"] == "pass" for r in encodings)
        else "fail",
        source=row,
        source_graph=entry["graphs"]["network"],
        implementation=implementation_identity(),
        checker_sha256=CHECKER_SHA256,
        device=device,
        execution=cfg.to_dict(),
        environment=capture(libraries=True),
        reference_environment=reference.environment,
        sample_count=actual.sample_count,
        tensor_count=len(actual.minmax),
        input_shapes=sorted(shapes),
        collection_seconds=elapsed,
        invariants=invariants,
        reference_statistics_identity=reference.statistics_identity,
        statistics_identity=actual.statistics_identity,
        statistics_identity_equal=reference.statistics_identity
        == actual.statistics_identity,
        plan_identity_equal=reference.plan_identity == actual.plan_identity,
        protection_equal=object_hash(reference.protection)
        == object_hash(actual.protection),
        statistics_comparison=statistics,
        encodings=encodings,
        generated={
            k: sha256(folder / k)
            for k in ("statistics.json", "histograms.json", "histograms.npz")
        },
        statistics_cache_reused=False,
        new_qdq_model_generated=False,
        adaround_training="not_run",
    )
    atomic_json(folder / "result.json", result)
    progress.update(state=result["state"], seconds=time.monotonic() - started)
    atomic_json(folder / "progress.json", progress)
    print(
        name,
        result["state"],
        "seconds",
        elapsed,
        "statistics",
        statistics["state"],
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).resolve().parent
        / "validation"
        / "calibration-plan.json",
    )
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument(
        "--models", required=True, help="Comma-separated plan model names"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import cv2

    cv2.setNumThreads(1)
    plan = read_json(args.plan)
    models = args.models.split(",")
    indexed = {r["model"]: r for r in plan["models"]}
    if len(set(models)) != len(models) or set(models) - set(indexed):
        parser.error("Models must be unique names from the plan")
    resolver = Resolver(args.bindings)
    failed = False
    for model in models:
        try:
            row = run_model(indexed[model], resolver, args.output, args.device)
            failed |= row["state"] != "pass"
        except Exception:
            failed = True
            traceback.print_exc()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
