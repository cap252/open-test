"""Sequential fixed-graph inference with verified condition/sample resume."""

from pathlib import Path
import tempfile
import hashlib
import numpy as np

from .._io import atomic_json, read_json, object_hash
from .._locking import RunLock
from .._feeds import feed_digest
from ..models.bundle import ModelBundle
from ..models.spec import ModelSpec
from ..models.adapter import ModelEvaluator
from ..ort.config import OrtConfig
from ..ort.session import create_session
from ..ort.profile import profile_sample
from .assets import (
    Resolver,
    selected,
    unseal,
    seal,
    implementation_identity,
    file_record,
    checked_file,
)
from .data import dataset_for, data_identity
from .checks import arrays
from .environment import capture


class ReplayBundle(ModelBundle):
    def __init__(self, entry, resolver, directory):
        super().__init__(
            ModelSpec(entry["recipe"]),
            directory,
            paths={k: resolver.check(v) for k, v in entry["graphs"].items()},
        )
        self.aliases = entry.get("output_aliases", {})

    def finish(self, outputs, metadata):
        result = super().finish(outputs, metadata)
        return {**result, **{k: result[v] for k, v in self.aliases.items()}}


def load_arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {k: source[k] for k in source.files}


def verify_complete(folder):
    result = unseal(folder / "result.json")
    for sample in result["samples"]:
        if sample.get("raw"):
            checked_file(folder, sample["raw"])
        elif not sample.get("raw_digest"):
            raise ValueError("Missing raw evidence")
        checked_file(folder, sample["prediction"])
    return result


def _reference(entry, resolver):
    ref = entry.get("reference")
    if not ref:
        return None
    try:
        manifest = resolver.json(ref["manifest"])
    except FileNotFoundError:
        return None
    if manifest["model_sha256"] != entry["graphs"]["network"]["sha256"]:
        raise ValueError("Reference model identity differs")
    return ref, manifest


def _condition(prepared, resolver, folder, limit, cfg, env, identity, storage="raw"):
    entry = prepared["entry"]
    if object_hash([entry, prepared["data_identity"], limit]) != prepared["identity"]:
        raise ValueError("Prepared model/profile identity is inconsistent")
    dataset = dataset_for(entry, resolver, limit)
    if data_identity(dataset, limit) != prepared["data_identity"]:
        raise ValueError("Dataset/order changed since prepare")
    resolver.check(entry["source_result"])
    bundle = ReplayBundle(entry, resolver, folder)
    audit = resolver.json(entry["audit"]) if entry.get("audit") else None
    # Validate every source even when a completed condition can be reused.
    if (folder / "result.json").exists():
        result = verify_complete(folder)
        if result["identity"] != identity:
            raise ValueError(
                "Completed result identity differs; use a new output directory"
            )
        return result
    folder.mkdir(parents=True, exist_ok=True)
    settings = folder / "identity.json"
    if settings.exists() and read_json(settings)["identity"] != identity:
        raise ValueError("Interrupted run identity differs")
    atomic_json(settings, dict(identity=identity))
    evaluator = ModelEvaluator(bundle, dataset)
    reference = _reference(entry, resolver)
    result = dict(
        id=entry["id"],
        model=entry["model"],
        task=entry["recipe"]["task"],
        condition=entry["condition"],
        identity=identity,
        prepared_identity=prepared["identity"],
        state="running",
        samples=[],
        environment=env,
        measurement=entry["measurement"],
        profile=entry["recipe"]["decode"].get("profile", "current_v1"),
        evaluation_contract=evaluator.contract.identity(),
        recipe_identity=object_hash(entry["recipe"]),
        primary_metric=entry["recipe"]["metric"],
        source_graphs=entry["graphs"],
        input_identity=prepared["data_identity"],
        execution=cfg.to_dict(),
        percentile=entry.get("percentile"),
        limit=limit,
        unit="pairs" if "evaluation_pairs" in dataset.info else "images",
        storage=storage,
    )
    with tempfile.TemporaryDirectory(dir=folder) as temp:
        executed = Path(temp) / "executed.onnx"
        session = create_session(
            bundle.paths["network"], cfg, optimized=executed if audit else None
        )
        result["actual_provider_options"] = session.get_provider_options()
        if audit:
            from ..quant.parameters import verify_parameters
            from ..quant.engine import audit_runtime

            result["parameter_audit"] = verify_parameters(
                bundle.paths["network"], audit
            )
            result["runtime_audit"] = audit_runtime(
                bundle.paths["network"], executed, audit
            )
        names = [v.name for v in session.get_outputs()]
        result["output_names"] = names
        shapes = set()
        for index, sample in enumerate(dataset.samples(bundle, limit=limit)):
            digest = feed_digest(sample.feeds)
            shapes.add(
                tuple((k, tuple(v.shape)) for k, v in sorted(sample.feeds.items()))
            )
            for value in sample.feeds.values():
                if (
                    value.dtype != np.float32
                    or value.shape[0] != 1
                    or not np.isfinite(value).all()
                ):
                    raise ValueError("Expected finite FP32 batch-one feeds")
            marker = folder / f"{index:06d}.json"
            if marker.exists():
                record = unseal(marker)
                if (
                    record["input_digest"] != digest
                    or record["id"] != sample.sample_id
                    or record["identity"] != identity
                ):
                    raise ValueError("Cached sample input/profile changed")
                if record.get("raw"):
                    values = load_arrays(checked_file(folder, record["raw"]))
                else:
                    values, repeat = repeat_outputs(session, names, sample.feeds)
                    if raw_digest(values) != record.get("raw_digest") or not all(
                        r["bitwise_equal"] for r in repeat.values()
                    ):
                        raise ValueError("Recomputed compact raw output differs")
                checked_file(folder, record["prediction"])
                if set(values) != set(names):
                    raise ValueError("Cached output names changed")
                evaluator.update(values, sample)
                if (
                    read_json(folder / record["prediction"]["path"])
                    != evaluator.prediction
                ):
                    raise ValueError("Cached postprocessing differs")
            else:
                values, repeat = repeat_outputs(session, names, sample.feeds)
                raw_path = folder / f"{index:06d}.npz"
                if storage == "raw":
                    with raw_path.with_suffix(".pending").open("wb") as stream:
                        np.savez(stream, **values)
                    raw_path.with_suffix(".pending").replace(raw_path)
                evaluator.update(values, sample)
                prediction_path = folder / f"{index:06d}.prediction.json"
                atomic_json(prediction_path, evaluator.prediction)
                record = dict(
                    id=sample.sample_id,
                    identity=identity,
                    input_digest=digest,
                    input_shapes={k: list(v.shape) for k, v in sample.feeds.items()},
                    raw=file_record(raw_path, folder) if storage == "raw" else None,
                    raw_digest=raw_digest(values),
                    prediction=file_record(prediction_path, folder),
                    repeat=repeat,
                    reference_state="pending",
                )
                record.update(
                    reference_diagnostic(
                        reference,
                        resolver,
                        sample.sample_id,
                        index,
                        digest,
                        values,
                        names,
                    )
                )
                seal(marker, record)
            if index == 0:
                result["placement"] = profile_sample(
                    bundle.paths["network"], sample.feeds, values, cfg
                )
            result["samples"].append(record)
            atomic_json(
                folder / "progress.json",
                dict(state="running", completed_samples=index + 1, identity=identity),
            )
        if data_identity(dataset, limit) != prepared["data_identity"]:
            raise ValueError("Dataset changed during execution")
        if not result["samples"]:
            raise ValueError("No evaluation samples")
        result["metrics"] = evaluator.finalize()
        if any(not np.isfinite(v) for v in result["metrics"].values()):
            raise ValueError("Nonfinite metric")
        if "smoke_pair_accuracy" in result["metrics"]:
            result["primary_metric"] = "smoke_pair_accuracy"
        result["metric_details"] = evaluator.metric_details
        result["metric_units"] = {k: "fraction" for k in result["metrics"]}
        result["sample_count"] = len(result["samples"])
        result["input_shapes"] = sorted(shapes)
        dynamic = any(not isinstance(d, int) for d in entry["recipe"]["input"]["shape"])
        result["dynamic_shape_coverage"] = (
            "pass" if not dynamic or len(shapes) > 1 else "pending"
        )
        result["repeat_bitwise"] = all(
            v["bitwise_equal"] for s in result["samples"] for v in s["repeat"].values()
        )
        result["state"] = "completed" if result["repeat_bitwise"] else "fail"
        result["environment"] = dict(
            capture(libraries=env["device"].startswith("cuda")),
            device=env["device"],
            execution=env["execution"],
            implementation=env["implementation"],
            device_identity=env.get("device_identity"),
        )
        expected_units = (
            6000 if result["unit"] == "pairs" else dataset.info.get("expected_samples")
        )
        result["stages"] = dict(
            materials="pass",
            postprocessing="executed",
            gpu_execution="pass" if env["device"].startswith("cuda") else "pending",
            gpu_comparison="pending",
            full_accuracy="executed" if limit == expected_units else "pending",
            retraining="not_run",
        )
        from .scoring import original_accuracy

        result["original_accuracy"] = original_accuracy(
            result, resolver.json(entry["source_result"]), folder
        )
        # A complete sample subset is not a claim about the original full evaluation.
        seal(folder / "result.json", result)
    return result


def run(
    *,
    package,
    bindings,
    output,
    device="cpu",
    models=None,
    conditions=None,
    storage="raw",
):
    from .environment import probe_gpu

    if storage not in ("raw", "predictions"):
        raise ValueError("storage must be raw or predictions")
    package, output = Path(package), Path(output)
    prepared = unseal(package / "manifest.json")
    chosen = selected([r["entry"] for r in prepared["conditions"]], models, conditions)
    selected_ids = {e["id"] for e in chosen}
    records = [r for r in prepared["conditions"] if r["entry"]["id"] in selected_ids]
    if device != "cpu" and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError("Device must be cpu or cuda:N")
    output.mkdir(parents=True, exist_ok=True)
    env = capture()
    if device.startswith("cuda"):
        probe = probe_gpu(int(device[5:]))
        atomic_json(output / "gpu_probe.json", probe)
        if env["version_mismatches"] or not probe["available"]:
            report = dict(
                state="pending",
                reason="GPU environment unavailable or version mismatch",
                environment=env,
                probe=probe,
                conditions=[
                    dict(id=r["entry"]["id"], state="pending") for r in records
                ],
            )
            seal(
                output
                / ("attempt.json" if (output / "run.json").exists() else "run.json"),
                report,
            )
            return report
        initialize_postprocessing()
        env = dict(
            capture(libraries=True), device_identity=probe.get("device_identity")
        )
    try:
        import torch

        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except ImportError:
        pass
    implementation = implementation_identity()
    resolver = Resolver(bindings)
    report = dict(
        schema_version=1,
        device=device,
        limit=prepared["limit"],
        implementation=implementation,
        storage=storage,
        conditions=[],
        gpu_comparison="pending",
        retraining="not_run",
    )
    with RunLock(output / ".lock"):
        if (output / "run.json").exists():
            previous = unseal(output / "run.json")
            if previous.get("schema_version") == 1:
                for key in ("device", "limit", "implementation", "storage"):
                    if previous.get(key) != report[key]:
                        raise ValueError(
                            "Run settings changed; use a new output directory"
                        )
                report["conditions"] = previous["conditions"]
        for row in records:
            entry = row["entry"]
            state = dict(
                id=entry["id"],
                model=entry["model"],
                condition=entry["condition"],
                state=row["state"],
            )
            report["conditions"] = [
                r for r in report["conditions"] if r["id"] != entry["id"]
            ]
            report["conditions"].append(state)
            if row["state"] == "ready":
                cfg = (
                    OrtConfig.cpu(strict_versions=True)
                    if device == "cpu"
                    else OrtConfig.cuda(
                        int(device[5:]),
                        memory_limit=entry["execution"]["gpu_mem_limit"],
                        strict_versions=True,
                    )
                )
                execution = cfg.to_dict()
                environment = dict(
                    env,
                    device=device,
                    execution=execution,
                    implementation=implementation,
                )
                identity = object_hash(
                    [row["identity"], implementation, environment, storage]
                )
                try:
                    result = _condition(
                        row,
                        resolver,
                        output / entry["id"],
                        row.get("limit", prepared["limit"]),
                        cfg,
                        environment,
                        identity,
                        storage,
                    )
                    state.update(
                        state=result["state"],
                        result=file_record(
                            output / entry["id"] / "result.json", output
                        ),
                        samples=result["sample_count"],
                        repeat_bitwise=result["repeat_bitwise"],
                    )
                except Exception as error:
                    state.update(state="fail", error=str(error))
            else:
                state["missing"] = row["missing"]
            seal(output / "run.json", report)
        report["reevaluation"] = reevaluation_metrics(output, report["conditions"])
        seal(output / "run.json", report)
    return dict(
        output=str(output),
        completed=sum(r["state"] == "completed" for r in report["conditions"]),
        failed=sum(r["state"] == "fail" for r in report["conditions"]),
        pending=sum(r["state"] == "pending" for r in report["conditions"]),
    )


def initialize_postprocessing():
    # torchvision loads its CUDA runtime even for CPU NMS. Load it before sealing
    # the environment so replay/resume identity does not depend on earlier models.
    import importlib

    try:
        importlib.import_module("torchvision")
    except ImportError:
        pass


def repeat_outputs(session, names, feeds):
    first = session.run(names, feeds)
    second = session.run(names, feeds)
    return dict(zip(names, first)), {
        n: arrays(x, y) for n, x, y in zip(names, first, second)
    }


def reevaluation_metrics(output, conditions):
    results = [
        unseal(checked_file(output, r["result"]))
        for r in conditions
        if r["state"] == "completed"
    ]
    fp32 = {r["model"]: r for r in results if r["condition"] == "fp32"}
    rows = []
    for r in results:
        metric = r["primary_metric"]
        value = r["metrics"].get(metric)
        base = fp32.get(r["model"])
        denominator = base["metrics"].get(metric) if base else None
        matched = base and all(
            r[k] == base[k]
            for k in ("recipe_identity", "input_identity", "profile", "limit")
        )
        recovery = (
            100 * value / denominator
            if matched
            and isinstance(value, (int, float))
            and isinstance(denominator, (int, float))
            and denominator > 0
            else None
        )
        rows.append(
            dict(
                model=r["model"],
                condition=r["condition"],
                metric=metric,
                value=value,
                unit=r["metric_units"].get(metric),
                recovery_percent=recovery,
                full_accuracy=r["stages"]["full_accuracy"],
                origin="this_execution",
            )
        )
    return rows


def reference_diagnostic(reference, resolver, sample_id, index, digest, values, names):
    result = dict(reference_state="pending")
    if not reference or index >= len(reference[1]["samples"]):
        return result
    descriptor, refs = reference
    src, ref = descriptor["samples"][index], refs["samples"][index]
    if src["id"] != sample_id or ref["id"] != sample_id:
        return dict(result, reference_reason="Reference sample/order differs")
    if src["inputs_sha256"] != ref["inputs_sha256"]:
        raise ValueError("GPU reference input binding differs")
    try:

        def load(relative, signature):
            return load_arrays(
                resolver.check(dict(id=descriptor["root"] + relative, sha256=signature))
            )

        inputs = load(src["inputs"], src["inputs_sha256"])
        expected = load(ref["outputs"], ref["outputs_sha256"])
    except FileNotFoundError as error:
        return dict(result, reference_reason=str(error))
    if feed_digest(inputs) != digest:
        return dict(
            result, reference_reason="Prepared input differs from GTX frozen input"
        )
    if names != refs["output_names"]:
        raise ValueError("Reference output names differ")
    result["reference_raw"] = {
        n: arrays(values[n], expected["output_" + str(i)]) for i, n in enumerate(names)
    }
    result["reference_state"] = (
        "pass"
        if all(v["numeric_passed"] for v in result["reference_raw"].values())
        else "fail"
    )
    return result


def raw_digest(values):
    return {
        name: dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        )
        for name, value in values.items()
    }
