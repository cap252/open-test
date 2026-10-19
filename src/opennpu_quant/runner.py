"""Run an experiment: for each model and condition, quantize, evaluate and record.

Every condition gets an execution identity (code, graphs, inputs, settings). A finished
condition with the same identity and intact files is reused instead of recomputed.
Layout of a run folder: docs/run-pipeline.md.
"""

import contextlib
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import tempfile
import time
from uuid import uuid4

from ._io import atomic_json, read_json, object_hash, sha256, now
from ._locking import RunLock
from .configuration import DEFAULT_INPUT_CACHE_BYTES, validate, runtime, scheme_for
from .models.spec import ModelSpec
from .models.bundle import ModelBundle
from .models.adapter import ModelEvaluator
from .data.samples import Dataset, preprocessing_implementation
from .graph.model import model_identity, save_model
from .api import calibrate, quantize, apply_adaround
from .quant.calibration import feed_digest
from .quant.storage import save_statistics, read_statistics
from .quant.config import AdaroundConfig, CalibrationConfig
from .quant.adaround_cache import ActivationCacheConfig
from .ort.session import environment, create_session
from .evaluation.loop import evaluate
from .report import ATTESTED_FILES, report, verify_artifacts


_REUSE_GUIDANCE = (
    "Use a new --output to keep previous results, or --force to recompute selected "
    "results (calibration caches and AdaRound checkpoints may still be reused)."
)
# Source files whose changes invalidate cached calibration statistics.
_STATISTICS_CODE = ("calibration.py", "histograms.py", "engine.py")


def _config_changes(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        paths = []
        for key in sorted(before.keys() | after.keys(), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                paths.append(path)
            else:
                paths.extend(_config_changes(before[key], after[key], path))
        return sorted(paths)
    return [prefix or "<root>"] if before != after else []


def _code_identity():
    base = Path(__file__).parent
    return object_hash(
        {str(p.relative_to(base)): sha256(p) for p in sorted(base.rglob("*.py"))}
    )


def runtime_audit(result, ort):
    from .quant.engine import audit_runtime

    with tempfile.TemporaryDirectory(
        dir=ort.scratch, prefix="opennpu_audit_"
    ) as folder:
        original = Path(folder) / "qdq.onnx"
        executed = Path(folder) / "executed.onnx"
        save_model(result.model, original)
        session = create_session(original, ort, optimized=executed)
        del session
        return audit_runtime(original, executed, result.audit)


def run(config, paths, *, force=False, recalibrate=False):
    """Execute configured models sequentially under one run-directory lock."""
    validate(config, models_dir=paths.models)
    if recalibrate and not any(config["conditions"].values()):
        raise ValueError("--recalibrate requires a quantized condition")
    out = paths.runs / config["name"]
    ort = runtime(config)
    code = _code_identity()
    with RunLock(out / ".lock"):
        recorded = out / "effective_config.json"
        if recorded.exists() and not force:
            previous = read_json(recorded)
            if previous != config:
                changed = _config_changes(previous, config)
                detail = ", ".join(changed[:10])
                if len(changed) > 10:
                    detail += f" (+{len(changed) - 10} more)"
                raise ValueError(
                    f"Run configuration changed. Changed keys: {detail}. "
                    + _REUSE_GUIDANCE
                )
        atomic_json(out / "effective_config.json", config)
        status = dict(
            state="running",
            attempt_id=uuid4().hex,
            started=now(),
            finished=None,
            models={
                name: dict(
                    state="pending",
                    conditions={c: {"state": "pending"} for c in config["conditions"]},
                )
                for name in config["models"]
            },
        )
        atomic_json(out / "status.json", status)
        try:
            if recalibrate:
                for name in config["models"]:
                    atomic_json(
                        out / name / "recalibration.json",
                        dict(generation=status["attempt_id"], state="pending"),
                    )
            for name in config["models"]:
                status["models"][name]["state"] = "running"
                atomic_json(out / "status.json", status)
                _run_model(name, config, paths, out, ort, code, force, status)
                status["models"][name]["state"] = "completed"
                atomic_json(out / "status.json", status)
                report(out)
        except BaseException as error:
            status["state"] = (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            )
            status["error"] = dict(type=type(error).__name__, message=str(error))
            status["finished"] = now()
            for state in status["models"].values():
                if state["state"] == "running":
                    state.update(state=status["state"], error=status["error"])
                for condition in state["conditions"].values():
                    if condition["state"] == "running":
                        condition.update(state=status["state"], error=status["error"])
            atomic_json(out / "status.json", status)
            # Best effort: a report failure must not hide the original error.
            with contextlib.suppress(Exception):
                report(out)
            raise
        status.update(state="completed", finished=now())
        atomic_json(out / "status.json", status)
        return report(out)


def _run_model(name, config, paths, out, ort, code, force, status):
    _ModelRun(name, config, paths, out, ort, code, status).run(force)


class _ModelRun:
    """One model's conditions in one run.

    Calibration statistics, the calibration input factory and RTN results are shared
    by the model's conditions; statistics are collected only when a condition needs them.
    """

    def __init__(self, name, config, paths, out, ort, code, status):
        self.name, self.config, self.paths, self.out = name, config, paths, out
        self.ort, self.code, self.status = ort, code, status
        from .models.evaluation_profiles import identity as profile_identity

        self.spec = ModelSpec.for_run(
            name,
            paths.models / name,
            profile=config["evaluation"].get("profiles", {}).get(name),
        )
        self.evaluation_profile = profile_identity(self.spec.recipe)
        self.bundle = ModelBundle(self.spec, paths.models / name)
        dataset_name = self.spec.recipe["dataset"]
        manifest = paths.data / dataset_name / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' is not prepared (missing {manifest}). "
                f"Run: opennpu-quant data prepare {shlex.quote(dataset_name)} "
                f"--root '<dataset folder>' --data-dir {shlex.quote(str(paths.data))}"
            )
        self.dataset = Dataset(manifest, root=paths.datasets.get(dataset_name))
        # Hash the materialized graph, including the bytes actually read from external data.
        # Bundle location and external-data filenames are not model identity.
        self.graphs = {k: model_identity(v) for k, v in self.bundle.paths.items()}
        self._identify_inputs(config["evaluation"]["limit"])
        self.count = config["calibration"]["samples"][dataset_name]
        self.factory = self.dataset.feeds(self.bundle, self.count)
        self.calibration_data = self.dataset.identity("calibration", self.count)
        self.request_path = out / name / "recalibration.json"
        self.request = (
            read_json(self.request_path) if self.request_path.exists() else None
        )
        self.generation = self.request["generation"] if self.request else None
        self.statistics = None
        self.calibration_ort = None
        self.input_cache = None
        self.rtn = {}  # scheme family -> RTN QuantizationResult

    def _identify_inputs(self, limit):
        """Preprocessing and evaluation identities; None for unidentifiable plugins."""
        recipe = self.spec.recipe
        implementation = preprocessing_implementation(recipe["preprocess"])
        self.preprocessing_identity = (
            object_hash(
                [
                    recipe["preprocess"],
                    recipe["input"],
                    self.bundle.mapping,
                    self.graphs.get("preprocess"),
                    implementation,
                ]
            )
            if implementation
            else None
        )
        self.evaluation_identity = (
            object_hash(
                [
                    self.dataset.identity("validation", limit),
                    recipe,
                    self.evaluation_profile,
                    environment(self.ort),
                    self.code,
                    {k: v for k, v in self.graphs.items() if k != "network"},
                    self.bundle.mapping,
                    self.preprocessing_identity,
                ]
            )
            if self.preprocessing_identity
            else None
        )

    def _save_status(self):
        atomic_json(self.out / "status.json", self.status)

    def run(self, force):
        for condition, policy in self.config["conditions"].items():
            self._run_condition(condition, policy, force)

    def _primary_metric(self):
        recipe = self.spec.recipe
        if recipe["task"] == "face_verification" and self.dataset.info.get("smoke"):
            metric = self.config["report"]["primary_metric"].get("face_verification")
            if metric != "smoke_pair_accuracy":
                raise ValueError(
                    "LFW smoke data requires the smoke_pair_accuracy report metric"
                )
            return metric
        return recipe["metric"]

    def _run_condition(self, condition, policy, force):
        current = self.status["models"][self.name]["conditions"][condition]
        current["state"] = "running"
        self._save_status()
        start = time.monotonic()
        directory = self.out / self.name / condition
        scheme = (
            scheme_for(self.config, self.name, policy["scheme"]) if policy else None
        )
        identity = self._identity(condition, policy, scheme)
        if not force and self._reuse(directory, condition, identity, policy, current):
            self._save_status()
            return
        result = audit = None
        if policy:
            self._collect_statistics()
            result = self._quantize(policy, scheme, directory, identity)
            audit = runtime_audit(result, self.ort)
        network = result.model if result else self.bundle.paths["network"]
        values = self._evaluate(network, directory)
        metric = self._primary_metric()
        if metric not in values.metrics:
            raise ValueError(
                f"Primary metric '{metric}' is undefined for {self.name}/{condition}; "
                f"available: {', '.join(sorted(values.metrics)) or 'none'}. "
                "Check the recipe metric or evaluate more samples."
            )
        # Files from an earlier attempt must not be attested by this result.
        for stale in ("model.onnx", *ATTESTED_FILES):
            (directory / stale).unlink(missing_ok=True)
        if self.config["output"]["save_predictions"]:
            (directory / "predictions.jsonl.gz.pending").replace(
                directory / "predictions.jsonl.gz"
            )
        record = self._record(condition, policy, scheme, identity, values, result)
        record.update(
            runtime_audit=audit, elapsed_s=time.monotonic() - start, finished_at=now()
        )
        if policy and self.generation:
            record["calibration_generation"] = self.generation
        self._write(directory, identity, policy, record, result)
        current.update(
            state="completed",
            identity=identity,
            result_sha256=sha256(directory / "result.json"),
            measurement_attempt_id=self.status["attempt_id"],
        )
        self._save_status()

    def _identity(self, condition, policy, scheme):
        """Execution identity of one condition; any change prevents result reuse."""
        return object_hash(
            [
                self.code,
                self.graphs,
                self.bundle.mapping,
                self.evaluation_identity,
                condition,
                policy,
                scheme.to_dict() if scheme else None,
                self.config["adaround"] if policy.get("adaround") else None,
                self.config.get("activation_cache"),
                self.count,
                self.calibration_data,
                self.config["calibration"],
                self.config["evaluation"]["limit"],
            ]
        )

    def _reuse(self, directory, condition, identity, policy, current):
        """Mark an intact earlier result as reused; False means recompute it."""
        complete = directory / "complete.json"
        if not complete.exists():
            return False
        if not self.preprocessing_identity:
            raise ValueError(
                "Cannot reuse results with unidentified custom preprocessing. "
                + _REUSE_GUIDANCE
            )
        proof = read_json(complete)
        if proof["identity"] != identity:
            raise ValueError(
                f"Run execution identity changed for condition '{condition}'. "
                "The stored identity does not identify the exact cause. "
                "Possible causes include code, model/host graphs, inputs or "
                "preprocessing, runtime or quantization settings. " + _REUSE_GUIDANCE
            )
        generation_ok = (
            not policy or proof.get("calibration_generation") == self.generation
        )
        if not (verify_artifacts(directory, proof) and generation_ok):
            # A tampered or incomplete result is recomputed from verified inputs.
            return False
        previous = read_json(directory / "result.json")
        current.update(
            state="reused",
            identity=identity,
            result_sha256=proof["result_sha256"],
            measurement_attempt_id=previous.get("attempt_id"),
            measurement_finished_at=previous.get("finished_at"),
            verification="same_execution_identity_and_artifact_hashes",
        )
        return True

    def _collect_statistics(self):
        """Load or collect FP32 calibration statistics once per model."""
        if self.statistics is not None:
            return
        calibration = self.config["calibration"]
        input_policy = calibration.get("input_cache", {})
        if input_policy.get("enabled", False):
            cache_start = time.monotonic()
            self.factory = self.dataset.cached_feeds(
                self.bundle,
                self.count,
                self.paths.cache / "inputs" / self.name,
                max_bytes=input_policy.get("max_bytes", DEFAULT_INPUT_CACHE_BYTES),
            )
            manifest = self.factory.manifest
            self.input_cache = dict(
                identity=manifest["identity"],
                samples_identity=manifest["samples_identity"],
                sample_count=manifest["sample_count"],
                bytes=sum(v["bytes"] for v in manifest["tensors"]),
                reused=self.factory.reused,
                prepare_seconds=time.monotonic() - cache_start,
                manifest_sha256=sha256(self.factory.directory / "manifest.json"),
            )
            atomic_json(self.out / self.name / "input_cache.json", self.input_cache)
        signatures = hashlib.sha256()
        for feed in self.factory():
            signatures.update(feed_digest(feed).encode())
        feed_id = signatures.hexdigest()
        expected = calibration.get("expected_samples_identity", {}).get(self.name)
        if expected is not None and feed_id != expected:
            raise ValueError(
                f"Calibration inputs differ from the selected source for {self.name}: "
                f"expected {expected}, observed {feed_id}. Check image order and preprocessing."
            )
        self.calibration_ort = (
            self.ort
            if calibration["device"] == "run"
            else runtime(self.config, calibration["device"])
        )
        key = object_hash(
            [
                self.graphs["network"],
                feed_id,
                calibration["histogram_bins"],
                environment(self.calibration_ort),
                {
                    p: sha256(Path(__file__).parent / "quant" / p)
                    for p in _STATISTICS_CODE
                },
            ]
        )
        cache = self.paths.cache / "calibration" / self.name / key
        if self.request:
            cache = cache / "requests" / self.generation
        pending = self.request is not None and self.request["state"] == "pending"
        options = CalibrationConfig(histogram_bins=calibration["histogram_bins"])
        network = self.bundle.paths["network"]
        if (cache / "complete.json").exists() and not pending:
            self.statistics = read_statistics(
                cache,
                network,
                config=options,
                ort=self.calibration_ort,
                samples_identity=feed_id,
            )
        else:
            self.statistics = calibrate(
                network, self.factory, config=options, ort=self.calibration_ort
            )
            save_statistics(self.statistics, cache)
        if self.request:
            identity = self.statistics.statistics_identity
            if not pending and self.request["statistics_identity"] != identity:
                raise ValueError(
                    "Recalibration statistics changed; request --recalibrate again"
                )
            self.request.update(state="completed", statistics_identity=identity)
            atomic_json(self.request_path, self.request)

    def _quantize(self, policy, scheme, directory, identity):
        """RTN result of the scheme family, then AdaRound when the condition asks for it."""
        network = self.bundle.paths["network"]
        family = policy["scheme"]
        if family not in self.rtn:
            # Use the collection execution profile to validate encodings, independent of evaluation device.
            self.rtn[family] = quantize(
                network, self.statistics, scheme, ort=self.calibration_ort
            )
        if not policy.get("adaround"):
            return self.rtn[family]
        cache = dict(self.config.get("activation_cache", {}))
        enabled = cache.pop("enabled", True)
        checkpoint = directory / "adaround" / identity
        return apply_adaround(
            network,
            self.rtn[family],
            self.factory,
            AdaroundConfig(**self.config["adaround"]),
            ort=self.ort,
            checkpoint_dir=checkpoint / self.generation
            if self.generation
            else checkpoint,
            activation_cache=ActivationCacheConfig(**cache) if enabled else None,
        )

    def _evaluate(self, network, directory):
        evaluator = ModelEvaluator(self.bundle, self.dataset)
        stream = None
        if self.config["output"]["save_predictions"]:
            # Predictions are streamed; the preprocessed dataset is never held in memory.
            directory.mkdir(parents=True, exist_ok=True)
            stream = gzip.open(directory / "predictions.jsonl.gz.pending", "wt")
            _tee_predictions(evaluator, stream)
        limit = self.config["evaluation"]["limit"]
        selected = None if limit is None else self.dataset.rows("validation", limit)
        try:
            return evaluate(
                network,
                self.dataset.samples(self.bundle, limit=limit),
                evaluator=evaluator,
                ort=self.ort,
                expected_samples=self.dataset.info["expected_samples"],
                # Selected rows: an LFW limit counts pairs, not image views.
                limit=None if selected is None else len(selected),
            )
        finally:
            if stream:
                stream.close()

    def _record(self, condition, policy, scheme, identity, values, result):
        recipe = self.spec.recipe
        metric = self._primary_metric()
        return dict(
            attempt_id=self.status["attempt_id"],
            model=self.name,
            title=recipe["title"],
            model_metadata={
                k: recipe[k]
                for k in (
                    "source",
                    "dataset",
                    "task",
                    "input",
                    "metric",
                    "preprocess",
                    "decode",
                )
            },
            condition=condition,
            metric=metric,
            value=values.metrics[metric],
            metrics=values.metrics,
            metric_units=values.metric_units,
            metric_details=values.metric_details,
            samples=values.samples,
            expected_samples=self.dataset.info["expected_samples"],
            complete=values.complete,
            functional_passed=True,
            export_validation=self.bundle.validation,
            full_evaluation=values.samples == self.dataset.info["expected_samples"],
            reference=bool(self.dataset.info.get("reference_bytes_verified"))
            and values.complete,
            config_identity=object_hash(self.config),
            evaluation_identity=self.evaluation_identity,
            evaluation_profile=self.evaluation_profile,
            preprocessing_identity=self.preprocessing_identity,
            identity=identity,
            model_identity=self.graphs["network"],
            graph_identities=self.graphs,
            scheme=scheme.to_dict() if scheme else None,
            calibration_samples=self.count if policy else None,
            calibration_identity=self.statistics.statistics_identity
            if policy
            else None,
            input_cache=self.input_cache if policy else None,
            adaround=result.audit.get("adaround") if result else None,
            runtime=values.runtime,
        )

    def _write(self, directory, identity, policy, record, result):
        """Write result.json and its files, then complete.json with their hashes."""
        model_hash = None
        if result and self.config["output"]["save_qdq_models"]:
            save_model(result.model, directory / "model.onnx")
            model_hash = sha256(directory / "model.onnx")
        if result:
            atomic_json(directory / "quantization.json", result.audit)
        atomic_json(directory / "result.json", record)
        files = {
            name: sha256(directory / name)
            for name in ATTESTED_FILES
            if (directory / name).is_file()
        }
        proof = dict(
            identity=identity,
            result_sha256=sha256(directory / "result.json"),
            model_sha256=model_hash,
            files=files,
        )
        if policy and self.generation:
            proof["calibration_generation"] = self.generation
        atomic_json(directory / "complete.json", proof)


def _tee_predictions(evaluator, stream):
    """Make evaluator.update also write each sample's prediction as one JSON line."""
    update = evaluator.update

    def update_and_write(outputs, sample):
        update(outputs, sample)
        prediction = dict(id=sample.sample_id, prediction=evaluator.prediction)
        stream.write(json.dumps(prediction) + "\n")

    evaluator.update = update_and_write
