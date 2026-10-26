"""External ONNX weights are part of fixed-model provenance and resume checks."""

import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx

from opennpu_quant._io import atomic_json, object_hash, read_json, sha256
from opennpu_quant.reproduction.assets import Resolver, seal, unseal
from opennpu_quant.reproduction.execution import run, _reference
from opennpu_quant.reproduction.prepare import prepare
from opennpu_quant.reproduction.report import report
from tests.test_gpu_reproduction import fixture


def external_fixture(root, kind="network", *, int8=False):
    source = fixture(root, int8=int8)
    selection = read_json(root / "selection.json")
    for entry in selection["entries"]:
        if kind == "network":
            path = source / (
                "model.onnx" if entry["condition"] == "fp32" else "qdq.onnx"
            )
            model = onnx.load(path)
        else:
            path = source / (kind + ".onnx")
            name = "images" if kind == "preprocess" else "logits"
            shape = [1, 3, 4, 4] if kind == "preprocess" else [1, 6]
            output = "host_" + name
            h, t = onnx.helper, onnx.TensorProto
            model = h.make_model(
                h.make_graph(
                    [h.make_node("Mul", [name, "gain"], [output])],
                    kind,
                    [h.make_tensor_value_info(name, t.FLOAT, shape)],
                    [h.make_tensor_value_info(output, t.FLOAT, shape)],
                    [onnx.numpy_helper.from_array(np.ones(1, np.float32), "gain")],
                ),
                opset_imports=[h.make_opsetid("", 18)],
                ir_version=10,
            )
            if kind == "preprocess":
                entry["recipe"]["network_input_map"] = {"images": output}
            else:
                entry["recipe"]["decode"]["output"] = output
        weights = path.name + ".weights"
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=weights,
            size_threshold=0,
        )
        entry["graphs"][kind] = dict(
            id="artifact://assets/" + path.name,
            sha256=sha256(path),
            external_data={weights: sha256(source / weights)},
        )
        entry["measurement"]["samples"] = 3
    atomic_json(root / "selection.json", selection)
    return source, selection


def prepared(root, output="prepared"):
    return prepare(
        bindings=root / "bindings.json",
        selection=root / "selection.json",
        output=root / output,
        limit="full",
    )


def execute(root, output="run"):
    return run(
        package=root / "prepared", bindings=root / "bindings.json", output=root / output
    )


class ExternalReproductionTests(unittest.TestCase):
    def test_missing_and_changed_weights_are_rejected_during_prepare(self):
        for kind in ("network", "preprocess", "postprocess"):
            for missing in (False, True):
                with (
                    self.subTest(kind=kind, missing=missing),
                    tempfile.TemporaryDirectory() as temp,
                ):
                    root = Path(temp)
                    source, selection = external_fixture(root, kind)
                    asset = selection["entries"][0]["graphs"][kind]
                    weight = source / next(iter(asset["external_data"]))
                    if missing:
                        weight.unlink()
                    else:
                        weight.write_bytes(b"changed")
                    self.assertEqual(
                        {"pending" if missing else "fail": 1}, prepared(root)["counts"]
                    )

    def test_expected_hashes_are_required_and_checked_per_complete_asset_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, selection = external_fixture(root)
            correct = selection["entries"][0]
            wrong = copy.deepcopy(correct)
            wrong["id"] = "resnet18__other"
            graph = wrong["graphs"]["network"]
            for index, value in enumerate(
                (None, {}, {"model.onnx.weights": "0" * 64}, {"extra": "0" * 64})
            ):
                if value is None:
                    graph.pop("external_data", None)
                else:
                    graph["external_data"] = value
                atomic_json(root / "selection.json", dict(entries=[correct, wrong]))
                self.assertEqual(
                    {"ready": 1, "fail": 1}, prepared(root, f"p{index}")["counts"]
                )

    def test_completed_results_cannot_hide_external_weight_changes(self):
        for kind in ("network", "preprocess", "postprocess"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, selection = external_fixture(root, kind)
                self.assertEqual({"ready": 1}, prepared(root)["counts"])
                self.assertEqual(1, execute(root)["completed"])
                result = root / "run/resnet18__fp32/result.json"
                before = result.read_bytes()
                weights = source / next(
                    iter(selection["entries"][0]["graphs"][kind]["external_data"])
                )
                original = weights.read_bytes()
                for missing in (False, True):
                    if missing:
                        weights.unlink()
                    else:
                        weights.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                    with (
                        patch(
                            "opennpu_quant.reproduction.execution.create_session"
                        ) as network,
                        patch("opennpu_quant.models.bundle.create_session") as host,
                    ):
                        self.assertEqual(1, execute(root)["failed"])
                    network.assert_not_called()
                    host.assert_not_called()
                    self.assertEqual(before, result.read_bytes())
                weights.write_bytes(original)
                self.assertEqual(1, execute(root)["completed"])

    def test_interrupted_sample_reuse_also_checks_external_weights(self):
        from opennpu_quant.models.adapter import ModelEvaluator

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = external_fixture(root)
            prepared(root)
            update = ModelEvaluator.update
            calls = []

            def interrupt(evaluator, values, sample):
                calls.append(sample.sample_id)
                if len(calls) == 2:
                    raise RuntimeError("interrupted")
                return update(evaluator, values, sample)

            with patch.object(ModelEvaluator, "update", interrupt):
                self.assertEqual(1, execute(root)["failed"])
            raw = root / "run/resnet18__fp32/000000.npz"
            before = raw.read_bytes()
            (source / "model.onnx.weights").write_bytes(b"different")
            with patch(
                "opennpu_quant.reproduction.execution.create_session"
            ) as session:
                self.assertEqual(1, execute(root)["failed"])
            session.assert_not_called()
            self.assertEqual(before, raw.read_bytes())

    def test_external_int8_relocation_report_and_verified_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = external_fixture(root, int8=True)
            self.assertEqual({"ready": 2}, prepared(root)["counts"])
            self.assertEqual(2, execute(root)["completed"])
            inline = root / "inline"
            inline.mkdir()
            fixture(inline, int8=True)
            prepared(inline)
            self.assertEqual(2, execute(inline)["completed"])
            for condition in ("resnet18__fp32", "resnet18__asym_pc_pot_rtn"):
                for index in range(3):
                    relative = Path("run") / condition / f"{index:06d}.npz"
                    with (
                        np.load(root / relative) as external,
                        np.load(inline / relative) as embedded,
                    ):
                        self.assertEqual(
                            external["logits"].tobytes(), embedded["logits"].tobytes()
                        )
            result = root / "run/resnet18__asym_pc_pot_rtn/result.json"
            before = result.read_bytes()
            moved = root / "relocated"
            shutil.move(source, moved)
            atomic_json(root / "bindings.json", dict(assets=str(moved)))
            self.assertEqual(2, execute(root)["completed"])
            self.assertEqual(before, result.read_bytes())
            tables = report(
                run=root / "run",
                package=root / "prepared",
                output=root / "tables",
                table_profile="opennpu_raw",
            )
            self.assertEqual(2, tables["full"])
            (moved / "qdq.onnx.weights").unlink()
            self.assertEqual(1, execute(root)["failed"])

    def test_old_unpinned_external_package_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            external_fixture(root)
            prepared(root)
            old = unseal(root / "prepared/manifest.json")
            row = old["conditions"][0]
            row["entry"]["graphs"]["network"].pop("external_data")
            row["identity"] = object_hash(
                [row["entry"], row["data_identity"], row["limit"]]
            )
            seal(root / "prepared/manifest.json", old)
            self.assertEqual(1, execute(root)["failed"])
            self.assertIn(
                "Missing external_data hashes",
                read_json(root / "run/run.json")["conditions"][0]["error"],
            )

    def test_external_paths_cannot_escape_the_graph_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, selection = external_fixture(root)
            path = source / "model.onnx"
            model = onnx.load(path, load_external_data=False)
            for value in model.graph.initializer[0].external_data:
                if value.key == "location":
                    value.value = "../outside.weights"
            path.write_bytes(model.SerializeToString())
            selection["entries"][0]["graphs"]["network"]["sha256"] = sha256(path)
            atomic_json(root / "selection.json", selection)
            self.assertEqual({"fail": 1}, prepared(root)["counts"])

    def test_raw_references_must_pin_the_same_external_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, selection = external_fixture(root)
            entry = selection["entries"][0]
            atomic_json(
                source / "reference.json",
                dict(model_sha256=entry["graphs"]["network"]["sha256"], samples=[]),
            )
            entry["reference"] = dict(
                manifest=dict(
                    id="artifact://assets/reference.json",
                    sha256=sha256(source / "reference.json"),
                ),
                root="artifact://assets/",
            )
            atomic_json(root / "selection.json", selection)
            self.assertEqual({"fail": 1}, prepared(root)["counts"])
            with self.assertRaisesRegex(ValueError, "external data identity"):
                _reference(entry, Resolver(root / "bindings.json"))


def sparse_fixture(
    root, *, component="values", kind="network", container="initializer"
):
    source, selection = external_fixture(root, kind)
    asset = selection["entries"][0]["graphs"][kind]
    path = source / Path(asset["id"]).name
    model = onnx.load(path)
    original = model.graph.initializer[0]
    values = onnx.numpy_helper.to_array(original)
    sparse = onnx.helper.make_sparse_tensor(
        onnx.numpy_helper.from_array(values.flatten(), original.name),
        onnx.numpy_helper.from_array(np.arange(values.size, dtype=np.int64), "indices"),
        list(values.shape),
    )
    tensor = getattr(sparse, component)
    weight = source / ("sparse-" + component + ".bin")
    weight.write_bytes(tensor.raw_data)
    onnx.external_data_helper.set_external_data(tensor, weight.name)
    tensor.ClearField("raw_data")
    del model.graph.initializer[0]
    if container == "initializer":
        model.graph.sparse_initializer.append(sparse)
    elif container == "constant":
        model.graph.node.insert(
            0,
            onnx.helper.make_node("Constant", [], [original.name], sparse_value=sparse),
        )
    elif container == "subgraph":
        branch = copy.deepcopy(model.graph)
        del branch.input[:]
        branch.sparse_initializer.append(sparse)
        del model.graph.node[:]
        model.graph.initializer.append(
            onnx.numpy_helper.from_array(np.array(True), "condition")
        )
        model.graph.node.append(
            onnx.helper.make_node(
                "If",
                ["condition"],
                [model.graph.output[0].name],
                then_branch=branch,
                else_branch=branch,
            )
        )
    else:
        attribute = (
            {"sparse_value": sparse}
            if container == "function"
            else {"sparse_values": [sparse]}
        )
        node = onnx.helper.make_node("Constant", [], [original.name], **attribute)
        model.functions.append(
            onnx.helper.make_function(
                "test",
                "SparseWeight",
                [],
                [original.name],
                [node],
                [onnx.helper.make_opsetid("", 18)],
            )
        )
        model.opset_import.append(onnx.helper.make_opsetid("test", 1))
        model.graph.node.insert(
            0, onnx.helper.make_node("SparseWeight", [], [original.name], domain="test")
        )
    path.write_bytes(model.SerializeToString())
    asset["sha256"] = sha256(path)
    asset["external_data"] = {weight.name: sha256(weight)}
    atomic_json(root / "selection.json", selection)
    return path, weight, selection


class SparseExternalReproductionTests(unittest.TestCase):
    def test_executable_sparse_graph_is_rejected_by_prepare_even_with_hashes(self):
        import onnxruntime as ort

        for component in ("values", "indices"):
            with (
                self.subTest(component=component),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                path, _, _ = sparse_fixture(root, component=component)
                # ORT 1.20 executes external values, but not external indices.
                if component == "values":
                    session = ort.InferenceSession(
                        str(path), providers=["CPUExecutionProvider"]
                    )
                    actual = session.run(
                        None, {"images": np.ones((1, 3, 4, 4), np.float32)}
                    )[0]
                    np.testing.assert_array_equal(actual, [[18, 21, 24, 27, 30, 33]])
                self.assertEqual({"fail": 1}, prepared(root)["counts"])
                row = unseal(root / "prepared/manifest.json")["conditions"][0]
                self.assertIn("External sparse ONNX", str(row["missing"]))

    def test_missing_sparse_values_and_indices_fail_for_network_and_host_graphs(self):
        for kind in ("network", "preprocess", "postprocess"):
            for component in ("values", "indices"):
                with (
                    self.subTest(kind=kind, component=component),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    root = Path(tmp)
                    _, weight, selection = sparse_fixture(
                        root, component=component, kind=kind
                    )
                    selection["entries"][0]["graphs"][kind].pop("external_data")
                    atomic_json(root / "selection.json", selection)
                    weight.unlink()
                    self.assertEqual({"fail": 1}, prepared(root)["counts"])
                    row = unseal(root / "prepared/manifest.json")["conditions"][0]
                    self.assertIn("External sparse ONNX", str(row["missing"]))

    def test_nested_sparse_attributes_and_functions_cannot_bypass_validation(self):
        from opennpu_quant.graph.model import external_data_files, load_model

        for container in ("constant", "subgraph", "function", "list_attribute"):
            for component in ("values", "indices"):
                with (
                    self.subTest(container=container, component=component),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    path, _, _ = sparse_fixture(
                        Path(tmp), component=component, container=container
                    )
                    for operation in (external_data_files, load_model):
                        with self.assertRaisesRegex(ValueError, "External sparse ONNX"):
                            operation(path)

    def test_public_load_identity_and_save_reject_unresolved_sparse_data(self):
        from opennpu_quant.graph.model import (
            load_model,
            model_identity,
            bundle_identity,
            save_model,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _, _ = sparse_fixture(root)
            for source in (path, onnx.load(path, load_external_data=False)):
                for operation in (load_model, model_identity):
                    with self.assertRaisesRegex(ValueError, "External sparse ONNX"):
                        operation(source)
                with self.assertRaisesRegex(ValueError, "External sparse ONNX"):
                    save_model(source, root / "copy.onnx")
            with self.assertRaisesRegex(ValueError, "External sparse ONNX"):
                bundle_identity(path)
            self.assertFalse((root / "copy.onnx").exists())

    def test_legacy_completed_and_interrupted_results_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, weight, selection = sparse_fixture(root)
            selection["entries"][0]["graphs"]["network"].pop("external_data")
            atomic_json(root / "selection.json", selection)
            # Create an old-style result with exactly the current run identity.
            with patch("opennpu_quant.graph.model._reject_external_sparse"):
                self.assertEqual({"ready": 1}, prepared(root)["counts"])
                self.assertEqual(1, execute(root)["completed"])
            folder = root / "run/resnet18__fp32"
            result = folder / "result.json"
            before = result.read_bytes()
            raw = folder / "000000.npz"
            raw_hash = sha256(raw)
            weights = weight.read_bytes()
            for missing in (False, True):
                if missing:
                    weight.unlink()
                else:
                    weight.write_bytes(bytes(len(weights)))
                with (
                    patch(
                        "opennpu_quant.reproduction.execution.verify_complete"
                    ) as reuse,
                    patch(
                        "opennpu_quant.reproduction.execution.create_session"
                    ) as session,
                ):
                    self.assertEqual(1, execute(root)["failed"])
                reuse.assert_not_called()
                session.assert_not_called()
                self.assertEqual(before, result.read_bytes())
            result.unlink()
            with patch(
                "opennpu_quant.reproduction.execution.create_session"
            ) as session:
                self.assertEqual(1, execute(root)["failed"])
            session.assert_not_called()
            self.assertEqual(raw_hash, sha256(raw))
            error = read_json(root / "run/run.json")["conditions"][0]["error"]
            self.assertIn("External sparse ONNX", error)

    def test_self_contained_sparse_model_still_loads_and_runs(self):
        from opennpu_quant.graph.model import load_model

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, weight, selection = sparse_fixture(root)
            model = onnx.load(path, load_external_data=False)
            tensor = model.graph.sparse_initializer[0].values
            tensor.raw_data = weight.read_bytes()
            tensor.ClearField("external_data")
            tensor.ClearField("data_location")
            path.write_bytes(model.SerializeToString())
            weight.unlink()
            graph = selection["entries"][0]["graphs"]["network"]
            graph["sha256"] = sha256(path)
            graph.pop("external_data")
            atomic_json(root / "selection.json", selection)
            self.assertEqual(
                tensor.raw_data,
                load_model(path).graph.sparse_initializer[0].values.raw_data,
            )
            self.assertEqual({"ready": 1}, prepared(root)["counts"])
            self.assertEqual(1, execute(root)["completed"])
