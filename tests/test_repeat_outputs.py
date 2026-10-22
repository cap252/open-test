"""Replay repeats inference through run_tensors, like the other session callers."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from onnx import TensorProto, helper

from opennpu_quant._io import atomic_json, read_json
from opennpu_quant.ort.config import OrtConfig
from opennpu_quant.ort.fetch import run_tensors
from opennpu_quant.ort.session import create_session
from opennpu_quant.reproduction import execution
from opennpu_quant.reproduction.assets import unseal
from opennpu_quant.reproduction.execution import repeat_outputs, run
from opennpu_quant.reproduction.prepare import prepare
from tests.test_gpu_reproduction import fixture

FEED = {"x": np.array([[-1.5, 2.0, -3.25, 4.0]], np.float32)}


def session():
    """y = Relu(x), z = Neg(x), and x itself as a passthrough graph output."""
    values = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 4]) for n in "yzx"
    ]
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"]), helper.make_node("Neg", ["x"], ["z"])],
        "repeat",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        values,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    return create_session(model, OrtConfig.cpu())


def recorded(calls):
    def call(*args):
        values = run_tensors(*args)
        calls.append(values)
        return values

    return call


class RepeatOutputsTests(unittest.TestCase):
    def test_computed_outputs_match_session_run_bitwise_in_requested_order(self):
        s = session()
        names = ["z", "y"]
        values, repeat = repeat_outputs(s, names, copy.deepcopy(FEED))
        self.assertEqual(names, list(values))
        for name in names:
            expected = s.run([name], FEED)[0]
            self.assertEqual(expected.dtype, values[name].dtype)
            self.assertEqual(expected.shape, values[name].shape)
            self.assertEqual(expected.tobytes(), values[name].tobytes())
            self.assertTrue(repeat[name]["bitwise_equal"])

    def test_passthrough_and_mixed_outputs_are_independent_copies(self):
        s = session()
        feed = copy.deepcopy(FEED)
        before = feed["x"].copy()
        calls = []
        with patch.object(execution, "run_tensors", side_effect=recorded(calls)):
            values, repeat = repeat_outputs(s, ["x", "y"], feed)
        first, second = calls
        self.assertEqual(before.tobytes(), values["x"].tobytes())
        self.assertEqual(s.run(["y"], FEED)[0].tobytes(), values["y"].tobytes())
        for a, b in (
            (first[0], feed["x"]),
            (second[0], feed["x"]),
            (first[0], second[0]),
        ):
            self.assertFalse(np.shares_memory(a, b))
        values["x"][0, 0] = 99
        self.assertEqual(before.tobytes(), feed["x"].tobytes())
        self.assertEqual(before.tobytes(), second[0].tobytes())
        self.assertTrue(all(r["bitwise_equal"] for r in repeat.values()))

    def test_missing_inputs_raise_and_extra_feed_keys_are_ignored(self):
        with self.assertRaisesRegex(KeyError, "x"):
            repeat_outputs(session(), ["y"], {"other": FEED["x"]})
        decoy = dict(FEED, y=np.zeros((1, 4), np.float32))
        values, _ = repeat_outputs(session(), ["y"], decoy)
        self.assertEqual(session().run(["y"], FEED)[0].tobytes(), values["y"].tobytes())


class ReplayFetchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        fixture(self.root)

    def prepare(self):
        prepare(
            bindings=self.root / "bindings.json",
            selection=self.root / "selection.json",
            output=self.root / "prepared",
        )
        return dict(
            package=self.root / "prepared", bindings=self.root / "bindings.json"
        )

    def test_missing_network_input_fails_the_condition_with_its_name(self):
        selection = read_json(self.root / "selection.json")
        selection["entries"][0]["recipe"]["network_input_map"] = {"other": "images"}
        atomic_json(self.root / "selection.json", selection)
        self.assertEqual(1, run(**self.prepare(), output=self.root / "run")["failed"])
        state = unseal(self.root / "run/run.json")["conditions"][0]
        self.assertEqual("fail", state["state"])
        self.assertIn("images", state["error"])

    def test_compact_resume_recomputes_raw_through_run_tensors(self):
        args = dict(self.prepare(), output=self.root / "compact", storage="predictions")
        self.assertEqual(1, run(**args)["completed"])
        folder = self.root / "compact/resnet18__fp32"
        digests = [s["raw_digest"] for s in unseal(folder / "result.json")["samples"]]
        (folder / "result.json").unlink()
        calls = []
        with patch.object(execution, "run_tensors", side_effect=recorded(calls)):
            self.assertEqual(1, run(**args)["completed"])
        # Three resumed samples, each recomputed twice and checked against its digest.
        self.assertEqual(6, len(calls))
        resumed = unseal(folder / "result.json")["samples"]
        self.assertEqual(digests, [s["raw_digest"] for s in resumed])
        self.assertFalse(list(folder.glob("*.npz")))


if __name__ == "__main__":
    unittest.main()
