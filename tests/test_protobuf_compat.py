"""Both descriptor APIs preserve repeated and singular message traversal."""

from types import SimpleNamespace
import unittest

import onnx

from opennpu_quant._protobuf import is_repeated
from opennpu_quant.graph.model import load_model
from opennpu_quant.export.contract import graph_contract
from tests.library_fixtures import toy_model
from tests.test_export_contract import graph


class ProtobufCompatibilityTests(unittest.TestCase):
    def test_modern_api_does_not_access_removed_label_even_for_singular_fields(self):
        class ModernField:
            def __init__(self, repeated):
                self.is_repeated = repeated

            @property
            def label(self):
                raise AssertionError("Modern descriptors must not read label")

        for repeated in (False, True):
            with self.subTest(repeated=repeated):
                self.assertIs(repeated, is_repeated(ModernField(repeated)))

    def test_legacy_optional_required_and_repeated_fields(self):
        for label, expected in ((1, False), (2, False), (3, True)):
            with self.subTest(label=label):
                self.assertIs(
                    expected,
                    is_repeated(SimpleNamespace(label=label, LABEL_REPEATED=3)),
                )

    def test_installed_descriptors_and_real_onnx_traversal(self):
        fields = onnx.ModelProto.DESCRIPTOR.fields_by_name
        self.assertTrue(is_repeated(fields["metadata_props"]))
        self.assertFalse(is_repeated(fields["graph"]))
        model = toy_model()
        before = model.SerializeToString(deterministic=True)
        copied = load_model(model)
        self.assertIsNot(copied, model)
        self.assertEqual(before, copied.SerializeToString(deterministic=True))
        self.assertEqual(before, model.SerializeToString(deterministic=True))
        dynamic = graph()
        original = graph_contract(dynamic)
        copied = load_model(dynamic)
        self.assertEqual(original, graph_contract(copied))
        copied.graph.input[0].type.tensor_type.shape.dim[2].dim_param = "different"
        self.assertNotEqual(original["structure"], graph_contract(copied)["structure"])
