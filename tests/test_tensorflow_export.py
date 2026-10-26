"""Check converter arguments with fake TF boundaries, not real export or parity."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import onnx

from opennpu_quant.export.tensorflow_models import export_model
from opennpu_quant.models.spec import ModelSpec


class ConversionReached(Exception):
    """Stop at the external converter; graph conversion is outside these unit tests."""


class TensorflowExportTests(unittest.TestCase):
    def check_opset(self, model, archive_name, converter_name):
        recipe = ModelSpec.load(model).recipe
        recipe["export"]["opset"] = 17
        tf = MagicMock(name="tensorflow")
        tf.function.side_effect = lambda **kwargs: lambda function: function
        signature = Mock()
        signature.structured_input_signature = ((), {"input": Mock(dtype=tf.uint8)})
        signature.structured_outputs = {"scores": object()}
        tf.saved_model.load.return_value.signatures = {"serving_default": signature}
        converter = Mock(side_effect=ConversionReached)
        conversion = Mock()
        setattr(conversion.convert, converter_name, converter)

        def extract(archive, destination):
            destination.mkdir(parents=True)
            (destination / archive_name).write_bytes(b"test fixture")

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(
                "sys.modules", {"tensorflow": tf, "tf2onnx": conversion, "onnx": onnx}
            ),
            patch("opennpu_quant.export.tensorflow_models.extract_archive", extract),
        ):
            target = Path(temp) / "raw.onnx"
            with self.assertRaises(ConversionReached):
                export_model(recipe, "weights.tar", [], target, temp)
            converter.assert_called_once()
            self.assertEqual(17, converter.call_args.kwargs["opset"])
            self.assertEqual(str(target), converter.call_args.kwargs["output_path"])
            if converter_name == "from_graph_def":
                self.assertEqual(
                    [recipe["export"]["graph_input"]],
                    converter.call_args.kwargs["input_names"],
                )
                self.assertEqual(
                    [recipe["export"]["graph_output"]],
                    converter.call_args.kwargs["output_names"],
                )

    def test_ssd_passes_the_recipe_opset_to_from_function(self):
        self.check_opset("ssd_mobilenet_v2_320", "saved_model.pb", "from_function")

    def test_deeplab_passes_the_recipe_opset_to_from_graph_def(self):
        self.check_opset(
            "deeplabv3_mnv2_voc", "frozen_inference_graph.pb", "from_graph_def"
        )


if __name__ == "__main__":
    unittest.main()
