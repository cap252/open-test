import copy
from pathlib import Path
import tempfile
import unittest

from opennpu_quant._io import read_json
from opennpu_quant.results import tables


class FinalEvidenceTests(unittest.TestCase):
    def test_real_decode_and_evaluation_settings_reject_old_fp32_pair(self):
        import onnx
        from onnx import helper as h
        import yaml
        from opennpu_quant import runner
        from tests.test_result_reliability import fixture

        for setting in ("decode", "evaluation"):
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as temp:
                paths, config = fixture(Path(temp), host="postprocess")
                post = paths.models / "toy/postprocess.onnx"
                model = onnx.load(post)
                model.graph.node.append(
                    h.make_node("Identity", ["scores"], ["alternate_scores"])
                )
                model.graph.output.append(
                    h.make_tensor_value_info("alternate_scores", 1, [1, 6])
                )
                onnx.save(model, post)
                runner.run(config, paths)
                out = paths.runs / config["name"]
                base = next(
                    r
                    for r in read_json(out / "records.json")["records"]
                    if r["condition"] == "fp32"
                )
                before = read_json(out / "toy/fp32/result.json")
                changed = copy.deepcopy(config)
                changed["name"] = setting
                if setting == "decode":
                    recipe_path = paths.models / "toy/recipe.yaml"
                    recipe = yaml.safe_load(recipe_path.read_text())
                    recipe["decode"]["output"] = "alternate_scores"
                    recipe_path.write_text(yaml.safe_dump(recipe))
                else:
                    changed["evaluation"]["limit"] = 1
                runner.run(changed, paths)
                after_dir = paths.runs / changed["name"]
                q = next(
                    r
                    for r in read_json(after_dir / "records.json")["records"]
                    if r["condition"] == "pot_rtn"
                )
                after = read_json(after_dir / "toy/pot_rtn/result.json")
                self.assertEqual(before["graph_identities"], after["graph_identities"])
                self.assertEqual(
                    before["preprocessing_identity"], after["preprocessing_identity"]
                )
                self.assertNotEqual(
                    before["evaluation_identity"], after["evaluation_identity"]
                )
                q["run_id"] = base["run_id"]
                row = tables([base, q])[0][1]
                self.assertEqual("mismatch", row["pairing_status"])
                self.assertIsNone(row["recovery_pct"])
                self.assertIsNotNone(row["int8_pct"])
