"""Code reuse identity covers Python bytes throughout the installed package."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from opennpu_quant import _identity


class CodeIdentityTests(unittest.TestCase):
    def test_nested_python_bytes_names_additions_and_deletions_change_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "__init__.py").write_text("# package\n")
            nested = root / "models/nested/evaluator.py"
            nested.parent.mkdir(parents=True)
            nested.write_text("def score():\n    return 1\n")
            with patch.object(_identity, "ROOT", root):
                before = _identity.code_identity()
                self.assertEqual(before, _identity.code_identity())
                original = nested.read_bytes()
                for content in (
                    b"# comment\n" + original,
                    original.replace(b"return 1", b"return 2"),
                ):
                    with self.subTest(content=content):
                        nested.write_bytes(content)
                        self.assertNotEqual(before, _identity.code_identity())
                        nested.write_bytes(original)
                        self.assertEqual(before, _identity.code_identity())
                renamed = nested.with_name("renamed.py")
                nested.rename(renamed)
                self.assertNotEqual(before, _identity.code_identity())
                renamed.rename(nested)
                nested.unlink()
                self.assertNotEqual(before, _identity.code_identity())
                nested.write_bytes(original)
                self.assertEqual(before, _identity.code_identity())
                added = root / "models/added.py"
                added.write_text("# new module\n")
                self.assertNotEqual(before, _identity.code_identity())
                added.unlink()
                self.assertEqual(before, _identity.code_identity())
                (root / "notes.txt").write_text("Not Python code")
                self.assertEqual(before, _identity.code_identity())
