"""Pinned source caches recover missing files without trusting links or changed code."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from opennpu_quant._io import sha256
from opennpu_quant.export.download import extract_archive
from opennpu_quant.export.source_archive import checkout_archive


class SourceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "sources/toy/commit"
        self.archive = self.folder.with_suffix(".zip")
        self.archive.parent.mkdir(parents=True)
        self.before_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.before_path))

    def archive_recipe(self, extra=None):
        with zipfile.ZipFile(self.archive, "w") as z:
            z.writestr("upstream/model.py", "version = 1\n")
            z.writestr("upstream/config.json", "{}")
            if extra:
                z.writestr(*extra)
        return {
            "name": "toy",
            "source": {
                "commit": "commit",
                "repo": "https://github.com/example/toy",
                "archive_sha256": sha256(self.archive),
            },
        }

    def test_clean_reuse_and_missing_cache_recovery(self):
        recipe = self.archive_recipe()
        self.folder.mkdir()
        first = checkout_archive(recipe, self.root)
        self.assertEqual("version = 1\n", (first / "model.py").read_text())
        with patch("opennpu_quant.export.source_archive.extract_archive") as extract:
            self.assertEqual(first, checkout_archive(recipe, self.root))
            extract.assert_not_called()
        (first / "config.json").unlink()
        self.assertEqual(first, checkout_archive(recipe, self.root))
        self.assertEqual("{}", (first / "config.json").read_text())

    def test_modified_source_is_not_repaired_or_overwritten(self):
        recipe = self.archive_recipe()
        source = checkout_archive(recipe, self.root)
        (source / "model.py").write_text("changed")
        (source / "config.json").unlink()
        with self.assertRaisesRegex(ValueError, "changed"):
            checkout_archive(recipe, self.root)
        self.assertEqual("changed", (source / "model.py").read_text())

    def test_explicit_source_missing_file_and_wrong_archive_hash(self):
        recipe = self.archive_recipe()
        explicit = self.root / "explicit"
        explicit.mkdir()
        explicit.with_suffix(".zip").write_bytes(self.archive.read_bytes())
        with self.assertRaisesRegex(FileNotFoundError, "Missing"):
            checkout_archive(recipe, self.root, explicit)
        self.assertEqual([], list(explicit.iterdir()))
        recipe["source"]["archive_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum"):
            checkout_archive(recipe, self.root)

    def test_interrupted_extraction_never_publishes_partial_source(self):
        recipe = self.archive_recipe()

        def interrupt(archive, destination, **kwargs):
            destination.mkdir()
            (destination / "partial").write_text("partial")
            raise KeyboardInterrupt()

        with (
            patch(
                "opennpu_quant.export.source_archive.extract_archive",
                side_effect=interrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            checkout_archive(recipe, self.root)
        self.assertFalse(self.folder.exists())
        self.assertFalse(list(self.folder.parent.glob("*.pending-*")))
        self.assertTrue((checkout_archive(recipe, self.root) / "model.py").is_file())

    def test_only_pinned_documentation_link_payload_is_extracted_as_text(self):
        link = zipfile.ZipInfo("upstream/docs/demo.md")
        link.external_attr = 0o120777 << 16
        self.archive_recipe((link, "../README.md"))
        target = self.root / "expanded"
        with self.assertRaisesRegex(ValueError, "links"):
            extract_archive(self.archive, target)
        extract_archive(
            self.archive, target, text_links={link.filename: "../README.md"}
        )
        value = target / link.filename
        self.assertFalse(value.is_symlink())
        self.assertEqual("../README.md", value.read_text())
        with self.assertRaisesRegex(ValueError, "links"):
            extract_archive(self.archive, target, text_links={link.filename: "wrong"})

    def test_unsafe_archive_path_is_rejected(self):
        recipe = self.archive_recipe(("upstream/../../escape.py", "bad"))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            checkout_archive(recipe, self.root)
        self.assertFalse((self.root / "escape.py").exists())

    def test_additional_import_files_are_rejected_without_modification(self):
        recipe = self.archive_recipe()
        source = checkout_archive(recipe, self.root)
        for name in (
            "model/__init__.py",
            "extra.py",
            "extra.pyc",
            "extension.so",
            "__pycache__/extra.cpython-311.pyc",
        ):
            with self.subTest(name=name):
                extra = source / name
                extra.parent.mkdir(parents=True, exist_ok=True)
                extra.write_bytes(b"added")
                with self.assertRaisesRegex(
                    ValueError, "Unpinned upstream import source"
                ):
                    checkout_archive(recipe, self.root)
                self.assertEqual(b"added", extra.read_bytes())
                extra.unlink()

    def test_git_ignored_import_files_are_also_rejected(self):
        import subprocess
        from opennpu_quant.export.torch_models import checkout

        source = self.root / "git-source"
        source.mkdir()
        for args in (
            ["init"],
            ["config", "user.email", "test@example.invalid"],
            ["config", "user.name", "Fixture"],
        ):
            subprocess.run(
                ["git", "-C", str(source), *args], check=True, capture_output=True
            )
        (source / "model.py").write_text("value = 1\n")
        (source / ".gitignore").write_text("model/\n")
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(source), "commit", "-m", "fixture"],
            check=True,
            capture_output=True,
        )
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        recipe = dict(name="toy", source=dict(repo=str(source), commit=commit))
        self.assertEqual(source, checkout(recipe, self.root, source))
        (source / "model").mkdir()
        (source / "model/__init__.py").write_text("value = 2\n")
        with self.assertRaisesRegex(ValueError, "Unpinned upstream import source"):
            checkout(recipe, self.root, source)
