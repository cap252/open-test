"""Small local archives exercise download, extraction and manifest preparation."""

from contextlib import nullcontext, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from opennpu_quant._io import read_json, sha256
from opennpu_quant.cli import main
from opennpu_quant.data import download as datasets
from opennpu_quant.data.prepare import prepare


class DatasetDownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.root = self.folder / "dataset"
        self.manifest = self.folder / "manifest.json"
        self.sources = self.folder / "sources"
        self.sources.mkdir()
        self.enterContext(
            patch(
                "opennpu_quant.data.prepare.COUNTS",
                {"coco": 2, "coco_keypoints": 2, "voc": 2, "widerface": 2},
            )
        )
        self.enterContext(
            patch(
                "opennpu_quant.data.prepare.identifiers",
                side_effect=lambda name: [["train"]]
                if name == "voc"
                else [["8" if name == "coco_keypoints" else "9"]],
            )
        )
        if sys.platform == "win32":
            # Production targets Linux. File locking itself is covered separately.
            for module in ("data.download", "export.download"):
                self.enterContext(
                    patch(
                        "opennpu_quant." + module + ".RunLock",
                        side_effect=lambda path: nullcontext(),
                    )
                )
        images = [{"id": i, "file_name": f"{i:012d}.jpg"} for i in (2, 1)]
        self.annotations = {
            "annotations/instances_val2017.json": json.dumps(
                {"images": images}
            ).encode(),
            "annotations/person_keypoints_val2017.json": json.dumps(
                {"images": images}
            ).encode(),
        }
        self.validation = {
            f"val2017/{i:012d}.jpg": f"validation-{i}".encode() for i in (1, 2)
        }
        self.training = {
            f"train2017/{i:012d}.jpg": f"training-{i}".encode() for i in (8, 9)
        }
        self.coco = {
            "annotations": self.archive("annotations.zip", self.annotations),
            "val": self.archive("val.zip", self.validation),
            "train": self.archive("train.zip", self.training),
        }
        self.enterContext(patch.object(datasets, "COCO_ARCHIVES", self.coco))
        self.voc_files = {
            "ImageSets/Segmentation/val.txt": b"b\na\n",
            "JPEGImages/train.jpg": b"train",
            "JPEGImages/a.jpg": b"a-image",
            "JPEGImages/b.jpg": b"b-image",
            "SegmentationClass/a.png": b"a-mask",
            "SegmentationClass/b.png": b"b-mask",
        }
        self.voc = self.archive("voc.tar", self.voc_files, prefix="VOCdevkit/VOC2012")
        self.enterContext(patch.object(datasets, "VOC_ARCHIVE", self.voc))

    def archive(self, filename, entries, *, prefix=""):
        path = self.sources / filename
        if filename.endswith(".zip"):
            with zipfile.ZipFile(path, "w") as archive:
                for name, content in entries.items():
                    archive.writestr(name, content)
        else:
            with tarfile.open(path, "w") as archive:
                for name, content in entries.items():
                    member = tarfile.TarInfo(f"{prefix}/{name}" if prefix else name)
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
        return datasets.Archive(filename, path.as_uri(), sha256(path), prefix)

    def test_coco_cli_download_and_keypoints_reuse(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main(
                [
                    "data",
                    "prepare",
                    "coco",
                    "--download",
                    "--root",
                    str(self.root),
                    "--data-dir",
                    str(self.folder / "data"),
                ]
            )
        result = json.loads(output.getvalue())
        self.assertEqual((2, 1), (result["validation"], result["calibration"]))
        manifest = read_json(result["manifest"])
        self.assertEqual([1, 2], [row["id"] for row in manifest["validation"]])
        self.assertEqual([9], [row["id"] for row in manifest["calibration"]])
        self.assertFalse(manifest["reference_bytes_verified"])
        with patch.object(datasets, "download", side_effect=AssertionError("network")):
            keypoints = prepare(
                "coco_keypoints", self.root, self.manifest, download=True
            )
        self.assertEqual([8], [row["id"] for row in keypoints["calibration"]])
        self.assertEqual(
            "annotations/person_keypoints_val2017.json",
            keypoints["annotations"]["coco"],
        )

    def test_voc_archive_prefix_is_removed_and_split_order_preserved(self):
        result = prepare("voc", self.root, self.manifest, download=True)
        self.assertEqual(str(self.root.resolve()), result["root"])
        self.assertEqual(["b", "a"], [row["id"] for row in result["validation"]])
        self.assertEqual(["train"], [row["id"] for row in result["calibration"]])
        self.assertTrue((self.root / "SegmentationClass/b.png").is_file())
        self.assertFalse((self.root / "VOCdevkit").exists())
        with patch.object(datasets, "download", side_effect=AssertionError("network")):
            repeated = prepare("voc", self.root, self.manifest, download=True)
            local = prepare("voc", self.root, self.manifest)
        self.assertEqual(result, repeated)
        self.assertEqual(result, local)

    def test_missing_image_is_restored_from_cached_archive(self):
        original = prepare("coco", self.root, self.manifest, download=True)
        kept = self.root / "val2017/000000000001.jpg"
        modified = kept.stat().st_mtime_ns
        missing = self.root / "val2017/000000000002.jpg"
        missing.unlink()
        with patch(
            "opennpu_quant.export.download.urllib.request.urlopen",
            side_effect=AssertionError("network"),
        ):
            repeated = prepare("coco", self.root, self.manifest, download=True)
        self.assertEqual(original, repeated)
        self.assertEqual(b"validation-2", missing.read_bytes())
        self.assertEqual(modified, kept.stat().st_mtime_ns)
        self.assertEqual([], list((self.root / ".downloads").glob("extract-*")))

    def test_no_download_without_flag(self):
        with patch.object(datasets, "download_dataset") as download:
            with self.assertRaises(FileNotFoundError):
                prepare("coco", self.root, self.manifest)
        download.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_checksum_failure_does_not_extract_or_write_manifest(self):
        self.coco["annotations"] = replace(self.coco["annotations"], sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "checksum"):
            prepare("coco", self.root, self.manifest, download=True)
        self.assertFalse(self.manifest.exists())
        self.assertFalse((self.root / "annotations").exists())
        self.assertFalse((self.root / ".downloads/annotations.zip.pending").exists())

    def test_corrupt_cached_archive_is_rejected(self):
        prepare("coco", self.root, self.manifest, download=True)
        self.manifest.unlink()
        (self.root / "val2017/000000000002.jpg").unlink()
        cached = self.root / ".downloads/val.zip"
        cached.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            prepare("coco", self.root, self.manifest, download=True)
        self.assertFalse(self.manifest.exists())

    def test_conflicting_file_is_preserved_before_other_files_are_installed(self):
        existing = self.root / "val2017/000000000001.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"user-image")
        with self.assertRaisesRegex(FileExistsError, "differs from archive"):
            prepare("coco", self.root, self.manifest, download=True)
        self.assertEqual(b"user-image", existing.read_bytes())
        self.assertFalse((self.root / "val2017/000000000002.jpg").exists())
        self.assertFalse(self.manifest.exists())

    def test_unsafe_zip_and_tar_are_rejected(self):
        for name, filename, prefix, entries in (
            ("coco", "unsafe.zip", "", self.annotations),
            ("voc", "unsafe.tar", "VOCdevkit/VOC2012", self.voc_files),
        ):
            with self.subTest(dataset=name):
                malicious = dict(entries)
                malicious["../../../../escape.txt"] = b"bad"
                archive = self.archive(filename, malicious, prefix=prefix)
                with (
                    patch.dict(self.coco, annotations=archive),
                    patch.object(datasets, "VOC_ARCHIVE", archive),
                ):
                    with self.assertRaisesRegex(ValueError, "escapes destination"):
                        prepare(name, self.root, self.manifest, download=True)
                self.assertFalse(self.manifest.exists())
                self.assertFalse((self.folder / "escape.txt").exists())

    def test_incomplete_voc_archive_is_not_installed(self):
        incomplete = dict(self.voc_files)
        del incomplete["SegmentationClass/a.png"]
        archive = self.archive("incomplete.tar", incomplete, prefix="VOCdevkit/VOC2012")
        with patch.object(datasets, "VOC_ARCHIVE", archive):
            with self.assertRaisesRegex(ValueError, "lacks required"):
                prepare("voc", self.root, self.manifest, download=True)
        self.assertFalse(self.manifest.exists())
        self.assertFalse((self.root / "JPEGImages").exists())

    def test_download_still_checks_calibration_validation_overlap(self):
        overlapping = dict(self.training)
        overlapping["train2017/000000000009.jpg"] = b"validation-1"
        self.coco["train"] = self.archive("overlap.zip", overlapping)
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare("coco", self.root, self.manifest, download=True)
        self.assertFalse(self.manifest.exists())

    def test_wrong_validation_count_stops_before_image_downloads(self):
        annotations = dict(self.annotations)
        annotations["annotations/instances_val2017.json"] = b'{"images": []}'
        self.coco["annotations"] = self.archive("wrong-count.zip", annotations)
        with self.assertRaisesRegex(ValueError, "Expected 2 validation images"):
            prepare("coco", self.root, self.manifest, download=True)
        self.assertFalse((self.root / ".downloads/train.zip").exists())
        self.assertFalse(self.manifest.exists())

    def test_unsupported_download_fails_before_creating_files(self):
        with self.assertRaisesRegex(ValueError, "--download supports"):
            prepare("widerface", self.root, self.manifest, download=True)
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
