"""Pinned COCO 2017 and VOC2012 archives, installed without replacing local data."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

from .._io import read_json, sha256
from .._locking import RunLock
from ..export.download import download, extract_archive


@dataclass(frozen=True)
class Archive:
    filename: str
    url: str
    sha256: str
    prefix: str = ""


# SHA256 values published by TensorFlow Datasets:
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/coco.txt
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/voc.txt
# The path-style URL addresses the official COCO bucket over valid HTTPS.
_COCO = "https://s3.amazonaws.com/images.cocodataset.org/"
COCO_ARCHIVES = {
    "annotations": Archive(
        "annotations_trainval2017.zip",
        _COCO + "annotations/annotations_trainval2017.zip",
        "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
    ),
    "train": Archive(
        "train2017.zip",
        _COCO + "zips/train2017.zip",
        "69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929",
    ),
    "val": Archive(
        "val2017.zip",
        _COCO + "zips/val2017.zip",
        "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
    ),
}
# This is the checksum-pinned mirror used by TensorFlow Datasets.
VOC_ARCHIVE = Archive(
    "VOCtrainval_11-May-2012.tar",
    "https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar",
    "e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb",
    "VOCdevkit/VOC2012",
)


def _target(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError("Dataset path escapes root: " + str(path))
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Dataset links are not supported: " + str(parent))
        if parent != path and parent.exists() and not parent.is_dir():
            raise FileExistsError("Dataset directory is a file: " + str(parent))
    return path


def _present(root, paths):
    # Check every path, including unsafe paths after a missing one.
    return all([_target(root, name).is_file() for name in paths])


def _install(root, archive, ready):
    if ready(root):
        return
    cache = root / ".downloads"
    print(f"Preparing {archive.filename}...", file=sys.stderr)
    downloaded = download(
        archive.url, cache / archive.filename, sha256_expected=archive.sha256
    )
    with TemporaryDirectory(prefix="extract-", dir=cache) as temporary:
        staging = Path(temporary)
        extract_archive(downloaded, staging)
        source = staging / archive.prefix
        if not ready(source):
            raise ValueError(
                "Archive lacks required dataset files: " + archive.filename
            )
        pending = []
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            target = _target(root, path.relative_to(source))
            if target.exists():
                if not target.is_file() or sha256(target) != sha256(path):
                    raise FileExistsError(
                        "Existing dataset file differs from archive: "
                        + str(target)
                        + "; use a new --root or resolve the conflicting file"
                    )
            else:
                pending.append((path, target))
        # Check all conflicts before publishing any files from this archive.
        for path, target in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)


def download_dataset(name, root, *, calibration, expected_samples):
    """Download missing inputs, then let prepare() perform its usual validation.

    Existing complete layouts need no network access. VOC's archive prefix is
    stripped so that --root always contains JPEGImages and SegmentationClass.
    Downloads are cached under --root/.downloads and checked before extraction.
    """
    if name not in ("coco", "coco_keypoints", "voc"):
        raise ValueError("Unsupported archive dataset: " + name)
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = _target(root, ".downloads")
    cache.mkdir(exist_ok=True)
    with RunLock(cache / "prepare.lock"):
        if name in ("coco", "coco_keypoints"):
            annotation = "annotations/" + (
                "person_keypoints_val2017.json"
                if name == "coco_keypoints"
                else "instances_val2017.json"
            )
            _install(
                root,
                COCO_ARCHIVES["annotations"],
                lambda folder: _present(folder, [annotation]),
            )
            images = read_json(root / annotation)["images"]
            if len(images) != expected_samples:
                raise ValueError(
                    f"Expected {expected_samples} validation images, found {len(images)}"
                )
            validation = ["val2017/" + row["file_name"] for row in images]
            training = [f"train2017/{int(row[0]):012d}.jpg" for row in calibration]
            _install(root, COCO_ARCHIVES["val"], lambda p: _present(p, validation))
            _install(root, COCO_ARCHIVES["train"], lambda p: _present(p, training))
        else:

            def ready(folder):
                split = _target(folder, "ImageSets/Segmentation/val.txt")
                if not split.is_file():
                    return False
                names = split.read_text().split()
                if len(names) != expected_samples:
                    raise ValueError(
                        f"Expected {expected_samples} validation images, found {len(names)}"
                    )
                paths = [f"JPEGImages/{row[0]}.jpg" for row in calibration]
                for item in names:
                    paths.extend(
                        (f"JPEGImages/{item}.jpg", f"SegmentationClass/{item}.png")
                    )
                return _present(folder, paths)

            _install(root, VOC_ARCHIVE, ready)
