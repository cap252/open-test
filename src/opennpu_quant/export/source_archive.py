"""Content-pinned source archives, including verified local extracted copies."""

import hashlib
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile
from .._io import sha256
from .._locking import RunLock
from .download import download, extract_archive


def _documentation_links(source):
    if source["archive_sha256"] != (
        "647bba7d16cc7e13feb5f2beca0296f86186c944f5a8fa9ccefc2629758bccba"
    ):
        return {}
    root = "YOLOX-6ddff4824372906469a7fae2dc3206c7aa4bbaee/docs/demo/"
    return {
        root + name + "_readme.md": "../../demo/" + target + "/README.md"
        for name, target in (
            ("megengine_cpp", "MegEngine/cpp"),
            ("megengine_py", "MegEngine/python"),
            ("ncnn_android", "ncnn/android"),
            ("ncnn_cpp", "ncnn/cpp"),
            ("onnx", "ONNXRuntime"),
            ("openvino_cpp", "OpenVINO/cpp"),
            ("openvino_py", "OpenVINO/python"),
            ("trt_cpp", "TensorRT/cpp"),
            ("trt_py", "TensorRT/python"),
        )
    }


def verify_source_files(target, expected):
    """Reject additional importable files, including ignored Git files and bytecode."""
    for root, directories, files in os.walk(target, followlinks=False):
        directories[:] = [name for name in directories if name != ".git"]
        for name in directories + files:
            path = Path(root) / name
            relative = path.relative_to(target).as_posix()
            if relative in expected:
                continue
            if path.is_symlink() or path.suffix.lower() in (
                ".py",
                ".pyc",
                ".pyo",
                ".so",
                ".pyd",
            ):
                raise ValueError("Unpinned upstream import source: " + relative)


def _verify(zipped, target, text_links):
    roots, missing, links, expected = set(), [], set(), set()
    for info in zipped.infolist():
        name = PurePosixPath(info.filename)
        if name.is_absolute() or ".." in name.parts or not name.parts:
            raise ValueError("Unsafe pinned archive path")
        roots.add(name.parts[0])
        if info.is_dir():
            continue
        rel = Path(*name.parts[1:])
        expected.add(rel.as_posix())
        value = zipped.read(info)
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            if text_links.get(info.filename, "").encode() != value:
                raise ValueError("Archive links are not supported")
            links.add(info.filename)
        path = (target / rel).resolve()
        if not path.is_relative_to(target.resolve()):
            raise ValueError("Unsafe pinned source: " + str(rel))
        if not path.is_file():
            missing.append(str(rel))
        elif hashlib.sha256(value).hexdigest() != sha256(path):
            raise ValueError("Pinned upstream source changed: " + str(rel))
    if len(roots) != 1 or links != set(text_links):
        raise ValueError("Pinned source archive layout changed")
    verify_source_files(target, expected)
    if missing:
        raise FileNotFoundError("Missing pinned source: " + missing[0])


def checkout_archive(recipe, directory, explicit=None):
    source = recipe["source"]
    folder = Path(directory) / "sources" / recipe["name"] / source["commit"]
    archive = (
        Path(explicit).with_suffix(".zip") if explicit else folder.with_suffix(".zip")
    )
    repository = source["repo"].removeprefix("https://github.com/")
    if not archive.is_file():
        archive = download(
            f"https://codeload.github.com/{repository}/zip/{source['commit']}",
            folder.with_suffix(".zip"),
            sha256_expected=source["archive_sha256"],
        )
    if sha256(archive) != source["archive_sha256"]:
        raise ValueError("Upstream source archive checksum mismatch")
    text_links = _documentation_links(source)
    with zipfile.ZipFile(archive) as zipped:
        if explicit:
            target = Path(explicit).resolve()
            _verify(zipped, target, text_links)
        else:
            with RunLock(folder.with_suffix(".extract.lock")):
                roots = {PurePosixPath(n).parts[0] for n in zipped.namelist()}
                if len(roots) != 1:
                    raise ValueError("Expected one upstream source archive root")
                root = next(iter(roots))
                target = (folder / root).resolve()
                try:
                    _verify(zipped, target, text_links)
                except FileNotFoundError:
                    with tempfile.TemporaryDirectory(
                        prefix=folder.name + ".pending-", dir=folder.parent
                    ) as temp:
                        pending = Path(temp) / "source"
                        extract_archive(archive, pending, text_links=text_links)
                        _verify(zipped, pending / root, text_links)
                        previous = Path(temp) / "incomplete"
                        if folder.exists():
                            folder.rename(previous)
                        try:
                            pending.rename(folder)
                        except BaseException:
                            if previous.exists():
                                previous.rename(folder)
                            raise
    sys.path.insert(0, str(target))
    return target
