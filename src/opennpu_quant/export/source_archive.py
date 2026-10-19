"""Content-pinned source archives, including verified local extracted copies."""

from pathlib import Path
import sys
import zipfile
from .._io import sha256
from .download import download, extract_archive


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
    if explicit:
        target = Path(explicit).resolve()
    else:
        if not folder.exists():
            extract_archive(archive, folder)
        children = [p for p in folder.iterdir() if p.is_dir()]
        if len(children) != 1:
            raise ValueError("Expected one upstream source archive root")
        target = children[0].resolve()
    # Verify source files against the original archive on every export. Cache
    # placement and directory names are not trusted source-version evidence.
    import hashlib

    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            if info.is_dir():
                continue
            rel = Path(*Path(info.filename).parts[1:])
            path = (target / rel).resolve()
            if not path.is_relative_to(target) or not path.is_file():
                raise ValueError("Missing/unsafe pinned source: " + str(rel))
            if hashlib.sha256(zipped.read(info)).hexdigest() != sha256(path):
                raise ValueError("Pinned upstream source changed: " + str(rel))
    sys.path.insert(0, str(target))
    return target
