"""Checksum-verified downloads and safe archive extraction for model export."""

from pathlib import Path
import shutil, tarfile, zipfile, urllib.request
from .._io import sha256
from .._locking import RunLock


def download(url, destination, *, sha256_expected):
    if not isinstance(sha256_expected, str) or len(sha256_expected) != 64:
        raise ValueError("Expected SHA256 is required")
    destination = Path(destination)
    with RunLock(destination.with_suffix(destination.suffix + ".lock")):
        if destination.exists():
            if sha256(destination) != sha256_expected:
                raise ValueError(
                    "Existing download checksum mismatch: " + destination.name
                )
            return destination
        temporary = destination.with_suffix(destination.suffix + ".pending")
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                temporary.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            if sha256(temporary) != sha256_expected:
                raise ValueError("Download checksum mismatch: " + destination.name)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def extract_archive(archive, destination, *, text_links=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    def checked(name):
        result = (destination / name).resolve()
        if not result.is_relative_to(destination.resolve()):
            raise ValueError("Archive path escapes destination")
        return result

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            allowed = text_links or {}
            found = set()
            for member in z.infolist():
                checked(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    if (
                        member.filename not in allowed
                        or z.read(member) != allowed[member.filename].encode()
                    ):
                        raise ValueError("Archive links are not supported")
                    found.add(member.filename)
            if found != set(allowed):
                raise ValueError("Expected archive documentation links changed")
            # ZipFile extracts link payloads as ordinary files, never as symlinks.
            z.extractall(destination)
    else:
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                checked(member.name)
                if not (member.isdir() or member.isfile()):
                    raise ValueError("Archive links/special files are not supported")
            t.extractall(destination, filter="data")
