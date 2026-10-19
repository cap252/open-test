"""Content-verified input cache that preserves each sample's spatial shape."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .._feeds import _VerifiedFeed, _fingerprint
from .._io import atomic_json, object_hash, read_json, sha256
from .._locking import RunLock
from .._feeds import feed_digest


def _layout(feed, names=None):
    if not feed or (names is not None and set(feed) != set(names)):
        raise ValueError("Calibration input names changed or empty feed")
    result = []
    for name, value in feed.items():
        value = np.asarray(value)
        if value.dtype != np.float32:
            raise ValueError("Calibration input dtype must be FP32: " + name)
        if value.ndim < 1 or value.shape[0] != 1 or any(d < 1 for d in value.shape):
            raise ValueError(
                "Calibration input must have nonempty batch-one shape: " + name
            )
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite calibration input: " + name)
        result.append(
            dict(
                name=name,
                shape=list(value.shape),
                dtype=str(value.dtype),
                payload_bytes=value.nbytes,
            )
        )
    return result


class VariableFeedCache:
    def __init__(self, directory, *, identity, verify_source):
        self.reused = False
        self.directory = Path(directory)
        self.verify_source = verify_source
        self.manifest = read_json(self.directory / "manifest.json")
        m = self.manifest
        if (
            m.get("schema_version") != 2
            or m.get("layout") != "per_sample"
            or m.get("state") != "completed"
        ):
            raise ValueError("Incomplete variable feed cache")
        if m["identity"] != object_hash(identity):
            raise ValueError("Feed cache preprocessing/source identity differs")
        signatures = m["feed_digests"]
        if (
            m["sample_count"] < 1
            or len(signatures) != m["sample_count"]
            or len(m["samples"]) != m["sample_count"]
            or hashlib.sha256("".join(signatures).encode()).hexdigest()
            != m["samples_identity"]
        ):
            raise ValueError("Feed cache sample identity is inconsistent")
        names = None
        seen = set()
        for sample in m["samples"]:
            current = [entry["name"] for entry in sample["tensors"]]
            if (
                not current
                or len(set(current)) != len(current)
                or (names is not None and set(current) != names)
            ):
                raise ValueError("Feed cache input names changed")
            names = set(current)
            for entry in sample["tensors"]:
                if entry["file"] in seen:
                    raise ValueError("Duplicate cache payload reference")
                seen.add(entry["file"])
        self.manifest["tensors"] = [
            e for sample in m["samples"] for e in sample["tensors"]
        ]
        files = self.verify()
        for sample, signature in zip(m["samples"], signatures):
            values = self._load(sample)
            _layout(values, names)
            if feed_digest(values) != signature:
                raise ValueError("Feed cache stored sample digest differs from payload")
        if any(_fingerprint(path) != fingerprint for path, fingerprint in files):
            raise ValueError("Feed cache changed while opening")

    def _path(self, entry):
        path = (self.directory / entry["file"]).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError("Feed cache path escaped cache directory")
        return path

    def _load(self, sample):
        values = {}
        for entry in sample["tensors"]:
            value = np.load(self._path(entry), mmap_mode="r", allow_pickle=False)
            if (
                list(value.shape) != entry["shape"]
                or str(value.dtype) != entry["dtype"]
            ):
                raise ValueError("Feed cache tensor layout differs")
            values[entry["name"]] = value
        return values

    def verify(self):
        self.verify_source()
        files = []
        for sample in self.manifest["samples"]:
            for entry in sample["tensors"]:
                path = self._path(entry)
                before = _fingerprint(path)
                if (
                    before[2] != entry["bytes"]
                    or sha256(path) != entry["sha256"]
                    or _fingerprint(path) != before
                ):
                    raise ValueError("Feed cache payload changed: " + entry["file"])
                files.append((path, before))
        return tuple(files)

    def __call__(self):
        files = self.verify()
        fingerprints = dict(files)
        for sample, signature in zip(
            self.manifest["samples"], self.manifest["feed_digests"]
        ):
            own_files = tuple(
                (self._path(e), fingerprints[self._path(e)]) for e in sample["tensors"]
            )
            feed = _VerifiedFeed(self._load(sample), signature, own_files)
            feed.verified_digest()
            yield feed
        self.verify_source()
        if any(_fingerprint(path) != fingerprint for path, fingerprint in files):
            raise ValueError("Cached payload changed during replay")


def prepare_variable_feed_cache(
    directory,
    factory,
    *,
    identity,
    verify_source,
    expected_samples,
    max_bytes,
    reserve=None,
):
    directory = Path(directory)
    if (
        type(expected_samples) is not int
        or type(max_bytes) is not int
        or expected_samples < 1
        or max_bytes < 1
        or not callable(factory)
    ):
        raise ValueError("Positive cache budget/count and feed factory required")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with RunLock(directory.with_name(directory.name + ".lock")):
        if directory.exists():
            cache = VariableFeedCache(
                directory, identity=identity, verify_source=verify_source
            )
            if cache.manifest["sample_count"] != expected_samples:
                raise ValueError("Reused calibration sample count changed")
            if (
                sum(e["bytes"] for s in cache.manifest["samples"] for e in s["tensors"])
                > max_bytes
            ):
                raise MemoryError("Existing calibration cache exceeds its disk budget")
            cache.reused = True
            return cache
        verify_source()
        layouts, signatures = [], []
        names = None
        estimated = 0
        for feed in factory():
            if len(layouts) >= expected_samples:
                raise ValueError("Calibration sample count changed")
            layout = _layout(feed, names)
            names = [e["name"] for e in layout]
            estimated += sum(e["payload_bytes"] + 4096 for e in layout)
            if estimated > max_bytes:
                raise MemoryError("Calibration cache exceeds its disk budget")
            layouts.append(layout)
            signatures.append(feed_digest(feed))
        if len(layouts) != expected_samples:
            raise ValueError("Calibration sample count changed")
        verify_source()
        if reserve is not None:
            reserve(estimated)
        temporary = Path(
            tempfile.mkdtemp(prefix=directory.name + ".building-", dir=directory.parent)
        )
        samples = []
        actual_bytes = 0
        try:
            for index, feed in enumerate(factory()):
                if (
                    index >= expected_samples
                    or _layout(feed, names) != layouts[index]
                    or feed_digest(feed) != signatures[index]
                ):
                    raise ValueError("Calibration replay changed while building cache")
                tensors = []
                for ti, entry in enumerate(layouts[index]):
                    name = f"sample_{index:06d}_{ti}.npy"
                    path = temporary / name
                    np.save(path, np.asarray(feed[entry["name"]]), allow_pickle=False)
                    actual_bytes += path.stat().st_size
                    if actual_bytes > max_bytes:
                        raise MemoryError("Calibration cache exceeds its disk budget")
                    tensors.append(
                        dict(
                            entry,
                            file=name,
                            bytes=path.stat().st_size,
                            sha256=sha256(path),
                        )
                    )
                    path.chmod(0o444)
                samples.append(dict(tensors=tensors))
            if len(samples) != expected_samples:
                raise ValueError("Calibration sample count changed")
            verify_source()
            record = dict(
                schema_version=2,
                layout="per_sample",
                state="completed",
                identity=object_hash(identity),
                source_identity=identity,
                sample_count=expected_samples,
                samples=samples,
                feed_digests=signatures,
                samples_identity=hashlib.sha256(
                    "".join(signatures).encode()
                ).hexdigest(),
            )
            atomic_json(temporary / "manifest.json", record)
            VariableFeedCache(temporary, identity=identity, verify_source=verify_source)
            os.replace(temporary, directory)
            return VariableFeedCache(
                directory, identity=identity, verify_source=verify_source
            )
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
