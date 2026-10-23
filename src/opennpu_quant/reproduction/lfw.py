"""Regenerate the pinned LFW input cache without distributing image tensors."""

from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
import hashlib
import json
import os
import platform
import tempfile

from .._identity import code_identity
from .._io import atomic_json, now, object_hash, read_json, sha256
from .._locking import cache_lock
from .assets import Resolver, catalog


def _reference():
    return read_json(
        files("opennpu_quant.resources.reproduction") / "lfw-input-reference.json"
    )


def _contract(selection, resolver):
    from ..data.lfw_protocol import require_full

    entries = [
        e
        for e in catalog(selection)["entries"]
        if e["dataset"].get("kind") == "frozen_lfw"
    ]
    if not entries:
        raise ValueError("Selection contains no frozen LFW conditions")
    descriptor, recipe = entries[0]["dataset"], entries[0]["recipe"]
    for entry in entries:
        if entry["dataset"] != descriptor or any(
            entry["recipe"][key] != recipe[key] for key in ("input", "preprocess")
        ):
            raise ValueError("Selected LFW conditions have different input contracts")
    reference = _reference()
    if (
        reference.get("schema_version") != 1
        or descriptor["samples"]["sha256"] != reference["samples_sha256"]
        or descriptor["cache"]["sha256"] != reference["cache_sha256"]
    ):
        raise ValueError("No matching packaged LFW input reference for this selection")
    samples = resolver.json(descriptor["samples"])
    require_full(samples)
    identity = samples["cache_identity"]["validation"]
    rows = samples["validation"]
    required = sorted({p[k] for p in samples["pairs"] for k in ("first", "second")})
    if (
        identity["rows"] != rows
        or identity["contract"] != samples["input_contract"]
        or [(r["image_key"], r["flip"]) for r in rows]
        != [(key, flip) for key in required for flip in (0, 1)]
        or any(r["image_id"] != r["image_key"] + f"|flip={r['flip']}" for r in rows)
    ):
        raise ValueError("LFW reference order or original/flip views changed")
    contract = identity["contract"]
    if (
        recipe["input"] != dict(name="images", shape=[1, 3, 112, 112], layout="NCHW")
        or recipe["preprocess"]
        != dict(kind="aligned_face", size=112, **contract["normalization"])
        or contract["input_size"] != [112, 112]
        or contract["source_color"] != "BGR"
        or contract["network_color"] != "RGB"
    ):
        raise ValueError("Unsupported LFW preprocessing contract")
    # Preserve the recorded contract, not a claim that its old code ran locally.
    # Actual generation provenance is written separately to verification.json.
    manifest = dict(
        schema_version=1,
        state="completed",
        identity=object_hash(identity),
        source_identity=identity,
        **{
            k: reference[k]
            for k in ("sample_count", "tensors", "feed_digests", "samples_identity")
        },
    )
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if hashlib.sha256(serialized.encode()).hexdigest() != descriptor["cache"]["sha256"]:
        raise ValueError("LFW input reference does not match the selected cache hash")
    if reference["sample_count"] != len(rows):
        raise ValueError("LFW reference sample count changed")
    return descriptor, recipe, samples, manifest


def _sources(root, samples):
    from ..data.lfw import PAIR_HASH, LANDMARK_HASH
    from ..evaluation.face_verification import read_pairs

    pins = samples["input_contract"]["dataset_files"]
    expected = {"pairs.txt": PAIR_HASH, "lfw_funneled.json": LANDMARK_HASH}
    if samples["pairs_sha256"] != PAIR_HASH or any(
        pins[k] != v for k, v in expected.items()
    ):
        raise ValueError("LFW source pins differ from the supported protocol")
    paths = {root / k: v for k, v in expected.items()}
    for path, digest in paths.items():
        if sha256(path) != digest:
            raise ValueError("LFW source file changed: " + path.name)
    if read_pairs(root / "pairs.txt") != samples["pairs"]:
        raise ValueError("LFW pair/fold order differs from the selected reference")
    landmarks = {}
    for row in read_json(root / "lfw_funneled.json")["samples"]:
        key = Path(row["full_path"]).relative_to("lfw_funneled").as_posix()
        if key in landmarks:
            raise ValueError("Duplicate LFW landmark key: " + key)
        landmarks[key] = row["landmarks"]
    image_root = (root / "data/lfw_funneled").resolve()
    for row in samples["validation"]:
        key = row["image_key"]
        path = (image_root / key).resolve()
        if not path.is_relative_to(image_root):
            raise ValueError("LFW image path escapes the local root")
        if landmarks.get(key) != row["landmarks"]:
            raise ValueError("LFW landmarks differ from reference: " + key)
        if path in paths and paths[path] != row["sha256"]:
            raise ValueError("Inconsistent original/flip source hash: " + key)
        paths[path] = row["sha256"]

    def verify():
        for path, digest in paths.items():
            if sha256(path) != digest:
                raise ValueError(
                    "LFW source file changed: " + str(path.relative_to(root))
                )

    verify()
    return image_root, landmarks, verify


def _feeds(image_root, landmarks, samples, recipe, reference):
    import cv2
    import numpy as np
    from PIL import Image
    from .._feeds import feed_digest
    from ..data.lfw import align_mobilefacenet
    from ..models.extended_preprocess import preprocess

    previous, aligned = None, None
    for index, row in enumerate(samples["validation"]):
        key = row["image_key"]
        if key != previous:
            # Decode the very bytes that were hashed, avoiding a hash/read gap.
            payload = (image_root / key).read_bytes()
            if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise ValueError("LFW source image changed: " + key)
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Unreadable LFW image: " + key)
            crop = align_mobilefacenet(image, landmarks[key])
            if crop.shape != (112, 112, 3) or crop.dtype != np.uint8:
                raise ValueError("Invalid aligned LFW crop: " + key)
            aligned, previous = Image.fromarray(crop[..., ::-1]), key
        image = (
            aligned.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if row["flip"]
            else aligned
        )
        tensor, _ = preprocess(image, recipe["preprocess"])
        if (
            tensor.shape != (1, 3, 112, 112)
            or tensor.dtype != np.float32
            or not np.isfinite(tensor).all()
        ):
            raise ValueError("Invalid LFW input shape, dtype or finite values")
        feed = {"images": tensor}
        digest = feed_digest(feed)
        expected = reference["feed_digests"][index]
        if digest != expected:
            raise ValueError(
                f"LFW input {index} ({row['image_id']}) differs from reference: "
                f"expected {expected}, got {digest}. Cache was not published."
            )
        yield feed


def prepare_lfw(*, bindings, root, output, selection=None, download=False):
    """Build an exact reference cache from local originals, or validate reuse.

    No ONNX model is executed. An existing output is never overwritten, and a
    new cache is published only after its selected manifest and payload match.
    """
    from ..data.feed_cache import FeedCache, prepare_feed_cache
    from ..data.lfw import _alignment_transform, download_inputs

    resolver = Resolver(bindings)
    descriptor, recipe, samples, reference = _contract(selection, resolver)
    root, output = Path(root).resolve(), Path(output).resolve()
    if output / "manifest.json" != resolver.path(descriptor["cache"]["id"]):
        raise ValueError(
            "--output must match the cache directory in --bindings (lfw-cache for a release bundle)"
        )
    if root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError(
            "LFW source root and output cache must be separate directories"
        )
    _alignment_transform()
    if download:
        download_inputs(root)
    image_root, landmarks, verify_source = _sources(root, samples)
    with cache_lock(output.with_name(output.name + ".prepare.lock")):
        if output.exists():
            resolver.check(descriptor["cache"])
            cache = FeedCache(
                output,
                identity=reference["source_identity"],
                verify_source=verify_source,
            )
            return dict(
                output=str(output),
                state="verified",
                reused=True,
                sample_count=cache.manifest["sample_count"],
                manifest_sha256=sha256(output / "manifest.json"),
                verification_code_identity=code_identity(),
            )
        with tempfile.TemporaryDirectory(
            prefix=output.name + ".building-", dir=output.parent
        ) as temporary:
            staging = Path(temporary) / "cache"
            prepare_feed_cache(
                staging,
                lambda: _feeds(image_root, landmarks, samples, recipe, reference),
                identity=reference["source_identity"],
                verify_source=verify_source,
                expected_samples=reference["sample_count"],
                max_bytes=sum(t["bytes"] for t in reference["tensors"]) + 1024**2,
            )
            if sha256(staging / "manifest.json") != descriptor["cache"]["sha256"]:
                raise ValueError(
                    "Generated LFW cache manifest or tensor file hash differs from reference"
                )
            verify_source()
            proof = dict(
                schema_version=1,
                state="verified",
                created_at=now(),
                code_identity=code_identity(),
                environment=dict(
                    python=platform.python_version(),
                    platform=platform.platform(),
                    packages={
                        name: version(name)
                        for name in (
                            "numpy",
                            "Pillow",
                            "opencv-python-headless",
                            "scikit-image",
                        )
                    },
                ),
                reference=dict(
                    samples=descriptor["samples"], cache=descriptor["cache"]
                ),
                sample_count=reference["sample_count"],
                pair_count=len(samples["pairs"]),
                samples_identity=reference["samples_identity"],
                checks=dict(
                    source_files=True,
                    pair_order=True,
                    original_flip_views=True,
                    shape_dtype_finite=True,
                    feed_digests=True,
                    tensor_files=True,
                    manifest_bytes=True,
                ),
            )
            atomic_json(staging / "verification.json", proof)
            os.rename(staging, output)
    return dict(
        output=str(output),
        state="verified",
        reused=False,
        sample_count=reference["sample_count"],
        manifest_sha256=descriptor["cache"]["sha256"],
        verification_code_identity=proof["code_identity"],
    )
