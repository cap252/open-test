"""Prepare the pinned LFW protocol; failed alignment never becomes a crop."""

from importlib.resources import files
from pathlib import Path
import hashlib
import numpy as np
from .._io import atomic_json, read_json, sha256
from ..evaluation.face_verification import read_pairs

DATASET = "lfw_mobilefacenet"
PAIR_HASH = "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"
LANDMARK_HASH = "a7ac7d5cbec62b79dd8b6c107c69754d59f32dcc0a534a280cff302f9eb6c68b"


def smoke_pairs(pairs):
    """Fixed first positive and first negative pair, chosen before any alignment."""
    return [pairs[0], pairs[300]]


def align_mobilefacenet(image, landmarks):
    """foamliu's 112-square early-return reference points (no inner padding)."""
    import cv2
    from skimage.transform import SimilarityTransform

    source = np.asarray(landmarks, np.float32).reshape(2, 5).T
    target = np.asarray(
        [
            [30.29459953, 51.69630051],
            [65.53179932, 51.50139999],
            [48.02519989, 71.73660278],
            [33.54930115, 92.3655014],
            [62.72990036, 92.20410156],
        ]
    )
    target[:, 0] += 8
    target = target.astype(np.float32)
    if source.shape != (5, 2) or not np.isfinite(source).all():
        raise ValueError("Invalid five-point alignment landmarks")
    fit = SimilarityTransform()
    if not fit.estimate(source, target) or not np.isfinite(fit.params).all():
        raise ValueError("Degenerate face alignment")
    return cv2.warpAffine(image, fit.params[:2], (112, 112), borderValue=0.0)


def download_inputs(root):
    from ..export.download import download, extract_archive

    root.mkdir(parents=True, exist_ok=True)
    download(
        "https://ndownloader.figshare.com/files/5976006",
        root / "pairs.txt",
        sha256_expected=PAIR_HASH,
    )
    base = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/datasets/lfw/v1/"
    archive = download(
        base + "lfw_ds.tar.gz",
        root / "lfw_ds.tar.gz",
        sha256_expected="e14a20c8d7f669a88ce156f80116efb80840c6fa803641a450a1631f23514135",
    )
    if not (root / "data").exists():
        extract_archive(archive, root / "extracted")
        (root / "extracted/data").rename(root / "data")
    download(
        base + "lfw_funneled.json",
        root / "lfw_funneled.json",
        sha256_expected=LANDMARK_HASH,
    )


def prepare_lfw(root, destination, *, smoke=False, download=False):
    import cv2
    from PIL import Image

    root, destination = Path(root).resolve(), Path(destination).resolve()
    if download:
        download_inputs(root)
    if sha256(root / "pairs.txt") != PAIR_HASH:
        raise ValueError("Official LFW pair content changed")
    all_pairs = read_pairs(root / "pairs.txt")
    pairs = smoke_pairs(all_pairs) if smoke else all_pairs
    all_required = {p[k] for p in all_pairs for k in ("first", "second")}
    required = sorted({p[k] for p in pairs for k in ("first", "second")})
    chosen = (
        files("opennpu_quant.resources.splits")
        .joinpath("lfw.txt")
        .read_text()
        .splitlines()
    )
    if smoke:
        chosen = chosen[:2]
    if set(chosen) & all_required:
        raise ValueError("Calibration keys overlap evaluation pairs")
    if sha256(root / "lfw_funneled.json") != LANDMARK_HASH:
        raise ValueError("Pinned landmarks changed")
    landmarks = {}
    for row in read_json(root / "lfw_funneled.json")["samples"]:
        key = Path(row["full_path"]).relative_to("lfw_funneled").as_posix()
        if key in landmarks:
            raise ValueError("Duplicate landmark key")
        landmarks[key] = row["landmarks"]
    image_root = root / "data/lfw_funneled"
    if len(list(image_root.glob("*/*.jpg"))) != 13233:
        raise ValueError("Expected all 13233 source LFW images")
    source_hashes = {
        key: sha256(image_root / key) for key in all_required | set(chosen)
    }
    if {source_hashes[k] for k in chosen} & {source_hashes[k] for k in all_required}:
        raise ValueError("Calibration/evaluation original content overlap")
    output = destination.parent
    output.mkdir(parents=True, exist_ok=True)
    calibration, validation, evidence, failures = [], [], {}, []
    crop_hashes = {"calibration": set(), "validation": set()}
    for split, keys, rows in [
        ("calibration", chosen, calibration),
        ("validation", required, validation),
    ]:
        for key in keys:
            try:
                image = cv2.imread(str(image_root / key))
                if image is None:
                    raise ValueError("Unreadable original image")
                crop = align_mobilefacenet(image, landmarks[key])
                if crop.shape != (112, 112, 3) or crop.dtype != np.uint8:
                    raise ValueError("Invalid aligned crop")
                digest = hashlib.sha256(crop.tobytes()).hexdigest()
                crop_hashes[split].add(digest)
                path = Path("aligned") / Path(key).with_suffix(".png")
                (output / path).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(crop[..., ::-1]).save(output / path)
                evidence[key] = dict(
                    source_sha256=source_hashes[key], crop_sha256=digest
                )
                for flip in (0,) if split == "calibration" else (0, 1):
                    rows.append(
                        dict(
                            id=key + f"|flip={flip}",
                            path=path.as_posix(),
                            image_key=key,
                            flip=flip,
                        )
                    )
            except (ValueError, KeyError) as error:
                failures.append(dict(image_key=key, error=str(error)))
    import skimage

    proof = dict(
        implementation_sha256=sha256(__file__),
        skimage_version=skimage.__version__,
        kind=DATASET,
        smoke=smoke,
        source=dict(
            landmarks_sha256=LANDMARK_HASH, alignment="foamliu supplied five points"
        ),
        pairs_sha256=PAIR_HASH,
        images=evidence,
        failures=failures,
    )
    atomic_json(output / "alignment.json", proof)
    if failures:
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"{len(failures)} LFW alignment failures; see {output / 'alignment.json'}"
        )
    if crop_hashes["calibration"] & crop_hashes["validation"]:
        raise ValueError("Calibration/evaluation aligned content overlap")
    (output / "pairs.txt").write_bytes((root / "pairs.txt").read_bytes())
    result = dict(
        schema_version=1,
        dataset=DATASET,
        root=str(output),
        source="official_lfw",
        smoke=smoke,
        reference_bytes_verified=False,
        expected_samples=2 * len(all_required),
        annotations=dict(pairs="pairs.txt", alignment="alignment.json"),
        calibration=calibration,
        validation=validation,
        evaluation_pairs=pairs,
    )
    atomic_json(destination, result)
    return result


def validate_manifest(info, root):
    """Reject changing the scoring protocol independently of the prepared images."""
    root = Path(root)
    pair_path = (root / info["annotations"]["pairs"]).resolve()
    alignment_path = (root / info["annotations"]["alignment"]).resolve()
    if not pair_path.is_relative_to(root) or not alignment_path.is_relative_to(root):
        raise ValueError("LFW annotation path escapes root")
    if sha256(pair_path) != PAIR_HASH:
        raise ValueError("Official LFW pairs changed")
    pairs = read_pairs(pair_path)
    expected = smoke_pairs(pairs) if info.get("smoke") else pairs
    if info.get("evaluation_pairs") != expected:
        raise ValueError(
            "LFW evaluation pairs/fold order differ from the prepared protocol"
        )
    required = {p[k] for p in expected for k in ("first", "second")}
    views = [(r.get("image_key"), r.get("flip")) for r in info["validation"]]
    if len(views) != 2 * len(required) or set(views) != {
        (k, f) for k in required for f in (0, 1)
    }:
        raise ValueError("LFW manifest is missing/duplicating an image view")
    original_keys = {p[k] for p in pairs for k in ("first", "second")}
    if info["expected_samples"] != 2 * len(original_keys):
        raise ValueError("LFW full evaluation count changed")
    if any(
        r.get("flip") != 0 or r.get("image_key") in original_keys
        for r in info["calibration"]
    ):
        raise ValueError("Invalid/disallowed LFW calibration image view")
    alignment = read_json(alignment_path)
    if (
        alignment.get("failures")
        or alignment["kind"] != info["dataset"]
        or alignment["smoke"] != info.get("smoke", False)
    ):
        raise ValueError("LFW alignment proof does not match the manifest")
