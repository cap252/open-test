from pathlib import Path
import argparse, tempfile, sys
from .._io import atomic_json, sha256, now
from .._locking import RunLock
from ..models.spec import ModelSpec
from ..models.preprocess import preprocess
from .fixtures import synthetic_image
from .torch_models import checkpoint, checkout
from .finalize import finalize
from .parity import diagnostic_lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--validation", choices=("strict", "functional"), default="strict")
    p.add_argument("--destination", required=True)
    p.add_argument("--weights-dir", required=True)
    p.add_argument("--weights")
    p.add_argument("--source-dir")
    p.add_argument("--voc-root")
    a = p.parse_args()
    recipe = ModelSpec.load(a.model).recipe
    pin = recipe["source"].get("package")
    if pin:
        from importlib.metadata import version

        installed = version(pin["name"]).split("+")[0]
        if installed != pin["version"]:
            raise ValueError(
                f"Pinned exporter requires {pin['name']}=={pin['version']}; found {installed}"
            )
    destination = Path(a.destination)
    destination.mkdir(parents=True, exist_ok=True)
    with RunLock(destination / ".lock"):
        weight = checkpoint(recipe, a.weights_dir, a.weights)
        source = (
            checkout(recipe, a.weights_dir, a.source_dir)
            if "repo" in recipe["source"]
            else None
        )
        shape = recipe["input"]["shape"]
        size = recipe["export"].get("fixture_size") or (
            shape[2] if recipe["input"]["layout"] == "NCHW" else shape[1]
        )
        feeds = []
        images = []
        for index in range(3):
            if recipe["export"].get("fixtures") == "voc_validation_first3":
                if not a.voc_root:
                    raise ValueError("VOC parity images required")
                from PIL import Image

                names = (
                    (Path(a.voc_root) / "ImageSets/Segmentation/val.txt")
                    .read_text()
                    .split()[:3]
                )
                with Image.open(
                    Path(a.voc_root) / "JPEGImages" / (names[index] + ".jpg")
                ) as opened:
                    image = opened.convert("RGB")
            else:
                w, h = (
                    (size + 83, size + 47)
                    if recipe["task"] == "classification"
                    else (size + 53, size + 21)
                )
                if recipe["task"] == "face_verification":
                    w, h = 112, 112
                if recipe["export"].get("dynamic_spatial"):
                    # Near-square images can all become the same shape after
                    # stride-aligned letterboxing; exercise square and both axes.
                    w, h = [(size, size), (size, size // 2), (size // 2, size)][index]
                image = synthetic_image(index, w, h)
            value, _ = preprocess(
                image, recipe["preprocess"], recipe["input"]["layout"]
            )
            if recipe["name"] == "yolox" and index:
                import numpy as np

                # Runtime padding stays square. Also exercise the released graph's
                # dynamic spatial axes by removing only the fixture's trailing pad.
                value = np.ascontiguousarray(value[:, :, :h, :w])
            feeds.append({"images": value})
            images.append(image)
        if recipe["export"].get("dynamic_spatial"):
            shapes = {feed["images"].shape for feed in feeds}
            if len(shapes) != len(feeds):
                raise ValueError(
                    "Dynamic export fixtures must have distinct input shapes"
                )
        with tempfile.TemporaryDirectory(prefix="export_", dir=destination) as scratch:
            raw = Path(scratch) / "raw.onnx"
            if recipe["export"]["family"].startswith("tf_"):
                from .tensorflow_models import export_model

                references = export_model(recipe, weight, feeds, raw, scratch)
            else:
                from .torch_models import export_model

                references = export_model(
                    recipe, weight, source, feeds, raw, fixture_images=images
                )
            result = finalize(
                raw, recipe, feeds, references, Path(scratch), validation=a.validation
            )
            result["export_settings"] = recipe["export"]
            protocol = raw.with_suffix(".protocol.json")
            if protocol.exists():
                from .._io import read_json

                result["independent_protocol"] = read_json(protocol)
            for name in recipe["graphs"].values():
                (Path(scratch) / name).replace(destination / name)
        from ..envcheck import info

        atomic_json(
            destination / "model.json",
            dict(
                schema_version=1,
                name=recipe["name"],
                source=recipe["source"],
                checkpoint_sha256=sha256(weight),
                files={
                    name: sha256(destination / name)
                    for name in recipe["graphs"].values()
                },
                network_input_map=result["network_input_map"],
                network_outputs=result["network_outputs"],
                validation=result,
                environment=info(),
                created=now(),
            ),
        )

        for line in diagnostic_lines(recipe["name"], result):
            print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
