from pathlib import Path
import shutil, subprocess, sys
import inspect
import numpy as np
from .._io import sha256
from .download import download


def checkpoint(recipe, directory, explicit=None):
    source = recipe["source"]
    if explicit:
        path = Path(explicit).resolve()
        if sha256(path) != source["sha256"]:
            raise ValueError("Original checkpoint checksum mismatch")
        return path
    if source.get("parts"):
        folder = Path(directory) / recipe["name"] / "mobilenet_v1_101"
        for part in source["parts"]:
            download(
                part["url"],
                folder / Path(part["url"]).name,
                sha256_expected=part["sha256"],
            )
        return folder / "manifest.json"
    filename = source.get("file") or Path(source["url"].split("?")[0]).name
    return download(
        source["url"],
        Path(directory) / recipe["name"] / filename,
        sha256_expected=source["sha256"],
    )


def checkout(recipe, directory, explicit=None):
    source = recipe["source"]
    if "archive_sha256" in source:
        from .source_archive import checkout_archive

        return checkout_archive(recipe, directory, explicit)
    commit = source["commit"]
    target = (
        Path(explicit).resolve()
        if explicit
        else Path(directory) / "sources" / recipe["name"] / commit
    )
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--no-checkout", source["repo"], str(target)],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(target), "checkout", "--detach", commit], check=True
            )
        except BaseException:
            # A partial checkout would make every later export fail.
            shutil.rmtree(target, ignore_errors=True)
            raise
    current = subprocess.check_output(
        ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
    ).strip()
    if current != commit:
        raise ValueError("Upstream source commit mismatch")
    changed = subprocess.check_output(
        ["git", "-C", str(target), "diff", "--name-only", "HEAD"], text=True
    ).strip()
    if changed:
        raise ValueError("Upstream tracked source has local changes")
    from .source_archive import verify_source_files

    tracked = subprocess.check_output(
        ["git", "-C", str(target), "ls-files", "-z"], text=True
    ).split("\0")
    verify_source_files(target, set(tracked))
    sys.path.insert(0, str(target))
    return target


def build(recipe, weight, source):
    import torch

    torch.set_num_threads(1)
    family = recipe["export"]["family"]
    transform = None
    if family == "torchvision":
        import torchvision

        name = recipe["source"]["builder"]
        kwargs = recipe["export"].get("kwargs", {})
        model = torchvision.models.get_model(name, weights=None, **kwargs)
        state = torch.load(weight, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        enum = torchvision.models.get_model_weights(name)
        transform = getattr(enum, recipe["source"]["weights"]).transforms()
    elif family == "retinaface":
        from models.retinaface import RetinaFace
        from data import cfg_mnet

        cfg = dict(cfg_mnet, pretrain=False)
        model = RetinaFace(cfg=cfg, phase="test")
        state = torch.load(weight, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(
            {k.removeprefix("module."): v for k, v in state.items()}, strict=True
        )
    elif family == "yolov5":
        from models.experimental import attempt_load

        # Official checkpoint contains framework objects. Its pinned SHA256 is verified first.
        base = attempt_load(
            str(weight), device=torch.device("cpu"), inplace=False, fuse=True
        )

        class First(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                value = self.model(x)
                return value[0] if isinstance(value, (tuple, list)) else value

        model = First(base)
    else:
        from .extended_torch import build as extended_build

        return extended_build(recipe, weight, source)
    return model.float().eval(), transform


def export_model(recipe, weight, source, feeds, target, *, fixture_images=None):
    import torch
    from .fixtures import synthetic_image
    from ..models.preprocess import preprocess

    model, transform = build(recipe, weight, source)
    if transform:
        crop = recipe["preprocess"]["crop"]
        for i, (w, h) in enumerate([(crop + 83, crop + 47), (321, 479), (479, 321)]):
            image = synthetic_image(i, w, h)
            actual, _ = preprocess(image, recipe["preprocess"])
            expected = transform(image).unsqueeze(0).numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    names = (
        recipe["export"].get("outputs")
        or {
            "classification": ["logits"],
            "detection": ["predictions"],
            "face": ["loc", "conf", "landmarks"],
            "segmentation": ["logits"],
            "keypoints": [
                "heatmaps",
                "offsets",
                "displacements_fwd",
                "displacements_bwd",
            ],
            "instance_segmentation": ["predictions", "prototypes"],
        }[recipe["task"]]
    )
    references = []
    with torch.inference_mode():
        for feed in feeds:
            output = model(torch.from_numpy(feed["images"]))
            values = list(output) if isinstance(output, (tuple, list)) else [output]
            references.append({n: v.detach().numpy() for n, v in zip(names, values)})
        if recipe["export"]["family"] == "mmseg" and fixture_images is not None:
            from .mmseg_protocol import check
            from .._io import atomic_json

            atomic_json(
                Path(target).with_suffix(".protocol.json"),
                check(model.model, recipe, fixture_images, feeds, references),
            )
        if recipe["export"]["family"] == "timm":
            from .extended_torch import static_same

            model = static_same(model, feeds)
        if recipe["export"]["family"] == "yolov7_face":
            from .extended_torch import dynamic_face_grids

            model = dynamic_face_grids(model, feeds)
        if recipe["name"] == "yolov7_full":
            from .extended_torch import dynamic_face_grids

            model = dynamic_face_grids(model, feeds, ("Detect", "IDetect"))
        dynamic = None
        if recipe["export"].get("dynamic_spatial"):
            dynamic = {"images": {2: "height", 3: "width"}}
            for name in names:
                dynamic[name] = (
                    {2: name + "_height", 3: name + "_width"}
                    if name in ("logits", "prototypes")
                    else {2 if recipe["export"]["family"] == "yolov9" else 1: "anchors"}
                )
        torch.onnx.export(
            model,
            torch.from_numpy(feeds[0]["images"]),
            str(target),
            opset_version=recipe["export"]["opset"],
            input_names=["images"],
            output_names=names,
            **(
                {"dynamo": False}
                if "dynamo" in inspect.signature(torch.onnx.export).parameters
                else {}
            ),
            dynamic_axes=dynamic,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )
    return references
