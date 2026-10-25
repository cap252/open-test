"""Optional, pinned framework loaders for the extended model catalog.

Imports stay inside the export worker; model evaluation does not import upstream
checkouts. Checkpoint/source verification is performed before these loaders run.
"""

import importlib.util
import inspect
import os
from pathlib import Path
import sys
import tempfile


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build(recipe, weight, source):
    import torch

    family = recipe["export"]["family"]
    transform = None
    if family == "timm":
        import timm
        from safetensors.torch import load_file

        model = timm.create_model(recipe["source"]["builder"], pretrained=False)
        model.load_state_dict(load_file(str(weight)), strict=True)
        cfg = timm.data.resolve_model_data_config(
            model, use_test_size=recipe["name"] == "mobilenet_v1"
        )
        transform = timm.data.create_transform(**cfg, is_training=False)
    elif family == "mobilefacenet_relu":
        cls = _module(
            source / "mobilefacenet.py", "opennpu_mobilefacenet_source"
        ).MobileFaceNet
        model = cls()
        model.load_state_dict(
            torch.load(weight, map_location="cpu", weights_only=True), strict=True
        )
        if any(isinstance(m, torch.nn.PReLU) for m in model.modules()):
            raise ValueError("Unexpected PReLU in the ReLU MobileFaceNet variant")
    elif family == "retinanet":
        from torchvision.models.detection import retinanet_resnet50_fpn

        base = retinanet_resnet50_fpn(weights=None, weights_backbone=None)
        base.load_state_dict(
            torch.load(weight, map_location="cpu", weights_only=True), strict=True
        )

        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = base

            def forward(self, images):
                out = self.model([images[0]])[0]
                return out["boxes"], out["labels"], out["scores"]

        model = Detector()
    elif family in ("yolov3", "yolov7"):
        # Rank 1 skips the legacy visualization-only font download at import.
        if family == "yolov3":
            os.environ["YOLOv5_AUTOINSTALL"] = "false"
            os.environ["RANK"] = "1"
        from models.experimental import attempt_load

        kwargs = (
            {"device": torch.device("cpu")}
            if "device" in inspect.signature(attempt_load).parameters
            else {"map_location": "cpu"}
        )
        if family == "yolov3":
            kwargs.update(inplace=False, fuse=True)
        base = attempt_load(str(weight), **kwargs)
        if recipe["name"] in ("yolov3_full", "yolov7_full"):
            for module in base.modules():
                if hasattr(module, "onnx_dynamic"):
                    module.onnx_dynamic = True
        if family == "yolov7":
            for module in base.modules():
                if hasattr(module, "export"):
                    module.export = False
                if hasattr(module, "concat"):
                    module.concat = True

        class First(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = base

            def forward(self, x):
                value = self.model(x)
                return value[0] if isinstance(value, (tuple, list)) else value

        model = First()
    elif family == "yolov4":
        from tool.darknet2pytorch import Darknet
        from .download import download

        cfg = recipe["source"]["config"]
        path = download(
            cfg["url"],
            Path(os.environ.get("OPENNPU_EXPORT_CACHE", tempfile.gettempdir()))
            / recipe["name"]
            / cfg["file"],
            sha256_expected=cfg["sha256"],
        )
        model = Darknet(str(path), inference=True)
        expected = 0
        for index, block in enumerate(model.blocks[1:]):
            if block["type"] == "convolutional":
                layer = model.models[index]
                expected += layer[0].weight.numel()
                expected += (
                    4 * layer[1].weight.numel()
                    if int(block.get("batch_normalize", 0))
                    else layer[0].bias.numel()
                )
        if Path(weight).stat().st_size != 20 + 4 * expected:
            raise ValueError("Incomplete Darknet parameter payload")
        model.load_weights(str(weight))
    elif family == "ultralytics":
        from ultralytics import YOLO

        model = YOLO(str(weight)).model
        for module in model.modules():
            if module.__class__.__name__ == "Detect":
                module.export, module.format, module.dynamic = True, "onnx", False
    elif family == "nanodet":
        from nanodet.util import cfg, load_config
        from nanodet.model.arch import build_model

        load_config(cfg, str(source / "config/nanodet-plus-m_320.yml"))
        model = build_model(cfg.model)
        checkpoint = torch.load(weight, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"].copy()
        for key in list(state):
            if key.startswith("avg_model."):
                state[key[4:]] = state.pop(key)
        for prefix in ("module.", "model."):
            if next(iter(state)).startswith(prefix):
                state = {k.removeprefix(prefix): v for k, v in state.items()}
        expected = model.state_dict()
        missing = [
            k
            for k in expected
            if k not in state and not k.endswith("num_batches_tracked")
        ]
        if missing:
            raise ValueError("Incomplete NanoDet checkpoint: " + str(missing))
        model.load_state_dict(
            {k: state.get(k, v) for k, v in expected.items()}, strict=True
        )
        model.head.forward = model.head._forward_onnx
    elif family == "posenet":
        from posenet import MobileNetV1

        model = MobileNetV1(101, output_stride=16)
        if Path(weight).suffix == ".json":
            from posenet.converter.tfjs2pytorch import load_variables

            state = load_variables(
                "mobilenet_v1_101", base_dir=str(Path(weight).parent.parent)
            )
        else:
            state = torch.load(weight, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    elif family == "unet":
        from nets.unet import Unet

        model = Unet(num_classes=21, backbone="resnet50", pretrained=False)
        model.load_state_dict(
            torch.load(weight, map_location="cpu", weights_only=True), strict=True
        )
    elif family == "mmseg":
        from mmseg.apis import init_model

        filename = recipe["export"].get("config") or (
            "deeplabv3/deeplabv3_r50-d8_4xb4-40k_voc12aug-512x512.py"
            if recipe["name"].startswith("deeplab")
            else "fcn/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py"
        )
        base = init_model(str(source / "configs" / filename), str(weight), device="cpu")

        class Segmenter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = base

            def forward(self, x):
                return self.model.decode_head.forward(self.model.extract_feat(x))

        model = Segmenter()
    elif family == "yolox":
        from yolox.exp import get_exp

        exp = get_exp(str(source / "exps/default/yolox_s.py"), None)
        model = exp.get_model()
        model.load_state_dict(
            torch.load(weight, map_location="cpu", weights_only=True)["model"],
            strict=True,
        )
        model.head.decode_in_inference = True
    elif family in ("yolov9", "yolov7_face", "yolov5_seg"):
        if family == "yolov7_face":
            import alfred
            from alfred.deprecated.dl.torch.common import print_shape

            # Upstream imports the old public spelling (only for commented debug calls).
            alfred.print_shape = print_shape
        state = torch.load(weight, map_location="cpu", weights_only=False)
        base = (state.get("ema") or state["model"]).float().eval()
        if hasattr(base, "fuse"):
            base = base.fuse()
        for module in base.modules():
            if hasattr(module, "dynamic"):
                module.dynamic = True
            if hasattr(module, "onnx_dynamic"):
                module.onnx_dynamic = True

        class Yolo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = base

            def forward(self, x):
                out = self.base(x)
                if family == "yolov5_seg":
                    return out[0], out[1]
                return out[0] if isinstance(out, (tuple, list)) else out

        model = Yolo()
    else:
        raise ValueError("Unknown Torch exporter: " + family)
    return model.float().eval(), transform


def static_same(model, feeds):
    """Replace timm dynamic SAME padding with its fixed-shape export variant."""
    import copy
    import torch
    from timm.layers.conv2d_same import Conv2dSame, Conv2dSameExport

    result = copy.deepcopy(model)
    for name, module in list(result.named_modules()):
        if isinstance(module, Conv2dSame):
            new = Conv2dSameExport(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
            )
            new.load_state_dict(module.state_dict(), strict=True)
            parent, _, child = name.rpartition(".")
            setattr(result.get_submodule(parent) if parent else result, child, new)
    result.eval()
    with torch.no_grad():
        for feed in feeds:
            x = torch.from_numpy(feed["images"])
            torch.testing.assert_close(result(x), model(x), rtol=0, atol=0)
    return result


def fixed_nearest_upsample(model, feeds):
    """Export integer nearest-neighbor upsampling without float shape arithmetic."""
    import copy
    import torch

    result = copy.deepcopy(model)
    changed = 0
    for module in result.modules():
        if isinstance(module, torch.nn.Upsample):
            scale = module.scale_factor
            factors = (scale,) if isinstance(scale, (int, float)) else scale
            if (
                module.mode != "nearest"
                or not factors
                or any(float(v) != int(v) for v in factors)
            ):
                raise ValueError(
                    "Only integer nearest upsampling can use fixed export scales"
                )
            module.recompute_scale_factor = False
            changed += 1
    if not changed:
        raise ValueError("Expected an upstream nearest upsample")
    with torch.no_grad():
        for feed in feeds:
            x = torch.from_numpy(feed["images"])
            torch.testing.assert_close(result(x), model(x), rtol=0, atol=0)
    return result


def dynamic_face_grids(model, feeds, head_types=("IKeypoint",)):
    """Rebuild legacy grids on every forward so tracing retains their shapes."""
    import copy
    import torch

    result = copy.deepcopy(model)

    def clear_grid(module, inputs):
        # The pinned head has no dynamic/onnx_dynamic switch. A warm grid would
        # otherwise become an ONNX constant when tracing the same input shape.
        module.grid = [inputs[0][0].new_zeros(1) for _ in range(module.nl)]

    heads = [m for m in result.modules() if m.__class__.__name__ in head_types]
    if len(heads) != 1:
        raise ValueError("Expected one legacy YOLO grid head")
    heads[0].register_forward_pre_hook(clear_grid)
    with torch.no_grad():
        for feed in feeds:
            x = torch.from_numpy(feed["images"])
            torch.testing.assert_close(result(x), model(x), rtol=0, atol=0)
    return result
