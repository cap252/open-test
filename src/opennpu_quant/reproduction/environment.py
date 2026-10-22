"""Record installed versions separately from libraries mapped by this process."""

import copy
import importlib.metadata
import platform
import os
import subprocess
from pathlib import Path

from ..ort.cuda_libs import loaded_libraries, loaded_driver_libraries, compare_libraries

EXPECTED = {
    "onnxruntime-gpu": "1.20.2",
    "nvidia-cuda-runtime-cu12": "12.4.127",
    "nvidia-cublas-cu12": "12.4.5.8",
    "nvidia-cudnn-cu12": "9.1.0.70",
}


def capture(*, libraries=False):
    packages = {}
    for name in (
        *EXPECTED,
        "numpy",
        "onnx",
        "torch",
        "torchvision",
        "pillow",
        "opencv-python-headless",
        "pycocotools",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu = dict(
            available=proc.returncode == 0,
            devices=proc.stdout.strip().splitlines(),
            diagnostic=(proc.stderr + proc.stdout).strip() if proc.returncode else None,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu = dict(available=False, diagnostic=str(error), devices=[])
    result = dict(
        packages=packages,
        python=platform.python_version(),
        system=platform.system(),
        machine=platform.machine(),
        cpu=platform.processor(),
        libc=platform.libc_ver(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu=gpu,
        kernel_driver=(
            Path("/proc/driver/nvidia/version").read_text().strip()
            if Path("/proc/driver/nvidia/version").is_file()
            else None
        ),
    )
    result["version_mismatches"] = {
        k: dict(expected=v, actual=packages[k])
        for k, v in EXPECTED.items()
        if packages[k] != v
    }
    if libraries:
        result["loaded_libraries"] = loaded_libraries()
        result["driver_libraries"] = loaded_driver_libraries()
    return result


def compare(actual, expected):
    gpu = actual["device"].startswith("cuda") and expected["device"].startswith("cuda")
    mismatches = []

    def normalized(value):
        value = copy.deepcopy(value)
        if isinstance(value, dict):
            value["provider_options"] = [
                [item for item in options if item[0] != "device_id"]
                for options in value.get("provider_options", [])
            ]
        return value

    for key in ("packages", "execution", "implementation", "python"):
        left, right = actual.get(key), expected.get(key)
        if key == "execution":
            left, right = normalized(left), normalized(right)
        if left != right:
            mismatches.append(key)
    libraries = compare_libraries(
        actual.get("loaded_libraries", {"libraries": []}),
        expected.get("loaded_libraries", {"libraries": []}),
    )
    if (
        "content_changed" in libraries.values()
        or gpu
        and bool(actual.get("version_mismatches"))
        or gpu
        and bool(expected.get("version_mismatches"))
    ):
        mismatches.append("CUDA versions/library contents")
    required = ("libcudart", "libcublas", "libcudnn", "libonnxruntime_providers_cuda")
    missing = {
        side: [
            prefix
            for prefix in required
            if not any(
                row["name"].startswith(prefix)
                for row in env.get("loaded_libraries", {}).get("libraries", [])
            )
        ]
        for side, env in (("actual", actual), ("expected", expected))
    }
    unverified = (
        not libraries
        or any(missing.values())
        or any(v in ("not_loaded", "unverifiable") for v in libraries.values())
    )
    return dict(
        state="fail" if mismatches else "pending" if gpu and unverified else "pass",
        mismatches=mismatches,
        libraries=libraries,
        not_loaded=missing,
        gpu_comparison=gpu,
    )


def probe_gpu(device=0):
    """A real Conv must execute on CUDA; installed providers alone are insufficient."""
    import tempfile
    from pathlib import Path
    import numpy as np
    import onnx
    from ..ort.config import OrtConfig
    from ..ort.session import create_session
    from ..ort.profile import summarize_profile, assert_cuda_observed
    from .._io import read_json

    h, t = onnx.helper, onnx.TensorProto
    graph = h.make_graph(
        [h.make_node("Conv", ["x", "w"], ["y"])],
        "probe",
        [h.make_tensor_value_info("x", t.FLOAT, [1, 1, 2, 2])],
        [h.make_tensor_value_info("y", t.FLOAT, [1, 1, 2, 2])],
        [h.make_tensor("w", t.FLOAT, [1, 1, 1, 1], [1])],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10)
    try:
        with tempfile.TemporaryDirectory() as temp:
            session = create_session(
                model,
                OrtConfig.cuda(device, strict_versions=True),
                profile=Path(temp) / "probe",
            )
            session.run(None, {"x": np.ones((1, 1, 2, 2), np.float32)})
            placement = summarize_profile(read_json(session.end_profiling()))
            assert_cuda_observed(session.get_providers(), placement)
        device_identity = None
        try:
            import torch

            properties = torch.cuda.get_device_properties(device)
            device_identity = dict(
                name=properties.name,
                uuid=str(getattr(properties, "uuid", "")),
                capability=[properties.major, properties.minor],
            )
        except (ImportError, RuntimeError, AssertionError):
            pass
        return dict(
            available=True, placement=placement, device_identity=device_identity
        )
    except Exception as error:
        return dict(available=False, diagnostic=str(error))
