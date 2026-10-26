import os, subprocess, sys
from pathlib import Path
from .._io import read_json


# Load only this package into the selected interpreter. Adding the enclosing
# site-packages to PYTHONPATH would also replace the worker's dependencies.
_WORKER_BOOTSTRAP = """
import importlib.util, runpy, sys
from pathlib import Path
package = Path(sys.argv.pop(1))
spec = importlib.util.spec_from_file_location(
    "opennpu_quant", package / "__init__.py",
    submodule_search_locations=[str(package)],
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
sys.argv[0] = "opennpu_quant.export.worker"
runpy.run_module(sys.argv[0], run_name="__main__")
"""


def export(
    model, paths, *, weights=None, source=None, python=None, validation="strict"
):
    """Run the exporter with this interpreter unless python is supplied."""
    from ..models.spec import ModelSpec, model_names

    if model not in model_names():
        raise ValueError(
            f"Unknown model: {model}. Export supports packaged models: "
            + ", ".join(model_names())
        )
    if validation not in ("strict", "functional"):
        raise ValueError("Unknown export validation mode")
    interpreter = (
        [
            python,
            "-I",
            "-c",
            _WORKER_BOOTSTRAP,
            str(Path(__file__).resolve().parents[1]),
        ]
        if python
        else [sys.executable, "-m", "opennpu_quant.export.worker"]
    )
    command = [
        *interpreter,
        model,
        "--destination",
        str(paths.models / model),
        "--weights-dir",
        str(paths.weights),
    ]
    if validation != "strict":
        command += ["--validation", validation]
    if weights:
        command += ["--weights", str(weights)]
    if source:
        command += ["--source-dir", str(source)]
    if (
        ModelSpec.load(model).recipe["export"].get("fixtures")
        == "voc_validation_first3"
    ):
        root = paths.datasets.get("voc")
        manifest = paths.data / "voc/manifest.json"
        if root is None and manifest.exists():
            root = read_json(manifest)["root"]
        if root is None:
            raise ValueError(
                "Prepare VOC data before exporting DeepLab; its preserved parity fixtures are the first three VOC validation images"
            )
        command += ["--voc-root", str(root)]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        TF_CPP_MIN_LOG_LEVEL="2",
        YOLOv5_AUTOINSTALL="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    cache = paths.cache / "export"
    cache.mkdir(parents=True, exist_ok=True)
    env.update(
        OPENNPU_EXPORT_CACHE=str(cache.resolve()),
        MPLCONFIGDIR=str(cache / "matplotlib"),
        YOLOV5_CONFIG_DIR=str(cache / "yolo"),
        TORCH_HOME=str(paths.weights / "torch"),
    )
    try:
        subprocess.run(command, env=env, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Export of {model} failed (exit code {error.returncode}); "
            "see the exporter messages above"
        ) from None
    return read_json(paths.models / model / "model.json")
