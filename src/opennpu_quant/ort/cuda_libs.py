import ctypes
import importlib.metadata
import sysconfig
from pathlib import Path

_handles = []
_loaded = False


def load_cuda_libraries():
    global _loaded
    if _loaded:
        return
    base = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    # ORT 1.20.2 links the split cuDNN libraries, not just libcudnn.so.
    # No LD_LIBRARY_PATH borrowed from a different environment is needed.
    names = (
        "cuda_runtime/lib/libcudart.so.12",
        "nvjitlink/lib/libnvJitLink.so.12",
        "cublas/lib/libcublasLt.so.12",
        "cublas/lib/libcublas.so.12",
        "cufft/lib/libcufft.so.11",
        "curand/lib/libcurand.so.10",
        "cuda_nvrtc/lib/libnvrtc.so.12",
        "cudnn/lib/libcudnn.so.9",
        "cudnn/lib/libcudnn_graph.so.9",
        "cudnn/lib/libcudnn_ops.so.9",
        "cudnn/lib/libcudnn_adv.so.9",
        "cudnn/lib/libcudnn_cnn.so.9",
        "cudnn/lib/libcudnn_engines_precompiled.so.9",
        "cudnn/lib/libcudnn_engines_runtime_compiled.so.9",
        "cudnn/lib/libcudnn_heuristic.so.9",
    )
    paths = {name: wheel_library(name, base) for name in names}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "CUDA wheel libraries are missing in this interpreter. "
            "Use the CUDA setup profile: " + ", ".join(missing)
        )
    for name in names:
        _handles.append(ctypes.CDLL(str(paths[name]), mode=ctypes.RTLD_GLOBAL))
    _loaded = True


def wheel_library(relative, base):
    """Resolve the active wheel as well as conventional site-packages layouts."""
    local = base / relative
    if local.is_file():
        return local
    family = relative.split("/", 1)[0].replace("_", "-")
    try:
        distribution = importlib.metadata.distribution("nvidia-" + family + "-cu12")
    except importlib.metadata.PackageNotFoundError:
        return local
    return Path(distribution.locate_file("nvidia/" + relative))


def loaded_libraries():
    """Actual process mappings, distinct from installed or eagerly loadable wheels."""
    return _mapped_libraries(
        (
            "libcudart",
            "libcublas",
            "libcudnn",
            "libnvrtc",
            "libnvJitLink",
            "libonnxruntime_providers_cuda",
        )
    )


def loaded_driver_libraries():
    """Driver mappings are separate from the pinned CUDA wheel libraries."""
    return _mapped_libraries(
        ("libcuda.so", "libnvidia-ml.so", "libnvidia-ptxjitcompiler.so")
    )


def _mapped_libraries(families):
    from .._io import sha256

    maps = Path("/proc/self/maps")
    if not maps.exists():
        return {"available": False, "libraries": [], "not_loaded": list(families)}
    paths = set()
    for line in maps.read_text().splitlines():
        parts = line.split(maxsplit=5)
        if (
            len(parts) == 6
            and parts[5].startswith("/")
            and any(n in Path(parts[5]).name for n in families)
        ):
            paths.add(parts[5])
    libraries = []
    for name in sorted(paths):
        path = Path(name)
        libraries.append(
            dict(
                name=path.name,
                sha256=sha256(path) if path.is_file() else None,
                state="loaded" if path.is_file() else "loaded_file_unavailable",
            )
        )
    return dict(
        available=True,
        libraries=libraries,
        not_loaded=[
            n for n in families if not any(r["name"].startswith(n) for r in libraries)
        ],
    )


def compare_libraries(actual, expected):
    """A missing mapping is unverified, not evidence of changed library contents."""

    def grouped(value):
        result = {}
        for row in value["libraries"]:
            result.setdefault(row["name"], set()).add(row["sha256"])
        return result

    left, right = grouped(actual), grouped(expected)
    return {
        name: "not_loaded"
        if name not in left or name not in right
        else "unverifiable"
        if None in left[name] or None in right[name]
        else "content_changed"
        if left[name] != right[name]
        else "unverifiable"
        if len(left[name]) != 1
        else "matched"
        for name in sorted(left.keys() | right.keys())
    }
