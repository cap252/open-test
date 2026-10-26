"""CPU numerical dispatch identity without host names or installation paths."""

from functools import lru_cache
import os
from pathlib import Path
import platform


@lru_cache(maxsize=1)
def _implementation():
    import numpy as np
    from .._io import sha256

    core = getattr(np, "_core", None) or np.core
    binary = Path(core._multiarray_umath.__file__)
    # The extension and bundled BLAS implementation distinguish builds with the
    # same version. Cache immutable installation files, not mutable dispatch flags.
    libraries = sorted(binary.parents[2].glob("numpy.libs/*"))
    contents = [sha256(binary)] + [sha256(p) for p in libraries if p.is_file()]
    cpu = Path("/proc/cpuinfo")
    fields = {
        "vendor_id",
        "cpu family",
        "model",
        "stepping",
        "model name",
        "CPU implementer",
        "CPU part",
    }
    identities = set()
    if cpu.is_file():
        for section in cpu.read_text().split("\n\n"):
            values = tuple(
                sorted(
                    (k.strip(), v.strip())
                    for line in section.splitlines()
                    if ":" in line
                    for k, v in [line.split(":", 1)]
                    if k.strip() in fields
                )
            )
            if values:
                identities.add(values)
    return dict(
        machine=platform.machine(),
        cpu=[dict(v) for v in sorted(identities)],
        libc=list(platform.libc_ver()),
        numpy_binary_sha256=contents,
    )


def cpu_environment():
    import numpy as np

    core = getattr(np, "_core", None) or np.core
    features = getattr(core._multiarray_umath, "__cpu_features__", None)
    flags = (
        "NPY_DISABLE_CPU_FEATURES",
        "OPENBLAS_CORETYPE",
        "OPENBLAS_NUM_THREADS",
        "MKL_CBWR",
        "MKL_ENABLE_INSTRUCTIONS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "MKL_DYNAMIC",
    )
    return dict(
        schema_version=1,
        implementation=_implementation(),
        features=None
        if features is None
        else {k: bool(v) for k, v in sorted(features.items())},
        dispatch={key: os.environ.get(key) for key in flags},
    )
