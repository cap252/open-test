from pathlib import Path
import numpy as np
from ..ort.config import OrtConfig
from ..ort.session import create_session
from ..ort.fetch import run_tensors
from .._io import read_json, sha256


class ModelBundle:
    """network.onnx plus optional CPU host graphs, and the network input mapping.

    from_directory() opens an exported model folder. Explicit graph paths, as reproduce
    passes them, read no model.json unless one is given. prepare() turns a preprocessed
    image into network feeds. finish() runs the postprocess graph on the outputs and
    adds the output aliases.
    """

    def __init__(self, spec, paths, *, manifest=None, aliases=None):
        self.spec = spec
        self.paths = {k: Path(v) for k, v in paths.items()}
        missing = [str(v) for v in self.paths.values() if not v.is_file()]
        if missing:
            raise FileNotFoundError(
                "Export " + spec.name + " first. Missing: " + ", ".join(missing)
            )
        manifest = Path(manifest) if manifest is not None else None
        recorded = read_json(manifest) if manifest and manifest.exists() else None
        self.validation = (recorded or {}).get("validation", {})
        if recorded is not None:
            for path in self.paths.values():
                if recorded["files"].get(path.name) != sha256(path):
                    raise ValueError("Model bundle changed: " + path.name)
        self.pre = (
            create_session(self.paths["preprocess"], OrtConfig.cpu())
            if "preprocess" in self.paths
            else None
        )
        self.post = (
            create_session(self.paths["postprocess"], OrtConfig.cpu())
            if "postprocess" in self.paths
            else None
        )
        default = spec.recipe.get("network_input_map", {"images": "images"})
        self.mapping = (
            recorded.get("network_input_map", default)
            if recorded is not None
            else default
        )
        self.aliases = dict(aliases or {})

    @classmethod
    def from_directory(cls, spec, directory):
        """An exported model folder, checked against its model.json when present."""
        directory = Path(directory)
        return cls(
            spec,
            {k: directory / v for k, v in spec.recipe["graphs"].items()},
            manifest=directory / "model.json",
        )

    def prepare(self, value, metadata):
        name = self.spec.recipe["input"]["name"]
        context = {name: value}
        if self.pre:
            names = [v.name for v in self.pre.get_outputs()]
            context.update(
                zip(
                    names,
                    run_tensors(
                        self.pre,
                        names,
                        {v.name: context[v.name] for v in self.pre.get_inputs()},
                    ),
                )
            )
        result = {name: context[source] for name, source in self.mapping.items()}
        if any(x.dtype != np.float32 for x in result.values()):
            raise ValueError("Network requires FP32 features")
        return result, dict(metadata, context=context)

    def finish(self, outputs, metadata):
        """Host outputs as a new dict. The caller's outputs are not modified."""
        if self.post:
            context = {**metadata["context"], **outputs}
            names = [v.name for v in self.post.get_outputs()]
            outputs = dict(
                zip(
                    names,
                    run_tensors(
                        self.post,
                        names,
                        {v.name: context[v.name] for v in self.post.get_inputs()},
                    ),
                )
            )
        return {**outputs, **{k: outputs[v] for k, v in self.aliases.items()}}
