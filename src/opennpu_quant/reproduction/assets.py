"""Portable artifact IDs and content identities for frozen inference."""

from importlib.resources import files
from pathlib import Path
import gzip
import json
import re

from .._io import read_json, sha256, object_hash, atomic_json


MODEL_SETS = ("core10", "extended", "all")


def catalog(path=None):
    return read_json(
        path or files("opennpu_quant.resources.reproduction") / "selected.json"
    )


def selected(entries, models=None, conditions=None):
    for field, requested in (("model", models), ("condition", conditions)):
        if requested and set(requested) - {e[field] for e in entries}:
            raise ValueError("Unknown " + field + ": " + str(requested))
    result = [
        e
        for e in entries
        if (not models or e["model"] in models)
        and (not conditions or e["condition"] in conditions)
    ]
    if not result:
        raise ValueError("No selected conditions")
    for entry in result:
        if not re.fullmatch(r"[a-z0-9_]+", entry["id"]):
            raise ValueError("Unsafe condition ID")
    return result


def select_catalog(*, selection=None, model_set=None, models=None, conditions=None):
    """Select new work without narrowing existing prepared packages."""
    if model_set is not None and models is not None:
        raise ValueError("Specify model_set or models, not both")
    if model_set is not None and model_set not in MODEL_SETS:
        raise ValueError("Unknown model set: " + str(model_set))
    if models is not None and not models:
        raise ValueError("models must contain at least one model name")
    if models is None:
        group = model_set or ("all" if selection is not None else "extended")
        if group != "all":
            from ..configuration import load_config

            models = load_config(group)["models"]
    source = catalog(selection)
    return source, selected(source["entries"], models, conditions)


class Resolver:
    def __init__(self, bindings):
        self.roots = (
            read_json(bindings) if isinstance(bindings, (str, Path)) else bindings
        )

    def path(self, uri):
        if not uri.startswith("artifact://"):
            raise ValueError("Expected artifact:// ID")
        root, _, relative = uri[11:].partition("/")
        if root not in self.roots:
            raise FileNotFoundError("Missing artifact binding: " + root)
        base = Path(self.roots[root]).resolve()
        path = (base / relative).resolve()
        if not path.is_relative_to(base):
            raise ValueError("Artifact path escapes binding")
        return path

    def check(self, asset):
        path = self.path(asset["id"])
        if not path.is_file():
            raise FileNotFoundError("Missing artifact: " + asset["id"])
        if sha256(path) != asset["sha256"]:
            raise ValueError("Artifact changed: " + asset["id"])
        return path

    def json(self, asset):
        path = self.check(asset)
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as stream:
                return json.load(stream)
        return read_json(path)


def seal(path, value):
    atomic_json(path, value)
    atomic_json(Path(path).with_suffix(".sha256.json"), {"sha256": sha256(path)})


def unseal(path):
    path = Path(path)
    if read_json(path.with_suffix(".sha256.json"))["sha256"] != sha256(path):
        raise ValueError("Manifest changed: " + path.name)
    return read_json(path)


def checked_file(root, record):
    root = Path(root).resolve()
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root) or sha256(path) != record["sha256"]:
        raise ValueError("Run artifact changed: " + record["path"])
    return path


def file_record(path, root):
    return dict(path=str(Path(path).relative_to(root)), sha256=sha256(path))


def implementation_identity():
    root = Path(__file__).parents[1]
    return object_hash(
        {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))}
    )
