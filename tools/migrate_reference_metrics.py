"""Copy reference records with canonical metric names and rebuild their hashes.

No values are recomputed. Originals, completed runs and prepared packages are
never modified. Use the generated bindings and selection for a new preparation.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import tempfile

from opennpu_quant._io import atomic_json, read_json, sha256
from opennpu_quant.reproduction.assets import Resolver, seal


RENAMES = {
    "qualcomm_angle_accuracy": "fixed_angle_accuracy",
    "qualcomm_fixed_angle_accuracy": "fixed_angle_accuracy",
    "qualcomm_angle_threshold_degrees": "fixed_angle_threshold_degrees",
    "qualcomm_reference": "fixed_angle_reference",
}


def exact(value):
    """Typed scalar bits, including signed zero, for name-only migration checks."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite value in reference record")
        return ("float64", struct.pack("!d", value).hex())
    if isinstance(value, dict):
        return ("dict", sorted((key, exact(item)) for key, item in value.items()))
    if isinstance(value, list):
        return ("list", [exact(item) for item in value])
    return (type(value).__name__, value)


def canonicalize(value, changes=None, path=""):
    """Convert known keys at the maintainer import boundary, rejecting conflicts."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = RENAMES.get(key, key)
            destination = path + "/" + name.replace("~", "~0").replace("/", "~1")
            converted = canonicalize(item, changes, destination)
            if name in result and exact(result[name]) != exact(converted):
                raise ValueError("Conflicting reference aliases at " + destination)
            result[name] = converted
            if changes is not None and name != key:
                changes.append(dict(parent=path, original=key, canonical=name))
        return result
    if isinstance(value, list):
        return [
            canonicalize(item, changes, path + "/" + str(i))
            for i, item in enumerate(value)
        ]
    exact(value)
    return value


def numeric_fingerprint(value):
    rows = []

    def visit(item, path=""):
        if isinstance(item, dict):
            for key, child in sorted(item.items()):
                visit(child, path + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, path + "/" + str(index))
        elif type(item) in (int, float):
            rows.append((path, exact(item)))

    visit(value)
    return dict(
        count=len(rows),
        sha256=hashlib.sha256(
            json.dumps(rows, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def migrate(*, selection, bindings, output, binding_name="reference-metrics-v1"):
    selection, output = Path(selection), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", binding_name):
        raise ValueError("Invalid binding name")
    resolver = Resolver(bindings)
    original_bytes = selection.read_bytes()
    original = read_json(selection)
    if original.get("metric_schema_version", 1) != 1:
        raise ValueError("Unsupported metric schema version")
    converted = copy.deepcopy(original)
    entries = converted.get("entries", [])
    ids = [entry["id"] for entry in entries]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Expected nonempty unique condition IDs")
    if any(not re.fullmatch(r"[a-z0-9_]+", key) for key in ids):
        raise ValueError("Unsafe condition ID")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".metric-migration-", dir=output.parent
    ) as temp:
        staging = Path(temp) / "bundle"
        staging.mkdir()
        verified, assets, conditions = {}, {}, []
        for entry, old in zip(entries, original["entries"]):
            asset = old["source_result"]
            key = (asset["id"], asset["sha256"])
            if key not in assets:
                source_path = resolver.check(asset)
                value = resolver.json(asset)
                changes = []
                new = canonicalize(value, changes)
                observed = new.get("result", new).get("metrics", {})
                converted_asset = asset
                if changes:
                    if binding_name in resolver.roots:
                        raise ValueError(
                            "Binding already exists. Choose a new --binding-name"
                        )
                    relative = "records/" + asset["sha256"] + ".json"
                    target = staging / relative
                    seal(target, new)
                    reread = read_json(target)
                    if exact(new) != exact(reread):
                        raise ValueError(
                            "Reference values changed during serialization"
                        )
                    converted_asset = dict(
                        id="artifact://" + binding_name + "/" + relative,
                        sha256=sha256(target),
                    )
                    verified[key] = dict(
                        original=asset,
                        converted=converted_asset,
                        changes=changes,
                        numeric_values=numeric_fingerprint(new),
                        values_bitwise_preserved=True,
                    )
                assets[key] = (converted_asset, observed)
                if sha256(source_path) != asset["sha256"]:
                    raise ValueError("Source changed during migration")
            converted_asset, observed = assets[key]
            changes = []
            entry["measurement"] = canonicalize(old.get("measurement", {}), changes)
            if exact(entry["measurement"].get("metrics", {})) != exact(observed):
                raise ValueError("Selected metrics differ from source: " + entry["id"])
            entry["source_result"] = converted_asset
            if changes or converted_asset != asset:
                conditions.append(
                    dict(
                        id=entry["id"],
                        changes=changes,
                        numeric_values=numeric_fingerprint(entry["measurement"]),
                    )
                )
        converted["metric_schema_version"] = 1
        if selection.read_bytes() != original_bytes:
            raise ValueError("Selection changed during migration")
        for old in original["entries"]:
            resolver.check(old["source_result"])
        bindings_out = dict(resolver.roots)
        if verified:
            bindings_out[binding_name] = str(output)
        archive = staging / "original-selection.json"
        archive.write_bytes(original_bytes)
        atomic_json(archive.with_suffix(".sha256.json"), dict(sha256=sha256(archive)))
        seal(staging / "selected.json", converted)
        seal(staging / "bindings.json", bindings_out)
        ledger = dict(
            schema_version=1,
            migration="canonical_metric_names_v1",
            original_selection_sha256=sha256(archive),
            converted_selection_sha256=sha256(staging / "selected.json"),
            tool_sha256=sha256(__file__),
            entries=len(entries),
            migrated_conditions=conditions,
            migrated_assets=list(verified.values()),
            numeric_values_unchanged=True,
            inference="not_run",
        )
        seal(staging / "migration.json", ledger)
        staging.rename(output)
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binding-name", default="reference-metrics-v1")
    args = parser.parse_args()
    result = migrate(**vars(args))
    print(
        json.dumps(
            dict(
                output=str(args.output),
                migrated_conditions=len(result["migrated_conditions"]),
                migrated_assets=len(result["migrated_assets"]),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
