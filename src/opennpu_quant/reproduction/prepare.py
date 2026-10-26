"""Verify selected model bytes and bind ordered evaluation data."""

from pathlib import Path
from collections import Counter
from .._io import object_hash
from .assets import select_catalog, Resolver, seal


def prepare(
    *,
    bindings,
    output,
    selection=None,
    model_set=None,
    models=None,
    conditions=None,
    limit=3,
):
    from .data import dataset_for, data_identity, full_count

    if limit != "full" and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer or full")
    source, entries = select_catalog(
        selection=selection, model_set=model_set, models=models, conditions=conditions
    )
    from ..data.lfw_protocol import require_full

    lfw = [e for e in entries if e["recipe"]["task"] == "face_verification"]
    if lfw:
        require_full(limit=limit)
    resolver = Resolver(bindings)
    for entry in lfw:
        try:
            full_count(entry, resolver)
        except FileNotFoundError:
            pass  # Missing assets are recorded as pending below.
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    records, data_cache, verified = [], {}, {}

    def check(asset, *, graph=False):
        key = (graph, object_hash(asset))
        if key not in verified:
            verified[key] = resolver.graph(asset) if graph else resolver.check(asset)
        return verified[key]

    for entry in entries:
        row = dict(entry=entry, state="ready", missing=[], gpu_reference="pending")
        try:
            if ":" in entry["recipe"]["preprocess"]["kind"]:
                raise ValueError(
                    "Fixed replay requires identifiable built-in preprocessing"
                )
            for asset in entry["graphs"].values():
                check(asset, graph=True)
            check(entry["source_result"])
            if entry.get("audit"):
                check(entry["audit"])
            elif entry["condition"] != "fp32":
                raise FileNotFoundError("Missing QDQ audit")
            count = full_count(entry, resolver) if limit == "full" else limit
            if entry["recipe"]["task"] == "face_verification":
                count = 6000
            row["limit"] = count
            key = object_hash([entry["dataset"], count])
            if key not in data_cache:
                data = dataset_for(entry, resolver, count)
                data_cache[key] = data_identity(data, count)
            row["data_identity"] = data_cache[key]
            reference = entry.get("reference")
            if reference:
                try:
                    ref = resolver.json(reference["manifest"])
                    if ref["model_sha256"] != entry["graphs"]["network"]["sha256"]:
                        raise ValueError("GPU reference is for a different model")
                    if ref.get("external_data", {}) != entry["graphs"]["network"].get(
                        "external_data", {}
                    ):
                        raise ValueError("GPU reference external data identity differs")
                    row["gpu_reference"] = (
                        "available" if len(ref["samples"]) >= count else "pending"
                    )
                    for sample in ref["samples"][:count]:
                        check(
                            dict(
                                id=reference["root"] + sample["outputs"],
                                sha256=sample["outputs_sha256"],
                            )
                        )
                except FileNotFoundError as error:
                    row["gpu_reference"] = "pending"
                    row["reference_missing"] = str(error)
            row["identity"] = object_hash([entry, row["data_identity"], count])
        except FileNotFoundError as error:
            row.update(state="pending", missing=[str(error)])
        except (ValueError, KeyError) as error:
            row.update(state="fail", missing=[str(error)])
        records.append(row)
    result = dict(
        schema_version=1,
        selection_identity=object_hash(source),
        limit=limit,
        conditions=records,
        counts=dict(Counter(r["state"] for r in records)),
        stages=[3, 32, 200],
        gpu_comparison="pending",
        retraining="not_run",
    )
    output.mkdir(parents=True)
    seal(output / "manifest.json", result)
    return dict(
        output=str(output),
        conditions=len(records),
        counts=result["counts"],
        gpu_references=dict(Counter(r["gpu_reference"] for r in records)),
    )
