"""Existing dataset adapters and frozen LFW caches, with pair-aware limits."""

from .._io import object_hash
from ..data.samples import Dataset
from ..data.lfw_protocol import require_full
from ..models.contracts import EvaluationSample


class FrozenLFW:
    """Frozen LFW feeds from a verified cache.

    Provides the Dataset members that evaluation reads: info, annotations,
    identity() and samples().
    """

    def __init__(self, descriptor, resolver, limit):
        from ..data.feed_cache import FeedCache

        samples = resolver.json(descriptor["samples"])
        require_full(samples, limit)
        limit = 6000
        cache_path = resolver.check(descriptor["cache"])
        manifest = resolver.json(descriptor["cache"])
        self.cache = FeedCache(
            cache_path.parent,
            identity=manifest["source_identity"],
            verify_source=lambda: resolver.check(descriptor["cache"]),
        )
        pairs = samples["pairs"][:limit]
        if len(pairs) != limit:
            raise ValueError("Not enough LFW pairs")
        required = {p[k] for p in pairs for k in ("first", "second")}
        self.rows = [
            (i, dict(id=r["image_id"], image_key=r["image_key"], flip=r["flip"]))
            for i, r in enumerate(samples["validation"])
            if r["image_key"] in required
        ]
        views = {(r["image_key"], r["flip"]) for _, r in self.rows}
        if views != {(k, f) for k in required for f in (0, 1)} or len(views) != len(
            self.rows
        ):
            raise ValueError("Incomplete LFW views")
        self.info = dict(
            evaluation_pairs=pairs, smoke=limit != 6000, expected_samples=6000
        )
        self.annotations = {}
        self.descriptor = descriptor

    def identity(self, split, limit):
        self.cache.verify()
        return object_hash([self.descriptor, self.info, self.rows])

    def samples(self, bundle, split="validation", limit=None):
        needed = dict(self.rows)
        for index, feeds in enumerate(self.cache()):
            if index in needed:
                row = needed[index]
                yield EvaluationSample(
                    row["id"], feeds, None, {"dataset_row": row, "context": feeds}
                )
            if index >= max(needed):
                break


def dataset_for(entry, resolver, limit):
    desc = entry["dataset"]
    if desc.get("kind") == "frozen_lfw":
        return FrozenLFW(desc, resolver, limit)
    dataset = Dataset(
        resolver.check(desc["manifest"]), root=resolver.path(desc["root"])
    )
    if (
        entry.get("recipe", {}).get("task") == "face_verification"
        or "evaluation_pairs" in dataset.info
    ):
        require_full(dataset.info, limit)
        return dataset
    if len(dataset.rows("validation")) < limit:
        raise ValueError("Not enough validation samples")
    return dataset


def data_identity(dataset, limit):
    return dataset.identity("validation", limit)


def full_count(entry, resolver):
    descriptor = entry["dataset"]
    if descriptor.get("kind") == "frozen_lfw":
        samples = resolver.json(descriptor["samples"])
        require_full(samples)
        return 6000
    info = resolver.json(descriptor["manifest"])
    if (
        entry.get("recipe", {}).get("task") == "face_verification"
        or "evaluation_pairs" in info
    ):
        require_full(info)
        return 6000
    expected = info.get("expected_samples")
    if type(expected) is not int or expected < 1 or len(info["validation"]) != expected:
        raise ValueError("Full evaluation requires the complete declared dataset")
    return expected


def require_full_lfw_rows(rows, default_limit, resolver=None):
    from ..data.lfw_protocol import reject_partial_result

    for row in rows:
        reject_partial_result(row)
        if row["entry"]["recipe"]["task"] == "face_verification":
            require_full(limit=row.get("limit", default_limit))
            if resolver is not None:
                try:
                    full_count(row["entry"], resolver)
                except FileNotFoundError:
                    pass  # Execution records unavailable assets without inference.
