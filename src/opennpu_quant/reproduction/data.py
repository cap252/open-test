"""Existing dataset adapters and frozen LFW caches, with pair-aware limits."""

from .._io import object_hash
from ..data.samples import Dataset
from ..models.contracts import EvaluationSample


class FrozenLFW:
    def __init__(self, descriptor, resolver, limit):
        from ..data.feed_cache import FeedCache

        samples = resolver.json(descriptor["samples"])
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
    if "evaluation_pairs" in dataset.info:
        pairs = dataset.info["evaluation_pairs"]
        if len(pairs) < limit:
            raise ValueError("Not enough LFW pairs")
        dataset.info = dict(
            dataset.info, evaluation_pairs=pairs[:limit], smoke=limit != 6000
        )
    elif len(dataset.rows("validation")) < limit:
        raise ValueError("Not enough validation samples")
    return dataset


def data_identity(dataset, limit):
    return dataset.identity("validation", limit)


def full_count(entry, resolver):
    descriptor = entry["dataset"]
    if descriptor.get("kind") == "frozen_lfw":
        samples = resolver.json(descriptor["samples"])
        if len(samples["pairs"]) != 6000:
            raise ValueError("Full LFW requires 6000 official pairs")
        return 6000
    info = resolver.json(descriptor["manifest"])
    if "evaluation_pairs" in info:
        if len(info["evaluation_pairs"]) != 6000 or info.get("smoke"):
            raise ValueError("Full LFW requires 6000 official pairs")
        return 6000
    expected = info.get("expected_samples")
    if type(expected) is not int or expected < 1 or len(info["validation"]) != expected:
        raise ValueError("Full evaluation requires the complete declared dataset")
    return expected
