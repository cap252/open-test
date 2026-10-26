"""Package source identities for reuse checks, and comparison of recorded evidence."""

from pathlib import Path

from ._io import object_hash, sha256

ROOT = Path(__file__).parent
# Session creation and output copies. Caches of values computed by ORT hash them.
ORT_CODE = ("ort/session.py", "ort/fetch.py", "ort/cpu_environment.py")


def code_identity():
    """Digest of every packaged .py file. run and reproduce reuse only on a match."""
    return object_hash(
        {str(p.relative_to(ROOT)): sha256(p) for p in sorted(ROOT.rglob("*.py"))}
    )


def source_hashes(*paths):
    """SHA256 of package files by relative path, for keys narrower than the code identity."""
    return {path: sha256(ROOT / path) for path in paths}


def compare_identity_evidence(left, right):
    """Compare recorded evidence. Missing values never establish compatibility."""
    missing, different = [], []
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key), right.get(key)
        if a is None or a == "" or b is None or b == "":
            missing.append(key)
        elif a != b:
            different.append(key)
    reasons = []
    if different:
        reasons.append("mismatch:" + ",".join(different))
    if missing:
        reasons.append("insufficient_evidence:" + ",".join(missing))
    return (
        "mismatch" if different else "insufficient_evidence" if missing else "passed",
        "; ".join(reasons) or None,
    )
